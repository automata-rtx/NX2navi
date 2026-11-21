#include "resource_core_effects.h"
#include "resource_core_shader_writes.h"
#include <limits.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#ifdef RC_EFFECT_DEBUG
#include <stdio.h>
#define TRACE(...) fprintf(stderr, __VA_ARGS__)
#else
#define TRACE(...) ((void)0)
#endif
#ifndef RC_EFFECT_STEP_LIMIT
#define RC_EFFECT_STEP_LIMIT 50000u
#endif
#ifndef RC_EFFECT_RECIPES
#define RC_EFFECT_RECIPES 1
#endif


uint32_t rc_effect_write_class(const uint32_t *w, size_t count) {
  if (!w || count < 5 || w[0] != 0x07230203 || !w[3] || w[3] > (1u << 20))
    return RC_EFFECT_WRITES_UNKNOWN;
  uint32_t bound = w[3], result = 0;
  uint32_t *types = calloc(bound, sizeof *types),
           *values = calloc(bound, sizeof *values);
  if (!types || !values) {
    result = RC_EFFECT_WRITES_UNKNOWN;
    goto done;
  }
  for (size_t i = 5; i < count;) {
    uint32_t n = w[i] >> 16, op = w[i] & 65535;
    if (!n || n > count - i) {
      result |= RC_EFFECT_WRITES_UNKNOWN;
      goto done;
    }
    if (op == 32) {
      if (n != 4 || w[i + 1] >= bound || w[i + 2] == UINT32_MAX) {
        result |= RC_EFFECT_WRITES_UNKNOWN;
        goto done;
      }
      types[w[i + 1]] = w[i + 2] + 1;
    }
    i += n;
  }
  for (size_t i = 5; i < count; i += w[i] >> 16) {
    uint32_t n = w[i] >> 16, op = w[i] & 65535;
    int typed = op == 1 || op == 46 || op == 55 || op == 57 ||
                (op >= 59 && op <= 61) || (op >= 65 && op <= 67) || op == 70 ||
                op == 83 || op == 143 || (op >= 120 && op <= 124) ||
                op == 169 || op == 245;
    if (typed) {
      if (n < 3 || w[i + 1] >= bound || w[i + 2] >= bound) {
        result |= RC_EFFECT_WRITES_UNKNOWN;
        goto done;
      }
      values[w[i + 2]] = types[w[i + 1]];
    }
  }
  for (size_t i = 5; i < count; i += w[i] >> 16) {
    uint32_t n = w[i] >> 16, op = w[i] & 65535, at = 0;
    if (op == 99) {
      result |= RC_EFFECT_WRITES_IMAGE;
      continue;
    }
    if ((op >= 62 && op <= 64) || op == 228 || op == 319)
      at = 1;
    else if ((op >= 229 && op <= 242) || op == 318 || op == 5614 ||
             op == 5615 || op == 6035)
      at = 3;
    else if (op >= 4096 && !rc_shader_extension_no_guest_write(op, n))
      result |= RC_EFFECT_WRITES_UNKNOWN;
    if (!at)
      continue;
    if (n <= at || w[i + at] >= bound || !values[w[i + at]]) {
      result |= RC_EFFECT_WRITES_UNKNOWN;
      continue;
    }
    switch (values[w[i + at]] - 1) {
    case 2:
    case 12:
      result |= RC_EFFECT_WRITES_BUFFER;
      break;
    case 5349:
      result |= RC_EFFECT_WRITES_GPA;
      break;
    case 11:
      result |= RC_EFFECT_WRITES_IMAGE;
      break;
    case 3:
    case 4:
    case 6:
    case 7:
    case 5402:
      // TaskPayloadWorkgroupEXT passes execution-owned data to a mesh stage. It is not a guest buffer;
      // translated fallback buffers, if present, still appear separately as StorageBuffer destinations.
      break; // Stage outputs, workgroup/task payload, private and function memory.
    default:
      result |= RC_EFFECT_WRITES_UNKNOWN;
      break;
    }
  }
done:
  free(types);
  free(values);
  return result;
}

// This is an abstract interpreter of address expressions, not a shader
// emulator. Unsupported data values become their entire integer domain. A write
// is narrowed only when all paths through its address computation are bounded.
enum {
  E_UNKNOWN,
  E_CONSTANT,
  E_READ,
  E_BUILTIN,
  E_CHOICE,
  E_UNION,
  E_FIRST_SPV = 1024
};
typedef struct {
  uint32_t op, a, b, c, bits;
  uint64_t value;
} Node;
typedef struct {
  uint32_t node, block, condition;
  uint8_t truth;
} Choice;
typedef struct {
  uint32_t node;
  uint8_t truth;
} Fact;
typedef struct {
  uint32_t node;
  uint8_t truth, a_exact, b_exact;
  uint64_t a, b;
} RuntimeFact;


typedef struct {
  uint32_t start, end, condition, yes, no, predecessor, guards, nguards, loop,
      merge;
} Block;
typedef struct {
  uint32_t block, next;
} Pred;
typedef struct {
  uint32_t kind, set, binding, low, high, bytes, block;
  uint32_t original, store_value;
  int gpa_twin;
  uint64_t affine_end;
} Effect;
typedef struct {
  uint32_t variable, block, node;
} Local;
typedef struct {
  uint32_t storage, set, binding, builtin, offset, pointee, buffer;
} Pointer;
struct RcEffectProgram {
  uint64_t identity;
  const uint32_t *w;
  size_t words;
  uint32_t bound, *definition, *id_block, *expression, *label_block, *set,
      *binding, *builtin;
  uint32_t *array_stride, *block_flags, *dom, *reach, *loop_writes, loop_words;
  Block *blocks;
  uint32_t nblocks, cblocks, dom_words;
  Pred *pred;
  uint32_t npred, cpred;
  Node *node;
  uint32_t nnode, cnode;
  Choice *choice;
  uint32_t nchoice, cchoice;
  Fact *guard;
  uint32_t nguard, cguard;
  Effect *effect;
  uint32_t neffect, ceffect;
  Effect *read;
  uint32_t nread, cread;
  int reads_unknown;
  Local *local;
  uint32_t nlocal, clocal;
  uint32_t *intern;
  uint32_t cintern;
  int supported, building, definite_stores, mirrored_gpa_stores;
  uint32_t original_effects, affine_local[3];
  int recipes;
};
typedef struct {
  uint64_t lo, hi;
} Interval;
typedef struct {
  Interval value;
  uint64_t context;
  uint32_t visiting, unstable;
} Cache;
typedef struct {
  const RcEffectProgram *p;
  const RcEffectInputs *in;
  Cache *cache;
  RuntimeFact fact[256];
  uint32_t nfacts, depth, ticks, visiting;
  uint64_t context, next_context;
  int exhausted;
  uint32_t last_unstable;

} Eval;

static int grow(void **p, uint32_t *cap, uint32_t count, size_t size) {
  if (count < *cap)
    return 1;
  uint32_t n = *cap ? *cap * 2 : 64;
  if (n < count + 1)
    n = count + 1;
  if (n > (1u << 22) || size > SIZE_MAX / n)
    return 0;
  void *q = realloc(*p, size * n);
  if (!q)
    return 0;
  *p = q;
  *cap = n;
  return 1;
}
static const uint32_t *def(const RcEffectProgram *p, uint32_t id) {
  return id < p->bound && p->definition[id] ? p->w + p->definition[id] : NULL;
}
static uint32_t opcode(const uint32_t *d) { return d ? d[0] & 65535 : 0; }
static uint32_t length(const uint32_t *d) { return d ? d[0] >> 16 : 0; }
static int constant(const RcEffectProgram *p, uint32_t id, uint64_t value) {
  const uint32_t *d = def(p, id);
  return opcode(d) == 43 && length(d) >= 4 && d[3] == (uint32_t)value &&
         (length(d) == 4 ? value <= UINT32_MAX
                         : length(d) == 5 && d[4] == (uint32_t)(value >> 32));
}
static uint32_t bits_type(const RcEffectProgram *p, uint32_t type) {
  const uint32_t *d = def(p, type);
  if (opcode(d) == 20)
    return 1;
  if ((opcode(d) == 21 || opcode(d) == 22) && (d[2] == 32 || d[2] == 64))
    return d[2];
  return 32;
}
static uint64_t mask_bits(uint32_t bits) {
  return bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
}
static uint32_t node_new(RcEffectProgram *p, Node n) {
  if (!grow((void **)&p->node, &p->cnode, p->nnode, sizeof *p->node)) {
    p->supported = 0;
    return 0;
  }
  uint32_t id = p->nnode++;
  p->node[id] = n;
  return id;
}
static uint64_t node_hash(Node n) {
  uint64_t h = UINT64_C(1469598103934665603);
  uint32_t v[] = {n.op,
                  n.a,
                  n.b,
                  n.c,
                  n.bits,
                  (uint32_t)n.value,
                  (uint32_t)(n.value >> 32)};
  for (unsigned i = 0; i < 7; i++)
    h = (h ^ v[i]) * UINT64_C(1099511628211);
  return h;
}
static uint32_t intern(RcEffectProgram *p, Node n) {
  if (!p->cintern || p->nnode * 2 >= p->cintern) {
    uint32_t cap = p->cintern ? p->cintern * 2 : 1024;
    while (cap < p->nnode * 2 + 2)
      cap *= 2;
    uint32_t *t = calloc(cap, sizeof *t);
    if (!t) {
      p->supported = 0;
      return 0;
    }
    for (uint32_t i = 1; i < p->nnode; i++)
      if (p->node[i].op != E_CHOICE) {
        uint32_t k = (uint32_t)node_hash(p->node[i]) & (cap - 1);
        while (t[k])
          k = (k + 1) & (cap - 1);
        t[k] = i;
      }
    free(p->intern);
    p->intern = t;
    p->cintern = cap;
  }
  uint32_t k = (uint32_t)node_hash(n) & (p->cintern - 1);
  while (p->intern[k]) {
    uint32_t i = p->intern[k];
    Node x = p->node[i];
    if (x.op == n.op && x.a == n.a && x.b == n.b && x.c == n.c &&
        x.bits == n.bits && x.value == n.value)
      return i;
    k = (k + 1) & (p->cintern - 1);
  }
  uint32_t i = node_new(p, n);
  p->intern[k] = i;
  return i;
}
static uint32_t cn(RcEffectProgram *p, uint64_t v, uint32_t bits) {
  return intern(p, (Node){E_CONSTANT, 0, 0, 0, bits, v & mask_bits(bits)});
}
static uint32_t canonical(const RcEffectProgram *p, uint32_t id) {
  for (unsigned i = 0; i < 256 && p->node[id].op == E_FIRST_SPV + 83; i++)
    id = p->node[id].a;
  return id;
}
static uint32_t operation(RcEffectProgram *p, uint32_t op, uint32_t a,
                          uint32_t b, uint32_t c, uint32_t bits) {
  a = canonical(p, a);
  b = canonical(p, b);
  c = canonical(p, c);
  if (op == 124 || op == 83) {
    if (p->node[a].bits != bits)
      return intern(p, (Node){E_UNKNOWN, a, op, 2, bits, 0});
    return a;
  }
  if ((op == 128 || op == 197 || op == 198) && p->node[b].op == E_CONSTANT &&
      !p->node[b].value)
    return a;
  if (op == 128 && p->node[a].op == E_CONSTANT && !p->node[a].value)
    return b;
  if (op == 169 && b == c)
    return b;
  return intern(p, (Node){E_FIRST_SPV + op, a, b, c, bits, 0});
}
static uint32_t expr(RcEffectProgram *p, uint32_t id);
static uint32_t local_at(RcEffectProgram *p, uint32_t var, uint32_t block,
                         uint32_t before);
static int dominates(const RcEffectProgram *p, uint32_t a, uint32_t b) {
  return a && b &&
         (p->dom[(size_t)b * p->dom_words + a / 32] & (1u << (a % 32)));
}
static uint32_t local_start(RcEffectProgram *p, uint32_t var, uint32_t block) {
  if (!block)
    return 0;
  for (uint32_t i = 0; i < p->nlocal; i++)
    if (p->local[i].variable == var && p->local[i].block == block)
      return p->local[i].node;
  if (!grow((void **)&p->local, &p->clocal, p->nlocal, sizeof *p->local)) {
    p->supported = 0;
    return 0;
  }
  uint32_t id = node_new(p, (Node){E_CHOICE, 0, 0, 0, 32, 0});
  p->local[p->nlocal++] = (Local){var, block, id};
  // Store edges separately until recursion finishes: nested local queries
  // append their own alternatives.
  Choice edge[64];
  uint32_t count = 0;
  for (uint32_t q = p->blocks[block].predecessor; q; q = p->pred[q].next) {
    if (count == 64) {
      p->supported = 0;
      return 0;
    }
    uint32_t b = p->pred[q].block;
    Block *x = &p->blocks[b];
    int back = dominates(p, block, b);
    if (back && !(p->loop_writes[(size_t)block * p->loop_words + var / 32] &
                  (1u << (var % 32))))
      continue;
    uint32_t condition = x->condition && !back ? expr(p, x->condition) : 0;
    uint32_t value =
        back ? intern(p, (Node){E_UNKNOWN, var, block, 1, 32, UINT64_MAX})
             : local_at(p, var, b, x->end);
    edge[count++] =
        (Choice){value, back ? 0 : b, condition, (uint8_t)(x->yes == block)};
  }
  if (!count) {
    const uint32_t *variable = def(p, var);
    if (opcode(variable) == 59 && length(variable) == 5) {
      uint32_t initializer = expr(p, variable[4]);
      p->node[id] = (Node){E_FIRST_SPV + 83, initializer, 0, 0,
                           p->node[initializer].bits, 0};
      return initializer;
    }
    p->node[id] = (Node){E_UNKNOWN, var, block, 2, 32, UINT64_MAX};
    return id;
  }
  uint32_t first = p->nchoice;
  for (uint32_t i = 0; i < count; i++) {
    if (!grow((void **)&p->choice, &p->cchoice, p->nchoice,
              sizeof *p->choice)) {
      p->supported = 0;
      return 0;
    }
    p->choice[p->nchoice++] = edge[i];
  }
  p->node[id].a = first;
  p->node[id].b = count;
  uint32_t same = canonical(p, edge[0].node);
  int identical = same != id;
  for (uint32_t i = 1; i < count; i++)
    identical &= canonical(p, edge[i].node) == same;
  if (identical) {
    p->node[id] = (Node){E_FIRST_SPV + 83, same, 0, 0, 32, 0};
    return same;
  }
  return id;
}
static uint32_t local_at(RcEffectProgram *p, uint32_t var, uint32_t block,
                         uint32_t before) {
  if (!block)
    return 0;
  uint32_t latest = 0;
  for (uint32_t i = p->blocks[block].start; i < before;) {
    const uint32_t *d = p->w + i;
    if (opcode(d) == 62 && d[1] == var)
      latest = d[2];
    i += length(d);
  }
  return latest ? expr(p, latest) : local_start(p, var, block);
}
static uint32_t scalar_size(const RcEffectProgram *p, uint32_t type) {
  const uint32_t *d = def(p, type);
  if (opcode(d) == 20)
    return 4;
  if (opcode(d) == 21 || opcode(d) == 22)
    return d[2] / 8;
  if (opcode(d) == 23)
    return scalar_size(p, d[2]) * d[3];
  return 0;
}
static int member_offset(const RcEffectProgram *p, uint32_t type,
                         uint32_t member, uint32_t *out) {
  for (size_t i = 5; i < p->words; i += p->w[i] >> 16)
    if ((p->w[i] & 65535) == 72 && (p->w[i] >> 16) == 5 &&
        p->w[i + 1] == type && p->w[i + 2] == member && p->w[i + 3] == 35) {
      *out = p->w[i + 4];
      return 1;
    }
  return 0;
}
static int pointer(RcEffectProgram *p, uint32_t id, Pointer *out,
                   uint32_t depth) {
  if (depth > 64)
    return 0;
  const uint32_t *d = def(p, id);
  if (!d)
    return 0;
  if (opcode(d) == 59) {
    const uint32_t *t = def(p, d[1]);
    if (opcode(t) != 32)
      return 0;
    *out = (Pointer){d[3],
                     p->set[id],
                     p->binding[id],
                     p->builtin[id],
                     cn(p, 0, 32),
                     t[3],
                     d[3] == 12 || p->block_flags[t[3]] == 3};
    return 1;
  }
  if (opcode(d) != 65 && opcode(d) != 66)
    return 0;
  if (!pointer(p, d[3], out, depth + 1))
    return 0;
  for (uint32_t j = 4; j < length(d); j++) {
    const uint32_t *t = def(p, out->pointee);
    if (!t)
      return 0;
    uint32_t off = 0, step = 0, index = expr(p, d[j]);
    if (opcode(t) == 30) {
      const uint32_t *c = def(p, d[j]);
      if (opcode(c) != 43 || c[3] + 2 >= length(t) ||
          !member_offset(p, out->pointee, c[3], &off))
        return 0;
      out->pointee = t[c[3] + 2];
      index = cn(p, off, 32);
    } else if (opcode(t) == 28 || opcode(t) == 29) {
      step = p->array_stride[out->pointee];
      out->pointee = t[2];
      if (!step)
        return 0;
    } else if (opcode(t) == 23) {
      step = scalar_size(p, t[2]);
      out->pointee = t[2];
      if (!step)
        return 0;
    } else
      return 0;
    if (step)
      index = operation(p, 132, index, cn(p, step, 32), 0, 32);
    out->offset = operation(p, 128, out->offset, index, 0, 32);
  }
  return 1;
}
// The affine store class addresses scalar/vector elements of a final storage array.
// Every preceding access selects a literal structure member, so the array's
// object bound is an independent check on its byte offset.
static uint64_t affine_storage_end(const RcEffectProgram *p, uint32_t ptr, uint32_t bytes) {
  const uint32_t *access = def(p, ptr);
  if ((opcode(access) != 65 && opcode(access) != 66) || length(access) < 5)
    return 0;
  const uint32_t *var = def(p, access[3]);
  if (opcode(var) != 59)
    return 0;
  const uint32_t *pointer = def(p, var[1]);
  if (opcode(pointer) != 32)
    return 0;
  uint32_t type = pointer[3];
  uint64_t base = 0;
  for (uint32_t i = 4; i + 1 < length(access); i++) {
    const uint32_t *structure = def(p, type), *member = def(p, access[i]);
    uint32_t offset;
    if (opcode(structure) != 30 || opcode(member) != 43 || length(member) != 4 ||
        member[3] >= length(structure) - 2 ||
        !member_offset(p, type, member[3], &offset) || offset > UINT32_MAX - base)
      return 0;
    base += offset;
    type = structure[member[3] + 2];
  }
  const uint32_t *array = def(p, type);
  if ((opcode(array) != 28 && opcode(array) != 29) ||
      scalar_size(p, array[2]) != bytes || !p->array_stride[type])
    return 0;
  const uint64_t limit = UINT64_C(1) << 32;
  if (opcode(array) == 29)
    return limit; // The retained Vulkan descriptor bounds a runtime array.
  const uint32_t *count = def(p, array[3]);
  if (opcode(count) != 43 || length(count) != 4 || !count[3])
    return 0;
  uint64_t end = (uint64_t)(count[3] - 1) * p->array_stride[type] + base + bytes;
  return end < limit ? end : limit;
}

static uint32_t expr(RcEffectProgram *p, uint32_t id) {
  if (!id || id >= p->bound)
    return 0;
  if (p->expression[id] != UINT32_MAX)
    return canonical(p, p->expression[id]);
  if (++p->building > 512) {
    p->building--;
    return intern(p, (Node){E_UNKNOWN, 0, 0, 2, 32, id});
  }
  p->expression[id] = intern(p, (Node){E_UNKNOWN, 0, 0, 2, 32, id});
  const uint32_t *d = def(p, id);
  uint32_t r = 0, op = opcode(d), bits = d ? bits_type(p, d[1]) : 32;
  if (op == 46 && (opcode(def(p, d[1])) == 20 || opcode(def(p, d[1])) == 21 ||
                   opcode(def(p, d[1])) == 22))
    r = cn(p, 0, bits);
  else if (op == 43)
    r = cn(p, d[3] | (length(d) == 5 ? (uint64_t)d[4] << 32 : 0), bits);
  else if (op == 41 || op == 42)
    r = cn(p, op == 41, 1);
  else if (op == 61 && length(d) >= 4) {
    const uint32_t *v = def(p, d[3]);
    if (opcode(v) == 59 && v[3] == 7)
      r = local_at(p, d[3], p->id_block[id], p->definition[id]);
    else {
      Pointer pt;
      if (pointer(p, d[3], &pt, 0)) {
        if (pt.storage == 1 && pt.builtin != UINT32_MAX)
          r = intern(p, (Node){E_BUILTIN, pt.offset, pt.builtin, 0, bits, 0});
        else if ((pt.storage == 2 || pt.storage == 12) &&
                 pt.binding != UINT32_MAX && bits <= 64 &&
                 (opcode(def(p, pt.pointee)) == 21 ||
                  opcode(def(p, pt.pointee)) == 22))
          // Distinct loads may observe different values when the runtime cannot
          // supply immutable bytes. Preserve their SSA result identities; a
          // guard on an earlier load cannot constrain a later one. The high word records
          // whether this is storage data rather than an immutable uniform expression.
          r = intern(p,
                     (Node){E_READ, pt.offset, pt.set, pt.binding, bits, ((uint64_t)pt.buffer << 32) | id});
      }
    }
  } else if (op == 81 && length(d) == 5) {
    const uint32_t *v = def(p, d[3]);
    Pointer pt;
    if (opcode(v) == 61 && pointer(p, v[3], &pt, 0) && pt.storage == 1 &&
        pt.builtin != UINT32_MAX)
      r = intern(
          p, (Node){E_BUILTIN, cn(p, d[4] * 4, 32), pt.builtin, 0, bits, 0});
    else if ((opcode(v) == 151 || opcode(v) == 152) && d[4] <= 1)
      r = operation(p, d[4] ? opcode(v) : 132, expr(p, v[3]), expr(p, v[4]), 0,
                    bits);
    else if ((opcode(v) == 149 || opcode(v) == 150) && d[4] <= 1)
      r = operation(p,
                    d[4]               ? opcode(v)
                    : opcode(v) == 149 ? 128
                                       : 130,
                    expr(p, v[3]), expr(p, v[4]), 0, bits);
  } else if ((op >= 113 && op <= 114) || op == 124 || op == 83 || op == 126 ||
             op == 168 || op == 200)
    r = operation(p, op, expr(p, d[3]), 0, 0, bits);
  else if ((op >= 128 && op <= 139) || (op >= 164 && op <= 179) ||
           (op >= 194 && op <= 199))
    r = operation(p, op, expr(p, d[3]), expr(p, d[4]),
                  op == 169 ? expr(p, d[5]) : 0, bits);
  else if (op == 204 || op == 205) {
    // Keep operand instability even when only the broad result range is needed.
    r = operation(p, op, expr(p, d[3]), 0, 0, bits);
  }
  // Unknown atoms, sampled data, floating-point math and subgroup results
  // retain their complete value domain.
  if (!r) {
    int memory_value = op == 61 || op == 227 || (op >= 229 && op <= 242) ||
                       op == 318 || op == 5614 || op == 5615 || op == 6035;
    if (memory_value) {
      const uint32_t *pointer_def = def(p, d[3]);
      const uint32_t *pointer_type = pointer_def ? def(p, pointer_def[1]) : NULL;
      memory_value = opcode(pointer_type) == 32 &&
                     (pointer_type[2] == 2 || pointer_type[2] == 12 || pointer_type[2] == 5349);
    }
    // Unsupported computations may carry undefined operands. Without their operand semantics they cannot
    // establish a correlated guard on another use. An opaque memory result retains its own SSA identity.
    r = intern(p, (Node){E_UNKNOWN, 0, 0, memory_value ? 0u : 2u, bits, id});
  }
  p->expression[id] = r;
  p->building--;
  return r;
}

// Validate the complete address-table lookup, including its page index and
// record stride. Merely finding a 64-bit addition near a physical pointer is
// insufficient to infer a guest address. The runtime also verifies this binding
// is its GPA table before accepting a GPA range.
static uint32_t unary(const RcEffectProgram *p, uint32_t id, uint32_t op) {
  const uint32_t *d = def(p, id);
  return opcode(d) == op && length(d) == 4 ? d[3] : 0;
}
static int binary_const(const RcEffectProgram *p, uint32_t id, uint32_t op,
                        uint64_t k, uint32_t *a) {
  const uint32_t *d = def(p, id);
  if (opcode(d) != op || length(d) != 5 || !constant(p, d[4], k))
    return 0;
  *a = d[3];
  return 1;
}
static int table_word(const RcEffectProgram *p, uint32_t id, uint32_t *table,
                      uint32_t *index) {
  const uint32_t *d = def(p, id);
  if (opcode(d) != 61)
    return 0;
  const uint32_t *a = def(p, d[3]);
  if (opcode(a) != 65 || length(a) != 6 || !constant(p, a[4], 0))
    return 0;
  if (*table && *table != a[3])
    return 0;
  *table = a[3];
  *index = a[5];
  return 1;
}
static int wide_pair(const RcEffectProgram *p, uint32_t id, uint32_t *lo,
                     uint32_t *hi) {
  const uint32_t *d = def(p, id);
  if (opcode(d) != 197 || length(d) != 5)
    return 0;
  *lo = unary(p, d[3], 113);
  uint32_t h;
  if (!*lo || !binary_const(p, d[4], 196, 32, &h) || !(*hi = unary(p, h, 113)))
    return 0;
  return 1;
}
static int gpa_pointer(RcEffectProgram *p, uint32_t id, uint32_t *low,
                       uint32_t *high, uint32_t *table, int *uses_twin) {
  uint32_t address = unary(p, id, 120);
  const uint32_t *s = def(p, address);
  if (opcode(s) != 169 || length(s) != 6)
    return 0;
  const uint32_t *device = def(p, s[4]), *dummy = def(p, s[5]);
  if (opcode(device) != 128 || opcode(dummy) != 128)
    return 0;
  uint32_t wl, dh, dl, th;
  if (!wide_pair(p, device[3], &wl, high) ||
      !binary_const(p, wl, 199, 0xfffffffc, low) ||
      !wide_pair(p, device[4], &dl, &dh))
    return 0;
  uint32_t rlo, rhi;
  *table = 0;
  if (!table_word(p, dl, table, &rlo) || !table_word(p, dh, table, &rhi))
    return 0;
  uint32_t recordword = 0, twin = 0;
  if (binary_const(p, rlo, 128, 2, &recordword))
    twin = 1;
  else
    recordword = rlo;
  uint32_t rhbase;
  if (!binary_const(p, rhi, 128, twin ? 3 : 1, &rhbase) || rhbase != recordword)
    return 0;
  const uint32_t *rw = def(p, recordword);
  uint32_t baseindex, record;
  if (opcode(rw) != 128 || !table_word(p, rw[3], table, &baseindex) ||
      !constant(p, baseindex, 2) || !binary_const(p, rw[4], 132, 4, &record))
    return 0;
  uint32_t ri;
  if (!table_word(p, record, table, &ri))
    return 0;
  const uint32_t *ra = def(p, ri);
  if (opcode(ra) != 128)
    return 0;
  uint32_t page, shift, blockmul, block;
  if (!binary_const(p, ra[4], 199, 0xfff, &shift) ||
      !binary_const(p, shift, 194, 12, &page) || page != *low ||
      !binary_const(p, ra[3], 128, 65552, &blockmul) ||
      !binary_const(p, blockmul, 132, 4096, &block))
    return 0;
  uint32_t bi, l1;
  if (!table_word(p, block, table, &bi) || !binary_const(p, bi, 128, 16, &l1))
    return 0;
  const uint32_t *ls = def(p, l1);
  if (opcode(ls) != 169 || !constant(p, ls[5], 0))
    return 0;
  uint32_t h;
  if (!binary_const(p, ls[3], 176, 256, &h) || h != *high)
    return 0;
  const uint32_t *ix = def(p, ls[4]);
  uint32_t il, ih;
  if (opcode(ix) != 197 || !binary_const(p, ix[3], 196, 8, &ih) ||
      ih != *high || !binary_const(p, ix[4], 194, 24, &il) || il != *low)
    return 0;
  uint32_t dummylo, dummyhi, di;
  if (!wide_pair(p, dummy[3], &dummylo, &dummyhi) ||
      !table_word(p, dummylo, table, &di) || !constant(p, di, twin ? 3 : 0) ||
      !table_word(p, dummyhi, table, &di) || !constant(p, di, twin ? 4 : 1) ||
      !(th = unary(p, dummy[4], 113)) ||
      !binary_const(p, th, 199, 0xffc, &di) || di != *low)
    return 0;
  uint32_t mapped = s[3];
  if (twin) {
    const uint32_t *both = def(p, mapped);
    if (opcode(both) != 167)
      return 0;
    mapped = both[3];
    uint32_t same = unary(p, both[4], 168);
    const uint32_t *eqs = def(p, same);
    if (opcode(eqs) != 167)
      return 0;
    const uint32_t *eqlo = def(p, eqs[3]), *eqhi = def(p, eqs[4]);
    uint32_t primarylo, primaryhi;
    if (opcode(eqlo) != 170 || opcode(eqhi) != 170 || eqlo[3] != dl ||
        eqhi[3] != dh || !table_word(p, eqlo[4], table, &primarylo) ||
        primarylo != recordword || !table_word(p, eqhi[4], table, &primaryhi) ||
        !binary_const(p, primaryhi, 128, 1, &rhbase) || rhbase != recordword)
      return 0;
  }
  uint32_t mr;
  if (!binary_const(p, mapped, 171, 0, &mr) || mr != record)
    return 0;
  *uses_twin = (int)twin;
  return 1;
}

static int no_result(uint32_t op) {
  switch (op) {
  case 0:
  case 3:
  case 4:
  case 5:
  case 6:
  case 8:
  case 10:
  case 14:
  case 15:
  case 16:
  case 17:
  case 56:
  case 62:
  case 63:
  case 64:
  case 71:
  case 72:
  case 73:
  case 74:
  case 75:
  case 99:
  case 218:
  case 219:
  case 220:
  case 221:
  case 224:
  case 225:
  case 228:
  case 246:
  case 247:
  case 249:
  case 250:
  case 251:
  case 252:
  case 253:
  case 254:
  case 255:
  case 256:
  case 257:
  case 317:
  case 319:
  case 330:
    return 1;
  default:
    return op >= 4096;
  }
}
static int instruction_length(uint32_t op, uint32_t n) {
  switch (op) {
  case 19:
  case 20:
  case 248:
  case 249:
    return n == 2;
  case 21:
  case 23:
  case 32:
  case 113:
  case 114:
  case 124:
  case 83:
  case 126:
  case 168:
  case 200:
    return n == 4;
  case 22:
  case 29:
  case 41:
  case 42:
  case 247:
    return n == 3;
  case 43:
    return n == 4 || n == 5;
  case 54:
  case 68:
    return n == 5;
  case 59:
    return n == 4 || n == 5;
  case 61:
    return n >= 4;
  case 62:
    return n >= 3;
  case 65:
  case 66:
    return n >= 5;
  case 71:
    return n >= 3;
  case 72:
    return n >= 4;
  case 81:
    return n >= 5;
  case 24:
  case 28:
  case 319:
    return n == 4;
  case 30:
    return n >= 2;
  case 33:
    return n >= 3;
  case 227:
  case 318:
    return n == 6;
  case 228:
    return n == 5;
  case 5614:
  case 5615:
  case 6035:
    return n >= 7;
  case 169:
    return n == 6;
  case 246:
    return n >= 4;
  case 250:
    return n == 4 || n == 6;
  case 56:
  case 253:
    return n == 1;
  default:
    if ((op >= 128 && op <= 152) || (op >= 164 && op <= 179) ||
        (op >= 194 && op <= 199))
      return n == 5;
    if (op >= 229 && op <= 242)
      return n >= 6;
    return n >= 1;
  }
}
static int edge(RcEffectProgram *p, uint32_t from, uint32_t to) {
  if (!to || !grow((void **)&p->pred, &p->cpred, p->npred, sizeof *p->pred))
    return 0;
  uint32_t i = p->npred++;
  p->pred[i] = (Pred){from, p->blocks[to].predecessor};
  p->blocks[to].predecessor = i;
  return 1;
}
static void mark_node(const RcEffectProgram *p, uint32_t id, uint32_t *map,
                      uint32_t *queue, uint32_t *n) {
  id = canonical(p, id);
  if (!map[id]) {
    map[id] = 1;
    queue[(*n)++] = id;
  }
}
static void mark_guards(const RcEffectProgram *p, uint32_t block, uint32_t *map,
                        uint32_t *queue, uint32_t *n) {
  if (!block)
    return;
  Block b = p->blocks[block];
  for (uint32_t i = 0; i < b.nguards; i++)
    mark_node(p, p->guard[b.guards + i].node, map, queue, n);
}
static int compact(RcEffectProgram *p) {
  uint32_t *map = calloc(p->nnode, sizeof *map),
           *queue = malloc((size_t)p->nnode * sizeof *queue);
  if (!map || !queue) {
    free(map);
    free(queue);
    return 0;
  }
  uint32_t count = 0;
  mark_node(p, 0, map, queue, &count);
  for (unsigned reads = 0; reads < 2; reads++)
    for (uint32_t i = 0; i < (reads ? p->nread : p->neffect); i++) {
      Effect f = reads ? p->read[i] : p->effect[i];
      mark_node(p, f.low, map, queue, &count);
      mark_node(p, f.high, map, queue, &count);
      mark_guards(p, f.block, map, queue, &count);
    }
  for (uint32_t at = 0; at < count; at++) {
    Node n = p->node[queue[at]];
    if (n.op == E_CHOICE)
      for (uint32_t i = 0; i < n.b; i++) {
        Choice c = p->choice[n.a + i];
        mark_node(p, c.node, map, queue, &count);
        if (c.condition)
          mark_node(p, c.condition, map, queue, &count);
        mark_guards(p, c.block, map, queue, &count);
      }
    else if (n.op == E_READ || n.op == E_BUILTIN)
      mark_node(p, n.a, map, queue, &count);
    else if (n.op >= E_FIRST_SPV) {
      mark_node(p, n.a, map, queue, &count);
      mark_node(p, n.b, map, queue, &count);
      mark_node(p, n.c, map, queue, &count);
    }
  }
  Node *nodes = malloc((size_t)count * sizeof *nodes);
  if (!nodes) {
    free(map);
    free(queue);
    return 0;
  }
  for (uint32_t i = 0; i < count; i++)
    map[queue[i]] = i;
  for (uint32_t i = 0; i < count; i++) {
    Node n = p->node[queue[i]];
    if (n.op == E_READ || n.op == E_BUILTIN)
      n.a = map[canonical(p, n.a)];
    else if (n.op >= E_FIRST_SPV) {
      n.a = map[canonical(p, n.a)];
      n.b = map[canonical(p, n.b)];
      n.c = map[canonical(p, n.c)];
    }
    nodes[i] = n;
  }
  for (uint32_t i = 0; i < p->nchoice; i++) {
    p->choice[i].node = map[canonical(p, p->choice[i].node)];
    if (p->choice[i].condition)
      p->choice[i].condition = map[canonical(p, p->choice[i].condition)];
  }
  for (uint32_t i = 0; i < p->nguard; i++)
    p->guard[i].node = map[canonical(p, p->guard[i].node)];
  for (unsigned reads = 0; reads < 2; reads++)
    for (uint32_t i = 0; i < (reads ? p->nread : p->neffect); i++) {
      Effect *f = reads ? &p->read[i] : &p->effect[i];
      f->low = map[canonical(p, f->low)];
      f->high = map[canonical(p, f->high)];
    }
  free(p->node);
  p->node = nodes;
  p->nnode = p->cnode = count;
  free(map);
  free(queue);
  // Runtime needs only the compact expression/guard graph and effect roots.
  // Drop all parsing, dominance and reaching-definition tables after
  // preparation instead of retaining a duplicate shader analysis working set.
#define DROP(field)                                                            \
  do {                                                                         \
    free(p->field);                                                            \
    p->field = NULL;                                                           \
  } while (0)
  DROP(definition);
  DROP(id_block);
  DROP(expression);
  DROP(label_block);
  DROP(set);
  DROP(binding);
  DROP(builtin);
  DROP(array_stride);
  DROP(block_flags);
  DROP(dom);
  DROP(reach);
  DROP(loop_writes);
  DROP(pred);
  DROP(local);
  DROP(intern);
  p->cintern = 0;
#undef DROP
  return 1;
}
#include "resource_core_effect_recipes.inc"
RcEffectProgram *rc_effect_compile(const uint32_t *w, size_t words) {
  static atomic_uint_fast64_t next_identity = 1;
  if (!w || words < 5 || words > UINT32_MAX || w[0] != 0x07230203 || !w[3] ||
      w[3] > (1u << 20))
    return NULL;
  RcEffectProgram *p = calloc(1, sizeof *p);
  if (!p)
    return NULL;
  uint_fast64_t identity =
      atomic_load_explicit(&next_identity, memory_order_relaxed);
  do {
    if (identity == UINT64_MAX) {
      free(p);
      return NULL;
    }
  } while (!atomic_compare_exchange_weak_explicit(
      &next_identity, &identity, identity + 1, memory_order_relaxed,
      memory_order_relaxed));
  p->identity = identity;
  p->w = w;
  p->words = words;
  p->bound = w[3];
  p->supported = 1;
  uint32_t **arrays[] = {&p->definition,  &p->id_block,     &p->expression,
                         &p->label_block, &p->set,          &p->binding,
                         &p->builtin,     &p->array_stride, &p->block_flags};
  for (unsigned i = 0; i < sizeof arrays / sizeof arrays[0]; i++)
    if (!(*arrays[i] = calloc(p->bound, sizeof(uint32_t))))
      goto bad;
  memset(p->expression, 255, p->bound * 4);
  memset(p->binding, 255, p->bound * 4);
  memset(p->builtin, 255, p->bound * 4);
  if (!grow((void **)&p->blocks, &p->cblocks, 0, sizeof *p->blocks) ||
      !grow((void **)&p->pred, &p->cpred, 0, sizeof *p->pred))
    goto bad;
  memset(p->blocks, 0, sizeof *p->blocks);
  p->nblocks = 1;
  p->npred = 1;
  if (node_new(p, (Node){E_UNKNOWN, 0, 0, 0, 32, 0}) != 0)
    goto bad;
  uint32_t block = 0, functions = 0, entries = 0, compute_entries = 0, returns = 0;
  int ends_in_return = 0, unresolved_local = 0;
  uint32_t compute_entry = 0, local_entry = 0, local[3] = {0}, workgroup_size = 0;
  uint32_t previous = 0;
  for (size_t i = 5; i < words;) {
    const uint32_t *d = w + i;
    uint32_t len = length(d), op = opcode(d), id = 0;
    if (!len || len > words - i || !instruction_length(op, len))
      goto bad;
    if ((op == 21 || op == 22) && d[2] != 32 && d[2] != 64)
      goto unsupported;
    if (!no_result(op) &&
        len < (((op >= 19 && op <= 39) || op == 248 || op == 7 || op == 11)
                   ? 2u
                   : 3u))
      goto bad;
    if (op == 54)
      functions++;
    if (op == 15) {
      entries++;
      if (len >= 4 && d[1] == 5) {
        compute_entries++;
        compute_entry = d[2];
      }
    }
    if (op == 16 && len >= 3 && d[2] == 17) { // LocalSize
      if (len != 6 || local_entry) unresolved_local = 1;
      else {
        local_entry = d[1];
        memcpy(local, d + 3, sizeof local);
      }
    }
    // ID-based execution modes and grouped decorations may change the effective local size.
    if (op == 331 || (op >= 73 && op <= 75)) unresolved_local = 1;
    if (op == 253) returns++;
    if (op == 56) ends_in_return = previous == 253;
    previous = op;
    if (op == 248) {
      if (!grow((void **)&p->blocks, &p->cblocks, p->nblocks,
                sizeof *p->blocks))
        goto bad;
      if (block)
        p->blocks[block].end = (uint32_t)i;
      block = p->nblocks++;
      p->blocks[block] = (Block){.start = (uint32_t)i};
      if (len != 2 || d[1] >= p->bound)
        goto bad;
      p->label_block[d[1]] = block;
    }
    if (!no_result(op))
      id = (op >= 19 && op <= 39) || op == 248 || op == 7 || op == 11
               ? d[1]
               : (len >= 3 ? d[2] : 0);
    if (id) {
      if (id >= p->bound || p->definition[id])
        goto bad;
      p->definition[id] = (uint32_t)i;
      p->id_block[id] = block;
    }
    if (op == 71 && len >= 3 && d[1] < p->bound) {
      if (d[2] == 33 && len == 4)
        p->binding[d[1]] = d[3];
      if (d[2] == 34 && len == 4)
        p->set[d[1]] = d[3];
      if (d[2] == 11 && len == 4) {
        p->builtin[d[1]] = d[3];
        if (d[3] == 25) { // WorkgroupSize may override the literal execution mode.
          unresolved_local |= workgroup_size != 0;
          workgroup_size = d[1];
        }
      }
      if (d[2] == 6 && len == 4)
        p->array_stride[d[1]] = d[3];
      if (d[2] == 2 || d[2] == 3)
        p->block_flags[d[1]] = d[2];
    }
    i += len;
  }
  if (block)
    p->blocks[block].end = (uint32_t)words;
  if (functions != 1)
    goto unsupported;
  p->definite_stores = p->nblocks == 2 && entries == 1 && compute_entries == 1 && returns == 1 && ends_in_return;
  if (p->definite_stores && !unresolved_local && compute_entry && compute_entry == local_entry &&
      local[0] && local[1] && local[2]) {
    const uint32_t *size = def(p, workgroup_size);
    if (!workgroup_size || (opcode(size) == 44 && length(size) == 6 &&
        constant(p, size[3], local[0]) && constant(p, size[4], local[1]) && constant(p, size[5], local[2])))
      memcpy(p->affine_local, local, sizeof local);
  }
  for (uint32_t b = 1; b < p->nblocks; b++)
    for (uint32_t i = p->blocks[b].start; i < p->blocks[b].end;
         i += w[i] >> 16) {
      const uint32_t *d = w + i;
      uint32_t op = opcode(d);
      if (op == 247 || op == 246) {
        if (d[1] >= p->bound)
          goto bad;
        p->blocks[b].merge = p->label_block[d[1]];
      }
      if (op == 249 || op == 250) {
        uint32_t yes = d[op == 249 ? 1 : 2];
        if (yes >= p->bound)
          goto bad;
        p->blocks[b].yes = p->label_block[yes];
        if (!edge(p, b, p->blocks[b].yes))
          goto bad;
        if (op == 250) {
          if (d[3] >= p->bound)
            goto bad;
          p->blocks[b].condition = d[1];
          p->blocks[b].no = p->label_block[d[3]];
          if (!edge(p, b, p->blocks[b].no))
            goto bad;
        }
      }
      if (op == 251)
        goto unsupported;
    }
  p->dom_words = (p->nblocks + 31) / 32;
  p->dom = malloc((size_t)p->nblocks * p->dom_words * 4);
  if (!p->dom)
    goto bad;
  for (uint32_t b = 0; b < p->nblocks; b++)
    for (uint32_t j = 0; j < p->dom_words; j++)
      p->dom[(size_t)b * p->dom_words + j] = b == 1 ? 0 : UINT32_MAX;
  p->dom[p->dom_words] |= 2;
  for (uint32_t pass = 0; pass < p->nblocks; pass++) {
    int changed = 0;
    for (uint32_t b = 2; b < p->nblocks; b++)
      for (uint32_t j = 0; j < p->dom_words; j++) {
        uint32_t v = UINT32_MAX;
        for (uint32_t q = p->blocks[b].predecessor; q; q = p->pred[q].next)
          v &= p->dom[(size_t)p->pred[q].block * p->dom_words + j];
        if (!p->blocks[b].predecessor)
          v = 0;
        if (j == b / 32)
          v |= 1u << (b % 32);
        uint32_t *old = &p->dom[(size_t)b * p->dom_words + j];
        changed |= *old != v;
        *old = v;
      }
    if (!changed)
      break;
  }
  // A static instruction in a loop executes more than once. Its condition is
  // not a fact about another iteration. Widen loop-carried values instead of
  // combining incompatible branch facts from separate iterations.
  p->loop_words = (p->bound + 31) / 32;
  p->loop_writes = calloc((size_t)p->nblocks * p->loop_words, 4);
  if (!p->loop_writes)
    goto bad;
  uint32_t *todo = malloc((size_t)p->nblocks * 4);
  uint8_t *seen = calloc(p->nblocks, 1);
  if (!todo || !seen) {
    free(todo);
    free(seen);
    goto bad;
  }
  for (uint32_t a = 1; a < p->nblocks; a++)
    for (unsigned k = 0; k < 2; k++) {
      uint32_t head = k ? p->blocks[a].no : p->blocks[a].yes;
      if (!head || !dominates(p, head, a))
        continue;
      memset(seen, 0, p->nblocks);
      uint32_t n = 0;
      seen[head] = 1;
      seen[a] = 1;
      todo[n++] = a;
      p->blocks[head].loop = 1;
      while (n) {
        uint32_t b = todo[--n];
        p->blocks[b].loop = 1;
        for (uint32_t q = p->blocks[b].predecessor; q; q = p->pred[q].next) {
          uint32_t pred = p->pred[q].block;
          if (!seen[pred]) {
            seen[pred] = 1;
            todo[n++] = pred;
          }
        }
      }
      for (uint32_t b = 1; b < p->nblocks; b++)
        if (seen[b])
          for (uint32_t i = p->blocks[b].start; i < p->blocks[b].end;
               i += w[i] >> 16)
            if ((w[i] & 65535) == 62 && w[i + 1] < p->bound)
              p->loop_writes[(size_t)head * p->loop_words + w[i + 1] / 32] |=
                  1u << (w[i + 1] % 32);
    }
  free(todo);
  free(seen);
  p->reach = calloc((size_t)p->nblocks * p->dom_words, 4);
  if (!p->reach)
    goto bad;
  for (uint32_t b = 1; b < p->nblocks; b++)
    p->reach[(size_t)b * p->dom_words + b / 32] |= 1u << (b % 32);
  for (uint32_t pass = 0; pass < p->nblocks; pass++) {
    int changed = 0;
    for (uint32_t b = p->nblocks - 1; b; b--)
      for (unsigned k = 0; k < 2; k++) {
        uint32_t next = k ? p->blocks[b].no : p->blocks[b].yes;
        if (!next || dominates(p, next, b))
          continue;
        for (uint32_t j = 0; j < p->dom_words; j++) {
          uint32_t *a = &p->reach[(size_t)b * p->dom_words + j],
                   v = *a | p->reach[(size_t)next * p->dom_words + j];
          changed |= v != *a;
          *a = v;
        }
      }
    if (!changed)
      break;
  }
  // Compile conditions before addresses; expression interning makes repeated
  // loads/copies share guard identities.
  for (uint32_t b = 1; b < p->nblocks; b++)
    if (p->blocks[b].condition)
      expr(p, p->blocks[b].condition);
  for (uint32_t b = 1; b < p->nblocks; b++) {
    p->blocks[b].guards = p->nguard;
    for (uint32_t a = 1; a < p->nblocks; a++)
      if (p->blocks[a].condition && dominates(p, a, b)) {
        int yes =
                !!(p->reach[(size_t)p->blocks[a].yes * p->dom_words + b / 32] &
                   (1u << (b % 32))),
            no = !!(p->reach[(size_t)p->blocks[a].no * p->dom_words + b / 32] &
                    (1u << (b % 32)));
        if (yes != no &&
            !dominates(p, yes ? p->blocks[a].yes : p->blocks[a].no, a)) {
          if (!grow((void **)&p->guard, &p->cguard, p->nguard,
                    sizeof *p->guard))
            goto bad;
          p->guard[p->nguard++] =
              (Fact){expr(p, p->blocks[a].condition), (uint8_t)yes};
          p->blocks[b].nguards++;
        }
      }
  }
  for (size_t i = 5; i < words; i += w[i] >> 16) {
    const uint32_t *d = w + i;
    uint32_t op = opcode(d), ptr = 0, bytes = 4;
    if (op == 68)
      p->reads_unknown = 1; // Shortening a descriptor would change ArrayLength
                            // even with bounded loads.
    if (op == 61 || op == 227) {
      const uint32_t *pd = def(p, d[3]), *pt = pd ? def(p, pd[1]) : NULL;
      if (opcode(pt) == 32 && (pt[2] == 2 || pt[2] == 12)) {
        Pointer pi;
        if (!pointer(p, d[3], &pi, 0))
          p->reads_unknown = 1;
        else if (pi.buffer) {
          uint32_t nb = scalar_size(p, pi.pointee);
          if (!nb)
            p->reads_unknown = 1;
          else {
            if (!grow((void **)&p->read, &p->cread, p->nread, sizeof *p->read))
              goto bad;
            p->read[p->nread++] =
                (Effect){RC_EFFECT_SSBO,   pi.set, pi.binding, pi.offset, 0, nb,
                         p->id_block[d[2]], 0, 0, 0, 0};
          }
        }
      }
    }
    if (op == 62 || op == 228 || op == 319)
      ptr = d[1];
    else if ((op >= 229 && op <= 242) || op == 318 || op == 5614 ||
             op == 5615 || op == 6035)
      ptr = d[3];
    else if (op == 63 || op == 64 || op == 57 ||
             (op >= 4096 && !rc_shader_extension_no_guest_write(op, length(d))))
      goto unsupported;
    if (!ptr)
      continue;
    const uint32_t *pd = def(p, ptr), *pt = pd ? def(p, pd[1]) : NULL;
    if (opcode(pt) != 32)
      goto unsupported;
    if (pt[2] == 7 || pt[2] == 4 || pt[2] == 3 || pt[2] == 6)
      continue;
    p->definite_stores &= op == 62;
    Effect e = {.bytes = bytes, .original = p->neffect, .store_value = op == 62 ? d[2] : 0};
    e.block = pd ? p->id_block[ptr] : 0;
    // The effect's block is the store, not necessarily the pointer's definition
    // block.
    for (uint32_t b = 1; b < p->nblocks; b++)
      if (i >= p->blocks[b].start && i < p->blocks[b].end) {
        e.block = b;
        break;
      }
    if (pt[2] == 5349) {
      uint32_t low, high, table;
      if (scalar_size(p, pt[3]) != 4)
        goto unsupported;
      if (!gpa_pointer(p, ptr, &low, &high, &table, &e.gpa_twin))
        goto unsupported;
      e.kind = RC_EFFECT_GPA;
      e.set = p->set[table];
      e.binding = p->binding[table];
      e.low = expr(p, low);
      e.high = expr(p, high);
    } else {
      Pointer pi;
      if (!pointer(p, ptr, &pi, 0) || (pi.storage != 2 && pi.storage != 12))
        goto unsupported;
      e.kind = RC_EFFECT_SSBO;
      e.set = pi.set;
      e.binding = pi.binding;
      e.low = pi.offset;
      e.bytes = scalar_size(p, pi.pointee);
      if (!e.bytes)
        goto unsupported;
      if (p->definite_stores && p->affine_local[0] && pi.buffer)
        e.affine_end = affine_storage_end(p, ptr, e.bytes);
    }
    if (!grow((void **)&p->effect, &p->ceffect, p->neffect, sizeof *p->effect))
      goto bad;
    p->effect[p->neffect++] = e;
  }
#ifdef RC_EFFECT_DEBUG
  if (p->neffect) {
    Block b = p->blocks[p->effect[0].block];
    TRACE("first effect %u guards", p->effect[0].block);
    for (uint32_t i = 0; i < b.nguards; i++) {
      Fact f = p->guard[b.guards + i];
      Node n = p->node[f.node];
      TRACE(" %u:%u(op%u,a%u,b%u)", f.node, f.truth, n.op, n.a, n.b);
    }
    TRACE("\n");
  }
  for (uint32_t b = 1; b < 60 && b < p->nblocks; b++)
    if (p->blocks[b].condition)
      TRACE("B%u start%u condSpv%u node%u yes%u no%u loop%u\n", b,
            p->blocks[b].start, p->blocks[b].condition,
            expr(p, p->blocks[b].condition), p->blocks[b].yes, p->blocks[b].no,
            p->blocks[b].loop);
#endif
  if (p->definite_stores && p->neffect && !(p->neffect & 1)) {
    p->mirrored_gpa_stores = 1;
    for (uint32_t j = 0; j < p->neffect; j += 2) {
      const Effect *a = &p->effect[j], *b = a + 1;
      if (a->kind != RC_EFFECT_GPA || b->kind != RC_EFFECT_GPA || a->gpa_twin || !b->gpa_twin ||
          a->low != b->low || a->high != b->high || a->set != b->set || a->binding != b->binding ||
          a->bytes != b->bytes || a->store_value != b->store_value) {
        p->mirrored_gpa_stores = 0;
        break;
      }
    }
  }
  if (!compact(p))
    goto bad;
  if (RC_EFFECT_RECIPES) {
    prepare_recipes(p);
    if (!compact(p)) goto bad;
  }
  return p;
unsupported:
  p->supported = 0;
  return p;
bad:
  rc_effect_free(p);
  return NULL;
}
void rc_effect_free(RcEffectProgram *p) {
  if (!p)
    return;
  free(p->definition);
  free(p->id_block);
  free(p->expression);
  free(p->label_block);
  free(p->set);
  free(p->binding);
  free(p->builtin);
  free(p->array_stride);
  free(p->block_flags);
  free(p->dom);
  free(p->reach);
  free(p->loop_writes);
  free(p->blocks);
  free(p->pred);
  free(p->node);
  free(p->choice);
  free(p->guard);
  free(p->effect);
  free(p->read);
  free(p->local);
  free(p->intern);
  free(p);
}
size_t rc_effect_workspace_size(const RcEffectProgram *p) {
  return p ? (size_t)p->nnode * sizeof(Cache) : 0;
}
size_t rc_effect_write_count(const RcEffectProgram *p) {
  return p ? (p->recipes ? p->original_effects : p->neffect) : 0;
}
int rc_effect_supported(const RcEffectProgram *p) { return p && p->supported; }

static Interval full(uint32_t bits) { return (Interval){0, mask_bits(bits)}; }
static Interval empty(void) { return (Interval){1, 0}; }
static int absent(Interval a) { return a.lo > a.hi; }
static Interval join(Interval a, Interval b) {
  if (absent(a))
    return b;
  if (absent(b))
    return a;
  return (Interval){a.lo < b.lo ? a.lo : b.lo, a.hi > b.hi ? a.hi : b.hi};
}
static Interval eval(Eval *e, uint32_t id);
static int fact(Eval *e, uint32_t id, int truth) {
  id = canonical(e->p, id);
  Node n = e->p->node[id];
  if (!id)
    return 1;
  if (n.op == E_FIRST_SPV + 168)
    return fact(e, n.a, !truth);
  if ((n.op == E_FIRST_SPV + 167 && truth) ||
      (n.op == E_FIRST_SPV + 166 && !truth))
    return fact(e, n.a, truth) && fact(e, n.b, truth);
  for (uint32_t i = 0; i < e->nfacts; i++)
    if (e->fact[i].node == id) {
      if (e->fact[i].truth != truth)
        TRACE("contradiction node %u op %u old %u new %d\n", id, n.op,
              e->fact[i].truth, truth);
      return e->fact[i].truth == truth;
    }
  if (e->nfacts == 256) {
    e->exhausted = 1;
    return 0;
  }
  Interval v = eval(e, id);
  if (absent(v) || (v.lo == v.hi && !!v.lo != truth)) {
    TRACE("guard excludes node %u op %u range %llu..%llu truth %d\n", id, n.op,
          (unsigned long long)v.lo, (unsigned long long)v.hi, truth);
    return 0;
  }
  if (e->last_unstable)
    return 1; // Separate uninitialized reads are not one reusable value or
              // branch fact.
  RuntimeFact f = {.node = id, .truth = (uint8_t)truth};
  if (n.op >= E_FIRST_SPV + 170 && n.op <= E_FIRST_SPV + 179) {
    Interval a = eval(e, n.a);
    f.a_exact = !e->last_unstable && a.lo == a.hi;
    f.a = a.lo;
    Interval b = eval(e, n.b);
    f.b_exact = !e->last_unstable && b.lo == b.hi;
    f.b = b.lo;
  }
  e->fact[e->nfacts++] = f;
  e->context = ++e->next_context;
  return 1;
}
static int block_facts(Eval *e, uint32_t block) {
  if (!block)
    return 1;
  Block b = e->p->blocks[block];
  for (uint32_t i = 0; i < b.nguards; i++) {
    Fact f = e->p->guard[b.guards + i];
    if (!fact(e, f.node, f.truth))
      return 0;
  }
  return 1;
}
static Interval constrained(Eval *e, uint32_t id, Interval v,
                            uint32_t unstable) {
  if (!id || absent(v) || unstable)
    return v;
  for (uint32_t i = 0; i < e->nfacts; i++) {
    RuntimeFact f = e->fact[i];
    if (f.node == id)
      return (Interval){f.truth, f.truth};
    Node n = e->p->node[f.node];
    uint32_t op = n.op - E_FIRST_SPV;
    if (op < 170 || op > 179)
      continue;
    uint32_t a = canonical(e->p, n.a), b = canonical(e->p, n.b);
    if (a != id && b != id)
      continue;
    uint32_t other = a == id ? b : a;
    uint64_t c;
    if (e->p->node[other].op == E_CONSTANT)
      c = e->p->node[other].value;
    else if (a == id && f.b_exact)
      c = f.b;
    else if (b == id && f.a_exact)
      c = f.a;
    else
      continue;
    if (a != id) {
      if (op == 172)
        op = 176;
      else if (op == 176)
        op = 172;
      else if (op == 174)
        op = 178;
      else if (op == 178)
        op = 174;
      else if (op == 173)
        op = 177;
      else if (op == 177)
        op = 173;
      else if (op == 175)
        op = 179;
      else if (op == 179)
        op = 175;
    }
    if (!f.truth) {
      if (op == 170)
        op = 171;
      else if (op == 171)
        op = 170;
      else if (op == 172)
        op = 178;
      else if (op == 174)
        op = 176;
      else if (op == 176)
        op = 174;
      else if (op == 178)
        op = 172;
      else if (op == 173)
        op = 179;
      else if (op == 175)
        op = 177;
      else if (op == 177)
        op = 175;
      else if (op == 179)
        op = 173;
    }
    // Signed comparisons only narrow when both the bound and candidate occupy
    // the same sign half.
    if (op == 173 || op == 175 || op == 177 || op == 179) {
      uint64_t sign = UINT64_C(1) << (e->p->node[id].bits - 1);
      if ((v.lo & sign) != (v.hi & sign) || (v.lo & sign) != (c & sign))
        continue;
      op--;
    }
    if (op == 170) {
      if (c < v.lo || c > v.hi)
        return empty();
      v = (Interval){c, c};
    } else if (op == 172) {
      if (c == UINT64_MAX)
        return empty();
      if (v.lo < c + 1)
        v.lo = c + 1;
    } else if (op == 174) {
      if (v.lo < c)
        v.lo = c;
    } else if (op == 176) {
      if (!c)
        return empty();
      if (v.hi > c - 1)
        v.hi = c - 1;
    } else if (op == 178) {
      if (v.hi > c)
        v.hi = c;
    } else if (op == 171 && v.lo == v.hi && v.lo == c)
      return empty();
  }
  return v;
}
static Interval arithmetic(uint32_t op, Interval a, Interval b, uint32_t bits) {
  if (absent(a) || absent(b))
    return empty();
  uint64_t mask = mask_bits(bits), lo = 0, hi = mask;
  if (a.lo == a.hi && b.lo == b.hi) {
    uint64_t x = a.lo, y = b.lo;
    switch (op) {
    case 128:
      lo = x + y;
      break;
    case 130:
      lo = x - y;
      break;
    case 132:
      lo = x * y;
      break;
    case 134:
      if (!y)
        return full(bits);
      lo = x / y;
      break;
    case 137:
      if (!y)
        return full(bits);
      lo = x % y;
      break;
    case 194:
      if (y >= bits)
        return full(bits);
      lo = x >> y;
      break;
    case 196:
      if (y >= bits)
        return full(bits);
      lo = x << y;
      break;
    case 197:
      lo = x | y;
      break;
    case 198:
      lo = x ^ y;
      break;
    case 199:
      lo = x & y;
      break;
    default:
      goto interval;
    }
    return (Interval){lo & mask, lo & mask};
  }
interval:
  switch (op) {
  case 128: {
    __uint128_t l = (__uint128_t)a.lo + b.lo, h = (__uint128_t)a.hi + b.hi;
    if ((l >> bits) == (h >> bits))
      return (Interval){(uint64_t)l & mask, (uint64_t)h & mask};
    break;
  }
  case 130:
    if (a.lo >= b.hi)
      return (Interval){a.lo - b.hi, a.hi - b.lo};
    if (a.hi < b.lo)
      return (Interval){(a.lo - b.hi) & mask, (a.hi - b.lo) & mask};
    break;
  case 132:
    if ((__uint128_t)a.hi * b.hi <= mask)
      return (Interval){a.lo * b.lo, a.hi * b.hi};
    break;
  case 151:
    return (Interval){(uint64_t)(((__uint128_t)a.lo * b.lo) >> bits),
                      (uint64_t)(((__uint128_t)a.hi * b.hi) >> bits)};
  case 149:
    return (Interval){(uint64_t)(((__uint128_t)a.lo + b.lo) >> bits),
                      (uint64_t)(((__uint128_t)a.hi + b.hi) >> bits)};
  case 150:
    return a.hi < b.lo    ? (Interval){1, 1}
           : a.lo >= b.hi ? (Interval){0, 0}
                          : full(1);
  case 134:
    if (b.lo)
      return (Interval){a.lo / b.hi, a.hi / b.lo};
    break;
  case 137:
    if (b.lo)
      return (Interval){0, b.hi - 1 < a.hi ? b.hi - 1 : a.hi};
    break;
  case 194:
    if (b.hi < bits)
      return (Interval){a.lo >> b.hi, a.hi >> b.lo};
    break;
  case 195:
    if (b.lo == b.hi && b.lo < bits) {
      uint64_t sign = UINT64_C(1) << (bits - 1);
      if (a.hi < sign)
        return (Interval){a.lo >> b.lo, a.hi >> b.lo};
      if (a.lo >= sign) {
        uint64_t fill = b.lo ? mask ^ (mask >> b.lo) : 0;
        return (Interval){(a.lo >> b.lo) | fill, (a.hi >> b.lo) | fill};
      }
    }
    break;
  case 196:
    if (b.hi < bits && a.hi <= (mask >> b.hi))
      return (Interval){a.lo << b.lo, a.hi << b.hi};
    break;
  case 199:
    hi = a.hi < b.hi ? a.hi : b.hi;
    break;
  case 197:
  case 198: {
    uint64_t m = a.hi | b.hi;
    m |= m >> 1;
    m |= m >> 2;
    m |= m >> 4;
    m |= m >> 8;
    m |= m >> 16;
    m |= m >> 32;
    hi = m & mask;
    break;
  }
  default:
    break;
  }
  return (Interval){lo, hi};
}
static Interval compare(uint32_t op, Interval a, Interval b, uint32_t bits) {
  if (absent(a) || absent(b))
    return empty();
  if (op == 164)
    op = 170;
  if (op == 165)
    op = 171;
  if (op == 166) {
    if (a.lo || b.lo)
      return (Interval){1, 1};
    if (!a.hi && !b.hi)
      return (Interval){0, 0};
    return full(1);
  }
  if (op == 167) {
    if (!a.hi || !b.hi)
      return (Interval){0, 0};
    if (a.lo && b.lo)
      return (Interval){1, 1};
    return full(1);
  }
  if (op == 173 || op == 175 || op == 177 || op == 179) {
    uint64_t sign = UINT64_C(1) << (bits - 1);
    if ((a.lo & sign) != (a.hi & sign) || (b.lo & sign) != (b.hi & sign))
      return full(1);
    a.lo ^= sign;
    a.hi ^= sign;
    b.lo ^= sign;
    b.hi ^= sign;
    op--;
  }
  int yes = 0, no = 0;
  switch (op) {
  case 170:
    yes = a.lo == a.hi && b.lo == b.hi && a.lo == b.lo;
    no = a.hi < b.lo || b.hi < a.lo;
    break;
  case 171:
    no = a.lo == a.hi && b.lo == b.hi && a.lo == b.lo;
    yes = a.hi < b.lo || b.hi < a.lo;
    break;
  case 172:
    yes = a.lo > b.hi;
    no = a.hi <= b.lo;
    break;
  case 174:
    yes = a.lo >= b.hi;
    no = a.hi < b.lo;
    break;
  case 176:
    yes = a.hi < b.lo;
    no = a.lo >= b.hi;
    break;
  case 178:
    yes = a.hi <= b.lo;
    no = a.lo > b.hi;
    break;
  default:
    break;
  }
  return yes ? (Interval){1, 1} : no ? (Interval){0, 0} : full(1);
}
typedef struct {
  uint64_t value;
  unsigned bits;
} Congruence;
static unsigned trailing_zero(uint64_t value, unsigned cap) {
  unsigned n = 0;
  while (n < cap && !(value & (UINT64_C(1) << n)))
    n++;
  return n;
}
static Congruence congruence_at(const RcEffectProgram *p, const Eval *e,
                                uint32_t id, unsigned depth) {
  Congruence unknown = {0, 0};
  if (depth > 64)
    return unknown;
  id = canonical(p, id);
  Node n = p->node[id];
  if (e && e->cache[id].context == e->context &&
      e->cache[id].value.lo == e->cache[id].value.hi)
    return (Congruence){e->cache[id].value.lo, n.bits};
  if (n.op == E_CONSTANT)
    return (Congruence){n.value, n.bits};
  if (n.op == E_CHOICE) {
    Congruence r = {0, 64};
    for (uint32_t i = 0; i < n.b; i++) {
      Congruence x = congruence_at(p, e, p->choice[n.a + i].node, depth + 1);
      if (!i)
        r = x;
      else {
        if (r.bits > x.bits)
          r.bits = x.bits;
        r.bits = trailing_zero(r.value ^ x.value, r.bits);
      }
    }
    return n.b ? r : unknown;
  }
  if (n.op < E_FIRST_SPV)
    return unknown;
  uint32_t op = n.op - E_FIRST_SPV;
  Congruence a = congruence_at(p, e, n.a, depth + 1), b = {0, 0}, r = {0, 0};
  if (op == 113 || op == 114 || op == 124 || op == 83) {
    r = a;
    goto done;
  }
  if (op == 126) {
    r = (Congruence){0 - a.value, a.bits};
    goto done;
  }
  if (op == 200) {
    r = (Congruence){~a.value, a.bits};
    goto done;
  }
  b = congruence_at(p, e, n.b, depth + 1);
  if (op == 128 || op == 130)
    r = (Congruence){op == 128 ? a.value + b.value : a.value - b.value,
                     a.bits < b.bits ? a.bits : b.bits};
  else if (op == 132) {
    if (b.bits == p->node[n.b].bits)
      r = (Congruence){a.value * b.value,
                       a.bits + trailing_zero(b.value, n.bits)};
    else if (a.bits == p->node[n.a].bits)
      r = (Congruence){a.value * b.value,
                       b.bits + trailing_zero(a.value, n.bits)};
    else
      r = (Congruence){0, trailing_zero(a.value, a.bits) +
                              trailing_zero(b.value, b.bits)};
  } else if ((op == 194 || op == 196) && b.bits == p->node[n.b].bits &&
             b.value < n.bits) {
    if (op == 194)
      r = (Congruence){a.value >> b.value,
                       a.bits > b.value ? a.bits - (unsigned)b.value : 0};
    else
      r = (Congruence){a.value << b.value, a.bits + (unsigned)b.value};
  } else if ((op == 199 || op == 197) && b.bits == p->node[n.b].bits) {
    r.value = op == 199 ? a.value & b.value : a.value | b.value;
    r.bits = a.bits;
    while (r.bits < n.bits &&
           !!(b.value & (UINT64_C(1) << r.bits)) == (op == 197))
      r.bits++;
  } else if (op == 198)
    r = (Congruence){a.value ^ b.value, a.bits < b.bits ? a.bits : b.bits};
done:
  if (r.bits > n.bits)
    r.bits = n.bits;
  return r;
}
#ifdef RC_EFFECT_DEBUG
void rc_effect_debug_dump(const RcEffectProgram *p, uint32_t id,
                          unsigned depth) {
  if (!p || id >= p->nnode || !depth)
    return;
  Node n = p->node[id];
  TRACE("%*s%u op%u a%u b%u c%u value%llu\n", (int)(12 - depth), "", id, n.op,
        n.a, n.b, n.c, (unsigned long long)n.value);
  if (n.op == E_CHOICE) {
    for (uint32_t i = 0; i < n.b; i++)
      rc_effect_debug_dump(p, p->choice[n.a + i].node, depth - 1);
  } else if (n.op >= E_FIRST_SPV) {
    rc_effect_debug_dump(p, n.a, depth - 1);
    if (n.b)
      rc_effect_debug_dump(p, n.b, depth - 1);
    if (n.c)
      rc_effect_debug_dump(p, n.c, depth - 1);
  } else if (n.op == E_READ || n.op == E_BUILTIN)
    rc_effect_debug_dump(p, n.a, depth - 1);
}
#endif
static Interval eval(Eval *e, uint32_t id) {
  id = canonical(e->p, id);
  Node n = e->p->node[id];
  Interval r = full(n.bits);
  uint32_t unstable = 0;
  ++e->ticks;
  ++e->depth;
  if (e->ticks > RC_EFFECT_STEP_LIMIT || e->depth > 256) {
    e->exhausted = 1;
    e->depth--;
    e->last_unstable = 1;
    return r;
  }
  Cache *memo = &e->cache[id];
  uint64_t context = e->context;
  if (memo->context == context) {
    e->depth--;
    e->last_unstable = memo->unstable;
    return memo->value;
  }
  // A recurrence can only widen the result. A dominating later guard can still
  // narrow its unknown integer.
  if (memo->visiting) {
    e->depth--;
    e->last_unstable = 1;
    return r;
  }
  memo->visiting = 1;
  for (uint32_t i = 0; i < e->nfacts; i++)
    if (e->fact[i].node == id) {
      r = (Interval){e->fact[i].truth, e->fact[i].truth};
      goto done;
    }
  if (n.op == E_CONSTANT)
    r = (Interval){n.value, n.value};
  else if (n.op == E_UNKNOWN) {
    unstable = n.c == 2;
    if (unstable)
      TRACE("uninitialized variable%u block%u node%u\n", n.a, n.b, id);
  } else if (n.op == E_CHOICE) {
    r = empty();
    for (uint32_t i = 0; i < n.b; i++) {
      Choice c = e->p->choice[n.a + i];
      uint32_t save = e->nfacts;
      uint64_t ctx = e->context;
      if (block_facts(e, c.block) &&
          (!c.condition || fact(e, c.condition, c.truth))) {
        Interval x = eval(e, c.node);
        if (!absent(x))
          unstable |= e->last_unstable;
        r = join(r, x);
      }
      e->nfacts = save;
      e->context = ctx;
    }
  } else if (n.op == E_READ) {
    Interval off = eval(e, n.a);
    unstable = e->last_unstable;
    r = empty();
    Congruence cg = congruence_at(e->p, e, n.a, 0);
    unsigned shift = cg.bits > 32 ? 32 : cg.bits;
    uint64_t step = UINT64_C(1) << shift;
    if (absent(off))
      goto done;
    if (!e->in->read || off.hi - off.lo > 16384 ||
        (off.hi - off.lo) / step > 4096 || off.hi > UINT64_MAX - n.bits / 8) {
      TRACE("read unbounded node%u binding%u offsetnode%u range%llu..%llu\n",
            id, n.c, n.a, (unsigned long long)off.lo,
            (unsigned long long)off.hi);
      r = full(n.bits);
      goto done;
    }
    uint64_t start = off.lo + ((cg.value - off.lo) & (step - 1));
    for (uint64_t at = start; at <= off.hi;) {
      uint64_t v = 0;
      if (!e->in->read(e->in->context, n.b, n.c, at, n.bits / 8, &v)) {
        r = full(n.bits);
        break;
      }
      r = join(r, (Interval){v, v});
      if (off.hi - at < step)
        break;
      at += step;
    }
  } else if (n.op == E_BUILTIN) {
    Interval component = eval(e, n.a);
    unstable = e->last_unstable;
    if (component.lo == component.hi && component.lo % 4 == 0 &&
        e->in->builtin) {
      uint64_t lo, hi;
      if (e->in->builtin(e->in->context, n.b, (uint32_t)(component.lo / 4), &lo,
                         &hi) &&
          lo <= hi && hi <= mask_bits(n.bits)) {
        r = (Interval){lo, hi};
        goto done;
      }
    }
    if (component.lo == component.hi && component.lo % 4 == 0 &&
        component.lo / 4 < 3 && e->in->direct_dispatch) {
      uint32_t axis = (uint32_t)(component.lo / 4);
      uint64_t count = n.b == 26   ? e->in->groups[axis]
                       : n.b == 27 ? e->in->local[axis]
                       : n.b == 28
                           ? (uint64_t)e->in->groups[axis] * e->in->local[axis]
                           : 0;
      if (count && count <= UINT32_MAX)
        r = (Interval){0, count - 1};
    }
  } else if (n.op >= E_FIRST_SPV) {
    uint32_t op = n.op - E_FIRST_SPV;
    Interval a = eval(e, n.a);
    unstable = e->last_unstable;
    if (op == 169) {
      Interval condition = a;
      r = empty();
      for (int truth = 0; truth <= 1; truth++)
        if (truth ? condition.hi : !condition.lo) {
          uint32_t save = e->nfacts;
          uint64_t ctx = e->context;
          if (fact(e, n.a, truth)) {
            Interval x = eval(e, truth ? n.b : n.c);
            if (!absent(x))
              unstable |= e->last_unstable;
            r = join(r, x);
          }
          e->nfacts = save;
          e->context = ctx;
        }
    } else if (op == 83 || op == 124 || op == 113 || op == 114) {
      r = a;
      if (a.hi > mask_bits(n.bits))
        r = full(n.bits);
      if (op == 114 && n.bits > e->p->node[n.a].bits) {
        uint64_t sign = UINT64_C(1) << (e->p->node[n.a].bits - 1);
        if (a.lo >= sign) {
          uint64_t fill = mask_bits(n.bits) ^ mask_bits(e->p->node[n.a].bits);
          r = (Interval){a.lo | fill, a.hi | fill};
        } else if (a.hi >= sign)
          r = full(n.bits);
      }
    } else if (op == 168)
      r = absent(a) ? empty() : (Interval){!a.hi, !a.lo};
    else if (op == 204)
      r = absent(a) ? empty() : full(n.bits);
    else if (op == 205)
      r = absent(a) ? empty() : (Interval){0, e->p->node[n.a].bits};
    else if (op == 126)
      r = arithmetic(130, (Interval){0, 0}, a, n.bits);
    else if (op == 200)
      r = absent(a)
              ? empty()
              : (Interval){mask_bits(n.bits) ^ a.hi, mask_bits(n.bits) ^ a.lo};
    else {
      Interval b = eval(e, n.b);
      unstable |= e->last_unstable;
      // Undefined integer operations cannot establish one reusable value merely because their operands are defined.
      if (!absent(a) && !absent(b)) {
        if ((op == 134 || op == 135 || op == 137 || op == 138 || op == 139) && !b.lo)
          unstable = 1;
        if ((op == 135 || op == 138 || op == 139) &&
            a.lo <= (UINT64_C(1) << (n.bits - 1)) &&
            a.hi >= (UINT64_C(1) << (n.bits - 1)) && b.hi == mask_bits(n.bits))
          unstable = 1;
        if ((op == 194 || op == 195 || op == 196) && b.hi >= n.bits)
          unstable = 1;
      }
      r = op >= 164 && op <= 179 ? compare(op, a, b, e->p->node[n.a].bits)
                                 : arithmetic(op, a, b, n.bits);
    }
  }
done:
  r = constrained(e, id, r, unstable);
  memo->visiting = 0;
  memo->context = context;
  memo->value = r;
  memo->unstable = unstable;
  e->last_unstable = unstable;
  e->depth--;
  return r;
}
static RcEffectResult evaluate_effects(const RcEffectProgram *p,
                                       const RcEffectInputs *in,
                                       RcEffectRange *ranges, size_t cap,
                                       size_t *count, int reads, uint32_t set,
                                       uint32_t binding, int definite) {
  if (count)
    *count = 0;
  if (!count || !p || !p->supported || (reads && p->reads_unknown) || !in ||
      !in->workspace || in->workspace_bytes < rc_effect_workspace_size(p) ||
      (!ranges && cap))
    return RC_EFFECT_UNKNOWN;
  if (in->direct_dispatch &&
      (!in->groups[0] || !in->groups[1] || !in->groups[2]))
    return RC_EFFECT_COMPLETE;
  memset(in->workspace, 0, rc_effect_workspace_size(p));
  Eval e = {.p = p,
            .in = in,
            .cache = in->workspace,
            .context = 1,
            .next_context = 1};
  size_t nout = 0;
  uint32_t seen = 0;
  uint32_t total = reads ? p->nread : p->neffect;
  for (uint32_t i = 0; i < total; i++) {
      Effect f = reads ? p->read[i] : p->effect[i];
      if (reads && (f.set != set || f.binding != binding)) continue;
      e.nfacts=0;e.context=++e.next_context;
      if (definite) {
        // An optimized recipe may split a store among speculative paths. Only an unguarded
        // singleton expression for each original store proves that all of these bytes were written.
        if (p->blocks[f.block].nguards || f.original >= 32 || (seen & (UINT32_C(1) << f.original)))
          return RC_EFFECT_UNKNOWN;
        seen |= UINT32_C(1) << f.original;
      }
      if (!block_facts(&e, f.block)) {
        TRACE("skip effect %u block %u\n", i, f.block);
        if (e.exhausted)
          return RC_EFFECT_UNKNOWN;
        continue;
      }
      Interval lo = eval(&e, f.low);
      int stable = !e.last_unstable;
      Interval hi = f.kind == RC_EFFECT_GPA ? eval(&e, f.high) : (Interval){0, 0};
      if (definite && (!stable || e.last_unstable || lo.lo != lo.hi || hi.lo != hi.hi))
        return RC_EFFECT_UNKNOWN;
      if (e.exhausted || absent(lo) || absent(hi) || lo.hi > UINT32_MAX ||
          hi.hi > 255) {
        TRACE("unknown effect %u kind%u binding%u low[%llu,%llu] "
              "high[%llu,%llu] ticks%u depth%u exhausted%d\n",
              i, f.kind, f.binding, (unsigned long long)lo.lo,
              (unsigned long long)lo.hi, (unsigned long long)hi.lo,
              (unsigned long long)hi.hi, e.ticks, e.depth, e.exhausted);
        return RC_EFFECT_UNKNOWN;
      }
      uint64_t first = (hi.lo << 32) | lo.lo, last = (hi.hi << 32) | lo.hi;
      if (f.kind == RC_EFFECT_GPA) {
        first &= ~UINT64_C(3);
        last &= ~UINT64_C(3);
      }
      if (last > UINT64_MAX - f.bytes || first > last)
        return RC_EFFECT_UNKNOWN;
      uint64_t end = last + f.bytes;
      // A complete 32-bit offset domain is not a useful subrange proof; retain
      // the descriptor authority.
      if (lo.lo == 0 && lo.hi == UINT32_MAX) {
        TRACE("full offset effect %u kind%u binding%u nodes%u,%u ticks%u\n", i,
              f.kind, f.binding, f.low, f.high, e.ticks);
        return RC_EFFECT_UNKNOWN;
      }
      RcEffectKind kind = definite && f.gpa_twin ? RC_EFFECT_GPA_TWIN : (RcEffectKind)f.kind;
      size_t k = 0;
      for (; k < nout; k++)
        if (ranges[k].kind == kind && ranges[k].set == f.set &&
            ranges[k].binding == f.binding &&
            first <= ranges[k].begin + ranges[k].size &&
            ranges[k].begin <= end) {
          uint64_t oldend = ranges[k].begin + ranges[k].size;
          if (first < ranges[k].begin)
            ranges[k].begin = first;
          if (end < oldend)
            end = oldend;
          ranges[k].size = end - ranges[k].begin;
          break;
        }
      if (k == nout) {
        if (nout == cap)
          return RC_EFFECT_OVERFLOW;
        ranges[nout++] = (RcEffectRange){kind, f.set, f.binding, first, end - first};
      }
  }
  if (definite && (uint32_t)__builtin_popcount(seen) != (p->recipes ? p->original_effects : p->neffect))
    return RC_EFFECT_UNKNOWN;
  *count = nout;
  return RC_EFFECT_COMPLETE;
}

int rc_effect_has_definite_store_plan(const RcEffectProgram *p) {
  return p && p->supported && p->definite_stores && p->neffect && p->neffect <= 32 &&
      (!p->recipes || p->original_effects <= 32);
}

int rc_effect_has_mirrored_gpa_store_plan(const RcEffectProgram *p) {
  return rc_effect_has_definite_store_plan(p) && p->mirrored_gpa_stores;
}
typedef struct {
  uint64_t first, last, stride[3];
} AffineStoreOffset;

// Every intermediate is a nonnegative affine form in the actual invocation
// coordinates. Reject wrap or narrowing at its own SPIR-V integer width.
static int affine_store_offset(const RcEffectProgram *p, uint32_t id, const uint64_t grid[3],
                               uint32_t depth, uint32_t *steps, AffineStoreOffset *out) {
  if (!id || id >= p->nnode || depth > 32 || *steps >= 512)
    return 0;
  ++*steps;
  const Node n = p->node[id];
  if (n.bits != 32 && n.bits != 64)
    return 0;
  if (n.op == E_CONSTANT) {
    *out = (AffineStoreOffset){.first = n.value, .last = n.value};
    return 1;
  }
  if (n.op == E_BUILTIN && n.b == 28 && n.bits == 32) { // GlobalInvocationId
    const Node component = p->node[n.a];
    if (component.op != E_CONSTANT || component.value % 4 || component.value / 4 >= 3)
      return 0;
    uint32_t axis = (uint32_t)(component.value / 4);
    *out = (AffineStoreOffset){.last = grid[axis] - 1};
    out->stride[axis] = grid[axis] > 1;
    return 1;
  }
  if (n.op != E_FIRST_SPV + 128 && n.op != E_FIRST_SPV + 132 && n.op != E_FIRST_SPV + 196)
    return 0;
  AffineStoreOffset a, b;
  if (!affine_store_offset(p, n.a, grid, depth + 1, steps, &a) ||
      !affine_store_offset(p, n.b, grid, depth + 1, steps, &b))
    return 0;
  uint64_t max = mask_bits(n.bits);
  if (n.op == E_FIRST_SPV + 128) { // IAdd
    if (a.last > max || b.last > max - a.last)
      return 0;
    *out = (AffineStoreOffset){.first = a.first + b.first, .last = a.last + b.last};
    for (unsigned d = 0; d < 3; d++) out->stride[d] = a.stride[d] + b.stride[d];
    return 1;
  }
  if (n.op == E_FIRST_SPV + 196) { // ShiftLeftLogical
    if (b.first != b.last || b.first >= n.bits)
      return 0;
    b.first = b.last = UINT64_C(1) << b.first;
  } else if (a.first == a.last && b.first != b.last) { // IMul
    AffineStoreOffset swap = a;
    a = b;
    b = swap;
  }
  if (b.first != b.last || (b.first && a.last > max / b.first))
    return 0;
  *out = (AffineStoreOffset){.first = a.first * b.first, .last = a.last * b.first};
  for (unsigned d = 0; d < 3; d++) out->stride[d] = a.stride[d] * b.first;
  return 1;
}

static RcEffectResult evaluate_affine_stores(const RcEffectProgram *p, const RcEffectInputs *in,
                                             RcEffectRange *ranges, size_t capacity, size_t *count) {
  uint64_t grid[3];
  for (unsigned d = 0; d < 3; d++) {
    grid[d] = (uint64_t)in->groups[d] * in->local[d];
    if (in->local[d] != p->affine_local[d] || !grid[d] || grid[d] > UINT32_MAX)
      return RC_EFFECT_UNKNOWN;
  }

  RcEffectRange parts[32];
  uint32_t seen = 0, steps = 0;
  for (uint32_t i = 0; i < p->neffect; i++) {
    const Effect *f = &p->effect[i];
    AffineStoreOffset address;
    if (f->kind != RC_EFFECT_SSBO || !f->affine_end || !f->block || p->blocks[f->block].nguards ||
        f->original >= 32 || (seen & (UINT32_C(1) << f->original)) ||
        !affine_store_offset(p, f->low, grid, 0, &steps, &address) ||
        f->bytes > f->affine_end || address.last > f->affine_end - f->bytes)
      return RC_EFFECT_UNKNOWN;
    seen |= UINT32_C(1) << f->original;

    // In ascending stride order each invocation must extend the tiled range
    // exactly once. Duplicated stores and holes cannot certify these bytes.
    uint64_t bytes = f->bytes;
    unsigned remaining = 0;
    for (unsigned d = 0; d < 3; d++) if (grid[d] > 1) remaining |= 1u << d;
    while (remaining) {
      unsigned next = 0;
      while (next < 3 && (!(remaining & (1u << next)) || address.stride[next] != bytes)) next++;
      if (next == 3 || grid[next] > UINT64_MAX / bytes)
        return RC_EFFECT_UNKNOWN;
      bytes *= grid[next];
      remaining &= ~(1u << next);
    }
    if (bytes != address.last - address.first + f->bytes)
      return RC_EFFECT_UNKNOWN;

    RcEffectRange part = {RC_EFFECT_SSBO, f->set, f->binding, address.first, bytes};
    uint32_t at = i;
    while (at && (part.set < parts[at - 1].set ||
        (part.set == parts[at - 1].set && (part.binding < parts[at - 1].binding ||
        (part.binding == parts[at - 1].binding && part.begin < parts[at - 1].begin))))) {
      parts[at] = parts[at - 1];
      at--;
    }
    parts[at] = part;
  }
  if ((uint32_t)__builtin_popcount(seen) != (p->recipes ? p->original_effects : p->neffect))
    return RC_EFFECT_UNKNOWN;

  uint32_t used = 0;
  for (uint32_t i = 0; i < p->neffect; i++) {
    RcEffectRange part = parts[i];
    RcEffectRange *prior = used ? &parts[used - 1] : NULL;
    if (prior && prior->set == part.set && prior->binding == part.binding) {
      uint64_t end = prior->begin + prior->size;
      if (end > part.begin)
        return RC_EFFECT_UNKNOWN; // Distinct stores may race across invocations.
      if (end == part.begin) {
        prior->size += part.size;
        continue;
      }
    }
    parts[used++] = part;
  }
  if (used > capacity)
    return RC_EFFECT_OVERFLOW;
  memcpy(ranges, parts, used * sizeof *ranges);
  *count = used;
  return RC_EFFECT_COMPLETE;
}

RcEffectResult rc_effect_evaluate_definite_stores(const RcEffectProgram *p, const RcEffectInputs *in,
    RcEffectRange *ranges, size_t capacity, size_t *count) {
  if (count) *count = 0;
  if (!count || !rc_effect_has_definite_store_plan(p) || !in || !in->direct_dispatch || (!ranges && capacity))
    return RC_EFFECT_UNKNOWN;
  for (unsigned d = 0; d < 3; ++d)
    if (in->groups[d] != 1 || in->local[d] != 1)
      return evaluate_affine_stores(p, in, ranges, capacity, count);
  return evaluate_effects(p, in, ranges, capacity, count, 0, 0, 0, 1);
}

RcEffectResult rc_effect_evaluate(const RcEffectProgram *p,
                                  const RcEffectInputs *in,
                                  RcEffectRange *ranges, size_t cap,
                                  size_t *count) {
  return evaluate_effects(p, in, ranges, cap, count, 0, 0, 0, 0);
}
RcEffectResult rc_effect_read_ranges(const RcEffectProgram *p,
                                     const RcEffectInputs *in, uint32_t set,
                                     uint32_t binding, RcEffectRange *ranges,
                                     size_t cap, size_t *count) {
  return evaluate_effects(p, in, ranges, cap, count, 1, set, binding, 0);
}

enum { MEMO_DEPENDENCIES = 256, MEMO_BUCKETS = 512, MEMO_RANGES = 128 };
typedef struct {
  uint32_t kind, set, binding, size, available;
  uint64_t offset, low, high;
} MemoDependency;
struct RcEffectMemo {
  uint64_t identity;
  uint32_t direct, groups[3], local[3], valid, overflow, ndeps;
  RcEffectResult result;
  size_t count;
  uint16_t buckets[MEMO_BUCKETS];
  MemoDependency deps[MEMO_DEPENDENCIES];
  RcEffectRange ranges[MEMO_RANGES];
};
typedef struct {
  RcEffectMemo *memo;
  const RcEffectInputs *input;
} MemoInput;
size_t rc_effect_memo_size(void) { return sizeof(RcEffectMemo); }
RcEffectMemo *rc_effect_memo_init(void *storage, size_t bytes) {
  if (!storage || bytes < sizeof(RcEffectMemo) ||
      (uintptr_t)storage % _Alignof(RcEffectMemo))
    return NULL;
  memset(storage, 0, sizeof(RcEffectMemo));
  return storage;
}
static unsigned dependency_bucket(MemoDependency d) {
  uint64_t h = d.offset ^ ((uint64_t)d.set << 32) ^ d.binding ^
               ((uint64_t)d.size << 17) ^ ((uint64_t)d.kind << 41);
  h ^= h >> 33;
  h *= UINT64_C(0xff51afd7ed558ccd);
  h ^= h >> 33;
  return (unsigned)h & (MEMO_BUCKETS - 1);
}
static int same_dependency(MemoDependency a, MemoDependency b) {
  return a.kind == b.kind && a.set == b.set && a.binding == b.binding &&
         a.offset == b.offset && a.size == b.size;
}
static MemoDependency dependency_read(const RcEffectInputs *in,
                                      MemoDependency d) {
  d.low = d.high = 0;
  if (d.kind)
    d.available = in->builtin &&
                  in->builtin(in->context, d.set, d.binding, &d.low, &d.high);
  else
    d.available = in->read && in->read(in->context, d.set, d.binding, d.offset,
                                       d.size, &d.low);
  d.available = !!d.available;
  return d;
}
static MemoDependency memo_dependency(MemoInput *input, MemoDependency d) {
  RcEffectMemo *m = input->memo;
  unsigned slot = dependency_bucket(d);
  while (m->buckets[slot]) {
    MemoDependency old = m->deps[m->buckets[slot] - 1];
    if (same_dependency(old, d))
      return old;
    slot = (slot + 1) & (MEMO_BUCKETS - 1);
  }
  d = dependency_read(input->input, d);
  if (m->ndeps == MEMO_DEPENDENCIES)
    m->overflow = 1;
  else {
    m->deps[m->ndeps] = d;
    m->buckets[slot] = (uint16_t)++m->ndeps;
  }
  return d;
}
static int memo_read(void *context, uint32_t set, uint32_t binding,
                     uint64_t offset, size_t bytes, void *dst) {
  MemoInput *input = context;
  if (bytes > 8) {
    input->memo->overflow = 1;
    return input->input->read &&
           input->input->read(input->input->context, set, binding, offset,
                              bytes, dst);
  }
  MemoDependency d =
      memo_dependency(input, (MemoDependency){.set = set,
                                              .binding = binding,
                                              .offset = offset,
                                              .size = (uint32_t)bytes});
  if (d.available)
    memcpy(dst, &d.low, bytes);
  return (int)d.available;
}
static int memo_builtin(void *context, uint32_t builtin, uint32_t component,
                        uint64_t *lo, uint64_t *hi) {
  MemoDependency d = memo_dependency(
      context,
      (MemoDependency){.kind = 1, .set = builtin, .binding = component});
  if (d.available) {
    *lo = d.low;
    *hi = d.high;
  }
  return (int)d.available;
}
RcEffectResult rc_effect_evaluate_memo(const RcEffectProgram *p,
                                       const RcEffectInputs *in,
                                       RcEffectMemo *m, RcEffectRange *ranges,
                                       size_t cap, size_t *count, int *reused) {
  if (reused)
    *reused = 0;
  if (count)
    *count = 0;
  if (!m || !p || !in || !count || (!ranges && cap))
    return RC_EFFECT_UNKNOWN;
  int hit = m->valid && m->identity == p->identity &&
            m->direct == in->direct_dispatch &&
            !memcmp(m->groups, in->groups, sizeof m->groups) &&
            !memcmp(m->local, in->local, sizeof m->local);
  for (uint32_t i = 0; hit && i < m->ndeps; i++) {
    MemoDependency a = m->deps[i], b = dependency_read(in, a);
    hit = a.available == b.available &&
          (!a.available || (a.low == b.low && a.high == b.high));
  }
  if (hit) {
    if (reused)
      *reused = 1;
    if (m->result == RC_EFFECT_COMPLETE) {
      if (m->count > cap)
        return RC_EFFECT_OVERFLOW;
      if (m->count)
        memcpy(ranges, m->ranges, m->count * sizeof *ranges);
      *count = m->count;
    }
    return m->result;
  }
  m->valid = m->overflow = m->ndeps = 0;
  memset(m->buckets, 0, sizeof m->buckets);
  MemoInput context = {m, in};
  RcEffectInputs wrapped = *in;
  wrapped.context = &context;
  wrapped.read = memo_read;
  wrapped.builtin = memo_builtin;
  RcEffectResult result = rc_effect_evaluate(p, &wrapped, ranges, cap, count);
  // Workspace refusal is not an input-derived proof and must not outlive a
  // later workspace allocation.
  if (!m->overflow && *count <= MEMO_RANGES && result != RC_EFFECT_OVERFLOW &&
      in->workspace && in->workspace_bytes >= rc_effect_workspace_size(p)) {
    m->identity = p->identity;
    m->direct = in->direct_dispatch;
    memcpy(m->groups, in->groups, sizeof m->groups);
    memcpy(m->local, in->local, sizeof m->local);
    m->result = result;
    m->count = *count;
    if (*count)
      memcpy(m->ranges, ranges, *count * sizeof *ranges);
    m->valid = 1;
  }
  return result;
}
