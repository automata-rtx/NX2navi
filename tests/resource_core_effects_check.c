// Host proofs use synthetic modules and values; no extracted shader or game
// bytes are embedded here.
#include "../runtime/resource_core_effects.c"
#include <assert.h>
#include <stdio.h>

typedef struct {
  uint32_t w[2048], n, next;
} Module;
static void emit(Module *m, uint32_t op, unsigned n, const uint32_t *a) {
  m->w[m->n++] = ((n + 1) << 16) | op;
  for (unsigned i = 0; i < n; i++)
    m->w[m->n++] = a[i];
}
#define OP(m, op, ...)                                                         \
  do {                                                                         \
    const uint32_t args_[] = {__VA_ARGS__};                                    \
    emit((m), (op), sizeof args_ / sizeof args_[0], args_);                    \
  } while (0)
static uint32_t id(Module *m) { return m->next++; }
static uint32_t lit(Module *m, uint32_t v) {
  uint32_t r = id(m);
  OP(m, 43, 1, r, v);
  return r;
}
static Module base(void) {
  Module m = {{0x07230203, 0x10000, 0, 1000, 0}, 5, 20};
  OP(&m, 19, 2);
  OP(&m, 21, 1, 32, 0);
  OP(&m, 20, 3);
  OP(&m, 33, 4, 2);
  OP(&m, 29, 5, 1);
  OP(&m, 30, 6, 5);
  OP(&m, 32, 7, 2, 6);
  OP(&m, 32, 8, 2, 1);
  OP(&m, 32, 9, 1, 1);
  OP(&m, 59, 7, 10, 2);
  OP(&m, 59, 9, 11, 1);
  OP(&m, 71, 5, 6, 4);
  OP(&m, 71, 6, 3);
  OP(&m, 72, 6, 0, 35, 0);
  OP(&m, 71, 10, 34, 0);
  OP(&m, 71, 10, 33, 7);
  OP(&m, 71, 11, 11, 42);
  return m;
}
static void start(Module *m, uint32_t entry) {
  OP(m, 54, 2, 12, 0, 4);
  OP(m, 248, entry);
}
static void finish(Module *m) {
  OP(m, 253);
  OP(m, 56);
  m->w[3] = m->next;
}
static void store_index(Module *m, uint32_t zero, uint32_t index) {
  uint32_t ptr = id(m);
  OP(m, 65, 8, ptr, 10, zero, index);
  OP(m, 62, ptr, zero);
}
static int builtin(void *ctx, uint32_t b, uint32_t component, uint64_t *lo,
                   uint64_t *hi) {
  (void)component;
  if (b != 42 || !ctx)
    return 0;
  *lo = 0;
  *hi = *(uint32_t *)ctx;
  return 1;
}
static RcEffectResult run(Module *m, RcEffectInputs *in, RcEffectRange *out,
                          size_t cap, size_t *n) {
  RcEffectProgram *p = rc_effect_compile(m->w, m->n);
  assert(p && rc_effect_supported(p));
  in->workspace_bytes = rc_effect_workspace_size(p);
  in->workspace = malloc(in->workspace_bytes);
  assert(in->workspace);
  RcEffectResult r = rc_effect_evaluate(p, in, out, cap, n);
  free(in->workspace);
  in->workspace = NULL;
  rc_effect_free(p);
  return r;
}
static void constant_and_builtin(void) {
  Module m = base();
  uint32_t z = lit(&m, 0), three = lit(&m, 3), entry = id(&m);
  start(&m, entry);
  store_index(&m, z, three);
  finish(&m);
  RcEffectInputs in = {0};
  RcEffectRange out[4];
  size_t n;
  assert(run(&m, &in, out, 4, &n) == RC_EFFECT_COMPLETE && n == 1 &&
         out[0].begin == 12 && out[0].size == 4);
  assert(run(&m, &in, out, 0, &n) == RC_EFFECT_OVERFLOW && n == 0);
  m = base();
  z = lit(&m, 0);
  entry = id(&m);
  start(&m, entry);
  uint32_t index = id(&m);
  OP(&m, 61, 1, index, 11);
  store_index(&m, z, index);
  finish(&m);
  assert(run(&m, &in, out, 4, &n) == RC_EFFECT_UNKNOWN && n == 0);
  uint32_t max = 23;
  in.builtin = builtin;
  in.context = &max;
  assert(run(&m, &in, out, 4, &n) == RC_EFFECT_COMPLETE && out[0].begin == 0 &&
         out[0].size == 96);
  max = 63;
  assert(run(&m, &in, out, 4, &n) == RC_EFFECT_COMPLETE && out[0].size == 256);
}
static void guards_and_merge(void) {
  for (unsigned early_return = 0; early_return < 2; early_return++) {
    Module m = base();
    uint32_t z = lit(&m, 0), limit = lit(&m, 4096), entry = id(&m),
             body = id(&m), merge = id(&m);
    start(&m, entry);
    uint32_t index = id(&m), condition = id(&m);
    OP(&m, 61, 1, index, 11);
    OP(&m, early_return ? 174 : 176, 3, condition, index, limit);
    OP(&m, 247, merge, 0);
    OP(&m, 250, condition, body, merge);
    OP(&m, 248, body);
    if (early_return) {
      OP(&m, 253);
    } else {
      store_index(&m, z, index);
      OP(&m, 249, merge);
    }
    OP(&m, 248, merge);
    if (early_return)
      store_index(&m, z, index);
    finish(&m);
    RcEffectInputs in = {0};
    RcEffectRange out[4];
    size_t n;
    assert(run(&m, &in, out, 4, &n) == RC_EFFECT_COMPLETE && n == 1 &&
           out[0].begin == 0 && out[0].size == 16384);
  }
  // The merge is reachable through either arm, so its false edge is not an
  // enduring condition.
  Module m = base();
  uint32_t z = lit(&m, 0), limit = lit(&m, 16), entry = id(&m), body = id(&m),
           merge = id(&m);
  start(&m, entry);
  uint32_t index = id(&m), condition = id(&m);
  OP(&m, 61, 1, index, 11);
  OP(&m, 174, 3, condition, index, limit);
  OP(&m, 247, merge, 0);
  OP(&m, 250, condition, body, merge);
  OP(&m, 248, body);
  OP(&m, 249, merge);
  OP(&m, 248, merge);
  store_index(&m, z, index);
  finish(&m);
  RcEffectInputs in = {0};
  RcEffectRange out[4];
  size_t n;
  assert(run(&m, &in, out, 4, &n) == RC_EFFECT_UNKNOWN && n == 0);
}
static Interval evaluate_node(Node *nodes, uint32_t n, uint32_t root) {
  RcEffectProgram p = {.node = nodes, .nnode = n};
  RcEffectInputs in = {0};
  Cache cache[16] = {0};
  Eval e = {
      .p = &p, .in = &in, .cache = cache, .context = 1, .next_context = 1};
  return eval(&e, root);
}
static void signed_and_bit_widths(void) {
  Node a[] = {{E_UNKNOWN, 0, 0, 0, 32, 0},
              {E_CONSTANT, 0, 0, 0, 32, 0x80000001},
              {E_FIRST_SPV + 114, 1, 0, 0, 64, 0}};
  Interval r = evaluate_node(a, 3, 2);
  assert(r.lo == UINT64_C(0xffffffff80000001) && r.hi == r.lo);
  a[1].value = 0x7fffffff;
  r = evaluate_node(a, 3, 2);
  assert(r.lo == 0x7fffffff && r.hi == r.lo);
  a[1] = (Node){E_UNKNOWN, 0, 0, 0, 32, 7};
  r = evaluate_node(a, 3, 2);
  assert(r.lo == 0 && r.hi == UINT64_MAX);
  a[2] = (Node){E_FIRST_SPV + 204, 1, 0, 0, 32, 0};
  r = evaluate_node(a, 3, 2);
  assert(r.lo == 0 && r.hi == UINT32_MAX);
  a[1].bits = 64;
  a[2].op = E_FIRST_SPV + 205;
  r = evaluate_node(a, 3, 2);
  assert(r.lo == 0 && r.hi == 64);
}
static void arithmetic_contains_concrete_values(void) {
  const uint32_t ops[] = {128, 130, 132, 194, 196, 197,
                          198, 199, 151, 149, 150};
  for (unsigned k = 0; k < sizeof ops / sizeof ops[0]; k++)
    for (uint32_t lo = 0; lo < 256; lo += 17)
      for (uint32_t hi = lo; hi < 256; hi += 19)
        for (uint32_t b = 0; b < 256; b += 13) {
          Interval r =
              arithmetic(ops[k], (Interval){lo, hi}, (Interval){b, b}, 8);
          for (uint32_t a = lo; a <= hi; a++) {
            uint32_t v = 0;
            switch (ops[k]) {
            case 128:
              v = (a + b) & 255;
              break;
            case 130:
              v = (a - b) & 255;
              break;
            case 132:
              v = (a * b) & 255;
              break;
            case 194:
              if (b >= 8)
                continue;
              v = a >> b;
              break;
            case 196:
              if (b >= 8)
                continue;
              v = (a << b) & 255;
              break;
            case 197:
              v = a | b;
              break;
            case 198:
              v = a ^ b;
              break;
            case 199:
              v = a & b;
              break;
            case 151:
              v = (a * b) >> 8;
              break;
            case 149:
              v = (a + b) >> 8;
              break;
            case 150:
              v = a < b;
              break;
            }
            assert(v >= r.lo && v <= r.hi);
          }
        }
}
static void stride_does_not_include_other_words(void) {
  Node nodes[] = {{E_UNKNOWN, 0, 0, 0, 32, 1},
                  {E_CONSTANT, 0, 0, 0, 32, 16},
                  {E_FIRST_SPV + 132, 0, 1, 0, 32, 0},
                  {E_CONSTANT, 0, 0, 0, 32, 68},
                  {E_FIRST_SPV + 128, 2, 3, 0, 32, 0}};
  RcEffectProgram p = {.node = nodes, .nnode = 5};
  Congruence c = congruence_at(&p, NULL, 4, 0);
  assert(c.bits == 4 && (c.value & 15) == 4);
}
static int captured_read(void *context, uint32_t set, uint32_t binding,
                         uint64_t offset, size_t size, void *dst) {
  if (!context || set || binding != 7 || size != 4 || offset >= 16 ||
      offset % 4)
    return 0;
  uint32_t value = *(uint32_t *)context + (uint32_t)offset * 4;
  memcpy(dst, &value, 4);
  return 1;
}
static void captured_reads_and_lengths(void) {
  Module m = base();
  uint32_t z = lit(&m, 0), entry = id(&m);
  start(&m, entry);
  uint32_t index = id(&m), ptr = id(&m), value = id(&m);
  OP(&m, 61, 1, index, 11);
  OP(&m, 65, 8, ptr, 10, z, index);
  OP(&m, 61, 1, value, ptr);
  store_index(&m, z, value);
  finish(&m);
  uint32_t max = 3;
  RcEffectInputs in = {
      .builtin = builtin, .context = &max, .read = captured_read};
  RcEffectRange out[4];
  size_t n;
  RcEffectProgram *p = rc_effect_compile(m.w, m.n);
  assert(p && p->supported);
  in.workspace_bytes = rc_effect_workspace_size(p);
  in.workspace = malloc(in.workspace_bytes);
  assert(rc_effect_read_ranges(p, &in, 0, 7, out, 4, &n) ==
             RC_EFFECT_COMPLETE &&
         n == 1 && out[0].begin == 0 && out[0].size == 16);
  assert(rc_effect_evaluate(p, &in, out, 4, &n) == RC_EFFECT_COMPLETE &&
         n == 1 && out[0].begin == 12 && out[0].size == 196);
  in.read = NULL;
  assert(rc_effect_evaluate(p, &in, out, 4, &n) == RC_EFFECT_UNKNOWN && n == 0);
  free(in.workspace);
  rc_effect_free(p);
  // Any array-length query makes a shortened descriptor observably different
  // even when loads are bounded.
  m.n -= 2;
  uint32_t len = id(&m);
  OP(&m, 68, 1, len, 10, 0);
  finish(&m);
  p = rc_effect_compile(m.w, m.n);
  assert(p);
  in.workspace_bytes = rc_effect_workspace_size(p);
  in.workspace = malloc(in.workspace_bytes);
  assert(rc_effect_read_ranges(p, &in, 0, 7, out, 4, &n) == RC_EFFECT_UNKNOWN &&
         n == 0);
  free(in.workspace);
  rc_effect_free(p);
}
static void loop_invariant_and_carried_values(void) {
  for (unsigned carried = 0; carried < 2; carried++) {
    Module m = base();
    OP(&m, 32, 15, 7, 1);
    uint32_t z = lit(&m, 0), three = lit(&m, 3), limit = lit(&m, 16);
    uint32_t entry = id(&m), head = id(&m), body = id(&m), cont = id(&m),
             merge = id(&m);
    start(&m, entry);
    OP(&m, 59, 15, 16, 7);
    OP(&m, 62, 16, three);
    OP(&m, 249, head);
    OP(&m, 248, head);
    uint32_t value = id(&m), condition = id(&m);
    OP(&m, 61, 1, value, 16);
    OP(&m, 176, 3, condition, value, limit);
    OP(&m, 246, merge, cont, 0);
    OP(&m, 250, condition, body, merge);
    OP(&m, 248, body);
    store_index(&m, z, value);
    OP(&m, 249, cont);
    OP(&m, 248, cont);
    if (carried) {
      uint32_t next = id(&m);
      OP(&m, 61, 1, next, 11);
      OP(&m, 62, 16, next);
    }
    OP(&m, 249, head);
    OP(&m, 248, merge);
    finish(&m);
    RcEffectInputs in = {0};
    RcEffectRange out[4];
    size_t n;
    assert(run(&m, &in, out, 4, &n) == RC_EFFECT_COMPLETE && n == 1);
    assert(out[0].begin == (carried ? 0 : 12) &&
           out[0].size == (carried ? 64 : 4));
  }
}
// Undefined propagation is checked through OpSelect in the adversarial fixture. Branching on an undefined
// predicate is undefined execution, so it cannot establish an expected output range.
static void separate_loads_are_not_correlated(void) {
  for (unsigned separate = 0; separate < 2; separate++) {
    Module m = base();
    uint32_t z = lit(&m, 0), limit = lit(&m, 16), entry = id(&m), body = id(&m),
             merge = id(&m);
    start(&m, entry);
    uint32_t ptr = id(&m), first = id(&m), condition = id(&m);
    OP(&m, 65, 8, ptr, 10, z, z);
    OP(&m, 61, 1, first, ptr);
    OP(&m, 176, 3, condition, first, limit);
    OP(&m, 247, merge, 0);
    OP(&m, 250, condition, body, merge);
    OP(&m, 248, body);
    uint32_t index = first;
    if (separate == 1) {
      index = id(&m);
      OP(&m, 61, 1, index, ptr);
    }
    store_index(&m, z, index);
    OP(&m, 249, merge);
    OP(&m, 248, merge);
    finish(&m);
    RcEffectInputs in = {0};
    RcEffectRange out[4];
    size_t n;
    RcEffectResult result = run(&m, &in, out, 4, &n);
    if (!separate)
      assert(result == RC_EFFECT_COMPLETE && n == 1 && out[0].size == 64);
    else
      assert(result == RC_EFFECT_UNKNOWN && n == 0);
  }
}
static void write_classification(void) {
  Module m = base();
  uint32_t z = lit(&m, 0), entry = id(&m);
  start(&m, entry);
  finish(&m);
  assert(rc_effect_write_class(m.w, m.n) == 0);
  m.n -= 2;
  store_index(&m, z, z);
  finish(&m);
  assert(rc_effect_write_class(m.w, m.n) == RC_EFFECT_WRITES_BUFFER);
  // Task payload writes must not create a whole-guest-memory writer. They also must not hide real writes
  // in the same module, including fallback payloads explicitly translated to StorageBuffer.
  const uint32_t task_classes[] = {5402, 12, 5349, 11, 9999};
  const uint32_t task_expected[] = {0, RC_EFFECT_WRITES_BUFFER, RC_EFFECT_WRITES_GPA,
                                  RC_EFFECT_WRITES_IMAGE, RC_EFFECT_WRITES_UNKNOWN};
  for (unsigned k = 0; k < sizeof task_classes / sizeof task_classes[0]; k++) {
    m = base();
    OP(&m, 32, 15, 5402, 1);
    OP(&m, 59, 15, 16, 5402);
    OP(&m, 32, 17, task_classes[k], 1);
    OP(&m, 59, 17, 18, task_classes[k]);
    z = lit(&m, 0); entry = id(&m);
    start(&m, entry);
    OP(&m, 62, 16, z);
    OP(&m, 62, 18, z);
    OP(&m, 5294, z, z, z, 16);
    finish(&m);
    assert(rc_effect_write_class(m.w, m.n) == task_expected[k]);
  }
  for (unsigned storage = 0; storage < 2; storage++) {
    m = base();
    OP(&m, 32, 15, storage ? 5349 : 3, 1);
    OP(&m, 59, 15, 16, storage ? 5349 : 3);
    z = lit(&m, 0);
    entry = id(&m);
    start(&m, entry);
    OP(&m, 62, 16, z);
    finish(&m);
    assert(rc_effect_write_class(m.w, m.n) ==
           (storage ? RC_EFFECT_WRITES_GPA : 0));
  }
  m = base();
  entry = id(&m);
  start(&m, entry);
  OP(&m, 99, 10, 10, 10);
  finish(&m);
  assert(rc_effect_write_class(m.w, m.n) == RC_EFFECT_WRITES_IMAGE);
  m.n -= 2;
  OP(&m, 62, 999, 10);
  finish(&m);
  assert(rc_effect_write_class(m.w, m.n) & RC_EFFECT_WRITES_UNKNOWN);
  assert(rc_effect_write_class(NULL, 0) == RC_EFFECT_WRITES_UNKNOWN);
  m.w[5] = 0;
  assert(rc_effect_write_class(m.w, m.n) == RC_EFFECT_WRITES_UNKNOWN);
}
typedef struct {
  uint32_t value, calls;
  int available;
} MemoFixture;
static int memo_fixture_read(void *context, uint32_t set, uint32_t binding,
                             uint64_t offset, size_t bytes, void *dst) {
  MemoFixture *f = context;
  f->calls++;
  if (!f->available || set || binding != 7 || offset || bytes != 4)
    return 0;
  memcpy(dst, &f->value, 4);
  return 1;
}
static void exact_input_memo(void) {
  Module m = base();
  uint32_t z = lit(&m, 0), entry = id(&m);
  start(&m, entry);
  uint32_t ptr = id(&m), index = id(&m);
  OP(&m, 65, 8, ptr, 10, z, z);
  OP(&m, 61, 1, index, ptr);
  store_index(&m, z, index);
  store_index(&m, z, index);
  finish(&m);
  RcEffectProgram *p = rc_effect_compile(m.w, m.n);
  assert(p);
  RcEffectMemo *memo = malloc(rc_effect_memo_size());
  assert(rc_effect_memo_init(memo, rc_effect_memo_size()));
  MemoFixture f = {3, 0, 1};
  RcEffectInputs in = {.read = memo_fixture_read, .context = &f};
  in.workspace_bytes = rc_effect_workspace_size(p);
  in.workspace = malloc(in.workspace_bytes);
  RcEffectRange out[4];
  size_t count;
  int hit;
  assert(rc_effect_evaluate_memo(p, &in, memo, out, 4, &count, &hit) ==
             RC_EFFECT_COMPLETE &&
         !hit);
  assert(count == 1 && out[0].begin == 12 &&
         f.calls ==
             1); // Repeated reads within one proof are pure and deduplicated.
  assert(rc_effect_evaluate_memo(p, &in, memo, out, 4, &count, &hit) ==
             RC_EFFECT_COMPLETE &&
         hit && f.calls == 2);
  assert(rc_effect_evaluate_memo(p, &in, memo, out, 0, &count, &hit) ==
             RC_EFFECT_OVERFLOW &&
         hit && !count);
  f.value = 8;
  assert(rc_effect_evaluate_memo(p, &in, memo, out, 4, &count, &hit) ==
             RC_EFFECT_COMPLETE &&
         !hit && out[0].begin == 32);
  in.groups[0] = 2;
  assert(rc_effect_evaluate_memo(p, &in, memo, out, 4, &count, &hit) ==
             RC_EFFECT_COMPLETE &&
         !hit);
  f.available = 0;
  assert(rc_effect_evaluate_memo(p, &in, memo, out, 4, &count, &hit) ==
             RC_EFFECT_UNKNOWN &&
         !hit && !count);
  assert(rc_effect_evaluate_memo(p, &in, memo, out, 4, &count, &hit) ==
             RC_EFFECT_UNKNOWN &&
         hit && !count);
  f.available = 1;
  assert(rc_effect_evaluate_memo(p, &in, memo, out, 4, &count, &hit) ==
             RC_EFFECT_COMPLETE &&
         !hit);
  RcEffectProgram *other = rc_effect_compile(m.w, m.n);
  assert(other && other->identity != p->identity);
  assert(rc_effect_evaluate_memo(other, &in, memo, out, 4, &count, &hit) ==
             RC_EFFECT_COMPLETE &&
         !hit);
  // Overflow never reuses a partial dependency trace, even if a valid result
  // would fit.
  MemoInput wrapper = {memo, &in};
  memo->ndeps = 0;
  memo->overflow = 0;
  memset(memo->buckets, 0, sizeof memo->buckets);
  for (unsigned i = 0; i <= MEMO_DEPENDENCIES; i++)
    memo_dependency(&wrapper,
                    (MemoDependency){.binding = 7, .offset = i, .size = 4});
  assert(memo->overflow && memo->ndeps == MEMO_DEPENDENCIES);
  rc_effect_free(other);
  rc_effect_free(p);
  free(in.workspace);
  free(memo);
}
static void definite_single_invocation_stores(void) {
  RcEffectInputs in = {.direct_dispatch = 1, .groups = {1, 1, 1}, .local = {1, 1, 1}};
  RcEffectRange spans[4];
  size_t count;
  for (int mode = 0; mode < 3; ++mode) {
    Module m = base();
    OP(&m, 15, 5, 12, 0x6e69616d, 0); // Compute entry 'main'.
    uint32_t zero = lit(&m, 0), first = lit(&m, 9), second = lit(&m, 17), entry = id(&m);
    uint32_t conditional = id(&m), end = id(&m);
    start(&m, entry);
    if (mode == 1) {
      uint32_t input = id(&m), condition = id(&m);
      OP(&m, 61, 1, input, 11);
      OP(&m, 170, 3, condition, input, zero);
      OP(&m, 247, end, 0);
      OP(&m, 250, condition, conditional, end);
      OP(&m, 248, conditional);
    }
    if (mode == 2) {
      uint32_t ptr = id(&m);
      OP(&m, 65, 8, ptr, 10, zero, first);
      OP(&m, 228, ptr, zero, zero, zero); // Unqualified atomics keep the may-write proof only.
    } else {
      store_index(&m, zero, first);
      store_index(&m, zero, second); // The stored value may equal old storage. Both writes still happened.
    }
    if (mode == 1) {
      OP(&m, 253);
      OP(&m, 248, end);
    }
    finish(&m);
    RcEffectProgram *p = rc_effect_compile(m.w, m.n);
    assert(p && rc_effect_supported(p));
    in.workspace_bytes = rc_effect_workspace_size(p);
    in.workspace = malloc(in.workspace_bytes);
    assert(in.workspace);
    assert(rc_effect_evaluate(p, &in, spans, 4, &count) == RC_EFFECT_COMPLETE && count == (mode == 2 ? 1 : 2));
    RcEffectResult actual = rc_effect_evaluate_definite_stores(p, &in, spans, 4, &count);
    if (mode) assert(actual == RC_EFFECT_UNKNOWN && !count);
    else {
      assert(actual == RC_EFFECT_COMPLETE && count == 2 && spans[0].kind == RC_EFFECT_SSBO && spans[0].binding == 7);
      assert(spans[0].begin == 36 && spans[0].size == 4 && spans[1].begin == 68 && spans[1].size == 4);
      in.groups[0] = 0;
      assert(rc_effect_evaluate_definite_stores(p, &in, spans, 4, &count) == RC_EFFECT_UNKNOWN && !count);
      in.groups[0] = 2;
      assert(rc_effect_evaluate_definite_stores(p, &in, spans, 4, &count) == RC_EFFECT_UNKNOWN && !count);
      in.groups[0] = 1;
      in.direct_dispatch = 0;
      assert(rc_effect_evaluate_definite_stores(p, &in, spans, 4, &count) == RC_EFFECT_UNKNOWN && !count);
      in.direct_dispatch = 1;
    }
    free(in.workspace);
    rc_effect_free(p);
  }
}

static void definite_affine_stores(void) {
  enum { DENSE, TILED, GAPPED, WRAPPED, CONDITIONAL, DIFFERENT_WORKGROUP };
  for (int mode = DENSE; mode <= DIFFERENT_WORKGROUP; ++mode) {
    Module m = base();
    OP(&m, 15, 5, 12, 0x6e69616d, 0);
    OP(&m, 16, 12, 17, 2, 1, 1); // The actual selected module's LocalSize.
    uint32_t zero = lit(&m, 0), one = lit(&m, 1), width = lit(&m, 8), length = lit(&m, 32);
    uint32_t gap = lit(&m, 2), max = lit(&m, UINT32_MAX);
    uint32_t vec = id(&m), gid_type = id(&m), gid = id(&m), array = id(&m);
    uint32_t record = id(&m), rec_ptr = id(&m), output = id(&m);
    OP(&m, 23, vec, 1, 3);
    OP(&m, 32, gid_type, 1, vec);
    OP(&m, 59, gid_type, gid, 1);
    OP(&m, 71, gid, 11, 28);
    OP(&m, 28, array, 1, length);
    OP(&m, 71, array, 6, 4);
    OP(&m, 30, record, array);
    OP(&m, 71, record, 3);
    OP(&m, 72, record, 0, 35, 0);
    OP(&m, 32, rec_ptr, 2, record);
    OP(&m, 59, rec_ptr, output, 2);
    OP(&m, 71, output, 34, 0);
    OP(&m, 71, output, 33, 8);
    if (mode == DIFFERENT_WORKGROUP) {
      uint32_t different = id(&m);
      OP(&m, 44, vec, different, one, one, one);
      OP(&m, 71, different, 11, 25);
    }
    uint32_t entry = id(&m), body = id(&m), end = id(&m);
    start(&m, entry);
    uint32_t coordinates = id(&m), x = id(&m), index = x;
    OP(&m, 61, vec, coordinates, gid);
    OP(&m, 81, 1, x, coordinates, 0);
    if (mode == TILED) {
      uint32_t y = id(&m), row = id(&m);
      index = id(&m);
      OP(&m, 81, 1, y, coordinates, 1);
      OP(&m, 132, 1, row, y, width);
      OP(&m, 128, 1, index, x, row);
    } else if (mode == GAPPED) {
      index = id(&m);
      OP(&m, 132, 1, index, x, gap);
    } else if (mode == WRAPPED) {
      uint32_t wrapped = id(&m);
      index = id(&m);
      OP(&m, 128, 1, wrapped, x, max);
      OP(&m, 128, 1, index, wrapped, one);
    } else if (mode == CONDITIONAL) {
      uint32_t condition = id(&m);
      OP(&m, 170, 3, condition, x, zero);
      OP(&m, 247, end, 0);
      OP(&m, 250, condition, body, end);
      OP(&m, 248, body);
    }
    uint32_t ptr = id(&m);
    OP(&m, 65, 8, ptr, output, zero, index);
    OP(&m, 62, ptr, one);
    if (mode == CONDITIONAL) {
      OP(&m, 249, end);
      OP(&m, 248, end);
    }
    finish(&m);

    RcEffectProgram *p = rc_effect_compile(m.w, m.n);
    assert(p && rc_effect_supported(p));
    RcEffectInputs in = {.direct_dispatch = 1, .groups = {16, 1, 1}, .local = {2, 1, 1}};
    if (mode == TILED) in.groups[0] = in.groups[1] = 4;
    if (mode == GAPPED) in.groups[0] = 8; // All 16 stores fit the array, with holes.
    RcEffectRange out[2];
    size_t n;
    RcEffectResult result = rc_effect_evaluate_definite_stores(p, &in, out, 2, &n);
    if (mode > TILED) assert(result == RC_EFFECT_UNKNOWN && !n);
    else {
      assert(result == RC_EFFECT_COMPLETE && n == 1 && out[0].kind == RC_EFFECT_SSBO &&
             out[0].set == 0 && out[0].binding == 8 && out[0].begin == 0 && out[0].size == 128);
      assert(rc_effect_evaluate_definite_stores(p, &in, NULL, 0, &n) == RC_EFFECT_OVERFLOW && !n);
      in.groups[1]++; // Past the shader array or an unpartitioned invocation axis.
      assert(rc_effect_evaluate_definite_stores(p, &in, out, 2, &n) == RC_EFFECT_UNKNOWN && !n);
      in.groups[1]--;
      in.local[0] = 1;
      assert(rc_effect_evaluate_definite_stores(p, &in, out, 2, &n) == RC_EFFECT_UNKNOWN && !n);
    }
    rc_effect_free(p);
  }
}

int main(void) {
  definite_single_invocation_stores();
  definite_affine_stores();
  constant_and_builtin();
  guards_and_merge();
  signed_and_bit_widths();
  arithmetic_contains_concrete_values();
  stride_does_not_include_other_words();
  captured_reads_and_lengths();
  loop_invariant_and_carried_values();
  separate_loads_are_not_correlated();
  write_classification();
  exact_input_memo();
  puts("resource_core_effects_check: synthetic control flow, captured index "
       "bounds, arithmetic and bit widths passed");
  return 0;
}
