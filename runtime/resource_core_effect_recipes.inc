// Compile local reaching alternatives into guarded scalar expressions. A guard belongs to the alternative
// which established it; it is never hoisted over a different address arm or a different write. Expansion is
// bounded and failure leaves the complete original effect graph in place.
enum { RECIPE_LIMIT = 65536, RECIPE_VARIANTS = 256, RECIPE_DEPTH = 128 };
typedef struct {
  uint32_t term, context, next;
} RecipeAlt;
typedef struct {
  uint32_t term, next;
  uint8_t truth;
} RecipeGuard;
typedef struct {
  uint32_t node, context, head;
  uint8_t state;
} RecipeMemo;
typedef struct {
  RcEffectProgram *p;
  RecipeAlt *alt;
  RecipeGuard *guard;
  RecipeMemo *memo;
  uint32_t *guard_hash;
  uint32_t *choice_tag, *widened;
  uint8_t *stable, *slice, *interest, *invariant;
  uint32_t nalts, nguards, nmemo, original_nodes, ticks, depth, operations;
  int failed;
} RecipeBuild;
typedef struct {
  uint32_t head, tail, count;
} RecipeList;
static Interval arithmetic(uint32_t, Interval, Interval, uint32_t);
static Interval compare(uint32_t, Interval, Interval, uint32_t);
static uint32_t recipe_hash(uint32_t a, uint32_t b) {
  uint32_t h = a * 0x9e3779b9u + b * 0x85ebca6bu;
  h ^= h >> 16;
  return h & (RECIPE_LIMIT * 2 - 1);
}
static void recipe_append(RecipeBuild *r, RecipeList *list, uint32_t term, uint32_t context) {
  if (r->failed)
    return;
  if (list->count == RECIPE_VARIANTS || r->nalts == RECIPE_LIMIT) {
    if (!r->failed)
      TRACE("recipe failure line%u ticks%u operations%u alts%u guards%u depth%u\n", __LINE__, r->ticks,
            r->operations, r->nalts, r->nguards, r->depth);
    r->failed = 1;
    return;
  }
  uint32_t at = ++r->nalts;
  r->alt[at] = (RecipeAlt){term, context, 0};
  if (list->tail)
    r->alt[list->tail].next = at;
  else
    list->head = at;
  list->tail = at;
  list->count++;
}
static void recipe_copy(RecipeBuild *r, RecipeList *out, uint32_t head) {
  for (uint32_t a = head; a && !r->failed; a = r->alt[a].next)
    recipe_append(r, out, r->alt[a].term, r->alt[a].context);
}
static int recipe_stable(RecipeBuild *r, uint32_t id, unsigned depth) {
  if (id >= RECIPE_LIMIT || depth > RECIPE_DEPTH)
    return 0;
  if (r->stable[id])
    return r->stable[id] == 2;
  r->stable[id] = 1;
  Node n = r->p->node[id];
  int good = 0;
  if (n.op == E_CONSTANT || n.op == E_UNION)
    good = 1;
  else if (n.op == E_UNKNOWN)
    good = n.c != 2;
  else if (n.op == E_READ || n.op == E_BUILTIN)
    good = recipe_stable(r, n.a, depth + 1);
  else if (n.op >= E_FIRST_SPV) {
    uint32_t op = n.op - E_FIRST_SPV;
    good = recipe_stable(r, n.a, depth + 1);
    if (op != 83 && op != 113 && op != 114 && op != 124 && op != 126 && op != 168 && op != 200 && op != 204 &&
        op != 205)
      good &= recipe_stable(r, n.b, depth + 1);
    if (op == 169)
      good &= recipe_stable(r, n.c, depth + 1);
    Node divisor = r->p->node[n.b];
    if (op == 134 || op == 135 || op == 137 || op == 138 || op == 139) {
      good &= divisor.op == E_CONSTANT && divisor.value != 0;
      if ((op == 135 || op == 138 || op == 139) && divisor.value == mask_bits(n.bits))
        good = 0;
    }
    if (op == 194 || op == 195 || op == 196)
      good &= divisor.op == E_CONSTANT && divisor.value < n.bits;
  }
  r->stable[id] = good ? 2 : 3;
  return good;
}
static uint32_t recipe_guard_node(RecipeBuild *r, uint32_t term, int truth, uint32_t next) {
  if (r->failed || ++r->operations > 1000000) {
    if (!r->failed)
      TRACE("recipe failure line%u ticks%u operations%u alts%u guards%u depth%u\n", __LINE__, r->ticks,
            r->operations, r->nalts, r->nguards, r->depth);
    r->failed = 1;
    return 0;
  }
  uint32_t slot = recipe_hash(term, (next << 1) | (uint32_t)truth);
  while (r->guard_hash[slot]) {
    uint32_t id = r->guard_hash[slot];
    RecipeGuard g = r->guard[id];
    if (g.term == term && g.next == next && g.truth == truth)
      return id;
    slot = (slot + 1) & (RECIPE_LIMIT * 2 - 1);
  }
  if (r->nguards == RECIPE_LIMIT) {
    if (!r->failed)
      TRACE("recipe failure line%u ticks%u operations%u alts%u guards%u depth%u\n", __LINE__, r->ticks,
            r->operations, r->nalts, r->nguards, r->depth);
    r->failed = 1;
    return 0;
  }
  uint32_t id = ++r->nguards;
  r->guard[id] = (RecipeGuard){term, next, (uint8_t)truth};
  r->guard_hash[slot] = id;
  return id;
}
static uint32_t recipe_binding(RecipeBuild *r, uint32_t context, uint32_t term, unsigned arm,
                               unsigned depth) {
  if (r->failed || ++r->operations > 1000000) {
    if (!r->failed)
      TRACE("recipe failure line%u ticks%u operations%u alts%u guards%u depth%u\n", __LINE__, r->ticks,
            r->operations, r->nalts, r->nguards, r->depth);
    r->failed = 1;
    return UINT32_MAX;
  }
  if (depth > RECIPE_DEPTH) {
    if (!r->failed)
      TRACE("recipe failure line%u ticks%u operations%u alts%u guards%u depth%u\n", __LINE__, r->ticks,
            r->operations, r->nalts, r->nguards, r->depth);
    r->failed = 1;
    return UINT32_MAX;
  }
  if (!context || r->guard[context].term > term)
    return recipe_guard_node(r, term, (int)arm, context);
  RecipeGuard g = r->guard[context];
  if (g.term == term)
    return g.truth == arm ? context : UINT32_MAX;
  uint32_t tail = recipe_binding(r, g.next, term, arm, depth + 1);
  return tail == UINT32_MAX ? tail : recipe_guard_node(r, g.term, g.truth, tail);
}
// UINT32_MAX denotes an impossible alternative. Select predicates with undefined origins retain both arms.
// Actual branch identities are represented separately: consuming an undefined branch predicate is undefined
// execution.
static uint32_t recipe_guard_add(RecipeBuild *r, uint32_t context, uint32_t term, int truth, unsigned depth) {
  if (r->failed || ++r->operations > 1000000) {
    if (!r->failed)
      TRACE("recipe failure line%u ticks%u operations%u alts%u guards%u depth%u\n", __LINE__, r->ticks,
            r->operations, r->nalts, r->nguards, r->depth);
    r->failed = 1;
    return UINT32_MAX;
  }
  if (depth > RECIPE_DEPTH) {
    if (!r->failed)
      TRACE("recipe failure line%u ticks%u operations%u alts%u guards%u depth%u\n", __LINE__, r->ticks,
            r->operations, r->nalts, r->nguards, r->depth);
    r->failed = 1;
    return UINT32_MAX;
  }
  Node n = r->p->node[term];
  if (n.op == E_CONSTANT)
    return !!n.value == truth ? context : UINT32_MAX;
  if (!recipe_stable(r, term, 0))
    return context;
  if (n.op == E_FIRST_SPV + 168)
    return recipe_guard_add(r, context, n.a, !truth, depth + 1);
  if ((n.op == E_FIRST_SPV + 167 && truth) || (n.op == E_FIRST_SPV + 166 && !truth)) {
    context = recipe_guard_add(r, context, n.a, truth, depth + 1);
    return context == UINT32_MAX ? context : recipe_guard_add(r, context, n.b, truth, depth + 1);
  }
  if (!context || r->guard[context].term > term)
    return recipe_guard_node(r, term, truth, context);
  RecipeGuard g = r->guard[context];
  if (g.term == term)
    return g.truth == truth ? context : UINT32_MAX;
  uint32_t tail = recipe_guard_add(r, g.next, term, truth, depth + 1);
  return tail == UINT32_MAX ? tail : recipe_guard_node(r, g.term, g.truth, tail);
}
static uint32_t recipe_term(RecipeBuild *r, Node n) {
  if (r->failed || ++r->operations > 1000000) {
    if (!r->failed)
      TRACE("recipe failure line%u ticks%u operations%u alts%u guards%u depth%u\n", __LINE__, r->ticks,
            r->operations, r->nalts, r->nguards, r->depth);
    r->failed = 1;
    return 0;
  }
  if (r->p->nnode >= RECIPE_LIMIT) {
    if (!r->failed)
      TRACE("recipe failure line%u ticks%u operations%u alts%u guards%u depth%u\n", __LINE__, r->ticks,
            r->operations, r->nalts, r->nguards, r->depth);
    r->failed = 1;
    return 0;
  }
  if (n.op >= E_FIRST_SPV) {
    uint32_t op = n.op - E_FIRST_SPV;
    Node a = r->p->node[n.a], b = r->p->node[n.b];
    if (a.op == E_CONSTANT && b.op == E_CONSTANT &&
        ((op >= 128 && op <= 152) || (op >= 164 && op <= 179) || (op >= 194 && op <= 199))) {
      Interval v = op >= 164 && op <= 179
                       ? compare(op, (Interval){a.value, a.value}, (Interval){b.value, b.value}, a.bits)
                       : arithmetic(op, (Interval){a.value, a.value}, (Interval){b.value, b.value}, n.bits);
      if (v.lo == v.hi)
        return cn(r->p, v.lo, n.bits);
    }
    if (op == 168 && a.op == E_CONSTANT)
      return cn(r->p, !a.value, 1);
    return operation(r->p, op, n.a, n.b, n.c, n.bits);
  }
  return intern(r->p, n);
}
static uint32_t recipe_normalize(RecipeBuild *, uint32_t, uint32_t);
static void recipe_slice(RecipeBuild *r, uint32_t id, unsigned depth) {
  if (depth > 256 || r->slice[id])
    return;
  r->slice[id] = 1;
  Node n = r->p->node[id];
  if (n.op == E_CHOICE) {
    for (unsigned i = 0; i < n.b; i++)
      recipe_slice(r, r->p->choice[n.a + i].node, depth + 1);
  } else if (n.op == E_READ || n.op == E_BUILTIN)
    recipe_slice(r, n.a, depth + 1);
  else if (n.op >= E_FIRST_SPV) {
    recipe_slice(r, n.a, depth + 1);
    recipe_slice(r, n.b, depth + 1);
    recipe_slice(r, n.c, depth + 1);
  }
}
static int recipe_invariant(RecipeBuild *r, uint32_t id, unsigned depth) {
  if (depth > 128)
    return 0;
  if (r->invariant[id])
    return r->invariant[id] == 2;
  r->invariant[id] = 1;
  Node n = r->p->node[id];
  int yes = n.op == E_CONSTANT || n.op == E_BUILTIN;
  if (n.op == E_READ)
    yes = !(n.value >> 32) && recipe_invariant(r, n.a, depth + 1);
  else if (n.op == E_CHOICE) {
    yes = 1;
    for (unsigned i = 0; i < n.b && yes; i++)
      yes = recipe_invariant(r, r->p->choice[n.a + i].node, depth + 1);
  } else if (n.op >= E_FIRST_SPV) {
    unsigned op = n.op - E_FIRST_SPV;
    yes = recipe_invariant(r, n.a, depth + 1);
    if (op != 83 && op != 113 && op != 114 && op != 124 && op != 126 && op != 168 && op != 200 && op != 204 &&
        op != 205)
      yes &= recipe_invariant(r, n.b, depth + 1);
    if (op == 169)
      yes &= recipe_invariant(r, n.c, depth + 1);
  }
  r->invariant[id] = yes ? 2 : 3;
  return yes;
}
// Conditions which cannot constrain an address retain only their branch identity; dropping their numeric
// interpretation admits extra alternatives rather than excluding any write.
static int recipe_interesting(RecipeBuild *r, uint32_t id, unsigned depth) {
  if (depth > 128)
    return 0;
  if (r->interest[id])
    return r->interest[id] == 2;
  r->interest[id] = 1;
  Node n = r->p->node[id];
  int keep = n.op != E_CONSTANT && r->slice[id];
  if (n.op == E_CHOICE) {
    for (unsigned i = 0; i < n.b && !keep; i++) {
      Choice c = r->p->choice[n.a + i];
      keep = recipe_interesting(r, c.node, depth + 1);
    }
  } else if (n.op >= E_FIRST_SPV) {
    unsigned op = n.op - E_FIRST_SPV;
    if (op >= 170 && op <= 179)
      keep = (r->slice[n.a] && r->p->node[n.a].op != E_CONSTANT && recipe_invariant(r, n.b, 0)) ||
             (r->slice[n.b] && r->p->node[n.b].op != E_CONSTANT && recipe_invariant(r, n.a, 0));
    else if (op >= 164 && op <= 169) {
      keep |= recipe_interesting(r, n.a, depth + 1);
      if (op != 168)
        keep |= recipe_interesting(r, n.b, depth + 1);
      if (op == 169)
        keep |= recipe_interesting(r, n.c, depth + 1);
    }
  }
  r->interest[id] = keep ? 2 : 3;
  return keep;
}
static RecipeList recipe_condition(RecipeBuild *r, RecipeList input, uint32_t condition, int truth,
                                   int branch) {
  condition = canonical(r->p, condition);
  Node predicate = r->p->node[condition];
  unsigned op = predicate.op - E_FIRST_SPV;
  if (op == 168)
    return recipe_condition(r, input, predicate.a, !truth, branch);
  // This records one branch result version, without evaluating unrelated computations or promoting its
  // operands. Repeated loads of one unchanged function-local store share a version; opaque external loads
  // retain SSA IDs.
  if (branch && predicate.op != E_CONSTANT) {
    uint32_t tag = recipe_term(r, (Node){.op = E_UNION, .c = 2, .bits = 1, .value = condition});
    RecipeList guarded = {0};
    for (uint32_t a = input.head; a && !r->failed; a = r->alt[a].next) {
      RecipeAlt source = r->alt[a];
      uint32_t context = recipe_binding(r, source.context, tag, truth, 0);
      if (context != UINT32_MAX)
        recipe_append(r, &guarded, source.term, context);
    }
    input = guarded;
  }
  if (predicate.op == E_CHOICE)
    return input;
  if (predicate.op != E_CONSTANT && !recipe_interesting(r, condition, 0))
    return input;
  if ((op == 167 && truth) || (op == 166 && !truth))
    return recipe_condition(r, recipe_condition(r, input, predicate.a, truth, branch), predicate.b, truth,
                            branch);
  if ((op == 167 && !truth) || (op == 166 && truth)) {
    if ((r->p->node[predicate.a].op != E_CONSTANT && !recipe_interesting(r, predicate.a, 0)) ||
        (r->p->node[predicate.b].op != E_CONSTANT && !recipe_interesting(r, predicate.b, 0)))
      return input;
    RecipeList out = {0}, a = recipe_condition(r, input, predicate.a, truth, branch),
               b = recipe_condition(r, input, predicate.b, truth, branch);
    recipe_copy(r, &out, a.head);
    recipe_copy(r, &out, b.head);
    return out;
  }

  RecipeList out = {0};
  // A scalar used by this actual comparison is defined on every defined execution reaching this branch.
  // Replace expensive data-dependent arithmetic only inside this guard context, preserving width and
  // identity. An unguarded use, a different load, a bypass arm and a previous iteration do not inherit this
  // replacement.
  if (branch && op >= 170 && op <= 179 && r->p->node[predicate.b].op == E_CONSTANT &&
      !recipe_invariant(r, predicate.a, 0)) {
    uint32_t from = canonical(r->p, predicate.a);
    if (!r->widened[from])
      r->widened[from] = recipe_term(r, (Node){.op = E_UNKNOWN,
                                               .bits = r->p->node[from].bits,
                                               .value = UINT64_C(0xffffffef00000000) | from});
    uint32_t to = r->widened[from];
    uint32_t tag =
        recipe_term(r, (Node){.op = E_UNION, .a = from, .b = to, .c = 1, .bits = 32, .value = from});
    Node comparison = predicate;
    comparison.a = to;
    uint32_t term = recipe_term(r, comparison);
    for (uint32_t a = input.head; a && !r->failed; a = r->alt[a].next) {
      RecipeAlt source = r->alt[a];
      uint32_t context = recipe_binding(r, source.context, tag, 1, 0);
      if (context != UINT32_MAX)
        context = recipe_guard_add(r, context, term, truth, 0);
      if (context != UINT32_MAX)
        recipe_append(r, &out, source.term, context);
    }
    return out;
  }
  for (uint32_t a = input.head; a && !r->failed; a = r->alt[a].next) {
    RecipeAlt source = r->alt[a];
    uint32_t head = recipe_normalize(r, condition, source.context);
    for (uint32_t b = head; b && !r->failed; b = r->alt[b].next) {
      RecipeAlt test = r->alt[b];
      uint32_t context = recipe_guard_add(r, test.context, test.term, truth, 0);
      if (context != UINT32_MAX)
        recipe_append(r, &out, source.term, context);
    }
  }
  return out;
}
static RecipeList recipe_block(RecipeBuild *r, RecipeList list, uint32_t block) {
  if (block) {
    Block b = r->p->blocks[block];
    for (uint32_t j = 0; j < b.nguards && !r->failed && list.head; j++) {
      Fact f = r->p->guard[b.guards + j];
      list = recipe_condition(r, list, f.node, f.truth, 1);
    }
  }
  return list;
}
static uint32_t recipe_normalize(RecipeBuild *r, uint32_t id, uint32_t context) {
  if (r->failed)
    return 0;
  r->ticks++;
  r->depth++;
  if (r->ticks > RECIPE_LIMIT * 4 || r->depth > RECIPE_DEPTH) {
    if (!r->failed)
      TRACE("recipe failure line%u ticks%u operations%u alts%u guards%u depth%u\n", __LINE__, r->ticks,
            r->operations, r->nalts, r->nguards, r->depth);
    r->failed = 1;
    r->depth--;
    return 0;
  }
  id = canonical(r->p, id);
  uint32_t slot = recipe_hash(id, context);
  while (r->memo[slot].state) {
    RecipeMemo m = r->memo[slot];
    if (m.node == id && m.context == context) {
      if (m.state == 1) {
        if (!r->failed)
          TRACE("recipe failure line%u ticks%u operations%u alts%u guards%u depth%u\n", __LINE__, r->ticks,
                r->operations, r->nalts, r->nguards, r->depth);
        r->failed = 1;
      }
      r->depth--;
      return m.head;
    }
    slot = (slot + 1) & (RECIPE_LIMIT * 2 - 1);
  }
  if (r->nmemo++ == RECIPE_LIMIT) {
    if (!r->failed)
      TRACE("recipe failure line%u ticks%u operations%u alts%u guards%u depth%u\n", __LINE__, r->ticks,
            r->operations, r->nalts, r->nguards, r->depth);
    r->failed = 1;
    r->depth--;
    return 0;
  }
  r->memo[slot] = (RecipeMemo){id, context, 0, 1};
  RecipeList out = {0};
  Node n = r->p->node[id];
  for (uint32_t q = context; q; q = r->guard[q].next) {
    Node replacement = r->p->node[r->guard[q].term];
    if (replacement.op == E_UNION && replacement.c == 1 && replacement.a == id) {
      recipe_append(r, &out, replacement.b, context);
      goto normalized;
    }
  }
  if (n.op == E_CHOICE) {
    if (!r->choice_tag[id])
      r->choice_tag[id] = recipe_term(r, (Node){.op = E_UNION, .bits = 32, .value = id});
    for (uint32_t j = 0; j < n.b && !r->failed; j++) {
      uint32_t bound = recipe_binding(r, context, r->choice_tag[id], j, 0);
      if (bound == UINT32_MAX)
        continue;
      Choice edge = r->p->choice[n.a + j];
      RecipeList list = {0};
      recipe_append(r, &list, 0, bound);
      list = recipe_block(r, list, edge.block);
      if (edge.condition)
        list = recipe_condition(r, list, edge.condition, edge.truth, 1);
      for (uint32_t a = list.head; a && !r->failed; a = r->alt[a].next)
        recipe_copy(r, &out, recipe_normalize(r, edge.node, r->alt[a].context));
    }
  } else if (n.op == E_FIRST_SPV + 169) {
    for (int truth = 0; truth <= 1 && !r->failed; truth++) {
      RecipeList list = {0};
      recipe_append(r, &list, 0, context);
      list = recipe_condition(r, list, n.a, truth, 0);
      for (uint32_t a = list.head; a && !r->failed; a = r->alt[a].next)
        recipe_copy(r, &out, recipe_normalize(r, truth ? n.b : n.c, r->alt[a].context));
    }
  } else if (n.op == E_READ || n.op == E_BUILTIN || n.op >= E_FIRST_SPV) {
    uint32_t left = recipe_normalize(r, n.a, context), op = n.op - E_FIRST_SPV;
    int binary = n.op >= E_FIRST_SPV && op != 83 && op != 113 && op != 114 && op != 124 && op != 126 &&
                 op != 168 && op != 200 && op != 204 && op != 205;
    for (uint32_t a = left; a && !r->failed; a = r->alt[a].next) {
      RecipeAlt x = r->alt[a];
      Node made = n;
      made.a = x.term;
      if (!binary)
        recipe_append(r, &out, recipe_term(r, made), x.context);
      else {
        uint32_t right = recipe_normalize(r, n.b, x.context);
        for (uint32_t b = right; b && !r->failed; b = r->alt[b].next) {
          RecipeAlt y = r->alt[b];
          made.b = y.term;
          recipe_append(r, &out, recipe_term(r, made), y.context);
        }
      }
    }
  } else
    recipe_append(r, &out, id, context);
normalized:
  r->memo[slot].head = out.head;
  r->memo[slot].state = 2;
  r->depth--;
  return out.head;
}
static void prepare_recipes(RcEffectProgram *p) {
  TRACE("prepare recipes nodes%u effects%u\n", p->nnode, p->neffect);
  if (!p->neffect || p->nnode >= RECIPE_LIMIT)
    return;
  RecipeBuild r = {.p = p, .original_nodes = p->nnode};
  r.alt = calloc(RECIPE_LIMIT + 1, sizeof *r.alt);
  r.guard = calloc(RECIPE_LIMIT + 1, sizeof *r.guard);
  r.memo = calloc(RECIPE_LIMIT * 2, sizeof *r.memo);
  r.guard_hash = calloc(RECIPE_LIMIT * 2, sizeof *r.guard_hash);
  r.stable = calloc(RECIPE_LIMIT, 1);
  r.choice_tag = calloc(RECIPE_LIMIT, sizeof *r.choice_tag);
  r.slice = calloc(RECIPE_LIMIT, 1);
  r.interest = calloc(RECIPE_LIMIT, 1);
  r.widened = calloc(RECIPE_LIMIT, 4);
  r.invariant = calloc(RECIPE_LIMIT, 1);
  Effect *result = NULL;
  uint32_t count = 0, capacity = 0;
  if (!r.alt || !r.guard || !r.memo || !r.guard_hash || !r.stable || !r.choice_tag || !r.slice ||
      !r.interest || !r.widened || !r.invariant)
    goto done;

  for (uint32_t i = 0; i < p->neffect && !r.failed; i++) {
    Effect f = p->effect[i];
    memset(r.slice, 0, RECIPE_LIMIT);
    memset(r.interest, 0, RECIPE_LIMIT);
    memset(r.memo, 0, RECIPE_LIMIT * 2 * sizeof *r.memo);
    memset(r.guard_hash, 0, RECIPE_LIMIT * 2 * sizeof *r.guard_hash);
    r.nalts = r.nguards = r.nmemo = r.ticks = r.operations = 0;
    recipe_slice(&r, f.low, 0);
    recipe_slice(&r, f.high, 0);
    TRACE("start effect %u guards%u variant%u\n", i, p->blocks[f.block].nguards, count);
    RecipeList list = {0};
    recipe_append(&r, &list, 0, 0);
    list = recipe_block(&r, list, f.block);
    for (uint32_t g = list.head; g && !r.failed; g = r.alt[g].next) {
      uint32_t low = recipe_normalize(&r, f.low, r.alt[g].context);
      for (uint32_t a = low; a && !r.failed; a = r.alt[a].next) {
        RecipeAlt lo = r.alt[a];
        uint32_t high = recipe_normalize(&r, f.high, lo.context);
        for (uint32_t b = high; b && !r.failed; b = r.alt[b].next) {
          RecipeAlt hi = r.alt[b];
          if (count == RECIPE_LIMIT || !grow((void **)&result, &capacity, count, sizeof *result) ||
              !grow((void **)&p->blocks, &p->cblocks, p->nblocks, sizeof *p->blocks)) {
            r.failed = 1;
            break;
          }
          Block block = {.guards = p->nguard};
          for (uint32_t c = hi.context; c; c = r.guard[c].next) {
            RecipeGuard guard = r.guard[c];
            if (p->node[guard.term].op == E_UNION)
              continue;
            if (!grow((void **)&p->guard, &p->cguard, p->nguard, sizeof *p->guard)) {
              r.failed = 1;
              break;
            }
            p->guard[p->nguard++] = (Fact){guard.term, guard.truth};
            block.nguards++;
          }
          p->blocks[p->nblocks] = block;
          Effect made = f;
          made.low = lo.term;
          made.high = hi.term;
          made.block = p->nblocks++;
          result[count++] = made;
        }
      }
    }
  }
  TRACE("effect recipes %u -> %u, nodes %u, alternatives %u, guards %u, failed %d\n", p->neffect, count,
        p->nnode, r.nalts, r.nguards, r.failed);
  if (!r.failed) {
    p->original_effects = p->neffect;
    p->recipes = 1;
    free(p->effect);
    p->effect = result;
    p->neffect = count;
    p->ceffect = capacity;
    result = NULL;
  }
done:
  free(result);
  free(r.alt);
  free(r.guard);
  free(r.memo);
  free(r.guard_hash);
  free(r.stable);
  free(r.choice_tag);
  free(r.slice);
  free(r.interest);
  free(r.widened);
  free(r.invariant);
}
