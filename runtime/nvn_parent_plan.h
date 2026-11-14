// A recording owns its command bytes. These hints permit borrowing only the immutable execution metadata of a
// still-live child. Interleaved recorders and thread changes may discard hints without affecting recording.
#define PLAN_PARENT_MIN (128u << 10)
#define PLAN_CHILD_MAX 16

typedef struct { uint64_t handle, identity; size_t begin, bytes; } child_hint;
static _Thread_local struct {
    cmd_t *owner;
    uint64_t cookie;
    unsigned count;
    child_hint child[PLAN_CHILD_MAX];
} recording_hints;
static uint64_t next_plan_cookie;

static void plan_note_child(cmd_t *o, uint64_t handle, uint64_t identity, size_t bytes) {
    if (!o->plan_cookie || o->plan_cookie != recording_hints.cookie || o != recording_hints.owner) {
        recording_hints.cookie = o->plan_cookie = __atomic_add_fetch(&next_plan_cookie, 1, __ATOMIC_RELAXED);
        recording_hints.owner = o;
        recording_hints.count = 0;
    }
    if (recording_hints.count < PLAN_CHILD_MAX)
        recording_hints.child[recording_hints.count++] = (child_hint){handle, identity, (size_t)(o->cur - o->seg), bytes};
}

typedef struct {
    const NvnExecStep *steps;
    const NvnFrontStep *front;
    const uint8_t *state;
    size_t n, nf, state_bytes;
    size_t first, end, delta;  // local byte interval and the offset from local to parent commands
    const copy_prefix *child;
    const NvnRecordSpan *spans;
    size_t nspans;
} plan_piece;
typedef struct {
    plan_piece part[2 * PLAN_CHILD_MAX + 1];
    plan_piece interior[PLAN_CHILD_MAX];
    exec_builder seam[PLAN_CHILD_MAX + 1];
    copy_prefix *held[PLAN_CHILD_MAX];
    size_t count, nseam, nheld, eligible, n, nf, bytes;
} parent_plan;

static int plan_cut(const NvnExecStep *step) {
    return step->opcode != NVN_EXEC_STATE && nvn_record_op_class(step->opcode) != NVN_PLAN_RUN;
}

// The compilers cannot cross either cut. A fresh compilation of this interval therefore produces exactly the
// same state groups and front summaries even when state continues across the original child boundaries.
static int child_interior(const copy_prefix *pf, const child_hint *hint, plan_piece *out) {
    const size_t *offsets = copy_offsets(pf);
    const NvnRecordSpan *spans = (const NvnRecordSpan *)(offsets + pf->nuniform + pf->ndeferred + pf->npinops + pf->ncapture);
    const NvnExecStep *steps = (const NvnExecStep *)(spans + pf->nspans);
    const NvnFrontStep *front = (const NvnFrontStep *)(steps + pf->nexec);
    size_t first = 1, end = pf->nexec - 1;
    while (first < end && !plan_cut(&steps[first])) first++;
    while (first < end && !plan_cut(&steps[end - 1])) end--;
    if (first >= end) return 0;
    size_t lo = steps[first].begin, hi = steps[end - 1].end;
    if (lo < sizeof(op_hdr) || hi > pf->bytes - sizeof(op_hdr) || hi <= lo || hi - lo < PLAN_CHILD_MIN / 2) return 0;
    size_t f = 0, g = pf->nfront;
    while (f < g && front[f].begin < lo) f++;
    while (f < g && front[g - 1].end > hi) g--;
    size_t a = 0, b = pf->nspans;
    while (a < b && spans[a].begin < lo) a++;
    while (a < b && spans[b - 1].end > hi) b--;
    if (a == b || spans[a].begin != lo || spans[b - 1].end != hi) return 0;
    *out = (plan_piece){steps + first, front + f, (const uint8_t *)(front + pf->nfront),
                        end - first, g - f, pf->state_bytes, lo, hi, hint->begin - sizeof(op_hdr),
                        pf, spans + a, b - a};
    return 1;
}

static int parent_seam(parent_plan *p, const uint8_t *commands, size_t lo, size_t hi) {
    if (lo == hi) return 1;
    exec_builder *b = &p->seam[p->nseam++];
    exec_build(b, commands + lo, hi - lo);
    if (!b->n) return 0;
    front_build(b, commands + lo, hi - lo);
    if (b->failed) return 0;
    p->part[p->count++] = (plan_piece){b->steps, b->front, b->state, b->n, b->nfront, b->bytes, 0, hi - lo, lo, NULL, NULL, 0};
    return 1;
}

static int parent_plan_select(parent_plan *p, size_t bytes) {
    child_hint hints[PLAN_CHILD_MAX];
    size_t eligible = 0, covered = 0, end = sizeof(op_hdr);
    plat_mutex_lock(&copy_lock);
    for (unsigned i = 0; i < recording_hints.count; i++) {
        child_hint h = recording_hints.child[i];
        if (h.begin < end || h.begin > bytes - sizeof(op_hdr) || h.bytes > bytes - sizeof(op_hdr) - h.begin) break;
        end = h.begin + h.bytes;
        copy_prefix *pf = copy_find(h.handle);
        if (!pf || !pf->plan_complete || pf->made_at != h.identity ||
            pf->bytes - 2 * sizeof(op_hdr) != h.bytes || pf->nexec < 3) continue;
        pf->plan_readers++;
        hints[p->nheld] = h;
        p->held[p->nheld++] = pf;
    }
    plat_mutex_unlock(&copy_lock);
    for (size_t i = 0; i < p->nheld; i++) {
        plan_piece *part = &p->interior[eligible];
        if (child_interior(p->held[i], &hints[i], part)) {
            covered += part->end - part->first;
            eligible++;
        }
    }
    if (covered < bytes / 2) return 0;
    p->eligible = eligible;
    return 1;
}

static int parent_plan_build(parent_plan *p, const uint8_t *commands, size_t bytes) {
    size_t cursor = 0;
    for (size_t i = 0; i < p->eligible; i++) {
        plan_piece part = p->interior[i];
        if (!parent_seam(p, commands, cursor, part.first + part.delta)) return 0;
        p->part[p->count++] = part;
        cursor = part.end + part.delta;
    }
    if (!parent_seam(p, commands, cursor, bytes)) return 0;
    for (size_t i = 0; i < p->count; i++) {
        plan_piece part = p->part[i];
        p->n += part.n;
        p->nf += part.nf;
        for (size_t k = 0; k < part.n; k++) p->bytes += part.steps[k].state_bytes;
        for (size_t k = 0; k < part.nf; k++) p->bytes += part.front[k].state_bytes;
        if (p->n * sizeof(NvnExecStep) + p->nf * sizeof(NvnFrontStep) + p->bytes > EXEC_PLAN_BYTES) return 0;
    }
    return 1;
}

static size_t lower_offset(const size_t *v, size_t n, size_t key) {
    size_t first = 0;
    while (first < n) {
        size_t mid = first + (n - first) / 2;
        if (v[mid] < key) first = mid + 1;
        else n = mid;
    }
    return first;
}

static void import_offsets(offset_list *list, const size_t *offsets, size_t count, const plan_piece *part) {
    size_t lo = lower_offset(offsets, count, part->first);
    size_t hi = lower_offset(offsets, count, part->end);
    size_t need = list->n + hi - lo;
    if (need > list->cap) {
        size_t cap = list->cap ? list->cap : 64;
        while (cap < need) cap *= 2;
        list->v = realloc_wait(list->v, cap * sizeof *list->v, "command offset import");
        if (!list->v) rt_fatal("nvn: command offset import allocation");
        list->cap = cap;
    }
    for (size_t i = lo; i < hi; i++) list->v[list->n++] = offsets[i] + part->delta;
}

static uint64_t import_resources(const plan_piece *part, const uint8_t *commands, offset_list *uniform,
                                 offset_list *deferred, offset_list *pinops, offset_list *capture,
                                 capture_set *captured, span_list *spans) {
    const copy_prefix *pf = part->child;
    const size_t *u = copy_offsets(pf), *d = u + pf->nuniform, *pins = d + pf->ndeferred;
    size_t old_u = uniform->n, old_pin = pinops->n;
    import_offsets(uniform, u, pf->nuniform, part);
    import_offsets(deferred, d, pf->ndeferred, part);
    import_offsets(pinops, pins, pf->npinops, part);
    // A capture's first occurrence may be outside the imported interval, so use all binds for global deduplication.
    for (size_t i = old_u; i < uniform->n; i++) {
        size_t off = uniform->v[i];
        if (capture_add(captured, (const p_bind_buf *)(commands + off + sizeof(op_hdr)))) offset_add(capture, off);
    }
    uint64_t named = 0;
    for (size_t i = old_pin; i < pinops->n; i++) {
        const op_hdr *h = (const op_hdr *)(commands + pinops->v[i]);
        const tex_t *pins_here[9];
        int k = op_pins(h->op, nvn_op_payload(h), pins_here);
        for (int j = 0; j < k; j++) pin_ref(pins_here[j], 1, 0);
        named += k;
    }
    span_finish_draw(spans);
    size_t need = spans->n + part->nspans;
    if (need > spans->cap) {
        size_t cap = spans->cap ? spans->cap : 64;
        while (cap < need) cap *= 2;
        spans->v = realloc_wait(spans->v, cap * sizeof *spans->v, "command span import");
        if (!spans->v) rt_fatal("nvn: command span import allocation");
        spans->cap = cap;
    }
    for (size_t i = 0; i < part->nspans; i++) {
        NvnRecordSpan s = part->spans[i];
        s.begin += part->delta;
        s.end += part->delta;
        spans->v[spans->n++] = s;
    }
    return named;
}

static void plan_drop_readers(copy_prefix **held, size_t count) {
    if (count) {
        copy_prefix *retired = NULL;
        plat_mutex_lock(&copy_lock);
        int sweep = 0;
        for (size_t i = 0; i < count; i++) {
            if (!--held[i]->plan_readers && !held[i]->magic) sweep = 1;
        }
        plat_mutex_unlock(&copy_lock);
        if (sweep) {
            uint64_t replayed = gfx_replayed_submits();
            plat_mutex_lock(&copy_lock);
            retired = zombie_sweep(replayed, NULL);
            plat_mutex_unlock(&copy_lock);
        }
        doom_free(retired);
    }
}

static void parent_plan_release(parent_plan *p) {
    plan_drop_readers(p->held, p->nheld);
    for (size_t i = 0; i < p->nseam; i++) {
        free(p->seam[i].steps);
        free(p->seam[i].front);
        free(p->seam[i].state);
    }
}

static int relocate_blob(const plan_piece *p, uint8_t *dst, uint32_t source, uint32_t size, size_t lo, size_t hi) {
    if (source > p->state_bytes || size > p->state_bytes - source || size < offsetof(ExecStateBuild, writes)) return 0;
    memcpy(dst, p->state + source, size);
    ExecStateBuild *b = (ExecStateBuild *)dst;
    if (b->count > EXEC_STATE_WRITES || size != offsetof(ExecStateBuild, writes) + b->count * sizeof(ExecStateWrite)) return 0;
    for (uint32_t i = 0; i < b->count; i++) {
        ExecStateWrite *w = &b->writes[i];
        if (w->operation == ES_ZERO) {
            if (w->source) return 0;
        } else {
            size_t read = w->operation == ES_BOOL ? sizeof(int32_t) : w->bytes;
            if (w->operation > ES_BOOL || w->source < lo || w->source >= hi || read > hi - w->source) return 0;
            w->source += p->delta;
        }
    }
    return 1;
}

static int parent_plan_write(const parent_plan *p, NvnExecStep *steps) {
    NvnFrontStep *front = (NvnFrontStep *)(steps + p->n);
    uint8_t *state = (uint8_t *)(front + p->nf);
    size_t used = 0;
    for (size_t i = 0; i < p->count; i++) {
        const plan_piece *part = &p->part[i];
        for (size_t k = 0; k < part->n; k++) {
            NvnExecStep s = part->steps[k];
            if (s.begin < part->first || s.end > part->end || s.begin >= s.end) return 0;
            if (s.state_bytes) {
                if (!relocate_blob(part, state + used, s.state_offset, s.state_bytes, s.begin, s.end)) return 0;
                s.state_offset = (uint32_t)used;
                used += s.state_bytes;
            }
            s.begin += part->delta;
            s.end += part->delta;
            *steps++ = s;
        }
    }
    for (size_t i = 0; i < p->count; i++) {
        const plan_piece *part = &p->part[i];
        for (size_t k = 0; k < part->nf; k++) {
            NvnFrontStep f = part->front[k];
            if (f.begin < part->first || f.end > part->end || f.begin >= f.end ||
                !relocate_blob(part, state + used, f.state_offset, f.state_bytes, f.begin, f.end)) return 0;
            f.state_offset = (uint32_t)used;
            used += f.state_bytes;
            f.begin += part->delta;
            f.end += part->delta;
            *front++ = f;
        }
    }
    return used == p->bytes;
}
