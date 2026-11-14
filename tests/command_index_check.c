// Exercise the actual copy, deferred-binding, clone and retirement paths without a game or GPU. Unused NVN entry
// points are removed by section garbage collection; the platform stubs below are deliberately single-threaded.
#include <assert.h>
#include <stdio.h>
#include <stdarg.h>

#include "../runtime/nvn_api_cmd.c"
#include "../runtime/vk_exec.h"
#define VK_EXEC_STATE_COMPILE_ONLY
#include "../runtime/vk_exec_state.h"

// The producer test does not emit Vulkan work. A real typed-handler identity still has to survive every clone.
static void test_draw(struct VkExec *ex, const void *payload) { (void)ex; (void)payload; }
NvnExecFn gfx_exec_prepared_handler(uint32_t op) { return op == OP_DRAW ? test_draw : NULL; }

static int test_mode = 2;
static uint64_t replayed;
unsigned rt_trace;
const char *cfg_get(const char *name) { (void)name; return NULL; }
void nvn_called(uint8_t *seen, const char *name) { (void)seen; (void)name; }
uint64_t rt_call(Cpu *c, uint64_t fn, int n, ...) { (void)c; (void)fn; (void)n; abort(); }
void perf_size(PerfSizes *s, const char *name, uint64_t count, uint64_t limit, uint64_t bytes) {
    (void)s; (void)name; (void)count; (void)limit; (void)bytes;
}
int resource_prep_mode(void) { return test_mode; }
uint64_t gfx_replayed_submits(void) { return replayed; }
void plat_mutex_lock(PlatMutex *p) { (void)p; }
void plat_mutex_unlock(PlatMutex *p) { (void)p; }
void plat_mutex_lock_shared(PlatMutex *p) { (void)p; }
void plat_mutex_unlock_shared(PlatMutex *p) { (void)p; }
uint64_t plat_now_ns(void) { return 0; }
void plat_sleep_ms(uint32_t ms) { (void)ms; abort(); }
void plat_commit_info(double *p, double *u, double *l) { *p = *u = *l = 0; }
void rt_log(const char *fmt, ...) { (void)fmt; }
_Noreturn void rt_fatal(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    abort();
}

static void finish(cmd_t *cmd) { put_hdr(cmd_alloc(cmd, 8), OP_END, 8); }
static uint64_t make_copy(cmd_t *cmd, uintptr_t control) {
    uint8_t *copy = flatten_recording(cmd->seg, (size_t)(cmd->cur - cmd->seg));
    copy_add(control, control + 1, copy);
    return (uintptr_t)copy;
}
static p_bind_handles *deferred_at(uint64_t handle, size_t off) {
    return (p_bind_handles *)((uint8_t *)(uintptr_t)handle + off + 8);
}

// Inspect the production compiler's clone-relative sources without invoking any rendering. The apply algorithm
// itself is compared with the scalar interpreter by exec_state_check; this fixture checks copy ownership/lifetime.
static uint64_t state_texture(const NvnRecordPlan *plan, size_t state_offset) {
    const ExecStateBuild *state = (const ExecStateBuild *)(plan->state_data + state_offset);
    for (uint32_t i = 0; i < state->count; i++) {
        const ExecStateWrite *w = &state->writes[i];
        if (w->notify != ES_TEXTURE) continue;
        assert(w->operation == ES_COPY && w->bytes >= sizeof(uint64_t));
        assert(w->source <= plan->bytes - sizeof(uint64_t));
        uint64_t value;
        memcpy(&value, plan->commands + w->source, sizeof value);
        return value;
    }
    assert(!"compiled block has no texture source");
    return 0;
}

static uint64_t planned_texture(const NvnRecordPlan *plan, size_t block_at) {
    size_t cursor = SIZE_MAX;
    const NvnExecStep *step = nvn_record_exec_at(plan, (const op_hdr *)(plan->commands + block_at), &cursor);
    assert(step && step->opcode == NVN_EXEC_STATE);
    return state_texture(plan, step->state_offset);
}

static void check_front_primary_boundaries(const NvnRecordPlan *plan) {
    size_t primary_bytes = 0;
    for (size_t i = 0; i < plan->nexec; i++) {
        const NvnExecStep *s = &plan->exec[i];
        if (s->opcode == NVN_EXEC_STATE) {
            assert(s->state_offset == primary_bytes);
            primary_bytes += s->state_bytes;
        }
    }
    size_t primary = 0;
    for (size_t i = 0; i < plan->nfront; i++) {
        const NvnFrontStep *f = &plan->front[i];
        assert(f->state_offset >= primary_bytes && f->state_offset <= plan->state_bytes);
        assert(f->state_bytes <= plan->state_bytes - f->state_offset);
        while (primary < plan->nexec && plan->exec[primary].begin < f->begin) primary++;
        assert(primary < plan->nexec && plan->exec[primary].begin == f->begin);
        size_t end = f->begin;
        uint32_t ops = 0, draws = 0;
        while (end < f->end) {
            assert(primary < plan->nexec);
            const NvnExecStep *s = &plan->exec[primary++];
            assert(s->begin == end && s->end <= f->end);
            end = s->end;
            ops += s->ops;
            draws += !!(s->flags & NVN_EXEC_DRAW);
        }
        assert(end == f->end && ops == f->ops && draws == f->draws);
    }
}

// Front summaries append to the same allocation as their primary inputs. Force the first append to grow it,
// then ensure all subsequent compositions still reference the unchanged primary prefix through the new address.
static void check_front_storage_growth(void) {
    cmd_t cmd = {0};
    put_hdr(cmd_alloc(&cmd, 8), OP_BEGIN, 8);
    for (int i = 0; i < 96; i++) {
        *REC(&cmd, OP_BIND_UBO, p_bind_buf) = (p_bind_buf){.stage = 0, .index = 3, .addr = 0x1000 + i * 256, .size = 256};
        *REC(&cmd, OP_BIND_UBO, p_bind_buf) = (p_bind_buf){.stage = 1, .index = 4, .addr = 0x2000 + i * 256, .size = 256};
        REC(&cmd, OP_DRAW, p_draw)->count = 3;
        REC(&cmd, OP_DRAW, p_draw)->count = 3;
    }
    finish(&cmd);
    size_t bytes = (size_t)(cmd.cur - cmd.seg);
    exec_builder b = {0};
    exec_build(&b, cmd.seg, bytes);
    assert(b.n && b.bytes > 4096);
    size_t primary_bytes = b.bytes;
    uint8_t *saved = malloc(primary_bytes);
    assert(saved);
    memcpy(saved, b.state, primary_bytes);
    uint8_t *exact = realloc(b.state, primary_bytes);
    assert(exact);
    b.state = exact, b.state_cap = primary_bytes;
    front_build(&b, cmd.seg, bytes);
    assert(!b.failed && b.nfront > 1 && b.state_cap > primary_bytes);
    assert(b.bytes > primary_bytes && !memcmp(saved, b.state, primary_bytes));
    NvnRecordPlan plan = {.commands = cmd.seg, .bytes = bytes, .exec = b.steps, .nexec = b.n,
        .front = b.front, .nfront = b.nfront, .state_data = b.state, .state_bytes = b.bytes};
    check_front_primary_boundaries(&plan);
    free(saved);
    free(b.steps);
    free(b.front);
    free(b.state);
    free(cmd.seg);
}

static void check_exec_slices(const NvnRecordPlan *plan) {
    size_t off[32], n = 0;
    for (size_t at = 0; at < plan->bytes;) {
        assert(n < sizeof off / sizeof off[0]);
        off[n++] = at;
        at += ((const op_hdr *)(plan->commands + at))->size;
    }
    for (size_t begin = 0; begin < n; begin++) for (size_t end = begin; end <= n; end++) {
        size_t cursor = SIZE_MAX, remaining = end - begin;
        const op_hdr *h = (const op_hdr *)(plan->commands + off[begin]);
        const op_hdr *limit = (const op_hdr *)(plan->commands + (end == n ? plan->bytes : off[end]));
        while (remaining) {
            const NvnExecStep *s = nvn_record_exec_slice(plan, h, limit, (uint32_t)remaining, &cursor);
            if (!s) { remaining--; h = nvn_op_next(h); continue; }
            const op_hdr *ref = h;
            for (uint32_t k = 0; k < s->ops; k++) ref = nvn_op_next(ref);
            assert((const uint8_t *)ref == plan->commands + s->end && ref <= limit);
            if (s->opcode == OP_DRAW) assert(s->run == test_draw && (s->flags & NVN_EXEC_DRAW));
            remaining -= s->ops, h = ref;
        }
        assert(h == limit);
    }
}

int main(void) {
    cmd_t cmd = {0};
    pin_t pin = {0};
    uint64_t table[2] = {11, 22};
    put_hdr(cmd_alloc(&cmd, 8), OP_BEGIN, 8);
    for (int i = 0; i < 7; i++) REC(&cmd, OP_DRAW, p_draw)->count = i + 1;
    size_t ub = (size_t)(cmd.cur - cmd.seg);
    *REC(&cmd, OP_BIND_UBO, p_bind_buf) = (p_bind_buf){.stage = 0, .index = 3, .addr = 0x1000, .size = 256};
    size_t def = (size_t)(cmd.cur - cmd.seg);
    p_bind_handles *b = RECN(&cmd, OP_BIND_TEX, p_bind_handles, h, 2);
    b->n = 2, b->src = (uintptr_t)table;
    size_t pinop = (size_t)(cmd.cur - cmd.seg);
    REC(&cmd, OP_CLEAR_TEX, p_clear_tex)->tex = &pin.t;
    finish(&cmd);

    uint64_t h = make_copy(&cmd, 0x10000);
    copy_prefix *pf = (copy_prefix *)(uintptr_t)h - 1;
    assert(pf->nuniform == 1 && pf->ndeferred == 1 && pf->npinops == 1);
    assert(copy_offsets(pf)[0] == ub && copy_offsets(pf)[1] == def && copy_offsets(pf)[2] == pinop);
    assert(pin.refs == 1);
    NvnRecordPlan plan;
    assert(nvn_record_plan(h, &plan));
    assert(plan.nexec && plan.state_bytes);
    check_exec_slices(&plan);
    size_t exec_cursor = SIZE_MAX;
    const NvnExecStep *state = nvn_record_exec_at(&plan, (const op_hdr *)(plan.commands + ub), &exec_cursor);
    assert(state && state->ops == 2 && state->end == pinop && state->opcode == NVN_EXEC_STATE);
    assert(!nvn_record_exec_at(&plan, (const op_hdr *)(plan.commands + def), &exec_cursor));
    const op_hdr *first_draw = (const op_hdr *)(plan.commands + sizeof(op_hdr));
    const op_hdr *after_draws = (const op_hdr *)(plan.commands + ub);
    size_t cursor = SIZE_MAX;
    uint32_t stride = 0;
    assert(nvn_record_draw_run(&plan, first_draw, after_draws, &cursor, &stride) == 7);
    // Command records include tail padding to their eight-byte stream alignment.
    assert(stride == ((sizeof(op_hdr) + sizeof(p_draw) + 7u) & ~(size_t)7u));
    assert(stride == first_draw->size);
    // Every interior chunk start and limit is compared with scalar traversal, including a limit in this run.
    for (uint32_t start = 0; start < 7; start++) {
        for (uint32_t stop = start; stop <= 7; stop++) {
            const op_hdr *from = first_draw, *to = first_draw;
            for (uint32_t k = 0; k < start; k++) from = nvn_op_next(from);
            for (uint32_t k = 0; k < stop; k++) to = nvn_op_next(to);
            cursor = SIZE_MAX;
            assert(nvn_record_draw_run(&plan, from, to, &cursor, &stride) == stop - start);
        }
    }
    cursor = SIZE_MAX;
    assert(!nvn_record_draw_run(&plan, after_draws, (const op_hdr *)(plan.commands + plan.bytes), &cursor, &stride));
    const size_t *offs = NULL;
    size_t count = 999;
    assert(nvn_record_uniform_offsets(h, &offs, &count) && count == 1 && offs[0] == ub);
    assert(!nvn_record_uniform_offsets(1, &offs, &count));
    assert(!nvn_record_uniform_offsets((uintptr_t)cmd.seg, &offs, &count));
    nvn_submit_prepare(&h, 1);
    assert(deferred_at(h, def)->h[0] == 11 && deferred_at(h, def)->h[1] == 22);
    assert(planned_texture(&plan, ub) == 11);

    // Resubmission while the first copy is queued must clone both command bytes and metadata. The original submit's
    // payload stays frozen even when the guest table changes; its pins live through both submissions.
    table[0] = 33;
    uint64_t clone;
    nvn_submit_prepare_async(&h, &clone, 1);
    assert(clone != h && pin.refs == 2 && n_clones == 1);
    assert(deferred_at(h, def)->h[0] == 11 && deferred_at(clone, def)->h[0] == 33);
    assert(nvn_record_uniform_offsets(clone, &offs, &count) && count == 1 && offs[0] == ub);
    assert(offs != copy_offsets(pf));
    assert(nvn_record_plan(clone, &plan));
    check_exec_slices(&plan);
    assert(planned_texture(&plan, ub) == 33);
    NvnRecordPlan original;
    assert(nvn_record_plan(h, &original) && planned_texture(&original, ub) == 11);
    assert(original.exec != plan.exec && original.state_data != plan.state_data);
    cursor = SIZE_MAX;
    assert(nvn_record_draw_run(&plan, (const op_hdr *)(plan.commands + sizeof(op_hdr)),
                              (const op_hdr *)(plan.commands + ub), &cursor, &stride) == 7);
    assert(!nvn_handle_live(clone));
    release_copies(0x10000, 0x10001);
    assert(!nvn_handle_live(h));
    assert(nvn_record_uniform_offsets(h, &offs, &count));
    assert(pin.refs == 2);
    replayed = nsubmits;
    release_copies(0x20000, 0x20001);
    assert(pin.refs == 0 && !copy_indexed.n && !nzombies);
    assert(!nvn_record_uniform_offsets(h, &offs, &count));
    assert(!nvn_record_uniform_offsets(clone, &offs, &count));

    // Repeated binds, zero-sized uniforms and every deferred kind remain indexed in recording order. Plain binds and
    // invalid deferred counts must not become reads from guest memory. Reuse the scratch builder for another copy.
    cmd.cur = cmd.seg;
    put_hdr(cmd_alloc(&cmd, 8), OP_BEGIN, 8);
    for (int i = 0; i < 200; i++) {
        *REC(&cmd, OP_BIND_UBO, p_bind_buf) = (p_bind_buf){.stage = i % 6, .index = 4, .addr = 0x2000, .size = i % 2 ? 0 : 64};
        b = RECN(&cmd, i % 2 ? OP_BIND_IMG : OP_BIND_SAMP, p_bind_handles, h, 2);
        b->n = 2, b->src = i % 3 ? (uintptr_t)table : 0;
    }
    finish(&cmd);
    h = make_copy(&cmd, 0x30000);
    pf = (copy_prefix *)(uintptr_t)h - 1;
    assert(pf->nuniform == 200 && pf->ndeferred == 133 && !pf->npinops);
    copy_index_check(pf);
    nvn_submit_prepare(&h, 1);
    replayed = nsubmits;
    release_copies(0x30000, 0x30001);

    // Short/interleaved draw sequences retain ordinary spans, so common bind/draw traffic does not allocate a new
    // structural span for every draw. Long runs split at query-state boundaries and at the fixed read-ahead bound.
    cmd.cur = cmd.seg;
    put_hdr(cmd_alloc(&cmd, 8), OP_BEGIN, 8);
    size_t first_bind = 0;
    for (int i = 0; i < 12; i++) {
        REC(&cmd, OP_DRAW, p_draw)->count = 3;
        if (!i) first_bind = (size_t)(cmd.cur - cmd.seg);
        *REC(&cmd, OP_BIND_UBO, p_bind_buf) = (p_bind_buf){.stage = 0, .index = 3, .addr = 0x1000, .size = 256};
        b = RECN(&cmd, OP_BIND_TEX, p_bind_handles, h, 2);
        b->n = 2, b->src = (uintptr_t)table;
    }
    size_t long_draws = (size_t)(cmd.cur - cmd.seg);
    for (int i = 0; i < 1030; i++) REC(&cmd, OP_DRAW, p_draw)->count = 3;
    size_t query = (size_t)(cmd.cur - cmd.seg);
    REC(&cmd, OP_REPORT_COUNTER, p_report_counter)->addr = 0;
    for (int i = 0; i < 3; i++) REC(&cmd, OP_DRAW, p_draw)->count = 3;
    finish(&cmd);
    h = make_copy(&cmd, 0x35000);
    assert(nvn_record_plan(h, &plan));
    assert(plan.nfront > 0);
    check_front_primary_boundaries(&plan);
    for (size_t i = 0; i < plan.nfront; i++) {
        const NvnFrontStep *f = &plan.front[i];
        assert(f->end <= query && f->draws >= 2 && f->draws <= 16);
        size_t fc = SIZE_MAX;
        assert(nvn_record_front_at(&plan, (const op_hdr *)(plan.commands + f->begin), &fc) == f);
        assert(!nvn_record_front_at(&plan, nvn_op_next((const op_hdr *)(plan.commands + f->begin)), &fc));
        for (uint32_t budget = 0; budget <= f->draws + 1; budget++) {
            const op_hdr *from = (const op_hdr *)(plan.commands + f->begin);
            for (const op_hdr *limit = from; (const uint8_t *)limit <= plan.commands + f->end;
                 limit = nvn_op_next(limit)) {
                fc = SIZE_MAX;
                const NvnFrontStep *fit = nvn_record_front_slice(&plan, from, limit, budget, &fc);
                assert((fit != NULL) == ((const uint8_t *)limit == plan.commands + f->end && budget >= f->draws));
                if ((const uint8_t *)limit == plan.commands + f->end) break;
            }
        }
    }
    unsigned draw_spans = 0;
    for (size_t i = 0; i < plan.nspans; i++) {
        const NvnRecordSpan *span = &plan.spans[i];
        if (span->draw_stride) {
            assert(span->begin >= long_draws && span->end <= query && span->ops >= 4);
            draw_spans++;
        }
    }
    assert(draw_spans == 2);
    cursor = SIZE_MAX;
    const op_hdr *at = (const op_hdr *)(plan.commands + long_draws), *limit = (const op_hdr *)(plan.commands + query);
    uint32_t planned = 0;
    while (at != limit) {
        uint32_t got = nvn_record_draw_run(&plan, at, limit, &cursor, &stride);
        assert(got);
        planned += got;
        at = (const op_hdr *)((const uint8_t *)at + (size_t)got * stride);
    }
    assert(planned == 1030);
    nvn_submit_prepare(&h, 1);
    assert(planned_texture(&plan, first_bind) == 33);
    assert(state_texture(&plan, plan.front[0].state_offset) == 33);
    table[0] = 123;
    nvn_submit_prepare_async(&h, &clone, 1);
    assert(clone != h && nvn_record_plan(clone, &original));
    assert(original.nfront == plan.nfront && original.front != plan.front);
    assert(!memcmp(original.front, plan.front, plan.nfront * sizeof *plan.front));
    assert(!memcmp(original.state_data, plan.state_data, plan.state_bytes));
    assert(planned_texture(&original, first_bind) == 123 && planned_texture(&plan, first_bind) == 33);
    assert(state_texture(&original, original.front[0].state_offset) == 123);
    assert(state_texture(&plan, plan.front[0].state_offset) == 33);
    check_front_primary_boundaries(&original);
    replayed = nsubmits;
    release_copies(0x35000, 0x35001);
    check_front_storage_growth();

    // Reference mode still takes the old walks; empty recordings have a valid empty prepared index.
    for (int mode = 0; mode <= 2; mode++) {
        test_mode = mode;
        cmd.cur = cmd.seg;
        put_hdr(cmd_alloc(&cmd, 8), OP_BEGIN, 8);
        finish(&cmd);
        h = make_copy(&cmd, 0x40000);
        assert(nvn_record_uniform_offsets(h, &offs, &count) == (mode != 0));
        if (mode) assert(!count);
        nvn_submit_prepare(&h, 1);
        replayed = nsubmits;
        release_copies(0x40000, 0x40001);
    }
    free(cmd.seg);
    puts("command index: clone isolation, retained lifetimes, deferred reads and reference fallback passed");
    return 0;
}
