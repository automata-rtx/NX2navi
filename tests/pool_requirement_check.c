#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "vk_exec.h"
#include "pool_requirement_constants.h"

typedef struct DrawCtx {
    int sf_mvprev;
    struct { uint64_t early_probes, early_known, early_prepared, early_learned, early_misses, early_prefix,
             early_prior, early_leaders, mesh_probed, mesh_matched, early_reject[VK_POOL_REFUSE_COUNT]; } pool_stats;
} DrawCtx;
static DrawCtx context, *dc = &context;
static VkCtx native = {.has_mesh_shader = 1};
VkCtx *vk_ctx(void) { return &native; }
static struct { uint64_t cur_fence; int ready, sealed; } PL;
static int n_probe_sh, deferred, skipped, calls, draw_calls, barriers, current_pool;
static uint32_t generation = 10;
static uint32_t resolved_generation;
static uint64_t resolved_texture;
static int defer_during_draw;
static unsigned prefix_effects[OP_COUNT_];
static uint64_t ready_generation;
static const char *early_setting, *mesh_setting;
static unsigned diagnostic_logs;
const char *cfg_get(const char *name) {
    if (!strcmp(name, "NX2NAVI_POOL_EARLY_TRACE")) return "1";
    if (!strcmp(name, "NX2NAVI_POOL_MESH_LEADER")) return mesh_setting;
    return early_setting;
}
void rt_log(const char *fmt, ...) { (void)fmt; diagnostic_logs++; }
const char *nvn_op_name(uint32_t op) { (void)op; return "fixture"; }
static int pool_on(void) { return 1; }
static int pool_current(void) { return current_pool; }
static int resource_prep_mode(void) { return 1; }
int vk_texture_deferred(void) { return deferred; }
uint32_t vk_texture_slots_gen(void) { return generation; }
uint64_t vk_exec_pool_generation(void) { return ready_generation; }
#include "vk_pool_requirement.h"

#define CHUNK_DRAWS 32u
static struct { uint64_t drained, out; } Q;
static struct { uint64_t serial_draws; } RW;
typedef struct { VkExecPoolRequirement requirement; } Group;
typedef struct { int idx; Group *g; } Chunk;
static Chunk *rw_chunk;
enum { GRAPHICS_BOUNDARY_NONE };
struct VkExecGraphicsDraw { unsigned used; };
static VkExecGraphicsDraw selected, *active_draw;
static VkExecGraphicsResult prepare_result = VK_EXEC_GRAPHICS_READY;
static unsigned preparations, declarations, noops;
static VkExecGraphicsResult prepare_ordered_graphics(VkExec *ex, const op_hdr *op, int boundary,
    RcRegistryPosition *attempted, RcRegistryPosition *replanned, VkExecGraphicsDraw **draw) {
    assert(ex && op && boundary == GRAPHICS_BOUNDARY_NONE && !attempted->scene && !replanned->scene);
    assert(!*draw && !active_draw);
    preparations++;
    *draw = &selected;
    return prepare_result;
}
static int graphics_input_finished(VkExec *ex, const op_hdr *op, VkExecGraphicsResult result,
                                   const VkExecGraphicsDraw *draw) {
    assert(ex && op && (!draw || draw == &selected));
    return result == VK_EXEC_GRAPHICS_READY || result == VK_EXEC_GRAPHICS_COMPAT || result == VK_EXEC_GRAPHICS_NOOP;
}
VkExecGraphicsDraw *vk_exec_graphics_draw_enter(VkExecGraphicsDraw *draw) {
    VkExecGraphicsDraw *previous = active_draw;
    active_draw = draw;
    return previous;
}
void vk_exec_graphics_draw_cancel(VkExecGraphicsDraw **draw) { assert(*draw != active_draw || !*draw); *draw = NULL; }
static void *resource_declare(VkExec *ex, const op_hdr *op, uint32_t count) {
    assert(op && count == 1 && !ex->resource_op);
    ex->resource_op = ++ex->resource_next;
    declarations++;
    return &declarations;
}
static void resource_declare_noop(VkExec *ex) {
    assert(!active_draw && !ex->resource_op);
    ex->resource_op = ++ex->resource_next;
}
void *vk_exec_effects_enter(void *effect) { assert(!effect || effect == &declarations); return NULL; }
void vk_exec_effects_recorded(void *effect, int emitted) { assert((!effect || effect == &declarations) && !emitted); }
void vk_exec_effects_leave(void *previous) { assert(!previous); }
void vk_exec_effects_release(void *effect) { assert(!effect || effect == &declarations); }
static void graphics_noop_record(VkExec *ex, const op_hdr *op) { assert(ex->resource_op && op && !active_draw); noops++; }
static void ahead_draw(VkExec *ex) { (void)ex; assert(!"Prediction must never resolve a shader or draw"); }
static void ahead_set_rt(VkExec *ex, const p_set_rt *rt);
void vk_exec_barrier(VkExec *ex) { barriers++; ex->in_pass = 0; }
static void op_one(VkExec *ex, const op_hdr *op) {
    assert(barriers > 0 && !ex->in_pass);
    calls++;
    if (op->op == OP_DRAW || op->op == OP_DRAW_IDX || op->op == OP_DRAW_IDX_IND) {
        draw_calls++;
        resolved_generation = generation;
        resolved_texture = ex->tex[NVN_STAGE_VS][2];
        if (defer_during_draw) deferred = 1;
        if (!skipped && !deferred) PL.cur_fence = ex->fence, PL.ready = 1, ready_generation++;
    } else if (op->op == OP_SET_RT) ahead_set_rt(ex, nvn_op_payload(op));
    else if (op->op == OP_RENDER_ENABLE) ex->render_disabled = !((const p_flag *)nvn_op_payload(op))->value;
    else if (op->op == OP_REPORT_COUNTER || op->op == OP_RESET_COUNTER || op->op == OP_CLEAR_DS) prefix_effects[op->op]++;
    else vk_exec_ahead_op(ex, op->op, nvn_op_payload(op));
}
#include "pool_leader_under_test.h"
typedef struct { op_hdr h; uint8_t payload[256]; } Record;
static Record records[132];
static void record(unsigned i, unsigned op) {
    records[i] = (Record){.h = {op, sizeof(Record)}};
}

int main(void) {
    prog_t program = {.serial = 123}, other = {.serial = 124};
    VkExec ex = {.dc = dc, .prog = &program, .prog_stages = 3, .fence = 1};
    ex.tex[NVN_STAGE_VS][2] = 22, ex.img[NVN_STAGE_FS][5] = 55;
    VkExecPoolRequirement learned = {0};
    pool_requirement_capture(&learned, &ex, 0);
    assert(learned.valid && pool_requirement_matches(&learned, &ex, 0));
    Group group = {0};
    Chunk first = {.g = &group}, second = {.g = &group, .idx = 1};
    rw_chunk = &first;
    VkExecPoolRequirement *sink = vk_exec_pool_requirement_begin();
    assert(sink == &group.requirement);
    // Ignored nonpool/mesh/skipped attempts and a deferred retry must not consume learning.
    assert(vk_exec_pool_requirement_begin() == sink);
    deferred = 1;
    pool_requirement_capture(sink, &ex, 0);
    assert(!sink->valid && vk_exec_pool_requirement_begin() == sink);
    deferred = 0;
    pool_requirement_capture(sink, &ex, 0);
    assert(sink->valid && !vk_exec_pool_requirement_begin());
    rw_chunk = &second;
    group.requirement.valid = 0;
    assert(!vk_exec_pool_requirement_begin());
    rw_chunk = NULL;
    assert(!vk_exec_pool_requirement_begin());
    assert(!vk_exec_draw_pool_requirement_known(&ex, 0));
    vk_exec_draw_pool_learn(&learned);
    assert(vk_exec_draw_pool_requirement_known(&ex, 0));
    ex.has_render_cond = 1; // historical marker persists after a condition has already resolved
    assert(vk_exec_draw_pool_requirement_known(&ex, 0));
    ex.render_disabled = 1;
    assert(!vk_exec_draw_pool_requirement_known(&ex, 0));
    ex.render_disabled = 0;
    ex.tex[NVN_STAGE_VS][3]++;
    assert(vk_exec_draw_pool_requirement_known(&ex, 0));
    ex.tex[NVN_STAGE_VS][2]++;
    generation++; // scheduling never reuses handles or resource versions from the learned draw
    assert(vk_exec_draw_pool_requirement_known(&ex, 0));
    ex.prog_stages++;
    assert(!vk_exec_draw_pool_requirement_known(&ex, 0));
    ex.prog_stages--;
    ex.pipe.color_fmt[0]++;
    assert(!vk_exec_draw_pool_requirement_known(&ex, 0));
    ex.pipe.color_fmt[0]--;
    ex.pipe.vattr[0] = 0xffffffffu;
    assert(!vk_exec_draw_pool_requirement_known(&ex, 0));
    ex.pipe.vattr[0] = 0;
    program.serial++;
    assert(!vk_exec_draw_pool_requirement_known(&ex, 0));
    program.serial--;
    deferred = 1;
    pool_requirement_capture(&learned, &ex, 0);
    assert(!learned.valid);
    deferred = 0;
    pool_requirement_capture(&learned, &ex, 0);
    assert(learned.valid);

    // Prefix prediction uses a copy; only canonical execution changes the real state, exactly once.
    record(0, OP_BIND_PROGRAM);
    p_bind_program bind = {.prog = &program, .stages = ex.prog_stages};
    memcpy(records[0].payload, &bind, sizeof bind);
    record(1, OP_DRAW); record(2, OP_DRAW); record(3, OP_END);
    ex.prog = &other;
    assert(pool_leader_draw(&ex, &records[0].h, &records[3].h) == &records[1].h);
    assert(ex.prog == &other && !calls);
    uint32_t draws = 65;
    ex.in_pass = 1;
    assert(pool_leader_run(&ex, &records[0].h, &records[3].h, &draws) == &records[2].h);
    assert(draws == 64 && calls == 2 && draw_calls == 1 && RW.serial_draws == 1 && ex.prog == &program);
    assert(preparations == 1 && declarations == 2 && !noops && !active_draw && !ex.resource_op);
    assert(context.pool_stats.early_prepared == 1);
    assert(resolved_generation == generation && resolved_texture == ex.tex[NVN_STAGE_VS][2]);
    assert(barriers == 1 && !ex.in_pass);
    PL.sealed = current_pool = 1;
    assert(!vk_exec_draw_pool_leader_needed(&ex));
    current_pool = 0;
    assert(vk_exec_draw_pool_leader_needed(&ex));
    PL.sealed = 0;
    // There is no leader when it would leave too few draws, when preparation already exists, or older work is live.
    int previous = calls;
    assert(pool_leader_run(&ex, &records[2].h, &records[3].h, &draws) == &records[2].h && calls == previous);
    draws = 100;
    assert(pool_leader_run(&ex, &records[2].h, &records[3].h, &draws) == &records[2].h && calls == previous);
    ex.fence++; Q.out = 1;
    assert(pool_leader_run(&ex, &records[2].h, &records[3].h, &draws) == &records[2].h && calls == previous);
    Q.drained = Q.out;
    unsigned before_declarations = declarations, before_preparations = preparations;
    prepare_result = VK_EXEC_GRAPHICS_UNAVAILABLE;
    assert(pool_leader_run(&ex, &records[2].h, &records[3].h, &draws) == &records[2].h);
    assert(calls == previous && draws == 100 && declarations == before_declarations && preparations == before_preparations + 1);
    prepare_result = VK_EXEC_GRAPHICS_NOOP;
    assert(pool_leader_run(&ex, &records[2].h, &records[3].h, &draws) == &records[3].h);
    assert(calls == previous && draws == 99 && declarations == before_declarations && noops == 1);
    prepare_result = VK_EXEC_GRAPHICS_READY;
    draws = 100;
    skipped = 1;
    assert(pool_leader_run(&ex, &records[2].h, &records[3].h, &draws) == &records[3].h && draws == 99);
    assert(context.pool_stats.early_prepared == 1); // skipped draw never claims ready resource publication
    skipped = 0;
    ex.fence++;
    defer_during_draw = 1;
    assert(pool_leader_run(&ex, &records[2].h, &records[3].h, &draws) == &records[3].h);
    assert(context.pool_stats.early_prepared == 1); // canonical deferral cannot publish a prepared result
    defer_during_draw = deferred = 0;
    record(0, OP_BEGIN); record(1, OP_DRAW);
    assert(pool_leader_draw(&ex, &records[0].h, &records[2].h) == &records[1].h);
    ex.render_disabled = 1;
    assert(!pool_leader_draw(&ex, &records[0].h, &records[2].h));
    ex.render_disabled = 0;
    record(0, OP_RENDER_ENABLE);
    ((p_flag *)records[0].payload)->value = 1;
    ex.render_disabled = 1;
    assert(pool_leader_draw(&ex, &records[0].h, &records[2].h) == &records[1].h);
    assert(ex.render_disabled == 1); // copied enable never changes the replay's live state
    ex.fence++;
    assert(pool_leader_run(&ex, &records[0].h, &records[2].h, &draws) == &records[2].h);
    assert(!ex.render_disabled && context.pool_stats.early_prepared == 2);
    ((p_flag *)records[0].payload)->value = 0;
    assert(!pool_leader_draw(&ex, &records[0].h, &records[2].h));
    record(0, OP_REPORT_COUNTER); record(1, OP_RESET_COUNTER); record(2, OP_CLEAR_DS);
    record(3, OP_DRAW); record(4, OP_END);
    assert(pool_leader_draw(&ex, &records[0].h, &records[4].h) == &records[3].h);
    assert(!prefix_effects[OP_REPORT_COUNTER] && !prefix_effects[OP_RESET_COUNTER] && !prefix_effects[OP_CLEAR_DS]);
    ex.fence++;
    unsigned old_calls = calls, old_draws = draw_calls;
    assert(pool_leader_run(&ex, &records[0].h, &records[4].h, &draws) == &records[4].h);
    assert(calls == (int)old_calls + 4 && draw_calls == (int)old_draws + 1);
    assert(prefix_effects[OP_REPORT_COUNTER] == 1 && prefix_effects[OP_RESET_COUNTER] == 1 && prefix_effects[OP_CLEAR_DS] == 1);
    const unsigned unsafe[] = {OP_CLEAR_COLOR, OP_RENDER_COND, OP_JUMP, OP_DRAW_MESH};
    for (unsigned i = 0; i < sizeof unsafe / sizeof *unsafe; i++) {
        record(0, unsafe[i]); record(1, OP_DRAW);
        assert(!pool_leader_draw(&ex, &records[0].h, &records[2].h));
    }
    VkExecPoolRequirement mesh = {0};
    pool_requirement_capture(&mesh, &ex, 1);
    vk_exec_draw_pool_learn(&mesh);
    assert(mesh.valid && !vk_exec_draw_pool_requirement_known(&ex, 1));
    mesh_setting = "1";
    vk_exec_draw_pool_schedule_init();
    assert(vk_exec_draw_pool_requirement_known(&ex, 1));
    native.mesh_fallback = 1;
    assert(!vk_exec_draw_pool_requirement_known(&ex, 1));
    native.mesh_fallback = 0;
    record(0, OP_DRAW_MESH); record(1, OP_END);
    ((p_draw_mesh *)records[0].payload)->count = 1;
    assert(pool_leader_draw(&ex, &records[0].h, &records[1].h) == &records[0].h);
    ((p_draw_mesh *)records[0].payload)->count = 0;
    assert(!pool_leader_draw(&ex, &records[0].h, &records[1].h));
    assert(context.pool_stats.mesh_probed && context.pool_stats.mesh_matched);
    for (unsigned i = 0; i < 130; i++) record(i, OP_BIND_UBO);
    record(130, OP_DRAW); record(131, OP_END);
    assert(!pool_leader_draw(&ex, &records[0].h, &records[131].h));
    early_setting = "0";
    assert(!vk_exec_draw_pool_leader_needed(&ex));
    early_setting = NULL;
    assert(pool_prefix_op_count[OP_RENDER_COND] && pool_prefix_op_count[OP_CLEAR_COLOR]);
    pool_requirement_diagnostics_collect(60);
    assert(!pool_prefix_op_count[OP_RENDER_COND] && !pool_prefix_op_count[OP_CLEAR_COLOR]);
    for (unsigned i = 0; i < 200; i++) pool_requirement_trace_family(1000 + i, 3, 0);
    assert(pool_family_trace_count == 128);
    unsigned logged = diagnostic_logs;
    pool_requirement_trace_family(2000, 3, 0);
    assert(diagnostic_logs == logged); // bounded diagnostics never grow after the cap
    pool_requirement_reset();
    assert(!vk_exec_draw_pool_requirement_known(&ex, 0) && !vk_exec_draw_pool_leader_needed(&ex));
    assert(context.pool_stats.early_reject[VK_POOL_REFUSE_DISABLED]);
    assert(context.pool_stats.early_reject[VK_POOL_REFUSE_FAMILY]);
    assert(context.pool_stats.early_reject[VK_POOL_REFUSE_FORMAT]);
    assert(context.pool_stats.early_reject[VK_POOL_REFUSE_ORDERED]);
    assert(context.pool_stats.early_reject[VK_POOL_REFUSE_LIMIT]);
    puts("pool requirements: scheduling families, current resource resolution, canonical once-only leader, enable/deferral and refusal reasons passed");
    return 0;
}
