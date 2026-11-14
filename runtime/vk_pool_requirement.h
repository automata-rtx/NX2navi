// Included after the pool's ordered synchronization helpers are declared. The cache is replay-thread owned;
// workers write only their retired group's first-draw record, never the cache itself.
#define POOL_REQUIREMENT_SETS 32u
#define POOL_REQUIREMENT_WAYS 2u
static VkExecPoolRequirement pool_requirements[POOL_REQUIREMENT_SETS][POOL_REQUIREMENT_WAYS];
static uint8_t pool_requirement_next[POOL_REQUIREMENT_SETS];
static int pool_requirement_any;
static uint64_t pool_prefix_op_count[OP_COUNT_];
static struct { uint64_t serial; int stages, learned; } pool_family_trace[128];
static unsigned pool_family_trace_count;

static int pool_requirement_trace_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *value = cfg_get("NX2NAVI_POOL_EARLY_TRACE");
        on = value && !strcmp(value, "1");
    }
    return on;
}

static void pool_requirement_trace_family(uint64_t serial, int stages, int learned) {
    if (!pool_requirement_trace_on()) return;
    for (unsigned i = 0; i < pool_family_trace_count; i++)
        if (pool_family_trace[i].serial == serial && pool_family_trace[i].stages == stages &&
            pool_family_trace[i].learned == learned) return;
    if (pool_family_trace_count == 128) return;
    unsigned i = pool_family_trace_count++;
    pool_family_trace[i].serial = serial, pool_family_trace[i].stages = stages, pool_family_trace[i].learned = learned;
    rt_log("pool family: %s serial %llu stages %d", learned ? "learned" : "probed", (unsigned long long)serial, stages);
}

void vk_exec_draw_pool_prefix_op(uint32_t opcode) {
    if (pool_requirement_trace_on() && opcode < OP_COUNT_) pool_prefix_op_count[opcode]++;
}

static void pool_requirement_diagnostics_collect(uint64_t frame) {
    if (!pool_requirement_trace_on()) return;
    for (unsigned op = 0; op < OP_COUNT_; op++) if (pool_prefix_op_count[op]) {
        rt_log("pool prefix: frame %llu: opcode %u (%s) refused %llu", (unsigned long long)frame, op,
               nvn_op_name(op), (unsigned long long)pool_prefix_op_count[op]);
        pool_prefix_op_count[op] = 0;
    }
}

static int pool_early_on(void) {
    const char *value = cfg_get("NX2NAVI_POOL_EARLY");
    return !value || strcmp(value, "0") != 0;
}

// Moving a mesh draw onto the submitting thread changes how much preparation overlaps worker recording.
// Keep this scheduling experiment independently selectable.
static int pool_mesh_leader;
void vk_exec_draw_pool_schedule_init(void) {
    const char *value = cfg_get("NX2NAVI_POOL_MESH_LEADER");
    pool_mesh_leader = value && !strcmp(value, "1");
}

static uint32_t pool_requirement_index(uint64_t serial) {
    return (uint32_t)((serial * 0x9e3779b97f4a7c15ull) >> 59);
}

static void pool_requirement_reset(void) {
    memset(pool_requirements, 0, sizeof pool_requirements);
    memset(pool_requirement_next, 0, sizeof pool_requirement_next);
    pool_requirement_any = 0;
    memset(pool_prefix_op_count, 0, sizeof pool_prefix_op_count);
    pool_family_trace_count = 0;
}

static int pool_requirement_family(const VkExecPoolRequirement *r, const VkExec *ex, int direct_mesh) {
    return r->valid && r->direct_mesh == direct_mesh && ex->prog && r->program == ex->prog &&
           r->serial == ex->prog->serial && r->stages == ex->prog_stages;
}

static int pool_requirement_matches(const VkExecPoolRequirement *r, const VkExec *ex, int direct_mesh) {
    if (!pool_requirement_family(r, ex, direct_mesh)) return 0;
    for (unsigned i = 0; i < 16; i++) {
        va_st attribute;
        memcpy(&attribute, &ex->pipe.vattr[i], sizeof attribute);
        if (r->vertex_format[i] != attribute.fmt) return 0;
    }
    if (memcmp(r->color_format, ex->pipe.color_fmt, sizeof r->color_format)) return 0;
    return 1;
}

static void pool_requirement_capture(VkExecPoolRequirement *r, const VkExec *ex, int direct_mesh) {
    r->valid = 0;
    if (!ex->prog || !ex->prog->serial || vk_texture_deferred()) return;
    memset(r, 0, sizeof *r);
    r->program = ex->prog, r->serial = ex->prog->serial;
    r->stages = ex->prog_stages;
    r->direct_mesh = (uint8_t)direct_mesh;
    for (unsigned i = 0; i < 16; i++) {
        va_st attribute;
        memcpy(&attribute, &ex->pipe.vattr[i], sizeof attribute);
        r->vertex_format[i] = (uint8_t)attribute.fmt;
    }
    memcpy(r->color_format, ex->pipe.color_fmt, sizeof r->color_format);
    r->valid = 1; // chunk completion, not this byte, publishes the record to the replay thread
}

void vk_exec_draw_pool_learn(const VkExecPoolRequirement *r) {
    if (!r->valid) return;
    pool_requirement_trace_family(r->serial, r->stages, 1);
    uint32_t index = pool_requirement_index(r->serial);
    VkExecPoolRequirement *set = pool_requirements[index];
    for (unsigned way = 0; way < POOL_REQUIREMENT_WAYS; way++)
        if (!memcmp(&set[way], r, sizeof *r)) return;
    unsigned way = !set[0].valid ? 0 : !set[1].valid ? 1 : pool_requirement_next[index]++ % POOL_REQUIREMENT_WAYS;
    set[way] = *r;
    pool_requirement_any = 1;
    dc->pool_stats.early_learned++;
}

int vk_exec_draw_pool_leader_needed(VkExec *ex) {
    if (!pool_requirement_any || !resource_prep_mode() || !pool_early_on() || !pool_on()) return 0;
    return PL.cur_fence != ex->fence || !PL.ready || (PL.sealed && !pool_current()) || !vk_exec_pool_generation();
}

int vk_exec_draw_pool_requirement_known(VkExec *ex, int direct_mesh) {
    DrawCtx *cx = ex->dc ? ex->dc : dc;
    if (direct_mesh && (!pool_mesh_leader || vk_ctx()->mesh_fallback || !vk_ctx()->has_mesh_shader)) return 0;
    if (direct_mesh) cx->pool_stats.mesh_probed++;
    cx->pool_stats.early_probes++;
    if (ex->prog) pool_requirement_trace_family(ex->prog->serial, ex->prog_stages, 0);
    int reason = !ex->prog || !ex->prog->serial ? VK_POOL_REFUSE_FAMILY :
                 ex->render_disabled ? VK_POOL_REFUSE_DISABLED :
                 cx->sf_mvprev || n_probe_sh || vk_texture_deferred() ? VK_POOL_REFUSE_SPECIAL : 0;
    if (reason) {
        cx->pool_stats.early_reject[reason]++;
        return cx->pool_stats.early_misses++, 0;
    }
    const VkExecPoolRequirement *set = pool_requirements[pool_requirement_index(ex->prog->serial)];
    int known = 0, family = 0;
    for (unsigned way = 0; way < POOL_REQUIREMENT_WAYS; way++) {
        family |= pool_requirement_family(&set[way], ex, direct_mesh);
        if (pool_requirement_matches(&set[way], ex, direct_mesh)) { known = 1; break; }
    }
    if (!known) {
        cx->pool_stats.early_reject[family ? VK_POOL_REFUSE_FORMAT : VK_POOL_REFUSE_FAMILY]++;
        return cx->pool_stats.early_misses++, 0;
    }
    cx->pool_stats.early_known++;
    if (direct_mesh) cx->pool_stats.mesh_matched++;
    return 1;
}

void vk_exec_draw_pool_leader_finished(VkExec *ex) {
    DrawCtx *cx = ex->dc ? ex->dc : dc;
    cx->pool_stats.early_leaders++;
    if (!vk_texture_deferred() && PL.cur_fence == ex->fence && PL.ready && !PL.sealed && vk_exec_pool_generation())
        cx->pool_stats.early_prepared++;
}

void vk_exec_draw_pool_refused(VkExec *ex, int reason) {
    DrawCtx *cx = ex->dc ? ex->dc : dc;
    if (reason) cx->pool_stats.early_prefix++;
    else cx->pool_stats.early_prior++;
    if ((unsigned)reason < VK_POOL_REFUSE_COUNT) cx->pool_stats.early_reject[reason]++;
}
