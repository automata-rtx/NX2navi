// Synthetic pipeline test. Build in place of the runtime's vk_pipeline.c, with this main
// as the entry point (rename main in runtime/main.c when linking its helpers). It creates
// a headless Vulkan device and never loads game files.
#include <assert.h>
#include "../runtime/vk.h"
#include "../runtime/platform/platform.h"

static volatile int32_t attempts, fail_next, active_marks, marks, entered;
static PlatEvent all_entered;
static _Thread_local int marker;
static int64_t capture_write(PlatFile *file, const void *bytes, uint32_t size, uint64_t offset);
#define plat_file_pwrite capture_write
static VKAPI_ATTR VkResult VKAPI_CALL counted_create(VkDevice device, VkPipelineCache cache, uint32_t count,
        const VkComputePipelineCreateInfo *info, const VkAllocationCallbacks *alloc, VkPipeline *out);
static VkCtx synthetic_context;
static const GfxShader *prepared_shader;
static const GfxShader *load_prepared_shader(const uint8_t id[32], int *owned) {
    assert(id[0] == 1);
    *owned = 0;
    return prepared_shader;
}
#define gfx_shader_load_private load_prepared_shader
#define vk_ctx() (&synthetic_context)
#define vkCreateComputePipelines counted_create
#include "../runtime/vk_pipeline.c"
#undef vkCreateComputePipelines
#undef plat_file_pwrite

static int64_t capture_write(PlatFile *file, const void *bytes, uint32_t size, uint64_t offset) {
    if (file != SK.journal) return plat_file_pwrite(file, bytes, size, offset);
    const uint8_t *rec = bytes;
    assert(size == JR_SLOT);
    if (rec[0]) {
        assert(!marker && rec[0] == offset / JR_SLOT + 1 && rec[32] == GFX_STAGE_CS);
        marker = 1;
        plat_inc32(&active_marks);
        plat_inc32(&marks);
    } else {
        assert(marker);
        marker = 0;
        plat_dec32(&active_marks);
    }
    return size;
}

static VKAPI_ATTR VkResult VKAPI_CALL counted_create(VkDevice device, VkPipelineCache cache, uint32_t count,
        const VkComputePipelineCreateInfo *info, const VkAllocationCallbacks *alloc, VkPipeline *out) {
    plat_inc32(&attempts);
    if (plat_xchg32(&fail_next, 0)) {
        assert(plat_event_wait_ns(&all_entered, 5000000000ull));
        assert(marker && __atomic_load_n(&active_marks, __ATOMIC_ACQUIRE) == 1);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return vkCreateComputePipelines(device, cache, count, info, alloc, out);
}

// A one-invocation compute shader with an empty main function.
static const uint32_t module[] = {
    0x07230203, 0x00010000, 0, 5, 0,
    0x20011, 1, 0x3000e, 0, 1, 0x5000f, 5, 3, 0x6e69616d, 0,
    0x60010, 3, 17, 1, 1, 1, 0x20013, 1, 0x30021, 2, 1,
    0x50036, 1, 3, 0, 2, 0x200f8, 4, 0x100fd, 0x10038
};

typedef struct {
    WarmBuilds *table;
    const GfxShader *shader;
    VkPipelineLayout layout;
    int slot;
    VkResult result;
} ComputeTest;

static void run_compute(void *arg) {
    ComputeTest *test = arg;
    VkPipeline pipe;
    GfxWarmEnt variant = {.stage = GFX_STAGE_CS};
    variant.iid[0] = (uint8_t)(test->slot + 1);
    if (plat_inc32(&entered) == 8) plat_event_set(&all_entered);
    Layout layout = {.layout = test->layout};
    test->result = compute_build(test->shader, &layout, 0, &pipe, test->table, test->slot, &variant);
    assert(!pipe);   // offline work leaves the result in the driver cache
}

// A read-only probe must reject another driver and torn binary records. All files here are synthetic.
static void check_cache_files(void) {
    char temp[PLAT_PATH_MAX], dir[PLAT_PATH_MAX * 2], path[PLAT_PATH_MAX * 2 + 32];
    assert(plat_temp_dir(temp, sizeof temp));
    snprintf(dir, sizeof dir, "%s/nx2navi-offline-%u-%llu", temp, plat_process_id(), (unsigned long long)plat_now_ns());
    assert(!plat_fs_mkdir(dir));
    snprintf(path, sizeof path, "%s/pipeline-binaries.bin", dir);
    PbHeader header = {{'N','X','2','N','P','B','I','N'}, PB_VERSION, 1, {42}};
    uint8_t binary[sizeof header + 2 * (4 + PB_HEAD + 8)] = {0};
    memcpy(binary, &header, sizeof header);
    uint32_t record[3] = {PB_HEAD + 8, PB_BLOB, 1};
    memcpy(binary + sizeof header, record, sizeof record);
    record[1] = PB_PIPE;
    memcpy(binary + sizeof header + 4 + record[0], record, sizeof record);
    assert(plat_file_write(path, binary, sizeof binary));
    assert(prepared_binary_bytes(dir, &header) == sizeof binary);
    assert(plat_file_write(path, binary, sizeof binary - 1));
    assert(!prepared_binary_bytes(dir, &header));
    binary[offsetof(PbHeader, gkey)]++;
    assert(plat_file_write(path, binary, sizeof binary));
    assert(!prepared_binary_bytes(dir, &header));
    plat_file_delete(path);

    snprintf(path, sizeof path, "%s/pipeline-cache.bin", dir);
    VkPhysicalDeviceProperties props = {.vendorID = 1, .deviceID = 2, .pipelineCacheUUID = {3}};
    struct { VkPipelineCacheHeaderVersionOne h; uint8_t body[16]; } cache =
        {{sizeof(VkPipelineCacheHeaderVersionOne), VK_PIPELINE_CACHE_HEADER_VERSION_ONE, 1, 2, {3}}, {4}};
    assert(plat_file_write(path, &cache, sizeof cache));
    PlatFile *f = plat_file_open(path);
    assert(f && cache_file_size(f, &props) == sizeof cache);
    props.pipelineCacheUUID[0]++;
    assert(!cache_file_size(f, &props));
    const void *view = plat_file_map(f);
    plat_file_close(f);
    assert(view && !memcmp(view, &cache, sizeof cache));
    plat_file_unmap(view, sizeof cache);
    plat_file_delete(path);
    assert(!plat_fs_rmdir(dir));
}

// Exercise WARM_ALL with a one-stage manifest. An actual driver create failure must reach the caller
// that writes the receipt, even when the archive contains no graphics programs.
static void check_failed_warm(const GfxShader *shader, VkPipelineLayout layout) {
    char temp[PLAT_PATH_MAX], dir[PLAT_PATH_MAX * 2], path[PLAT_PATH_MAX * 2 + 32];
    assert(plat_temp_dir(temp, sizeof temp));
    snprintf(dir, sizeof dir, "%s/nx2navi-warm-%u-%llu", temp, plat_process_id(), (unsigned long long)plat_now_ns());
    assert(!plat_fs_mkdir(dir));
    snprintf(path, sizeof path, "%s/shader-prepared.bin", dir);
    struct { char magic[8]; uint32_t version, stride; uint8_t record[36]; } manifest = {"NX2NSPRP", 1, 36, {GFX_STAGE_CS}};
    manifest.record[4] = 1;
    assert(plat_file_write(path, &manifest, sizeof manifest));
    snprintf(P.cache_path, sizeof P.cache_path, "%s/pipeline-cache.bin", dir);
    Bindings bindings = {0};
    uint64_t hash = FNV0;
    assert(add_stage(&bindings, shader, 0) && bindings_sort(&bindings, &hash) > 0);
    Layout *entry = calloc(1, sizeof *entry + bindings.n * sizeof *bindings.b);
    assert(entry);
    entry->hash = hash, entry->n = bindings.n, entry->layout = layout;
    memcpy(entry->b, bindings.b, bindings.n * sizeof *bindings.b);
    assert(!layouts[hash % LAYOUT_BUCKETS]);
    layouts[hash % LAYOUT_BUCKETS] = entry;
    prepared_shader = shader;
    int previous = attempts;
    fail_next = 1;
    assert(lb_warm_prepared(dir, WARM_ALL) != 0);
    assert(attempts == previous + 1 && LB.fail[2] == 1 && !active_marks);
    layouts[hash % LAYOUT_BUCKETS] = NULL;
    free(entry);
    plat_file_delete(path);
    assert(!plat_fs_rmdir(dir));
}

int main(void) {
    check_cache_files();
    VkInstanceCreateInfo ic = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    VkInstance instance;
    assert(vkCreateInstance(&ic, NULL, &instance) == VK_SUCCESS);
    uint32_t n = 1;
    VkPhysicalDevice phys;
    VkResult r = vkEnumeratePhysicalDevices(instance, &n, &phys);
    assert((r == VK_SUCCESS || r == VK_INCOMPLETE) && n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, NULL);
    VkQueueFamilyProperties *families = calloc(n, sizeof *families);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, families);
    uint32_t family = 0;
    while (family < n && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;
    assert(family < n);
    free(families);
    float priority = 1;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
                                  .queueFamilyIndex = family, .pQueuePriorities = &priority};
    VkDeviceCreateInfo dc = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
    assert(vkCreateDevice(phys, &dc, NULL, &P.device) == VK_SUCCESS);
    VkPipelineLayoutCreateInfo lc = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout layout, other_layout;
    assert(vkCreatePipelineLayout(P.device, &lc, NULL, &layout) == VK_SUCCESS);
    VkPushConstantRange push = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 4};
    lc.pushConstantRangeCount = 1, lc.pPushConstantRanges = &push;
    assert(vkCreatePipelineLayout(P.device, &lc, NULL, &other_layout) == VK_SUCCESS);
    WarmBuilds table = {0};
    uint32_t different[sizeof module / 4];
    memcpy(different, module, sizeof module);
    different[18] = 2;   // LocalSize X
    const GfxShader *a = gfx_shader_own(module, sizeof module / 4, "{\"stage\":\"cs\"}");
    const GfxShader *b = gfx_shader_own(module, sizeof module / 4, "{ \"stage\":\"cs\" }");
    const GfxShader *c = gfx_shader_own(different, sizeof different / 4, "{\"stage\":\"cs\"}");
    assert(a && b && c && memcmp(gfx_shader_key(a), gfx_shader_key(b), 32));
    ComputeTest jobs[8];
    PlatThread *threads[8];
    SK.journal = (PlatFile *)(uintptr_t)1;
    plat_event_init(&all_entered, 1, 0);
    fail_next = 1;
    for (int i = 0; i < 8; i++) {
        jobs[i] = (ComputeTest){.table = &table, .shader = i & 1 ? a : b, .layout = layout, .slot = i};
        threads[i] = plat_thread_new(run_compute, &jobs[i], "compute sharing check");
        assert(threads[i]);
    }
    int failures = 0;
    for (int i = 0; i < 8; i++) {
        plat_thread_join(threads[i]);
        failures += jobs[i].result != VK_SUCCESS;
    }
    assert(failures == 1 && attempts == 2 && table.shared == 6);
    ComputeTest next = {.table = &table, .shader = c, .layout = layout};
    run_compute(&next);
    assert(next.result == VK_SUCCESS && attempts == 3);
    next.shader = a, next.layout = other_layout;
    run_compute(&next);
    assert(next.result == VK_SUCCESS && attempts == 4);
    uint32_t declared[sizeof module / 4 + 2];
    memcpy(declared, module, 7 * 4);
    declared[7] = 0x20011, declared[8] = 5378;   // unused pixel-interlock capability stripped by prepare_module_words
    memcpy(declared + 9, module + 7, sizeof module - 7 * 4);
    const GfxShader *alias = gfx_shader_own(declared, sizeof declared / 4, "{\"stage\":\"cs\"}");
    assert(alias);
    next.shader = alias, next.layout = layout;
    run_compute(&next);
    assert(next.result == VK_SUCCESS && attempts == 4 && marks == 4 && !active_marks);
    Layout live_layout = {.layout = layout};
    VkPipeline live = compute_create(a, &live_layout, 0);
    assert(live && attempts == 5);
    vkDestroyPipeline(P.device, live, NULL);

    WarmComputeKey k, other;
    VkComputePipelineCreateInfo ci = {.stage = {.stage = VK_SHADER_STAGE_COMPUTE_BIT, .pName = "main"}, .layout = layout};
    assert(warm_compute_key(&k, module, sizeof module / 4, &ci));
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sg = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, .requiredSubgroupSize = 32};
    ci.stage.pNext = &sg;
    assert(warm_compute_key(&other, module, sizeof module / 4, &ci) && memcmp(&k, &other, sizeof k));
    ci.stage.pNext = NULL, ci.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    assert(warm_compute_key(&other, module, sizeof module / 4, &ci) && memcmp(&k, &other, sizeof k));
    VkSpecializationInfo spec = {0};
    ci.stage.pSpecializationInfo = &spec;
    assert(!warm_compute_key(&other, module, sizeof module / 4, &ci));

    assert(!active_marks && marks == 4);
    SK.journal = NULL;
    check_failed_warm(a, layout);
    WarmBuilds graphics = {0};
    P.dyn_vi = 1;
    VkPipeState state;
    vk_pipe_state_defaults(&state);
    GKey gk;
    gkey_fill(&state, &gk);
    WarmBuild *owner;
    assert(warm_claim(&graphics, &gk, sizeof gk, &owner) && owner);
    warm_complete(&graphics, owner, 1);
    state.raster_discard = !state.raster_discard;   // dynamic state has one canonical driver key
    gkey_fill(&state, &gk);
    assert(!warm_claim(&graphics, &gk, sizeof gk, &owner));
    state.samples = 4;
    gkey_fill(&state, &gk);
    assert(warm_claim(&graphics, &gk, sizeof gk, &owner) && owner);
    warm_complete(&graphics, owner, 1);
    assert(graphics.shared == 1);
    warm_builds_free(&graphics);
    warm_builds_free(&table);
    gfx_shader_release_private(alias);
    gfx_shader_release_private(a), gfx_shader_release_private(b), gfx_shader_release_private(c);
    vkDestroyPipelineLayout(P.device, other_layout, NULL);
    vkDestroyPipelineLayout(P.device, layout, NULL);
    vkDestroyDevice(P.device, NULL);
    vkDestroyInstance(instance, NULL);
    puts("offline pipeline check: concurrent aliases, failure retry, module fixups, layouts, subgroup flags, journal owners, dynamic graphics states, live compute, failed warm propagation and cache file validation passed");
    return 0;
}
