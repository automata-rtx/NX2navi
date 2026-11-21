"""Exercise the native mesh marker's combined VMA and host admission."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
VMA_FIXTURE = Path(__file__).with_name("resource_core_vma_check.c")


PREP_STUBS = r"""
volatile int32_t vk_lost;
struct VkMemSpanCapture { unsigned refs; };
typedef struct { unsigned saved[16]; } NativeRmwTail;
typedef struct {
    VkBuffer mirror, import;
    RcRegistryBufferSpan native;
    uint64_t descriptor_size;
    RcRegistryPosition read;
} SelectedMirrorWord;
struct VkMemNativeRmwWord {
    SelectedMirrorWord source;
    VkMemSpan descriptor;
    RcBudgetTicket host;
    VkBuffer marker;
    VmaAllocation allocation;
    volatile uint32_t *mapped;
    VkMemSpanCapture *provider;
    RcRegistryPosition output;
    uint64_t marker_generation;
    unsigned references;
    NativeRmwTail *workspace;
    VkCommandBuffer body;
    MemLazyWriteRange lazy_write;
};
enum { RMW_WORD_LIMIT = 16 };
static struct { unsigned word_count; VkMemNativeRmwWord *scene; uint64_t generation; } RW;
static unsigned host_cancels, flushes;
static int marker_mapped;
static VkResult flush_status;
static uint32_t marker_words[2];
static MemLazyWriteResult lazy_result[2];
static unsigned lazy_attempts, parks, releases, selections, stale_selection;
static int stop_after_work;
static MemLazyWriteRange *owned_lazy;
static PlatMutex pool_lock;

void plat_mutex_lock_shared(PlatMutex *lock) { assert(lock == &pool_lock); }
void plat_mutex_unlock_shared(PlatMutex *lock) { assert(lock == &pool_lock); }
int vk_exec_shared(void) { return 0; }
RcRegistryPosition vk_exec_resource_position(void) { return (RcRegistryPosition){7, 2}; }
int rc_registry_provisional_read_current(RcRegistry *r, RcRegistryPosition read) {
    return !r && !rc_registry_position_compare(read, vk_exec_resource_position());
}
VkSyncProgressResult vk_sync_progress_prepare_pipeline(int (*prepare)(void *), void *context) {
    assert(++parks == 1);
    return prepare(context) && !stop_after_work ? VK_SYNC_PROGRESS_COMPLETE : VK_SYNC_PROGRESS_STOPPED;
}

int vk_sync_stopping(void) { return 0; }
RcRegistry *rc_registry_runtime(void) { return NULL; }
RcRegistryResult rc_registry_buffer_access_check(RcRegistry *registry, RcRegistryBufferSpan span,
                                                 RcRegistryPosition position) {
    assert(!registry && span.native.offset == 4 && position.scene == 7 && position.ordinal == 2);
    return RC_REGISTRY_OK;
}
VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *capture) { capture->refs++; return capture; }
void vk_mem_span_capture_release(VkMemSpanCapture **capture) { (*capture)->refs--; *capture = NULL; }

static RcRegistryResult selected_mirror_word(VkMemSpanCapture *provider, const VkMemSpan *descriptor,
    uint64_t address, uint64_t size, RcRegistryPosition read, SelectedMirrorWord *out) {
    assert(provider->refs && descriptor->pool && address == 0x1004 && size == 64 && read.scene == 7);
    const uint32_t mapping = ++selections == stale_selection ? 12 : 11;
    *out = (SelectedMirrorWord){.mirror = (VkBuffer)(uintptr_t)9, .import = (VkBuffer)(uintptr_t)10,
        .native = {{address, mapping, {13, 4, 4}}, {18, 19, 4}}, .descriptor_size = size, .read = read};
    return RC_REGISTRY_OK;
}
#define selected_mirror_word_locked selected_mirror_word
static int native_rmw_pipeline_ready(VkCtx *c) { return c == &context; }
static int native_rmw_tail_supported(const void *pool, uint64_t offset) { return pool && offset == 4; }
static int native_rmw_word_bitmap_refuses(const void *pool, const SelectedMirrorWord *source, VkCommandBuffer body) {
    assert(pool && source->native.native.offset == 4 && !body);
    return 0; // This admission fixture has no issued bitmap receipt.
}
MemLazyWriteResult mem_lazy_write_acquire(MemLazyWriteRange *range, uint64_t address, uint64_t bytes,
    uint64_t mirror, uint64_t import, uint64_t offset, const void *owner, int parked) {
    assert(range && !range->size && !owned_lazy && owner && ((const VkMemSpanCapture *)owner)->refs == 2);
    assert(address == 0x1004 && bytes == 4 && mirror == 9 && import == 10 && offset == 4);
    assert(lazy_attempts++ == (unsigned)parked);
    if (lazy_result[parked] == MEM_LAZY_WRITE_READY) {
        range->size = bytes;
        owned_lazy = range;
    }
    return lazy_result[parked];
}
void mem_lazy_write_release(MemLazyWriteRange *range) {
    if (!range->size) return;
    assert(owned_lazy == range);
    owned_lazy = NULL;
    range->size = 0;
    releases++;
}
int rt_knob(const char *name, const char *fallback) {
    assert(!strcmp(name, "GRAPHICS_PREP_TRACE") && !strcmp(fallback, "0"));
    return 0;
}
void rt_log(const char *format, ...) { (void)format; assert(0); }
VkResult vmaFlushAllocation(VmaAllocator allocator, VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size) {
    assert(allocator == context.vma && allocation && !offset && size == sizeof marker_words);
    assert(!marker_words[0] && !marker_words[1]);
    flushes++;
    return flush_status;
}

static VkResult mesh_test_allocate_mapped(VkCtx *c, size_t host_bytes, VkDeviceSize marker_bytes, RcBudgetTicket *host,
    RcVmaDependency *dependency, VkBuffer *marker, VmaAllocation *allocation, VmaAllocationInfo *info) {
    assert(marker_bytes == 2 * sizeof(uint32_t));
    VkResult result = native_rmw_marker_allocate(c, host_bytes, marker_bytes, host, dependency, marker, allocation, info);
    if (result == VK_SUCCESS && marker_mapped) info->pMappedData = marker_words;
    return result;
}
static RcAdmissionStatus mesh_test_cancel_host(RcBudgetTicket *host) {
    assert(host->token.ledger);
    host_cancels++;
    return resource_core_budget_cancel(host);
}
#define native_rmw_marker_allocate mesh_test_allocate_mapped
#define resource_core_budget_cancel mesh_test_cancel_host
"""


CASES = r"""
static void release_mesh(VkBuffer marker, VmaAllocation allocation, RcBudgetTicket *host, uint64_t host_bytes) {
    RcBudgetSnapshot usage;
    resource_core_budget_snapshot(&usage);
    assert(usage.reserved.amount[RC_BUDGET_RAM] == host_bytes);
    assert(usage.reserved.amount[RC_BUDGET_COMMIT] == host_bytes);
    const RcAdmissionVector retained = {.amount = {
        [RC_BUDGET_RAM] = host_bytes, [RC_BUDGET_COMMIT] = host_bytes}};
    assert(resource_core_budget_materialize(host, &retained) == RC_ADMISSION_OK);
    resource_core_vma_destroy_buffer(context.vma, marker, allocation);
    assert(resource_core_budget_retire(host) == RC_ADMISSION_OK);
    assert(resource_core_budget_destroyed(host) == RC_ADMISSION_OK);
    resource_core_budget_snapshot(&usage);
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; ++d) {
        assert(!usage.reserved.amount[d]);
        assert(!usage.producer[RC_BUDGET_BUFFER].retained.amount[d]);
        assert(!usage.producer[RC_BUDGET_BUFFER].pending.amount[d]);
    }
}

static void rejected_old_request(void) {
    start(256);
    RcBudgetTicket host = {0};
    RcVmaDependency dep = {0};
    const RcVmaRequest invalid = {.producer = RC_BUDGET_BUFFER, .required = 1,
        .dependency = &dep, .host_bytes = MB, .host_ticket = &host};
    const VkBufferCreateInfo info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 8};
    const VmaAllocationCreateInfo placement = {0};
    VkBuffer marker;
    VmaAllocation allocation;
    RcAdmissionStatus why;
    assert(resource_core_vma_create_buffer(&invalid, context.vma, &info, &placement, &marker,
        &allocation, NULL, &why) == VK_ERROR_INITIALIZATION_FAILED);
    assert(!marker && !allocation && !host.token.ledger && !dep.selected && !native.reuse_calls && !native.new_calls);
    assert(why == RC_ADMISSION_INVALID);
}

static void successful_marker(int reuse) {
    start(256);
    native.type = 1;
    native.reuse = reuse;
    RcBudgetTicket host = {0};
    RcVmaDependency dep = {0};
    VkBuffer marker = VK_NULL_HANDLE;
    VmaAllocation allocation = NULL;
    uint64_t bytes = MB;
    test_reuse_commit = reuse ? 17 : 0;
    assert(native_rmw_marker_allocate(&context, bytes, 2 * sizeof(uint32_t), &host, &dep, &marker, &allocation, NULL) == VK_SUCCESS);
    assert(marker && allocation && host.token.ledger && dep.selected && dep.memory_type == 1);
    assert(dep.reuse.amount[RC_BUDGET_RAM] == native.need + bytes);
    assert(dep.reuse.amount[RC_BUDGET_COMMIT] == bytes + test_reuse_commit);
    assert(dep.peak.amount[RC_BUDGET_COMMIT] == native.expected_growth + bytes);
    assert(native.reuse_calls == 1 && native.new_calls == (unsigned)!reuse);
    release_mesh(marker, allocation, &host, bytes);
    assert(native.destroys == 1 && !host.token.ledger);
    test_reuse_commit = 0;
}

static void refusal_and_retry(void) {
    start(256);
    native.type = 1;
    native.reuse = 1;
    RcBudgetTicket host = {0};
    RcVmaDependency dep = {0};
    VkBuffer marker = VK_NULL_HANDLE;
    VmaAllocation allocation = NULL;
    RcBudgetReadiness before;
    resource_core_budget_readiness(1, &before);
    uint64_t bytes = before.available.amount[RC_BUDGET_COMMIT] + 1;
    assert(native_rmw_marker_allocate(&context, bytes, 2 * sizeof(uint32_t), &host, &dep, &marker, &allocation, NULL) == VK_NOT_READY);
    assert(!marker && !allocation && !host.token.ledger && !native.new_calls);
    assert(dep.selected && dep.peak.amount[RC_BUDGET_COMMIT] == bytes);
    assert(dep.peak.amount[RC_BUDGET_RAM] == native.need + bytes);
    assert(native.reuse_calls == 1 && native.destroys == 1);
    assert(!resource_core_vma_dependency_ready(&dep, &before));
    sample.free.amount[RC_BUDGET_COMMIT] += 1024 * MB;
    sample.free.amount[RC_BUDGET_RAM] += 1024 * MB;
    clock_ns += 1000000000;
    resource_core_budget_refresh();
    RcBudgetReadiness after;
    resource_core_budget_readiness(1, &after);
    assert(resource_core_vma_dependency_ready(&dep, &after));
    assert(native_rmw_marker_allocate(&context, bytes, 2 * sizeof(uint32_t), &host, &dep, &marker, &allocation, NULL) == VK_SUCCESS);
    release_mesh(marker, allocation, &host, bytes);
    assert(native.reuse_calls == 2 && native.destroys == 2 && !host.token.ledger);
}

static void failed_marker(int overflow) {
    start(256);
    native.type = 1;
    native.fail = !overflow;
    test_reuse_commit = overflow ? UINT64_MAX : 0;
    RcBudgetTicket host = {0};
    RcVmaDependency dep = {0};
    VkBuffer marker = VK_NULL_HANDLE;
    VmaAllocation allocation = NULL;
    VkResult expected = overflow ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_ERROR_OUT_OF_DEVICE_MEMORY;
    assert(native_rmw_marker_allocate(&context, MB, 2 * sizeof(uint32_t), &host, &dep, &marker, &allocation, NULL) == expected);
    assert(!marker && !allocation && !host.token.ledger);
    assert(native.new_calls == (unsigned)!overflow && native.reuse_calls == (unsigned)!overflow);
    RcBudgetSnapshot usage;
    resource_core_budget_snapshot(&usage);
    assert(!usage.reserved.amount[RC_BUDGET_RAM] && !usage.reserved.amount[RC_BUDGET_COMMIT]);
    assert(!usage.producer[RC_BUDGET_BUFFER].retained.amount[RC_BUDGET_COMMIT]);
    test_reuse_commit = 0;
}

static void prepare_mapping(int which) {
    start(256);
    native.type = 1;
    native.reuse = 1;
    host_cancels = flushes = lazy_attempts = parks = releases = selections = 0;
    marker_mapped = which != 0;
    marker_words[0] = marker_words[1] = 19;
    flush_status = which == 1 ? VK_ERROR_MEMORY_MAP_FAILED : VK_SUCCESS;
    VkMemSpanCapture capture = {.refs = 1};
    VkMemSpan descriptor = {.pool = &capture};
    RcVmaDependency dep = {0};
    VkMemNativeRmwWord *word = NULL;
    VkResult result = vk_mem_native_rmw_word_prepare(&capture, &descriptor, 64, 0x1004,
        (RcRegistryPosition){7, 2}, &dep, &word);
    assert(flushes == (unsigned)marker_mapped);
    if (which == 2) {
        assert(result == VK_SUCCESS && word && word->references == 1 && word->workspace && capture.refs == 2);
        assert(!host_cancels && !native.destroys && lazy_attempts == 1 && !parks);
        native_rmw_word_release(word);
        assert(releases == 1 && !owned_lazy);
    } else {
        assert(result == (which ? VK_ERROR_MEMORY_MAP_FAILED : VK_ERROR_OUT_OF_HOST_MEMORY));
        assert(!word && host_cancels == 1 && !lazy_attempts && !releases);
    }
    assert(native.destroys == 1 && capture.refs == 1);
    RcBudgetSnapshot usage;
    resource_core_budget_snapshot(&usage);
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; ++d) {
        assert(!usage.reserved.amount[d]);
        assert(!usage.producer[RC_BUDGET_BUFFER].retained.amount[d]);
        assert(!usage.producer[RC_BUDGET_BUFFER].pending.amount[d]);
    }
}

static void prepare_lazy(MemLazyWriteResult first, MemLazyWriteResult parked, unsigned remap_selection,
    int stopped, VkResult expected, unsigned expected_attempts, unsigned expected_releases) {
    start(256);
    native.type = 1;
    native.reuse = 1;
    marker_mapped = 1;
    flush_status = VK_SUCCESS;
    host_cancels = flushes = lazy_attempts = parks = releases = selections = 0;
    lazy_result[0] = first;
    lazy_result[1] = parked;
    stale_selection = remap_selection;
    stop_after_work = stopped;
    VkMemSpanCapture capture = {.refs = 1};
    VkMemSpan descriptor = {.pool = &capture};
    VkMemNativeRmwWord *word = NULL;
    VkResult result = vk_mem_native_rmw_word_prepare(&capture, &descriptor, 64, 0x1004,
        (RcRegistryPosition){7, 2}, NULL, &word);
    assert(result == expected && lazy_attempts == expected_attempts);
    assert(parks == (first == MEM_LAZY_WRITE_BUSY || first == MEM_LAZY_WRITE_NEEDS_FETCH));
    if (result == VK_SUCCESS) {
        assert(owned_lazy == &word->lazy_write);
        // The issued callback and the physical binding each hold a reference. Either can release first.
        word->references++;
        native_rmw_word_release(word);
        assert(owned_lazy == &word->lazy_write && !releases && !native.destroys && capture.refs == 2);
        native_rmw_word_release(word);
    } else assert(!word);
    assert(!owned_lazy && releases == expected_releases && native.destroys == 1 && capture.refs == 1);
    assert(!host_cancels && flushes == 1);
    RcBudgetSnapshot usage;
    resource_core_budget_snapshot(&usage);
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; ++d) {
        assert(!usage.reserved.amount[d]);
        assert(!usage.producer[RC_BUDGET_BUFFER].retained.amount[d]);
        assert(!usage.producer[RC_BUDGET_BUFFER].pending.amount[d]);
    }
}

int main(void) {
    rejected_old_request();
    successful_marker(0);
    successful_marker(1);
    refusal_and_retry();
    failed_marker(0);
    failed_marker(1);
    for (int i = 0; i < 3; ++i) prepare_mapping(i);
    prepare_lazy(MEM_LAZY_WRITE_BUSY, MEM_LAZY_WRITE_READY, 0, 0, VK_SUCCESS, 2, 1);
    prepare_lazy(MEM_LAZY_WRITE_NEEDS_FETCH, MEM_LAZY_WRITE_READY, 0, 0, VK_SUCCESS, 2, 1);
    prepare_lazy(MEM_LAZY_WRITE_FOREIGN, MEM_LAZY_WRITE_READY, 0, 0, VK_ERROR_FEATURE_NOT_PRESENT, 1, 0);
    prepare_lazy(MEM_LAZY_WRITE_BUSY, MEM_LAZY_WRITE_BUSY, 0, 0, VK_ERROR_FEATURE_NOT_PRESENT, 2, 0);
    prepare_lazy(MEM_LAZY_WRITE_NEEDS_FETCH, MEM_LAZY_WRITE_READY, 2, 0, VK_ERROR_FEATURE_NOT_PRESENT, 1, 0);
    prepare_lazy(MEM_LAZY_WRITE_NEEDS_FETCH, MEM_LAZY_WRITE_READY, 3, 0, VK_ERROR_INITIALIZATION_FAILED, 2, 1);
    prepare_lazy(MEM_LAZY_WRITE_NEEDS_FETCH, MEM_LAZY_WRITE_READY, 0, 1, VK_ERROR_FEATURE_NOT_PRESENT, 2, 1);
    puts("native mesh admission: rollback, mapping and flush lifetime, lazy receipt, parking, remap and final release passed");
    return 0;
}
"""


def main():
    compiler = os.environ.get("CC") or shutil.which("clang")
    if not compiler:
        raise SystemExit("Clang is required (set CC)")
    cc = Path(shutil.which(compiler) or compiler).resolve()
    include = Path(os.environ.get("VMA_INCLUDE", cc.parent.parent / "include/vma"))
    memory = (RUNTIME / "vk_memory.c").read_text(encoding="utf-8")
    start = memory.index("static VkResult native_rmw_marker_allocate(")
    end = memory.index("static MemLazyWriteResult native_rmw_word_acquire_lazy(", start)
    allocation = memory[start:end]
    prepare = memory[end:memory.index("int vk_mem_native_rmw_word_order(", end)]
    destroy_start = memory.index("static void native_rmw_word_destroy(")
    destroy = memory[destroy_start:memory.index("static void native_rmw_word_binding_canceled(", destroy_start)]
    # The real VMA reuse query normally has no commit growth. Exercise checked addition even if a query
    # reports an existing commit component, while keeping the real query and allocation paths underneath.
    prefix = f'''
#define main shared_vma_fixture_main
#include "{VMA_FIXTURE.as_posix()}"
#undef main
#include "{(RUNTIME / 'host.h').as_posix()}"

static uint64_t test_reuse_commit;
static VkResult mesh_test_peak(const RcVmaRequest *request, const VkBufferCreateInfo *buffer,
    const VmaAllocationCreateInfo *requested, VmaAllocationCreateInfo *chosen, RcAdmissionVector *peak) {{
    VkResult result = resource_core_vma_buffer_peak(request, buffer, requested, chosen, peak);
    if (result == VK_SUCCESS && request->dependency)
        request->dependency->reuse.amount[RC_BUDGET_COMMIT] = test_reuse_commit;
    return result;
}}
#define resource_core_vma_buffer_peak mesh_test_peak
'''
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="native-mesh-admission-") as temporary:
        source = Path(temporary) / "check.c"
        exe = source.with_suffix(".exe" if os.name == "nt" else "")
        source.write_text(prefix + allocation + "\n#undef resource_core_vma_buffer_peak\n" + PREP_STUBS +
                          "\nstatic void native_rmw_word_release(void *);\n" + prepare +
                          "\n#undef native_rmw_marker_allocate\n#undef resource_core_budget_cancel\n" + destroy + CASES,
                          encoding="utf-8")
        subprocess.run([str(cc), "-std=c11", "-O1", "-fsanitize=undefined", "-fsanitize-trap=undefined",
                        f"-I{include}", str(source), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()
