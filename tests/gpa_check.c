// runtime/vk_gpa.c: the device address table against a model of what a shader reads through it, with the allocator,
// the device and the completion queue replaced by the stand-ins below. Built and run by tests/gpa_check.py; prints each
// failed check and exits 1 on any.
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "vk_gpa.h"
#include "resource_host_write.h"
#include "resource_core_vma.h"
#include "platform/platform.h"

static int fails;
#define CHECK(c, ...) do { if (!(c) && fails++ < 40) { printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

// ---- stand-ins for what vk_gpa.c links against
static VkCtx ctx = {.has_buffer_device_address = 1, .vma = (VmaAllocator)(uintptr_t)1};
static RcBudgetSample budget_sample;
static uint64_t budget_clock_ns = 1000000000;
uint64_t rc_budget_test_now_ns(void) { return budget_clock_ns; }
void rc_budget_test_lock(void) {}
void rc_budget_test_unlock(void) {}
void rc_budget_test_sample(RcBudgetSample *out) { *out = budget_sample; }
void rc_budget_test_wait(uint32_t milliseconds) { (void)milliseconds; abort(); }
int plat_commit_counts_vram(void) { return 1; }
VkDeviceSize vk_vma_block_size(void) { return 64u << 20; }
VkCtx *vk_ctx(void) { return &ctx; }
static int gpa_diagnostic_disabled;
const char *cfg_get(const char *name) {
    return gpa_diagnostic_disabled && !strcmp(name, "NX2NAVI_GPA") ? "0" : NULL;
}
void plat_mutex_lock(PlatMutex *m) { (void)m; }
void plat_mutex_unlock(PlatMutex *m) { (void)m; }
void plat_cond_init(PlatCond *cond) {
    CHECK(!cond->p, "registry condition is initialized once");
    cond->p = cond;
}
int plat_cond_wait_ns(PlatCond *cond, PlatMutex *mutex, uint64_t ns) {
    (void)cond;
    (void)mutex;
    (void)ns;
    fputs("gpa_check: unexpected GPU capture wait\n", stderr);
    abort();
}
void plat_cond_broadcast(PlatCond *cond) {
    CHECK(cond->p == cond, "registry condition is initialized before notification");
}
void plat_once(PlatOnce *o, void (*fn)(void *), void *arg) {
    if (!o->state) {
        fn(arg);
        o->state = 1;
    }
}
uint64_t plat_now_ns(void) { static uint64_t now; return ++now; }

static jmp_buf fatal_jump;
static int expect_required_fatal;
void rt_fatal(const char *fmt, ...) {
    if (expect_required_fatal && strstr(fmt, "required device address table was not prepared"))
        longjmp(fatal_jump, 1);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    abort();
}

RcBackingResult mem_backing_capture(uint64_t a, uint64_t n, MemBackingCapture **out) {
    (void)a; (void)n; *out = NULL; return RC_BACKING_ABSENT;
}
const RcInterval *mem_backing_intervals(const MemBackingCapture *capture, uint32_t *count) {
    (void)capture; *count = 0; return NULL;
}
void mem_backing_release(MemBackingCapture **capture) { *capture = NULL; }

// Test the real table owner. These adapters stand in only for the selected pool span pin.
struct VkMemSpanCapture { uint64_t address, size; };
static RcInterval donor_mapping;
static VkBuffer donor_mirror, donor_import;
static unsigned donor_refs;
RcRegistryResult vk_mem_span_native_input(uint64_t address, uint64_t size, int clamp, VkMemSpan *span,
                                          VkMemSpanCapture **hold) {
    *hold = NULL;
    uint64_t start = donor_mapping.address;
    if (clamp || !donor_mirror || address < start || address - start > donor_mapping.range.size ||
        size > donor_mapping.range.size - (address - start)) return RC_REGISTRY_UNSUPPORTED;
    VkMemSpanCapture *capture = malloc(sizeof *capture);
    if (!capture) return RC_REGISTRY_CAPACITY;
    *capture = (VkMemSpanCapture){address, size};
    *span = (VkMemSpan){.buffer = donor_mirror, .offset = address - start, .avail = size};
    *hold = capture;
    donor_refs++;
    return RC_REGISTRY_OK;
}
void vk_mem_span_capture_release(VkMemSpanCapture **hold) {
    if (!*hold) return;
    free(*hold);
    *hold = NULL;
    donor_refs--;
}
uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *hold, uint64_t address, uint64_t size, RcInterval **out) {
    if (!hold || address != hold->address || size != hold->size) { *out = NULL; return 0; }
    return resource_host_interval_slice(&donor_mapping, 1, address, size, out);
}
int vk_mem_span_gpa_donor(const VkMemSpanCapture *hold, const VkMemSpan *span,
                          RcRegistryBufferLocation *location, VkBuffer *twin) {
    CHECK(hold && span->buffer == donor_mirror, "same selected native mirror");
    *location = (RcRegistryBufferLocation){31, 3, span->offset};
    *twin = donor_import;
    return 1;
}

static int out_of_blocks_lines;
static char last_gpa_blocker[512];
void perf_size(PerfSizes *s, const char *name, uint64_t count, uint64_t limit, uint64_t bytes) {
    (void)s, (void)name, (void)count, (void)limit, (void)bytes;   // vk_gpa_sizes is not called here
}

void rt_log(const char *fmt, ...) {
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    if (strstr(line, "out of L2 blocks")) out_of_blocks_lines++;
    if (strstr(line, "vk_gpa: source blocker")) snprintf(last_gpa_blocker, sizeof last_gpa_blocker, "%s", line);
}

// The native model keeps the ledger and placement adapter live during the GPA fixture.
enum { L1_WORD = 16, L1_N = 65536, L2_WORD = L1_WORD + L1_N, L2_N = 4096, BLOCKS = 1024, RECS = 65536 };
#define MiB (1ull << 20)
#define TABLE_BYTES ((L2_WORD + BLOCKS * L2_N + RECS * 4) * sizeof(uint32_t))
#define HOST_BYTES (TABLE_BYTES + RECS * sizeof(uint32_t))
static const uint32_t *table;
static uintptr_t next_handle = 0x1000;
static unsigned create_calls, attempt_base, destroys, fail_create, fail_map;
static int fail_address;

VKAPI_ATTR void VKAPI_CALL vkGetDeviceBufferMemoryRequirements(VkDevice device,
    const VkDeviceBufferMemoryRequirements *query, VkMemoryRequirements2 *out) {
    (void)device;
    out->memoryRequirements = (VkMemoryRequirements){query->pCreateInfo->size, 256, 3};
}
VkResult vmaFindMemoryTypeIndexForBufferInfo(VmaAllocator a, const VkBufferCreateInfo *bi,
    const VmaAllocationCreateInfo *ai, uint32_t *type) {
    (void)a; (void)bi;
    *type = !!(ai->flags & VMA_ALLOCATION_CREATE_MAPPED_BIT);
    CHECK(ai->memoryTypeBits & (1u << *type), "the native placement uses an eligible memory type");
    return VK_SUCCESS;
}
VkResult vmaCreateBuffer(VmaAllocator a, const VkBufferCreateInfo *bi, const VmaAllocationCreateInfo *ai, VkBuffer *buf,
                         VmaAllocation *alloc, VmaAllocationInfo *info) {
    (void)a;
    unsigned call = ++create_calls - attempt_base;
    const uint32_t memory_type = call < 3;
    CHECK((ai->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) && ai->memoryTypeBits == (1u << memory_type),
          "native call %u keeps its bounded, selected placement", call);
    if (call == 1) {
        RcBudgetSnapshot snapshot;
        resource_core_budget_snapshot(&snapshot);
        CHECK(snapshot.reserved.amount[RC_BUDGET_RAM] == HOST_BYTES + TABLE_BYTES + 4096 &&
              snapshot.reserved.amount[RC_BUDGET_COMMIT] == HOST_BYTES + TABLE_BYTES + 8192 &&
              snapshot.reserved.amount[RC_BUDGET_GPU_FIRST] == 4096 &&
              snapshot.reserved.amount[RC_BUDGET_GPU_FIRST + 1] == TABLE_BYTES + 4096,
              "the first native create holds the whole native and host peak");
    }
    if (call == fail_create) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    void *mem = calloc(1, (size_t)bi->size);
    if (!mem) return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (call == 1) table = mem;
    *buf = (VkBuffer)(next_handle += 0x1000);
    *alloc = (VmaAllocation)mem;
    *info = (VmaAllocationInfo){.size = bi->size, .memoryType = memory_type,
        .pMappedData = memory_type && call != fail_map ? mem : NULL};
    return VK_SUCCESS;
}
void vmaDestroyBuffer(VmaAllocator a, VkBuffer buf, VmaAllocation alloc) {
    (void)a; (void)buf;
    if ((const void *)table == (const void *)alloc) table = NULL;
    destroys++;
    free(alloc);
}

// The linked adapter supports other allocation kinds. This fixture must only request buffers.
VKAPI_ATTR void VKAPI_CALL vkGetDeviceImageMemoryRequirements(VkDevice device,
    const VkDeviceImageMemoryRequirements *query, VkMemoryRequirements2 *out) { (void)device; (void)query; (void)out; abort(); }
VkResult vmaFindMemoryTypeIndexForImageInfo(VmaAllocator a, const VkImageCreateInfo *i,
    const VmaAllocationCreateInfo *c, uint32_t *type) { (void)a; (void)i; (void)c; (void)type; abort(); }
VkResult vmaFindMemoryTypeIndex(VmaAllocator a, uint32_t bits, const VmaAllocationCreateInfo *c, uint32_t *type) {
    (void)a; (void)bits; (void)c; (void)type; abort();
}
VkResult vmaCreatePool(VmaAllocator a, const VmaPoolCreateInfo *c, VmaPool *out) { (void)a; (void)c; (void)out; abort(); }
VkResult vmaCreateImage(VmaAllocator a, const VkImageCreateInfo *i, const VmaAllocationCreateInfo *c,
    VkImage *out, VmaAllocation *allocation, VmaAllocationInfo *info) {
    (void)a; (void)i; (void)c; (void)out; (void)allocation; (void)info; abort();
}
VkResult vmaAllocateMemory(VmaAllocator a, const VkMemoryRequirements *r, const VmaAllocationCreateInfo *c,
    VmaAllocation *allocation, VmaAllocationInfo *info) { (void)a; (void)r; (void)c; (void)allocation; (void)info; abort(); }
void vmaDestroyImage(VmaAllocator a, VkImage image, VmaAllocation allocation) { (void)a; (void)image; (void)allocation; abort(); }
void vmaFreeMemory(VmaAllocator a, VmaAllocation allocation) { (void)a; (void)allocation; abort(); }

// A buffer's device address: its handle far above the GPU addresses the tests map.
static uint64_t dev_of(VkBuffer b) { return (uint64_t)(uintptr_t)b << 28; }
VkDeviceAddress vkGetBufferDeviceAddress(VkDevice d, const VkBufferDeviceAddressInfo *i) {
    (void)d;
    return fail_address ? 0 : dev_of(i->buffer);
}

static void budget_setup(int refuse) {
    ctx.props.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    ctx.mem_props.memoryTypeCount = ctx.mem_props.memoryHeapCount = 2;
    ctx.mem_props.memoryTypes[0] = (VkMemoryType){VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
    ctx.mem_props.memoryTypes[1] = (VkMemoryType){VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1};
    ctx.mem_props.memoryHeaps[0] = (VkMemoryHeap){2048 * MiB, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    ctx.mem_props.memoryHeaps[1] = (VkMemoryHeap){2048 * MiB, 0};
    budget_sample.known_mask = (1u << (RC_BUDGET_GPU_FIRST + 2)) - 1;
    for (unsigned d = 0; d < RC_BUDGET_GPU_FIRST + 2; d++) {
        budget_sample.total.amount[d] = 2048 * MiB;
        budget_sample.free.amount[d] = 1024 * MiB;
    }
    // This fits all three native buffers. The companion shadow makes the joint peak exceed the commit allowance.
    if (refuse) budget_sample.free.amount[RC_BUDGET_COMMIT] = 128 * MiB + TABLE_BYTES + 8192;
}

static void check_storage_budget(int live) {
    RcBudgetSnapshot snapshot;
    resource_core_budget_snapshot(&snapshot);
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++)
        CHECK(!snapshot.reserved.amount[d] && !snapshot.pending.amount[d], "domain %u has no unsettled owner", d);
    CHECK(snapshot.retained.amount[RC_BUDGET_RAM] == (live ? HOST_BYTES + TABLE_BYTES + 4096 : 0) &&
          snapshot.retained.amount[RC_BUDGET_COMMIT] == (live ? HOST_BYTES + TABLE_BYTES + 8192 : 0) &&
          snapshot.retained.amount[RC_BUDGET_GPU_FIRST] == (live ? 4096 : 0) &&
          snapshot.retained.amount[RC_BUDGET_GPU_FIRST + 1] == (live ? TABLE_BYTES + 4096 : 0),
          "the native and CPU storage are each charged exactly once while retained");
}

// The completion queue: a callback runs once the GPU is past its submission (gpu_done).
static struct { uint64_t fence; void (*fn)(void *); void *arg; } queue[1 << 16];
static int nqueue;
static uint64_t issued;
uint64_t vk_sync_issued(void) { return issued; }
void vk_completion_add(uint64_t fence, void (*fn)(void *), void *arg) {
    if (nqueue == (int)(sizeof queue / sizeof *queue)) { printf("FAIL: completion queue full\n"); exit(1); }
    queue[nqueue].fence = fence, queue[nqueue].fn = fn, queue[nqueue].arg = arg;
    nqueue++;
}
static void gpu_done(void) {   // the submission being recorded is issued and finishes
    issued++;
    int k = 0;
    for (int i = 0; i < nqueue; i++)
        if (queue[i].fence <= issued) queue[i].fn(queue[i].arg);
        else queue[k++] = queue[i];
    nqueue = k;
}

// ---- the table as a shader reads it (the layout at the top of vk_gpa.c)
static uint32_t block_of(uint64_t addr) { return table[L1_WORD + (addr >> 24)]; }
static uint32_t record_of(uint64_t addr) {
    uint32_t blk = block_of(addr);
    return blk ? table[L2_WORD + blk * L2_N + ((addr >> 12) & (L2_N - 1))] : 0;
}
static uint64_t read_record(uint64_t addr, uint32_t rec, int twin) {
    if (!rec) return 0;
    const uint32_t *r = table + table[2] + rec * 4 + (twin ? 2 : 0);
    return ((uint64_t)r[0] | (uint64_t)r[1] << 32) + addr;
}
static uint64_t read_through(uint64_t addr, int twin) {
    return read_record(addr, record_of(addr), twin);
}

static VkBuffer new_buf(void) { return (VkBuffer)(next_handle += 0x1000); }

// Every page of [lo, hi) reads buf (whose byte 0 is at buf_lo), or the zero buffer when buf is none.
static void expect(uint64_t lo, uint64_t hi, VkBuffer buf, uint64_t buf_lo, const char *what) {
    for (uint64_t a = lo; a < hi; a += 4096) {
        uint64_t got = read_through(a, 0), want = buf ? dev_of(buf) + (a - buf_lo) : 0;
        if (got != want) {
            CHECK(0, "%s: %#llx reads %#llx, expected %#llx", what, (unsigned long long)a, (unsigned long long)got,
                  (unsigned long long)want);
            return;
        }
    }
}

static uint32_t cover_rec;
static uint64_t cover_lo, cover_hi;
static uint32_t cover(uint64_t a) { return a >= cover_lo && a + 4096 <= cover_hi ? cover_rec : 0; }

static VkGpaVirtualBinding virtual_binding(uint64_t lo, uint64_t hi, VkBuffer buffer, uint64_t offset) {
    RcSourceRange coverage = buffer ? (RcSourceRange){(uintptr_t)buffer, offset, hi - lo} : (RcSourceRange){0};
    return (VkGpaVirtualBinding){.lo=lo, .hi=hi, .buffer=buffer, .backing=coverage};
}

static void test_virtual_map(uint64_t lo, uint64_t hi, VkBuffer buffer, uint64_t base) {
    if (!buffer) { vk_gpa_virtual_unmap(lo, hi); return; }
    VkGpaVirtualBinding binding = virtual_binding(lo, hi, buffer, lo - base);
    VkGpaVirtualUpdate update;
    RcBackingResult result = vk_gpa_virtual_prepare(&binding, 1, &update);
    CHECK(result == RC_BACKING_OK, "prepare virtual alias");
    if (result == RC_BACKING_OK) vk_gpa_virtual_commit(&update);
}

static void virtual_projection(void) {
    const uint64_t page = 4096;
    const uint64_t lo = 0x3700000000ull + 16 * MiB - 2 * page;
    const uint64_t end = lo + 6 * page, hole = lo + 2 * page;
    const uint64_t first_offset = 9 * page, second_offset = 4 * page;
    const VkBuffer first = new_buf(), second = new_buf(), adjacent = new_buf();

    test_virtual_map(lo - page, lo, adjacent, lo - page);
    test_virtual_map(lo, end, first, lo - first_offset);
    uint32_t first_rec = record_of(lo);
    CHECK(first_rec && record_of(end - page) == first_rec, "one virtual range spans two L2 blocks");
    expect(lo, end, first, lo - first_offset, "initial virtual buffer");

    test_virtual_map(hole - page, hole + page, second, hole - page - second_offset);
    uint32_t second_rec = record_of(hole);
    CHECK(second_rec && second_rec != first_rec, "overlapping replacement has a fresh record");
    expect(lo - page, lo, adjacent, lo - page, "adjacent virtual pool");
    expect(lo, lo + page, first, lo - first_offset, "first alias before overlay");
    expect(hole - page, hole + page, second, hole - page - second_offset, "cross-block overlay");
    expect(hole + page, end, first, lo - first_offset, "first alias after overlay");
    CHECK(read_through(hole, 1) == dev_of(second) + second_offset + page, "virtual target is its own twin");

    RcRegistry *registry = rc_registry_runtime();
    RcRegistryHandle h = {0}, neighbor = {0};
    const RcSourceRange bytes = {1, hole + 16, 32};
    CHECK(rc_registry_source(registry, bytes.offset, bytes, &h) == RC_REGISTRY_OK, "hole source registered");
    const RcSourceRange other = {1, hole + 64, 16};
    CHECK(rc_registry_source(registry, other.offset, other, &neighbor) == RC_REGISTRY_OK, "neighbor source registered");
    RcRegistryHandle exact_old = {0}, exact_new = {0}, exact_unrelated = {0};
    RcSourceRange old_bytes = {(uint64_t)(uintptr_t)second, second_offset + page + 16, 32};
    RcSourceRange new_bytes = {(uint64_t)(uintptr_t)first, first_offset + 2 * page + 16, 32};
    RcSourceRange unrelated = {(uint64_t)(uintptr_t)adjacent, page + 16, 32};
    CHECK(rc_registry_source(registry, 10, old_bytes, &exact_old) == RC_REGISTRY_OK &&
          rc_registry_source(registry, 11, new_bytes, &exact_new) == RC_REGISTRY_OK &&
          rc_registry_source(registry, 12, unrelated, &exact_unrelated) == RC_REGISTRY_OK, "canonical sources registered");

    const VkGpaEffectRange effect = {bytes.offset, bytes.size};
    VkGpaWriteScope *writer = vk_gpa_write_begin_ranges(&effect, 1);
    RcRegistryObservation observation;
    CHECK(rc_registry_observe(registry, h, 1, &observation) != RC_REGISTRY_OK, "pending writer excludes source bytes");
    CHECK(rc_registry_observe(registry, neighbor, 1, &observation) == RC_REGISTRY_OK, "exact writer spares adjacent bytes");
    CHECK(rc_registry_observe(registry, exact_old, 1, &observation) != RC_REGISTRY_OK &&
          rc_registry_observe(registry, exact_new, 1, &observation) != RC_REGISTRY_OK,
          "predeclared guard excludes projected storage and an arriving target");
    RcRegistryReadBlocker detail;
    CHECK(rc_registry_diagnose_read_blocker(registry, old_bytes, (RcRegistryPosition){0}, &detail) &&
          detail.guard_reason == RC_REGISTRY_GUARD_GPA_PROJECTION, "find the live guard by exact ticket");
    VkGpaWriteOrigin origin = {.program = 719, .opcode = 9, .stage = 5, .set = 0, .binding = 4, .shader_key = {0xab}};
    vk_gpa_write_note_recorded(writer, &origin);
    vk_gpa_report_write_blocker(detail.sample.writer, old_bytes);
    CHECK(strstr(last_gpa_blocker, "recorded 1 program 719 op 9 stage 5 set 0 binding 4 shader ab") &&
          strstr(last_gpa_blocker, "guest bounded 1 ranges") &&
          strstr(last_gpa_blocker, "canonical projected 1 uncertain 1 unresolved 0"), "describe the actual bound writer and projection");
    vk_gpa_report_write_blocker(detail.sample.writer, new_bytes);
    CHECK(strstr(last_gpa_blocker, "target overlap projected 0:0+0 uncertain 0:0+0"), "distinguish symbolic guard from projected overlap");

    test_virtual_map(hole, hole + page, VK_NULL_HANDLE, 0);
    expect(hole, hole + page, VK_NULL_HANDLE, 0, "cleared middle alias page");
    expect(hole - page, hole, second, hole - page - second_offset, "second alias remains partially mapped");
    vk_gpa_write_complete(writer, issued + 1);
    vk_gpa_report_write_blocker(detail.sample.writer, old_bytes);
    CHECK(strstr(last_gpa_blocker, "no longer has a live scope"), "recycled slot cannot report the old writer");
    CHECK(rc_registry_observe(registry, h, 1, &observation) != RC_REGISTRY_OK,
          "scope retains uncertainty from the old in-flight target");
    CHECK(rc_registry_reconcile_cpu(registry, bytes) == RC_REGISTRY_OK, "reconciled old alias write");
    CHECK(rc_registry_observe(registry, exact_old, 1, &observation) != RC_REGISTRY_OK &&
          rc_registry_observe(registry, exact_new, 1, &observation) == RC_REGISTRY_OK,
          "completed guard retains uncertainty only on the old target");

    writer = vk_gpa_write_begin_ranges(&effect, 1);
    vk_gpa_write_complete(writer, issued + 1);
    CHECK(rc_registry_observe(registry, h, 1, &observation) == RC_REGISTRY_OK,
          "new scope has no alias in the hole even though the second record still names an adjacent page");

    writer = vk_gpa_write_begin_ranges(&effect, 1);
    test_virtual_map(hole, hole + page, first, lo - first_offset);
    test_virtual_map(hole, hole + page, VK_NULL_HANDLE, 0);
    vk_gpa_write_complete(writer, issued + 1);
    CHECK(rc_registry_observe(registry, h, 1, &observation) != RC_REGISTRY_OK,
          "publication joins an already live scope and its later removal preserves uncertainty");
    CHECK(rc_registry_reconcile_cpu(registry, bytes) == RC_REGISTRY_OK, "reconciled publication during scope");
    CHECK(rc_registry_observe(registry, exact_new, 1, &observation) != RC_REGISTRY_OK &&
          rc_registry_observe(registry, exact_unrelated, 1, &observation) == RC_REGISTRY_OK,
          "a new target joins the existing canonical guard and unrelated storage is released");
    CHECK(rc_registry_reconcile_cpu(registry, old_bytes) == RC_REGISTRY_OK &&
          rc_registry_reconcile_cpu(registry, new_bytes) == RC_REGISTRY_OK, "reconciled canonical targets");

    RcRegistryHandle mapped = {0};
    const RcSourceRange mapped_bytes = {1, lo + 16, 32};
    CHECK(rc_registry_source(registry, mapped_bytes.offset, mapped_bytes, &mapped) == RC_REGISTRY_OK,
          "mapped source registered");
    writer = vk_gpa_write_begin();
    vk_gpa_write_complete(writer, issued + 1);
    CHECK(rc_registry_observe(registry, mapped, 1, &observation) != RC_REGISTRY_OK,
          "whole-domain scope preserves the current alias's uncertainty");
    CHECK(rc_registry_observe(registry, h, 1, &observation) == RC_REGISTRY_OK,
          "whole-domain scope excludes an empty page between active alias records");
    CHECK(rc_registry_reconcile_cpu(registry, (RcSourceRange){1, lo - page, end - lo + page}) == RC_REGISTRY_OK,
          "reconciled whole-domain alias writes");

    // Retirement waits for the last page, then the prior graphics use. The mock shader keeps an old record ID.
    const uint64_t scratch = end + 2 * page;
    gpu_done();
    CHECK(read_record(hole - page, second_rec, 0) == dev_of(second) + second_offset,
          "partially displaced record survives a fence");
    test_virtual_map(hole - page, hole, VK_NULL_HANDLE, 0);
    for (int i = 0; i < 16; i++) {
        test_virtual_map(scratch, scratch + page, adjacent, scratch);
        CHECK(record_of(scratch) != second_rec, "last-page retirement does not reuse an in-flight record");
    }
    CHECK(read_record(hole - page, second_rec, 0) == dev_of(second) + second_offset,
          "in-flight shader still reads the displaced record payload");
    gpu_done();
    test_virtual_map(scratch, scratch + page, adjacent, scratch);
    CHECK(record_of(scratch) != first_rec, "record with surviving pages stays live");
    expect(lo, lo + page, first, lo - first_offset, "surviving first alias prefix after retirement");
    expect(hole + page, end, first, lo - first_offset, "surviving first alias suffix after retirement");

    test_virtual_map(lo + page, end, second, lo + page);
    test_virtual_map(lo, lo + page, VK_NULL_HANDLE, 0);
    CHECK(read_record(lo, first_rec, 0) == dev_of(first) + first_offset, "whole-record retirement keeps old payload");
    gpu_done();
    test_virtual_map(scratch, scratch + page, adjacent, scratch);
    CHECK(record_of(scratch) == first_rec, "last-page departure makes the record reusable after actual completion");

    for (int i = 0; i < 128; i++) {
        test_virtual_map(lo + 2 * page, lo + 3 * page, first, lo + 2 * page);
        test_virtual_map(lo + 2 * page, lo + 3 * page, second, lo + page);
        gpu_done();
    }
    expect(lo + page, end, second, lo + page, "repeated partial remaps");
    expect(lo - page, lo, adjacent, lo - page, "adjacent pool survives all replacements");

    test_virtual_map(lo - page, scratch + page, VK_NULL_HANDLE, 0);
    CHECK(!block_of(lo) && !block_of(end), "clearing the last aliases releases both blocks");
    gpu_done();
}

static void canonical_buffer_writers(void) {
    RcRegistry *registry = rc_registry_runtime();
    RcInterval native[2] = {{0x2000, 100, {10001, 16, 64}}, {0x2040, 101, {10002, 32, 64}}};
    RcInterval *parts = NULL;
    uint32_t n = resource_host_interval_slice(native, 2, 0x2020, 64, &parts);
    CHECK(n == 2 && parts && parts[0].range.offset == 48 && parts[0].range.size == 32 &&
          parts[1].range.offset == 32 && parts[1].range.size == 32, "exact affine clipped write coverage");
    RcRegistryHandle old = {0}, adjacent = {0}, successor = {0}, compatibility = {0};
    RcSourceRange old_bytes = {10001, 48, 16}, other_bytes = {10002, 32, 16}, new_bytes = {10003, 48, 16};
    CHECK(rc_registry_source(registry, 20, old_bytes, &old) == RC_REGISTRY_OK &&
          rc_registry_source(registry, 21, other_bytes, &adjacent) == RC_REGISTRY_OK &&
          rc_registry_source(registry, 22, new_bytes, &successor) == RC_REGISTRY_OK &&
          rc_registry_source(registry, 23, (RcSourceRange){1, 0x2020, 16}, &compatibility) == RC_REGISTRY_OK,
          "register old and replacement mappings");
    RcRegistryObservation observation;
    ResourceHostWrite owner = resource_host_write_prepare(parts, n, RC_REGISTRY_BUFFER, 0);
    CHECK(owner.tickets && rc_registry_observe(registry, old, 1, &observation) != RC_REGISTRY_OK &&
          rc_registry_observe(registry, adjacent, 1, &observation) != RC_REGISTRY_OK &&
          rc_registry_observe(registry, successor, 1, &observation) == RC_REGISTRY_OK,
          "a reservation covers every captured extent and spares successor backing");
    resource_host_write_dispose(&owner, 0, 0, 0);
    CHECK(rc_registry_observe(registry, old, 1, &observation) == RC_REGISTRY_OK &&
          rc_registry_observe(registry, adjacent, 1, &observation) == RC_REGISTRY_OK, "cancel keeps bytes observable");

    n = resource_host_interval_slice(native, 2, 0x2020, 64, &parts);
    owner = resource_host_write_prepare(parts, n, RC_REGISTRY_BUFFER, 0);
    resource_host_write_dispose(&owner, issued + 1, 1, 0);
    CHECK(rc_registry_observe(registry, old, 1, &observation) != RC_REGISTRY_OK &&
          rc_registry_observe(registry, adjacent, 1, &observation) != RC_REGISTRY_OK &&
          rc_registry_observe(registry, successor, 1, &observation) == RC_REGISTRY_OK,
          "an old GPU completion retains only its actual target authority");
    CHECK(rc_registry_reconcile_cpu(registry, (RcSourceRange){10001, 48, 32}) == RC_REGISTRY_OK &&
          rc_registry_reconcile_cpu(registry, (RcSourceRange){10002, 32, 32}) == RC_REGISTRY_OK,
          "explicit proven visibility removes both old obligations");

    RcRegistryObservation before = {0};
    CHECK(rc_registry_observe(registry, old, 1, &before) == RC_REGISTRY_OK, "observe before directly visible write");
    n = resource_host_interval_slice(native, 2, 0x2020, 64, &parts);
    owner = resource_host_write_prepare(parts, n, RC_REGISTRY_BUFFER, 0);
    resource_host_write_dispose(&owner, issued + 1, 1, 1);
    CHECK(!rc_registry_content_current(registry, &before) &&
          rc_registry_observe(registry, old, 1, &observation) == RC_REGISTRY_OK,
          "a visible GPU write still changes exact retained content provenance");

    native[1].address++;
    CHECK(!resource_host_interval_slice(native, 2, 0x2020, 64, &parts) && !parts,
          "partial mapping cannot authorize a complete write");
    owner = resource_host_write_prepare(parts, 0, RC_REGISTRY_BUFFER, 0);
    CHECK(owner.guard.id && rc_registry_observe(registry, successor, 1, &observation) != RC_REGISTRY_OK &&
          rc_registry_observe(registry, compatibility, 1, &observation) == RC_REGISTRY_OK,
          "unresolved writer reserves a canonical-only guard");
    resource_host_write_dispose(&owner, 0, 0, 0);
    CHECK(rc_registry_observe(registry, successor, 1, &observation) == RC_REGISTRY_OK, "skipped guard is canceled");
}

static void ordered_prefix(void) {
    const uint64_t lo = UINT64_C(0x400000000), page = 4096, scene = 400;
    const VkGpaEffectRange footprint = {lo + 16, 32};
    RcRegistry *registry = rc_registry_runtime();
    VkBuffer old_buffer = new_buf(), current_buffer = new_buf();
    const RcInterval old_host = {lo, 111, {90001, 0, page}}, current_host = {lo, 112, {90002, 0, page}};
    const RcInterval sample = {0x20000, 113, {90002, 16, 32}};
    uint32_t old_rec = vk_gpa_map(lo, lo + page, old_buffer, lo, 1, &old_host, 1);
    CHECK(old_rec, "initial GPA target is disjoint from the sampled backing");
    RcRegistryHandle input = {0};
    CHECK(rc_registry_source(registry, 112, sample.range, &input) == RC_REGISTRY_OK, "register the required input");
    RcRegistryPosition producer = {scene, 41}, read = {scene, 134}, output = {scene, 135};
    const RcSourceRange untouched = {90003, 32, 32};
    const RcSourceRange captured[] = {sample.range, untouched};
    RcRegistryGpuCapture uniforms = {0};
    CHECK(rc_registry_gpu_capture_begin(registry, captured, 2, &uniforms) == RC_REGISTRY_OK,
          "capture the two uniform ranges before the GPU writer");
    VkGpaWriteScope *prior = vk_gpa_write_begin_at(&footprint, 1, 1, producer);
    RcRegistryWrite current;
    CHECK(rc_registry_write_begin_at(registry, sample.range, RC_REGISTRY_IMAGE, output, &current) == RC_REGISTRY_OK,
          "declaration of the unrecorded requesting operation");
    CHECK(rc_registry_declare_through(registry, output) == RC_REGISTRY_OK, "seal the requesting operation");
    RcRegistryReadBlocker detail;
    CHECK(rc_registry_diagnose_read_blocker(registry, sample.range, read, &detail) &&
          detail.guard_reason == RC_REGISTRY_GUARD_GPA_PROJECTION, "the earlier symbolic projection blocks an otherwise disjoint input");
    const VkGpaPrefixRequest request = {detail.sample.writer, producer, read, sample};
    CHECK(!vk_gpa_write_prefix_ready(&request), "unrecorded GPA claims cannot justify a split");
    vk_gpa_write_note_recorded(prior, &(VkGpaWriteOrigin){.program=719});
    CHECK(vk_gpa_write_prefix_ready(&request) && vk_gpa_write_scene_may_need_prefix(scene),
          "a central recorder can prepare a real earlier visible prefix");
    CHECK(!vk_gpa_write_prefix_complete(&request), "declaration alone cannot settle a GPA writer");
    CHECK(rc_registry_reopen_unemitted(registry, read) == RC_REGISTRY_BUSY, "do not rewind live output claims");
    CHECK(rc_registry_write_cancel(registry, current) == RC_REGISTRY_OK &&
          rc_registry_reopen_unemitted(registry, read) == RC_REGISTRY_OK, "reopen only the last canceled operation");

    // A mapping published before the real part executes can redirect the earlier command after acquisition.
    uint32_t amended = vk_gpa_map(lo, lo + page, current_buffer, lo, 1, &current_host, 1);
    CHECK(amended && record_of(lo) == amended, "the supplied writer can reach the newly projected sampled storage");
    CHECK(rc_registry_read_range_at(registry, sample.range, read) == RC_REGISTRY_BLOCKED,
          "a disjoint initial projection does not permit reading before real execution");
    CHECK(rc_registry_gpu_capture_read_at(registry, uniforms, &sample.range, 1, read) == RC_REGISTRY_BLOCKED &&
          rc_registry_gpu_capture_read_at(registry, uniforms, &untouched, 1, read) == RC_REGISTRY_BLOCKED,
          "an immutable snapshot cannot bypass a live mutable projection on either range");
    uint32_t sampled_pixels = 0x11223344;
    if (read_through(lo + 16, 0) == dev_of(current_buffer) + 16)
        sampled_pixels ^= 0x00ffffff; // test device store through the amended table, before the host-visible marker
    CHECK(rc_registry_handoff_through(registry, (RcRegistryPosition){scene,133}) == RC_REGISTRY_OK &&
          vk_gpa_write_prefix_complete(&request), "complete exactly the covered earlier scope");
    CHECK(rc_registry_position_open(registry, read) && !vk_gpa_write_scene_may_need_prefix(scene),
          "native prefix retires the writer without handing the current input or the final guest scene");
    RcRegistryGpuReadEvidence evidence;
    CHECK(rc_registry_gpu_capture_read_evidence(registry, uniforms, &sample.range, 1, read, &evidence) == RC_REGISTRY_STALE &&
          evidence == RC_GPU_CAPTURE_EARLIER_VISIBLE,
          "a real earlier overlap survives settlement and expires the original uniform snapshot");
    CHECK(rc_registry_gpu_capture_read_evidence(registry, uniforms, &untouched, 1, read, &evidence) == RC_REGISTRY_OK &&
          evidence == RC_GPU_CAPTURE_NO_EVIDENCE,
          "disjoint immutable uniform bytes become usable at their original read without recapture");
    CHECK(rc_registry_gpu_capture_end(registry, &uniforms) == RC_REGISTRY_OK, "retire the capture once");
    RcRegistryReadObservation observation;
    CHECK(rc_registry_observe_at(registry, input, 1, read, &observation) == RC_REGISTRY_OK && sampled_pixels == 0x11ddccbb,
          "retry observes the new coherent overlapped bytes at the same input position");
    CHECK(rc_registry_write_begin_at(registry, sample.range, RC_REGISTRY_IMAGE, output, &current) == RC_REGISTRY_OK &&
          rc_registry_declare_through(registry, output) == RC_REGISTRY_OK,
          "redeclare the canceled output at its original ordinal after visibility");
    CHECK(rc_registry_write_cancel(registry, current) == RC_REGISTRY_OK, "test tail cancellation");
    vk_gpa_write_cancel(prior); // final scene's callback owns the shell after early settlement
    vk_gpa_unmap(lo, lo + page, amended, NULL);
    vk_gpa_unmap(lo, lo + page, old_rec, NULL);
    gpu_done();
}

static void definite_mirror_stores(void) {
    RcRegistry *registry = rc_registry_runtime();
    const uint64_t page = 4096;
    for (unsigned mode = 0; mode < 4; mode++) {
        const uint64_t base = UINT64_C(0x420000000) + mode * (page * 4), scene = 500 + mode;
        const RcInterval mapping = {base, 120 + mode, {90100 + mode, 0, page}};
        donor_mapping = mapping;
        donor_import = new_buf();
        donor_mirror = new_buf();
        uint32_t imported = vk_gpa_map(base, base + page, donor_import, base, 1, &mapping, 1);
        int twin;
        uint32_t live = vk_gpa_remap_exact(base, base + page, imported, donor_mirror, base, 1, 1, &twin);
        CHECK(live && twin, "actual table has both routes");
        const VkGpaEffectRange ranges[] = {{base + 36, 4}, {base + 68, 4}};
        const RcRegistryPosition producer = {scene, 3}, read = {scene, 8};
        RcRegistryGpuCapture capture = {0};
        CHECK(rc_registry_gpu_capture_begin(registry, &mapping.range, 1, &capture) == RC_REGISTRY_OK,
              "immutable base precedes the actual writer");
        VkGpaWriteScope *scope = vk_gpa_write_begin_at(ranges, 2, 1, producer);
        const VkBuffer saved = donor_import;
        if (mode == 3) donor_import = new_buf();
        vk_gpa_write_record_definite(scope, ranges, 2);
        donor_import = saved;
        CHECK(donor_refs == (mode == 3 ? 0u : 2u), "retain both exact affected native spans, wrong twin refuses");
        const RcInterval extra = {base + page, 130 + mode, {90200 + mode, 0, page}};
        uint32_t disjoint = vk_gpa_map(base + page, base + 2 * page, new_buf(), base + page, 1, &extra, 1);
        CHECK(disjoint, "unrelated page edit while the writer still has live mappings");
        uint32_t replacement = 0;
        if (mode == 1) {
            replacement = vk_gpa_map(base, base + page, new_buf(), base, 1, &mapping, 1);
            CHECK(replacement && record_of(base) == replacement, "a relevant projection now has a different primary");
        }
        CHECK(rc_registry_declare_through(registry, read) == RC_REGISTRY_OK, "declare future original read");
        if (mode == 2) {
            vk_gpa_write_cancel(scope);
            CHECK(!donor_refs, "an unsubmitted recorded claim releases both provider captures");
        } else {
            RcRegistryReadBlocker blocker;
            CHECK(rc_registry_diagnose_read_blocker(registry, mapping.range, read, &blocker), "prior GPA guard still active");
            const VkGpaPrefixRequest request = {blocker.sample.writer, producer, read, mapping};
            vk_gpa_write_note_recorded(scope, &(VkGpaWriteOrigin){.program = 1});
            CHECK(rc_registry_handoff_through(registry, (RcRegistryPosition){scene, 7}) == RC_REGISTRY_OK &&
                  vk_gpa_write_prefix_complete(&request), "only real earlier prefix settles the projection");
            if (mode == 0) CHECK(donor_refs == 2, "completion callback still owns the recorded provider");
        }
        const RcRegistryBufferSpan demand = {mapping, {31, 3, 0}};
        RcRegistryBufferFragment fragments[2];
        uint32_t count = 31;
        const RcRegistryResult result = rc_registry_gpu_capture_buffer_fragments(registry, capture, &demand, 1, read,
                                                                                   fragments, 2, &count);
        if (mode == 0) {
            CHECK(result == RC_REGISTRY_OK && count == 2 && fragments[0].interval.range.offset == 36 &&
                  fragments[1].interval.range.offset == 68 && fragments[0].interval.range.size == 4 &&
                  fragments[1].interval.range.size == 4 &&
                  rc_registry_position_compare(fragments[1].output, producer) == 0,
                  "both genuine written words remain exact at the original read, no claim on immutable gaps");
        } else if (mode == 2) CHECK(result == RC_REGISTRY_OK && count == 0, "canceled writer contributes no output");
        else CHECK(result == RC_REGISTRY_BLOCKED && count == 0, "wrong or replaced projection cannot donate bytes");
        if (mode != 2) vk_gpa_write_complete(scope, scene);
        CHECK(donor_refs == 0, "each provider pin is released after the scope's actual lifetime");
        CHECK(rc_registry_gpu_capture_end(registry, &capture) == RC_REGISTRY_OK, "retire the original capture once");
        vk_gpa_unmap(base, base + page, replacement ? replacement : live, NULL);
        if (replacement) vk_gpa_unmap(base, base + page, live, NULL);
        vk_gpa_unmap(base + page, base + 2 * page, disjoint, NULL);
        gpu_done();
    }
    donor_mirror = donor_import = VK_NULL_HANDLE;
}

static void partial_remap_refuses(void) {
    const uint64_t lo = UINT64_C(0x4e01234000), page = 4096, hi = lo + 3 * page;
    VkBuffer original = new_buf(), overlay = new_buf(), proposed = new_buf();
    uint32_t old = vk_gpa_map(lo, hi, original, lo, 0, NULL, 0);
    uint32_t other = vk_gpa_map(lo + page, lo + 2 * page, overlay, lo + page, 0, NULL, 0);
    CHECK(old && other && old != other, "projected import and its overlapping neighbor");
    int twin = -1;
    CHECK(!vk_gpa_remap_exact(lo, hi, old, proposed, lo, 1, 0, &twin) && !twin,
          "overlapping page rejects the whole mirror installation");
    expect(lo, lo + page, original, lo, "first original page survives refusal");
    expect(lo + page, lo + 2 * page, overlay, lo + page, "unrelated mapping survives refusal");
    expect(lo + 2 * page, hi, original, lo, "last original page survives refusal");
    vk_gpa_unmap(lo + page, lo + 2 * page, other, NULL);
    CHECK(!vk_gpa_remap_exact(lo, hi, old, proposed, lo, 1, 0, &twin),
          "a missing page also blocks the whole promotion");
    expect(lo, lo + page, original, lo, "original mapping survives the hole");
    expect(lo + page, lo + 2 * page, VK_NULL_HANDLE, 0, "unmapped byte stays unmapped");
    vk_gpa_unmap(lo, hi, old, NULL);
    gpu_done();
}

static void check_required_binding_guard(void) {
    expect_required_fatal = 1;
    if (!setjmp(fatal_jump)) {
        (void)vk_gpa_info();
        CHECK(0, "a required unprepared table must never produce a zero binding");
    }
    expect_required_fatal = 0;
}

static int check_startup_failure(const char *mode) {
    int retry_after_relief = !strcmp(mode, "retry-after-relief");
    int retry_after_cancel = !strcmp(mode, "retry-after-cancel");
    int refused = !strcmp(mode, "refuse") || retry_after_relief;
    gpa_diagnostic_disabled = !strcmp(mode, "diagnostic-disabled");
    if (!strcmp(mode, "no-capability")) ctx.has_buffer_device_address = 0;
    if (!strcmp(mode, "fail-table")) fail_create = 1;
    if (!strcmp(mode, "fail-zero")) fail_create = 2;
    if (!strcmp(mode, "fail-sink") || retry_after_cancel) fail_create = 3;
    if (!strcmp(mode, "no-table-map")) fail_map = 1;
    if (!strcmp(mode, "no-zero-map")) fail_map = 2;
    if (!strcmp(mode, "no-device-address")) fail_address = 1;
    budget_setup(refused);
    VkResult failure;
    if (!vk_gpa_wanted()) {
        CHECK(vk_gpa_prepare(&failure) == VK_GPA_DISABLED && failure == VK_SUCCESS,
              "missing shader capability or explicit diagnostic disable uses compatibility");
        CHECK(!vk_gpa_on() && !vk_gpa_info().buffer && !create_calls,
              "disabled GPA neither allocates nor provides a shader table");
        check_storage_budget(0);
        VkGpaVirtualBinding binding = virtual_binding(MiB, 2 * MiB, new_buf(), 0);
        VkGpaVirtualUpdate update;
        CHECK(vk_gpa_virtual_prepare(&binding, 1, &update) == RC_BACKING_OK,
              "an explicitly disabled projection accepts a canonical-only map");
        vk_gpa_virtual_commit(&update);
        CHECK(!table && !create_calls, "a disabled map does not create storage");
    } else {
        const unsigned attempted = refused ? 0 : fail_create ? fail_create : fail_map ? fail_map : 3;
        VkGpaPrepareResult expected = refused || fail_create ? VK_GPA_NO_CAPACITY : VK_GPA_ERROR;
        CHECK(vk_gpa_prepare(&failure) == expected && failure != VK_SUCCESS,
              "a required startup failure returns a typed result");
        CHECK(!vk_gpa_on() && !table, "failed startup exposes no renderer buffer");
        check_required_binding_guard();
        CHECK(!vk_gpa_on() && create_calls == attempted, "a first-use query does not allocate or rearm startup");
        CHECK(destroys == attempted - (fail_create != 0), "startup failure destroyed every allocated native member");
        check_storage_budget(0);
        vk_gpa_shutdown();
        CHECK(destroys == attempted - (fail_create != 0), "shutdown does not destroy rolled-back storage twice");
        check_storage_budget(0);
        if (retry_after_relief || retry_after_cancel) {
            attempt_base = create_calls;
            fail_create = 0;
            budget_setup(0);
            budget_clock_ns += 100000000;
            resource_core_budget_refresh();
            CHECK(vk_gpa_prepare(&failure) == VK_GPA_READY && vk_gpa_info().buffer && table && table[2],
                  "a later explicit startup after measured relief or cancellation can publish");
            check_storage_budget(1);
            vk_gpa_shutdown();
            check_storage_budget(0);
        }
    }
    printf("gpa_check (%s): %s\n", mode, fails ? "FAILED" : "ok");
    return fails != 0;
}

int main(int argc, char **argv) {
    if (argc == 2) return check_startup_failure(argv[1]);
    budget_setup(0);
    VkResult failure;
    CHECK(!vk_gpa_on() && !create_calls, "a readiness query performs no allocation");
    CHECK(vk_gpa_prepare(&failure) == VK_GPA_READY && failure == VK_SUCCESS && vk_gpa_on() && table && table[2],
          "the required table exists before any mappings or shader bindings");
    CHECK(vk_gpa_prepare(&failure) == VK_GPA_READY && create_calls == 3,
          "a second preparation reuses the published allocation group");
    if (fails) return 1;
    check_storage_budget(1);
    virtual_projection();
    canonical_buffer_writers();
    ordered_prefix();
    definite_mirror_stores();
    partial_remap_refuses();

    // a buffer over three 16 MiB ranges, not aligned to them
    uint64_t a0 = 0x600000000ull + 5 * MiB, a1 = a0 + 40 * MiB;
    VkBuffer b0 = new_buf();
    uint32_t r0 = vk_gpa_map(a0, a1, b0, a0, 1, NULL, 0);
    CHECK(r0, "a first buffer is mapped");
    expect(a0, a1, b0, a0, "first buffer");
    expect(a0 - 16 * MiB, a0, VK_NULL_HANDLE, 0, "below the first buffer");
    expect(a1, a1 + 16 * MiB, VK_NULL_HANDLE, 0, "above the first buffer");

    // a mirror takes the pages with the imported buffer as twin, and gives them back
    VkBuffer m0 = new_buf();
    int twin = 0;
    uint32_t mirrored = vk_gpa_remap_exact(a0, a1, r0, m0, a0, 1, 1, &twin);
    CHECK(mirrored && mirrored != r0 && twin, "the full mirror has a new live record and an import twin");
    r0 = mirrored;
    expect(a0, a1, m0, a0, "mirrored");
    CHECK(read_through(a0 + 4096, 1) == dev_of(b0) + 4096, "the twin of a mirrored page is the imported buffer");
    r0 = vk_gpa_remap_exact(a0, a1, r0, b0, a0, 0, 1, &twin);
    CHECK(r0 && r0 != mirrored && !twin, "complete reverse projection restores the import");
    expect(a0, a1, b0, a0, "mirror evicted");
    CHECK(read_through(a0 + 4096, 1) == dev_of(b0) + 4096, "a plain page is its own twin");
    gpu_done();

    // an unmapped range reads the zero buffer at once, and its blocks serve no other range while the GPU may still read
    // the old L1 entries
    uint32_t old_blk[3] = {block_of(a0), block_of(a0 + 16 * MiB), block_of(a0 + 32 * MiB)};
    CHECK(old_blk[0] && old_blk[1] && old_blk[2], "three blocks for three ranges");
    vk_gpa_unmap(a0, a1, r0, NULL);
    expect(a0, a1, VK_NULL_HANDLE, 0, "unmapped");
    CHECK(!block_of(a0) && !block_of(a0 + 16 * MiB) && !block_of(a0 + 32 * MiB), "empty blocks leave L1");
    uint64_t c0 = 0x700000000ull;
    VkBuffer b1 = new_buf();
    uint32_t r1 = vk_gpa_map(c0, c0 + 48 * MiB, b1, c0, 1, NULL, 0);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            CHECK(block_of(c0 + (uint64_t)i * 16 * MiB) != old_blk[j], "a block still waiting for the GPU serves another range");
    for (int j = 0; j < 3; j++)
        for (uint32_t k = 0; k < L2_N; k++)
            if (table[L2_WORD + old_blk[j] * L2_N + k]) { CHECK(0, "a waiting block holds an entry"); break; }
    // the same range mapped again before the GPU is done takes its block back
    VkBuffer b2 = new_buf();
    uint32_t r2 = vk_gpa_map(a0, a1, b2, a0, 1, NULL, 0);
    CHECK(block_of(a0) == old_blk[0] && block_of(a0 + 16 * MiB) == old_blk[1] && block_of(a0 + 32 * MiB) == old_blk[2],
          "a range mapped again takes its waiting blocks back");
    expect(a0, a1, b2, a0, "mapped again");
    gpu_done();   // the callbacks of the first release find the blocks in use
    expect(a0, a1, b2, a0, "mapped again, after the GPU");
    VkBuffer b3 = new_buf();
    uint32_t r3 = vk_gpa_map(0x780000000ull, 0x780000000ull + 16 * MiB, b3, 0x780000000ull, 1, NULL, 0);
    for (int j = 0; j < 3; j++) CHECK(block_of(0x780000000ull) != old_blk[j], "a block taken back was freed by its old callback");
    vk_gpa_unmap(0x780000000ull, 0x780000000ull + 16 * MiB, r3, NULL);
    vk_gpa_unmap(a0, a1, r2, NULL);
    vk_gpa_unmap(c0, c0 + 48 * MiB, r1, NULL);
    gpu_done();

    // released, taken back and released within one submission, many times: one block, whatever the callbacks pending
    for (int i = 0; i < 3000; i++) {
        VkBuffer b = new_buf();
        uint32_t r = vk_gpa_map(a0, a0 + 8 * MiB, b, a0, 1, NULL, 0);
        if (i % 500 == 0) expect(a0, a0 + 8 * MiB, b, a0, "freed and made again at its address");
        vk_gpa_unmap(a0, a0 + 8 * MiB, r, NULL);
        if (i % 1000 == 999) gpu_done();
    }
    gpu_done();
    CHECK(!out_of_blocks_lines, "a pool freed and made again at its address ran out of blocks");

    // addresses that climb through 64 GiB, as the game's allocator hands them out: 20 MiB pools 48 MiB apart, eight
    // alive at a time, far more ranges than there are blocks
    enum { ALIVE = 8, POOLS = 1300 };
    struct { uint64_t lo, hi; VkBuffer b; uint32_t rec; } live[ALIVE] = {0};
    for (int i = 0; i < POOLS; i++) {
        int s = i % ALIVE;
        if (live[s].rec) vk_gpa_unmap(live[s].lo, live[s].hi, live[s].rec, NULL);
        live[s].lo = 0x800000000ull + (uint64_t)i * 48 * MiB + 3 * MiB, live[s].hi = live[s].lo + 20 * MiB;
        live[s].b = new_buf();
        live[s].rec = vk_gpa_map(live[s].lo, live[s].hi, live[s].b, live[s].lo, 1, NULL, 0);
        CHECK(live[s].rec, "climbing pool %d has a record", i);
        if (i % 3 == 0) gpu_done();
        if (i % 97 == 0 || i == POOLS - 1)
            for (int k = 0; k < ALIVE; k++)
                if (live[k].rec) expect(live[k].lo, live[k].hi, live[k].b, live[k].lo, "climbing pool");
    }
    CHECK(!out_of_blocks_lines, "climbing pool addresses ran out of blocks (%d lines)", out_of_blocks_lines);
    for (int k = 0; k < ALIVE; k++) vk_gpa_unmap(live[k].lo, live[k].hi, live[k].rec, NULL);
    gpu_done();

    // overlapping imports: the pages of the one that goes return to the other, and their block stays
    uint64_t o0 = 0x500000000ull, o1 = o0 + 16 * MiB;
    VkBuffer ba = new_buf(), bb = new_buf();
    uint32_t ra = vk_gpa_map(o0, o1, ba, o0, 1, NULL, 0), rb = vk_gpa_map(o0 + 4 * MiB, o0 + 8 * MiB, bb, o0 + 4 * MiB, 1, NULL, 0);
    expect(o0 + 4 * MiB, o0 + 8 * MiB, bb, o0 + 4 * MiB, "the later import holds the shared pages");
    cover_rec = ra, cover_lo = o0, cover_hi = o1;
    vk_gpa_unmap(o0 + 4 * MiB, o0 + 8 * MiB, rb, cover);
    expect(o0, o1, ba, o0, "the shared pages return to the other import");
    cover_rec = 0;
    vk_gpa_unmap(o0, o1, ra, NULL);
    gpu_done();

    // every block in use at once: the next range is left out and said so once, the ranges after it and the mapped
    // ones are untouched, and a block given back serves the range that was left out
    static struct { VkBuffer b; uint32_t rec; } all[BLOCKS];
    uint64_t f0 = 0x1000000000ull;
    for (uint32_t i = 0; i < BLOCKS - 1; i++) {
        all[i].b = new_buf();
        all[i].rec = vk_gpa_map(f0 + (uint64_t)i * 32 * MiB, f0 + (uint64_t)i * 32 * MiB + MiB, all[i].b, f0 + (uint64_t)i * 32 * MiB, 1, NULL, 0);
    }
    CHECK(!out_of_blocks_lines, "the table holds %u ranges at once", BLOCKS - 1);
    uint64_t x0 = 0x2000000000ull;
    VkBuffer bx = new_buf();
    uint32_t rx = vk_gpa_map(x0, x0 + MiB, bx, x0, 1, NULL, 0);
    CHECK(out_of_blocks_lines == 1, "a range past the last block is reported (%d lines)", out_of_blocks_lines);
    expect(x0, x0 + MiB, VK_NULL_HANDLE, 0, "a range past the last block");
    const uint64_t page = 4096;
    VkGpaVirtualBinding pending[] = {
        virtual_binding(f0, f0 + page, bx, 0),
        virtual_binding(f0 + MiB - page, f0 + MiB, VK_NULL_HANDLE, 0),
        virtual_binding(x0, x0 + page, bx, page),
    };
    VkGpaVirtualUpdate admission;
    unsigned last = record_of(f0);
    CHECK(vk_gpa_virtual_prepare(pending, 3, &admission) == RC_BACKING_CAPACITY,
          "a new L2 block cannot leave an earlier map or unmap partially published");
    CHECK(record_of(f0) == last && record_of(f0 + MiB - page) == last && !block_of(x0),
          "shader resolver retains existing mappings on L2 exhaustion");
    CHECK(vk_gpa_virtual_prepare(pending, 2, &admission) == RC_BACKING_OK,
          "existing L2 blocks admit a remap at capacity");
    vk_gpa_virtual_cancel(&admission);
    CHECK(record_of(f0) == last && record_of(f0 + MiB - page) == last,
          "prepared map canceled by canonical failure has no native effect");
    fail_address = 1;
    CHECK(vk_gpa_virtual_prepare(pending, 2, &admission) == RC_BACKING_STALE &&
          record_of(f0) == last, "missing physical device address cannot clear the previous mapping");
    fail_address = 0;

    uint64_t split = f0 + 16 * MiB;
    test_virtual_map(split - page, split, all[0].b, split - page);
    VkGpaVirtualBinding cross_block = virtual_binding(split - page, split + page, bx, 0);
    last = record_of(split - page);
    CHECK(vk_gpa_virtual_prepare(&cross_block, 1, &admission) == RC_BACKING_CAPACITY &&
          record_of(split - page) == last && !record_of(split),
          "exhaustion cannot publish just the first block of one request");
    vk_gpa_virtual_unmap(split - page, split);

    for (uint32_t i = 0; i < BLOCKS - 1; i += 37)
        expect(f0 + (uint64_t)i * 32 * MiB, f0 + (uint64_t)i * 32 * MiB + MiB, all[i].b, f0 + (uint64_t)i * 32 * MiB, "a full table");
    vk_gpa_unmap(x0, x0 + MiB, rx, NULL);
    vk_gpa_unmap(f0, f0 + MiB, all[0].rec, NULL);
    rx = vk_gpa_map(x0, x0 + MiB, bx, x0, 1, NULL, 0);
    expect(x0, x0 + MiB, VK_NULL_HANDLE, 0, "a block the GPU may still read");
    vk_gpa_unmap(x0, x0 + MiB, rx, NULL);
    gpu_done();
    rx = vk_gpa_map(x0, x0 + MiB, bx, x0, 1, NULL, 0);
    expect(x0, x0 + MiB, bx, x0, "a block given back");

    uint64_t reuse = f0 + 32 * MiB;
    vk_gpa_unmap(reuse, reuse + MiB, all[1].rec, NULL);
    CHECK(!block_of(reuse), "vacant L2 block waits for the earlier issued fence");
    VkGpaVirtualBinding at_same_l1 = virtual_binding(reuse, reuse + page, bx, 2 * page);
    CHECK(vk_gpa_virtual_prepare(&at_same_l1, 1, &admission) == RC_BACKING_OK,
          "the same L1 can re-admit its waiting block before the old fence");
    vk_gpa_virtual_commit(&admission);
    gpu_done();
    CHECK(read_through(reuse, 0) == dev_of(bx) + 2 * page &&
          read_through(reuse, 1) == dev_of(bx) + 2 * page,
          "old block cancellation callback leaves valid remap words and their twin");

    // No completion is needed to exhaust immutable records on an existing L2 page.
    for (uint32_t i = 0; i < RECS; ++i) {
        if (!vk_gpa_map(x0, x0 + page, bx, x0, 0, NULL, 0)) break;
    }
    last = record_of(reuse);
    CHECK(vk_gpa_virtual_prepare(&at_same_l1, 1, &admission) == RC_BACKING_CAPACITY &&
          record_of(reuse) == last && read_through(reuse, 0) == dev_of(bx) + 2 * page,
          "record pressure does not invalidate an earlier live virtual alias");
    vk_gpa_virtual_unmap(reuse, reuse + page);
    expect(reuse, reuse + page, VK_NULL_HANDLE, 0, "unmap cannot be blocked by exhausted records");


    vk_gpa_report();
    gpu_done();
    vk_gpa_shutdown();
    CHECK(!vk_gpa_on() && !table && create_calls == 3 && destroys == 3,
          "all three native owners retire once at shutdown");
    check_storage_budget(0);
    printf("gpa_check: %s\n", fails ? "FAILED" : "ok");
    return fails != 0;
}
