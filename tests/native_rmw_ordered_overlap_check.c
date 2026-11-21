#include "vk.h"
#include "vk_exec.h"
#include "resource_core_budget.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { POOL_IMPORT = 1, WORDS = 12, BITMAP_FIRST = 3, BITMAP_WORDS = 6, OVERLAP = 4,
       BITMAP_EQUAL = 5, BITMAP_SKIP = 6, SNL_ONLY = 7, SCENE = 7 };
#define GUEST UINT64_C(0x1000)
#define CANONICAL UINT64_C(0x200)
#define BACKING UINT64_C(42)
#define BODY ((VkCommandBuffer)(uintptr_t)1)
#define OTHER_BODY ((VkCommandBuffer)(uintptr_t)2)
#define TAIL ((VkCommandBuffer)(uintptr_t)3)
#define IMPORT ((VkBuffer)(uintptr_t)11)
#define MIRROR ((VkBuffer)(uintptr_t)12)
#define SNAPSHOT ((VkBuffer)(uintptr_t)13)
#define WORD_MARKER ((VkBuffer)(uintptr_t)14)
#define BITMAP_MARKER ((VkBuffer)(uintptr_t)15)
#define WORD_PIPE ((VkPipeline)(uintptr_t)21)
#define BITMAP_PIPE ((VkPipeline)(uintptr_t)22)
#define OK(call) assert((call) == RC_REGISTRY_OK)

typedef struct PoolRec {
    VkBuffer buf, mbuf;
    uint64_t publication, import_lo, import_hi, wb_fence;
    uint32_t acc_sub, wb_sub;
    int kind, gpa_host_coherent;
    void *smap;
} PoolRec;
struct VkMemSpanCapture {
    uint64_t address, size, mirror_incarnation;
    PoolRec *provider, *window;
    MemBackingCapture *host;
    int native_input, current;
    unsigned refs;
};

static const uint32_t msub = 1;
static PoolRec pool = {.buf = IMPORT, .mbuf = MIRROR, .publication = 8, .import_lo = GUEST,
    .import_hi = GUEST + 4 * WORDS, .acc_sub = 1, .kind = POOL_IMPORT, .gpa_host_coherent = 1};
static RcInterval backing = {GUEST, 3, {BACKING, CANONICAL, 4 * WORDS}};
static VkMemSpanCapture word_hold, bitmap_hold;
static RcRegistry *registry;
static RcRegistryPosition recorder;
static const RcRegistryPosition word_read = {SCENE, 2}, word_output = {SCENE, 3};
static const RcRegistryPosition bitmap_read = {SCENE, 4}, bitmap_output = {SCENE, 5};
static const VkMemSpan word_span = {.pool = &pool, .buffer = MIRROR, .offset = 4 * OVERLAP, .avail = 4};
static const VkMemSpan bitmap_span = {.pool = &pool, .buffer = MIRROR, .offset = 4 * BITMAP_FIRST,
                                      .avail = 4 * BITMAP_WORDS};
static VkCtx context;

static int span_capture_match(const VkMemSpanCapture *hold, const VkMemSpan *span) {
    return hold && hold->current && hold->provider == span->pool && span->buffer == hold->provider->mbuf &&
        span->offset == hold->address - GUEST;
}
const RcInterval *mem_backing_intervals(const MemBackingCapture *host, uint32_t *count) {
    assert(host == (MemBackingCapture *)&backing);
    *count = 1;
    return &backing;
}

int vk_exec_shared(void) { return 0; }
RcRegistryPosition vk_exec_resource_position(void) { return recorder; }
RcRegistry *rc_registry_runtime(void) { return registry; }
VkCtx *vk_ctx(void) { return &context; }
int vk_mem_span_capture_current(const VkMemSpanCapture *hold, const VkMemSpan *span) {
    return span_capture_match(hold, span);
}
VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *hold) {
    if (hold) { assert(hold->refs); ++hold->refs; }
    return hold;
}
void vk_mem_span_capture_release(VkMemSpanCapture **owned) {
    if (owned && *owned) { assert((*owned)->refs > 1); --(*owned)->refs; *owned = NULL; }
}
static uint32_t copy_pool_host_intervals(const PoolRec *p, uint64_t addr, uint64_t size, RcInterval **out) {
    assert(p == &pool && addr >= GUEST && addr + size <= GUEST + 4 * WORDS);
    *out = malloc(sizeof **out);
    assert(*out);
    **out = (RcInterval){addr, backing.mapping, {BACKING, CANONICAL + addr - GUEST, size}};
    return 1;
}
int rt_knob(const char *name, const char *value) {
    assert(!strcmp(name, "GRAPHICS_PREP_TRACE") && !strcmp(value, "0"));
    return 0;
}
void rt_log(const char *fmt, ...) { (void)fmt; abort(); }
void rt_fatal(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    abort();
}

#define plat_mutex_lock(lock) ((void)(lock))
#define plat_mutex_unlock(lock) ((void)(lock))
#define plat_mutex_lock_shared(lock) ((void)(lock))
#define plat_mutex_unlock_shared(lock) ((void)(lock))
static int pool_lock, mirror_stat_lock;

typedef struct {
    void (*done)(void *);
    void (*cancel)(void *, int);
    void *arg;
} Callback;
static Callback callbacks[12];
static unsigned callback_count;
void vk_completion_add_owned(uint64_t fence, void (*done)(void *), void (*cancel)(void *, int), void *arg) {
    assert(fence == SCENE && callback_count < sizeof callbacks / sizeof callbacks[0]);
    callbacks[callback_count++] = (Callback){done, cancel, arg};
}

/* production declarations */

typedef struct { PoolRec *r; uint64_t off, size; } MRange;
static MRange *wbl;
static int nwbl, cwbl, lz_nskip, nsnl, nswl;
static struct { VkMemSpan sp; uint64_t size; } lz_skip[2];
static SnapRange snapshots[1];
static SnapRange *snl = snapshots;
static struct { VkBuffer buf; uint64_t size; } chunks[] = {{SNAPSHOT, 4 * (WORDS + 3)}};
static const int nchunks = sizeof chunks / sizeof chunks[0];
static int shadow_on, mirror_verify;
static uint64_t snap_completion_fence;
static struct { uint64_t merge_bytes, wb_bytes, wb_copies, wb_ranges; } macc, mrep;
int vk_mirror_on = 1;

/* production visibility */

static unsigned word_releases, bitmap_releases, stage_index;
static unsigned notified[2][WORDS];
static void native_rmw_word_release(void *opaque) {
    VkMemNativeRmwWord *word = opaque;
    assert(word->references);
    if (--word->references) return;
    vk_mem_span_capture_release(&word->provider);
    word_releases++;
}
static void native_rmw_bitmap_release(void *opaque) {
    VkMemNativeRmwBitmap *bitmap = opaque;
    assert(bitmap->references);
    if (--bitmap->references) return;
    vk_mem_span_capture_release(&bitmap->provider);
    bitmap_releases++;
}
VkResult vmaInvalidateAllocation(VmaAllocator vma, VmaAllocation allocation, VkDeviceSize off, VkDeviceSize size) {
    assert(!vma && !off && size == 8 && ((uintptr_t)allocation == (uintptr_t)WORD_MARKER ||
                                        (uintptr_t)allocation == (uintptr_t)BITMAP_MARKER));
    return VK_SUCCESS;
}
void vk_mem_buffer_write_completed(void *provider, VkBuffer buffer, uint64_t addr, uint64_t size, int mirrored) {
    assert(provider == &pool && buffer == MIRROR && mirrored && stage_index < 2);
    assert(addr >= GUEST && addr + size <= GUEST + 4 * WORDS && ((addr | size) & 3) == 0);
    for (uint64_t word = (addr - GUEST) / 4; word < (addr - GUEST + size) / 4; ++word)
        notified[stage_index][word]++;
}
static int native_rmw_bitmap_overlaps(const PoolRec *, uint64_t, uint64_t);
static const char *native_rmw_bitmap_tail_refusal(const PoolRec *, uint64_t, uint64_t, uint64_t,
                                                  VkMemNativeRmwBitmapFailure *);

/* production receipts */

typedef struct {
    VkPipeline pipeline;
    VkDescriptorBufferInfo buffers[4];
    uint32_t push[6], count;
    VkBuffer copy_src, copy_dst;
    VkBufferCopy copy;
    SnapRange ordinary;
} Command;
static Command commands[64], next_command;
static unsigned command_count;
static int marker_copy_ready, marker_copy_needs_visibility;
static uint32_t imported[WORDS], mirrored[WORDS], donor[WORDS + 3];
static uint32_t word_marker[2], bitmap_marker[3];
static unsigned physical_write_count[WORDS];
static uint32_t *src, *mirror_view, *marks, *dst;
static uint32_t invocation;
typedef uint32_t uint;
static struct { uint32_t mode, count, source_word, mirror_word, mask_word, mask_plane; } pc;
static struct { uint32_t mode, source_index, mirror_index, import_index; } selection;

static void write_import(unsigned index, uint32_t value) {
    assert(dst >= imported && dst < imported + WORDS && index < (unsigned)(imported + WORDS - dst));
    physical_write_count[dst + index - imported]++;
    dst[index] = value;
}
#define atomicOr(target, bits) ((target) |= (bits))
/* production kernels */
#undef atomicOr

static uint32_t *buffer_range(VkDescriptorBufferInfo info) {
    assert((info.offset | info.range) % 4 == 0 && info.range);
    uint32_t *memory = NULL;
    size_t size = 0;
    if (info.buffer == IMPORT) { memory = imported; size = sizeof imported; }
    else if (info.buffer == MIRROR) { memory = mirrored; size = sizeof mirrored; }
    else if (info.buffer == SNAPSHOT) { memory = donor; size = sizeof donor; }
    else if (info.buffer == WORD_MARKER) { memory = word_marker; size = sizeof word_marker; }
    else if (info.buffer == BITMAP_MARKER) { memory = bitmap_marker; size = sizeof bitmap_marker; }
    assert(memory && info.offset <= size && info.range <= size - info.offset);
    return memory + info.offset / 4;
}
static void queue_command(Command command) {
    assert(command_count < sizeof commands / sizeof commands[0]);
    commands[command_count++] = command;
}
static void gpu_execute_tail(void) {
    for (unsigned i = 0; i < command_count; ++i) {
        const Command *cmd = &commands[i];
        if (cmd->copy.size) {
            uint32_t *source = buffer_range((VkDescriptorBufferInfo){cmd->copy_src, cmd->copy.srcOffset, cmd->copy.size});
            uint32_t *target = buffer_range((VkDescriptorBufferInfo){cmd->copy_dst, cmd->copy.dstOffset, cmd->copy.size});
            memmove(target, source, (size_t)cmd->copy.size);
        } else if (cmd->ordinary.size) {
            const SnapRange s = cmd->ordinary;
            assert(s.r == &pool && s.chunk == 0);
            dst = imported;
            for (uint64_t j = 0; j < s.size / 4; ++j) {
                const unsigned target = (unsigned)(s.off / 4 + j);
                const unsigned before = (unsigned)(s.soff / 4 + j);
                if (mirrored[target] != donor[before]) write_import(target, mirrored[target]);
            }
        } else {
            src = buffer_range(cmd->buffers[0]);
            mirror_view = buffer_range(cmd->buffers[1]);
            marks = buffer_range(cmd->buffers[2]);
            dst = buffer_range(cmd->buffers[3]);
            if (cmd->pipeline == BITMAP_PIPE) {
                memcpy(&pc, cmd->push, sizeof pc);
                assert(cmd->count == (pc.count + 63) / 64);
                for (invocation = 0; invocation < cmd->count * 64; ++invocation) bitmap_kernel();
            } else {
                assert(cmd->pipeline == WORD_PIPE && cmd->count == 1);
                memcpy(&selection, cmd->push, sizeof selection);
                word_kernel();
            }
        }
    }
}
static void mbarrier(VkCommandBuffer cb, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
    VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    assert(cb == TAIL && src_stage && src_access && dst_stage && dst_access);
    if (dst_stage == VK_PIPELINE_STAGE_2_COPY_BIT) {
        assert(src_stage == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT && src_access == VK_ACCESS_2_MEMORY_WRITE_BIT &&
               (dst_access & VK_ACCESS_2_TRANSFER_READ_BIT) && (dst_access & VK_ACCESS_2_TRANSFER_WRITE_BIT));
        marker_copy_ready = 1;
    }
    if (marker_copy_needs_visibility && dst_stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) {
        assert(src_stage == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT && src_access == VK_ACCESS_2_MEMORY_WRITE_BIT &&
               (dst_access & VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
        marker_copy_needs_visibility = 0;
    }
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer source, VkBuffer target,
                                          uint32_t count, const VkBufferCopy *copies) {
    assert(cb == TAIL && marker_copy_ready && count == 1 && copies);
    assert(source == WORD_MARKER && target == WORD_MARKER &&
           copies[0].srcOffset == 0 && copies[0].dstOffset == 4 && copies[0].size == 4);
    marker_copy_needs_visibility = 1;
    queue_command((Command){.copy_src = source, .copy_dst = target, .copy = copies[0]});
}
VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipeline pipeline) {
    assert(cb == TAIL && point == VK_PIPELINE_BIND_POINT_COMPUTE);
    next_command.pipeline = pipeline;
}
void rec_push(PFN_vkCmdPushDescriptorSetKHR fn, VkCommandBuffer cb, VkPipelineBindPoint point,
              VkPipelineLayout layout, uint32_t set, uint32_t count, const VkWriteDescriptorSet *writes) {
    assert(!fn && cb == TAIL && point == VK_PIPELINE_BIND_POINT_COMPUTE && !layout && !set && count == 4);
    for (unsigned i = 0; i < count; ++i) {
        assert(writes[i].dstBinding == i && writes[i].descriptorCount == 1 &&
               writes[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        next_command.buffers[i] = *writes[i].pBufferInfo;
    }
}
VKAPI_ATTR void VKAPI_CALL vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout layout,
    VkShaderStageFlags flags, uint32_t offset, uint32_t size, const void *data) {
    assert(cb == TAIL && !layout && flags == VK_SHADER_STAGE_COMPUTE_BIT && !offset);
    assert(size == (next_command.pipeline == BITMAP_PIPE ? sizeof pc : sizeof selection));
    memcpy(next_command.push, data, size);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    assert(cb == TAIL && !marker_copy_needs_visibility && x && y == 1 && z == 1);
    next_command.count = x;
    queue_command(next_command);
}

static int wb_merge_record(VkCommandBuffer cb, const SnapRange *ranges, int count) {
    assert(cb == TAIL);
    int recorded = 0;
    for (int i = 0; i < count; ++i) {
        const SnapRange r = ranges[i];
        if (native_rmw_bitmap_overlaps(r.r, r.off, r.size)) continue;
        queue_command((Command){.ordinary = r});
        ++recorded;
    }
    return recorded;
}
static void lazy_part_check(void) {}
static void swb_job(uint64_t fence) { (void)fence; abort(); }
static void snap_transfer(uint64_t fence) { assert(fence == SCENE && snap_completion_fence == 0); }
int vk_mem_mirror_pending(void) { return 0; }
int vk_gpu_passes_capturing(void) { return 0; }
void vk_gpu_gap_boundary(VkCommandBuffer cb) { (void)cb; abort(); }
void vk_gpu_gap_note(int kind, uint64_t units) { (void)kind; (void)units; abort(); }
static int verify_record(VkCommandBuffer cb, uint64_t fence) { (void)cb; (void)fence; abort(); }
static void range_add(MRange **ranges, int *count, int *cap, PoolRec *p, uint64_t off, uint64_t size) {
    (void)ranges; (void)count; (void)cap; (void)p; (void)off; (void)size;
    abort();
}
static int copy_ranges(VkCommandBuffer cb, MRange *ranges, int count, int to_mirror,
                       uint64_t *bytes, uint64_t *copies, CpuVisibility *visibility, uint64_t upload_scene) {
    (void)cb; (void)ranges; (void)count; (void)to_mirror; (void)bytes; (void)copies; (void)visibility; (void)upload_scene;
    abort();
}

/* production tail */

static NativeRmwTail workspace;
static NativeRmwBitmapEdge edges[8];
static NativeRmwBitmapSlice slices[8];
static VkMemNativeRmwWord scalar;
static VkMemNativeRmwBitmap *bitmap;
static uint64_t source_registration = 100;

static void begin_case(void) {
    registry = rc_registry_create((RcRegistryConfig){128, 128, 16});
    assert(registry && !callback_count && !RW.scene && !RB.scene);
    context.props.limits.maxComputeWorkGroupCount[0] = 1;
    context.props.limits.minStorageBufferOffsetAlignment = 16;
    context.props.limits.maxStorageBufferRange = 1024;
    RW.pipeline = WORD_PIPE;
    RB.pipeline = BITMAP_PIPE;
    RB.edge = edges;
    RB.slice = slices;
    RB.capacity = sizeof edges / sizeof edges[0] / 2;
    word_hold = (VkMemSpanCapture){GUEST + word_span.offset, word_span.avail, 9,
        &pool, &pool, (MemBackingCapture *)&backing, 1, 1, 1};
    bitmap_hold = (VkMemSpanCapture){GUEST + bitmap_span.offset, bitmap_span.avail, 9,
        &pool, &pool, (MemBackingCapture *)&backing, 1, 1, 1};
    memset(&workspace, 0, sizeof workspace);
    memset(word_marker, 0, sizeof word_marker);
    memset(bitmap_marker, 0, sizeof bitmap_marker);
    memset(notified, 0, sizeof notified);
    memset(physical_write_count, 0, sizeof physical_write_count);
    word_releases = bitmap_releases = command_count = 0;
    marker_copy_ready = marker_copy_needs_visibility = 0;
    for (unsigned i = 0; i < WORDS; ++i)
        mirrored[i] = imported[i] = UINT32_C(0x3c120000) + 0x101u * i;
    memset(donor, 0xdd, sizeof donor);
    // Offset the saved payload within its allocation, and straddle both bitmap boundaries.
    snapshots[0] = (SnapRange){&pool, 8, 32, 16, 0};
    memcpy(donor + 4, mirrored + 2, 32);
    nsnl = 1;
    scalar = (VkMemNativeRmwWord){.descriptor = word_span, .provider = vk_mem_span_capture_retain(&word_hold),
        .marker = WORD_MARKER, .allocation = (VmaAllocation)(uintptr_t)WORD_MARKER, .mapped = word_marker,
        .workspace = &workspace, .marker_generation = 1, .references = 1};
    OK(selected_mirror_access_locked(&word_hold, &word_span, word_hold.address, 4, 4, word_read, &scalar.source));
    const unsigned extent_capacity = (BITMAP_WORDS + 1) / 2;
    bitmap = calloc(1, sizeof *bitmap + 2 * extent_capacity * sizeof bitmap->extents[0]);
    assert(bitmap);
    *bitmap = (VkMemNativeRmwBitmap){.descriptor = bitmap_span, .provider = vk_mem_span_capture_retain(&bitmap_hold),
        .marker = BITMAP_MARKER, .allocation = (VmaAllocation)(uintptr_t)BITMAP_MARKER, .mapped = bitmap_marker,
        .words = BITMAP_WORDS, .map_dwords = 1, .references = 1};
    OK(selected_mirror_access_locked(&bitmap_hold, &bitmap_span, bitmap_hold.address,
        bitmap_hold.size, bitmap_hold.size, bitmap_read, &bitmap->source));
}
static void end_case(void) {
    assert(!RW.scene && !RB.scene && !RW.word_count && !RB.count && !callback_count);
    assert(word_hold.refs == 1 && bitmap_hold.refs == 1 && word_releases == 1 && bitmap_releases == 1);
    assert(!rc_registry_stats(registry).fail_closed);
    rc_registry_destroy(registry);
    free(bitmap);
}
static void position(RcRegistryPosition read) {
    recorder = read;
    OK(rc_registry_declare_through(registry, read));
}
static void declare_word(void) {
    position(word_read);
    OK(vk_mem_native_rmw_word_declare(&scalar, word_output));
    position(word_output);
}
static void issue_word(VkCommandBuffer cb) {
    VkMemNativeRmwWord *owned = &scalar;
    assert(vk_mem_native_rmw_word_bind(&scalar, cb));
    vk_mem_native_rmw_word_issued(&owned, cb);
    assert(!owned && RW.scene == &scalar);
}
static void issue_bitmap(VkPipelineStageFlags2 stage, VkCommandBuffer cb) {
    bitmap->writer_stage = stage;
    VkMemNativeRmwBitmapFailure failure;
    if (!vk_mem_native_rmw_bitmap_current_report(bitmap, &bitmap_span, bitmap_read, &failure)) {
        fprintf(stderr, "compatible earlier issued scalar refused by %s (registry %d)\n", failure.gate, failure.registry);
        exit(1);
    }
    OK(vk_mem_native_rmw_bitmap_declare(bitmap, bitmap_output));
    position(bitmap_output);
    assert(vk_mem_native_rmw_bitmap_bind(bitmap, cb));
    VkMemNativeRmwBitmap *owned = bitmap;
    vk_mem_native_rmw_bitmap_issued(&owned, cb);
    assert(!owned && RB.scene == bitmap);
}
static void cancel_word(void) {
    VkMemNativeRmwWord *owned = &scalar;
    vk_mem_native_rmw_word_cancel(&owned);
    assert(!owned);
}
static void cancel_bitmap(void) {
    VkMemNativeRmwBitmap *owned = bitmap;
    vk_mem_native_rmw_bitmap_cancel(&owned);
    assert(!owned);
}
static RcRegistryResult authority(uint64_t backing_id, unsigned word) {
    const uint64_t base = backing_id == BACKING ? CANONICAL : GUEST;
    RcRegistryHandle handle;
    RcRegistryObservation observation;
    OK(rc_registry_source(registry, source_registration++, (RcSourceRange){backing_id, base + 4 * word, 4}, &handle));
    return rc_registry_observe(registry, handle, 1, &observation);
}
static void finish_callbacks(int mode) {
    for (unsigned i = 0; i < callback_count; ++i) {
        const Callback c = callbacks[i];
        if (mode) {
            c.cancel(c.arg, 0);
        } else {
            if (c.done == native_rmw_word_completed) stage_index = 0;
            if (c.done == native_rmw_bitmap_completed) stage_index = 1;
            c.done(c.arg);
        }
    }
    callback_count = 0;
}

enum Action { SKIP, EQUAL, CHANGE, RESTORE };
typedef struct { enum Action first, later; int donor_word, canceled, stale_completion; VkPipelineStageFlags2 stage; } Scenario;
static void run_scenario(Scenario s) {
    begin_case();
    declare_word();
    issue_word(BODY);
    position(bitmap_read);
    assert(!rc_registry_position_handed(registry, word_output));
    OK(rc_registry_buffer_access_check(registry, bitmap->source.native, bitmap_read));
    issue_bitmap(s.stage, BODY);
    assert(authority(BACKING, OVERLAP) == RC_REGISTRY_BLOCKED);
    assert(authority(RC_GUEST_COMPATIBILITY_BACKING, OVERLAP) == RC_REGISTRY_BLOCKED);

    const uint32_t preimage = mirrored[OVERLAP];
    const uint32_t equal_preimage = mirrored[BITMAP_EQUAL];
    if (s.donor_word) mirrored[OVERLAP] = preimage + 3;
    mirrored[SNL_ONLY] ^= UINT32_C(0x81020304);
    const uint32_t expected_first = preimage + (s.donor_word ? 3 : 0);
    const uint32_t expected_later = expected_first - (s.first == CHANGE ? 1 : 0);
    uint32_t first = 0, later = 0;
    if (!s.canceled || s.canceled == 2) {
        if (s.first != SKIP) {
            first = mirrored[OVERLAP];
            if (s.first == CHANGE) --mirrored[OVERLAP];
            word_marker[0] = 1;
        }
        assert(first == (s.first == SKIP ? 0 : expected_first));
    }
    if (!s.canceled) {
        if (s.later != SKIP) {
            later = mirrored[OVERLAP];
            if (s.later == CHANGE) mirrored[OVERLAP] ^= UINT32_C(0x10203040);
            if (s.later == RESTORE) mirrored[OVERLAP] = preimage;
            bitmap_marker[1] |= 1u << (OVERLAP - BITMAP_FIRST);
        }
        assert(later == (s.later == SKIP ? 0 : expected_later));
        const uint32_t equal_return = mirrored[BITMAP_EQUAL];
        bitmap_marker[1] |= 1u << (BITMAP_EQUAL - BITMAP_FIRST);
        assert(equal_return == equal_preimage);
    }
    const uint32_t original_bitmap_marks = bitmap_marker[1];

    // Inject distinct one-byte CPU edits into every word after recording the guest producers.
    uint32_t late_cpu[WORDS];
    for (unsigned i = 0; i < WORDS; ++i) {
        ((unsigned char *)imported)[4 * i + 1] ^= (unsigned char)(0x81u + i);
        late_cpu[i] = imported[i];
    }
    assert(vk_mem_mirror_end(TAIL, SCENE) > 0);
    assert(memcmp(imported, late_cpu, sizeof imported) == 0 && command_count && !RW.scene && !RB.scene);
    if (!s.canceled) {
        OK(rc_registry_handoff_through(registry, bitmap_output));
        gpu_execute_tail();
        assert(bitmap_marker[1] == original_bitmap_marks && word_marker[0] == (s.first != SKIP));
        if (s.stale_completion) {
            word_hold.current = bitmap_hold.current = 0;
        }
    } else if (s.canceled == 2) {
        OK(rc_registry_handoff_through(registry, word_output));
    }
    finish_callbacks(s.canceled);

    for (unsigned i = 0; i < WORDS; ++i) {
        const int scalar_executed = i == OVERLAP && !s.canceled && s.first != SKIP;
        const int bitmap_executed = !s.canceled && (i == BITMAP_EQUAL || (i == OVERLAP && s.later != SKIP));
        const int snl_difference = !s.canceled && (i == SNL_ONLY || (i == OVERLAP && s.donor_word));
        const unsigned writes = scalar_executed || bitmap_executed || snl_difference;
        assert(physical_write_count[i] == writes);
        assert(imported[i] == (writes ? mirrored[i] : late_cpu[i]));
        assert(notified[0][i] == (unsigned)(scalar_executed && !s.stale_completion));
        assert(notified[1][i] == (unsigned)(bitmap_executed && !s.stale_completion));
    }
    for (unsigned i = 0; i < WORDS; ++i) {
        const int bitmap_range = i >= BITMAP_FIRST && i < BITMAP_FIRST + BITMAP_WORDS;
        const int blocked = (s.stale_completion && bitmap_range) || (s.canceled == 2 && i == OVERLAP);
        const RcRegistryResult expected = blocked ? RC_REGISTRY_BLOCKED : RC_REGISTRY_OK;
        assert(authority(BACKING, i) == expected);
        assert(authority(RC_GUEST_COMPATIBILITY_BACKING, i) == expected);
    }
    assert(imported[BITMAP_SKIP] == late_cpu[BITMAP_SKIP]);
    end_case();
}

static void reverse_overlap(int bitmap_executes, int word_executes) {
    begin_case();
    position(bitmap_read);
    issue_bitmap(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, BODY);
    scalar.source.read = (RcRegistryPosition){SCENE, 6};
    const RcRegistryPosition output = {SCENE, 7};
    position(scalar.source.read);
    OK(rc_registry_buffer_access_check(registry, scalar.source.native, scalar.source.read));
    OK(vk_mem_native_rmw_word_declare(&scalar, output));
    position(output);
    assert(!vk_mem_native_rmw_word_bind(&scalar, OTHER_BODY));
    issue_word(BODY);

    const uint32_t preimage = mirrored[OVERLAP];
    if (bitmap_executes) {
        mirrored[OVERLAP] = preimage + 1;
        bitmap_marker[1] |= 1u << (OVERLAP - BITMAP_FIRST);
    }
    const uint32_t marks = bitmap_marker[1];
    if (word_executes) {
        assert(mirrored[OVERLAP] == preimage + bitmap_executes);
        // A scalar equal-result atomic still owns its exact shared word.
        word_marker[0] = 1;
    }
    const uint32_t late = imported[OVERLAP] ^ UINT32_C(0x900);
    imported[OVERLAP] = late;
    assert(vk_mem_mirror_end(TAIL, SCENE) > 0 && imported[OVERLAP] == late);
    OK(rc_registry_handoff_through(registry, output));
    gpu_execute_tail();
    assert(bitmap_marker[1] == marks && word_marker[0] == (unsigned)word_executes);
    finish_callbacks(0);
    assert(physical_write_count[OVERLAP] == (unsigned)(bitmap_executes || word_executes));
    assert(imported[OVERLAP] == (bitmap_executes || word_executes ? mirrored[OVERLAP] : late));
    assert(notified[0][OVERLAP] == (unsigned)word_executes && notified[1][OVERLAP] == (unsigned)bitmap_executes);
    assert(authority(BACKING, OVERLAP) == RC_REGISTRY_OK);
    end_case();
}

static void refusals(void) {
    begin_case();
    declare_word();
    position(bitmap_read);
    VkMemNativeRmwBitmapFailure failure;
    assert(!vk_mem_native_rmw_bitmap_current_report(bitmap, &bitmap_span, bitmap_read, &failure));
    assert(failure.registry == RC_REGISTRY_BLOCKED);
    assert(!RW.scene);
    cancel_word();
    assert(vk_mem_native_rmw_bitmap_current_report(bitmap, &bitmap_span, bitmap_read, &failure));
    cancel_bitmap();
    end_case();

    begin_case();
    declare_word();
    issue_word(BODY);
    position(bitmap_read);
    assert(vk_mem_native_rmw_bitmap_current_report(bitmap, &bitmap_span, bitmap_read, &failure));
    backing.mapping++;
    assert(!vk_mem_native_rmw_bitmap_current_report(bitmap, &bitmap_span, bitmap_read, &failure));
    backing.mapping--;
    VkMemSpan stale_address = bitmap_span;
    stale_address.offset += 4;
    assert(!vk_mem_native_rmw_bitmap_current_report(bitmap, &stale_address, bitmap_read, &failure));
    // A native allocation can leave valid canonical tickets behind on a different retained buffer.
    scalar.source.mirror = (VkBuffer)(uintptr_t)99;
    assert(!vk_mem_native_rmw_bitmap_current_report(bitmap, &bitmap_span, bitmap_read, &failure));
    assert(failure.gate && !strcmp(failure.gate, "bitmap-word-mirror"));
    scalar.source.mirror = MIRROR;
    scalar.output.ordinal = bitmap_read.ordinal + 1;
    assert(!vk_mem_native_rmw_bitmap_current_report(bitmap, &bitmap_span, bitmap_read, &failure));
    assert(failure.gate && !strcmp(failure.gate, "bitmap-word-order"));
    scalar.output = word_output;
    RcRegistryWrite unknown;
    OK(rc_registry_write_begin(registry, (RcSourceRange){BACKING, CANONICAL + 4 * OVERLAP, 4},
                               RC_REGISTRY_BUFFER, &unknown));
    assert(!vk_mem_native_rmw_bitmap_current_report(bitmap, &bitmap_span, bitmap_read, &failure));
    assert(failure.registry == RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(registry, unknown));
    cancel_bitmap();
    assert(vk_mem_mirror_end(TAIL, SCENE) > 0);
    finish_callbacks(1);
    end_case();
}

static void separated_command_buffers(void) {
    for (int prefix_handed = 0; prefix_handed <= 1; ++prefix_handed) {
        begin_case();
        declare_word();
        issue_word(BODY);
        if (prefix_handed) OK(rc_registry_handoff_through(registry, word_output));
        position(bitmap_read);
        assert(vk_mem_native_rmw_bitmap_current(bitmap, &bitmap_span, bitmap_read));
        OK(vk_mem_native_rmw_bitmap_declare(bitmap, bitmap_output));
        position(bitmap_output);
        assert(vk_mem_native_rmw_bitmap_bind(bitmap, OTHER_BODY) == prefix_handed);
        cancel_bitmap();
        assert(vk_mem_mirror_end(TAIL, SCENE) > 0);
        finish_callbacks(1);
        assert(authority(BACKING, OVERLAP) == (prefix_handed ? RC_REGISTRY_BLOCKED : RC_REGISTRY_OK));
        end_case();
    }
}

int main(void) {
    const VkPipelineStageFlags2 fragment = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    const VkPipelineStageFlags2 mesh = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
    const Scenario scenarios[] = {
        {CHANGE, SKIP, 0, 0, 0, fragment}, {EQUAL, SKIP, 0, 0, 0, mesh},
        {SKIP, SKIP, 0, 0, 0, fragment}, {SKIP, SKIP, 1, 0, 0, mesh},
        {CHANGE, CHANGE, 0, 0, 0, mesh}, {CHANGE, EQUAL, 0, 0, 0, fragment},
        {CHANGE, RESTORE, 0, 0, 0, fragment}, {SKIP, CHANGE, 0, 0, 0, mesh},
        {CHANGE, CHANGE, 0, 1, 0, fragment}, {CHANGE, CHANGE, 0, 2, 0, mesh},
        {CHANGE, EQUAL, 0, 0, 1, fragment},
    };
    for (unsigned i = 0; i < sizeof scenarios / sizeof scenarios[0]; ++i) {
        fprintf(stderr, "ordered overlap case %u\n", i + 1);
        run_scenario(scenarios[i]);
    }
    for (int bitmap = 0; bitmap <= 1; ++bitmap)
        for (int scalar = 0; scalar <= 1; ++scalar) reverse_overlap(bitmap, scalar);
    refusals();
    separated_command_buffers();
    puts("ordered scalar and fragment/mesh bitmap: both orders, one sparse writeback, exact execution flags, late CPU bytes, cancellation and authority passed");
    return 0;
}

