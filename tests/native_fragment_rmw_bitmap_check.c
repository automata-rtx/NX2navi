#include "vk.h"
#include "resource_core_budget.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    VkBuffer buf, mbuf;
    uint64_t publication, import_lo, import_hi;
    int kind, smap, gpa_host_coherent; uint32_t acc_sub;
} PoolRec;
enum { POOL_IMPORT = 1 };
struct VkMemSpanCapture {
    uint64_t address, size, mirror_incarnation;
    PoolRec *provider, *window;
    MemBackingCapture *host; int native_input;
};
typedef struct { PoolRec *r; uint64_t off, size, soff; int chunk; } SnapRange;
static const SnapRange *snl;
static struct { uint64_t size; } chunks[6];
static int nsnl, nchunks = sizeof chunks / sizeof chunks[0];
static uint32_t msub = 1;
static RcInterval backing;
static int span_capture_match(const VkMemSpanCapture *capture, const VkMemSpan *span) {
    return capture->provider == span->pool;
}
const RcInterval *mem_backing_intervals(const MemBackingCapture *host, uint32_t *count) {
    assert(host);
    *count = 1;
    return &backing;
}
int nvn_gpu_only(uint64_t addr) { return addr >= UINT64_C(0x8000000000); }

/* production declarations */

struct MeshWordFixture {
    struct MeshWordFixture *next;
    VkMemSpan descriptor;
    SelectedMirrorWord source;
    RcRegistryPosition output;
};
static struct { struct MeshWordFixture *scene; } RW;
typedef struct MeshWordFixture VkMemNativeRmwWordFixture;
#define VkMemNativeRmwWord VkMemNativeRmwWordFixture
static struct { PoolRec *r; uint64_t off, size; } wbl[2];
static struct { VkMemSpan sp; uint64_t size; } lz_skip[2];
static int nwbl, lz_nskip;
static VkCtx context;
VkCtx *vk_ctx(void) { return &context; }

static int current, handed, fail_closed;
static unsigned begins, submitted, finished, complete_attempts, physical, releases, abandons, unknown, access_records;
static RcRegistryResult completion_result;
static VkMemNativeRmwBitmap *pending;
static void (*done_cb)(void *);
static void (*cancel_cb)(void *, int);
static RcSourceRange visibility[8], canonical[8], fallback[8];
static unsigned actual_count, visible_count, uncertainty[2];
RcRegistry *rc_registry_runtime(void) { return NULL; }
void rc_registry_fail_closed(RcRegistry *registry) { assert(!registry); ++fail_closed; }
int rc_registry_position_handed(RcRegistry *registry, RcRegistryPosition pos) {
    assert(!registry && pos.scene == 7 && pos.ordinal == 3);
    return handed;
}
RcRegistryResult rc_registry_write_begin_extents(RcRegistry *registry, const RcRegistryWriteExtent *extents,
    uint32_t count, RcRegistryPosition out, RcRegistryWrite *tickets) {
    assert(!registry && out.scene == 7 && out.ordinal == 3 && count == 2 && !begins);
    assert(extents[0].range.backing == 42 && extents[1].range.backing == RC_GUEST_COMPATIBILITY_BACKING);
    assert(extents[0].range.size == extents[1].range.size);
    for (unsigned i = 0; i < count; ++i) {
        assert(extents[i].kind == RC_REGISTRY_BUFFER);
        tickets[i] = (RcRegistryWrite){.id = i + 1, .index = i + 1};
    }
    begins += count;
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_buffer_access_record(RcRegistry *registry, RcRegistryWrite write, RcRegistryBufferSpan span,
    RcRegistryPosition out) {
    assert(!registry && write.id && span.native.allocation == 8 && span.native.incarnation == 9);
    assert(span.interval.mapping == 3 && span.interval.range.backing == 42 && out.scene == 7 && out.ordinal == 3);
    ++access_records;
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_write_abandon(RcRegistry *registry, RcRegistryWrite write, int may_have_executed) {
    (void)registry; (void)write; (void)may_have_executed;
    abort(); // Both tickets must be settled together.
}
static void check_pair(RcRegistry *registry, const RcRegistryWrite *tickets, uint32_t count) {
    assert(!registry && count == 2 && tickets[0].id == 1 && tickets[1].id == 2);
}
RcRegistryResult rc_registry_write_submit_extents(RcRegistry *registry, const RcRegistryWrite *tickets,
    uint32_t count, uint64_t scene) {
    check_pair(registry, tickets, count);
    assert(!submitted && scene == 7);
    submitted += count;
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_write_cancel_extents(RcRegistry *registry, const RcRegistryWrite *tickets,
    uint32_t count) {
    check_pair(registry, tickets, count);
    assert(!submitted && !abandons);
    abandons += count;
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_write_abandon_extents(RcRegistry *registry, const RcRegistryWrite *tickets,
    uint32_t count) {
    check_pair(registry, tickets, count);
    assert(!abandons);
    abandons += count;
    unknown += count;
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_write_complete_extents(RcRegistry *registry, const RcRegistryWriteCompletion *c, uint32_t n) {
    assert(!registry && n == 2 && submitted == 2 && c[0].written_count == c[1].written_count);
    ++complete_attempts;
    if (completion_result != RC_REGISTRY_OK) return completion_result;
    ++finished;
    actual_count = c[0].written_count;
    assert(actual_count < 8);
    for (unsigned i = 0; i < actual_count; ++i) {
        canonical[i] = c[0].written[i];
        fallback[i] = c[1].written[i];
    }
    uncertainty[0] = c[0].uncertain_count;
    uncertainty[1] = c[1].uncertain_count;
    return RC_REGISTRY_OK;
}
int vk_mem_span_capture_current(const VkMemSpanCapture *capture, const VkMemSpan *span) {
    assert(capture && span->pool);
    return current;
}
VkResult vmaInvalidateAllocation(VmaAllocator a, VmaAllocation b, VkDeviceSize off, VkDeviceSize size) {
    assert(!a && b && !off && size == 12);
    ++physical;
    return VK_SUCCESS;
}
void vk_mem_buffer_write_completed(void *pool, VkBuffer mirror, uint64_t addr, uint64_t size, int mirrored) {
    assert(pool && mirror && mirrored && visible_count < 8);
    visibility[visible_count++] = (RcSourceRange){RC_GUEST_COMPATIBILITY_BACKING, addr, size};
}
void vk_completion_add_owned(uint64_t scene, void (*done)(void *), void (*cancel)(void *, int), void *opaque) {
    assert(scene == 7 && !pending);
    pending = opaque;
    done_cb = done;
    cancel_cb = cancel;
}
static void native_rmw_bitmap_release(void *opaque) {
    assert(opaque);
    ++releases;
}
int cfg_bool(const char *name) { assert(name); return 0; }
int rt_knob(const char *name, const char *def) {
    assert(!strcmp(name, "GRAPHICS_PREP_TRACE") && !strcmp(def, "0"));
    return 0;
}
void rt_log(const char *fmt, ...) { (void)fmt; assert(0); }
void rt_fatal(const char *fmt, ...) { (void)fmt; assert(0); abort(); }

static PoolRec pool = {.buf = (VkBuffer)22, .mbuf = (VkBuffer)21, .publication = 8, .import_lo = 0x1000, .import_hi = 0x3000,
    .kind = POOL_IMPORT, .gpa_host_coherent = 1, .acc_sub = 1};
static PoolRec disjoint;
static VkMemSpanCapture host = {.address = 0x1000, .size = 512, .mirror_incarnation = 9, .host = (MemBackingCapture *)1,
    .native_input = 1, .provider = &pool, .window = &pool};

/* production functions */

static void prepare_cases(void) {
    backing = (RcInterval){0x1000, 3, {42, 0x200, 256}};
    VkMemSpan span = {.pool = &pool, .buffer = (VkBuffer)21, .offset = 0, .avail = 512};
    SelectedMirrorWord s;
    RcRegistryPosition read = {7, 2};
    assert(selected_mirror_access_locked(&host,&span,0x1000,256,256,read,&s) == RC_REGISTRY_OK);
    assert(s.mirror == span.buffer && s.import == pool.buf && s.native.interval.range.size == 256);
    assert(s.native.interval.range.offset == 0x200 && s.native.native.allocation == 8 && s.native.native.incarnation == 9);
    assert(selected_mirror_access_locked(&host,&span,0x1004,4,256,read,&s) == RC_REGISTRY_OK);
    assert(s.native.interval.range.offset == 0x204 && s.native.native.offset == 4);
    assert(selected_mirror_access_locked(&host,&span,0x1000,260,260,read,&s) == RC_REGISTRY_UNSUPPORTED);
    assert(selected_mirror_access_locked(&host,&span,0x1001,4,256,read,&s) == RC_REGISTRY_UNSUPPORTED);
    span.offset = 0x1ffc;
    assert(selected_mirror_access_locked(&host,&span,0x1000,256,256,read,&s) == RC_REGISTRY_INVALID);
    span.offset = 0;
    pool.acc_sub = 2;
    assert(selected_mirror_access_locked(&host,&span,0x1000,256,256,read,&s) == RC_REGISTRY_UNSUPPORTED);
    pool.acc_sub = 1;
}

static void init_bitmap(VkMemNativeRmwBitmap *r, PoolRec *p, uint64_t off, uint64_t bytes) {
    *r = (VkMemNativeRmwBitmap){.source = {
        .mirror = (VkBuffer)21, .import = (VkBuffer)22,
        .native = {{0x1000 + off, 3, {42, 0x200 + off, bytes}}, {8, 9, off}},
        .descriptor_size = bytes, .read = {7, 2}},
        .descriptor = {.pool = p}, .provider = &host, .output = {7, 3}, .body = (VkCommandBuffer)1};
}

static void check_complement(const NativeRmwBitmapTail *tail, const SnapRange *source,
    const SnapRange *expected, size_t count) {
    size_t next = native_rmw_bitmap_first_clip(tail, source);
    assert(next < tail->count && native_rmw_bitmap_snapshot_clipped(tail, source));
    uint64_t cursor = source->off;
    SnapRange part;
    for (size_t i = 0; i < count; ++i) {
        assert(native_rmw_bitmap_next_complement(tail, source, &next, &cursor, &part));
        const SnapRange wanted = expected[i];
        assert(part.r == wanted.r && part.off == wanted.off && part.size == wanted.size &&
               part.soff == wanted.soff && part.chunk == wanted.chunk);
        assert(!native_rmw_bitmap_snapshot_clipped(tail, &part));
    }
    assert(!native_rmw_bitmap_next_complement(tail, source, &next, &cursor, &part));
}

static void geometry_cases(void) {
    NativeRmwBitmapEdge edges[8];
    NativeRmwBitmapSlice slices[8];
    VkMemNativeRmwBitmap a, b, c;
    init_bitmap(&a,&pool,32,64);
    init_bitmap(&b,&pool,64,64);
    init_bitmap(&c,&pool,160,32);
    a.next = &b;
    b.next = &c;
    RB.edge = edges;
    RB.slice = slices;
    RB.capacity = 4;
    RB.scene = &a;
    RB.count = 3;
    NativeRmwBitmapTail tail = {0};
    assert(native_rmw_bitmap_slices(7,&tail) && tail.count == 4);
    assert(slices[0].owner == &a && slices[0].off == 32 && slices[0].size == 32);
    assert(slices[1].owner == &a && slices[1].off == 64 && slices[1].size == 32);
    assert(slices[2].owner == &b && slices[2].off == 96 && slices[2].size == 32);
    assert(slices[3].owner == &c && slices[3].off == 160 && slices[3].size == 32);
    assert(native_rmw_bitmap_overlaps(&pool,96,8) && !native_rmw_bitmap_overlaps(&pool,128,32));
    assert(!native_rmw_bitmap_overlaps(&disjoint,96,8));

    const SnapRange snapshots[] = {
        {&pool,16,208,320,4}, {&pool,48,64,992,5}, {&disjoint,16,208,32,1}, {&pool,120,172,256,2},
    };
    const SnapRange after_mesh[] = {
        {&pool,28,196,332,4}, {&pool,16,8,320,4}, snapshots[1], snapshots[2], snapshots[3],
    };
    const SnapRange expected[] = {{&pool,28,4,332,4},{&pool,128,32,432,4},{&pool,192,32,496,4}};
    const SnapRange late[] = {{&pool,128,32,264,2},{&pool,192,100,328,2}};
    snl = snapshots;
    nsnl = 3; // The three marker owners already have their preadmitted shapes.
    for (int i = 0; i < nchunks; ++i) chunks[i].size = 2048;
    context.props.limits.maxComputeWorkGroupCount[0] = 1;
    context.props.limits.minStorageBufferOffsetAlignment = 16;
    context.props.limits.maxStorageBufferRange = 96;

    // A later ordinary writer appends a different overlapping snapshot. The mesh ordinary view also
    // omits the independent word at 24, while the prologue still has the four original recipes.
    ++nsnl;
    tail = (NativeRmwBitmapTail){0};
    assert(native_rmw_bitmap_tail_plan(&tail, 7, after_mesh, 5));
    assert(tail.ordinary == after_mesh && tail.ordinary_count == 5 && tail.dispatch_bytes == 80);
    check_complement(&tail, &after_mesh[0], expected, 3);
    check_complement(&tail, &after_mesh[2], NULL, 0);
    check_complement(&tail, &after_mesh[4], late, 2);
    assert(!native_rmw_bitmap_snapshot_clipped(&tail, &after_mesh[1]));
    assert(!native_rmw_bitmap_snapshot_clipped(&tail, &after_mesh[3]));
    assert(snl == snapshots && nsnl == 4 && snapshots[0].size == 208 && snapshots[3].soff == 256);
    assert(native_rmw_spans_overlap(48,64,32,64) && !native_rmw_spans_overlap(32,32,64,32));

    // Pending plain and borrowed lazy ranges are checked by actual overlap.
    nwbl = lz_nskip = 1;
    wbl[0].r = &pool; wbl[0].off = 216; wbl[0].size = 32;
    lz_skip[0].sp.pool = &pool; lz_skip[0].sp.offset = 140; lz_skip[0].size = 8;
    assert(native_rmw_bitmap_tail_supported(&pool,32,96,7));
    assert(!native_rmw_bitmap_tail_supported(&pool,136,16,7));
    assert(!native_rmw_bitmap_tail_supported(&pool,208,16,7));
    nwbl = lz_nskip = 0;

    b.source.native.interval.mapping++;
    tail.count = 0;
    assert(!native_rmw_bitmap_slices(7,&tail));
    b.source.native.interval.mapping--;
    tail.count = 0;
    assert(!native_rmw_bitmap_slices(8,&tail));
    RB.scene = NULL;
    RB.count = 0;
    snl = NULL;
    nsnl = 0;
}

static void reset(void) {
    handed = current = 1;
    fail_closed = 0;
    begins = submitted = finished = complete_attempts = physical = releases = abandons = unknown = access_records = 0;
    completion_result = RC_REGISTRY_OK;
    actual_count = visible_count = uncertainty[0] = uncertainty[1] = 0;
    pending = NULL;
}
static void completion_case(int mode) {
    reset();
    const unsigned words = 35, max_extents = (words + 1) / 2;
    VkMemNativeRmwBitmap *owner = calloc(1,sizeof *owner + 2 * max_extents * sizeof(RcSourceRange));
    init_bitmap(owner,&pool,0,4 * words);
    uint32_t marker[] = {0x80000000u, (1u<<1) | (1u<<31), (1u<<0) | (1u<<2)};
    owner->mapped = marker;
    owner->words = words;
    owner->map_dwords = 2;
    owner->allocation = (VmaAllocation)1;
    owner->bound = owner->body;
    owner->body = NULL;
    owner->tail_recorded = 1;
    const RcRegistryPosition out = {7,3};
    assert(vk_mem_native_rmw_bitmap_declare(owner,(RcRegistryPosition){7,5}) == RC_REGISTRY_INVALID && !begins);
    assert(vk_mem_native_rmw_bitmap_declare(owner,out) == RC_REGISTRY_OK && begins == 2);
    if (mode == 3) {
        VkMemNativeRmwBitmap *loan = owner;
        vk_mem_native_rmw_bitmap_cancel(&loan);
        assert(!loan && !pending && !submitted && !physical && abandons == 2 && !unknown && releases == 1);
    } else {
        VkMemNativeRmwBitmap *loan = owner;
        vk_mem_native_rmw_bitmap_issued(&loan,owner->bound);
        assert(!loan && pending == owner && RB.scene == owner && RB.count == 1 && access_records == 2);
        RB.scene = NULL;
        RB.count = 0;
        if (mode == 4 || mode == 5) {
            handed = mode != 4;
            cancel_cb(owner,0);
            assert(!submitted && !physical && abandons == 2 && unknown == (handed ? 2u : 0u));
        } else {
            current = mode != 1 && mode != 6;
            if (mode == 2) marker[2] |= 1u<<3;
            if (mode == 6) marker[1] = marker[2] = 0;
            if (mode == 7) completion_result = RC_REGISTRY_CAPACITY;
            done_cb(owner);
            if (mode == 2) {
                assert(!submitted && physical == 1 && abandons == 2 && unknown == 2 && !visible_count);
            } else if (mode == 1 || mode == 7) {
                assert(submitted == 2 && physical == 1 && abandons == 2 && unknown == 2 && !visible_count);
                assert(!finished && complete_attempts == (mode == 7 ? 1u : 0u));
            } else if (mode == 6) {
                assert(finished == 1 && submitted == 2 && physical == 1 && !actual_count && !visible_count && !abandons);
            } else {
                assert(finished == 1 && submitted == 2 && physical == 1 && actual_count == 3 && !abandons);
                assert(canonical[0].offset == 0x204 && canonical[0].size == 4);
                assert(canonical[1].offset == 0x27c && canonical[1].size == 8);
                assert(canonical[2].offset == 0x288 && canonical[2].size == 4);
                assert(fallback[0].offset == 0x1004 && fallback[1].offset == 0x107c && fallback[2].offset == 0x1088);
                assert(!uncertainty[0] && !uncertainty[1] && visible_count == 3);
                for (unsigned i=0; i<visible_count; ++i) assert(rc_range_equal(visibility[i],fallback[i]));
            }
        }
        assert(releases == 1 && !fail_closed);
    }
    free(owner);
}

int main(void) {
    prepare_cases();
    geometry_cases();
    for (int i = 0; i < 8; ++i) completion_case(i);
    puts("native fragment RMW bitmap: selected mapping, streamed snapshot complements, precise execution extents, paired stale/capacity abandonment and cancellation passed");
}
