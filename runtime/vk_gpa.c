// The device address table: GPU address pages to Vulkan buffer device addresses, for the global memory accesses the
// translator finds no storage buffer for (translator patch 0030: GlobalToStorage TryKeepDeviceAddressAccess, and the
// SPIR-V back end's GetDeviceAddressPointer, which reads this layout). Ocarina's foliage, instance and light data reach
// their buffers through pointers the shaders read from other buffers at an index (a phi of constant-buffer pointers and
// loaded ones, a pointer spilled to local memory and loaded again); those accesses read 0 before, which drew interiors
// and shadowed areas black and dropped instanced foliage.
//
// One storage buffer of 32-bit words, bound whole to every shader that declares it (res.json "table", cb_slot 0x40):
//   words 0-1    the device address of a zeroed 4 KiB buffer, which an address no page maps reads and writes instead
//   word 2       the word index of the records (REC_WORD)
//   words 3-4    the device address of a 4 KiB scratch buffer, where a twin write goes when its page has no twin
//   words 16..   L1: 65536 entries, one per 16 MiB of the 40-bit GPU address space, the index of its L2 block (0: none)
//   word 65552   L2 blocks of 4096 entries, one per 4 KiB page: the record of the buffer that holds the page (0: none)
//   REC_WORD     records of four words: the buffer's device address minus the GPU address of its byte 0, then the same
//                for the page's twin (low words first)
// The twin: a pool with a VRAM mirror (vk_memory.c mirror_create) has the mirror in its record's first address and its
// imported buffer, guest memory, in the second; any other record has the same buffer in both. Shaders read through the
// first and write through both (translator patch 0034): the GPU's writes through the table are not recorded as GPU
// writes (vk_exec_draw.c ssbo_info), so with the mirror alone the CPU never saw them, and a sync of a page the CPU wrote
// copied the stale guest page over them. The game's compute passes write tree and foliage records this way, and a
// record overwritten so drew a tree stretched into a column reaching the sky (Hyrule Field, Kakariko, Death Mountain).
// A record a page entry names is never written: a new mapping takes a new record, and a record goes back to the free
// list only once the GPU finished every submission issued while a page could still name it (vk_completion_add). Page
// and L1 entries are single 32-bit stores, so a shader that reads the table while it changes sees the old mapping or
// the new one, never half of each. The CPU keeps a copy (shadow) for its reads: the GPU copy is often device memory
// behind the PCIe BAR, slow to read back.
//
// An L2 block whose last page is unmapped leaves its L1 entry at once (blocks_release) and serves another 16 MiB range
// once the GPU finished every submission issued while a shader could still have read the old L1 entry: until then it
// holds only zeros, which read as unmapped. The game's allocator hands out pool addresses that climb through the 64 GiB
// alias region, so blocks kept for good ran out after 16 GiB of ranges ever mapped (39 minutes of play, six areas), and
// every pool created after that stayed out of the table: its pointers read zeros, as before the table existed. A range
// mapped again while its block still waits takes that block back (block_take), so a pool freed and made again at its
// address within a frame holds no second block.
//
// NX2NAVI_GPA=0 turns the table off: the translator is told the host has no device addresses (those accesses read 0).
// NX2NAVI_GPA_EMPTY=1 keeps it empty, so every access through it reads the zero buffer (diagnosis).
// Mirrored pools are not synced for the table's readers (vk_exec_draw.c ssbo_info says why). With shadow storage the
// pools are copies of guest memory, which a guest submission that binds the table (vk_exec_draw.c ssbo_info) brings up
// to date with the pages the CPU wrote, once, before the part is submitted (vk_memory.c vk_mem_shadow_sync). There
// the twin of a mirrored pool is its shadow copy, not guest memory, and what a shader stores through the table is not
// noted as a GPU write: it does not come back to guest memory, and the next whole copy-in of its page overwrites it.
// vk_memory.c "Pages a shader stores into through the device address table" has the sequence, what was measured, and
// the switch that keeps the stores (NX2NAVI_SHADOW_TABLE_STORES).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "vk_gpa.h"
#include "platform/platform.h"
#include "resource_core_registry.h"
#include "resource_core_vma.h"

#define L1_WORD 16u
#define L1_N 65536u
#define L2_WORD (L1_WORD + L1_N)
#define L2_N 4096u                     // pages per block: 16 MiB
#define BLOCKS 1024u                   // L2 blocks: 16 GiB of mapped GPU address space; block 0 stays zero
#define REC_WORD (L2_WORD + BLOCKS * L2_N)
#define RECS 65536u                    // record 0 means none
#define REC_WORDS 4u                   // words per record
#define WORDS (REC_WORD + RECS * REC_WORDS)
#define PAGE_SHIFT 12

enum { GPA_TABLE, GPA_ZERO, GPA_SINK, GPA_BUFFERS };
typedef struct {
    VkBuffer buffer[GPA_BUFFERS];
    VmaAllocation allocation[GPA_BUFFERS];
    volatile uint32_t *gpu;
    uint32_t *cpu, *free_recs;
    RcBudgetTicket host_budget;
} GpaStorage;

static struct {
    int ready;                         // published only after complete device startup preparation
    GpaStorage storage;
    uint32_t nblocks;                  // L2 blocks ever handed out (block 0 is the zero block)
    uint16_t blk_pages[BLOCKS];        // pages each block maps
    uint16_t blk_alias_pages[BLOCKS];  // virtual pages in the current projection
    uint16_t blk_l1[BLOCKS];           // the L1 entry that names it, or named it while it waits
    uint8_t blk_wait[BLOCKS];          // off L1, waiting for the GPU (blocks_release)
    uint32_t blk_gen[BLOCKS];          // releases so far: a block taken back and released again has two callbacks
    uint16_t free_blks[BLOCKS];        // empty blocks whose fence completed
    uint32_t nfree_blk, blk_waiting;
    uint64_t blk_returned;             // blocks taken off L1 so far
    uint32_t nrec;                     // records handed out, free ones aside
    uint32_t nfree;        // records whose fence completed
    uint64_t mapped_pages, maps, unmaps, refused;
    struct {
        uint64_t lo, hi;
        RcInterval *host;
        uint32_t host_count, next, prev;
        int cpu_visible;
    } authority[RECS];
    struct {
        RcSourceRange backing;
        uint64_t lo;
        uint32_t pages, retire_next;
        int alias;
    } aliases[RECS];
    uint32_t authority_head;
} G;
static PlatMutex gpa_lock = PLAT_MUTEX_INIT;

typedef struct {
    uint64_t lo, hi;
    RcRegistryWrite ticket;
    RcSourceRange *uncertain;
    uint32_t count, capacity;
} GpaWriteRange;
typedef struct {
    RcSourceRange *items;
    uint32_t count, capacity;
} GpaRanges;
typedef struct {
    GpaRanges written, uncertain;
    int unknown;
} GpaCoverage;
typedef struct GpaSceneManifest {
    struct GpaSceneManifest *next;
    uint64_t scene;
    uint32_t refs;
    GpaWriteRange uncertainty;
    GpaCoverage canonical;
} GpaSceneManifest;
static GpaSceneManifest *scene_manifests;
// One single-invocation effect plan has at most 32 stores, or 16 primary/twin pairs.
enum { GPA_DEFINITE_RANGES = 16, GPA_DEFINITE_SPANS = 32 };
typedef struct {
    VkMemSpanCapture *holds[GPA_DEFINITE_RANGES];
    RcRegistryBufferSpan spans[GPA_DEFINITE_SPANS];
    uint32_t held, count;
    int invalid;
} GpaDefiniteWrite;

struct VkGpaWriteScope {
    struct VkGpaWriteScope *next;
    GpaWriteRange *ranges;
    uint32_t count;
    int whole_domain;
    RcRegistryWrite guard;
    GpaCoverage canonical;
    GpaSceneManifest *manifest;
    RcRegistryPosition position;
    VkGpaWriteOrigin origin;
    GpaDefiniteWrite *definite;
    int origin_recorded, settled;
};
static VkGpaWriteScope *write_scopes;
static uint32_t definite_scopes;
static struct {uint64_t scopes,mappings,uncertainty,begin_ns,end_ns,map_joins,manifests,manifest_reuses;uint32_t live,peak;} writer_cost;

// The pending reservation is symbolic: an unknown shader needs one domain ticket, not one ticket per mapped pool.
// Only uncertain visibility intervals are materialized, and only at real completion do they replace the reservation.
static void write_range_uncertain(GpaWriteRange *r, uint64_t lo, uint64_t hi) {
    if (hi <= lo) return;
    for (uint32_t i=0; i<r->count;) {
        RcSourceRange old=r->uncertain[i]; uint64_t end=old.offset+old.size;
        if (end < lo || old.offset > hi) { i++; continue; }
        if (old.offset < lo) lo=old.offset;
        if (end > hi) hi=end;
        r->uncertain[i]=r->uncertain[--r->count];
        i=0;
    }
    if (r->count==r->capacity) {
        uint32_t n=r->capacity?r->capacity*2:8;
        void *p=realloc(r->uncertain,(size_t)n*sizeof *r->uncertain);
        if (!p) rt_fatal("vk_gpa: uncertainty allocation failed");
        r->uncertain=p; r->capacity=n;
    }
    r->uncertain[r->count++]=(RcSourceRange){1,lo,hi-lo};writer_cost.uncertainty++;
}

static void coverage_range(GpaRanges *ranges, RcSourceRange range) {
    uint64_t lo = range.offset, end = lo + range.size;
    for (uint32_t i = 0; i < ranges->count;) {
        RcSourceRange prior = ranges->items[i];
        uint64_t hi = prior.offset + prior.size;
        if (prior.backing != range.backing || hi < lo || end < prior.offset) { ++i; continue; }
        if (prior.offset < lo) lo = prior.offset;
        if (hi > end) end = hi;
        ranges->items[i] = ranges->items[--ranges->count];
        i = 0;
    }
    if (ranges->count == ranges->capacity) {
        uint32_t capacity = ranges->capacity ? 2 * ranges->capacity : 8;
        RcSourceRange *items = realloc(ranges->items, (size_t)capacity * sizeof *items);
        if (!items) rt_fatal("vk_gpa: canonical writer coverage allocation failed");
        ranges->items = items;
        ranges->capacity = capacity;
    }
    ranges->items[ranges->count++] = (RcSourceRange){range.backing, lo, end - lo};
}

static void coverage_record(GpaCoverage *coverage, uint64_t lo, uint64_t hi, int visible,
                            const RcInterval *source, uint32_t count) {
    if (hi <= lo) return;
    uint64_t covered = lo;
    for (uint32_t i = 0; i < count && covered < hi; i++) {
        RcInterval span = source[i];
        uint64_t end = span.address + span.range.size;
        if (end <= covered) continue;
        if (span.address > covered || !rc_range_valid(span.range)) break;
        if (end > hi) end = hi;
        RcSourceRange range = {span.range.backing, span.range.offset + (covered - span.address), end - covered};
        coverage_range(&coverage->written, range);
        if (!visible) coverage_range(&coverage->uncertain, range);
        covered = end;
    }
    if (covered != hi) coverage->unknown = 1;
}

static void coverage_free(GpaCoverage *coverage) {
    free(coverage->written.items);
    free(coverage->uncertain.items);
}

static void write_scope_canonical(VkGpaWriteScope *scope, uint64_t lo, uint64_t hi, int visible,
                                   const RcInterval *source, uint32_t count) {
    if (scope->whole_domain) {
        GpaCoverage *coverage = scope->manifest ? &scope->manifest->canonical : &scope->canonical;
        coverage_record(coverage, lo, hi, visible, source, count);
        return;
    }
    for (uint32_t i = 0; i < scope->count; i++) {
        GpaWriteRange *range = &scope->ranges[i];
        coverage_record(&scope->canonical, range->lo > lo ? range->lo : lo, range->hi < hi ? range->hi : hi,
                        visible, source, count);
    }
}

// gpa_lock held. A virtual page can hide GPU writes from the CPU at the virtual address. The backing owner
// protects canonical CPU storage separately, including when it is no longer the current virtual target.
static void write_scope_mapping(VkGpaWriteScope *scope, uint64_t lo, uint64_t hi, int cpu_visible) {
    if (cpu_visible) return;
    if (scope->manifest) {
        write_range_uncertain(&scope->manifest->uncertainty,lo,hi);
        return;
    }
    for (uint32_t i=0; i<scope->count; i++) {
        GpaWriteRange *r=&scope->ranges[i];
        uint64_t begin=r->lo>lo?r->lo:lo, end=r->hi<hi?r->hi:hi;
        write_range_uncertain(r,begin,end);
    }
}

// An unknown effect's uncertainty is shared by its scene, but its producer order and ticket are not.
// Add the new virtual/imported coverage before making the GPU page entries visible. Existing scopes keep every
// target they could already have reached, regardless of what later takes its place.
static void authority_publish(uint64_t lo, uint64_t hi, int cpu_visible, const RcInterval *source, uint32_t count) {
    for (GpaSceneManifest *m=scene_manifests;m;m=m->next) {
        writer_cost.map_joins++;
        if (!cpu_visible) write_range_uncertain(&m->uncertainty,lo,hi);
        coverage_record(&m->canonical, lo, hi, cpu_visible, source, count);
    }
    for (VkGpaWriteScope *s=write_scopes;s;s=s->next) if (!s->manifest && s->count) {
        writer_cost.map_joins++;
        write_scope_mapping(s,lo,hi,cpu_visible);
        write_scope_canonical(s, lo, hi, cpu_visible, source, count);
    }
}

static void authority_map_owned(uint32_t rec, uint64_t lo, uint64_t hi, int cpu_visible,
                                 RcInterval *host, uint32_t count) {
    G.authority[rec]=(typeof(G.authority[rec])){.lo=lo,.hi=hi,.host=host,.host_count=count,
                                              .next=G.authority_head,.cpu_visible=cpu_visible};
    if (G.authority_head) G.authority[G.authority_head].prev=rec;
    G.authority_head=rec;
    authority_publish(lo, hi, cpu_visible, host, count);
}

static void authority_map(uint32_t rec, uint64_t lo, uint64_t hi, int cpu_visible,
                          const RcInterval *source, uint32_t count) {
    RcInterval *host = count ? calloc(count, sizeof *host) : NULL;
    if (host) memcpy(host, source, count * sizeof *host);
    authority_map_owned(rec, lo, hi, cpu_visible, host, host ? count : 0);
}

static void write_scope_alias_pages(VkGpaWriteScope *scope);

static VkGpaWriteScope *write_scope_begin(const VkGpaEffectRange *effects,uint32_t count,int whole_domain,
                                         RcRegistryPosition position) {
    uint64_t start=plat_now_ns();
    VkGpaWriteScope *scope=calloc(1,sizeof *scope);
    if (!scope) rt_fatal("vk_gpa: write scope allocation failed");
    scope->position=position;
    scope->whole_domain=whole_domain;
    scope->count=whole_domain?1:count;
    if (scope->count) {
        scope->ranges=calloc(scope->count,sizeof *scope->ranges);
        if (!scope->ranges) rt_fatal("vk_gpa: writer allocation failed");
    }
    plat_mutex_lock(&gpa_lock);
    for (uint32_t i=0;i<scope->count;i++) {
        GpaWriteRange *r=&scope->ranges[i];
        r->lo=whole_domain?0:effects[i].addr;
        r->hi=whole_domain?UINT64_MAX:r->lo+effects[i].size;
        RcSourceRange range={1,r->lo,r->hi-r->lo};
        if (position.scene) rc_registry_write_begin_at(rc_registry_runtime(),range,RC_REGISTRY_BUFFER,position,&r->ticket);
        else rc_registry_write_begin(rc_registry_runtime(),range,RC_REGISTRY_BUFFER,&r->ticket);
    }
    if (scope->count && rc_registry_write_guard_begin_reason_at(rc_registry_runtime(), RC_REGISTRY_BUFFER, position,
                                         RC_REGISTRY_GUARD_GPA_PROJECTION, &scope->guard) != RC_REGISTRY_OK)
        rc_registry_fail_closed(rc_registry_runtime());
    int snapshot=scope->count!=0;
    if(whole_domain && position.scene) {
        GpaSceneManifest *m=scene_manifests;
        while(m && m->scene!=position.scene)m=m->next;
        if(m) {m->refs++;snapshot=0;writer_cost.manifest_reuses++;}
        else {
            m=calloc(1,sizeof *m);if(!m)rt_fatal("vk_gpa: scene manifest allocation failed");
            m->scene=position.scene;m->refs=1;m->next=scene_manifests;scene_manifests=m;writer_cost.manifests++;
        }
        scope->manifest=m;
    }
    // The first unknown scope snapshots current mappings; later scopes borrow the same conservative scene union.
    // A union may retain a mapping removed earlier in this scene, never omit one that this command could reach.
    // The last scope destroys it, so uncertainty cannot grow indefinitely across a long session.
    for (uint32_t i=snapshot?G.authority_head:0;i;i=G.authority[i].next) {
        writer_cost.mappings++;
        write_scope_mapping(scope,G.authority[i].lo,G.authority[i].hi,G.authority[i].cpu_visible);
        write_scope_canonical(scope, G.authority[i].lo, G.authority[i].hi, G.authority[i].cpu_visible,
                              G.authority[i].host, G.authority[i].host_count);
    }
    if (snapshot) write_scope_alias_pages(scope);
    scope->next=write_scopes;write_scopes=scope;writer_cost.scopes++;
    if(++writer_cost.live>writer_cost.peak)writer_cost.peak=writer_cost.live;
    writer_cost.begin_ns+=plat_now_ns()-start;
    plat_mutex_unlock(&gpa_lock);
    return scope;
}
VkGpaWriteScope *vk_gpa_write_begin_at(const VkGpaEffectRange *ranges,uint32_t count,int complete,
                                      RcRegistryPosition position) {
    if (complete && count && !ranges) complete=0;
    for (uint32_t i=0;complete && i<count;i++)
        if (!ranges[i].size || ranges[i].size>UINT64_MAX-ranges[i].addr) complete=0;
    return write_scope_begin(complete?ranges:NULL,complete?count:0,!complete,position);
}
VkGpaWriteScope *vk_gpa_write_begin(void) {return vk_gpa_write_begin_at(NULL,0,0,(RcRegistryPosition){0});}
VkGpaWriteScope *vk_gpa_write_begin_ranges(const VkGpaEffectRange *ranges,uint32_t count) {
    return vk_gpa_write_begin_at(ranges,count,1,(RcRegistryPosition){0});
}

void vk_gpa_write_note_recorded(VkGpaWriteScope *scope, const VkGpaWriteOrigin *origin) {
    if (!scope || !origin) return;
    // The recorder still owns the scope. Completion cannot remove it before this one publication.
    scope->origin = *origin;
    __atomic_store_n(&scope->origin_recorded, 1, __ATOMIC_RELEASE);
}

static RcSourceRange matching_projection(const GpaRanges *ranges, RcSourceRange target) {
    for (uint32_t i = 0; i < ranges->count; ++i)
        if (rc_range_overlap(ranges->items[i], target)) return ranges->items[i];
    return (RcSourceRange){0};
}

// gpa_lock held by callers. A new projection can change eligibility until the actual native part completes.
static int prefix_eligible(const VkGpaWriteScope *scope, RcRegistryPosition read) {
    return !scope->whole_domain && scope->count && scope->position.scene == read.scene &&
           rc_registry_position_compare(scope->position, read) < 0 &&
           __atomic_load_n(&scope->origin_recorded, __ATOMIC_ACQUIRE) &&
           !scope->canonical.unknown && !scope->canonical.uncertain.count;
}

static int prefix_requested(const VkGpaWriteScope *scope, const VkGpaPrefixRequest *request) {
    return scope->guard.id == request->guard.id && scope->guard.index == request->guard.index &&
           !rc_registry_position_compare(scope->position, request->producer) && prefix_eligible(scope, request->read);
}

int vk_gpa_write_prefix_ready(const VkGpaPrefixRequest *request) {
    if (!request || !request->guard.id || !request->source.mapping || !rc_range_valid(request->source.range) ||
        !rc_registry_position_valid(request->read)) return 0;
    plat_mutex_lock(&gpa_lock);
    const VkGpaWriteScope *scope = write_scopes;
    while (scope && !prefix_requested(scope, request)) scope = scope->next;
    int ready = scope != NULL;
    plat_mutex_unlock(&gpa_lock);
    return ready;
}

int vk_gpa_write_scene_may_need_prefix(uint64_t scene) {
    plat_mutex_lock(&gpa_lock);
    const VkGpaWriteScope *scope = write_scopes;
    RcRegistryPosition end = {scene, UINT64_MAX};
    while (scope && !prefix_eligible(scope, end)) scope = scope->next;
    int ready = scope != NULL;
    plat_mutex_unlock(&gpa_lock);
    return ready;
}

void vk_gpa_report_write_blocker(RcRegistryWrite guard, RcSourceRange target) {
    if (!guard.id) return;
    VkGpaWriteOrigin origin = {0};
    RcSourceRange mapped = {0}, uncertain = {0};
    uint32_t count = 0, mapped_count = 0, uncertain_count = 0;
    int recorded = 0, whole = 0, coverage_unknown = 0;
    uint64_t first_lo = 0, first_hi = 0;
    plat_mutex_lock(&gpa_lock);
    const VkGpaWriteScope *scope = write_scopes;
    while (scope && (scope->guard.id != guard.id || scope->guard.index != guard.index)) scope = scope->next;
    if (scope) {
        const GpaCoverage *coverage = scope->manifest ? &scope->manifest->canonical : &scope->canonical;
        mapped = matching_projection(&coverage->written, target);
        uncertain = matching_projection(&coverage->uncertain, target);
        mapped_count = coverage->written.count;
        uncertain_count = coverage->uncertain.count;
        coverage_unknown = coverage->unknown;
        recorded = __atomic_load_n(&scope->origin_recorded, __ATOMIC_ACQUIRE);
        if (recorded) origin = scope->origin;
        whole = scope->whole_domain;
        count = scope->count;
        if (count) first_lo = scope->ranges[0].lo, first_hi = scope->ranges[0].hi;
    }
    int found = scope != NULL;
    plat_mutex_unlock(&gpa_lock);
    if (!found) {
        rt_log("vk_gpa: source blocker %llu/%u no longer has a live scope", (unsigned long long)guard.id, guard.index);
        return;
    }
    char shader[65];
    for (size_t i = 0; i < sizeof origin.shader_key; ++i)
        snprintf(shader + i * 2, 3, "%02x", origin.shader_key[i]);
    rt_log("vk_gpa: source blocker %llu/%u recorded %d program %llu op %u stage %u set %u binding %u shader %s; "
           "guest %s %u ranges first %#llx..%#llx; canonical projected %u uncertain %u unresolved %d; "
           "target overlap projected %llu:%#llx+%#llx uncertain %llu:%#llx+%#llx",
           (unsigned long long)guard.id, guard.index, recorded, (unsigned long long)origin.program, origin.opcode,
           origin.stage, origin.set, origin.binding, shader, whole ? "whole-domain" : "bounded", count,
           (unsigned long long)first_lo, (unsigned long long)first_hi, mapped_count, uncertain_count, coverage_unknown,
           (unsigned long long)mapped.backing, (unsigned long long)mapped.offset, (unsigned long long)mapped.size,
           (unsigned long long)uncertain.backing, (unsigned long long)uncertain.offset, (unsigned long long)uncertain.size);
}
static void finish_canonical_scope(VkGpaWriteScope *scope, uint64_t fence, int abandoned) {
    if (!scope->guard.id) return;
    RcRegistry *registry = rc_registry_runtime();
    if (!fence && !abandoned) {
        rc_registry_write_cancel(registry, scope->guard);
        return;
    }
    if (!fence || rc_registry_write_submit(registry, scope->guard, fence) != RC_REGISTRY_OK) {
        rc_registry_write_guard_accumulate(registry, scope->guard);
        return;
    }
    const GpaCoverage *coverage = scope->manifest ? &scope->manifest->canonical : &scope->canonical;
    RcRegistryWriteCompletion completion = {scope->guard, coverage->written.items, coverage->uncertain.items,
                                             coverage->written.count, coverage->uncertain.count};
    const GpaDefiniteWrite *definite = scope->definite;
    if (!coverage->unknown && definite && !definite->invalid &&
        rc_registry_write_complete_gpa_projection(registry, &completion, definite->spans, definite->count) == RC_REGISTRY_OK)
        return;
    if (coverage->unknown || rc_registry_write_complete_extents(registry, &completion, 1) != RC_REGISTRY_OK)
        rc_registry_write_guard_accumulate(registry, scope->guard);
}

static void write_scope_settle_locked(VkGpaWriteScope *scope,uint64_t fence,int abandoned_execution) {
    uint64_t start=plat_now_ns();
    VkGpaWriteScope **at=&write_scopes;
    while (*at && *at!=scope) at=&(*at)->next;
    if (*at!=scope) rt_fatal("vk_gpa: write scope ended twice");
    *at=scope->next;
    for (uint32_t i=0;i<scope->count;i++) {
        GpaWriteRange *r=&scope->ranges[i];
        if (r->ticket.id) {
            if (fence) {
                if (rc_registry_write_submit(rc_registry_runtime(),r->ticket,fence)==RC_REGISTRY_OK) {
                    const GpaWriteRange *visibility=scope->manifest?&scope->manifest->uncertainty:r;
                    if(rc_registry_write_complete_ranges(rc_registry_runtime(),r->ticket,visibility->uncertain,visibility->count)!=RC_REGISTRY_OK)
                        rc_registry_write_complete(rc_registry_runtime(),r->ticket,0);
                } else rc_registry_write_abandon(rc_registry_runtime(),r->ticket,1);
                // Refusal retains full foreign authority with real completion provenance, not an ownerless pending ticket.
            } else rc_registry_write_abandon(rc_registry_runtime(),r->ticket,abandoned_execution);
        }
        free(r->uncertain);
    }
    finish_canonical_scope(scope, fence, abandoned_execution);
    if (scope->definite) --definite_scopes;
    coverage_free(&scope->canonical);
    if(scope->manifest && !--scope->manifest->refs) {
        GpaSceneManifest *m=scope->manifest,**link=&scene_manifests;
        while(*link && *link!=m)link=&(*link)->next;
        if(*link!=m)rt_fatal("vk_gpa: scene manifest ownership lost");
        *link=m->next;
        free(m->uncertainty.uncertain);
        coverage_free(&m->canonical);
        free(m);
    }
    free(scope->ranges);
    scope->ranges = NULL;
    scope->settled = 1;
    writer_cost.live--;writer_cost.end_ns+=plat_now_ns()-start;
}

static void definite_release(GpaDefiniteWrite *write) {
    if (!write) return;
    for (uint32_t i = 0; i < write->held; i++) vk_mem_span_capture_release(&write->holds[i]);
    free(write);
}

static void write_scope_end(VkGpaWriteScope *scope,uint64_t fence,int abandoned_execution) {
    if (!scope) return;
    plat_mutex_lock(&gpa_lock);
    if (!scope->settled) write_scope_settle_locked(scope, fence, abandoned_execution);
    plat_mutex_unlock(&gpa_lock);
    definite_release(scope->definite);
    free(scope);
}

int vk_gpa_write_prefix_complete(const VkGpaPrefixRequest *request) {
    if (!request || !request->guard.id || request->read.ordinal < 2 ||
        !rc_registry_position_handed(rc_registry_runtime(), (RcRegistryPosition){request->read.scene, request->read.ordinal - 1})) return 0;
    plat_mutex_lock(&gpa_lock);
    int resolved = 0;
    for (VkGpaWriteScope *scope = write_scopes, *next; scope; scope = next) {
        next = scope->next;
        if (!prefix_eligible(scope, request->read)) continue;
        if (prefix_requested(scope, request)) resolved = 1;
        // The native fence callback still owns the shell. It will only free it after this early settlement.
        write_scope_settle_locked(scope, request->read.scene, 0);
    }
    plat_mutex_unlock(&gpa_lock);
    return resolved;
}
void vk_gpa_write_cancel(VkGpaWriteScope *scope) {write_scope_end(scope,0,0);}
void vk_gpa_write_abandon(VkGpaWriteScope *scope,int may_have_executed) {write_scope_end(scope,0,may_have_executed);}
void vk_gpa_write_complete(VkGpaWriteScope *scope,uint64_t fence) {
    if (!fence) rt_fatal("vk_gpa: completion requires a submission fence");
    write_scope_end(scope,fence,0);
}

static void put(uint32_t word, uint32_t v) {
    G.storage.cpu[word] = v;
    __atomic_store_n(&G.storage.gpu[word], v, __ATOMIC_RELAXED);
}

int vk_gpa_wanted_ctx(const VkCtx *c) {
    static int knob = -1;
    if (knob < 0) {
        const char *e = cfg_get("NX2NAVI_GPA");
        knob = !(e && !strcmp(e, "0"));
    }
    return knob && c && c->has_buffer_device_address;
}

int vk_gpa_wanted(void) { return vk_gpa_wanted_ctx(vk_ctx()); }

uint64_t vk_gpa_device_address(VkBuffer b) {
    VkBufferDeviceAddressInfo ai = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b};
    return b ? vkGetBufferDeviceAddress(vk_ctx()->device, &ai) : 0;
}

static void storage_destroy(VkCtx *c, GpaStorage *storage) {
    for (int i = GPA_BUFFERS - 1; i >= 0; i--)
        if (storage->allocation[i])
            resource_core_vma_destroy_buffer(c->vma, storage->buffer[i], storage->allocation[i]);
    if (storage->cpu) {
        if (resource_core_budget_retire(&storage->host_budget) != RC_ADMISSION_OK)
            rt_fatal("vk_gpa: CPU table lost its capacity owner");
        free(storage->cpu);
        free(storage->free_recs);
        if (resource_core_budget_destroyed(&storage->host_budget) != RC_ADMISSION_OK)
            rt_fatal("vk_gpa: CPU table capacity destruction failed");
    } else if (storage->host_budget.token.ledger && resource_core_budget_cancel(&storage->host_budget) != RC_ADMISSION_OK) {
        rt_fatal("vk_gpa: CPU table reservation cancellation failed");
    }
    *storage = (GpaStorage){0};
}

static void touch_host(void *memory, size_t bytes) {
    // Commit the working set while its RAM allowance is still reserved.
    volatile uint8_t *pages = memory;
    for (size_t offset = 0; offset < bytes; offset += 4096) pages[offset] = 0;
    pages[bytes - 1] = 0;
}

static VkResult create(VkCtx *c, GpaStorage *storage) {
    const size_t table_bytes = WORDS * sizeof(uint32_t), record_bytes = RECS * sizeof(uint32_t);
    const VkBufferUsageFlags table = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkBufferUsageFlags addressable = table | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    const VkBufferCreateInfo buffers[GPA_BUFFERS] = {
        [GPA_TABLE] = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = table_bytes, .usage = table},
        [GPA_ZERO] = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096, .usage = addressable},
        [GPA_SINK] = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096, .usage = addressable},
    };
    // The table and the two pages live until shutdown. Use dedicated placements so the joint peak tracks
    // their native requirements without reserving three possible default-pool blocks.
    const VmaAllocationCreateInfo device = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT};
    VmaAllocationCreateInfo mapped = device;
    mapped.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    RcVmaRequest request = {.producer = RC_BUDGET_BUFFER, .required = 1};
    RcAdmissionVector host = {{0}};
    host.amount[RC_BUDGET_RAM] = host.amount[RC_BUDGET_COMMIT] = table_bytes + record_bytes;
    RcAdmissionVector peak = host;
    VmaAllocationCreateInfo placements[GPA_BUFFERS];
    for (int i = 0; i < GPA_BUFFERS; i++) {
        RcAdmissionVector part;
        VkResult result = resource_core_vma_buffer_peak(&request, &buffers[i], i == GPA_SINK ? &device : &mapped,
                                                       &placements[i], &part);
        if (result != VK_SUCCESS) return result;
        for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
            if (part.amount[d] > UINT64_MAX - peak.amount[d]) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            peak.amount[d] += part.amount[d];
        }
    }
    RcBudgetTicket joint = {0};
    if (resource_core_budget_reserve(request.producer, request.required, &peak, &joint) != RC_ADMISSION_OK)
        return VK_NOT_READY;
    request.peak_ticket = &joint;
    request.host_bytes = host.amount[RC_BUDGET_RAM];
    request.host_ticket = &storage->host_budget;
    VmaAllocationInfo info[GPA_BUFFERS];
    VkResult result = VK_SUCCESS;
    for (int i = 0; i < GPA_BUFFERS; i++) {
        result = resource_core_vma_create_buffer(&request, c->vma, &buffers[i], &placements[i],
                                               &storage->buffer[i], &storage->allocation[i], &info[i], NULL);
        request.host_bytes = 0;
        request.host_ticket = NULL;
        if (result != VK_SUCCESS) break;
        if (i != GPA_SINK && !info[i].pMappedData) {
            result = VK_ERROR_MEMORY_MAP_FAILED;
            break;
        }
    }
    if (resource_core_budget_cancel(&joint) != RC_ADMISSION_OK)
        rt_fatal("vk_gpa: buffer group lost its joint reservation");
    if (result != VK_SUCCESS) return result;

    uint32_t *cpu = calloc(WORDS, sizeof *cpu), *free_recs = malloc(record_bytes);
    if (!cpu || !free_recs) {
        free(cpu);
        free(free_recs);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    touch_host(cpu, table_bytes);
    touch_host(free_recs, record_bytes);
    if (!rc_admission_accounted(resource_core_budget_materialize(&storage->host_budget, &host)))
        rt_fatal("vk_gpa: CPU table capacity settlement failed");
    storage->cpu = cpu;
    storage->free_recs = free_recs;
    storage->gpu = info[GPA_TABLE].pMappedData;
    memset(info[GPA_TABLE].pMappedData, 0, table_bytes);
    memset(info[GPA_ZERO].pMappedData, 0, 4096);
    return VK_SUCCESS;
}

VkGpaPrepareResult vk_gpa_prepare(VkResult *failure) {
    *failure = VK_SUCCESS;
    VkCtx *c = vk_ctx();
    if (!c) {
        *failure = VK_ERROR_INITIALIZATION_FAILED;
        return VK_GPA_ERROR;
    }
    if (!vk_gpa_wanted_ctx(c)) return VK_GPA_DISABLED;
    if (vk_gpa_on()) return VK_GPA_READY;

    GpaStorage prepared = {0};
    VkResult result = create(c, &prepared);
    uint64_t zero = 0, sink = 0;
    if (result == VK_SUCCESS) {
        zero = vk_gpa_device_address(prepared.buffer[GPA_ZERO]);
        sink = vk_gpa_device_address(prepared.buffer[GPA_SINK]);
        if (!zero || !sink) result = VK_ERROR_INITIALIZATION_FAILED;
    }
    if (result != VK_SUCCESS) {
        storage_destroy(c, &prepared);
        *failure = result;
        if (result == VK_NOT_READY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY)
            return VK_GPA_NO_CAPACITY;
        return VK_GPA_ERROR;
    }

    plat_mutex_lock(&gpa_lock);
    G.storage = prepared;
    put(0, (uint32_t)zero);
    put(1, (uint32_t)(zero >> 32));
    put(2, REC_WORD);
    put(3, (uint32_t)sink);
    put(4, (uint32_t)(sink >> 32));
    G.nblocks = 1;
    G.nrec = 1;
    __atomic_store_n(&G.ready, 1, __ATOMIC_RELEASE);
    plat_mutex_unlock(&gpa_lock);
    rt_log("vk_gpa: device address table on (%.1f MiB, %u blocks of 16 MiB, %u records)",
           WORDS * sizeof(uint32_t) / 1048576.0, BLOCKS, RECS);
    return VK_GPA_READY;
}

int vk_gpa_on(void) { return __atomic_load_n(&G.ready, __ATOMIC_ACQUIRE); }

VkDescriptorBufferInfo vk_gpa_info(void) {
    if (vk_gpa_on()) return (VkDescriptorBufferInfo){G.storage.buffer[GPA_TABLE], 0, VK_WHOLE_SIZE};
    if (vk_gpa_wanted()) rt_fatal("vk_gpa: required device address table was not prepared before recording");
    return (VkDescriptorBufferInfo){0};
}

// A block for L1 entry l1, 0 when none is left: the one the entry gave up and that still waits for the GPU (it maps
// nothing, and a shader that read the old entry finds the new pages in it), else a free one, else one never used.
static uint32_t block_take(uint32_t l1) {
    uint32_t blk = 0;
    for (uint32_t b = 1; G.blk_waiting && b < G.nblocks && !blk; b++)
        if (G.blk_wait[b] && G.blk_l1[b] == l1) blk = b;
    if (blk) G.blk_wait[blk] = 0, G.blk_waiting--;   // its callback finds it in use (blk_free_cb)
    else if (G.nfree_blk) blk = G.free_blks[--G.nfree_blk];
    else if (G.nblocks < BLOCKS) blk = G.nblocks++;
    else return 0;
    G.blk_l1[blk] = (uint16_t)l1;
    return blk;
}

// The L2 entry word of a page; 0 when its block does not exist and alloc is 0 or no block is left.
static uint32_t page_word(uint64_t page, int alloc) {
    uint64_t l1 = page >> 12;   // 4096 pages of 4 KiB per 16 MiB
    if (l1 >= L1_N) return 0;
    uint32_t blk = G.storage.cpu[L1_WORD + l1];
    if (!blk) {
        if (!alloc || !(blk = block_take((uint32_t)l1))) return 0;
        put(L1_WORD + (uint32_t)l1, blk);
    }
    return L2_WORD + blk * L2_N + (uint32_t)(page & (L2_N - 1));
}

// Exact selected host mapping and actual primary/twin addresses for every page the emitted store can touch.
static int definite_mapping_current(RcInterval span, uint64_t primary_delta, uint64_t twin_delta) {
    const uint64_t end = span.address + span.range.size;
    for (uint64_t at = span.address; at < end;) {
        const uint64_t page_end = ((at >> PAGE_SHIFT) + 1) << PAGE_SHIFT;
        uint64_t until = page_end < end ? page_end : end;
        uint32_t word = page_word(at >> PAGE_SHIFT, 0);
        uint32_t r = word ? G.storage.cpu[word] : 0;
        if (!r || !G.authority[r].cpu_visible || G.aliases[r].alias ||
            G.authority[r].lo > at || G.authority[r].hi < until) return 0;
        const uint32_t *record = &G.storage.cpu[REC_WORD + r * REC_WORDS];
        uint64_t primary = ((uint64_t)record[1] << 32) | record[0];
        uint64_t twin = ((uint64_t)record[3] << 32) | record[2];
        if (primary != primary_delta || twin != twin_delta || twin == primary) return 0;
        const RcInterval *host = G.authority[r].host;
        for (uint32_t i = 0; i < G.authority[r].host_count && at < until; i++) {
            const RcInterval map = host[i];
            const uint64_t map_end = map.address + map.range.size;
            if (map_end <= at) continue;
            if (map.address > at || map.mapping != span.mapping || map.range.backing != span.range.backing ||
                map.range.offset + (at - map.address) != span.range.offset + (at - span.address)) return 0;
            at = map_end < until ? map_end : until;
        }
        if (at != until) return 0;
    }
    return 1;
}

void vk_gpa_write_record_definite(VkGpaWriteScope *scope, const VkGpaEffectRange *ranges, uint32_t count) {
    if (!scope || !ranges || !count || count > GPA_DEFINITE_RANGES || scope->whole_domain || !scope->guard.id)
        return;
    GpaDefiniteWrite *write = calloc(1, sizeof *write);
    if (!write) return;
    uint64_t primary[GPA_DEFINITE_SPANS], twin[GPA_DEFINITE_SPANS];
    uint64_t remaining = GPA_DEFINITE_RANGES * sizeof(uint32_t);
    for (uint32_t i = 0; i < count; i++) {
        const VkGpaEffectRange range = ranges[i];
        if (!range.size || range.size > remaining || ((range.addr | range.size) & 3) ||
            i >= scope->count || range.addr != scope->ranges[i].lo ||
            range.size != scope->ranges[i].hi - scope->ranges[i].lo) goto refuse;
        remaining -= range.size;
        VkMemSpan span;
        VkMemSpanCapture **hold = &write->holds[write->held];
        if (vk_mem_span_native_input(range.addr, range.size, 0, &span, hold) != RC_REGISTRY_OK) goto refuse;
        write->held++;
        RcRegistryBufferLocation location;
        VkBuffer twin_buffer;
        if (!vk_mem_span_gpa_donor(*hold, &span, &location, &twin_buffer)) goto refuse;
        uint64_t primary_address = vk_gpa_device_address(span.buffer), twin_address = vk_gpa_device_address(twin_buffer);
        if (!primary_address || !twin_address || primary_address == twin_address ||
            span.offset > UINT64_MAX - primary_address || span.offset > UINT64_MAX - twin_address)
            goto refuse;
        RcInterval *mapped = NULL;
        uint32_t mapped_count = vk_mem_span_input_intervals(*hold, range.addr, range.size, &mapped);
        uint64_t at = range.addr, end = range.addr + range.size;
        for (uint32_t j = 0; j < mapped_count && write->count < GPA_DEFINITE_SPANS; j++) {
            RcInterval part = mapped[j];
            if (!part.mapping || !rc_range_valid(part.range) || part.address != at || part.range.size > end - at)
                break;
            uint32_t k = write->count++;
            write->spans[k] = (RcRegistryBufferSpan){part, {location.allocation, location.incarnation,
                                                          location.offset + (at - range.addr)}};
            primary[k] = primary_address + span.offset - range.addr;
            twin[k] = twin_address + span.offset - range.addr;
            at += part.range.size;
        }
        free(mapped);
        if (at != end) goto refuse;
    }
    plat_mutex_lock(&gpa_lock);
    int current = G.ready && scope->count == count && !scope->definite;
    for (uint32_t i = 0; i < write->count && current; i++)
        current = definite_mapping_current(write->spans[i].interval, primary[i], twin[i]);
    if (current) {
        scope->definite = write;
        definite_scopes++;
    }
    plat_mutex_unlock(&gpa_lock);
    if (current) return;
refuse:
    definite_release(write);
}

// Capture only virtual pages currently projected by this block. Older scopes already hold their prior coverage.
static void write_scope_alias_block(GpaWriteRange *range, GpaCoverage *coverage, uint32_t blk,
                                    uint64_t base, uint64_t lo, uint64_t hi) {
    uint32_t first = L2_WORD + blk * L2_N;
    uint64_t at = lo;
    while (at < hi) {
        uint64_t start = at;
        uint32_t record = G.storage.cpu[first + ((at - base) >> PAGE_SHIFT)];
        do {
            at = ((at >> PAGE_SHIFT) + 1) << PAGE_SHIFT;
        } while (at < hi && G.storage.cpu[first + ((at - base) >> PAGE_SHIFT)] == record);
        if (G.aliases[record].alias) {
            uint64_t stop = at < hi ? at : hi;
            RcInterval projected = {G.aliases[record].lo, 0, G.aliases[record].backing};
            write_range_uncertain(range, start, stop);
            coverage_record(coverage, start, stop, 0, &projected, rc_range_valid(projected.range));
        }
    }
}

static void write_scope_alias_pages(VkGpaWriteScope *scope) {
    if (!G.storage.cpu) return;
    const uint64_t block_size = (uint64_t)L2_N << PAGE_SHIFT;
    if (scope->whole_domain) {
        GpaWriteRange *range = scope->manifest ? &scope->manifest->uncertainty : &scope->ranges[0];
        GpaCoverage *coverage = scope->manifest ? &scope->manifest->canonical : &scope->canonical;
        for (uint32_t blk = 1; blk < G.nblocks; blk++) {
            if (!G.blk_alias_pages[blk]) continue;
            uint64_t base = G.blk_l1[blk] * block_size;
            writer_cost.mappings++;
            write_scope_alias_block(range, coverage, blk, base, base, base + block_size);
        }
        return;
    }
    for (uint32_t i=0;i<scope->count;i++) {
        GpaWriteRange *range = &scope->ranges[i];
        for (uint64_t l1=range->lo/block_size; l1<L1_N && l1<=(range->hi-1)/block_size; l1++) {
            uint32_t blk = G.storage.cpu[L1_WORD+l1];
            if (!blk || !G.blk_alias_pages[blk]) continue;
            uint64_t base=l1*block_size, end=base+block_size;
            writer_cost.mappings++;
            write_scope_alias_block(range, &scope->canonical, blk, base,
                                    range->lo > base ? range->lo : base, range->hi < end ? range->hi : end);
        }
    }
}

// Store an L2 entry and transfer current alias page ownership. Last-page departure queues retirement separately
// from the corresponding writer scopes, which still hold every location this record could have reached.
static void page_put(uint32_t w, uint32_t r, uint32_t *retired) {
    uint32_t blk = (w - L2_WORD) / L2_N, old = G.storage.cpu[w];
    if (old == r) return;
    if (definite_scopes) {
        uint64_t page = (uint64_t)G.blk_l1[blk] * L2_N + (w - L2_WORD) % L2_N;
        for (VkGpaWriteScope *scope = write_scopes; scope; scope = scope->next) {
            GpaDefiniteWrite *write = scope->definite;
            for (uint32_t i = 0; write && !write->invalid && i < write->count; i++) {
                RcInterval span = write->spans[i].interval;
                if ((span.address >> PAGE_SHIFT) <= page && page <= ((span.address + span.range.size - 1) >> PAGE_SHIFT))
                    write->invalid = 1;
            }
        }
    }
    put(w, r);
    if (!old && r) G.blk_pages[blk]++, G.mapped_pages++;
    else if (old && !r) G.blk_pages[blk]--, G.mapped_pages--;
    if (G.aliases[r].alias) {
        G.aliases[r].pages++;
        G.blk_alias_pages[blk]++;
    }
    if (G.aliases[old].alias) {
        G.blk_alias_pages[blk]--;
        if (!--G.aliases[old].pages) {
            G.aliases[old].retire_next = *retired;
            *retired = old;
        }
    }
}

static void blk_free_cb(void *arg) {
    uint32_t blk = (uint32_t)((uintptr_t)arg & 0xffff), gen = (uint32_t)((uintptr_t)arg >> 16);
    plat_mutex_lock(&gpa_lock);
    if (G.ready && G.blk_wait[blk] && G.blk_gen[blk] == gen) {   // not taken back since (block_take)
        G.blk_wait[blk] = 0, G.blk_waiting--;
        G.free_blks[G.nfree_blk++] = (uint16_t)blk;
    }
    plat_mutex_unlock(&gpa_lock);
}

// After pages of [p0, p1) were unmapped: the blocks of that range that map nothing leave L1, and go back to the free
// list once the GPU finished the submissions issued until now and the one being recorded, as a record does.
static void blocks_release(uint64_t p0, uint64_t p1) {
    if (p1 <= p0) return;
    for (uint64_t l1 = p0 >> 12; l1 <= (p1 - 1) >> 12 && l1 < L1_N; l1++) {
        plat_mutex_lock(&gpa_lock);
        uint32_t blk = G.storage.cpu[L1_WORD + l1], gen = 0;
        if (blk && !G.blk_pages[blk]) {
            put(L1_WORD + (uint32_t)l1, 0);
            G.blk_wait[blk] = 1, G.blk_waiting++, G.blk_returned++;
            gen = ++G.blk_gen[blk];
        } else {
            blk = 0;
        }
        plat_mutex_unlock(&gpa_lock);
        if (blk)
            vk_completion_add(vk_sync_issued() + 1, blk_free_cb, (void *)((uintptr_t)blk | (uintptr_t)gen << 16));
    }
}

// gpa_lock held: a page found no block. Pointers into it read the zero buffer until its buffer is mapped again.
static void out_of_blocks(uint64_t page) {
    if (!G.refused++)
        rt_log("vk_gpa: out of L2 blocks at %#llx (%u in use, %u waiting for the GPU): shaders read zeros through "
               "pointers into the pages left out", (unsigned long long)(page << PAGE_SHIFT),
               G.nblocks - 1 - G.nfree_blk - G.blk_waiting, G.blk_waiting);
}

static uint32_t rec_alloc(uint64_t delta, uint64_t twin) {
    uint32_t r;
    if (G.nfree) r = G.storage.free_recs[--G.nfree];
    else if (G.nrec < RECS) r = G.nrec++;
    else return 0;
    // The record before any page names it: the page store below is ordered after these by the release fence.
    put(REC_WORD + r * REC_WORDS, (uint32_t)delta);
    put(REC_WORD + r * REC_WORDS + 1, (uint32_t)(delta >> 32));
    put(REC_WORD + r * REC_WORDS + 2, (uint32_t)twin);
    put(REC_WORD + r * REC_WORDS + 3, (uint32_t)(twin >> 32));
    __atomic_thread_fence(__ATOMIC_RELEASE);
    return r;
}

static void rec_free_cb(void *arg) {
    uint32_t r = (uint32_t)(uintptr_t)arg;
    plat_mutex_lock(&gpa_lock);
    if (G.ready) {
        if (G.aliases[r].alias) {
            if (G.aliases[r].pages) rt_fatal("vk_gpa: retiring a live virtual record");
            memset(&G.aliases[r], 0, sizeof G.aliases[r]);
        } else {
            if (G.authority[r].prev) G.authority[G.authority[r].prev].next = G.authority[r].next;
            else G.authority_head = G.authority[r].next;
            if (G.authority[r].next) G.authority[G.authority[r].next].prev = G.authority[r].prev;
            free(G.authority[r].host);
            memset(&G.authority[r], 0, sizeof G.authority[r]);
        }
        G.storage.free_recs[G.nfree++] = r;
    }
    plat_mutex_unlock(&gpa_lock);
}

// vk_sync_issued advances after the current recording is queued. Its split parts may already be executing.
// Include that recording in the retirement fence and wait for its actual graphics timeline completion.
static void retire_virtual_records(uint32_t retired) {
    if (!retired) return;
    uint64_t fence = vk_sync_issued() + 1;
    while (retired) {
        uint32_t next = G.aliases[retired].retire_next;
        vk_completion_add(fence, rec_free_cb, (void *)(uintptr_t)retired);
        retired = next;
    }
}

static int table_empty(void) {
    static int empty = -1;
    if (empty < 0) { const char *e = cfg_get("NX2NAVI_GPA_EMPTY"); empty = e && !strcmp(e, "1"); }
    return empty;
}

uint32_t vk_gpa_map(uint64_t lo, uint64_t hi, VkBuffer buf, uint64_t buf_lo, int cpu_visible_after_fence,
                    const RcInterval *host, uint32_t host_count) {
    if (!vk_gpa_on() || !buf || hi <= lo) return 0;
    uint64_t dev = vk_gpa_device_address(buf);
    if (!dev) return 0;
    // Pages the buffer holds whole: [lo, hi) rounded inwards (imports are aligned outwards, so this is all of them)
    uint64_t p0 = (lo >> PAGE_SHIFT) + !!(lo & 4095), p1 = hi >> PAGE_SHIFT;
    if (p1 <= p0) return 0;
    plat_mutex_lock(&gpa_lock);
    if (table_empty()) { plat_mutex_unlock(&gpa_lock); return 0; }
    uint32_t r = rec_alloc(dev - buf_lo, dev - buf_lo);
    if (!r) {
        if (!G.refused++) rt_log("vk_gpa: out of records: a buffer at %#llx is not in the table", (unsigned long long)lo);
        plat_mutex_unlock(&gpa_lock);
        return 0;
    }
    authority_map(r, p0 << PAGE_SHIFT, p1 << PAGE_SHIFT, cpu_visible_after_fence, host, host_count);
    uint32_t retired = 0;
    for (uint64_t p = p0; p < p1; p++) {
        uint32_t w = page_word(p, 1);
        if (!w) {
            out_of_blocks(p);
            p |= L2_N - 1;   // the rest of this 16 MiB has no block either; the next range may have one
            continue;
        }
        page_put(w, r, &retired);
    }
    G.maps++;
    plat_mutex_unlock(&gpa_lock);
    retire_virtual_records(retired);
    return r;
}

void vk_gpa_unmap(uint64_t lo, uint64_t hi, uint32_t rec, uint32_t (*cover)(uint64_t page_addr)) {
    if (!rec || !G.ready) return;
    uint64_t p0 = (lo + 4095) >> PAGE_SHIFT, p1 = hi >> PAGE_SHIFT;
    uint32_t retired = 0;
    plat_mutex_lock(&gpa_lock);
    for (uint64_t p = p0; p < p1; p++) {
        uint32_t w = page_word(p, 0);
        if (w && G.storage.cpu[w] == rec) page_put(w, 0, &retired);
    }
    G.unmaps++;
    plat_mutex_unlock(&gpa_lock);
    // Pages another live buffer also holds whole (overlapping imports of the same guest memory) get its record back.
    if (cover) {
        for (uint64_t p = p0; p < p1; p++) {
            plat_mutex_lock(&gpa_lock);
            uint32_t w = page_word(p, 0), now = w ? G.storage.cpu[w] : 0;
            plat_mutex_unlock(&gpa_lock);
            if (!w || now) continue;
            uint32_t other = cover(p << PAGE_SHIFT);
            if (!other || other == rec) continue;
            plat_mutex_lock(&gpa_lock);
            if (!G.storage.cpu[w]) page_put(w, other, &retired);
            plat_mutex_unlock(&gpa_lock);
        }
    }
    retire_virtual_records(retired);
    blocks_release(p0, p1);   // after the cover: a block another buffer's pages went back into stays
    vk_completion_add(vk_sync_issued() + 1, rec_free_cb, (void *)(uintptr_t)rec);
}

RcBackingResult vk_gpa_virtual_prepare(VkGpaVirtualBinding *bindings, uint32_t count, VkGpaVirtualUpdate *update) {
    *update = (VkGpaVirtualUpdate){0};
    if (!count || !vk_gpa_wanted()) return RC_BACKING_OK;
    if (!vk_gpa_on()) return RC_BACKING_STALE;
    // This explicit diagnostic mode makes all guest pointers read zeros.
    plat_mutex_lock(&gpa_lock);
    int empty = table_empty();
    plat_mutex_unlock(&gpa_lock);
    if (empty) return RC_BACKING_OK;
    uint32_t records = 0;
    const uint64_t table_end = (uint64_t)L1_N * L2_N << PAGE_SHIFT;
    for (uint32_t i = 0; i < count; ++i) {
        VkGpaVirtualBinding *binding = &bindings[i];
        if (binding->hi <= binding->lo || binding->hi > table_end ||
            ((binding->lo | binding->hi) & ((1u << PAGE_SHIFT) - 1))) return RC_BACKING_INVALID;
        if (!binding->backing.backing) {
            if (binding->buffer) return RC_BACKING_INVALID;
            continue;
        }
        if (!rc_range_valid(binding->backing) || binding->backing.size != binding->hi - binding->lo)
            return RC_BACKING_INVALID;
        uint64_t dev = vk_gpa_device_address(binding->buffer);
        if (!dev) return RC_BACKING_STALE;
        if (dev > UINT64_MAX - (binding->backing.offset + binding->backing.size)) return RC_BACKING_INVALID;
        binding->delta = dev + binding->backing.offset - binding->lo;
        ++records;
    }

    plat_mutex_lock(&gpa_lock);
    uint64_t admitted[L1_N / 64] = {0};
    for (uint32_t i = 1; G.blk_waiting && i < G.nblocks; ++i) {
        if (!G.blk_wait[i]) continue;
        uint32_t l1 = G.blk_l1[i];
        admitted[l1 / 64] |= UINT64_C(1) << (l1 % 64);
    }
    uint32_t fresh = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const VkGpaVirtualBinding *binding = &bindings[i];
        if (!binding->buffer) continue;
        const unsigned shift = PAGE_SHIFT + 12;
        uint32_t last = (uint32_t)((binding->hi - 1) >> shift);
        for (uint32_t l1 = (uint32_t)(binding->lo >> shift); l1 <= last; ++l1) {
            const uint64_t bit = UINT64_C(1) << (l1 % 64);
            if (G.storage.cpu[L1_WORD + l1] || (admitted[l1 / 64] & bit)) continue;
            admitted[l1 / 64] |= bit;
            ++fresh;
        }
    }
    if (records > RECS - G.nrec + G.nfree || fresh > BLOCKS - G.nblocks + G.nfree_blk) {
        ++G.refused;
        plat_mutex_unlock(&gpa_lock);
        return RC_BACKING_CAPACITY;
    }
    *update = (VkGpaVirtualUpdate){bindings, count};
    return RC_BACKING_OK;
}

void vk_gpa_virtual_cancel(VkGpaVirtualUpdate *update) {
    if (update->count) plat_mutex_unlock(&gpa_lock);
    *update = (VkGpaVirtualUpdate){0};
}

// gpa_lock held. An unmap cannot allocate. A map has already admitted every needed record and L2 block.
static void virtual_publish(const VkGpaVirtualBinding *binding, uint32_t *retired) {
    const uint64_t p0 = binding->lo >> PAGE_SHIFT, p1 = binding->hi >> PAGE_SHIFT;
    uint32_t record = 0;
    if (binding->buffer) {
        record = rec_alloc(binding->delta, binding->delta);
        if (!record) rt_fatal("vk_gpa: virtual mapping lost its admitted record");
        G.aliases[record] = (typeof(G.aliases[record])){.backing = binding->backing, .lo = binding->lo, .alias = 1};
        RcInterval projection = {binding->lo, 0, binding->backing};
        authority_publish(binding->lo, binding->hi, 0, &projection, 1);
    }
    for (uint64_t page = p0; page < p1;) {
        uint64_t end = (page | (L2_N - 1)) + 1;
        if (end > p1) end = p1;
        uint32_t word = page_word(page, record != 0);
        if (!word && record) rt_fatal("vk_gpa: virtual mapping lost its admitted L2 block");
        if (word)
            for (uint32_t i = 0; i < end - page; ++i) page_put(word + i, record, retired);
        page = end;
    }
    if (record) ++G.maps;
    else ++G.unmaps;
}

void vk_gpa_virtual_commit(VkGpaVirtualUpdate *update) {
    if (!update->count) return;
    const VkGpaVirtualUpdate admitted = *update;
    uint32_t retired = 0;
    for (uint32_t i = 0; i < admitted.count; ++i) virtual_publish(&admitted.bindings[i], &retired);
    vk_gpa_virtual_cancel(update);
    retire_virtual_records(retired);
    for (uint32_t i = 0; i < admitted.count; ++i) {
        const VkGpaVirtualBinding *binding = &admitted.bindings[i];
        if (!binding->buffer) blocks_release(binding->lo >> PAGE_SHIFT, binding->hi >> PAGE_SHIFT);
    }
}

void vk_gpa_virtual_unmap(uint64_t lo, uint64_t hi) {
    if (!vk_gpa_on() || hi <= lo) return;
    const VkGpaVirtualBinding binding = {.lo = lo, .hi = hi};
    plat_mutex_lock(&gpa_lock);
    VkGpaVirtualUpdate update = {&binding, 1};
    vk_gpa_virtual_commit(&update);
}

uint32_t vk_gpa_remap_exact(uint64_t lo, uint64_t hi, uint32_t old_rec, VkBuffer buf, uint64_t buf_lo,
                            int keep_twin, int cpu_visible_after_fence, int *uses_twin) {
    if (uses_twin) *uses_twin = 0;
    if (!old_rec || !buf || (lo & 4095) || (hi & 4095) || lo >= hi || !vk_gpa_on()) return 0;
    uint64_t dev = vk_gpa_device_address(buf);
    if (!dev) return 0;
    uint64_t p0 = lo >> PAGE_SHIFT, p1 = hi >> PAGE_SHIFT;
    if (p1 > (uint64_t)L1_N * L2_N) return 0;
    plat_mutex_lock(&gpa_lock);
    if (old_rec >= G.nrec || G.aliases[old_rec].alias || G.authority[old_rec].lo != lo ||
        G.authority[old_rec].hi != hi) {
        plat_mutex_unlock(&gpa_lock);
        return 0;
    }
    for (uint64_t page = p0; page < p1; page++) {
        uint32_t word = page_word(page, 0);
        if (!word || G.storage.cpu[word] != old_rec) {
            plat_mutex_unlock(&gpa_lock);
            return 0;
        }
    }

    // Neither missing pages nor metadata/record pressure may replace a subset of the live import.
    uint32_t count = G.authority[old_rec].host_count;
    RcInterval *host = count ? calloc(count, sizeof *host) : NULL;
    if (count && !host) { plat_mutex_unlock(&gpa_lock); return 0; }
    if (host) memcpy(host, G.authority[old_rec].host, count * sizeof *host);
    static int twins = -1;
    if (twins < 0) { const char *e = cfg_get("NX2NAVI_GPA_TWIN"); twins = !(e && !strcmp(e, "0")); }
    int twin = keep_twin && twins;
    uint32_t offset = REC_WORD + old_rec * REC_WORDS;
    uint64_t delta = twin ? (uint64_t)G.storage.cpu[offset] | (uint64_t)G.storage.cpu[offset + 1] << 32 : dev - buf_lo;
    uint32_t record = rec_alloc(dev - buf_lo, delta);
    if (!record) {
        free(host);
        plat_mutex_unlock(&gpa_lock);
        return 0;
    }
    authority_map_owned(record, lo, hi, cpu_visible_after_fence && (!keep_twin || twin), host, count);
    uint32_t retired = 0;
    for (uint64_t page = p0; page < p1; page++) page_put(page_word(page, 0), record, &retired);
    plat_mutex_unlock(&gpa_lock);
    retire_virtual_records(retired);
    vk_completion_add(vk_sync_issued() + 1, rec_free_cb, (void *)(uintptr_t)old_rec);
    if (uses_twin) *uses_twin = twin;
    return record;
}

void vk_gpa_report(void) {
    if (!G.ready) return;
    plat_mutex_lock(&gpa_lock);
    rt_log("vk_gpa: %llu pages (%.0f MiB) mapped, %u of %u blocks in use (%u ever, %u waiting for the GPU, %llu given "
           "back), %u records in use, %llu maps, %llu unmaps, %llu refused",
           (unsigned long long)G.mapped_pages, G.mapped_pages * 4096.0 / 1048576.0,
           G.nblocks - 1 - G.nfree_blk - G.blk_waiting, BLOCKS - 1, G.nblocks - 1, G.blk_waiting,
           (unsigned long long)G.blk_returned, G.nrec - 1 - G.nfree, (unsigned long long)G.maps,
           (unsigned long long)G.unmaps, (unsigned long long)G.refused);
    plat_mutex_unlock(&gpa_lock);
}

void vk_gpa_shutdown(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    // The backend has stopped creating users and drained the graphics and upload queues.
    storage_destroy(c, &G.storage);
    if (!G.ready) return;
    for (uint32_t i = 1; i < G.nrec; i++) free(G.authority[i].host);
    memset(&G, 0, sizeof G);
    __atomic_store_n(&G.ready, 0, __ATOMIC_RELEASE);
}

// NX2NAVI_TRACE=perf "mem: sizes" (perf.c): device address table records and L2 blocks in use (those waiting for a
// fence included), against their fixed sizes (when they run out, shaders read zeros).
void vk_gpa_sizes(PerfSizes *s) {
    plat_mutex_lock(&gpa_lock);
    uint64_t recs = G.nrec > 1 + G.nfree ? G.nrec - 1 - G.nfree : 0;
    uint64_t blks = G.nblocks > 1 + G.nfree_blk + G.blk_waiting ? G.nblocks - 1 - G.nfree_blk - G.blk_waiting : 0;
    plat_mutex_unlock(&gpa_lock);
    perf_size(s, "device address records", recs, RECS, PERF_SIZE_NONE);
    perf_size(s, "device address blocks", blks, BLOCKS, PERF_SIZE_NONE);
}

void vk_gpa_effects_report(uint64_t frame) {
    plat_mutex_lock(&gpa_lock);
    rt_log("resource GPA effects: frame %llu scopes %llu mappings_visited %llu uncertainty_insertions %llu "
        "map_scope_joins %llu begin_ns %llu end_ns %llu live %u peak %u scene_manifests %llu manifest_reuses %llu",
        (unsigned long long)frame,(unsigned long long)writer_cost.scopes,(unsigned long long)writer_cost.mappings,
        (unsigned long long)writer_cost.uncertainty,(unsigned long long)writer_cost.map_joins,
        (unsigned long long)writer_cost.begin_ns,(unsigned long long)writer_cost.end_ns,writer_cost.live,writer_cost.peak,
        (unsigned long long)writer_cost.manifests,(unsigned long long)writer_cost.manifest_reuses);
    uint32_t live=writer_cost.live;memset(&writer_cost,0,sizeof writer_cost);writer_cost.live=writer_cost.peak=live;
    plat_mutex_unlock(&gpa_lock);
}
