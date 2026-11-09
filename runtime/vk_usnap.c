// Uniform blocks as of the submit call. NVN starts the GPU when nvnQueueSubmitCommands returns and the GPU reads the
// constants soon after; here the replay records for several milliseconds, and the GPU reads a pool's bytes when it
// runs the part that uses them (the VRAM mirror's sync copies run then, vk_memory.c "VRAM mirror"). Meanwhile the game,
// having handed the frame over, rewrites per-object uniform blocks on other threads with no fence: late draws read
// another object's matrix (phantom rocks, a flat far field; NX2navi 1.2.1's producer f_70008c6460, 14-24 ms after the
// submit). So at the start of a submission's recording every uniform range it binds is copied into an arena of its own,
// and the uniform bindings of the replay use the copy.
// - Only ranges of imported pools (memory the CPU writes) are copied, aligned out to 256 bytes (the largest
//   minUniformBufferOffsetAlignment) and at most 64 KiB from the bound address, overlapping ones merged.
// - A range the GPU wrote in a submission that has not completed is left out: its bytes reach guest memory only with
//   that submission's write-back, the mirror holds them until then.
// - A GPU write within the submission (UpdateUniformBuffer, a storage buffer a shader writes, a buffer copy or fill,
//   an image-to-buffer copy: vk_usnap_drop) drops the copies it touches from then on, so later draws read the result.
// - The arena goes back to a free list once the submission's fence is done (completion thread).
// The copy is taken where vk_exec_record starts: at the submit call when the replay runs on the submitting thread, later
// with NX2NAVI_REPLAY_ASYNC. NX2NAVI_UBO_SNAPSHOT=0 binds live memory as before; =ab alternates (NX2NAVI_AB).
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "nvn_api.h"
#include "vk.h"
#include "vk_dlaa.h"
#include "platform/platform.h"

#define ALIGN 256u
#define MAX_RANGE (64u << 10)
#define ARENA_MIN (4u << 20)
#define CALL_DEPTH 16

typedef struct Arena {
    struct Arena *next;
    unsigned owners;
    VkBuffer buf;
    VmaAllocation alloc;
    uint8_t *map;
    uint8_t *cpu;    // with DLSS (vk_dlaa_available), the same bytes in host memory for the CPU's reads (vk_usnap_bytes):
                     // map is often device memory through the BAR, written fast but read back at about a microsecond a
                     // cache line. Without DLSS nothing reads them back, and the bytes go into map in one copy.
    uint64_t size;
} Arena;

typedef struct { uint64_t lo, hi; } Rng;
typedef struct { uint64_t lo, hi, off; int dropped; } Reg;
typedef struct { uint64_t lo, hi, fence; } Wrote;

// A deferred queue keeps its original arena and sorted range table. All accesses to the tables
// are serialized by the replay lock. Only the arena return runs on the GPU completion thread.
struct VkUniformSnapshot {
    Arena *arena;
    Reg *reg;
    int nreg;
    VkUniformSnapshot *next;
};
static VkUniformSnapshot submit_snapshot, empty_snapshot;
static VkUniformSnapshot *active = &submit_snapshot, *held;

static struct {
    int mode;                   // rt_knob: 0 off, 1 on, 2 ab
    uint64_t fence, gen;
    Rng *col;                   // collected ranges
    int ncol, ccol;
    int creg;                   // capacity of the reusable submission range table
    Wrote *wr;                  // GPU writes of submissions not known to be complete
    int nwr, cwr;
    // since the last summary
    uint64_t subs, regions, bytes, copy_ns, hits, drops, inflight, refused, off_subs;
} U;

static Arena *free_arenas;
static PlatMutex free_lock = PLAT_MUTEX_INIT;

static void *grow(void *p, int *cap, size_t elem) {
    int n = *cap ? *cap * 2 : 1024;
    void *q = realloc(p, (size_t)n * elem);
    if (!q) rt_fatal("vk_usnap: out of memory");
    *cap = n;
    return q;
}

static Arena *arena_get(uint64_t need) {
    plat_mutex_lock(&free_lock);
    Arena **pp = &free_arenas, *a = NULL;
    for (; *pp; pp = &(*pp)->next)
        if ((*pp)->size >= need) {
            a = *pp, *pp = a->next;
            break;
        }
    plat_mutex_unlock(&free_lock);
    if (a) {
        a->owners = 1;
        return a;
    }
    a = calloc(1, sizeof *a);
    if (!a) rt_fatal("vk_usnap: out of memory");
    a->owners = 1;
    a->size = need < ARENA_MIN ? ARENA_MIN : (need + (1u << 20) - 1) & ~(uint64_t)((1u << 20) - 1);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = a->size,
                             // transfer source: DLAA copies the camera block out (vk_dlaa.c mv_pass)
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    // Device-local through the BAR when the device has it (the GPU reads constants on every draw), else host memory.
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    VkResult res = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &a->buf, &a->alloc, &info);
    if (res != VK_SUCCESS) {
        rt_log("vk_usnap: arena of %llu bytes failed (%d): this submission binds live memory",
               (unsigned long long)a->size, (int)res);
        free(a);
        return NULL;
    }
    a->map = info.pMappedData;
    rt_log("vk_usnap: new arena of %.1f MiB", a->size / 1048576.0);
    return a;
}

static void arena_done(void *arg) {   // completion thread: the GPU has read it
    Arena *a = arg;
    if (__atomic_sub_fetch(&a->owners, 1, __ATOMIC_ACQ_REL)) return;
    plat_mutex_lock(&free_lock);
    a->next = free_arenas, free_arenas = a;
    plat_mutex_unlock(&free_lock);
}

static void collect_binding(const p_bind_buf *b) {
    if (!b->addr || !b->size) return;
    uint64_t n = b->size < MAX_RANGE ? b->size : MAX_RANGE;
    // the vertex stage's object and bone blocks: also the game's other copy on either side, which still
    // holds last frame's (vk_mvprev.c)
    uint64_t step = vk_mvprev_step(b->stage, b->index, b->size);
    // which neighbours lie in an imported pool, per block and vk_mem_gen: the same blocks are bound by
    // many draws, and two resolves for each binding were a quarter of the snapshot's time
    static struct { uint64_t addr, size, gen; int ok; } nc[1024];
    uint32_t ci = (uint32_t)((b->addr ^ n << 48) * 0x9E3779B97F4A7C15ull >> 54);
    uint64_t gen = vk_mem_gen();
    int known = step && nc[ci].addr == b->addr && nc[ci].size == n && nc[ci].gen == gen, ok = 0;
    for (int k = step ? -1 : 0; k <= (step ? 1 : 0); k++) {
        uint64_t at = b->addr + (uint64_t)((int64_t)k * (int64_t)step);
        if (k < 0 && b->addr < step) continue;
        // a neighbour past its pool would make the whole merged region fail the pool check below, and
        // the bindings in it read live memory again
        VkMemSpan sp;
        uint64_t lo = at & ~(uint64_t)(ALIGN - 1);
        if (k && known) {
            if (!(nc[ci].ok >> (k + 1) & 1)) continue;
        } else if (k) {
            if (!vk_mem_resolve(lo, at + n - lo, &sp) || sp.alloc || !sp.pool || sp.avail < at + n - lo ||
                (sp.cls != VK_MCLS_HOST_CACHED && sp.cls != VK_MCLS_HOST_UNC))
                continue;
            ok |= 1 << (k + 1);
        }
        if (U.ncol == U.ccol) U.col = grow(U.col, &U.ccol, sizeof *U.col);
        U.col[U.ncol++] = (Rng){at & ~(uint64_t)(ALIGN - 1), at + n};
    }
    if (step && !known) nc[ci].addr = b->addr, nc[ci].size = n, nc[ci].gen = gen, nc[ci].ok = ok;
}

static void collect(const op_hdr *h, int depth, p_bind_buf bindings[8][18], int capture) {
    for (;;) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_)
            rt_fatal("vk_usnap: corrupt command record at %p: op %u size %u", (const void *)h, h->op, h->size);
        const void *a = nvn_op_payload(h);
        if (h->op == OP_END) return;
        if (h->op == OP_JUMP) {
            h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target;
            continue;
        }
        if (h->op == OP_CALL && depth < CALL_DEPTH) collect(nvn_op_first(((const p_jump *)a)->target), depth + 1, bindings, capture);
        if (h->op == OP_BIND_UBO) {
            const p_bind_buf *b = a;
            if (bindings && (unsigned)b->stage < 8 && (unsigned)b->index < 18)
                bindings[b->stage][b->index] = *b;
            if (capture) collect_binding(b);
        }
        h = nvn_op_next(h);
    }
}

static int by_lo(const void *a, const void *b) {
    uint64_t x = ((const Rng *)a)->lo, y = ((const Rng *)b)->lo;
    return x < y ? -1 : x > y;
}
static int wr_by_lo(const void *a, const void *b) {
    uint64_t x = ((const Wrote *)a)->lo, y = ((const Wrote *)b)->lo;
    return x < y ? -1 : x > y;
}

// Sorts the collected ranges by lo. A town frame binds about 10,000 uniform ranges and qsort took half a millisecond
// of the copy's 0.96 ms on the game thread, so they go through an LSD radix sort on (lo - lowest) / ALIGN, three
// passes of 11 bits; qsort is left for a span that does not fit in 33 bits (2 TiB).
#define RADIX_BITS 11
static void sort_col(void) {
    static Rng *tmp;
    static int ctmp;
    int n = U.ncol;
    if (n < 2) return;
    uint64_t lo = UINT64_MAX, hi = 0;
    for (int i = 0; i < n; i++) {
        if (U.col[i].lo < lo) lo = U.col[i].lo;
        if (U.col[i].lo > hi) hi = U.col[i].lo;
    }
    if ((hi - lo) / ALIGN >> (3 * RADIX_BITS)) {
        qsort(U.col, (size_t)n, sizeof *U.col, by_lo);
        return;
    }
    if (ctmp < n) {
        free(tmp);
        ctmp = U.ccol;
        if (!(tmp = malloc((size_t)ctmp * sizeof *tmp))) rt_fatal("vk_usnap: out of memory");
    }
    Rng *src = U.col, *dst = tmp;
    for (int pass = 0; pass < 3; pass++) {
        int shift = pass * RADIX_BITS, count[1 << RADIX_BITS] = {0};
        for (int i = 0; i < n; i++) count[(src[i].lo - lo) / ALIGN >> shift & ((1 << RADIX_BITS) - 1)]++;
        for (int k = 0, sum = 0; k < 1 << RADIX_BITS; k++) {
            int c = count[k];
            count[k] = sum, sum += c;
        }
        for (int i = 0; i < n; i++) dst[count[(src[i].lo - lo) / ALIGN >> shift & ((1 << RADIX_BITS) - 1)]++] = src[i];
        Rng *t = src;
        src = dst, dst = t;
    }
    if (src != U.col) memcpy(U.col, src, (size_t)n * sizeof *src);   // three passes end in tmp
}

// Keeps the GPU writes of submissions that are not done, merged into disjoint ranges sorted by lo (the terrain rewrites
// one block before every draw: hundreds of equal entries), each with the newest fence among its parts.
static void wr_prune(void) {
    uint64_t done = vk_sync_fence_done();
    int n = 0;
    for (int i = 0; i < U.nwr; i++)
        if (U.wr[i].fence > done) U.wr[n++] = U.wr[i];
    qsort(U.wr, (size_t)n, sizeof *U.wr, wr_by_lo);
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (m && U.wr[i].lo <= U.wr[m - 1].hi) {
            if (U.wr[i].hi > U.wr[m - 1].hi) U.wr[m - 1].hi = U.wr[i].hi;
            if (U.wr[i].fence > U.wr[m - 1].fence) U.wr[m - 1].fence = U.wr[i].fence;
        } else {
            U.wr[m++] = U.wr[i];
        }
    }
    U.nwr = m;
}
static int wr_hits(uint64_t lo, uint64_t hi) {   // wr_prune ran: disjoint, sorted
    int a = 0, b = U.nwr;
    while (a < b) {
        int k = (a + b) / 2;
        if (U.wr[k].hi <= lo) a = k + 1;
        else b = k;
    }
    return a < U.nwr && U.wr[a].lo < hi;
}

static void summary(void) {
    if (U.subs + U.off_subs < 3600) return;
    double s = U.subs ? (double)U.subs : 1.0;
    rt_log("vk_usnap: %llu submissions with a snapshot, %llu without; per submission %.0f ranges, %.1f KiB copied in "
           "%.3f ms, %.0f uniform bindings from the copy, %.1f dropped by GPU writes, %.1f left out (GPU writes in "
           "flight), %.1f refused (not an imported pool)",
           (unsigned long long)U.subs, (unsigned long long)U.off_subs, U.regions / s, U.bytes / s / 1024.0,
           U.copy_ns / s / 1e6, U.hits / s, U.drops / s, U.inflight / s, U.refused / s);
    U.subs = U.off_subs = U.regions = U.bytes = U.copy_ns = U.hits = U.drops = U.inflight = U.refused = 0;
}

void vk_usnap_begin(const uint64_t *handles, int n, uint64_t fence, p_bind_buf bindings[8][18]) {
    if (!U.mode) U.mode = rt_knob("UBO_SNAPSHOT", "1") + 1;   // stored +1: 0 is "not read yet"
    active = &submit_snapshot;
    active->nreg = 0;
    U.fence = fence;
    U.gen++;
    summary();
    int capture = RT_KNOB_ON(U.mode - 1);
    if (!capture) {
        U.off_subs++;
        if (U.mode == 1 || !bindings) return;
    }
    uint64_t t0 = kern_now_ns();
    U.ncol = 0;
    // A submission can draw with a uniform bound by an earlier, still-blocked submission of
    // its queue. Advance the queue's binding table in submit order, independently of replay.
    if (capture && bindings) {
        for (int st = 0; st < 8; ++st)
            for (int slot = 0; slot < 18; ++slot)
                if (bindings[st][slot].size) collect_binding(&bindings[st][slot]);
    }
    for (int i = 0; i < n; i++)
        if (handles[i]) collect(nvn_op_first(handles[i]), 0, bindings, capture);
    if (!capture) return;
    sort_col();
    wr_prune();
    // merge, keeping only what lies wholly in one imported pool and no GPU write in flight touches
    uint64_t total = 0;
    for (int i = 0; i < U.ncol;) {
        uint64_t lo = U.col[i].lo, hi = U.col[i].hi;
        for (i++; i < U.ncol && U.col[i].lo <= hi; i++)
            if (U.col[i].hi > hi) hi = U.col[i].hi;
        VkMemSpan sp;
        if (!vk_mem_resolve(lo, hi - lo, &sp) || sp.alloc || !sp.pool || sp.avail < hi - lo ||
            (sp.cls != VK_MCLS_HOST_CACHED && sp.cls != VK_MCLS_HOST_UNC)) {
            U.refused++;
            continue;
        }
        if (wr_hits(lo, hi)) {
            U.inflight++;
            continue;
        }
        if (active->nreg == U.creg) active->reg = grow(active->reg, &U.creg, sizeof *active->reg);
        active->reg[active->nreg++] = (Reg){lo, hi, total, 0};
        total += (hi - lo + ALIGN - 1) & ~(uint64_t)(ALIGN - 1);
    }
    if (!active->nreg) return;
    if (!(active->arena = arena_get(total))) return;
    uint8_t *dst = active->arena->map;
    if (vk_dlaa_available()) {
        if (!active->arena->cpu && !(active->arena->cpu = malloc(active->arena->size))) rt_fatal("vk_usnap: out of memory");
        dst = active->arena->cpu;
    }
    for (int i = 0; i < active->nreg; i++)
        memcpy(dst + active->reg[i].off, (const void *)(uintptr_t)active->reg[i].lo, active->reg[i].hi - active->reg[i].lo);
    if (dst != active->arena->map) memcpy(active->arena->map, dst, total);   // one sequential write into the device copy
    vmaFlushAllocation(vk_ctx()->vma, active->arena->alloc, 0, total);   // a no-op on coherent memory
    U.subs++, U.regions += (uint64_t)active->nreg, U.bytes += total, U.copy_ns += kern_now_ns() - t0;
}

VkUniformSnapshot *vk_usnap_hold(void) {
    if (!active->arena) return &empty_snapshot;
    VkUniformSnapshot *s = malloc(sizeof *s + sizeof *s->reg * (size_t)active->nreg);
    if (!s) rt_fatal("vk_usnap: out of memory (held uniforms)");
    s->arena = active->arena;
    __atomic_add_fetch(&s->arena->owners, 1, __ATOMIC_RELAXED);
    s->nreg = active->nreg;
    s->reg = (Reg *)(s + 1);
    memcpy(s->reg, active->reg, sizeof *s->reg * (size_t)s->nreg);
    s->next = held;
    held = s;
    return s;
}

void vk_usnap_use(VkUniformSnapshot *s) {
    active = s ? s : &submit_snapshot;
    U.gen++;
}

void vk_usnap_release(VkUniformSnapshot *s, uint64_t fence) {
    if (s == &empty_snapshot) return;
    VkUniformSnapshot **slot = &held;
    while (*slot && *slot != s) slot = &(*slot)->next;
    if (!*slot || s == active) rt_fatal("vk_usnap: releasing an invalid or active snapshot");
    *slot = s->next;
    vk_completion_add(fence, arena_done, s->arena);
    free(s);
}

void vk_usnap_end(uint64_t fence) {
    if (active != &submit_snapshot) rt_fatal("vk_usnap: a deferred snapshot is still selected");
    if (active->arena) vk_completion_add(fence, arena_done, active->arena);
    active->arena = NULL;
    active->nreg = 0;
    U.gen++;
}

// First region whose end is above a.
static int first_above(const VkUniformSnapshot *s, uint64_t a) {
    int lo = 0, hi = s->nreg;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (s->reg[m].hi <= a) lo = m + 1;
        else hi = m;
    }
    return lo;
}

int vk_usnap_find(uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out) {
    if (!active->arena || !n) return 0;
    int i = first_above(active, addr);
    if (i >= active->nreg) return 0;
    const Reg *r = &active->reg[i];
    if (r->dropped || addr < r->lo || n > r->hi - addr) return 0;
    uint64_t off = r->off + (addr - r->lo);
    if (off % align) return 0;
    *out = (VkDescriptorBufferInfo){active->arena->buf, off, n};
    U.hits++;
    return 1;
}

static void drop_range(VkUniformSnapshot *s, uint64_t lo, uint64_t hi) {
    for (int i = first_above(s, lo); i < s->nreg && s->reg[i].lo < hi; i++) {
        if (!s->reg[i].dropped) {
            s->reg[i].dropped = 1;
            U.drops++;
            U.gen++;
        }
    }
}

void vk_usnap_drop(uint64_t addr, uint64_t n) {
    if (!U.mode || !n) return;
    if (U.nwr == U.cwr) U.wr = grow(U.wr, &U.cwr, sizeof *U.wr);
    U.wr[U.nwr++] = (Wrote){addr, addr + n, U.fence};
    // A producer can overwrite a waiting consumer's input, even when the consumer resumes after
    // the producer's physical fence has already retired. Mark the overlapping snapshots now.
    drop_range(&submit_snapshot, addr, addr + n);
    for (VkUniformSnapshot *s = held; s; s = s->next) drop_range(s, addr, addr + n);
}

uint64_t vk_usnap_gen(void) { return U.gen; }

// the host copy where there is one (DLSS), else the device copy
static uint8_t *host_copy(void) { return active->arena->cpu ? active->arena->cpu : active->arena->map; }

const uint8_t *vk_usnap_map(void) { return active->arena ? host_copy() : NULL; }

const uint8_t *vk_usnap_bytes(uint64_t addr, uint64_t n) {
    VkDescriptorBufferInfo bi;
    return vk_usnap_find(addr, n, 1, &bi) ? host_copy() + bi.offset : NULL;
}

// Retain the draw's uniform bytes across submissions. A directly bound range is
// readable only in an imported CPU pool and while no outstanding GPU writer overlaps it.
int vk_usnap_read_draw(uint64_t addr, size_t n, void *dst) {
    if (!n || addr > UINT64_MAX - n) return 0;
    const uint8_t *bytes = vk_usnap_bytes(addr, n);
    if (!bytes) {
        VkMemSpan span;
        if (U.mode <= 1 || !vk_mem_resolve(addr, n, &span) || !span.pool || span.alloc || span.avail < n ||
            (span.cls != VK_MCLS_HOST_UNC && span.cls != VK_MCLS_HOST_CACHED)) return 0;
        // vk_usnap_drop appends unsorted writes after the submission's initial prune.
        for (int i = 0; i < U.nwr; i++)
            if (addr < U.wr[i].hi && U.wr[i].lo < addr + n) return 0;
        bytes = (const uint8_t *)(uintptr_t)addr;
    }
    memcpy(dst, bytes, n);
    return 1;
}

// The arenas back on the free list (vk_exec_draw_shutdown: the device is idle, every completion ran).
void vk_usnap_shutdown(void) {
    plat_mutex_lock(&free_lock);
    for (Arena *a = free_arenas, *next; a; a = next) next = a->next, vmaDestroyBuffer(vk_ctx()->vma, a->buf, a->alloc), free(a->cpu), free(a);
    free_arenas = NULL;
    plat_mutex_unlock(&free_lock);
}
