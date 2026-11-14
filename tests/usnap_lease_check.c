// Snapshot lookup and prepared-binding leases against an independent interval oracle. The actual snapshot code runs
// with host allocations in place of Vulkan arenas, so capture, drop ordering and asynchronous adoption need no GPU.
#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include "../runtime/vk_usnap.c"

static uint64_t epoch = 1, completed, ticks;
static VkCtx ctx;
static uint8_t *guest;
static size_t guest_size;
static int indexed, preparation = 1;
int resource_prep_mode(void) { return preparation; }
static size_t uniform_offsets[] = {8, 8 + 8 + sizeof(p_bind_buf)};
VkCtx *vk_ctx(void) { return &ctx; }
int vk_exec_shared(void) { return 1; }
uint64_t vk_exec_epoch(int write) { (void)write; return epoch; }
uint64_t vk_sync_fence_done(void) { return completed; }
uint64_t kern_now_ns(void) { return ++ticks; }
int rt_knob(const char *name, const char *def) { (void)name; (void)def; return 1; }
int rt_ab(void) { return 0; }
int vk_dlaa_available(void) { return 0; }
uint64_t vk_mvprev_step(int stage, int index, uint64_t size) { (void)stage; (void)index; (void)size; return 0; }
uint64_t vk_mem_gen(void) { return 1; }
int vk_mem_lookup(uint64_t addr, uint64_t n, VkMemSpan *out) {
    uint64_t lo = (uintptr_t)guest;
    if (addr < lo || addr - lo >= guest_size || n > guest_size - (addr - lo)) return 0;
    *out = (VkMemSpan){.pool = (void *)guest, .avail = guest_size - (addr - lo), .cls = VK_MCLS_HOST_CACHED};
    return 1;
}
int vk_mem_resolve(uint64_t a, uint64_t n, VkMemSpan *out) { return vk_mem_lookup(a, n, out); }
int nvn_record_plan(uint64_t h, NvnRecordPlan *out) {
    *out = (NvnRecordPlan){0};
    if (!indexed) return 0;
    out->commands = (const uint8_t *)(uintptr_t)h;
    out->capture_offsets = uniform_offsets;
    out->ncapture = 2;
    return 1;
}
void vk_exec_par(void (*fn)(void *, int, int), void *arg, int n, int slice) { (void)slice; fn(arg, 0, n); }
void vk_completion_add(uint64_t fence, void (*fn)(void *), void *arg) { (void)fence; fn(arg); }
void plat_mutex_lock(PlatMutex *p) { (void)p; }
void plat_mutex_unlock(PlatMutex *p) { (void)p; }
void rt_log(const char *fmt, ...) { (void)fmt; }
_Noreturn void rt_fatal(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    abort();
}
void perf_size(PerfSizes *s, const char *name, uint64_t count, uint64_t limit, uint64_t bytes) {
    (void)s; (void)name; (void)count; (void)limit; (void)bytes;
}
VKAPI_ATTR VkResult VKAPI_CALL vmaCreateBuffer(VmaAllocator a, const VkBufferCreateInfo *bi,
        const VmaAllocationCreateInfo *ai, VkBuffer *b, VmaAllocation *allocation, VmaAllocationInfo *info) {
    (void)a; (void)ai;
    void *data = malloc((size_t)bi->size);
    assert(data);
    *b = (VkBuffer)data, *allocation = (VmaAllocation)data;
    memset(info, 0, sizeof *info);
    info->pMappedData = data;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vmaDestroyBuffer(VmaAllocator a, VkBuffer b, VmaAllocation allocation) {
    (void)a; (void)b; free(allocation);
}
VKAPI_ATTR VkResult VKAPI_CALL vmaFlushAllocation(VmaAllocator a, VmaAllocation allocation, VkDeviceSize off, VkDeviceSize n) {
    (void)a; (void)allocation; (void)off; (void)n; return VK_SUCCESS;
}

int main(void) {
    guest_size = 65536;
    uint8_t *allocation = malloc(guest_size + ALIGN);
    assert(allocation);
    guest = (uint8_t *)(((uintptr_t)allocation + ALIGN - 1) & ~(uintptr_t)(ALIGN - 1));
    memset(guest, 0x11, guest_size);
    struct { op_hdr begin, bind; p_bind_buf b; op_hdr bind2; p_bind_buf b2; op_hdr end; } cmd = {
        {OP_BEGIN, 8}, {OP_BIND_UBO, 8 + sizeof(p_bind_buf)}, {0, 3, (uintptr_t)guest, 1024},
        {OP_BIND_UBO, 8 + sizeof(p_bind_buf)}, {0, 4, (uintptr_t)guest + 2048, 512}, {OP_END, 8}};
    uint64_t handle = (uintptr_t)&cmd;
    VkDescriptorBufferInfo bi;
    VkUsnapLease lease;
    vk_usnap_begin(&handle, 1, 1);
    assert(vk_usnap_find_lease((uintptr_t)guest + 256, 128, 256, &bi, &lease));
    assert(vk_usnap_lease_valid(&lease) && bi.range == 128 && bi.offset == 256);
    assert(vk_usnap_bytes((uintptr_t)guest, 1)[0] == 0x11);
    VkUsnapProof proof;
    vk_usnap_proof_begin(&proof);
    assert(vk_usnap_proof_add(&proof, &lease) && vk_usnap_proof_end(&proof));
    assert(vk_usnap_proof_valid(&proof));
    VkUsnapProof invalid_proof;
    VkUsnapLease invalid_lease = lease;
    invalid_lease.region = UINT32_MAX;
    vk_usnap_proof_begin(&invalid_proof);
    assert(!vk_usnap_proof_add(&invalid_proof, &invalid_lease) && !vk_usnap_proof_end(&invalid_proof));
    uint64_t sealed_serial = proof.serial;
    uint64_t capture = vk_usnap_capture(), gen = vk_usnap_gen();
    vk_usnap_drop((uintptr_t)guest + 2048, 64);
    assert(vk_usnap_capture() == capture && vk_usnap_gen() != gen && vk_usnap_lease_valid(&lease));
    // A different dependency bucket changed: keep the aggregate without rechecking the descriptor's lease.
    assert(vk_usnap_proof_valid(&proof) && proof.serial != sealed_serial);
    gen = vk_usnap_gen();

    // The main replay may reach a later write while a worker is preparing an earlier draw. Both lookup and lease
    // validation must still read the old snapshot for that worker, then reject it once its own epoch reaches the write.
    epoch = 20;
    vk_usnap_drop((uintptr_t)guest + 300, 4);
    assert(vk_usnap_capture() == capture && vk_usnap_gen() != gen);
    assert(!vk_usnap_proof_valid(&proof));
    epoch = 1;
    vk_usnap_proof_begin(&proof);
    assert(vk_usnap_proof_add(&proof, &lease) && vk_usnap_proof_end(&proof));
    assert(proof.horizon == 20);
    for (epoch = 1; epoch <= 30; epoch++) {
        int expected = epoch < 20;
        VkUsnapLease another;
        assert(vk_usnap_lease_valid(&lease) == expected);
        assert(vk_usnap_proof_valid(&proof) == expected);
        assert(vk_usnap_find_lease(lease.addr, lease.size, lease.align, &bi, &another) == expected);
        if (!expected) assert(!another.capture);
    }
    // Moving an already recorded drop earlier must publish another invalidation, even though it is still in the
    // future of the current consumer. The replacement proof must remember the earlier horizon.
    epoch = 10;
    vk_usnap_drop((uintptr_t)guest + 300, 4);
    epoch = 1;
    assert(!vk_usnap_proof_valid(&proof));
    vk_usnap_proof_begin(&proof);
    assert(vk_usnap_proof_add(&proof, &lease) && vk_usnap_proof_end(&proof) && proof.horizon == 10);
    epoch = 10;
    assert(!vk_usnap_proof_valid(&proof));
    epoch = 1;
    vk_usnap_proof_begin(&proof);
    assert(vk_usnap_proof_add(&proof, &lease));
    epoch = 2;
    assert(!vk_usnap_proof_end(&proof));   // construction cannot cross a consumer ordering boundary
    epoch = 1;
    vk_usnap_proof_begin(&proof);
    assert(vk_usnap_proof_add(&proof, &lease));
    epoch = 5;
    vk_usnap_drop((uintptr_t)guest + 300, 4);
    epoch = 1;
    assert(!vk_usnap_proof_end(&proof));   // construction cannot mix publication states
    vk_usnap_proof_begin(&proof);
    assert(vk_usnap_proof_add(&proof, &lease) && vk_usnap_proof_end(&proof));
    vk_usnap_end(1);
    assert(!vk_usnap_capture() && !vk_usnap_lease_valid(&lease));
    assert(!vk_usnap_proof_valid(&proof));

    // Reusing an arena and the exact same region index cannot revive an old lease. Indexed gathering must produce
    // the same capture as the raw command walk, including its bytes and alignment rules.
    completed = 1, indexed = 1, epoch = 1;
    memset(guest, 0x22, guest_size);
    vk_usnap_begin(&handle, 1, 2);
    assert(vk_usnap_capture() != capture && !vk_usnap_lease_valid(&lease));
    assert(!vk_usnap_proof_valid(&proof));
    assert(vk_usnap_find_lease((uintptr_t)guest, 1024, 256, &bi, &lease));
    assert(vk_usnap_bytes((uintptr_t)guest, 1)[0] == 0x22);
    for (uint64_t off = 0; off < 1400; off += 7) {
        for (uint64_t n = 0; n < 600; n += 13) {
            int expected = n && off + n <= 1024 && off % 16 == 0;
            VkUsnapLease probe;
            int got = vk_usnap_find_lease((uintptr_t)guest + off, n, 16, &bi, &probe);
            assert(got == expected && vk_usnap_lease_valid(&probe) == expected);
        }
    }
    VkUsnapLease bad;
    assert(!vk_usnap_find_lease(UINT64_MAX - 7, 32, 16, &bi, &bad));
    assert(!vk_usnap_find_lease((uintptr_t)guest, 16, 0, &bi, &bad));
    vk_usnap_end(2);

    // Async capture is taken before the game overwrites its uniform block and adopted later by the replay.
    completed = 2;
    vk_usnap_take(&handle, 1);
    memset(guest, 0x33, guest_size);
    vk_usnap_begin(&handle, 1, 3);
    assert(vk_usnap_bytes((uintptr_t)guest, 1)[0] == 0x22);
    assert(!vk_usnap_lease_valid(&lease));
    vk_usnap_end(3);

    // Reference mode retains ordinary drops and lookup validity without publishing aggregate proof metadata.
    completed = 3, preparation = 0;
    vk_usnap_begin(&handle, 1, 4);
    assert(vk_usnap_find_lease((uintptr_t)guest, 1024, 256, &bi, &lease));
    vk_usnap_proof_begin(&proof);
    assert(!vk_usnap_proof_add(&proof, &lease) && !vk_usnap_proof_end(&proof));
    uint64_t reference_serial = U.proof_serial, reference_gen = vk_usnap_gen();
    vk_usnap_drop((uintptr_t)guest, 4);
    assert(!vk_usnap_lease_valid(&lease) && vk_usnap_gen() != reference_gen && U.proof_serial == reference_serial);
    vk_usnap_end(4);

    // More regions than dependency buckets exercise a conservative collision: the original lease remains valid,
    // but its aggregate must fall back when a different region sharing the bucket changes.
    struct { op_hdr begin; struct { op_hdr bind; p_bind_buf buffer; } binds[65]; op_hdr end; } many = {
        .begin = {OP_BEGIN, sizeof(op_hdr)}, .end = {OP_END, sizeof(op_hdr)}};
    for (unsigned i = 0; i < 65; i++) {
        many.binds[i].bind = (op_hdr){OP_BIND_UBO, sizeof many.binds[i]};
        many.binds[i].buffer = (p_bind_buf){.stage = 0, .index = 3, .addr = (uintptr_t)guest + i * 768, .size = 128};
    }
    uint64_t many_handle = (uintptr_t)&many;
    completed = 4, preparation = 1, indexed = 0;
    vk_usnap_begin(&many_handle, 1, 5);
    assert(vk_usnap_find_lease((uintptr_t)guest, 128, 256, &bi, &lease));
    vk_usnap_proof_begin(&proof);
    assert(vk_usnap_proof_add(&proof, &lease) && vk_usnap_proof_end(&proof));
    vk_usnap_drop((uintptr_t)guest + 64 * 768, 4);
    assert(vk_usnap_lease_valid(&lease) && !vk_usnap_proof_valid(&proof));
    vk_usnap_proof_begin(&proof);
    assert(vk_usnap_proof_add(&proof, &lease) && vk_usnap_proof_end(&proof));
    assert(vk_usnap_proof_valid(&proof));
    vk_usnap_end(5);

    // Incremental renewal inherits unchanged members only inside a new publication bracket. Added members and
    // conservative old dependencies must both reject a later drop; no old proof is mutated during inheritance.
    completed = 5, epoch = 1;
    vk_usnap_begin(&many_handle, 1, 6);
    VkUsnapLease added;
    assert(vk_usnap_find_lease((uintptr_t)guest, 128, 256, &bi, &lease));
    assert(vk_usnap_find_lease((uintptr_t)guest + 768, 128, 256, &bi, &added));
    vk_usnap_proof_begin(&proof);
    assert(vk_usnap_proof_add(&proof, &lease) && vk_usnap_proof_end(&proof));
    VkUsnapProof renewal;
    vk_usnap_proof_begin(&renewal);
    assert(vk_usnap_proof_inherit(&renewal, &proof));
    assert(vk_usnap_proof_add(&renewal, &added) && vk_usnap_proof_end(&renewal));
    assert(renewal.buckets == (proof.buckets | (1ull << (added.region % VK_USNAP_PROOF_BUCKETS))));
    uint64_t old_serial = proof.serial;
    vk_usnap_proof_begin(&invalid_proof);
    vk_usnap_drop((uintptr_t)guest + 2 * 768, 4);
    assert(!vk_usnap_proof_inherit(&invalid_proof, &proof) && !vk_usnap_proof_end(&invalid_proof));
    vk_usnap_proof_begin(&invalid_proof);
    assert(vk_usnap_proof_inherit(&invalid_proof, &proof) && vk_usnap_proof_end(&invalid_proof));
    assert(proof.serial == old_serial);   // validating a private copy must not advance the donor
    assert(vk_usnap_proof_valid(&renewal));
    epoch = 3;
    vk_usnap_drop((uintptr_t)guest + 768, 4);
    epoch = 1;
    assert(!vk_usnap_proof_valid(&renewal) && vk_usnap_proof_valid(&proof));
    vk_usnap_proof_begin(&renewal);
    assert(vk_usnap_proof_inherit(&renewal, &proof));
    assert(vk_usnap_proof_add(&renewal, &added) && vk_usnap_proof_end(&renewal));
    assert(renewal.horizon == 3);
    vk_usnap_proof_begin(&invalid_proof);
    epoch = 2;
    assert(!vk_usnap_proof_inherit(&invalid_proof, &proof));
    epoch = 3;
    assert(!vk_usnap_proof_valid(&renewal) && vk_usnap_proof_valid(&proof));
    epoch = 1;
    vk_usnap_proof_begin(&invalid_proof);
    assert(vk_usnap_proof_inherit(&invalid_proof, &proof));
    vk_usnap_drop((uintptr_t)guest + 3 * 768, 4);
    assert(!vk_usnap_proof_end(&invalid_proof));
    epoch = 4;
    vk_usnap_drop((uintptr_t)guest, 4);
    epoch = 1;
    vk_usnap_proof_begin(&invalid_proof);
    assert(!vk_usnap_proof_inherit(&invalid_proof, &proof));
    vk_usnap_proof_begin(&proof);
    assert(vk_usnap_proof_add(&proof, &lease) && vk_usnap_proof_end(&proof));
    vk_usnap_proof_begin(&renewal);
    assert(vk_usnap_proof_inherit(&renewal, &proof));
    assert(vk_usnap_proof_add(&renewal, &added) && vk_usnap_proof_end(&renewal) && renewal.horizon == 3);
    vk_usnap_end(6);
    completed = 6;
    vk_usnap_begin(&many_handle, 1, 7);
    vk_usnap_proof_begin(&invalid_proof);
    assert(!vk_usnap_proof_inherit(&invalid_proof, &proof));
    vk_usnap_proof_begin(&invalid_proof);
    assert(!vk_usnap_proof_inherit(&invalid_proof, &invalid_proof));
    vk_usnap_end(7);
    vk_usnap_shutdown();
    free(allocation);
    puts("snapshot leases/proofs: intervals, ordered drops, publication, renewal, collisions, capture reuse and async adoption passed");
    return 0;
}
