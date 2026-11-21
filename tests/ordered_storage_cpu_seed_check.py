"""Exercise an ordered CPU seed's exact input and capacity ownership before image emission."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from present_source_check import ROOT, functions, section


def fixture(source):
    entry = section(source, 'static int write_target_capture(', 'static int write_target_current_locked(')
    assert 'else if (cpu_allowed)' in entry and 'target->cpu_seed = write_target_capture_cpu' in entry
    begin = section(source, 'const VkTexImage *vk_texture_write_target_begin(', 'static void explicit_written(')
    assert begin.index('write_target_cpu_record(') < begin.index('e->pub.image = target->image')
    assert begin.index('e->pub.image = target->image') < begin.index('write_target_cpu_destroy(')
    return r"""
#include <assert.h>
#include <stdio.h>
#include "nvn_api.h"
#include "resource_core_vma.h"
#include "texture_source.h"
typedef struct { uint64_t id; } TextureSourceAuthority;
typedef struct { uint32_t count; TextureSourceAuthority *observed; } TextureSourceMapping;
typedef struct TextureMaterial TextureMaterial;
typedef struct { void *material; } MockInstance;
typedef struct { MockInstance *instance; } MockUse;
typedef struct { VkBufferImageCopy *regions; uint8_t *scratch; size_t workspace, bytes; uint32_t count; } TextureRecoveryCopy;
typedef struct { int unused; } TexVkFormat;
typedef struct Tex {
    VkTexImage pub;
    tex_t t;
    TexVkFormat f;
    uint8_t *source_copy;
    uint64_t content_revision;
    int source_canonical, source_blocked, pin_owned, pin_detached;
    TextureSourceMapping *source_mapping;
    void *pending_write;
    MemBackingCapture *image_host_backing;
    TextureMaterial *material;
} Tex;
""" + section(source, 'typedef struct {\n    TextureMaterial *previous;', 'struct VkTextureWriteTarget {') + r"""
struct VkTextureWriteTarget {
    Tex *entry;
    uint64_t revision;
    struct {TextureMaterial *material;} *seed;
    MockUse source_use;
    VkImage selected;
    TextureWriteCpuSeed *cpu_seed;
    TextureWriteStorageSelection storage;
};
static int write_target_storage_current_locked(const VkTextureWriteTarget *target) { (void)target; abort(); }
static VkCtx context;
VkCtx *vk_ctx(void) { return &context; }
static RcAdmission ledger;
static bool mapping_current = true, copies_current = true, locked, borrowed, writable = true;
static unsigned copies, live_work, stage_released, stage_allocated, flushes;
static uint8_t exact_staging[256];
static RcRegistryPosition read = {7,12};
static TextureSourceObservation expected;
static void lock(void) { assert(!locked); locked = true; }
static void unlock(void) { assert(locked); locked = false; }
RcRegistryPosition vk_exec_resource_position(void) { return read; }
int mem_backing_current(const MemBackingCapture *backing) { return mapping_current && backing; }
int mem_mapping_range_unchanged(uint64_t addr,uint64_t size,uint64_t stamp) { return addr && size && stamp == expected.stamp.mapping; }
int mem_protection_range_unchanged(uint64_t addr,uint64_t size,uint64_t stamp) { return addr && size && stamp == expected.stamp.protection; }
static int source_private_sample(const Tex *e) { return e->pub.mirrored && !e->pub.gpu_owned; }
static int source_observe(Tex *e, const TextureSourceStamp *old, TextureSourceObservation *out) {
    assert(locked && e && !old);
    *out = expected;
    return TEX_SOURCE_DIRTY;
}
static int texture_authority_same(const TextureSourceAuthority *a, const TextureSourceAuthority *b) {
    return a && b && a->id == b->id;
}
static int source_snapshot_hold(Tex *e) {
    assert(locked && e->source_mapping && !borrowed);
    borrowed = true;
    return 1;
}
void mem_read_release(void) { assert(locked && borrowed); borrowed = false; }
static int source_snapshot_verify(Tex *e, const TextureSourceStamp *before, const TextureSourceAuthority *proof) {
    assert(locked && !borrowed && e && proof && before->generation == expected.stamp.generation);
    return copies_current;
}
static int recovery_copy_layout(const Tex *e, TextureRecoveryCopy *copy) {
    assert(e && copy);
    copy->count = 1;
    copy->bytes = 128;
    copy->workspace = sizeof(VkBufferImageCopy);
    if (copy->regions) copy->regions[0].bufferOffset = 0;
    return writable;
}
static int recovery_copy_fill(const Tex *recipe,const TextureRecoveryCopy *copy,uint8_t *output) {
    assert(locked && borrowed && copy->count == 1 && output == exact_staging);
    memcpy(output,recipe->source_copy,copy->bytes);
    ++copies;
    return 1;
}
VkResult vk_staging_prepare_plan(VkDeviceSize bytes, VkPreparedStaging *stage, RcAdmissionVector *peak) {
    assert(bytes == 128 && !stage->buffer);
    *peak=(RcAdmissionVector){0};
    peak->amount[RC_BUDGET_STAGING] = bytes;
    return VK_SUCCESS;
}
VkResult vk_staging_prepare_allocate(VkPreparedStaging *stage, RcBudgetTicket *peak, int required) {
    assert(peak->token.ledger == &ledger && required && !stage->buffer);
    stage->map = exact_staging;
    stage->buffer = (VkBuffer)(uintptr_t)9;
    stage->allocation = (VmaAllocation)(uintptr_t)10;
    ++stage_allocated;
    return VK_SUCCESS;
}
void vk_staging_prepare_cancel(VkPreparedStaging *stage) {
    if (stage->buffer) ++stage_released;
    *stage=(VkPreparedStaging){0};
}
RcAdmissionStatus resource_core_budget_split(RcBudgetTicket *ticket, const RcAdmissionVector *cost,
    RcBudgetProducer producer, RcBudgetTicket *out) {
    assert(ticket->token.ledger == &ledger && cost->amount[RC_BUDGET_RAM] == sizeof(VkBufferImageCopy));
    assert(cost->amount[RC_BUDGET_COMMIT] == sizeof(VkBufferImageCopy) && producer == RC_BUDGET_TEXTURE);
    out->token.ledger = &ledger;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *ticket, const RcAdmissionVector *actual) {
    assert(ticket->token.ledger == &ledger && actual->amount[RC_BUDGET_RAM]);
    ++live_work;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *ticket) {
    assert(ticket->token.ledger == &ledger && live_work == 1); return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *ticket) {
    assert(ticket->token.ledger == &ledger && live_work == 1);
    --live_work;
    ticket->token.ledger = NULL; return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) {
    assert(ticket->token.ledger == &ledger && !live_work);
    ticket->token.ledger = NULL; return RC_ADMISSION_OK;
}
VkResult vmaFlushAllocation(VmaAllocator vma,VmaAllocation alloc,VkDeviceSize begin,VkDeviceSize bytes) {
    assert(vma == context.vma && alloc == (VmaAllocation)(uintptr_t)10 && !begin && bytes == 128 && !locked);
    ++flushes;
    return VK_SUCCESS;
}
""" + functions(source, 'source_stamp_equal', 'restore_peak_add', 'write_target_capture_cpu',
                'write_target_cpu_current', 'write_target_current_locked', 'write_target_cpu_peak', 'write_target_cpu_stage',
                'write_target_cpu_rollback', 'write_target_cpu_destroy') + r"""
int main(void) {
    uint8_t cpu_bytes[128];
    for (unsigned i = 0; i < sizeof cpu_bytes; ++i) cpu_bytes[i] = (uint8_t)(3*i + 1);
    expected.stamp = (TextureSourceStamp){.mapping=1,.protection=2,.lineage=3,.generation=4};
    TextureSourceAuthority extents[] = {{1},{2}};
    TextureSourceMapping mapping = {.count=2,.observed=extents};
    Tex original = {.pub={.image=(VkImage)(uintptr_t)33,.addr=0x1000,.mirrored=1},
        .t={.mem=cpu_bytes,.L.size=128}, .source_canonical=1,.pin_owned=1,
        .source_mapping=&mapping, .image_host_backing=(MemBackingCapture *)&original, .content_revision=5};
    assert(!write_target_capture_cpu(&original,&(TextureSourceObservation){0}));
    original.source_blocked=1;
    assert(!write_target_capture_cpu(&original,&expected));
    original.source_blocked=0;
    original.pub.gpu_owned=1;
    assert(!write_target_capture_cpu(&original,&expected));
    original.pub.gpu_owned=0;
    VkTextureWriteTarget target = {.entry=&original,.revision=5,
        .cpu_seed=write_target_capture_cpu(&original,&expected)};
    assert(target.cpu_seed);
    original.pending_write=&target;
    RcAdmissionVector peak={0};
    peak.amount[RC_BUDGET_GPU_FIRST]=8192;
    assert(write_target_cpu_peak(target.cpu_seed,&peak)==VK_SUCCESS);
    assert(peak.amount[RC_BUDGET_GPU_FIRST]==8192 && peak.amount[RC_BUDGET_STAGING]==128 &&
        peak.amount[RC_BUDGET_RAM]==sizeof(VkBufferImageCopy) && peak.amount[RC_BUDGET_COMMIT]==sizeof(VkBufferImageCopy));
    RcBudgetTicket ticket={.token.ledger=&ledger};
    ++read.ordinal;
    assert(write_target_cpu_stage(&target,&ticket)==VK_ERROR_INITIALIZATION_FAILED && !copies);
    --read.ordinal;
    write_target_cpu_rollback(target.cpu_seed);
    assert(!live_work && stage_released==1);
    ++expected.stamp.generation;
    assert(write_target_cpu_stage(&target,&ticket)==VK_ERROR_INITIALIZATION_FAILED && !copies);
    --expected.stamp.generation;
    write_target_cpu_rollback(target.cpu_seed);
    extents[1].id = 9; // First normalized extent is unchanged.
    assert(write_target_cpu_stage(&target,&ticket)==VK_ERROR_INITIALIZATION_FAILED && !copies);
    write_target_cpu_rollback(target.cpu_seed);
    extents[1].id = 2;
    copies_current=false;
    assert(write_target_cpu_stage(&target,&ticket)==VK_ERROR_INITIALIZATION_FAILED && copies==1 && !borrowed);
    write_target_cpu_rollback(target.cpu_seed);
    copies_current=true;
    assert(write_target_cpu_stage(&target,&ticket)==VK_SUCCESS && copies==2 && flushes==1);
    assert(!memcmp(exact_staging,cpu_bytes,sizeof cpu_bytes));
    assert(original.pub.image==(VkImage)(uintptr_t)33 && !original.pub.gpu_owned && original.content_revision==5);
    write_target_cpu_destroy(&target.cpu_seed);
    assert(!target.cpu_seed && !live_work && stage_released==stage_allocated);
    puts("Ordered CPU seed: shared peak, stale read/version/authority refusal, post-copy proof and cancellation passed");
}
"""


def main():
    cc = Path(os.environ.get('CC') or shutil.which('clang') or '').resolve()
    vma = Path(os.environ.get('VMA_INCLUDE', cc.parent.parent / 'include' / 'vma'))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get('PATH', ''))
    runtime = Path(env.get('STORAGE_CPU_SEED_SOURCE', ROOT / 'runtime'))
    with tempfile.TemporaryDirectory(prefix='storage-seed-') as temp:
        source = Path(temp) / 'check.c'
        executable = Path(temp) / ('check.exe' if os.name == 'nt' else 'check')
        source.write_text(fixture((runtime / 'vk_texture.c').read_text()))
        subprocess.run([str(cc), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=undefined', '-fsanitize-trap=undefined',
                        '-I' + str(ROOT / 'runtime'), '-I' + str(vma), str(source), '-o', str(executable)],
                       env=env, check=True)
        subprocess.run([str(executable)], env=env, check=True)


if __name__ == '__main__':
    main()
