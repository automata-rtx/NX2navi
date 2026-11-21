"""Exercise the actual import capability seam without granting registry permission."""
from pathlib import Path
import os
import subprocess
import tempfile
from pass_end_lock_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
SOURCE = (ROOT / "runtime/vk_memory.c").read_text()
HEADER = (ROOT / "runtime/resource_host_write.h").read_text()
DECLS = HEADER[HEADER.index("typedef enum VkMemCanonicalSourceQualificationStage {"):
               HEADER.index("// Qualifies an ordinary tracked import")]
PREFIX = r"""
#include <assert.h>
#include <stdio.h>
#include "resource_core_registry.h"
typedef struct { const RcInterval *intervals; uint32_t count; int current; } MemBackingCapture;
enum {POOL_IMPORT=1, POOL_PHYS, POOL_VIRT};
typedef struct {
    int kind, unpublished, buf;
    const MemBackingCapture *host_backing;
    uint64_t import_lo, import_hi;
    uint64_t lo, size, publication, host_mapping_cutoff;
} PoolRec;
typedef struct { PoolRec *buf; } pidx_ent;
static pidx_ent selected;
static int pool_index, pool_lock, change_after_find;
static MemBackingCapture *changed;
static const RcInterval *mem_backing_intervals(const MemBackingCapture *capture, uint32_t *count) {
    *count=capture ? capture->count : 0;
    return capture ? capture->intervals : NULL;
}
static int mem_backing_current(const MemBackingCapture *capture) { assert(!pool_lock);return capture->current; }
static void plat_mutex_lock_shared(int *lock) { assert(!*lock);*lock=1; }
static void plat_mutex_unlock_shared(int *lock) { assert(*lock);*lock=0; }
static int nvn_gpu_only(uint64_t addr) { return addr>=0x10000; }
static const pidx_ent *pidx_find(int *index, uint64_t address, uint64_t size) {
    assert(index==&pool_index && pool_lock && address && size);
    if (change_after_find) changed->current=0;
    return selected.buf?&selected:0;
}
"""
CASES = r"""
int main(void) {
    RcInterval imported[]={{0x500,6,{2,0,0x100}},{0x600,7,{3,0,0x180}}};
    RcInterval subset[]={{0x570,6,{2,0x70,0x90}},{0x600,7,{3,0,0x80}},{0x680,7,{3,0x80,0x40}}};
    MemBackingCapture provider={imported,2,1}, source={subset,3,1};
    PoolRec primary={POOL_IMPORT,0,1,&provider,0x500,0x780,0x500,0x280,1,4};
    PoolRec alias={POOL_IMPORT,0,1,&provider,0x540,0x720,0x540,0x1e0,2,5};
    VkMemCanonicalSourceQualification report;
    assert(vk_mem_canonical_source_qualify_report(&source,0x570,0x150,&report)==RC_REGISTRY_UNSUPPORTED);
    assert(report.stage==VK_MEM_CANONICAL_SOURCE_PROVIDER_MISSING && report.source_count==3);
    selected.buf=&primary;
    assert(vk_mem_canonical_source_qualify(&source,0x570,0x150)==RC_REGISTRY_OK);
    selected.buf=&alias;
    assert(vk_mem_canonical_source_qualify(&source,0x570,0x150)==RC_REGISTRY_OK);
    assert(vk_mem_canonical_source_qualify(&source,0x570,0x151)==RC_REGISTRY_STALE);
    subset[2].mapping++;
    assert(vk_mem_canonical_source_qualify_report(&source,0x570,0x150,&report)==RC_REGISTRY_STALE);
    assert(report.stage==VK_MEM_CANONICAL_SOURCE_VECTOR && report.mismatch_address==0x680 &&
           report.source_interval.mapping==8 && report.provider_interval.mapping==7 &&
           report.source_count==3 && report.provider_count==2 &&
           report.provider_publication==2 && report.provider_mapping_cutoff==5);
    subset[2].mapping--;
    alias.unpublished=1;
    assert(vk_mem_canonical_source_qualify(&source,0x570,0x150)==RC_REGISTRY_UNSUPPORTED);
    alias.unpublished=0;
    alias.kind=POOL_PHYS;
    assert(vk_mem_canonical_source_qualify(&source,0x570,0x150)==RC_REGISTRY_UNSUPPORTED);
    alias.kind=POOL_IMPORT;
    changed=&source;change_after_find=1;
    assert(vk_mem_canonical_source_qualify_report(&source,0x570,0x150,&report)==RC_REGISTRY_STALE);
    assert(report.stage==VK_MEM_CANONICAL_SOURCE_LEASE_AFTER && report.provider_publication==2);
    assert(vk_mem_canonical_source_qualify_report(&source,0x570,0x150,&report)==RC_REGISTRY_STALE);
    assert(report.stage==VK_MEM_CANONICAL_SOURCE_LEASE_BEFORE && !report.provider_publication);
    source.current=1;change_after_find=0;
    RcRegistry *r=rc_registry_create((RcRegistryConfig){8,4,2});assert(r);
    RcRegistryHandle source_record;
    assert(rc_registry_source(r,1,subset[1].range,&source_record)==RC_REGISTRY_OK);
    RcRegistryWrite write;
    assert(rc_registry_write_begin(r,(RcSourceRange){3,0,80},RC_REGISTRY_BUFFER,&write)==RC_REGISTRY_OK);
    assert(vk_mem_canonical_source_qualify(&source,0x570,0x150)==RC_REGISTRY_OK);
    RcRegistryObservation proof;
    assert(rc_registry_observe(r,source_record,1,&proof)==RC_REGISTRY_BLOCKED);
    assert(rc_registry_write_cancel(r,write)==RC_REGISTRY_OK);
    assert(rc_registry_write_guard_begin_reason_at(r,RC_REGISTRY_UNKNOWN,(RcRegistryPosition){0},
        RC_REGISTRY_GUARD_COMMAND_OPAQUE,&write)==RC_REGISTRY_OK);
    assert(vk_mem_canonical_source_qualify(&source,0x570,0x150)==RC_REGISTRY_OK);
    assert(rc_registry_observe(r,source_record,1,&proof)==RC_REGISTRY_BLOCKED);
    assert(rc_registry_write_guard_accumulate(r,write)==RC_REGISTRY_OK);
    assert(rc_registry_observe(r,source_record,1,&proof)==RC_REGISTRY_BLOCKED);
    rc_registry_destroy(r);
    puts("ordinary source qualification: exact import and alias coverage, remap/churn, typed foreign and opaque exclusion passed");
}
"""
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
work = ROOT / "build/test"
work.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="import-qualification-", dir=work) as temporary:
    case = Path(temporary) / "check.c"
    case.write_text(PREFIX + "\n" + DECLS + "\n" + function(SOURCE, "host_interval_vectors_match") + "\n" +
                    function(SOURCE, "vk_mem_canonical_source_qualify_report") + "\n" +
                    function(SOURCE, "vk_mem_canonical_source_qualify") + "\n" + CASES)
    exe = case.with_suffix(".exe")
    subprocess.run([str(CC), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-fsanitize-trap=all", "-DRC_REGISTRY_SINGLE_THREADED", "-I", str(ROOT/"runtime"),
                    str(case), str(ROOT/"runtime/resource_core_registry.c"), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)
