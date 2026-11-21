"""CPU-visible physical backing excludes every overlapping ordinary-source registration."""
from pathlib import Path
import os
import subprocess
import tempfile
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = (ROOT / "runtime/vk_memory.c").read_text()
host_write = (ROOT / "runtime/resource_host_write.c").read_text()
host_header = (ROOT / "runtime/resource_host_write.h").read_text()
fixture = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "resource_core_registry.h"
static RcRegistry *registry;
static RcRegistry *fixture_registry(void) { return registry; }
#define rc_registry_runtime fixture_registry
enum {POOL_IMPORT=1,POOL_PHYS,POOL_VIRT};
static int nvn_gpu_only(uint64_t a) {return a>=0x100000;}
static uint32_t resource_host_current_intervals(uint64_t address, uint64_t size, RcInterval **out) {
    assert(address == 0x1000 && size == 0x2000);
    *out = malloc(sizeof **out);
    assert(*out);
    **out = (RcInterval){address, 9, {37, 0, size}};
    return 1;
}
'''
fixture += host_header[host_header.index("typedef struct {"):host_header.index("} ResourceHostWrite;") + len("} ResourceHostWrite;")]
fixture += "\ntypedef struct {int kind;uint64_t lo,size;RcRegistryWrite physical_exclusion;ResourceHostWrite physical_host_authority;} PoolRec;\n"
fixture += "\n".join(function(host_write, name) for name in (
    "resource_host_write_prepare", "complete_guard", "complete_tickets", "resource_host_write_dispose"))
fixture += "\n".join(function(source, name) for name in ("pool_authority_begin", "pool_authority_end"))
fixture += r'''
int main(void) {
    registry=rc_registry_create((RcRegistryConfig){16,16,4});assert(registry);
    PoolRec p={.kind=POOL_PHYS,.lo=0x1000,.size=0x2000};
    pool_authority_begin(&p);assert(p.physical_exclusion.id && p.physical_host_authority.tickets);
    RcRegistryHandle alias={0},canonical={0},outside={0};RcRegistryObservation o;
    assert(rc_registry_source(registry,1,(RcSourceRange){1,0x1800,32},&alias)==RC_REGISTRY_OK);
    assert(rc_registry_source(registry,2,(RcSourceRange){1,0x8000,32},&outside)==RC_REGISTRY_OK);
    assert(rc_registry_source(registry,3,(RcSourceRange){37,0x800,32},&canonical)==RC_REGISTRY_OK);
    assert(rc_registry_observe(registry,alias,1,&o)!=RC_REGISTRY_OK);
    assert(rc_registry_observe(registry,canonical,1,&o)!=RC_REGISTRY_OK);
    assert(rc_registry_observe(registry,outside,1,&o)==RC_REGISTRY_OK);
    pool_authority_end(&p);assert(!p.physical_exclusion.id && !p.physical_host_authority.tickets);
    assert(rc_registry_observe(registry,alias,1,&o)!=RC_REGISTRY_OK); // Retirement alone is not CPU visibility.
    assert(rc_registry_reconcile_cpu(registry,(RcSourceRange){1,0x1000,0x2000})==RC_REGISTRY_OK);
    assert(rc_registry_observe(registry,alias,1,&o)==RC_REGISTRY_OK);
    assert(rc_registry_observe(registry,canonical,1,&o)!=RC_REGISTRY_OK);
    assert(rc_registry_reconcile_cpu(registry,(RcSourceRange){37,0,0x2000})==RC_REGISTRY_OK);
    assert(rc_registry_observe(registry,canonical,1,&o)==RC_REGISTRY_OK);
    p=(PoolRec){.kind=POOL_IMPORT,.lo=0x1000,.size=0x2000};pool_authority_begin(&p);assert(!p.physical_exclusion.id);
    p=(PoolRec){.kind=POOL_PHYS,.lo=0x100000,.size=0x2000};pool_authority_begin(&p);assert(!p.physical_exclusion.id);
    rc_registry_destroy(registry);
    puts("physical backing exclusion: overlapping later registration blocked; unrelated ordinary sources eligible; retirement remains foreign");
}
'''
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="physical-exclusion-", dir=ROOT / "build/test") as tmp:
    path = Path(tmp) / "pool.c"
    path.write_text(fixture)
    exe = path.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-DRC_REGISTRY_SINGLE_THREADED",
                    "-I", str(ROOT / "runtime"), str(path), str(ROOT / "runtime/resource_core_registry.c"),
                    "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)
