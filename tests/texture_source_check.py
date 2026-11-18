"""Exercise production source observations, subresource masks and generation discontinuities."""
from pathlib import Path
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/test/texture-source'
CC = Path(os.environ.get('CC', r'C:\msys64\ucrt64\bin\clang.exe'))
if not CC.is_file():
    CC = Path(r'C:\Users\Andrew\AppData\Local\Programs\2NX2Navi\toolchain\msys64\ucrt64\bin\clang.exe')
sys.path.insert(0, str(ROOT / 'tools'))
from check_content import GENERATED_MARKER

SOURCE = (ROOT / 'runtime/vk_memory.c').read_text()


def function(name):
    match = re.search(r'^(?:static )?[^\n]*\b' + name + r'\([^;]*?\)\s*\{', SOURCE, re.M)
    if not match:
        raise RuntimeError('Missing production function ' + name)
    start = SOURCE.index('{', match.start())
    depth, end = 1, start + 1
    while depth:
        depth += (SOURCE[end] == '{') - (SOURCE[end] == '}')
        end += 1
    return SOURCE[match.start():end] + '\n'


PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "texture_source.h"
#define RT_ADDRESS_LIMIT (128ull << 20)
#define SNAP_SHIFT 21
#define REG_SHIFT 21
#define GEN_PAGE_SHIFT 12
#define SNAP_REGIONS (RT_ADDRESS_LIMIT >> SNAP_SHIFT)
static uint32_t gen_now;
static uint64_t source_generation_lineage = 1;
static int gen_lock, locked;
static uint32_t reg_max[SNAP_REGIONS], snap_at[SNAP_REGIONS];
static uint8_t snap_untracked[SNAP_REGIONS];
static uint64_t snap_protection[SNAP_REGIONS], snap_mapping[SNAP_REGIONS];
static uint64_t map_region[SNAP_REGIONS], protect_region[SNAP_REGIONS];
static uint32_t pages[RT_ADDRESS_LIMIT >> GEN_PAGE_SHIFT];
static uint64_t map_serial, protect_serial, n_snap_calls, n_snap_checks, n_snap_untracked;
static uint64_t page_reads, os_queries;
static int query_untracked, map_check_count, mutate_at_map_check;
static void plat_mutex_lock(int *p) { (void)p; assert(!locked); locked = 1; }
static void plat_mutex_unlock(int *p) { (void)p; assert(locked==1); locked = 0; }
static void plat_mutex_lock_shared(int *p) { (void)p; assert(!locked); locked = 2; }
static void plat_mutex_unlock_shared(int *p) { (void)p; assert(locked==2); locked = 0; }
static uint64_t mem_mapping_serial(void) { return map_serial; }
static uint64_t mem_protection_serial(void) { return protect_serial; }
static uint64_t mem_mapping_region(uint32_t r) { return map_region[r]; }
static uint64_t mem_protection_region(uint32_t r) { return protect_region[r]; }
static int unchanged(uint64_t addr, uint64_t size, uint64_t observed, uint64_t serial, uint64_t *regions) {
    if (!observed || !serial || observed > serial || !size || addr >= RT_ADDRESS_LIMIT || size > RT_ADDRESS_LIMIT-addr) return 0;
    for (uint64_t r=addr>>REG_SHIFT;r<=(addr+size-1)>>REG_SHIFT;r++) if (!regions[r] || regions[r]>observed) return 0;
    return 1;
}
static int mem_mapping_range_unchanged(uint64_t addr, uint64_t size, uint64_t observed) {
    if (++map_check_count == mutate_at_map_check) map_region[addr>>REG_SHIFT] = ++map_serial;
    return unchanged(addr,size,observed,map_serial,map_region);
}
static int mem_protection_range_unchanged(uint64_t addr, uint64_t size, uint64_t observed) {
    return unchanged(addr,size,observed,protect_serial,protect_region);
}
static uint32_t *gen_slot(uint64_t addr, int create) {
    assert(locked);assert(!create);page_reads++;
    assert(addr < RT_ADDRESS_LIMIT);return &pages[addr>>GEN_PAGE_SHIFT];
}
'''

SNAP = r'''
static uint64_t snap_regions(uint64_t first, uint64_t last, uint32_t epoch) {
    assert(locked);
    uint64_t calls=0;
    for (uint64_t r=first;r<=last;r++) {
        if (snap_current((uint32_t)r,epoch)) continue;
        generation_next();calls++;os_queries++;
        snap_at[r]=epoch;snap_untracked[r]=(uint8_t)query_untracked;
        snap_mapping[r]=map_serial;snap_protection[r]=protect_serial;
    }
    return calls;
}
static void reset(void) {
    memset(reg_max,0,sizeof reg_max);memset(snap_at,0,sizeof snap_at);memset(pages,0,sizeof pages);
    memset(snap_untracked,0,sizeof snap_untracked);memset(snap_mapping,0,sizeof snap_mapping);memset(snap_protection,0,sizeof snap_protection);
    gen_now=0;source_generation_lineage=1;map_serial=protect_serial=2;locked=0;
    n_snap_calls=n_snap_checks=n_snap_untracked=page_reads=os_queries=0;
    query_untracked=map_check_count=mutate_at_map_check=0;
    for(unsigned r=0;r<SNAP_REGIONS;r++) map_region[r]=protect_region[r]=2;
}
static void write_page(uint64_t addr) {
    uint32_t g=generation_next();pages[addr>>GEN_PAGE_SHIFT]=g;reg_max[addr>>REG_SHIFT]=g;
}
static void assert_all(const TextureSourceObservation *o, unsigned count) {
    for(unsigned i=0;i<count;i++) assert(texture_source_span_dirty(o,i));
    for(unsigned i=count;i<TEXTURE_SOURCE_MAX_SPANS;i++) assert(!texture_source_span_dirty(o,i));
}
'''

TESTS = r'''
int main(void) {
    const uint64_t addr=4ull<<20, span=2ull<<20, size=8*span;
    TextureSourceSpan spans[8];for(unsigned i=0;i<8;i++)spans[i]=(TextureSourceSpan){i*span,span};
    TextureSourceObservation o;
    reset();assert(vk_mem_observe_subresources(addr,size,NULL,1,spans,8,&o)==TEX_SOURCE_UNKNOWN);
    assert(texture_source_stamp_valid(&o.stamp));assert_all(&o,8);assert(n_snap_calls==8);TextureSourceStamp resident=o.stamp;
    uint64_t queries=os_queries;
    assert(vk_mem_observe_subresources(addr,size,&resident,1,spans,8,&o)==TEX_SOURCE_CLEAN);
    assert(!o.dirty[0]&&!page_reads&&os_queries==queries);
    // Distant writes dirty exactly their covered layers, not the bounding envelope between them.
    write_page(addr+span+4096);write_page(addr+7*span+4096);
    assert(vk_mem_observe_subresources(addr,size,&resident,1,spans,8,&o)==TEX_SOURCE_DIRTY);
    assert(o.dirty[0]==((1ull<<1)|(1ull<<7))&&!o.dirty[1]);
    assert(page_reads<=2*(span>>GEN_PAGE_SHIFT));
    resident=o.stamp;
    assert(vk_mem_observe_subresources(addr,size,&resident,1,spans,8,&o)==TEX_SOURCE_CLEAN);
    // A write after the pre-copy observation is still dirty when that old stamp is committed.
    TextureSourceStamp before_copy=o.stamp;write_page(addr+3*span);
    assert(vk_mem_observe_subresources(addr,size,&before_copy,1,spans,8,&o)==TEX_SOURCE_DIRTY);
    assert(o.dirty[0]==(1ull<<3));resident=o.stamp;
    // A same-epoch remap re-queries the old snapshot lease and forbids partial retention.
    map_region[addr>>REG_SHIFT]=++map_serial;queries=os_queries;
    assert(vk_mem_observe_subresources(addr,size,&resident,1,spans,8,&o)==TEX_SOURCE_UNKNOWN);
    assert_all(&o,8);assert(texture_source_stamp_valid(&o.stamp)&&os_queries>queries);resident=o.stamp;
    protect_region[addr>>REG_SHIFT]=++protect_serial;
    assert(vk_mem_observe_subresources(addr,size,&resident,1,spans,8,&o)==TEX_SOURCE_UNKNOWN);
    assert_all(&o,8);resident=o.stamp;
    // Stable mapping is insufficient when the write-watch query was untracked.
    query_untracked=1;
    assert(vk_mem_observe_subresources(addr,size,&resident,2,spans,8,&o)==TEX_SOURCE_UNKNOWN);
    assert_all(&o,8);assert(!texture_source_stamp_valid(&o.stamp));query_untracked=0;
    assert(vk_mem_observe_subresources(addr,size,NULL,3,spans,8,&o)==TEX_SOURCE_UNKNOWN);resident=o.stamp;
    assert(texture_source_stamp_valid(&resident));
    // Busy/unavailable mapping facts never grant a certificate.
    map_region[addr>>REG_SHIFT]=0;
    assert(vk_mem_observe_subresources(addr,size,&resident,3,spans,8,&o)==TEX_SOURCE_UNKNOWN);
    assert(!texture_source_stamp_valid(&o.stamp));map_region[addr>>REG_SHIFT]=map_serial;
    // A mapping mutation during the final validation clears the otherwise-current stamp.
    map_check_count=0;mutate_at_map_check=3;
    assert(vk_mem_observe_subresources(addr,size,&resident,4,spans,8,&o)==TEX_SOURCE_UNKNOWN);
    assert_all(&o,8);assert(!texture_source_stamp_valid(&o.stamp));mutate_at_map_check=0;
    // Page-generation wrap has a separate lineage. An old small generation cannot revive after a complete cycle.
    reset();assert(vk_mem_observe_subresources(addr,size,NULL,1,spans,8,&o)==TEX_SOURCE_UNKNOWN);resident=o.stamp;
    gen_now=UINT32_MAX;assert(generation_next()==0&&source_generation_lineage==2);
    assert(vk_mem_observe_subresources(addr,size,&resident,2,spans,8,&o)==TEX_SOURCE_UNKNOWN);
    assert_all(&o,8);assert(o.stamp.lineage==2&&texture_source_stamp_valid(&o.stamp));resident=o.stamp;
    assert(vk_mem_observe_subresources(addr,size,&resident,2,spans,8,&o)==TEX_SOURCE_CLEAN);
    gen_now=UINT32_MAX;source_generation_lineage=UINT64_MAX;generation_next();assert(!source_generation_lineage);
    assert(vk_mem_observe_subresources(addr,size,&resident,3,spans,8,&o)==TEX_SOURCE_UNKNOWN);
    assert(!texture_source_stamp_valid(&o.stamp));
    // Extent/count overflow fails closed without querying storage or writing beyond the bounded mask.
    reset();TextureSourceSpan bad={UINT64_MAX,4096};
    assert(vk_mem_observe_subresources(UINT64_MAX-7,4096,NULL,1,spans,8,&o)==TEX_SOURCE_UNKNOWN);
    assert(vk_mem_observe_subresources(addr,size,NULL,1,&bad,1,&o)==TEX_SOURCE_UNKNOWN);
    assert(vk_mem_observe_subresources(addr,size,NULL,1,spans,257,&o)==TEX_SOURCE_UNKNOWN);
    assert_all(&o,256);assert(!os_queries);
    // Adjacent subresources sharing a page both invalidate; a layout gap cannot be declared clean.
    TextureSourceSpan small[2]={{0,2048},{2048,2048}};
    assert(vk_mem_observe_subresources(addr,8192,NULL,1,small,2,&o)==TEX_SOURCE_UNKNOWN);resident=o.stamp;
    write_page(addr);
    assert(vk_mem_observe_subresources(addr,8192,&resident,1,small,2,&o)==TEX_SOURCE_DIRTY);assert_all(&o,2);resident=o.stamp;
    write_page(addr+4096);
    assert(vk_mem_observe_subresources(addr,8192,&resident,1,small,2,&o)==TEX_SOURCE_DIRTY);assert_all(&o,2);
    puts("texture source observations: generations, sparse masks, remaps, unknown tracking and wrap passed");return 0;
}
'''

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
    fixture = PREFIX + function('generation_next') + function('snap_lease_valid') + function('snap_current') + SNAP
    fixture += function('vk_mem_observe_subresources') + TESTS
    source = OUT / 'source.c'
    source.write_text('// ' + marker + '\n' + fixture)
    exe = OUT / 'source.exe'
    env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get('PATH', ''))
    subprocess.run([str(CC), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined', '-fsanitize-trap=all',
                    '-I', str(ROOT / 'runtime'), str(source), '-o', str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)
