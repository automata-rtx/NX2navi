"""Exercise the production texture reclaim policy with bounded synthetic lifecycles; no game or GPU."""
from pathlib import Path
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/test/texture-reclaim-policy'
CC = Path(os.environ.get('CC', r'C:\msys64\ucrt64\bin\clang.exe'))
if not CC.is_file():
    CC = Path(r'C:\Users\Andrew\AppData\Local\Programs\2NX2Navi\toolchain\msys64\ucrt64\bin\clang.exe')
sys.path.insert(0, str(ROOT / 'tools'))
from check_content import GENERATED_MARKER


def function(source, name):
    match = re.search(r'^static [^\n]*\b' + re.escape(name) + r'\([^;{}]*\)\s*\{', source, re.M)
    if not match:
        raise RuntimeError('Missing production function: ' + name)
    start = match.start()
    opening = source.index('{', match.start())
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end] + '\n'


BUDGET = r'''
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "resource_budget.h"
static void near(double a, double b) { if(fabs(a-b) >= 0.000001) fprintf(stderr, "actual %.12f expected %.12f\n", a, b); assert(fabs(a-b) < 0.000001); }
int main(void) {
    int pressure = 0;
    double floor = resource_commit_headroom_mb(58517, -1);
    double recover = resource_commit_recovery_mb(floor);
    near(floor, 5851.7); near(recover, floor + floor/32);
    // A small crossing asks for a recovery band, not eviction down to a fixed tiny retained-cache size.
    double deficit = resource_commit_deficit_mb(&pressure, 58517-floor+0.25, 58517, floor);
    assert(pressure); near(deficit, recover-floor+0.25); assert(deficit < 184);
    // Repeated readings around the entry line stay under pressure, with a continuously sized request.
    for (int i=0;i<1000;i++) {
        double free_mb = floor + ((i&1) ? 2 : -2);
        near(resource_commit_deficit_mb(&pressure, 58517-free_mb, 58517, floor), recover-free_mb);
        assert(pressure);
    }
    near(resource_commit_deficit_mb(&pressure, 58517-recover, 58517, floor), 0); assert(!pressure);
    near(resource_commit_deficit_mb(&pressure, 61440, 65536, 4096), 0); assert(!pressure);
    // Headroom growth is bounded even on a very large commit limit. Critical deficits remain explicit.
    near(resource_commit_recovery_mb(16384), 16640);
    near(resource_commit_deficit_mb(&pressure, 58517-100, 58517, floor), recover-100); assert(pressure);
    near(resource_build_headroom_mb(58517, 4096, -1), recover);
    near(resource_build_headroom_mb(58517, 9000, -1), 9000);
    near(resource_build_headroom_mb(58517, 0, -1), 0);
    // Disabled or unknown inputs clear pressure instead of preserving an obsolete eviction request.
    near(resource_commit_deficit_mb(&pressure, 58000, 58517, 0), 0); assert(!pressure);
    pressure=1; near(resource_commit_deficit_mb(&pressure, -1, 58517, floor), 0); assert(!pressure);
    pressure=1; near(resource_commit_deficit_mb(&pressure, 58000, -1, floor), 0); assert(!pressure);
    pressure=1; near(resource_commit_deficit_mb(&pressure, 58000, 0, floor), 0); assert(!pressure);
    puts("texture reclamation commit targets: passed");
    return 0;
}
'''


def run(name, fixture):
    OUT.mkdir(parents=True, exist_ok=True)
    marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
    source = OUT / (name + '.c')
    source.write_text('// ' + marker + '\n' + fixture)
    exe = OUT / (name + '.exe')
    env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get('PATH', ''))
    subprocess.run([str(CC), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=undefined', '-fsanitize-trap=all', '-I', str(ROOT/'runtime'),
                    str(source), '-o', str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)




def lifecycle_fixture():
    source = (ROOT/'runtime/vk_texture.c').read_text()
    names = ('PRESENTS', 'VRAM_PRESSURE_DEFAULT', 'PRESSURE_STEP', 'IDLE_ROUNDS',
             'STALE_FRAMES', 'GPU_STALE_FRAMES', 'GAP_TIMES', 'OVERWRITTEN_KEEP', 'RECLAIM_LOOK', 'EVICT_STEP', 'ROOM_CHECKS')
    defines = '\n'.join(re.search(r'^#define '+name+r'\b[^\n]*', source, re.M)[0] for name in names)
    prefix = r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MB (1ull<<20)
#define RT_TRACE_PERF 1
static unsigned tex_rate=256, rt_trace;
typedef struct Tex {
    struct { int mirrored, gpu_owned; } pub;
    unsigned gpu_dirty, overwritten, retiring, released, back_gap;
    uint32_t use_round;
    uint64_t bytes, rel_frame, use_frame, ow_frame;
    struct Tex *rel_prev, *rel_next;
    unsigned fixture_id, dropped;
} Tex;
static Tex *rel_head,*rel_tail,*sweep_at,*reclaim_at,*overwritten_at;
static int reclaim_gpu_phase,reclaim_cpu_seen;
static uint64_t rel_bytes, clock_frame=10000, now_ns, sync_done=100, sync_issued=100;
static uint32_t look_round=1000,submit_epoch;
static struct { uint32_t cap; Tex **vals; } by_key;
static struct { uint64_t frames; } gfx_counters;
static struct { uint64_t checks, inspected, scheduled, pending, ns; } reclaim_stats;
static uint64_t evict_wait,n_evict,n_evict_bytes,room_level,n_pressure_drops,n_commit_drops,n_cap_drops,n_commit_past;
static uint32_t room_checks;
static int64_t room_wanted;
static int evict_logs,vram_pressure=80,tex_evict=1,commit_evict_on=1,commit_critical;
static uint64_t commit_deficit, fake_pending, fake_cap, fake_used, fake_budget;
static int fake_low,fake_held,fake_known=1;
static uint64_t total_dropped, disuse_want,n_over_drops;
static unsigned reasons[16];
enum { DROP_VRAM=1,DROP_COMMIT,DROP_CAP,DROP_IDLE_VRAM,DROP_IDLE_COMMIT,DROP_OVERWRITTEN };
static uint64_t frames_now(void) { return clock_frame; }
static uint64_t kern_now_ns(void) { return ++now_ns; }
static void ru_unlink(Tex *e) { (void)e; }
static int keep_taken(void) { return 1; }
static void rt_log(const char *fmt,...) { (void)fmt; }
static const char *cfg_get(const char *n) { (void)n;return NULL; }
static int rt_knob(const char *n,const char *fallback) { (void)n;return atoi(fallback); }
static uint64_t vk_sync_fence_done(void) { return sync_done; }
static uint64_t vk_sync_issued(void) { return sync_issued; }
static int64_t plat_xchg64(int64_t *p,int64_t n) { int64_t old=*p;*p=n;return old; }
static int vk_mem_vram_live(uint64_t *u,uint64_t *b) { *u=fake_used;*b=fake_budget;return fake_known; }
static int vk_mem_vram(uint64_t *u,uint64_t *b,int *exact) { *exact=1;return vk_mem_vram_live(u,b); }
static void retire_sweep(void) {}
static int commit_is_low(void) { return fake_low; }
static int commit_drops_held(void) { return fake_held; }
static uint64_t released_cap_bytes(void) { return fake_cap; }
static uint64_t vk_mem_reclaim_pending(void) { return fake_pending; }
static uint64_t evict_disused(uint64_t want) { disuse_want=want;return 0; }
"""
    unlink = function(source,'rel_unlink')
    drop = r"""
static void drop_tex(Tex *e,int why) {
    assert(!e->dropped); e->dropped=1;
    total_dropped+=e->bytes;reasons[why]++;
    if(e->released)rel_unlink(e);
    if(by_key.vals)for(unsigned i=0;i<by_key.cap;i++)if(by_key.vals[i]==e)by_key.vals[i]=NULL;
}
"""
    functions='\n'.join(function(source,name) for name in ('comes_back_intact','idle','stale_after','reclaim_released','drop_overwritten','evict_idle','sweep_pressure'))
    main=r"""
static Tex textures[1600];static Tex *table[1600];
static void reset(void) {
    memset(textures,0,sizeof textures);memset(table,0,sizeof table);memset(reasons,0,sizeof reasons);
    memset(&reclaim_stats,0,sizeof reclaim_stats);memset(&by_key,0,sizeof by_key);
    rel_head=rel_tail=sweep_at=reclaim_at=overwritten_at=NULL;reclaim_gpu_phase=reclaim_cpu_seen=0;rel_bytes=total_dropped=disuse_want=0;
    fake_pending=0;fake_low=fake_held=commit_critical=0;commit_deficit=0;fake_cap=UINT64_MAX;
    fake_known=1;fake_used=500*MB;fake_budget=1000*MB;vram_pressure=80;tex_evict=commit_evict_on=1;
    room_wanted=room_level=room_checks=0;evict_wait=0;sync_done=sync_issued=100;
    gfx_counters.frames=clock_frame;submit_epoch=0;
}
static Tex *append(unsigned i,uint64_t bytes,int protected) {
    Tex *e=&textures[i];*e=(Tex){.bytes=bytes,.released=1,.fixture_id=i,.rel_frame=1};
    e->pub.mirrored=1;e->use_round=protected?look_round:1;e->use_frame=protected?clock_frame:1;
    e->rel_prev=rel_tail;if(rel_tail)rel_tail->rel_next=e;else rel_head=e;rel_tail=e;rel_bytes+=bytes;return e;
}
static Tex *live(unsigned i,int protected) {
    Tex *e=&textures[i];*e=(Tex){.bytes=MB,.fixture_id=i};e->pub.mirrored=1;
    e->use_round=protected?look_round:1;e->use_frame=protected?clock_frame:1;table[i]=e;return e;
}
int main(void) {
    uint64_t count=0,got;
    reset();for(unsigned i=0;i<700;i++)append(i,MB,1);textures[650].use_round=textures[650].use_frame=1;
    assert(reclaim_released(MB,&count,DROP_COMMIT)==0&&reclaim_stats.inspected==RECLAIM_LOOK);
    assert(reclaim_at==&textures[512]);
    rel_unlink(&textures[512]);assert(reclaim_at==&textures[513]);
    got=reclaim_released(MB,&count,DROP_COMMIT);assert(got==MB&&textures[650].dropped);
    assert(reclaim_stats.inspected<=2*RECLAIM_LOOK);
    // Recoverable contents get priority even when a much older GPU-only entry heads the list.
    reset();Tex *gpu=append(0,MB,0);gpu->pub.mirrored=0;append(1,MB,0);
    assert(reclaim_released(MB,&count,DROP_COMMIT)==MB&&textures[1].dropped&&!gpu->dropped);
    assert(reclaim_released(MB,&count,DROP_VRAM)==0&&!gpu->dropped);
    assert(reclaim_released(MB,&count,DROP_COMMIT)==0&&!gpu->dropped);
    assert(reclaim_released(MB,&count,DROP_COMMIT)==MB&&gpu->dropped);
    // A GPU-only entry in the first slice cannot bypass CPU-backed entries in later slices.
    reset();gpu=append(0,MB,0);gpu->pub.mirrored=0;
    for(unsigned i=1;i<700;i++)append(i,MB,1);
    textures[650].use_round=textures[650].use_frame=1;
    assert(!reclaim_released(MB,&count,DROP_COMMIT)&&!gpu->dropped&&!reclaim_gpu_phase);
    assert(reclaim_released(MB,&count,DROP_COMMIT)==MB&&textures[650].dropped&&!gpu->dropped&&!reclaim_gpu_phase);
    for(unsigned i=0;i<4&&!gpu->dropped;i++)reclaim_released(MB,&count,DROP_COMMIT);
    assert(gpu->dropped);
    // Meeting a small target must not allow a GPU-only entry to jump ahead of CPU candidates skipped this cycle.
    reset();gpu=append(0,MB,0);gpu->pub.mirrored=0;
    for(unsigned i=1;i<81;i++)append(i,MB,0);
    unsigned passes=0;
    while(!gpu->dropped && passes++<1000) {
        reclaim_released(MB,&count,DROP_COMMIT);
        if(gpu->dropped)for(unsigned i=1;i<81;i++)assert(textures[i].dropped);
    }
    assert(gpu->dropped&&passes<1000);
    reset();gpu=append(0,MB,0);gpu->pub.mirrored=0;gpu->rel_frame=clock_frame-GPU_STALE_FRAMES+1;
    append(1,MB,1);assert(!reclaim_released(20*MB,&count,DROP_COMMIT));
    // Prior takeback history extends the residency protection; it is not lost on a pressure check.
    reset();Tex *back=append(0,MB,0);back->back_gap=2000;
    assert(!reclaim_released(MB,&count,DROP_COMMIT)&&!back->dropped);
    reset();append(0,8*MB,0);append(1,8*MB,0);append(2,8*MB,0);
    assert(reclaim_released(10*MB,&count,DROP_COMMIT)==16*MB&&!textures[2].dropped);
    // Cap + measured VRAM + commit need overlap: one 30MiB request, not 20+30+25MiB.
    reset();for(unsigned i=0;i<80;i++)append(i,MB,0);
    fake_cap=60*MB;fake_used=810*MB;fake_low=1;commit_deficit=25*MB;
    sweep_pressure();assert(total_dropped==30*MB&&reclaim_stats.scheduled==30*MB&&!disuse_want);
    // Actual pending allocator destruction suspends ordinary batches; a later fresh sample can resume.
    reset();for(unsigned i=0;i<80;i++)append(i,MB,0);fake_low=1;commit_deficit=30*MB;fake_pending=8*MB;
    sweep_pressure();assert(!total_dropped&&reclaim_stats.inspected<=RECLAIM_LOOK&&reclaim_stats.pending==1);
    fake_pending=0;sweep_pressure();assert(total_dropped==30*MB);
    // Critical pressure still progresses, crediting pending destruction once across all reasons.
    reset();for(unsigned i=0;i<80;i++)append(i,MB,0);fake_low=commit_critical=1;commit_deficit=30*MB;fake_pending=10*MB;
    sweep_pressure();assert(total_dropped==20*MB);
    reset();for(unsigned i=0;i<80;i++)append(i,MB,0);fake_low=commit_critical=1;commit_deficit=30*MB;fake_pending=40*MB;
    sweep_pressure();assert(!total_dropped);
    // Pending frees already left the retained-byte count; they cannot pay the remaining cap excess twice.
    reset();for(unsigned i=0;i<80;i++)append(i,MB,0);
    fake_low=commit_critical=1;commit_deficit=30*MB;fake_pending=40*MB;fake_cap=70*MB;
    sweep_pressure();assert(total_dropped==10*MB&&!disuse_want);
    // A cache-size cap never authorizes discarding live resources to satisfy that cap.
    reset();for(unsigned i=0;i<20;i++)append(i,MB,1);fake_cap=0;
    by_key.cap=64;by_key.vals=table;live(30,0);sweep_pressure();assert(!total_dropped&&!textures[30].dropped&&!disuse_want);
    // Live eviction uses a bounded table slice, excludes current/retiring/GPU-only resources, and waits on fences.
    reset();by_key.cap=1024;by_key.vals=table;
    for(unsigned i=0;i<1024;i++)live(i,1);
    textures[0].use_round=textures[0].use_frame=1; textures[0].pub.gpu_owned=1;
    textures[1].use_round=textures[1].use_frame=1; textures[1].retiring=1;
    textures[2].use_round=textures[2].use_frame=1;
    assert(evict_idle(MB,0,0,DROP_IDLE_COMMIT)==MB&&textures[2].dropped);
    assert(!textures[0].dropped&&!textures[1].dropped&&reclaim_stats.inspected==RECLAIM_LOOK);
    uint64_t inspected=reclaim_stats.inspected;
    assert(!evict_idle(MB,0,0,DROP_IDLE_COMMIT)&&reclaim_stats.inspected==inspected);
    sync_done=102;textures[700].use_round=textures[700].use_frame=1;
    assert(evict_idle(MB,0,0,DROP_IDLE_COMMIT)==MB&&textures[700].dropped);
    assert(reclaim_stats.inspected==2*RECLAIM_LOOK);
    assert(n_evict>=2 && n_evict_bytes>=2*MB && n_commit_past==0);
    // Overwritten maintenance has its own bounded, removal-safe cursor instead of rescanning all protected entries.
    reset();for(unsigned i=0;i<700;i++)append(i,MB,1);
    textures[600].overwritten=1;textures[600].ow_frame=1;
    drop_overwritten();assert(reclaim_stats.inspected==RECLAIM_LOOK&&overwritten_at==&textures[512]&&!total_dropped);
    rel_unlink(&textures[512]);assert(overwritten_at==&textures[513]);
    drop_overwritten();assert(textures[600].dropped&&total_dropped==MB&&n_over_drops>0);
    assert(reclaim_stats.inspected<=2*RECLAIM_LOOK);
    puts("texture reclamation lifecycle: passed");return 0;
}
"""
    return prefix+defines+'\n'+unlink+drop+functions+main

if __name__ == '__main__':
    run('budget', BUDGET)
    run('lifecycle', lifecycle_fixture())


