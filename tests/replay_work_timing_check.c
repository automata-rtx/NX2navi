// Controlled clocks exercise the production diagnostic implementation, including nested owner-help work.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "replay_work_trace.h"

#if NX2NAVI_REPLAY_WORK
#define TL_CLOSED (1u << 31)
static uint64_t wall = 100, cpu = 1000;
static unsigned clock_reads, recording = 1;
static uint64_t plat_now_ns(void) { clock_reads++; return wall; }
static uint32_t plat_thread_id(void) { return 7; }
static uint64_t plat_thread_cpu_ns(uint32_t tid) { assert(tid == 7); return cpu; }
static int tl_recording(void) { return (int)recording; }
static unsigned inside_gaps,outside_gaps;
static void rt_log(const char *format, ...) {
    inside_gaps+=strstr(format,"rw unscoped phase")!=NULL;
    outside_gaps+=strstr(format,"rw outside-window unscoped phase")!=NULL;
}
#include "replay_work_impl.h"

static uint64_t exclusive_sum(const RwtUnit *u) {
    uint64_t n = 0;
    for (unsigned i = 0; i < RWT_REGION_COUNT; i++) n += u->region_ns[i];
    return n;
}

int main(void) {
    RwtUnitScope owner = gfx_rwt_unit_begin(RWT_ROLE_OWNER, 1, 1, 0, 0);
    wall = 110; cpu = 1010;
    RwtRegionScope sync = gfx_rwt_region_begin(RWT_REGION_POOL_SYNC);
    wall = 120;
    RwtRegionScope lock = gfx_rwt_region_begin(RWT_REGION_POOL_LOCK);
    wall = 130; gfx_rwt_region_end(&lock);
    wall = 140; cpu = 1030;
    RwtUnitScope helper = gfx_rwt_unit_begin(RWT_ROLE_HELP, 1, 1, 1, 0);
    wall = 150;
    RwtRegionScope entries = gfx_rwt_region_begin(RWT_REGION_POOL_ENTRIES);
    wall = 170; gfx_rwt_region_end(&entries);
    wall = 180; cpu = 1060; gfx_rwt_unit_end(&helper);
    wall = 200; gfx_rwt_region_end(&sync);
    wall = 220; cpu = 1100; gfx_rwt_unit_end(&owner);
    const RwtUnit *p = owner.unit, *h = helper.unit;
    assert(p && h && !p->invalid && !h->invalid && p->cpu_valid && h->cpu_valid);
    assert(p->children_ns == 40 && p->cpu_ns == 100 && p->children_cpu_ns == 30);
    assert(p->region_ns[RWT_REGION_POOL_SYNC] == 40);
    assert(p->region_inclusive_ns[RWT_REGION_POOL_SYNC] == 50);
    assert(p->region_ns[RWT_REGION_POOL_LOCK] == 10);
    assert(p->region_ns[RWT_REGION_RESIDUAL] == 30);
    assert(exclusive_sum(p) == p->end - p->begin - p->children_ns);
    assert(exclusive_sum(h) == 40 && h->region_ns[RWT_REGION_POOL_ENTRIES] == 20);
    assert(h->region_ns[RWT_REGION_RESIDUAL] == 20 && h->cpu_ns == 30);
    gfx_rwt_region_end(&sync); gfx_rwt_unit_end(&helper); gfx_rwt_unit_end(&owner);
    assert(p->children_ns == 40 && exclusive_sum(p) == 80); // Cleanup after explicit end is harmless.

    // Complete command classes nest preparation, bindings, effects and existing pool waits in one budget.
    // A completion inside recording remains in the parent's budget; a background completion gets its own unit.
    wall=230;cpu=1110;
    RwtUnitScope chain=gfx_rwt_unit_begin(RWT_ROLE_OWNER,2,2,0,0);
    RwtRegionScope draw=gfx_rwt_region_begin(RWT_REGION_OP_DRAW);
    wall=240;RwtRegionScope prepare=gfx_rwt_region_begin(RWT_REGION_DRAW_PREPARE);
    wall=250;RwtRegionScope bind=gfx_rwt_region_begin(RWT_REGION_DRAW_BIND);
    wall=260;RwtRegionScope pool_wait=gfx_rwt_region_begin(RWT_REGION_POOL_WAIT);
    wall=270;gfx_rwt_region_end(&pool_wait);
    wall=280;gfx_rwt_region_end(&bind);gfx_rwt_region_end(&prepare);
    RwtRegionScope effect=gfx_rwt_region_begin(RWT_REGION_EFFECT_RECORDED);
    wall=290;gfx_rwt_region_end(&effect);
    RwtDrawScope sampled=gfx_rwt_draw_begin(RWT_KIND_DRAW);gfx_rwt_draw_activate(&sampled);
    RwtUnitScope parent_completion=gfx_rwt_completion_begin(2);assert(!parent_completion.entered);
    RwtRegionScope complete=gfx_rwt_region_begin(RWT_REGION_COMPLETION);
    wall=300;gfx_rwt_region_end(&complete);gfx_rwt_unit_end(&parent_completion);gfx_rwt_draw_end(&sampled);
    gfx_rwt_region_end(&draw);
    RwtRegionScope compute=gfx_rwt_region_begin(RWT_REGION_OP_COMPUTE);
    wall=310;gfx_rwt_region_end(&compute);cpu=1190;gfx_rwt_unit_end(&chain);
    assert(!chain.unit->invalid&&exclusive_sum(chain.unit)==80);
    assert(chain.unit->region_ns[RWT_REGION_OP_DRAW]==10&&chain.unit->region_ns[RWT_REGION_DRAW_PREPARE]==10);
    assert(chain.unit->region_ns[RWT_REGION_DRAW_BIND]==20&&chain.unit->region_ns[RWT_REGION_POOL_WAIT]==10);
    assert(chain.unit->region_ns[RWT_REGION_EFFECT_RECORDED]==10&&chain.unit->region_ns[RWT_REGION_COMPLETION]==10);
    assert(chain.unit->region_ns[RWT_REGION_OP_COMPUTE]==10&&!chain.unit->children_ns);
    wall=320;RwtUnitScope background=gfx_rwt_completion_begin(2);
    assert(background.unit&&background.unit->role==RWT_ROLE_COMPLETION&&!background.unit->parent);
    complete=gfx_rwt_region_begin(RWT_REGION_COMPLETION);
    wall=340;gfx_rwt_region_end(&complete);gfx_rwt_unit_end(&background);
    assert(!background.unit->invalid&&exclusive_sum(background.unit)==20);

    unsigned before = clock_reads;
    recording = 0;
    RwtUnitScope off_completion=gfx_rwt_completion_begin(2);
    gfx_rwt_unit_end(&off_completion);assert(!off_completion.unit);
    RwtUnitScope off = gfx_rwt_unit_begin(RWT_ROLE_OWNER, 2, 2, 0, 0);
    RwtRegionScope off_region = gfx_rwt_region_begin(RWT_REGION_POOL_SYNC);
    gfx_rwt_region_end(&off_region); gfx_rwt_unit_end(&off);
    assert(clock_reads == before && !rwt_unscoped_regions[RWT_REGION_POOL_SYNC]);
    recording = 1;
    off_region = gfx_rwt_region_begin(RWT_REGION_POOL_SYNC);
    assert(!off_region.unit && rwt_unscoped_regions[RWT_REGION_POOL_SYNC] == 1);
    assert(rwt_unscoped_count==1&&rwt_unscoped_committed[0]&&rwt_unscoped[0].at==wall);

    wall = 300; cpu = 1200;
    RwtUnitScope invalid_cpu = gfx_rwt_unit_begin(RWT_ROLE_WORKER, 3, 3, 0, 0);
    wall = 310; cpu = 0; gfx_rwt_unit_end(&invalid_cpu);
    assert(!invalid_cpu.unit->cpu_valid && !invalid_cpu.unit->invalid);
    assert(exclusive_sum(invalid_cpu.unit) == 10);
    wall = 330; cpu = 1300;
    RwtUnitScope closing = gfx_rwt_unit_begin(RWT_ROLE_OWNER, 3, 3, 0, 0);
    wall = 340; recording = 0;
    RwtUnitScope missed_child = gfx_rwt_unit_begin(RWT_ROLE_HELP, 3, 3, 1, 0);
    wall = 350; gfx_rwt_unit_end(&missed_child);
    wall = 360; cpu = 1330; gfx_rwt_unit_end(&closing);
    assert(closing.unit->invalid && !closing.unit->cpu_valid);
    assert(closing.unit->children_ns == 10 && exclusive_sum(closing.unit) == 20);
    recording = 1;
    wall = 400; cpu = 1500;
    RwtUnitScope bad = gfx_rwt_unit_begin(RWT_ROLE_WORKER, 4, 4, 0, 0);
    RwtRegionScope outer = gfx_rwt_region_begin(RWT_REGION_POOL_SYNC);
    RwtRegionScope inner = gfx_rwt_region_begin(RWT_REGION_POOL_LOCK);
    gfx_rwt_region_end(&outer); gfx_rwt_region_end(&inner);
    wall = 410; cpu = 1510; gfx_rwt_unit_end(&bad);
    assert(bad.unit->invalid && !rwt_committed[bad.unit->id - 1]);
    off_region=gfx_rwt_region_begin(RWT_REGION_POOL_SYNC);assert(!off_region.unit);
    rwt_dump(350,1000);
    assert(inside_gaps==1&&outside_gaps==1); // Pre-window recording is not an in-window attribution gap.
    puts("replay timing: nested exclusive budgets, child CPU subtraction, residuals, disabled capture and invalid scopes passed");
    return 0;
}
#else
int main(void) {
    unsigned n = 0;
    RWT_REGION(off, RWT_REGION_POOL_SYNC);
    assert(RWT_REGION_CALL(RWT_REGION_POOL_LOCK, ++n) == 1 && n == 1);
    RWT_REGION_END(off);
    RWT_CHAIN(chain_off,RWT_REGION_OP_DRAW);
    RWT_COMPLETION(completion_off,1);
    puts("replay timing: compiled-off expression evaluated once");
    return 0;
}
#endif
