"""Production provider giveback, pool demand and domain-specific pressure transitions."""
from pathlib import Path
import tempfile

import texture_material_integration_check as material


PRESSURE = r'''
static int vram_pressure=80,tex_evict=1,commit_evict_on=1;
enum { VRAM_PRESSURE_DEFAULT=80,DROP_VRAM=1,DROP_COMMIT,DROP_CAP,DROP_IDLE_VRAM,DROP_IDLE_COMMIT };
static uint64_t commit_deficit,pressure_used,pressure_budget=1000,host_deficit;
static unsigned commit_critical,host_releases_commit=1,sample_calls,refreshes;
static uint64_t n_pressure_drops,n_commit_drops,n_cap_drops,n_commit_past;
static struct {uint64_t checks,inspected,scheduled,pending,ns;} reclaim_stats;
static Tex *rel_head;
static uint64_t freed_at_sample,freed_provider_bytes;
static const char *cfg_get(const char *key){(void)key;return NULL;}
static int rt_knob(const char *key,const char *value){(void)key;(void)value;return 1;}
static void rt_log(const char *fmt,...){(void)fmt;}
static void retire_sweep(void){}
static int vk_mem_vram_live(uint64_t *used,uint64_t *budget){*used=pressure_used;*budget=pressure_budget;return 1;}
static int vk_mem_vram(uint64_t *used,uint64_t *budget,int *exact){*exact=1;return vk_mem_vram_live(used,budget);}
static int commit_is_low(void){
 sample_calls++;uint64_t returned=host_releases_commit?freed_provider_bytes-freed_at_sample:0;
 commit_deficit=host_deficit>returned?host_deficit-returned:0;return commit_deficit!=0;
}
void resource_core_budget_refresh(void){refreshes++;}
static int commit_drops_held(void){return 0;}
static void drop_overwritten(void){}
static uint64_t released_cap_bytes(void){return UINT64_MAX;}
static uint64_t vk_mem_reclaim_pending(void){return 0;}
static void vk_exec_texture_publications_request_reclaim(void){}
static uint64_t reclaim_released(uint64_t n,uint64_t *count,int why,uint32_t domain){(void)n;(void)count;(void)why;(void)domain;return 0;}
static uint64_t evict_idle(uint64_t n,uint64_t used,uint64_t budget,int why,uint32_t domain){(void)n;(void)used;(void)budget;(void)why;(void)domain;return 0;}
static void evict_disused(uint64_t n,uint32_t domain){(void)n;(void)domain;}
/* production pressure sweep */

static void *tracked_bytes[8];static unsigned tracked_count,tracked_frees;
static void pressure_free(void *p){
 for(unsigned i=0;i<tracked_count;i++)if(p==tracked_bytes[i]){
  tracked_bytes[i]=NULL;tracked_frees++;freed_provider_bytes+=64;break;
 }
 free(p);
}
static void freed_before_budget_destroy(void){assert(tracked_frees);}
static void track_provider(TextureMaterial *m){
 assert(m->provider && tracked_count<8);tracked_bytes[tracked_count++]=((MaterialSource*)m->provider)->bytes;
}
static void begin_case(uint64_t host,uint64_t gpu){
 assert(!material_ledger.owners && !material_snapshot_bytes);
 host_deficit=host;pressure_used=gpu;freed_at_sample=freed_provider_bytes;
 host_releases_commit=1;sample_calls=refreshes=0;reclaim_stats=(__typeof__(reclaim_stats)){0};
 submit_epoch=(submit_epoch+8)&~7u;head=NULL;tex_deferred=0;tracked_count=tracked_frees=0;
}
static void seed(Tex *e,uint64_t addr,uint8_t *bytes,unsigned image){
 *e=make_tex(addr,bytes);observes=0;assert(!content_early_prepare(e));publish_snapshot(e,image);
 track_provider(e->material);
}
static void end_case(Tex *e){
 if(e->ph){uint32_t id=(uint32_t)e->ph;if(pref[id-1].refs)vk_texture_pool_hold_locked(id,0);pool_forget(e);}
 if(e->material)content_release(e,1);
 content_collect(UINT64_MAX);
}
int main(void){
 uint8_t bytes[64],other[64];memset(bytes,21,64);memset(other,47,64);
 // Current pool selection survives submit changes, including a released/finalized association. Last release
 // makes it eligible again; a stale PRef pointing elsewhere is never mistaken for current demand.
 begin_case(0,0);Tex selected,cold;seed(&selected,4096,bytes,10);
 uint32_t id=pool_ref(&selected);vk_texture_pool_hold_locked(0,id);
 texture_material_set_live(selected.material,0);selected.released=1;submit_epoch++;
 assert(!content_reclaim(64,1) && selected.pub.image==10);
 vk_texture_pool_hold_locked(id,0);assert(content_reclaim(64,1)==64 && !selected.pub.image);
 content_collect(UINT64_MAX);assert(!content_drop_providers(64) && selected.material->provider);
 assert(content_restore(&selected,selected.material,NULL));submit_epoch++;
 vk_texture_pool_hold_locked(0,id);pref[id-1].e=&cold;
 assert(content_reclaim(64,1)==64);pref[id-1].e=&selected;
 end_case(&selected);assert(!material_ledger.owners);

 // Commit-only pressure drops the optional host copy first. Actual post-free sampling removes the deficit,
 // leaving the demanded GPU image and its existing bindings untouched across repeated clean accesses.
 begin_case(64,0);seed(&selected,8192,bytes,11);id=pool_ref(&selected);vk_texture_pool_hold_locked(0,id);
 budget_destroy_hook=freed_before_budget_destroy;
 unsigned before_generations=generations,before_allocations=allocations,before_hashes=hashes;
 sweep_pressure();budget_destroy_hook=NULL;
 assert(sample_calls==2 && refreshes==1 && !commit_deficit && !selected.material->provider);
 assert(selected.pub.image==11 && !reclaim_stats.scheduled && generations==before_generations);
 for(unsigned i=0;i<100;i++)assert(content_require_resident(&selected) && make_fresh_source(&selected));
 assert(allocations==before_allocations && hashes==before_hashes && !selected.material->provider);
 end_case(&selected);

 // Simultaneous pressure prefers the demanded image's raw copy; the cold image keeps reconstruction bytes
 // and gives back GPU residency. Host bytes cannot reduce the independently measured 120-byte GPU deficit.
 begin_case(64,900);seed(&selected,12288,bytes,12);seed(&cold,16384,other,13);
 id=pool_ref(&selected);vk_texture_pool_hold_locked(0,id);submit_epoch+=8;
 sweep_pressure();assert(!selected.material->provider && selected.pub.image==12);
 assert(!cold.pub.image && cold.material->provider && reclaim_stats.scheduled==64);
 assert(!commit_deficit && refreshes==1 && material_ledger.provider_bytes==64);
 end_case(&cold);end_case(&selected);

 // GPU-only pressure does not discard host providers. The cold exact version remains reconstructible.
 begin_case(0,900);seed(&cold,20480,bytes,14);submit_epoch+=8;sweep_pressure();
 assert(!cold.pub.image && cold.material->provider && !refreshes && reclaim_stats.scheduled==64);
 end_case(&cold);

 // Freeing host allocations need not return committed heap pages immediately. Resampling must keep that
 // real deficit, and an essential current pool set must stop after bounded no-progress work, never thrash.
 begin_case(512,900);host_releases_commit=0;seed(&selected,24576,bytes,15);
 id=pool_ref(&selected);vk_texture_pool_hold_locked(0,id);submit_epoch+=8;
 sweep_pressure();assert(commit_deficit==512 && refreshes==1 && selected.pub.image==15);
 assert(!selected.material->provider && !reclaim_stats.scheduled);
 before_allocations=allocations;before_generations=generations;
 for(unsigned i=0;i<100;i++){submit_epoch+=8;sweep_pressure();}
 assert(selected.pub.image==15 && !reclaim_stats.scheduled && refreshes==1 && commit_deficit==512);
 assert(allocations==before_allocations && generations==before_generations);end_case(&selected);

 // Provider drop retains the sole exact physical version for a finalized alias and a recorded consumer.
 // Mutating the current alias performs COW and cannot alter or detach the old finalized image.
 begin_case(0,0);seed(&selected,28672,bytes,16);Tex final=make_tex(32768,bytes);observes=0;
 assert(content_early_prepare(&final) && final.material==selected.material);
 TextureMaterial *old=selected.material;texture_material_set_live(old,0);final.released=1;final.t.mem=NULL;
 TextureMaterialUse recording={0};assert(texture_material_use_begin(old,&recording)==RC_LIFETIME_OK);
 assert(content_drop_providers(64)==64 && !old->provider);
 assert(!texture_material_take_residency(old).image);
 assert(content_write_begin(&selected,1) && !selected.material && final.pub.image==16);
 physical[selected.pub.image][0]=99;assert(physical[16][0]==21 && recording.instance->value.image==16);
 submit_epoch++;assert(!content_reclaim(64,1) && !texture_material_take_residency(old).image);
 end_case(&final);assert(material_ledger.pending_bytes==64 && material_ledger.pending_instances==1);
 assert(texture_material_use_cancel(&recording)==RC_LIFETIME_OK);content_collect(UINT64_MAX);
 assert(!material_ledger.owners && !material_ledger.pending_instances);

 // An unreferenced cache owner can drop its provider and then retire completely; no permanent resident pin.
 begin_case(0,0);seed(&cold,36864,bytes,17);TextureMaterial *cached=cold.material;
 content_release(&cold,0);assert(cached->cached && !cached->refs);
 assert(content_drop_providers(64)==64 && content_reclaim(64,0)==64);content_collect(UINT64_MAX);
 assert(!material_ledger.owners && !material_snapshot_bytes && !material_ledger.provider_bytes && !n_commit_past);
 free(pref);
 puts("provider pressure: exact pool demand, host/GPU separation, observed relief, bounded no-progress and COW lifetime passed");
 return 0;
}
'''


def fixture():
    source = (material.ROOT / 'runtime/vk_texture.c').read_text()
    result = material.content_fixture().replace('int main(void){', 'int content_baseline_main(void){', 1)
    result = result.replace('int last_ch,ph;', 'int last_ch,ph,pidx;')
    begin = source.index('typedef struct {', source.index('// RESOURCE_MEMBERSHIP_BEGIN:'))
    end = source.index('static void pool_touch(void)', begin)
    membership = source[begin:end]
    membership = membership.replace('static void touch(Tex *e);', '')
    minimal = 'typedef struct { Tex *e; uint32_t refs; } PRef;\nstatic PRef *pref;\nstatic int npref;'
    stubs = '''static struct {unsigned dependency_detach;} resource_path;
static void pool_unlist(Tex *e){e->pidx=0;}
static void pool_list(Tex *e){e->pidx=1;}
static void rt_fatal(const char *fmt,...);
'''
    assert minimal in result
    result = result.replace(minimal, stubs + membership)
    result = result.replace('RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *t){',
                            'static void (*budget_destroy_hook)(void);\n'
                            'RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *t){'
                            'if(budget_destroy_hook)budget_destroy_hook();')
    result = result.replace('#include "vk_texture_content.h"',
                            'static void pressure_free(void *);\n#define free pressure_free\n'
                            '#include "vk_texture_content.h"\n#undef free')
    begin = source.index('static void sweep_pressure(void)')
    end = source.index('\n}\n', begin) + 3
    return result + PRESSURE.replace('/* production pressure sweep */', source[begin:end])


if __name__ == '__main__':
    material.OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='provider-pressure-', dir=material.OUT) as temp:
        material.OUT = Path(temp)
        material.compile_and_run('provider_pressure', fixture())
