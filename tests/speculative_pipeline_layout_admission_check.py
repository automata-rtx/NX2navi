"""Check ordinary speculative graphics layout admission using the real shared ledger."""

import os
from pathlib import Path
import subprocess
import tempfile

import resource_pipeline_progress_check as harness


ROOT = Path(__file__).resolve().parents[1]

MODEL = r'''
typedef int GfxShader;
enum {GFX_STAGE_VS=1,GFX_STAGE_FS=2};
typedef struct {GfxShader *vs,*fs;unsigned native_rmw_marker;} VkPipeState;
typedef struct {unsigned topo_class;} GKey;
typedef struct {unsigned n;} Bindings;
typedef struct {int set1,half[2];} Layout;
typedef struct {int kind;} FixtureKey;
typedef FixtureKey ViKey,PrKey,FsKey,FoKey;
typedef struct Lib Lib;
typedef struct {Lib *b; int kind; Layout *l; VkPipeState s; GKey k; int optional;} LibJob;
struct Lib {LibJob *job;};
enum {LIB_VI,LIB_PR,LIB_FS,LIB_FO,LIB_COUNT};
static struct {int on,closed;} A={.on=1};
static struct {int on;uint64_t queued,waits;} U={.on=1};
static struct {int lock,gpl,indep,ready;struct {uint64_t hints,hint_libs;} g;} P={.gpl=1,.indep=1,.ready=1};
static uint64_t recording_scene=9;
static unsigned layout_cached,lock_calls,native_recheck,pipe_published,hint_jobs;
static Layout layout;
static Lib cached[LIB_COUNT];
static int hint_on(void){return 1;}
static int is_mesh(const VkPipeState *s){(void)s;return 0;}
static int gfx_shader_stage(const GfxShader *s){return *s;}
static void gkey_fill(const VkPipeState *s,GKey *k){assert(s->vs);*k=(GKey){0};}
static uint64_t key_hash(const void *bytes,size_t n){assert(bytes&&n);return 7;}
static int plat_mutex_trylock(int *p){
    assert(!budget_depth);
    if(*p)return 0;
    (*p)++; mutex_depth++;lock_calls++;
    if(native_recheck && lock_calls==2)pipe_published=1;
    return 1;
}
static void plat_mutex_unlock(int *p){assert(*p==1&&mutex_depth==1);(*p)--;mutex_depth--;}
static int pipe_find(const GKey *k,size_t n,uint64_t h){assert(P.lock&&k&&n&&h==7);return pipe_published;}
static int add_stage(Bindings *b,const GfxShader *s,unsigned first){assert(s&&!first);++b->n;return 1;}
static Layout *layout_pair_select(Bindings *b,Bindings *f,VkResult *status,int create){
    assert(P.lock&&recording_scene==9&&b->n==1&&f->n==1&&*status==VK_NOT_READY);
    if(layout_cached)return &layout;
    if(!create)return NULL;
    assert(pipe_budget.ticket.token.ledger&&pipe_budget_enter_native());
    driver_calls+=3;
    layout_cached=1;
    layout.set1=1;
    return &layout;
}
static int workers_start(void){return 1;}
static void lib_keys(const VkPipeState *s,const GKey *k,ViKey *vi,PrKey *pr,FsKey *fs,FoKey *fo){
    assert(s&&k);*vi=(FixtureKey){LIB_VI};*pr=(FixtureKey){LIB_PR};*fs=(FixtureKey){LIB_FS};*fo=(FixtureKey){LIB_FO};
}
static Lib *lib_find(uint64_t hash,const void *key,uint32_t size){
    assert(hash==7&&size==sizeof(FixtureKey));int kind=((const FixtureKey *)key)->kind;
    return cached[kind].job ? &cached[kind] : NULL;
}
static Lib *lib_new(uint64_t hash,const void *key,uint32_t size){
    assert(hash==7&&size==sizeof(FixtureKey));return &cached[((const FixtureKey *)key)->kind];
}
static void u_promote(LibJob *j){assert(j);j->optional=0;}
static void u_push(LibJob *j){assert(j&&j->optional);hint_jobs++;}
static int half_layout(Layout *l,int half){
    assert(P.lock&&l==&layout);
    if(l->half[half])return 1;
    assert(pipe_budget.ticket.token.ledger&&pipe_budget.native_entered);
    driver_calls++;
    return l->half[half]=1;
}
void vk_pipeline_stop_preparation(void){pipe_budget_stopping=1;}
static void reset_hint(uint64_t room){
    assert(recording_scene==9&&!P.lock);
    for(unsigned i=0;i<LIB_COUNT;i++){free(cached[i].job);cached[i].job=NULL;}
    reset_ledger(room,room);
    native_recheck=pipe_published=hint_jobs=layout_cached=lock_calls=0;
    layout=(Layout){0};
    P.g.hint_libs=P.g.hints=0;
}
'''

CHECK = r'''
int main(void){
    GfxShader vs=GFX_STAGE_VS,fs=GFX_STAGE_FS;
    VkPipeState state={.vs=&vs,.fs=&fs};
    reset_hint(ROOM);
    assert(vk_pipeline_hint(&state)==0&&!driver_calls&&!hint_jobs);
    assert(pipeline_stats().refused==1&&!layout_cached);
    assert(pipe_budget_try_begin(ROOM,1,NULL)==PIPE_BUDGET_OWNED);
    pipe_budget_end(PIPE_BUDGET_OWNED);
    assert_charge(0,0);

    reset_hint(ROOM);
    layout_cached=1; layout=(Layout){1,{1,1}};
    assert(vk_pipeline_hint(&state)==2&&!driver_calls&&hint_jobs==2);
    assert(!pipeline_stats().refused&&!pipeline_stats().admitted);
    assert(vk_pipeline_hint(&state)==0&&hint_jobs==2);
    assert_charge(0,0);

    reset_hint(ROOM);
    layout_cached=1;layout=(Layout){1,{1,0}};
    assert(vk_pipeline_hint(&state)==1&&hint_jobs==1&&!driver_calls&&!layout.half[1]);
    assert(cached[LIB_PR].job&&!cached[LIB_FS].job&&pipeline_stats().refused==1);

    reset_hint(2048);
    assert(vk_pipeline_hint(&state)==2&&driver_calls==5&&hint_jobs==2);
    assert(pipeline_stats().admitted==1&&pipeline_stats().opaque_finished==1);
    assert_charge(0,ROOM);
    assert(vk_pipeline_hint(&state)==0&&driver_calls==5&&hint_jobs==2);

    reset_hint(2048);
    native_recheck=1;
    assert(vk_pipeline_hint(&state)==0&&pipe_published&&lock_calls==2&&!driver_calls&&!hint_jobs);
    assert(pipeline_stats().admitted==1&&!pipeline_stats().opaque_finished);
    assert_charge(0,0);

    reset_hint(ROOM);
    RcBudgetTicket pending={0};
    RcAdmissionVector other=pipe_budget_cost(16),need=pipe_budget_cost(ROOM);
    assert(resource_core_budget_reserve(RC_BUDGET_BUFFER,1,&other,&pending)==RC_ADMISSION_OK);
    assert(resource_core_budget_materialize(&pending,&other)==RC_ADMISSION_OK);
    assert(resource_core_budget_retire(&pending)==RC_ADMISSION_OK);
    assert(vk_pipeline_hint(&state)==0&&!driver_calls);
    RcBudgetReadiness ready;
    resource_core_budget_readiness(1,&ready);
    assert(!resource_core_budget_peak_ready(&need,&ready));
    assert(resource_core_budget_destroyed(&pending)==RC_ADMISSION_OK);
    resource_core_budget_readiness(1,&ready);
    assert(!resource_core_budget_peak_ready(&need,&ready));
    change_room(ROOM,ROOM);test_clock+=1000000000;resource_core_budget_refresh();
    assert(pipe_budget_try_begin(ROOM,1,NULL)==PIPE_BUDGET_OWNED);
    pipe_budget_end(PIPE_BUDGET_OWNED);

    assert(recording_scene==9&&!P.lock&&!wait_count&&!service_count);
    puts("Speculative pipeline: optional refusal leaves required room; warm/partial hits and one owned cold request; same-key arrival and real retirement passed");
}
'''

def main():
    result_type = next(line for line in harness.PIPELINE.splitlines()
                       if "LIB_QUEUE_UNAVAILABLE, LIB_QUEUE_READY" in line)
    functions = ("lib_queue", "pipeline_hint_queue_locked", "vk_pipeline_hint")
    code = "\n".join((harness.base, harness.admission, harness.optional, MODEL, result_type,
                      *(harness.function(name) for name in functions), CHECK))
    cc = Path(os.environ["CC"]).resolve()
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ["PATH"])
    with tempfile.TemporaryDirectory(prefix="speculative-layout-admission-") as directory:
        source = Path(directory) / "check.c"
        source.write_text(code, encoding="utf-8")
        exe = source.with_suffix(".exe")
        subprocess.run([str(cc), "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I" + str(ROOT / "runtime"),
                        str(source), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True, timeout=20)


if __name__ == "__main__":
    main()
