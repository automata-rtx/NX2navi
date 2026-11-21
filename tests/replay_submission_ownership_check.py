"""Verify capture transfer, cancellation and shutdown order at the backend caller boundary."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from bind_recipe_check import function

source = (ROOT / "runtime/vk_backend.c").read_text()
metadata = (ROOT / "runtime/vk_usnap.c").read_text()
owner = source[source.index("enum { RI_SUBMIT"):source.index("static void replay_handles_release(")]
accounting = (metadata[metadata.rindex("static RcAdmissionVector arena_host_cost("):
                       metadata.index("static void arena_cancel_host_reservation(")] +
              metadata[metadata.rindex("static VkResult arena_admission_failure("):
                       metadata.rindex("static VkResult metadata_allocate(")])
selected = "\n".join(function(source, name) for name in
                     ("replay_frontend_begin", "replay_frontend_end", "replay_handles_release",
                      "replay_main", "replay_enqueue", "replay_handles_clone",
                      "replay_registration_cutoff", "replay_registration_pending", "replay_registration_wait",
                      "replay_copy_texture", "vk_be_texture_release", "replay_capture", "replay_submit",
                      "vk_be_submit", "vk_backend_shutdown"))

fixture = r'''
#include <assert.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum {DR_FULL, DR_FINISH, DR_GATE, DR_CAPTURE, DR_TEX_RELEASE, DR_REGISTRATION, RQ_CAP=4};
typedef struct {int alive;} VkUsnapCapture;
typedef struct {int alive;} VkDrawScratchCredit;
typedef struct {VkUsnapCapture *snapshot;VkDrawScratchCredit *scratch;} VkReplayCaptureBundle;
typedef int VkResult;
enum {VK_SUCCESS, VK_NOT_READY, VK_ERROR_INITIALIZATION_FAILED, VK_ERROR_TOO_MANY_OBJECTS,
      VK_ERROR_FEATURE_NOT_PRESENT, VK_ERROR_OUT_OF_HOST_MEMORY};
enum {RC_BUDGET_RAM, RC_BUDGET_COMMIT, RC_ADMISSION_MAX_DOMAINS};
typedef enum {RC_ADMISSION_OK, RC_ADMISSION_REFUSED, RC_ADMISSION_NO_RECORDS, RC_ADMISSION_UNSUPPORTED} RcAdmissionStatus;
typedef enum {RC_BUDGET_SNAPSHOT} RcBudgetProducer;
typedef struct {struct {const void *ledger;} token; RcBudgetProducer producer;} RcBudgetTicket;
typedef struct {uint64_t amount[RC_ADMISSION_MAX_DOMAINS];} RcAdmissionVector;
typedef struct {int ready;} RcBudgetReadiness;
typedef struct {uint64_t event;} RcBudgetProgress;
typedef enum {VK_REPLAY_CAPTURE_WAIT_NONE,VK_REPLAY_CAPTURE_WAIT_BUDGET,
              VK_REPLAY_CAPTURE_WAIT_PRODUCER,VK_REPLAY_CAPTURE_WAIT_DRAIN_OLDER} VkReplayCaptureWaitKind;
typedef struct {RcAdmissionVector peak;RcBudgetReadiness before;uint64_t producer_epoch;
                VkReplayCaptureWaitKind kind;} VkReplayCaptureWait;
typedef struct {int alive;uint64_t publication;} VkTextureRegistrationScope;
typedef struct {uint64_t address;} p_bind_buf;
typedef struct {
    const uint64_t *handles;
    int nhandles;
    const p_bind_buf *inherited;
    size_t ninherited;
} VkUsnapInputs;
typedef struct {p_bind_buf slots[8*18];} VkUsnapBindingState;
typedef struct {void *mem;} tex_t;
static VkUsnapBindingState r_uniforms;
''' + owner + r'''
static ReplayItem rq[RQ_CAP];
static int rq_head,rq_n,r_busy,r_started=1,r_frontend_calls,rl,rcv,replay_registration_order;
static uint64_t r_submit_done,r_presents_done;
static uint64_t dr_calls[8],dr_tcalls[8],next_fence;
static uint64_t r_registration_issued, r_registration_done, r_registration_publication, registration_publication=1;
static uint64_t vk_texture_registration_record_revision(void){return 0;}
static int stopping,lost,abort_before_copy,stop_after_copy,scope_releases,capture_releases;
static int fixed_blocked, fixed_attempts;
static int scratch_releases, drain_capture, capture_steps;
static int producer_refused_once, producer_waits;
static int capture_race, copied_under_registration_order;
static int protect_cold, registration_waits;
static unsigned gate_closed, gate_steps, gate_drains;
static uint64_t observed_inherited;
static int vk_sub_in_frame,nvn_submits_in_frame;
static struct {uint64_t frames;} gfx_counters;
static VkUsnapCapture arena;
static VkDrawScratchCredit scratch;
static VkTextureRegistrationScope registrations;
static int shutdown_hooks,waited,simulated_submitter,loss_in_pipeline_shutdown,loss_in_window_release;
static int mesh_idle_failure,mesh_failure_reported;
static jmp_buf mesh_failure;
static int worker,worker_submits;
static jmp_buf worker_done;
static void replay_main(void *);
static void finish_test_replay(void);
static int budget_state,budget_blocked,budget_progress_now,budget_stop,budget_exhausted;
static int budget_allocations,budget_retires,budget_destroys,budget_reclaims,budget_waits,budget_cancels;
static int fail_malloc, expect_fatal, project_failure;
static jmp_buf allocation_failure;
static uint64_t budget_size;
static const uint64_t *attempt_copy;
static int attempt_allocations;
static void *metadata_malloc(size_t bytes) { return fail_malloc ? NULL : malloc(bytes); }
static void plat_mutex_lock(int *m){assert((m==&rl||m==&replay_registration_order)&&!*m);*m=1;}
static void plat_mutex_unlock(int *m){assert((m==&rl||m==&replay_registration_order)&&*m);*m=0;}
static void plat_cond_wait(int *c,int *m){
    if(worker){
        assert(c==&rcv&&m==&rl&&*m&&!rq_n&&!r_busy);
        *m=0;longjmp(worker_done,1);
    }
    if(protect_cold){
        assert(c==&rcv&&m==&rl&&*m&&!replay_registration_order&&rq_n==1&&r_frontend_calls==1);
        *m=0;finish_test_replay();*m=1;return;
    }
    assert(c==&rcv&&m==&rl&&*m&&stopping&&r_frontend_calls==simulated_submitter);
    assert(shutdown_hooks==2);waited++;r_frontend_calls=0;
}
static void plat_cond_broadcast(int *c){assert(c==&rcv&&rl);}
static int vk_sync_stopping(void){return stopping;}
static int vk_device_is_lost(void){return lost;}
static int plat_thread_spawn(void (*main)(void *),void *p,const char *name,int flags){
    (void)main;(void)p;(void)name;(void)flags;abort();
}
static _Noreturn void rt_fatal(const char *text,...){
    if(mesh_idle_failure&&strstr(text,"cannot retire mesh fallback resources without idle proof"))
        longjmp(mesh_failure,1);
    if(expect_fatal&&strstr(text,"vk:"))longjmp(allocation_failure,1);
    puts(text);abort();
}
static uint64_t perf_begin(void){return 0;}
static int core_away_unlocked(void){assert(rl&&protect_cold);return 0;}
static void core_back_unlocked(int *away){assert(rl&&protect_cold&&!*away);}
static void wait_counted(int why,uint64_t t0){assert(rl&&why==DR_REGISTRATION&&!t0&&protect_cold);registration_waits++;}
static uint64_t vk_sync_reserve(void){return ++next_fence;}
static void vk_sync_start(void){assert(!rl);}
static int vk_sync_prepare_draw(void){
    assert(!rl&&r_frontend_calls&&!registrations.alive);
    fixed_attempts++;
    return !fixed_blocked;
}
static int gfx_replay_async(void){return 1;}
static uint64_t nvn_submit_count(void){return 33;}
static void vk_dlssg_render_start(uint64_t frame){assert(worker&&frame==1&&!rl);}
static void perf_replay_thread(void){assert(worker&&!rl);}
static void present_image(const tex_t *t,const int *crop,uint64_t at,uint64_t queued,uint64_t frame,uint64_t value){
    (void)t;(void)crop;(void)at;(void)queued;(void)frame;(void)value;abort();
}
static void vk_texture_release_replayed(const tex_t *t){(void)t;abort();}
static uint64_t submit_sync(const uint64_t *h,int n){(void)h;(void)n;abort();}
static int vk_exec_gate_wait(const uint64_t *h,int n){
    assert(!rl&&r_frontend_calls&&h&&n==1);
    if(gate_closed){gate_steps=1;gate_closed--;return 0;}
    if(gate_steps)assert(gate_steps==4);
    return 1;
}
static int vk_usnap_project(const uint64_t *handles,int n,VkUsnapBindingState *candidate){
    assert(n==1&&!rl&&r_frontend_calls);
    candidate->slots[0].address=*handles;
    return !project_failure;
}
static int nvn_submission_ordered_scratch_max(const uint64_t *handles,int n,uint64_t *bytes){
    assert(handles&&n==1);*bytes=*handles;return 1;
}
static VkTextureRegistrationScope *vk_texture_registration_scope_begin(void){
    assert(r_frontend_calls&&!registrations.alive);
    registrations=(VkTextureRegistrationScope){1,registration_publication};
    return &registrations;
}
static int vk_texture_registration_scope_current(const VkTextureRegistrationScope *scope){
    assert(rl&&replay_registration_order&&scope->alive);
    return scope->publication==registration_publication;
}
static uint64_t vk_texture_registration_scope_publication(const VkTextureRegistrationScope *scope){
    assert(rl&&scope&&scope->alive);
    return scope->publication;
}
static void vk_texture_registration_scope_end(VkTextureRegistrationScope *scope){
    if(scope){assert(scope==&registrations&&scope->alive);scope->alive=0;scope_releases++;}
}
static VkResult vk_usnap_queue_bundle(const VkUsnapInputs *inputs,uint64_t bytes,
                                    VkReplayCaptureBundle *out,VkReplayCaptureWait *wait){
    assert(!rl&&r_frontend_calls&&inputs->handles&&inputs->nhandles==1&&!arena.alive);
    assert(inputs->ninherited==8*18&&bytes==*inputs->handles&&!out->snapshot&&!out->scratch);
    assert(!gate_closed&&registrations.alive&&budget_state==2&&budget_size==sizeof *inputs->handles);
    if(attempt_copy)assert(attempt_copy==inputs->handles&&attempt_allocations==budget_allocations);
    attempt_copy=inputs->handles;attempt_allocations=budget_allocations;
    if(drain_capture){drain_capture=0;capture_steps=1;wait->kind=VK_REPLAY_CAPTURE_WAIT_DRAIN_OLDER;return VK_NOT_READY;}
    if(producer_refused_once){
        producer_refused_once=0;wait->kind=VK_REPLAY_CAPTURE_WAIT_PRODUCER;wait->producer_epoch=42;return VK_NOT_READY;
    }
    observed_inherited=inputs->inherited[0].address;
    stopping=abort_before_copy||stop_after_copy;
    if(abort_before_copy)return VK_ERROR_INITIALIZATION_FAILED;
    if(capture_race){assert(!replay_registration_order);registration_publication++;capture_race=0;}
    copied_under_registration_order+=replay_registration_order;
    arena.alive=scratch.alive=1;
    *out=(VkReplayCaptureBundle){&arena,&scratch};
    return VK_SUCCESS;
}
static void vk_usnap_queue_bundle_cancel(VkReplayCaptureBundle *captures){
    if(captures->snapshot){assert(captures->snapshot==&arena&&arena.alive);arena.alive=0;capture_releases++;}
    if(captures->scratch){assert(captures->scratch==&scratch&&scratch.alive);scratch.alive=0;scratch_releases++;}
    *captures=(VkReplayCaptureBundle){0};
}
static uint64_t submit_sync_capture(const uint64_t *h,int n,VkReplayCaptureBundle *captures,
                                    VkTextureRegistrationScope *scope){
    assert(worker&&r_busy&&!rl&&!replay_registration_order&&!stopping&&!lost);
    assert(n==1&&h==attempt_copy&&*h==r_uniforms.slots[0].address&&budget_state==2);
    worker_submits++;
    vk_usnap_queue_bundle_cancel(captures);
    vk_texture_registration_scope_end(scope);
    return next_fence;
}
static void finish_test_replay(void){
    assert(!rl&&!replay_registration_order&&rq_n==1&&rq[rq_head].kind==RI_SUBMIT&&budget_state==2);
    worker=1;
    if(!setjmp(worker_done)){replay_main(NULL);abort();}
    worker=0;
    assert(!rl&&!rq_n&&!r_busy&&!budget_state&&r_submit_done==33&&!r_presents_done&&
           r_registration_done==r_registration_issued);
    rq_head=0;
    attempt_copy=NULL;
}
static int vk_texture_finalize_registration_ordered(const tex_t *texture,uint64_t queued_publication){
    assert(replay_registration_order&&!rl&&texture->mem&&r_frontend_calls==1);
    if(queued_publication && protect_cold){
        assert(rq_n==1&&rq[0].registrations->publication==queued_publication);
        assert(queued_publication==registration_publication);
        return 0;
    }
    assert(!protect_cold||registration_waits==1);
    registration_publication++;
    return 1;
}
static int vk_texture_release_ordered(const tex_t *texture,uint64_t publication){(void)texture;(void)publication;abort();}
static void vk_usnap_wait_producer(uint64_t epoch,uint32_t ms){
    assert(epoch==42&&ms&&!rl&&!replay_registration_order&&!registrations.alive&&!arena.alive&&!scratch.alive);
    producer_waits++;
}
static void vk_queue_lock(void){abort();}
static void vk_queue_unlock(void){abort();}
static void check_peak(const RcAdmissionVector *peak,uint64_t size){
    assert(size&&peak->amount[RC_BUDGET_RAM]==size&&peak->amount[RC_BUDGET_COMMIT]==size);
}
static void resource_core_budget_service(int required,const RcAdmissionVector *peak,RcBudgetProgress *out){
    assert(required&&!rl&&!replay_registration_order&&!budget_state&&!stopping&&!lost);
    check_peak(peak,sizeof(uint64_t));budget_reclaims++;
    out->event=314;
    if(budget_progress_now)budget_blocked=0;
}
static void resource_core_budget_readiness(int required,RcBudgetReadiness *out){
    assert(required&&!rl&&!replay_registration_order);
    out->ready=!budget_blocked;
}
static int resource_core_budget_peak_ready(const RcAdmissionVector *peak,const RcBudgetReadiness *now){
    check_peak(peak,sizeof(uint64_t));return now->ready;
}
static int resource_core_budget_peak_progress(const RcAdmissionVector *peak,const RcBudgetReadiness *old,
                                              const RcBudgetReadiness *now){(void)peak;(void)old;(void)now;abort();}
static void resource_core_budget_wait_since(uint64_t event,unsigned ms){
    assert(event==314&&ms&&!rl&&!replay_registration_order&&!budget_state);
    budget_waits++;
    if(budget_stop)stopping=1;
    else budget_blocked=0;
}
static RcAdmissionStatus resource_core_budget_split(RcBudgetTicket *joint,const RcAdmissionVector *peak,
                                                    RcBudgetProducer producer,RcBudgetTicket *out){
    (void)joint;(void)peak;(void)producer;(void)out;abort();
}
static RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer,int required,
                                                      const RcAdmissionVector *peak,RcBudgetTicket *out){
    assert(producer==RC_BUDGET_SNAPSHOT&&required&&!out->token.ledger&&!rl&&!replay_registration_order&&!budget_state);
    check_peak(peak,peak->amount[RC_BUDGET_RAM]);
    if(budget_exhausted)return RC_ADMISSION_NO_RECORDS;
    if(budget_blocked)return RC_ADMISSION_REFUSED;
    budget_size=peak->amount[RC_BUDGET_RAM];
    out->token.ledger=&budget_state;
    out->producer=producer;
    budget_state=1;return RC_ADMISSION_OK;
}
static RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *ticket,const RcAdmissionVector *actual){
    assert(budget_state==1&&ticket->token.ledger==&budget_state&&ticket->producer==RC_BUDGET_SNAPSHOT);
    check_peak(actual,budget_size);budget_allocations++;budget_state=2;return RC_ADMISSION_OK;
}
static RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket){
    assert(budget_state==1&&ticket->token.ledger==&budget_state&&!rl&&!replay_registration_order);
    *ticket=(RcBudgetTicket){0};budget_cancels++;budget_state=0;budget_size=0;return RC_ADMISSION_OK;
}
static RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *ticket){
    assert(budget_state==2&&ticket->token.ledger==&budget_state&&!rl&&!replay_registration_order);
    budget_retires++;budget_state=3;return RC_ADMISSION_OK;
}
static RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *ticket){
    assert(budget_state==3&&ticket->token.ledger==&budget_state&&!rl&&!replay_registration_order);
    *ticket=(RcBudgetTicket){0};budget_destroys++;budget_state=0;budget_size=0;return RC_ADMISSION_OK;
}
static int rc_admission_accounted(RcAdmissionStatus r){return r==RC_ADMISSION_OK;}
static void vk_sync_stop_submissions(void){assert(!shutdown_hooks++);stopping=1;}
static void vk_pipeline_stop_preparation(void){assert(shutdown_hooks++==1);}
static void vk_exec_draw_mfb_stop(void){assert(shutdown_hooks++==2&&!r_frontend_calls);}
static void replay_drain(int why){
    if(why==DR_GATE){assert(gate_steps++==1&&!registrations.alive);gate_drains++;return;}
    if(why==DR_CAPTURE){assert(capture_steps++==1&&!registrations.alive);return;}
    assert(why==DR_FINISH&&shutdown_hooks++==3&&!r_frontend_calls&&!rq_n);
}
static void vk_exec_discard_held(void){assert(shutdown_hooks++==4);}
static void gfx_exit_log(void){assert(shutdown_hooks++==5);}
static void vk_present_drain(void){
    if(capture_steps==2){capture_steps++;return;}
    if(!shutdown_hooks){assert(gate_steps++==2);return;}
    assert(shutdown_hooks++==6);
}
static void vk_sync_finish(void){
    if(capture_steps==3){capture_steps++;gate_closed=1;return;}
    assert(gate_steps++==3);
}
static void vk_gpu_passes_stop_capture(void){assert(shutdown_hooks++==7);}
static void vk_sync_shutdown(void){assert(shutdown_hooks++==8);}
static void vk_pipeline_shutdown(void){assert(shutdown_hooks++==9&&!lost);lost=loss_in_pipeline_shutdown;}
static void gfx_window_release_vk(void){assert(shutdown_hooks++==10&&!lost);lost=loss_in_window_release;}
static VkResult vk_meshfb_shutdown(void){
    assert(shutdown_hooks++==11&&!lost);
    return mesh_idle_failure ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS;
}
static void vk_device_lost_check(VkResult result,const char *operation){
    assert(mesh_idle_failure&&result==VK_ERROR_INITIALIZATION_FAILED&&strstr(operation,"mesh fallback"));
    mesh_failure_reported++;
}
static void vk_exec_draw_mfb_shutdown(void){assert(shutdown_hooks++==12&&!lost);}
static void vk_exec_draw_native_mesh_shutdown(void){assert(shutdown_hooks++==13&&!lost);}
static void vk_dlaa_size_shutdown(void){assert(shutdown_hooks++==14&&!lost);}
static void vk_dlaa_hud_shutdown(void){assert(shutdown_hooks++==15&&!lost);}
static void vk_usnap_shutdown(void){assert(shutdown_hooks++==16&&!arena.alive);}
static void vk_exec_draw_scratch_shutdown(void){assert(shutdown_hooks++==17);}
static void vk_exec_draw_descriptor_shutdown(void){assert(shutdown_hooks++==18&&!lost);}
static void vk_exec_draw_bindless_shutdown(void){assert(shutdown_hooks++==19&&!lost);}
static void vk_exec_draw_fallback_shutdown(void){assert(shutdown_hooks++==20&&!lost);}
static void vk_exec_draw_fixed_shutdown(void){assert(shutdown_hooks++==21);}
static void vk_mem_shutdown(void){assert(shutdown_hooks++==22);}
static void vk_shutdown(void){assert(shutdown_hooks++==23);}
#define malloc metadata_malloc
''' + accounting + r'''
#undef malloc
''' + selected + r'''
int main(void){
    uint64_t handle=71;
    fixed_blocked=1;
    assert(!vk_be_submit(&handle,1)&&fixed_attempts==1&&!rq_n&&!next_fence&&!r_frontend_calls);
    assert(!scope_releases&&!capture_releases&&!registrations.alive&&!r_uniforms.slots[0].address);
    assert(!budget_state&&!budget_allocations);
    fixed_blocked=0;

    ReplayItem item={.kind=RI_SUBMIT};
    assert(replay_handles_clone(&item,NULL)&&!item.h&&!item.handle_budget.token.ledger);
    replay_handles_release(&item);
    assert(!budget_allocations&&!budget_destroys);
    uint64_t handles[]={5,3,7};
    item.n=3;
    assert(replay_handles_clone(&item,handles)&&item.h!=handles&&!memcmp(item.h,handles,sizeof handles));
    assert(budget_state==2&&budget_size==sizeof handles);
    replay_handles_release(&item);
    assert(!item.h&&!item.handle_budget.token.ledger&&!budget_state);

    expect_fatal=budget_exhausted=1;
    if(!setjmp(allocation_failure)){replay_handles_clone(&item,handles);abort();}
    assert(!item.h&&!item.handle_budget.token.ledger&&!budget_state&&!budget_reclaims);
    budget_exhausted=0;fail_malloc=1;
    if(!setjmp(allocation_failure)){replay_handles_clone(&item,handles);abort();}
    assert(!item.h&&!item.handle_budget.token.ledger&&!budget_state&&budget_cancels==1);
    fail_malloc=0;expect_fatal=0;

    budget_stop=budget_blocked=1;
    assert(!vk_be_submit(&handle,1)&&stopping&&!budget_state&&!rq_n&&!next_fence&&!r_frontend_calls);
    assert(budget_allocations==1&&budget_destroys==1&&budget_waits==1&&budget_reclaims==1);
    budget_stop=budget_blocked=stopping=0;

    abort_before_copy=1;
    assert(!vk_be_submit(&handle,1)&&!rq_n&&!next_fence&&!r_frontend_calls);
    assert(scope_releases==1&&!capture_releases&&!registrations.alive&&!budget_state);
    attempt_copy=NULL;abort_before_copy=stopping=0;stop_after_copy=1;
    assert(!vk_be_submit(&handle,1)&&!rq_n&&!next_fence&&!r_frontend_calls);
    assert(scope_releases==2&&capture_releases==1&&scratch_releases==1&&!registrations.alive&&!arena.alive&&!scratch.alive);
    assert(!vk_be_submit(&handle,1)&&scope_releases==2); // Late entry never starts capture.
    assert(!r_uniforms.slots[0].address&&!budget_state); // A rejected submission never advances FIFO state.
    assert(budget_allocations==3&&budget_retires==3&&budget_destroys==3);

    attempt_copy=NULL;stop_after_copy=stopping=0;
    budget_blocked=budget_progress_now=1;
    gate_closed=1;
    assert(vk_be_submit(&handle,1)==1&&rq_n==1&&!r_frontend_calls);
    assert(gate_steps==4&&gate_drains==1&&!observed_inherited&&r_uniforms.slots[0].address==handle);
    assert(rq[0].h!=&handle&&rq[0].h[0]==handle&&rq[0].submit_no==33);
    assert(rq[0].captures.snapshot==&arena&&rq[0].captures.scratch==&scratch&&rq[0].registrations==&registrations);
    assert(budget_state==2&&rq[0].handle_budget.token.ledger&&budget_reclaims==2&&budget_waits==1);
    finish_test_replay();
    assert(worker_submits==1&&budget_destroys==4);
    handle=75;
    drain_capture=1;producer_refused_once=1;
    assert(vk_be_submit(&handle,1)==2&&observed_inherited==71&&r_uniforms.slots[0].address==75&&capture_steps==4);
    assert(gate_steps==4&&gate_drains==2&&scope_releases==5&&producer_waits==1);
    assert(budget_allocations==5&&budget_destroys==4);
    stopping=1;finish_test_replay();stopping=0;
    assert(worker_submits==1&&budget_destroys==5);

    handle=81;capture_race=1;
    assert(vk_be_submit(&handle,1)==3&&r_registration_issued==3);
    assert(rq[0].registrations->publication==registration_publication&&copied_under_registration_order==1);
    assert(!replay_registration_order&&capture_releases==4&&scope_releases==7);
    uint64_t older_publication=registration_publication;
    protect_cold=1;
    tex_t texture={.mem=&handle};
    vk_be_texture_release(&texture);
    assert(registration_waits==1&&r_registration_done==3&&registration_publication==older_publication+1);
    assert(rq_n==1&&rq[0].kind==RI_RELEASE&&rq[0].tex!=&texture&&rq[0].tex->mem==texture.mem);
    protect_cold=0;
    handle=87;
    assert(vk_be_submit(&handle,1)==4&&rq_n==2&&r_registration_issued==4);
    assert(rq[1].registrations->publication==registration_publication);
    free(rq[0].tex);rq[0]=(ReplayItem){0};rq_head=1;rq_n=1;
    lost=1;finish_test_replay();lost=0;
    assert(worker_submits==2&&budget_allocations==7&&budget_retires==7&&budget_destroys==7&&!budget_state);

    expect_fatal=project_failure=1;r_frontend_calls=1;
    if(!setjmp(allocation_failure)){replay_submit(&handle,1);abort();}
    assert(!rq_n&&!budget_state&&budget_allocations==8&&budget_retires==8&&budget_destroys==8);
    expect_fatal=project_failure=0;r_frontend_calls=0;

    r_frontend_calls=simulated_submitter=1;
    vk_backend_shutdown();
    assert(shutdown_hooks==24&&waited==1);
    uint64_t final_publication=registration_publication;
    vk_be_texture_release(&texture);
    assert(!rq_n&&!r_frontend_calls&&registration_publication==final_publication);
    shutdown_hooks=0;lost=1;
    vk_backend_shutdown();
    assert(shutdown_hooks==9&&waited==1); // Unproven GPU owners remain on device loss.

    shutdown_hooks=lost=0;loss_in_pipeline_shutdown=1;
    vk_backend_shutdown();
    assert(shutdown_hooks==10&&waited==1&&lost); // Loss while draining pipelines prevents later teardown.

    shutdown_hooks=lost=loss_in_pipeline_shutdown=0;loss_in_window_release=1;
    vk_backend_shutdown();
    assert(shutdown_hooks==11&&waited==1&&lost); // A lost window generation also blocks memory teardown.

    shutdown_hooks=lost=loss_in_window_release=0;mesh_idle_failure=1;
    if(!setjmp(mesh_failure)) { vk_backend_shutdown(); abort(); }
    assert(shutdown_hooks==12&&mesh_failure_reported==1); // A failed mesh idle check blocks later destruction.
    puts("replay handle RAM/commit, refusal, FIFO retry, worker retirement and shutdown ownership: ok");
}
'''

cc = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
work = ROOT / "build/test"
work.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="replay-submission-ownership-", dir=work) as directory:
    out = Path(directory)
    src, exe = out / "check.c", out / "check.exe"
    src.write_text(fixture)
    subprocess.run([str(cc), "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-fsanitize-trap=all", str(src), "-o", str(exe)], check=True, env=env)
    subprocess.run([str(exe)], check=True, env=env)
