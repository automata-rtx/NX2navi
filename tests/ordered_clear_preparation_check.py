"""Exercise ordered clear pressure with the production allocator prerequisite query and replay edge."""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXEC = (ROOT / "runtime/vk_exec.c").read_text()
MEMORY = (ROOT / "runtime/vk_memory.c").read_text()


def definition(source, signature):
    start = source.index(signature)
    return source[start:source.index("\n}\n", start) + 3]


PREFIX = r'''
#include <assert.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resource_core_budget.h"
typedef struct { uint32_t memory_type; int selected, owner_full, can_reuse; RcAdmissionVector peak;
                 RcBudgetReadiness before; } RcVmaDependency;
typedef struct { uint32_t type; int custom_pool; } Allocation;
typedef struct { unsigned readers; } VkMemoryImageHold;
typedef struct { Allocation *alloc; uint64_t tl, seq; const VkMemoryImageHold *image_hold; } Dead;
static Dead dead[8];
static int ndead, cmd_once, cmd_lock;
static uint64_t done_seq = 16, handed = 8, done;
static uint64_t vk_sync_handed_fence(void) { return handed; }
static void cmd_init(void) {}
static void plat_once(int *once, void (*fn)(void), void *ignored) { assert(once && !ignored); fn(); }
static void plat_rmutex_lock(int *lock) { assert(!*lock); *lock = 1; }
static void plat_rmutex_unlock(int *lock) { assert(*lock); *lock = 0; }
static int resource_core_vma_dependency_retirement(const RcVmaDependency *dependency, const Allocation *allocation) {
    assert(cmd_lock);
    return dependency->can_reuse && !allocation->custom_pool && allocation->type == dependency->memory_type;
}

typedef struct { uint64_t fence, resource_next, resource_op; } VkExec;
typedef struct { uint64_t scene, op; } RcRegistryPosition;
typedef struct { uint64_t serial, mapping; } Tex;
typedef struct { const Tex *tex; } p_clear_tex;
typedef p_clear_tex p_copy_b2t;
enum { OP_COPY_B2T = 1, RC_REGISTRY_INVALID = 2 };
typedef int RcRegistryResult;
typedef struct { uint64_t serial, mapping; } VkTextureWriteTarget;
typedef struct { VkTextureWriteTarget *target; } VkExecClearTexture;
typedef struct { VkTextureWriteTarget *target; } VkExecBufferToTexture;
typedef struct { VkTextureWriteTarget *target; struct { int sType; } scratch_recipe; } VkExecTextureToTexture;
typedef enum {VK_SUCCESS, VK_NOT_READY, VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_OUT_OF_HOST_MEMORY,
              VK_ERROR_DEVICE_LOST, VK_ERROR_FEATURE_NOT_PRESENT, VK_ERROR_INITIALIZATION_FAILED} VkResult;
typedef enum {VK_SYNC_PROGRESS_UNAVAILABLE,VK_SYNC_PROGRESS_COMPLETE,VK_SYNC_PROGRESS_STOPPED} VkSyncProgressResult;
static VkSyncProgressResult vk_exec_texture_capacity(VkExec *,const RcVmaDependency *,VkResult,
    int (*spare)(void *),void *,uint64_t *);
enum { RC_REGISTRY_OK, RC_REGISTRY_STALE, RC_REGISTRY_CAPACITY };
enum {RECORDING_RESOURCE_UNAVAILABLE=1};
static int recording_failed;
static int vk_exec_recording_failed(void){return recording_failed;}
static void rt_log(const char *text,...){assert(text);}
static int preserve_alias, preserve_refused, preservations, shared_image_pressure;
static VkExec *recording, *x_cur;
static int rw_chunk;
enum { PW_OP };
static VkTextureWriteTarget retained;
static Tex texture;
static VkResult refusal;
static VkSyncProgressResult wait_result;
static jmp_buf terminal;
static int declared, captures, attempts, waits, collected, canceled, lost, stopped, relevant, handled, invalid, failures;
static int stop_after_allocation, wait_changes_mapping, gained_externally;
static int stale_native_source, stale_at_prerequisite, touched_source;
static int vk_sync_stopping(void) { return stopped; }
static int vk_device_is_lost(void) { return lost; }
static void vk_device_lost_check(VkResult result, const char *context_name) {
    assert(result == VK_ERROR_DEVICE_LOST && context_name);
    lost = 1;
}
static void rt_fatal(const char *message, ...) {
    assert(message);
    failures++;
    longjmp(terminal, 1);
}
static RcRegistryPosition vk_exec_resource_position(void) {
    assert(recording->resource_op == recording->resource_next + 1);
    return (RcRegistryPosition){recording->fence, recording->resource_op * 2};
}
static void *rc_registry_runtime(void) { return recording; }
static int rc_registry_declare_through(void *registry, RcRegistryPosition position) {
    assert(registry == recording && position.scene == 9 && position.op == 16 && !attempts);
    declared++;
    return RC_REGISTRY_OK;
}
static int vk_exec_clear_alias_candidate(const p_clear_tex *op) {return op->tex==&texture&&preserve_alias;}
static int vk_exec_buffer_to_texture_alias_candidate(VkExec *ex,const p_copy_b2t *op) {
    assert(ex==recording);return vk_exec_clear_alias_candidate(op);
}
static int prepare_bound_texture_relocation(VkExec *ex,const Tex *t,int ordered) {
    assert(ex==recording&&t==&texture&&ordered&&declared==1&&!captures&&!attempts);
    assert(ex->resource_op==8);preservations++;return preserve_refused ? -1 : 0;
}
static int vk_exec_clear_capture(const p_clear_tex *op, VkExecClearTexture *clear) {
    assert(declared && recording->resource_next == 7 && recording->resource_op == 8 && !clear->target);
    captures++;
    if (handled && !invalid) {
        retained = (VkTextureWriteTarget){op->tex->serial, op->tex->mapping};
        clear->target = &retained;
    }
    return handled;
}
static VkResult vk_texture_write_target_allocate(VkTextureWriteTarget *target, RcVmaDependency *dep) {
    assert(target == &retained && target->serial == 101 && target->mapping == 44);
    assert(recording->resource_next == 7 && !recording->resource_op && !cmd_lock);
    *dep = (RcVmaDependency){.selected=1,.memory_type=0,.can_reuse=1};
    dep->before.records_available=1;
    if (shared_image_pressure) dep->peak.amount[RC_BUDGET_GPU_FIRST]=64;
    if (stop_after_allocation) stopped = 1;
    return ++attempts == 1 ? refusal : VK_SUCCESS;
}
static VkResult vk_exec_texture_to_texture_allocate(VkExecTextureToTexture *copy, RcVmaDependency *dep) {
    assert(copy->scratch_recipe.sType);
    return vk_texture_write_target_allocate(copy->target, dep);
}
static int vk_texture_write_target_current(const VkTextureWriteTarget *target) {
    assert(target == &retained && target->serial == 101 && target->mapping == 44);
    return 1; // an address remap need not change the retained writable version
}
static int vk_exec_buffer_to_texture_capture(VkExec *ex, const p_copy_b2t *op, VkExecBufferToTexture *copy) {
    assert(ex == recording && !copy->target);
    VkExecClearTexture clear = {0};
    int r = vk_exec_clear_capture(op, &clear);
    copy->target = clear.target;
    return r;
}
static int vk_exec_buffer_to_texture_source_current(const VkExecBufferToTexture *copy) {
    assert(copy->target == &retained);
    return !stale_native_source;
}
static int vk_exec_buffer_to_texture_prepare_source(VkExec *ex, VkExecBufferToTexture *copy) {
    assert(ex == recording && ex->resource_op==ex->resource_next+1);
    assert(copy->target == &retained && !stale_native_source && !cmd_lock);
    touched_source++;
    return RC_REGISTRY_OK;
}
static void vk_exec_required_graphics_unavailable(VkExec *ex,int opcode,const char *reason) {
    assert(ex==recording && opcode==OP_COPY_B2T && reason);
    recording_failed=RECORDING_RESOURCE_UNAVAILABLE;
}
static uint64_t vk_sync_fence_done(void) { return done; }
static VkSyncProgressResult vk_sync_progress_wait_prior(uint64_t supplier) {
    assert(supplier == 5 && supplier < recording->fence && !cmd_lock);
    waits++;
    if (wait_result != VK_SYNC_PROGRESS_COMPLETE) return wait_result;
    done = supplier;
    if (wait_changes_mapping) texture.mapping = 55;
    if (stale_at_prerequisite) stale_native_source = 1;
    return VK_SYNC_PROGRESS_COMPLETE;
}
static void vk_mem_collect(void) {
    assert(!cmd_lock);
    collected++;
    relevant = done == 5 || gained_externally;
}
void resource_core_budget_readiness(int required, RcBudgetReadiness *out) {
    assert(required && out);
    *out = (RcBudgetReadiness){0};
}
static int resource_core_vma_dependency_ready(const RcVmaDependency *d, const RcBudgetReadiness *budget) {
    assert(d->selected && budget);
    return relevant;
}
static void vk_texture_write_target_cancel(VkTextureWriteTarget *target) {
    assert(target == &retained);
    canceled++;
}
static void vk_exec_buffer_to_texture_cancel(VkExecBufferToTexture *copy) {
    vk_texture_write_target_cancel(copy->target);
    copy->target = NULL;
}
typedef VkTextureWriteTarget VkTextureRestore;
typedef void VkTexImage;
typedef void VkTextureAliasRead;
typedef struct { uint64_t backing,offset,size; } RcSourceRange;
typedef struct { uint64_t address,mapping; RcSourceRange range; } RcInterval;
enum { VK_TEXTURE_ALIAS_MAPPING, VK_TEXTURE_ALIAS_CAPACITY };
typedef struct { int phase,status;RcInterval receiver;} VkTextureAliasFailure;
typedef struct {VkTextureRestore *texture;const VkTexImage *image;VkTextureAliasRead *alias;uint64_t destination;} VkExecRestoreCopy;
static int upload_waits, budget_waits, published, lost_destination, losing_destination, ready_after_upload;
static int upload_parked, stop_during_upload;
static int already_affordable;
static uint64_t ready_upload;
static int vk_exec_restore_copy_capture(VkExec *ex,const p_copy_b2t *op,VkExecRestoreCopy *copy,VkTextureAliasFailure *failure){
 assert(failure&&!failure->status);
 assert(ex==recording && declared && ex->resource_op==8 && !copy->texture);
 captures++;if(!handled)return 0;
 retained=(VkTextureRestore){op->tex->serial,op->tex->mapping};
 *copy=(VkExecRestoreCopy){.texture=&retained,.destination=900};return 1;
}
static int vk_exec_restore_copy_current(const p_copy_b2t *op,const VkExecRestoreCopy *copy){
 assert(op->tex==&texture && copy->destination==900);return !lost_destination;
}
static VkResult vk_texture_restore_prepare(VkTextureRestore *input,RcVmaDependency *dep){
 assert(input==&retained && input->mapping==44 && !recording->resource_op && recording->resource_next==7);
 *dep=(RcVmaDependency){.selected=1,.can_reuse=1,.peak.amount[RC_BUDGET_RAM]=64};
 if(already_affordable)dep->before=(RcBudgetReadiness){.records_available=1,.usable_mask=1u<<RC_BUDGET_RAM,
                                                       .available.amount[RC_BUDGET_RAM]=64};
 return ++attempts==1?refusal:VK_SUCCESS;
}
static int vk_texture_restore_spare_ready(void *input){assert(input==&retained);return relevant;}
static uint64_t vk_texture_restore_upload_supplier(const VkTextureRestore *input,uint64_t after){
 assert(input==&retained);return ready_upload>after?ready_upload:0;
}
static int vk_staging_prepare_wait(uint64_t upload){
 assert(upload==ready_upload && upload_parked && !cmd_lock && !recording->resource_op);upload_waits++;
 if(stop_during_upload){stopped=1;return 0;}
 texture.mapping=55;lost_destination=losing_destination;relevant=ready_after_upload;return 1;
}
static void group_drain(VkExec *ex,int why){
 assert(ex==recording && why==PW_OP && !upload_parked && !cmd_lock && !ex->resource_op && ex->resource_next==7);
}
static VkSyncProgressResult vk_sync_progress_wait_upload_staging(uint64_t upload){
 upload_parked=1;
 int ready=vk_staging_prepare_wait(upload);
 upload_parked=0;
 if(stopped)return VK_SYNC_PROGRESS_STOPPED;
 return ready?VK_SYNC_PROGRESS_COMPLETE:VK_SYNC_PROGRESS_UNAVAILABLE;
}
static VkSyncProgressResult vk_sync_progress_wait_capacity(const RcVmaDependency *dep,int (*spare)(void *),void *input){
 assert(dep->selected && !spare(input));budget_waits++;
 if (shared_image_pressure) {
    assert(dep->peak.amount[RC_BUDGET_GPU_FIRST]==64 && !cmd_lock && !recording->resource_op);
    if (stale_at_prerequisite) {
        stale_native_source=1;
        assert(spare(input));
    } else relevant=1;
    return wait_result;
 }
 if(losing_destination){lost_destination=1;assert(spare(input));}
 return wait_result;
}
static const void *vk_texture_restore_publish(VkTextureRestore *input){
 assert(input==&retained && !recording->resource_op && recording->resource_next==7);
 published++;if(stop_after_allocation)stopped=1;return input;
}
static void vk_exec_restore_copy_cancel(VkExecRestoreCopy *copy){
 assert(copy->texture==&retained);canceled++;*copy=(VkExecRestoreCopy){0};
}
'''

MAIN = r'''
static Allocation match = {.type=0}, other_type = {.type=1}, custom = {.type=0,.custom_pool=1};
static const VkMemoryImageHold live_reader = {.readers=1};
static void reset(VkExec *ex) {
    *ex = (VkExec){.fence=9, .resource_next=7};
    recording = x_cur = ex;
    texture = (Tex){101,44};
    done = declared = captures = attempts = waits = collected = canceled = lost = stopped = relevant = failures = 0;
    wait_changes_mapping = stop_after_allocation = gained_externally = invalid = 0;
    stale_native_source = stale_at_prerequisite = touched_source = 0;
    handled = 1;
    refusal = VK_NOT_READY;
    wait_result = VK_SYNC_PROGRESS_COMPLETE;
    ndead = 0;
    upload_waits=budget_waits=published=lost_destination=losing_destination=ready_after_upload=already_affordable=recording_failed=0;
    ready_upload=upload_parked=stop_during_upload=0;
    preserve_alias=preserve_refused=preservations=shared_image_pressure=0;
}
int main(void) {
    VkExec ex;
    VkExecClearTexture clear = {0};
    const p_clear_tex op = {&texture};
    reset(&ex);
    dead[0] = (Dead){&match,9,16,NULL};       // current final fence cannot supply progress
    dead[1] = (Dead){&match,8,17,NULL};       // upload prerequisite has not completed
    dead[2] = (Dead){&other_type,7,16,NULL};  // same heap but no actual compatible block slot
    dead[3] = (Dead){&custom,6,16,NULL};      // same memory type in an unrelated fixed pool
    dead[4] = (Dead){&match,8,16,&live_reader}; // Ending its scene cannot release an independent native reader.
    dead[5] = (Dead){&match,5,16,NULL};       // actual compatible earlier final scene
    ndead = 6;
    wait_changes_mapping = 1;
    assert(prepare_ordered_clear(&ex, &op, &clear) == 1);
    assert(wait_changes_mapping && texture.mapping == 55 && retained.mapping == 44);
    assert(captures == 1 && attempts == 2 && waits == 1 && declared == 1 && collected == 1 && !canceled);
    assert(ex.resource_next == 7 && !ex.resource_op);

    clear = (VkExecClearTexture){0}; reset(&ex); ndead = 5;
    if (!setjmp(terminal)) { prepare_ordered_clear(&ex, &op, &clear); assert(0); }
    assert(failures == 1 && attempts == 1 && !waits && collected == 1 && canceled == 1);

    clear = (VkExecClearTexture){0}; reset(&ex); gained_externally = 1;
    assert(prepare_ordered_clear(&ex, &op, &clear) == 1);
    assert(captures == 1 && attempts == 2 && collected == 1 && !waits); // relevant release raced the query

    clear = (VkExecClearTexture){0}; reset(&ex); preserve_alias=preserve_refused=1;
    assert(prepare_ordered_clear(&ex,&op,&clear)==-1 && recording_failed && preservations==1);
    assert(!clear.target&&!attempts&&!captures&&!ex.resource_op);
    reset(&ex); preserve_alias=1; shared_image_pressure=1;
    assert(prepare_ordered_clear(&ex,&op,&clear)==1 && clear.target==&retained && preservations==1);
    assert(budget_waits==1&&attempts==2&&!waits&&collected==1&&relevant);
    clear = (VkExecClearTexture){0}; reset(&ex); ndead = 6; wait_result = VK_SYNC_PROGRESS_STOPPED;
    assert(prepare_ordered_clear(&ex, &op, &clear) == -1 && !clear.target);
    assert(captures == 1 && attempts == 1 && waits == 1 && canceled == 1 && !collected);
    clear = (VkExecClearTexture){0}; reset(&ex); refusal = VK_SUCCESS; stop_after_allocation = 1;
    assert(prepare_ordered_clear(&ex, &op, &clear) == -1 && !clear.target && canceled == 1);
    clear = (VkExecClearTexture){0}; reset(&ex); refusal = VK_ERROR_DEVICE_LOST;
    assert(prepare_ordered_clear(&ex, &op, &clear) == -1 && lost && canceled == 1 && !collected);

    clear = (VkExecClearTexture){0}; reset(&ex); invalid = 1;
    assert(prepare_ordered_clear(&ex, &op, &clear) == 1 && !clear.target && !attempts);
    reset(&ex); handled = 0;
    assert(!prepare_ordered_clear(&ex, &op, &clear) && !clear.target && !attempts);
    VkExecBufferToTexture incoming={0};
    reset(&ex); ndead=6;
    assert(prepare_ordered_buffer_to_texture(&ex,&op,&incoming)==1 && incoming.target==&retained);
    assert(captures==1 && attempts==2 && waits==1 && touched_source==1 && ex.resource_next==7 && !ex.resource_op);
    vk_exec_buffer_to_texture_cancel(&incoming);
    reset(&ex); ndead=6; stale_at_prerequisite=1;
    assert(prepare_ordered_buffer_to_texture(&ex,&op,&incoming)==-1 && !incoming.target && recording_failed);
    assert(captures==1 && attempts==1 && waits==1 && !touched_source && canceled==1);
    reset(&ex); shared_image_pressure=stale_at_prerequisite=1;
    assert(prepare_ordered_buffer_to_texture(&ex,&op,&incoming)==-1 && recording_failed);
    assert(budget_waits==1&&captures==1&&attempts==1&&!touched_source&&canceled==1);
    reset(&ex); stop_after_allocation=1; refusal=VK_SUCCESS;
    assert(prepare_ordered_buffer_to_texture(&ex,&op,&incoming)==-1 && !incoming.target && !recording_failed);
    assert(!touched_source && canceled==1);
    reset(&ex); ndead=6; retained=(VkTextureWriteTarget){101,44};
    VkExecTextureToTexture overlap={.target=&retained,.scratch_recipe={1}};
    VkResult terminal_result;
    assert(prepare_ordered_write_target(&ex,&retained,NULL,&overlap,&terminal_result)==VK_SYNC_PROGRESS_COMPLETE);
    assert(attempts==2 && waits==1 && ex.resource_next==7 && !ex.resource_op);
    VkExecRestoreCopy copy={0};
    reset(&ex);ready_upload=31;ready_after_upload=1;
    assert(prepare_ordered_texture_copy(&ex,&op,&copy)==1 && captures==1 && attempts==2 && published==1);
    assert(upload_waits==1 && !budget_waits && texture.mapping==55 && retained.mapping==44 && !recording_failed);
    vk_exec_restore_copy_cancel(&copy);
    reset(&ex);ready_upload=31;ready_after_upload=1;losing_destination=1;
    assert(prepare_ordered_texture_copy(&ex,&op,&copy)==-1 && recording_failed && canceled==1 && !published && attempts==1);
    assert(upload_waits==1 && !budget_waits && !upload_parked && ex.resource_next==7);
    reset(&ex);ready_upload=31;stop_during_upload=1;
    assert(prepare_ordered_texture_copy(&ex,&op,&copy)==-1 && !recording_failed && canceled==1 && !published && attempts==1);
    assert(upload_waits==1 && !budget_waits && !upload_parked && ex.resource_next==7);
    reset(&ex);ready_upload=31;already_affordable=1;
    if(!setjmp(terminal)){prepare_ordered_texture_copy(&ex,&op,&copy);assert(0);}
    assert(upload_waits==1 && !ready_after_upload && attempts==1 && !published && !budget_waits && canceled==1);
    reset(&ex);wait_result=VK_SYNC_PROGRESS_STOPPED;
    assert(prepare_ordered_texture_copy(&ex,&op,&copy)==-1 && budget_waits==1 && canceled==1 && !published);
    reset(&ex);losing_destination=1;
    assert(prepare_ordered_texture_copy(&ex,&op,&copy)==-1 && budget_waits==1 && canceled==1 &&
           attempts==1 && !published && recording_failed);
    reset(&ex);refusal=VK_SUCCESS;stop_after_allocation=1;
    assert(prepare_ordered_texture_copy(&ex,&op,&copy)==-1 && published==1 && canceled==1 && !copy.texture);
    puts("ordered texture restore: exact handed staging, retained input, remap and stop cancellation passed");
    puts("ordered clear and transfers: useful prior, shared texture giveback, exact source cancellation and no-effect boundary passed");
}
'''

memory = definition(MEMORY, "uint64_t vk_mem_progress_prior(")
capacity_start=EXEC.index("typedef struct {\n    VkTextureWriteTarget *target;\n    const VkExecBufferToTexture *buffer;")
capacity_end=EXEC.index("} TextureWriteCapacity;",capacity_start) + len("} TextureWriteCapacity;\n")
prepare = EXEC[capacity_start:capacity_end] + "\n".join(definition(EXEC, signature) for signature in (
    "static int texture_write_input_current(", "static int texture_write_input_changed(",
    "static void image_write_input_position(", "static VkSyncProgressResult prepare_ordered_write_target(",
    "static int prepare_ordered_clear(", "static int prepare_ordered_buffer_to_texture("))
header = (ROOT / "runtime/vk_exec.h").read_text()
types = header[header.index("typedef enum {\n    VK_EXEC_RESTORE_READY"):header.index("VkExecRestoreResult vk_exec_prepare_texture_restore(")]
restore_start = EXEC.index("static int restore_progress_ready(")
copy_current = EXEC.index("static int restore_copy_current(", restore_start)
restore_end = EXEC.index("\n}\n", copy_current) + 3
progress_start = EXEC.index("static VkSyncProgressResult upload_staging_progress(")
progress_end = EXEC.index("static void progress_adopt_tail(", progress_start)
restore = EXEC[progress_start:progress_end] + EXEC[restore_start:restore_end] + r'''
static const RcInterval *vk_texture_alias_read_intervals(const VkTextureAliasRead *alias,uint32_t *count){
 (void)alias;(void)count;abort();
}
VkExecRestoreResult vk_exec_prepare_alias_texture_read(VkExec *ex,VkTextureAliasRead *const *alias,
    uint32_t count,int (*current)(void *),void *scope,const VkTexImage **image){
 (void)ex;(void)alias;(void)count;(void)current;(void)scope;(void)image;abort();
}
''' + definition(EXEC, "static int prepare_ordered_texture_copy(")
walk = EXEC[EXEC.index('static const op_hdr *walk(VkExec *ex,', EXEC.index("static int prepare_ordered_clear(")):]
write = 'resource_declare_texture_write(ex, h->op, target)'
assert walk.index('exact_clear = prepare_ordered_clear') < walk.index(write) < walk.index('vk_exec_clear_commit(ex, &clear)')
assert walk.index('exact_b2t = prepare_ordered_buffer_to_texture') < walk.index(write) < walk.index('vk_exec_buffer_to_texture_commit(ex, &b2t)')
assert walk.index('exact_t2t = prepare_ordered_texture_to_texture') < walk.index(write) < walk.index('vk_exec_texture_to_texture_commit(ex, &t2t)')
declare_t2b = 'vk_exec_effects_declare_texture_buffer(ex, a, &restored,'
commit_t2b = walk.index('vk_exec_restore_copy_commit(ex, a, &restored)')
assert (walk.index('prepare_ordered_texture_copy(ex, a, &restored)')
        < walk.index('vk_exec_restore_copy_prepare(ex, a, &restored)')
        < walk.index(declare_t2b)
        < commit_t2b
        < walk.index('vk_exec_restore_copy_cancel(&restored)', commit_t2b))
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    source, binary = d / 'clear_check.c', d / 'clear_check.exe'
    source.write_text(PREFIX + memory + prepare + types + restore + MAIN)
    cc = Path(os.environ.get("CC", "clang"))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    subprocess.run([str(cc), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=undefined', '-fsanitize-trap=all', '-I', str(ROOT / 'runtime'),
                    str(source), '-o', str(binary)], env=env, check=True)
    subprocess.run([str(binary)], env=env, check=True)
