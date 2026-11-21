"""Exercise the production ordered pool grant, exact restore and native lifetime handoff."""
import os
from pathlib import Path
import subprocess
import tempfile

from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.getenv('ORDERED_POOL_DRAW_SOURCE', ROOT / 'runtime/vk_exec_draw.c')).read_text()
EXEC_SOURCE = (ROOT / 'runtime/vk_exec.c').read_text()
CC = Path(os.environ['CC'])

PREFIX = r'''
#include <assert.h>
#include <stdint.h>
static uint32_t pool_incompatible_comparison_kinds(uint32_t id,uint32_t kinds);
#define VK_GRAPHICS_INACTIVE_POOL_ATTESTED UINT64_C(1)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#define VK_POOL_TEX 4
#define VK_POOL_KINDS 16
#define VK_POOL_SAMP 2
#define VK_POOL_COMPARISON_BINDING 18
#define POOL_IMAGE_ENTRIES (VK_POOL_KINDS * VK_POOL_TEX)
#define POOL_DESCRIPTOR_ENTRIES (POOL_IMAGE_ENTRIES + 2 * VK_POOL_SAMP)
#define POOL_LOG 8
#define GFX_SHADER_POOL_INDEX_LIMIT 4
#define VK_EXEC_ALIAS_READ_LIMIT 2
#define POOL_SETS 4
#define VK_NULL_HANDLE 0
#define GFX_RES_TEX 0
#define VK_SUCCESS 0
#define VK_NOT_READY 1
#define VK_ERROR_INITIALIZATION_FAILED -3
#define RWT_REGION_CALL(t, code) (code)
enum {OP_DRAW=25, OP_DRAW_IDX, OP_DRAW_IDX_IND, OP_DRAW_MESH, OP_DRAW_MESH_IND};
enum {NVN_INDEX_TYPE_UNSIGNED_BYTE, NVN_INDEX_TYPE_UNSIGNED_SHORT, NVN_INDEX_TYPE_UNSIGNED_INT};
typedef struct {int prim, count;} p_draw;
typedef struct {int prim, count, itype; uint64_t ibuf;} p_draw_idx;
typedef struct {int prim, itype; uint64_t ind, ibuf;} p_draw_idx_ind;
typedef struct {uint64_t ind;} p_draw_mesh_ind;
typedef struct {uint64_t offset;} VkMemSpan;
typedef struct VkMemSpanCapture VkMemSpanCapture;
typedef struct {uint32_t ignored[5];} VkDrawIndexedIndirectCommand;
typedef struct {uint32_t ignored[3];} VkDrawMeshTasksIndirectCommandEXT;
typedef uint64_t VkDescriptorSet, VkDescriptorSetLayout, VkImage, VkImageView, VkSampler;
enum { GFX_POOL_SAMPLER_COMPARE = 1, GFX_POOL_SAMPLER_ORDINARY = 2 };
typedef struct { uint64_t compare[4096 / 64], ordinary[4096 / 64]; } GfxShaderPoolSamplers;
typedef int dpool_t;
typedef int VkResult;
typedef struct { unsigned owner; } RcBudgetTicket;
enum { RC_BUDGET_TEXTURE = 1 };
typedef struct {uint64_t scene, ordinal;} RcRegistryPosition;
typedef struct {uint64_t fence, tex_pool; const void *samp_pool; VkImage fbl_img[1]; int nfbl;} VkExec;
typedef struct {int type; VkImage image; unsigned samples;} VkTexImage;
typedef struct {int target;} view_t;
typedef struct {VkTexImage image; view_t view; int has_view;} VkTextureRestoreShape;
typedef struct {const VkTexImage *image; uint64_t registration; view_t view;
 int native_slot,has_view,historical,canonical,projected;} VkTextureSample;
typedef struct {uint64_t registration; int native_slot,canonical,unbacked,projected;} VkTextureReadKey;
typedef struct {VkTextureReadKey selected; uint32_t selected_kinds,stamp,fmask;} PoolSlot;
typedef struct {uint64_t mapping;} RcInterval;
typedef int VkTextureAliasPhase, RcRegistryResult;
typedef struct {int gate;} RcRegistryNativeImageFailure;
typedef struct {VkTextureAliasPhase phase; RcRegistryResult status; RcInterval receiver;
 uint64_t registration,material_serial; VkImage image; RcRegistryNativeImageFailure coverage;} VkTextureAliasFailure;
typedef struct { unsigned retired; } VkTextureRegistrationScope;
typedef struct VkTextureAliasRead VkTextureAliasRead;
typedef struct {int unused;} ViewKey;
typedef struct {int logical; uint64_t observed;} VkTextureRestore;
typedef struct {int pool, type;} GfxShaderRes;
typedef struct {GfxShaderRes resources[2]; int count;} GfxShader;
typedef struct {const GfxShader *vs, *fs, *ts;} VkPipeState;
typedef struct {int pool;} VkPipeInfo;
typedef struct VkExecPoolGrant VkExecPoolGrant;
typedef struct {const VkPipeInfo *info; VkExecPoolGrant *grant; uint32_t opcode;} VkExecPoolBindScope;
typedef struct {VkImage installed[VK_POOL_KINDS * VK_POOL_TEX];} TextureImagePublication;
typedef int VkExecGraphicsPass;
typedef int VkExecGpaSampleScope;
typedef enum {VK_EXEC_GRAPHICS_COMPAT,VK_EXEC_GRAPHICS_NOOP,VK_EXEC_GRAPHICS_READY,VK_EXEC_GRAPHICS_NEEDS_ORDERED,
 VK_EXEC_GRAPHICS_STOPPED,VK_EXEC_GRAPHICS_UNAVAILABLE,VK_EXEC_GRAPHICS_INVALIDATED} VkExecGraphicsResult;
typedef enum {VK_EXEC_RESTORE_READY,VK_EXEC_RESTORE_STOPPED,VK_EXEC_RESTORE_STALE,VK_EXEC_RESTORE_FAILED} VkExecRestoreResult;
enum {VK_TEXTURE_ALIAS_ORDER,VK_TEXTURE_ALIAS_LAYOUT,VK_TEXTURE_ALIAS_OWNER,VK_TEXTURE_ALIAS_COVERAGE};
enum {RC_REGISTRY_BLOCKED=4,RC_REGISTRY_STALE,RC_REGISTRY_UNSUPPORTED=7};
enum {VK_EXEC_PREPARATION_POOL_IMAGE=4};
typedef enum {VK_EXEC_PREPARATION_ORDER,VK_EXEC_PREPARATION_MAPPING,VK_EXEC_PREPARATION_ATTACHMENT,
 VK_EXEC_PREPARATION_PLACEMENT,VK_EXEC_PREPARATION_PRESSURE,VK_EXEC_PREPARATION_DESCRIPTOR_SET} VkExecPreparationReason;
typedef struct {VkTextureAliasFailure alias;uint64_t registration;RcRegistryPosition read;} VkExecSampledFailure;
typedef struct {int owner; VkExecPreparationReason reason; VkExecGraphicsResult result; int detail;
 uint32_t role,slot; RcInterval range; uint64_t registration,material_serial; VkImage image;
 RcRegistryNativeImageFailure coverage; const char *stage; VkExecSampledFailure sampled;} VkExecPreparationFailure;
typedef enum {VK_SYNC_PROGRESS_UNAVAILABLE,VK_SYNC_PROGRESS_COMPLETE,VK_SYNC_PROGRESS_STOPPED,
 VK_SYNC_PROGRESS_STOPPED_AFTER_CUT} VkSyncProgressResult;
typedef enum {VK_TEXTURE_SAMPLE_READY, VK_TEXTURE_SAMPLE_UNAVAILABLE, VK_TEXTURE_SAMPLE_ABSENT,
 VK_TEXTURE_SAMPLE_GONE, VK_TEXTURE_SAMPLE_CANONICAL_REFUSED, VK_TEXTURE_SAMPLE_UNBACKED,
 VK_TEXTURE_SAMPLE_FALLBACK, VK_TEXTURE_SAMPLE_DEFERRED} VkTextureSampleStatus;
enum {VK_TEXTURE_REGISTRATION_CURRENT=0,VK_TEXTURE_REGISTRATION_EXACT=1,VK_TEXTURE_REGISTRATION_ABSENT=-1,
 VK_TEXTURE_READ_REQUIRED=1};
enum {VK_IMAGE_TYPE_2D=1,VK_IMAGE_VIEW_TYPE_3D=4};
static int flat3d_pool_consuming;
static unsigned member_revision[VK_POOL_TEX], selected_notes[VK_POOL_TEX];
static uint32_t previously_installed;
static int view_failure, create_deferred;
static uint64_t member_registration(int id){return (uint64_t)id+90+16*member_revision[id];}
static unsigned pool_mt, texture_lock;
static uint64_t publication,scope,readable,done,pool_ready_generation,refused_at;
static VkTextureRegistrationScope registration, newer_registration, *owned_scope = &registration;
static unsigned scope_refs, finite_capture, finite_inputs, finite_commits, finite_fallbacks;
static uint32_t sampler_generation;
static int sampler_zero_registered, sampler_zero_comparison, sampler_pool_comparison, sampler_one_comparing;
static uint64_t comparison_mask, ordinary_mask;
static uint32_t vk_sampler_gen(void){return sampler_generation;}
static int vk_sampler_slot_registered_locked(int id){assert(texture_lock&&!id);return sampler_zero_registered;}
static int vk_sampler_slot_comparison_locked(int id){
    assert(texture_lock && (id == 0 || id == 1));
    return id ? sampler_one_comparing : vk_sampler_slot_registered_locked(id) && sampler_zero_comparison;
}
static VkSampler vk_sampler_by_id_locked(int id){
    assert(texture_lock && (id == 0 || id == 1));
    return id ? 999 : 778;
}
static int vk_sampler_pool_comparison_locked(void){assert(texture_lock);return sampler_pool_comparison;}
static unsigned word_captures, word_releases, exact_load[VK_POOL_TEX];
static uint32_t finite_member, finite_member_kinds;
static int finite_block, layout_block, read_closed;
static unsigned exact_checks, exact_drift_at, exact_drift_mask, exact_remap_member;
typedef struct { unsigned slot; uint64_t mapping; } MemMappingReadRange;
typedef struct { MemMappingReadRange *ranges; uint32_t count,capacity; int incomplete; } VkTexturePoolMappingProof;
static VkTexturePoolMappingProof *host_proof;
static unsigned map_validations;
static void vk_texture_pool_mapping_proof_locked(VkTexturePoolMappingProof *proof) {
    assert(texture_lock&&pool_mt);
    host_proof=proof;
}
static uint64_t mem_mapping_validate_read_ranges(const MemMappingReadRange *ranges,uint32_t count,uint64_t before);
static int exact_drift_repeat, exact_close_read, exact_recycle_set;
static uint64_t exact_mapping, exact_protection;
static uint64_t pool_finite_native_bindings;
static int registry;
typedef int RcRegistry;
static RcRegistryPosition position;
static int pass_current, stop_request, ordered_current, mapping_changed, restore_stop, scope_useful, record_failed;
static unsigned captures, cancels, native_restores, native_effects, prefix_cuts, prior_waits, refused_reports, history_reports;
static unsigned native_part, native_part_uses[2];
static int phase_recorded, stop_after_cut;
static int available[VK_POOL_TEX], actual_kinds[VK_POOL_TEX], missing_reason[VK_POOL_TEX];
static VkTextureSampleStatus selected_status[VK_POOL_TEX];
static int scoped_absent[VK_POOL_TEX];
static VkTexImage resident[VK_POOL_TEX];
static TextureImagePublication installed[POOL_SETS];
static int pool_retry;
static uint32_t pool_defer[VK_POOL_TEX],pool_nw;
enum { DV_DREF_2D = 9, DV_DREF_1D, DV_DREF_1D_ARRAY, DV_DREF_2D_ARRAY, DV_DREF_CUBE, DV_DREF_CUBE_ARRAY };
static struct { struct {VkImageView view;} img[DV_DREF_CUBE_ARRAY + 1]; VkSampler samp, samp_shadow; } Z;
static uint64_t cells[POOL_SETS][POOL_DESCRIPTOR_ENTRIES];
static uint32_t native_writes[POOL_SETS];
static uint32_t vk_pool_kinds_used;
static int optional_comparison_refused;
static const TextureImagePublication *pool_transfer_images;
static void pool_restore_comparison_sampler0(int index);
static void pool_catch_up_to(int index, uint64_t from, uint64_t to);
static struct {
 VkDescriptorSet set,sets[POOL_SETS];
 TextureImagePublication *publication[POOL_SETS];
 uint64_t cur_fence,set_fence[POOL_SETS],set_generation[POOL_SETS],set_synced[POOL_SETS],logn,
  registration_pool,registration_scope,registration_publication,comparison_generation,catchup,written,rewrites;
 uint64_t last[POOL_DESCRIPTOR_ENTRIES];
 uint32_t log[POOL_LOG];
 VkSampler samp[VK_POOL_SAMP],comparison[VK_POOL_SAMP];
 uint8_t set_comparison_sampler0[POOL_SETS],set_comparison_bound[POOL_SETS];
 uint32_t set_prepare_refs[POOL_SETS],set_registration_gone[POOL_SETS],kinds,registration_gone_count,sgen;
 uint64_t tok[VK_POOL_TEX];
 PoolSlot ps[VK_POOL_TEX];
 VkImage image[VK_POOL_KINDS*VK_POOL_TEX];
 VkImageView view[VK_POOL_KINDS*VK_POOL_TEX];
 uint8_t registration_gone[VK_POOL_TEX],set_rebuild[POOL_SETS],set_prefix_complete[POOL_SETS],set_registration_missing[POOL_SETS];
 int cur,nsets,sealed,registration_bound,wanted_reconcile,registration_live,finite_reconcile,split_samplers;
} PL;
static void plat_mutex_lock(unsigned *lock){assert(!*lock);(*lock)++;}
static void plat_mutex_unlock(unsigned *lock){assert(*lock);(*lock)--;}
static void vk_texture_lock(void){assert(!texture_lock);texture_lock++;}
static void vk_texture_unlock(void){assert(texture_lock);texture_lock--;}
static VkTextureRegistrationScope *vk_exec_texture_registration_scope(void){return owned_scope;}
static void vk_texture_registration_scope_retain(VkTextureRegistrationScope *lease){
 assert(lease==owned_scope&&!lease->retired);scope_refs++;
}
static void vk_texture_registration_scope_end(VkTextureRegistrationScope *lease){assert(lease&&scope_refs);scope_refs--;}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *read){(void)read;abort();}
static void vk_replay_metadata_free(void *data,RcBudgetTicket *ticket){assert(!data&&!ticket->owner);}
static RcRegistry *rc_registry_runtime(void){return &registry;}
static int rc_registry_position_open(RcRegistry *r,RcRegistryPosition p){
 assert(r==&registry);return !read_closed&&p.scene==position.scene&&p.ordinal==position.ordinal;
}
static uint64_t rc_registry_authority_epoch(RcRegistry *r){assert(r==&registry);return readable;}
static uint64_t mem_mapping_serial(void){return exact_mapping;}
static uint64_t mem_protection_serial(void){return exact_protection;}
static int in_fbl(const VkExec *ex,VkImage image){return ex->nfbl&&image==ex->fbl_img[0];}
static int vk_texture_pool_gpu_layout_current(TextureImagePublication *p,const VkImage *feedback,uint32_t count){
 assert(pool_mt&&p&&feedback);return !count&&!layout_block;
}
static VkTextureAliasRead *vk_texture_publication_alias_locked(const TextureImagePublication *p,uint32_t slot,VkImage image){
 assert(pool_mt&&texture_lock&&p&&slot<VK_POOL_KINDS*VK_POOL_TEX&&p->installed[slot]==image);return NULL;
}
static int vk_texture_alias_published_current_locked(int slot,VkTextureAliasRead *read,RcRegistryPosition p) {
 (void)slot;(void)read;(void)p;abort();
}
static uint64_t vk_texture_registration_publication(void){return publication;}
static RcRegistryPosition vk_exec_resource_position(void){return position;}
static int rc_registry_position_valid(RcRegistryPosition p){return p.scene&&p.ordinal;}
static int rc_registry_position_compare(RcRegistryPosition a,RcRegistryPosition b){return a.scene!=b.scene||a.ordinal!=b.ordinal;}
static int vk_texture_deferred(void){return 0;}
static int vk_sync_stopping(void){return stop_request;}
static int vk_device_is_lost(void){return 0;}
static int vk_exec_recording_failed(void){return record_failed;}
static uint64_t vk_sync_fence_done(void){return done;}
static int pool_current(void){return !PL.finite_reconcile&&ordered_current&&PL.cur_fence==position.scene&&PL.registration_publication==publication;}
static int pool_current_for(uint32_t kinds){return pool_current()&&(PL.kinds&kinds)==kinds;}
static int pool_order_current(void){return !PL.finite_reconcile&&ordered_current;}
static int pool_order_current_for(uint32_t kinds){return pool_order_current()&&(PL.kinds&kinds)==kinds;}
static void fixed_require(void){}
static VkDescriptorSetLayout vk_pool_set_layout(void){return 1;}
static int vk_exec_graphics_pass_current(VkExec *ex,VkExecGraphicsPass *pass){(void)ex;return pass&&pass_current;}
static void rt_fatal(const char *reason,...){fputs(reason,stderr);abort();}
static void rt_log(const char *format,...){(void)format;}
static void vk_exec_required_graphics_unavailable(VkExec *ex,unsigned opcode,const char *reason){
 (void)ex;(void)opcode;(void)reason;record_failed=1;
}
static unsigned res_view_type(uint16_t k){return k&255;}
static unsigned dummy_index(unsigned k,int ms){assert(!ms);return k;}
static uint64_t pool_slot_token(int id){
 return scoped_absent[id]?0:((uint64_t)member_revision[id]<<32)|((uint64_t)id+21);
}
static int pool_need_restore(uint32_t id,uint32_t mask);
static void pool_member_invalidate(uint32_t id){assert(id<VK_POOL_TEX);}
static void pool_set_image(uint32_t kind,uint32_t id,VkImage image,VkImageView view){
 uint32_t entry=kind*VK_POOL_TEX+id;
 if (kind==10 && (PL.view[entry]!=view || PL.image[entry]!=image)) {
  PL.last[entry]=PL.logn;
  PL.log[PL.logn++ % POOL_LOG]=entry;
  cells[PL.cur][entry]=view;
 }
 PL.view[entry]=view;PL.image[entry]=image;installed[PL.cur].installed[entry]=image;
}
static int gfx_shader_resources(const GfxShader*s,int k,const GfxShaderRes **r){assert(k==GFX_RES_TEX);*r=s->resources;return s->count;}
static int texture_view_key(const VkTexImage *s,const view_t*v,int hv,uint16_t rtype,int storage,ViewKey *o){
 (void)v;(void)hv;(void)o;assert(!storage);return s->type==rtype;
}
static VkImageView get_view_actual(VkExec *ex,const VkTexImage *s,const view_t*v,int hv,uint16_t rtype,int storage,
 VkImage *actual,int required,VkTextureReadKey key){
 (void)ex;(void)v;(void)hv;(void)required;(void)key;
 assert(texture_lock&&pool_mt&&!storage&&s->type==rtype);
 if(view_failure)return 0;
 *actual=s->image;return s->image;
}
static int vk_texture_registration_expected_locked(int id,VkImage *image){
 assert(texture_lock&&id>=0&&id<VK_POOL_TEX);
 if(image)*image=0;
 return scoped_absent[id]?VK_TEXTURE_REGISTRATION_ABSENT:VK_TEXTURE_REGISTRATION_CURRENT;
}
static void vk_texture_registration_note_locked(int id,const VkTexImage *image,const view_t *view,int hv) {
 (void)view;(void)hv;assert(texture_lock&&image==&resident[id]);
}
static int vk_texture_pool_note_selected_locked(const VkTextureSample *selected,uint32_t *out,VkTextureReadKey *key){
 const int id=selected->native_slot;
 assert(texture_lock&&selected->image==&resident[id]&&id>=0&&id<VK_POOL_TEX);
 assert(!key->registration||key->registration==member_registration(id));
 *out=(uint32_t)id+1;key->registration=member_registration(id);selected_notes[id]++;return 1;
}
static VkTextureSampleStatus vk_texture_history_flat3d_locked(int logical,const VkTexImage **image,void *plan){
 (void)logical;(void)image;(void)plan;abort();
}
static uint32_t pool_required_previous_kinds(uint32_t id,uint32_t kinds){
 assert(id<VK_POOL_TEX);return previously_installed&kinds;
}
static int vk_texture_create_deferred_locked(void){
 int d=create_deferred;create_deferred=0;return d;
}
static void pool_association_detach(uint32_t id){assert(texture_lock&&id<VK_POOL_TEX);}
static void pool_hold(uint32_t id,uint32_t ent){assert(texture_lock&&id<VK_POOL_TEX&&(!ent||ent==id+1));}
static void pool_unres(uint32_t id,int got){assert(texture_lock&&id<VK_POOL_TEX&&(got||selected_status[id]==VK_TEXTURE_SAMPLE_UNBACKED));}
static int vk_texture_sample_restore_plan_locked(int id,VkTextureRestoreShape *shape,VkTextureRestore **out){
 assert(texture_lock&&pool_mt);if(!available[id])return 0;
 *shape=(VkTextureRestoreShape){.image={.type=actual_kinds[id]}};
 if(out){*out=malloc(sizeof **out);**out=(VkTextureRestore){id,readable};captures++;}return 1;
}
static VkTextureRestore *vk_texture_sample_restore_capture_locked(int id){
 VkTextureRestoreShape shape;VkTextureRestore *restore=NULL;
 vk_texture_sample_restore_plan_locked(id,&shape,&restore);return restore;
}
static void vk_texture_restore_cancel(VkTextureRestore *restore){if(restore){cancels++;free(restore);}}
static int vk_texture_sample_restore_current(void *input){
 VkTextureRestore *restore=input;return restore->observed==readable;
}
static int vk_texture_sample_restore_install_locked(VkTextureRestore *restore,VkTextureSample *sample){
 assert(texture_lock&&pool_mt);if(restore->observed!=readable)return VK_TEXTURE_SAMPLE_UNAVAILABLE;
 available[restore->logical]=0;
 *sample=(VkTextureSample){.image=&resident[restore->logical],.registration=(uint32_t)restore->logical+90,
                          .native_slot=restore->logical,.canonical=1};
 return VK_TEXTURE_SAMPLE_READY;
}
static VkExecRestoreResult vk_exec_prepare_texture_restore(VkExec *ex,VkTextureRestore *restore,
 int(*current)(void *),void *ctx,const VkTexImage **out,VkResult *failure){
 (void)failure;assert(ex->fence==position.scene&&!pool_mt&&!texture_lock);native_restores++;
 assert(current(ctx));
 if (restore_stop) return VK_EXEC_RESTORE_STOPPED;
 if (mapping_changed) readable++;
 if (!current(ctx)) return VK_EXEC_RESTORE_STALE;
 resident[restore->logical]=(VkTexImage){.type=actual_kinds[restore->logical],.image=100+restore->logical};
 *out=&resident[restore->logical];return VK_EXEC_RESTORE_READY;
}
static void vk_pool_retry_remove(int *r,uint32_t id){(void)r;(void)id;}
static void pool_write_to(VkDescriptorSet set, uint32_t binding, uint32_t id, VkImageView view, VkSampler sampler) {
 assert(pool_mt && !(!view && !sampler));
 int index = -1;
 for (int i = 0; i < PL.nsets; ++i) if (PL.sets[i] == set) index = i;
 assert(index >= 0 && !PL.set_prepare_refs[index]);
 uint32_t entry = binding == VK_POOL_COMPARISON_BINDING ? POOL_IMAGE_ENTRIES + VK_POOL_SAMP + id :
  binding * VK_POOL_TEX + id;
 if (binding < VK_POOL_KINDS) {
  assert(texture_lock);
  if (pool_transfer_images) {
   assert(pool_transfer_images == PL.publication[PL.cur] && index != PL.cur);
   assert(pool_transfer_images->installed[entry] == PL.image[entry]);
  }
  assert(PL.publication[index] == &installed[index]);
  PL.publication[index]->installed[entry] = PL.image[entry];
 }
 cells[index][entry] = view ? view : sampler;
 native_writes[index]++;
}
static void pool_flush(void){assert(pool_mt);}
static void vk_texture_publication_use(TextureImagePublication *images){
 assert(pool_mt&&images&&native_part<2);native_effects++;native_part_uses[native_part]++;
}
static VkExecGpaSampleScope wait_scope;
static VkExecGpaSampleScope *vk_exec_gpa_sample_enter(VkExec *ex,RcRegistryPosition attempted){
 (void)ex;(void)attempted;assert(!pool_mt&&!texture_lock);return &wait_scope;
}
static int vk_texture_sample_wait_gpa_locked(VkExec *ex,int id){(void)ex;assert(texture_lock);refused_at=id+1;return scope_useful;}
static int vk_exec_gpa_sample_leave(VkExecGpaSampleScope *p){assert(p==&wait_scope);return scope_useful;}
static void vk_exec_gpa_sample_cancel(VkExecGpaSampleScope *p){assert(p==&wait_scope);}
static void vk_exec_sample_source_refused(VkExec *ex,int id){(void)ex;assert(id<0);refused_reports++;}
static void vk_exec_registration_gone(VkExec *ex,int id){(void)ex;assert(id<0);history_reports++;}
'''

STRUCTS = r'''
typedef struct PoolIndexInputs {int alive;} PoolIndexInputs;
typedef struct ComputePoolIndexInputs {PoolIndexInputs inputs;} ComputePoolIndexInputs;
static PoolIndexInputs receipt;
typedef struct {
 uint64_t program_serial;
 VkPipeState state; struct {const VkPipeInfo *info; uint64_t pipeline; const void *revision; uint64_t fs_query_size_lod0_mask;} use; VkExecPoolGrant *pool_grant;
 VkExecGraphicsPass *pass; VkExecPoolIndices pool_indices; uint32_t opcode, sample_count; int compatibility, prim, mesh, direct_mesh;
 int restart, exact_indirect; void *widened_index; VkMemSpan index, indirect; VkMemSpanCapture *index_hold, *indirect_hold;
 int pool_index_retry;
 const GfxShaderRes *pool_index_import_pressure,*pool_index_import_required;
 int pool_index_import_pressure_stage,pool_index_import_required_stage;
 PoolIndexInputs *pool_index_inputs; RcRegistryPosition pool_index_read,read_position;
 const void *sampler_pool; uint32_t sampler_generation;
 const GfxShader *pool_index_shaders[3];
 const void *payload;
 VkExecPreparationFailure failure;
} VkExecGraphicsDraw;
static void vk_exec_graphics_failure_set(VkExecGraphicsDraw *draw,const VkExecPreparationFailure *failure) {
 if (!draw->failure.owner) draw->failure=*failure;
}
static VkExecPoolBindScope pool_binding;
static int graphics_pool_retained_indices_current(VkExec *,VkExecGraphicsDraw *,const GfxShader *const [3]);
static int domain_capture, domain_attempts, domain_refused, domain_comparison_proved;
static int import_pressure, imported_required, refuse_import;
static int gfx_shader_has_pool_index_domain(const GfxShader *shader) { return shader && domain_capture; }
static int gfx_shader_has_pool_comparison_domain(const GfxShader *shader) {
    return shader && domain_capture && shader->count && shader->resources[0].type == 0x403;
}
static unsigned gfx_shader_pool_sampler_modes(const GfxShader *shader) {
 return gfx_shader_has_pool_comparison_domain(shader) ? GFX_POOL_SAMPLER_COMPARE :
     shader && shader->count == 2 && shader->resources[1].pool == 2 ? GFX_POOL_SAMPLER_ORDINARY : 0;
}
static int gfx_shader_has_pool_index_plan(const GfxShader *shader) { return shader && !domain_capture; }
static int graphics_pool_capture_indices(VkExec *ex,VkExecGraphicsDraw *draw,const GfxShader *const shaders[3],int domain){
 assert(ex&&draw&&shaders[0]==draw->state.vs);
 if (graphics_pool_retained_indices_current(ex,draw,shaders)) return 1;
 if (import_pressure && !draw->pool_index_import_required) {
   draw->pool_index_import_pressure=shaders[0]->resources;
   draw->pool_index_import_pressure_stage=0;
   return 0;
 }
 if (draw->pool_index_import_required) {
   assert(draw->pool_index_import_required==shaders[0]->resources && !draw->pool_index_import_required_stage);
   imported_required++;
   if (refuse_import) return 0;
 } else if (domain) { domain_attempts++; if (domain_refused || !domain_capture) return 0; }
 else if (!finite_capture) return 0;
 assert(!draw->pool_index_inputs&&!receipt.alive);
 draw->pool_indices=(VkExecPoolIndices){.count=1,.domain=domain,.entries={{finite_member,finite_member_kinds}}};
 if(domain && domain_comparison_proved) {
   draw->pool_indices.comparison_sampler_generation=sampler_generation;
   draw->pool_indices.comparison_samplers.compare[0]=comparison_mask;
   draw->pool_indices.comparison_samplers.ordinary[0]=ordinary_mask;
   draw->pool_indices.comparison_absent=!sampler_zero_registered;
 }
 memcpy(draw->pool_index_shaders,shaders,sizeof draw->pool_index_shaders);
 draw->pool_index_read=position;
 draw->pool_index_inputs=&receipt;receipt.alive=1;word_captures++;
 finite_inputs=1;return 1;
}
static int vk_exec_pool_index_inputs_current(VkExec *ex,VkExecGraphicsDraw *draw){assert(ex&&draw);return finite_inputs;}
static int pool_index_inputs_current(const VkExec *ex,RcRegistryPosition read,const PoolIndexInputs *inputs){
 (void)ex;(void)read;(void)inputs;assert(0);return 0;
}
static void compute_pool_index_inputs_release(ComputePoolIndexInputs *inputs){assert(!inputs);}

static void vk_exec_pool_index_inputs_commit(VkExecGraphicsDraw *draw,int finite){
 draw->pool_index_retry=0;
 if (finite) {assert(finite_inputs&&draw->pool_grant->finite);finite_commits++;return;}
 if (draw->pool_index_inputs) {
  assert(receipt.alive&&draw->pool_index_inputs==&receipt);
  receipt.alive=0;word_releases++;finite_fallbacks++;finite_inputs=0;draw->pool_index_inputs=NULL;
 }
 draw->pool_indices=(VkExecPoolIndices){0};draw->pool_index_read=(RcRegistryPosition){0};
 memset(draw->pool_index_shaders,0,sizeof draw->pool_index_shaders);
}
static void vk_exec_graphics_draw_release_gpa_inputs(VkExecGraphicsDraw *draw);
static VkExecGraphicsResult graphics_required_device_inputs(VkExec *ex,VkExecGraphicsDraw *draw){
 assert(ex&&draw);return VK_EXEC_GRAPHICS_READY;
}
'''

NATIVE = r'''
// Projection and physical lookups are mocked as for image restoration; the actual grant walks the
// resolved membership, seals the generation, preserves its registration horizon and issues the bind.
enum { POOL_EXACT_READY, POOL_EXACT_SET, POOL_EXACT_REFUSED, POOL_EXACT_READ, POOL_EXACT_CHANGED, POOL_EXACT_MAP };
static unsigned pool_selected_member_status(const VkExec *ex,uint32_t id,uint32_t kinds,int prepare,uint32_t *failed_kind) {
 assert(ex&&pool_mt&&texture_lock&&id<VK_POOL_TEX&&kinds&&failed_kind&&(!host_proof||!prepare));
 *failed_kind=UINT32_MAX;
 ++exact_checks;
 if (exact_drift_at && (exact_checks==exact_drift_at || exact_drift_repeat)) {
    if (exact_drift_mask&1) ++readable;
    if (exact_drift_mask&2) ++exact_mapping;
    if (exact_drift_mask&4) ++exact_protection;
    if (exact_remap_member) ++member_revision[exact_remap_member];
    if (exact_close_read) read_closed=1;
    if (exact_recycle_set) ++PL.set_generation[PL.cur];
 }
 if (finite_block || PL.registration_gone[id] || PL.tok[id]!=pool_slot_token((int)id)) return POOL_EXACT_REFUSED;
 if ((kinds & (1u<<10)) && scoped_absent[id] &&
     (!PL.view[10*VK_POOL_TEX+id] || PL.view[10*VK_POOL_TEX+id]!=pool_absent_view(10)))
  return POOL_EXACT_REFUSED;
 for (uint32_t k=0;k<VK_POOL_KINDS;k++) if ((kinds>>k&1) &&
     ((!PL.image[k*VK_POOL_TEX+id]&&!scoped_absent[id]&&
       !(PL.ps[id].selected.unbacked&&selected_status[id]==VK_TEXTURE_SAMPLE_UNBACKED))||
     PL.publication[PL.cur]->installed[k*VK_POOL_TEX+id]!=PL.image[k*VK_POOL_TEX+id])) {
   *failed_kind=k;return POOL_EXACT_REFUSED;
 }
 if (host_proof) host_proof->ranges[host_proof->count++]=(MemMappingReadRange){id,member_revision[id]};
 return POOL_EXACT_READY;
}
static uint64_t mem_mapping_validate_read_ranges(const MemMappingReadRange *ranges,uint32_t count,uint64_t before) {
    assert(!host_proof&&texture_lock&&pool_mt&&before&&count);
    map_validations++;
    for(uint32_t i=0;i<count;i++) if (member_revision[ranges[i].slot]!=ranges[i].mapping) return 0;
    return exact_mapping;
}
typedef void (*VkAliasPoolMember)(int logical,const VkTextureSample *,void *);
static void graphics_pool_plan_dynamic_alias(int id,const VkTextureSample *s,void *p){(void)id;(void)s;(void)p;abort();}
static void pool_alias_refuse(PoolAliases *,uint32_t,uint32_t,const char *,const VkTextureAliasFailure *);
static int fail_pool_alias;
static void vk_texture_alias_pool_members_locked(uint64_t pool,VkAliasPoolMember visit,void *context){
 assert(pool&&visit&&context&&pool_mt&&texture_lock);
 if (fail_pool_alias) {
  GraphicsPoolRestore *request=context;
  assert(!request->indices && request->kinds);
  VkTextureAliasFailure fail={.phase=VK_TEXTURE_ALIAS_COVERAGE,.status=RC_REGISTRY_BLOCKED,
   .receiver={41},.registration=96,.material_serial=123,.image=456,.coverage={10}};
  pool_alias_refuse(request->aliases,2,2,"capture",&fail);
 }
}
static const RcInterval *vk_texture_alias_read_intervals(const VkTextureAliasRead *alias,uint32_t *count){
 (void)alias;*count=0;return NULL;
}
static int graphics_pool_alias_install(GraphicsPoolRestore *request){(void)request;abort();}
static VkExecGraphicsResult graphics_pool_alias_prepare(VkExec *ex,PoolAliases *aliases){(void)ex;(void)aliases;abort();}
static void pool_order_publish(void){assert(pool_mt);}
static void pool_sync_selected_entries(GraphicsPoolRestore *request);
static int pool_keep_selected(uint32_t id,uint32_t kinds) {
 assert(texture_lock);
 return selected_status[id]==VK_TEXTURE_SAMPLE_READY&&PL.ps[id].selected.canonical&&
        PL.ps[id].selected.registration==member_registration(id)&&(PL.ps[id].selected_kinds&kinds)==kinds;
}
static int pool_need_alias_locked(GraphicsPoolRestore *request,uint32_t id,uint32_t kind,int force) {
 assert(texture_lock&&request->indices&&id<VK_POOL_TEX&&kind<VK_POOL_KINDS&&!force);return 0;
}
static uint32_t pool_incompatible_comparison_kinds(uint32_t id,uint32_t kinds) {
 (void)id;(void)kinds;return 0;  // No registered integer interpretations in this finite-draw fixture.
}
static int pool_domain_comparison_supported(GraphicsPoolRestore *request,uint32_t id,uint32_t k,const ViewKey *key,uint64_t reg) {
 (void)request;(void)id;(void)k;(void)reg;assert(key);return 1;
}
static VkTextureSampleStatus vk_texture_resolve_sample_locked(int id,unsigned flags,VkTextureSample *sample) {
 assert(pool_mt&&texture_lock&&flags==VK_TEXTURE_READ_REQUIRED);exact_load[id]++;
 if(selected_status[id]==VK_TEXTURE_SAMPLE_UNBACKED)
  *sample=(VkTextureSample){.registration=member_registration(id),.native_slot=id};
 if(selected_status[id]==VK_TEXTURE_SAMPLE_DEFERRED)create_deferred=1;
 if(selected_status[id]!=VK_TEXTURE_SAMPLE_READY&&selected_status[id]!=VK_TEXTURE_SAMPLE_FALLBACK)
  return selected_status[id];
 resident[id]=(VkTexImage){.type=actual_kinds[id],.image=100+id+16*member_revision[id]};
 *sample=(VkTextureSample){.image=&resident[id],.native_slot=id};
 if(selected_status[id]==VK_TEXTURE_SAMPLE_READY)sample->registration=member_registration(id),sample->canonical=1;
 return selected_status[id];
}
static int pool_prepare_depth_alias_locked(GraphicsPoolRestore *request,uint32_t slot,uint32_t kinds,const VkTextureSample *sample) {
    (void)request;(void)slot;(void)kinds;(void)sample;return 0;
}
static int vk_texture_sample_unbacked_current_locked(VkTextureReadKey key) {
 assert(texture_lock);
 return key.unbacked && key.native_slot>=0 && key.native_slot<VK_POOL_TEX &&
     key.registration==member_registration(key.native_slot) && selected_status[key.native_slot]==VK_TEXTURE_SAMPLE_UNBACKED;
}
static void pool_sync(VkExec *ex){
 assert(pool_mt&&graphics_pool_restore);int n=graphics_pool_restore->target_set;
 PL.finite_reconcile=graphics_pool_restore->indices!=NULL;
 assert(n>=0&&n<PL.nsets&&!PL.set_prepare_refs[n]);
 assert(!PL.set_fence[n]||PL.set_fence[n]==ex->fence||PL.set_fence[n]<=done);
 if (PL.cur != n) PL.sealed=PL.registration_bound=0;
 PL.cur=n;PL.set=PL.sets[n];PL.cur_fence=PL.set_fence[n]=ex->fence;PL.set_generation[n]++;
 PL.registration_pool=ex->tex_pool;PL.registration_scope=scope;PL.registration_publication=publication;
 PL.sgen=sampler_generation;
 if(optional_comparison_refused){
  PL.kinds=pool_refresh_kinds(graphics_pool_restore);
  missing_reason[3]=PL.kinds&(1u<<10)?POOL_SOURCE_REFUSED:0;
 }
 PL.set_synced[n]=PL.logn;PL.set_registration_gone[n]=PL.set_registration_missing[n]=0;
 vk_texture_lock();
 for(uint32_t id=0;id<VK_POOL_TEX;id++) {
  if(missing_reason[id])pool_sample_availability(id,missing_reason[id]);
  else if(!graphics_pool_restore->indices&&!pool_need_restore(id,PL.kinds))pool_sample_availability(id,0);
  if(PL.registration_gone[id]&&!PL.set_registration_gone[n]){
   PL.set_registration_gone[n]=id+1;PL.set_registration_missing[n]=PL.registration_gone[id];
  }
 }
 if(graphics_pool_restore->indices)pool_sync_selected_entries(graphics_pool_restore);
 PL.set_synced[n]=PL.logn;
 if (PL.set_comparison_sampler0[n]) pool_restore_comparison_sampler0(n);
 vk_texture_unlock();
 ordered_current=!PL.set_registration_gone[n];
}
static VkSyncProgressResult vk_exec_graphics_pool_progress(VkExec *ex,uint64_t prior){
 assert(!texture_lock&&!pool_mt);
 if(prior){prior_waits++;done=prior;}
 else {
  assert(!phase_recorded);
  prefix_cuts++;
  vk_exec_pool_registration_prefix_complete(ex->fence);
  pass_current=0;
  if(stop_after_cut)return VK_SYNC_PROGRESS_STOPPED_AFTER_CUT;
 }
 return VK_SYNC_PROGRESS_COMPLETE;
}
'''

COMPAT = r'''
typedef struct {
 RcRegistryPosition attempted;
 VkExecGpaSampleScope **pending;
 VkExecGraphicsResult result;
 int saturated;
} GraphicsCompatibilityPool;
static struct {int mesh_fallback, has_index_type_uint8;} caps;
static VkExecGraphicsDraw *draw_current;
static unsigned selection_calls;
static int expected_prim, expected_count, expected_restart, expected_mesh, expected_direct;
#define vk_ctx() (&caps)
static int pool_on(void){return 1;}
static int draw_mesh_fn(void){return 1;}
static int draw_mesh_ind_fn(void){return 1;}
static VkExecGraphicsResult graphics_compat_argument_prepare(uint64_t address,uint64_t extent,uint32_t alignment,
    VkMemSpan *span,VkMemSpanCapture **hold){
 assert(extent&&alignment&&hold&&!*hold);
 *span=(VkMemSpan){address};
 return address && !(address % alignment) ? VK_EXEC_GRAPHICS_READY : VK_EXEC_GRAPHICS_COMPAT;
}
static VkExecGraphicsResult graphics_prepare_byte_indices(VkExec *ex,VkExecGraphicsDraw *draw,const p_draw_idx *indices,int ordered){
 (void)ex;(void)draw;(void)indices;(void)ordered;abort();
}
static VkExecGraphicsResult graphics_argument_select(uint64_t address,uint64_t extent,uint32_t alignment,
    VkMemSpan *span,VkMemSpanCapture **hold,int *exact){
 VkExecGraphicsResult result=graphics_compat_argument_prepare(address,extent,alignment,span,hold);
 *exact=result==VK_EXEC_GRAPHICS_READY;
 return result;
}
static int graphics_compat_arguments_current(const VkExec *ex,const VkExecGraphicsDraw *draw){
 assert(ex&&draw);return 1;
}
static int restart_for(const VkExec *ex,int prim,int itype){
 (void)ex;
 return prim==7&&itype==NVN_INDEX_TYPE_UNSIGNED_SHORT;
}
static VkExecGraphicsDraw *vk_exec_graphics_draw_enter(VkExecGraphicsDraw *draw){
 VkExecGraphicsDraw *previous=draw_current;
 draw_current=draw;
 return previous;
}
static int vk_exec_graphics_draw_current(VkExec *ex,uint32_t opcode,const void *payload,VkExecGraphicsDraw *draw) {
 assert(opcode==draw->opcode&&payload==draw->payload);
 return vk_exec_pool_grant_current(ex,draw->pool_grant);
}
static void vk_exec_graphics_draw_release_gpa_inputs(VkExecGraphicsDraw *draw) {
 if(!draw)return;
 vk_exec_pool_index_inputs_commit(draw,0);
 vk_exec_pool_grant_release(&draw->pool_grant);
}
static void graphics_release_gpa_inputs(VkExecGraphicsDraw *draw,int keep_arguments) {
 assert(keep_arguments==(draw&&draw->compatibility));
 vk_exec_graphics_draw_release_gpa_inputs(draw);
}
static int prepare_draw_phase(VkExec *ex,int prim,int32_t count,int restart,int mesh,int direct_mesh,
                             GraphicsCompatibilityPool *prepare){
 assert(draw_current&&draw_current->compatibility&&prepare&&!phase_recorded);
 assert(prim==expected_prim&&count==expected_count&&restart==expected_restart&&mesh==expected_mesh&&
        direct_mesh==expected_direct);
 selection_calls++;
 VkExecGraphicsResult result=graphics_pool_prepare(ex,draw_current,1,1,prepare->attempted,prepare->pending,!prepare->saturated);
 if(result!=VK_EXEC_GRAPHICS_READY)prepare->result=result;
 return 0;
}
'''

GROUP = r'''
typedef struct {uint32_t op;} op_hdr;
typedef struct {int appended;} Group;
enum {GRAPHICS_BOUNDARY_REPLAN=1,GRAPHICS_BOUNDARY_ORDERED=2};
static VkExecGraphicsDraw *front_draw;
static int graphics_opcode(uint32_t opcode){return opcode==25;}
static VkExecGraphicsResult graphics_input(VkExec *ex,const op_hdr *op,int ordered,RcRegistryPosition attempted,
    VkExecGpaSampleScope **pending,VkExecGraphicsDraw **draw){
 assert(op->op==25&&!ordered&&!pending&&!*draw);
 *draw=front_draw;
 return graphics_pool_prepare(ex,*draw,ordered,1,attempted,pending,0);
}
static int graphics_input_finished(VkExec *ex,const op_hdr *op,VkExecGraphicsResult result,const VkExecGraphicsDraw *draw){
 (void)ex;(void)op;(void)draw;return result==VK_EXEC_GRAPHICS_READY||result==VK_EXEC_GRAPHICS_NOOP;
}
static void vk_exec_graphics_draw_cancel(VkExecGraphicsDraw **draw){
 vk_exec_pool_grant_release(&(*draw)->pool_grant);*draw=NULL;
}
static void resource_group_append(Group *g,VkExec *ex,const op_hdr *op,unsigned count,
    VkExecGraphicsDraw *draw,int noop){
 (void)ex;(void)op;assert(count==1&&draw&&!draw->pool_grant&&!noop);g->appended++;
}
static const op_hdr *exercise_group_front(Group *g,VkExec *ex,const op_hdr *o,
    VkExecGraphicsDraw **boundary,int *ordered_boundary){
 const op_hdr *end=NULL;
 do {
 /* production group-front decision */
 } while (0);
 return end;
}
'''

MAIN = r'''
static const GfxShader vs={{{1,3},{2,0}},2}, fs={{{1,0x103},{0,0}},1};
static const VkPipeInfo pooled={1}, unpooled={0};
static const RcRegistryPosition none={0};
static void reset(VkExec *ex,VkExecGraphicsDraw *draw){
 memset(&PL,0,sizeof PL);memset(installed,0,sizeof installed);memset(resident,0,sizeof resident);
 memset(available,0,sizeof available);memset(missing_reason,0,sizeof missing_reason);
 memset(selected_status,0,sizeof selected_status);memset(scoped_absent,0,sizeof scoped_absent);
 memset(member_revision,0,sizeof member_revision);memset(selected_notes,0,sizeof selected_notes);
 previously_installed=0;view_failure=create_deferred=0;
 memset(exact_load,0,sizeof exact_load);word_captures=word_releases=0;receipt.alive=0;
 available[1]=available[2]=1;actual_kinds[1]=3;actual_kinds[2]=4;
 PL.nsets=2;PL.sets[0]=11;PL.sets[1]=22;PL.publication[0]=&installed[0];PL.publication[1]=&installed[1];
 PL.kinds=(1u<<2)|(1u<<3)|(1u<<4);PL.registration_live=VK_POOL_TEX;
 vk_pool_kinds_used=PL.kinds;optional_comparison_refused=0;
 assert(!scope_refs);
 owned_scope=&registration;registration.retired=newer_registration.retired=0;
 finite_capture=finite_inputs=finite_commits=finite_fallbacks=0;
 sampler_generation=1;sampler_zero_registered=sampler_zero_comparison=sampler_pool_comparison=sampler_one_comparing=0;
 comparison_mask=1;ordinary_mask=0;
 finite_member=1;finite_member_kinds=1u<<2;finite_block=layout_block=read_closed=fail_pool_alias=0;
 assert(!host_proof);
 map_validations=0;
 exact_checks=exact_drift_at=exact_drift_mask=exact_remap_member=0;
 exact_drift_repeat=exact_close_read=exact_recycle_set=0;
 exact_mapping=exact_protection=1;
 domain_capture=domain_attempts=domain_refused=domain_comparison_proved=0;
 import_pressure=imported_required=refuse_import=0;
 *ex=(VkExec){.fence=7,.tex_pool=91};*draw=(VkExecGraphicsDraw){.state={.vs=&vs,.fs=&fs},.use={.info=&pooled},.opcode=25,
  .sampler_generation=sampler_generation};
 pool_binding=(VkExecPoolBindScope){0};position=(RcRegistryPosition){7,8};publication=17;scope=18;readable=19;done=0;
 pass_current=1;stop_request=ordered_current=mapping_changed=restore_stop=scope_useful=record_failed=0;
 captures=cancels=native_restores=native_effects=prefix_cuts=prior_waits=refused_reports=history_reports=0;refused_at=0;
 native_part=0;
 phase_recorded=stop_after_cut=0;
 memset(native_part_uses,0,sizeof native_part_uses);
 memset(cells,0,sizeof cells);
 memset(native_writes,0,sizeof native_writes);
}
static void saturate_compatibility_pool(const VkExec *ex){
 PL.nsets=4;
 PL.cur_fence=ex->fence;
 PL.registration_bound=PL.sealed=1;
 for(int j=0;j<PL.nsets;j++){
  PL.sets[j]=11*(j+1);
  PL.publication[j]=&installed[j];
  PL.set_fence[j]=ex->fence;
 }
}
static void compatibility_opcode(VkExec *ex,VkExecGraphicsDraw *draw,uint32_t op,const void *payload,
                                 int prim,int count,int restart,int mesh,int direct){
 reset(ex,draw);
 saturate_compatibility_pool(ex);
 draw->opcode=op;
 draw->payload=payload;
 draw->prim=prim;
 draw->compatibility=1;
 draw->mesh=mesh;
 draw->direct_mesh=direct;
 expected_prim=prim;
 expected_count=count;
 expected_restart=restart;
 expected_mesh=mesh;
 expected_direct=direct;
 selection_calls=0;
 assert(graphics_compat_pool_prepare(ex,draw,none,NULL)==VK_EXEC_GRAPHICS_COMPAT);
 assert(selection_calls==1&&!draw_current&&prefix_cuts==1&&!native_effects&&draw->pool_grant);
 vk_exec_pool_grant_release(&draw->pool_grant);
}
int main(void){
 VkExec ex;
 VkExecGraphicsDraw draw;
 reset(&ex,&draw);
 const op_hdr op={25};Group group={0};int ordered_boundary=0;VkExecGraphicsDraw *boundary=NULL;
 front_draw=&draw;
 assert(exercise_group_front(&group,&ex,&op,&boundary,&ordered_boundary)==&op);
 assert(boundary==&draw&&ordered_boundary==GRAPHICS_BOUNDARY_ORDERED&&!group.appended&&!draw.pool_grant);
 assert(!captures&&!native_restores&&!native_effects);
 draw.use.info=&unpooled;boundary=NULL;ordered_boundary=0;
 assert(!exercise_group_front(&group,&ex,&op,&boundary,&ordered_boundary));
 assert(!boundary&&!ordered_boundary&&group.appended==1);

 reset(&ex,&draw);
 assert((vk_exec_pool_shader_kinds(draw.state.vs)|vk_exec_pool_shader_kinds(draw.state.fs))==((1u<<2)|(1u<<3)));
 assert(graphics_pool_prepare(&ex,&draw,0,1,none,NULL,0)==VK_EXEC_GRAPHICS_NEEDS_ORDERED);
 assert(!captures&&!native_restores&&!native_effects);
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 assert(captures==1&&cancels==1&&native_restores==1&&available[2]);
 assert((draw.pool_grant&&draw.pool_grant->held)&&vk_exec_pool_grant_current(&ex,draw.pool_grant)&&!native_effects);
 int i=draw.pool_grant->index;assert(installed[i].installed[2*VK_POOL_TEX+1]==101);
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY&&PL.set_prepare_refs[i]==1);
 vk_exec_pool_registration_prefix_complete(ex.fence);
 assert(PL.set_fence[i]==ex.fence&&PL.set_prefix_complete[i]&&vk_exec_pool_grant_current(&ex,draw.pool_grant));
 VkDescriptorSet set=0;
 assert(!graphics_pool_bind(&ex,&unpooled,&set)&&!native_effects);
 VkExecPoolBindScope old_scope=vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,draw.pool_grant,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set==PL.sets[i]);
 assert(native_effects==1&&draw.pool_grant->held&&!PL.set_prefix_complete[i]&&PL.set_fence[i]==ex.fence&&PL.sealed);
 vk_exec_pool_registration_prefix_complete(ex.fence);
 assert(PL.set_fence[i]==ex.fence&&PL.set_prefix_complete[i]);
 native_part=1;
 assert(vk_exec_pool_grant_current(&ex,draw.pool_grant));
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set==PL.sets[i]);
 assert(draw.pool_grant->held&&native_part_uses[0]==1&&native_part_uses[1]==1);
 vk_exec_pool_bind_enter(old_scope);
 vk_exec_pool_grant_release(&draw.pool_grant);
 assert(native_effects==2&&PL.set_fence[i]==ex.fence&&!PL.set_prepare_refs[i]);

 // An earlier whole-kind grant must still mean the same read and physical set at issue.
 // Changing its coverage cannot cause a second slot/member selection or consume capacity.
 reset(&ex,&draw);
 draw.compatibility=1;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 VkExecPoolGrant *owned_whole=draw.pool_grant;
 const uint64_t spare_generation=PL.set_generation[1-owned_whole->index];
 const unsigned captures_before_issue=captures;
 assert(!owned_whole->finite&&graphics_pool_prepare(&ex,&draw,1,0,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 ordered_current=0;
 assert(graphics_pool_prepare(&ex,&draw,1,0,none,NULL,0)==VK_EXEC_GRAPHICS_INVALIDATED);
 assert(draw.pool_grant==owned_whole&&owned_whole->held&&PL.set_generation[1-owned_whole->index]==spare_generation);
 assert(captures==captures_before_issue&&!prior_waits&&!prefix_cuts&&scope_refs==1);
 assert(draw.failure.owner==VK_EXEC_PREPARATION_POOL_IMAGE&&draw.failure.reason==VK_EXEC_PREPARATION_DESCRIPTOR_SET&&
        draw.failure.detail==RC_REGISTRY_STALE&&draw.failure.slot==owned_whole->index&&
        !strcmp(draw.failure.stage,"pool-retained-publication"));
 ordered_current=1;
 position.ordinal+=2;
 draw.failure=(VkExecPreparationFailure){0};
 assert(graphics_pool_prepare(&ex,&draw,1,0,none,NULL,0)==VK_EXEC_GRAPHICS_INVALIDATED&&
        draw.pool_grant==owned_whole&&!strcmp(draw.failure.stage,"pool-retained-read"));
 position=owned_whole->read;
 assert(graphics_pool_prepare(&ex,&draw,1,0,none,NULL,0)==VK_EXEC_GRAPHICS_READY&&draw.pool_grant==owned_whole);
 vk_exec_pool_grant_release(&draw.pool_grant);
 assert(!scope_refs);

 reset(&ex,&draw);
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);i=draw.pool_grant->index;
 old_scope=vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,draw.pool_grant,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set==PL.sets[i]);
 vk_exec_pool_bind_enter(old_scope);
 vk_exec_pool_grant_release(&draw.pool_grant);
 assert(native_effects==1&&PL.set_fence[i]==ex.fence&&!PL.set_prepare_refs[i]);
 vk_exec_pool_registration_prefix_complete(ex.fence);
 assert(!PL.set_fence[i]);

 reset(&ex,&draw);
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);i=draw.pool_grant->index;
 publication++;
 assert(!vk_exec_pool_grant_current(&ex,draw.pool_grant));
 vk_exec_pool_registration_prefix_complete(ex.fence);assert(PL.set_fence[i]==ex.fence);
 vk_exec_pool_grant_release(&draw.pool_grant);
 assert(!PL.set_fence[i]&&!PL.set_prepare_refs[i]&&!native_effects&&native_restores==1);

 reset(&ex,&draw);mapping_changed=1;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(captures==1&&cancels==1&&native_restores==1&&!native_effects&&!(draw.pool_grant&&draw.pool_grant->held));
 reset(&ex,&draw);restore_stop=1;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_STOPPED);
 assert(captures==1&&cancels==1&&native_restores==1&&!native_effects&&!(draw.pool_grant&&draw.pool_grant->held));

 reset(&ex,&draw);missing_reason[3]=POOL_MISSING_HISTORY;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(history_reports==1&&captures==1&&cancels==1&&!native_restores&&!native_effects);
 reset(&ex,&draw);missing_reason[3]=POOL_SOURCE_REFUSED;scope_useful=1;
 VkExecGpaSampleScope *pending=NULL;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,&pending,0)==VK_EXEC_GRAPHICS_NEEDS_ORDERED);
 assert(pending==&wait_scope&&refused_at==4&&!refused_reports&&captures==cancels&&!native_restores&&!native_effects);
 reset(&ex,&draw);missing_reason[3]=POOL_SOURCE_REFUSED;scope_useful=1;
 assert(graphics_pool_prepare(&ex,&draw,1,0,none,NULL,0)==VK_EXEC_GRAPHICS_NEEDS_ORDERED);
 assert(refused_at==4&&!refused_reports&&captures==cancels&&!native_restores&&!native_effects);

 reset(&ex,&draw);missing_reason[3]=POOL_SOURCE_REFUSED;
 assert(graphics_pool_prepare(&ex,&draw,1,0,none,NULL,0)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(refused_at==4&&refused_reports==1&&captures==cancels&&!native_restores&&!native_effects&&
        !prior_waits&&!prefix_cuts&&!draw.pool_grant);


 reset(&ex,&draw);PL.set_fence[1]=6;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY&&prior_waits==1&&!prefix_cuts);
 vk_exec_pool_grant_release(&draw.pool_grant);
 reset(&ex,&draw);VkExecGraphicsPass pass=1;draw.pass=&pass;
 PL.cur_fence=PL.set_fence[0]=PL.set_fence[1]=ex.fence;PL.registration_bound=1;
 assert(vk_exec_pool_prepare_info(&ex,1u<<2,&draw.pool_grant,1,0,none,NULL,&pass)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!prefix_cuts&&!native_effects);
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_INVALIDATED);
 assert(prefix_cuts==1&&!PL.set_fence[0]&&!PL.set_fence[1]&&!native_restores&&!native_effects);
 draw.pass=NULL;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY&&prefix_cuts==1);
 vk_exec_pool_grant_release(&draw.pool_grant);
 // Field profile: four already-bound sets belonging to the unfinished scene.
 reset(&ex,&draw);
 saturate_compatibility_pool(&ex);
 draw.compatibility=1;
 assert(graphics_pool_prepare(&ex,&draw,1,0,none,NULL,0)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!prefix_cuts&&!draw.pool_grant&&!native_effects);
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 i=draw.pool_grant->index;
 assert(prefix_cuts==1&&PL.set_prepare_refs[i]==1&&done<ex.fence&&!native_effects);
 for(int j=0;j<PL.nsets;j++)if(j!=i)assert(!PL.set_fence[j]);
 // The binding owns the publication across a covered prefix until it is used or canceled.
 vk_exec_pool_registration_prefix_complete(ex.fence);
 assert(PL.set_fence[i]==ex.fence&&PL.set_prefix_complete[i]&&PL.set_prepare_refs[i]==1);
 assert(vk_exec_pool_grant_current(&ex,draw.pool_grant));
 vk_exec_pool_grant_release(&draw.pool_grant);
 assert(!PL.set_fence[i]&&!PL.set_prepare_refs[i]&&!native_effects);

 reset(&ex,&draw);
 saturate_compatibility_pool(&ex);
 stop_after_cut=1;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_STOPPED);
 assert(prefix_cuts==1&&done<ex.fence&&!draw.pool_grant&&!native_effects);
 for(int j=0;j<PL.nsets;j++)assert(!PL.set_prepare_refs[j]);

 reset(&ex,&draw);
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 i=draw.pool_grant->index;
 old_scope=vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,draw.pool_grant,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set==PL.sets[i]);
 vk_exec_pool_bind_enter(old_scope);
 phase_recorded=1;
 saturate_compatibility_pool(&ex);
 publication++;
 VkExecPoolGrant *late=NULL;
 assert(vk_exec_pool_prepare_info(&ex,1u<<2,&late,1,0,none,NULL,NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!late&&!prefix_cuts&&native_effects==1&&PL.set_fence[i]==ex.fence&&PL.set_prepare_refs[i]==1);
 vk_exec_pool_grant_release(&draw.pool_grant);
 assert(!PL.set_prepare_refs[i]&&PL.set_fence[i]==ex.fence&&native_effects==1);
 // A finite shader reaches slot 1. Slot 2 is also a cold kind-2 member and slot 3 is genuinely missing
 // for a broad request. The selected set may remain exact at its own read even when the aggregate cannot.
 reset(&ex,&draw);
 draw.state.fs=NULL;finite_capture=1;actual_kinds[2]=3;missing_reason[3]=POOL_MISSING_HISTORY;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 assert(draw.pool_grant->finite&&draw.pool_grant->indices.count==1&&draw.pool_grant->indices.entries[0].slot==1);
 assert(draw.pool_grant->read.ordinal==8&&scope_refs==1&&finite_commits==1&&!finite_fallbacks);
 assert(captures==1&&cancels==1&&native_restores==1&&available[2]&&!history_reports&&!native_effects);
 i=draw.pool_grant->index;
 assert(PL.registration_gone[3]==POOL_MISSING_HISTORY&&!pool_order_current()&&vk_exec_pool_grant_current(&ex,draw.pool_grant));
 const uint64_t before_finite=pool_finite_native_bindings;
 old_scope=vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,draw.pool_grant,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set==PL.sets[i]);
 assert(pool_finite_native_bindings==before_finite+1&&native_effects==1&&scope_refs==1);
 // Numeric publication and scope tokens alone cannot permit use through a different retained scope object.
 owned_scope=&newer_registration;
 assert(!vk_exec_pool_grant_current(&ex,draw.pool_grant));
 owned_scope=&registration;
 read_closed=1;
 assert(!vk_exec_pool_grant_current(&ex,draw.pool_grant));
 read_closed=0;
 ex.fbl_img[0]=101;ex.nfbl=1;
 assert(graphics_pool_bind(&ex,&pooled,&set)&&!set&&record_failed);
 assert(pool_finite_native_bindings==before_finite+1&&native_effects==1);
 vk_exec_pool_bind_enter(old_scope);
 vk_exec_pool_grant_release(&draw.pool_grant);
 assert(!scope_refs&&!PL.set_prepare_refs[i]);

 // The warm global certificate skipped a cold, actually selected token. Its sealed set is busy.
 // Installation belongs to another set and services only the requested registration.
 reset(&ex,&draw);
 draw.state.fs=NULL;draw.compatibility=1;finite_capture=1;available[1]=0;fail_pool_alias=1;
 PL.cur_fence=PL.set_fence[0]=ex.fence;PL.set_generation[0]=2;PL.set=PL.sets[0];
 PL.registration_pool=ex.tex_pool;PL.registration_publication=publication;PL.registration_scope=scope;
 PL.sealed=PL.registration_bound=ordered_current=1;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,1)==VK_EXEC_GRAPHICS_READY);
 assert(draw.pool_grant->finite&&draw.pool_grant->index==1&&PL.tok[1]==pool_slot_token(1));
 assert(exact_load[1]==1&&!exact_load[0]&&!exact_load[2]&&!exact_load[3]&&!native_restores);
 assert(!installed[0].installed[2*VK_POOL_TEX+1]&&installed[1].installed[2*VK_POOL_TEX+1]==101);
 assert(graphics_pool_prepare(&ex,&draw,1,0,none,NULL,0)==VK_EXEC_GRAPHICS_READY&&exact_load[1]==1);
 const GfxShader *original=draw.state.vs;
 draw.state.vs=&fs;
 assert(graphics_pool_prepare(&ex,&draw,1,0,none,NULL,0)==VK_EXEC_GRAPHICS_INVALIDATED&&exact_load[1]==1);
 draw.state.vs=original;
 vk_exec_graphics_draw_release_gpa_inputs(&draw);
 assert(!scope_refs&&!receipt.alive&&word_releases==1);

 // Exact refill changes the registration on the same native slot. The set seen by a
 // native draw retains its original view and material while the next set is prepared.
 reset(&ex,&draw);
 draw.state.fs=NULL;finite_capture=1;available[1]=0;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 const unsigned prior_set=draw.pool_grant->index;
 old_scope=vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,draw.pool_grant,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set==PL.sets[prior_set]);
 vk_exec_pool_bind_enter(old_scope);
 vk_exec_graphics_draw_release_gpa_inputs(&draw);
 assert(PL.set_fence[prior_set]==ex.fence&&installed[prior_set].installed[2*VK_POOL_TEX+1]==101);
 member_revision[1]++;publication++;position.ordinal+=2;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 assert(draw.pool_grant->index!=prior_set&&PL.ps[1].selected.registration==member_registration(1));
 assert(exact_load[1]==2&&selected_notes[1]==2&&PL.image[2*VK_POOL_TEX+1]==117);
 assert(installed[prior_set].installed[2*VK_POOL_TEX+1]==101&&PL.set_fence[prior_set]==ex.fence);
 vk_exec_graphics_draw_release_gpa_inputs(&draw);

 // A broad member uses the one already selected image for all compatible kinds. A
 // failed realizable view or a deferred source leaves that installed cohort intact.
 reset(&ex,&draw);
 plat_mutex_lock(&pool_mt);vk_texture_lock();
 PL.cur=0;PL.set=PL.sets[0];Z.img[1].view=600;
 assert(pool_prepare_selected_member(&ex,1,(1u<<2)|1,0));
 assert(exact_load[1]==1&&selected_notes[1]==1&&PL.image[2*VK_POOL_TEX+1]==101&&!PL.image[1]);
 member_revision[1]++;
 view_failure=1;
 assert(!pool_prepare_selected_member(&ex,1,1u<<2,0));
 assert(exact_load[1]==2&&selected_notes[1]==1&&PL.image[2*VK_POOL_TEX+1]==101);
 assert(PL.registration_gone[1]==POOL_SOURCE_REFUSED);
 view_failure=0;selected_status[1]=VK_TEXTURE_SAMPLE_DEFERRED;
 assert(!pool_prepare_selected_member(&ex,1,1u<<2,0));
 assert(exact_load[1]==3&&selected_notes[1]==1&&PL.image[2*VK_POOL_TEX+1]==101);
 selected_status[1]=VK_TEXTURE_SAMPLE_FALLBACK;
 assert(!pool_prepare_selected_member(&ex,1,1u<<2,1));
 assert(exact_load[1]==4&&selected_notes[1]==1&&PL.image[2*VK_POOL_TEX+1]==101);
 assert(pool_prepare_selected_member(&ex,1,1u<<2,0));
 assert(exact_load[1]==5&&selected_notes[1]==2&&PL.image[2*VK_POOL_TEX+1]==117);
 assert(PL.ps[1].selected.registration==member_registration(1));
 available[1]=1;
 GraphicsPoolRestore staged={.ex=&ex,.kinds=1u<<2};
 graphics_pool_restore=&staged;
 assert(pool_need_restore(1,1u<<2)&&staged.restore&&staged.logical==1);
 assert(PL.registration_gone[1]==POOL_SOURCE_RESTORE&&PL.image[2*VK_POOL_TEX+1]==117&&
        installed[0].installed[2*VK_POOL_TEX+1]==117);
 vk_texture_restore_cancel(staged.restore);
 graphics_pool_restore=NULL;
 vk_texture_unlock();plat_mutex_unlock(&pool_mt);

 // A returned ABSENT is not authority when the same scope says the registration is live.
 reset(&ex,&draw);
 draw.state.fs=NULL;finite_capture=1;available[1]=0;fail_pool_alias=1;
 selected_status[1]=VK_TEXTURE_SAMPLE_ABSENT;Z.img[3].view=600;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,1)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!draw.pool_grant&&!PL.tok[1]&&!PL.image[2*VK_POOL_TEX+1]&&word_releases==1);
 assert(draw.failure.slot==1&&draw.failure.owner==VK_EXEC_PREPARATION_POOL_IMAGE);
 reset(&ex,&draw);
 draw.state.fs=NULL;finite_capture=1;available[1]=0;scoped_absent[1]=1;
 selected_status[1]=VK_TEXTURE_SAMPLE_ABSENT;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,1)==VK_EXEC_GRAPHICS_READY);
 assert(draw.pool_grant->finite&&!PL.tok[1]&&!PL.image[2*VK_POOL_TEX+1]&&exact_load[1]==1);
 vk_exec_graphics_draw_release_gpa_inputs(&draw);

 // Known absence installs the depth view only in comparison kind 10.
 reset(&ex,&draw);
 scoped_absent[0]=1;selected_status[0]=VK_TEXTURE_SAMPLE_ABSENT;
 Z.img[3].view=600;Z.img[DV_DREF_2D].view=610;
 plat_mutex_lock(&pool_mt);vk_texture_lock();
 assert(pool_prepare_selected_member(&ex,0,(1u<<2)|(1u<<10),1));
 assert(PL.view[2*VK_POOL_TEX]==600&&PL.view[10*VK_POOL_TEX]==610);
 assert(!PL.image[2*VK_POOL_TEX]&&!PL.image[10*VK_POOL_TEX]);
 vk_texture_unlock();plat_mutex_unlock(&pool_mt);

 // A genuinely absent comparison slot cannot reach the generic color/noncomparison stand-ins.
 reset(&ex,&draw);
 const GfxShader comparison={{{1,0x403}},1};
 draw.state.vs=&comparison;draw.state.fs=NULL;finite_capture=1;
 finite_member=0;finite_member_kinds=1u<<10;scoped_absent[0]=1;
 VkExecPoolIndices comparison_pair={.count=1,.comparison_sampler_generation=1,.domain=1,.entries={{0,1u<<10}}};
 vk_texture_lock();
 assert(pool_comparison_absent_pair_locked(&comparison_pair));
 sampler_zero_registered=1;
 assert(!pool_comparison_absent_pair_locked(&comparison_pair));
 sampler_zero_registered=0;sampler_generation=2;
 assert(!pool_comparison_absent_pair_locked(&comparison_pair));
 sampler_generation=1;scoped_absent[0]=0;
 assert(!pool_comparison_absent_pair_locked(&comparison_pair));
 scoped_absent[0]=1;comparison_pair.entries[0].slot=1;
 assert(!pool_comparison_absent_pair_locked(&comparison_pair));
 vk_texture_unlock();
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(draw.failure.owner==VK_EXEC_PREPARATION_POOL_IMAGE&&draw.failure.reason==VK_EXEC_PREPARATION_ATTACHMENT&&
        draw.failure.detail==RC_REGISTRY_UNSUPPORTED&&draw.failure.slot==0&&draw.failure.role==10&&
        !strcmp(draw.failure.stage,"missing-comparison-standin"));
 assert(!draw.pool_grant&&!prefix_cuts&&!native_effects&&word_releases==1&&!exact_load[0]);

 // A real mapping hole uses its typed stand-in at this read. Once populated, the same
 // registration must receive actual material rather than inheriting the old dummy.
 reset(&ex,&draw);
 draw.state.fs=NULL;finite_capture=1;available[1]=0;
 selected_status[1]=VK_TEXTURE_SAMPLE_UNBACKED;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,1)==VK_EXEC_GRAPHICS_READY);
 assert(draw.pool_grant->finite&&PL.tok[1]&&PL.ps[1].selected.unbacked&&!PL.image[2*VK_POOL_TEX+1]);
 vk_exec_graphics_draw_release_gpa_inputs(&draw);
 selected_status[1]=VK_TEXTURE_SAMPLE_READY;
 position.ordinal+=2;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,1)==VK_EXEC_GRAPHICS_READY);
 assert(draw.pool_grant->finite&&!PL.ps[1].selected.unbacked&&PL.image[2*VK_POOL_TEX+1]==resident[1].image);
 vk_exec_graphics_draw_release_gpa_inputs(&draw);

 // Selected GPA refusal offers one real retry. It carries the words, never an unissued pool grant.
 reset(&ex,&draw);
 draw.state.fs=NULL;draw.compatibility=1;finite_capture=1;available[1]=0;
 selected_status[1]=VK_TEXTURE_SAMPLE_CANONICAL_REFUSED;scope_useful=1;
 VkExecGpaSampleScope *selected_pending=NULL;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,&selected_pending,1)==VK_EXEC_GRAPHICS_NEEDS_ORDERED);
 assert(selected_pending==&wait_scope&&draw.pool_index_retry&&draw.pool_index_inputs==&receipt&&!draw.pool_grant);
 assert(word_captures==1&&!word_releases&&!finite_commits&&!refused_reports);
 vk_exec_graphics_draw_release_gpa_inputs_retry(&draw);
 assert(word_captures==1&&!word_releases&&draw.pool_index_inputs==&receipt&&draw.pool_indices.count==1);
 assert(draw.pool_index_read.ordinal==position.ordinal&&!draw.pool_grant);
 scope_useful=0;selected_status[1]=VK_TEXTURE_SAMPLE_READY;selected_pending=NULL;
 assert(graphics_pool_prepare(&ex,&draw,1,1,position,&selected_pending,1)==VK_EXEC_GRAPHICS_READY);
 assert(!selected_pending&&draw.pool_grant->finite&&word_captures==1&&finite_commits==1&&!word_releases);
 vk_exec_graphics_draw_release_gpa_inputs(&draw);
 vk_exec_graphics_draw_release_gpa_inputs(&draw);
 assert(word_releases==1&&!receipt.alive&&!scope_refs);

 // Narrow refusal cancels the plan and independently verifies the whole native set before recording.
 reset(&ex,&draw);
 draw.state.fs=NULL;finite_capture=1;finite_block=1;actual_kinds[2]=3;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 assert(draw.pool_grant->held&&!draw.pool_grant->finite&&!finite_inputs&&!draw.pool_indices.count);
 assert(!finite_commits&&finite_fallbacks==1&&native_restores==2&&!available[2]&&scope_refs==1);
 old_scope=vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,draw.pool_grant,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set&&native_effects==1);
 assert(pool_finite_native_bindings==before_finite+2);
 vk_exec_pool_bind_enter(old_scope);
 vk_exec_pool_grant_release(&draw.pool_grant);
 assert(!scope_refs);

 // A certified table fallback runs only after the whole cohort refuses. The same retained
 // finite grant reaches compatibility issue without a second capture or any unselected cell.
 vk_exec_graphics_draw_release_gpa_inputs(&draw);
 reset(&ex,&draw);
 draw.state.fs=NULL;domain_capture=1;fail_pool_alias=1;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 assert(draw.pool_grant->finite&&draw.pool_grant->indices.domain&&domain_attempts==1&&word_captures==1&&finite_commits==1);
 assert(draw.pool_indices.count==1&&draw.pool_indices.entries[0].slot==1&&!draw.failure.owner);
 assert(graphics_pool_prepare(&ex,&draw,1,0,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 assert(domain_attempts==1&&word_captures==1&&!word_releases);
 vk_exec_graphics_draw_release_gpa_inputs(&draw);
 assert(word_releases==1&&!scope_refs);
 reset(&ex,&draw);
 draw.state.fs=NULL;domain_capture=1;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 assert(!domain_attempts&&!word_captures&&draw.pool_grant&&!draw.pool_grant->finite);
 vk_exec_graphics_draw_release_gpa_inputs(&draw);
 reset(&ex,&draw);
 draw.state.fs=NULL;domain_capture=1;domain_refused=1;fail_pool_alias=1;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(domain_attempts==1&&!word_captures&&!draw.pool_grant&&draw.failure.slot==2);

 // A raw storage replica is optional unless the real whole publication refuses.
 reset(&ex,&draw);draw.state.fs=NULL;import_pressure=1;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 assert(!imported_required&&draw.pool_grant&&!draw.pool_grant->finite&&!receipt.alive);
 vk_exec_graphics_draw_release_gpa_inputs(&draw);
 reset(&ex,&draw);draw.state.fs=NULL;import_pressure=1;fail_pool_alias=1;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 assert(imported_required==1&&draw.pool_grant->finite&&draw.pool_index_inputs&&receipt.alive);
 assert(!draw.pool_index_import_required);
 vk_exec_graphics_draw_release_gpa_inputs(&draw);
 assert(!receipt.alive);
 reset(&ex,&draw);draw.state.fs=NULL;import_pressure=1;fail_pool_alias=1;refuse_import=1;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(imported_required==1&&!draw.pool_grant&&!receipt.alive&&!draw.pool_index_import_required);


 // A complete ordinary descriptor set does not certify its noncomparison default for a known Dref.
 reset(&ex,&draw);
 const GfxShader comparison_domain={{{1,0x403},{2,0}},2};
 draw.state.vs=&comparison_domain;draw.state.fs=NULL;
 PL.kinds |= 1u<<10;
 domain_capture=domain_comparison_proved=1;finite_member=0;finite_member_kinds=1u<<10;scoped_absent[0]=1;
 selected_status[0]=VK_TEXTURE_SAMPLE_ABSENT;
 Z.img[DV_DREF_2D].view=610;
 Z.samp_shadow=888;
 PL.samp[0]=777;
 PL.nsets=1;
 assert(vk_exec_pool_prepare_info(&ex,1u<<10,&draw.pool_grant,1,1,none,NULL,NULL)==VK_EXEC_GRAPHICS_READY);
 assert(!draw.pool_grant->finite);
 VkDescriptorSet ordinary_route_set=draw.pool_grant->set;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!draw.pool_grant&&domain_attempts==1&&word_captures==1&&!native_effects);
 assert(draw.failure.reason==VK_EXEC_PREPARATION_PRESSURE&&!strcmp(draw.failure.stage,"pool-ring-no-progress"));
 draw.failure=(VkExecPreparationFailure){0};
 PL.nsets=2;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 assert(draw.pool_grant&&draw.pool_grant->finite&&draw.pool_grant->indices.comparison_absent&&
        draw.pool_grant->set!=ordinary_route_set&&!native_effects&&PL.samp[0]==777);
 assert(cells[draw.pool_grant->index][10*VK_POOL_TEX]==610&&
        cells[draw.pool_grant->index][VK_POOL_KINDS*VK_POOL_TEX]==888);
 assert(graphics_pool_prepare(&ex,&draw,1,0,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 vk_exec_graphics_draw_release_gpa_inputs(&draw);

 // Two authored comparison cells in the regular table need neither a copy nor ordinary-cell rewriting.
 reset(&ex,&draw);
 draw.state.vs=&comparison_domain;draw.state.fs=NULL;
 domain_capture=domain_comparison_proved=1;comparison_mask=3;
 sampler_zero_registered=sampler_zero_comparison=sampler_one_comparing=1;
 finite_member=1;finite_member_kinds=1u<<10;actual_kinds[1]=0x403;PL.kinds|=1u<<10;
 PL.samp[0]=778;PL.samp[1]=999;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 assert(draw.pool_grant&&draw.pool_grant->finite&&!draw.pool_grant->indices.comparison_absent&&
        draw.pool_grant->set==PL.set&&draw.pool_grant->indices.comparison_samplers.compare[0]==3);
 vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,draw.pool_grant,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set==PL.set);
 assert(!PL.set_comparison_sampler0[PL.cur]&&PL.samp[0]==778&&PL.samp[1]==999);
 PL.samp[1]=777; // Same generation and texture publication, wrong effective sampler descriptor.
 assert(graphics_pool_bind(&ex,&pooled,&set)&&!set&&record_failed);
 PL.samp[1]=999;record_failed=0;sampler_one_comparing=0;
 assert(!vk_exec_pool_grant_current(&ex,draw.pool_grant));
 vk_exec_pool_bind_enter((VkExecPoolBindScope){0});
 vk_exec_graphics_draw_release_gpa_inputs(&draw);

 // Separate, fully proved ordinary and comparison indices can share the faithful normal publication.
 reset(&ex,&draw);
 const GfxShader ordinary_domain={{{1,3},{2,0}},2};
 draw.state.vs=&comparison_domain;draw.state.fs=&ordinary_domain;
 domain_capture=domain_comparison_proved=1;comparison_mask=1;ordinary_mask=2;
 sampler_zero_registered=sampler_zero_comparison=1;sampler_one_comparing=0;
 finite_member=1;finite_member_kinds=1u<<10;actual_kinds[1]=0x403;PL.kinds|=1u<<10;
 PL.samp[0]=778;PL.samp[1]=999;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 assert(draw.pool_grant&&draw.pool_grant->finite&&draw.pool_grant->set==PL.set&&
        draw.pool_grant->indices.comparison_samplers.compare[0]==1&&
        draw.pool_grant->indices.comparison_samplers.ordinary[0]==2);
 vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,draw.pool_grant,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set==PL.set);
 draw.pool_grant->indices.comparison_samplers.ordinary[0]=1; // This single cell cannot serve both modes.
 assert(graphics_pool_bind(&ex,&pooled,&set)&&!set&&record_failed);
 vk_exec_pool_bind_enter((VkExecPoolBindScope){0});
 vk_exec_graphics_draw_release_gpa_inputs(&draw);

 // When every shared sampler is known comparing, reuse the legal whole publication without a table copy.
 reset(&ex,&draw);
 draw.state.vs=&comparison_domain;draw.state.fs=NULL;domain_capture=sampler_pool_comparison=1;
 PL.kinds |= 1u<<10;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 assert(draw.pool_grant&&!draw.pool_grant->finite&&!domain_attempts&&!word_captures);
 assert(graphics_pool_prepare(&ex,&draw,1,0,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 PL.sgen=0;
 assert(graphics_pool_prepare(&ex,&draw,1,0,none,NULL,0)==VK_EXEC_GRAPHICS_INVALIDATED);
 assert(!strcmp(draw.failure.stage,"pool-retained-comparison-sampler"));
 draw.failure=(VkExecPreparationFailure){0};PL.sgen=sampler_generation;
 sampler_generation++;
 assert(graphics_pool_prepare(&ex,&draw,1,0,none,NULL,0)==VK_EXEC_GRAPHICS_INVALIDATED);
 assert(!strcmp(draw.failure.stage,"pool-retained-comparison-sampler"));
 vk_exec_graphics_draw_release_gpa_inputs(&draw);

 // A genuinely broad alias capture refusal arrives at the selected draw with its actual scope and
 // exact member, disposition and producer classification intact. No binding or prefix was issued.
 reset(&ex,&draw);
 fail_pool_alias=1;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!draw.pool_grant&&!prefix_cuts&&!native_effects&&draw.failure.owner==VK_EXEC_PREPARATION_POOL_IMAGE&&
  draw.failure.reason==VK_EXEC_PREPARATION_ORDER&&draw.failure.slot==2&&draw.failure.role==2&&
  draw.failure.registration==96&&draw.failure.material_serial==123&&draw.failure.image==456&&
  draw.failure.detail==RC_REGISTRY_BLOCKED&&draw.failure.range.mapping==41&&draw.failure.coverage.gate==10&&
  !rc_registry_position_compare(draw.failure.sampled.read,position));
 const p_draw direct={4,64};
 const p_draw_idx indexed={7,99,NVN_INDEX_TYPE_UNSIGNED_SHORT,32};
 const p_draw_idx_ind indirect={7,NVN_INDEX_TYPE_UNSIGNED_SHORT,16,32};
 const p_draw_mesh_ind mesh_indirect={16};
 compatibility_opcode(&ex,&draw,OP_DRAW,&direct,4,64,0,0,0);
 compatibility_opcode(&ex,&draw,OP_DRAW_IDX,&indexed,7,99,1,0,0);
 compatibility_opcode(&ex,&draw,OP_DRAW_IDX_IND,&indirect,7,INT32_MAX,1,0,0);
 compatibility_opcode(&ex,&draw,OP_DRAW_MESH,NULL,3,INT32_MAX,0,1,1);
 compatibility_opcode(&ex,&draw,OP_DRAW_MESH_IND,&mesh_indirect,3,INT32_MAX,0,1,0);
 reset(&ex,&draw);
 saturate_compatibility_pool(&ex);
 draw.opcode=OP_DRAW_MESH_IND;
 draw.payload=&mesh_indirect;
 draw.compatibility=1;
 caps.mesh_fallback=1;
 selection_calls=0;
 assert(graphics_compat_pool_prepare(&ex,&draw,none,NULL)==VK_EXEC_GRAPHICS_COMPAT);
 assert(!selection_calls&&!prefix_cuts&&!native_effects);

 // A private comparison grant survives alongside the already issued ordinary set.
 reset(&ex,&draw);
 scoped_absent[0]=1;
 selected_status[0]=VK_TEXTURE_SAMPLE_ABSENT;
 Z.img[DV_DREF_2D].view=610;
 Z.samp_shadow=888;
 PL.samp[0]=777;
 PL.cur=1;
 PL.kinds|=1u<<10;
 PL.view[10*VK_POOL_TEX]=610;
 PL.logn=2;
 PL.log[0]=10*VK_POOL_TEX;
 const uint32_t carried=2*VK_POOL_TEX+1;
 PL.log[1]=carried;
 PL.last[carried]=1;
 cells[0][10*VK_POOL_TEX]=610;
 cells[0][VK_POOL_KINDS*VK_POOL_TEX]=cells[1][VK_POOL_KINDS*VK_POOL_TEX]=777;
 VkExecPoolGrant *ordinary=NULL, *shadow=NULL, *retry=NULL;
 VkExecPreparationFailure pressure;
 assert(vk_exec_pool_prepare_info(&ex,1u<<2,&ordinary,1,0,none,NULL,NULL)==VK_EXEC_GRAPHICS_READY);
 unsigned ordinary_index=ordinary->index;
 assert(ordinary_index==0);
 old_scope=vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,ordinary,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set==PL.sets[ordinary_index]);
 vk_exec_pool_bind_enter(old_scope);
 PL.image[carried]=101;
 PL.view[carried]=501;
 PL.publication[0]->installed[carried]=101;
 cells[0][carried]=501;
 uint64_t ordinary_generation=ordinary->generation;
 comparison_pair.entries[0].slot=0;
 comparison_pair.comparison_absent=1;
 unsigned writes_before=native_writes[ordinary_index];
 PL.nsets=1;
 assert(pool_prepare_indices(&ex,1u<<10,&shadow,1,1,none,NULL,NULL,&comparison_pair,&pressure)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!shadow&&pressure.reason==VK_EXEC_PREPARATION_PRESSURE&&!prefix_cuts);
 PL.nsets=2;
 PL.set_fence[1]=ex.fence-1;
 assert(pool_prepare_indices(&ex,1u<<10,&shadow,1,0,none,NULL,NULL,&comparison_pair,&pressure)==VK_EXEC_GRAPHICS_READY);
 unsigned shadow_index=shadow->index;
 assert(shadow_index==1&&prior_waits==1&&PL.set_synced[1]==PL.logn);
 assert(cells[1][10*VK_POOL_TEX]==610&&cells[1][VK_POOL_KINDS*VK_POOL_TEX]==888);
 assert(cells[1][carried]==501&&PL.publication[1]->installed[carried]==101&&
        PL.publication[0]->installed[carried]==101&&cells[0][carried]==501);
 assert(cells[0][VK_POOL_KINDS*VK_POOL_TEX]==777&&PL.samp[0]==777&&PL.cur==0&&PL.set==ordinary->set&&
        PL.set_generation[0]==ordinary_generation&&native_writes[0]==writes_before&&vk_exec_pool_grant_current(&ex,ordinary));
 assert(pool_prepare_indices(&ex,1u<<10,&retry,1,1,none,NULL,NULL,&comparison_pair,&pressure)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!retry&&pressure.reason==VK_EXEC_PREPARATION_PRESSURE&&!prefix_cuts);
 uint64_t first_shadow_generation=shadow->generation;
 uint64_t before_cancel_log=PL.logn;
 sampler_generation++;
 assert(!vk_exec_pool_grant_current(&ex,shadow));
 // Cancel before any native bind. A sampler log delta is unnecessary to restore the cell.
 vk_exec_pool_grant_release(&shadow);
 assert(!PL.set_fence[1]&&!PL.set_comparison_sampler0[1]&&cells[1][VK_POOL_KINDS*VK_POOL_TEX]==777&&
        PL.logn==before_cancel_log);
 comparison_pair.comparison_sampler_generation=sampler_generation;
 assert(pool_prepare_indices(&ex,1u<<10,&shadow,1,0,none,NULL,NULL,&comparison_pair,&pressure)==VK_EXEC_GRAPHICS_READY);
 assert(shadow->generation!=first_shadow_generation);
 old_scope=vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,shadow,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set==PL.sets[1]);
 vk_exec_pool_bind_enter(old_scope);
 vk_exec_pool_grant_release(&shadow);
 assert(PL.set_fence[1]==ex.fence&&cells[1][VK_POOL_KINDS*VK_POOL_TEX]==888);
 assert(pool_prepare_indices(&ex,1u<<10,&retry,1,0,none,NULL,NULL,&comparison_pair,&pressure)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!retry&&pressure.reason==VK_EXEC_PREPARATION_PRESSURE&&!prefix_cuts);
 assert(vk_exec_pool_grant_current(&ex,ordinary));
 vk_exec_pool_grant_release(&ordinary);
 assert(pool_prepare_indices(&ex,1u<<10,&shadow,1,1,none,NULL,NULL,&comparison_pair,&pressure)==VK_EXEC_GRAPHICS_READY);
 assert(prefix_cuts==1&&PL.set_fence[shadow->index]==ex.fence);
 old_scope=vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,shadow,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set==shadow->set);
 vk_exec_pool_bind_enter(old_scope);
 shadow_index=shadow->index;
 vk_exec_pool_grant_release(&shadow);
 // A native prefix released the old read. An ordinary publication can now catch up
 // and remove that private sampler even after generic history advanced independently.
 vk_exec_pool_registration_prefix_complete(ex.fence);
 PL.samp[0]=779;
 PL.log[PL.logn % POOL_LOG]=2*VK_POOL_TEX+1;
 PL.last[2*VK_POOL_TEX+1]=PL.logn++;
 PL.set_rebuild[shadow_index]=1;
 plat_mutex_lock(&pool_mt);vk_texture_lock();
 pool_catch_up_to((int)shadow_index,PL.set_synced[shadow_index],PL.logn);
 pool_restore_comparison_sampler0((int)shadow_index);
 pool_flush();
 vk_texture_unlock();plat_mutex_unlock(&pool_mt);
 assert(PL.rewrites==1&&!PL.set_rebuild[shadow_index]&&!PL.set_comparison_sampler0[shadow_index]);
 assert(cells[shadow_index][VK_POOL_KINDS*VK_POOL_TEX]==779);
 // Layout recreation resets local generations and markers with the ring storage.
 reset(&ex,&draw);
 assert(!PL.comparison_generation&&!PL.set_comparison_sampler0[1]&&!native_writes[1]);

 // The real ordered selector needs no sampler-index proof once both typed tables are published.
 reset(&ex,&draw);
 PL.split_samplers=1; Z.img[DV_DREF_2D].view=404;
 draw.state.vs=&comparison_domain;draw.state.fs=&ordinary_domain;
 PL.kinds|=1u<<10;
 assert(graphics_pool_prepare(&ex,&draw,1,1,none,NULL,0)==VK_EXEC_GRAPHICS_READY);
 assert(draw.pool_grant&&!draw.pool_grant->finite&&draw.pool_grant->sampler_generation==sampler_generation);
 assert(vk_exec_pool_grant_current(&ex,draw.pool_grant));
 ++sampler_generation;
 assert(!vk_exec_pool_grant_current(&ex,draw.pool_grant));
 --sampler_generation;
 const uint32_t held=draw.pool_grant->index, candidate=1-held;
 PL.samp[0]=811;PL.comparison[0]=812;PL.comparison[1]=813;
 const uint32_t companion0=POOL_IMAGE_ENTRIES+VK_POOL_SAMP;
 PL.log[PL.logn%POOL_LOG]=companion0;PL.last[companion0]=PL.logn++;
 plat_mutex_lock(&pool_mt);vk_texture_lock();
 pool_catch_up_to((int)candidate,0,PL.logn);
 assert(cells[candidate][companion0]==812&&!cells[candidate][POOL_IMAGE_ENTRIES]&&!cells[held][companion0]);
 PL.set_rebuild[candidate]=1;
 pool_catch_up_to((int)candidate,0,PL.logn);
 assert(cells[candidate][companion0]==812&&cells[candidate][companion0+1]==813&&cells[candidate][POOL_IMAGE_ENTRIES]==811);
 vk_texture_unlock();plat_mutex_unlock(&pool_mt);
 vk_exec_pool_grant_release(&draw.pool_grant);
 assert(!scope_refs);

 // A globally compiled Dref type has no authority over an ordinary original consumer.
 reset(&ex,&draw);
 PL.split_samplers=1; Z.img[DV_DREF_2D].view=404;
 vk_pool_kinds_used|=1u<<10;
 optional_comparison_refused=1;
 VkExecPoolGrant *cohort=NULL;
 VkExecPreparationFailure optional={0};
 assert(pool_prepare_indices(&ex,1u<<2,&cohort,1,1,none,NULL,NULL,NULL,&optional)==VK_EXEC_GRAPHICS_READY);
 assert(cohort&&PL.kinds==(1u<<2)&&!PL.registration_gone[3]&&vk_exec_pool_grant_current(&ex,cohort));
 assert(pool_refresh_kinds(NULL)==vk_pool_kinds_used);
 assert(!pool_order_current_for(1u<<10));
 old_scope=vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,cohort,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set==cohort->set);
 vk_exec_pool_bind_enter(old_scope);
 vk_exec_pool_grant_release(&cohort);
 assert(pool_prepare_indices(&ex,1u<<10,&cohort,1,1,none,NULL,NULL,NULL,&optional)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!cohort&&optional.slot==3&&optional.detail==POOL_SOURCE_REFUSED);
 assert(!scope_refs);
 assert(pool_prepare_indices(&ex,1u<<2,&cohort,1,1,none,NULL,NULL,NULL,&optional)==VK_EXEC_GRAPHICS_READY);
 assert(cohort&&!PL.registration_gone[3]&&vk_exec_pool_grant_current(&ex,cohort));
 vk_exec_pool_grant_release(&cohort);
 assert(!scope_refs);

 // Unreachable declarations can bind an immutable finite set without installing
 // the irrelevant incompatible cell. A later actual use still refuses that cell.
 reset(&ex,&draw);
 PL.split_samplers=1;
 optional_comparison_refused=1;
 VkExecPoolIndices inactive={.domain=1,.inactive=1};
 assert(pool_prepare_indices(&ex,1u<<10,&cohort,1,1,none,NULL,NULL,&inactive,&optional)==VK_EXEC_GRAPHICS_READY);
 assert(cohort&&cohort->finite&&cohort->indices.inactive&&!cohort->indices.count&&vk_exec_pool_grant_current(&ex,cohort));
 draw.pool_grant=cohort;
 draw.pool_indices=inactive;
 draw.pool_index_read=position;
 const GfxShader *inactive_shaders[]={draw.state.vs,draw.state.fs,draw.state.ts};
 memcpy(draw.pool_index_shaders,inactive_shaders,sizeof inactive_shaders);
 finite_inputs=1;
 draw.use.revision=&pooled;
 draw.use.fs_query_size_lod0_mask=VK_GRAPHICS_INACTIVE_POOL_ATTESTED;
 assert(graphics_pool_issue_retained(&ex,&draw,inactive_shaders,1u<<10)==VK_EXEC_GRAPHICS_READY);
 draw.use.fs_query_size_lod0_mask=0;
 assert(graphics_pool_issue_retained(&ex,&draw,inactive_shaders,1u<<10)==VK_EXEC_GRAPHICS_INVALIDATED);
 assert(!strcmp(draw.failure.stage,"pool-inactive-module-changed"));
 draw.pool_grant=NULL;
 old_scope=vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,cohort,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set==cohort->set);
 vk_exec_pool_bind_enter(old_scope);
 ++sampler_generation;
 assert(!vk_exec_pool_grant_current(&ex,cohort));
 --sampler_generation;
 vk_exec_pool_grant_release(&cohort);
 assert(!scope_refs);
 assert(pool_prepare_indices(&ex,1u<<10,&cohort,1,1,none,NULL,NULL,NULL,&optional)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!cohort&&optional.slot==3&&optional.detail==POOL_SOURCE_REFUSED);
 // An earlier cell can acquire a content version during the last selected cell's check.
 // Proof must restart at the first cell of the same finite publication, never the whole table.
 reset(&ex,&draw);
 available[1]=available[2]=0;actual_kinds[2]=0x103;missing_reason[3]=POOL_MISSING_HISTORY;
 const VkExecPoolIndices selected_pair={
   .count=2,.entries={{1,1u<<2},{2,1u<<3}}
 };
 VkExecPoolGrant *finite_grant=NULL;
 VkExecPreparationFailure finite_failure={0};
 exact_drift_at=2;exact_drift_mask=7;
 assert(pool_prepare_indices(&ex,(1u<<2)|(1u<<3),&finite_grant,1,0,none,NULL,NULL,
    &selected_pair,&finite_failure)==VK_EXEC_GRAPHICS_READY);
 assert(finite_grant&&finite_grant->finite&&exact_checks==4&&map_validations==1&&!host_proof&&!prefix_cuts&&!history_reports);
 assert(PL.registration_gone[3]==POOL_MISSING_HISTORY&&!pool_order_current());
 old_scope=vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,finite_grant,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&set==finite_grant->set);
 vk_exec_pool_bind_enter(old_scope);
 vk_exec_pool_grant_release(&finite_grant);
 assert(!scope_refs);

 // A selected earlier member actually changed: recheck must fail at its original cell.
 reset(&ex,&draw);
 available[1]=available[2]=0;actual_kinds[2]=0x103;exact_drift_at=2;exact_drift_mask=2;exact_remap_member=1;
 assert(pool_prepare_indices(&ex,(1u<<2)|(1u<<3),&finite_grant,1,0,none,NULL,NULL,
    &selected_pair,&finite_failure)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!finite_grant&&!scope_refs&&exact_checks==3&&finite_failure.slot==1&&
    !strcmp(finite_failure.stage,"finite-pool-member"));

 reset(&ex,&draw);
 available[1]=available[2]=0;actual_kinds[2]=0x103;exact_drift_at=1;exact_drift_mask=4;exact_drift_repeat=1;
 assert(pool_prepare_indices(&ex,(1u<<2)|(1u<<3),&finite_grant,1,0,none,NULL,NULL,
    &selected_pair,&finite_failure)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!finite_grant&&!scope_refs&&exact_checks==4&&finite_failure.detail==4&&
    !strcmp(finite_failure.stage,"finite-pool-authority-race"));

 reset(&ex,&draw);
 available[1]=available[2]=0;actual_kinds[2]=0x103;exact_drift_at=2;exact_close_read=1;
 assert(pool_prepare_indices(&ex,(1u<<2)|(1u<<3),&finite_grant,1,0,none,NULL,NULL,
    &selected_pair,&finite_failure)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!finite_grant&&!scope_refs&&exact_checks==2&&!strcmp(finite_failure.stage,"finite-pool-original-read"));

 reset(&ex,&draw);
 available[1]=available[2]=0;actual_kinds[2]=0x103;exact_drift_at=2;exact_recycle_set=1;
 assert(pool_prepare_indices(&ex,(1u<<2)|(1u<<3),&finite_grant,1,0,none,NULL,NULL,
    &selected_pair,&finite_failure)==VK_EXEC_GRAPHICS_UNAVAILABLE);
 assert(!finite_grant&&!scope_refs&&exact_checks==2&&!strcmp(finite_failure.stage,"finite-pool-physical-set"));

 // Continuous unrelated mapping changes are tolerated after exact selected topology is proved.
 reset(&ex,&draw);
 available[1]=available[2]=0;actual_kinds[2]=0x103;
 exact_drift_at=1;exact_drift_mask=2;exact_drift_repeat=1;
 assert(pool_prepare_indices(&ex,(1u<<2)|(1u<<3),&finite_grant,1,0,none,NULL,NULL,
    &selected_pair,&finite_failure)==VK_EXEC_GRAPHICS_READY);
 assert(finite_grant&&map_validations==1&&exact_mapping==5&&!host_proof);
 i=finite_grant->index;
 // An unrelated later mapping can revalidate, while a selected remap cannot reach bind.
 exact_drift_at=0;
 ++exact_mapping;
 assert(vk_exec_pool_grant_current(&ex,finite_grant));
 ++exact_mapping;
 ++member_revision[1];
 old_scope=vk_exec_pool_bind_enter((VkExecPoolBindScope){&pooled,finite_grant,25});
 assert(graphics_pool_bind(&ex,&pooled,&set)&&!set&&record_failed);
 assert(!native_effects&&PL.set_prepare_refs[i]==1);
 vk_exec_pool_bind_enter(old_scope);
 vk_exec_pool_grant_release(&finite_grant);
 assert(!scope_refs&&!PL.set_prepare_refs[i]&&!host_proof);

 puts("ordered pool grant: finite and typed whole tables, scoped kinds, publication, lifetime and pressure passed");
}
'''

names = (
    'pool_kind_type', 'pool_absent_view', 'pool_can_stay', 'pool_next_set', 'pool_release_completed_set',
    'vk_exec_pool_registration_prefix_complete', 'pool_sample_availability', 'pool_need_restore',
    'graphics_pool_identity_locked', 'pool_comparison_sampler_cells_locked', 'graphics_pool_grant_read_current',
    'vk_exec_pool_grant_current', 'graphics_pool_release_locked',
    'vk_exec_pool_grant_release', 'vk_exec_pool_bind_enter', 'graphics_pool_layout_matches', 'graphics_pool_bind', 'vk_exec_pool_shader_kinds',
    'pool_install_selected_member', 'pool_prepare_selected_member', 'pool_sync_selected_entries',
    'graphics_pool_snapshot', 'graphics_pool_restore_current', 'graphics_pool_restore_member', 'graphics_pool_missing',
    'pool_selected_epoch', 'pool_selected_epoch_changed', 'pool_selected_validate_locked',
    'pool_selected_members_current', 'pool_hold_grant_locked', 'graphics_pool_alias_pending',
    'image_alias_preparation_reason','pool_alias_refuse','pool_alias_progress_refused', 'pool_preparation_failure',
    'pool_entry_to', 'pool_restore_comparison_sampler0', 'pool_catch_up_to',
    'pool_comparison_absent_pair_locked', 'pool_comparison_absent', 'pool_comparison_indices_current',
    'pool_comparison_target_locked', 'pool_hold_comparison_locked', 'pool_wait_for_set', 'pool_prepare_indices',
    'vk_exec_pool_prepare_info', 'graphics_shader_uses_sampler_pool', 'graphics_pool_comparison_usage',
    'graphics_pool_comparison_samplers_current', 'graphics_pool_retained_indices_current', 'graphics_pool_issue_retained',
    'graphics_pool_has_domain', 'graphics_pool_prepare_selected', 'graphics_pool_prepare',
)

def definition(name):
    import re
    declaration = re.search(r'^[a-zA-Z_0-9 *]+[ *]' + name + r'\([^;{]*\)\s*\{', SOURCE, re.M)
    assert declaration, name
    return function(SOURCE[declaration.start():], name)

draw_pool = SOURCE[SOURCE.index('enum { POOL_MISSING_HISTORY'):SOURCE.index('#define POOL_SETS VK_POOL_SETS')]
for helper in ('pool_alias_member', 'pool_aliases_append'):
    if helper + '(' in draw_pool:
        draw_pool = draw_pool.replace(definition(helper), '')
text = PREFIX + draw_pool + STRUCTS
text += '\n'.join(definition(name) for name in names[:7])
text += definition('pool_refresh_kinds') + '\n'
text += NATIVE
# This harness isolates existing graphics and comparison set behavior.
text += r'''
static int compute_pool_query_current(const VkExec *ex,const VkExecPoolGrant *grant,const VkPipeInfo *info) {
    (void)ex;(void)grant;(void)info;abort();
}
static VkDescriptorSet compute_pool_query_bind(VkExec *ex,VkExecPoolGrant *grant,const VkPipeInfo *info) {
    (void)ex;(void)grant;(void)info;abort();
}
static void compute_pool_query_release(void *query) {(void)query;abort();}
static int graphics_pool_comparison_feedback(const VkExec *ex,const VkPipeInfo *info,const VkExecPoolGrant *grant) {
    (void)ex;(void)info;(void)grant;return 0; // Covered by retained_comparison_depth_check.py.
}
static void graphics_pool_comparison_refusal(VkExec *ex,const VkExecGraphicsDraw *draw,
    const GfxShader *const shaders[3],int comparison,int finite,VkExecPreparationFailure *failure) {
    assert(ex&&draw&&shaders&&comparison&&failure->stage);
    (void)finite;
}
static void pool_index_scalar_failure_report(RcRegistryPosition read) {(void)read;}
static void pool_index_guard_failure_before_whole(RcRegistryPosition read) {(void)read;}
static VkExecSampledFailure sampled_pool_receipt(const VkExecPreparationFailure *failure,RcRegistryPosition read,
    uint64_t program,uint64_t pipeline,const void *revision,const GfxShader *const shaders[3]) {
    (void)read;(void)program;(void)pipeline;(void)revision;(void)shaders;return failure->sampled;
}
'''

text += 'static int pool_comparison_absent_pair_locked(const VkExecPoolIndices *indices);\n'
if 'static const char pool_missing_comparison_standin[]' in SOURCE:
    start = SOURCE.index('static const char pool_missing_comparison_standin[]')
    text += SOURCE[start:SOURCE.index('\n', start)] + '\n'
start = SOURCE.index('enum { GRAPHICS_POOL_COMPARISON_NONE')
text += SOURCE[start:SOURCE.index('\n', start)] + '\n'
text += definition('pool_sampler_descriptors') + '\n'
text += """static unsigned pool_selected_epoch_changed(PoolSelectedEpoch checked);
static int pool_selected_members_current(const VkExec *ex, const VkExecPoolGrant *base,
    const VkExecPoolIndices *indices, RcRegistryPosition read, int prepare, PoolSelectedProof *proof);
"""
text += '\n'.join(definition(name) for name in names[7:]
                 if name not in ('pool_preparation_failure', 'graphics_pool_issue_retained') or name + '(' in SOURCE)
text += COMPAT
text += '\n'.join(definition(name) for name in ('graphics_index_needs_widening', 'graphics_indirect_select',
                                                'graphics_compat_pool_saturated', 'graphics_compat_pool_prepare',
                                                'vk_exec_graphics_draw_release_gpa_inputs_retry'))
group = function(EXEC_SOURCE, 'group_start')
start = group.index('        VkExecGraphicsDraw *draw = NULL;')
cut = group.index('        if (cut)', start)
append = group.index('        resource_group_append(g, ex, o, 1, draw, graphics_noop);', cut)
text += GROUP.replace('/* production group-front decision */', group[start:cut] + group[append:group.index('\n', append)])
text += MAIN

with tempfile.TemporaryDirectory(prefix='ordered-pool-', dir=ROOT / 'build/test') as tmp:
    cfile = Path(tmp) / 'ordered_pool.c'
    cfile.write_text(text)
    exe = cfile.with_suffix('.exe')
    env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get('PATH', ''))
    subprocess.run([str(CC), '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined',
                    '-fsanitize-trap=all', str(cfile), '-o', str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)
