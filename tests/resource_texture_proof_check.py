"""Exercise production same-operation proof reuse with the real authority registry."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = (ROOT / "runtime/vk_texture.c").read_text()
end = source.index("} TextureBindingProof;") + len("} TextureBindingProof;")
proof_type = source[source.rfind("typedef struct {", 0, end):end]
start = source.index("static int source_binding_ordered(")
end = source.index("\nstatic int source_binding_receipt(", start)
production = (function(source, "source_access_anchor") + function(source, "source_latest_read") + function(source, "source_stamp_equal") + function(source[source.index("static int source_permission_at(Tex *e, RcRegistryPosition position, RcRegistryReadObservation *out) {"):], "source_permission_at") + function(source[source.index("static int source_permission_current(Tex *e) {"):], "source_permission_current") + function(source[source.index("static void source_commit(Tex *e, const TextureSourceStamp *stamp, int same_contents) {"):], "source_commit") + function((ROOT / "runtime/vk_texture_content.h").read_text(), "content_version") + source[start:end])
prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#define RC_REGISTRY_SINGLE_THREADED 1
#define rc_registry_runtime unused_registry_runtime
#include "resource_core_registry.c"
#undef rc_registry_runtime
#include "texture_source.h"
#include "resource_texture_source.h"
typedef uint64_t VkImage;
''' + proof_type + r'''
typedef struct {int ready;struct {VkImage image;} resident;RcPreparedInput input;} TextureMaterial;
typedef TextureMaterial Material;
typedef struct {const void *lease;RcRegistrySourceAccess *access;TextureSourceAuthority *observed,*copied;uint32_t count;} TextureSourceMapping;
typedef struct {
 struct {VkImage image;uint64_t addr;} pub;
 struct {struct {uint64_t size;} L;} t;
 uint32_t checked,wgen,wt_n;uint64_t checked_authority;
 int source_blocked,uploaded,up_retry;
 TextureSourceStamp source_stamp;
 RcRegistrySourceAccess source_access;
 TextureSourceMapping *source_mapping;RcRegistryReadObservation source_vector_read;uint8_t source_canonical;
 TextureSourceAuthority source_authority,source_copy_authority;
 TextureSourceStamp source_authority_stamp,source_copy_stamp;void *source_copy;
 RcRegistryBindingReceipt compatibility_receipt;
 uint64_t compatibility_image,compatibility_mapping,compatibility_protection;
 uint32_t compatibility_wgen,compatibility_writes;
 TextureBindingProof binding_proof,pool_binding_proof;RcRegistryWrite image_authority;
 Material *material;RcRegistryReadVersion material_version;
} Tex;
static RcRegistry *registry;
static RcRegistry *rc_registry_runtime(void){return registry;}
static RcRegistryPosition position={1,2};static uint32_t submit_epoch=1;
static uint64_t mapping=2,protection=2;
static unsigned map_checks,protect_checks,ordinary_checks,compat_checks,effect_checks,ordinary_renews,compat_renews;
static int mapping_ok=1,protection_ok=1,effects_ok=1,race;
static int dependency_clean;static unsigned dependency_checks;
static int pool_deps_certificate(Tex *e){(void)e;dependency_checks++;return dependency_clean;}
static RcRegistryPosition vk_exec_resource_position(void){return position;}
static uint64_t mem_mapping_serial(void){return mapping;}
static uint64_t mem_protection_serial(void){return protection;}
static int mem_mapping_range_unchanged(uint64_t a,uint64_t n,uint64_t s){(void)a;(void)n;map_checks++;return mapping_ok&&mapping&&s&&s<=mapping;}
static int mem_protection_range_unchanged(uint64_t a,uint64_t n,uint64_t s){(void)a;(void)n;protect_checks++;return protection_ok&&protection&&s&&s<=protection;}
static int vk_exec_effects_read_allowed(uint64_t a,uint64_t n){(void)a;(void)n;effect_checks++;return effects_ok;}
static int source_effects_read_allowed(const Tex *e){return vk_exec_effects_read_allowed(e->pub.addr,e->t.L.size);}
static int mem_backing_current(const void *lease){(void)lease;return mapping_ok;}
static int host_hold,mutate_backing,extent_reads,held_extent_reads;
static RcSourceRange vector_ranges[]={{7,100,32},{8,200,32}};
static RcRegistryWrite intersecting;
static RcRegistryResult mapped_refusal;
enum {SOURCE_REFUSAL_PERMISSION=1};
static void source_note_refusal(Tex *e,int stage,RcRegistryResult result){
 (void)e;assert(!host_hold&&stage==SOURCE_REFUSAL_PERMISSION);mapped_refusal=result;
}
static int mem_backing_read_hold(const void *lease){
 assert(lease&&!host_hold);
 if(!mapping_ok)return 0;
 host_hold=1;
 return 1;
}
static void mem_read_release(void){assert(host_hold);host_hold=0;}
static RcRegistryResult count_mapped(RcRegistry *r,RcRegistrySourceAccess *a,uint32_t count,uint64_t scope,
 RcRegistryPosition p,RcRegistryReadObservation *out){
 assert(count==2);extent_reads++;held_extent_reads+=host_hold;
 if(mutate_backing==1)
  assert(rc_registry_write_begin(r,vector_ranges[1],RC_REGISTRY_BUFFER,&intersecting)==RC_REGISTRY_OK);
 RcRegistryResult result=rc_registry_source_access_read_extents(r,a,count,scope,p,out);
 if(mutate_backing==2)assert(rc_registry_cpu_changed(r,(RcSourceRange){8,1000,1})==RC_REGISTRY_OK);
 if(mutate_backing==3)assert(rc_registry_cpu_changed(r,vector_ranges[1])==RC_REGISTRY_OK);
 mutate_backing=0;
 return result;
}
static void inject_race(void){
 if(race==1)mapping++;
 if(race==2)assert(rc_registry_cpu_changed(registry,(RcSourceRange){1,1000,1})==RC_REGISTRY_OK);
 race=0;
}
static RcRegistryResult count_ordinary(RcRegistry *r,RcRegistrySourceAccess *a,uint64_t scope,
 RcRegistryPosition p,RcRegistryReadObservation *out){
 ordinary_checks++;RcRegistryStats before=rc_registry_stats(r);
 RcRegistryResult result=rc_registry_source_access_read(r,a,scope,p,out);
 RcRegistryStats after=rc_registry_stats(r);
 ordinary_renews+=(unsigned)(after.read_renew_fast+after.read_renew_queries-before.read_renew_fast-before.read_renew_queries);
 inject_race();return result;
}
static RcRegistryResult count_compat(RcRegistry *r,const RcRegistryBindingReceipt *old,uint64_t scope,
 RcRegistryPosition p,RcRegistryBindingReceipt *out){
 compat_checks++;RcRegistryResult result=rc_registry_binding_revalidate(r,old,scope,p,out);inject_race();return result;
}
static RcRegistryResult renew_compat(RcRegistry *r,const RcRegistryBindingReceipt *old,uint64_t scope,
 RcRegistryPosition p,RcRegistryBindingReceipt *out){
 compat_renews++;RcRegistryResult result=rc_registry_binding_renew(r,old,scope,p,out);inject_race();return result;
}
#define rc_registry_source_access_read count_ordinary
#define rc_registry_source_access_read_extents count_mapped
#define rc_registry_binding_revalidate count_compat
#define rc_registry_binding_renew renew_compat
static struct {uint64_t uses,use_empty,use_position,use_identity,use_feedback,use_mapping,use_protection,
 use_epoch,use_scope,use_backwards,use_edge,use_registry,use_ready;} compatibility_path;
static struct {uint64_t unordered,position,unchecked,stamp,feedback,material,mapping,protection,registry,backwards,scope,race;} ordinary_path;
''' + production
main = r'''
#define OK(x) assert((x)==RC_REGISTRY_OK)
int main(void){
 registry=rc_registry_create((RcRegistryConfig){8,8,2});assert(registry);
 RcRegistryHandle source_handle;OK(rc_registry_source(registry,7,(RcSourceRange){1,100,64},&source_handle));
 OK(rc_registry_declare_through(registry,(RcRegistryPosition){1,100}));
 Tex e={.pub={7,100},.t={{64}},.checked=1,.uploaded=1,.source_stamp={2,2,1,1}};
 OK(rc_registry_source_access_acquire(registry,7,(RcSourceRange){1,100,64},&e.source_access));
 TextureSourceAuthority prepared;
 OK(rc_registry_source_access_observe(registry,&e.source_access,1,position,&prepared));
 OK(rc_registry_source_access_commit(registry,&e.source_access,&prepared,0,NULL));
 for(unsigned alias=0;alias<400;alias++)for(unsigned kind=0;kind<16;kind++)assert(source_binding_current(&e));
 assert(ordinary_checks==1&&map_checks==1&&protect_checks==1&&effect_checks==6400);
 position.ordinal=4;assert(source_binding_current(&e));assert(ordinary_checks==2);
 assert(source_binding_current(&e)&&ordinary_checks==2); // Only the latest-query memo advances.
 // Later consumers do not replace the upload origin or exclude an earlier recording worker.
 assert(e.source_access._content.ordered.position.ordinal==2);
 assert(e.source_access._read.position.ordinal==2);
 position.ordinal=2;assert(source_binding_current(&e));position.ordinal=4;assert(source_binding_current(&e));
 RcRegistryReadObservation saved=e.source_access._read;
 e.source_access._read.position.ordinal=6;e.binding_proof=(TextureBindingProof){0};
 assert(source_binding_current(&e));
 assert(e.source_access._read.position.ordinal==4);
 e.source_access._read=saved;e.binding_proof=(TextureBindingProof){0};assert(source_binding_current(&e));
 unsigned n=ordinary_checks;e.pub.image++;assert(source_binding_current(&e)&&ordinary_checks==n+1);
 n=ordinary_checks;e.source_stamp.generation++;assert(source_binding_current(&e)&&ordinary_checks==n+1);
 n=ordinary_checks;mapping++;assert(source_binding_current(&e)&&ordinary_checks==n+1);
 n=ordinary_checks;protection++;assert(source_binding_current(&e)&&ordinary_checks==n+1);
 n=ordinary_checks;OK(rc_registry_cpu_changed(registry,(RcSourceRange){1,1000,1}));
 assert(source_binding_current(&e)&&ordinary_checks==n+1);
 effects_ok=0;assert(!source_binding_current(&e));effects_ok=1;
 Material material={.ready=1,.resident.image=e.pub.image,.input={1,2,3,4,RC_PROVIDER_CAPTURE}};
 e.material=&material;assert(source_binding_current(&e)&&e.material_version.id); // Existing exact material acquires permission without re-preparation.
 assert(e.material_version.id&&source_binding_current(&e));
 material.ready=0;assert(!source_binding_current(&e));material.ready=1;
 material.resident.image++;assert(!source_binding_current(&e));material.resident.image--;
 material.input.content++;assert(!source_binding_current(&e));material.input.content--;e.material=NULL;
 submit_epoch++;assert(!source_binding_current(&e));
 dependency_clean=1;unsigned dependencies=dependency_checks;
 assert(source_binding_current(&e)&&e.checked==submit_epoch&&dependency_checks==dependencies+1);
 assert(source_binding_current(&e)&&dependency_checks==dependencies+1);
 dependency_clean=0;submit_epoch--;e.checked=submit_epoch;
 // Changes between source checks and publication reject the result rather than populate the reusable proof.
 position.ordinal=6;race=1;n=ordinary_checks;assert(!source_binding_current(&e)&&ordinary_checks==n+1);
 assert(source_binding_current(&e)&&ordinary_checks==n+2);
 position.ordinal=8;race=2;assert(!source_binding_current(&e));assert(source_binding_current(&e));
 mapping++;mapping_ok=0;n=ordinary_checks;assert(!source_binding_current(&e)&&ordinary_checks==n);
 mapping_ok=1;assert(source_binding_current(&e));
 protection++;protection_ok=0;assert(!source_binding_current(&e));protection_ok=1;assert(source_binding_current(&e));

 // Compatibility receipts have distinct provenance, but share only their successful exact-operation validation.
 e.binding_proof=(TextureBindingProof){0};e.compatibility_image=e.pub.image;
 e.compatibility_mapping=mapping;e.compatibility_protection=protection;
 e.compatibility_wgen=e.wgen;e.compatibility_writes=e.wt_n;
 RcRegistryBindingObservation observed;OK(rc_registry_binding_begin(registry,source_handle,1,position,&observed));
 OK(rc_registry_binding_commit(registry,&observed,&e.compatibility_receipt));
 unsigned maps=map_checks,protects=protect_checks;n=compat_checks;
 for(unsigned alias=0;alias<400;alias++)for(unsigned kind=0;kind<16;kind++)assert(source_binding_compat_receipt(&e,NULL));
 assert(compat_checks==n+1&&map_checks==maps+1&&protect_checks==protects+1);
 RcRegistryBindingReceipt saved_compat=e.compatibility_receipt;
 // Replacing a successful untyped receipt with asserted image intent must not borrow its cached validation.
 // The local image/hold matches the new assertion, but the registry never issued this physical owner token.
 e.image_authority=(RcRegistryWrite){123456,1};
 e.compatibility_receipt.observation.image_binding=(RcRegistryImageBinding){e.pub.image,e.image_authority};
 assert(!source_binding_compat_receipt(&e,NULL)&&compat_checks==n+2);
 e.compatibility_receipt=saved_compat;e.image_authority=(RcRegistryWrite){0};n++;
 e.compatibility_receipt.observation.position.ordinal=position.ordinal+2;
 assert(!source_binding_compat_receipt(&e,NULL));e.compatibility_receipt=saved_compat;
 assert(source_binding_compat_receipt(&e,NULL)&&compat_checks==n+1);
 position.ordinal+=2;assert(source_binding_compat_receipt(&e,NULL)&&compat_checks==n+2);
 assert(source_binding_compat_receipt(&e,NULL)&&compat_checks==n+2);
 e.compatibility_receipt.observation.revision++;assert(!source_binding_compat_receipt(&e,NULL));e.compatibility_receipt=saved_compat;
 e.compatibility_receipt.observation.handle.id++;assert(!source_binding_compat_receipt(&e,NULL));e.compatibility_receipt=saved_compat;
 e.compatibility_receipt.observation.range.size--;assert(!source_binding_compat_receipt(&e,NULL));e.compatibility_receipt=saved_compat;
 e.compatibility_receipt.observation.next_read=position;assert(!source_binding_compat_receipt(&e,NULL));e.compatibility_receipt=saved_compat;
 e.pub.image++;assert(!source_binding_compat_receipt(&e,NULL));e.pub.image--;
 e.wgen++;assert(!source_binding_compat_receipt(&e,NULL));e.wgen--;
 e.wt_n++;assert(!source_binding_compat_receipt(&e,NULL));e.wt_n--;
 e.up_retry=1;assert(!source_binding_compat_receipt(&e,NULL));e.up_retry=0;
 effects_ok=0;assert(!source_binding_compat_receipt(&e,NULL));effects_ok=1;
 submit_epoch++;assert(!source_binding_compat_receipt(&e,NULL));submit_epoch--;
 n=compat_checks;mapping++;assert(source_binding_compat_receipt(&e,NULL)&&compat_checks==n+1);
 n=compat_checks;protection++;assert(source_binding_compat_receipt(&e,NULL)&&compat_checks==n+1);
 n=compat_checks;OK(rc_registry_cpu_changed(registry,(RcSourceRange){1,1000,1}));
 assert(source_binding_compat_receipt(&e,NULL)&&compat_checks==n+1);
 position.ordinal+=2;race=1;assert(!source_binding_compat_receipt(&e,NULL));assert(source_binding_compat_receipt(&e,NULL));
 position.ordinal+=2;race=2;assert(!source_binding_compat_receipt(&e,NULL));assert(source_binding_compat_receipt(&e,NULL));
 // Real overlapping writes revoke both modes; failures never install a success-cache entry.
 OK(rc_registry_cpu_changed(registry,(RcSourceRange){1,110,1}));
 assert(!source_binding_compat_receipt(&e,NULL));assert(!source_binding_current(&e));
 n=compat_checks;assert(!source_binding_compat_receipt(&e,NULL)&&compat_checks==n+1);
 rc_registry_destroy(registry);
 // A real typed receipt retains its exact physical-image owner at every source-helper entry.
 registry=rc_registry_create((RcRegistryConfig){8,8,2});assert(registry);
 OK(rc_registry_source(registry,7,(RcSourceRange){1,100,64},&source_handle));
 RcRegistryImageBinding image={.image=11};
 OK(rc_registry_hold_begin(registry,(RcSourceRange){1,100,64},RC_REGISTRY_IMAGE,&image.hold));
 OK(rc_registry_image_hold_bind(registry,image.hold,image.image));
 OK(rc_registry_declare_through(registry,(RcRegistryPosition){1,100}));position=(RcRegistryPosition){1,4};
 e=(Tex){.pub={11,100},.t={{64}},.uploaded=1,.compatibility_image=11,
  .compatibility_mapping=mapping,.compatibility_protection=protection,.image_authority=image.hold};
 OK(rc_registry_binding_begin_image(registry,source_handle,1,position,image,&observed));
 OK(rc_registry_binding_commit(registry,&observed,&e.compatibility_receipt));
 n=compat_checks;assert(source_binding_compat_receipt(&e,NULL)&&compat_checks==n+1);
 assert(source_binding_compat_receipt(&e,NULL)&&compat_checks==n+1);
 e.image_authority.id++;assert(!source_binding_compat_receipt(&e,NULL));e.image_authority=image.hold;
 e.image_authority.index++;assert(!source_binding_compat_receipt(&e,NULL));e.image_authority=image.hold;
 e.pub.image++;e.compatibility_image=e.pub.image;assert(!source_binding_compat_receipt(&e,NULL));
 rc_registry_destroy(registry);

 // A prefix boundary renews authority once; every alias then uses the refreshed proof without resolution/upload.
 registry=rc_registry_create((RcRegistryConfig){8,8,2});assert(registry);
 OK(rc_registry_source(registry,7,(RcSourceRange){1,100,64},&source_handle));
 OK(rc_registry_declare_through(registry,(RcRegistryPosition){1,4}));position=(RcRegistryPosition){1,2};
 e=(Tex){.pub={7,100},.t={{64}},.checked=1,.uploaded=1,.source_stamp={mapping,protection,1,1}};
 OK(rc_registry_source_access_acquire(registry,7,(RcSourceRange){1,100,64},&e.source_access));
 OK(rc_registry_source_access_observe(registry,&e.source_access,1,position,&prepared));
 OK(rc_registry_source_access_commit(registry,&e.source_access,&prepared,0,NULL));
 assert(source_binding_current(&e));
 OK(rc_registry_declare_through(registry,(RcRegistryPosition){1,20}));position.ordinal=6;
 n=ordinary_renews;assert(source_binding_current(&e)&&ordinary_renews==n+1);
 assert(e.source_access._read.sealed_through.ordinal==20);
 for(unsigned alias=0;alias<6400;alias++)assert(source_binding_current(&e));
 assert(ordinary_renews==n+1);
 RcRegistryWrite producer;
 OK(rc_registry_write_begin_at(registry,(RcSourceRange){1,100,64},RC_REGISTRY_BUFFER,(RcRegistryPosition){1,23},&producer));
 OK(rc_registry_declare_through(registry,(RcRegistryPosition){1,30}));position.ordinal=24;
 assert(!source_binding_current(&e)); // Old bytes cannot bypass a newly discovered producer.
 OK(rc_registry_write_cancel(registry,producer));
 assert(source_binding_current(&e));
 // Positionless maintenance retains the exact immutable material and content origin.
 Material retained={.ready=1,.resident.image=e.pub.image,.input={41,42,43,44,RC_PROVIDER_CAPTURE}};
 e.material=&retained;content_version(&e,&retained);assert(e.material_version.id);
 uint64_t version=e.material_version.id;RcRegistryPosition origin=e.source_access._content.ordered.position;
 position=(RcRegistryPosition){0};e.source_authority_stamp=e.source_stamp;
 OK(rc_registry_observe(registry,source_handle,submit_epoch,&e.source_authority.legacy));
 source_commit(&e,&e.source_stamp,1);content_version(&e,&retained);
 assert(e.material_version.id==version&&e.source_access._content.ordered.position.ordinal==origin.ordinal);
 assert(source_permission_current(&e));
 position=(RcRegistryPosition){1,24};assert(source_binding_current(&e));
 submit_epoch=2;assert(!source_binding_current(&e));e.checked=2;assert(source_binding_current(&e));
 n=ordinary_checks;assert(source_binding_current(&e)&&ordinary_checks==n);
 assert(e.material_version.id==version&&e.source_access._content.ordered.position.ordinal==origin.ordinal);
 OK(rc_registry_cpu_changed(registry,(RcSourceRange){1,100,1}));assert(!source_binding_current(&e));
 e.material=NULL;submit_epoch=1;
 // An executed image write changes provenance even after its reservation has completed CPU-visible.
 OK(rc_registry_source_access_acquire(registry,7,(RcSourceRange){1,100,64},&e.source_access));
 OK(rc_registry_source_access_observe(registry,&e.source_access,1,position,&prepared));
 OK(rc_registry_source_access_commit(registry,&e.source_access,&prepared,0,NULL));e.checked=1;
 OK(rc_registry_write_begin(registry,(RcSourceRange){1,100,64},RC_REGISTRY_IMAGE,&producer));
 OK(rc_registry_declare_through(registry,(RcRegistryPosition){1,40}));
 OK(rc_registry_write_submit(registry,producer,1));OK(rc_registry_write_complete(registry,producer,1));
 position.ordinal=34;assert(!source_binding_current(&e));
 position.ordinal=24;assert(!source_binding_current(&e)); // Never backdate permission over changed bytes.
 // An initially positionless exact material is adopted by the first ordered consumer without an upload.
 e.source_authority=(TextureSourceAuthority){0};position=(RcRegistryPosition){0};
 OK(rc_registry_observe(registry,source_handle,1,&e.source_authority.legacy));
 e.source_authority_stamp=e.source_stamp;source_commit(&e,&e.source_stamp,0);
 e.material=&retained;content_version(&e,&retained);assert(!e.material_version.id&&source_permission_current(&e));
 position=(RcRegistryPosition){1,36};assert(source_binding_current(&e)&&e.material_version.id);
 version=e.material_version.id;retained.input.payload++;
 assert(!source_binding_current(&e));content_version(&e,&retained);assert(e.material_version.id!=version);
 e.material=NULL;

 // The exact same-image compatibility path renews its seal without weakening a real producer boundary.
 image=(RcRegistryImageBinding){.image=11};
 OK(rc_registry_hold_begin(registry,(RcSourceRange){1,100,64},RC_REGISTRY_IMAGE,&image.hold));
 OK(rc_registry_image_hold_bind(registry,image.hold,image.image));
 e=(Tex){.pub={11,100},.t={{64}},.uploaded=1,.compatibility_image=11,
  .compatibility_mapping=mapping,.compatibility_protection=protection,.image_authority=image.hold};
 OK(rc_registry_binding_begin_image(registry,source_handle,1,position,image,&observed));
 OK(rc_registry_binding_commit(registry,&observed,&e.compatibility_receipt));
 assert(source_binding_compat_receipt(&e,NULL));
 OK(rc_registry_declare_through(registry,(RcRegistryPosition){1,50}));position.ordinal=42;
 n=compat_renews;assert(source_binding_compat_receipt(&e,NULL)&&compat_renews==n+1);
 assert(e.compatibility_receipt.observation.sealed_through.ordinal==50);
 for(unsigned alias=0;alias<6400;alias++)assert(source_binding_compat_receipt(&e,NULL));
 assert(compat_renews==n+1);
 OK(rc_registry_declare_through(registry,(RcRegistryPosition){1,60}));position.ordinal=52;race=1;
 assert(!source_binding_compat_receipt(&e,NULL)); // Guard race cannot seed a success cache after renewal.
 assert(source_binding_compat_receipt(&e,NULL));
 rc_registry_destroy(registry);
 // The actual mapped receipt linearizes against both source ranges. Another relevant producer cannot
 // slip in before that registry observation, and unrelated global churn cannot invalidate it.
 registry=rc_registry_create((RcRegistryConfig){8,8,2});assert(registry);
 RcRegistrySourceAccess access[2]={0};
 RcRegistrySourceProof vector_proof[2];
 TextureSourceMapping mapped={.lease=&mapped,.access=access,.count=2};
 position=(RcRegistryPosition){7,4};submit_epoch=3;
 OK(rc_registry_declare_through(registry,(RcRegistryPosition){7,10}));
 e=(Tex){.pub={2,100},.t={{64}},.checked=3,.uploaded=1,.source_stamp={mapping,protection,1,1},
   .source_mapping=&mapped,.source_canonical=1};
 for(unsigned i=0;i<2;i++)OK(rc_registry_source_access_acquire(registry,21,vector_ranges[i],&access[i]));
 OK(rc_registry_source_access_observe_extents(registry,access,2,3,position,vector_proof));
 OK(rc_registry_source_access_commit_extents(registry,access,vector_proof,2,0,NULL));
 uint64_t epoch=rc_registry_ordered_epoch(registry);
 mutate_backing=2;
 assert(source_binding_current(&e)&&extent_reads==1&&held_extent_reads==1&&!host_hold);
 assert(rc_registry_ordered_epoch(registry)!=epoch&&e.binding_proof.authority==epoch);
 assert(source_binding_current(&e)&&extent_reads==2);
 assert(source_binding_current(&e)&&extent_reads==2);
 position.ordinal=6;mutate_backing=1;
 assert(!source_binding_current(&e)&&extent_reads==3&&mapped_refusal==RC_REGISTRY_BLOCKED&&!host_hold);
 OK(rc_registry_write_cancel(registry,intersecting));
 OK(rc_registry_source_access_observe_extents(registry,access,2,3,position,vector_proof));
 OK(rc_registry_source_access_commit_extents(registry,access,vector_proof,2,1,NULL));
 assert(source_binding_current(&e)&&extent_reads==4);
 // The mapped host lock and permission stamp still reject an actual mapping/protection change.
 mapping++;mapping_ok=0;assert(!source_binding_current(&e)&&extent_reads==4&&mapped_refusal==RC_REGISTRY_STALE);
 mapping_ok=1;protection++;protection_ok=0;assert(!source_binding_current(&e)&&extent_reads==5);
 protection_ok=1;mutate_backing=3;
 assert(source_binding_current(&e)&&extent_reads==6); // This write follows the observation itself.
 assert(!source_binding_current(&e)&&extent_reads==7&&mapped_refusal==RC_REGISTRY_STALE);
 rc_registry_destroy(registry);
 puts("texture proofs: immutable provenance, maintenance adoption, finite permissions, alias reuse and racing writes passed");
}
'''
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="texture-proof-", dir=ROOT / "build/test") as tmp:
    path = Path(tmp) / "check.c"
    path.write_text(prefix + main)
    exe = path.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-Wno-unused-function", "-fsanitize=undefined", "-fsanitize-trap=all",
                    "-I", str(ROOT / "runtime"), str(path), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)
