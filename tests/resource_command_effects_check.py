"""Exercise actual ordered command ownership, sharing, cancellation and completion without Vulkan."""
from pathlib import Path
import os, subprocess, tempfile
from pass_end_lock_check import function
ROOT=Path(__file__).resolve().parents[1]
CC=Path(os.environ['CC'])
source=(ROOT/'runtime/resource_command_effects.c').read_text()
source='\n'.join(line for line in source.splitlines() if not line.startswith('#include'))
image=(ROOT/'runtime/resource_image_write.c').read_text()
image='\n'.join(line for line in image.splitlines() if not line.startswith('#include'))
host=(ROOT/'runtime/resource_host_write.c').read_text()
host=host[host.index('ResourceHostWrite resource_host_write_prepare('):]
host=host.replace('resource_host_write_prepare(', 'actual_host_prepare(').replace('resource_host_write_dispose(', 'actual_host_dispose(')
prefix=r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "resource_command_effects.h"
#include "resource_image_write.h"
typedef struct {
    RcInterval *intervals;
    RcRegistryWrite *tickets;
    RcRegistryWrite guard;
    uint32_t count;
} ResourceHostWrite;
static ResourceHostWrite actual_host_prepare(RcInterval *,uint32_t,RcRegistryWriterKind,int);
static void actual_host_dispose(ResourceHostWrite *,uint64_t,int,int);
static int actual_host;
static ResourceHostWrite resource_host_write_prepare(RcInterval *i,uint32_t n,RcRegistryWriterKind kind,int declared) {
    if (actual_host) return actual_host_prepare(i,n,kind,declared);
    (void)kind;(void)declared;return (ResourceHostWrite){.intervals=i,.count=n};
}
static void resource_host_write_dispose(ResourceHostWrite *w,uint64_t fence,int emitted,int visible) {
    if (actual_host) { actual_host_dispose(w,fence,emitted,visible); return; }
    (void)fence;(void)emitted;(void)visible;free(w->intervals);*w=(ResourceHostWrite){0};
}
static RcRegistry *registry;
static RcRegistry *fixture_registry(void) { return registry; }
#define rc_registry_runtime() fixture_registry()
typedef int PlatMutex;
#define PLAT_MUTEX_INIT 0
static void plat_mutex_lock(PlatMutex *p){assert(!*p);*p=1;}
static void plat_mutex_unlock(PlatMutex *p){assert(*p);*p=0;}
static int trace_requested;
static unsigned trace_log_count;
static char trace_log[4096];
static const char *trace_getenv(const char *key) {
 assert(!strcmp(key,"NX2NAVI_IMAGE_PRODUCER_TRACE"));
 return trace_requested ? "1" : NULL;
}
#define getenv trace_getenv
static void rt_log(const char *s,...) {
 if (strncmp(s,"resource image producer ",24)) return;
 va_list args; va_start(args,s); vsnprintf(trace_log,sizeof trace_log,s,args); va_end(args);
 assert(trace_log_count<64);
 trace_log_count++;
}
static void rt_fatal(const char *s){fputs(s,stderr);abort();}
static int nvn_gpu_only(uint64_t a){return a>=UINT64_C(0xf000000000000000);}
static struct {uint64_t fence;void(*done)(void*);void(*cancel)(void*,int);void *arg;} q[8300];
static unsigned nq;
static void vk_completion_add_owned(uint64_t f,void(*done)(void*),void(*cancel)(void*,int),void *arg){
 assert(nq<sizeof q/sizeof q[0]);q[nq++]=(typeof(q[0])){f,done,cancel,arg};
}
'''
draw=(ROOT/'runtime/vk_exec_draw.c').read_text()
start=draw.index('    uint32_t write_class=gfx_shader_guest_writes(shader);')
end=draw.index('    if ((write_class || queried_img) && !effect_storage_phase_owns_stage(d, st))', start)
fallback=draw[draw.index('compatibility:\n',end):draw.index('\n}',draw.index('compatibility:\n',end))]
semantic=r'''
#include "resource_core_effects.h"
typedef struct {unsigned flags,cbuf_slot;} GfxShaderRes;
typedef struct {uint32_t classification;GfxShaderRes reflection;} GfxShader;
typedef struct {
 ResourceCommandEffects *commands;
 RcRegistryPosition position;
 const GfxShader *shader[8];
 unsigned image_count,gpa_unknown,gpa_count,gpa_recorded;
} EffectDeclaration;
static EffectDeclaration *effect_current;
static RcRegistryPosition source_read_position;
static RcRegistryPosition vk_exec_resource_position(void){return source_read_position;}
static void effect_images_truncate(EffectDeclaration *d,unsigned n){d->image_count=n;}
// The reflected-semantic fixture binds no storage writers.
typedef struct {int unused;} VkExec;
static void effect_ordinary_storage(EffectDeclaration *d,VkExec *ex,int stage,const GfxShader *shader,int opaque){
    assert(d&&ex&&stage==0&&shader&&opaque);
}
static void effect_declare_cold_storage(EffectDeclaration *d,int stage){assert(d&&stage==0);}
#define GFX_RES_IMG 0
#define GFX_RES_SSBO 1
#define VK_GPA_CB_SLOT 0x40
static struct {unsigned readonly_stages,opaque_stages;} effect_stats;
static uint32_t gfx_shader_guest_writes(const GfxShader *s){return s->classification;}
static int gfx_shader_write_inventory_complete(const GfxShader *s){return !s->classification;}
static void semantic_stage(EffectDeclaration *d,const GfxShader *shader){
 unsigned first_image=0;int st=0;VkExec state={0},*ex=&state;
'''+draw[start:end]+'''if(!write_class){effect_stats.readonly_stages++;return;}
'''+fallback+"}\n"
ordered_read_gate=function(draw,'vk_exec_effects_read_allowed_range')

tests=r'''
static RcRegistryHandle source(uint64_t a){RcRegistryHandle h;assert(rc_registry_source(registry,a,(RcSourceRange){1,a,8},&h)==RC_REGISTRY_OK);return h;}
static int visible(RcRegistryHandle h){RcRegistryObservation o;return rc_registry_observe(registry,h,1,&o)==RC_REGISTRY_OK;}
static ResourceCommandEffects *command(uint64_t scene,uint64_t ordinal){
 assert(rc_registry_declare_through(registry,(RcRegistryPosition){scene,ordinal-1})==RC_REGISTRY_OK);
 return resource_effects_create((RcRegistryPosition){scene,ordinal});
}
static void seal(uint64_t scene,uint64_t ordinal){assert(rc_registry_declare_through(registry,(RcRegistryPosition){scene,ordinal})==RC_REGISTRY_OK);}
static void handoff(uint64_t scene,uint64_t ordinal){assert(rc_registry_handoff_through(registry,(RcRegistryPosition){scene,ordinal})==RC_REGISTRY_OK);}
static void projected_native_image_fate(void) {
 const RcInterval interval={UINT64_C(0xf000000000001000),7,{6,0,0x2000}};
 const RcRegistryPosition output={1,5},read={1,6};
 for (int disposition=0;disposition<4;disposition++) {
  registry=rc_registry_create((RcRegistryConfig){8,8,4});assert(registry);
  ResourceCommandEffects *e=command(output.scene,output.ordinal);
  assert(resource_effects_add_intervals(e,&interval,1,&interval,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
  seal(output.scene,output.ordinal);
  RcRegistryImageWriteExtent owner={.interval=interval};
  assert(rc_registry_image_hold_begin_extents(registry,777,&interval,1,&owner.hold)==RC_REGISTRY_OK);
  ResourceCommandEffects *prior=resource_effects_enter(e);
  ResourceImageScope scope={0};
  resource_image_scope_begin(&scope,output.scene);
  resource_image_prepare_mapped(777,interval.address,interval.range.size,&interval,1);
  if (disposition!=3) {
   resource_image_written(777,interval.address);
   resource_image_owned_mapped(777,interval.address,&interval,&owner.hold,1);
   const RcRegistryNativeImageWrite projection={owner,&interval.range,1};
   assert(rc_registry_native_image_write_fragments(registry,777,&projection,1,output)==RC_REGISTRY_OK);
  }
  resource_image_scope_end(&scope);
  resource_effects_leave(prior);
  resource_effects_release(e);
  seal(read.scene,read.ordinal);
  RcRegistryNativeImageFragment fragment;
  uint32_t count;
  RcRegistryResult observed=rc_registry_native_image_fragments(registry,&interval,1,read,&fragment,1,&count);
  if (disposition==3) {
   assert(observed==RC_REGISTRY_BLOCKED&&!count&&!nq);
  } else {
   assert(observed==RC_REGISTRY_OK&&count==1&&fragment.binding.hold.id==owner.hold.id&&nq==1);
   handoff(output.scene,output.ordinal);
   if (disposition==0) q[0].done(q[0].arg);
   else q[0].cancel(q[0].arg,disposition==2);
   nq=0;
   observed=rc_registry_native_image_fragments(registry,&interval,1,read,&fragment,1,&count);
   assert(observed==(disposition==0?RC_REGISTRY_OK:RC_REGISTRY_BLOCKED));
   assert(rc_registry_stats(registry).writer_foreign==(unsigned)(disposition==2));
  }
  rc_registry_destroy(registry);
 }
}
static void ordered_retained_image_effects(void) {
 const RcInterval old={UINT64_C(0xf000000000001000),7,{6,0,0x2000}};
 const RcInterval current={old.address,old.mapping,{6,0,0x4000}};
 const RcRegistryPosition before={1,3}, replace={2,5}, read={2,6};
 registry=rc_registry_create((RcRegistryConfig){12,16,4});assert(registry);
 RcRegistryWrite old_hold,new_hold;
 ResourceCommandEffects *a=command(before.scene,before.ordinal);
 assert(resource_effects_add_intervals(a,&old,1,&old,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
 seal(before.scene,before.ordinal);
 ResourceCommandEffects *other=resource_effects_enter(a);
 {RESOURCE_IMAGE_SCOPE(scope,before.scene);
  resource_image_prepare_mapped(601,old.address,old.range.size,&old,1);
  assert(rc_registry_image_hold_begin_extents(registry,601,&old,1,&old_hold)==RC_REGISTRY_OK);
  assert(!resource_effects_image_emitted_intervals(601,&old,&old_hold,1,before,1));
  resource_image_written(601,old.address);
  assert(!resource_effects_image_emitted_intervals(601,&old,&old_hold,1,before,1));
  resource_image_owned_mapped(601,old.address,&old,&old_hold,1);
  assert(!resource_effects_image_emitted_intervals(602,&old,&old_hold,1,before,1));
  assert(resource_effects_image_emitted_intervals(601,&old,&old_hold,1,before,1));
 }
 resource_effects_leave(other);
 resource_effects_release(a);
 assert(nq==1); // exact legacy fate owns the positioned provider through handoff
 handoff(before.scene,before.ordinal);
 q[0].done(q[0].arg);nq=0;
 a=command(replace.scene,replace.ordinal);
 assert(resource_effects_add_intervals(a,&current,1,&current,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
 seal(read.scene,read.ordinal);
 other=resource_effects_enter(a);
 {RESOURCE_IMAGE_SCOPE(scope,replace.scene);
  resource_image_prepare_mapped(602,current.address,current.range.size,&current,1);
  resource_image_written(602,current.address);
  assert(rc_registry_image_hold_begin_extents(registry,602,&current,1,&new_hold)==RC_REGISTRY_OK);
  resource_image_owned_mapped(602,current.address,&current,&new_hold,1);
  assert(resource_effects_image_emitted_intervals(602,&current,&new_hold,1,replace,1));
  RcRegistryNativeImageWrite output={{current,new_hold},&current.range,1};
  assert(rc_registry_native_image_write_fragments(registry,602,&output,1,replace)==RC_REGISTRY_OK);
  resource_effects_recorded(a,1);
  assert(rc_registry_native_image_covers(registry,602,&current,1,read));
 }
 resource_effects_leave(other);
 resource_effects_release(a);
 assert(nq==1);
 handoff(replace.scene,replace.ordinal);
 q[0].done(q[0].arg);nq=0;
 assert(rc_registry_native_image_covers(registry,602,&current,1,read));
 rc_registry_destroy(registry);
}
typedef struct {int known_unhanded, disposed;} ResidentTestPart;
static int retained_part_known_unhanded(const void *part) {
 return ((const ResidentTestPart *)part)->known_unhanded;
}
static void retained_part_dispose(void *part,void *owner,int known_unhanded) {
 ResidentTestPart *actual=part;
 assert(part==owner && actual->known_unhanded==known_unhanded && !actual->disposed);
 actual->disposed=1;
}
static void resident_image_split_fate(void) {
 const RcInterval plane={UINT64_C(0xf000000003f48e00),1349,{1346,0x3f48e00,0xb64000}};
 const RcRegistryPosition initial={1,3}, output={1,9}, read={1,12};
 enum { COMPLETED, KNOWN_UNHANDED, UNCERTAIN };
 for (int fate=COMPLETED;fate<=UNCERTAIN;fate++) {
  registry=rc_registry_create((RcRegistryConfig){12,16,4});assert(registry);
  ResidentTestPart part={.known_unhanded=fate==KNOWN_UNHANDED};
  const uint64_t image=104032;
  RcRegistryImageWriteExtent holder={.interval=plane};
  assert(rc_registry_image_hold_begin_extents(registry,image,&plane,1,&holder.hold)==RC_REGISTRY_OK);
  ResourceCommandEffects *seed=command(initial.scene,initial.ordinal);
  assert(resource_effects_add_pinned_image(seed,image,&plane,1)==RC_REGISTRY_OK);
  seal(initial.scene,initial.ordinal);
  ResourceCommandEffects *prior=resource_effects_enter(seed);
  resource_effects_emitted();
  assert(resource_effects_image_prepare_pinned(image,&plane,1)>0);
  resource_effects_image_written_interval(image,plane);
  resource_effects_image_owned_interval(image,plane,holder.hold);
  assert(resource_effects_image_emitted_intervals(image,&plane,&holder.hold,1,initial,1));
  const RcRegistryNativeImageWrite ready={holder,&plane.range,1};
  assert(rc_registry_native_image_write_fragments(registry,image,&ready,1,initial)==RC_REGISTRY_OK);
  resource_effects_leave(prior);
  resource_effects_release(seed);
  assert(nq==1);
  handoff(initial.scene,initial.ordinal);
  q[0].done(q[0].arg);nq=0;

  ResourceCommandEffects *atomic=command(output.scene,output.ordinal);
  assert(resource_effects_add_pinned_image(atomic,image,&plane,1)==RC_REGISTRY_OK);
  seal(read.scene,read.ordinal);
  prior=resource_effects_enter(atomic);
  resource_effects_emitted();
  assert(resource_effects_image_prepare_pinned(image,&plane,1)>0);
  resource_effects_image_written_interval(image,plane);
  resource_effects_image_owned_interval(image,plane,holder.hold);
  assert(resource_effects_image_emitted_intervals(image,&plane,&holder.hold,1,output,0));
  assert(resource_effects_image_preserved_intervals(image,&plane,&holder.hold,1,output,0));
  resource_effects_retain_part_disposition(&part,retained_part_known_unhanded,&part,retained_part_dispose);
  resource_effects_leave(prior);
  resource_effects_release(atomic);
  assert(nq==1);
  assert(!rc_registry_native_image_covers(registry,image,&plane,1,read));
  assert(rc_registry_native_image_resident_covers(registry,image,&holder.hold,1,&plane,1,read));

  if (fate==COMPLETED) handoff(output.scene,output.ordinal);
  // The generic completion reports success after the prefix. The exact tail witness overrides it.
  if (fate==UNCERTAIN) q[0].cancel(q[0].arg,1);
  else q[0].done(q[0].arg);
  nq=0;
  assert(part.disposed);
  assert(rc_registry_native_image_covers(registry,image,&plane,1,read)==(fate==KNOWN_UNHANDED));
  assert(rc_registry_native_image_resident_covers(registry,image,&holder.hold,1,&plane,1,read)==(fate!=UNCERTAIN));
  rc_registry_destroy(registry);
 }
}
static void bound_whole_image_fate(void) {
 const RcInterval interval={UINT64_C(0xf000000000001000),7,{6,0,0x2000}};
 const RcRegistryPosition output={1,5},next_read={2,2};
 enum { WHOLE, PARTIAL, CANCELED, UNCERTAIN, SKIPPED };
 for (int mode=WHOLE;mode<=SKIPPED;mode++) {
  registry=rc_registry_create((RcRegistryConfig){8,8,4});assert(registry);
  ResourceCommandEffects *e=command(output.scene,output.ordinal);
  assert(resource_effects_add_intervals(e,&interval,1,&interval,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
  seal(output.scene,output.ordinal);
  RcRegistryImageWriteExtent owner={.interval=interval};
  assert(rc_registry_image_hold_begin_extents(registry,778,&interval,1,&owner.hold)==RC_REGISTRY_OK);
  ResourceCommandEffects *prior=resource_effects_enter(e);
  {RESOURCE_IMAGE_SCOPE(scope,output.scene);
   resource_image_prepare_mapped(778,interval.address,interval.range.size,&interval,1);
   if (mode!=SKIPPED) {
    resource_image_written(778,interval.address);
    resource_image_owned_mapped(778,interval.address,&interval,&owner.hold,1);
    assert(rc_registry_native_image_written(registry,778,&interval,&owner.hold,1,output));
    if (mode!=PARTIAL) assert(rc_registry_native_image_whole_written(registry,778,&owner,1,output));
   }
  }
  resource_effects_leave(prior);
  resource_effects_release(e);
  const unsigned owns_fate=mode!=PARTIAL&&mode!=SKIPPED;
  assert(nq==owns_fate);
  if (mode==WHOLE||mode==UNCERTAIN) handoff(output.scene,output.ordinal);
  if (mode==WHOLE) q[0].done(q[0].arg);
  else if (owns_fate) q[0].cancel(q[0].arg,mode==UNCERTAIN);
  nq=0;
  seal(next_read.scene,next_read.ordinal);
  handoff(next_read.scene,1);
  RcRegistryResult move=rc_registry_native_image_relocate(registry,778,779,&owner,1,output,next_read);
  assert((move==RC_REGISTRY_OK)==(mode==WHOLE));
  assert(rc_registry_native_image_covers(registry,779,&interval,1,next_read)==(mode==WHOLE));
  assert(rc_registry_stats(registry).writer_foreign==(unsigned)(mode==UNCERTAIN));
  rc_registry_destroy(registry);
 }
}
static void native_buffer_output_fate(void) {
 const RcRegistryPosition output={1,5},read={1,6};
 const RcRegistryBufferSpan write={{0x9014,14,{42,100,24}},{30,1,208}};
 RcRegistryBufferSpan selected={{0xa000,15,{42,96,32}},{30,1,204}};
 for (int action=0;action<4;action++) {
  registry=rc_registry_create((RcRegistryConfig){8,12,3});assert(registry);
  RcRegistryGpuCapture base={0};
  assert(rc_registry_gpu_capture_begin(registry,&selected.interval.range,1,&base)==RC_REGISTRY_OK);
  ResourceCommandEffects *e=command(output.scene,output.ordinal);
  assert(resource_effects_add_intervals(e,&write.interval,1,&write.interval,1,RC_REGISTRY_BUFFER)==RC_REGISTRY_OK);
  seal(output.scene,read.ordinal);
  ResourceCommandEffects *prior=resource_effects_enter(e);
  assert(resource_effects_buffer_prepare_intervals(&write.interval,1,1));
  resource_effects_buffer_prepare_native(&write.interval,1,write.interval.address,write.native);
  RcRegistryBufferFragment f;
  uint32_t count;
  assert(rc_registry_gpu_capture_buffer_fragments(registry,base,&selected,1,read,&f,1,&count)==RC_REGISTRY_BLOCKED&&!count);
  if (action!=3) {
   resource_effects_buffers_commit(); /* analogous to actual VK copy/fill emitted immediately before */
   assert(rc_registry_gpu_capture_buffer_fragments(registry,base,&selected,1,read,&f,1,&count)==RC_REGISTRY_OK&&count==1);
   assert(f.interval.address==0xa004&&f.interval.mapping==15&&f.interval.range.size==24&&f.output.ordinal==5);
  }
  resource_effects_leave(prior);
  resource_effects_release(e);
  assert(nq==(unsigned)(action!=3));
  if (action==0) {
   handoff(output.scene,output.ordinal);q[0].done(q[0].arg);
  } else if (action==1) q[0].cancel(q[0].arg,0);
  else if (action==2) q[0].cancel(q[0].arg,1);
  nq=0;
  assert(rc_registry_gpu_capture_buffer_fragments(registry,base,&selected,1,read,&f,1,&count)==
    (action==2?RC_REGISTRY_BLOCKED:RC_REGISTRY_OK));
  assert(count==(unsigned)(action==0));
  assert(rc_registry_gpu_capture_end(registry,&base)==RC_REGISTRY_OK);
  rc_registry_destroy(registry);
 }
}

typedef struct { RcRegistryBufferSpan placement; int initialized, stale, checks, releases; } PreserveCheck;
static int preserve_current(void *opaque,RcRegistryBufferSpan span,RcRegistryPosition read,int *initialized) {
 PreserveCheck *check=opaque;
 check->checks++;
 assert(read.scene==1 && read.ordinal==4);
 assert(span.native.allocation==check->placement.native.allocation && span.native.offset==check->placement.native.offset);
 assert(span.interval.mapping==check->placement.interval.mapping && rc_range_equal(span.interval.range,check->placement.interval.range));
 *initialized=check->initialized;
 return !check->stale;
}
static void preserve_release(void *opaque){((PreserveCheck *)opaque)->releases++;}
static void preserved_buffer_output_fate(void) {
 const RcRegistryPosition output={1,5},read={1,6};
 const RcInterval interval={0x9014,14,{42,100,24}};
 for (unsigned mode=0;mode<5;mode++) {
  registry=rc_registry_create((RcRegistryConfig){8,12,3});assert(registry);
  PreserveCheck proof={.placement={interval,{30,mode&1,208}},.initialized=mode!=1,.stale=mode==4};
  RcRegistryGpuCapture capture={0};
  assert(rc_registry_gpu_capture_begin(registry,&interval.range,1,&capture)==RC_REGISTRY_OK);
  RcRegistryWrite earlier={0};
  if (mode==1) {
   seal(1,2);assert(rc_registry_write_begin_at(registry,interval.range,RC_REGISTRY_BUFFER,(RcRegistryPosition){1,3},&earlier)==RC_REGISTRY_OK);
   seal(1,3);assert(rc_registry_buffer_write_record(registry,earlier,proof.placement,(RcRegistryPosition){1,3})==RC_REGISTRY_OK);
  }
  ResourceCommandEffects *e=command(output.scene,output.ordinal);
  assert(resource_effects_add_intervals(e,&interval,1,&interval,1,RC_REGISTRY_BUFFER)==RC_REGISTRY_OK);
  seal(output.scene,read.ordinal);
  ResourceCommandEffects *old=resource_effects_enter(e);
  assert(resource_effects_buffer_prepare_intervals(&interval,1,0));
  assert(resource_effects_buffer_preserve_native(&interval,1,interval.address,proof.placement.native,
      &proof,preserve_current,preserve_release));
  uint32_t count;
  RcRegistryBufferFragment fragment;
  assert(rc_registry_gpu_capture_buffer_fragments(registry,capture,&proof.placement,1,read,&fragment,1,&count)==RC_REGISTRY_BLOCKED);
  assert(!proof.checks&&!count);
  if (mode!=2) {
   resource_effects_buffers_commit();
   assert(proof.checks==1);
   const RcRegistryResult result=rc_registry_gpu_capture_buffer_fragments(registry,capture,&proof.placement,1,read,&fragment,1,&count);
   if(mode==4) assert(result==RC_REGISTRY_BLOCKED&&!count);
   else assert(result==RC_REGISTRY_OK && count==1 && fragment.output.ordinal==5);
  }
  resource_effects_leave(old);resource_effects_release(e);
  assert(proof.releases==1 && nq==(unsigned)(mode!=2));
  if (mode==3) q[0].cancel(q[0].arg,0);
  else if (mode!=2) { handoff(1,5);q[0].done(q[0].arg); }
  nq=0;
  assert(rc_registry_gpu_capture_end(registry,&capture)==RC_REGISTRY_OK);
  rc_registry_destroy(registry);
 }
}
static void native_buffer_access_is_not_write_coverage(void) {
 const RcRegistryPosition output={1,5},read={1,6};
 const RcRegistryBufferSpan write={{0x9014,14,{42,100,24}},{30,1,208}};
 const RcRegistryBufferSpan word={{0x901c,14,{42,108,4}},{30,1,216}};
 registry=rc_registry_create((RcRegistryConfig){8,12,3});assert(registry);
 RcRegistryGpuCapture base={0};
 assert(rc_registry_gpu_capture_begin(registry,&word.interval.range,1,&base)==RC_REGISTRY_OK);
 ResourceCommandEffects *e=command(output.scene,output.ordinal);
 RcSourceRange legacy={RC_GUEST_COMPATIBILITY_BACKING,write.interval.address,write.interval.range.size};
 resource_effects_add(e,legacy,legacy,RC_REGISTRY_BUFFER);
 assert(resource_effects_add_intervals(e,&write.interval,1,&write.interval,1,RC_REGISTRY_BUFFER)==RC_REGISTRY_OK);
 seal(output.scene,read.ordinal);
 ResourceCommandEffects *prior=resource_effects_enter(e);
 assert(resource_effects_buffer_prepare(write.interval.address,write.interval.range.size,1));
 assert(resource_effects_buffer_prepare_intervals(&write.interval,1,1));
 resource_effects_buffer_access_native(&write.interval,1,write.interval.address,write.native);
 assert(rc_registry_buffer_access_check(registry,word,read)==RC_REGISTRY_BLOCKED);
 resource_effects_buffers_commit();
 resource_effects_recorded(e,1);
 assert(rc_registry_buffer_access_check(registry,word,read)==RC_REGISTRY_OK);
 RcRegistryBufferFragment fragment;uint32_t count=0;
 assert(rc_registry_gpu_capture_buffer_fragments(registry,base,&word,1,read,&fragment,1,&count)==RC_REGISTRY_BLOCKED);
 assert(!count); /* Access provenance never becomes definite native output coverage. */
 resource_effects_leave(prior);
 resource_effects_release(e);
 assert(nq==1);
 handoff(output.scene,output.ordinal);q[0].done(q[0].arg);nq=0;
 assert(rc_registry_gpu_capture_end(registry,&base)==RC_REGISTRY_OK);
 rc_registry_destroy(registry);
}
static void sampled_input_before_gpa(void){
 registry=rc_registry_create((RcRegistryConfig){32,16,4});assert(registry);
 RcRegistryPosition read={14421,40}, output={14421,41}, later={14421,42};
 RcSourceRange contents={14,0xa7051c00,0xab400}, unrelated={14,0x5000,16};
 RcRegistryWrite older, gpa;
 assert(rc_registry_write_begin_at(registry,contents,RC_REGISTRY_BUFFER,(RcRegistryPosition){14421,39},&older)==RC_REGISTRY_OK);
 ResourceCommandEffects *effects=command(output.scene,output.ordinal);
 assert(rc_registry_write_guard_begin_reason_at(registry,RC_REGISTRY_BUFFER,output,RC_REGISTRY_GUARD_GPA_PROJECTION,&gpa)==RC_REGISTRY_OK);
 RcInterval actual={.address=0x10000,.mapping=13,.range=unrelated};
 assert(resource_effects_add_intervals(effects,&actual,1,&actual,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
 EffectDeclaration declaration={.commands=effects,.position=output,.gpa_unknown=1};effect_current=&declaration;
 source_read_position=read;
 assert(!vk_exec_effects_read_allowed_range(contents,1)); // Output position is not sealed.
 seal(output.scene,output.ordinal);
 assert(vk_exec_effects_read_allowed_range(contents,1));
 assert(!vk_exec_effects_read_allowed_range(contents,0)); // Same-operation feedback or mutable source.
 assert(!vk_exec_effects_read_allowed_range(unrelated,1)); // Specific same-operation image overlap remains excluded.
 assert(rc_registry_read_range_at(registry,contents,read)==RC_REGISTRY_BLOCKED); // Still sees the earlier writer.
 handoff(read.scene,read.ordinal-1);
 assert(rc_registry_write_submit(registry,older,read.scene)==RC_REGISTRY_OK);
 assert(rc_registry_read_range_at(registry,contents,read)==RC_REGISTRY_BLOCKED);
 assert(rc_registry_write_complete(registry,older,1)==RC_REGISTRY_OK);
 assert(rc_registry_read_range_at(registry,contents,read)==RC_REGISTRY_OK);
 assert(rc_registry_stats(registry).writer_pending>=1); // The GPAs are still guarded for subsequent readers.
 source_read_position=(RcRegistryPosition){read.scene,38};
 assert(!vk_exec_effects_read_allowed_range(contents,1));
 source_read_position=(RcRegistryPosition){0};
 assert(!vk_exec_effects_read_allowed_range(contents,1));
 source_read_position=read;declaration.gpa_recorded=1;
 assert(!vk_exec_effects_read_allowed_range(contents,1));
 declaration.gpa_recorded=0;
 ResourceCommandEffects *prior=resource_effects_enter(effects);
 resource_effects_emitted();
 assert(!vk_exec_effects_read_allowed_range(contents,1));
 resource_effects_leave(prior);
 effect_current=NULL;
 resource_effects_release(effects);
 assert(nq==1);q[0].done(q[0].arg);nq=0;
 seal(later.scene,later.ordinal+1);
 assert(rc_registry_read_range_at(registry,contents,later)==RC_REGISTRY_BLOCKED);
 handoff(output.scene,output.ordinal);
 assert(rc_registry_write_submit(registry,gpa,read.scene)==RC_REGISTRY_OK);
 RcRegistryWriteCompletion completion={.ticket=gpa,.written=&contents,.uncertain=&contents,.written_count=1,.uncertain_count=1};
 assert(rc_registry_write_complete_extents(registry,&completion,1)==RC_REGISTRY_OK);
 assert(rc_registry_read_range_at(registry,contents,later)==RC_REGISTRY_BLOCKED);
 rc_registry_destroy(registry);
}
static void normalized(void){
 const RcInterval descriptors[]={{0x8000,71,{30,10,64}},{0x8040,72,{31,10,64}}};
 const RcInterval actuals[]={{0x8010,71,{30,26,12}},{0x8050,72,{31,26,12}}};
 registry=rc_registry_create((RcRegistryConfig){16,1,4});assert(registry);
 ResourceCommandEffects *e=command(1,3);
 assert(resource_effects_add_intervals(e,actuals,2,descriptors,2,RC_REGISTRY_BUFFER)==RC_REGISTRY_CAPACITY);
 assert(!rc_registry_stats(registry).writers && resource_effects_read_allowed_range(e,actuals[0].range));
 resource_effects_release(e);rc_registry_destroy(registry);

 registry=rc_registry_create((RcRegistryConfig){16,8,4});assert(registry);
 RcRegistryHandle first,second,unwritten;
 assert(rc_registry_source(registry,1,actuals[0].range,&first)==RC_REGISTRY_OK);
 assert(rc_registry_source(registry,2,actuals[1].range,&second)==RC_REGISTRY_OK);
 assert(rc_registry_source(registry,3,(RcSourceRange){31,40,8},&unwritten)==RC_REGISTRY_OK);
 RcRegistryObservation prior;
 assert(rc_registry_observe(registry,unwritten,1,&prior)==RC_REGISTRY_OK);
 e=command(1,3);
 assert(resource_effects_add_intervals(e,actuals,2,descriptors,2,RC_REGISTRY_BUFFER)==RC_REGISTRY_OK);seal(1,3);
 assert(!resource_effects_read_allowed_range(e,actuals[0].range));
 assert(resource_effects_read_allowed_range(e,(RcSourceRange){32,26,12}));
 assert(resource_effects_read_allowed(e,0x8010,12)); // Compatibility observers use their separate producer path.
 ResourceCommandEffects *old=resource_effects_enter(e);
 RcInterval replaced[]={actuals[0],actuals[1]};replaced[1].mapping++;
 assert(!resource_effects_buffer_prepare_intervals(replaced,2,1));
 resource_effects_buffers_commit();resource_effects_leave(old);
 resource_effects_release(e);
 assert(!nq && !rc_registry_stats(registry).writers && visible(first) && visible(second));

 e=command(2,3);
 assert(resource_effects_add_intervals(e,actuals,2,descriptors,2,RC_REGISTRY_BUFFER)==RC_REGISTRY_OK);seal(2,3);
 old=resource_effects_enter(e);
 assert(resource_effects_buffer_prepare_intervals(actuals,2,1));
 assert(resource_effects_buffer_prepare_intervals(actuals+1,1,0));
 resource_effects_buffers_commit();resource_effects_emitted();resource_effects_leave(old);
 resource_effects_release(e);handoff(2,3);
 assert(nq==1 && !visible(first) && !visible(second));q[0].done(q[0].arg);nq=0;
 assert(visible(first) && !visible(second) && rc_registry_content_current(registry,&prior));
 assert(rc_registry_reconcile_cpu(registry,actuals[1].range)==RC_REGISTRY_OK);

 e=command(3,3);
 assert(resource_effects_add_intervals(e,actuals,2,descriptors,2,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);seal(3,3);
 old=resource_effects_enter(e);
 assert(resource_effects_image_prepare_intervals(133,actuals,2));
 resource_effects_image_written_interval(133,actuals[0]);
 resource_effects_image_written_interval(133,actuals[1]);
 resource_effects_emitted();resource_effects_leave(old);
 resource_effects_release(e);handoff(3,3);
 assert(nq==1);q[0].cancel(q[0].arg,1);nq=0;
 assert(!visible(first) && !visible(second) && visible(unwritten));
 unsigned foreign=rc_registry_stats(registry).writer_foreign;
 e=command(4,3);assert(resource_effects_opaque_intervals(e,NULL)==RC_REGISTRY_OK);seal(4,3);
 assert(!resource_effects_read_allowed_range(e,actuals[0].range) && resource_effects_read_allowed(e,0x8010,12));
 resource_effects_release(e);assert(!nq && rc_registry_stats(registry).writer_foreign==foreign);
 e=command(5,3);assert(resource_effects_opaque_intervals(e,NULL)==RC_REGISTRY_OK);seal(5,3);
 old=resource_effects_enter(e);resource_effects_emitted();
 RcSourceRange exact={35,0,16};
 assert(resource_effects_resolve_opaque_intervals(&exact,1,&exact,1)==RC_REGISTRY_OK);
 resource_effects_leave(old);resource_effects_release(e);handoff(5,3);
 assert(nq==1);q[0].done(q[0].arg);nq=0;
 assert(rc_registry_stats(registry).writer_foreign==++foreign && !rc_registry_unresolved_guard(registry).id);
 for (uint64_t scene=6;scene<12;scene++) {
  e=command(scene,3);assert(resource_effects_opaque_intervals(e,NULL)==RC_REGISTRY_OK);seal(scene,3);
  resource_effects_recorded(e,1);resource_effects_release(e);handoff(scene,3);
  assert(nq==1);q[0].done(q[0].arg);nq=0;
  assert(rc_registry_stats(registry).writer_foreign==foreign+1 && rc_registry_unresolved_guard(registry).id);
 }
 rc_registry_destroy(registry);
}
static void typed_output_does_not_retain_dormant_command_guard(void) {
 const RcInterval output={0x9000,72,{31,20,16}};
 const RcSourceRange sample={32,22,8};
 registry=rc_registry_create((RcRegistryConfig){16,8,4});assert(registry);
 RcRegistryHandle output_source,sample_source;
 assert(rc_registry_source(registry,1,output.range,&output_source)==RC_REGISTRY_OK);
 assert(rc_registry_source(registry,2,sample,&sample_source)==RC_REGISTRY_OK);
 for(int emitted=0;emitted<=1;emitted++) {
  uint64_t scene=1+emitted;
  ResourceCommandEffects *e=command(scene,3);
  assert(resource_effects_opaque_intervals(e,NULL)==RC_REGISTRY_OK);
  assert(resource_effects_add_intervals(e,&output,1,&output,1,RC_REGISTRY_BUFFER)==RC_REGISTRY_OK);
  seal(scene,3);
  ResourceCommandEffects *old=resource_effects_enter(e);
  assert(resource_effects_buffer_prepare_intervals(&output,1,1));
  resource_effects_buffers_commit();
  if(emitted) resource_effects_emitted();
  resource_effects_leave(old);
  assert(!visible(sample_source) && !visible(output_source));
  resource_effects_release(e);
  seal(scene,4);
  assert(nq==1 && rc_registry_read_range_at(registry,output.range,(RcRegistryPosition){scene,4})==RC_REGISTRY_BLOCKED);
  if(emitted) assert(rc_registry_read_range_at(registry,sample,(RcRegistryPosition){scene,4})==RC_REGISTRY_BLOCKED);
  else assert(rc_registry_read_range_at(registry,sample,(RcRegistryPosition){scene,4})==RC_REGISTRY_OK);
  handoff(scene,3);
  q[0].done(q[0].arg);nq=0;
  assert(!!rc_registry_unresolved_guard(registry).id==emitted);
  assert(visible(output_source)==!emitted && visible(sample_source)==!emitted);
 }
 rc_registry_destroy(registry);
}
static void transferred_cpu_result(void){
 const RcInterval dst[]={{0x8ff8,71,{31,80,8}},{0x9000,72,{32,0,8}}};
 RcInterval remapped[]={dst[0],dst[1]};
 remapped[1].mapping++;
 RcSourceRange raw={1,0x8ff8,16};
 registry=rc_registry_create((RcRegistryConfig){16,3,4});assert(registry);
 RcRegistryHandle old_a,old_b,successor,neighbour;
 assert(rc_registry_source(registry,1,dst[0].range,&old_a)==RC_REGISTRY_OK);
 assert(rc_registry_source(registry,2,dst[1].range,&old_b)==RC_REGISTRY_OK);
 assert(rc_registry_source(registry,3,(RcSourceRange){33,0,8},&successor)==RC_REGISTRY_OK);
 assert(rc_registry_source(registry,4,(RcSourceRange){32,8,8},&neighbour)==RC_REGISTRY_OK);
 RcRegistryObservation successor_before,neighbour_before;
 uint64_t prior_transfers=__atomic_load_n(&cpu_results_transferred,__ATOMIC_RELAXED);
 assert(rc_registry_observe(registry,successor,1,&successor_before)==RC_REGISTRY_OK);
 assert(rc_registry_observe(registry,neighbour,1,&neighbour_before)==RC_REGISTRY_OK);
 for(uint64_t scene=1;scene<=2;scene++){
  ResourceCommandEffects *e=command(scene,3);
  resource_effects_add(e,raw,raw,RC_REGISTRY_BUFFER);
  assert(resource_effects_add_intervals(e,dst,2,dst,2,RC_REGISTRY_BUFFER)==RC_REGISTRY_OK);seal(scene,3);
  ResourceCommandEffects *old=resource_effects_enter(e);
  RcRegistryWrite tickets[3]={0};
  assert(!resource_effects_buffer_result_take(raw.offset,raw.size,remapped,2,&tickets[0],tickets+1));
  assert(!tickets[0].id && !tickets[1].id && !tickets[2].id);
  assert(resource_effects_buffer_result_take(raw.offset,raw.size,dst,2,&tickets[0],tickets+1));
  assert(!resource_effects_buffer_result_take(raw.offset,raw.size,dst,2,&tickets[0],tickets+1));
  assert(tickets[0].id && tickets[1].id && tickets[2].id);
  assert(!resource_effects_read_allowed(e,raw.offset,raw.size) && !resource_effects_read_allowed_range(e,dst[1].range));
  resource_effects_buffers_commit();resource_effects_leave(old);
  resource_effects_recorded(e,1);resource_effects_release(e);
  assert(!nq && rc_registry_stats(registry).writers==3 && !visible(old_a) && !visible(old_b));
  if(scene==1) assert(rc_registry_write_cancel_extents(registry,tickets,3)==RC_REGISTRY_OK);
  else {
   handoff(scene,3);
   assert(rc_registry_write_submit_extents(registry,tickets,3,scene)==RC_REGISTRY_OK);
   const RcRegistryWriteCompletion actuals[]={
    {tickets[0],&raw,NULL,1,0},{tickets[1],&dst[0].range,NULL,1,0},{tickets[2],&dst[1].range,NULL,1,0}};
   assert(rc_registry_write_complete_extents(registry,actuals,3)==RC_REGISTRY_OK);
  }
  assert(!rc_registry_stats(registry).writers && visible(old_a) && visible(old_b));
  assert(rc_registry_content_current(registry,&successor_before) && rc_registry_content_current(registry,&neighbour_before));
 }
 assert(!rc_registry_stats(registry).fail_closed);
 assert(__atomic_load_n(&cpu_results_transferred,__ATOMIC_RELAXED)==prior_transfers+2);
 rc_registry_destroy(registry);
}
static void independent_upload(void) {
 const RcInterval span={0x8100,44,{19,0x100,16}};
 registry=rc_registry_create((RcRegistryConfig){16,8,4});
 assert(registry);
 RcRegistryHandle mapped,raw,next;
 assert(rc_registry_source(registry,1,span.range,&mapped)==RC_REGISTRY_OK);
 assert(rc_registry_source(registry,2,(RcSourceRange){1,span.address,span.range.size},&raw)==RC_REGISTRY_OK);
 assert(rc_registry_source(registry,3,(RcSourceRange){20,span.range.offset,span.range.size},&next)==RC_REGISTRY_OK);
 actual_host=1;
 ResourceCommandEffects *command_scope=resource_effects_create((RcRegistryPosition){1,3});
 ResourceCommandEffects *old=resource_effects_enter(command_scope);
 ResourceImageScope upload={0};
 resource_image_scope_begin(&upload,0);
 resource_image_prepare_mapped(71,span.address,span.range.size,&span,1);
 assert(rc_registry_stats(registry).writer_pending==2 && !visible(raw) && !visible(mapped) && visible(next));
 resource_image_scope_end_upload(&upload);
 assert(!nq && !rc_registry_stats(registry).writers && visible(raw) && visible(mapped));
 resource_image_scope_begin(&upload,0);
 resource_image_prepare_mapped(71,span.address,span.range.size,&span,1);
 resource_image_written(71,span.address);
 RcRegistryWrite raw_hold,mapped_hold;
 assert(rc_registry_hold_begin(registry,(RcSourceRange){1,span.address,span.range.size},RC_REGISTRY_IMAGE,&raw_hold)==RC_REGISTRY_OK);
 assert(rc_registry_image_hold_bind(registry,raw_hold,71)==RC_REGISTRY_OK);
 assert(rc_registry_image_hold_begin_extents(registry,71,&span,1,&mapped_hold)==RC_REGISTRY_OK);
 resource_image_owned(71,span.address,raw_hold);
 resource_image_owned_mapped(71,span.address,&span,&mapped_hold,1);
 resource_image_scope_end_upload(&upload);
 assert(!nq && !rc_registry_stats(registry).writer_pending && rc_registry_stats(registry).writer_holds==2);
 assert(!visible(raw) && !visible(mapped) && visible(next));
 assert(rc_registry_hold_release(registry,raw_hold,1)==RC_REGISTRY_OK);
 assert(rc_registry_image_hold_release_extents(registry,&mapped_hold,1,1)==RC_REGISTRY_OK);
 resource_image_scope_begin(&upload,0);
 resource_image_prepare_mapped(71,span.address,span.range.size,&span,1);
 resource_image_written(71,span.address);
 resource_image_scope_end_upload(&upload); // A recorded upload without a durable holder remains unknown.
 assert(!nq && rc_registry_stats(registry).writer_foreign==2 && !visible(raw) && !visible(mapped) && visible(next));
 resource_effects_leave(old);
 resource_effects_release(command_scope);
 actual_host=0;
 assert(!rc_registry_stats(registry).fail_closed);
 rc_registry_destroy(registry);
}
static void repeated_mapped_images(void){
 enum {repeats=8193};
 const RcInterval target={0x9000,7,{41,0,32}};
 const RcInterval additional[]={{0x9010,7,{41,16,32}},{0x9000,8,{42,0,32}}};
 registry=rc_registry_create((RcRegistryConfig){16000,2,4});assert(registry);
 RcRegistryHandle first,successor,gap;
 assert(rc_registry_source(registry,1,target.range,&first)==RC_REGISTRY_OK);
 assert(rc_registry_source(registry,2,additional[1].range,&successor)==RC_REGISTRY_OK);
 assert(rc_registry_source(registry,3,(RcSourceRange){41,32,16},&gap)==RC_REGISTRY_OK);
 unsigned positions=0;
 for(unsigned i=0;i<repeats;i++){
  ResourceCommandEffects *e=command(1,i+3);
  assert(resource_effects_add_intervals(e,&target,1,&target,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
  if(i==0){assert(e->claims->shared->refs==1);positions=rc_registry_stats(registry).image_positions;}
  seal(1,i+3);
  resource_effects_recorded(e,1);resource_effects_release(e);
 }
 RcRegistryStats peak=rc_registry_stats(registry);
 assert(peak.writers==1 && peak.writer_pending==1 && !peak.fail_closed &&
        peak.image_positions==positions+repeats-1 && !peak.image_position_refusals);
 assert(nq==repeats && !visible(first) && visible(successor) && visible(gap));

 ResourceCommandEffects *e=command(1,repeats+3);
 assert(resource_effects_add_intervals(e,additional,2,additional,2,RC_REGISTRY_IMAGE)==RC_REGISTRY_CAPACITY);
 assert(!e->claims && rc_registry_stats(registry).writers==1 && visible(gap) && visible(successor));
 assert(resource_effects_add_intervals(e,additional,1,additional,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
 assert(rc_registry_stats(registry).writers==2 && !visible(gap) && visible(successor));
 seal(1,repeats+3);
 resource_effects_release(e); // A skipped overlapping output cannot cancel the earlier exact group.
 assert(rc_registry_stats(registry).writers==1 && !visible(first) && visible(gap));
 handoff(1,repeats+3);
 for(unsigned i=0;i<nq;i++) q[i].done(q[i].arg);
 nq=0;
 RcRegistryStats settled=rc_registry_stats(registry);
 assert(!shared_images && settled.writers==1 && settled.writer_foreign==1 && !settled.image_positions &&
        !settled.writer_pending && !settled.fail_closed);
 assert(!visible(first) && visible(gap) && visible(successor));
 assert(rc_registry_reconcile_cpu(registry,target.range)==RC_REGISTRY_OK && visible(first));
 rc_registry_destroy(registry);
}
static void pinned_image_projection(void) {
 const RcInterval target={0x9600,8,{41,0,32}};
 registry=rc_registry_create((RcRegistryConfig){32,8,4});assert(registry);
 RcRegistryHandle sampled;
 assert(rc_registry_source(registry,1,target.range,&sampled)==RC_REGISTRY_OK);
 RcRegistryObservation initial;
 assert(rc_registry_observe(registry,sampled,1,&initial)==RC_REGISTRY_OK);

 ResourceCommandEffects *e=command(1,3);
 seal(1,3);
 ResourceCommandEffects *previous=resource_effects_enter(e);
 {RESOURCE_IMAGE_SCOPE(scope,1);
  resource_image_prepare_pinned(49,target.address,target.range.size,&target,1,1);
  assert(!scope.writes && !resource_image_pinned_projected(49,target.address));
  resource_effects_emitted();}
 resource_effects_leave(previous);
 resource_effects_release(e);
 handoff(1,3);
 assert(!nq && rc_registry_observation_current(registry,&initial));

 // A real ordinary alias still declares and records its mapped output. Private image 49 must not adopt it.
 e=command(2,3);
 RcSourceRange raw={1,target.address,target.range.size};
 resource_effects_add(e,raw,raw,RC_REGISTRY_IMAGE);
 assert(resource_effects_add_intervals(e,&target,1,&target,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
 seal(2,3);
 previous=resource_effects_enter(e);
 RcRegistryWrite hold,raw_hold;
 {RESOURCE_IMAGE_SCOPE(scope,2);
  resource_image_prepare_pinned(49,target.address,target.range.size,&target,1,1);
  assert(!scope.writes);
  resource_image_prepare_mapped(52,target.address,target.range.size,&target,1);
  resource_image_written(52,target.address);
  resource_effects_emitted();
  assert(rc_registry_hold_begin(registry,raw,RC_REGISTRY_IMAGE,&raw_hold)==RC_REGISTRY_OK);
  assert(rc_registry_image_hold_bind(registry,raw_hold,52)==RC_REGISTRY_OK);
  resource_image_owned(52,target.address,raw_hold);
  assert(rc_registry_image_hold_begin_extents(registry,52,&target,1,&hold)==RC_REGISTRY_OK);
  resource_image_owned_mapped(52,target.address,&target,&hold,1);}
 resource_effects_leave(previous);resource_effects_release(e);handoff(2,3);
 assert(!nq && !visible(sampled));
 assert(rc_registry_hold_release(registry,raw_hold,1)==RC_REGISTRY_OK);
 assert(rc_registry_image_hold_release_extents(registry,&hold,1,1)==RC_REGISTRY_OK && visible(sampled));

 // An earlier live declaration remains the exact projection if the selected image detaches before recording.
 for(unsigned emitted=0;emitted<2;emitted++) {
  uint64_t scene=3+emitted;
  e=command(scene,3);
  assert(resource_effects_add_pinned_image(e,49,&target,1)==RC_REGISTRY_OK);
  seal(scene,3);previous=resource_effects_enter(e);
  {RESOURCE_IMAGE_SCOPE(scope,scene);
   RcInterval wrong=target;
   wrong.mapping++;
   assert(resource_effects_image_prepare_pinned(49,&wrong,1)==-1);
   assert(!resource_effects_image_prepare_intervals(52,&target,1));
   resource_image_prepare_pinned(49,target.address,target.range.size,&target,1,1);
   assert(scope.writes && resource_image_pinned_projected(49,target.address));
   if(emitted) {
    resource_image_written(49,target.address);resource_effects_emitted();
    assert(rc_registry_image_hold_begin_extents(registry,49,&target,1,&hold)==RC_REGISTRY_OK);
    resource_image_owned_mapped(49,target.address,&target,&hold,1);
   }}
  resource_effects_leave(previous);resource_effects_release(e);handoff(scene,3);
  assert(!nq && visible(sampled)==!emitted);
 }
 assert(rc_registry_image_hold_release_extents(registry,&hold,1,1)==RC_REGISTRY_OK && visible(sampled));
 assert(!rc_registry_stats(registry).writers && !rc_registry_stats(registry).fail_closed);
 rc_registry_destroy(registry);
}
static void canceled_exact_phase(void) {
 registry=rc_registry_create((RcRegistryConfig){64,16,4});assert(registry);
 const RcInterval task={0xe100,31,{61,0,64}}, final={0xe200,31,{61,128,64}}, uncertain={0xe300,31,{61,256,64}};
 RcRegistryHandle task_source,final_source,unknown_source;
 assert(rc_registry_source(registry,1,task.range,&task_source)==RC_REGISTRY_OK);
 assert(rc_registry_source(registry,2,final.range,&final_source)==RC_REGISTRY_OK);
 assert(rc_registry_source(registry,3,uncertain.range,&unknown_source)==RC_REGISTRY_OK);
 RcRegistryObservation prior;
 assert(rc_registry_observe(registry,final_source,1,&prior)==RC_REGISTRY_OK);
 ResourceCommandEffects *effects=command(1,3);
 assert(resource_effects_add_pinned_image(effects,55,&task,1)==RC_REGISTRY_OK);
 assert(resource_effects_add_pinned_image(effects,56,&final,1)==RC_REGISTRY_OK);
 assert(resource_effects_add_intervals(effects,&uncertain,1,&uncertain,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
 seal(1,3);
 ResourceCommandEffects *previous=resource_effects_enter(effects);
 RcRegistryWrite issue;
 assert(rc_registry_image_hold_begin_extents(registry,55,&task,1,&issue)==RC_REGISTRY_OK);
 RcRegistryImageWriteExtent actual={task,issue};
 resource_effects_image_record_extents(55,&actual,1);
 resource_effects_cancel_unissued_image(effects,55); // Same image was already issued by an earlier phase.
 resource_effects_cancel_unissued_image(effects,56);
 resource_effects_cancel_unissued_image(effects,57); // Cannot cancel an unqualified output by guessing its image.
 resource_effects_recorded(effects,1);
 resource_effects_leave(previous);
 resource_effects_release(effects);
 handoff(1,3);
 for(unsigned i=0;i<nq;i++) q[i].done(q[i].arg);
 nq=0;
 assert(visible(final_source) && !visible(task_source) && !visible(unknown_source));
 assert(rc_registry_content_current(registry,&prior));
 assert(rc_registry_image_binding_current(registry,(RcRegistryImageBinding){55,issue}));
 assert(rc_registry_image_hold_release_extents(registry,&issue,1,1)==RC_REGISTRY_OK && visible(task_source));
 assert(!rc_registry_stats(registry).fail_closed);
 rc_registry_destroy(registry);
}
static void captured_image_extent(void){
 registry=rc_registry_create((RcRegistryConfig){64,8,4});assert(registry);
 RcInterval whole={0xa000,21,{51,32,64}}, written={0xa010,21,{51,48,16}};
 RcRegistryHandle reader;
 assert(rc_registry_source(registry,1,written.range,&reader)==RC_REGISTRY_OK);
 RcRegistryImageWriteExtent capture={.interval=whole};
 assert(rc_registry_image_hold_begin_extents(registry,9,&whole,1,&capture.hold)==RC_REGISTRY_OK);
 RcRegistryImageBinding owner={9,capture.hold};
 RcRegistryBindingObservation observed;

 ResourceCommandEffects *e=command(1,3);
 assert(resource_effects_add_intervals(e,&written,1,&written,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
 seal(1,4);
 ResourceCommandEffects *previous=resource_effects_enter(e);
 resource_effects_image_record_extents(9,&capture,1);
 resource_effects_leave(previous);resource_effects_recorded(e,1);
 assert(rc_registry_binding_begin_image(registry,reader,1,(RcRegistryPosition){1,4},owner,&observed)==RC_REGISTRY_OK);
 resource_effects_release(e);handoff(1,4);
 assert(!nq && rc_registry_stats(registry).writer_holds==1 && !rc_registry_stats(registry).writer_pending);

 e=command(2,3);
 assert(resource_effects_add_intervals(e,&written,1,&written,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
 seal(2,4);
 capture.interval.mapping++;
 previous=resource_effects_enter(e);
 resource_effects_image_record_extents(9,&capture,1);
 resource_effects_leave(previous);resource_effects_recorded(e,1);
 assert(rc_registry_binding_begin_image(registry,reader,1,(RcRegistryPosition){2,4},owner,&observed)!=RC_REGISTRY_OK);
 resource_effects_release(e);handoff(2,4);
 assert(nq==1);q[0].cancel(q[0].arg,1);nq=0;
 assert(!shared_images && rc_registry_stats(registry).writer_foreign==1);
 assert(rc_registry_image_hold_release_extents(registry,&capture.hold,1,1)==RC_REGISTRY_OK);
 rc_registry_destroy(registry);
}
static RcRegistryWrite shared_positioned_owned(ResourceCommandEffects *effects, RcInterval whole,
    RcSourceRange exact, uint64_t image, RcRegistryPosition output) {
 ResourceCommandEffects *prior=resource_effects_enter(effects);
 RcRegistryImageWriteExtent owner={.interval=whole};
 assert(rc_registry_image_hold_begin_extents(registry,image,&whole,1,&owner.hold)==RC_REGISTRY_OK);
 {RESOURCE_IMAGE_SCOPE(scope,output.scene);
  resource_image_prepare_mapped(image,whole.address,whole.range.size,&whole,1);
  resource_image_written(image,whole.address);
  resource_image_owned_mapped(image,whole.address,&whole,&owner.hold,1);
  RcRegistryNativeImageWrite projection={owner,&exact,1};
  assert(rc_registry_native_image_write_fragments(registry,image,&projection,1,output)==RC_REGISTRY_OK);
  resource_effects_recorded(effects,1);
 }
 resource_effects_leave(prior);
 return owner.hold;
}
static RcInterval shared_positioned_span(RcInterval whole, uint64_t offset, uint64_t size) {
 whole.address+=offset;
 whole.range.offset+=offset;
 whole.range.size=size;
 return whole;
}
static ResourceCommandEffects *shared_positioned_unowned(RcRegistryPosition output, RcInterval whole, RcInterval actual) {
 ResourceCommandEffects *effects=command(output.scene,output.ordinal);
 assert(resource_effects_add_intervals(effects,&whole,1,&whole,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
 seal(output.scene,output.ordinal);
 ResourceCommandEffects *prior=resource_effects_enter(effects);
 assert(resource_effects_image_prepare_intervals(1902,&actual,1));
 resource_effects_image_written_interval(1902,actual);
 resource_effects_recorded(effects,1);
 resource_effects_leave(prior);
 return effects;
}
static void shared_positioned_completion(void) {
 const RcInterval whole={UINT64_C(0xf000000000001000),7,{6,0,0x4000}};
 const RcInterval left=shared_positioned_span(whole,0,0x1000), middle=shared_positioned_span(whole,0x1000,0x1000),
                  right=shared_positioned_span(whole,0x3000,0x1000);
 RcRegistryPosition old={1,3}, unknown={1,5}, latest={1,7}, read={2,2};
 enum { PARTIAL, FULL, KNOWN_CANCELED, DRAINED, TAIL_CANCELED, SYNTHETIC };
 for (int mode=PARTIAL;mode<=SYNTHETIC;mode++) {
  registry=rc_registry_create((RcRegistryConfig){16,16,4});assert(registry);
  ResourceCommandEffects *a=command(old.scene,old.ordinal);
  assert(resource_effects_add_intervals(a,&whole,1,&whole,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
  seal(old.scene,old.ordinal);
  RcRegistryWrite previous=shared_positioned_owned(a,whole,whole.range,1901,old);
  resource_effects_release(a);
  ResourceCommandEffects *b;
  if (mode==SYNTHETIC) {
   b=command(unknown.scene,unknown.ordinal);
   assert(resource_effects_add_intervals(b,&whole,1,&whole,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
   seal(unknown.scene,unknown.ordinal);
   resource_effects_recorded(b,1);
  } else b=shared_positioned_unowned(unknown,whole,shared_positioned_span(whole,0,0x2000));
  ResourceCommandEffects *c=command(latest.scene,latest.ordinal);
  assert(resource_effects_add_intervals(c,&whole,1,&whole,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
  assert(b->claims->shared==c->claims->shared && b->claims->shared->refs==3);
  seal(latest.scene,latest.ordinal);
  const RcSourceRange exact=mode==FULL?whole.range:left.range;
  RcRegistryWrite newer=shared_positioned_owned(c,whole,exact,1903,latest);
  resource_effects_release(b);
  resource_effects_release(c);
  assert(nq==3);
  handoff(latest.scene,mode==TAIL_CANCELED?unknown.ordinal:latest.ordinal);
  seal(read.scene,read.ordinal);
  q[0].done(q[0].arg);
  if (mode==KNOWN_CANCELED) q[1].cancel(q[1].arg,0);
  else if (mode==DRAINED) q[1].cancel(q[1].arg,1);
  else q[1].done(q[1].arg);
  if (mode==TAIL_CANCELED) q[2].cancel(q[2].arg,0);
  else q[2].done(q[2].arg);
  nq=0;
  RcRegistryNativeImageFragment got;
  uint32_t count=0;
  RcRegistryNativeImageFailure failure;
  RcRegistryResult result=rc_registry_native_image_fragments_checked(registry,&right,1,read,&got,1,&count,&failure);
  if (mode==SYNTHETIC) {
   assert(result==RC_REGISTRY_BLOCKED&&!count&&failure.cause==RC_NATIVE_IMAGE_BLOCK_FOREIGN);
   assert(!rc_registry_position_compare(failure.blocking_output,unknown));
  } else assert(result==RC_REGISTRY_OK&&count==1&&got.binding.hold.id==(mode==FULL?newer.id:previous.id));
  result=rc_registry_native_image_fragments_checked(registry,&left,1,read,&got,1,&count,&failure);
  if (mode==TAIL_CANCELED) {
   assert(result==RC_REGISTRY_BLOCKED&&!count&&failure.cause==RC_NATIVE_IMAGE_BLOCK_FOREIGN);
   assert(!rc_registry_position_compare(failure.blocking_output,unknown));
  } else assert(result==RC_REGISTRY_OK&&count==1&&got.binding.hold.id==newer.id);
  result=rc_registry_native_image_fragments_checked(registry,&middle,1,read,&got,1,&count,&failure);
  if (mode==FULL||mode==KNOWN_CANCELED) assert(result==RC_REGISTRY_OK&&count==1&&got.binding.hold.id==(mode==FULL?newer.id:previous.id));
  else {
   assert(result==RC_REGISTRY_BLOCKED&&!count&&failure.cause==RC_NATIVE_IMAGE_BLOCK_FOREIGN);
   assert(!rc_registry_position_compare(failure.blocking_output,unknown));
  }
  rc_registry_destroy(registry);
 }
}
static void shared_positioned_completion_reverse(void) {
 const RcInterval whole={UINT64_C(0xf000000000001000),7,{6,0,0x4000}};
 const RcInterval early=shared_positioned_span(whole,0,0x1000), late=shared_positioned_span(whole,0x1000,0x1000),
                  unwritten=shared_positioned_span(whole,0x3000,0x1000);
 const RcRegistryPosition first={1,3}, owned={1,5}, last={1,7}, read={2,2};
 for (int capacity=0;capacity<2;capacity++) {
  registry=rc_registry_create((RcRegistryConfig){16,capacity?2:16,4});assert(registry);
  ResourceCommandEffects *a=shared_positioned_unowned(first,whole,early);
  ResourceCommandEffects *b=command(owned.scene,owned.ordinal);
  assert(resource_effects_add_intervals(b,&whole,1,&whole,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
  seal(owned.scene,owned.ordinal);
  const RcRegistryWrite owned_hold=shared_positioned_owned(b,whole,whole.range,1911,owned);
  ResourceCommandEffects *c=shared_positioned_unowned(last,whole,late);
  assert(a->claims->shared==c->claims->shared && b->claims->shared==c->claims->shared);
  resource_effects_release(a);
  resource_effects_release(b);
  resource_effects_release(c);
  assert(nq==3);
  handoff(last.scene,last.ordinal);
  seal(read.scene,read.ordinal);
  // Completion callback order must not determine writer order.
  q[2].done(q[2].arg);q[1].done(q[1].arg);q[0].done(q[0].arg);nq=0;
  const RcRegistryStats stats=rc_registry_stats(registry);
  assert(!stats.fail_closed&&!stats.writer_submitted&&!stats.writer_pending&&stats.writer_holds==1);
  assert(stats.completion_range_refusals==(unsigned)capacity&&stats.writer_foreign==(capacity?1u:2u));
  RcRegistryNativeImageFragment got;
  RcRegistryNativeImageFailure failure;
  uint32_t count=0;
  RcRegistryResult result=rc_registry_native_image_fragments(registry,&early,1,read,&got,1,&count);
  if (capacity) assert(result==RC_REGISTRY_BLOCKED&&!count);
  else assert(result==RC_REGISTRY_OK&&count==1&&got.binding.hold.id==owned_hold.id);
  assert(rc_registry_native_image_fragments_checked(registry,&late,1,read,&got,1,&count,&failure)==RC_REGISTRY_BLOCKED);
  assert(!count&&failure.cause==RC_NATIVE_IMAGE_BLOCK_FOREIGN);
  assert(capacity?!rc_registry_position_valid(failure.blocking_output):
                  !rc_registry_position_compare(failure.blocking_output,last));
  result=rc_registry_native_image_fragments(registry,&unwritten,1,read,&got,1,&count);
  if (capacity) assert(result==RC_REGISTRY_BLOCKED&&!count);
  else assert(result==RC_REGISTRY_OK&&count==1&&got.binding.hold.id==owned_hold.id);
  rc_registry_destroy(registry);
 }
}

static void shared_positioned_completion_repeated(void) {
 const RcInterval whole={UINT64_C(0xf000000000001000),7,{6,0,0x4000}};
 const RcInterval dirty=shared_positioned_span(whole,0,0x2000);
 registry=rc_registry_create((RcRegistryConfig){16,2,4});assert(registry);
 for (uint64_t scene=1;scene<=96;scene++) {
  ResourceCommandEffects *effects=shared_positioned_unowned((RcRegistryPosition){scene,3},whole,dirty);
  resource_effects_release(effects);
  assert(nq==1);
  handoff(scene,3);
  q[0].done(q[0].arg);nq=0;
  const RcRegistryStats stats=rc_registry_stats(registry);
  assert(!stats.fail_closed&&!stats.writer_pending&&stats.writer_foreign==1);
 }
 const RcRegistryPosition last={96,3},read={97,2};
 seal(read.scene,read.ordinal);
 RcRegistryNativeImageFragment got;
 RcRegistryNativeImageFailure failure;
 uint32_t count=0;
 assert(rc_registry_native_image_fragments_checked(registry,&dirty,1,read,&got,1,&count,&failure)==RC_REGISTRY_BLOCKED);
 assert(!count&&failure.cause==RC_NATIVE_IMAGE_BLOCK_FOREIGN);
 assert(!rc_registry_position_compare(failure.blocking_output,last));
 rc_registry_destroy(registry);
}

static void tensor_optional_role_fate(void) {
 const RcInterval whole={UINT64_C(0xf000000000001000),7,{6,0,0x4000}};
 const RcInterval other={UINT64_C(0xf000000000008000),8,{8,0,0x4000}};
 const RcRegistryPosition old={1,3}, actual={1,5}, read={2,2};
 enum { SKIPPED, SAME_MAPPING, SAME_IMAGE, ISSUED_EDGE, CANCEL_TAIL, LEGACY_SKIP };
 for (int mode=SKIPPED;mode<=LEGACY_SKIP;mode++) {
  const int alias=mode==SAME_MAPPING||mode==SAME_IMAGE;
  const RcInterval optional=mode==LEGACY_SKIP?(RcInterval){0x1000,0,{1,0x1000,64}}:whole;
  const RcInterval primary=alias?optional:other;
  registry=rc_registry_create((RcRegistryConfig){16,16,4});assert(registry);
  ResourceCommandEffects *a=NULL;
  RcRegistryWrite prior_hold={0};
  if (mode!=LEGACY_SKIP) {
   a=command(old.scene,old.ordinal);
   assert(resource_effects_add_intervals(a,&optional,1,&optional,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
   seal(old.scene,old.ordinal);
   prior_hold=shared_positioned_owned(a,optional,optional.range,1921,old);
   resource_effects_release(a);
  }
  ResourceCommandEffects *b=command(actual.scene,actual.ordinal);
  assert(resource_effects_add_intervals(b,&primary,1,&primary,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
  ResourceImageEffectRole border={0};
  resource_effects_image_role_begin(b,&border);
  if (mode==LEGACY_SKIP) {
   resource_effects_add(b,optional.range,optional.range,RC_REGISTRY_IMAGE);
  } else if (mode==SAME_IMAGE) {
   const RcInterval descriptor=optional;
   assert(resource_effects_add_pinned_image(b,1922,&descriptor,1)==RC_REGISTRY_OK);
  } else {
   assert(resource_effects_add_intervals(b,&optional,1,&optional,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
  }
  resource_effects_image_role_end(b,&border);
  seal(actual.scene,actual.ordinal);
  if (mode==ISSUED_EDGE) {
   ResourceCommandEffects *prior=resource_effects_enter(b);
   const RcInterval edge=shared_positioned_span(optional,0x1000,0x1000);
   assert(resource_effects_image_prepare_intervals(1923,&edge,1));
   resource_effects_image_role_finish(&border,1);
   resource_effects_image_written_interval(1923,edge);
   resource_effects_leave(prior);
  }
  resource_effects_image_role_finish(&border,0);
  const RcRegistryWrite primary_hold=shared_positioned_owned(b,primary,primary.range,1922,actual);
  resource_effects_release(b);
  assert(nq==(mode==LEGACY_SKIP?1u:2u));
  handoff(actual.scene,mode==CANCEL_TAIL?old.ordinal:actual.ordinal);
  seal(read.scene,read.ordinal);
  if (mode!=LEGACY_SKIP) q[0].done(q[0].arg);
  if (mode==CANCEL_TAIL) q[1].cancel(q[1].arg,0);
  else q[nq-1].done(q[nq-1].arg);
  nq=0;
  RcRegistryNativeImageFragment got;
  RcRegistryNativeImageFailure failure;
  uint32_t count=0;
  if (mode!=LEGACY_SKIP) {
   const RcInterval first=shared_positioned_span(optional,0,0x1000);
   assert(rc_registry_native_image_fragments_checked(registry,&first,1,read,&got,1,&count,&failure)==RC_REGISTRY_OK);
   assert(count==1&&got.binding.hold.id==(alias?primary_hold.id:prior_hold.id));
  } else {
   assert(!rc_registry_stats(registry).writer_foreign);
  }
  if (mode!=CANCEL_TAIL) {
   assert(rc_registry_native_image_fragments_checked(registry,&primary,1,read,&got,1,&count,&failure)==RC_REGISTRY_OK);
   assert(count==1&&got.binding.hold.id==primary_hold.id);
  }
  if (mode==ISSUED_EDGE) {
   const RcInterval edge=shared_positioned_span(optional,0x1000,0x1000);
   assert(rc_registry_native_image_fragments_checked(registry,&edge,1,read,&got,1,&count,&failure)==RC_REGISTRY_BLOCKED);
   assert(!count&&failure.cause==RC_NATIVE_IMAGE_BLOCK_FOREIGN);
   assert(!rc_registry_position_compare(failure.blocking_output,actual));
  }
  rc_registry_destroy(registry);
 }
}
static void image_producer_trace(void) {
 const RcInterval interval={UINT64_C(0xf000000000001000),7,{6,0,0x2000}};
 const RcRegistryPosition output={1,5}, read={1,6};
 assert(!resource_effects_image_trace_enabled()&&!image_trace);
 trace_requested=1;
 image_trace_mode=0;  // The production switch is fixed for the life of one process.
 assert(resource_effects_image_trace_enabled()&&image_trace);
 for (int disposition=0;disposition<3;disposition++) {
  registry=rc_registry_create((RcRegistryConfig){8,8,4});assert(registry);
  ResourceCommandEffects *e=command(output.scene,output.ordinal);
  assert(resource_effects_add_intervals(e,&interval,1,&interval,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
  seal(output.scene,output.ordinal);
  const RcRegistryWrite writer=image_ticket(e->claims);
  RcRegistryImageProducerState state;
  assert(rc_registry_image_producer_state(registry,writer,output,&state)&&state.unresolved==1);
  assert(!rc_registry_image_producer_state(registry,writer,read,&state));
  RcRegistryImageWriteExtent owner={.interval=interval};
  ResourceCommandEffects *prior=resource_effects_enter(e);
  ResourceImageScope scope={0};
  resource_image_scope_begin(&scope,output.scene);
  resource_image_prepare_mapped(777,interval.address,interval.range.size,&interval,1);
  if (disposition!=2) {
   assert(rc_registry_image_hold_begin_extents(registry,777,&interval,1,&owner.hold)==RC_REGISTRY_OK);
   resource_image_written(777,interval.address);
   resource_image_owned_mapped(777,interval.address,&interval,&owner.hold,1);
   const ResourceImageTraceDetails view={.format=9,.aspect=1,.mip=2,.mips=1,.layer=1,.layers=1,.selected_view=1,
     .baseline=RESOURCE_IMAGE_TRACE_CURRENT_PLANE,.placement=RESOURCE_IMAGE_TRACE_PRIVATE};
   resource_effects_image_trace_details(NULL,777,view);
   const ResourceImageTraceProjection publication=disposition==0 ? RESOURCE_IMAGE_TRACE_PLANE_PUBLISHED :
       RESOURCE_IMAGE_TRACE_LEGACY_WRITE;
   resource_effects_image_trace_details(NULL,777,(ResourceImageTraceDetails){.projection=publication});
   if (disposition==0) {
    const RcRegistryNativeImageWrite projection={owner,&interval.range,1};
    assert(rc_registry_native_image_write_fragments(registry,777,&projection,1,output)==RC_REGISTRY_OK);
   }
  }
  resource_effects_recorded_image_trace(e,1,27,49,RESOURCE_IMAGE_TRACE_DISPATCH_COMPAT);
  assert(rc_registry_image_producer_state(registry,writer,output,&state));
  assert(state.active==1&&!state.unresolved&&state.unknown==(disposition==2)
     &&state.projected==(disposition==0));
  assert(state.image.image==(disposition==2?0:777)&&rc_range_equal(state.interval.range,interval.range));
  const RcRegistryNativeImageFailure failure={.blocker=writer,.blocking_output=output,.blocking_interval=interval};
  const unsigned before=trace_log_count;
  resource_effects_image_trace_blocker(read,&failure);
  assert(trace_log_count==before+1&&strstr(trace_log,"output=1/5")&&strstr(trace_log,"observer=retained")
      &&strstr(trace_log,"route=dispatch-compat-selected program=49")&&strstr(trace_log,"mapping=7 backing=6"));
  if (disposition==0) assert(strstr(trace_log,"projected=1")&&strstr(trace_log,"plane=published"));
  if (disposition==1) assert(strstr(trace_log,"missing-actual=0 actual-known=1")&&strstr(trace_log,"projected=0")
      &&strstr(trace_log,"selected-view=1 format=9 aspect=1 mip=2+1 layer=1+1")
      &&strstr(trace_log,"baseline=current-plane placement=private plane=legacy-write"));
  if (disposition==2) {
   assert(strstr(trace_log,"missing-actual=1 actual-known=0")&&strstr(trace_log,"unknown=1 projected=0"));
   RcRegistryNativeImageFailure wrong=failure;
   wrong.blocker.id++;
   resource_effects_image_trace_blocker(read,&wrong);
   assert(strstr(trace_log,"observer=unavailable registry=unavailable"));
   wrong=failure;
   wrong.blocking_output.ordinal+=2;
   resource_effects_image_trace_blocker(read,&wrong);
   assert(strstr(trace_log,"observer=unavailable registry=unavailable"));
   for (unsigned i=0;i<IMAGE_TRACE_REPORTS;i++) resource_effects_image_trace_blocker(read,&failure);
   assert(trace_log_count==IMAGE_TRACE_REPORTS);
  }
  resource_image_scope_end(&scope);
  resource_effects_leave(prior);
  resource_effects_release(e);
  while(nq) {nq--;q[nq].cancel(q[nq].arg,0);}
  rc_registry_destroy(registry);
 }
 free(image_trace);
 image_trace=NULL;
 trace_requested=0;
 image_trace_mode=0;
 assert(!resource_effects_image_trace_enabled()&&!image_trace);
}

static void replacement_attachment_shader_effects(void) {
 const RcInterval attachment={0x10000,7,{12,0x1000,64}};
 const RcInterval disjoint[]={{0x20000,8,{12,0x2000,20}},{0x20014,9,{13,0x400,12}}};
 const RcInterval aliased={0x30000,10,{12,0x1010,4}};
 const RcInterval image={0x40000,11,{14,0x100,48}};
 registry=rc_registry_create((RcRegistryConfig){128,32,8});
 assert(registry);
 for(int mode=0;mode<5;++mode) {
  const uint64_t output=3+2*(uint64_t)mode;
  ResourceCommandEffects *e=command(1,output);
  ResourceImageEffectRole load={0};
  resource_effects_image_role_begin(e,&load);
  resource_effects_add(e,(RcSourceRange){1,attachment.address,attachment.range.size},
      (RcSourceRange){1,attachment.address,attachment.range.size},RC_REGISTRY_IMAGE);
  assert(resource_effects_add_intervals(e,&attachment,1,&attachment,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
  resource_effects_image_role_end(e,&load);
  RcSourceRange writer={1,disjoint[0].address,32};
  resource_effects_add(e,writer,writer,RC_REGISTRY_BUFFER);
  assert(resource_effects_add_intervals(e,disjoint,mode==1?1:2,disjoint,2,RC_REGISTRY_BUFFER)==RC_REGISTRY_OK);
  assert(resource_effects_add_intervals(e,&image,1,&image,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
  if(mode==2) assert(resource_effects_add_intervals(e,&aliased,1,&aliased,1,RC_REGISTRY_BUFFER)==RC_REGISTRY_OK);
  if(mode==3) resource_effects_opaque(e);
  if(mode==4) assert(resource_effects_add_intervals(e,&attachment,1,&attachment,1,RC_REGISTRY_IMAGE)==RC_REGISTRY_OK);
  RcInterval overlap={0};
  const RcRegistryResult got=resource_effects_shader_write_peer(e,&load,attachment,&overlap);
  if(mode==0) assert(got==RC_REGISTRY_OK && !overlap.mapping);
  if(mode==1 || mode==3) assert(got==RC_REGISTRY_UNSUPPORTED && !overlap.mapping);
  if(mode==2) assert(got==RC_REGISTRY_BLOCKED && overlap.mapping==aliased.mapping);
  if(mode==4) assert(got==RC_REGISTRY_BLOCKED && overlap.mapping==attachment.mapping);
  seal(1,output);
  resource_effects_release(e);
  assert(!rc_registry_stats(registry).writers && !nq);
 }
 rc_registry_destroy(registry);
 registry=NULL;
}

int main(void){
 replacement_attachment_shader_effects();
 image_producer_trace();
 registry=rc_registry_create((RcRegistryConfig){64,64,8});assert(registry);
 RcRegistryHandle image=source(0x1000),buffer=source(0x2010),gap=source(0x2040);
 ResourceCommandEffects *a=command(1,3);
 resource_effects_add(a,(RcSourceRange){1,0x1000,128},(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE);seal(1,3);
 ResourceCommandEffects *b=command(1,5);
 resource_effects_add(b,(RcSourceRange){1,0x1000,128},(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE);seal(1,5);
 assert(rc_registry_stats(registry).writers==1 && !visible(image));
 resource_effects_release(a);assert(!visible(image)); // canceled earlier declaration cannot cancel later owner.
 ResourceCommandEffects *old=resource_effects_enter(b);
 assert(resource_effects_image_prepare(7,0x1000,128));resource_effects_image_written(7,0x1000);
 resource_effects_leave(old);resource_effects_release(b);assert(nq==1);
 handoff(1,5);q[0].done(q[0].arg);nq=0;assert(!visible(image));
 assert(rc_registry_reconcile_cpu(registry,(RcSourceRange){1,0x1000,128})==RC_REGISTRY_OK);
 a=command(2,3);resource_effects_add(a,(RcSourceRange){1,0x1000,128},(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE);seal(2,3);
 old=resource_effects_enter(a);RcRegistryWrite hold={0};
 {RESOURCE_IMAGE_SCOPE(scope,2);resource_image_prepare(7,0x1000,128);
  {RESOURCE_IMAGE_SCOPE(nested,2);resource_image_written(7,0x1000);} // nested helper has no matching own preparation.
  assert(a->claims->actual && !a->claims->actual->recorded);
  assert(rc_registry_hold_begin(registry,(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE,&hold)==RC_REGISTRY_OK);
  assert(rc_registry_image_hold_bind(registry,hold,7)==RC_REGISTRY_OK);
  resource_image_written(7,0x1000);resource_image_owned(7,0x1000,hold);}
 resource_effects_leave(old);resource_effects_release(a);assert(!nq && !visible(image));
 assert(rc_registry_stats(registry).writers==1); // persistent hold, not one pending ticket per draw.
 assert(rc_registry_hold_release(registry,hold,1)==RC_REGISTRY_OK);assert(visible(image));
 a=command(3,3);resource_effects_add(a,(RcSourceRange){1,0x2010,16},(RcSourceRange){1,0x2000,128},RC_REGISTRY_BUFFER);seal(3,3);
 assert(!visible(buffer)&&visible(gap));old=resource_effects_enter(a);
 assert(resource_effects_buffer_prepare(0x2000,128,1));resource_effects_buffers_commit();resource_effects_leave(old);
 resource_effects_release(a);handoff(3,3);assert(nq==1);q[0].done(q[0].arg);nq=0;
 assert(visible(buffer)&&visible(gap));
 a=command(4,3);resource_effects_add(a,(RcSourceRange){1,0x2010,16},(RcSourceRange){1,0x2000,128},RC_REGISTRY_BUFFER);seal(4,3);
 old=resource_effects_enter(a);assert(resource_effects_buffer_prepare(0x2000,128,0));resource_effects_leave(old);
 resource_effects_release(a);assert(!nq && visible(buffer)); // preparation alone is not a recorded write.
 a=command(5,3);resource_effects_add(a,(RcSourceRange){1,0x2010,16},(RcSourceRange){1,0x2000,128},RC_REGISTRY_BUFFER);seal(5,3);
 old=resource_effects_enter(a);resource_effects_buffer_prepare(0x2000,128,0);resource_effects_buffers_commit();resource_effects_leave(old);
 resource_effects_release(a);assert(nq==1);q[0].cancel(q[0].arg,1);nq=0;assert(!visible(buffer));
 assert(rc_registry_reconcile_cpu(registry,(RcSourceRange){1,0,UINT64_MAX})==RC_REGISTRY_OK);
 assert(!rc_registry_stats(registry).writers && !shared_images);
 // Loaded semantics report a write, while carried reflection is empty/readonly. Production consumer selects
 // unknown authority; only an actual emitted command may retain it beyond cancellation.
 GfxShader missing={.classification=RC_EFFECT_WRITES_BUFFER,.reflection={0}};
 a=command(6,3);EffectDeclaration declaration={.commands=a};semantic_stage(&declaration,&missing);seal(6,3);
 assert(!visible(buffer)&&effect_stats.opaque_stages==1);
 resource_effects_release(a);assert(!nq&&visible(buffer));
 a=command(7,3);declaration.commands=a;semantic_stage(&declaration,&missing);seal(7,3);
 old=resource_effects_enter(a);resource_effects_opaque_commit();resource_effects_leave(old);
 resource_effects_release(a);handoff(7,3);assert(nq==1);q[0].done(q[0].arg);nq=0;
 assert(!visible(buffer)&&rc_registry_stats(registry).writer_foreign==1);
 assert(rc_registry_reconcile_cpu(registry,(RcSourceRange){1,0,UINT64_MAX})==RC_REGISTRY_OK);
 missing.classification=0;a=command(8,3);declaration.commands=a;semantic_stage(&declaration,&missing);seal(8,3);
 assert(visible(buffer)&&effect_stats.readonly_stages==1);resource_effects_release(a);assert(!nq);
 // A 43-byte GPU descriptor may have a 48-byte robust2 write envelope. Its empty marker must never erase
 // the five-byte tail of the actual claim. Check both the previous marker shape and the aligned owner shape.
 RcRegistryHandle tail=source(0x402b),outside=source(0x4030);
 for(unsigned pass=0;pass<4;pass++){
  uint64_t scene=9+pass;a=command(scene,3);
  resource_effects_empty(a,(RcSourceRange){1,0x4000,pass?48:43});
  resource_effects_add(a,(RcSourceRange){1,0x4000,48},(RcSourceRange){1,0x4000,48},RC_REGISTRY_BUFFER);seal(scene,3);
  assert(!visible(tail)&&visible(outside));old=resource_effects_enter(a);
  assert(resource_effects_buffer_prepare(0x4000,48,pass==2));
  if(pass!=1)resource_effects_buffers_commit();
  resource_effects_leave(old);resource_effects_release(a);
  if(pass==1)assert(!nq&&visible(tail)); // a prepared-but-canceled command releases the whole envelope.
  else{
   assert(nq==1);handoff(scene,3);
   if(pass==3)q[0].cancel(q[0].arg,1);else q[0].done(q[0].arg);
   nq=0;assert(visible(tail)==(pass==2)&&visible(outside));
  }
  assert(rc_registry_reconcile_cpu(registry,(RcSourceRange){1,0x4000,48})==RC_REGISTRY_OK);
 }
 // Frontend enter/leave does not mean a draw was emitted. Its unresolved shared position survives until
 // actual worker recording, which can certify a subsequent input before the whole group is released.
 a=command(13,3);resource_effects_add(a,(RcSourceRange){1,0x1000,128},(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE);seal(13,3);
 b=command(13,9);resource_effects_add(b,(RcSourceRange){1,0x1000,128},(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE);seal(13,9);
 RcRegistryImageBinding binding={.image=7};
 assert(rc_registry_hold_begin(registry,(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE,&binding.hold)==RC_REGISTRY_OK);
 assert(rc_registry_image_hold_bind(registry,binding.hold,7)==RC_REGISTRY_OK);
 RcRegistryBindingObservation observation;
 old=resource_effects_enter(a);resource_effects_leave(old);resource_effects_recorded(a,0);
 assert(rc_registry_binding_begin_image(registry,image,1,(RcRegistryPosition){13,4},binding,&observation)!=RC_REGISTRY_OK);
 old=resource_effects_enter(a);assert(resource_effects_image_prepare(7,0x1000,128));
 resource_effects_image_written(7,0x1000);resource_effects_image_owned(7,0x1000,binding.hold);resource_effects_leave(old);
 assert(rc_registry_binding_begin_image(registry,image,1,(RcRegistryPosition){13,4},binding,&observation)!=RC_REGISTRY_OK);
 resource_effects_recorded(a,1);
 assert(rc_registry_binding_begin_image(registry,image,1,(RcRegistryPosition){13,4},binding,&observation)==RC_REGISTRY_OK);
 assert(observation.next_read.scene==13&&observation.next_read.ordinal==8);
 resource_effects_release(a);assert(!nq);
 assert(rc_registry_binding_begin_image(registry,image,1,(RcRegistryPosition){13,4},binding,&observation)==RC_REGISTRY_OK);
 resource_effects_release(b);assert(!nq);
 assert(rc_registry_hold_release(registry,binding.hold,1)==RC_REGISTRY_OK);assert(visible(image));
 // A captured hold can retire before the actual command's finish. That write must retain completion work,
 // rather than cancel its broad shared claim merely because an earlier notification marked it owned.
 a=command(14,3);resource_effects_add(a,(RcSourceRange){1,0x1000,128},(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE);seal(14,3);
 assert(rc_registry_hold_begin(registry,(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE,&binding.hold)==RC_REGISTRY_OK);
 assert(rc_registry_image_hold_bind(registry,binding.hold,7)==RC_REGISTRY_OK);
 old=resource_effects_enter(a);assert(resource_effects_image_prepare(7,0x1000,128));
 resource_effects_image_written(7,0x1000);resource_effects_image_owned(7,0x1000,binding.hold);resource_effects_leave(old);
 assert(rc_registry_hold_release(registry,binding.hold,1)==RC_REGISTRY_OK);
 resource_effects_recorded(a,1);resource_effects_recorded(a,0);resource_effects_release(a);assert(nq==1&&!visible(image));
 handoff(14,3);q[0].done(q[0].arg);nq=0;assert(!visible(image));
 assert(rc_registry_reconcile_cpu(registry,(RcSourceRange){1,0x1000,128})==RC_REGISTRY_OK);
 // An actual command with unavailable attachment capture is unknown execution, never cancellation.
 a=command(15,3);resource_effects_add(a,(RcSourceRange){1,0x1000,128},(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE);seal(15,3);
 resource_effects_recorded(a,1);resource_effects_recorded(a,0);resource_effects_release(a);assert(nq==1&&!visible(image));
 handoff(15,3);q[0].done(q[0].arg);nq=0;assert(!visible(image));
 assert(rc_registry_reconcile_cpu(registry,(RcSourceRange){1,0x1000,128})==RC_REGISTRY_OK);
 a=command(16,3);resource_effects_add(a,(RcSourceRange){1,0x1000,128},(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE);seal(16,3);
 old=resource_effects_enter(a);resource_effects_leave(old);resource_effects_recorded(a,0);
 resource_effects_release(a);assert(!nq&&visible(image));
 // The attachment hook can retain an actual image with an unavailable ownership hold. Reporting hold zero
 // cannot discard the real write or certify a typed receipt; it needs ordinary unknown completion.
 a=command(17,3);resource_effects_add(a,(RcSourceRange){1,0x1000,128},(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE);seal(17,3);
 old=resource_effects_enter(a);assert(resource_effects_image_prepare(7,0x1000,128));
 resource_effects_image_written(7,0x1000);resource_effects_image_owned(7,0x1000,(RcRegistryWrite){0});resource_effects_leave(old);
 resource_effects_recorded(a,1);resource_effects_release(a);assert(nq==1&&!visible(image));
 handoff(17,3);q[0].done(q[0].arg);nq=0;assert(!visible(image));
 assert(rc_registry_reconcile_cpu(registry,(RcSourceRange){1,0x1000,128})==RC_REGISTRY_OK);
 // Losing typed image identity must not erase the still-live broad range owner or manufacture FOREIGN bytes.
 a=command(18,3);resource_effects_add(a,(RcSourceRange){1,0x1000,128},(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE);seal(18,3);
 assert(rc_registry_hold_begin(registry,(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE,&binding.hold)==RC_REGISTRY_OK);
 assert(rc_registry_image_hold_bind(registry,binding.hold,7)==RC_REGISTRY_OK);
 rc_registry_image_hold_forget(registry,binding.hold);
 assert(!rc_registry_image_binding_current(registry,binding));
 assert(rc_registry_image_hold_covers(registry,binding.hold,(RcSourceRange){1,0x1000,128}));
 old=resource_effects_enter(a);assert(resource_effects_image_prepare(8,0x1000,128));
 resource_effects_image_written(8,0x1000);resource_effects_image_owned(8,0x1000,binding.hold);
 resource_effects_emitted();resource_effects_leave(old);resource_effects_recorded(a,0);
 assert(rc_registry_binding_begin_image(registry,image,1,(RcRegistryPosition){18,4},binding,&observation)!=RC_REGISTRY_OK);
 resource_effects_release(a);assert(!nq && !visible(image));
 assert(rc_registry_stats(registry).writer_holds==1 && !rc_registry_stats(registry).writer_foreign);
 assert(rc_registry_hold_release(registry,binding.hold,1)==RC_REGISTRY_OK);assert(visible(image));
 // A substituted dispatch declares unknown image coverage but writes one actual output. Omitting its written
 // notification reproduces whole-domain FOREIGN; forwarding prepare/written/owned retains only the live target.
 for(unsigned forwarded=0;forwarded<2;forwarded++){
  uint64_t scene=19+forwarded;
  a=command(scene,3);resource_effects_add(a,(RcSourceRange){1,0,UINT64_MAX},(RcSourceRange){1,0,UINT64_MAX},RC_REGISTRY_IMAGE);seal(scene,3);
  assert(rc_registry_hold_begin(registry,(RcSourceRange){1,0x1000,128},RC_REGISTRY_IMAGE,&binding.hold)==RC_REGISTRY_OK);
  assert(rc_registry_image_hold_bind(registry,binding.hold,7)==RC_REGISTRY_OK);
  old=resource_effects_enter(a);
  {RESOURCE_IMAGE_SCOPE(scope,scene);resource_image_prepare(7,0x1000,128);
   resource_effects_emitted();
   if(forwarded)resource_image_written(7,0x1000);
   resource_image_owned(7,0x1000,binding.hold);}
  resource_effects_leave(old);resource_effects_recorded(a,0);resource_effects_release(a);
  if(!forwarded){
   assert(nq==1);handoff(scene,3);q[0].done(q[0].arg);nq=0;
   assert(!visible(buffer) && rc_registry_stats(registry).writer_foreign==1);
  }else{
   assert(!nq && visible(buffer) && !visible(image));
   assert(rc_registry_stats(registry).writer_holds==1 && !rc_registry_stats(registry).writer_foreign);
  }
  assert(rc_registry_hold_release(registry,binding.hold,1)==RC_REGISTRY_OK);
  assert(rc_registry_reconcile_cpu(registry,(RcSourceRange){1,0,UINT64_MAX})==RC_REGISTRY_OK);
  assert(visible(buffer)&&visible(image));
 }
 // A GPU-only substitution still reports its exact output to the declaration owner, while its scoped legacy
 // record has no CPU reservation. Its actual completion must not poison unrelated CPU sampled sources.
 RcRegistryObservation image_before,buffer_before,image_after,buffer_after;
 assert(rc_registry_observe(registry,image,1,&image_before)==RC_REGISTRY_OK);
 assert(rc_registry_observe(registry,buffer,1,&buffer_before)==RC_REGISTRY_OK);
 a=command(21,3);resource_effects_add(a,(RcSourceRange){1,0,UINT64_MAX},(RcSourceRange){1,0,UINT64_MAX},RC_REGISTRY_IMAGE);seal(21,3);
 old=resource_effects_enter(a);
 const uint64_t gpu_only=UINT64_C(0xf000000000001000);
 RcRegistryDiagnostics diagnostic_before,diagnostic_after;
 rc_registry_diagnostics(registry,&diagnostic_before);
 {RESOURCE_IMAGE_SCOPE(scope,21);resource_image_prepare(99,gpu_only,128);
  assert(scope.writes && !scope.writes->ticket.id);
  resource_effects_emitted();resource_image_written(99,gpu_only);}
 resource_effects_leave(old);resource_effects_recorded(a,0);resource_effects_release(a);
 handoff(21,3);for(unsigned i=0;i<nq;i++)q[i].done(q[i].arg);nq=0;
 assert(visible(buffer)&&visible(image));
 assert(rc_registry_observe(registry,image,1,&image_after)==RC_REGISTRY_OK);
 assert(rc_registry_observe(registry,buffer,1,&buffer_after)==RC_REGISTRY_OK);
 assert(image_after.content_revision==image_before.content_revision && buffer_after.content_revision==buffer_before.content_revision);
 assert(image_after.revision!=image_before.revision && buffer_after.revision!=buffer_before.revision);
 assert(rc_registry_stats(registry).writer_foreign==1);
 rc_registry_diagnostics(registry,&diagnostic_after);
 assert(diagnostic_after.counts[RC_REGISTRY_DIAG_MISSING_IMAGE]==diagnostic_before.counts[RC_REGISTRY_DIAG_MISSING_IMAGE]);
 assert(rc_registry_reconcile_cpu(registry,(RcSourceRange){1,gpu_only,128})==RC_REGISTRY_OK);
 assert(!rc_registry_stats(registry).writers);
 rc_registry_destroy(registry);
 registry=rc_registry_create((RcRegistryConfig){16,1,4});assert(registry);
 a=command(1,3);resource_effects_add(a,(RcSourceRange){1,0x3000,256},(RcSourceRange){1,0x3000,256},RC_REGISTRY_BUFFER);seal(1,3);
 old=resource_effects_enter(a);resource_effects_buffer_prepare(0x3000,16,0);resource_effects_buffer_prepare(0x3080,16,0);
 resource_effects_buffers_commit();resource_effects_leave(old);resource_effects_release(a);handoff(1,3);
 assert(nq==1);q[0].done(q[0].arg);nq=0;
 RcRegistryStats stats=rc_registry_stats(registry);
 assert(stats.completion_range_refusals==1 && stats.writer_foreign==1 && !stats.writer_submitted);
 assert(rc_registry_reconcile_cpu(registry,(RcSourceRange){1,0x3000,256})==RC_REGISTRY_OK);
 rc_registry_destroy(registry);
 normalized();
 typed_output_does_not_retain_dormant_command_guard();
 transferred_cpu_result();
 independent_upload();
 repeated_mapped_images();
 pinned_image_projection();
 canceled_exact_phase();
 captured_image_extent();
 projected_native_image_fate();
 bound_whole_image_fate();
 ordered_retained_image_effects();
 resident_image_split_fate();
 native_buffer_output_fate();
 preserved_buffer_output_fate();
 native_buffer_access_is_not_write_coverage();
 sampled_input_before_gpa();
 shared_positioned_completion();
 shared_positioned_completion_reverse();
 shared_positioned_completion_repeated();
 tensor_optional_role_fate();
 puts("ordered command effects: shared images, mapping-aware recording, atomic mixed-backing completion and abandonment passed");
}
'''
env=dict(os.environ,PATH=str(CC.parent)+os.pathsep+os.environ.get('PATH',''))
with tempfile.TemporaryDirectory(prefix='command-effects-',dir=ROOT/'build/test') as tmp:
 path=Path(tmp)/'check.c';path.write_text(prefix+host+source+image+semantic+ordered_read_gate+tests);exe=Path(tmp)/'check.exe'
 subprocess.run([str(CC),'-std=gnu11','-O1','-Wall','-Wextra','-Werror','-fsanitize=undefined','-fsanitize-trap=all','-DRC_REGISTRY_SINGLE_THREADED','-I',str(ROOT/'runtime'),str(path),str(ROOT/'runtime/resource_core_registry.c'),'-o',str(exe)],env=env,check=True)
 subprocess.run([str(exe)],env=env,check=True)
 # Negative control changes only the historical GPU-only early return in this generated host fixture.
 old_image=image.replace('if (!image_scope || !image || !size) return;',
     'if (!image_scope || !image || !size || nvn_gpu_only(addr)) return;',1)
 assert old_image!=image
 old_tests=tests.replace('assert(scope.writes && !scope.writes->ticket.id);','')
 # Keep the diagnostic proof fixture active on the current adapter. This historical negative deliberately removes
 # the recording callback it observes, and should still fail at the original CPU-source exclusion assertion.
 old_tests=old_tests.replace(' image_producer_trace();',' if (0) image_producer_trace();')
 path.write_text(prefix+host+source+old_image+semantic+ordered_read_gate+old_tests)
 subprocess.run([str(CC),'-std=gnu11','-O1','-Wall','-Wextra','-Werror','-fsanitize=undefined','-fsanitize-trap=all','-DRC_REGISTRY_SINGLE_THREADED','-I',str(ROOT/'runtime'),str(path),str(ROOT/'runtime/resource_core_registry.c'),'-o',str(exe)],env=env,check=True)
 result=subprocess.run([str(exe)],env=env,capture_output=True,text=True)
 assert result.returncode and 'visible(buffer)&&visible(image)' in result.stderr, result.stderr
 print('Negative control: GPU-only early return reproduces unrelated CPU-source exclusion')
 # The former completion path changed every source in the broad reservation, even when actual writes were
 # narrowed correctly. Exercise that independently from the notification bug through the real adapter above.
 registry=(ROOT/'runtime/resource_core_registry.c').read_text()
 old_registry=registry.replace('    const uint32_t diagnostic_flags = provenance.diagnostic_flags;\n'
                               '    invalidate_topology(r, w->interval.range);',
                               '    const uint32_t diagnostic_flags = provenance.diagnostic_flags;\n'
                               '    invalidate(r, w->interval.range);',1)
 assert old_registry!=registry
 old_path=Path(tmp)/'registry.c';old_path.write_text(old_registry)
 path.write_text(prefix+host+source+image+semantic+ordered_read_gate+tests)
 subprocess.run([str(CC),'-std=gnu11','-O1','-Wall','-Wextra','-Werror','-fsanitize=undefined','-fsanitize-trap=all','-DRC_REGISTRY_SINGLE_THREADED','-I',str(ROOT/'runtime'),str(path),str(old_path),'-o',str(exe)],env=env,check=True)
 result=subprocess.run([str(exe)],env=env,capture_output=True,text=True)
 assert result.returncode and 'image_after.content_revision==image_before.content_revision' in result.stderr, result.stderr
 print('Negative control: broad completion reproduces unrelated CPU-content version changes')
 # Replay previously rejected every private source when this operation owned any GPA scope.
 old_gate=ordered_read_gate.replace('if (declaration->gpa_unknown || declaration->gpa_count) {',
     '(void)private_sample;\n    if (declaration->gpa_unknown || declaration->gpa_count) return 0;\n    if (0) {',1)
 assert old_gate!=ordered_read_gate
 path.write_text(prefix+host+source+image+semantic+old_gate+tests)
 subprocess.run([str(CC),'-std=gnu11','-O1','-Wall','-Wextra','-Werror','-fsanitize=undefined','-fsanitize-trap=all','-DRC_REGISTRY_SINGLE_THREADED','-I',str(ROOT/'runtime'),str(path),str(ROOT/'runtime/resource_core_registry.c'),'-o',str(exe)],env=env,check=True)
 result=subprocess.run([str(exe)],env=env,capture_output=True,text=True)
 assert result.returncode and 'vk_exec_effects_read_allowed_range(contents,1)' in result.stderr, result.stderr
 print('Negative control: blanket same-operation GPA exclusion reproduces the ordered sampled input refusal')
