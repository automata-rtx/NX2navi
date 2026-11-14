#include <assert.h>
#include "shader_selection_under_test.h"

struct GfxShader { const uint8_t *control; int stage; GfxShaderAnswers answers; uint32_t mask; };
static struct GfxShader services[256];
static unsigned services_used, service_calls, shader_locks, texture_locks;
static unsigned avoided_calls;
static int mode=1, memo_mode=1, deferred, defer_handle, pending_service, lock_depth;
static uint32_t generation=1, service_mask=1u<<2;
static uint64_t tokens[64];
static uint8_t types[64], formats[64];
static int mutate_from, mutate_to, mutate_on_lock;
static uint8_t control[0xe0], code[64];
static prog_t programs[4];
static DrawCtx context;
static VkExec ex;
VkExecFrame vk_exec_frame;
static unsigned route_advanced, primary_draws, motion_draws;
static unsigned turns, ended_passes;
static int motion_lock_depth, descriptor_defer, ordered;
static struct { int failed; uint64_t ns_check; } MO;
static PlatMutex mo_lock = PLAT_MUTEX_INIT;
static int gate_motion(VkExec *ex);
static int mvobj_pick(VkExec *ex,const GfxShader **vs,uint32_t *banks,uint64_t t0) {
    (void)t0; *vs=NULL; *banks=0; return gate_motion(ex);
}
static void mvobj_draw_(VkExec *ex,const p_draw_idx *d,VkBuffer buffer,VkDeviceSize offset,uint64_t t0,const GfxShader *vs,uint32_t banks) {
    (void)d;(void)buffer;(void)offset;(void)t0;(void)vs;(void)banks;
    assert(ex->dc->B.st_valid && motion_lock_depth);
    if(descriptor_defer && !ordered) { deferred=1; return; }
    motion_draws++;
}
void rc_vkCmdDrawIndexed(VkCommandBuffer cb,uint32_t count,uint32_t instances,uint32_t first,int32_t vertex,uint32_t base) {
    (void)cb;(void)count;(void)instances;(void)first;(void)vertex;(void)base; primary_draws++;
}
#include "shader_caller_gates.h"

void rt_log(const char *fmt, ...) { (void)fmt; }
_Noreturn void rt_fatal(const char *fmt, ...) {
    va_list args; va_start(args,fmt); vfprintf(stderr,fmt,args); va_end(args); fputc('\n',stderr); abort();
}
int resource_prep_mode(void) { return mode; }
static int shader_memo_on(void) { return memo_mode; }
int vk_exec_shared(void) { return 1; }
int vk_pipeline_async_on(void) { return 1; }
uint32_t vk_texture_slots_gen(void) { return generation; }
uint64_t vk_texture_slot_token(int id) { assert((unsigned)id<64); return tokens[id]; }
int vk_texture_deferred(void) { return deferred; }
int vk_texture_deferred_take(void) { int was=deferred;deferred=0;return was; }
int vk_mvprev_on(void) { return 1; }
int rt_knob(const char *a,const char *b) { (void)a;(void)b;return 0; }
uint64_t kern_now_ns(void) { return 0; }
void vk_exec_pass_end(VkExec *state) {
    assert(!motion_lock_depth && !deferred && state->dc->B.st_valid && primary_draws==1);
    ended_passes++; state->in_pass=0;
}
void vk_exec_turn(void) { assert(!motion_lock_depth && !deferred); ordered=1;defer_handle=0;turns++; }
void plat_mutex_lock(PlatMutex *m) {
    if(m==&mo_lock) { assert(!motion_lock_depth);motion_lock_depth=1;return; }
    assert(m==&shader_lock); shader_locks++;
}
void plat_mutex_unlock(PlatMutex *m) {
    if(m==&mo_lock) { assert(motion_lock_depth);motion_lock_depth=0;return; }
    assert(m==&shader_lock);
}
static void mutate(int slot) { tokens[slot]++; generation++; types[slot]++; formats[slot]++; }
void vk_texture_lock(void) {
    assert(!lock_depth); lock_depth=1; texture_locks++;
    if (mutate_on_lock) { mutate(mutate_on_lock); mutate_on_lock=0; }
}
void vk_texture_unlock(void) { assert(lock_depth); lock_depth=0; }
static uint8_t type_of_handle(uint64_t h) {
    assert(lock_depth && h<64);
    if (h && (int)h==defer_handle) { deferred=1; return 0; }
    if (h && (int)h==mutate_from) { mutate(mutate_to); mutate_from=0; }
    return h ? types[h] : 0;
}
static uint8_t format_of_handle(uint64_t h) { assert(lock_depth && h<64); return h ? formats[h] : 0; }
VkFormat vk_vertex_format(int fmt,int *type) { if(type)*type=fmt%3; return VK_FORMAT_R32_UINT; }
static VkFormat target_format(int fmt,int *depth,int *stencil) { (void)depth;(void)stencil;return (VkFormat)fmt; }
static int format_type(VkFormat fmt) { return (unsigned)fmt%3; }
uint32_t gfx_shader_slot_mask(const GfxShader *s) { return s->mask; }
static const GfxShader *service(int stage,const uint8_t *ctl,const GfxShaderAnswers *a,int *pending) {
    service_calls++;
    if (pending) *pending=pending_service==1;
    if (pending_service) return NULL;
    for(unsigned i=0;i<services_used;i++)
        if(services[i].control==ctl && services[i].stage==stage && !memcmp(&services[i].answers,a,sizeof *a)) return &services[i];
    assert(services_used<256);
    services[services_used]=(struct GfxShader){ctl,stage,*a,service_mask};
    return &services[services_used++];
}
const GfxShader *gfx_shader_get(int st,const uint8_t *ctl,const uint8_t *data,uint64_t len,const uint8_t *c1,const GfxShaderAnswers *a) {
    (void)data;(void)len;(void)c1; return service(st,ctl,a,NULL);
}
const GfxShader *gfx_shader_get_async(int st,const uint8_t *ctl,const uint8_t *data,uint64_t len,const GfxShaderAnswers *a,int *pending) {
    (void)data;(void)len; return service(st,ctl,a,pending);
}
const GfxShader *gfx_shader_await(int st,const uint8_t *ctl,const uint8_t *data,uint64_t len,const GfxShaderAnswers *a,uint32_t ms,int *pending) {
    (void)data;(void)len;(void)ms; return service(st,ctl,a,pending);
}

static void reset(int stage) {
    memset(&context,0,sizeof context); memset(&ex,0,sizeof ex); memset(programs,0,sizeof programs);
    memset(shader_tab,0,sizeof shader_tab); memset(control,0,sizeof control);
    uint32_t magic=0x98761234u, st=(uint32_t)stage<<16, len=sizeof code;
    memcpy(control,&magic,4); memcpy(control+0x54,&st,4); memcpy(control+0x38,&len,4);
    for(unsigned i=0;i<4;i++) {
        programs[i].serial=i+1; programs[i].nstage=1;
        programs[i].sd[0].control=control; programs[i].sd[0].data=(uintptr_t)code;
    }
    for(unsigned i=0;i<64;i++) tokens[i]=i+100,types[i]=(uint8_t)(i%9+1),formats[i]=(uint8_t)(i%4+1);
    ex.dc=&context; ex.prog=&programs[0]; ex.tex[stage][2]=1; ex.img[stage][2]=2;
    mode=memo_mode=1; deferred=defer_handle=pending_service=lock_depth=0;
    mutate_from=mutate_to=mutate_on_lock=n_probe_sh=0;
    service_mask=1u<<2; generation=1;
    services_used=service_calls=shader_locks=answer_calls=texture_locks=resolve_calls=0;
    turns=ended_passes=0; motion_lock_depth=descriptor_defer=ordered=0;
}
static const GfxShader *select_ready(VkExec *state,int stage) {
    int present=0,pending=0;
    const GfxShader *s=shader_for(state,stage,&present,&pending,0);
    assert(s && present && !pending && !deferred && !lock_depth); return s;
}
static const GfxShader *prime(int stage) { select_ready(&ex,stage); return select_ready(&ex,stage); }
static void assert_hit(VkExec *state,int stage,const GfxShader *expected) {
    unsigned calls=resolve_calls,locks=shader_locks,answers=answer_calls;
    assert(select_ready(state,stage)==expected);
    assert(resolve_calls==calls && shader_locks==locks && answer_calls==answers); avoided_calls++;
}

static void dependencies(void) {
    const int stages[]={NVN_STAGE_VS,NVN_STAGE_FS,NVN_STAGE_CS,NVN_STAGE_TASK,NVN_STAGE_MESH};
    for(unsigned i=0;i<sizeof stages/sizeof stages[0];i++) {
        int st=stages[i]; reset(st); const GfxShader *s=prime(st);
        assert_hit(&ex,st,s);
        memset(&ex.pipe.blend,0x5a,sizeof ex.pipe.blend); memset(&ex.pipe.ds,0xa5,sizeof ex.pipe.ds);
        ex.pipe.depth_fmt++; ex.pipe.samples=4; ex.pipe.poly.cull=1; ex.pipe.vstream[0].stride=48;
        ex.ubo[st][0].addr=1234; ex.samp[st][0]=53; ex.tex[st][3]=33;
        assert_hit(&ex,st,s);
        va_st attribute; memcpy(&attribute,&ex.pipe.vattr[0],sizeof attribute);
        attribute.offset=137; attribute.stream=7; memcpy(&ex.pipe.vattr[0],&attribute,sizeof attribute);
        assert_hit(&ex,st,s);
        unsigned before=resolve_calls;
        if(st==NVN_STAGE_VS) ex.pipe.vattr[0]++;
        else if(st==NVN_STAGE_FS) ex.pipe.color_fmt[0]++;
        else ex.pipe.vattr[0]++,ex.pipe.color_fmt[0]++;
        s=select_ready(&ex,st);
        assert(resolve_calls==before+(st==NVN_STAGE_VS || st==NVN_STAGE_FS));
        before=resolve_calls; ex.tex[st][2]=3; s=select_ready(&ex,st); assert(resolve_calls==before+1);
        before=resolve_calls; ex.img[st][2]=4; s=select_ready(&ex,st); assert(resolve_calls==before+1);
        before=resolve_calls; generation++; s=select_ready(&ex,st); assert(resolve_calls==before+1);
        assert_hit(&ex,st,s);
        VkExec copied=ex; copied.tex[st][2]=5; // Equal tex_serial and the same context cannot conceal different values.
        before=resolve_calls; const GfxShader *changed=select_ready(&copied,st); assert(resolve_calls==before+1 && changed!=s);
        assert_hit(&ex,st,s);
    }
    reset(NVN_STAGE_VS); service_mask=0; const GfxShader *s=prime(NVN_STAGE_VS);
    generation++; ex.tex[0][0]=31; assert_hit(&ex,NVN_STAGE_VS,s);
    unsigned before=resolve_calls; context.sf_mvprev=1; const GfxShader *mv=select_ready(&ex,0);
    assert(resolve_calls==before+1 && mv!=s && mv->answers.mvprev);
    context.sf_mvprev=0; assert_hit(&ex,0,s);
    before=resolve_calls; programs[0].serial=998; select_ready(&ex,0); assert(resolve_calls==before+1);
}

static void interleaving_and_modes(void) {
    reset(NVN_STAGE_VS); const GfxShader *a=prime(0);
    ex.prog=&programs[1]; const GfxShader *b=prime(0);
    for(unsigned i=0;i<100;i++) { ex.prog=&programs[i&1]; assert_hit(&ex,0,i&1?b:a); }
    mode=2; unsigned calls=resolve_calls; select_ready(&ex,0); assert(resolve_calls==calls+1 && context.shader_packets.checked==1);
    mode=0; calls=resolve_calls; select_ready(&ex,0); select_ready(&ex,0); assert(resolve_calls==calls+2);
    mode=1; n_probe_sh=1; probe_sh[0].ctl=control; probe_sh[0].s=&services[0];
    calls=resolve_calls; select_ready(&ex,0); assert(resolve_calls==calls+1);
}

static void incomplete(void) {
    reset(0); defer_handle=1; int present=0,pending=0;
    assert(!shader_for(&ex,0,&present,&pending,0) && present && deferred && !pending && !service_calls);
    assert(!context.AC[0].ttok[2]);
    deferred=defer_handle=0; types[1]=9; const GfxShader *s=prime(0); assert(s->answers.tex_type[2]==9);
    reset(0); pending_service=1;
    assert(!shader_for(&ex,0,&present,&pending,0) && pending && !context.shader_packets.last[0]);
    pending_service=0; prime(0); assert(context.shader_packets.last[0]);
    reset(0); pending_service=2;
    assert(!shader_for(&ex,0,&present,&pending,0) && !pending && !context.shader_packets.last[0]);
    reset(0); programs[0].nstage=0;
    assert(!shader_for(&ex,0,&present,&pending,0) && !present && !context.shader_packets.last[0]);
    reset(0); s=prime(0); mode=2; memo_mode=2; defer_handle=1;
    assert(!shader_for(&ex,0,&present,&pending,0) && deferred && !context.shader_packets.last[0]->shader);
    deferred=defer_handle=0; memo_mode=1; s=select_ready(&ex,0); assert(s && !deferred);
}

static void answer_publications(void) {
    reset(0); GfxShaderAnswers a;
    ex.tex[0][0]=1; ex.tex[0][1]=2; ex.img[0][0]=2;
    assert(answers_for(&ex,0,control,3,&a));
    mutate(1); mutate_from=1; mutate_to=2;
    assert(answers_for(&ex,0,control,3,&a));
    assert(a.tex_type[1]==types[2] && a.img_type[0]==types[2] && a.img_format[0]==formats[2]);
    context.AC[0].ttok[0]=0; mutate_on_lock=1;
    assert(answers_for(&ex,0,control,1,&a));
    assert(context.AC[0].tgen[0]==generation && a.tex_type[0]==types[1]);
    memo_mode=2; defer_handle=1;
    assert(!answers_for(&ex,0,control,1,&a) && deferred && !context.AC[0].ttok[0]);
    deferred=defer_handle=0; memo_mode=1; mutate(1);
    assert(answers_for(&ex,0,control,1,&a) && a.tex_type[0]==types[1]);
}

static unsigned packet_set(uint64_t serial,unsigned stage) {
    return (unsigned)((serial*0x9e3779b97f4a7c15ull ^ (uint64_t)stage*0xd6e8feb86659fd93ull)>>58);
}

static void eviction(void) {
    reset(0); const GfxShader *a=prime(0);
    ShaderPacket *old_last=context.shader_packets.last[0];
    unsigned target=packet_set(programs[0].serial,0);
    uint8_t fragment_control[sizeof control]; memcpy(fragment_control,control,sizeof control);
    uint32_t fragment_stage=1u<<16; memcpy(fragment_control+0x54,&fragment_stage,4);
    uint64_t serial=2;
    for(unsigned i=1;i<3;i++) {
        while(packet_set(serial,1)!=target) serial++;
        programs[i].serial=serial++; programs[i].sd[0].control=fragment_control;
        ex.prog=&programs[i]; prime(1);
    }
    assert(context.shader_packets.last[0]==old_last && old_last->stage==1);
    ex.prog=&programs[0]; unsigned before=resolve_calls;
    assert(select_ready(&ex,0)==a && resolve_calls==before+1);
    assert_hit(&ex,0,a);
    assert(sizeof context.shader_packets<=128u*1024u);
}

static void caller_routing(void) {
    reset(NVN_STAGE_CS); route_advanced=0; defer_handle=1;
    gate_op_dispatch(&ex); assert(deferred && !route_advanced);
    deferred=0; gate_op_dispatch_ind(&ex); assert(deferred && !route_advanced);
    deferred=defer_handle=0; gate_op_dispatch(&ex); gate_op_dispatch_ind(&ex); assert(route_advanced==2);
    reset(NVN_STAGE_MESH); route_advanced=0; defer_handle=1;
    gate_mfb_draw(&ex); assert(deferred && !route_advanced);
    deferred=defer_handle=0; gate_mfb_draw(&ex); assert(route_advanced==1);
    reset(NVN_STAGE_VS); primary_draws=motion_draws=0; vk_exec_frame.draws=0; context.B.st_valid=1;
    p_draw_idx draw={.itype=NVN_INDEX_TYPE_UNSIGNED_SHORT,.count=3,.instances=1}; VkMemSpan span={0};
    defer_handle=1; gate_indexed(&ex,&draw,span);
    assert(!deferred && primary_draws==1 && motion_draws==1 && vk_exec_frame.draws==1 && !context.sf_mvprev);
    assert(turns==1 && ended_passes==1 && context.B.st_valid && ex.dirty==VK_EXEC_DIRTY_ALL && !ex.rt_resolved);
    reset(NVN_STAGE_VS); primary_draws=motion_draws=0; vk_exec_frame.draws=0; context.B.st_valid=1;
    descriptor_defer=1; gate_indexed(&ex,&draw,span);
    assert(!deferred && primary_draws==1 && motion_draws==1 && vk_exec_frame.draws==1 && turns==1 && ended_passes==1);
    draw.itype=NVN_INDEX_TYPE_UNSIGNED_BYTE; gate_indexed(&ex,&draw,span);
    assert(primary_draws==2 && motion_draws==1 && vk_exec_frame.draws==2);
}

int main(void) {
    dependencies(); interleaving_and_modes(); incomplete(); answer_publications(); eviction(); caller_routing();
    printf("shader_packet_check: %u ready selections avoided the original resolver, answer construction and shared shader lock; all five stages, exact dependencies, copied state, reference/check modes, pending/deferred results, answer publication, dispatch routing and one-primary motion retries passed\n",avoided_calls);
    return 0;
}
