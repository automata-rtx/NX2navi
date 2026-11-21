"""Production effect-memo ownership and immutable-input consumer adapter, CPU only."""
from pathlib import Path
import os,re,subprocess,tempfile
from bind_recipe_check import function
ROOT=Path(__file__).resolve().parents[1];CC=Path(os.environ["CC"])
s=(ROOT/"runtime/vk_exec_draw.c").read_text()
def extract(name):
 m=re.search(r"^static [^\n]*\b"+name+r"\([^;{]*\)\s*\{",s,re.M);assert m,name
 return function(s[m.start():],name)
prefix=r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "resource_core_effects.h"
struct RcEffectMemo {int valid,value;};
struct RcEffectProgram {int supported;};
typedef struct {RcEffectProgram program;} GfxShader;
typedef struct {RcEffectMemo *effect_memo[16];size_t effect_memo_bytes;void *effect_workspace;size_t effect_workspace_bytes;} DrawCtx;
typedef struct {DrawCtx *dc;} VkExec;
typedef struct {VkExec *ex;const GfxShader *shader;int stage;uint32_t groups[3];int value,current,reads;
 int mirrored_gpa;uint32_t gpa_set,gpa_binding;} EffectInputCapture;
enum {NVN_STAGE_CS=5};
#define EFFECT_MEMO_SLOTS 16u
#define EFFECT_MEMO_CONTEXT_BYTES (512u*1024u)
static uint64_t effect_memo_live_bytes,effect_memo_live_slots;
static struct {uint64_t memo_hits,memo_misses;} effect_stats;
static size_t memo_bytes=sizeof(RcEffectMemo);
static unsigned evaluations,uncached;
size_t rc_effect_memo_size(void){return memo_bytes;}
RcEffectMemo *rc_effect_memo_init(void *storage,size_t bytes){assert(bytes==memo_bytes);memset(storage,0,bytes);return storage;}
static const RcEffectProgram *gfx_shader_effects(const GfxShader *s){return &s->program;}
int rc_effect_supported(const RcEffectProgram *p){return p->supported;}
int rc_effect_has_mirrored_gpa_store_plan(const RcEffectProgram *p){(void)p;return 0;}
RcEffectResult rc_effect_evaluate_definite_stores(const RcEffectProgram *p,const RcEffectInputs *in,
 RcEffectRange *ranges,size_t capacity,size_t *count) {
 (void)p;(void)in;(void)ranges;(void)capacity;(void)count;abort();
}
size_t rc_effect_workspace_size(const RcEffectProgram *p){(void)p;return 64;}
static void gfx_shader_local_size(const GfxShader *s,uint32_t *v){(void)s;v[0]=v[1]=v[2]=1;}
static int effect_input_read(void *ctx,uint32_t set,uint32_t binding,uint64_t offset,size_t size,void *out){
 EffectInputCapture *c=ctx;assert(set==0&&binding==0&&offset==0&&size==4);c->reads++;memcpy(out,&c->value,4);return 1;
}
static int effect_input_builtin(void *c,uint32_t b,uint32_t x,uint64_t *lo,uint64_t *hi){(void)c;(void)b;(void)x;*lo=*hi=0;return 1;}
static int effect_input_current(const EffectInputCapture *c){return c->current;}
RcEffectResult rc_effect_evaluate_memo(const RcEffectProgram *p,const RcEffectInputs *in,RcEffectMemo *m,RcEffectRange *ranges,size_t cap,size_t *count,int *reused){
 (void)p;assert(m&&in->workspace_bytes>=64&&cap);int value;assert(in->read(in->context,0,0,0,4,&value));
 *reused=m->valid&&m->value==value;if(!*reused)evaluations++;
 m->valid=1;m->value=value;*count=1;ranges[0]=(RcEffectRange){.kind=RC_EFFECT_GPA,.begin=(uint64_t)value,.size=4};
 return RC_EFFECT_COMPLETE;
}
RcEffectResult rc_effect_evaluate(const RcEffectProgram *p,const RcEffectInputs *in,RcEffectRange *ranges,size_t cap,size_t *count){
 (void)p;(void)in;(void)ranges;(void)cap;uncached++;*count=0;return RC_EFFECT_UNKNOWN;
}
'''
body="\n".join(extract(n) for n in ("effect_memo_for","effect_memo_release","effect_mirrored_gpa","effect_shader_ranges"))
tests=r'''
int main(void){
 DrawCtx c={0};VkExec ex={&c};GfxShader shader={.program={1}};
 EffectInputCapture capture={.ex=&ex,.shader=&shader,.stage=2,.value=4096,.current=1};
 RcEffectRange ranges[4];size_t count;
 assert(effect_shader_ranges(&capture,1,ranges,4,&count)==RC_EFFECT_COMPLETE&&count==1);
 assert(evaluations==1&&capture.reads==1&&effect_memo_live_slots==1);
 assert(effect_shader_ranges(&capture,1,ranges,4,&count)==RC_EFFECT_COMPLETE&&effect_stats.memo_hits==1&&capture.reads==2);
 capture.current=0;
 assert(effect_shader_ranges(&capture,1,ranges,4,&count)==RC_EFFECT_UNKNOWN&&!count); // A hit cannot skip lease validation.
 capture.current=1;capture.value=8192;
 assert(effect_shader_ranges(&capture,1,ranges,4,&count)==RC_EFFECT_COMPLETE&&ranges[0].begin==8192&&evaluations==2);
 // No per-command reset; release the context once and account all retained storage exactly.
 effect_memo_release(&c);assert(!c.effect_memo_bytes&&!effect_memo_live_slots&&!effect_memo_live_bytes);
 effect_memo_release(&c);assert(!effect_memo_live_bytes);
 memo_bytes=EFFECT_MEMO_CONTEXT_BYTES+1;
 assert(effect_shader_ranges(&capture,1,ranges,4,&count)==RC_EFFECT_UNKNOWN&&uncached==1&&!c.effect_memo_bytes);
 memo_bytes=sizeof(RcEffectMemo);
 for(int stage=0;stage<16;stage++)assert(effect_memo_for(&c,&shader,stage));
 assert(effect_memo_live_slots<=16&&effect_memo_live_bytes<=EFFECT_MEMO_CONTEXT_BYTES);
 effect_memo_release(&c);free(c.effect_workspace);assert(!effect_memo_live_bytes&&!effect_memo_live_slots);
 puts("effect memo consumer: immutable dependencies, final lease refusal, bounded context storage, fallback and shutdown accounting passed");
}
'''
env=dict(os.environ,PATH=str(CC.parent)+os.pathsep+os.environ.get("PATH",""))
with tempfile.TemporaryDirectory(prefix="effect-memo-consumer-",dir=ROOT/"build/test") as tmp:
 p=Path(tmp)/"consumer.c";p.write_text(prefix+body+tests);exe=p.with_suffix(".exe")
 subprocess.run([str(CC),"-std=gnu11","-O1","-Wall","-Wextra","-Werror","-fsanitize=undefined","-fsanitize-trap=all","-I",str(ROOT/"runtime"),str(p),"-o",str(exe)],env=env,check=True)
 subprocess.run([str(exe)],env=env,check=True)
