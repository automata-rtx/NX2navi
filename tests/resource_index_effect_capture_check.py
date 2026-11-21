"""Exercise the exact aligned-index input used by bounded shader-effect evaluation."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DRAW = (ROOT / "runtime/vk_exec_draw.c").read_text()
CC = Path(os.environ["CC"])


def extract(signature):
    start = DRAW.index(signature)
    end = DRAW.index("\n}", start) + 2
    return DRAW[start:end]


PREFIX = r"""
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "resource_core_registry.h"
#define CHUNK_BYTES 2048
#define NVN_INDEX_TYPE_UNSIGNED_SHORT 1
#define NVN_INDEX_TYPE_UNSIGNED_INT 2
typedef uint64_t VkBuffer, VkDeviceSize;
typedef struct {VkBuffer buffer; VkDeviceSize offset,range;} VkDescriptorBufferInfo;
typedef struct {VkBuffer buffer;} VkMemSpan;
typedef struct {int held;} VkMemSpanCapture;
typedef struct {int prim,itype,count,base_vertex;uint64_t ibuf;} p_draw_idx;
typedef struct {int live;} VkExec;
typedef struct {
    int compatibility,exact_index;
    VkMemSpanCapture *index_hold;
    VkMemSpan index;
    RcRegistryPosition read_position;
} VkExecGraphicsDraw;
typedef struct {
    RcRegistryPosition position;
    uint32_t vertex_lo,vertex_hi,index_extents;
    int index_captured;
    uint64_t index_addr;
    VkDescriptorBufferInfo indices;
    RcInterval *index_mapping;
    void *commands;
} EffectDeclaration;
typedef struct {const uint8_t *bytes; void *opaque;} VkExecHostInputRead;
enum {VK_EXEC_HOST_INPUT_READY,VK_EXEC_HOST_INPUT_BLOCKED,VK_EXEC_HOST_INPUT_STALE,VK_EXEC_HOST_INPUT_CHANGED};

static VkExecGraphicsDraw *graphics_current;
static unsigned begins, finishes, copies, compatibility;
static int cpu_visible=1, earlier_gpu, settled_during, source_changed, alias, retained_current=1;
static int vk_mem_span_input_cpu_visible(const VkMemSpanCapture *span){assert(span->held);return cpu_visible;}
static int vk_mem_span_held_argument_current(const VkMemSpanCapture *span,uint64_t address,uint64_t size,
                                             const VkMemSpan *selected) {
    assert(span->held&&address&&size==6&&selected->buffer==77);
    return retained_current;
}
static uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *span,uint64_t address,uint64_t size,
                                           RcInterval **out) {
    assert(span->held&&size==6);
    *out=malloc(2*sizeof **out);assert(*out);
    (*out)[0]=(RcInterval){address,17,{12,40,2}};
    (*out)[1]=(RcInterval){address+2,18,{14,100,4}};
    return 2;
}
static int vk_exec_host_input_begin(const VkMemSpanCapture *owner,const VkMemSpan *span,uint64_t address,
                                   uint64_t size,RcRegistryPosition cut,VkExecHostInputRead *read) {
    assert(owner->held&&span->buffer==77&&size==6&&cut.scene==7&&cut.ordinal==12&&!read->opaque);
    begins++;
    if(earlier_gpu)return VK_EXEC_HOST_INPUT_BLOCKED;
    read->bytes=(const uint8_t *)(uintptr_t)address;read->opaque=read;
    return VK_EXEC_HOST_INPUT_READY;
}
static int vk_exec_host_input_finish(VkExecHostInputRead *read) {
    assert(read->opaque==read&&read->bytes);*read=(VkExecHostInputRead){0};finishes++;
    if(settled_during)return VK_EXEC_HOST_INPUT_STALE;
    return source_changed?VK_EXEC_HOST_INPUT_CHANGED:VK_EXEC_HOST_INPUT_READY;
}
static int effect_index_compatibility(VkExec *ex,uint64_t addr,uint64_t bytes,RcRegistryPosition read,
                                      VkDescriptorBufferInfo *descriptor,const uint8_t **captured) {
    assert(ex->live&&addr&&bytes==6&&read.scene==7&&read.ordinal==12);compatibility++;
    *descriptor=(VkDescriptorBufferInfo){42,0,bytes};
    *captured=(const uint8_t *)(uintptr_t)addr;
    return 1;
}
static int restart_for(const VkExec *ex,int prim,int type){assert(ex->live&&prim==2&&type==1);return 1;}
static uint8_t owned_bytes[6];
typedef enum { SCRATCH_OPTIONAL, SCRATCH_REQUIRED, SCRATCH_REQUIRED_PRIOR } ScratchUse;
static uint8_t *scratch(VkExec *ex,uint64_t size,VkBuffer *buffer,VkDeviceSize *offset,ScratchUse use) {
    assert(use == SCRATCH_OPTIONAL);
    assert(ex->live&&size==sizeof owned_bytes);*buffer=89;*offset=8;copies++;return owned_bytes;
}
static int resource_effects_read_allowed_range(void *commands,RcSourceRange range) {
    assert(commands&&rc_range_valid(range));
    return !alias || !rc_range_overlap(range,(RcSourceRange){14,102,1});
}
"""
CASES = r"""
int main(void) {
    VkExec ex={.live=1};
    VkMemSpanCapture binding={1};
    VkExecGraphicsDraw graphics={.exact_index=1,.index_hold=&binding,.index={77},.read_position={7,12}};
    graphics_current=&graphics;
    uint16_t indices[]={5,UINT16_MAX,1};
    p_draw_idx cmd={.prim=2,.itype=1,.count=3,.base_vertex=3,.ibuf=(uint64_t)(uintptr_t)indices};
    EffectDeclaration effect={.position={7,13},.commands=&ex};
    assert(effect_capture_indices(&effect,&ex,&cmd));
    assert(effect.vertex_lo==4&&effect.vertex_hi==8&&effect.indices.buffer==89&&effect.indices.offset==8);
    assert(!memcmp(owned_bytes,indices,6)&&copies==1&&begins==1&&finishes==1&&!compatibility);
    assert(effect.index_extents==2&&effect_captured_mapping_clear(&effect,effect.index_mapping,effect.index_extents,6));
    alias=1; // Same backing-relative interval via a different guest range.
    assert(!effect_captured_mapping_clear(&effect,effect.index_mapping,effect.index_extents,6));
    assert(effect_captured_mapping_clear(&effect,effect.index_mapping,effect.index_extents,4));
    free(effect.index_mapping);
    effect=(EffectDeclaration){.position={7,13},.commands=&ex};

    earlier_gpu=1;
    assert(!effect_capture_indices(&effect,&ex,&cmd)&&begins==2&&finishes==1&&copies==1&&!compatibility);
    earlier_gpu=0;settled_during=1;
    assert(!effect_capture_indices(&effect,&ex,&cmd)&&begins==3&&finishes==2&&copies==1&&!compatibility);
    settled_during=0;source_changed=1;
    assert(!effect_capture_indices(&effect,&ex,&cmd)&&begins==4&&finishes==3&&copies==1&&!compatibility);
    source_changed=0;cpu_visible=0;
    assert(!effect_capture_indices(&effect,&ex,&cmd)&&begins==4&&copies==1&&!compatibility);
    cpu_visible=1;graphics.read_position.ordinal=10;
    assert(!effect_capture_indices(&effect,&ex,&cmd)&&begins==4&&copies==1&&!compatibility);
    graphics.compatibility=1;
    assert(!effect_capture_indices(&effect,&ex,&cmd)&&begins==4&&copies==1&&!compatibility);
    graphics.read_position.ordinal=12;retained_current=0;
    assert(!effect_capture_indices(&effect,&ex,&cmd)&&begins==4&&copies==1&&!compatibility);
    retained_current=1;
    assert(effect_capture_indices(&effect,&ex,&cmd)&&!compatibility&&effect.indices.buffer==89&&copies==2);
    free(effect.index_mapping);
    effect=(EffectDeclaration){.position={7,13},.commands=&ex};
    graphics.index_hold=NULL;
    assert(effect_capture_indices(&effect,&ex,&cmd)&&compatibility==1&&effect.indices.buffer==42);
    assert(!effect.index_mapping&&!effect.index_extents&&copies==2);
    puts("aligned-index effect input: retained cutoff, mapped aliases and typed rejection passed");
    return 0;
}
"""

source = PREFIX + "\n".join(extract(signature) for signature in (
    "static int effect_index_protected(",
    "static int effect_capture_indices(",
    "static int effect_captured_mapping_clear(",
)) + CASES
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
work = ROOT / "build/test"
work.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="index-effects-", dir=work) as tmp:
    c = Path(tmp) / "check.c"
    c.write_text(source)
    exe = c.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-I" + str(ROOT / "runtime"),
                    str(c), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)
