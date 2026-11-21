"""Native color producer effect association with the real registry (CPU only)."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

import argparse
import tempfile

args = argparse.ArgumentParser(description=__doc__)
args.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
args.add_argument("--draw", type=Path)
args.add_argument("--dlaa", type=Path)
args.add_argument("--nested-baseline", action="store_true", help="Reproduce the unprepared nested output")
args.add_argument("--baseline", action="store_true", help="Reproduce the previous tentative producer")
opts = args.parse_args()
root = opts.root.resolve()
temp = tempfile.TemporaryDirectory(prefix="private-color-blit-effect-")
here = Path(temp.name)
sys.path.insert(0, str(root / "tests"))
from bind_recipe_check import function
draw = (opts.draw or root / "runtime/vk_exec_draw.c").read_text(encoding="utf-8")
corrected = not opts.baseline
runtime = root/"runtime"
texture = (runtime/"vk_texture.c").read_text()
dlaa = (opts.dlaa or runtime/"vk_dlaa.c").read_text(encoding="utf-8")
fixture = (root/"tests/tensor_effect_outputs_check.py").read_text()
prefix = fixture.split("prefix = r'''",1)[1].split("'''",1)[0]
prefix = prefix[:prefix.index("static void effect_declare_host(")]
prefix = prefix.replace('#include "resource_core_registry.h"',
    '#include "resource_core_registry.h"\n#include "resource_command_effects.h"\n#include "resource_image_write.h"')
prefix = prefix.replace("typedef struct {void *owner;} ResourceImageEffectRole;\n", "")
prefix = prefix.replace("#define RESOURCE_IMAGE_SCOPE(name,fence) (void)(fence)\n","")
prefix = prefix.replace("#define rc_registry_runtime() registry", "static RcRegistry *fixture_registry(void){return registry;}\n#define rc_registry_runtime() fixture_registry()")
prefix = prefix.replace("void *commands;EffectFallbackCapture", "ResourceCommandEffects *commands;EffectFallbackCapture")
prefix = prefix.replace("static int nvn_gpu_only(uint64_t a){return a>=UINT64_C(0xf000000000000000);}",
                        "static int nvn_gpu_only(uint64_t a){return a>=UINT64_C(0x8000000000);}")
prefix = prefix.replace("uint32_t image_private_mapping_count,image_mapped_hold_count;",
    "uint32_t image_private_mapping_count,image_mapped_hold_count; RcRegistryWrite *image_mapped_holds;")
prefix += r'''
enum { CAPACITY=8 };
typedef int PlatMutex;
#define PLAT_MUTEX_INIT 0
static void plat_mutex_lock(PlatMutex *p){assert(!*p);*p=1;}
static void plat_mutex_unlock(PlatMutex *p){assert(*p);*p=0;}
static void rt_fatal(const char *s,...){fprintf(stderr,"%s\n",s);abort();}
static void rt_log(const char *s,...){(void)s;}
static struct {void (*complete)(void *);void (*cancel)(void *,int);void *arg;} queue[CAPACITY];
static unsigned queued;
static void vk_completion_add_owned(uint64_t fence,void (*complete)(void *),void (*cancel)(void *,int),void *arg){
    assert(fence==read_position.scene && queued<CAPACITY);
    queue[queued++]=(typeof(queue[0])){complete,cancel,arg};
}
typedef struct {RcInterval *intervals;RcRegistryWrite *tickets;RcRegistryWrite guard;uint32_t count;} ResourceHostWrite;
typedef struct {void *alloc;} VkMemSpan;
static RcInterval live_mapping;
static int raw_write_refused;
static unsigned raw_attempts;
static int vk_mem_lookup(uint64_t addr,uint64_t size,VkMemSpan *span){
    if(addr!=live_mapping.address || size!=live_mapping.range.size)return 0;
    span->alloc=(void *)1;return 1;
}
static uint32_t vk_mem_writer_intervals(const VkMemSpan *span,uint64_t a,uint64_t n,RcInterval **out){
    assert(span->alloc&&a==live_mapping.address&&n==live_mapping.range.size);
    raw_attempts++;
    if(raw_write_refused){*out=NULL;return 0;}
    *out=malloc(sizeof **out);assert(*out);**out=live_mapping;return 1;
}
static uint32_t resource_host_current_intervals(uint64_t a,uint64_t n,RcInterval **out){
    (void)a;(void)n;*out=NULL;return 0;
}
static ResourceHostWrite resource_host_write_prepare(RcInterval *,uint32_t,RcRegistryWriterKind,int);
static void resource_host_write_dispose(ResourceHostWrite *,uint64_t,int,int);
static const RcInterval *image_backing_intervals(const Tex *e,uint32_t *n){
    *n=e->image_private_mapping_count;return e->image_private_mapping;
}
static int placement_current(const Tex *e){
    const RcInterval *m=e->image_private_mapping;
    return m && m->mapping==live_mapping.mapping && m->address==live_mapping.address &&
        rc_range_equal(m->range,live_mapping.range);
}
static void vk_texture_storage_blit_written(const VkTextureColorBlitSource *, const VkImageBlit *);
static int overlap_scene;
static int vk_dlaa_scene_effects(VkDlaaSceneEffects *capture){
    *capture=(VkDlaaSceneEffects){.complete=1};
    if(overlap_scene) {
        capture->images[0]=99;
        capture->ranges[0]=(RcSourceRange){1,live_mapping.address,live_mapping.range.size};
    }
    return 1;
}
static void vk_dlaa_scene_prepare_edges(VkDlaaSceneEffects *scene,const VkTextureColorBlitSource *target){
    (void)scene;(void)target; // This fixture executes primary output only.
}
static void vk_texture_projected_transfer_cancel(VkTextureProjectedTransfer **p){assert(!*p);}
static RcRegistryResult vk_texture_projected_transfer_declare(const VkTextureProjectedTransfer *p,ResourceCommandEffects *e,RcRegistryPosition o){
    (void)p;(void)e;(void)o;abort();
}
'''
host = (runtime/"resource_host_write.c").read_text()
core = (runtime/"resource_command_effects.c").read_text()
scope = (runtime/"resource_image_write.c").read_text()
strip = lambda s: "\n".join(line for line in s.splitlines() if not line.startswith("#include"))+"\n"
# Keep real scope and effect ownership. IRQ/CPU mapping of unrelated resources is outside this fixture.
prefix += "\n"+strip(core)+strip(scope)
prefix += function(host,"resource_host_interval_slice")
prefix += r'''
static ResourceHostWrite resource_host_write_prepare(RcInterval *intervals,uint32_t count,
        RcRegistryWriterKind kind,int declared) {
    assert(kind==RC_REGISTRY_IMAGE);
    (void)declared;
    return (ResourceHostWrite){.intervals=intervals,.count=count};
}
static void resource_host_write_dispose(ResourceHostWrite *write,uint64_t fence,int done,int visible){
    (void)fence;(void)done;(void)visible;free(write->intervals);memset(write,0,sizeof *write);
}
'''
prefix += function(texture,"texture_color_blit_capture")
prefix += function(texture,"texture_color_blit_complete")
texture_prepare = texture[texture.index("static void tex_image_prepare(const Tex *e, uint64_t size) {"):]
for name in ("tex_image_prepare", "image_write_prepare_locked", "vk_texture_image_write_prepare"):
    prefix += function(texture_prepare, name)
prefix += function(dlaa,"dlaa_guest_target")
prefix += function(dlaa,"dlaa_compute_blit_written")
prefix += r'''
static RcRegistryResult projection;
static int mapped_effect;
static RcRegistryImageWriteExtent extent;
static void vk_texture_storage_blit_written(const VkTextureColorBlitSource *output,const VkImageBlit *region){
    Tex *e=owners;
    assert(e->pub.image==output->image);
    uint32_t count;
    const RcInterval *mapping=image_backing_intervals(e,&count);
    assert(count==1&&!e->image_mapped_hold_count);
    assert(rc_registry_image_hold_begin_extents(registry,e->pub.image,mapping,count,e->image_mapped_holds)==RC_REGISTRY_OK);
    e->image_mapped_hold_count=count;
    extent=(RcRegistryImageWriteExtent){*mapping,*e->image_mapped_holds};
    resource_image_owned_mapped((uint64_t)e->pub.image,e->pub.addr,mapping,e->image_mapped_holds,count);
    RcRegistryPosition odd=vk_exec_resource_position();odd.ordinal |= 1;
    mapped_effect=resource_effects_image_emitted_intervals(e->pub.image,mapping,e->image_mapped_holds,count,odd,1);
    if (texture_color_blit_complete(e,output,region) && placement_current(e)) {
        RcSourceRange meaningful=mapping->range;meaningful.size=0x127a800;
        RcRegistryNativeImageWrite write={extent,&meaningful,1};
        projection=rc_registry_native_image_write_fragments(registry,e->pub.image,&write,1,odd);
    }
}
'''
evaluate = function(dlaa, "evaluate")
scope_begin = evaluate.index("    RESOURCE_IMAGE_SCOPE(image_writes, ex->fence);")
scope_end = evaluate.index("\n", scope_begin)
copy_begin = evaluate.index("    RcSourceRange output = dlaa_guest_target(dst);")
copy_end = evaluate.index("    full_barrier(cb);", copy_begin)
prefix += r'''
#define VK_FILTER_LINEAR 1
#define VK_FILTER_NEAREST 0
static unsigned nested_blits;
static int baseline_evaluate;
static void vk_texture_storage_written(VkImage image, uint64_t address) {
    assert(image && address);
}
static void vkCmdBlitImage(VkCommandBuffer cb,VkImage src,int src_layout,VkImage dst,int dst_layout,
        unsigned count,const VkImageBlit *box,int filter) {
    assert(cb && src && src!=dst && src_layout==VK_IMAGE_LAYOUT_GENERAL && dst_layout==src_layout);
    assert(count==1 && box->dstSubresource.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT && filter==VK_FILTER_NEAREST);
    assert(!mapped_effect && projection==RC_REGISTRY_INVALID);
    assert(image_scope && image_scope->parent && image_scope->parent->writes);
    int prepared_here=0;
    for (ResourceImageWrite *w=image_scope->writes;w;w=w->next)
        if (w->image==dst && w->addr==live_mapping.address && !w->recorded) prepared_here++;
    assert(prepared_here==!baseline_evaluate);
    nested_blits++;
}
static void evaluate_copy(VkExec *ex,const VkExecTarget *dst,const VkTextureColorBlitSource *compute_output) {
    const VkCommandBuffer cb=ex->cb;
    const VkImage shown=21;
    const unsigned ow=dst->width, w=1720;
    const struct {int early;} D={0};
    VkImageBlit b={.dstSubresource={1,0,0,1},.dstOffsets={{0,0,0},{3440,1440,1}}};
'''
prefix += evaluate[scope_begin:scope_end] + "\n" + evaluate[copy_begin:copy_end] + "}\n"
tests = r'''
static int run_case(int deny_raw,int stale,int canceled,int success,int abandon,int overlap,int nested){
    registry=rc_registry_create((RcRegistryConfig){64,64,8});assert(registry&&!queued);
    const RcRegistryPosition placement={14057,46962},write={14057,46963},read={14057,47126};
    read_position=placement;
    const RcInterval captured={0x80185d7e00,1349,{1346,0x185d7e00,0x1428000}};
    live_mapping=captured;
    raw_write_refused=deny_raw;raw_attempts=0;projection=RC_REGISTRY_INVALID;mapped_effect=0;overlap_scene=overlap;
    nested_blits=0;
    baseline_evaluate=NESTED_BASELINE;
    RcRegistryWrite hold={0};
    const VkImage image=0x19afdfc0cd0;

    Tex output={
      .pub={.image=image,.addr=captured.address,.width=3440,.height=1440,.samples=1,.aspect=1,
            .type=VK_IMAGE_TYPE_2D,.format=122,.levels=1,.layers=1},
      .t={.L.size=0x1428000,.serial=17,.d={.width=3440,.height=1440,.target=2,.format=63}},
      .image_private_mapping=&captured,.image_private_mapping_count=1,.image_mapped_hold_count=0,
      .independent=1,.image_mapped_holds=&hold
    };
    owners=&output;slots[1]=&output;
    const Program program={118};
    VkExec ex={.prog=(Program *)&program,.fence=placement.scene,.cb=1};ex.img[0][0]=1;
    EffectDeclaration d={.commands=resource_effects_create(write),.program=(Program *)&program,.program_serial=program.serial};
    assert(rc_registry_declare_through(registry,placement)==RC_REGISTRY_OK);
    assert(effect_tensor_outputs(&d,&ex)&&d.fallback->blit.mapping_count==1&&color_holds==1);
    assert(rc_registry_declare_through(registry,read)==RC_REGISTRY_OK);
    if(!canceled){
        RESOURCE_IMAGE_SCOPE(issue,ex.fence);
        ResourceCommandEffects *prior=resource_effects_enter(d.commands);
        if(stale) live_mapping.mapping++;
        if(overlap) {
            resource_image_prepare_mapped(99,captured.address,captured.range.size,&captured,1);
            resource_image_written(99,captured.address);
        }
        vk_texture_image_write_prepare(image,captured.address,captured.range.size);
        if (nested) {
            // vk_dlaa_scene_upscale's late target has an actual image snapshot and no guest tex_t.
            const VkExecTarget target={.image=image,.format=output.pub.format,.width=output.pub.width,
                .height=output.pub.height,.img=&d.fallback->output};
            evaluate_copy(&ex,&target,&d.fallback->blit);
        } else {
            const VkImageBlit box={.dstSubresource={1,0,0,1},.dstOffsets={{0,0,0},{3440,1440,1}}};
            dlaa_compute_blit_written(&d.fallback->blit,&box);
        }
        resource_effects_leave(prior);
    }
    assert(nested_blits==(unsigned)(nested&&!canceled));
    resource_effects_recorded(d.commands,0);
    resource_effects_release(d.commands);
    vk_mem_image_release(d.fallback->blit.hold);
    free(d.fallback);
    RcRegistryNativeImageFragment piece;
    RcRegistryNativeImageFailure receipt;
    uint32_t count=0;
    RcInterval demand=live_mapping;demand.range.size=0x127a800;
    RcRegistryResult result=rc_registry_native_image_fragments_checked(registry,&demand,1,read,&piece,1,&count,&receipt);
    if(canceled||stale) assert(result!=RC_REGISTRY_OK&&count==0&&projection==RC_REGISTRY_INVALID);
    else if(success) assert(mapped_effect&&projection==RC_REGISTRY_OK&&result==RC_REGISTRY_OK&&count==1&&
        piece.binding.image==image&&piece.binding.hold.id==hold.id&&piece.output.ordinal==write.ordinal);
    else assert(!mapped_effect&&projection==RC_REGISTRY_OK&&result!=RC_REGISTRY_OK&&!count&&
        receipt.gate==RC_NATIVE_IMAGE_READ_TENTATIVE&&receipt.selected.image==image&&
        receipt.selected_output.ordinal==write.ordinal);
    if(abandon){
        while(queued){unsigned i=--queued;queue[i].cancel(queue[i].arg,0);}
        result=rc_registry_native_image_fragments_checked(registry,&demand,1,read,&piece,1,&count,&receipt);
        assert(result!=RC_REGISTRY_OK&&!count);
    }else while(queued){unsigned i=--queued;queue[i].cancel(queue[i].arg,canceled?0:1);}
    assert(!color_holds);
    if(hold.id) assert(rc_registry_image_hold_release_extents(registry,&hold,1,0)==RC_REGISTRY_OK);
    rc_registry_destroy(registry);slots[1]=NULL;owners=NULL;
    return raw_attempts;
}
int main(void){
    assert(run_case(0,0,0,1,0,0,0)==(CORRECTED?0:1));
    assert(run_case(1,0,0,CORRECTED,0,0,0)==(CORRECTED?0:1));
    if(CORRECTED) {
        assert(!run_case(1,1,0,0,0,0,0));
        assert(!run_case(1,0,1,0,0,0,0));
        assert(!run_case(1,0,0,1,1,0,0));
        assert(!run_case(1,0,0,!NESTED_BASELINE,0,0,1));
        if (NESTED_BASELINE) puts("red: late blit left the exact prepared owner unrecorded in its inner scope");
        else {
            assert(!run_case(1,1,0,0,0,0,1));
            assert(!run_case(1,0,1,0,0,0,1));
            assert(!run_case(1,0,0,1,1,0,1));
            assert(run_case(0,0,0,0,0,1,1)==1);
            puts("green: late blit confirmed its exact prepared owner in the inner scope");
        }
    }
    assert(run_case(0,0,0,0,0,1,0)==(CORRECTED?1:2));
    puts(CORRECTED?"captured image declaration, raw refusal, remap, overlapping border and cancel passed":"red: retained projection is tentative if raw writer placement refuses");
}
'''
compiler = os.environ.get("CC") or shutil.which("clang")
if not compiler:
    raise SystemExit("Clang is required (set CC)")
cc = Path(shutil.which(compiler) or compiler).resolve()
compiler_env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
label = "green" if corrected else "red"
body = "\n".join(function(draw, name) for name in
                 ("fallback_destination_locked", "effect_declare_host", "effect_emit", "effect_tensor_outputs"))
source = here / (label + ".c")
source.write_text(prefix + body + tests, encoding="utf-8")
exe = source.with_suffix(".exe" if os.name == "nt" else "")
subprocess.run(
    [str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
     "-Wno-unused-variable", "-Wno-unused-but-set-global", "-fsanitize=undefined", "-fsanitize-trap=all",
     "-DRC_REGISTRY_SINGLE_THREADED", "-DCORRECTED=" + str(int(corrected)),
     "-DNESTED_BASELINE=" + str(int(opts.nested_baseline)), "-I", str(runtime),
     str(source), str(runtime / "resource_core_registry.c"), "-o", str(exe)],
    env=compiler_env, check=True)
subprocess.run([str(exe)], env=compiler_env, check=True)
