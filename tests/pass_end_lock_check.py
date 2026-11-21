"""Check the production pass-end lock order and saved target identities without Vulkan."""
from pathlib import Path
import argparse
import os
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_content import GENERATED_MARKER


def function(source, name):
    match = re.search(r"^(?:static )?[^\n]*\b" + name + r"\([^;]*?\)\s*\{", source, re.M)
    assert match, name
    begin = source.index("{", match.start())
    depth, end = 1, begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


PREFIX = r'''
#include <windows.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "vk_target_frames.h"
typedef uint64_t VkImage, VkCommandBuffer;
#define VK_NULL_HANDLE 0
typedef struct { VkImage image; uint64_t addr; } VkTexImage;
typedef struct { struct { uint64_t size; } L; } tex_t;
typedef struct { const tex_t *pin; VkImage image; } VkPinnedTextureWrite;
typedef struct { void *extents; VkPinnedTextureWrite pinned; } VkExecTargetBinding;
typedef struct { VkImage image; uint32_t level,base_layer; VkTexImage *img; const tex_t *tex; int exact_snapshot; } VkExecTarget;
typedef struct { uint32_t draws; uint64_t last[2]; } Mark;
typedef struct {
    int in_pass,rt_resolved; uint32_t pass_ncolor,recorded,target_binding_mask,bc_target_mask,bc_written_mask; uint64_t fence;
    VkExecTargetBinding target_bindings[9];
    VkCommandBuffer cb; struct { uint32_t width; } tonemap_extent;
    Mark *mark; uint64_t mark_draws0; VkExecTarget color[8],depth;
} VkExec;
typedef struct { HANDLE handle; int target; } PlatMutex;
static PlatMutex tfr_lock,tex_lock;
static HANDLE texture_held,other_notifying;
static volatile LONG waits_for_tfr,waits_for_tex;
static _Thread_local int who,tfr_depth;
static _Thread_local VkExec *current;
static unsigned notifications[3],prepares[3],ended[3],copied[3],queries[3],dumps[3],tonemaps[3],locks[3];
static unsigned notifications_under_history[3];
static VkImage notified_image[3][16];static uint64_t notified_addr[3][16];
static int parallel,mutate_after_snapshot;
static struct {uint64_t frames;} gfx_counters;
static struct {uint64_t draws;} vk_exec_frame;
static int vk_dump_armed;
void vk_exec_pass_end(VkExec *ex);
static void plat_mutex_lock(PlatMutex *m) {
    if (parallel && who==1 && m==&tfr_lock) InterlockedExchange(&waits_for_tfr,1);
    if (parallel && who==2 && m==&tex_lock) InterlockedExchange(&waits_for_tex,1);
    assert(WaitForSingleObject(m->handle,INFINITE)==WAIT_OBJECT_0);
    if(m==&tfr_lock){tfr_depth++;locks[who]++;}
}
static void plat_mutex_unlock(PlatMutex *m) {
    if(m==&tfr_lock){assert(tfr_depth==1);tfr_depth--;}
    assert(ReleaseMutex(m->handle));
}
static int vk_exec_shared(void){return parallel;}
#define RESOURCE_IMAGE_SCOPE(name,fence) ((void)(fence))
static void vk_texture_image_write_prepare(VkImage image,uint64_t addr,uint64_t size) {
    assert(image&&addr&&size);prepares[who]++;
}
#define resource_image_prepare vk_texture_image_write_prepare
static void resource_image_written(uint64_t image,uint64_t addr) {
    unsigned n=notifications[who];assert(n<16&&image&&addr);
    notified_image[who][n]=image;notified_addr[who][n]=addr;
}
static void vk_texture_storage_written(VkImage image,uint64_t addr) {
    assert(current&&!current->in_pass&&ended[who]);
    if(tfr_depth)notifications_under_history[who]++;
    if(parallel&&who==2&&!notifications[who])assert(SetEvent(other_notifying));
    plat_mutex_lock(&tex_lock);
    unsigned n=notifications[who]++;
    assert(notified_image[who][n]==image&&notified_addr[who][n]==addr);
    // Texture initialization can reenter the same execution context after its rendering was ended.
    vk_exec_pass_end(current);
    if(mutate_after_snapshot&&!n){
        current->color[1].image=999;current->color[1].img->image=998;
        current->color[1].img->addr=997;current->color[1].level=99;
        current->depth.image=996;current->depth.img->image=995;current->depth.img->addr=994;
    }
    plat_mutex_unlock(&tex_lock);
}
static int vk_texture_render_target_written(VkImage image,uint64_t addr) {
    (void)image;(void)addr;assert(0);return 0;
}
static void vk_texture_pinned_write_release(VkPinnedTextureWrite *capture) {
    assert(!capture->pin);*capture=(VkPinnedTextureWrite){0};
}
static void vk_exec_oq_pass_end(VkExec *ex){assert(ex==current&&ex->in_pass);queries[who]++;}
static void vkCmdEndRendering(VkCommandBuffer cb){assert(current&&current->cb==cb);ended[who]++;}
static void vk_dlaa_tonemap_end(VkExec *ex){assert(ex==current&&!ex->in_pass);tonemaps[who]++;}
static void vk_exec_draw_shaders(uint64_t *a,uint64_t *b){*a=17;*b=18;}
static void vk_pass_mark_end(VkCommandBuffer cb,Mark *mark){assert(cb==current->cb&&mark==current->mark);}
static void test_bc_copy_back(VkExec *ex,const VkExecTarget *target){
    assert(ex==current&&!ex->in_pass&&target->img&&target->image!=target->img->image);copied[who]++;
    if (ACTUAL_ATTACHMENT_WRITERS) {
        vk_texture_image_write_prepare(target->img->image,target->img->addr,target->tex->L.size);
        resource_image_written(target->img->image,target->img->addr);
        vk_texture_storage_written(target->img->image,target->img->addr);
    }
}
#define bc_copy_back(ex, target, ...) test_bc_copy_back(ex, target)
static void vk_dump_pass_end(VkExec *ex){assert(ex==current&&!ex->in_pass);dumps[who]++;}
'''

CASES = r'''
static VkTexImage owners[3][4];static tex_t texture={.L.size=4096};
static VkExec execution[3];
static VkExec setup(int index,uint64_t base) {
    for(int i=0;i<4;i++)owners[index][i]=(VkTexImage){base+i,0x10000+base+i};
    VkExec ex={.in_pass=1,.pass_ncolor=2,.target_binding_mask=0x101,.bc_target_mask=2,.bc_written_mask=2,
        .fence=77,.cb=50+(uint64_t)index};
    ex.color[0]=(VkExecTarget){base,1,2,&owners[index][0],&texture,0};
    ex.color[1]=(VkExecTarget){base+10,3,4,&owners[index][1],&texture,0}; // BC stand-in owns another image.
    ex.color[6]=(VkExecTarget){base+20,9,10,NULL,NULL,0}; // Backend-only target still has frame history.
    ex.color[7]=(VkExecTarget){base+2,5,6,&owners[index][2],&texture,0}; // Preserve the existing all-color history rule.
    ex.depth=(VkExecTarget){base+3,7,8,&owners[index][3],&texture,0};
    return ex;
}
static void check_frame(uint64_t image,uint32_t level,uint32_t layer,uint64_t owner,uint64_t stamp) {
    TfrTarget t={image,level,layer,owner};const TargetFrames *f=tfr_find(&t,0);
    assert(f&&f->image==image&&f->level==level&&f->layer==layer&&f->owner==owner&&f->last==stamp&&f->n==1);
}
static void check_execution(int index,uint64_t base) {
    VkExec *ex=&execution[index];assert(!ex->in_pass&&ex->recorded==1&&!ex->target_binding_mask);
    assert(!notifications_under_history[index]);
    assert(ended[index]==1&&queries[index]==1&&copied[index]==1);
    if (ACTUAL_ATTACHMENT_WRITERS) {
        assert(prepares[index]==1&&notifications[index]==1);
        assert(notified_image[index][0]==base+1&&notified_addr[index][0]==0x10000+base+1);
    } else {
        assert(prepares[index]==3&&notifications[index]==4);
        for(unsigned i=0;i<4;i++)assert(notified_image[index][i]==base+i&&notified_addr[index][i]==0x10000+base+i);
    }
    check_frame(base,1,2,0x10000+base,42);
    check_frame(base+10,3,4,0x10000+base+1,42);
    check_frame(base+2,5,6,0x10000+base+2,42);
    check_frame(base+3,7,8,0x10000+base+3,42);
    check_frame(base+20,9,10,0,42);
}
static DWORD WINAPI texture_owner(void *unused) {
    (void)unused;who=1;current=&execution[1];plat_mutex_lock(&tex_lock);
    assert(SetEvent(texture_held));assert(WaitForSingleObject(other_notifying,2000)==WAIT_OBJECT_0);
    vk_exec_pass_end(current);plat_mutex_unlock(&tex_lock);return 0;
}
static DWORD WINAPI other_pass(void *unused) {
    (void)unused;who=2;current=&execution[2];
    assert(WaitForSingleObject(texture_held,2000)==WAIT_OBJECT_0);vk_exec_pass_end(current);return 0;
}
int main(void) {
    tfr_lock=(PlatMutex){CreateMutex(NULL,FALSE,NULL),1};tex_lock=(PlatMutex){CreateMutex(NULL,FALSE,NULL),0};
    texture_held=CreateEvent(NULL,TRUE,FALSE,NULL);other_notifying=CreateEvent(NULL,TRUE,FALSE,NULL);
    assert(tfr_lock.handle&&tex_lock.handle&&texture_held&&other_notifying);
    gfx_counters.frames=41;parallel=1;execution[1]=setup(1,100);execution[2]=setup(2,200);
    HANDLE threads[2]={CreateThread(NULL,0,texture_owner,NULL,0,NULL),CreateThread(NULL,0,other_pass,NULL,0,NULL)};
    assert(threads[0]&&threads[1]);DWORD result=WaitForMultipleObjects(2,threads,TRUE,3000);
    if(result==WAIT_TIMEOUT&&InterlockedCompareExchange(&waits_for_tfr,0,0)&&InterlockedCompareExchange(&waits_for_tex,0,0)) {
        fputs("pass end: reproduced texture/target-history ABBA; both lock attempts remain blocked\n",stderr);
        return 86; // Process exit disposes the deliberately blocked negative-control threads.
    }
    assert(result==WAIT_OBJECT_0);check_execution(1,100);check_execution(2,200);
    assert(locks[1]==1&&locks[2]==1);
    for(unsigned i=0;i<2;i++){DWORD code;assert(GetExitCodeThread(threads[i],&code)&&!code);CloseHandle(threads[i]);}

    // Nonparallel callbacks may mutate live texture metadata. The entire pass's history and notification identity
    // must have been copied first, rather than taking later color/depth identities from already-mutated targets.
    parallel=0;who=0;execution[0]=setup(0,300);current=&execution[0];
    mutate_after_snapshot=!ACTUAL_ATTACHMENT_WRITERS;
    Mark mark={0};current->mark=&mark;current->mark_draws0=9;vk_exec_frame.draws=12;
    current->tonemap_extent.width=64;vk_dump_armed=1;
    vk_exec_pass_end(current);check_execution(0,300);
    assert(!locks[0]&&!current->mark&&mark.draws==3&&mark.last[0]==17&&mark.last[1]==18);
    assert(tonemaps[0]==1&&dumps[0]==1);
    vk_exec_pass_end(current);check_execution(0,300); // An ended pass cannot notify or record a second time.
    CloseHandle(texture_held);CloseHandle(other_notifying);CloseHandle(tfr_lock.handle);CloseHandle(tex_lock.handle);
    puts("pass end: concurrent texture reentry completes; actual copy-back and target history remain ordered");
    return 0;
}
'''


def fixture(source):
    names = ["resource_target_prepare", "resource_target_written", "tfr_of"]
    if "static void target_frames_written(" in source:
        names.append("target_frames_written")
    if "static void resource_targets_forget(" in source:
        names.append("resource_targets_forget")
    names.append("vk_exec_pass_end")
    actual = int("vk_texture_render_target_written(" in source)
    return f"#define ACTUAL_ATTACHMENT_WRITERS {actual}\n" + PREFIX + "\n".join(
        function(source, name) for name in names) + CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-source", type=Path, help="Archived source zip whose old pass-end must reproduce ABBA")
    args = parser.parse_args()
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    work = ROOT / "build/test"
    work.mkdir(parents=True, exist_ok=True)
    work.resolve().relative_to(ROOT.resolve())
    sources = [("current", (ROOT / "runtime/vk_exec.c").read_text(), 0)]
    if args.negative_source:
        with zipfile.ZipFile(args.negative_source) as archive:
            paths = [name for name in archive.namelist() if name.replace("\\", "/").endswith("runtime/vk_exec.c")]
            assert len(paths) == 1, paths
            sources.append(("negative", archive.read(paths[0]).decode(), 86))
    with tempfile.TemporaryDirectory(prefix="pass-end-lock-", dir=work) as temporary:
        for label, source, expected in sources:
            path, exe = Path(temporary) / (label + ".c"), Path(temporary) / (label + ".exe")
            marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
            path.write_text("// " + marker + "\n" + fixture(source))
            subprocess.run([str(cc), "-std=gnu11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                            "-Wno-unused-function", "-Wno-sign-compare", "-fsanitize=undefined", "-fsanitize-trap=all",
                            "-I", str(ROOT / "runtime"), str(path), "-o", str(exe)], env=env, check=True)
            result = subprocess.run([str(exe)], env=env, timeout=8)
            assert result.returncode == expected, (label, result.returncode, expected)
            if expected:
                print("Archived pass-end negative control reproduced the bounded lock cycle (exit 86).")


if __name__ == "__main__":
    main()
