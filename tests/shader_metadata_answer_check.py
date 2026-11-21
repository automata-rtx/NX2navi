"""Run the production specialization answer bridge without content preparation or a GPU."""
from pathlib import Path
import os, re, subprocess, tempfile
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = (ROOT / "runtime/vk_exec_draw.c").read_text()
start = re.search(r"^static int answer_of_handle\(", source, re.M).start()
body = function(source[start:], "answer_of_handle")
prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
typedef unsigned VkFormat;
#define VK_FORMAT_UNDEFINED 0
#define NVN_HANDLE_TEX(h) ((int)((h)&0xfffff))
typedef struct {unsigned view_type,samples,format;uint64_t image;} VkTexImage;
typedef struct {unsigned set_format,format;} view_t;
static uint64_t effect_metadata_unavailable;
static VkTexImage metadata={3,1,4,0},actual={6,1,4,900};
static view_t selected={1,4};
static int result=1,deferred,fallback_deferred,fallback_absent;
static unsigned metadata_calls,content_calls;
static int vk_texture_metadata_locked(int id,VkTexImage*out,view_t*v,int*hv){
 assert(id==7);metadata_calls++;*out=metadata;*v=selected;*hv=1;return result;
}
static const VkTexImage *vk_texture_current_registration_locked(int id,view_t*v,int*hv){
 assert(id==7);content_calls++;*v=selected;*hv=1;deferred=fallback_deferred;
 return fallback_absent?NULL:&actual;
}
static int vk_texture_deferred(void){return deferred;}
static int gfx_type_of(const VkTexImage*image,const view_t*v,int hv){(void)v;(void)hv;return image->view_type;}
static int int_view_format(unsigned format,VkFormat*out){if(out)*out=format;return format;}
static unsigned texel_bytes(VkFormat format){return format;}
enum ShaderAnswerResource { SHADER_ANSWER_TEXTURE, SHADER_ANSWER_IMAGE, SHADER_ANSWER_COMPUTE_IMAGE };
static uint8_t storage_image_format_answer(const VkTexImage *image,const view_t *view,int hv,int compute) {
    (void)compute;
    VkFormat format;
    return hv && view->set_format && int_view_format(view->format,&format) && texel_bytes(format) &&
        texel_bytes(format)==texel_bytes(image->format) ? (uint8_t)view->format : 0;
}
static void rt_log(const char*format,...){(void)format;}
'''
tests = r'''
int main(void){
 uint8_t type=99,format=99;
 assert(answer_of_handle(0,1,&type,&format,0)&&!type&&!format&&!metadata_calls&&!content_calls);
 assert(answer_of_handle(7,1,&type,&format,0)&&type==3&&format==4&&!content_calls);
 deferred=1;assert(answer_of_handle(7,0,&type,&format,0)&&type==3&&!format&&deferred&&!content_calls);
 deferred=0;result=0;assert(answer_of_handle(7,1,&type,&format,0)&&!type&&!format&&!content_calls);
 result=1;selected.format=8;assert(answer_of_handle(7,1,&type,&format,0)&&type==3&&!format&&!content_calls);
 selected.format=4;result=-1;assert(answer_of_handle(7,1,&type,&format,0)&&type==6&&format==4&&content_calls==1);
 fallback_deferred=1;assert(!answer_of_handle(7,1,&type,&format,0)&&deferred&&content_calls==2);
 fallback_deferred=0;fallback_absent=1;assert(answer_of_handle(7,1,&type,&format,0)&&!type&&!format&&content_calls==3);
 assert(effect_metadata_unavailable==3);
 puts("shader metadata answers: known and absent avoid content; ambiguous uses compatibility; deferred never publishes");
}
'''
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="shader-metadata-answer-", dir=ROOT / "build/test") as tmp:
    path = Path(tmp) / "check.c"
    path.write_text(prefix + body + tests)
    exe = path.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", str(path), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)
