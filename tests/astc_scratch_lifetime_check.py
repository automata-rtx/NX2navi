"""Run the actual ASTC scratch replacement policy against recorded upload identities."""
from pathlib import Path
import os
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "runtime/vk_astc.c").read_text()
start = source.index("static VkBuffer scratch_get(")
begin = source.index("{", start)
depth, end = 1, begin + 1
while depth:
    depth += (source[end] == "{") - (source[end] == "}")
    end += 1
body = source[start:end]
owner = re.search(r"static struct \{ VkBuffer buf;[^\n]+\} S;", source).group()
prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef uint64_t VkBuffer,VmaAllocation,VkDeviceSize,VkCommandBuffer,VkDevice;
typedef int VkResult,RcAdmissionStatus;
typedef struct {VkDevice device;int vma;} VkCtx;
typedef struct {int sType;uint64_t size;int usage;} VkBufferCreateInfo;
typedef struct {int usage;} VmaAllocationCreateInfo;
typedef struct {int producer,required;} RcVmaRequest;
enum {VK_SUCCESS=0,VK_ERROR_OUT_OF_DEVICE_MEMORY=-1,VK_ERROR_OUT_OF_HOST_MEMORY=-2,VK_NOT_READY=1,
 VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO=2,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT=4,VK_BUFFER_USAGE_TRANSFER_SRC_BIT=8,
 VK_BUFFER_USAGE_TRANSFER_DST_BIT=16,VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE=1,RC_BUDGET_TEXTURE=1,VK_TS_SCRATCH=1};
#define VK_NULL_HANDLE 0
#define MB (1024ull*1024ull)
#define SCRATCH_MIN (32*MB)
#define SCRATCH_SHRINK_NS 5000000000ull
static int keep,g_lock,depth,fail_allocations;static uint64_t current_seq=1,current_cb=1,clock_ns=1,next_buffer;
static uint64_t n_scratch_new,n_scratch_reuse;static volatile int32_t scratch_fail_n;
static struct {VkBuffer buffer;uint64_t seq;} retired[16];static unsigned nretired;
static int scratch_keep(void){return keep;}
static uint64_t vk_upload_recording_sequence(VkCommandBuffer cb){assert(!depth);return cb==current_cb?current_seq:0;}
static int plat_cas32(volatile int32_t*p,int a,int b){(void)a;(void)b;return *p;}
static int plat_add32(volatile int32_t*p,int n){return *p+=n;}
static int plat_inc32(volatile int32_t*p){return ++*p;}
static void plat_mutex_lock(int*p){(void)p;assert(!depth);depth++;}
static void plat_mutex_unlock(int*p){(void)p;assert(depth==1);depth--;}
static uint64_t kern_now_ns(void){return clock_ns;}
static void rt_fatal(const char*s){fputs(s,stderr);abort();}
static void rt_log(const char*s,...){(void)s;}
static void vk_texstat_time(int n,uint64_t t){(void)n;(void)t;}
static VkResult resource_core_vma_create_buffer(const RcVmaRequest*r,int v,const VkBufferCreateInfo*b,
 const VmaAllocationCreateInfo*a,VkBuffer*out,VmaAllocation*memory,void*info,RcAdmissionStatus*admission){
 (void)r;(void)v;(void)b;(void)a;(void)info;*admission=0;
 if(fail_allocations){fail_allocations--;return VK_NOT_READY;}*out=++next_buffer;*memory=*out;return VK_SUCCESS;
}
static void vk_upload_defer_buffer(VkBuffer buffer,VmaAllocation allocation,uint64_t sequence){
 assert(depth==1&&buffer==allocation&&sequence&&sequence<=current_seq);
 retired[nretired++]=(typeof(retired[0])){buffer,sequence};
}
'''
tests = r'''
int main(void){
 VkCtx c={1,1};int reused;
 VkBuffer first=scratch_get(&c,current_cb,8*MB,&reused);assert(first&&!reused&&S.last_seq==1);
 assert(scratch_get(&c,current_cb,8*MB,&reused)==first&&reused&&!nretired);
 // Per-batch replacement owns exactly its old upload, not the scene or the newly opened batch.
 current_cb=2;current_seq=2;VkBuffer second=scratch_get(&c,current_cb,8*MB,&reused);
 assert(second!=first&&!reused&&nretired==1&&retired[0].buffer==first&&retired[0].seq==1);
 VkBuffer larger=scratch_get(&c,current_cb,80*MB,&reused);
 assert(larger!=second&&nretired==2&&retired[1].seq==2&&S.last_seq==2);
 // Shared reuse updates the last-use identity even though the physical allocation stays the same.
 keep=1;current_cb=3;current_seq=3;assert(scratch_get(&c,current_cb,64*MB,&reused)==larger&&reused&&S.last_seq==3);
 current_cb=4;current_seq=4;fail_allocations=2;
 assert(!scratch_get(&c,current_cb,150*MB,&reused)&&nretired==2&&S.buf==larger&&S.last_seq==3);
 VkBuffer grown=scratch_get(&c,current_cb,150*MB,&reused);
 assert(grown!=larger&&nretired==3&&retired[2].buffer==larger&&retired[2].seq==3&&S.last_seq==4);
 current_cb=5;current_seq=5;clock_ns+=SCRATCH_SHRINK_NS+1;
 assert(scratch_get(&c,current_cb,8*MB,&reused)!=grown&&nretired==4&&retired[3].seq==4&&S.size==SCRATCH_MIN);
 assert(n_scratch_new==5&&n_scratch_reuse==2&&!depth);
 puts("ASTC scratch: per-batch replacement, shared reuse, same-batch growth, failed growth and shrink retain exact upload identity");
}
'''
cc = Path(os.environ.get("CC", "C:/Users/Andrew/AppData/Local/Programs/2NX2Navi/toolchain/msys64/ucrt64/bin/clang.exe"))
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="astc-scratch-lifetime-", dir=ROOT / "build/test") as tmp:
    p = Path(tmp) / "check.c"
    p.write_text(prefix + owner + body + tests)
    exe = p.with_suffix(".exe")
    subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-fsanitize-trap=all", str(p), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)
