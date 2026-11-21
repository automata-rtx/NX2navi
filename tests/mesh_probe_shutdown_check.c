
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "platform/platform.h"
#include "vk.h"

typedef struct {
    uint64_t fence;
    void (*fn)(void *);
    void *arg;
    void (*cancel)(void *, int);
} Pending;

struct VkGraphicsPipelineRevision {
    VkPipeline pipeline;
    uint32_t users;
    uint64_t last_scene;
    int retired, legacy;
};

static struct { PlatMutex lock; VkDevice device; } P;
static PlatMutex st_lock;
static PlatMutex *held_lock;
static VkGraphicsPipelineRevision revision;
static Pending pending;
static int queued, phase, loss_at_sync;
static uint64_t scene=21, completed=20, handed=20;
volatile int32_t vk_lost;
void plat_mutex_lock(PlatMutex *lock) { assert(!held_lock); held_lock=lock; }
void plat_mutex_unlock(PlatMutex *lock) { assert(held_lock==lock); held_lock=NULL; }
_Noreturn void rt_fatal(const char *s, ...) { fprintf(stderr,"%s\n",s); abort(); }
VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(VkDevice dev,VkPipeline pipe,const VkAllocationCallbacks *alloc) {
    (void)dev; (void)pipe; (void)alloc; abort();
}
void vk_completion_add_owned(uint64_t fence,void (*fn)(void*),void (*cancel)(void*,int),void *arg) {
    assert(fence==scene && fn && cancel && arg==&revision && !queued);
    pending=(Pending){fence,fn,arg,cancel};
    queued=1;
}

#include "meshfb-owners.inc"

static void probe_finish(void) {
    assert(!phase++);
    assert(queued && revision.users); // Only issued values completed here.
}
void vk_pipeline_stop_preparation(void) { assert(phase++==1); }
void vk_exec_draw_mfb_stop(void) { assert(phase++==2); }
void vk_sync_shutdown(void) {
    assert(phase++==3 && revision.users && queued);
    vk_lost=loss_at_sync;
    Pending *queue=&pending;
    int head=0;
    pending_dispose_queue(&queue,&head,&queued,completed,handed,!vk_lost);
    assert(queued==loss_at_sync && revision.users==(uint32_t)loss_at_sync);
}
void vk_pipeline_shutdown(void) {
    if (revision.users) rt_fatal("probe reached pipeline destruction with an outstanding scene revision");
    assert(phase++==4 && !queued && !vk_lost);
}
void vk_exec_draw_shutdown(void) { assert(phase++==5 && !queued && !vk_lost); }
void vk_exec_xfer_shutdown(void) { assert(phase++==6 && !queued && !vk_lost); }
void vk_backend_shutdown(void) { assert(phase++==(loss_at_sync?4:7)); }

#include "meshfb-finish.inc"

static void check(int lose) {
    loss_at_sync=lose;
    phase=queued=vk_lost=0;
    handed=lose ? scene : completed;
    VkPipeInfo info={0};
    VkPipeline pipeline=(VkPipeline)(uintptr_t)1;
    revision=(VkGraphicsPipelineRevision){.pipeline=pipeline,.users=1};
    VkGraphicsPipelineUse use={.info=&info,.pipeline=pipeline,.revision=&revision};
    vk_pipeline_graphics_record(&use,scene);
    assert(queued && revision.users==1 && !use.revision);
    const GfxBackend b={.finish=probe_finish};
    meshfb_finish_probe(&b);
    assert(phase==(lose?5:8) && queued==lose && revision.users==(uint32_t)lose);
}
int main(void) {
    check(0);
    check(1);
    puts("mesh probe shutdown: known-unsent revision retired before pipeline, uncertain loss owner retained");
}
