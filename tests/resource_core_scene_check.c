#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
typedef uint64_t VkImage;
#define VK_NULL_HANDLE 0
typedef void *VmaAllocation;
typedef struct { uint32_t width, height; } ImgKey;
#include "../runtime/vk_texture_material.h"
typedef struct { TextureMaterial *material; struct { VkImage image; } pub; } Tex;
typedef struct { uint64_t serial; } tex_t;
static const Tex pinned_owner = {.pub.image = 8001};
static const tex_t pinned_texture = {77};
static unsigned pinned_refs, pinned_released;
static uint64_t recording_fence;
static int locked;
static void lock(void) { assert(!locked); locked = 1; }
static void unlock(void) { assert(locked); locked = 0; }
// The scene test supplies a fake memory owner. The real publication and recording code below decide
// when to acquire, transfer and release this physical hold, independent of the tested material owners.
typedef struct { VkImage image; unsigned refs; int retired; } VkMemoryImageHold;
static VkMemoryImageHold native_owner = {.image = 8001};
static VkMemoryImageHold *vk_mem_image_hold(VkImage image) {
    assert(locked && image == native_owner.image && !native_owner.retired);
    native_owner.refs++;
    return &native_owner;
}
static VkMemoryImageHold *vk_mem_image_hold_retain(VkMemoryImageHold *hold) {
    assert(locked && hold == &native_owner && hold->refs);
    hold->refs++;
    return hold;
}
static VkMemoryImageHold *vk_mem_image_hold_retain_exact(VkMemoryImageHold *hold, VkImage image) {
    if (!hold || hold->image != image) return NULL;
    return vk_mem_image_hold_retain(hold);
}
static int vk_mem_image_hold_retired(const VkMemoryImageHold *hold) {
    assert(locked && hold == &native_owner && hold->refs);
    return hold->retired;
}
static void vk_mem_image_release(VkMemoryImageHold *hold) {
    assert(locked && hold == &native_owner && hold->refs);
    hold->refs--;
}
// The fixture supplies the alias's own refcount; the real publication and scene code choose its lifetime.
typedef struct { unsigned refs; } VkTextureAliasRead;
static VkTextureAliasRead *vk_texture_alias_read_retain(VkTextureAliasRead *alias) {
    if (alias) { assert(locked && alias->refs); alias->refs++; }
    return alias;
}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *alias) {
    assert(locked && alias && alias->refs);
    alias->refs--;
}
static uint64_t vk_exec_recording_fence(void) { return recording_fence; }
static void content_collect(uint64_t limit) { (void)limit; assert(locked); }
static const tex_t *nvn_texture_pin_retain_attachment(uint64_t serial, const void *owner) {
    assert(locked && owner == &pinned_owner && serial == pinned_texture.serial);
    pinned_refs++;
    return &pinned_texture;
}
static void nvn_texture_pin_release(const tex_t *pin) {
    assert(locked && pin == &pinned_texture && pinned_refs);
    pinned_refs--;
    pinned_released++;
}
__attribute__((noreturn)) static void rt_fatal(const char *format, ...) {
    va_list args; va_start(args, format); vfprintf(stderr, format, args); va_end(args); abort();
}
#include "../runtime/replay_work_trace.h"
static uint64_t publication_allocations, publication_frees;
static void *publication_calloc(size_t count, size_t size) {
    publication_allocations++; return calloc(count, size);
}
static void publication_free(void *p) { publication_frees += p != NULL; free(p); }
#define calloc publication_calloc
#define free publication_free
#include "../runtime/vk_texture_uses.h"
#undef calloc
#undef free

static void init(TextureMaterial *m, TextureMaterialLedger *ledger, VkImage image, uint64_t upload) {
    ImgKey key = {16, 16};
    assert(texture_material_init(m, ledger, image, (void *)(uintptr_t)image, &key, 1024, 1));
    texture_material_set_upload(m, upload); m->ready = 1;
    lock(); content_use_image_add(m->resident.instance); unlock();
}
static void destroy(TextureMaterialInstance *instance, TextureMaterial *expected) {
    lock();
    assert(texture_material_instance_defer_ready(instance));
    content_use_image_remove(instance);
    texture_material_instance_defer_begin(instance);
    texture_material_destroyed(instance);
    assert(texture_material_instance_reap(instance) == expected);
    unlock();
}
static TextureMaterialInstance *retire(TextureMaterial *m) {
    assert(texture_material_release(m, 1));
    return texture_material_retire(m).instance;
}

static void submitted_split_and_cached_publication(void) {
    TextureMaterialLedger ledger = {0}; TextureMaterial material;
    init(&material, &ledger, 101, 7);
    VkImage images[] = {101, 101, 999}; // Unknown images are owned by the legacy/nonordinary adapter.
    TextureImagePublication *publication = vk_texture_publication_create(images, 3, NULL);
    assert(!vk_texture_publication_has_retired(NULL));
    assert(!vk_texture_publication_has_retired(publication));
    TextureMaterialInstance *instance = material.resident.instance;
    assert(instance->lifetime.captures == 1);
    recording_fence = 20;
    vk_texture_publication_use(publication);
    vk_texture_publication_use(publication);
    assert(instance->lifetime.captures == 2);
    TextureUseBatch *first = vk_texture_uses_seal(20, 7);
    assert(first && first->handover == TEXTURE_BATCH_SEALED);
    assert(first->head->use.capture.state == RC_LIFETIME_CAPTURE_RECORDING);
    assert(retire(&material) == instance && !texture_material_instance_defer_ready(instance));
    assert(vk_texture_publication_has_retired(publication));

    // A second recorded part with the same final fence needs its own hand-over. Reusing the set retains its exact
    // detached instance, rather than consulting the material's now-empty current slot.
    vk_texture_publication_use(publication);
    TextureUseBatch *second = vk_texture_uses_seal(20, 7);
    assert(second && second != first && instance->lifetime.captures == 3);
    vk_texture_uses_handed(first);
    vk_texture_uses_complete(19, 7);
    assert(instance->lifetime.captures == 3);
    vk_texture_uses_abandon(second);
    vk_texture_uses_complete(20, 7);
    assert(instance->lifetime.captures == 1 && !texture_uses.batches);

    // Rebuilding a cached set transfers detached image ownership before releasing the previous publication.
    TextureImagePublication *replacement = vk_texture_publication_create(images, 3, publication);
    vk_texture_publication_release(publication);
    assert(instance->lifetime.captures == 1);
    // A different cached set can adopt the same detached image through the adapter's surviving exact hold.
    TextureImagePublication *other_set = vk_texture_publication_create(images, 3, NULL);
    vk_texture_publication_release(replacement);
    replacement = other_set;
    assert(instance->lifetime.captures == 1);
    recording_fence = 21;
    vk_texture_publication_use(replacement);
    TextureUseBatch *last = vk_texture_uses_seal(21, 7);
    assert(vk_texture_publication_has_retired(replacement) && instance->lifetime.captures == 2);
    vk_texture_publication_release(replacement);
    assert(instance->lifetime.captures == 1 && !texture_material_instance_defer_ready(instance));
    vk_texture_uses_handed(last);
    vk_texture_uses_complete(21, 7);
    assert(!instance->lifetime.captures);
    destroy(instance, &material);
    assert(texture_material_can_free(&material) && !ledger.pending_bytes);
}

static void current_images_cancel_and_shutdown(void) {
    TextureMaterialLedger ledger = {0}; TextureMaterial material;
    init(&material, &ledger, 201, 0);
    TextureMaterialInstance *instance = material.resident.instance;
    Tex e = {.material = &material};
    recording_fence = 30;
    lock(); content_use(&e); content_use(&e); unlock();
    VkImage images[] = {201, 201};
    vk_texture_use_images(images, 2);
    assert(instance->lifetime.captures == 1);
    vk_texture_uses_cancel(30);
    assert(!instance->lifetime.captures && !texture_uses.batches);
    assert(!vk_texture_uses_seal(30, 0));

    recording_fence = 31;
    vk_texture_use_images(images, 2);
    assert(instance->lifetime.captures == 1);
    vk_texture_uses_shutdown(); // Unsubmitted recordings are canceled, never reported GPU-complete.
    assert(!instance->lifetime.captures);
    assert(retire(&material) == instance);
    destroy(instance, &material);
    recording_fence = 0;
    vk_texture_use_images(images, 2);
    assert(!texture_uses.batches);
}

static void publication_refuses_copied_source_ticket(void) {
    TextureMaterialLedger ledger = {0}; TextureMaterial material;
    init(&material, &ledger, 301, 0);
    TextureMaterialUse owning = {0}, fork = {0};
    assert(texture_material_use_begin(&material, &owning) == RC_LIFETIME_OK);
    TextureMaterialUse copied = owning;
    assert(texture_material_use_fork(&copied, &fork) == RC_LIFETIME_INVALID);
    assert(texture_material_use_fork(&owning, &fork) == RC_LIFETIME_OK);
    assert(texture_material_use_cancel(&owning) == RC_LIFETIME_OK);
    assert(texture_material_use_cancel(&fork) == RC_LIFETIME_OK);
    TextureMaterialInstance *instance = retire(&material);
    destroy(instance, &material);
}
static void descriptor_pool_union_survives_split(void) {
    TextureMaterialLedger ledger = {0}; TextureMaterial first, second;
    init(&first, &ledger, 401, 0); init(&second, &ledger, 402, 0);
    TextureMaterialInstance *a = first.resident.instance, *b = second.resident.instance;
    TextureImagePublication *pool = vk_texture_publication_create(NULL, 0, NULL);
    recording_fence = 40;
    VkImage images[] = {401, 402};
    vk_texture_publication_add_and_use(pool, images, 1);
    vk_texture_publication_add_and_use(pool, images, 2);
    vk_texture_publication_add_and_use(pool, images, 2);
    assert(a->lifetime.captures == 2 && b->lifetime.captures == 2);
    TextureUseBatch *part = vk_texture_uses_seal(40, 0);
    vk_texture_uses_handed(part);
    assert(retire(&first) == a && retire(&second) == b);
    // The whole-binding fast return can reuse its old descriptor pool in a later part without resolving resources.
    vk_texture_publication_use(pool);
    TextureUseBatch *last = vk_texture_uses_seal(40, 0);
    assert(a->lifetime.captures == 3 && b->lifetime.captures == 3);
    vk_texture_uses_handed(last);
    vk_texture_publication_release(pool); // The backend completion callback owns the descriptor pool itself.
    assert(a->lifetime.captures == 2 && b->lifetime.captures == 2);
    vk_texture_uses_complete(40, 0);
    destroy(a, &first); destroy(b, &second);
}
static void idle_drain_and_unsubmitted_loss_cleanup(void) {
    TextureMaterialLedger ledger = {0}; TextureMaterial material;
    init(&material, &ledger, 501, 0);
    TextureMaterialInstance *instance = material.resident.instance;
    VkImage image = 501;
    recording_fence = 50;
    vk_texture_use_images(&image, 1);
    TextureUseBatch *handed = vk_texture_uses_seal(50, 0);
    vk_texture_uses_handed(handed);
    recording_fence = 51;
    vk_texture_use_images(&image, 1);
    TextureUseBatch *held = vk_texture_uses_seal(51, 0);
    vk_texture_uses_abandon(held);
    recording_fence = 52;
    vk_texture_use_images(&image, 1);
    assert(instance->lifetime.captures == 3);
    vk_texture_uses_cancel_unsubmitted(50);
    assert(instance->lifetime.captures == 1 && texture_uses.batches == 1);
    uint64_t observed = texture_uses.completed;
    vk_texture_uses_idle(50, 0); // Real idle covers the handed partial scene without inventing a final signal.
    assert(!instance->lifetime.captures && !texture_uses.batches && texture_uses.completed == observed);
    assert(retire(&material) == instance);
    destroy(instance, &material);
}
static TextureUseNode *entry_use(TextureImagePublication *p, uint32_t entry) {
    return p->entry_capacity ? p->entries[texture_publication_entry_at(p,entry)].use : NULL;
}
static void indexed_publication_updates(void) {
    TextureMaterialLedger ledger = {0}; TextureMaterial first,second;
    init(&first,&ledger,601,11);init(&second,&ledger,602,12);
    TextureMaterialInstance *a=first.resident.instance,*b=second.resident.instance;
    TextureImagePublication *p=vk_texture_publication_create_indexed(16*32768);
    recording_fence=60;
    uint64_t allocated=publication_allocations;
    for(uint32_t i=0;i<37;i++)vk_texture_publication_update(p,i*1000,i&1?0:999);
    assert(!p->entries&&!p->entry_count&&publication_allocations==allocated);
    vk_texture_publication_update(p,4,601);vk_texture_publication_update(p,32768+4,601);
    TextureUseNode *saved=entry_use(p,4);
    assert(saved==entry_use(p,32768+4)&&saved->publication_refs==2&&a->lifetime.captures==1);
    assert(texture_publication_image_locked(p,4,601)==a);
    assert(texture_publication_image_locked(p,32768+4,601)==a);
    assert(!texture_publication_image_locked(p,5,601));
    assert(!texture_publication_image_locked(p,4,602));
    assert(vk_texture_publication_contains_image_locked(p,4,601));
    assert(!vk_texture_publication_contains_image_locked(p,5,601));
    assert(!vk_texture_publication_contains_image_locked(p,4,602));
    assert(p->entry_capacity==16&&p->entry_count==2);
    vk_texture_publication_use(p);
    uint64_t batch=p->last_batch,alloc_before=publication_allocations,free_before=publication_frees;
    // The measured workload rebuilt whole owners on view writes. Changing37 views over the same image must
    // keep stable owning records, do no allocation/cancellation and never repeat the scene capture walk.
    for(uint32_t i=0;i<37;i++) {
        vk_texture_publication_update(p,4,601);vk_texture_publication_commit_updates(p);
        vk_texture_publication_use(p);
        assert(p->last_batch==batch&&entry_use(p,4)==saved&&a->lifetime.captures==2);
    }
    assert(publication_allocations==alloc_before&&publication_frees==free_before);
    vk_texture_publication_update(p,4,0);vk_texture_publication_update(p,32768+4,0);
    assert(p->pending==saved&&!saved->publication_refs&&a->lifetime.captures==2);
    vk_texture_publication_update(p,4,601);vk_texture_publication_update(p,32768+4,601);
    vk_texture_publication_commit_updates(p);assert(entry_use(p,4)==saved&&saved->publication_refs==2&&!p->pending);
    vk_texture_publication_update(p,32768+4,0);assert(saved->publication_refs==1&&!p->pending);
    vk_texture_publication_update(p,4,602);
    assert(!saved->publication_refs&&saved->publication_pending&&a->lifetime.captures==2);
    assert(!texture_publication_image_locked(p,4,601)); // Pending writes retain A but no longer select it in this slot.
    assert(texture_publication_image_locked(p,4,602)==b);
    assert(!texture_publication_image_locked(p,32768+4,602));
    assert(b->lifetime.captures==2); // Newly added after first use: independent capture already belongs to this scene.
    TextureUseNode *other=entry_use(p,4);
    vk_texture_publication_update(p,4,601); // A->B->A before native flush revives the exact original owning record.
    assert(entry_use(p,4)==saved&&saved->publication_refs==1&&other->publication_pending);
    assert(a->lifetime.captures==2&&b->lifetime.captures==2);
    vk_texture_publication_commit_updates(p);
    assert(!p->pending&&a->lifetime.captures==2&&b->lifetime.captures==1);
    assert(retire(&first)==a&&retire(&second)==b);
    assert(vk_texture_publication_has_retired(p));

    assert(!first.resident.image && texture_publication_image_locked(p,4,601)==a);

    // A second set catches up through exact detached instances; no mutable residency is consulted.
    TextureImagePublication *copy=vk_texture_publication_create_indexed(16*32768);
    vk_texture_publication_update(copy,9,601);assert(a->lifetime.captures==3);
    assert(!texture_publication_image_locked(copy,4,601));
    assert(texture_publication_image_locked(copy,9,601)==a);
    vk_texture_publication_update(p,4,0);vk_texture_publication_commit_updates(p);
    assert(!p->entry_count&&!p->head&&a->lifetime.captures==2);
    vk_texture_publication_release(p);
    TextureUseBatch *scene=vk_texture_uses_seal(60,12);assert(scene);
    vk_texture_uses_handed(scene);

    // The archived exact owner still supplies the same detached material to a new recording.
    recording_fence=61;
    lock();
    const TextureUseNode *published=texture_publication_owner_locked(copy,9,601);
    assert(published&&published->use.instance==a);
    texture_use_scene_material_locked(a,&published->use);
    unlock();
    vk_texture_publication_release(copy);
    assert(a->lifetime.captures==2&&b->lifetime.captures==1);
    vk_texture_uses_complete(59,12);assert(a->lifetime.captures==2);
    vk_texture_uses_complete(60,12);assert(a->lifetime.captures==1&&!b->lifetime.captures);
    TextureUseBatch *history=vk_texture_uses_seal(61,12);assert(history);
    vk_texture_uses_handed(history);
    vk_texture_uses_complete(61,12);assert(!a->lifetime.captures);
    uint64_t old_id=a->value.id;destroy(a,&first);destroy(b,&second);

    // The same backend handle, after real retirement, identifies a new instance and a new owning record.
    TextureMaterial reused;init(&reused,&ledger,601,0);a=reused.resident.instance;assert(a->value.id!=old_id);
    p=vk_texture_publication_create_indexed(1);recording_fence=62;
    vk_texture_publication_update(p,0,601);vk_texture_publication_use(p);
    assert(entry_use(p,0)->use.instance==a&&a->lifetime.captures==2);
    vk_texture_publication_update(p,0,0);assert(p->pending&&a->lifetime.captures==2);
    vk_texture_publication_release(p); // Cancellation disposes pending removals once, with no commit required.
    assert(a->lifetime.captures==1);vk_texture_uses_cancel(62);
    assert(retire(&reused)==a);destroy(a,&reused);
}
static void queued_descriptor_write_holds(void) {
    TextureMaterialLedger ledger={0};TextureMaterial material;init(&material,&ledger,701,0);
    TextureMaterialInstance *instance=material.resident.instance;
    TextureImagePublication *p=vk_texture_publication_create_indexed(524288);
    recording_fence=0;vk_texture_publication_update(p,500000,701);
    assert(retire(&material)==instance);
    vk_texture_publication_update(p,500000,0);
    assert(!p->entry_count&&p->pending&&instance->lifetime.captures==1);
    // Native descriptor writes still name701. Reclaim between queued writes must not schedule its destruction.
    lock();content_collect(UINT64_MAX);assert(!texture_material_instance_defer_ready(instance));unlock();
    vk_texture_publication_commit_updates(p);assert(!p->head&&!p->pending&&!instance->lifetime.captures);
    assert(texture_material_instance_defer_ready(instance));
    vk_texture_publication_release(p);destroy(instance,&material);
}
static void empty_indexed_first_use(void) {
    TextureMaterialLedger ledger={0};TextureMaterial material;init(&material,&ledger,702,13);
    TextureMaterialInstance *instance=material.resident.instance;
    TextureImagePublication *p=vk_texture_publication_create_indexed(1024);
    recording_fence=62;assert(!p->last_batch);
    vk_texture_publication_use(p);assert(p->last_batch&&!p->head);
    vk_texture_publication_update(p,999,702);assert(instance->lifetime.captures==2);
    vk_texture_publication_update(p,999,0);vk_texture_publication_commit_updates(p);
    assert(instance->lifetime.captures==1);vk_texture_publication_release(p);
    TextureUseBatch *batch=vk_texture_uses_seal(62,13);
    assert(batch&&batch->head&&batch->head->use.instance==instance&&batch->upload_submitted==13);
    vk_texture_uses_handed(batch);assert(retire(&material)==instance);
    vk_texture_uses_complete(61,13);assert(instance->lifetime.captures==1);
    vk_texture_uses_complete(62,13);destroy(instance,&material);
}
static void indexed_publication_repeat_across_parts(void) {
    TextureMaterialLedger ledger = {0};
    TextureMaterial material;
    init(&material, &ledger, 703, 0);
    TextureMaterialInstance *instance = material.resident.instance;
    TextureImagePublication *set = vk_texture_publication_create_indexed(4);
    recording_fence = 63;
    vk_texture_publication_update(set, 2, 703);
    vk_texture_publication_use(set);
    TextureUseBatch *first = vk_texture_uses_seal(63, 0);
    assert(first && instance->lifetime.captures == 2);
    vk_texture_uses_handed(first);
    vk_texture_publication_use(set);
    lock();
    TextureUseBatch *tail = texture_use_active(63, 0);
    unlock();
    assert(tail && tail != first && instance->lifetime.captures == 3);
    vk_texture_publication_release(set);
    vk_texture_uses_cancel(63);
    assert(instance->lifetime.captures == 1);
    assert(retire(&material) == instance);
    vk_texture_uses_complete(63, 0);
    destroy(instance, &material);
}

static void sparse_index_churn(void) {
    enum {N=257,M=7};TextureMaterialLedger ledger={0};TextureMaterial materials[M];
    for(unsigned i=0;i<M;i++)init(&materials[i],&ledger,800+i,0);
    TextureImagePublication *p=vk_texture_publication_create_indexed(524288);
    uint32_t expected[N]={0},random=1;recording_fence=0;
    for(unsigned step=0;step<20000;step++) {
        random=random*1664525u+1013904223u;unsigned slot=(random>>8)%N;
        random=random*1664525u+1013904223u;unsigned value=(random>>8)%(M+1);
        uint32_t index=slot*1999;
        expected[slot]=value;vk_texture_publication_update(p,index,value?799+value:0);
        if(step%31)continue;
        vk_texture_publication_commit_updates(p);
        uint32_t counts[M]={0},entries=0;
        for(unsigned i=0;i<N;i++) {
            TextureUseNode *node=entry_use(p,i*1999);
            if(!expected[i])assert(!node);
            else {assert(node&&node->use.instance==materials[expected[i]-1].resident.instance);counts[expected[i]-1]++;entries++;}
        }
        assert(p->entry_count==entries&&p->entry_capacity<=1024);
        for(unsigned i=0;i<M;i++) {
            TextureMaterialInstance *instance=materials[i].resident.instance;
            TextureUseNode *node=texture_use_find(p->buckets,instance);
            assert(instance->lifetime.captures==(counts[i]!=0));
            assert((node?node->publication_refs:0)==counts[i]);
        }
    }
    vk_texture_publication_release(p);
    for(unsigned i=0;i<M;i++){TextureMaterialInstance *instance=retire(&materials[i]);destroy(instance,&materials[i]);}
}
static void prepared_manifest_exact_handoff(void) {
    TextureMaterialLedger ledger={0};TextureMaterial first,second;
    init(&first,&ledger,901,17);init(&second,&ledger,902,18);
    TextureMaterialInstance *a=first.resident.instance,*b=second.resident.instance;
    TextureImagePublication *selected=vk_texture_publication_create_indexed(3);
    TextureImagePublication *native=vk_texture_publication_create(NULL,0,NULL);
    recording_fence=0;
    vk_texture_publication_update(selected,0,901);
    vk_texture_publication_update(selected,1,901);
    assert(a->lifetime.captures==1&&entry_use(selected,0)==entry_use(selected,1));
    TextureUseNode *selection=entry_use(selected,0);
    vk_texture_publication_clear(selected);
    assert(!selected->entry_count&&selected->pending==selection&&a->lifetime.captures==1);
    vk_texture_publication_update(selected,0,901);
    vk_texture_publication_update(selected,1,901);
    vk_texture_publication_commit_updates(selected);
    assert(entry_use(selected,0)==selection&&!selected->pending&&selection->publication_refs==2);
    assert(retire(&first)==a); // Logical residency disappears before native packet installation.
    recording_fence=70;
    vk_texture_publication_capture(native,selected);
    assert(texture_use_find(native->buckets,a)&&a->lifetime.captures==3);
    TextureUseBatch *part1=vk_texture_uses_seal(70,17);assert(part1);

    // The same owner and fence are reused by a distinct recording part. Already-owned instances must still
    // acquire its separate batch capture, without consulting the now-empty material residency.
    vk_texture_publication_capture(native,selected);
    TextureUseBatch *part2=vk_texture_uses_seal(70,17);
    assert(part2&&part2!=part1&&part2->id!=part1->id&&a->lifetime.captures==4);
    vk_texture_uses_handed(part1);vk_texture_uses_abandon(part2);
    vk_texture_uses_complete(70,17);assert(a->lifetime.captures==2);

    // A replacement remains held before native capture, while removed entries cannot leak into the next native
    // manifest. A prior native owner independently retains A until all of its immutable packets are finished.
    recording_fence=71;
    vk_texture_publication_clear(selected);
    vk_texture_publication_update(selected,0,902);
    vk_texture_publication_update(selected,1,0);
    assert(selected->pending&&a->lifetime.captures==2&&b->lifetime.captures==1);
    TextureImagePublication *next_native=vk_texture_publication_create(NULL,0,NULL);
    vk_texture_publication_capture(next_native,selected);
    assert(!texture_use_find(next_native->buckets,a)&&texture_use_find(next_native->buckets,b));
    assert(b->lifetime.captures==3);
    vk_texture_publication_commit_updates(selected);
    assert(a->lifetime.captures==1);
    vk_texture_publication_release(native);assert(!a->lifetime.captures);destroy(a,&first);
    assert(retire(&second)==b);
    vk_texture_publication_release(selected);assert(b->lifetime.captures==2);
    TextureUseBatch *part3=vk_texture_uses_seal(71,18);assert(part3);
    vk_texture_uses_handed(part3);
    vk_texture_publication_release(next_native);assert(b->lifetime.captures==1);
    vk_texture_uses_complete(71,18);assert(!b->lifetime.captures);destroy(b,&second);

    // Reusing a numeric backend handle after true destruction acquires a new physical instance.
    TextureMaterial reused;init(&reused,&ledger,901,0);b=reused.resident.instance;
    selected=vk_texture_publication_create_indexed(1);native=vk_texture_publication_create(NULL,0,NULL);
    recording_fence=72;vk_texture_publication_update(selected,0,901);
    vk_texture_publication_capture(native,selected);
    assert(texture_use_find(native->buckets,b)&&b->lifetime.captures==3);
    vk_texture_publication_release(selected);vk_texture_publication_release(native);
    vk_texture_uses_cancel(72);assert(!b->lifetime.captures);
    assert(retire(&reused)==b);destroy(b,&reused);

    // Abandoned direct preparation has no native owner or recording use: clearing and canceling the builder
    // releases its exact holds immediately, including a provider detached during preparation.
    TextureMaterial abandoned;init(&abandoned,&ledger,903,0);b=abandoned.resident.instance;
    selected=vk_texture_publication_create_indexed(1);recording_fence=0;
    vk_texture_publication_update(selected,0,903);assert(retire(&abandoned)==b);
    vk_texture_publication_clear(selected);vk_texture_publication_commit_updates(selected);
    assert(!b->lifetime.captures&&!selected->head&&!selected->pending);
    vk_texture_publication_release(selected);destroy(b,&abandoned);
}
static void pinned_target_native_handoff(void) {
    recording_fence = 90;
    lock();
    texture_use_pin(&pinned_owner, &pinned_texture);
    texture_use_pin(&pinned_owner, &pinned_texture); // same pass/part retains the owner once
    unlock();
    assert(pinned_refs == 1);
    TextureUseBatch *handed = vk_texture_uses_seal(90, 0);
    vk_texture_uses_handed(handed);
    lock(); texture_use_pin(&pinned_owner, &pinned_texture); unlock();
    TextureUseBatch *abandoned = vk_texture_uses_seal(90, 0);
    vk_texture_uses_abandon(abandoned);
    recording_fence = 91;
    lock(); texture_use_pin(&pinned_owner, &pinned_texture); unlock();
    assert(pinned_refs == 3);
    vk_texture_uses_cancel_unsubmitted(90);
    assert(pinned_refs == 1 && pinned_released == 2);
    vk_texture_uses_complete(89, 0);
    assert(pinned_refs == 1);
    vk_texture_uses_idle(90, 0); // real native idle can discharge a handed prefix without its guest final fence
    assert(!pinned_refs && pinned_released == 3 && texture_uses.completed == 89);
}

static void indexed_native_descriptor_handoff(void) {
    TextureImagePublication *publication = vk_texture_publication_create_indexed(2);
    recording_fence = 100;
    lock();
    assert(texture_publication_native_retain_locked(publication, 0, native_owner.image));
    assert(texture_publication_native_retain_locked(publication, 1, native_owner.image));
    unlock();
    assert(native_owner.refs == 2);
    vk_texture_publication_use(publication);
    vk_texture_publication_use(publication);
    assert(native_owner.refs == 3);
    TextureUseBatch *handed = vk_texture_uses_seal(100, 0);
    assert(handed && handed->native && !handed->native->next);
    vk_texture_publication_use(publication);
    TextureUseBatch *canceled = vk_texture_uses_seal(100, 0);
    assert(canceled && canceled != handed && native_owner.refs == 4);

    native_owner.retired = 1;
    assert(vk_texture_publication_has_retired(publication));
    vk_texture_publication_clear(publication);
    assert(native_owner.refs == 4);
    vk_texture_publication_commit_updates(publication);
    assert(native_owner.refs == 2);
    vk_texture_publication_release(publication);
    vk_texture_uses_handed(handed);
    vk_texture_uses_abandon(canceled);
    vk_texture_uses_complete(99, 0);
    assert(native_owner.refs == 2);
    vk_texture_uses_idle(100, 0);
    assert(!native_owner.refs && texture_uses.completed == 99);
}

static void indexed_alias_descriptor_handoff(void) {
    TextureImagePublication *publication = vk_texture_publication_create_indexed(2);
    VkTextureAliasRead alias = {1};
    const VkImage image = 8002;
    recording_fence = 101;
    lock();
    assert(vk_texture_publication_retain_alias_locked(publication, 0, image, &alias));
    assert(vk_texture_publication_retain_alias_locked(publication, 1, image, &alias));
    assert(vk_texture_publication_alias_locked(publication, 0, image) == &alias);
    assert(!vk_texture_publication_alias_locked(publication, 0, image + 1));
    vk_texture_alias_read_cancel(&alias); // The descriptor edges now own the original alias.
    unlock();
    assert(alias.refs == 2);
    vk_texture_publication_use(publication);
    vk_texture_publication_use(publication);
    TextureUseBatch *handed = vk_texture_uses_seal(101, 0);
    assert(handed && handed->native && handed->native->alias == &alias && !handed->native->next && alias.refs == 3);
    vk_texture_publication_use(publication);
    TextureUseBatch *unissued = vk_texture_uses_seal(101, 0);
    assert(unissued && unissued != handed && alias.refs == 4);
    vk_texture_publication_update(publication, 0, VK_NULL_HANDLE);
    assert(alias.refs == 4 && publication->pending_native);
    vk_texture_publication_commit_updates(publication);
    assert(alias.refs == 3);
    vk_texture_publication_release(publication);
    vk_texture_uses_handed(handed);
    vk_texture_uses_abandon(unissued);
    vk_texture_uses_cancel_unsubmitted(101);
    vk_texture_uses_complete(100, 0);
    assert(alias.refs == 1 && !native_owner.refs);
    vk_texture_uses_complete(101, 0);
    assert(!alias.refs && !native_owner.refs);
}


static void part_disposition_survives_batch_release(void) {
    const uint64_t scene = 120;
    recording_fence = scene;
    lock();
    TextureUseBatch *batch = texture_use_active(scene, 1);
    TextureUsePartDisposition *recording = texture_use_part_retain(batch);
    TextureUsePartDisposition *also_recording = texture_use_part_retain(batch);
    unlock();
    assert(recording == also_recording && recording->refs == 3);
    assert(texture_use_part_known_unhanded(recording));
    vk_texture_uses_cancel(scene);
    assert(recording->refs == 2 && texture_use_part_known_unhanded(recording));
    texture_use_part_release(also_recording);
    assert(recording->refs == 1);
    texture_use_part_release(recording);

    lock();
    TextureUsePartDisposition *handed = texture_use_part_retain(texture_use_active(scene, 1));
    unlock();
    TextureUseBatch *prefix = vk_texture_uses_seal(scene, 0);
    assert(prefix && prefix->part == handed && !texture_use_part_known_unhanded(handed));
    vk_texture_uses_handed(prefix);

    lock();
    TextureUsePartDisposition *abandoned = texture_use_part_retain(texture_use_active(scene, 1));
    unlock();
    TextureUseBatch *tail = vk_texture_uses_seal(scene, 0);
    assert(tail && tail->part == abandoned && !texture_use_part_known_unhanded(abandoned));
    vk_texture_uses_complete(scene - 1, 0);
    assert(abandoned->refs == 2 && handed->refs == 2);
    vk_texture_uses_abandon(tail);
    vk_texture_uses_cancel_unsubmitted(scene);
    assert(abandoned->refs == 1 && texture_use_part_known_unhanded(abandoned));
    assert(handed->refs == 2 && !texture_use_part_known_unhanded(handed));
    vk_texture_uses_idle(scene, 0);
    assert(handed->refs == 1 && !texture_use_part_known_unhanded(handed));
    assert(texture_uses.completed == scene - 1);
    texture_use_part_release(handed);
    texture_use_part_release(abandoned);
}

int main(void) {
    submitted_split_and_cached_publication();
    current_images_cancel_and_shutdown();
    publication_refuses_copied_source_ticket();
    descriptor_pool_union_survives_split();
    idle_drain_and_unsubmitted_loss_cleanup();
    indexed_publication_updates();
    queued_descriptor_write_holds();
    empty_indexed_first_use();
    indexed_publication_repeat_across_parts();
    sparse_index_churn();
    prepared_manifest_exact_handoff();
    pinned_target_native_handoff();
    indexed_native_descriptor_handoff();
    indexed_alias_descriptor_handoff();
    part_disposition_survives_batch_release();
    assert(!texture_uses.nodes && !texture_uses.batches && !texture_uses.publications && !locked);
    for (unsigned i = 0; i < TEXTURE_IMAGE_BUCKETS; ++i) assert(!texture_uses.images[i]);
    puts("resource core scene captures: passed");
    return 0;
}
