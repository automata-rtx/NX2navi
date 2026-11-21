#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define RC_REGISTRY_SINGLE_THREADED 1
#include "../runtime/resource_core_registry.c"

enum { VK_POOL_TEX = 8, VK_IMAGE_TYPE_2D = 2, NVN_TEXTURE_TARGET_1D, NVN_TEXTURE_TARGET_1D_ARRAY,
       NVN_TEXTURE_TARGET_3D, NVN_TEXTURE_TARGET_BUFFER, NVN_TEXTURE_TARGET_2D };
enum { NVN_MEMORY_POOL_FLAGS_PHYSICAL = 1, NVN_MEMORY_POOL_FLAGS_VIRTUAL = 2 };
#define VK_NULL_HANDLE 0
enum { VK_TEXTURE_REGISTRATION_CURRENT = 0, VK_TEXTURE_REGISTRATION_EXACT = 1,
       VK_TEXTURE_REGISTRATION_ABSENT = -1, VK_TEXTURE_REGISTRATION_GONE = -2 };

typedef uint64_t VkImage;
typedef struct { VkImage image; int gpu_owned, skip, type, gone; } VkTexImage;
typedef struct {struct {uint64_t id;} value;} TextureMaterialInstance;
typedef struct {TextureMaterialInstance *instance;} TextureMaterialUse;
typedef struct {TextureMaterialUse use;} TextureUseNode;
typedef struct {struct {TextureMaterialInstance *instance;} resident;unsigned refs;} TextureMaterial;
typedef struct TextureFlat3D {
    TextureMaterial *material;
    struct TextureFlat3D *next;
    VkTexImage image;
    VkImage source;
    uint64_t source_instance,revision;
    uint32_t upload,layer;
    int ordered;
} TextureFlat3D;
typedef struct { uint64_t view; int set_layers, set_format; uint32_t min_layer, format; } view_t;
typedef struct {
    void *mem;
    uint64_t serial;
    struct { uint64_t size; } L;
    struct { int samples, target, format; } d;
    unsigned pool_flags;
} tex_t;
typedef struct TextureRegistration TextureRegistration;
typedef struct { uintptr_t addr; int target,format; } TexKey;
typedef struct {
    tex_t t;
    TexKey key;
    int uploaded, up_retry, source_canonical, released;
    TextureMaterial *material;
    TextureFlat3D *flat3d;
    uint32_t n_up;
    uint64_t content_revision;
    void *source_mapping;
    TextureRegistration *source_registrations;
    struct { uint64_t id; } material_version;
    VkTexImage pub;
} Tex;
typedef struct VkTextureRegistrationScope VkTextureRegistrationScope;
typedef struct {
    tex_t t;
    uint32_t cgen;
    int used, sample_owned, chas_view, finalized;
    view_t cview;
    Tex *ce;
    TextureRegistration *registration;
    uint64_t pool_owner;
    int logical_id, free_slot;
} TexSlot;
typedef struct { uint64_t images[VK_POOL_TEX * 2 + 1]; } TextureImagePublication;

static TexSlot tslots[32];
static int ntslots = 32, pool_slots;
static uint32_t agen[1] = {1}, slots_gen;
static uint64_t sinfo[32];
static SRWLOCK texture_lock = SRWLOCK_INIT;
static _Thread_local int locked;
static int holds[4], scene_uses[4];
static VkTextureRegistrationScope *selected;
static uint64_t selected_pool;

static void lock(void) { assert(!locked); AcquireSRWLockExclusive(&texture_lock); locked = 1; }
static void unlock(void) { assert(locked); locked = 0; ReleaseSRWLockExclusive(&texture_lock); }
static _Noreturn void rt_fatal(const char *message, ...) { puts(message); abort(); }
static void rt_log(const char *message, ...) { (void)message; }
static void texture_source_report_locked(int logical) { assert(locked && logical >= 0); }
static int nvn_gpu_only(uintptr_t addr) { return addr > UINT32_MAX; }
static int tex_mirrored(const tex_t *t) { return t->mem != NULL; }
static int source_private_sample(const Tex *e) { return !e->pub.gpu_owned && !e->pub.skip; }
static void make_key(const tex_t *t,TexKey *key) {
    *key = (TexKey){(uintptr_t)t->mem,t->d.target,t->d.format};
}
static void slot_result_view(const TexSlot *slot,const Tex *entry,view_t *view,int *has);
static void dirty_push(uint32_t id);
static void vk_backend_registration_record_progress(void) { assert(!locked); }
static TextureMaterialInstance resident[] = {{{0}}, {{101}}, {{102}}, {{103}}};
static TextureMaterialInstance *content_use_image_find(uint64_t image) {
    return image && image < 4 ? &resident[image] : NULL;
}
static void texture_material_acquire(TextureMaterial *material,int live) {
    assert(live&&material->refs);material->refs++;
}
static void flat3d_retire(TextureFlat3D *child) {
    assert(child->material->refs>1);child->material->refs--;free(child);
}
static VkTextureRegistrationScope *vk_exec_texture_registration_scope(void) { return selected; }
static uint64_t vk_exec_texture_pool(void) { return selected_pool; }

static TextureImagePublication *vk_texture_publication_create_indexed(size_t count) {
    assert(count == VK_POOL_TEX * 2 + 1);
    return calloc(1, sizeof(TextureImagePublication));
}
static void vk_texture_publication_update(TextureImagePublication *publication, size_t index, uint64_t image) {
    uint64_t old = publication->images[index];
    if (old) holds[old]--;
    publication->images[index] = image;
    if (image) holds[image]++;
}
static void vk_texture_publication_commit_updates(TextureImagePublication *publication) { assert(publication); }
static void vk_texture_publication_release(TextureImagePublication *publication) {
    if (!publication) return;
    for (size_t i = 0; i < VK_POOL_TEX * 2 + 1; ++i)
        vk_texture_publication_update(publication, i, 0);
    free(publication);
}
static TextureUseNode indexed[] = {{{NULL}}, {{&resident[1]}}, {{&resident[2]}}, {{&resident[3]}}};
static const TextureUseNode *alternate_owner;
static const TextureUseNode *texture_publication_owner_locked(const TextureImagePublication *publication,
    uint32_t index, VkImage expected) {
    if (!publication || !expected || publication->images[index] != expected) return NULL;
    return alternate_owner ? alternate_owner : &indexed[expected];
}
static void texture_use_scene_material_locked(TextureMaterialInstance *instance, const TextureMaterialUse *owner) {
    assert(instance && owner && owner->instance == instance && holds[instance - resident]);
    scene_uses[instance - resident]++;
}

#include "../runtime/nvn_hmap.h"
#include "../runtime/vk_texture_pool_names.h"
#include "../runtime/vk_texture_registrations.h"

static void slot_result_view(const TexSlot *slot,const Tex *entry,view_t *view,int *has) {
    assert(slot->registration->source == entry);
    *view=slot->registration->view;
    *has=slot->registration->has_view;
}
static void slot_unplace(int id) { assert(id >= 0 && id < ntslots); }
static void dirty_push(uint32_t id) { assert(id < (uint32_t)ntslots); }

typedef struct { const void *at; VkTextureRegistrationScope *registration_scope; int registration_deferred; } Cont;
typedef struct {const VkTexImage *image; view_t view; int has_view, native_slot, historical; uint64_t registration;} VkTextureSample;
#include "registration_resume_under_test.h"

static tex_t descriptor(uint64_t serial) {
    return (tex_t){.mem = (void *)(uintptr_t)4096, .serial = serial, .L.size = 64,
                   .d.target = NVN_TEXTURE_TARGET_2D, .d.samples = 1};
}
static int publish_pool(uint64_t owner, int logical, const tex_t *t, const view_t *view, Tex *ready) {
    lock();
    int id = texture_pool_claim(owner, logical);
    TexSlot *slot = &tslots[id];
    slot->pool_owner = owner;
    slot->logical_id = logical;
    texture_registration_publish(slot, id, t, view);
    assert(slot->registration->descriptor.serial == t->serial);
    slot->t = slot->registration->descriptor;
    slot->used = 1;
    slot->ce = ready;
    slot->sample_owned = ready != NULL;
    slot->cgen = 1;
    slot->chas_view = view != NULL;
    slot->cview = view ? *view : (view_t){0};
    unlock();
    return id;
}
static void finalize(int id) {
    lock();
    texture_registration_finalize(&tslots[id], id);
    unlock();
}
static int resolve(int id, uint64_t expected_image, uint64_t expected_view) {
    const VkTexImage *image = NULL;
    view_t view = {0};
    int has_view = 0;
    uint64_t bytes = 0;
    lock();
    int result = vk_texture_registration_image_locked(id, &image, &view, &has_view, &bytes);
    VkImage expected = 0;
    assert(vk_texture_registration_expected_locked(id, &expected) == result);
    assert(expected == expected_image);
    if (result > 0) assert(image->image == expected_image && view.view == expected_view && has_view && bytes == 64);
    else assert(!image && !has_view && !bytes);
    unlock();
    return result;
}

static void visited(int slot, void *arg) {
    assert(slot >= 0 && slot < VK_POOL_TEX);
    *(uint32_t *)arg |= 1u << slot;
}

static DWORD WINAPI check_sparse_selectors(void *arg) {
    unsigned passes = *(const unsigned *)arg;
    for (unsigned i = 0; i < passes; i++) {
        lock();
        int live = vk_texture_pool_slot_locked(100, 1);
        assert(live == -1 || (live == 1 && tslots[live].pool_owner == 100));
        live = vk_texture_pool_slot_locked(101, 1);
        assert(live == -1 || (live == 1 && tslots[live].pool_owner == 101));
        assert(vk_texture_pool_slot_locked(100, VK_POOL_TEX - 1) == -1);
        assert(vk_texture_pool_slot_locked(101, VK_POOL_TEX - 1) == -1);
        unlock();
    }
    return 0;
}

static void sparse_primary_selectors(void) {
    ntslots = 2;
    const unsigned passes = 4096;
    HANDLE reader = CreateThread(NULL, 0, check_sparse_selectors, (void *)&passes, 0, NULL);
    assert(reader);
    for (unsigned i = 0; i < passes; i++) {
        uint64_t owner = 100 + i % 2;
        lock();
        assert(vk_texture_pool_slot_locked(owner, 1) == -1);
        assert(texture_pool_claim(owner, 1) == 1);
        assert(vk_texture_pool_slot_locked(owner, 1) == -1); // The claimed slot is still empty.
        assert(vk_texture_pool_slot_locked(owner, -1) == -1);
        assert(vk_texture_pool_slot_locked(owner, 1 << 20) == -1);
        assert(vk_texture_pool_slot_locked(owner, VK_POOL_TEX - 1) == -1); // First high slot is unallocated.
        assert(texture_pool_claim(owner, VK_POOL_TEX - 1) == VK_POOL_TEX - 1);
        assert(vk_texture_pool_slot_locked(owner, VK_POOL_TEX - 1) == -1);
        tslots[1] = (TexSlot){.used = 1, .pool_owner = owner, .logical_id = 1};
        assert(vk_texture_pool_slot_locked(owner, 1) == 1);
        assert(vk_texture_pool_slot_locked(201 - owner, 1) == -1);
        unlock();
        lock();
        tslots[1] = (TexSlot){0};
        texture_pool_names_shutdown();
        unlock();
    }
    assert(WaitForSingleObject(reader, INFINITE) == WAIT_OBJECT_0);
    assert(CloseHandle(reader));
    ntslots = 32;
}

int main(void) {
    sparse_primary_selectors();
    tex_t old = descriptor(1), replacement = descriptor(2);
    view_t original_view = {.view=101,.set_layers=1,.min_layer=1,.set_format=1,.format=11};
    view_t replacement_view = {.view=102,.set_format=1,.format=10};
    TextureMaterial old_material={.resident.instance=&resident[1],.refs=1};
    TextureMaterial new_material={.resident.instance=&resident[2],.refs=1},derived_material={.refs=1};
    TextureFlat3D old_child={.material=&derived_material,.source=1,.source_instance=101,.layer=1,
                             .image={.image=3}};
    Tex first = {.t = old, .uploaded = 1, .material = &old_material, .material_version = {201},
                 .flat3d=&old_child,.pub = {.image=1,.type=VK_IMAGE_TYPE_2D}};
    Tex second = {.t = replacement, .uploaded = 1, .material = &new_material, .material_version = {202},
                  .pub = {.image=2,.type=VK_IMAGE_TYPE_2D}};
    publish_pool(0, 3, &old, &original_view, &first);
    int cold = publish_pool(0, 4, &old, &original_view, NULL);
    lock();
    uint32_t format;
    assert(vk_texture_registration_view_format_locked(3,&format) && format==11);
    assert(vk_texture_registration_view_format_locked(4,&format) && format==11);
    assert(!vk_texture_registration_view_format_locked(5,&format));
    first.source_canonical=1;first.source_mapping=&first;make_key(&old,&first.key);
    assert(texture_registration_source_attach(&tslots[3],&first));
    tslots[3].cgen=0;tslots[3].ce=(Tex *)(uintptr_t)1;
    uint32_t source_slots=0;
    texture_registration_source_dirty(&first,visited,&source_slots);
    assert(source_slots==(1u<<3));
    texture_registration_source_clear(&first);
    assert(!first.source_registrations && !texture_registration_source(&tslots[3]));
    assert(texture_registration_source_attach(&tslots[3],&first));
    slots_gen=0;
    unlock();

    VkTextureRegistrationScope *enqueue = vk_texture_registration_scope_begin();
    VkTextureRegistrationScope *same_registrations = vk_texture_registration_scope_begin();
    selected = enqueue;
    assert(!vk_texture_registration_scope_token() && !resolve(3, 0, 0));
    assert(vk_texture_registration_scope_current(enqueue));
    assert(texture_registration_same(&tslots[cold], &old, &original_view));
    assert(!texture_registration_same(&tslots[cold], &replacement, &original_view));
    uint64_t queued = vk_texture_registration_scope_publication(enqueue);
    assert(texture_registration_pending(&tslots[cold], queued));
    assert(!texture_registration_pending(&tslots[cold], tslots[cold].registration->publication - 1));
    assert(!vk_texture_pool_release_ordered(0, queued) && vk_texture_registration_scope_current(enqueue));
    Cont paused = {.registration_scope = vk_texture_registration_scope_retain(enqueue)};
    Cont unstarted = {.registration_scope = vk_texture_registration_scope_retain(enqueue)};
    cont_registration_defer(&unstarted);
    assert(!unstarted.registration_scope && unstarted.registration_deferred);
    vk_texture_registration_scope_end(enqueue);

    publish_pool(0, 3, &replacement, &replacement_view, &second); // Same guest address, new registration/material.
    assert(derived_material.refs==2);
    assert(texture_retired_registration->flat3d && texture_retired_registration->flat3d->image.image==3);
    assert(texture_retired_registration->flat3d!=&old_child);
    publish_pool(0, 4, &replacement, &replacement_view, &second);
    publish_pool(0, 5, &replacement, &replacement_view, &second);
    assert(holds[1] == 1 && !holds[2]);
    assert(!vk_texture_registration_scope_current(selected));
    assert(cont_registration_resume(&paused) == selected); // Queued, but no wait operation was reached.
    vk_texture_registration_record_end(paused.registration_scope,1);
    uint64_t exact_registrations = vk_texture_registration_scope_token();
    assert(exact_registrations);
    selected = same_registrations;
    // Distinct queue owners with identical command-visible registrations use the same selector publication.
    assert(vk_texture_registration_scope_token() == exact_registrations);
    selected = paused.registration_scope;
    vk_texture_registration_scope_end(same_registrations);
    assert(vk_texture_registration_publication() == exact_registrations);
    uint64_t current_registrations = texture_registration_publication;
    uint32_t changed = 0;
    lock();
    assert(!vk_texture_registration_view_format_locked(3,&format)); // Held exact view still resolves separately.
    assert(!vk_texture_registration_view_format_locked(4,&format)); // Lost historical source is not absence proof.
    assert(vk_texture_registration_changes_locked(0, current_registrations, exact_registrations, visited, &changed));
    assert(changed == ((1u << 3) | (1u << 4) | (1u << 5)));
    changed = 0;
    assert(vk_texture_registration_changes_locked(0, exact_registrations, current_registrations, visited, &changed));
    assert(changed == ((1u << 3) | (1u << 4) | (1u << 5)));
    unlock();
    assert(resolve(3, 1, 101) == 1 && scene_uses[1] == 1);
    assert(derived_material.refs==2);
    assert(resolve(4, 0, 0) == VK_TEXTURE_REGISTRATION_GONE); // An old registration cannot reread new bytes.
    assert(resolve(5, 0, 0) == VK_TEXTURE_REGISTRATION_ABSENT);
    selected = cont_registration_resume(&unstarted);
    assert(selected && !unstarted.registration_deferred && !vk_texture_registration_scope_token());
    assert(!resolve(3, 0, 0) && !resolve(5, 0, 0)); // Both replacements are current at the late observation.
    vk_texture_registration_record_end(unstarted.registration_scope,1);
    vk_texture_registration_scope_end(unstarted.registration_scope);

    paused.at = &paused;
    selected = cont_registration_resume(&paused);
    vk_texture_registration_record_end(paused.registration_scope,1);
    paused.at = NULL;
    assert(!vk_texture_registration_scope_token() && !resolve(3, 0, 0));
    assert(!holds[1] && scene_uses[1] == 1); // Registration retirement discharges its hold, while submitted use remains.
    assert(derived_material.refs==1);
    assert(!resolve(5, 0, 0));

    assert(rc_registry_retire_registration(texture_registrations, texture_registration_key(0, 3), old.serial) ==
           RC_REGISTRY_STALE);
    assert(!resolve(3, 0, 0)); // Late finalization of the old owner did not retire the replacement.
    finalize(3);
    assert(holds[2] == 1 && resolve(3, 2, 102) == 1);
    VkTextureRegistrationScope *after_finalize = vk_texture_registration_scope_begin();
    selected = after_finalize;
    assert(resolve(3, 0, 0) == VK_TEXTURE_REGISTRATION_ABSENT);
    vk_texture_registration_scope_end(after_finalize);
    selected = NULL;
    vk_texture_registration_scope_end(paused.registration_scope); // Cancellation also discharges history.
    assert(!holds[2] && scene_uses[2] == 1);

    int a = publish_pool(10, 3, &old, &original_view, &first);
    int b = publish_pool(20, 3, &replacement, &replacement_view, &second);
    assert(a != b && a >= VK_POOL_TEX && b >= VK_POOL_TEX);
    assert(vk_texture_pool_count_locked(10) == 4 && vk_texture_pool_count_locked(20) == 4);
    assert(vk_texture_pool_logical_locked(10, b) == -1 && vk_texture_pool_logical_locked(20, b) == 3);
    selected = vk_texture_registration_scope_begin();
    selected_pool = 10;
    tslots[a].sample_owned = 0; // Actual bindless selection can retain an old provider without the direct marker.
    lock();
    vk_texture_registration_note_locked(a, &first.pub, &original_view, 1);
    // Eviction or an unrelated address-generation change invalidates the slot's pointer. The chosen
    // material incarnation remains owned by the native publication until the registration can fork it.
    tslots[a].ce = NULL;
    tslots[a].cgen = 0;
    unlock();
    uint64_t first_pool_horizon = vk_texture_registration_publication();
    // Earlier selected residency cannot prove the bytes for a later, unread exact scope.
    assert(!vk_texture_pool_release_ordered(10, first_pool_horizon));
    assert(vk_texture_pool_release_ordered(10, tslots[a].registration->publication - 1));
    assert(holds[1] == 1 && vk_texture_pool_slot_locked(10, 3) == -1);
    int reincarnated = publish_pool(11, 3, &replacement, &replacement_view, &second);
    assert(reincarnated == a && vk_texture_pool_slot_locked(20, 3) == b);
    VkTextureSample saved = {0};
    int live_slot = 999;
    assert(texture_history_sample_locked(3, &saved, &live_slot, 0) == VK_TEXTURE_REGISTRATION_EXACT);
    assert(saved.image->image == 1 && saved.view.view == 101 && saved.registration &&
           saved.native_slot == -1 && saved.historical == VK_TEXTURE_REGISTRATION_EXACT && live_slot == -1);
    const uint64_t archived_registration = saved.registration;
    assert(vk_texture_history_sample_current_locked(3, archived_registration, 1));
    assert(!vk_texture_history_sample_current_locked(3, archived_registration + 1, 1));
    assert(!vk_texture_history_sample_current_locked(3, archived_registration, 2));
    const int prior_uses = scene_uses[1];
    alternate_owner = &indexed[2]; // Even a matching raw VkImage value needs the original material incarnation.
    assert(!vk_texture_history_sample_current_locked(3, archived_registration, 1));
    assert(resolve(3, 0, 0) == VK_TEXTURE_REGISTRATION_GONE && scene_uses[1] == prior_uses);
    alternate_owner = NULL;
    assert(resolve(3, 1, 101) == VK_TEXTURE_REGISTRATION_EXACT && scene_uses[1] == prior_uses + 1);
    selected_pool = 20;
    assert(resolve(3, 0, 0) == VK_TEXTURE_REGISTRATION_CURRENT);
    selected_pool = 11;
    assert(resolve(3, 0, 0) == VK_TEXTURE_REGISTRATION_ABSENT);
    lock();
    changed = 0;
    assert(vk_texture_registration_changes_locked(20, first_pool_horizon, texture_registration_publication,
                                                visited, &changed));
    assert(!changed); // Pool 10 retirement and pool 11 publication never invalidate pool 20's slot 3.
    unlock();
    vk_texture_registration_scope_end(selected);
    assert(!holds[1]);
    selected = NULL;
    vk_texture_pool_release(11);
    vk_texture_pool_release(20);
    assert(slots_gen == 3 && !pool_slots);

    lock();
    for (int i = 0; i < TEXTURE_REGISTRATION_CHANGES; ++i) texture_registration_changed(0, 7);
    changed = 0;
    assert(!vk_texture_registration_changes_locked(0, exact_registrations, texture_registration_publication, visited, &changed));
    assert(!changed); // History loss never supplies a partial repair.
    unlock();
    texture_registration_shutdown();
    texture_pool_names_shutdown();
    puts("texture registration history, queued continuations and late observations: ok");
}
