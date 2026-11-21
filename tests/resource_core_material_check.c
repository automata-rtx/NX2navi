#include <assert.h>
#include <stdint.h>
#include <stdio.h>
typedef uint64_t VkImage;
typedef void *VmaAllocation;
typedef struct { uint32_t width, height; } ImgKey;
#include "../runtime/vk_texture_material.h"

#define OK(call) assert((call) == RC_LIFETIME_OK)
#define IS(call, result) assert((call) == (result))
static RcLifetimeCompletion point(uint64_t domain, uint64_t value) {
    RcLifetimeCompletion done = {0};
    OK(rc_lifetime_completion_add(&done, domain, value));
    return done;
}
static TextureMaterial *destroy(TextureMaterialInstance *instance) {
    assert(texture_material_instance_defer_ready(instance));
    TextureMaterialResidency value = texture_material_instance_defer_begin(instance);
    assert(value.instance == instance && value.image);
    assert(!texture_material_instance_reap(instance));
    texture_material_destroyed(instance);
    return texture_material_instance_reap(instance);
}

static void recording_outlives_logical_retirement(void) {
    TextureMaterialLedger ledger = {0}; TextureMaterial m;
    ImgKey key = {64, 64};
    assert(texture_material_init(&m, &ledger, 11, (void *)(uintptr_t)21, &key, 4096, 1));
    texture_material_set_upload(&m, 9);
    m.ready = 1;
    TextureMaterialUse use = {0};
    OK(texture_material_use_begin(&m, &use));
    TextureMaterialInstance *instance = use.instance;
    assert(texture_material_release(&m, 1));
    TextureMaterialResidency old = texture_material_retire(&m);
    assert(old.image == 11 && old.instance == instance && !m.resident.image && !texture_material_can_free(&m));
    assert(!texture_material_instance_defer_ready(instance));
    assert(ledger.pending_bytes == 4096 && ledger.pending_instances == 1);
    RcLifetimeCompletion done = point(TEXTURE_MATERIAL_GRAPHICS_DOMAIN, 50);
    OK(rc_lifetime_completion_add(&done, 3, 8));
    IS(texture_material_use_submit(&use, 8, &done), RC_LIFETIME_DEPENDENCY);
    OK(texture_material_use_submit(&use, 9, &done));
    RcLifetimeCompletion graphics_only = point(TEXTURE_MATERIAL_GRAPHICS_DOMAIN, 50);
    IS(texture_material_use_complete(&use, 9, &graphics_only), RC_LIFETIME_BUSY);
    assert(!texture_material_instance_defer_ready(instance));
    OK(texture_material_use_complete(&use, 9, &done));
    assert(destroy(instance) == &m);
    assert(texture_material_can_free(&m) && !ledger.pending_bytes && !ledger.pending_instances);
    assert(m.lifetime.state == RC_LIFETIME_VERSION_RETIRED);
}

static void reinstall_keeps_old_instance_separate(void) {
    TextureMaterialLedger ledger = {0}; TextureMaterial m; ImgKey key = {32, 32};
    assert(texture_material_init(&m, &ledger, 1, (void *)(uintptr_t)1, &key, 100, 1));
    m.ready = 1;
    TextureMaterialUse old_use = {0}, new_use = {0};
    OK(texture_material_use_begin(&m, &old_use));
    TextureMaterialResidency old = texture_material_take_residency(&m);
    assert(old.image == 1 && old_use.instance == old.instance);
    TextureMaterialResidency fresh = texture_material_install_residency(&m, 2, (void *)(uintptr_t)2, 200);
    texture_material_set_upload(&m, 7);
    assert(fresh.image == 2 && fresh.id != old.id && m.instances == 2);
    assert(old_use.instance->value.image == 1);
    OK(texture_material_use_begin(&m, &new_use));
    RcLifetimeCompletion done = point(TEXTURE_MATERIAL_GRAPHICS_DOMAIN, 3);
    OK(texture_material_use_submit(&old_use, 0, &done));
    OK(texture_material_use_complete(&old_use, 0, &done));
    assert(!destroy(old.instance));
    assert(m.instances == 1 && m.resident.image == 2 && ledger.pending_bytes == 0);
    OK(texture_material_use_cancel(&new_use));
    assert(texture_material_release(&m, 1));
    TextureMaterialResidency last = texture_material_retire(&m);
    // Deferral owns the actual upload drain even with no graphics consumer for this instance.
    assert(last.instance->lifetime.upload == RC_LIFETIME_UPLOAD_PENDING);
    assert(destroy(last.instance) == &m);
    assert(!ledger.owners && !ledger.resident_bytes && !ledger.pending_bytes);
}

static void finalized_descriptor_needs_exact_recovery(void) {
    TextureMaterialLedger ledger = {0}; TextureMaterial m; ImgKey key = {16, 16};
    assert(texture_material_init(&m, &ledger, 1, (void *)(uintptr_t)1, &key, 100, 1));
    texture_material_set_live(&m, 0);
    assert(m.lifetime.exact_holds == 1);
    assert(!texture_material_take_residency(&m).image && m.resident.image == 1);
    OK(rc_lifetime_recovery_set(&m.lifetime, m.lifetime.id, RC_LIFETIME_RECOVERY_IMMUTABLE));
    TextureMaterialResidency old = texture_material_take_residency(&m);
    assert(old.image && !destroy(old.instance));
    assert(!m.instances && m.lifetime.state == RC_LIFETIME_VERSION_READY);
    assert(texture_material_release(&m, 0));
    assert(!texture_material_retire(&m).image);
    assert(texture_material_can_free(&m) && m.lifetime.state == RC_LIFETIME_VERSION_RETIRED);
}

static void copied_use_does_not_release_another_recording(void) {
    TextureMaterialLedger ledger = {0}; TextureMaterial m; ImgKey key = {16, 16};
    assert(texture_material_init(&m, &ledger, 1, (void *)(uintptr_t)1, &key, 100, 1));
    m.ready = 1;
    TextureMaterialUse a = {0}, b = {0};
    OK(texture_material_use_begin(&m, &a)); OK(texture_material_use_begin(&m, &b));
    TextureMaterialUse copied = a;
    IS(texture_material_use_cancel(&copied), RC_LIFETIME_INVALID);
    assert(m.resident.instance->lifetime.captures == 2);
    OK(texture_material_use_cancel(&a)); OK(texture_material_use_cancel(&b));
    assert(texture_material_release(&m, 1));
    TextureMaterialResidency old = texture_material_retire(&m);
    assert(destroy(old.instance) == &m);
}

int main(void) {
    recording_outlives_logical_retirement();
    reinstall_keeps_old_instance_separate();
    finalized_descriptor_needs_exact_recovery();
    copied_use_does_not_release_another_recording();
    puts("resource core material checks passed");
    return 0;
}
