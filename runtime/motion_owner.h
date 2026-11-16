#pragma once
#include <stdint.h>

#define MOTION_OWNER_BINDINGS 128
enum MotionEquipmentKind { MOTION_SWORD = 1, MOTION_SHIELD, MOTION_SHEATH, MOTION_PLAYER };
typedef struct {
    uint64_t player, actor, link, model, resource, part, lifetime;
    uint32_t generation, kind;
} MotionOwner;
typedef struct {
    uint64_t address, size, observed, buffer;
    MotionOwner owner;
    int ambiguous;
} MotionOwnerBinding;
typedef struct {
    MotionOwnerBinding bindings[MOTION_OWNER_BINDINGS];
    unsigned count, refs;
} MotionOwnerSnapshot;

// Capture identities alongside the submission's uniform bytes, before asynchronous replay.
MotionOwnerSnapshot *motion_owner_acquire(void);
void motion_owner_release(MotionOwnerSnapshot *snapshot);
void motion_owner_shutdown(void);
void motion_owner_present(void);
void motion_owner_forget(uint64_t buffer);
void motion_owner_forget_range(uint64_t address, uint64_t size);
int vk_usnap_motion_owner(uint64_t address, uint64_t size, MotionOwner *out);
int motion_owner_enabled(void);
int motion_owner_player_camera_interpolating(void);
