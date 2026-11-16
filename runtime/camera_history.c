#include "camera_history.h"
#include "motion_owner.h"
#include "nvn_api.h"
#include "platform/platform.h"

// The guest usually prepares the next picture while the renderer records the previous one.
// Keep two adjacent poses and one frame of overlap. A second view or notification check
// in the same present interval makes the association ambiguous.
typedef struct {
    uint64_t frame, history, manager;
    float pose[12];
    int checked, observed, ambiguous, accepted, interpolating;
} CameraSample;

static CameraSample samples[3];
static PlatMutex camera_lock = PLAT_MUTEX_INIT;
static _Thread_local int accepted_before_check;
static uint64_t enabled_at;

static uint64_t next_present(void) {
    return __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
}

void camera_history_enable(int on) {
    __atomic_store_n(&enabled_at, on ? next_present() : 0, __ATOMIC_RELAXED);
}

// Called with camera_lock held. All producers take the frame number under the same lock.
static CameraSample *writing(void) {
    uint64_t frame = next_present();
    CameraSample *s = &samples[frame % 3];
    if (s->frame != frame) *s = (CameraSample){.frame = frame};
    return s;
}

// Ocarina's view update stores the discontinuity reason returned by the game. x20 is
// the selected view packet. Its matrix is the one copied into the scene uniform.
void camera_history_view(Cpu *c) {
    unsigned reason = (uint8_t)c->x[0];
    ST8(c->x[19] + 0x12, reason);
    if (!__atomic_load_n(&enabled_at, __ATOMIC_RELAXED)) return;
    uint64_t frame = next_present();
    int blending = !reason && motion_owner_player_camera_interpolating();
    float pose[12];
    memcpy(pose, (const void *)(uintptr_t)(c->x[20] + 0xa8), sizeof pose);
    plat_mutex_lock(&camera_lock);
    CameraSample *s = writing();
    s->ambiguous |= s->observed || s->frame != frame;
    s->observed = 1;
    s->history = c->x[19];
    memcpy(s->pose, pose, sizeof pose);
    s->interpolating = blending;
    plat_mutex_unlock(&camera_lock);
}

// Acceptance and completion are in the same guest routine. Carry acceptance through
// completion even when the two instructions fall on opposite sides of a present.
void camera_history_accepted(Cpu *c) {
    c->x[8] = LD64(c->x[20] + 0xbf0);
    if (!__atomic_load_n(&enabled_at, __ATOMIC_RELAXED)) return;
    accepted_before_check = 1;
    plat_mutex_lock(&camera_lock);
    writing()->accepted = 1;
    plat_mutex_unlock(&camera_lock);
}

void camera_history_checked(Cpu *c) {
    c->x[8] = LD8(c->x[20] + 0xe42);
    if (!__atomic_load_n(&enabled_at, __ATOMIC_RELAXED)) return;
    plat_mutex_lock(&camera_lock);
    CameraSample *s = writing();
    s->ambiguous |= s->checked;
    s->checked = 1;
    s->manager = c->x[20];
    s->accepted |= accepted_before_check;
    accepted_before_check = 0;
    plat_mutex_unlock(&camera_lock);
}

static int complete(const CameraSample *s, uint64_t frame) {
    return s->frame == frame && s->observed && s->checked && !s->ambiguous;
}

int camera_history_continuous(uint64_t frame, const float previous[12], const float current[12]) {
    uint64_t first = __atomic_load_n(&enabled_at, __ATOMIC_RELAXED);
    if (frame < 3 || !first || !RT_V111) return 0;
    int keep = 0;
    plat_mutex_lock(&camera_lock);
    for (uint64_t age = 0; age <= 1; ++age) {
        uint64_t f = frame - age;
        const CameraSample *s = &samples[f % 3], *p = &samples[(f - 1) % 3];
        if (f - 1 > first && complete(s, f) && complete(p, f - 1) && s->history == p->history && s->manager == p->manager &&
            s->interpolating && !s->accepted && !memcmp(s->pose, current, sizeof s->pose) &&
            !memcmp(p->pose, previous, sizeof p->pose)) {
            keep = 1;
            break;
        }
    }
    const CameraSample *latest = &samples[frame % 3];
    if (latest->frame == frame && latest->accepted) keep = 0;
    plat_mutex_unlock(&camera_lock);
    return keep;
}
