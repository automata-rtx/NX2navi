// Link with runtime/camera_history.c. Simulate the producer one frame ahead of recording.
#include "camera_history.h"
#include "nvn_api.h"
#include "platform/platform.h"
#include <assert.h>

GfxStats gfx_counters;
const int rt_game_version = 111;
static int blending;
int motion_owner_player_camera_interpolating(void) { return blending; }
void plat_mutex_lock(PlatMutex *m) { (void)m; }
void plat_mutex_unlock(PlatMutex *m) { (void)m; }

static Cpu cpu;
static unsigned char manager[0xe50], history[32], packet[0xe0];
static const float a[12] = {1}, b[12] = {2};

static void publish(unsigned frame, const float *pose, int interpolating, unsigned reason, int checked, int cut) {
    gfx_counters.frames = frame - 1;
    cpu.x[20] = (uintptr_t)manager;
    if (cut) {
        ST64(cpu.x[20] + 0xbf0, 0x123456789ull);
        camera_history_accepted(&cpu);
        assert(cpu.x[8] == 0x123456789ull);
    }
    if (checked) {
        manager[0xe42] = 42;
        camera_history_checked(&cpu);
        assert(cpu.x[8] == 42);
    }
    blending = interpolating;
    cpu.x[0] = reason;
    cpu.x[19] = (uintptr_t)history;
    cpu.x[20] = (uintptr_t)packet;
    memcpy(packet + 0xa8, pose, 48);
    history[0x12] = 0xff;
    camera_history_view(&cpu);
    assert(history[0x12] == reason && cpu.x[0] == reason);
}

int main(void) {
    camera_history_enable(1);
    publish(10, a, 0, 0, 1, 0);
    publish(11, b, 1, 0, 1, 0);
    assert(camera_history_continuous(12, a, b));
    assert(!camera_history_continuous(12, b, a));
    assert(!camera_history_continuous(13, a, b));
    publish(12, a, 1, 4, 1, 0);
    assert(!camera_history_continuous(13, b, a));
    publish(13, b, 1, 0, 1, 1);
    assert(!camera_history_continuous(14, a, b));
    publish(14, a, 0, 0, 1, 0);
    assert(!camera_history_continuous(15, b, a));
    publish(15, b, 1, 0, 0, 0);
    assert(!camera_history_continuous(16, a, b));
    publish(16, a, 1, 0, 1, 0);
    assert(!camera_history_continuous(17, b, a));
    publish(17, b, 1, 0, 1, 0);
    publish(17, b, 1, 0, 1, 0);
    assert(!camera_history_continuous(18, a, b));
    camera_history_enable(0);
    publish(20, a, 1, 0, 1, 0);
    publish(21, b, 1, 0, 1, 0);
    assert(!camera_history_continuous(22, a, b));
    camera_history_enable(1);
    assert(!camera_history_continuous(22, a, b));
    return 0;
}
