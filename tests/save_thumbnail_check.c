// Link with runtime/save_thumbnail.c. Check the source pixels visible at each viewport edge.
#include "gfx.h"
#include <assert.h>
#include <math.h>

static unsigned picture_w, picture_h;
void gfx_game_picture_size(unsigned *w, unsigned *h) {
    *w = picture_w;
    *h = picture_h;
}

static void check(unsigned target, unsigned w, unsigned h, double kept_w, double kept_h) {
    Cpu c = {0};
    uint64_t stack[42] = {0}, manager[140] = {0};
    float bounds[] = {0, 0, 0, 0, 543, 306};
    // The full scene has already been fitted to a fixed-size intermediate texture.
    float scale[] = {543.0f / 1920, 306.0f / 1080};
    picture_w = w;
    picture_h = h;
    const float offset[] = {0, 0};
    c.sp = (uintptr_t)stack;
    c.x[29] = c.sp + 240;
    c.x[19] = (uintptr_t)manager;
    ST32(c.x[19] + 1088, target);
    uint64_t args = c.x[29] - 56;
    ST64(args + 8, (uintptr_t)bounds);
    ST64(args + 16, (uintptr_t)scale);
    ST64(args + 24, (uintptr_t)offset);

    save_thumbnail_crop(&c);
    assert(c.x[2] == args);
    const float *origin = (const float *)(uintptr_t)LD64(args + 24);
    double x = w / (1920.0 * scale[0]), y = h / (1080.0 * scale[1]);
    assert(fabs(-origin[0] * x - (w - kept_w) / 2) < 0.002);
    assert(fabs(-origin[1] * y - (h - kept_h) / 2) < 0.002);
    assert(fabs((543 - origin[0]) * x - (w + kept_w) / 2) < 0.002);
    assert(fabs((306 - origin[1]) * y - (h + kept_h) / 2) < 0.002);
    if (kept_w == w && kept_h == h) assert(origin == offset);
}

int main(void) {
    for (unsigned target = 0; target <= 1; ++target) {
        check(target, 3440, 1440, 2560, 1440);
        check(target, 5120, 1440, 2560, 1440);
        check(target, 1280, 800, 1280, 720);
        check(target, 1440, 1440, 1440, 810);
        check(target, 720, 1280, 720, 405);
        check(target, 1080, 1920, 1080, 607.5);
        check(target, 999, 1133, 999, 561.9375);
        check(target, 2560, 1440, 2560, 1440);
        check(target, 1, 1, 1, 0.5625);
        check(target, 0, 0, 0, 0);
    }
    check(2, 1440, 1440, 1440, 1440);
    return 0;
}
