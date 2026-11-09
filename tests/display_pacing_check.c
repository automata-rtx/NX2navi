// Exercise the real pacer and the rate shown in the overlay with controlled monitor and limiter inputs.
// Link with runtime/frame_rate.c using the runtime build's includes.
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "vk_dlssg.h"
#include "vk_fg.h"
#include "platform/platform.h"

static double monitor;
static const char *limit = "display";
static unsigned version;
static int64_t rtss_request;
static int fsr, dlss, driver_generating, external_limiter;

const int rt_game_version = 111;
unsigned rt_trace;
uint64_t rt_guest_start_ns;
uint64_t kern_now_ns(void) { return 0; }
int rt_patch_enabled(int set) { (void)set; return 0; }
void *rt_patch_page(void) { return NULL; }
void rt_patch_disable(int set) { (void)set; }
int plat_mem_read(uint64_t addr, void *out, size_t n) { (void)addr; (void)out; (void)n; return 0; }
int plat_mem_readable(uint64_t addr, uint64_t n) { (void)addr; (void)n; return 0; }

double gfx_window_refresh_hz(void) { return monitor; }
const char *cfg_get(const char *name) { return strcmp(name, "NX2NAVI_FPS_LIMIT") == 0 ? limit : NULL; }
unsigned cfg_version(void) { return version; }
int rt_patch_set(const char *name) { (void)name; return -1; }
void rt_patch_switch(int set, int on) { (void)set; (void)on; }
void rt_log(const char *fmt, ...) { (void)fmt; }
int vk_fg_wanted(void) { return fsr; }
int vk_dlssg_wanted(void) { return dlss; }
int vk_fg_generating(void) { return fsr && driver_generating; }
void rtss_limit_want(int64_t mhz) { rtss_request = mhz; }
int rtss_limit_active(void) { return external_limiter; }

static void check_hz(double measured, double expected) {
    monitor = measured;
    assert(frame_step_cap_ns() == (uint64_t)llround(1e9 / expected));
    assert(frame_step_limit_hz() == expected);
    assert(rtss_request == llround(expected * 1000));
}

int main(int argc, char **argv) {
    if (argc == 2) {
        fsr = strcmp(argv[1], "fsr3") == 0;
        dlss = strcmp(argv[1], "dlss") == 0;
        assert(fsr || dlss);
        monitor = 120;
        external_limiter = 1;
        assert(frame_step_cap_ns() == 8333333 && rtss_request == -1);
        driver_generating = 1;
        assert(frame_step_cap_ns() == (fsr ? 16666667 : 8333333) && rtss_request == -1);
        limit = "unlimited";
        ++version;
        assert(frame_step_cap_ns() == 0 && rtss_request == -1);
        return 0;
    }
    check_hz(0, 60);
    check_hz(24, 30);
    check_hz(25, 30);
    check_hz(29.97, 30);
    check_hz(30, 30);
    check_hz(59.94, 59.94);
    check_hz(144, 144);
    external_limiter = 1;
    assert(frame_step_cap_ns() == 0 && rtss_request == 144000);
    external_limiter = 0;
    limit = "120";
    ++version;
    check_hz(24, 120);
    limit = "unlimited";
    ++version;
    assert(frame_step_cap_ns() == 0 && frame_step_limit_hz() == 0 && rtss_request == 0);
    return 0;
}
