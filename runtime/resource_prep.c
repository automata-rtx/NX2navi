#include <string.h>

#include "cfg.h"
#include "resource_prep.h"
#include "nvn_exec_plan.h"

// Front ends linked without the Vulkan execution compiler retain their ordinary command representation.
__attribute__((weak)) int nvn_exec_state_compile(const uint8_t *commands, size_t begin, size_t limit,
                                                NvnExecStateEmit emit, void *user, NvnExecStateSpan *out) {
    (void)commands; (void)begin; (void)limit; (void)emit; (void)user; (void)out;
    return 0;
}
__attribute__((weak)) NvnExecFn gfx_exec_prepared_handler(uint32_t opcode) { (void)opcode; return NULL; }
__attribute__((weak)) int nvn_exec_front_compile(const uint8_t *commands, size_t begin, size_t limit,
                                                NvnExecStateEmit emit, void *user, NvnExecFrontSpan *out) {
    (void)commands; (void)begin; (void)limit; (void)emit; (void)user; (void)out;
    return 0;
}
__attribute__((weak)) int nvn_exec_front_compose(const uint8_t *commands, size_t command_bytes,
                                                const NvnExecStep *steps, size_t nsteps,
                                                const uint8_t *primary_state_data, size_t primary_state_bytes,
                                                size_t begin_step, NvnExecStateEmit emit, void *user,
                                                NvnExecFrontComposeSpan *out) {
    (void)commands; (void)command_bytes; (void)steps; (void)nsteps;
    (void)primary_state_data; (void)primary_state_bytes; (void)begin_step;
    (void)emit; (void)user; (void)out;
    return 0;
}

int resource_prep_mode(void) {
    static int cached;
    int value = __atomic_load_n(&cached, __ATOMIC_ACQUIRE);
    if (!value) {
        const char *s = cfg_value("NX2NAVI_RESOURCE_PREP");
        value = s && (!strcmp(s, "check") || !strcmp(s, "2")) ? 3 : s && !strcmp(s, "1") ? 2 : 1;
        __atomic_store_n(&cached, value, __ATOMIC_RELEASE);
    }
    return value - 1;
}
