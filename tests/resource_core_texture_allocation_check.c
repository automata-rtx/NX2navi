#include <assert.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../runtime/resource_core_budget.h"

typedef uint64_t VkImage;
typedef void *VmaAllocation;
typedef int VkResult;
enum { VK_SUCCESS = 0, VK_NOT_READY = 1, VK_ERROR_OUT_OF_HOST_MEMORY = -1,
       VK_ERROR_OUT_OF_DEVICE_MEMORY = -2, VK_ERROR_FORMAT_NOT_SUPPORTED = -11 };
typedef struct { void *vma; } VkCtx;
typedef struct { struct { uint32_t width, height, depth; } extent; int format; } VkImageCreateInfo;
typedef struct { unsigned flags; } VmaAllocationCreateInfo;
typedef struct { RcBudgetProducer producer; int required; } RcVmaRequest;

static VkResult responses[2];
static RcAdmissionStatus reasons[2];
static unsigned attempts, reclaims, refreshes, fatals;
static int expected_required;
static uint64_t reclaim_report;
static char fatal_message[1024];
static jmp_buf stopped;

static VkResult resource_core_vma_create_image(const RcVmaRequest *request, void *allocator,
                                               const VkImageCreateInfo *image, const VmaAllocationCreateInfo *allocation,
                                               VkImage *out, VmaAllocation *memory, void *info, RcAdmissionStatus *status) {
    (void)allocator; (void)image; (void)allocation; (void)info;
    assert(attempts < 2 && request->producer == RC_BUDGET_TEXTURE && request->required == expected_required);
    unsigned i = attempts++;
    *out = responses[i] == VK_SUCCESS ? 42 : 0;
    *memory = responses[i] == VK_SUCCESS ? (void *)(uintptr_t)1 : NULL;
    *status = reasons[i];
    return responses[i];
}
static uint64_t vk_mem_give_back(void) { reclaims++; return reclaim_report; }
void resource_core_budget_refresh(void) { refreshes++; }
void resource_core_budget_snapshot(RcBudgetSnapshot *out) { memset(out, 0, sizeof *out); }
static void rt_fatal(const char *format, ...) {
    va_list ap;
    va_start(ap, format);
    vsnprintf(fatal_message, sizeof fatal_message, format, ap);
    va_end(ap);
    fatals++;
    longjmp(stopped, 1);
}
#include "../runtime/vk_texture_allocation.h"

static void start(int required, VkResult first, RcAdmissionStatus reason, VkResult second) {
    expected_required = required;
    responses[0] = first; responses[1] = second;
    reasons[0] = reason; reasons[1] = second == VK_SUCCESS ? RC_ADMISSION_OK : reason;
    attempts = reclaims = refreshes = fatals = 0;
    reclaim_report = 64ull << 20;
    fatal_message[0] = 0;
}
static VkResult attempt(int *capacity) {
    VkCtx c = {0};
    VkImageCreateInfo ci = {.extent = {256, 128, 1}, .format = 37};
    VmaAllocationCreateInfo ai = {0};
    VkImage image = 0;
    VmaAllocation memory = NULL;
    VkResult result = texture_image_allocate(&c, &ci, &ai, &image, &memory, expected_required, "test texture", capacity);
    assert((result == VK_SUCCESS) == (image != 0 && memory != NULL));
    return result;
}

int main(void) {
    int capacity;
    start(1, VK_SUCCESS, RC_ADMISSION_OK, VK_SUCCESS);
    assert(attempt(&capacity) == VK_SUCCESS && !capacity);
    assert(attempts == 1 && !reclaims && !refreshes && !fatals);

    start(1, VK_NOT_READY, RC_ADMISSION_REFUSED, VK_SUCCESS);
    assert(attempt(&capacity) == VK_SUCCESS && !capacity);
    assert(attempts == 2 && reclaims == 1 && refreshes == 1 && !fatals);

    // Reporting reclaimed bytes does not grant admission. Unchanged refusal terminates before a caller can
    // publish an empty image or enter its native-error hold, even though reclaim reported 64 MiB.
    start(1, VK_NOT_READY, RC_ADMISSION_REFUSED, VK_NOT_READY);
    if (!setjmp(stopped)) { (void)attempt(&capacity); assert(!"required refusal returned to a substitution path"); }
    assert(attempts == 2 && reclaims == 1 && refreshes == 1 && fatals == 1);
    assert(strstr(fatal_message, "required test texture") && strstr(fatal_message, "no substitute texture"));

    start(0, VK_NOT_READY, RC_ADMISSION_REFUSED, VK_SUCCESS);
    assert(attempt(&capacity) == VK_NOT_READY && capacity);
    assert(attempts == 1 && !reclaims && !refreshes && !fatals); // optional narrowing/speculation keeps its prior owner

    start(0, VK_ERROR_OUT_OF_DEVICE_MEMORY, RC_ADMISSION_OK, VK_SUCCESS);
    assert(attempt(&capacity) == VK_ERROR_OUT_OF_DEVICE_MEMORY && capacity);
    assert(attempts == 1 && !reclaims && !fatals);

    start(1, VK_NOT_READY, RC_ADMISSION_UNSUPPORTED, VK_NOT_READY);
    if (!setjmp(stopped)) { (void)attempt(&capacity); assert(!"missing telemetry waited indefinitely"); }
    assert(attempts == 2 && reclaims == 1 && refreshes == 1 && fatals == 1);

    // Unsupported content is the caller's existing format-handling case, not capacity pressure.
    start(1, VK_ERROR_FORMAT_NOT_SUPPORTED, RC_ADMISSION_INVALID, VK_SUCCESS);
    assert(attempt(&capacity) == VK_ERROR_FORMAT_NOT_SUPPORTED && !capacity);
    assert(attempts == 1 && !reclaims && !fatals);
    puts("resource_core_texture_allocation_check: passed");
    return 0;
}
