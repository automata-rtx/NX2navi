// Command-scoped image write reservations. Preparation never claims that a GPU command ran. Each nested helper
// owns its reservations; skipped work cancels, emitted work transfers to persistent ownership or fence completion.
#pragma once
#include <stdint.h>
#include "resource_core_registry.h"

typedef struct ResourceImageWrite ResourceImageWrite;
typedef struct ResourceImageScope {
    struct ResourceImageScope *parent;
    ResourceImageWrite *writes;
    uint64_t fence;
    int active;
} ResourceImageScope;

void resource_image_scope_begin(ResourceImageScope *scope, uint64_t fence);
void resource_image_scope_end(ResourceImageScope *scope);
// Independent synchronous upload. Emitted work without a durable image hold retains foreign coverage.
// A recorded upload may already have reached the queue even if its wait encounters device loss.
void resource_image_scope_end_upload(ResourceImageScope *scope);
void resource_image_prepare(uint64_t image, uint64_t addr, uint64_t size);
// The image association lends its immutable captured logical host storage to this operation.
void resource_image_prepare_mapped(uint64_t image, uint64_t addr, uint64_t size, const RcInterval *, uint32_t);
// A pinned image matches its exact declared instance and mapped backing. A detached private image can carry
// a guest effect when it already declared that projection while live.
void resource_image_prepare_pinned(uint64_t image, uint64_t addr, uint64_t size, const RcInterval *, uint32_t, int detached);
int resource_image_pinned_projected(uint64_t image, uint64_t addr);
void resource_image_written(uint64_t image, uint64_t addr);
void resource_image_owned(uint64_t image, uint64_t addr, RcRegistryWrite hold);
void resource_image_owned_mapped(uint64_t image, uint64_t addr, const RcInterval *, const RcRegistryWrite *, uint32_t count);

// GNU cleanup covers early returns as well as the normal exit. It is used only for lexical command scopes.
#define RESOURCE_IMAGE_SCOPE(name, submission) \
    ResourceImageScope name __attribute__((cleanup(resource_image_scope_end))) = {0}; \
    resource_image_scope_begin(&name, (submission))
