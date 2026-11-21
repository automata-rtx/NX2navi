#include "resource_image_write.h"
#include "resource_core_registry.h"
#include "resource_command_effects.h"
#include "resource_host_write.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"
#include <stdlib.h>

struct ResourceImageWrite {
    ResourceImageWrite *next;
    uint64_t image, addr, size, fence;
    RcRegistryWrite ticket;
    ResourceHostWrite host;
    int recorded, owned, canonical_owned, pinned;
};
static __thread ResourceImageScope *image_scope;

void resource_image_scope_begin(ResourceImageScope *scope, uint64_t fence) {
    if (!scope || scope->active) rt_fatal("resource image: invalid scope begin");
    *scope = (ResourceImageScope){.parent = image_scope, .fence = fence, .active = 1};
    image_scope = scope;
}

enum ImagePreparation { IMAGE_COMPATIBILITY, IMAGE_PINNED_LIVE, IMAGE_PINNED_DETACHED };

static void image_prepare(uint64_t image, uint64_t addr, uint64_t size,
                          const RcInterval *captured, uint32_t captured_count, enum ImagePreparation mode) {
    // Resource look-ahead outside a recording scope does not acquire writable execution authority.
    if (!image_scope || !image || !size) return;
    // Ordered declarations still need the actual destination of GPU-only output. Otherwise an emitted command
    // with a conservative declaration retains that whole declaration instead of its precise destination. Scope
    // allocation is deduplicated separately from each command's effect notification. A GPU-only scope record
    // carries no CPU reservation, but still guards notifications against consuming an outer scope's preparation.
    int compatibility = mode == IMAGE_COMPATIBILITY, pinned = mode >= IMAGE_PINNED_LIVE;
    int declared = image_scope->fence && compatibility && resource_effects_image_prepare(image, addr, size);
    int canonical_declared = 0;
    if (pinned && image_scope->fence)
        canonical_declared = resource_effects_image_prepare_pinned(image, captured, captured_count);
    if (mode == IMAGE_PINNED_DETACHED && !canonical_declared) return;
    if (pinned && canonical_declared != 1)
        rt_fatal("resource image: pinned destination differs from its declared image and backing");
    RcInterval *intervals = captured && captured_count ? calloc(captured_count, sizeof *intervals) : NULL;
    uint32_t count = intervals ? captured_count : 0;
    if (count) memcpy(intervals, captured, (size_t)count * sizeof *intervals);
    if (!pinned) canonical_declared = image_scope->fence && count && resource_effects_image_prepare_intervals(image, intervals, count);
    if (pinned && !count) rt_fatal("resource image: pinned backing reservation allocation failed");
    for (ResourceImageWrite *w = image_scope->writes; w; w = w->next) {
        if (w->image == image && w->addr == addr && w->size == size) {
            w->pinned |= pinned;
            free(intervals);
            return;
        }
    }
    ResourceImageWrite *w = calloc(1, sizeof *w);
    if (!w) rt_fatal("resource image: write reservation allocation failed");
    w->image = image; w->addr = addr; w->size = size; w->fence = image_scope->fence; w->pinned = pinned;
    if (compatibility && !declared && !nvn_gpu_only(addr))
        rc_registry_write_begin(rc_registry_runtime(), (RcSourceRange){1, addr, size}, RC_REGISTRY_IMAGE, &w->ticket);
    if (captured_count || (compatibility && !nvn_gpu_only(addr)))
        w->host = resource_host_write_prepare(intervals, count, RC_REGISTRY_IMAGE, canonical_declared);
    w->next = image_scope->writes;
    image_scope->writes = w;
}

void resource_image_prepare_mapped(uint64_t image, uint64_t addr, uint64_t size,
                                   const RcInterval *captured, uint32_t captured_count) {
    image_prepare(image, addr, size, captured, captured_count, IMAGE_COMPATIBILITY);
}

void resource_image_prepare_pinned(uint64_t image, uint64_t addr, uint64_t size,
                                  const RcInterval *captured, uint32_t captured_count, int detached) {
    if (!image_scope || !image_scope->fence || !image || !size || !captured || !captured_count)
        rt_fatal("resource image: missing pinned transfer scope or backing");
    image_prepare(image, addr, size, captured, captured_count, detached ? IMAGE_PINNED_DETACHED : IMAGE_PINNED_LIVE);
}

int resource_image_pinned_projected(uint64_t image, uint64_t addr) {
    for (const ResourceImageWrite *w = image_scope ? image_scope->writes : NULL; w; w = w->next)
        if (w->image == image && w->addr == addr && w->pinned) return 1;
    return 0;
}

void resource_image_prepare(uint64_t image, uint64_t addr, uint64_t size) {
    resource_image_prepare_mapped(image, addr, size, NULL, 0);
}

void resource_image_written(uint64_t image, uint64_t addr) {
    // Search the nearest scope only. A nested helper cannot consume a reservation belonging to its caller.
    if (!image_scope) return;
    for (ResourceImageWrite *w = image_scope->writes; w; w = w->next)
        if (w->image == image && w->addr == addr) {
            w->recorded = 1;
            w->owned = w->canonical_owned = 0;
            if (image_scope->fence) {
                resource_effects_image_written(image, addr);
                for (uint32_t i = 0; i < w->host.count; i++)
                    resource_effects_image_written_interval(image, w->host.intervals[i]);
            }
        }
}

void resource_image_owned(uint64_t image, uint64_t addr, RcRegistryWrite hold) {
    if (!image_scope) return;
    for (ResourceImageWrite *w = image_scope->writes; w; w = w->next)
        if (w->recorded && w->image == image && w->addr == addr) {
            w->owned = 1;
            if (image_scope->fence) resource_effects_image_owned(image,addr,hold);
        }
}

void resource_image_owned_mapped(uint64_t image, uint64_t addr, const RcInterval *intervals,
                                const RcRegistryWrite *holds, uint32_t count) {
    if (!image_scope || !intervals || !holds || !count) return;
    RcRegistry *core = rc_registry_runtime();
    for (ResourceImageWrite *w = image_scope->writes; w; w = w->next) {
        if (!w->recorded || w->image != image || w->addr != addr || w->host.count != count) continue;
        uint32_t i = 0;
        for (; i < count; i++) {
            RcInterval actual = w->host.intervals[i], retained = intervals[i];
            if (actual.address != retained.address || actual.mapping != retained.mapping ||
                !rc_range_equal(actual.range, retained.range) ||
                !rc_registry_image_hold_covers(core, holds[i], retained.range) ||
                !rc_registry_image_binding_current(core, (RcRegistryImageBinding){image, holds[i]})) break;
        }
        if (i != count) continue;
        w->canonical_owned = 1;
        if (image_scope->fence)
            for (i = 0; i < count; i++) resource_effects_image_owned_interval(image, intervals[i], holds[i]);
    }
}

static void image_write_completed(void *arg) {
    ResourceImageWrite *w = arg;
    resource_host_write_dispose(&w->host, w->fence, 1, 0);
    if (w->ticket.id) {
        rc_registry_write_submit(rc_registry_runtime(), w->ticket, w->fence);
        // The image may no longer have a legacy Tex association. The range obligation survives that retirement;
        // only real readback/replacement may restore CPU authority.
        rc_registry_write_complete(rc_registry_runtime(), w->ticket, 0);
    }
    free(w);
}

static void image_write_canceled(void *arg, int may_have_executed) {
    ResourceImageWrite *w = arg;
    resource_host_write_dispose(&w->host, 0, may_have_executed, 0);
    if (w->ticket.id) rc_registry_write_abandon(rc_registry_runtime(), w->ticket, may_have_executed);
    free(w);
}

void resource_image_scope_end(ResourceImageScope *scope) {
    if (!scope || !scope->active) return;
    if (image_scope != scope) rt_fatal("resource image: scopes must end in nesting order");
    image_scope = scope->parent;
    scope->active = 0;
    ResourceImageWrite *w = scope->writes;
    scope->writes = NULL;
    while (w) {
        ResourceImageWrite *next = w->next;
        if (!w->recorded || w->owned) {
            if (w->ticket.id) rc_registry_write_cancel(rc_registry_runtime(), w->ticket);
            w->ticket = (RcRegistryWrite){0};
        }
        if (!w->recorded || w->canonical_owned) resource_host_write_dispose(&w->host, 0, 0, 0);
        if (!w->recorded || (!w->ticket.id && !w->host.tickets && !w->host.guard.id)) {
            free(w);
        } else {
            if (!w->fence) rt_fatal("resource image: recorded write has no completion fence");
            vk_completion_add_owned(w->fence, image_write_completed, image_write_canceled, w);
        }
        w = next;
    }
}

void resource_image_scope_end_upload(ResourceImageScope *scope) {
    if (!scope || !scope->active || image_scope != scope || scope->fence)
        rt_fatal("resource image: invalid independent upload scope");
    image_scope = scope->parent;
    scope->active = 0;
    ResourceImageWrite *w = scope->writes;
    scope->writes = NULL;
    while (w) {
        ResourceImageWrite *next = w->next;
        if (w->owned && w->ticket.id) {
            rc_registry_write_cancel(rc_registry_runtime(), w->ticket);
            w->ticket = (RcRegistryWrite){0};
        }
        if (w->canonical_owned) resource_host_write_dispose(&w->host, 0, 0, 0);
        image_write_canceled(w, w->recorded);
        w = next;
    }
}
