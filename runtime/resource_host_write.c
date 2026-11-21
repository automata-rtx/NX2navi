#include "resource_host_write.h"
#include "host.h"

#include <stdlib.h>

uint32_t resource_host_interval_slice(const RcInterval *source, uint32_t count, uint64_t address, uint64_t size,
                                      RcInterval **out) {
    *out = NULL;
    if (!source || !size || size > UINT64_MAX - address) return 0;
    uint64_t end = address + size, covered = address;
    uint32_t first = 0;
    while (first < count && source[first].address <= address &&
           source[first].range.size <= address - source[first].address) ++first;
    uint32_t i = first;
    for (; i < count && covered < end; i++) {
        RcInterval span = source[i];
        if (!span.mapping || span.range.backing <= 1 || !rc_range_valid(span.range) ||
            span.range.size > UINT64_MAX - span.address || span.address > covered ||
            (i != first && span.address != covered) ||
            span.address + span.range.size <= covered) return 0;
        covered = span.address + span.range.size;
    }
    if (covered < end) return 0;
    uint32_t used = i - first;
    RcInterval *result = calloc(used, sizeof *result);
    if (!result) return 0;
    for (i = 0; i < used; i++) {
        RcInterval span = source[first + i];
        uint64_t lo = span.address > address ? span.address : address;
        uint64_t hi = span.address + span.range.size;
        if (hi > end) hi = end;
        result[i] = (RcInterval){lo, span.mapping, {span.range.backing, span.range.offset + (lo - span.address), hi - lo}};
    }
    *out = result;
    return used;
}

uint32_t resource_host_current_intervals(uint64_t address, uint64_t size, RcInterval **out) {
    *out = NULL;
    MemBackingCapture *capture;
    if (mem_backing_capture(address, size, &capture) != RC_BACKING_OK) return 0;
    uint32_t count;
    const RcInterval *source = mem_backing_intervals(capture, &count);
    uint32_t copied = resource_host_interval_slice(source, count, address, size, out);
    mem_backing_release(&capture);
    return copied;
}

ResourceHostWrite resource_host_write_prepare(RcInterval *intervals, uint32_t count, RcRegistryWriterKind kind,
                                               int declared) {
    ResourceHostWrite write = {.intervals = intervals, .count = count};
    if (declared) return write;
    RcRegistry *registry = rc_registry_runtime();
    if (count) {
        RcRegistryWriteExtent *extents = calloc(count, sizeof *extents);
        RcRegistryWrite *tickets = calloc(count, sizeof *tickets);
        if (extents && tickets) {
            for (uint32_t i = 0; i < count; i++) extents[i] = (RcRegistryWriteExtent){intervals[i].range, kind};
            RcRegistryResult result = rc_registry_write_begin_extents(registry, extents, count,
                                                                       (RcRegistryPosition){0}, tickets);
            if (result == RC_REGISTRY_OK) {
                write.tickets = tickets;
                tickets = NULL;
            }
        }
        free(tickets);
        free(extents);
        if (write.tickets) return write;
    }
    RcRegistryWriterKind guard_kind = kind == RC_REGISTRY_IMAGE ? RC_REGISTRY_UNKNOWN : kind;
    RcRegistryGuardReason reason = count ? RC_REGISTRY_GUARD_RESERVATION_REFUSED : RC_REGISTRY_GUARD_MAPPING_UNAVAILABLE;
    if (rc_registry_write_guard_begin_reason_at(registry, guard_kind, (RcRegistryPosition){0}, reason,
                                               &write.guard) != RC_REGISTRY_OK)
        rc_registry_fail_closed(registry);
    return write;
}

static void complete_guard(RcRegistry *registry, const ResourceHostWrite *write, uint64_t fence, int visible) {
    if (rc_registry_write_submit(registry, write->guard, fence) != RC_REGISTRY_OK) {
        rc_registry_write_guard_accumulate(registry, write->guard);
        return;
    }
    RcSourceRange *parts = write->count ? calloc(write->count, sizeof *parts) : NULL;
    if (parts) {
        for (uint32_t i = 0; i < write->count; i++) parts[i] = write->intervals[i].range;
        RcRegistryWriteCompletion completion = {write->guard, parts, visible ? NULL : parts,
                                                 write->count, visible ? 0 : write->count};
        RcRegistryResult result = rc_registry_write_complete_extents(registry, &completion, 1);
        free(parts);
        if (result == RC_REGISTRY_OK) return;
    }
    // The GPU may have affected storage outside any reconstructible vector. Preserve the guard as foreign.
    rc_registry_write_guard_accumulate(registry, write->guard);
}

static void complete_tickets(RcRegistry *registry, const ResourceHostWrite *write, uint64_t fence, int visible) {
    if (rc_registry_write_submit_extents(registry, write->tickets, write->count, fence) != RC_REGISTRY_OK) {
        rc_registry_write_abandon_extents(registry, write->tickets, write->count);
        return;
    }
    RcRegistryWriteCompletion *completions = calloc(write->count, sizeof *completions);
    if (completions) {
        for (uint32_t i = 0; i < write->count; i++) {
            const RcSourceRange *range = &write->intervals[i].range;
            completions[i] = (RcRegistryWriteCompletion){write->tickets[i], range, visible ? NULL : range, 1, !visible};
        }
        RcRegistryResult result = rc_registry_write_complete_extents(registry, completions, write->count);
        free(completions);
        if (result == RC_REGISTRY_OK) return;
    }
    for (uint32_t i = 0; i < write->count; i++) rc_registry_write_complete(registry, write->tickets[i], 0);
}

void resource_host_write_dispose(ResourceHostWrite *write, uint64_t fence, int may_have_executed, int cpu_visible) {
    RcRegistry *registry = rc_registry_runtime();
    if (write->tickets) {
        if (fence) complete_tickets(registry, write, fence, cpu_visible);
        else if (may_have_executed) rc_registry_write_abandon_extents(registry, write->tickets, write->count);
        else rc_registry_write_cancel_extents(registry, write->tickets, write->count);
    } else if (write->guard.id) {
        if (fence) complete_guard(registry, write, fence, cpu_visible);
        else if (may_have_executed) rc_registry_write_guard_accumulate(registry, write->guard);
        else rc_registry_write_cancel(registry, write->guard);
    }
    free(write->tickets);
    free(write->intervals);
    *write = (ResourceHostWrite){0};
}
