"""Exercise the graphics input frontier in the actual replay owner."""

import os
from pathlib import Path
import subprocess
import tempfile

from ordered_dispatch_preparation_check import REPLAY, between, function


STUBS = r'''
#include <assert.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct { uint32_t op, payload; } op_hdr;
typedef struct { uint64_t serial; } HostProgram;
typedef struct { uint64_t fence, resource_op, resource_next; const HostProgram *prog; } VkExec;
typedef struct { uint64_t scene, ordinal; } RcRegistryPosition;
typedef struct { unsigned selected; } VkExecGraphicsDraw;
typedef struct { RcRegistryPosition read; } VkExecGpaSampleScope;
typedef enum {
    VK_EXEC_GRAPHICS_COMPAT, VK_EXEC_GRAPHICS_NOOP, VK_EXEC_GRAPHICS_READY,
    VK_EXEC_GRAPHICS_NEEDS_ORDERED, VK_EXEC_GRAPHICS_STOPPED, VK_EXEC_GRAPHICS_UNAVAILABLE,
    VK_EXEC_GRAPHICS_INVALIDATED,
} VkExecGraphicsResult;
typedef struct { uint64_t backing, offset, size; } RcSourceRange;
typedef struct { uint64_t address, mapping; RcSourceRange range; } RcInterval;
typedef struct { uint64_t id; uint32_t index; } RcRegistryWrite;
typedef struct { uint64_t image; RcRegistryWrite hold; } RcRegistryImageBinding;
typedef void *VkImage, *VkBuffer, *VkPipeline;
typedef struct VkGraphicsPipelineRevision VkGraphicsPipelineRevision;
typedef int RcRegistryResult, VkResult, VkExecHostInputResult;
''' + between(REPLAY.with_name("texture_source.h").read_text(),
              "typedef enum {\n    TEX_SOURCE_TRACKING_NONE,", "} TextureSourceTrackingFailure;") + "} TextureSourceTrackingFailure;\n" + between(
    REPLAY.with_name("texture_source.h").read_text(), "static inline const char *texture_source_tracking_failure_name(",
    "// A producer-identified freeze") + between(REPLAY.with_name("resource_core_registry.h").read_text(),
              "typedef enum {\n    RC_NATIVE_IMAGE_READ_OK,", "// Optional maximum authored geometry") + between(
    REPLAY.with_name("texture_resource.h").read_text(), "typedef enum {\n    VK_TEXTURE_SAMPLE_READY,", "} VkTextureSampleRefusal;") + "} VkTextureSampleRefusal;\n" + between(
    REPLAY.with_name("vk.h").read_text(), "typedef struct {\n    const char *gate;\n    RcRegistryResult registry;", "} VkMemNativeRmwBitmapFailure;") + "} VkMemNativeRmwBitmapFailure;\n" + r'''
typedef struct VkExecPreparationFailure VkExecPreparationFailure;
''' + between(REPLAY.with_name("vk_exec.h").read_text(), "typedef enum {\n    VK_EXEC_PREPARATION_NONE,", "void vk_exec_graphics_failure_set(") + r'''
typedef enum {
    VK_SYNC_PROGRESS_UNAVAILABLE, VK_SYNC_PROGRESS_COMPLETE, VK_SYNC_PROGRESS_STOPPED,
    VK_SYNC_PROGRESS_STOPPED_AFTER_CUT,
} VkSyncProgressResult;
enum { OP_DRAW = 1, OP_DRAW_IDX, OP_DRAW_IDX_IND, OP_DRAW_MESH, OP_DRAW_MESH_IND };
enum { RC_REGISTRY_OK = 0, PW_OP = 3 };
static VkExec *active;
static VkExecGraphicsDraw earlier = {11}, blocked = {12};
static VkExecPreparationFailure graphics_failure;
static const VkExecPreparationFailure *vk_exec_graphics_failure(const VkExecGraphicsDraw *draw) {
    return draw == &blocked && graphics_failure.owner ? &graphics_failure : NULL;
}
static char failure_text[448], storage_text[1024];
static unsigned seals, probes, ordered_calls, drains, declarations, cancellations;
static unsigned physical_drops, prefix_settlements, replans, pending_cancellations;
static int stopping, lost, failed, prior_writer, unsafe, stale_after_cut, stop_replan, repend;
static VkSyncProgressResult settle_result;
static VkExecGpaSampleScope sample_scope;
static enum { COLD, WARM_STALE, PROVIDER_STALE, STALE_AGAIN, MISSING_MANIFEST, PRESSURE } mode;

static int vk_sync_stopping(void) { return stopping; }
static int vk_device_is_lost(void) { return lost; }
static int vk_exec_recording_failed(void) { return failed; }
static int *rc_registry_runtime(void) { return &failed; }
static RcRegistryPosition vk_exec_resource_position(void) {
    return (RcRegistryPosition){active->fence, active->resource_op * 2};
}
static int rc_registry_position_compare(RcRegistryPosition a, RcRegistryPosition b) {
    return a.scene != b.scene || a.ordinal != b.ordinal;
}
static int rc_registry_declare_through(int *registry, RcRegistryPosition position) {
    assert(registry == &failed);
    assert(position.scene == active->fence && position.ordinal == (active->resource_next + 1) * 2);
    ++seals;
    return RC_REGISTRY_OK;
}
static const void *nvn_op_payload(const op_hdr *op) { return &op->payload; }
static void rt_fatal(const char *message) { (void)message; abort(); }
static void rt_log(const char *format, ...) {
    const char *prefix = "vk: required graphics storage";
    if (strncmp(format, prefix, strlen(prefix))) return;
    const size_t used = strlen(storage_text);
    va_list arguments;
    va_start(arguments, format);
    assert(vsnprintf(storage_text + used, sizeof storage_text - used, format, arguments) > 0);
    va_end(arguments);
}

static VkExecGraphicsResult vk_exec_graphics_draw_prepare(VkExec *ex, uint32_t opcode,
        const void *payload, int ordered, RcRegistryPosition attempted,
        VkExecGpaSampleScope **sample_pending, VkExecGraphicsDraw **draw) {
    assert(ex == active && opcode == OP_DRAW && ex->resource_op == ex->resource_next + 1);
    uint32_t token = *(const uint32_t *)payload;
    if (!ordered) {
        ++probes;
        assert(!*draw && !sample_pending);
        if (mode == MISSING_MANIFEST) return VK_EXEC_GRAPHICS_INVALIDATED;
        *draw = token == earlier.selected ? &earlier : &blocked;
        if (*draw == &earlier) return VK_EXEC_GRAPHICS_READY;
        if (mode == PROVIDER_STALE) return VK_EXEC_GRAPHICS_INVALIDATED;
        return mode == WARM_STALE ? VK_EXEC_GRAPHICS_READY : VK_EXEC_GRAPHICS_NEEDS_ORDERED;
    }
    ++ordered_calls;
    assert(drains && *draw == &blocked && blocked.selected == token && sample_pending && !*sample_pending);
    if (mode == PRESSURE) return VK_EXEC_GRAPHICS_NEEDS_ORDERED;
    if (mode == STALE_AGAIN) return VK_EXEC_GRAPHICS_INVALIDATED;
    RcRegistryPosition read = vk_exec_resource_position();
    if (prior_writer && (rc_registry_position_compare(attempted, read) || repend)) {
        sample_scope.read = read;
        *sample_pending = &sample_scope;
        return VK_EXEC_GRAPHICS_NEEDS_ORDERED;
    }
    if (prior_writer) assert(!rc_registry_position_compare(attempted, read));
    return VK_EXEC_GRAPHICS_READY;
}
static int vk_exec_graphics_draw_current(const VkExec *ex, uint32_t op, const void *payload,
                                         VkExecGraphicsDraw *draw) {
    assert(ex == active && op == OP_DRAW && ex->resource_op == ex->resource_next + 1);
    assert(draw->selected == *(const uint32_t *)payload);
    return draw != &blocked || replans || (mode != WARM_STALE && !(stale_after_cut && prefix_settlements));
}
static int vk_exec_graphics_draw_replan(VkExec *ex, uint32_t op, const void *payload, VkExecGraphicsDraw *draw) {
    assert(ex == active && op == OP_DRAW && draw == &blocked && draw->selected == *(const uint32_t *)payload);
    assert(ex->resource_op == ex->resource_next + 1 && !failed && drains && !replans);
    ++replans;
    if (stop_replan) stopping = 1;
    return !unsafe && !stop_replan;
}
static void vk_exec_graphics_draw_cancel(VkExecGraphicsDraw **draw) {
    assert(*draw);
    *draw = NULL;
    ++cancellations;
}
static void group_drain(VkExec *ex, int why) {
    assert(ex == active && !ex->resource_op && why == PW_OP);
    ++drains;
}
static void cancel_recording(void) { ++cancellations; }
static void vk_exec_gpa_sample_cancel(VkExecGpaSampleScope *scope) {
    assert(scope == &sample_scope);
    ++pending_cancellations;
}
static void vk_exec_graphics_draw_release_gpa_inputs_retry(VkExecGraphicsDraw *draw) {
    assert(draw == &blocked && !active->resource_op);
    ++physical_drops;
}
static VkSyncProgressResult vk_exec_gpa_sample_settle(VkExecGpaSampleScope *scope, RcRegistryPosition *attempted) {
    assert(scope == &sample_scope && !active->resource_op && scope->read.ordinal == 2 * (active->resource_next + 1));
    assert(physical_drops == 1 && !prefix_settlements);
    ++prefix_settlements;
    if (settle_result == VK_SYNC_PROGRESS_COMPLETE) *attempted = scope->read;
    return settle_result;
}
static void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t op, const char *why) {
    assert(ex == active && op == OP_DRAW && why);
    snprintf(failure_text,sizeof failure_text,"%s",why);
    failed = 1;
}
'''


CHECKS = r'''
static VkExec ex;
static const op_hdr ops[2] = {{OP_DRAW, 11}, {OP_DRAW, 12}};
static VkExecGraphicsDraw *pending;
static RcRegistryPosition attempted, replanned;

static void reset(void) {
    ex = (VkExec){.fence = 20, .resource_next = 6};
    active = &ex;
    failure_text[0] = storage_text[0] = 0;
    pending = NULL;
    attempted = replanned = (RcRegistryPosition){0};
    seals = probes = ordered_calls = drains = declarations = cancellations = 0;
    physical_drops = prefix_settlements = replans = pending_cancellations = 0;
    stopping = lost = failed = prior_writer = unsafe = stale_after_cut = stop_replan = repend = 0;
    settle_result = VK_SYNC_PROGRESS_COMPLETE;
    mode = COLD;
}

static void group_cut(int expect) {
    int boundary = GRAPHICS_BOUNDARY_NONE;
    assert(probe_group(&ex, ops, ops + 2, &pending, &boundary) == &ops[1]);
    assert(pending == &blocked && boundary == expect);
    assert(probes == 2 && declarations == 1 && seals == 2 && !drains);
    assert(ex.resource_next == 7 && !ex.resource_op && !failed);
}

static VkExecGraphicsResult ordered(int boundary) {
    uint64_t before = ex.resource_next;
    unsigned declared = declarations;
    VkExecGraphicsResult result = prepare_ordered_graphics(&ex, &ops[1], boundary, &attempted, &replanned, &pending);
    assert(ex.resource_next == before && !ex.resource_op && declarations == declared);
    return result;
}

int main(void) {
    reset();
    group_cut(GRAPHICS_BOUNDARY_ORDERED);
    assert(ordered(GRAPHICS_BOUNDARY_ORDERED) == VK_EXEC_GRAPHICS_READY);
    assert(probes == 2 && ordered_calls == 1 && seals == 3 && drains == 1 && !replans);
    vk_exec_graphics_draw_cancel(&pending);

    reset();
    mode = WARM_STALE;
    group_cut(GRAPHICS_BOUNDARY_REPLAN);
    assert(ordered(GRAPHICS_BOUNDARY_REPLAN) == VK_EXEC_GRAPHICS_READY);
    assert(replans == 1 && ordered_calls == 1 && !prefix_settlements && !attempted.scene);
    assert(replanned.scene == 20 && replanned.ordinal == 16 && declarations == 1);
    assert(ordered(GRAPHICS_BOUNDARY_REPLAN) == VK_EXEC_GRAPHICS_INVALIDATED);
    assert(replans == 1 && ordered_calls == 1); // Still the same read, even in a second helper invocation.
    vk_exec_graphics_draw_cancel(&pending);

    reset();
    mode = PROVIDER_STALE;
    prior_writer = 1;
    group_cut(GRAPHICS_BOUNDARY_REPLAN);
    assert(ordered(GRAPHICS_BOUNDARY_REPLAN) == VK_EXEC_GRAPHICS_READY);
    assert(replans == 1 && ordered_calls == 2 && physical_drops == 1 && prefix_settlements == 1);
    assert(attempted.scene == 20 && attempted.ordinal == 16 && replanned.ordinal == 16);
    vk_exec_graphics_draw_cancel(&pending);

    reset();
    prior_writer = stale_after_cut = 1;
    group_cut(GRAPHICS_BOUNDARY_ORDERED);
    assert(ordered(GRAPHICS_BOUNDARY_ORDERED) == VK_EXEC_GRAPHICS_READY);
    assert(replans == 1 && ordered_calls == 3 && physical_drops == 1 && prefix_settlements == 1);
    assert(attempted.scene == 20 && attempted.ordinal == 16 && replanned.ordinal == 16);
    vk_exec_graphics_draw_cancel(&pending);

    reset();
    mode = PROVIDER_STALE;
    prior_writer = repend = 1;
    group_cut(GRAPHICS_BOUNDARY_REPLAN);
    assert(ordered(GRAPHICS_BOUNDARY_REPLAN) == VK_EXEC_GRAPHICS_UNAVAILABLE && failed);
    assert(replans == 1 && prefix_settlements == 1 && pending_cancellations == 1 && attempted.ordinal == 16);
    vk_exec_graphics_draw_cancel(&pending);

    reset();
    mode = PROVIDER_STALE;
    unsafe = 1;
    group_cut(GRAPHICS_BOUNDARY_REPLAN);
    assert(ordered(GRAPHICS_BOUNDARY_REPLAN) == VK_EXEC_GRAPHICS_INVALIDATED && !failed);
    assert(replans == 1 && !ordered_calls && !prefix_settlements);
    graphics_failure = (VkExecPreparationFailure){.owner=VK_EXEC_PREPARATION_IMAGE,
        .reason=VK_EXEC_PREPARATION_FEEDBACK,.result=VK_EXEC_GRAPHICS_UNAVAILABLE,.role=4,.slot=2,.detail=19,
        .range={.mapping=17,.range={18,0x400,0x80}},.peer={.mapping=38,.range={18,0x420,0x10}}};
    assert(!graphics_input_finished(&ex, &ops[1], VK_EXEC_GRAPHICS_INVALIDATED, &blocked) && failed);
    assert(strstr(failure_text,"image/feedback") && strstr(failure_text,"role=4 slot=2") &&
           strstr(failure_text,"mapping=17 backing=18 offset=0x400") &&
           strstr(failure_text,"mapping=38 backing=18 offset=0x420"));
    assert(!storage_text[0]);
    graphics_failure = (VkExecPreparationFailure){0};
    vk_exec_graphics_draw_cancel(&pending);

    reset();
    graphics_failure = (VkExecPreparationFailure){.owner=VK_EXEC_PREPARATION_DESCRIPTOR,
        .reason=VK_EXEC_PREPARATION_DESCRIPTOR_SOURCE,.result=VK_EXEC_GRAPHICS_UNAVAILABLE,.role=7,.slot=5,.detail=-8,
        .storage={.gate="bitmap-authority",.read={20,16},.stage=7,.set=0,.binding=5,.marker=9,.sites=5,
            .revision=1,.detail=-8,.bitmap={.gate="bitmap-authority",.result=-8,.blocker_read={20,3},
                .blocker_offset=0x420,.blocker_bytes=0x10}}};
    assert(graphics_input_finished(&ex, &ops[1], VK_EXEC_GRAPHICS_READY, &blocked) && !failed && !storage_text[0]);
    assert(!graphics_input_finished(&ex, &ops[1], VK_EXEC_GRAPHICS_UNAVAILABLE, &blocked) && failed);
    assert(strstr(failure_text,"descriptor/descriptor-source") &&
           strstr(storage_text,"read=20/16 gate=bitmap-authority") && strstr(storage_text,"stage=7 set=0 binding=5 marker=9") &&
           strstr(storage_text,"native-result=-8") && strstr(storage_text,"blocker=20/3 blocker-offset=0x420 blocker-bytes=0x10"));
    graphics_failure = (VkExecPreparationFailure){0};

    reset();
    mode = STALE_AGAIN;
    group_cut(GRAPHICS_BOUNDARY_ORDERED);
    assert(ordered(GRAPHICS_BOUNDARY_ORDERED) == VK_EXEC_GRAPHICS_INVALIDATED);
    assert(replans == 1 && ordered_calls == 2 && !prefix_settlements);
    vk_exec_graphics_draw_cancel(&pending);

    reset();
    mode = PRESSURE;
    group_cut(GRAPHICS_BOUNDARY_ORDERED);
    assert(ordered(GRAPHICS_BOUNDARY_ORDERED) == VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(!replans && !prefix_settlements && ordered_calls == 1);
    vk_exec_graphics_draw_cancel(&pending);

    reset();
    mode = MISSING_MANIFEST;
    assert(ordered(GRAPHICS_BOUNDARY_NONE) == VK_EXEC_GRAPHICS_UNAVAILABLE && failed);
    assert(!pending && !drains && !replans && !ordered_calls);

    reset();
    mode = PROVIDER_STALE;
    stop_replan = 1;
    group_cut(GRAPHICS_BOUNDARY_REPLAN);
    assert(ordered(GRAPHICS_BOUNDARY_REPLAN) == VK_EXEC_GRAPHICS_STOPPED && !failed);
    assert(replans == 1 && !ordered_calls && !prefix_settlements);
    assert(!graphics_input_finished(&ex, &ops[1], VK_EXEC_GRAPHICS_STOPPED, NULL) && cancellations == 1);
    vk_exec_graphics_draw_cancel(&pending);

    reset();
    mode = PROVIDER_STALE;
    prior_writer = 1;
    settle_result = VK_SYNC_PROGRESS_STOPPED_AFTER_CUT;
    group_cut(GRAPHICS_BOUNDARY_REPLAN);
    assert(ordered(GRAPHICS_BOUNDARY_REPLAN) == VK_EXEC_GRAPHICS_STOPPED && !failed);
    assert(replans == 1 && physical_drops == 1 && prefix_settlements == 1 && !attempted.scene);
    vk_exec_graphics_draw_cancel(&pending);

    reset();
    stopping = 1;
    assert(ordered(GRAPHICS_BOUNDARY_NONE) == VK_EXEC_GRAPHICS_STOPPED);
    assert(!failed && !pending && !seals && !replans);
}
'''


def main():
    source = REPLAY.read_text(encoding="utf-8")
    boundary = between(source, "enum { GRAPHICS_BOUNDARY_NONE", "static VkExecGraphicsResult graphics_input(")
    frontier = "\n".join(function(source, n) for n in
                         ("graphics_input", "vk_exec_graphics_storage_failure_log", "graphics_preparation_unavailable",
                          "graphics_input_finished", "prepare_ordered_graphics"))
    group = function(source, "group_start")
    probe = between(group, "        VkExecGraphicsDraw *draw = NULL;", "        if (cut)")
    wrapper = """
static const op_hdr *probe_group(VkExec *ex, const op_hdr *start, const op_hdr *end,
                                 VkExecGraphicsDraw **boundary, int *ordered_boundary) {
    for (const op_hdr *o = start; o != end; ++o) {
""" + probe + """
        ++ex->resource_next;
        declarations += !graphics_noop;
        assert(draw == &earlier);
    }
    return end;
}
"""
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="ordered-graphics-frontier-") as tmp:
        c = Path(tmp) / "check.c"
        exe = c.with_suffix(".exe")
        c.write_text("\n".join((STUBS, function(source, "graphics_opcode"), boundary, frontier, wrapper, CHECKS)), encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", str(c), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True, timeout=20)
    print("Ordered graphics: owned cut, bounded same-read replan, GPA prefix and stop passed")


if __name__ == "__main__":
    main()
