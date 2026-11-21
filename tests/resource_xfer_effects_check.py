"""Exercise real transfer footprints and command effects without a device."""
from pathlib import Path
import os
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = (ROOT / "runtime/vk_exec_xfer.c").read_text()


def function(name):
    m = re.search(r"^(?:static )?(?:int|void|uint32_t|RcRegistryResult) " + name + r"\([^;]*?\)\s*\{", source, re.M)
    if not m:
        raise RuntimeError(name)
    end = source.index("{", m.start()) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[m.start():end]


prefix = r"""
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_exec.h"
#include "resource_command_effects.h"
#include "resource_host_write.h"

enum { BASE = 32768, NATIVE_OFFSET = 128, FOOTPRINT = 368 };
static unsigned ndeclared, nwriteback, ncompleted, native_calls, live_charges;
static uint64_t address[8], bytes[8];
static RcRegistryWriterKind output_kind;
static uint8_t host[FOOTPRINT + 16], native[FOOTPRINT];
static uint64_t wb_offset[8], wb_bytes[8];
static int stale, stale_after_barrier, failed, refuse_metadata;
static RcRegistry *registry;
static RcAdmission ledger;
static RcRegistry *fixture_registry(void) { return registry; }
#define rc_registry_runtime() fixture_registry()

typedef int PlatMutex;
#define PLAT_MUTEX_INIT 0
static void plat_mutex_lock(PlatMutex *p) { assert(!*p); *p = 1; }
static void plat_mutex_unlock(PlatMutex *p) { assert(*p); *p = 0; }
void rt_fatal(const char *s, ...) { fputs(s, stderr); abort(); }
void rt_log(const char *s, ...) { (void)s; }

static struct { void (*done)(void *); void (*cancel)(void *, int); void *arg; } queued[2];
static unsigned nqueued;
void vk_completion_add_owned(uint64_t fence, void (*done)(void *), void (*cancel)(void *, int), void *arg) {
    assert(fence == 1 && nqueued < 2);
    queued[nqueued++] = (typeof(queued[0])){done, cancel, arg};
}
VkResult vk_replay_metadata_allocate(size_t size, int clear, RcBudgetProducer producer,
                                    RcBudgetTicket *ticket, void **allocation) {
    assert(size && !ticket->token.ledger && producer == RC_BUDGET_BUFFER);
    *allocation = NULL;
    if (refuse_metadata) return VK_NOT_READY;
    *allocation = clear ? calloc(1, size) : malloc(size);
    assert(*allocation);
    ticket->token.ledger = &ledger;
    live_charges++;
    return VK_SUCCESS;
}
void vk_replay_metadata_free(void *allocation, RcBudgetTicket *ticket) {
    assert(allocation && live_charges && ticket->token.ledger == &ledger);
    free(allocation);
    *ticket = (RcBudgetTicket){0};
    live_charges--;
}
static void capture(void *context, uint64_t addr, uint64_t size, RcRegistryWriterKind kind) {
    assert(context == &ndeclared && ndeclared < 8);
    address[ndeclared] = addr;
    bytes[ndeclared++] = size;
    output_kind = kind;
}
int nvn_fmt_bpb(int format, int *bw, int *bh) { *bw = *bh = format == 1 ? 4 : 1; return format == 1 ? 16 : 4; }

enum { W_FORMAT, W_REGION };
static void warn_once(int where, const char *message, ...) { (void)where; (void)message; }
static uint32_t max1(uint32_t n) { return n ? n : 1; }

struct VkMemSpanCapture { RcRegistryBufferLocation location; };
static const VkBuffer buffer = (VkBuffer)(uintptr_t)2;
static const VkImage image = (VkImage)(uintptr_t)3;
int vk_mirror_on = 1;
static int cold_move, import_fallback, span_recorded;
static RcRegistryResult bound_result;

RcRegistryResult vk_mem_span_capture_bound_input(uint64_t addr, uint64_t size, const VkMemSpan *span,
                                                  VkMemSpanCapture **hold) {
    assert(addr == BASE && size == FOOTPRINT && span->pool && !*hold);
    return bound_result;
}
RcRegistryResult vk_mem_span_prepare_native_input(uint64_t addr, uint64_t size, int clamp, VkMemMirrorInputPolicy policy,
                                                  VkMemSpan *selected, VkMemSpanCapture **hold) {
    assert(addr == BASE && size == FOOTPRINT && !clamp && policy == VK_MEM_MIRROR_IMPORT && !*hold);
    import_fallback++;
    *selected = (VkMemSpan){.buffer = buffer, .offset = NATIVE_OFFSET + !!cold_move,
                           .pool = (void *)(uintptr_t)1, .avail = FOOTPRINT};
    *hold = malloc(sizeof **hold);
    assert(*hold);
    (*hold)->location = (RcRegistryBufferLocation){6, 7, selected->offset};
    return RC_REGISTRY_OK;
}
int vk_mem_span_record_input(VkMemSpanCapture *hold, const VkMemSpan *selected) {
    assert(hold && selected->offset == NATIVE_OFFSET);
    span_recorded++;
    return 1;
}

int vk_mem_span_capture_current(const VkMemSpanCapture *hold, const VkMemSpan *selected) {
    assert(hold && selected->buffer == buffer && selected->offset == NATIVE_OFFSET);
    return !stale;
}
int vk_mem_span_buffer_location(const VkMemSpanCapture *hold, const VkMemSpan *selected,
                                RcRegistryBufferLocation *location) {
    if (!vk_mem_span_capture_current(hold, selected)) return 0;
    *location = hold->location;
    return 1;
}
void vk_mem_span_capture_release(VkMemSpanCapture **hold) { free(*hold); *hold = NULL; }
RcRegistryResult vk_mem_canonical_source_qualify(const struct MemBackingCapture *mapping, uint64_t addr, uint64_t size) {
    assert(mapping && addr == BASE && size == FOOTPRINT);
    return stale ? RC_REGISTRY_STALE : RC_REGISTRY_OK;
}
int vk_texture_alias_read_current(VkTextureAliasRead *alias) { assert(alias); return !stale; }
void vk_texture_present_release(VkTexturePresentSource *source) { assert(source); }
void vk_texture_restore_use(const VkTextureRestore *restore) { assert(restore); }

int vk_texture_transfer_input_current(const VkTextureTransferInput *input) {
    assert(!input->texture);
    return 1;
}
int vk_texture_transfer_input_use(const VkTextureTransferInput *input) {
    assert(!input->texture);
    return 1;
}

int vk_texture_transfer_read_use(const VkTexturePresentSource *source, VkImage selected) {
    assert(source && selected == image);
    return 1;
}

int vk_exec_recording_failed(void) { return failed; }
void vk_contact_flush(VkExec *ex) { assert(ex); }
void vk_exec_pass_end(VkExec *ex) { assert(ex); }
void vk_exec_barrier(VkExec *ex) { assert(ex); stale |= stale_after_barrier; }
void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t op, const char *reason) {
    assert(ex && op == OP_COPY_T2B && reason);
    failed = 1;
}
void vk_gpu_gap_note(int kind, uint64_t units) { assert(kind == VK_GAP_COPY_T2B && units == 45); }
void vk_mem_count(int cls, int use, uint64_t size) { assert(!cls && use == VK_MUSE_GPU_WRITE && size == 288); }
void vk_usnap_drop(uint64_t addr, uint64_t size) { assert(addr == BASE && size == FOOTPRINT); }
void vk_mem_gpu_pending_span(const VkMemSpan *span, uint64_t addr, uint64_t size) {
    assert(span->buffer == buffer && addr == BASE && size == FOOTPRINT);
}
void vk_mem_gpu_wrote_exact(const VkMemSpan *row, uint64_t size) {
    assert(row->buffer == buffer && nwriteback < 8);
    wb_offset[nwriteback] = row->offset - NATIVE_OFFSET;
    wb_bytes[nwriteback++] = size;
}
void vk_mem_buffer_write_completed(void *provider, VkBuffer selected, uint64_t addr, uint64_t size, int mirror) {
    assert(!provider && selected == buffer && addr == BASE + 64 * ncompleted && size == 48 && mirror);
    ncompleted++;
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer(VkCommandBuffer cb, VkImage src, VkImageLayout layout,
        VkBuffer dst, uint32_t count, const VkBufferImageCopy *r) {
    assert(cb && src == image && layout == VK_IMAGE_LAYOUT_GENERAL && dst == buffer && count == 1);
    assert(r->bufferOffset == NATIVE_OFFSET && r->bufferRowLength == 16 && r->bufferImageHeight == 8);
    assert(r->imageSubresource.mipLevel == 1 && r->imageSubresource.baseArrayLayer == 3);
    assert(r->imageSubresource.layerCount == 3 && r->imageExtent.width == 9 && r->imageExtent.height == 5);
    for (unsigned row = 0; row < 6; row++) memset(native + row * 64, (int)(row + 1), 48);
    native_calls++;
}
"""

main = r"""
static void cold_destination(void) {
    VkExecTextureBufferWrite write = {.address = BASE, .pitch = {.size = FOOTPRINT},
        .span = {.buffer = buffer, .pool = (void *)(uintptr_t)1, .offset = NATIVE_OFFSET, .avail = FOOTPRINT}};
    bound_result = RC_REGISTRY_UNSUPPORTED;
    assert(texture_buffer_hold_output(&write) && import_fallback == 1 && span_recorded == 1);
    assert(write.location.allocation == 6 && write.location.offset == NATIVE_OFFSET);
    vk_mem_span_capture_release(&write.hold);
    cold_move = 1;
    assert(!texture_buffer_hold_output(&write) && import_fallback == 2 && span_recorded == 1);
    vk_mem_span_capture_release(&write.hold);
    bound_result = RC_REGISTRY_CAPACITY;
    assert(!texture_buffer_hold_output(&write) && import_fallback == 2 && span_recorded == 1 && !write.hold);
}

static void extraction(void) {
    VkExec ex = {.copy_stride = 64};
    tex_t tex = {0};
    tex.mem = (void *)(uintptr_t)16384; tex.L.size = 1024; tex.d.format = 1;
    p_copy_b2t op = {.src = BASE, .tex = &tex, .r = {.width = 9, .height = 5, .depth = 3}};
    assert(vk_exec_xfer_effects(&ex, OP_COPY_T2B, &op, capture, &ndeclared));
    assert(ndeclared == 6 && output_kind == RC_REGISTRY_BUFFER);
    for (unsigned row = 0; row < ndeclared; row++) assert(address[row] == BASE + 64 * row && bytes[row] == 48);
    ndeclared = 0; ex.copy_stride = 0;
    assert(vk_exec_xfer_effects(&ex, OP_COPY_T2B, &op, capture, &ndeclared));
    assert(ndeclared == 1 && address[0] == BASE && bytes[0] == 288);
    assert(vk_exec_xfer_effects(&ex, OP_COPY_B2T, &op, capture, &ndeclared));
    assert(address[1] == 16384 && bytes[1] == 1024 && output_kind == RC_REGISTRY_IMAGE);
    op.r.width = 0;
    assert(!vk_exec_xfer_effects(&ex, OP_COPY_T2B, &op, capture, &ndeclared));
    op.r.width = 9; ex.copy_stride = INT64_MAX - 15;
    assert(!vk_exec_xfer_effects(&ex, OP_COPY_T2B, &op, capture, &ndeclared));
    ex.copy_stride = 49;
    assert(vk_exec_xfer_effects(&ex, OP_COPY_T2B, &op, capture, &ndeclared) && ndeclared == 2);
    ex.copy_stride = 64; op.src = UINT64_MAX - 367;
    assert(!vk_exec_xfer_effects(&ex, OP_COPY_T2B, &op, capture, &ndeclared) && ndeclared == 2);
    p_copy_b2b copy = {.src = 4096, .dst = 8192, .size = 200};
    assert(!vk_exec_xfer_effects(&ex, OP_COPY_B2B, &copy, capture, &ndeclared) && ndeclared == 2);
    copy.dst = UINT64_MAX - 100;
    assert(!vk_exec_xfer_effects(&ex, OP_COPY_B2B, &copy, capture, &ndeclared));
    p_clear_buf clear = {.addr = 8193, .size = 64};
    assert(!vk_exec_xfer_effects(&ex, OP_CLEAR_BUF, &clear, capture, &ndeclared) && ndeclared == 2);
    p_report_counter report = {.addr = 65536};
    assert(vk_exec_xfer_effects(&ex, OP_REPORT_COUNTER, &report, capture, &ndeclared));
    assert(bytes[2] == sizeof(NVNcounterData));
    p_signal_event event = {.val = (void *)(uintptr_t)65536};
    assert(vk_exec_xfer_effects(&ex, OP_SIGNAL_EVENT, &event, capture, &ndeclared) && bytes[3] == 4);
    assert(!vk_exec_xfer_effects(&ex, OP_DRAW, NULL, capture, &ndeclared));
}

static RcRegistryPosition read_position;
static int readable(uint64_t backing, uint64_t offset, uint64_t size) {
    return rc_registry_read_range_at(registry, (RcSourceRange){backing, offset, size},
                                    read_position) == RC_REGISTRY_OK;
}

enum { BUFFER_UNRECORDED, BUFFER_UNSENT, BUFFER_UNKNOWN, BUFFER_COMPLETE };

static void retained_buffer_effects(uint32_t opcode, int disposition) {
    registry = rc_registry_create((RcRegistryConfig){32, 32, 4});
    assert(registry);
    nqueued = 0;
    RcInterval part[] = {{BASE, 11, {12, 256, 8}}, {BASE + 8, 13, {14, 512, 4}},
                         {BASE + 12, 13, {14, 516, 8}}};
    BufferTransferLeg legs[] = {{.src = BASE - 32, .dst = BASE, .size = 12, .writes = part, .write_count = 2},
                                {.src = BASE - 20, .dst = BASE + 12, .size = 8, .writes = part + 2, .write_count = 1}};
    VkExecBufferTransfer transfer = {.legs = legs, .count = 2, .src = BASE - 32, .dst = BASE,
        .size = 32, .read = {1, 2}, .opcode = opcode, .stop = BUFFER_TRANSFER_HOLE, .missing = 1,
        .missing_address = BASE + 20};
    const RcRegistryPosition output = {1, 3};
    read_position = (RcRegistryPosition){1, 4};
    assert(rc_registry_declare_through(registry, transfer.read) == RC_REGISTRY_OK);
    ResourceCommandEffects *effects = resource_effects_create(output);
    assert(effects);
    assert(vk_exec_buffer_transfer_declare(&transfer, effects, (RcRegistryPosition){1, 5}) == RC_REGISTRY_STALE);
    assert(!rc_registry_stats(registry).writers);
    assert(vk_exec_buffer_transfer_declare(&transfer, effects, output) == RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry, read_position) == RC_REGISTRY_OK);
    assert(!readable(1, BASE, 20) && !readable(12, 256, 8) && !readable(14, 512, 12));
    assert(readable(1, BASE - 32, 32) && readable(1, BASE + 20, 12));
    assert(readable(12, 264, 4) && readable(14, 524, 12));

    // Model emission with the same effect calls used by buffer_transfer_record for each retained leg.
    if (disposition != BUFFER_UNRECORDED) {
        ResourceCommandEffects *prior = resource_effects_enter(effects);
        for (unsigned i = 0; i < 2; i++) {
            const BufferTransferLeg *leg = &legs[i];
            int visible = i == 0;
            assert(resource_effects_buffer_prepare_intervals(leg->writes, leg->write_count, visible));
            assert(resource_effects_buffer_prepare(leg->dst, leg->size, visible));
            resource_effects_buffer_prepare_native(leg->writes, leg->write_count,
                leg->dst, (RcRegistryBufferLocation){30 + i, i, NATIVE_OFFSET});
        }
        resource_effects_buffers_commit();
        resource_effects_leave(prior);
    }
    resource_effects_release(effects);
    assert(nqueued == (disposition != BUFFER_UNRECORDED));
    if (disposition >= BUFFER_UNKNOWN) {
        assert(rc_registry_handoff_through(registry, read_position) == RC_REGISTRY_OK);
        read_position = (RcRegistryPosition){2, 2};
        assert(rc_registry_declare_through(registry, read_position) == RC_REGISTRY_OK);
    }
    if (disposition == BUFFER_COMPLETE) queued[0].done(queued[0].arg);
    else if (nqueued) queued[0].cancel(queued[0].arg, disposition == BUFFER_UNKNOWN);

    const int direct_visible = disposition != BUFFER_UNKNOWN;
    const int mirror_visible = disposition <= BUFFER_UNSENT;
    assert(readable(1, BASE, 12) == direct_visible && readable(1, BASE + 12, 8) == mirror_visible);
    assert(readable(12, 256, 8) == direct_visible && readable(14, 512, 4) == direct_visible);
    assert(readable(14, 516, 8) == mirror_visible);
    assert(readable(1, BASE - 32, 32) && readable(1, BASE + 20, 12));
    assert(readable(12, 264, 4) && readable(14, 524, 12));
    assert(!rc_registry_stats(registry).fail_closed);
    rc_registry_destroy(registry);
}

enum { COMPLETE, CANCEL, UNKNOWN, STALE_BEFORE_DECLARATION, STALE_BEFORE_RECORD };
static void transfer(int disposition) {
    registry = rc_registry_create((RcRegistryConfig){32, 32, 4});
    assert(registry);
    nqueued = nwriteback = ncompleted = native_calls = failed = 0;
    stale = disposition == STALE_BEFORE_DECLARATION;
    stale_after_barrier = disposition == STALE_BEFORE_RECORD;
    memset(host, 0xaa, sizeof host);
    memset(native, 0xbb, sizeof native);
    VkExec ex = {.copy_stride = 64, .cb = (VkCommandBuffer)(uintptr_t)1, .fence = 1};
    tex_t tex = {0};
    p_copy_b2t op = {.tex = &tex, .src = BASE, .has_view = 1, .r = {.zoffset = 1, .depth = 3},
                     .view = {.set_levels = 1, .base_level = 1, .set_layers = 1, .min_layer = 2}};
    VkTexImage selected = {.image = image, .type = VK_IMAGE_TYPE_2D, .width = 18, .height = 10,
        .layers = 6, .levels = 2, .aspect = VK_IMAGE_ASPECT_COLOR_BIT};
    Sub sub = {.img = &selected, .bpb = 16, .bw = 4, .bh = 4};
    RcBudgetTicket charge = {0};
    void *allocation;
    assert(vk_replay_metadata_allocate(sizeof(VkExecTextureBufferWrite) + 6 * sizeof(RcRegistryWrite), 1,
                                      RC_BUDGET_BUFFER, &charge, &allocation) == VK_SUCCESS);
    VkExecTextureBufferWrite *write = allocation;
    write->memory = charge;
    assert(sub_region(&tex, &op.view, 1, &op.r, 1, &sub));
    assert(buffer_layout(&ex, &sub, &write->pitch, &write->region));
    assert(write->pitch.size == FOOTPRINT);
    write->region.bufferOffset = NATIVE_OFFSET;
    write->span = (VkMemSpan){.buffer = buffer, .offset = NATIVE_OFFSET, .avail = FOOTPRINT};
    write->image = image; write->address = BASE; write->mirror = 1;
    write->resident = (VkTexturePresentSource *)(uintptr_t)1;
    write->location = (RcRegistryBufferLocation){6, 7, NATIVE_OFFSET};
    write->hold = malloc(sizeof *write->hold);
    assert(write->hold);
    write->hold->location = write->location;
    write->mapped_count = 2;
    write->mapped = malloc(2 * sizeof *write->mapped);
    assert(write->mapped);
    write->mapped[0] = (RcInterval){BASE, 11, {12, 256, 82}};
    write->mapped[1] = (RcInterval){BASE + 82, 13, {14, 512, FOOTPRINT - 82}};
    RcSourceRange cover[] = {write->mapped[0].range, write->mapped[1].range};
    RcRegistryBufferSpan demand[] = {
        {write->mapped[0], write->location},
        {write->mapped[1], {6, 7, NATIVE_OFFSET + 82}},
    };
    RcRegistryGpuCapture capture = {0};
    assert(rc_registry_gpu_capture_begin(registry, cover, 2, &capture) == RC_REGISTRY_OK);
    refuse_metadata = 1;
    assert(!texture_buffer_written_intervals(&write->pitch, BASE, write->mapped, 2,
                                            &write->written_memory, &write->written));
    assert(!write->written && live_charges == 1);
    refuse_metadata = 0;
    write->written_count = texture_buffer_written_intervals(&write->pitch, BASE, write->mapped, 2,
                                                           &write->written_memory, &write->written);
    assert(write->written_count == 7 && write->written[1].range.size == 18 &&
           write->written[2].range.size == 30 && write->written[6].address == BASE + 320);
    RcRegistryPosition position = {1, 3}, read = {1, 4};
    read_position = read;
    assert(rc_registry_declare_through(registry, (RcRegistryPosition){1, 2}) == RC_REGISTRY_OK);
    ResourceCommandEffects *effects = resource_effects_create(position);
    VkExecRestoreCopy copy = {.output = write};
    RcRegistryResult declared = vk_exec_restore_copy_declare(&op, &copy, effects, position);
    assert(declared == (stale ? RC_REGISTRY_STALE : RC_REGISTRY_OK));
    assert(rc_registry_declare_through(registry, read) == RC_REGISTRY_OK);
    assert(readable(1, BASE + 48, 16) && readable(12, 256 + 48, 16));
    if (!stale) assert(!readable(1, BASE, 48) && !readable(12, 256, 48));
    ResourceCommandEffects *prior = resource_effects_enter(effects);
    if (!stale) vk_exec_restore_copy_commit(&ex, &op, &copy);
    resource_effects_leave(prior);
    if (disposition >= STALE_BEFORE_DECLARATION) {
        assert(!native_calls && !nwriteback && copy.output && !nqueued);
        texture_buffer_write_cancel(copy.output, 0);
        resource_effects_release(effects);
        assert(readable(1, BASE, 48) && readable(12, 256, 48));
    } else {
        assert(!copy.output && native_calls == 1 && nwriteback == 6 && nqueued == 1 && live_charges == 2);
        RcRegistryBufferFragment fragments[8];
        uint32_t count;
        assert(rc_registry_gpu_capture_buffer_fragments(registry, capture, demand, 2, read, fragments, 8,
                                                        &count) == RC_REGISTRY_OK && count == 7);
        for (uint32_t i = 0; i < count; i++) {
            uint64_t offset = fragments[i].interval.address - BASE;
            assert(fragments[i].output.ordinal == 3 && offset % 64 < 48 &&
                   fragments[i].interval.range.size <= 48 - offset % 64);
        }
        resource_effects_release(effects);
        assert(nqueued == 2);
        for (unsigned row = 0; row < 6; row++) {
            assert(wb_offset[row] == 64 * row && wb_bytes[row] == 48);
            memset(host + 64 * row + 48, 0xcc, 16);  // CPU writes padding after native recording.
        }
        if (disposition == COMPLETE) {
            assert(rc_registry_handoff_through(registry, read) == RC_REGISTRY_OK);
            read_position = (RcRegistryPosition){2, 2};
            assert(rc_registry_declare_through(registry, read_position) == RC_REGISTRY_OK);
            RcRegistryVisibility proof;
            rc_registry_visibility_begin(registry, RC_REGISTRY_VISIBILITY_ORDERED, &proof);
            proof.submission = 1;
            for (unsigned row = 0; row < 6; row++)
                memcpy(host + wb_offset[row], native + wb_offset[row], wb_bytes[row]);
            for (unsigned i = 0; i < nqueued; i++) queued[i].done(queued[i].arg);
            assert(ncompleted == 6);
            assert(!readable(1, BASE, 48) && !readable(12, 256, 48));
            for (unsigned row = 0; row < 6; row++)
                rc_registry_visibility_apply(registry, &proof, (RcSourceRange){1, BASE + wb_offset[row], wb_bytes[row]});
            for (uint32_t i = 0; i < count; i++)
                rc_registry_visibility_apply(registry, &proof, fragments[i].interval.range);
            assert(readable(1, BASE, FOOTPRINT) && readable(12, 256, 82) && readable(14, 512, FOOTPRINT - 82));
        } else {
            for (unsigned i = 0; i < nqueued; i++) queued[i].cancel(queued[i].arg, disposition == UNKNOWN);
            assert(!ncompleted);
            assert(readable(1, BASE, 48) == (disposition == CANCEL));
            assert(readable(12, 256, 48) == (disposition == CANCEL));
        }
        for (unsigned row = 0; row < 6; row++) {
            for (unsigned col = 0; col < 48; col++)
                assert(host[64 * row + col] == (disposition == COMPLETE ? row + 1 : 0xaa));
            for (unsigned col = 48; col < 64; col++) assert(host[64 * row + col] == 0xcc);
        }
        assert(readable(1, BASE + 48, 16) && readable(12, 256 + 48, 16));
    }
    assert(!live_charges && !rc_registry_stats(registry).fail_closed);
    assert(rc_registry_gpu_capture_end(registry, &capture) == RC_REGISTRY_OK);
    rc_registry_destroy(registry);
}

int main(void) {
    extraction();
    cold_destination();
    for (int disposition = COMPLETE; disposition <= STALE_BEFORE_RECORD; disposition++) transfer(disposition);
    for (int disposition = BUFFER_UNRECORDED; disposition <= BUFFER_COMPLETE; ++disposition) {
        retained_buffer_effects(OP_COPY_B2B, disposition);
        retained_buffer_effects(OP_CLEAR_BUF, disposition);
    }
    puts("transfer effects: exact T2B rows, retained B2B and ClearBuffer claims, cancellation and visibility passed");
}
"""

effects = (ROOT / "runtime/resource_command_effects.c").read_text()
effects = "\n".join(line for line in effects.splitlines() if not line.startswith("#include"))
sub = source[source.index("typedef struct {\n    const VkTexImage *img;"):source.index("static int sub_format(")]
layout = source[source.index("static VkImageSubresourceLayers sub_layers("):source.index("// A VRAM mirror supplies")]
write = source[source.index("struct VkExecTextureBufferWrite {"):source.index("int vk_exec_restore_copy_prepare(")]
begin = source.index("typedef struct {\n    VkMemSpan source, destination;")
buffer_plan = "typedef struct Range Range;\n" + source[begin:source.index("static int buffer_transfer_probe;", begin)]
selected = ("vk_exec_restore_copy_declare", "vk_exec_restore_copy_commit", "vk_exec_buffer_transfer_declare",
            "effect_emit_range", "effect_emit_texture",
            "vk_exec_xfer_effects")
work = ROOT / "build/test"
work.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="resource-xfer-", dir=work) as tmp:
    c = Path(tmp) / "check.c"
    exe = c.with_suffix(".exe")
    c.write_text(prefix + effects + "\n" + sub + layout + write + buffer_plan + "\n".join(function(n) for n in selected) + main)
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-fsanitize-trap=all", "-DRC_REGISTRY_SINGLE_THREADED", "-I" + str(ROOT / "runtime"),
                    "-I" + os.environ.get("VMA_INCLUDE", str(CC.parent.parent / "include/vma")),
                    str(c), str(ROOT / "runtime/resource_core_registry.c"), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)
