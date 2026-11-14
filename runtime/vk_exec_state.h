// Compiled state deltas contain source offsets in the current command copy and destinations in VkExec.
// They retain no resource objects or captured bytes. Include after DrawCtx and the ordinary state handlers.
#pragma once

#include "nvn_exec_state_blob.h"

_Static_assert(sizeof(VkExec) < UINT32_MAX, "compiled state destination width");

// Writes stay sorted and disjoint. Locate the overlap without scanning unrelated destinations, then replace
// only that slice. A later write retains either surviving edge; each insertion grows the list by at most two.
static void exec_state_write(ExecStateBuild *b, uint32_t dst, uint32_t bytes, size_t source,
                             unsigned operation, unsigned notify) {
    if (!bytes) return;
    uint32_t end = dst + bytes;
    ExecStateWrite value = {source, dst, (uint16_t)bytes, (uint8_t)operation, (uint8_t)notify};
    if (!b->count || b->writes[b->count - 1].destination + b->writes[b->count - 1].bytes <= dst) {
        b->writes[b->count++] = value;
        return;
    }
    uint32_t first = 0, last = b->count;
    while (first < last) {
        uint32_t mid = first + (last - first) / 2;
        const ExecStateWrite *w = &b->writes[mid];
        if (w->destination + w->bytes <= dst) first = mid + 1;
        else last = mid;
    }
    if (b->writes[first].destination == dst && b->writes[first].bytes == bytes) {
        b->writes[first] = value;
        return;
    }
    last = first;
    while (last < b->count && b->writes[last].destination < end) last++;
    ExecStateWrite replacement[3];
    uint32_t n = 0;
    if (first < last && b->writes[first].destination < dst) {
        replacement[n] = b->writes[first];
        replacement[n].bytes = (uint16_t)(dst - replacement[n].destination);
        n++;
    }
    replacement[n++] = value;
    if (first < last && b->writes[last - 1].destination + b->writes[last - 1].bytes > end) {
        ExecStateWrite right = b->writes[last - 1];
        right.bytes = (uint16_t)(right.destination + right.bytes - end);
        if (right.operation == ES_COPY) right.source += end - right.destination;
        right.destination = end;
        replacement[n++] = right;
    }
    uint32_t tail = b->count - last;
    if (tail && first + n != last)
        memmove(&b->writes[first + n], &b->writes[last], (size_t)tail * sizeof b->writes[0]);
    memcpy(&b->writes[first], replacement, (size_t)n * sizeof b->writes[0]);
    b->count = first + n + tail;
}

static int exec_state_compile_op(ExecStateBuild *b, const uint8_t *commands, size_t at, size_t limit) {
    if (limit - at < sizeof(op_hdr)) return 0;
    const op_hdr *h = (const op_hdr *)(commands + at);
    if (h->size < sizeof *h || (h->size & 7) || h->size > limit - at) return 0;
    const void *payload = nvn_op_payload(h);
    size_t source = at + sizeof *h, available = h->size - sizeof *h;
#define REQUIRE(type) if (available < sizeof(type)) return 0
#define WRITE(member, src, n, notify) exec_state_write(b, (uint32_t)offsetof(VkExec, member), (uint32_t)(n), (src), ES_COPY, (notify))
#define FIELD(member, type, input) WRITE(member, source + offsetof(type, input), sizeof(((VkExec *)0)->member), ES_NO_NOTIFY)
    switch (h->op) {
    case OP_BIND_PROGRAM:
        REQUIRE(p_bind_program);
        FIELD(prog, p_bind_program, prog);
        FIELD(prog_stages, p_bind_program, stages);
        b->bind_increments++;
        break;
    case OP_BIND_STATE: {
        REQUIRE(p_bind_state);
        const p_bind_state *v = payload;
        uint32_t dst, bytes;
        switch (v->kind) {
        case 0: {
            blend_st blend;
            memcpy(&blend, v->bytes, sizeof blend);
            if (blend.target >= 8) return 0;
            dst = (uint32_t)offsetof(VkExec, pipe.blend) + (uint32_t)blend.target * sizeof blend;
            bytes = sizeof blend;
            break;
        }
        case 1: dst = offsetof(VkExec, pipe.color), bytes = sizeof(color_st); break;
        case 2: dst = offsetof(VkExec, pipe.chmask), bytes = sizeof(chmask_st); break;
        case 3: dst = offsetof(VkExec, pipe.ms), bytes = sizeof(ms_st); break;
        case 4: dst = offsetof(VkExec, pipe.poly), bytes = sizeof(poly_st); break;
        case 5: dst = offsetof(VkExec, pipe.ds), bytes = sizeof(ds_st); break;
        case 6: dst = offsetof(VkExec, point_size), bytes = sizeof(float); break;
        default: return 0;
        }
        exec_state_write(b, dst, bytes, source + offsetof(p_bind_state, bytes), ES_COPY, ES_NO_NOTIFY);
        break;
    }
    case OP_BIND_VATTR: case OP_BIND_VSTREAM: {
        if (available < offsetof(p_bind_vattr, st)) return 0;
        const p_bind_vattr *v = payload;
        size_t stride = h->op == OP_BIND_VATTR ? sizeof(uint32_t) : sizeof(vs_st);
        if (v->first < 0 || v->first > 16 || v->n < 0 || v->n > 16 - v->first ||
            available < offsetof(p_bind_vattr, st) + (size_t)v->n * stride) return 0;
        uint32_t dst = h->op == OP_BIND_VATTR ? offsetof(VkExec, pipe.vattr) : offsetof(VkExec, pipe.vstream);
        exec_state_write(b, dst, (uint32_t)(16 * stride), 0, ES_ZERO, ES_NO_NOTIFY);
        exec_state_write(b, dst + (uint32_t)(v->first * stride), (uint32_t)(v->n * stride),
                         source + offsetof(p_bind_vattr, st), ES_COPY, ES_NO_NOTIFY);
        break;
    }
    case OP_BIND_VBUF: {
        if (available < offsetof(p_bind_vbuf, r)) return 0;
        const p_bind_vbuf *v = payload;
        if (v->first < 0 || v->first > 16 || v->n < 0 || v->n > 16 - v->first ||
            available < offsetof(p_bind_vbuf, r) + (size_t)v->n * sizeof v->r[0]) return 0;
        exec_state_write(b, offsetof(VkExec, vbuf) + (uint32_t)v->first * sizeof(NVNbufferRange),
                         (uint32_t)v->n * sizeof(NVNbufferRange), source + offsetof(p_bind_vbuf, r), ES_COPY, ES_NO_NOTIFY);
        break;
    }
    case OP_BIND_UBO: case OP_BIND_SSBO: {
        REQUIRE(p_bind_buf);
        const p_bind_buf *v = payload;
        unsigned slots = h->op == OP_BIND_UBO ? 18 : 16;
        if (v->stage < 0 || v->stage >= 8 || v->index < 0 || (unsigned)v->index >= slots) return 0;
        // Task/mesh SSBO binds retain their once-per-slot diagnostic in the ordinary typed handler.
        if (h->op == OP_BIND_SSBO && v->stage >= 6) return 0;
        uint32_t dst = h->op == OP_BIND_UBO ? offsetof(VkExec, ubo) : offsetof(VkExec, ssbo);
        dst += ((uint32_t)v->stage * slots + (uint32_t)v->index) * sizeof(p_bind_buf);
        exec_state_write(b, dst, sizeof(p_bind_buf), source, ES_COPY, h->op == OP_BIND_UBO ? ES_UBO : ES_SSBO);
        b->bind_increments++;
        break;
    }
    case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: {
        if (available < offsetof(p_bind_handles, h)) return 0;
        const p_bind_handles *v = payload;
        unsigned slots = h->op == OP_BIND_IMG ? 32 : 128;
        if (v->stage < 0 || v->stage >= 8 || v->first < 0 || (unsigned)v->first > slots ||
            v->n < 0 || v->n > 32 || (unsigned)v->n > slots - (unsigned)v->first ||
            available < offsetof(p_bind_handles, h) + (size_t)v->n * sizeof v->h[0]) return 0;
        uint32_t dst = h->op == OP_BIND_TEX ? offsetof(VkExec, tex) :
                       h->op == OP_BIND_IMG ? offsetof(VkExec, img) : offsetof(VkExec, samp);
        dst += ((uint32_t)v->stage * slots + (uint32_t)v->first) * sizeof(uint64_t);
        unsigned notify = h->op == OP_BIND_TEX ? ES_TEXTURE : h->op == OP_BIND_IMG ? ES_IMAGE : ES_NO_NOTIFY;
        exec_state_write(b, dst, (uint32_t)v->n * sizeof(uint64_t), source + offsetof(p_bind_handles, h), ES_COPY, notify);
        if (h->op != OP_BIND_SAMP) b->bind_increments++, b->texture_increments++;
        break;
    }
    case OP_PUSH_CC: {
        REQUIRE(p_push_cc);
        const p_push_cc *v = payload;
        if (v->offset > sizeof(((VkExec *)0)->cs_push) || v->size > sizeof(((VkExec *)0)->cs_push) - v->offset ||
            v->size > available - sizeof *v) return 0;
        exec_state_write(b, offsetof(VkExec, cs_push) + v->offset, v->size, source + sizeof *v, ES_COPY, ES_NO_NOTIFY);
        b->bind_increments++;
        break;
    }
    case OP_VIEWPORTS: {
        REQUIRE(p_viewports);
        const p_viewports *v = payload;
        if (v->first < 0 || v->first > 16 || v->n < 0 || v->n > 16 - v->first) return 0;
        exec_state_write(b, offsetof(VkExec, vp) + (uint32_t)v->first * sizeof(v->v[0]),
                         (uint32_t)v->n * sizeof(v->v[0]), source + offsetof(p_viewports, v), ES_COPY, ES_NO_NOTIFY);
        b->flags |= ES_HAS_VIEWPORT, b->dirty |= VK_EXEC_DIRTY_VIEWPORT;
        break;
    }
    case OP_SCISSOR:
        REQUIRE(p_scissor);
        WRITE(scissor, source, sizeof(p_scissor), ES_NO_NOTIFY);
        b->flags |= ES_HAS_SCISSOR, b->dirty |= VK_EXEC_DIRTY_SCISSOR;
        break;
    case OP_DEPTH_RANGE:
        REQUIRE(p_depth_range);
        WRITE(depth_range, source, sizeof(p_depth_range), ES_NO_NOTIFY);
        b->dirty |= VK_EXEC_DIRTY_VIEWPORT;
        break;
    case OP_STENCIL: {
        REQUIRE(p_stencil);
        const p_stencil *v = payload;
        uint32_t dst = v->which == 0 ? offsetof(VkExec, st_vmask) : v->which == 1 ? offsetof(VkExec, st_wmask) : offsetof(VkExec, st_ref);
        source += v->which == 0 ? offsetof(p_stencil, value_mask) : v->which == 1 ? offsetof(p_stencil, mask) : offsetof(p_stencil, ref);
        for (unsigned f = 0; f < 2; f++) if (v->face & (1u << f))
            exec_state_write(b, dst + f * sizeof(uint32_t), sizeof(uint32_t), source, ES_COPY, ES_NO_NOTIFY);
        b->dirty |= VK_EXEC_DIRTY_STENCIL;
        break;
    }
    case OP_POLY_OFFSET:
        REQUIRE(p_poly_offset);
        WRITE(poly_offset, source, sizeof(p_poly_offset), ES_NO_NOTIFY);
        b->dirty |= VK_EXEC_DIRTY_DEPTH_BIAS;
        break;
    case OP_BLEND_COLOR:
        REQUIRE(p_blend_color);
        FIELD(blend_color, p_blend_color, rgba);
        b->dirty |= VK_EXEC_DIRTY_BLEND_CONST;
        break;
    case OP_DEPTH_CLAMP: case OP_RASTER_DISCARD:
        REQUIRE(p_flag);
        exec_state_write(b, h->op == OP_DEPTH_CLAMP ? offsetof(VkExec, pipe.depth_clamp) : offsetof(VkExec, pipe.raster_discard),
                         1, source + offsetof(p_flag, value), ES_BOOL, ES_NO_NOTIFY);
        break;
    case OP_PRIM_RESTART:
        REQUIRE(p_prim_restart);
        WRITE(prim_restart, source, sizeof(p_prim_restart), ES_NO_NOTIFY);
        break;
    case OP_COPY_STRIDE:
        REQUIRE(p_copy_stride);
        FIELD(copy_stride, p_copy_stride, stride);
        break;
    default: return 0;
    }
#undef FIELD
#undef WRITE
#undef REQUIRE
    return 1;
}

static void exec_state_emit(ExecStateBuild *b, NvnExecStateEmit emit, void *user) {
    // Construction already maintains destination order; emission needs no second sorting pass.
    size_t bytes = offsetof(ExecStateBuild, writes) + b->count * sizeof b->writes[0];
    emit(user, b, bytes);
}

int nvn_exec_state_compile(const uint8_t *commands, size_t begin, size_t limit,
                           NvnExecStateEmit emit, void *user, NvnExecStateSpan *out) {
    if (!commands || !emit || !out || begin >= limit || (begin & 7)) return 0;
    ExecStateBuild b;
    // Only the header is live initially. Each insertion initializes its write before increasing count.
    memset(&b, 0, offsetof(ExecStateBuild, writes));
    uint32_t ops = 0;
    size_t end = begin;
    // A command can add two writes, each splitting one older write. Reserve eight slots before accepting it.
    while (ops < EXEC_STATE_OPS && b.count + 8 <= EXEC_STATE_WRITES && end < limit) {
        if (!exec_state_compile_op(&b, commands, end, limit)) break;
        end += ((const op_hdr *)(commands + end))->size;
        ops++;
    }
    if (ops < 2) return 0;
    exec_state_emit(&b, emit, user);
    *out = (NvnExecStateSpan){.end = end, .ops = ops};
    return 1;
}

// The planning front pass observes state only at cuts and ordered operations; it never executes these draws.
// A summary can therefore discard state overwritten across draws, provided its entire interval fits one chunk.
int nvn_exec_front_compile(const uint8_t *commands, size_t begin, size_t limit,
                           NvnExecStateEmit emit, void *user, NvnExecFrontSpan *out) {
    if (!commands || !emit || !out || begin >= limit || (begin & 7)) return 0;
    ExecStateBuild b;
    memset(&b, 0, offsetof(ExecStateBuild, writes));
    uint32_t ops = 0, draws = 0;
    size_t end = begin;
    while (ops < EXEC_STATE_OPS && draws < 16 && b.count + 8 <= EXEC_STATE_WRITES && end < limit) {
        if (limit - end < sizeof(op_hdr)) break;
        const op_hdr *h = (const op_hdr *)(commands + end);
        if (h->size < sizeof *h || (h->size & 7) || h->size > limit - end) break;
        size_t need = h->op == OP_DRAW ? sizeof(p_draw) : h->op == OP_DRAW_IDX ? sizeof(p_draw_idx) :
                      h->op == OP_DRAW_IDX_IND ? sizeof(p_draw_idx_ind) : 0;
        if (need) {
            if (h->size - sizeof *h < need) break;
            draws++;
        } else if (!exec_state_compile_op(&b, commands, end, limit)) break;
        end += h->size, ops++;
    }
    // Homogeneous draws already have a compact front-pass span. Require actual repeated state work here.
    if (draws < 2 || ops - draws < 2) return 0;
    exec_state_emit(&b, emit, user);
    *out = (NvnExecFrontSpan){.end = end, .ops = ops, .draws = draws};
    return 1;
}

static int exec_state_merge(ExecStateBuild *dst, const ExecStateBuild *src) {
    for (uint32_t i = 0; i < src->count; i++) {
        if (dst->count + 2 > EXEC_STATE_WRITES) return 0;
        const ExecStateWrite *w = &src->writes[i];
        exec_state_write(dst, w->destination, w->bytes, w->source, w->operation, w->notify);
    }
    dst->dirty |= src->dirty, dst->flags |= src->flags;
    dst->bind_increments += src->bind_increments, dst->texture_increments += src->texture_increments;
    return 1;
}

// Only a near-capacity merge needs a rollback copy. Keep it off the common composer's stack and copy path.
static __attribute__((noinline)) int exec_state_merge_bounded(ExecStateBuild *dst, const ExecStateBuild *src) {
    ExecStateBuild candidate;
    memcpy(&candidate, dst, offsetof(ExecStateBuild, writes) + dst->count * sizeof(ExecStateWrite));
    if (!exec_state_merge(&candidate, src)) return 0;
    memcpy(dst, &candidate, offsetof(ExecStateBuild, writes) + candidate.count * sizeof(ExecStateWrite));
    return 1;
}

static int exec_state_blob_valid(const ExecStateBuild *b, size_t bytes, size_t command_bytes) {
    // The composer can adopt the first blob directly. Check the ordered, disjoint representation it relies on,
    // as well as storage bounds, without repeating the compiler's field classification.
    if (bytes < offsetof(ExecStateBuild, writes) || b->count > EXEC_STATE_WRITES || b->reserved ||
        b->count > (bytes - offsetof(ExecStateBuild, writes)) / sizeof(ExecStateWrite) ||
        (b->flags & ~(ES_HAS_VIEWPORT | ES_HAS_SCISSOR)) || (b->dirty & ~VK_EXEC_DIRTY_ALL)) return 0;
    for (uint32_t i = 0; i < b->count; i++) {
        const ExecStateWrite *w = &b->writes[i];
        if (!w->bytes || w->destination > sizeof(VkExec) || w->bytes > sizeof(VkExec) - w->destination ||
            w->operation > ES_BOOL || w->notify > ES_IMAGE) return 0;
        if (i && b->writes[i - 1].destination + b->writes[i - 1].bytes > w->destination) return 0;
        size_t read_bytes = w->operation == ES_BOOL ? sizeof(int32_t) : w->operation == ES_COPY ? w->bytes : 0;
        if (read_bytes && (w->source > command_bytes || read_bytes > command_bytes - w->source)) return 0;
        if (w->operation == ES_BOOL && w->bytes != 1) return 0;
    }
    return 1;
}

int nvn_exec_front_compose(const uint8_t *commands, size_t command_bytes,
                           const NvnExecStep *steps, size_t nsteps,
                           const uint8_t *state_data, size_t state_bytes, size_t begin_step,
                           NvnExecStateEmit emit, void *user, NvnExecFrontComposeSpan *out) {
    if (!commands || ((uintptr_t)commands & 7) || !steps || !emit || !out || begin_step >= nsteps ||
        (state_bytes && (!state_data || ((uintptr_t)state_data & 7)))) return 0;
    ExecStateBuild b;
    memset(&b, 0, offsetof(ExecStateBuild, writes));
    uint32_t ops = 0, draws = 0;
    size_t end = steps[begin_step].begin, i = begin_step;
    for (; i < nsteps && ops < EXEC_STATE_OPS && draws < 16; i++) {
        const NvnExecStep *step = &steps[i];
        if (step->begin != end || (step->begin & 7) || (step->end & 7) || step->end <= end ||
            step->end > command_bytes || !step->ops || step->ops > (step->end - end) / sizeof(op_hdr) ||
            step->reserved || step->flags != nvn_record_exec_flags(step->opcode)) return 0;
        if (step->ops > EXEC_STATE_OPS - ops) break;
        if (step->opcode == NVN_EXEC_STATE) {
            if (!state_data || step->run || step->ops < 2 || (step->state_offset & 7) || (step->state_bytes & 7) ||
                step->state_offset > state_bytes || step->state_bytes > state_bytes - step->state_offset ||
                step->state_bytes < offsetof(ExecStateBuild, writes)) return 0;
            const ExecStateBuild *src = (const ExecStateBuild *)(state_data + step->state_offset);
            if (!exec_state_blob_valid(src, step->state_bytes, command_bytes) ||
                src->bind_increments > step->ops || src->texture_increments > step->ops) return 0;
            if (!ops) memcpy(&b, src, offsetof(ExecStateBuild, writes) + src->count * sizeof(ExecStateWrite));
            else if (b.count + 2 * src->count <= EXEC_STATE_WRITES) {
                if (!exec_state_merge(&b, src)) return 0;
            } else if (!exec_state_merge_bounded(&b, src)) break;
        } else if (step->opcode == OP_DRAW || step->opcode == OP_DRAW_IDX || step->opcode == OP_DRAW_IDX_IND) {
            size_t payload = step->opcode == OP_DRAW ? sizeof(p_draw) : step->opcode == OP_DRAW_IDX ?
                             sizeof(p_draw_idx) : sizeof(p_draw_idx_ind);
            const op_hdr *h = (const op_hdr *)(commands + end);
            if (step->ops != 1 || step->end - end < sizeof(op_hdr) + payload || h->op != step->opcode ||
                h->size != step->end - end || step->state_bytes || step->state_offset) return 0;
            draws++;
        } else {
            if (step->ops != 1 || step->state_bytes || step->state_offset) return 0;
            if (!(step->flags & NVN_EXEC_STATE_INPUT) || b.count + 8 > EXEC_STATE_WRITES) break;
            const op_hdr *h = (const op_hdr *)(commands + end);
            if (h->op != step->opcode || h->size != step->end - end) return 0;
            if (!exec_state_compile_op(&b, commands, end, step->end)) break;
        }
        ops += step->ops, end = step->end;
    }
    if (draws < 2 || ops - draws < 2) return 0;
    // No primary/state pointers are read after this call: the emitter may grow the allocation they came from.
    exec_state_emit(&b, emit, user);
    *out = (NvnExecFrontComposeSpan){.end = end, .end_step = i, .ops = ops, .draws = draws};
    return 1;
}

#ifndef VK_EXEC_STATE_COMPILE_ONLY
void vk_exec_draw_state_apply(VkExec *ex, const uint8_t *commands, const void *prepared) {
    const ExecStateBuild *b = prepared;
    if (!ex->dc) ex->dc = dc;
    DrawCtx *cx = ex->dc;
    uint32_t dirty_buffer = 0, dirty_texture = 0;
    for (uint32_t i = 0; i < b->count; i++) {
        const ExecStateWrite *w = &b->writes[i];
        uint8_t *dst = (uint8_t *)ex + w->destination;
        const uint8_t *src = commands + w->source;
        if (w->notify == ES_UBO || w->notify == ES_SSBO) {
            p_bind_buf old, next;
            memcpy(&old, dst, sizeof old), memcpy(&next, src, sizeof next);
            if (old.addr != next.addr || old.size != next.size) {
                unsigned slots = w->notify == ES_UBO ? 18 : 16;
                uint32_t base = w->notify == ES_UBO ? offsetof(VkExec, ubo) : offsetof(VkExec, ssbo);
                unsigned index = (w->destination - base) / sizeof(p_bind_buf);
                dirty_buffer |= w->notify == ES_UBO ? cx->prep_ubo[index / slots][index % slots] :
                                                    cx->prep_ssbo[index / slots][index % slots];
            }
        } else if (w->notify == ES_TEXTURE || w->notify == ES_IMAGE) {
            unsigned slots = w->notify == ES_TEXTURE ? 128 : 32;
            uint32_t base = w->notify == ES_TEXTURE ? offsetof(VkExec, tex) : offsetof(VkExec, img);
            unsigned index = (w->destination - base) / sizeof(uint64_t);
            for (unsigned k = 0; k < w->bytes / sizeof(uint64_t); k++) {
                uint64_t old, next;
                memcpy(&old, dst + k * sizeof old, sizeof old), memcpy(&next, src + k * sizeof next, sizeof next);
                if (old == next) continue;
                unsigned slot = index + k;
                if (w->notify == ES_TEXTURE) dirty_texture |= cx->prep_tex[slot / slots][slot % slots];
                else dirty_buffer |= cx->prep_img[slot / slots][slot % slots];
            }
        }
        if (w->operation == ES_COPY) memcpy(dst, src, w->bytes);
        else if (w->operation == ES_ZERO) memset(dst, 0, w->bytes);
        else {
            int32_t value;
            memcpy(&value, src, sizeof value);
            *dst = value != 0;
        }
    }
    cx->prep_dirty[VK_BR_GROUP_BUFFER] |= dirty_buffer;
    cx->prep_dirty[VK_BR_GROUP_TEXTURE] |= dirty_texture;
    cx->bind_gen += b->bind_increments;
    ex->tex_serial += b->texture_increments;
    ex->dirty |= b->dirty;
    if (b->flags & ES_HAS_VIEWPORT) ex->has_vp = 1;
    if (b->flags & ES_HAS_SCISSOR) ex->has_scissor = 1;
}
#endif
