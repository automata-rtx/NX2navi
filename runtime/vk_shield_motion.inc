// Each supported opaque player or equipment draw uses the game's translated vertex program twice. Keep a small, owned
// copy of its draw inputs until the scene resolve. Scratch buffers and descriptors from the original
// submission may already have retired by then. All Vulkan allocations below belong to the resolve.
#include "motion_owner.h"

enum { PLAYER_DRAWS = 80, EQUIPMENT_DRAWS = 48, SHIELD_DRAWS = PLAYER_DRAWS + EQUIPMENT_DRAWS, SHIELD_BYTES = 2 * 1024 * 1024 };
typedef struct {
    uint32_t offset, bytes;
} ShieldSlice;
typedef struct {
    MotionOwner owner;
    VkPipeState state;
    VkViewport viewport;
    VkRect2D scissor;
    VkImage depth;
    uint64_t camera, order, fence;
    p_draw_idx draw;
    float bias[3], support[3];
    ShieldSlice indices, vertex[16], bank[17];
    const uint8_t *shared_bank[17];
    uint32_t size, geom_size, geometry_id, vertex_mask;
    uint64_t first_vertex;
    uint32_t capacity;
    unsigned char *data;
} ShieldDraw;

typedef struct {
    uint64_t frame;
    unsigned count;
    unsigned players, overflow, bank_count;
    struct {
        uint64_t capture, address;
        uint32_t size, capacity;
        uint8_t *data;
    } banks[64];
    ShieldDraw draws[SHIELD_DRAWS];
} ShieldFrame;

static ShieldFrame shield_frames[2];
static uint64_t shield_last_frame;
static int shield_had_scene;
static VkImage shield_last_depth;
static uint64_t shield_last_camera;
static uint32_t shield_width, shield_height;
static PlatMutex shield_lock = PLAT_MUTEX_INIT;
static int shield_trace(void) {
    static int trace = -1;
    if (trace < 0) trace = rt_knob("SHIELD_TRACE", "0");
    return trace;
}

static int shield_owner(VkExec *ex, MotionOwner *owner) {
    const GfxShaderRes *rs;
    int count = gfx_shader_resources(ex->dc->B.st.vs, GFX_RES_UBO, &rs);
    for (int i = 0; i < count; ++i) {
        int bank = rs[i].slot;
        if (bank < 3 || bank > 16) continue;
        const p_bind_buf *u = &ex->ubo[0][bank - 3];
        MotionOwner part;
        if (!vk_usnap_motion_owner(u->addr, u->size, &part)) continue;
        if (owner->model && (owner->model != part.model || owner->part != part.part)) return 0;
        *owner = part;
    }
    return owner->part != 0;
}

static int shield_copy(ShieldDraw *p, ShieldSlice *dst, const void *src, uint64_t addr, uint64_t size) {
    uint32_t aligned = (p->size + 15u) & ~15u;
    if (aligned > SHIELD_BYTES) return 0;
    if (!size || size > SHIELD_BYTES - aligned) return 0;
    // Reuse the allocation on the next frame of this ring slot. A larger mesh only
    // grows its own packet; the common one-shape equipment keeps its small allocation.
    if (aligned + size > p->capacity) {
        uint32_t capacity = p->capacity ? p->capacity : 4096;
        while (capacity < aligned + size) capacity *= 2;
        unsigned char *data = realloc(p->data, capacity);
        if (!data) return 0;
        p->capacity = capacity;
        p->data = data;
    }
    memset(p->data + p->size, 0, aligned - p->size);
    p->size = aligned;
    uint8_t *to = p->data + p->size;
    if (src)
        memcpy(to, src, size);
    else if (!addr || !vk_usnap_read(addr, size, to))
        return 0;
    *dst = (ShieldSlice){p->size, (uint32_t)size};
    p->size += (uint32_t)size;
    return 1;
}

// Static, weighted vertices are shared by all poses. The history comparison proves the
// current packet still has the previous geometry. Keep that geometry on the GPU and
// upload only the new pose. Never overwrite bytes referenced by an in-flight frame.
enum { PLAYER_MESH_BYTES = 12u << 20, PLAYER_MESH_ENTRIES = 64 };
static struct {
    VkBuffer buffer;
    VmaAllocation allocation;
    uint8_t *mapped;
    struct { uint32_t offset, size; uint8_t *bytes; int uploaded; } entries[PLAYER_MESH_ENTRIES];
    unsigned count, used, cpu_used;
    int attempted;
} player_mesh;

static const uint8_t *geometry_bytes(const ShieldDraw *p) {
    return p->geometry_id ? player_mesh.entries[p->geometry_id - 1].bytes : p->data;
}

// A mesh seen by another depth pass can still recur every frame. Keep its CPU
// data independently of the choice of scene depth. Only selected meshes need GPU space.
static unsigned remember_geometry(const ShieldDraw *p) {
    for (unsigned i = 0; i < player_mesh.count; ++i)
        if (player_mesh.entries[i].size == p->geom_size &&
            !memcmp(player_mesh.entries[i].bytes, p->data, p->geom_size)) return i + 1;
    if (player_mesh.count == PLAYER_MESH_ENTRIES || p->geom_size > PLAYER_MESH_BYTES - player_mesh.cpu_used) return 0;
    uint8_t *bytes = malloc(p->geom_size);
    if (!bytes) return 0;
    memcpy(bytes, p->data, p->geom_size);
    unsigned i = player_mesh.count++;
    player_mesh.entries[i].bytes = bytes;
    player_mesh.entries[i].size = p->geom_size;
    player_mesh.cpu_used += p->geom_size;
    return i + 1;
}

// Test the actual bound ranges against the retained index and vertex bytes. A changed
// stream, layout or topology takes the full capture path and has to earn fresh history.
static int reuse_motion_geometry(VkExec *ex, ShieldDraw *p, const ShieldFrame *last) {
    for (unsigned i = 0; i < last->count; ++i) {
        const ShieldDraw *q = &last->draws[i];
        if (q->owner.lifetime != p->owner.lifetime || q->owner.part != p->owner.part ||
            q->draw.ibuf != p->draw.ibuf || q->draw.count != p->draw.count || q->draw.itype != p->draw.itype ||
            q->draw.base_vertex != p->draw.base_vertex || q->vertex_mask != p->vertex_mask ||
            memcmp(q->state.vattr, p->state.vattr, sizeof p->state.vattr) ||
            memcmp(q->state.vstream, p->state.vstream, sizeof p->state.vstream)) continue;
        const uint8_t *data = geometry_bytes(q);
        if (!vk_usnap_equal_tail(p->draw.ibuf, q->indices.bytes, 0, data + q->indices.offset)) continue;
        unsigned j = 0;
        for (; j < 16; ++j) {
            ShieldSlice v = q->vertex[j];
            if (!v.bytes) continue;
            // The indices already matched. Per-vertex attributes have unsigned offsets,
            // so this draw cannot fetch any byte before its smallest effective vertex.
            uint64_t skip = q->first_vertex * p->state.vstream[j].stride;
            if (v.bytes > ex->vbuf[j].size || skip >= v.bytes ||
                !vk_usnap_equal_tail(ex->vbuf[j].address, v.bytes, skip, data + v.offset)) break;
        }
        if (j != 16) continue;
        // A cache entry must earn its place by recurring with the same fetched data.
        // Rigid equipment shares the same proof and lifetime checks as the animated mesh.
        unsigned id = q->geometry_id ? q->geometry_id : remember_geometry(q);
        if (!id) return 0;
        p->first_vertex = q->first_vertex;
        p->indices = q->indices;
        memcpy(p->vertex, q->vertex, sizeof p->vertex);
        p->geom_size = q->geom_size;
        p->geometry_id = id;
        return 1;
    }
    return 0;
}

static int shield_geometry(VkExec *ex, ShieldDraw *p) {
    const p_draw_idx *d = &p->draw;
    unsigned width = d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 4;
    if (d->base_vertex < 0 || !shield_copy(p, &p->indices, NULL, d->ibuf, (uint64_t)d->count * width)) return 0;
    uint32_t low = UINT32_MAX, high = 0;
    if (width == 2) {
        const uint16_t *indices = (const uint16_t *)p->data;
        for (int i = 0; i < d->count; ++i) {
            if (indices[i] > high) high = indices[i];
            if (indices[i] < low) low = indices[i];
        }
    } else {
        const uint32_t *indices = (const uint32_t *)p->data;
        for (int i = 0; i < d->count; ++i) {
            if (indices[i] > high) high = indices[i];
            if (indices[i] < low) low = indices[i];
        }
    }
    p->first_vertex = (uint64_t)d->base_vertex + low;
    for (unsigned i = 0; i < 16; ++i) {
        if (!(p->vertex_mask & (1u << i))) continue;
        va_st a;
        memcpy(&a, &p->state.vattr[i], sizeof a);
        if (!a.fmt) continue;
        if (a.stream >= 16) return 0;
        if (p->vertex[a.stream].bytes) continue;
        const NVNbufferRange *v = &ex->vbuf[a.stream];
        const vs_st *stream = &p->state.vstream[a.stream];
        uint64_t bytes = ((uint64_t)high + d->base_vertex + 1) * stream->stride;
        if (stream->divisor || !bytes || bytes > v->size ||
            !shield_copy(p, &p->vertex[a.stream], NULL, v->address, bytes))
            return 0;
    }
    p->geom_size = p->size;
    return 1;
}

// vk_usnap_bytes validates the range at the draw's replay epoch. Equal ranges
// within that immutable submission snapshot share one owned copy for both poses.
static const uint8_t *retain_bank(ShieldFrame *f, uint64_t capture, uint64_t address, uint32_t size, const uint8_t *src) {
    if (size < 512 || size > 65536 || !capture) return NULL;
    for (unsigned i = f->bank_count; i--;) {
        if (f->banks[i].capture == capture && f->banks[i].address == address && f->banks[i].size == size)
            return f->banks[i].data;
    }
    if (f->bank_count == sizeof f->banks / sizeof *f->banks) return NULL;
    unsigned i = f->bank_count;
    if (f->banks[i].capacity < size) {
        uint8_t *data = realloc(f->banks[i].data, size);
        if (!data) return NULL;
        f->banks[i].data = data;
        f->banks[i].capacity = size;
    }
    memcpy(f->banks[i].data, src, size);
    f->banks[i].capture = capture;
    f->banks[i].address = address;
    f->banks[i].size = size;
    ++f->bank_count;
    return f->banks[i].data;
}

static const uint8_t *pose_bank(const ShieldDraw *p, unsigned slot) {
    return p->shared_bank[slot] ? p->shared_bank[slot] : p->data + p->bank[slot].offset;
}

static int same_pose(const ShieldDraw *a, const ShieldDraw *b) {
    for (unsigned i = 0; i < 17; ++i) {
        uint32_t bytes = a->bank[i].bytes;
        if (bytes != b->bank[i].bytes) return 0;
        if (bytes && pose_bank(a, i) != pose_bank(b, i) && memcmp(pose_bank(a, i), pose_bank(b, i), bytes)) return 0;
    }
    return 1;
}

static int shield_uniforms(VkExec *ex, ShieldFrame *f, ShieldDraw *p) {
    const GfxShaderRes *r;
    int n = gfx_shader_resources(p->state.vs, GFX_RES_UBO, &r);
    for (int i = 0; i < n; ++i) {
        int slot = r[i].slot;
        if (slot < 0 || slot > 16 || r[i].count > 1) return 0;
        if (p->bank[slot].bytes) continue;
        if (slot == 0) {
            uint8_t bank[VK_C0_BYTES] = {0};
            if (c0_on()) vk_c0_fill(bank, 0, ex->ssbo[0], ex->ubo[0], ex->dc->c0_dim, ex->dc->c0_binst);
            if (!shield_copy(p, &p->bank[0], bank, 0, sizeof bank)) return 0;
        } else if (slot == 1) {
            const uint8_t *ctl, *code;
            uint32_t len;
            if (!find_stage(ex->dc, ex->prog, NVN_STAGE_VS, &ctl, &code, &len)) return 0;
            uint32_t off = ctl_c1(ctl, len);
            if (ctl_nvn2(ctl)) memcpy(&len, ctl + 0x44, 4);
            if (off >= len || !shield_copy(p, &p->bank[1], code + off, 0, len - off)) return 0;
        } else {
            if (slot < 3) return 0;
            const p_bind_buf *u = &ex->ubo[0][slot - 3];
            const uint8_t *snapshot = vk_usnap_bytes(u->addr, u->size);
            if (!snapshot || !u->size || u->size > SHIELD_BYTES) return 0;
            p->shared_bank[slot] = retain_bank(f, vk_usnap_capture(), u->addr, (uint32_t)u->size, snapshot);
            if (p->shared_bank[slot]) p->bank[slot].bytes = (uint32_t)u->size;
            else if (!shield_copy(p, &p->bank[slot], snapshot, 0, u->size)) return 0;
        }
    }
    return 1;
}

static void shield_capture(VkExec *ex, const p_draw_idx *d) {
    const VkPipeState *s = &ex->dc->B.st;
    if (!motion_owner_enabled() || !vk_dlaa_available() || !ex->dc->B.st_valid || !s->vs ||
        ex->ubo[0][0].size != 0x3500 || !ex->color[0].image || !s->ds.dt ||
        (!s->ds.dw && s->ds.df != NVN_DEPTH_FUNC_EQUAL) || s->color.blend_en || s->ds.st || s->ms.a2c ||
        !ex->depth.image || ex->depth.format != VK_FORMAT_D32_SFLOAT || ex->depth.level || ex->depth.base_layer ||
        ex->pass_layers != 1 || ex->depth.layers != 1 || s->samples > 1 || s->raster_discard || d->instances > 1 ||
        d->base_instance || d->itype == NVN_INDEX_TYPE_UNSIGNED_BYTE)
        return;
    MotionOwner owner = {0};
    if (!shield_owner(ex, &owner)) return;
    const GfxShaderRes *resources;
    if (gfx_shader_resources(s->vs, GFX_RES_IMG, &resources) || gfx_shader_resources(s->vs, GFX_RES_TEX, &resources) ||
        gfx_shader_resources(s->vs, GFX_RES_SSBO, &resources))
        return;
    VkViewport v = draw_viewport(ex);
    if (v.width != ex->depth.width || -v.height != ex->depth.height || fabsf(v.x) > 1 || fabsf(v.y + v.height) > 1)
        return;
    uint64_t frame = vk_dlaa_scene_frame();
    plat_mutex_lock(&shield_lock);
    ShieldFrame *f = &shield_frames[frame & 1];
    if (f->frame != frame) f->frame = frame, f->count = f->players = f->overflow = f->bank_count = 0;
    int player = owner.kind == MOTION_PLAYER;
    if (player ? f->players == PLAYER_DRAWS : f->count - f->players == EQUIPMENT_DRAWS) {
        f->overflow |= player ? 1u : 2u;
        plat_mutex_unlock(&shield_lock);
        return;
    }
    ShieldDraw *p = &f->draws[f->count];
    memset(p, 0, offsetof(ShieldDraw, capacity));
    p->owner = owner;
    p->draw = *d;
    p->state = *s;
    p->depth = ex->depth.image;
    p->camera = ex->ubo[0][0].addr;
    p->order = vk_exec_stream_pos();
    p->fence = ex->fence;
    p->viewport = v;
    p->scissor = vk_exec_scissor(ex);
    support_in(ex, p->support);
    p->bias[0] = ex->poly_offset.units;
    p->bias[1] = ex->poly_offset.clamp;
    p->bias[2] = ex->poly_offset.factor;
    p->vertex_mask = gfx_shader_inputs(p->state.vs, NULL);
    const ShieldFrame *last = &shield_frames[(frame & 1) ^ 1];
    int geometry = (last->frame + 1 == frame && reuse_motion_geometry(ex, p, last)) || shield_geometry(ex, p);
    int uniforms = geometry && shield_uniforms(ex, f, p);
    if (uniforms) {
        int pending = 0, has = 0;
        ex->dc->sf_mvprev = 1;
        const GfxShader *variant = shader_for(ex, NVN_STAGE_VS, &has, &pending, 0);
        ex->dc->sf_mvprev = 0;
        if (variant && !pending) {
            GfxShaderIo io[32];
            uint32_t extra = gfx_shader_outputs(variant, io) & ~gfx_shader_outputs(s->vs, io);
            unsigned banks = gfx_shader_mvprev_banks(variant);
            if (extra && (banks & (1u << 3)) && (banks & ~(1u << 3))) {
                p->state.fs = mo_fs(31 - __builtin_clz(extra));
                p->state.vs = variant;
                if (p->state.fs) {
                    ++f->count;
                    f->players += player;
                }
            }
        }
    }
    if (shield_trace() && (frame % 30 == 0 || f->count < 2))
        rt_log("shield capture: frame %llu part %llu geometry %d uniforms %d retained %u bytes %u",
               (unsigned long long)frame, (unsigned long long)owner.part, geometry, uniforms, f->count, p->size);
    plat_mutex_unlock(&shield_lock);
}

static const ShieldDraw *shield_previous(const ShieldFrame *f, const ShieldDraw *p) {
    const ShieldDraw *found = NULL;
    for (unsigned i = 0; i < f->count; ++i) {
        const ShieldDraw *q = f->draws + i;
        if (q->camera != shield_last_camera || q->depth != shield_last_depth ||
            q->owner.lifetime != p->owner.lifetime || q->owner.part != p->owner.part ||
            q->owner.model != p->owner.model || q->state.vs != p->state.vs || q->draw.ibuf != p->draw.ibuf ||
            q->draw.count != p->draw.count || q->draw.base_vertex != p->draw.base_vertex ||
            q->draw.itype != p->draw.itype || q->geom_size != p->geom_size ||
            memcmp(q->state.vattr, p->state.vattr, sizeof p->state.vattr) ||
            memcmp(q->state.vstream, p->state.vstream, sizeof p->state.vstream) ||
            memcmp(q->vertex, p->vertex, sizeof p->vertex) || (!(p->geometry_id && p->geometry_id == q->geometry_id) &&
             memcmp(geometry_bytes(q), geometry_bytes(p), p->geom_size)))
            continue;
        if (found && !same_pose(q, found)) return NULL;
        found = q;
    }
    return found;
}

static VkDescriptorBufferInfo shield_upload(VkExec *ex, const void *src, uint32_t bytes) {
    VkDescriptorBufferInfo d = {.range = bytes};
    memcpy(scratch(ex, bytes, &d.buffer, &d.offset), src, bytes);
    return d;
}

static void player_mesh_shutdown(void) {
    if (player_mesh.buffer) vmaDestroyBuffer(vk_ctx()->vma, player_mesh.buffer, player_mesh.allocation);
    for (unsigned i = 0; i < player_mesh.count; ++i) free(player_mesh.entries[i].bytes);
    memset(&player_mesh, 0, sizeof player_mesh);
    for (unsigned f = 0; f < 2; ++f)
        for (unsigned i = 0; i < SHIELD_DRAWS; ++i) free(shield_frames[f].draws[i].data);
    for (unsigned f = 0; f < 2; ++f)
        for (unsigned i = 0; i < sizeof shield_frames[f].banks / sizeof *shield_frames[f].banks; ++i)
            free(shield_frames[f].banks[i].data);
    memset(shield_frames, 0, sizeof shield_frames);
    shield_had_scene = 0;
}

static int motion_geometry_uploaded(unsigned id) {
    if (!id) return 0;
    unsigned i = id - 1;
    if (player_mesh.entries[i].uploaded) return 1;
    if (!player_mesh.attempted) {
        player_mesh.attempted = 1;
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = PLAYER_MESH_BYTES,
                                 .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                      .requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &player_mesh.buffer, &player_mesh.allocation, &info) == VK_SUCCESS)
            player_mesh.mapped = info.pMappedData;
    }
    if (!player_mesh.mapped) return 0;
    unsigned offset = (player_mesh.used + 15u) & ~15u;
    unsigned size = player_mesh.entries[i].size;
    if (size > PLAYER_MESH_BYTES - offset) return 0;
    memcpy(player_mesh.mapped + offset, player_mesh.entries[i].bytes, size);
    player_mesh.entries[i].offset = offset;
    player_mesh.entries[i].uploaded = 1;
    player_mesh.used = offset + size;
    return 1;
}

// Bones and camera constants often serve several shapes. The source packets stay
// immutable for this resolve and scratch descriptors live in this command buffer.
typedef struct {
    const void *data;
    VkDescriptorBufferInfo buffer;
} PoseUpload;

static int shield_emit(VkExec *ex, ShieldDraw *p, const ShieldDraw *prev, uint32_t w, uint32_t h, PoseUpload uploads[32]) {
    VkPipeState state = p->state;
    memset(state.color_fmt, 0, sizeof state.color_fmt);
    state.color_fmt[0] = NVN_FORMAT_RG16F;
    state.chmask.mask = 3;
    state.feedback = state.color.logic_op = 0;
    state.ds.dw = 0;
    // Match the final visible depth, including holes in the depth-writing silhouette.
    state.ds.df = NVN_DEPTH_FUNC_EQUAL;
    int pending = 0;
    const VkPipeInfo *info = NULL;
    VkPipeline pipe = vk_pipeline_graphics_async(&state, &info, &pending, 0);
    if (pending || !pipe || !info || info->pool || info->n_bindings > MAX_BINDINGS) return 0;
    VkDescriptorBufferInfo inputs[MAX_BINDINGS];
    VkWriteDescriptorSet writes[MAX_BINDINGS];
    float screen[4] = {(float)w, (float)h, p->viewport.x, p->viewport.y - h};
    uint8_t support[SUPPORT_BYTES];
    support_fill(support, p->support[0], p->support[1], p->support[2]);
    for (uint32_t i = 0; i < info->n_bindings; ++i) {
        const VkPipeBinding *b = info->bindings + i;
        const void *data;
        uint32_t bytes;
        PoseUpload *shared = NULL;
        if (b->type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || b->count > 1) return 0;
        if (b->kind == VK_PIPE_SUPPORT) {
            data = support;
            bytes = sizeof support;
        } else if (b->kind != GFX_RES_UBO)
            return 0;
        else if (nvn_stage_of(b->stage) == NVN_STAGE_FS) {
            data = screen;
            bytes = sizeof screen;
        } else {
            int slot = b->res->slot;
            const ShieldDraw *source = p;
            if (slot >= 18) {
                slot -= 15;
                source = prev;
            }
            if (slot < 0 || slot > 16 || !source->bank[slot].bytes) return 0;
            ShieldSlice bank = source->bank[slot];
            data = pose_bank(source, slot);
            bytes = bank.bytes;
            if (bytes >= 512) shared = &uploads[b->res->slot];
        }
        if (shared && shared->buffer.range == bytes &&
            (data == shared->data || !memcmp(data, shared->data, bytes))) {
            inputs[i] = shared->buffer;
        } else {
            inputs[i] = shield_upload(ex, data, bytes);
            if (shared) *shared = (PoseUpload){data, inputs[i]};
        }
        writes[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                           .dstBinding = b->binding,
                                           .descriptorCount = 1,
                                           .descriptorType = b->type,
                                           .pBufferInfo = inputs + i};
    }
    VkCommandBuffer cb = ex->cb;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vk_pipeline_set_dynamic(cb, &state);
    vkCmdSetViewportWithCount(cb, 1, &p->viewport);
    vkCmdSetScissorWithCount(cb, 1, &p->scissor);
    vkCmdSetDepthBias(cb, p->bias[0], p->bias[1], p->bias[2]);
    for (uint32_t set = 0, first = 0; first < info->n_bindings && set < 2; ++set) {
        uint32_t end = first;
        while (end < info->n_bindings && (info->n_sets < 2 || info->bindings[end].set == set))
            ++end;
        if (end == first) continue;
        if (set ? info->push1 : info->push)
            rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_GRAPHICS, info->layout, set, end - first,
                     writes + first);
        else {
            VkDescriptorSet ds = alloc_set(ex, set ? info->set_layout1 : info->set_layout);
            for (uint32_t i = first; i < end; ++i)
                writes[i].dstSet = ds;
            vkUpdateDescriptorSets(vk_ctx()->device, end - first, writes + first, 0, NULL);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, info->layout, set, 1, &ds, 0, NULL);
        }
        first = end;
    }
    int uploaded = motion_geometry_uploaded(p->geometry_id);
    const uint8_t *geometry = geometry_bytes(p);
    VkDeviceSize mesh_offset = uploaded ? player_mesh.entries[p->geometry_id - 1].offset : 0;
    MoVb vb = {0};
    for (unsigned i = 0; i < 17; ++i) {
        vb.b[i] = Z.zbuf;
        vb.sz[i] = ZERO_BYTES;
        if (i == 16) continue;
        vb.str[i] = state.vstream[i].stride;
        ShieldSlice v = p->vertex[i];
        if (v.bytes && uploaded) {
            vb.b[i] = player_mesh.buffer;
            vb.o[i] = mesh_offset + v.offset;
            vb.sz[i] = v.bytes;
        } else if (v.bytes) {
            VkDescriptorBufferInfo u = shield_upload(ex, geometry + v.offset, v.bytes);
            vb.b[i] = u.buffer;
            vb.o[i] = u.offset;
            vb.sz[i] = u.range;
        }
    }
    vkCmdBindVertexBuffers2(cb, 0, 17, vb.b, vb.o, vb.sz, vk_pipeline_vertex_input_dynamic() ? NULL : vb.str);
    VkIndexType type = p->draw.itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    if (uploaded)
        vkCmdBindIndexBuffer(cb, player_mesh.buffer, mesh_offset + p->indices.offset, type);
    else
        bind_index(ex, geometry + p->indices.offset, p->indices.bytes, type);
    vkCmdDrawIndexed(cb, p->draw.count, 1, 0, p->draw.base_vertex, 0);
    return 1;
}

int vk_shield_resolve(VkExec *ex, uint64_t camera, VkImage depth, VkImageView zv, VkImageView mv, uint32_t w,
                      uint32_t h, int reset) {
    if (!motion_owner_enabled()) return 0;
    uint64_t frame = vk_dlaa_scene_frame();
    ShieldFrame *now = &shield_frames[frame & 1], *last = &shield_frames[(frame & 1) ^ 1];
    unsigned written = 0, cross = 0, matched = 0;
    // The scene hook drains earlier recording groups. Use the original stream order for coplanar ties.
    if (!reset && shield_had_scene && shield_last_frame + 1 == frame && now->frame == frame &&
        last->frame == frame - 1 && shield_width == w && shield_height == h) {
        ShieldDraw *sorted[SHIELD_DRAWS];
        unsigned n = 0;
        for (unsigned i = 0; i < now->count; ++i) {
            ShieldDraw *p = now->draws + i;
            if ((now->overflow | last->overflow) & (p->owner.kind == MOTION_PLAYER ? 1u : 2u)) continue;
            if (p->camera != camera || p->depth != depth) continue;
            unsigned j = n++;
            while (j && (sorted[j - 1]->fence > p->fence ||
                         (sorted[j - 1]->fence == p->fence && sorted[j - 1]->order > p->order))) {
                sorted[j] = sorted[j - 1];
                --j;
            }
            sorted[j] = p;
        }
        PoseUpload uploads[32] = {0};
        for (unsigned i = 0; i < n; ++i) {
            ShieldDraw *p = sorted[i];
            const ShieldDraw *q = shield_previous(last, p);
            if (!q) continue;
            if (!matched++) {
                VkRenderingAttachmentInfo color = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                                                   .imageView = mv,
                                                   .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                   .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                                   .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
                VkRenderingAttachmentInfo z = color;
                z.imageView = zv;
                VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                                      .renderArea = {{0, 0}, {w, h}},
                                      .layerCount = 1,
                                      .colorAttachmentCount = 1,
                                      .pColorAttachments = &color,
                                      .pDepthAttachment = &z};
                vkCmdBeginRendering(ex->cb, &ri);
            }
            if (shield_emit(ex, p, q, w, h, uploads)) {
                ++written;
                cross += p->fence != ex->fence;
            }
        }
        if (matched) {
            vkCmdEndRendering(ex->cb);
            vk_exec_draw_new_cb();
            ex->dirty = VK_EXEC_DIRTY_ALL;
            ex->recorded += written + 2;
        }
    }
    if (shield_trace() && (frame % 15 == 0 || written))
        rt_log("shield resolve: frame %llu captures %u matched %u emitted %u across submissions %u reset %d",
               (unsigned long long)frame, now->frame == frame ? now->count : 0, matched, written, cross, reset);
    shield_had_scene = 1;
    shield_last_frame = frame;
    shield_last_camera = camera;
    shield_last_depth = depth;
    shield_width = w;
    shield_height = h;
    return written != 0;
}
