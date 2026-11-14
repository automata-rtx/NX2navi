// Included after shader_for_resolve. Packets keep only completed shader selections; resources and draw ordering
// still pass through the ordinary renderer. Program code/control data is immutable for its nvnProgramSetShaders serial.
#pragma once

static int shader_packet_matches(const ShaderPacket *p, const VkExec *ex, int stage, uint32_t gen, uint32_t mask) {
    if (!p->shader || p->prog != ex->prog || p->serial != ex->prog->serial || p->stage != stage || p->mask != mask ||
        (mask && p->gen != gen) || p->mvprev != (stage == NVN_STAGE_VS && ex->dc->sf_mvprev != 0)) return 0;
    // vk_pipeline_answers reads only these fields. Blend, depth/stencil, culling and vertex strides select a
    // pipeline/dynamic state, but do not select a different translated shader.
    if (stage == NVN_STAGE_VS) for (unsigned i = 0; i < 16; i++) {
        va_st attribute;
        memcpy(&attribute, &ex->pipe.vattr[i], sizeof attribute);
        if (p->vformat[i] != attribute.fmt) return 0;
    }
    if (stage == NVN_STAGE_FS && memcmp(p->color_fmt, ex->pipe.color_fmt, sizeof p->color_fmt)) return 0;
    // Compare the selected values directly, including on copied/speculative states with equal tex_serial values.
    for (uint32_t m = mask; m; m &= m - 1) {
        unsigned slot = (unsigned)__builtin_ctz(m);
        if (p->tex[slot] != ex->tex[stage][slot] || p->img[slot] != ex->img[stage][slot]) return 0;
    }
    return 1;
}

static void shader_packet_store(ShaderPacket *p, const VkExec *ex, int stage, uint32_t gen, uint32_t mask,
                                 const GfxShader *shader) {
    p->prog = ex->prog, p->serial = ex->prog->serial, p->stage = (uint8_t)stage;
    p->gen = gen, p->mask = mask, p->mvprev = stage == NVN_STAGE_VS && ex->dc->sf_mvprev != 0;
    if (stage == NVN_STAGE_VS) for (unsigned i = 0; i < 16; i++) {
        va_st attribute;
        memcpy(&attribute, &ex->pipe.vattr[i], sizeof attribute);
        p->vformat[i] = (uint8_t)attribute.fmt;
    }
    if (stage == NVN_STAGE_FS) memcpy(p->color_fmt, ex->pipe.color_fmt, sizeof p->color_fmt);
    for (uint32_t m = mask; m; m &= m - 1) {
        unsigned slot = (unsigned)__builtin_ctz(m);
        p->tex[slot] = ex->tex[stage][slot], p->img[slot] = ex->img[stage][slot];
    }
    p->shader = shader;
}

static ShaderPacket *shader_packet_find(ShaderPackets *cache, const VkExec *ex, int stage, uint32_t gen, uint32_t mask,
                                        uint32_t *index) {
    ShaderPacket *last = cache->last[stage];
    if (last && shader_packet_matches(last, ex, stage, gen, mask)) return last;
    *index = (uint32_t)((ex->prog->serial * 0x9e3779b97f4a7c15ull ^ (uint64_t)stage * 0xd6e8feb86659fd93ull) >> 58);
    for (unsigned way = 0; way < SHADER_PACKET_WAYS; way++) {
        ShaderPacket *p = &cache->entry[*index][way];
        if (p != last && shader_packet_matches(p, ex, stage, gen, mask)) return cache->last[stage] = p;
    }
    return NULL;
}

static const GfxShader *shader_for(VkExec *ex, int stage, int *present, int *pending, uint32_t await_ms) {
    int mode = resource_prep_mode();
    if (!mode || !ex->prog || !ex->prog->serial || (unsigned)stage >= 8 || n_probe_sh || vk_texture_deferred())
        return shader_for_resolve(ex, stage, present, pending, await_ms);
    ShaderPackets *cache = &ex->dc->shader_packets;
    uint64_t serial = ex->prog->serial;
    uint32_t mask = slot_mask_of(ex->dc, serial, stage), gen = vk_texture_slots_gen();
    uint32_t index = 0;
    ShaderPacket *p = shader_packet_find(cache, ex, stage, gen, mask, &index);
    if (p) {
        const GfxShader *result = p->shader;
        if (mode == 2 || shader_memo_on() == 2) {
            int actual_present = 0, actual_pending = 0;
            const GfxShader *actual = shader_for_resolve(ex, stage, &actual_present, &actual_pending, await_ms);
            cache->checked++;
            if (vk_texture_deferred() || (mask && vk_texture_slots_gen() != gen) ||
                slot_mask_of(ex->dc, serial, stage) != mask) {
                p->shader = NULL, cache->incomplete++;
                *present = actual_present;
                if (pending) *pending = actual_pending;
                return actual;
            }
            if (!actual_present || actual_pending || actual != result)
                rt_fatal("vk_exec: shader packet of program %llu stage %d differs from ordered selection",
                         (unsigned long long)serial, stage);
        }
        cache->hits[stage]++;
        *present = 1;
        if (pending) *pending = 0;
        return result;
    }
    cache->misses[stage]++;
    const GfxShader *result = shader_for_resolve(ex, stage, present, pending, await_ms);
    if (!result || !*present || (pending && *pending) || vk_texture_deferred() ||
        (mask && vk_texture_slots_gen() != gen) || slot_mask_of(ex->dc, serial, stage) != mask) {
        cache->incomplete++;
        return result;
    }
    // Two ways bound both memory and comparisons. Replacement occurs on misses only, with no large MRU copies.
    ShaderPacket *set = cache->entry[index];
    unsigned way = !set[0].shader ? 0 : !set[1].shader ? 1 : cache->next[index]++ % SHADER_PACKET_WAYS;
    shader_packet_store(&set[way], ex, stage, gen, mask, result);
    cache->last[stage] = &set[way];
    return result;
}
