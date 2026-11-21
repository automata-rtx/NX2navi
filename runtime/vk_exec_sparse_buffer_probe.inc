#include "sparse_buffer_probe_reader_spv.h"
#include "sparse_buffer_probe_writer_spv.h"

static void sparse_buffer_probe_program(ProbeSampleProgram *definition, const GfxShader *shader) {
    const uint32_t stage = GFX_STAGE_CS, size = sizeof definition->code[0];
    definition->program = (prog_t){.magic = NVN_PROG_MAGIC, .serial = nvn_new_serial(), .nstage = 1};
    memcpy(definition->control[0] + 0x704, &size, sizeof size);
    memcpy(definition->control[0] + 0x714, &stage, sizeof stage);
    definition->program.sd[0] = (NVNshaderData){.data = (uintptr_t)definition->code[0],
                                                .control = definition->control[0]};
    vk_exec_probe_shader(definition->control[0], shader);
}

static int sparse_buffer_probe_guard(const void *data, size_t bytes, uint8_t expected) {
    const uint8_t *byte = data;
    for (size_t i = 0; i < bytes; i++)
        if (byte[i] != expected) return 0;
    return 1;
}

static int test_sparse_buffer_producer(const GfxBackend *b, uint8_t *guest) {
    enum { WORDS = 1024, BYTES = WORDS * sizeof(uint32_t), GUARD = 64 };
    const GfxShader *writer = gfx_shader_own(sparse_buffer_probe_writer_spv,
        sizeof sparse_buffer_probe_writer_spv / sizeof *sparse_buffer_probe_writer_spv,
        "{\"stage\":\"cs\",\"textures\":[],\"images\":[],\"key\":\"sparse-buffer-probe-writer\",\"cbufs\":[],\"sbufs\":[{\"binding\":1,\"slot\":0,\"cb_slot\":0,\"cb_offset\":196,\"flags\":1}]}");
    const GfxShader *reader = gfx_shader_own(sparse_buffer_probe_reader_spv,
        sizeof sparse_buffer_probe_reader_spv / sizeof *sparse_buffer_probe_reader_spv,
        "{\"stage\":\"cs\",\"textures\":[],\"images\":[],\"key\":\"sparse-buffer-probe-reader\",\"cbufs\":[{\"binding\":1,\"slot\":3}],\"sbufs\":[{\"binding\":2,\"slot\":0,\"cb_slot\":0,\"cb_offset\":132,\"flags\":0},{\"binding\":3,\"slot\":0,\"cb_slot\":0,\"cb_offset\":196,\"flags\":1}]}");
    CHECK(writer && reader && gfx_shader_guest_writes(writer) == RC_EFFECT_WRITES_BUFFER &&
          gfx_shader_write_inventory_complete(writer), "qualified original compute SSBO writer and two reader bindings");
    ProbeSampleProgram write_program = {0}, read_program = {0};
    sparse_buffer_probe_program(&write_program, writer);
    sparse_buffer_probe_program(&read_program, reader);

    pool_t source_pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xc00000,
                          .game_mem = guest + 0xc00000, .size = 0x100000};
    pool_t report_pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xe00000,
                          .game_mem = guest + 0xe00000, .size = 0x100000};
    CHECK(b->pool_init(&source_pool, &source_pool) && b->pool_init(&report_pool, &report_pool),
          "initialized independent source and report pools");
    CHECK(vk_mirror_on && vk_caps.storage_write_alignment && BYTES % vk_caps.storage_write_alignment == 0,
          "real mirror mode and bounded storage writes on this device");
    uint8_t *const source = source_pool.mem + 0x20000, *const prefix = report_pool.mem + 0x30000;
    uint8_t *const report = report_pool.mem + 0x50000;
    const uint64_t address = (uintptr_t)source;
    uint32_t original[WORDS], result[WORDS];
    for (unsigned i = 0; i < WORDS; i++) original[i] = 0x4d37a100u + i * 109u;
    memcpy(source, original, BYTES);
    memset(source - GUARD, 0xb7, GUARD);
    memset(source + BYTES, 0x2d, GUARD);
    memset(prefix - GUARD, 0xe5, BYTES + 2 * GUARD);
    memset(report - GUARD, 0x6f, 2 * BYTES + 2 * GUARD);
    b->guest_written(address - GUARD, BYTES + 2 * GUARD);
    b->guest_written((uintptr_t)(prefix - GUARD), BYTES + 2 * GUARD);
    b->guest_written((uintptr_t)(report - GUARD), 2 * BYTES + 2 * GUARD);

    CHECK(vk_mem_mirror_probe(address), "created the real source mirror through its ordinary first-use setup");
    VkMemSpan imported, mirrored;
    VkMemSpanCapture *hold = NULL;
    CHECK(vk_mem_lookup(address, BYTES, &imported) &&
          vk_mem_span_native_input(address, BYTES, 0, &mirrored, &hold) == RC_REGISTRY_OK &&
          mirrored.buffer != imported.buffer, "selected an actual retained mirror distinct from the host import");
    vk_mem_span_capture_release(&hold);
    _Alignas(8) uint8_t stream[4096];
    uint8_t *next = put(stream, OP_BEGIN, NULL, 0);
    const p_copy_b2b warm = {address, (uintptr_t)prefix, BYTES};
    next = put(next, OP_COPY_B2B, &warm, sizeof warm);
    CHECK(xf_submit(b, stream, next), "mirror initialization and native pre-producer copy completed");
    b->guest_read((uintptr_t)prefix, BYTES);
    CHECK(!memcmp(prefix, original, BYTES), "native pre-producer mirror bytes match the initial CPU pattern");

    // CPU updates after the earlier mirror read belong to the next source version. Include a skipped
    // word, an executed store with an equal value and an executed store with a different value.
    original[3] = 0x7a1b2c00u;
    original[7] = 0xdab0392bu;
    original[9] = 0x6ab4300bu;
    memcpy(source + 3 * sizeof(uint32_t), original + 3, 7 * sizeof(uint32_t));
    b->guest_written(address + 3 * sizeof(uint32_t), 7 * sizeof(uint32_t));
    unsigned changed = 0, equal = 0, skipped = 0;
    for (unsigned i = 0; i < WORDS; i++) {
        uint32_t word = original[i];
        result[i] = word;
        if ((word & 31u) != 11u) skipped++;
        else if (word & 32u) equal++;
        else result[i] ^= 0x57a536c0u, changed++;
    }
    CHECK(changed && equal && skipped && result[3] == original[3] && result[7] == original[7] &&
          result[9] != original[9], "data covers taken, skipped and equal compute stores");

    next = put(stream, OP_BEGIN, NULL, 0);
    const p_bind_program write = {&write_program.program, 1 << GFX_STAGE_CS};
    const p_bind_program read = {&read_program.program, 1 << GFX_STAGE_CS};
    const p_bind_buf src = {GFX_STAGE_CS, 0, address, BYTES};
    const p_bind_buf dst = {GFX_STAGE_CS, 0, (uintptr_t)report, 2 * BYTES};
    const p_dispatch sparse = {.x = WORDS / 64, .y = 1, .z = 1};
    const p_dispatch consume = {.x = WORDS / (4 * 64), .y = 1, .z = 1};
    const p_barrier order = {0, 0};
    next = put(next, OP_BIND_PROGRAM, &write, sizeof write);
    next = put(next, OP_BIND_SSBO, &src, sizeof src);
    next = put(next, OP_DISPATCH, &sparse, sizeof sparse);
    next = put(next, OP_BARRIER, &order, sizeof order);
    next = put(next, OP_BIND_PROGRAM, &read, sizeof read);
    next = put(next, OP_BIND_UBO, &src, sizeof src);
    next = put(next, OP_BIND_SSBO, &dst, sizeof dst);
    next = put(next, OP_DISPATCH, &consume, sizeof consume);
    CHECK(xf_submit(b, stream, next), "ordinary conditional producer and its original-cut consumers completed");
    b->guest_read(address - GUARD, BYTES + 2 * GUARD);
    b->guest_read((uintptr_t)(report - GUARD), 2 * BYTES + 2 * GUARD);
    CHECK(!memcmp(report, result, BYTES), "every original uniform byte matches the GPU sparse result");
    CHECK(!memcmp(report + BYTES, result, BYTES), "every original readonly storage byte matches the GPU sparse result");
    CHECK(!memcmp(source, result, BYTES), "guest storage reconciled exactly after the GPU producer");
    CHECK(sparse_buffer_probe_guard(source - GUARD, GUARD, 0xb7) &&
          sparse_buffer_probe_guard(source + BYTES, GUARD, 0x2d) &&
          sparse_buffer_probe_guard(report - GUARD, GUARD, 0x6f) &&
          sparse_buffer_probe_guard(report + 2 * BYTES, GUARD, 0x6f),
          "all 256 source and output guard bytes unchanged");

    printf("probe-vkexec: sparse SSBO stores changed %u, equal %u, skipped %u, verified %u uniform and readonly bytes and %u guard bytes\n",
           changed, equal, skipped, 2 * BYTES, 4 * GUARD);
    b->pool_fini(&report_pool);
    b->pool_fini(&source_pool);
    return 0;
}
