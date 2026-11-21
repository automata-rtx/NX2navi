typedef enum { PAYLOAD_EXACT, PAYLOAD_UNHANDED, PAYLOAD_REMAP, PAYLOAD_UNKNOWN } PayloadCase;

static void cross_scene_case(PayloadCase mode) {
    begin((FixtureRange){.address = UINT64_C(0x1586ac000), .offset = UINT64_C(0x286ac000), .size = MAX_BYTES,
                         .update_offset = 32, .update_size = 64, .scene = 12731,
                         .update_op = 12609, .read_op = 517});
    cross_scene_fixture = 1;
    const unsigned initial_parts = payload_parts, initial_drops = payload_part_drop;
    const RcRegistryPosition prior_output = {recorder.fence, recorder.resource_op * 2 + 1};
    RcRegistryWrite prior = declare(input.update_offset, input.update_size, prior_output.ordinal);
    emit_inline(prior, input.update_offset, input.update_size);
    const uint64_t archive_bytes = sizeof(VkUniformPayload) + sizeof(RcInterval) + input.update_size;
    assert(budget.retained.amount[RC_BUDGET_RAM] == archive_bytes + sizeof(p_update_ub) &&
           budget.retained.amount[RC_BUDGET_COMMIT] == archive_bytes + sizeof(p_update_ub));
    assert(active_part.part && !vk_recording_part_disposition_handed(active_part.part) &&
           payload_parts == initial_parts + 1);
    finish_fixture_part(mode != PAYLOAD_UNHANDED);
    OK(rc_registry_handoff_through(registry, prior_output));
    OK(rc_registry_write_submit(registry, prior, prior_output.scene));
    OK(rc_registry_write_complete(registry, prior, 1));
    vk_exec_draw_update_history_end(recorder.fence);
    assert(!UPD.fence && !UPD.n && payload_part_drop == initial_drops);
    assert(budget.retained.amount[RC_BUDGET_RAM] == archive_bytes && budget.retained.amount[RC_BUDGET_COMMIT] == archive_bytes);
    // The live CPU-writable import has meanwhile changed. Only the immutable command bytes are authoritative.
    memset(buffers[NATIVE] + OUT_OFFSET, 0x4f, MAX_BYTES);

    recorder.fence += 7;
    U.fence = recorder.fence;
    recorder.resource_op = 511;
    OK(rc_registry_declare_through(registry, vk_exec_resource_position()));
    input.update_offset = 16;
    input.update_size = 16;
    copies = barriers = notes = commits = gpu_writes = 0;
    RcRegistryWrite current = declare(input.update_offset, input.update_size, recorder.resource_op * 2 + 1);
    emit_inline(current, input.update_offset, input.update_size);
    finish_fixture_part(0);
    advance(513);
    input.update_offset = 48;
    input.update_size = 16;
    copies = barriers = notes = commits = gpu_writes = 0;
    RcRegistryWrite overlap = declare(input.update_offset, input.update_size, recorder.resource_op * 2 + 1);
    emit_inline(overlap, input.update_offset, input.update_size);
    finish_fixture_part(0);
    RcRegistryWrite unknown = {0};
    if (mode == PAYLOAD_UNKNOWN)
        OK(rc_registry_write_begin_at(registry, (RcSourceRange){BACKING, input.offset + 200, 4},
            RC_REGISTRY_UNKNOWN, (RcRegistryPosition){recorder.fence, 1029}, &unknown));
    advance(input.read_op);

    UboMergePlan plan = {0};
    GraphicsPayloadDemand demand = {.alignment = ALIGN, .bytes = OUT_OFFSET};
    VkDescriptorBufferInfo descriptor = {0};
    assert(ubo_merge_plan(&recorder, &pipeline, recorder.resource_op, &demand, &plan, NULL));
    if (mode == PAYLOAD_EXACT || mode == PAYLOAD_REMAP) {
        assert(plan.count == 1 && plan.entries[0].fragment_count == 4 && !plan.entries[0].native_count);
        assert(plan.entries[0].gap_count == 2 && plan.entries[0].bytes == MAX_BYTES);
        const GraphicsUboFragment *winners = graphics_ubo_sealed(plan.entries);
        assert(winners[0].offset == 16 && winners[0].bytes == 16 && !winners[0].payload);
        assert(winners[1].offset == 32 && winners[1].bytes == 16 && winners[1].payload);
        assert(winners[2].offset == 48 && winners[2].bytes == 16 && !winners[2].payload);
        assert(winners[3].offset == 64 && winners[3].bytes == 32 && winners[3].payload == winners[1].payload);
        assert(!memcmp(winners[1].data, (uint8_t[]){0xa1,0xa2,0xa3,0xa4}, 4));
        if (mode == PAYLOAD_REMAP) {
            live_mapping++;
            assert(!graphics_ubo_merge_current(&recorder, plan.entries));
            assert(!ubo_merges_initialize(&recorder, &plan, buffers[MERGED]));
        } else {
            int copied = 0;
            assert(ubo_merges_initialize(&recorder, &plan, buffers[MERGED]));
            assert(ubo_merges_record_native(&recorder, &plan, MERGED, 0, &copied) && !copied);
            UpdateSlab slab = {.chunk = {.buf = MERGED, .map = buffers[MERGED]}};
            VkExecGraphicsScratch graphics = {.ubo_merges = plan.entries, .ubo_merge_count = 1,
                                             .slab = &slab, .native_materialized = 1};
            graphics_payload_bound = &graphics;
            assert(uniform_buffer_descriptor(&recorder, &bindings[0], NVN_STAGE_VS, 3, input.address, input.size, ALIGN,
                                             1, VK_MUSE_UBO, &descriptor, NULL));
            graphics_payload_bound = NULL;
            assert(descriptor.buffer == MERGED && descriptor.offset == OUT_OFFSET && descriptor.range == MAX_BYTES);
            const uint8_t *bytes = descriptor_bytes(descriptor);
            for (unsigned i = 0; i < MAX_BYTES; ++i) {
                uint8_t expected = original_byte(i);
                if (i >= 16 && i < 32) expected = (uint8_t)(0xa1 + i - 16);
                else if (i >= 48 && i < 64) expected = (uint8_t)(0xa1 + i - 48);
                else if (i >= 32 && i < 96) expected = (uint8_t)(0xa1 + i - 32);
                assert(bytes[i] == expected);
            }
        }
    } else {
        assert(!plan.count && !plan.entries);
    }
    if (mode != PAYLOAD_EXACT) {
        assert(!descriptor.buffer && !descriptor.range);
        assert(!uniform_buffer_descriptor(&recorder, &bindings[0], NVN_STAGE_VS, 3, input.address, input.size, ALIGN,
                                          1, VK_MUSE_UBO, &descriptor, NULL));
        assert(recording_failed && !descriptor.buffer);
    }
    if (mode == PAYLOAD_REMAP) live_mapping--;
    ubo_merges_release(&plan);
    OK(rc_registry_write_cancel(registry, current));
    OK(rc_registry_write_cancel(registry, overlap));
    if (unknown.id) OK(rc_registry_write_cancel(registry, unknown));
    vk_exec_draw_update_history_end(recorder.fence);
    OK(rc_registry_gpu_capture_end(registry, &U.producer.read));
    rc_registry_destroy(registry);
    registry = NULL;
    vk_uniform_payload_reap();
    assert(payload_parts - initial_parts == 3 && payload_part_drop - initial_drops == 3);
    assert(!active_part.part && !native_holds && !UPD.watched && !budget.retained.amount[RC_BUDGET_RAM] &&
           !budget.retained.amount[RC_BUDGET_COMMIT] && !budget.pending.amount[RC_BUDGET_RAM]);
    cross_scene_fixture = 0;
}

static void cross_scene_cases(void) {
    cross_scene_case(PAYLOAD_EXACT);
    cross_scene_case(PAYLOAD_UNHANDED);
    cross_scene_case(PAYLOAD_REMAP);
    cross_scene_case(PAYLOAD_UNKNOWN);
    puts("cross-scene ordered uniform: full 64 KiB, handed payload and overlapping local writers; unhanded, remap, unknown refused");
}
