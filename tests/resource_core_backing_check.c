#include <assert.h>
#include <stdio.h>

#define RC_BACKING_SINGLE_THREADED 1
#include "../runtime/resource_core_backing.c"

#define OK(call) assert((call) == RC_BACKING_OK)

static RcBackingCapture *capture(RcBackingStore *store, RcBackingMapId map, uint64_t address, uint64_t size,
                                 RcBackingCaptureMode mode) {
    RcBackingCapture *captured;
    OK(rc_backing_capture(store, map, address, size, mode, &captured));
    return captured;
}

static const RcBackingSegment *segments(const RcBackingCapture *captured, uint32_t expected) {
    uint32_t count;
    const RcBackingSegment *result = rc_backing_capture_segments(captured, &count);
    assert(result && count == expected);
    return result;
}

static void check_segment(const RcBackingSegment *actual, uint64_t address, uint64_t mapping, uint64_t backing,
                          uint64_t offset, uint64_t size, uintptr_t provider) {
    assert(actual->address == address && actual->mapping == mapping && actual->provider == provider);
    assert(rc_range_equal(actual->range, (RcSourceRange){backing, offset, size}));
}

static void retire(RcBackingStore *store, RcBackingId id, uintptr_t expected) {
    uintptr_t provider;
    OK(rc_backing_finalize(store, id));
    OK(rc_backing_reclaim(store, id, &provider));
    assert(provider == expected);
}

static void split_mapping_and_finalize(void) {
    RcBackingStore *store = rc_backing_store_create((RcBackingConfig){4, 4, 8, 8, 16});
    assert(store);
    RcBackingId physical, replacement;
    RcBackingMapId virtual, alias;
    OK(rc_backing_create(store, 128, 111, &physical));
    OK(rc_backing_create(store, 40, 222, &replacement));
    OK(rc_backing_map_create(store, 1000, 100, &virtual));
    OK(rc_backing_map_create(store, 4000, 20, &alias));
    OK(rc_backing_map_replace(store, virtual, 1010, 80, physical, 20));
    OK(rc_backing_map_replace(store, alias, 4000, 20, physical, 50));
    RcBackingCapture *old = capture(store, virtual, 1010, 80, RC_BACKING_REQUIRE_MAPPED);
    RcBackingCapture *unaffected = capture(store, virtual, 1015, 10, RC_BACKING_REQUIRE_MAPPED);
    RcBackingCapture *aliased = capture(store, alias, 4005, 10, RC_BACKING_REQUIRE_MAPPED);
    const RcBackingSegment *old_vector = segments(old, 1);
    uint64_t original = old_vector[0].mapping;
    assert(original && original != physical.id);
    assert(rc_range_overlap(old_vector[0].range, segments(aliased, 1)[0].range));
    check_segment(&old_vector[0], 1010, original, physical.id, 20, 80, 111);

    OK(rc_backing_map_replace(store, virtual, 1040, 20, replacement, 5));
    RcBackingCapture *split = capture(store, virtual, 1030, 40, RC_BACKING_REQUIRE_MAPPED);
    const RcBackingSegment *parts = segments(split, 3);
    uint64_t inserted = parts[1].mapping;
    assert(inserted && inserted != original);
    check_segment(&parts[0], 1030, original, physical.id, 40, 10, 111);
    check_segment(&parts[1], 1040, inserted, replacement.id, 5, 20, 222);
    check_segment(&parts[2], 1060, original, physical.id, 70, 10, 111);
    assert(!rc_backing_capture_current(store, old));
    assert(rc_backing_capture_current(store, unaffected));
    assert(rc_backing_capture_current(store, aliased));
    assert(rc_backing_capture_current(store, split));

    // A rewrite of the same native provider and offsets is still a new mapping incarnation.
    OK(rc_backing_map_replace(store, virtual, 1040, 20, replacement, 5));
    assert(!rc_backing_capture_current(store, split));
    assert(rc_backing_capture_current(store, unaffected));
    OK(rc_backing_finalize(store, physical));
    assert(!rc_backing_capture_current(store, aliased));
    assert(!rc_backing_capture_current(store, unaffected));
    assert(rc_backing_map_replace(store, alias, 4000, 20, physical, 50) == RC_BACKING_STALE);
    RcBackingCapture *holes;
    assert(rc_backing_capture(store, virtual, 1000, 100, RC_BACKING_REQUIRE_MAPPED, &holes) == RC_BACKING_ABSENT);
    assert(!holes);
    holes = capture(store, virtual, 1000, 100, RC_BACKING_ALLOW_HOLES);
    parts = segments(holes, 3);
    check_segment(&parts[0], 1000, 0, 0, 0, 40, 0);
    assert(parts[1].mapping != inserted);
    check_segment(&parts[1], 1040, parts[1].mapping, replacement.id, 5, 20, 222);
    check_segment(&parts[2], 1060, 0, 0, 0, 40, 0);
    assert(rc_backing_capture_current(store, holes));
    check_segment(&old_vector[0], 1010, original, physical.id, 20, 80, 111);

    uintptr_t provider;
    assert(rc_backing_reclaim(store, physical, &provider) == RC_BACKING_BUSY);
    OK(rc_backing_map_close(store, virtual));
    OK(rc_backing_map_close(store, alias));
    assert(!rc_backing_capture_current(store, holes));
    assert(rc_backing_reclaim(store, physical, &provider) == RC_BACKING_BUSY);
    rc_backing_capture_release(&aliased);
    rc_backing_capture_release(&unaffected);
    rc_backing_capture_release(&split);
    rc_backing_capture_release(&old);
    assert(!old);
    rc_backing_capture_release(&old);
    OK(rc_backing_reclaim(store, physical, &provider));
    assert(provider == 111);
    assert(rc_backing_reclaim(store, physical, &provider) == RC_BACKING_STALE);
    rc_backing_capture_release(&holes);
    retire(store, replacement, 222);

    RcBackingId reused;
    RcBackingMapId reused_map;
    OK(rc_backing_create(store, 128, 111, &reused));
    assert(reused.slot == physical.slot && reused.id != physical.id);
    OK(rc_backing_map_create(store, 1000, 100, &reused_map));
    assert(reused_map.slot == virtual.slot && reused_map.id != virtual.id);
    assert(rc_backing_map_close(store, virtual) == RC_BACKING_STALE);
    assert(rc_backing_map_replace(store, reused_map, 1010, 80, physical, 20) == RC_BACKING_STALE);
    OK(rc_backing_map_close(store, reused_map));
    retire(store, reused, 111);
    OK(rc_backing_store_destroy(store));
}

static void imports_share_only_live_overlap(void) {
    RcBackingStore *store = rc_backing_store_create((RcBackingConfig){12, 8, 24, 10, 24});
    assert(store);
    RcBackingMapId middle, left_right, other_namespace, new_mapping;
    OK(rc_backing_map_import(store, 7, 2000, 100, 60, &middle));
    RcBackingCapture *old = capture(store, middle, 2000, 60, RC_BACKING_REQUIRE_MAPPED);
    const RcBackingSegment *original = segments(old, 1);
    uint64_t original_id = original[0].range.backing;
    assert(original_id > 1);

    OK(rc_backing_map_import(store, 7, 4000, 80, 100, &left_right));
    RcBackingCapture *overlap = capture(store, left_right, 4000, 100, RC_BACKING_REQUIRE_MAPPED);
    const RcBackingSegment *wide = segments(overlap, 3);
    check_segment(&wide[1], 4020, wide[0].mapping, original_id, 0, 60, 0);
    assert(wide[0].range.backing != original_id && wide[2].range.backing != original_id);
    assert(!rc_range_overlap(wide[0].range, wide[2].range));
    OK(rc_backing_map_import(store, 8, 2000, 100, 60, &other_namespace));
    RcBackingCapture *other = capture(store, other_namespace, 2000, 60, RC_BACKING_REQUIRE_MAPPED);
    assert(!rc_range_overlap(original[0].range, segments(other, 1)[0].range));

    // Cutting the same physical bytes affects both aliases even though their logical addresses differ.
    RcBackingCapture *before = capture(store, middle, 2000, 10, RC_BACKING_REQUIRE_MAPPED);
    OK(rc_backing_import_invalidate(store, 7, 110, 20));
    assert(!rc_backing_capture_current(store, old) && !rc_backing_capture_current(store, overlap));
    assert(rc_backing_capture_current(store, before) && rc_backing_capture_current(store, other));
    RcBackingCapture *cut = capture(store, left_right, 4020, 60, RC_BACKING_ALLOW_HOLES);
    const RcBackingSegment *remains = segments(cut, 3);
    check_segment(&remains[0], 4020, wide[0].mapping, original_id, 0, 10, 0);
    check_segment(&remains[1], 4030, 0, 0, 0, 20, 0);
    check_segment(&remains[2], 4050, wide[0].mapping, original_id, 30, 30, 0);

    OK(rc_backing_map_close(store, middle));
    OK(rc_backing_map_import(store, 7, 3000, 105, 30, &new_mapping));
    RcBackingCapture *reused = capture(store, new_mapping, 3000, 30, RC_BACKING_REQUIRE_MAPPED);
    remains = segments(reused, 3);
    check_segment(&remains[0], 3000, remains[1].mapping, original_id, 5, 5, 0);
    assert(remains[1].range.backing != original_id);
    check_segment(&remains[1], 3005, remains[1].mapping, remains[1].range.backing, 0, 20, 0);
    check_segment(&remains[2], 3025, remains[1].mapping, original_id, 30, 5, 0);
    assert(original[0].range.backing == original_id && original[0].range.size == 60);
    OK(rc_backing_map_close(store, left_right));
    OK(rc_backing_map_close(store, other_namespace));
    OK(rc_backing_map_close(store, new_mapping));
    assert(rc_backing_store_destroy(store) == RC_BACKING_BUSY);
    rc_backing_capture_release(&reused);
    rc_backing_capture_release(&cut);
    rc_backing_capture_release(&before);
    rc_backing_capture_release(&other);
    rc_backing_capture_release(&overlap);

    // Retained history alone never restores a live import identity when the last current alias is gone.
    OK(rc_backing_map_import(store, 7, 2000, 100, 60, &middle));
    RcBackingCapture *reopened = capture(store, middle, 2000, 60, RC_BACKING_REQUIRE_MAPPED);
    assert(segments(reopened, 1)[0].range.backing != original_id);
    assert(!rc_backing_capture_current(store, old));
    rc_backing_capture_release(&old);
    rc_backing_capture_release(&reopened);
    OK(rc_backing_map_close(store, middle));
    OK(rc_backing_store_destroy(store));
}

static void capacity_is_transactional(void) {
    RcBackingStore *store = rc_backing_store_create((RcBackingConfig){2, 3, 2, 2, 2});
    assert(store);
    RcBackingMapId larger, map, failed;
    OK(rc_backing_map_import(store, 1, 0, 0, 100, &larger));
    RcBackingCapture *older = capture(store, larger, 0, 100, RC_BACKING_REQUIRE_MAPPED);
    assert(rc_backing_map_import(store, 1, 99, 99, 2, &failed) == RC_BACKING_CAPACITY);
    assert(!failed.id);
    assert(rc_backing_capture_current(store, older));
    OK(rc_backing_map_import(store, 1, 5, 5, 2, &map));
    OK(rc_backing_import_invalidate(store, 1, 5, 2)); // This removal funds the split of the larger map.
    assert(!rc_backing_capture_current(store, older));
    RcBackingCapture *hole = capture(store, map, 5, 2, RC_BACKING_ALLOW_HOLES);
    check_segment(segments(hole, 1), 5, 0, 0, 0, 2, 0);
    assert(rc_backing_capture_current(store, hole));
    RcBackingCapture *failed_capture;
    assert(rc_backing_capture(store, map, 5, 2, RC_BACKING_ALLOW_HOLES, &failed_capture) == RC_BACKING_CAPACITY);
    assert(!failed_capture);
    assert(rc_backing_import_invalidate(store, 1, 1, 1) == RC_BACKING_CAPACITY);
    rc_backing_capture_release(&hole);
    rc_backing_capture_release(&older);
    OK(rc_backing_map_close(store, map));
    RcBackingCapture *left = capture(store, larger, 0, 5, RC_BACKING_REQUIRE_MAPPED);
    assert(segments(left, 1)[0].range.size == 5);
    rc_backing_capture_release(&left);
    OK(rc_backing_map_close(store, larger));

    RcBackingId backing;
    OK(rc_backing_create(store, 20, 321, &backing));
    OK(rc_backing_map_create(store, UINT64_MAX - 20, 20, &map));
    OK(rc_backing_map_replace(store, map, UINT64_MAX - 20, 20, backing, 0));
    older = capture(store, map, UINT64_MAX - 20, 20, RC_BACKING_REQUIRE_MAPPED);
    assert(rc_backing_map_replace(store, map, UINT64_MAX - 10, 5, backing, 10) == RC_BACKING_CAPACITY);
    assert(rc_backing_map_replace(store, map, UINT64_MAX - 10, 11, backing, 10) == RC_BACKING_INVALID);
    assert(rc_backing_map_replace(store, map, UINT64_MAX - 10, 10, backing, 11) == RC_BACKING_INVALID);
    assert(rc_backing_capture_current(store, older));
    OK(rc_backing_map_replace(store, map, UINT64_MAX - 10, 5, (RcBackingId){0}, 0));
    assert(!rc_backing_capture_current(store, older));
    assert(rc_backing_capture(store, map, UINT64_MAX - 20, 20, RC_BACKING_ALLOW_HOLES, &failed_capture) == RC_BACKING_CAPACITY);
    rc_backing_capture_release(&older);
    OK(rc_backing_map_close(store, map));
    retire(store, backing, 321);
    OK(rc_backing_store_destroy(store));
}

static void storage_publication_and_import_projections(void) {
    RcBackingStore *store = rc_backing_store_create((RcBackingConfig){8, 4, 12, 12, 24});
    assert(store);
    RcBackingMapId host, first, second, replacement;
    OK(rc_backing_storage_map_create(store, 9, 1000, 4000, 200, &host));
    RcBackingCapture *absent;
    assert(rc_backing_capture(store, host, 1020, 60, RC_BACKING_REQUIRE_MAPPED, &absent) == RC_BACKING_ABSENT);
    OK(rc_backing_storage_map_publish(store, host, 1020, 60));
    RcBackingCapture *whole = capture(store, host, 1020, 60, RC_BACKING_REQUIRE_MAPPED);
    uint64_t original = segments(whole, 1)[0].range.backing;
    assert(original > 1);

    OK(rc_backing_map_import(store, 9, 2000, 4030, 40, &first));
    OK(rc_backing_map_import(store, 9, 3000, 4040, 30, &second));
    RcBackingCapture *native_a = capture(store, first, 2010, 20, RC_BACKING_REQUIRE_MAPPED);
    RcBackingCapture *native_b = capture(store, second, 3000, 20, RC_BACKING_REQUIRE_MAPPED);
    RcSourceRange first_coverage = segments(native_a, 1)[0].range;
    RcSourceRange second_coverage = segments(native_b, 1)[0].range;
    assert(rc_range_equal(first_coverage, (RcSourceRange){original, 20, 20}));
    assert(rc_range_equal(first_coverage, second_coverage));
    assert(rc_backing_storage_map_publish(store, first, 2000, 10) == RC_BACKING_INVALID);
    assert(rc_backing_storage_map_publish(store, host, 1010, 20) == RC_BACKING_INVALID);
    assert(rc_backing_capture_current(store, whole));

    OK(rc_backing_map_close(store, first));
    assert(!rc_backing_capture_current(store, native_a) && rc_backing_capture_current(store, native_b));
    RcBackingCapture *unchanged = capture(store, host, 1020, 20, RC_BACKING_REQUIRE_MAPPED);
    OK(rc_backing_import_invalidate(store, 9, 4040, 20));
    assert(rc_backing_capture_current(store, unchanged));
    assert(!rc_backing_capture_current(store, whole) && !rc_backing_capture_current(store, native_b));
    assert(rc_backing_capture(store, host, 1040, 20, RC_BACKING_REQUIRE_MAPPED, &absent) == RC_BACKING_ABSENT);
    OK(rc_backing_storage_map_publish(store, host, 1040, 20));
    RcBackingCapture *now = capture(store, host, 1020, 60, RC_BACKING_REQUIRE_MAPPED);
    const RcBackingSegment *updated = segments(now, 3);
    assert(updated[0].range.backing == original && updated[2].range.backing == original);
    assert(updated[1].range.backing != original && updated[1].range.offset == 0);
    assert(updated[2].range.offset == 40);
    assert(rc_range_equal(first_coverage, segments(native_b, 1)[0].range));
    assert(rc_backing_capture(store, second, 3000, 20, RC_BACKING_REQUIRE_MAPPED, &absent) == RC_BACKING_ABSENT);
    OK(rc_backing_map_import(store, 9, 2000, 4040, 20, &replacement));
    RcBackingCapture *new_import = capture(store, replacement, 2000, 20, RC_BACKING_REQUIRE_MAPPED);
    assert(rc_range_equal(updated[1].range, segments(new_import, 1)[0].range));

    rc_backing_capture_release(&new_import);
    rc_backing_capture_release(&now);
    rc_backing_capture_release(&unchanged);
    rc_backing_capture_release(&native_a);
    rc_backing_capture_release(&native_b);
    rc_backing_capture_release(&whole);
    OK(rc_backing_map_close(store, replacement));
    OK(rc_backing_map_close(store, second));
    OK(rc_backing_map_close(store, host));
    OK(rc_backing_store_destroy(store));
}

static void identity_exhaustion_preserves_current_topology(void) {
    RcBackingConfig config = {2, 2, 2, 2, 2};
    RcBackingStore *a = rc_backing_store_create(config), *b = rc_backing_store_create(config);
    assert(a && b);
    RcBackingId backing, other;
    RcBackingMapId map;
    OK(rc_backing_create(a, 10, 123, &backing));
    OK(rc_backing_create(b, 10, 123, &other));
    OK(rc_backing_map_create(a, 0, 10, &map));
    assert(backing.slot == other.slot && backing.id != other.id);
    assert(rc_backing_map_replace(a, map, 0, 10, other, 0) == RC_BACKING_STALE);

    atomic_store_explicit(&last_identity, UINT64_MAX - 1, memory_order_relaxed);
    RcBackingMapId import;
    assert(rc_backing_map_import(a, 1, 0, 0, 10, &import) == RC_BACKING_EXHAUSTED);
    OK(rc_backing_map_replace(a, map, 0, 10, backing, 0));
    RcBackingCapture *saved = capture(a, map, 0, 10, RC_BACKING_REQUIRE_MAPPED);
    check_segment(segments(saved, 1), 0, UINT64_MAX, backing.id, 0, 10, 123);
    assert(rc_backing_map_replace(a, map, 0, 10, backing, 0) == RC_BACKING_EXHAUSTED);
    assert(rc_backing_capture_current(a, saved));
    assert(!rc_backing_capture_current(b, saved));
    OK(rc_backing_map_replace(a, map, 0, 10, (RcBackingId){0}, 0));
    assert(!rc_backing_capture_current(a, saved));
    rc_backing_capture_release(&saved);
    OK(rc_backing_map_close(a, map));
    retire(a, backing, 123);
    retire(b, other, 123);
    OK(rc_backing_store_destroy(a));
    OK(rc_backing_store_destroy(b));
}

static void value_intervals_reject_reused_backing_slots(void) {
    RcBackingStore *store = rc_backing_store_create((RcBackingConfig){1, 1, 1, 1, 1});
    assert(store);
    RcBackingMapId map;
    OK(rc_backing_storage_map_create(store, 1, 100, 0, 128, &map));
    OK(rc_backing_storage_map_publish(store, map, 100, 128));
    RcInterval old, replacement;
    uint32_t count;
    OK(rc_backing_map_intervals(store, map, 100, 128, &old, 1, &count));
    assert(count == 1 && old.range.backing > 1 && old.mapping);
    assert(rc_backing_map_intervals_current(store, map, &old, 1, 120, 16));
    OK(rc_backing_import_invalidate(store, 1, 0, 128));
    assert(!rc_backing_map_intervals_current(store, map, &old, 1, 120, 16));
    assert(!rc_backing_store_usage(store).backings);
    OK(rc_backing_storage_map_publish(store, map, 100, 128));
    OK(rc_backing_map_intervals(store, map, 100, 128, &replacement, 1, &count));
    assert(count == 1 && replacement.range.backing != old.range.backing && replacement.mapping != old.mapping);
    assert(!rc_backing_map_intervals_current(store, map, &old, 1, 120, 16));
    assert(rc_backing_map_intervals_current(store, map, &replacement, 1, 120, 16));
    OK(rc_backing_map_close(store, map));
    OK(rc_backing_store_destroy(store));
}

static void map_batch_preserves_mapping_pieces(void) {
    RcBackingStore *store = rc_backing_store_create((RcBackingConfig){2, 1, 4, 2, 8});
    assert(store);
    RcBackingId first, second;
    RcBackingMapId map;
    OK(rc_backing_create(store, 128, 5, &first));
    OK(rc_backing_create(store, 128, 6, &second));
    OK(rc_backing_map_create(store, 100, 128, &map));
    OK(rc_backing_map_replace(store, map, 100, 128, first, 0));
    RcBackingCapture *original = NULL;
    OK(rc_backing_capture(store, map, 100, 128, RC_BACKING_REQUIRE_MAPPED, &original));
    const RcBackingSegment prior = rc_backing_capture_segments(original, NULL)[0];

    const RcBackingReplace refused[] = {{116, 16, second, 0}, {164, 16, second, 32}};
    assert(rc_backing_map_replace_many(store, map, refused, 2) == RC_BACKING_CAPACITY);
    assert(rc_backing_capture_current(store, original) && rc_backing_store_usage(store).current_segments == 1);
    const RcBackingReplace stale[] = {{116, 16, second, 0}, {164, 16, {second.id + 1, second.slot}, 0}};
    assert(rc_backing_map_replace_many(store, map, stale, 2) == RC_BACKING_STALE);
    assert(rc_backing_capture_current(store, original));

    const RcBackingReplace ordered[] = {{100, 96, second, 16}, {132, 16, {0}, 0}};
    OK(rc_backing_map_replace_many(store, map, ordered, 2));
    assert(!rc_backing_capture_current(store, original) &&
           rc_backing_capture_range_current(store, original, 196, 32));
    RcInterval actual[4];
    uint32_t count;
    OK(rc_backing_map_intervals(store, map, 100, 128, actual, 4, &count));
    assert(count == 4 && !actual[1].mapping && actual[1].range.size == 16);
    assert(actual[0].mapping != prior.mapping && actual[0].mapping == actual[2].mapping &&
           actual[2].range.offset == 64 && actual[3].mapping == prior.mapping && actual[3].range.offset == 96);
    assert(rc_backing_capture_segments(original, NULL)[0].provider == 5);

    OK(rc_backing_finalize(store, first));
    uintptr_t provider;
    assert(rc_backing_reclaim(store, first, &provider) == RC_BACKING_BUSY);
    rc_backing_capture_release(&original);
    OK(rc_backing_reclaim(store, first, &provider));
    assert(provider == 5);
    OK(rc_backing_map_close(store, map));
    OK(rc_backing_finalize(store, second));
    OK(rc_backing_reclaim(store, second, &provider));
    OK(rc_backing_store_destroy(store));
}

int main(void) {
    split_mapping_and_finalize();
    imports_share_only_live_overlap();
    capacity_is_transactional();
    storage_publication_and_import_projections();
    value_intervals_reject_reused_backing_slots();
    map_batch_preserves_mapping_pieces();
    identity_exhaustion_preserves_current_topology();
    puts("resource core backing: split remaps, retained vectors, host publication, import aliases, capacity and ID exhaustion passed");
    return 0;
}
