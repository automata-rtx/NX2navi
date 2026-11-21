// Per-representation source ownership uses the production registry, without native memory/provider stubs.
#include <assert.h>
#include <stdio.h>
#define RC_REGISTRY_SINGLE_THREADED 1
#include "../runtime/resource_core_registry.c"
#define OK(x) assert((x) == RC_REGISTRY_OK)
static RcRegistryPosition pos(uint64_t ordinal) { return (RcRegistryPosition){1, ordinal}; }
static void commit_at(RcRegistry *r, RcRegistrySourceAccess *a, RcRegistryPosition p) {
    RcRegistrySourceProof proof; int changed = 0;
    OK(rc_registry_source_access_observe(r, a, 1, p, &proof));
    OK(rc_registry_source_access_commit(r, a, &proof, 0, &changed));
    assert(changed);
}
int main(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){32, 16, 4}); assert(r);
    RcSourceRange range = {1, 4096, 64}, unrelated = {1, 8192, 64};
    RcRegistrySourceAccess a = {0}, b = {0};
    OK(rc_registry_source_access_acquire(r, 7, range, &a));
    OK(rc_registry_source_access_acquire(r, 7, range, &b));
    assert(a.owner.handle.id == b.owner.handle.id && a.owner.id != b.owner.id);
    RcRegistrySource owner = a.owner;
    OK(rc_registry_source_access_acquire(r, 7, range, &a));
    assert(a.owner.id == owner.id && rc_registry_stats(r).source_leases == 2);
    OK(rc_registry_declare_through(r, pos(4)));
    commit_at(r, &a, pos(2));
    RcRegistrySourceProof original = rc_registry_source_access_content(&a);
    RcRegistryReadObservation read;
    assert(rc_registry_source_access_read(r, &b, 1, pos(2), &read) != RC_REGISTRY_OK);
    OK(rc_registry_source_access_read(r, &a, 1, pos(4), &read));
    assert(read.position.ordinal == 4 && a._read.position.ordinal == 2);
    // No consumer query advances the canonical byte origin or silently extends its finite permission.
    OK(rc_registry_declare_through(r, pos(10)));
    OK(rc_registry_source_access_read(r, &a, 1, pos(8), &read));
    assert(a._read.position.ordinal == 2 && a._read.sealed_through.ordinal == 10);
    assert(!memcmp(&original, &a._content, sizeof original));
    // A newly declared producer appears in the installed-binding schedule even inside an older seal.
    RcRegistryWrite writer;
    OK(rc_registry_write_begin_at(r, range, RC_REGISTRY_BUFFER, pos(13), &writer));
    RcRegistryReadSchedule schedule;
    OK(rc_registry_source_access_schedule(r, &a, 1, pos(8), &schedule));
    assert(schedule.next_read.ordinal == 12 && schedule.epoch == rc_registry_authority_epoch(r));
    OK(rc_registry_declare_through(r, pos(20)));
    assert(rc_registry_source_access_read(r, &a, 1, pos(14), &read) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r, writer));
    OK(rc_registry_source_access_read(r, &a, 1, pos(14), &read));
    // Maintenance of unchanged bytes cannot replace an earlier consumer's anchored permission.
    RcRegistrySourceProof maintenance;
    OK(rc_registry_source_access_observe(r, &a, 1, (RcRegistryPosition){0}, &maintenance));
    int changed = 1;
    OK(rc_registry_source_access_commit(r, &a, &maintenance, 1, &changed));
    assert(!changed && !memcmp(&original, &a._content, sizeof original));
    assert(rc_registry_source_access_current(r, &a, 1, (RcRegistryPosition){0}));
    OK(rc_registry_source_access_read(r, &a, 1, pos(2), &read));
    // A new scope queries real authority but can retain the exact byte provenance.
    OK(rc_registry_source_access_read(r, &a, 2, pos(4), &read));
    assert(read.scope == 2 && a._read.position.ordinal == 2);
    OK(rc_registry_cpu_changed(r, unrelated));
    OK(rc_registry_source_access_read(r, &a, 2, pos(4), &read));
    // An actual changed source defeats both permissions and stale-copy commit, even if memory stamps match.
    OK(rc_registry_cpu_changed(r, range));
    assert(rc_registry_source_access_read(r, &a, 2, pos(4), &read) == RC_REGISTRY_STALE);
    assert(rc_registry_source_access_commit(r, &a, &original, 1, &changed) == RC_REGISTRY_STALE);
    assert(!rc_registry_source_access_current(r, &a, 1, (RcRegistryPosition){0}));
    commit_at(r, &b, pos(4));
    assert(!rc_registry_source_proof_same(&a._content, &b._content));
    assert(rc_registry_source_access_commit(r, &a, &b._content, 1, &changed) == RC_REGISTRY_OK && changed);
    // A copied owner is not another lease; releasing one invalidates every stale copy atomically.
    RcRegistrySourceAccess stale = a;
    OK(rc_registry_source_access_release(r, &a));
    assert(!a.owner.id && !a._content.ordered.handle.id);
    assert(rc_registry_source_access_read(r, &stale, 1, pos(4), &read) == RC_REGISTRY_STALE);
    assert(rc_registry_source_access_commit(r, &stale, &b._content, 1, &changed) == RC_REGISTRY_STALE);
    OK(rc_registry_source_access_read(r, &b, 1, pos(4), &read));
    OK(rc_registry_finalize_object(r, 7, range));
    assert(rc_registry_source_access_read(r, &b, 1, pos(4), &read) == RC_REGISTRY_STALE);
    assert(rc_registry_source_access_acquire(r, 7, range, &b) == RC_REGISTRY_STALE);
    OK(rc_registry_source_access_release(r, &b));
    OK(rc_registry_source_access_acquire(r, 8, range, &b));
    assert(b.owner.handle.id != owner.handle.id);
    assert(rc_registry_source_access_commit(r, &b, &original, 1, &changed) == RC_REGISTRY_STALE);
    commit_at(r, &b, pos(4));
    OK(rc_registry_range_gone(r, range));
    assert(rc_registry_source_access_read(r, &b, 1, pos(4), &read) == RC_REGISTRY_STALE);
    OK(rc_registry_source_access_release(r, &b));
    OK(rc_registry_source_access_acquire(r, 9, range, &b));
    commit_at(r, &b, pos(4));
    poison(r);
    assert(rc_registry_source_access_read(r, &b, 1, pos(4), &read) == RC_REGISTRY_BLOCKED);
    assert(rc_registry_source_access_commit(r, &b, &b._content, 1, &changed) != RC_REGISTRY_OK);
    rc_registry_destroy(r);
    puts("source access: one owned decision state, anchored permissions, schedule renewal, shared leases and stale identities passed");
}
