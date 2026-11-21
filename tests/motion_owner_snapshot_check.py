"""Check production motion-owner publication and accounting with the real host ledger."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runtime"
SOURCE = Path(sys.argv[2]) if len(sys.argv) > 2 else RUNTIME / "motion_owner.c"

PRELUDE = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "motion_owner.h"
#include "resource_core_budget.h"

typedef int PlatMutex;
static PlatMutex owner_lock;
static struct {
    unsigned count;
    uint64_t producer_frame;
    int valid;
    MotionOwnerSnapshot cached, *published;
} O;

static RcAdmission budget;
static RcAdmissionRecord records[8];
static unsigned allocations, destructions, reservations, cancellations;
static int fail_malloc, advance_during_malloc, clear_during_malloc;
static uint64_t source_buffer;

static void plat_mutex_lock(PlatMutex *lock) { assert(!*lock); *lock = 1; }
static void plat_mutex_unlock(PlatMutex *lock) { assert(*lock); *lock = 0; }
int motion_owner_enabled(void) { return 1; }
static int rt_knob(const char *key, const char *value) { (void)key; (void)value; return 0; }
static void rt_log(const char *fmt, ...) { (void)fmt; }
_Noreturn void rt_fatal(const char *fmt, ...) { puts(fmt); abort(); }

static void refresh(uint64_t frame, int force) {
    assert(owner_lock && frame == O.producer_frame && force);
    O.valid = 1;
    O.count = source_buffer != 0;
}
static void capture_model_buffers(MotionOwnerSnapshot *out) {
    if (!O.count) return;
    out->count = 1;
    out->bindings[0] = (MotionOwnerBinding){
        .address = 0x1000, .size = 256, .observed = O.producer_frame, .buffer = source_buffer};
}

RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer, int required,
                                              const RcAdmissionVector *cost, RcBudgetTicket *ticket) {
    assert(!owner_lock && !required && producer == RC_BUDGET_SNAPSHOT && !ticket->token.ledger);
    assert(cost->amount[RC_BUDGET_RAM] >= sizeof(MotionOwnerSnapshot));
    assert(cost->amount[RC_BUDGET_RAM] == cost->amount[RC_BUDGET_COMMIT]);
    RcAdmissionStatus status = rc_admission_reserve(&budget, producer, cost, &ticket->token);
    if (status == RC_ADMISSION_OK) {
        ticket->producer = producer;
        reservations++;
    }
    return status;
}
RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *ticket, const RcAdmissionVector *cost) {
    assert(!owner_lock);
    return rc_admission_materialize(&budget, ticket->token, ticket->producer, cost, &ticket->token);
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) {
    assert(!owner_lock);
    cancellations++;
    return rc_admission_cancel(&budget, ticket->token, ticket->producer);
}
RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *ticket) {
    assert(!owner_lock);
    return rc_admission_retire(&budget, ticket->token, ticket->producer, &ticket->token);
}
RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *ticket) {
    assert(!owner_lock);
    destructions++;
    return rc_admission_destroyed(&budget, ticket->token, ticket->producer);
}
static void *snapshot_malloc(size_t size) {
    assert(!owner_lock && budget.reserved.amount[RC_BUDGET_RAM] == size);
    assert(budget.reserved.amount[RC_BUDGET_COMMIT] == size);
    if (fail_malloc) return NULL;
    if (advance_during_malloc) {
        source_buffer = clear_during_malloc ? 0 : source_buffer + 1;
        motion_owner_present();
    }
    allocations++;
    return malloc(size);
}
static void snapshot_free(void *ptr) {
    assert(!owner_lock && budget.pending.amount[RC_BUDGET_RAM] >= sizeof(MotionOwnerSnapshot));
    free(ptr);
}
'''

CASES = r'''
static void expect_charges(uint64_t bytes) {
    for (unsigned d = 0; d < 2; d++) {
        assert(budget.retained.amount[d] == bytes);
        assert(!budget.reserved.amount[d] && !budget.pending.amount[d]);
    }
}

int main(void) {
    const RcAdmissionDomain domains[] = {
        [RC_BUDGET_RAM] = {RC_ADMISSION_RAM, 0, UINT64_MAX},
        [RC_BUDGET_COMMIT] = {RC_ADMISSION_COMMIT, 0, UINT64_MAX}};
    assert(rc_admission_init(&budget, 1, domains, 2, records, 8) == RC_ADMISSION_OK);
    assert(!motion_owner_acquire() && !reservations);
    source_buffer = 1;
    motion_owner_present();
    MotionOwnerSnapshot *queued = motion_owner_acquire();
    assert(queued && queued->bindings[0].buffer == 1 && queued->bindings[0].observed == 1);
    uint64_t charge = budget.retained.amount[RC_BUDGET_RAM];
    assert(charge >= sizeof *queued && allocations == 1 && reservations == 1);
    expect_charges(charge);
    MotionOwnerSnapshot *recording = motion_owner_acquire();
    assert(recording == queued && reservations == 1);

    source_buffer = 2;
    motion_owner_present();
    expect_charges(charge); // The queued and recording references still own the original placement.
    for (unsigned d = 0; d < 2; d++) {
        budget.domain[d].limit = 2 * charge - 1;
        assert(!motion_owner_acquire()); // No stale generation is substituted on refusal.
        expect_charges(charge);
        budget.domain[d].limit = 2 * charge;
    }
    assert(allocations == 1 && reservations == 1);

    fail_malloc = 1;
    assert(!motion_owner_acquire() && cancellations == 1);
    expect_charges(charge);
    fail_malloc = 0;
    advance_during_malloc = 1;
    MotionOwnerSnapshot *current = motion_owner_acquire();
    advance_during_malloc = 0;
    assert(current && current != queued && current->bindings[0].buffer == 3);
    assert(current->bindings[0].observed == 3 && queued->bindings[0].buffer == 1);
    expect_charges(2 * charge);

    motion_owner_release(queued);
    assert(!destructions);
    expect_charges(2 * charge);
    motion_owner_release(recording);
    assert(destructions == 1);
    expect_charges(charge);

    motion_owner_forget(3);
    assert(!O.published && current->bindings[0].buffer == 3);
    expect_charges(charge);
    motion_owner_release(current);
    expect_charges(0);
    current = motion_owner_acquire();
    assert(current);
    motion_owner_release(current); // The cache owns the only remaining reference.
    motion_owner_present();
    expect_charges(0);
    current = motion_owner_acquire();
    assert(current);
    motion_owner_shutdown();
    expect_charges(charge);
    motion_owner_release(current);
    expect_charges(0);

    source_buffer = 4;
    motion_owner_present();
    advance_during_malloc = clear_during_malloc = 1;
    assert(!motion_owner_acquire()); // A newly empty generation drops the unused admitted allocation.
    expect_charges(0);
    assert(allocations == 5 && reservations == 6 && cancellations == 1 && destructions == 5);
    puts("motion owner snapshot admission, generation retention, cancellation and reclamation passed");
    return 0;
}
'''

source = SOURCE.read_text()
order = source[source.index("static int binding_address_order("):source.index("static void capture_model_buffers(")]
start = source.index("void motion_owner_release(")
allocation = source.rfind("typedef struct {\n    MotionOwnerSnapshot snapshot;", 0, start)
production = source[allocation if allocation >= 0 else start:]
with tempfile.TemporaryDirectory(prefix="motion-owner-snapshot-") as temporary:
    fixture = Path(temporary) / "motion_owner_snapshot_check.c"
    fixture.write_text(PRELUDE + order + "\n#define malloc snapshot_malloc\n#define free snapshot_free\n" +
                       production + "\n#undef malloc\n#undef free\n" + CASES)
    exe = fixture.with_suffix(".exe")
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    subprocess.run([str(cc), "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-fsanitize-trap=undefined", "-I", str(RUNTIME), str(fixture), "-o", str(exe)],
                   env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)
