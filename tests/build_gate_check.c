// runtime/vk_build_gate.h: the memory gate of the background pipeline builds (runtime/vk_pipeline.c gate_enter). Built
// and run by tests/build_gate_check.py; prints each failed check and exits 1 on any.
#include <stdio.h>

#include "vk_build_gate.h"
#include "resource_budget.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(void) {
    // A machine with 73,154 MB of commit and 4,096 MB kept free; each build is estimated at 64 MB.
    const double limit = 73154, keep = 4096, est = 64;
    BuildGate g = {.free_mb = keep};

    // Plenty of room: every one of 16 builders gets in, and each one's estimate is reserved.
    for (int i = 0; i < 16; i++) CHECK(build_gate_enter(&g, 42000, limit, est), "builder %d refused with 31 GB free", i);
    CHECK(g.reserved_mb == 16 * est, "reserved %.0f MB, expected %.0f", g.reserved_mb, 16 * est);
    for (int i = 0; i < 16; i++) build_gate_leave(&g, est);
    CHECK(g.reserved_mb == 0, "reserved %.0f MB after every build ended", g.reserved_mb);

    // The case the reservations are for: the commit in use does not move while the builders start (their compiles
    // have taken nothing yet), and the room above the kept part holds four estimates. A gate without reservations
    // lets all sixteen in; this one lets four in and refuses the rest.
    double used = limit - keep - 4 * est;
    int in = 0;
    for (int i = 0; i < 16; i++) in += build_gate_enter(&g, used, limit, est);
    CHECK(in == 4, "%d builders started in room for 4", in);
    CHECK(g.reserved_mb == 4 * est, "a refused build reserved something: %.0f MB", g.reserved_mb);
    // One ends: exactly one more fits.
    build_gate_leave(&g, est);
    CHECK(build_gate_enter(&g, used, limit, est), "no build started after one ended");
    CHECK(!build_gate_enter(&g, used, limit, est), "a fifth build started");
    for (int i = 0; i < 4; i++) build_gate_leave(&g, est);

    // The boundary: exactly the kept part plus one estimate free is enough, one MB less is not.
    CHECK(build_gate_admits(&g, limit - keep - est, limit, est), "refused with exactly enough room");
    CHECK(!build_gate_admits(&g, limit - keep - est + 1, limit, est), "admitted one MB short");

    // The commit of the night this was written for: 66,000 of 73,154 MB in use. One ordinary build fits (7,154 MB
    // free against 4,160 needed); a large compute build estimated at 4,096 MB does not, and while one of those runs
    // with 8,254 MB free (room for it, and then 4,158 MB for the rest), ordinary builds are refused too.
    CHECK(build_gate_admits(&g, 66000, limit, est), "an ordinary build refused at 66 of 73 GB");
    CHECK(!build_gate_admits(&g, 66000, limit, 4096), "a large compute build admitted at 66 of 73 GB");
    CHECK(build_gate_enter(&g, 64900, limit, 4096), "a large compute build refused with 8,254 MB free");
    CHECK(!build_gate_admits(&g, 64900, limit, est), "an ordinary build admitted beside a large one with no room for both");
    build_gate_leave(&g, 4096);

    // Above the limit already (the commit in use can pass a limit that shrank): refused, not a negative room taken
    // for a large one.
    CHECK(!build_gate_admits(&g, limit + 500, limit, est), "admitted above the limit");

    // No check: the setting at 0, or a platform that reports no limit (-1). Reservations are still counted, so the
    // numbers a log line prints stay true.
    BuildGate off = {.free_mb = 0};
    CHECK(build_gate_enter(&off, limit, limit, est), "refused with the gate off");
    CHECK(off.reserved_mb == est, "the gate that is off did not count its build");
    CHECK(build_gate_admits(&g, 1e9, -1, est), "refused with an unknown limit");

    // More leaves than enters never leave a negative reservation behind, which would admit builds for nothing.
    build_gate_leave(&g, est);
    CHECK(g.reserved_mb == 0, "reserved %.0f MB after a leave too many", g.reserved_mb);

    // Speculative work shares the texture-pressure floor, including explicit overrides and gate-off behavior.
    CHECK(resource_commit_headroom_mb(32768, -1) == 4096, "small-system automatic floor");
    CHECK(resource_commit_headroom_mb(60000, -1) == 6000, "large-system automatic floor");
    CHECK(resource_commit_headroom_mb(60000, 0) == 0, "explicit texture floor disable");
    CHECK(resource_commit_headroom_mb(60000, 8192) == 8192, "explicit texture floor");
    CHECK(resource_build_headroom_mb(60000, 4096, -1) == 6187.5, "builds must leave the texture recovery margin");
    CHECK(resource_build_headroom_mb(60000, 8192, -1) == 8192, "larger build margin must survive");
    CHECK(resource_build_headroom_mb(60000, 4096, 0) == 4096, "texture disable must retain build floor");
    CHECK(resource_build_headroom_mb(60000, 0, -1) == 0, "explicit build gate disable");

    // Two estimates fit. Completing both before a new sample must not admit an unlimited stream of builds using
    // the old room: the retained results have not yet reached that sample. A measured release restores capacity.
    ResourceBudget b = {.active = {.free_mb = 6000}};
    const double b_limit = 60000, b_used = b_limit - b.active.free_mb - 128;
    CHECK(resource_budget_enter(&b, b_used, b_limit, 64), "first bounded build refused");
    CHECK(resource_budget_enter(&b, b_used, b_limit, 64), "second bounded build refused");
    resource_budget_leave(&b, 64);
    CHECK(b.active.reserved_mb == 64 && b.completed_mb == 64, "completion did not retain the estimate");
    CHECK(!resource_budget_enter(&b, b_used, b_limit, 64), "completed build reused stale headroom");
    resource_budget_leave(&b, 64);
    CHECK(!resource_budget_enter(&b, b_used, b_limit, 64), "serial completions bypassed the gate");
    resource_budget_sample(&b);
    CHECK(b.completed_mb == 0, "fresh sample retained settled debt");
    CHECK(!resource_budget_enter(&b, b_used + 128, b_limit, 64), "measured retained results ignored");
    CHECK(resource_budget_enter(&b, b_used, b_limit, 64), "measured free capacity did not return");

    // Refreshing usage must not cancel reservations of compilers still working. Completion then carries the
    // estimate only to the next sample; even a duplicate completion cannot invent additional pending debt.
    resource_budget_sample(&b);
    CHECK(b.active.reserved_mb == 64, "fresh sample dropped a live compiler reservation");
    resource_budget_leave(&b, 64);
    resource_budget_leave(&b, 64);
    CHECK(b.completed_mb == 64 && b.active.reserved_mb == 0, "duplicate completion invented debt");
    resource_budget_sample(&b);
    CHECK(b.completed_mb == 0, "debt persisted beyond its next sample");
    CHECK(resource_budget_enter(&b, 1e9, -1, 64), "unknown system limit blocked progress");
    resource_budget_leave(&b, 64);
    b.active.free_mb = 0;
    CHECK(resource_budget_enter(&b, 1e9, b_limit, 64), "disabled gate blocked preparation");

    printf("build_gate_check: %s\n", fails ? "FAILED" : "passed");
    return fails ? 1 : 0;
}
