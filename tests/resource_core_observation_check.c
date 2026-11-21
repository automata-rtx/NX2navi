#include "../runtime/resource_core_observation.h"
#include <assert.h>
#include <stdio.h>
int main(void) {
    RcSourceDomain d;
    RcSourceRange r = {1, 4096, 8192};
    RcObservation old, now;
    assert(rc_source_init(&d, 7, r));
    assert(rc_source_observe(&d, &old));
    assert(rc_source_changed(&d, (RcSourceRange){1, 0, 4096}, 1));
    assert(rc_observation_current(&d, &old));
    assert(rc_source_new_scope(&d)); // late CPU event: an original submit token is not enough
    assert(!rc_observation_current(&d, &old));
    assert(!rc_observation_revalidate(&d, &old, 0, &now));
    assert(rc_observation_revalidate(&d, &old, 1, &now));
    old = now;
    assert(rc_source_changed(&d, (RcSourceRange){1, 4095, 2}, 1));
    assert(!rc_observation_revalidate(&d, &old, 1, &now));
    assert(rc_source_observe(&d, &old));
    RcAuthorityTransfer t = {0}, stale;
    assert(rc_authority_begin(&d, &t));
    stale = t;
    assert(!rc_source_observe(&d, &now)); // pending buffer writes exclude CPU certification too
    assert(rc_authority_end(&d, &t, RC_AUTH_GPU, 0));
    assert(!rc_observation_current(&d, &old));
    assert(!rc_authority_end(&d, &stale, RC_AUTH_GPU, 1));
    assert(rc_authority_begin(&d, &t));
    assert(rc_authority_end(&d, &t, RC_AUTH_GPU, 1));
    assert(!rc_source_observe(&d, &now));
    assert(rc_authority_begin(&d, &t));
    assert(rc_source_changed(&d, (RcSourceRange){1, 4100, 4}, 1));
    assert(rc_authority_end(&d, &t, RC_AUTH_CPU, 0));
    assert(d.authority == RC_AUTH_COMPATIBILITY); // Cancel does not undo an independent CPU mutation.
    assert(rc_authority_begin(&d, &t));
    assert(rc_authority_end(&d, &t, RC_AUTH_CPU, 1));
    assert(rc_source_observe(&d, &old));
    assert(rc_authority_begin(&d, &t));
    assert(rc_source_changed(&d, (RcSourceRange){1, 4100, 4}, 1));
    assert(!rc_authority_end(&d, &t, RC_AUTH_GPU, 1));
    assert(rc_authority_end(&d, &t, RC_AUTH_COMPATIBILITY, 1));
    assert(!rc_source_observe(&d, &now));
    assert(rc_authority_begin(&d, &t));
    assert(rc_authority_end(&d, &t, RC_AUTH_CPU, 1));
    assert(rc_source_observe(&d, &old));
    rc_source_finalize(&d);
    assert(!rc_observation_current(&d, &old));
    assert(rc_source_init(&d, 8, r)); // identical address, new registration
    assert(!rc_observation_current(&d, &old));
    d.revision = UINT64_MAX;
    assert(!rc_source_changed(&d, (RcSourceRange){0}, 0));
    assert(!rc_source_observe(&d, &now));
    assert(!rc_range_valid((RcSourceRange){1, UINT64_MAX, 1}));
    RcPreparedInput p = {1, 2, 3, 4, RC_PROVIDER_CONDITIONAL_GUEST};
    assert(!rc_input_exact(&p));
    p.kind = RC_PROVIDER_CAPTURE;
    assert(rc_input_exact(&p));
    puts("resource core observation: scopes, aliases, handoff, finalization and provider checks passed");
    return 0;
}
