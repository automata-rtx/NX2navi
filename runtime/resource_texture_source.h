// CPU-source proof adaptation. Ordered proofs describe a read position; exact prepared versions own their bytes
// independently. A proof for an earlier draw is never a current-source shortcut for a later draw.
#pragma once
#include "resource_core_registry.h"
typedef RcRegistrySourceProof TextureSourceAuthority;
static inline RcRegistryResult texture_authority_observe(RcRegistry *r, RcRegistryHandle source,
    uint64_t scope, RcRegistryPosition position, TextureSourceAuthority *out) {
    *out = (TextureSourceAuthority){0};
    return rc_registry_position_valid(position) ? rc_registry_observe_at(r, source, scope, position, &out->ordered) :
        rc_registry_observe(r, source, scope, &out->legacy);
}
static inline int texture_authority_current(RcRegistry *r, const TextureSourceAuthority *a,
                                            RcRegistryPosition position) {
    if (rc_registry_position_valid(a->ordered.position))
        return !rc_registry_position_compare(position, a->ordered.position) &&
            rc_registry_read_observation_current(r, &a->ordered);
    return rc_registry_observation_current(r, &a->legacy);
}
// Retaining the cutoff of bytes already uploaded does not grant another read or publish a current resource.
// The next source observation must acquire permission again before it can reuse this byte provenance.
static inline int texture_authority_content_current(RcRegistry *r, const TextureSourceAuthority *a) {
    if (!rc_registry_position_valid(a->ordered.position)) return rc_registry_content_current(r, &a->legacy);
    RcRegistryObservation content = {.handle = a->ordered.handle, .range = a->ordered.range,
        .content_revision = a->ordered.content_revision, .guard = a->ordered.guard};
    return rc_registry_content_current(r, &content);
}
static inline int texture_authority_same(const TextureSourceAuthority *a, const TextureSourceAuthority *b) {
    return rc_registry_source_proof_same(a, b);
}
