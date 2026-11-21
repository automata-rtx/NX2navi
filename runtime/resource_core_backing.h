#ifndef RESOURCE_CORE_BACKING_H
#define RESOURCE_CORE_BACKING_H

#include "resource_core_observation.h"

typedef struct RcBackingStore RcBackingStore;
typedef struct RcBackingCapture RcBackingCapture;
typedef struct { uint64_t id; uint32_t slot; } RcBackingId;
typedef struct { uint64_t id; uint32_t slot; } RcBackingMapId;
typedef enum {
    RC_BACKING_OK, RC_BACKING_INVALID, RC_BACKING_STALE, RC_BACKING_ABSENT,
    RC_BACKING_CAPACITY, RC_BACKING_EXHAUSTED, RC_BACKING_BUSY
} RcBackingResult;
typedef struct {
    uint32_t backings, maps, current_segments, captures, captured_segments;
} RcBackingConfig;
typedef enum { RC_BACKING_REQUIRE_MAPPED, RC_BACKING_ALLOW_HOLES } RcBackingCaptureMode;
typedef struct {
    uint64_t address, mapping;
    RcSourceRange range;
    uintptr_t provider;
} RcBackingSegment;
typedef struct {
    uint64_t address;
    RcSourceRange range;
    uintptr_t provider;
} RcBackingRun;

// A store serializes its topology and pins internally. Enter it after adapter topology locks, and release those
// locks before waiting for retirement. Operations call no adapters. Mutations either publish in full or fail.
// Destroy requires external exclusivity and all maps, captures and explicit backings closed/reclaimed.
RcBackingStore *rc_backing_store_create(RcBackingConfig);
RcBackingResult rc_backing_store_destroy(RcBackingStore *);
RcBackingConfig rc_backing_store_usage(RcBackingStore *);

// The adapter owns the native allocation. A successful finalize removes every current alias of this backing.
// Reclaim transfers the provider token exactly once, after all retained captures have released it. The adapter
// must also discharge any native use obligations, including submitted GPU work, before destroying the provider.
RcBackingResult rc_backing_create(RcBackingStore *, uint64_t size, uintptr_t provider, RcBackingId *);
RcBackingResult rc_backing_finalize(RcBackingStore *, RcBackingId);
RcBackingResult rc_backing_reclaim(RcBackingStore *, RcBackingId, uintptr_t *provider);
// Query an exact explicit backing, including finalized records that still retain their provider.
// Given a known identity and a zero slot, remember the record's slot for later incarnation checks.
int rc_backing_explicit_retained(RcBackingStore *, RcBackingId *);

// A map is one adapter-selected address window. Different windows can overlap. Publishing any nonempty replace
// gives the inserted interval a new mapping incarnation. Surviving fragments retain their prior mapping identity
// and affine backing offsets. A zero backing handle with offset zero punches a hole.
RcBackingResult rc_backing_map_create(RcBackingStore *, uint64_t address, uint64_t size, RcBackingMapId *);
RcBackingResult rc_backing_map_replace(RcBackingStore *, RcBackingMapId, uint64_t address, uint64_t size,
                                      RcBackingId, uint64_t offset);
typedef struct {
    uint64_t address, size;
    RcBackingId backing;
    uint64_t offset;
} RcBackingReplace;
// Apply one request list in order. Failure preserves every original mapping and retained capture.
RcBackingResult rc_backing_map_replace_many(RcBackingStore *, RcBackingMapId, const RcBackingReplace *, uint32_t count);
RcBackingResult rc_backing_map_close(RcBackingStore *, RcBackingMapId);

// Read up to capacity consecutive current runs, starting at address. Adjacent mappings of the same backing at
// consecutive offsets form one native run even when their mapping incarnations differ. Holes have only a size.
// The provider tokens are borrowed. Hold the adapter's topology lock across lookup and use, or acquire a capture.
RcBackingResult rc_backing_map_runs(RcBackingStore *, RcBackingMapId, uint64_t address,
                                   RcBackingRun *, uint32_t capacity, uint32_t *count);
// Copy current interval identities, including holes, without pinning backing records. With no output and zero
// capacity, count the full vector. With a supplied buffer, copy up to its capacity in address order. The caller
// holds its topology lock across successive calls and protects any actual native access separately.
RcBackingResult rc_backing_map_intervals(RcBackingStore *, RcBackingMapId, uint64_t address, uint64_t size,
                                       RcInterval *, uint32_t capacity, uint32_t *count);
int rc_backing_map_intervals_current(RcBackingStore *, RcBackingMapId, const RcInterval *, uint32_t count,
                                    uint64_t address, uint64_t size);

// Imports share canonical byte identity wherever live maps overlap in the same storage namespace. The caller
// supplies stable storage coordinates and must invalidate replaced storage before publishing the native change
// to consumers. The namespace is an adapter identity for the coordinate system, independent of guest VA and
// Vulkan buffer identity. New imports reuse identity through current coverage. Exact history alone cannot
// restore a retired import extent to current topology.
// Invalidating a part of an import leaves the other parts current. Publish a new import to map replacement bytes.
RcBackingResult rc_backing_map_import(RcBackingStore *, uint64_t storage_namespace, uint64_t address,
                                     uint64_t storage_offset, uint64_t size, RcBackingMapId *);
// An authoritative storage map starts sparse. Publish a newly available span only after the native commit
// succeeds. The whole range must be a hole in this map. Storage invalidation removes a range from every
// projection in the namespace before native storage replacement, without dropping retained captures.
RcBackingResult rc_backing_storage_map_create(RcBackingStore *, uint64_t storage_namespace, uint64_t address,
                                              uint64_t storage_offset, uint64_t size, RcBackingMapId *);
RcBackingResult rc_backing_storage_map_publish(RcBackingStore *, RcBackingMapId, uint64_t address, uint64_t size);
RcBackingResult rc_backing_import_invalidate(RcBackingStore *, uint64_t storage_namespace,
                                            uint64_t storage_offset, uint64_t size);

// A capture is owned by its caller and movable by transferring its pointer. Borrowed segments are immutable
// until release. Synchronize borrowers with the sole releasing owner. Capture pins metadata and explicit
// providers. The adapter must separately preserve any promised bytes and lease readable host pages as needed.
// The allow-holes mode returns a complete vector with range.size set and all identity fields zero for holes.
// Current checks topology only. A hole matches current absence, and unchanged mappings can carry newer bytes.
RcBackingResult rc_backing_capture(RcBackingStore *, RcBackingMapId, uint64_t address, uint64_t size,
                                  RcBackingCaptureMode, RcBackingCapture **);
const RcBackingSegment *rc_backing_capture_segments(const RcBackingCapture *, uint32_t *count);
int rc_backing_capture_current(RcBackingStore *, const RcBackingCapture *);
int rc_backing_capture_range_current(RcBackingStore *, const RcBackingCapture *, uint64_t address, uint64_t size);
void rc_backing_capture_release(RcBackingCapture **);

#endif
