// Ordered command reservations, adopted by actual recording instead of duplicated by late worker preparation.
#pragma once
#include "resource_core_registry.h"

typedef struct ResourceCommandEffects ResourceCommandEffects;
typedef struct { ResourceCommandEffects *owner; } ResourceImageEffectRole;
// Associate new image claims with one optional output of this command. Close the declaration
// scope immediately after adding that role. A recorder can discard this role's unissued refs
// before any other output records, even when both roles share an image or a normalized claim.
void resource_effects_image_role_begin(ResourceCommandEffects *, ResourceImageEffectRole *);
void resource_effects_image_role_end(ResourceCommandEffects *, ResourceImageEffectRole *);
void resource_effects_image_role_finish(ResourceImageEffectRole *, int issued);
ResourceCommandEffects *resource_effects_create(RcRegistryPosition position);
void resource_effects_add(ResourceCommandEffects *, RcSourceRange effect, RcSourceRange descriptor,
                          RcRegistryWriterKind kind);
void resource_effects_empty(ResourceCommandEffects *, RcSourceRange descriptor);
// Stable backing/mapping intervals captured at declaration. Empty effect coverage declares complete no-write
// descriptor coverage. An unsuccessful declaration confers no permission to issue the operation.
// The exact native buffer descriptors enter the ordinary effect ledger and expose its own writer tickets.
RcRegistryResult resource_effects_add_buffer_phase(ResourceCommandEffects *, const RcInterval *, uint32_t,
                                                  RcRegistryBufferPhaseExtent *);
RcRegistryResult resource_effects_add_intervals(ResourceCommandEffects *, const RcInterval *effects, uint32_t effect_count,
                                                const RcInterval *descriptors, uint32_t descriptor_count,
                                                RcRegistryWriterKind);
RcRegistryResult resource_effects_add_pinned_image(ResourceCommandEffects *, uint64_t image, const RcInterval *, uint32_t count);
// Unsupported semantic writers retain unknown authority only if a GPU command is actually emitted.
void resource_effects_opaque(ResourceCommandEffects *);
void resource_effects_opaque_commit(void);
// Protect canonical outputs when a bounded declaration is unavailable. Preparation alone does not make the
// guard foreign. Executed unknown output joins the registry's recoverable unresolved guard at completion.
RcRegistryResult resource_effects_opaque_intervals(ResourceCommandEffects *, RcRegistryWrite *out_optional);
// After actual recording, report exhaustive coverage for the active command's canonical guard. Copies ranges.
RcRegistryResult resource_effects_resolve_opaque_intervals(const RcSourceRange *written, uint32_t written_count,
                                                          const RcSourceRange *uncertain, uint32_t uncertain_count);
int resource_effects_read_allowed(const ResourceCommandEffects *, uint64_t addr, uint64_t size);
int resource_effects_read_allowed_range(const ResourceCommandEffects *, RcSourceRange);
// Compare an exact attachment interval with the command's separately declared shader writes.
// The attachment role's own writes are excluded; opaque or raw-only shader effects cannot establish disjointness.
RcRegistryResult resource_effects_shader_write_peer(const ResourceCommandEffects *, const ResourceImageEffectRole *,
                                                    RcInterval, RcInterval *overlap);
ResourceCommandEffects *resource_effects_enter(ResourceCommandEffects *);
void resource_effects_leave(ResourceCommandEffects *previous);
// A successful guest draw/dispatch/output command was emitted in the current scope. Preparation and barriers
// alone must not call this; output coverage missing at the recording boundary remains conservatively unknown.
void resource_effects_emitted(void);
int resource_effects_before_emission(const ResourceCommandEffects *);
int resource_effects_unrecorded(const ResourceCommandEffects *);
int resource_effects_buffer_prepare(uint64_t addr, uint64_t size, int cpu_visible);
int resource_effects_buffer_prepare_intervals(const RcInterval *, uint32_t count, int cpu_visible);
void resource_effects_buffer_prepare_native(const RcInterval *, uint32_t count, uint64_t address,
                                            RcRegistryBufferLocation);
// Tag an actual retained descriptor for may-write provenance only. This must not publish exact write coverage.
void resource_effects_buffer_access_native(const RcInterval *, uint32_t count, uint64_t address,
                                           RcRegistryBufferLocation);
// Take an exact bound-provider receipt on success. Only actual command commit can publish its result.
typedef int (*ResourceBufferPreserveCheck)(void *, RcRegistryBufferSpan, RcRegistryPosition, int *initialized);
int resource_effects_buffer_preserve_native(const RcInterval *, uint32_t count, uint64_t address,
    RcRegistryBufferLocation, void *receipt, ResourceBufferPreserveCheck, void (*release)(void *));
void resource_effects_buffers_commit(void);
// An issued immutable update can attach its bytes to its exact native CPU-visible normalized writes.
void resource_effects_buffers_commit_payload(RcRegistryBufferPayload *);
// Transfer the raw and all exact mapped declarations to the CPU result callback as one ownership change.
// Refusal leaves the declarations and caller-owned output unchanged.
int resource_effects_buffer_result_take(uint64_t address, uint64_t size, const RcInterval *, uint32_t count,
                                        RcRegistryWrite *raw, RcRegistryWrite *mapped);
int resource_effects_image_prepare(uint64_t image, uint64_t addr, uint64_t size);
int resource_effects_image_prepare_intervals(uint64_t image, const RcInterval *, uint32_t count);
// Returns 1 for the exact declared image and backing, 0 if that image has no guest output, -1 on a mismatch.
int resource_effects_image_prepare_pinned(uint64_t image, const RcInterval *, uint32_t count);
void resource_effects_image_written(uint64_t image, uint64_t addr);
void resource_effects_image_written_interval(uint64_t image, RcInterval);
void resource_effects_image_owned(uint64_t image, uint64_t addr, RcRegistryWrite hold);
void resource_effects_image_owned_interval(uint64_t image, RcInterval, RcRegistryWrite hold);
int resource_effects_image_emitted_intervals(uint64_t image, const RcInterval *, const RcRegistryWrite *,
                                              uint32_t count, RcRegistryPosition output, int first_write);
int resource_effects_image_preserved_intervals(uint64_t image, const RcInterval *, const RcRegistryWrite *,
                                               uint32_t count, RcRegistryPosition output, int exact_projection_layout);
// An exact independent image projection has no written bytes in this otherwise prepared mapping.
void resource_effects_image_unchanged_extent(uint64_t image, RcRegistryImageWriteExtent, RcRegistryPosition output);
void resource_effects_image_record_extents(uint64_t image, const RcRegistryImageWriteExtent *, uint32_t count);
// Actual operation recording has finished (after any retry), not prediction/preparation. Complete recorded image
// claims may publish their exact representation; missing/prepared-only claims remain unresolved until disposal.
// emitted is persistent evidence that this operation recorded GPU work; missing image coverage then retains the
// declared output range through completion instead of treating absent callbacks as a skipped command.
// The selected exact physical output's phase never began, even if other phases emitted commands.
void resource_effects_cancel_unissued_image(ResourceCommandEffects *, uint64_t image);
void resource_effects_recorded(ResourceCommandEffects *, int emitted);

// NX2NAVI_IMAGE_PRODUCER_TRACE=1. Scalar diagnostics scoped to an exact recorded output.
typedef enum {
    RESOURCE_IMAGE_TRACE_COMMAND, RESOURCE_IMAGE_TRACE_DRAW, RESOURCE_IMAGE_TRACE_DRAW_RETAINED,
    RESOURCE_IMAGE_TRACE_DISPATCH, RESOURCE_IMAGE_TRACE_DISPATCH_COMPAT, RESOURCE_IMAGE_TRACE_DISPATCH_RETAINED,
} ResourceImageTraceRoute;
typedef enum {
    RESOURCE_IMAGE_TRACE_NO_PROJECTION_NOTE, RESOURCE_IMAGE_TRACE_LEGACY_WRITE,
    RESOURCE_IMAGE_TRACE_PLANE_UNSUPPORTED, RESOURCE_IMAGE_TRACE_PLANE_PLACEMENT,
    RESOURCE_IMAGE_TRACE_PLANE_DEMAND, RESOURCE_IMAGE_TRACE_PLANE_PUBLICATION,
    RESOURCE_IMAGE_TRACE_PLANE_PUBLISHED,
} ResourceImageTraceProjection;
typedef enum {
    RESOURCE_IMAGE_TRACE_BASELINE_UNOBSERVED, RESOURCE_IMAGE_TRACE_FULL_OVERWRITE,
    RESOURCE_IMAGE_TRACE_CURRENT_PLANE, RESOURCE_IMAGE_TRACE_ALIAS_SEED,
} ResourceImageTraceBaseline;
typedef enum {
    RESOURCE_IMAGE_TRACE_PLACEMENT_UNOBSERVED, RESOURCE_IMAGE_TRACE_PRIVATE,
    RESOURCE_IMAGE_TRACE_POOL_BOUND, RESOURCE_IMAGE_TRACE_OTHER_PLACEMENT,
} ResourceImageTracePlacement;
typedef enum {
    RESOURCE_IMAGE_TRACE_ADMISSION_UNOBSERVED, RESOURCE_IMAGE_TRACE_STORAGE_LAYOUT,
    RESOURCE_IMAGE_TRACE_STORAGE_BINDING, RESOURCE_IMAGE_TRACE_STORAGE_PEER,
    RESOURCE_IMAGE_TRACE_STORAGE_TARGET, RESOURCE_IMAGE_TRACE_STORAGE_PLANE, RESOURCE_IMAGE_TRACE_STORAGE_SELECTED,
} ResourceImageTraceAdmission;
typedef struct {
    uint32_t format, aspect, mip, mips, layer, layers;
    int selected_view, independent, pinned, replacement;
    ResourceImageTraceProjection projection;
    ResourceImageTraceBaseline baseline;
    ResourceImageTracePlacement placement;
    ResourceImageTraceAdmission admission;
} ResourceImageTraceDetails;
int resource_effects_image_trace_enabled(void);
void resource_effects_recorded_image_trace(ResourceCommandEffects *, int emitted, uint32_t opcode,
                                           uint64_t program, ResourceImageTraceRoute);
// A null effects argument denotes the current recorder scope.
void resource_effects_image_trace_details(ResourceCommandEffects *, uint64_t image, ResourceImageTraceDetails);
// Optional receipt for a retained original storage binding after its command emitted.
typedef struct {
    uint64_t program, shader, buffer, offset, size;
    uint32_t opcode, stage, set, binding;
    int borrowed;
} ResourceBufferIssueTrace;
void resource_effects_trace_buffer_issued(ResourceCommandEffects *, const RcInterval *, uint32_t,
                                          ResourceBufferIssueTrace);
void resource_effects_image_trace_blocker(RcRegistryPosition read, const RcRegistryNativeImageFailure *);
// The optional recorder-part witness outlives this command's effects. Settlement precedes the owner callback.
void resource_effects_retain_part_disposition(void *part, int (*known_unhanded)(const void *),
    void *owner, void (*release)(void *part, void *owner, int known_unhanded));
void resource_effects_release(ResourceCommandEffects *);

void resource_effects_report(uint64_t frame);
