// Explicit preparation boundary for CPU-provided sampled textures. All calls require the texture lock.
#pragma once
#include "vk.h"
#include "resource_core_registry.h"

typedef struct VkTextureRegistrationScope VkTextureRegistrationScope;
VkTextureRegistrationScope *vk_texture_registration_scope_begin(void);
VkTextureRegistrationScope *vk_texture_registration_scope_retain(VkTextureRegistrationScope *);
int vk_texture_registration_scope_current(const VkTextureRegistrationScope *);
uint64_t vk_texture_registration_scope_publication(const VkTextureRegistrationScope *);
void vk_texture_registration_scope_end(VkTextureRegistrationScope *);
// An actual recorder can resolve cold descriptors until its CPU part or joined continuation stops.
VkTextureRegistrationScope *vk_texture_registration_record_begin(VkTextureRegistrationScope *carried, int resumed);
void vk_texture_registration_record_end(VkTextureRegistrationScope *, int resumed);
uint64_t vk_texture_registration_record_revision(void);
void vk_backend_registration_record_progress(void);
int vk_texture_release_ordered(const tex_t *, uint64_t queued_publication);
// Selected by the command recorder. A scope preserves metadata; exact previously selected image instances
// have their own recording/publication holds. A null scope selects the current registration.
VkTextureRegistrationScope *vk_exec_texture_registration_scope(void);
VkTextureRegistrationScope *vk_exec_texture_registration_select(VkTextureRegistrationScope *);
uint64_t vk_exec_texture_pool(void);
int vk_texture_pool_slot_locked(uint64_t pool, int logical); // maps a guest descriptor slot to its native adapter slot
int vk_texture_pool_count_locked(uint64_t pool);
int vk_texture_pool_primary(uint64_t pool);
int vk_texture_pool_logical_locked(uint64_t pool, int native_slot);
uint64_t vk_texture_pool_slot_token(uint64_t pool, int logical);
uint64_t vk_texture_registration_scope_token(void); // zero when the selected horizon still matches current publications
uint64_t vk_texture_registration_publication(void); // immutable descriptor publication selected by this scope
int vk_texture_registration_changes_locked(uint64_t pool, uint64_t from, uint64_t to, void (*visit)(int, void *), void *arg);
enum { VK_TEXTURE_REGISTRATION_CURRENT = 0, VK_TEXTURE_REGISTRATION_EXACT = 1,
       VK_TEXTURE_REGISTRATION_ABSENT = -1, VK_TEXTURE_REGISTRATION_GONE = -2 };
int vk_texture_registration_image_locked(int id, const VkTexImage **, view_t *, int *has_view, uint64_t *source_size);
int vk_texture_registration_expected_locked(int id, VkImage *expected); // metadata only, no native use
int vk_texture_registration_view_format_locked(int logical, uint32_t *format);
void vk_texture_registration_report_unavailable(int logical);
const VkTexImage *vk_texture_current_registration_locked(int logical, view_t *, int *has_view);
void vk_texture_registration_note_locked(int native_slot, const VkTexImage *image, const view_t *view, int has_view);

typedef enum {
    VK_TEXTURE_SAMPLE_READY,
    VK_TEXTURE_SAMPLE_PREPARE,
    VK_TEXTURE_SAMPLE_FALLBACK,
    VK_TEXTURE_SAMPLE_DEFERRED,
    VK_TEXTURE_SAMPLE_UNAVAILABLE,
    VK_TEXTURE_SAMPLE_ABSENT,
    VK_TEXTURE_SAMPLE_GONE,
    VK_TEXTURE_SAMPLE_CANONICAL_REFUSED,
    // Current descriptor with a verified hole in its qualified storage mapping.
    VK_TEXTURE_SAMPLE_UNBACKED
} VkTextureSampleStatus;

typedef struct {
    const char *stage;
    RcRegistryPosition position;
    RcRegistryResult result;
    VkImage image;
    uint64_t registration, material, address, bytes, upload_retry;
    uint64_t material_version, material_payload, source_copy_version;
    RcInterval mapped_source;
    VkTexImage material_shape;
    uint8_t recorded, uploaded, tracking_failure, material_provider;
} VkTextureSampleRefusal;

typedef struct {
    uint64_t registration;
    int native_slot;
    uint8_t canonical, refused, unbacked, projected;
} VkTextureReadKey;

// A caller with an exact sampled selection can prepare the derived view without rediscovering its source.
const VkTexImage *vk_texture_flat3d_selected_locked(const VkTexImage *, VkTextureReadKey, uint32_t layer,
                                                    VkCommandBuffer, int required);
int vk_texture_flat3d_selected_current_locked(VkTextureReadKey, VkImage child);
const VkTexImage *vk_texture_current_registration_read_locked(int logical, view_t *, int *, VkTextureReadKey *);

typedef struct VkTextureHistorical3D VkTextureHistorical3D;
typedef struct RcVmaDependency RcVmaDependency;
typedef struct TextureImagePublication TextureImagePublication;
typedef struct VkTextureQueryImage VkTextureQueryImage;
VkTextureSampleStatus vk_texture_history_flat3d_locked(int logical, const VkTexImage **image, VkTextureHistorical3D **plan);
int vk_texture_history_flat3d_installed_locked(int logical, const TextureImagePublication *set_images,
                                              uint32_t descriptor, VkImage expected);
int vk_texture_history_flat3d_current(void *plan);
VkResult vk_texture_history_flat3d_prepare(void *plan, RcVmaDependency *dependency);
int vk_texture_history_flat3d_record_locked(VkTextureHistorical3D *plan, VkCommandBuffer cb);
void vk_texture_history_flat3d_cancel(VkTextureHistorical3D *plan);

typedef struct {
    const VkTexImage *image;
    view_t view;
    int has_view, native_slot, historical;
    uint64_t registration;
    int canonical;
    // Current independently placed GPU material with complete ordered native coverage.
    int projected;
    VkTextureQueryImage *query_image; // Only an attested geometry descriptor can bind this carrier.
} VkTextureSample;

// A historical READY sample carries its finalized registration and an immutable provider. No current native slot.
// Original-read proof for an already admitted ordinary private host upload. Neither function resolves the current slot.
void vk_texture_retain_compat_sample_locked(VkTexturePresentSource *, VkTextureReadKey, VkImage);
int vk_texture_retained_compat_sample_current_locked(const VkTexturePresentSource *, uint64_t registration, VkImage);
void vk_texture_report_retained_compat_sample_locked(const VkTexturePresentSource *, uint64_t registration, VkImage);

int vk_texture_history_sample_current_locked(int logical, uint64_t registration, VkImage image);
VkTexturePresentSource *vk_texture_history_sample_retain_locked(int logical, uint64_t registration, VkImage image);

// Attribute activity to the physical member already selected under the texture lock. No slot refill.
int vk_texture_pool_note_selected_locked(const VkTextureSample *, uint32_t *entry, VkTextureReadKey *key);

typedef struct VkTextureAliasRead VkTextureAliasRead;
enum { VK_TEXTURE_ALIAS_SAMPLE_FORCE = 1, VK_TEXTURE_ALIAS_SAMPLE_LINEAR_RGBA8_VIEW = 2,
       VK_TEXTURE_ALIAS_SAMPLE_ONLY = 4 };
// Only used with the original prepared sample / pool publication and the texture lock held.
const RcInterval *vk_texture_selected_sample_intervals_locked(VkTextureReadKey, VkImage, uint32_t *count);
// Placement of the already qualified native sample. These extents confer no raw read permission.
const RcInterval *vk_texture_native_sample_intervals_locked(RcRegistryPosition read, const VkTextureSample *,
    const VkTexturePresentSource *, VkImage, VkImageMemoryPlacement *, uint32_t *count);
// Terminal inspection under the original ordered input. Shares selection and coverage checks with capture;
// no native storage, registrations, physical references, or command effects are acquired.
int vk_texture_alias_sample_inspect_locked(int logical, RcRegistryPosition read, int force, VkTextureAliasFailure *);
void vk_texture_alias_sample_report_layout_locked(int logical, const VkTextureAliasFailure *);
int vk_texture_alias_sample_current_locked(int logical, VkTextureAliasRead *);
int vk_texture_alias_published_inspect_locked(int logical, const VkTextureAliasRead *, RcRegistryPosition read);
int vk_texture_alias_published_current_locked(int logical, VkTextureAliasRead *, RcRegistryPosition read);
typedef void (*VkAliasPoolMember)(int logical, const VkTextureSample *, void *);
void vk_texture_alias_pool_members_locked(uint64_t pool, VkAliasPoolMember, void *);
VkTextureAliasRead *vk_texture_publication_alias_locked(const TextureImagePublication *, uint32_t descriptor, VkImage);
int vk_texture_publication_retain_alias_locked(TextureImagePublication *, uint32_t descriptor, VkImage, VkTextureAliasRead *);

typedef struct VkTextureSampleGeometry {
    const VkTextureRegistrationScope *scope;
    RcRegistryPosition read;
    RcRegistryScope history_scope;
    uint64_t history_capture;
    uint64_t pool, publication, registration, material, layout_revision;
    VkImage image;
    VmaAllocation allocation;
    RcInterval mapping;
    VkImageSubresourceRange planes;
    VkExtent2D extent;
    VkFormat format;
    int native_slot;
    uint8_t registration_only;
    uint8_t association_only;
} VkTextureSampleGeometry;

typedef enum {
    VK_TEXTURE_SAMPLE_GEOMETRY_READY,
    VK_TEXTURE_SAMPLE_GEOMETRY_ORDER,
    VK_TEXTURE_SAMPLE_GEOMETRY_REGISTRATION,
    VK_TEXTURE_SAMPLE_GEOMETRY_PROVIDER,
    VK_TEXTURE_SAMPLE_GEOMETRY_SHAPE,
    VK_TEXTURE_SAMPLE_GEOMETRY_VIEW,
    VK_TEXTURE_SAMPLE_GEOMETRY_MAPPING,
    VK_TEXTURE_SAMPLE_GEOMETRY_LAYOUT_UNOBSERVED,
    VK_TEXTURE_SAMPLE_GEOMETRY_LAYOUT_UNORDERED,
    VK_TEXTURE_SAMPLE_GEOMETRY_GONE,
    VK_TEXTURE_SAMPLE_GEOMETRY_UNBACKED,
    VK_TEXTURE_SAMPLE_GEOMETRY_ALTERNATE
} VkTextureSampleGeometryResult;

// Optional refusal report for an original query-only sample. The selector never logs on its own.
VkTextureSampleGeometryResult vk_texture_sample_geometry_report_locked(int logical, VkTextureSample *out,
    VkTextureSampleGeometry *geometry);
// A cold registration can prove the exact shape for an isolated carrier. The sample has no native image.
VkTextureSampleGeometryResult vk_texture_sample_registered_geometry_locked(int logical, VkTextureSample *out,
    VkTextureSampleGeometry *geometry);
// The carrier may use an already selected native association, or a shape agreed by all possible providers.
VkTextureSampleGeometryResult vk_texture_sample_associated_geometry_locked(int logical, const VkTextureSample *selected,
    VkTextureSample *out, VkTextureSampleGeometry *geometry);
// Exact historical selection carries a geometry witness from its retained scope. Only a carrier can bind it.
VkTextureSampleGeometryResult vk_texture_sample_historical_geometry_locked(int logical, VkTextureSample *out,
    VkTextureSampleGeometry *geometry);
const char *vk_texture_sample_geometry_status_name(VkTextureSampleGeometryResult);

// For an otherwise proven direct 2D size-query binding only. This proves geometry, native layout and
// placement, and supplies no texel authority. The caller retains the exact image and registration scope.
int vk_texture_sample_geometry_locked(int logical, VkTextureSample *out, VkTextureSampleGeometry *geometry);
int vk_texture_sample_geometry_current_locked(int logical, const VkTextureSample *captured,
    const VkTexImage *exact, const VkTextureSampleGeometry *geometry, const RcInterval **ranges, uint32_t *count);

struct VkExec;
// Only a binding attested to query 2D size at LOD zero can use the returned view. It has no guest mapping
// or texel authority. The caller retains the original exact selection and records outside rendering.
VkResult vk_texture_query_image_create_locked(int logical, const VkTextureSample *original,
    const VkTextureSampleGeometry *geometry, VkTextureQueryImage **out);
int vk_texture_query_image_current_locked(const VkTextureQueryImage *);
int vk_texture_query_image_record_locked(struct VkExec *, VkTextureQueryImage *);
VkImageView vk_texture_query_image_view(const struct VkExec *, const VkTextureQueryImage *, VkImage *native_image);
void vk_texture_query_image_release(VkTextureQueryImage **);

// Query never allocates, uploads, revives an entry or changes its lifetime. A ready result remains valid only
// while the caller holds the texture lock; recording use is explicit and must precede releasing that lock.
VkTextureSampleStatus vk_texture_sample_query_locked(int id, VkTextureSample *out);
// Preparation may allocate/upload/revive. Cold eligible slots enter this path; unrelated resource classes use
// FALLBACK. Deferred work preserves the replay's existing ordered retry signal and is never published as ready.
VkTextureSampleStatus vk_texture_sample_prepare_locked(int id, VkTextureSample *out);
void vk_texture_sample_use_locked(const VkTextureSample *sample);
// FORCE_COMPAT selects the explicit legacy substitution even if the registered image has an exact provider.
// EXACT_DETAIL preserves full mip detail for typed positional reads. A qualified native projection already has it.
enum { VK_TEXTURE_READ_REQUIRED = 1, VK_TEXTURE_READ_FORCE_COMPAT = 2,
       VK_TEXTURE_READ_LINEAR_RGBA8_VIEW = 4, VK_TEXTURE_READ_EXACT_DETAIL = 8 };
VkTextureSampleStatus vk_texture_resolve_sample_locked(int logical, unsigned flags, VkTextureSample *out);
// On a definitive refusal, export facts from the same selector. No second provider lookup.
VkTextureSampleStatus vk_texture_resolve_sample_report_locked(int logical, unsigned flags, VkTextureSample *out,
                                                              VkTextureSampleRefusal *refused);
// Peek through the selected registration scope and query its exact current provider under the same lock.
// Historical material still enters ordered acquisition. No provider may be created or adopted here.
VkTextureSampleStatus vk_texture_select_sample_locked(int logical, unsigned flags, VkTextureSample *out);
// A READY alias owns the composed texel plan for this read. Preparation, recording and cleanup belong
// to the caller. SAMPLE_ONLY queries only the composed provider for pool preflight.
VkTextureSampleStatus vk_texture_ordered_sample_select_locked(int logical, RcRegistryPosition read, unsigned flags,
    unsigned alias_options, VkTextureSample *sample, VkTextureAliasRead **alias, VkTexImage *shape,
    VkTextureAliasFailure *failure);
// Record a dependency only for a required current sample whose observation hit an earlier positioned GPA writer.
int vk_texture_sample_wait_gpa_locked(struct VkExec *ex, int logical);
VkFormat vk_texture_color_depth_format(VkFormat color);
int vk_texture_alias_depth_compare_locked(VkTextureAliasRead *, VkTexImage *);
int vk_texture_alias_read_uses_image(const VkTextureAliasRead *, VkImage);
int vk_texture_alias_comparison_source(const VkTextureAliasRead *, VkImage);
int vk_texture_alias_comparison_range_refused(const VkTextureAliasRead *);
int vk_texture_publication_comparison_source_locked(const TextureImagePublication *, uint32_t kinds, VkImage);
VkTextureSampleStatus vk_texture_comparison_alias_capture_locked(int logical, RcRegistryPosition,
    VkTextureSample *, VkTextureAliasRead **, VkTexImage *, VkTextureAliasFailure *);
// A current registered ordinary material can borrow the same exact staged reconstruction as texture-to-buffer.
// Capture and install hold the texture lock. The recorder waits outside it at the original input position.
VkTextureRestore *vk_texture_sample_restore_capture_locked(int logical);
typedef struct {
    VkTexImage image;
    view_t view;
    int has_view;
} VkTextureRestoreShape;
// Inspect an exact ordinary nonresident recipe and its view. A requested receipt retains this same proof.
int vk_texture_sample_restore_plan_locked(int logical, VkTextureRestoreShape *, VkTextureRestore **);
int vk_texture_sample_restore_current(void *);
VkTextureSampleStatus vk_texture_sample_restore_install_locked(VkTextureRestore *, VkTextureSample *);
// Validate a retained descriptor at the current logical read position. This is a source/instance check, not a
// lifetime acquisition; the descriptor owner must already retain the expected image through its submitted use.
// Source-dependent compatibility resources retain the caller's registration, epoch and authority guards.
int vk_texture_binding_current_locked(int id, VkImage expected);
// A successful receipt permits unchanged outer guards to reuse before the returned read-phase boundary of the
// next writer or sealed declaration prefix. A zero next_write.scene means no current-source boundary. Retain the originating read position and mapping/protection
// serials when aggregating; the writer's own read phase must recheck its writable aliases.
// Positive results distinguish an exact physical binding from one whose readiness depends on guest-source
// authority. Both retain registration/lifetime guards; SOURCE needs ordered registry-write invalidation. LEGACY
// additionally requires the legacy authority epoch even within its single-operation interval.
enum { VK_TEXTURE_BINDING_DIRECT = 1, VK_TEXTURE_BINDING_SOURCE = 2, VK_TEXTURE_BINDING_LEGACY = 4 };
int vk_texture_binding_receipt_locked(int id, VkImage expected, RcRegistryPosition *next_write);
// Refresh the selected ordinary source's proof using the exact registration already chosen for the descriptor.
// The caller owns the selected native image and holds the texture lock.
int vk_texture_resolved_receipt_locked(VkTextureReadKey key, VkImage expected, RcRegistryPosition *next_write);
int vk_texture_sample_unbacked_current_locked(VkTextureReadKey);
// Bindless aggregate only: requires source/schedule notifications, topology epoch races, an open-position gate,
// and native descriptor ownership. The returned actual producer edge does not authorize a source capture.
int vk_texture_binding_schedule_locked(int id, VkImage expected, RcRegistryPosition *next_read);
int vk_texture_resolved_schedule_locked(VkTextureReadKey key, VkImage expected, RcRegistryPosition *next_read);
int vk_texture_projected_sample_receipt_locked(VkTextureReadKey, VkImage, VkTextureSample *);
// The caller protects the selected native set and its indexed descriptor. Only the exact ordinary material
// held by that descriptor can renew its scoped source permission after logical residency is detached.
int vk_texture_published_sample_schedule_locked(const TextureImagePublication *set_images, uint32_t descriptor,
                                               VkTextureReadKey key, VkImage expected, RcRegistryPosition *next_read);
int vk_texture_published_alias_schedule_locked(const TextureImagePublication *, uint32_t descriptor, int logical,
    VkTextureReadKey receiver, VkImage expected, const VkTexImage **source, view_t *, int *has_view,
    RcRegistryPosition *next_read);
// Live GPU-native producer. Its actual writer must precede the read, and its published pool view must still match.
int vk_texture_pool_current_view_locked(const TextureImagePublication *set, uint32_t descriptor,
    int logical, VkImage expected, VkTexImage *image, view_t *view, int *has_view);
int vk_texture_pool_gpu_sample_locked(int logical, VkImage expected, VkTextureSample *);
int vk_texture_pool_gpu_hold_locked(TextureImagePublication *, uint32_t descriptor, const VkTextureSample *);
int vk_texture_pool_gpu_layout_current(const TextureImagePublication *, const VkImage *feedback, uint32_t count);
// Indexed physical ownership only. The caller must still prove the selected source at its read position.
int vk_texture_publication_contains_image_locked(const TextureImagePublication *, uint32_t descriptor, VkImage expected);
int vk_texture_publication_contains_native_locked(const TextureImagePublication *, uint32_t descriptor, VkImage expected);
typedef struct {
    MemMappingReadRange *ranges;
    uint32_t count, capacity;
    int incomplete;
} VkTexturePoolMappingProof;
// The exact selected texture receipts optionally report their host mapping requirements during
// a finite recheck. All observations and final validation occur under the texture lock.
void vk_texture_pool_mapping_proof_locked(VkTexturePoolMappingProof *);
// Copy a ring cell's exact native lease from its still-owned publication. 0 means no matching native cell, -1 means its lease could not be transferred.
int vk_texture_publication_copy_native_locked(TextureImagePublication *destination, const TextureImagePublication *source,
                                               uint32_t descriptor, VkImage expected);
