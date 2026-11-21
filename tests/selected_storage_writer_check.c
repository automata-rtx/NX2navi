#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gfx.h"
#include "nvn_api.h"
#include "host.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_gpa.h"
#include "resource_core_effects.h"
#include "resource_command_effects.h"
#include "resource_core_vma.h"
// SPIR-V from the QA ssbo_alignment_write.comp. Source SHA-256 5c134b6aaf16e655a434f2ad357d220eda0d047518ceac27f369569781d596bd.
static const uint32_t ssbo_alignment_write_spv[] = {
    0x07230203, 0x00010300, 0x000d000b, 0x00000017, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0005000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x00060010, 0x00000004, 0x00000011,
    0x00000001, 0x00000001, 0x00000001, 0x00040047, 0x00000007, 0x00000006, 0x00000004, 0x00030047,
    0x00000008, 0x00000002, 0x00050048, 0x00000008, 0x00000000, 0x00000023, 0x00000000, 0x00040047,
    0x0000000a, 0x00000021, 0x00000003, 0x00040047, 0x0000000a, 0x00000022, 0x00000000, 0x00040047,
    0x00000016, 0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
    0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x0003001d, 0x00000007, 0x00000006, 0x0003001e,
    0x00000008, 0x00000007, 0x00040020, 0x00000009, 0x0000000c, 0x00000008, 0x0004003b, 0x00000009,
    0x0000000a, 0x0000000c, 0x00040015, 0x0000000b, 0x00000020, 0x00000001, 0x0004002b, 0x0000000b,
    0x0000000c, 0x00000000, 0x0004002b, 0x0000000b, 0x0000000d, 0x00000001, 0x0004002b, 0x00000006,
    0x0000000e, 0x93bc471d, 0x00040020, 0x0000000f, 0x0000000c, 0x00000006, 0x0004002b, 0x0000000b,
    0x00000011, 0x00000007, 0x0004002b, 0x00000006, 0x00000012, 0x4ea710ad, 0x00040017, 0x00000014,
    0x00000006, 0x00000003, 0x0004002b, 0x00000006, 0x00000015, 0x00000001, 0x0006002c, 0x00000014,
    0x00000016, 0x00000015, 0x00000015, 0x00000015, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x00060041, 0x0000000f, 0x00000010, 0x0000000a, 0x0000000c,
    0x0000000d, 0x0003003e, 0x00000010, 0x0000000e, 0x00060041, 0x0000000f, 0x00000013, 0x0000000a,
    0x0000000c, 0x00000011, 0x0003003e, 0x00000013, 0x00000012, 0x000100fd, 0x00010038,
};

// Synthetic module declaring one u32 runtime array and one store at GlobalInvocationId.x.
#define SPV(words, op) (((words) << 16) | (op))
static const uint32_t affine_ssbo_writer_spv[] = {
    0x07230203, 0x00010300, 0, 26, 0,
    SPV(2, 17), 1, SPV(3, 14), 0, 1,
    SPV(6, 15), 5, 15, 0x6e69616d, 0, 13,
    SPV(6, 16), 15, 17, 1, 1, 1,
    SPV(4, 71), 3, 6, 4, SPV(3, 71), 4, 2, SPV(5, 72), 4, 0, 35, 0,
    SPV(4, 71), 6, 34, 0, SPV(4, 71), 6, 33, 3, SPV(4, 71), 13, 11, 28,
    SPV(2, 19), 1, SPV(4, 21), 2, 32, 0, SPV(3, 29), 3, 2,
    SPV(3, 30), 4, 3, SPV(4, 32), 5, 12, 4, SPV(4, 59), 5, 6, 12,
    SPV(4, 32), 7, 12, 2, SPV(4, 43), 2, 8, 0, SPV(4, 43), 2, 9, 1,
    SPV(4, 23), 11, 2, 3, SPV(4, 32), 12, 1, 11, SPV(4, 59), 12, 13, 1,
    SPV(3, 33), 14, 1, SPV(5, 54), 1, 15, 0, 14, SPV(2, 248), 16,
    SPV(4, 61), 11, 17, 13, SPV(5, 81), 2, 18, 17, 0,
    SPV(6, 65), 7, 19, 6, 8, 18, SPV(3, 62), 19, 9, SPV(1, 253), SPV(1, 56),
};
#undef SPV

#define NVN_STAGE_CS 5
#include "selected_types.inc"

struct VkMemSpanCapture { VkMemSpan span; RcInterval mapping; unsigned revision; int cold_writer, published, retained; };
VkCaps vk_caps = {.sbuf_align = 8192, .storage_write_alignment = 4};
int vk_mirror_on;
static struct { uint64_t ubo_max, sbo_max, sbo_align; } LIM = {65536, 65536, 256};
static const uint64_t guest_address = 0x2000, cold_guest_address = 0x6000;
static const VkBuffer second_native = (VkBuffer)(uintptr_t)3;
static VkBuffer native_buffer = (VkBuffer)(uintptr_t)1;
static VkBuffer mirror_buffer = (VkBuffer)(uintptr_t)2;
static unsigned revision = 1, active, stopped, mirror, credit_state, cancel_count, dispose_count;
static unsigned blocked_transition, ordinary_mirror_issued;
static unsigned transition_plans_allowed, blocked_publications, permit_reduced_cold, change_late_mirror, stop_on_refusal;
static unsigned cohort_publications, reduced_publications;
static RcRegistryResult ordinary_publication_result;
static const VkPipeBinding *marker_candidate;
int vk_pipeline_fragment_rmw_candidate(const VkPipeState *state,const VkPipeBinding *binding) {
    assert(state); return binding == marker_candidate;
}
int vk_pipeline_mesh_rmw_bitmap_candidate(const VkPipeState *state,const VkPipeBinding *binding) {
    assert(state && binding); return 0;
}
int resource_effects_image_trace_enabled(void) { return 0; }
const uint8_t *gfx_shader_key(const GfxShader *shader) { (void)shader; abort(); }
void resource_effects_trace_buffer_issued(ResourceCommandEffects *effects,const RcInterval *mapped,
                                           uint32_t count,ResourceBufferIssueTrace trace) {
    (void)effects; (void)mapped; (void)count; (void)trace; abort();
}
static RcRegistryResult mirror_input_result;
static uint64_t reserved, actual;
static unsigned releases, writebacks, drops, donations, scene_begin;
static unsigned cold_publish, cold_issue, cold_checked, cold_adopted, cold_noted, binding_refused;
static unsigned note_refused, preserve_enabled, preserve_refused, preserve_calls, preserve_released;
static VkMemSpanCapture *preserved;
static void (*preserved_release)(void *);
static uint64_t pointer_value;
static unsigned host_table_reads, fallback_writes;
static uint64_t out_offsets[4], out_sizes[4];
static struct { int watched; } UPD;
typedef struct { EffectStorageCapture *storage; uint32_t storage_count; int native_bindings; } EffectInputCapture;
struct EffectDeclaration {
    SelectedStorageWriters *selected_writers;
    EffectInputCapture *capture[8];
    const GfxShader *shader[8];
    uint64_t program_serial;
    ResourceCommandEffects *commands;
    DeviceStoragePointer *pointer_sources;
    uint32_t pointer_count, opcode;
};
typedef struct EffectDeclaration EffectDeclaration;
static EffectDeclaration *effect_current;

enum { NVN_STAGE_MESH = 7 };
static void limits_init(void) { abort(); }
static int nvn_stage_of(VkShaderStageFlags flags) {
    if (flags == GFX_STAGE_MS) return NVN_STAGE_MESH;
    assert(flags == VK_SHADER_STAGE_COMPUTE_BIT || flags == VK_SHADER_STAGE_FRAGMENT_BIT || flags == GFX_STAGE_FS);
    return flags == VK_SHADER_STAGE_COMPUTE_BIT ? NVN_STAGE_CS : NVN_STAGE_FS;
}
static int ssbo_slot(const GfxShaderRes *r, int st, int *ubo) {
    (void)st; *ubo = 0; return r->slot;
}
static int ssbo_bound_recipe(const VkExec *ex, int stage, const GfxShaderRes *r, p_bind_buf *out, int *uniform) {
    int slot = ssbo_slot(r, stage, uniform);
    if (slot < 0 || slot >= 16 || r->cbuf_slot != 0) return 0;
    *out = ex->ssbo[stage][slot];
    return 1;
}
static int ptr_addr_checked(const VkExec *ex, int stage, const GfxShaderRes *r, uint64_t *out) {
    assert(ex && stage == NVN_STAGE_FS && r->ptr);
    host_table_reads++;
    *out = pointer_value;
    return 1;
}
RcRegistryResult rc_registry_read_device_buffer_at(RcRegistry *registry, const RcInterval *ranges,
    uint32_t count, RcRegistryPosition read) {
    assert(registry == rc_registry_runtime() && ranges && count == 1 &&
           !rc_registry_position_compare(read, vk_exec_resource_position()));
    return RC_REGISTRY_OK;
}
static int enable_c0;
static int c0_on(void) { return enable_c0; }
static int c0_ssbo_base(int stage) { (void)stage; abort(); }
int rc_native_rmw_word_address(const RcNativeRmwWord *plan, uint32_t bound, uint64_t address, uint64_t size, uint64_t *word) {
    (void)plan; (void)bound; (void)address; (void)size; (void)word; abort();
}
void vk_mem_native_rmw_word_cancel(VkMemNativeRmwWord **rmw) { assert(rmw && !*rmw); }
int vk_mem_native_rmw_word_current(const VkMemNativeRmwWord *rmw, const VkMemSpan *span, uint64_t address,
    RcRegistryPosition read) { (void)rmw; (void)span; (void)address; (void)read; abort(); }
VkResult vk_mem_native_rmw_word_prepare(VkMemSpanCapture *hold, const VkMemSpan *span, uint64_t size,
    uint64_t word, RcRegistryPosition read, RcVmaDependency *dependency, VkMemNativeRmwWord **rmw) {
    (void)hold; (void)span; (void)size; (void)word; (void)read; (void)dependency; (void)rmw; abort();
}
void vk_mem_native_rmw_word_issued(VkMemNativeRmwWord **rmw, VkCommandBuffer cb) { (void)rmw; (void)cb; abort(); }
struct VkMemNativeRmwBitmap { const VkMemSpanCapture *hold; RcRegistryPosition read; VkPipelineStageFlags2 stage; };
static VkMemNativeRmwBitmap bitmap;
static unsigned bitmap_prepared, bitmap_canceled, bitmap_issued;
static int bitmap_refuse;
VkResult vk_mem_native_rmw_bitmap_prepare_report(VkMemSpanCapture *hold, const VkMemSpan *span, uint64_t size,
    VkPipelineStageFlags2 stage, RcRegistryPosition read, RcVmaDependency *dependency, VkMemNativeRmwBitmap **out,
    VkMemNativeRmwBitmapFailure *failure) {
    assert(!*out && span->buffer == mirror_buffer && size == 64 && span->offset == 4096);
    assert(hold && hold->span.buffer == span->buffer && hold->span.offset == span->offset && dependency);
    if (failure) *failure = (VkMemNativeRmwBitmapFailure){0};
    if (bitmap_refuse) {
        if (failure) *failure = (VkMemNativeRmwBitmapFailure){.gate = "fixture-tail-overlap", .registry = RC_REGISTRY_BLOCKED,
            .range = hold->mapping, .result = VK_ERROR_FEATURE_NOT_PRESENT, .blocker_read = {read.scene, 1},
            .mirror_offset = span->offset, .blocker_offset = span->offset + 16, .blocker_bytes = 4};
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    bitmap = (VkMemNativeRmwBitmap){hold, read, stage};
    *out = &bitmap;
    bitmap_prepared++;
    return VK_SUCCESS;
}
int vk_mem_native_rmw_bitmap_current_report(const VkMemNativeRmwBitmap *owner, const VkMemSpan *span,
    RcRegistryPosition read, VkMemNativeRmwBitmapFailure *failure) {
    const int current = owner == &bitmap && !rc_registry_position_compare(owner->read, read) &&
        vk_mem_span_capture_current(owner->hold, span);
    if (failure) *failure = current ? (VkMemNativeRmwBitmapFailure){0} :
        (VkMemNativeRmwBitmapFailure){.gate = "fixture-provider-changed", .registry = RC_REGISTRY_STALE};
    return current;
}
void vk_mem_native_rmw_bitmap_cancel(VkMemNativeRmwBitmap **owned) {
    if (!*owned) return;
    assert(*owned == &bitmap);
    *owned = NULL;
    bitmap_canceled++;
}
void vk_mem_native_rmw_bitmap_issued(VkMemNativeRmwBitmap **owned, VkCommandBuffer cb) {
    assert(*owned == &bitmap && cb == (VkCommandBuffer)(uintptr_t)4);
    *owned = NULL;
    bitmap_issued++;
}
int resource_core_vma_dependency_ready(const RcVmaDependency *dependency, const RcBudgetReadiness *now) {
    (void)dependency; (void)now; abort();
}
void vk_mem_gpu_wrote_shader(const VkMemSpan *sp, uint64_t size) {
    assert(sp->offset == 4096 && size == 64);
    if (sp->buffer == mirror_buffer) ordinary_mirror_issued++;
    else { assert(sp->buffer == native_buffer); fallback_writes++; }
}
int vk_mem_lookup(uint64_t addr, uint64_t bytes, VkMemSpan *out) {
    assert((addr == guest_address || addr == cold_guest_address) && bytes == 1);
    *out = (VkMemSpan){.buffer=addr == guest_address ? native_buffer : second_native,
        .avail=4096,.offset=4096,.pool=(void *)(uintptr_t)(addr == guest_address ? 6 : 7)};
    return 1;
}
RcBackingResult mem_backing_capture_plan(uint64_t addr, uint64_t bytes, MemBackingCapturePlan *plan) {
    assert((addr == guest_address || addr == cold_guest_address) && bytes == 64);
    *plan = (MemBackingCapturePlan){.address=addr,.size=bytes,.count=1,.bytes=32};
    return RC_BACKING_OK;
}
RcRegistryResult vk_mem_span_import_input_plan(const VkMemSpan *span,const MemBackingCapturePlan *map,
                                               VkMemImportInputPlan *out) {
    const unsigned active_mirror = map->address == guest_address ? mirror : 0;
    assert(span->buffer == (map->address == guest_address ? native_buffer : second_native) &&
           span->offset == 4096 && map->size == 64);
    if (active_mirror && mirror_input_result != RC_REGISTRY_OK) return mirror_input_result;
    *out = (VkMemImportInputPlan){.imported=*span,.selected=*span,.publication=revision,
        .mirror_incarnation=active_mirror,.bytes=sizeof(VkMemSpanCapture) + map->bytes};
    if (active_mirror) out->selected.buffer = mirror_buffer;
    return RC_REGISTRY_OK;
}
int vk_mem_span_import_writer_candidate(uint64_t address, const VkMemSpan *span) {
    assert((address == guest_address && span->buffer == native_buffer) ||
           (address == cold_guest_address && span->buffer == second_native));
    return vk_mirror_on ? (mirror && address == guest_address ? VK_MEM_IMPORT_WRITER_ACTIVE : VK_MEM_IMPORT_WRITER_COLD) : 0;
}
RcRegistryResult vk_mem_span_import_writer_plan(const VkMemSpan *span, const MemBackingCapturePlan *map,
                                                VkMemImportInputPlan *plan) {
    assert(vk_mirror_on);
    if (mirror && map->address == guest_address) {
        if (!transition_plans_allowed) {
            blocked_transition++;
            return RC_REGISTRY_BLOCKED;
        }
        RcRegistryResult result = vk_mem_span_import_input_plan(span, map, plan);
        if (result == RC_REGISTRY_OK) {
            plan->selected = *span;
            plan->mirror_incarnation = 0;
            plan->transition_mirror = mirror;
            plan->cold_writer = 1;
        }
        return result;
    }
    RcRegistryResult result = vk_mem_span_import_input_plan(span, map, plan);
    plan->cold_writer = 1;
    return result;
}
RcRegistryResult vk_mem_span_import_input_capture(const VkMemImportInputPlan *plan,
                                                  const MemBackingCapturePlan *map, VkMemSpanCapture **out) {
    assert(!*out && (credit_state == 1 || credit_state == 2));
    if (plan->publication != revision) return RC_REGISTRY_STALE;
    VkMemSpanCapture *capture = calloc(1, sizeof *capture);
    assert(capture);
    *capture = (VkMemSpanCapture){.span=plan->selected,.revision=revision,.cold_writer=plan->cold_writer,
        .mapping={.range={.backing=5,.offset=128,.size=map->size},.address=map->address,.mapping=6}};
    *out = capture; active++;
    return RC_REGISTRY_OK;
}
uint32_t vk_mem_span_import_input_ranges(const VkMemSpanCapture *capture, const RcInterval **out) {
    *out = &capture->mapping; return 1;
}
int vk_mem_span_capture_current(const VkMemSpanCapture *capture, const VkMemSpan *span) {
    return capture && capture->revision == revision && capture->span.buffer == span->buffer &&
        capture->span.offset == span->offset && (!capture->cold_writer || capture->published);
}
int vk_mem_span_input_cpu_visible(const VkMemSpanCapture *capture) { return capture->span.buffer == native_buffer; }
int vk_mem_span_record_input(VkMemSpanCapture *capture, const VkMemSpan *span) {
    cold_checked++;
    return vk_mem_span_capture_current(capture, span);
}
VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *capture) {
    assert(capture);
    VkMemSpanCapture *held=malloc(sizeof *held);
    assert(held);
    *held=*capture;
    held->retained=1;
    return held;
}
void vk_exec_record_native_input_provider(VkExec *ex, VkMemSpanCapture **capture) {
    assert(ex && *capture && (*capture)->retained);
    free(*capture); // The fake recording owner consumes its fork.
    *capture = NULL;
    cold_adopted++;
}
static EffectInputCapture *effect_scope_capture(VkExec *ex, int stage) {
    assert(ex && effect_current && stage >= 0 && stage < 8);
    return effect_current->capture[stage];
}
static int note_buffer_written(VkExec *ex, uint64_t addr, uint64_t size, const VkMemSpan *span,
                                int visible, const RcInterval *mapping, uint32_t count) {
    assert(ex && addr == guest_address && size == 64 && span->buffer == native_buffer && span->offset == 4096 &&
           visible && count == 1 && mapping->address == addr && mapping->range.size == size);
    cold_noted++;
    return !note_refused;
}
int vk_mem_span_buffer_preserve_current(void *opaque,RcRegistryBufferSpan result,RcRegistryPosition read,int *initialized) {
    VkMemSpanCapture *hold=opaque;
    assert(hold->retained && result.native.incarnation==hold->revision && result.native.offset==hold->span.offset);
    assert(!rc_registry_position_compare(read,vk_exec_resource_position()));
    *initialized=1;
    return 1;
}
int resource_effects_buffer_preserve_native(const RcInterval *mapping,uint32_t count,uint64_t address,
    RcRegistryBufferLocation location,void *hold,int (*current)(void *,RcRegistryBufferSpan,RcRegistryPosition,int *),
    void (*release)(void *)) {
    assert(preserve_enabled && !preserved && count==1 && mapping->address==address && mapping->mapping==6 &&
           mapping->range.backing==5 && mapping->range.offset==128 && mapping->range.size==64 &&
           location.allocation==19 && location.offset==4096);
    int initialized=0;
    assert(current(hold,(RcRegistryBufferSpan){*mapping,location},vk_exec_resource_position(),&initialized)&&initialized);
    ++preserve_calls;
    if (preserve_refused) return 0;
    preserved=hold;
    preserved_release=release;
    return 1;
}
void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t op, const char *reason) {
    assert(ex && op == 1 && reason);
    binding_refused++;
}
void vk_mem_span_capture_release(VkMemSpanCapture **capture) {
    if (*capture) {
        if ((*capture)->retained) preserve_released++;
        else { assert((credit_state == 1 || credit_state == 2) && active); active--; releases++; }
        free(*capture);
        *capture = NULL;
    }
}
RcRegistryResult vk_mem_span_import_writers_publish(VkMemSpanCapture *const *batch, uint32_t count,
                                                    RcRegistryPosition read) {
    assert(count && !rc_registry_position_compare(read, vk_exec_resource_position()));
    assert(credit_state == 2);
    for (unsigned i = 0; i < count; ++i) assert(batch[i]->cold_writer && !batch[i]->published);
    if (permit_reduced_cold) {
        cohort_publications++;
        if (count == 1) {
            assert(batch[0]->mapping.address == cold_guest_address && batch[0]->span.buffer == second_native);
            reduced_publications++;
        } else assert(count == 2 && batch[0]->mapping.address == guest_address &&
                      batch[1]->mapping.address == cold_guest_address);
    }
    if (ordinary_publication_result != RC_REGISTRY_OK && !(permit_reduced_cold && count == 1)) {
        blocked_publications++;
        if (change_late_mirror) mirror++;
        if (stop_on_refusal) stopped=1;
        return ordinary_publication_result;
    }
    for (unsigned i = 0; i < count; ++i) {
        batch[i]->published = 1;
        cold_publish++;
    }
    return RC_REGISTRY_OK;
}
void vk_mem_span_import_writer_issued(VkMemSpanCapture *capture, uint64_t fence) {
    assert(fence == 1 && capture);
    if (capture->cold_writer) {
        assert(capture->published);
        cold_issue++;
    }
}
VkSyncProgressResult vk_sync_progress_prepare_pipeline(int (*prepare)(void *), void *context) {
    assert(prepare(context)); return stopped ? VK_SYNC_PROGRESS_STOPPED_AFTER_CUT : VK_SYNC_PROGRESS_COMPLETE;
}
int vk_sync_stopping(void) { return stopped != 0; }
int vk_exec_recording_failed(void) { return 0; }
volatile int32_t vk_lost;
RcRegistryPosition vk_exec_resource_position(void) { return (RcRegistryPosition){1, 2}; }
static RcAdmission budget_id;
static int refuse;
RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer, int required, const RcAdmissionVector *cost,
                                               RcBudgetTicket *ticket) {
    assert(producer == RC_BUDGET_BUFFER && required && !credit_state);
    assert(cost->amount[RC_BUDGET_RAM] && cost->amount[RC_BUDGET_RAM] == cost->amount[RC_BUDGET_COMMIT]);
    for (unsigned i = RC_BUDGET_GPU_FIRST; i < RC_ADMISSION_MAX_DOMAINS; ++i) assert(!cost->amount[i]);
    if (refuse) return RC_ADMISSION_REFUSED;
    ticket->token.ledger = &budget_id; credit_state = 1; reserved = cost->amount[RC_BUDGET_RAM];
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *ticket, const RcAdmissionVector *cost) {
    assert(ticket->token.ledger == &budget_id && credit_state == 1);
    actual = cost->amount[RC_BUDGET_RAM];
    assert(actual && actual <= reserved && actual == cost->amount[RC_BUDGET_COMMIT]);
    credit_state = 2;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) {
    assert(ticket->token.ledger == &budget_id && credit_state == 1 && !active);
    *ticket = (RcBudgetTicket){0}; credit_state = 0; cancel_count++;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *ticket) {
    assert(ticket->token.ledger == &budget_id && credit_state == 2 && !active);
    credit_state = 3;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *ticket) {
    assert(ticket->token.ledger == &budget_id && credit_state == 3);
    *ticket = (RcBudgetTicket){0}; credit_state = 0; dispose_count++;
    return RC_ADMISSION_OK;
}
void resource_core_budget_service(int required, const RcAdmissionVector *wanted, RcBudgetProgress *out) {
    assert(required && wanted->amount[RC_BUDGET_RAM]); *out = (RcBudgetProgress){0};
}
void resource_core_budget_readiness(int required, RcBudgetReadiness *out) {
    assert(required); *out = (RcBudgetReadiness){0};
}
void rt_fatal(const char *s, ...) { fputs(s, stderr); abort(); }
void rt_log(const char *s, ...) { (void)s; }
void gfx_shader_local_size(const GfxShader *shader, uint32_t local[3]) {
    assert(shader); local[0] = local[1] = local[2] = 1;
}
uint32_t gfx_shader_guest_writes(const GfxShader *shader) { assert(shader); return RC_EFFECT_WRITES_BUFFER; }
int gfx_shader_resources(const GfxShader *shader, int kind, const GfxShaderRes **resources) {
    static const GfxShaderRes writer = {.flags = 1};
    assert(shader && kind == GFX_RES_SSBO);
    *resources = &writer;
    return 1;
}
RcRegistry *rc_registry_runtime(void) { return (RcRegistry *)(uintptr_t)1; }
RcRegistryResult rc_registry_produced_scene_begin(RcRegistry *r, uint64_t scene) {
    assert(r && scene == 1); scene_begin++; return RC_REGISTRY_OK;
}
int vk_mem_span_buffer_location(const VkMemSpanCapture *capture, const VkMemSpan *span, RcRegistryBufferLocation *location) {
    if (!vk_mem_span_capture_current(capture, span)) return 0;
    if (span->buffer == mirror_buffer) {
        *location=(RcRegistryBufferLocation){7,3,span->offset};
        return 1;
    }
    if (!preserve_enabled || span->buffer != native_buffer) return 0;
    assert(capture->retained);
    *location=(RcRegistryBufferLocation){19,capture->revision,span->offset};
    return 1;
}
void vk_mem_gpu_wrote_exact(const VkMemSpan *part, uint64_t size) {
    assert(part->buffer == mirror_buffer && writebacks < 4);
    out_offsets[writebacks] = part->offset; out_sizes[writebacks++] = size;
}
void vk_usnap_drop(uint64_t address, uint64_t size) {
    assert(address >= guest_address && address - guest_address + size <= 64); drops++;
}
void resource_effects_buffer_prepare_native(const RcInterval *intervals, uint32_t count, uint64_t address,
                                            RcRegistryBufferLocation location) {
    assert(count == 1 && address == intervals->address && location.allocation == 7 && location.incarnation == 3);
    assert(location.offset == 4096 + address - guest_address &&
           (intervals->range.size == 4 || intervals->range.size == 64));
    donations++;
}
#include "selected_functions.inc"

static void selected_bitmap(int mesh) {
    const int stage = mesh ? NVN_STAGE_MESH : NVN_STAGE_FS;
    const VkPipelineStageFlags2 stage_bit = mesh ?
        VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    GfxShaderRes resource = {.slot = 0, .flags = 1, .binding = 2, .count = 1};
    VkPipeBinding binding = {.stage = mesh ? GFX_STAGE_MS : GFX_STAGE_FS, .set = !mesh, .binding = 2, .kind = GFX_RES_SSBO,
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .res = &resource, .count = 1};
    VkPipeInfo info = {.bindings = &binding, .n_bindings = 1};
    VkExec ex = {.fence = 1, .cb = (VkCommandBuffer)(uintptr_t)4};
    ex.ssbo[stage][0] = (p_bind_buf){.addr = guest_address, .size = 64};
    SelectedStorageWriters *writers = NULL;
    const VkPipeBinding *blocked = NULL;
    enable_c0 = vk_mirror_on = mirror = 1;
    bitmap_prepared = bitmap_issued = bitmap_canceled = 0;
    assert(selected_storage_writers_prepare(&ex, &info, 0, NULL, NULL, NULL, &writers, NULL, 0, &blocked, NULL, NULL)
           == VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(!writers && !blocked && !bitmap_prepared);
    assert(selected_storage_writers_prepare(&ex, &info, 1, NULL, NULL, NULL, &writers, NULL, 0, &blocked, NULL, NULL)
           == VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!writers && blocked == &binding && !bitmap_prepared);
    info.native_rmw = (RcNativeRmwWord){.atomic_word = 123, .execution_model = mesh ? 5365 : 4,
        .storage_set = !mesh, .storage_binding = 2, .index = 98, .storage_class = 2, .bitmap_sites = mesh ? 5 : 1};
    info.native_rmw_binding = 5;
    vk_caps.storage_write_alignment = 16;
    VkExecStorageFailure reason = {0};
    ex.ssbo[stage][0].size = 60;
    assert(selected_storage_writers_prepare(&ex, &info, 1, NULL, NULL, NULL, &writers, NULL, 0, NULL, NULL, &reason)
           == VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!writers && !bitmap_prepared && !strcmp(reason.gate, "writer-marked-range-alignment"));
    assert(reason.stage == binding.stage && reason.binding == binding.binding && reason.address == guest_address &&
        reason.bytes == 60 && !reason.bitmap.gate);
    ex.ssbo[stage][0].size = 64;
    reason = (VkExecStorageFailure){0};
    bitmap_refuse = 1;
    assert(selected_storage_writers_prepare(&ex, &info, 1, NULL, NULL, NULL, &writers, NULL, 0, NULL, NULL, &reason)
           == VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!writers && !bitmap_prepared && !active && !credit_state && !strcmp(reason.gate, "fixture-tail-overlap"));
    assert(reason.detail == VK_ERROR_FEATURE_NOT_PRESENT && reason.registry == RC_REGISTRY_BLOCKED &&
        reason.address == guest_address && reason.bytes == 64 && reason.bitmap.blocker_bytes == 4 &&
        reason.bitmap.blocker_offset == 4112 && reason.range.range.size == 64);
    bitmap_refuse = 0;
    reason = (VkExecStorageFailure){0};
    assert(selected_storage_writers_prepare(&ex, &info, 1, NULL, NULL, NULL, &writers, NULL, 0, NULL, NULL, &reason)
           == VK_EXEC_GRAPHICS_READY);
    assert(!reason.gate && !reason.bitmap.gate);
    assert(writers->rmw_bitmap == &bitmap && !writers->stores && bitmap_prepared == 1 && active == 1 &&
        bitmap.stage == stage_bit);
    EffectDeclaration effects = {.selected_writers = writers};
    effect_current = &effects;
    VkDescriptorBufferInfo descriptor = {0};
    assert(selected_storage_rmw_descriptor(&ex, &info, stage, &resource, &descriptor));
    assert(descriptor.buffer == mirror_buffer && descriptor.range == 64 && writers->rmw_bound);
    selected_storage_writers_issued(&ex, &effects);
    assert(!writers->rmw_bitmap && bitmap_issued == 1 && !bitmap_canceled);
    selected_storage_writers_release(&writers);
    effect_current = NULL;

    assert(selected_storage_writers_prepare(&ex, &info, 1, NULL, NULL, NULL, &writers, NULL, 0, NULL, NULL, NULL)
           == VK_EXEC_GRAPHICS_READY);
    revision++;
    reason = (VkExecStorageFailure){0};
    assert(selected_storage_writers_prepare(&ex, &info, 1, NULL, NULL, NULL, &writers, NULL, 0, NULL, NULL, &reason)
           == VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!strcmp(reason.gate, "fixture-provider-changed") && reason.registry == RC_REGISTRY_STALE);
    assert(!writers && bitmap_prepared == 2 && bitmap_canceled == 1 && bitmap_issued == 1 && !active && !credit_state);
}

int main(void) {
    GfxShaderRes resource = {.slot=0,.flags=1,.binding=3,.count=1};
    VkPipeBinding binding = {.stage=VK_SHADER_STAGE_COMPUTE_BIT,.kind=GFX_RES_SSBO,
        .type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.res=&resource,.count=1};
    VkPipeInfo info = {.bindings=&binding,.n_bindings=1};
    VkExec ex = {.fence=1};
    ex.ssbo[NVN_STAGE_CS][0] = (p_bind_buf){.addr=guest_address,.size=64};
    ex.ssbo[NVN_STAGE_FS][0] = ex.ssbo[NVN_STAGE_CS][0];
    SelectedStorageWriters *owner = NULL;
    assert(selected_storage_writers_prepare(&ex,&info,0,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(!owner && !credit_state && !active);
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_READY);
    assert(owner && credit_state == 2 && active == 1 && reserved == actual);
    SelectedStorageWriter *selected = selected_storage_writer(owner,NVN_STAGE_CS,&resource);
    assert(selected && selected->storage.descriptor.buffer == native_buffer && selected->storage.span.offset == 4096);
    assert(!owner->store_count && selected->storage.visible && !drops && !writebacks);
    // A changed retained mapping returns INVALIDATED with no writer, releases admission and permits a real replan.
    revision++;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!owner && !credit_state && !active && releases == 1 && dispose_count == 1 && !drops);
    binding.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_READY);
    assert(selected_storage_writer(owner,NVN_STAGE_FS,&resource) && !owner->store_count);
    selected_storage_writers_release(&owner);
    assert(!owner && !credit_state && !active && dispose_count == 2 && !drops);
    refuse = stopped = 1;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_STOPPED);
    refuse = stopped = 0;

    mirror = 1;
    binding.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!owner && !credit_state && !active);
    const GfxShader *shader = (const GfxShader *)(uintptr_t)1;
    info.compute_effect_shader = shader;
    assert(shader_has_storage_writer(shader) && !shader_has_storage_writer(NULL));
    RcEffectProgram *program = rc_effect_compile(ssbo_alignment_write_spv,
        sizeof ssbo_alignment_write_spv / sizeof *ssbo_alignment_write_spv);
    p_dispatch dispatch = {.x=2,.y=1,.z=1};
    assert(rc_effect_has_definite_store_plan(program));
    assert(selected_storage_writers_prepare(&ex,&info,1,shader,program,&dispatch,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!owner && !credit_state && !active && cancel_count == 1);
    dispatch.x = 1;
    assert(selected_storage_writers_prepare(&ex,&info,1,shader,program,&dispatch,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_READY);
    assert(owner->store_count == 2 && owner->stores[0].offset == 4 && owner->stores[1].offset == 28);
    assert(reserved > actual && owner->entries[0].storage.span.buffer == mirror_buffer && !owner->entries[0].storage.visible);
    assert(!drops && !writebacks && !donations);
    selected_storage_writers_release(&owner);
    assert(!owner && !credit_state && !active && !drops && !writebacks && !donations);

    assert(selected_storage_writers_prepare(&ex,&info,1,shader,program,&dispatch,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_READY);
    EffectStorageCapture bound = owner->entries[0].storage;
    bound.borrowed_writer = bound.recorded = 1;
    EffectInputCapture inputs = {.storage=&bound,.storage_count=1};
    EffectDeclaration effect = {.selected_writers=owner,.capture={[NVN_STAGE_CS]=&inputs}};
    selected_storage_writers_issued(&ex,&effect);
    selected_storage_writers_issued(&ex,&effect);
    assert(out_offsets[0] == 4100 && out_offsets[1] == 4124 && out_sizes[0] == 4 && out_sizes[1] == 4);
    assert(writebacks == 2 && drops == 2 && donations == 2 && scene_begin == 1);
    selected_storage_writers_release(&owner);
    rc_effect_free(program);
    assert(!owner && !credit_state && !active && dispose_count == 4);

    mirror = 0;
    vk_mirror_on = 1;
    vk_caps.sbuf_align = 16; // Ordinary native alignment, no high-alignment diagnostic.
    binding.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    assert(selected_storage_writers_prepare(&ex,&info,0,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(!owner && !cold_publish && !credit_state);
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_READY);
    assert(owner && cold_publish == 1 && !cold_issue && owner->entries[0].storage.visible);
    EffectDeclaration opaque = {.selected_writers=owner,.opcode=1};
    effect_current = &opaque;
    VkDescriptorBufferInfo descriptor = {0};
    assert(selected_storage_cold_descriptor(&ex,&info,NVN_STAGE_FS,&resource,&descriptor));
    assert(descriptor.buffer == native_buffer && descriptor.offset == 4096 && descriptor.range == 64 &&
           cold_checked == 1 && cold_adopted == 1 && cold_noted == 1);
    revision++;
    assert(selected_storage_cold_descriptor(&ex,&info,NVN_STAGE_FS,&resource,&descriptor));
    assert(!descriptor.buffer && binding_refused == 1 && cold_noted == 1);
    effect_current = NULL;
    selected_storage_writers_release(&owner);
    assert(!owner && !active && !credit_state && !cold_issue);
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_READY);
    opaque = (EffectDeclaration){.selected_writers=owner,.opcode=1};
    effect_current = &opaque;
    assert(selected_storage_cold_descriptor(&ex,&info,NVN_STAGE_FS,&resource,&descriptor));
    selected_storage_writers_issued(&ex,&opaque);
    effect_current = NULL;
    assert(cold_publish == 2 && cold_issue == 1 && writebacks == 2 && drops == 3);
    selected_storage_writers_release(&owner);
    assert(!owner && !active && !credit_state && dispose_count == 6);

    // Native writable arrays repeat the one reflected binding. A different inventory is not that alias.
    resource.count = 2;
    resource.ptr = 1;
    binding.count = 2;
    LIM.sbo_max = 64;
    pointer_value = guest_address;
    assert(!effect_storage_fixed(&ex,NVN_STAGE_FS,&resource));
    assert(selected_storage_writers_prepare(&ex,&info,0,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(!owner && !active && !credit_state);
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_READY);
    assert(owner && owner->count == 1 && owner->entries[0].storage.descriptor.range == 64);
    pointer_value += 64;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!owner && !active && !credit_state && cold_issue == 1 && cold_publish == 3);
    binding.count = 3;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_READY);
    assert(!owner && !active && !credit_state && cold_publish == 3);
    // A normally aligned output keeps the active mirror when a same-pool write prevents conversion.
    program = rc_effect_compile(affine_ssbo_writer_spv,
        sizeof affine_ssbo_writer_spv / sizeof *affine_ssbo_writer_spv);
    assert(rc_effect_has_definite_store_plan(program));
    binding.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    binding.count = resource.count = 1;
    resource.ptr = 0;
    mirror = 1;
    dispatch.x = 17; // The runtime array fits in the shader, but the actual descriptor would discard a store.
    assert(selected_storage_writers_prepare(&ex,&info,1,shader,program,&dispatch,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!owner && !active && !credit_state && blocked_transition == 1);
    dispatch.x = 16;
    info.compute_effect_shader = NULL;
    assert(selected_storage_writers_prepare(&ex,&info,1,shader,program,&dispatch,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!owner && !active && !credit_state && blocked_transition == 2);
    info.compute_effect_shader = shader;
    assert(selected_storage_writers_prepare(&ex,&info,1,shader,program,&dispatch,&owner,NULL,0,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_READY);
    assert(reserved == actual && blocked_transition == 3 && owner->store_count == 1 &&
           owner->stores[0].offset == 0 && owner->stores[0].interval.range.size == 64);
    assert(owner->entries[0].storage.descriptor.buffer == mirror_buffer && cold_publish == 3);
    bound = owner->entries[0].storage;
    bound.borrowed_writer = bound.recorded = 1;
    inputs = (EffectInputCapture){.storage=&bound,.storage_count=1};
    effect = (EffectDeclaration){.selected_writers=owner,.capture={[NVN_STAGE_CS]=&inputs}};
    selected_storage_writers_issued(&ex,&effect);
    assert(out_offsets[2] == 4096 && out_sizes[2] == 64 && writebacks == 3 && drops == 4 && donations == 3);
    selected_storage_writers_release(&owner);
    rc_effect_free(program);
    assert(!owner && !credit_state && !active);
    assert(selected_storage_writers_current(&ex,owner,&info,vk_exec_resource_position()));
    // A different host word must not select or validate the retained GPU word's target.
    mirror = 0;
    resource.ptr = 1;
    const unsigned host_reads = host_table_reads;
    VkMemSpanCapture table = {.span={.buffer=native_buffer,.offset=16},.revision=revision};
    RcInterval table_range = {.address=0x9000,.range={.backing=7,.offset=16,.size=8}};
    const int stages[] = {NVN_STAGE_FS, NVN_STAGE_CS};
    for (unsigned i = 0; i < sizeof stages / sizeof stages[0]; i++) {
        int stage = stages[i];
        binding.stage = stage == NVN_STAGE_CS ? VK_SHADER_STAGE_COMPUTE_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        ex.ssbo[stage][0] = (p_bind_buf){.addr=table_range.address,.size=16};
        DeviceStoragePointer pointer = {.stage=stage,.resource=&resource,.span=table.span,.capture=&table,
            .ranges=&table_range,.count=1,.read=vk_exec_resource_position(),.binding=ex.ssbo[stage][0],
            .prepared=1,.value=guest_address};
        assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,&pointer,1,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_READY);
        assert(owner&&owner->count==1&&owner->entries[0].device_pointer&&
               owner->entries[0].binding.addr==guest_address&&host_table_reads==host_reads);
        assert(selected_storage_writers_device_current(&ex,owner,&info,pointer.read,&pointer,1));
        assert(!selected_storage_writers_device_current(&ex,owner,&info,pointer.read,NULL,0));
        unsigned previous_issue = cold_issue;
        if (stage == NVN_STAGE_CS) {
            EffectDeclaration committed = {.selected_writers=owner,.opcode=1,.pointer_sources=&pointer,.pointer_count=1};
            effect_current = &committed;
            assert(selected_storage_cold_descriptor(&ex,&info,stage,&resource,&descriptor));
            assert(descriptor.buffer==native_buffer&&descriptor.range==64);
            selected_storage_writers_issued(&ex,&committed);
            effect_current = NULL;
            assert(cold_issue==previous_issue+1);
        }
        ex.ssbo[stage][0].size++;
        assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,&pointer,1,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_INVALIDATED);
        assert(!owner&&!credit_state&&!active&&host_table_reads==host_reads);
        assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,&pointer,1,NULL, NULL, NULL)==VK_EXEC_GRAPHICS_INVALIDATED);
        assert(!owner&&!credit_state&&!active&&host_table_reads==host_reads);
        if (stage == NVN_STAGE_FS) assert(cold_issue==previous_issue);
    }
    // With no imported writer, an opaque stage binds the exact prepared fallback capture.
    ex.ssbo[NVN_STAGE_CS][0] = (p_bind_buf){.addr=table_range.address,.size=16};
    VkMemSpan gpu_span = {.buffer=native_buffer,.offset=4096,.avail=4096};
    VkMemSpanCapture gpu_target = {.span=gpu_span,.revision=revision};
    RcInterval gpu_mapping = {.address=guest_address,.range={.backing=5,.offset=128,.size=64}};
    DeviceStoragePointer gpu_pointer = {.stage=NVN_STAGE_CS,.resource=&resource,.span=table.span,.capture=&table,
        .ranges=&table_range,.count=1,.read=vk_exec_resource_position(),.binding=ex.ssbo[NVN_STAGE_CS][0],
        .prepared=1,.value=guest_address,.output={.resource=&resource,.native=&gpu_target,
            .span=gpu_span,.descriptor={native_buffer,4096,64},.addr=guest_address,.authority_size=64,
            .mapped=&gpu_mapping,.mapped_count=1,.written=1,.visible=1}};
    EffectDeclaration gpu_opaque = {.opcode=1,.pointer_sources=&gpu_pointer,.pointer_count=1};
    effect_current = &gpu_opaque;
    unsigned previous_adoptions = cold_adopted, previous_drops = drops;
    assert(selected_storage_cold_descriptor(&ex,&info,NVN_STAGE_CS,&resource,&descriptor));
    assert(descriptor.buffer==native_buffer&&descriptor.offset==4096&&descriptor.range==64);
    assert(selected_storage_cold_descriptor(&ex,&info,NVN_STAGE_CS,&resource,&descriptor));
    assert(cold_adopted==previous_adoptions+1&&fallback_writes==1&&drops==previous_drops+1);
    assert(gpu_pointer.output.native==&gpu_target&&host_table_reads==host_reads);
    ex.ssbo[NVN_STAGE_CS][0].addr+=8;
    assert(selected_storage_cold_descriptor(&ex,&info,NVN_STAGE_CS,&resource,&descriptor));
    assert(!descriptor.buffer&&cold_adopted==previous_adoptions+1&&fallback_writes==1);
    effect_current = NULL;
    // Preservation needs successful exact declaration, including across repeated descriptor preparation.
    resource.ptr=0;
    ex.ssbo[NVN_STAGE_CS][0]=(p_bind_buf){.addr=guest_address,.size=64};
    binding.stage=VK_SHADER_STAGE_COMPUTE_BIT;
    preserve_enabled=1;
    for (unsigned rejected=1; ; --rejected) {
        preserve_refused=rejected;
        unsigned previous_released=preserve_released;
        unsigned previous_preserve=preserve_calls;
        assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL,NULL)==VK_EXEC_GRAPHICS_READY);
        EffectDeclaration accepted={.selected_writers=owner,.opcode=1};
        effect_current=&accepted;
        note_refused=1;
        assert(selected_storage_cold_descriptor(&ex,&info,NVN_STAGE_CS,&resource,&descriptor));
        assert(!owner->entries[0].storage.writer_noted && preserve_calls==previous_preserve && !preserved);
        note_refused=0;
        assert(selected_storage_cold_descriptor(&ex,&info,NVN_STAGE_CS,&resource,&descriptor));
        assert(owner->entries[0].storage.writer_noted && preserve_calls==previous_preserve+1);
        assert(selected_storage_cold_descriptor(&ex,&info,NVN_STAGE_CS,&resource,&descriptor));
        assert(preserve_calls==previous_preserve+1);
        effect_current=NULL;
        selected_storage_writers_release(&owner);
        if (rejected) {
            assert(!preserved && preserve_released==previous_released+1);
        } else {
            assert(preserved && preserve_released==previous_released);
            preserved_release(preserved);
            preserved=NULL;
            assert(preserve_released==previous_released+1);
            break;
        }
    }
    assert(!active&&!credit_state);
    preserve_enabled=0;
    selected_bitmap(0);
    selected_bitmap(1);
    // A blocked CPU-import transition may keep the exact active carrier selected for a native compute write.
    mirror = 1;
    binding.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
    info.compute_effect_shader = NULL;
    ex.ssbo[NVN_STAGE_CS][0]=(p_bind_buf){.addr=guest_address,.size=64};
    const unsigned before_blocked = blocked_transition, before_cold = cold_publish;
    const unsigned before_exact = writebacks, before_donated = donations, before_issued = cold_issue;
    VkExecStorageFailure mirror_failure = {0};
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL,&mirror_failure)==VK_EXEC_GRAPHICS_READY);
    assert(owner && owner->count==1 && active==1 && !owner->store_count && !owner->rmw && !owner->rmw_bitmap);
    assert(blocked_transition==before_blocked+1 && cold_publish==before_cold);
    const EffectStorageCapture *retained_mirror = &owner->entries[0].storage;
    assert(retained_mirror->selected_binding==&binding && retained_mirror->descriptor.buffer==mirror_buffer &&
           retained_mirror->descriptor.offset==4096 && retained_mirror->descriptor.range==64 && retained_mirror->mapped_count==1 &&
           retained_mirror->mapped[0].mapping==6 && retained_mirror->native && !retained_mirror->visible);
    selected_storage_writers_release(&owner);
    assert(!owner && !active && !ordinary_mirror_issued && cold_issue==before_issued);

    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL,&mirror_failure)==VK_EXEC_GRAPHICS_READY);
    bound=owner->entries[0].storage;
    bound.borrowed_writer=bound.recorded=bound.writer_noted=1;
    inputs=(EffectInputCapture){.storage=&bound,.storage_count=1};
    effect=(EffectDeclaration){.selected_writers=owner,.capture={[NVN_STAGE_CS]=&inputs}};
    ordinary_storage_writers_issued(&effect);
    selected_storage_writers_issued(&ex,&effect);
    ordinary_storage_writers_issued(&effect);
    assert(ordinary_mirror_issued==1 && bound.recorded==2 && writebacks==before_exact &&
           donations==before_donated && cold_publish==before_cold && cold_issue==before_issued);
    revision++;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL,&mirror_failure)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!owner && !active && !credit_state && !strcmp(mirror_failure.gate,"writer-descriptor-current"));

    mirror_input_result=RC_REGISTRY_STALE;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL,&mirror_failure)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!owner && !active && !credit_state && !strcmp(mirror_failure.gate,"writer-provider-stale"));
    mirror_input_result=RC_REGISTRY_UNSUPPORTED;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL, NULL,&mirror_failure)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!owner && !active && !credit_state && !strcmp(mirror_failure.gate,"writer-provider-plan"));
    assert(ordinary_mirror_issued==1 && cold_issue==before_issued);

    // A clean plan can lose its optional raw-import publication without losing either captured mirror binding.
    mirror_input_result = RC_REGISTRY_OK;
    transition_plans_allowed = 1;
    ordinary_publication_result = RC_REGISTRY_BLOCKED;
    GfxShaderRes second_resource = resource;
    second_resource.binding++;
    VkPipeBinding cohort[2] = {binding, binding};
    cohort[1].res = &second_resource;
    cohort[1].binding++;
    info.bindings = cohort;
    info.n_bindings = 2;
    const unsigned before_cohort_published = cold_publish;
    VkExecGraphicsResult recovery = selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL,NULL,&mirror_failure);
    if (recovery != VK_EXEC_GRAPHICS_READY) {
        puts("late optional raw import refused an otherwise intact ordinary mirror cohort");
        return 7;
    }
    assert(owner && owner->count==2 && active==2 && !owner->rmw && !owner->store_count &&
           blocked_publications==1 && cold_publish==before_cohort_published);
    for (unsigned i=0; i<2; ++i) {
        const EffectStorageCapture *m = &owner->entries[i].storage;
        assert(m->selected_binding==&cohort[i] && m->descriptor.buffer==mirror_buffer &&
               m->native && !m->visible && m->native->revision==revision && m->mapped_count==1);
    }
    selected_storage_writers_release(&owner);
    assert(!owner && !active && !credit_state && ordinary_mirror_issued==1 && cold_issue==before_issued);

    ordinary_publication_result=RC_REGISTRY_CAPACITY;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL,NULL,&mirror_failure)==VK_EXEC_GRAPHICS_READY);
    assert(owner && owner->count==2 && blocked_publications==2 && cold_publish==before_cohort_published);
    selected_storage_writers_release(&owner);
    ordinary_publication_result=RC_REGISTRY_STALE;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL,NULL,&mirror_failure)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!owner && !active && !credit_state && blocked_publications==3 &&
        mirror_failure.detail==RC_REGISTRY_STALE && cold_publish==before_cohort_published);

    // Ordinary graphics can use the same mirror, and a known shader marker takes precedence.
    info.bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
    for (unsigned i=0; i<2; i++) cohort[i].stage=VK_SHADER_STAGE_FRAGMENT_BIT;
    ex.ssbo[NVN_STAGE_FS][0]=ex.ssbo[NVN_STAGE_CS][0];
    VkPipeState graphics={0};
    const VkPipeBinding *blocked_import=NULL;
    transition_plans_allowed=0;
    ordinary_publication_result=RC_REGISTRY_OK;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,&blocked_import,&graphics,&mirror_failure)
           ==VK_EXEC_GRAPHICS_READY);
    assert(!blocked_import && owner && owner->entries[0].storage.selected_binding==&cohort[0]);
    selected_storage_writers_release(&owner);
    marker_candidate=&cohort[1];
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,&blocked_import,&graphics,&mirror_failure)
           ==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(blocked_import==marker_candidate && !owner && !active && mirror_failure.detail==-3);
    blocked_import=NULL;
    transition_plans_allowed=1;
    ordinary_publication_result=RC_REGISTRY_BLOCKED;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,&blocked_import,&graphics,&mirror_failure)
           ==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(blocked_import==marker_candidate && !owner && !active && mirror_failure.detail==RC_REGISTRY_BLOCKED);
    assert(cold_publish==before_cohort_published && ordinary_mirror_issued==1 && cold_issue==before_issued);
    // One independently cold import keeps its own authority when the distinct active mirror
    // cannot migrate. Only that cold entry is ever published, in the second atomic batch.
    marker_candidate=NULL;
    info.bind_point=VK_PIPELINE_BIND_POINT_COMPUTE;
    cohort[0].stage=cohort[1].stage=VK_SHADER_STAGE_COMPUTE_BIT;
    second_resource.slot=1;
    ex.ssbo[NVN_STAGE_CS][1]=(p_bind_buf){.addr=cold_guest_address,.size=64};
    permit_reduced_cold=1;
    const unsigned before_mixed=cold_publish, previous_blocked=blocked_publications;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL,NULL,&mirror_failure)==VK_EXEC_GRAPHICS_READY);
    assert(owner && owner->count==2 && cohort_publications==2 && reduced_publications==1 &&
           blocked_publications==previous_blocked+1 && cold_publish==before_mixed+1);
    assert(owner->entries[0].storage.descriptor.buffer==mirror_buffer && !owner->entries[0].cold_writer &&
           owner->entries[1].storage.descriptor.buffer==second_native && owner->entries[1].cold_writer &&
           owner->entries[1].storage.native->published && !owner->entries[0].storage.native->published);
    selected_storage_writers_release(&owner);
    assert(!owner && !active && !credit_state && cold_issue==before_issued);

    const unsigned before_refused=cold_publish;
    change_late_mirror=1;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL,NULL,&mirror_failure)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!owner && !active && !credit_state && cold_publish==before_refused &&
           mirror_failure.detail==RC_REGISTRY_STALE && reduced_publications==1);
    change_late_mirror=0;
    stop_on_refusal=1;
    assert(selected_storage_writers_prepare(&ex,&info,1,NULL,NULL,NULL,&owner,NULL,0,NULL,NULL,&mirror_failure)==VK_EXEC_GRAPHICS_STOPPED);
    assert(!owner && !active && !credit_state && cold_publish==before_refused && reduced_publications==1);
    puts("Selected SSBO storage: direct, late block and capacity, hot/cold reduced batch, stale, stop and marker controls");
}
