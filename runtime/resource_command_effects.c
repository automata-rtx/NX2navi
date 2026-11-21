#include "resource_command_effects.h"
#include "host.h"
#include "vk.h"
#include <stdlib.h>
#include "platform/platform.h"

typedef struct EffectBufferPreservation {
    struct EffectBufferPreservation *next;
    void *receipt;
    ResourceBufferPreserveCheck check;
    void (*release)(void *);
} EffectBufferPreservation;
typedef struct EffectActual {
    struct EffectActual *next;
    RcInterval interval;
    uint64_t image;
    RcRegistryWrite hold;
    int prepared, recorded, visible, owned;
    RcRegistryBufferLocation native_buffer;
    int native_buffer_conflict;
    RcRegistryBufferLocation preserved_location;
    EffectBufferPreservation *preservation;
    int preservation_conflict;
    RcRegistryBufferLocation native_access;
    RcInterval native_access_interval;
    int native_access_conflict;
} EffectActual;
typedef struct SharedImageClaim {
    struct SharedImageClaim *next;
    RcInterval interval;
    uint64_t scene;
    RcRegistryWrite ticket;
    uint32_t refs, count, capacity;
    RcRegistryImageEffectSpan *spans;
    int executed, broad;
} SharedImageClaim;
static SharedImageClaim *shared_images;
static PlatMutex images_lock=PLAT_MUTEX_INIT;
static uint64_t image_requests,image_unique,mapped_image_requests,mapped_image_unique;
static uint64_t cpu_results_transferred;

static SharedImageClaim *shared_image_find(SharedImageClaim *p,RcInterval interval,uint64_t scene) {
    for(;p;p=p->next)
        if(p->scene==scene && p->interval.address==interval.address && p->interval.mapping==interval.mapping &&
           rc_range_equal(p->interval.range,interval.range)) return p;
    return NULL;
}

static SharedImageClaim *shared_image_take(RcSourceRange range,RcRegistryPosition position) {
    plat_mutex_lock(&images_lock);image_requests++;
    RcInterval interval={.address=range.offset,.range=range};
    SharedImageClaim *p=shared_image_find(shared_images,interval,position.scene);
    if(p) {
        if(p->ticket.id)rc_registry_image_producer_add(rc_registry_runtime(),p->ticket,position);
        p->refs++;plat_mutex_unlock(&images_lock);return p;
    }
    p=calloc(1,sizeof *p);
    if(!p)rt_fatal("resource effects: shared image allocation failed");
    p->interval=interval;p->scene=position.scene;p->refs=1;image_unique++;
    rc_registry_write_begin_at(rc_registry_runtime(),range,RC_REGISTRY_IMAGE,position,&p->ticket);
    if(p->ticket.id)rc_registry_image_producer_add(rc_registry_runtime(),p->ticket,position);
    p->next=shared_images;shared_images=p;plat_mutex_unlock(&images_lock);return p;
}
static void shared_image_note(RcRegistryPosition *first, RcRegistryPosition *last, RcRegistryPosition output) {
    if (!rc_registry_position_valid(*first) || rc_registry_position_compare(output, *first) < 0) *first = output;
    if (rc_registry_position_compare(output, *last) > 0) *last = output;
}

static int shared_image_part(SharedImageClaim *claim, RcSourceRange range, RcRegistryPosition output, int completed) {
    uint32_t i = 0;
    while (i < claim->count && !rc_range_equal(claim->spans[i].range, range)) ++i;
    if (i == claim->count) {
        if (i == claim->capacity) {
            if (claim->capacity > UINT32_MAX / 2) return 0;
            uint32_t capacity = claim->capacity ? claim->capacity * 2 : 4;
            RcRegistryImageEffectSpan *spans = realloc(claim->spans, (size_t)capacity * sizeof *spans);
            if (!spans) return 0;
            claim->spans = spans;
            claim->capacity = capacity;
        }
        claim->spans[claim->count++] = (RcRegistryImageEffectSpan){.range = range};
    }
    RcRegistryImageEffectSpan *span = &claim->spans[i];
    shared_image_note(&span->first, &span->latest, output);
    if (completed) shared_image_note(&span->completed_first, &span->completed_latest, output);
    return 1;
}

static void shared_image_release(SharedImageClaim *p, EffectActual *actuals, RcRegistryPosition output,
                                 int completed, int maybe) {
    plat_mutex_lock(&images_lock);
    RcRegistry *r = rc_registry_runtime();
    int unowned = 0, owned = 0;
    for (EffectActual *a = actuals; a; a = a->next) {
        if (!a->recorded) continue;
        if (a->owned) { owned = 1; continue; }
        unowned = 1;
        if (completed || maybe) {
            p->executed = 1;
            if (!p->broad && (a->visible || !shared_image_part(p, a->interval.range, output, completed)))
                p->broad = 1;
        }
    }
    if (unowned && p->ticket.id) {
        if ((owned && (completed || maybe)) ||
            rc_registry_image_producer_settle(r, p->ticket, output, completed || maybe) != RC_REGISTRY_OK)
            p->broad = 1;
    }

    if (--p->refs) { plat_mutex_unlock(&images_lock); return; }
    SharedImageClaim **at = &shared_images;
    while (*at && *at != p) at = &(*at)->next;
    if (*at != p) rt_fatal("resource effects: shared image ownership lost");
    *at = p->next;
    if (p->ticket.id) {
        if (!p->executed) rc_registry_write_cancel(r, p->ticket);
        else if (p->broad || rc_registry_image_effects_settle(r, p->ticket, p->spans, p->count) != RC_REGISTRY_OK)
            rc_registry_write_abandon(r, p->ticket, 1);
    }
    free(p->spans);
    free(p);
    plat_mutex_unlock(&images_lock);
}
typedef struct EffectClaim {
    struct EffectClaim *next;
    RcInterval effect, descriptor;
    RcRegistryWriterKind kind;
    uint64_t pinned_image;
    ResourceImageEffectRole *role;
    int opaque;
    RcRegistryWrite ticket;
    SharedImageClaim *shared;
    EffectActual *actual;
    EffectActual inline_actual;
    uint8_t inline_used, position_resolved;
} EffectClaim;
typedef struct EffectPartDisposition {
    struct EffectPartDisposition *next;
    void *part, *owner;
    int (*known_unhanded)(const void *);
    void (*release)(void *part, void *owner, int known_unhanded);
} EffectPartDisposition;

struct ResourceCommandEffects {
    RcRegistryPosition position;
    EffectPartDisposition *dispositions;
    EffectBufferPreservation *preservations;
    EffectClaim *claims;
    ResourceImageEffectRole *declaring_role;
    RcRegistryWriteCompletion opaque;
    RcSourceRange *opaque_ranges;
    int emitted, opaque_resolved;
    int buffer_trace_issued;
};
static _Thread_local ResourceCommandEffects *current;

enum { IMAGE_TRACE_SLOTS = 8192, IMAGE_TRACE_WAYS = 4, IMAGE_TRACE_REPORTS = 64 };
typedef struct {
    uint64_t program;
    uint32_t opcode;
    ResourceImageTraceRoute route;
} ImageTraceCommand;
typedef struct {
    RcRegistryPosition output;
    uint64_t sequence;
    uint64_t image;
    ResourceImageTraceDetails details;
} ImageTraceDetails;
typedef struct {
    RcRegistryPosition output;
    RcRegistryWrite writer;
    RcInterval claim;
    RcRegistryImageBinding image;
    ResourceImageTraceDetails details;
    uint64_t program, declared_image, sequence;
    uint32_t opcode, prepared, recorded, owned;
    ResourceImageTraceRoute route;
    RcRegistryResult resolved;
    int emitted, known, missing_actual;
} ImageTraceEntry;
typedef struct {
    RcRegistryPosition output;
    RcRegistryWrite writer;
    RcInterval claim;
    ResourceBufferIssueTrace binding;
    RcRegistryBufferLocation access, qualified;
    uint64_t sequence;
    int conflict, proof, status, initialized;
} BufferTraceEntry;
typedef struct {
    BufferTraceEntry buffers[IMAGE_TRACE_SLOTS];
    uint32_t buffer_reports;
    ImageTraceDetails details[IMAGE_TRACE_SLOTS];
    ImageTraceEntry entries[IMAGE_TRACE_SLOTS];
    uint64_t sequence;
    uint32_t reports;
} ImageTraceLedger;
static ImageTraceLedger *image_trace;
static PlatMutex image_trace_lock = PLAT_MUTEX_INIT;
static int image_trace_mode;

int resource_effects_image_trace_enabled(void) {
    int mode = __atomic_load_n(&image_trace_mode, __ATOMIC_ACQUIRE);
    if (mode) return mode == 2;
    const char *enabled = getenv("NX2NAVI_IMAGE_PRODUCER_TRACE");
    if (!enabled || enabled[0] != '1' || enabled[1]) {
        __atomic_store_n(&image_trace_mode, 1, __ATOMIC_RELEASE);
        return 0;
    }
    plat_mutex_lock(&image_trace_lock);
    mode = __atomic_load_n(&image_trace_mode, __ATOMIC_RELAXED);
    if (!mode) {
        image_trace = calloc(1, sizeof *image_trace);
        mode = image_trace ? 2 : 1;
        __atomic_store_n(&image_trace_mode, mode, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&image_trace_lock);
    return mode == 2;
}

static unsigned image_trace_index(RcRegistryPosition output, uint64_t key) {
    uint64_t hash = (output.scene * UINT64_C(0xc2b2ae3d27d4eb4f)) ^ output.ordinal ^
                    (key * UINT64_C(0x9e3779b97f4a7c15));
    hash ^= hash >> 33;
    hash *= UINT64_C(0xff51afd7ed558ccd);
    return (unsigned)(hash ^ (hash >> 33)) & (IMAGE_TRACE_SLOTS - IMAGE_TRACE_WAYS);
}

static ImageTraceDetails *image_trace_details_slot(RcRegistryPosition output, uint64_t image, int insert) {
    ImageTraceDetails *bucket = &image_trace->details[image_trace_index(output, image)];
    ImageTraceDetails *oldest = bucket;
    for (unsigned i = 0; i < IMAGE_TRACE_WAYS; ++i) {
        ImageTraceDetails *entry = &bucket[i];
        if (entry->image == image && !rc_registry_position_compare(entry->output, output)) return entry;
        if (entry->sequence < oldest->sequence) oldest = entry;
    }
    if (!insert) return NULL;
    *oldest = (ImageTraceDetails){.output = output, .image = image};
    return oldest;
}

static ImageTraceEntry *image_trace_entry_slot(RcRegistryPosition output, RcRegistryWrite writer, int insert) {
    ImageTraceEntry *bucket = &image_trace->entries[image_trace_index(output, writer.id)];
    ImageTraceEntry *oldest = bucket;
    for (unsigned i = 0; i < IMAGE_TRACE_WAYS; ++i) {
        ImageTraceEntry *entry = &bucket[i];
        if (entry->writer.id == writer.id && entry->writer.index == writer.index &&
            !rc_registry_position_compare(entry->output, output)) return entry;
        if (entry->sequence < oldest->sequence) oldest = entry;
    }
    return insert ? oldest : NULL;
}

void resource_effects_image_trace_details(ResourceCommandEffects *effects, uint64_t image,
                                          ResourceImageTraceDetails details) {
    if (!effects) effects = current;
    if (!effects || !image || !resource_effects_image_trace_enabled()) return;
    const RcRegistryPosition output = effects->position;
    plat_mutex_lock(&image_trace_lock);
    ImageTraceDetails *entry = image_trace_details_slot(output, image, 1);
    entry->sequence = ++image_trace->sequence;
    if (details.selected_view || (!entry->details.selected_view && details.aspect)) {
        entry->details.format = details.format;
        entry->details.aspect = details.aspect;
        entry->details.mip = details.mip;
        entry->details.mips = details.mips;
        entry->details.layer = details.layer;
        entry->details.layers = details.layers;
        entry->details.selected_view = details.selected_view;
    }
    entry->details.independent |= details.independent;
    entry->details.pinned |= details.pinned;
    entry->details.replacement |= details.replacement;
    if (!entry->details.format) entry->details.format = details.format;
    if (details.projection) entry->details.projection = details.projection;
    if (details.baseline) entry->details.baseline = details.baseline;
    if (details.placement) entry->details.placement = details.placement;
    if (details.admission) entry->details.admission = details.admission;
    plat_mutex_unlock(&image_trace_lock);
}

static void image_trace_record(const ResourceCommandEffects *effects, const EffectClaim *claim,
                               RcRegistryWrite writer, RcRegistryResult resolved, int known, int missing_actual,
                               const ImageTraceCommand *command) {
    ImageTraceEntry row = {
        .output = effects->position, .writer = writer, .claim = claim->effect,
        .declared_image = claim->pinned_image, .program = command ? command->program : 0,
        .opcode = command ? command->opcode : 0, .route = command ? command->route : RESOURCE_IMAGE_TRACE_COMMAND,
        .resolved = resolved, .emitted = effects->emitted, .known = known, .missing_actual = missing_actual,
    };
    for (const EffectActual *actual = claim->actual; actual; actual = actual->next) {
        row.prepared += actual->prepared != 0;
        row.owned += actual->recorded && actual->owned;
        if (actual->recorded) {
            if (!row.recorded) row.image = (RcRegistryImageBinding){actual->image, actual->hold};
            row.recorded++;
        }
    }
    // A single exact actual can supply a source view and publication note. An ambiguous set is left unlabelled.
    const uint64_t image = row.recorded == 1 ? row.image.image : 0;
    plat_mutex_lock(&image_trace_lock);
    if (image) {
        const ImageTraceDetails *details = image_trace_details_slot(row.output, image, 0);
        if (details) row.details = details->details;
    }
    row.sequence = ++image_trace->sequence;
    *image_trace_entry_slot(row.output, writer, 1) = row;
    plat_mutex_unlock(&image_trace_lock);
}

// Reuse the opted-in bounded image flight recorder. No resource custody is acquired by this ledger.
static BufferTraceEntry *buffer_trace_slot(RcRegistryPosition output, RcRegistryWrite writer, int insert) {
    BufferTraceEntry *bucket=&image_trace->buffers[image_trace_index(output,writer.id)], *oldest=bucket;
    for (unsigned i=0;i<IMAGE_TRACE_WAYS;i++) {
        BufferTraceEntry *row=&bucket[i];
        if (row->sequence && row->writer.id==writer.id && row->writer.index==writer.index &&
            !rc_registry_position_compare(row->output,output)) return row;
        if (row->sequence < oldest->sequence) oldest=row;
    }
    return insert ? oldest : NULL;
}

static void buffer_trace_proof(const ResourceCommandEffects *e, const EffectClaim *claim,
    RcRegistryBufferLocation native, int proof, int status, int initialized) {
    if (!e->buffer_trace_issued) return;
    plat_mutex_lock(&image_trace_lock);
    BufferTraceEntry *row=buffer_trace_slot(e->position,claim->ticket,0);
    if (row) {
        row->qualified=native;
        row->proof=proof;
        row->status=status;
        row->initialized=initialized;
    }
    plat_mutex_unlock(&image_trace_lock);
}

static void buffer_trace_blocker(RcRegistryPosition read, const RcRegistryNativeImageFailure *failure) {
    enum { MATCH_LIMIT=4 };
    BufferTraceEntry rows[MATCH_LIMIT], exact={0};
    unsigned found=0;
    const RcRegistryPosition output=failure->blocking_output;
    const RcSourceRange demand=rc_range_valid(failure->interval.range) ? failure->interval.range :
        failure->blocking_interval.range;
    plat_mutex_lock(&image_trace_lock);
    if (image_trace->buffer_reports == 16) {
        plat_mutex_unlock(&image_trace_lock);
        return;
    }
    image_trace->buffer_reports++;
    const BufferTraceEntry *match=buffer_trace_slot(output,failure->blocker,0);
    if (match) exact=*match;
    else if (rc_range_valid(demand)) for (unsigned i=0;i<IMAGE_TRACE_SLOTS;i++) {
        const BufferTraceEntry *row=&image_trace->buffers[i];
        const RcSourceRange claim=row->claim.range;
        if (!row->sequence || rc_registry_position_compare(row->output,output) || claim.backing!=demand.backing ||
            claim.offset>=demand.offset+demand.size || demand.offset>=claim.offset+claim.size) continue;
        if (found<MATCH_LIMIT) rows[found]=*row;
        found++;
    }
    plat_mutex_unlock(&image_trace_lock);
    rt_log("vk: original buffer blocker read=%llu/%llu output=%llu/%llu ticket=%llu/%u fate=%u cause=%d "
           "need=%llu:%#llx+%#llx exact=%d normalized-overlaps=%u",
           (unsigned long long)read.scene,(unsigned long long)read.ordinal,(unsigned long long)output.scene,
           (unsigned long long)output.ordinal,(unsigned long long)failure->blocker.id,failure->blocker.index,
           failure->writer_state,failure->cause,(unsigned long long)demand.backing,(unsigned long long)demand.offset,
           (unsigned long long)demand.size,exact.sequence!=0,found);
    const unsigned count=exact.sequence?1:found<MATCH_LIMIT?found:MATCH_LIMIT;
    for (unsigned i=0;i<count;i++) {
        const BufferTraceEntry *row=exact.sequence?&exact:&rows[i];
        const ResourceBufferIssueTrace *b=&row->binding;
        static const char *const proof[]={"unobserved","exact-write","preserved","physical-refusal"};
        rt_log("vk: original buffer issued output=%llu/%llu ticket=%llu/%u join=%s ambiguous=%d "
               "op=%u program=%llu shader-key64=%016llx stage=%u set=%u binding=%u borrowed=%d "
               "guest=%#llx map=%llu backing=%llu range=%#llx+%#llx descriptor=%#llx:%#llx+%#llx "
               "native-access=%llu/%llu/%#llx proof=%s result=%d initialized=%d output-native=%llu/%llu/%#llx",
               (unsigned long long)output.scene,(unsigned long long)output.ordinal,(unsigned long long)row->writer.id,
               row->writer.index,exact.sequence?"ticket":"range",row->conflict,b->opcode,(unsigned long long)b->program,
               (unsigned long long)b->shader,b->stage,b->set,b->binding,b->borrowed,(unsigned long long)row->claim.address,
               (unsigned long long)row->claim.mapping,(unsigned long long)row->claim.range.backing,
               (unsigned long long)row->claim.range.offset,(unsigned long long)row->claim.range.size,
               (unsigned long long)b->buffer,(unsigned long long)b->offset,(unsigned long long)b->size,
               (unsigned long long)row->access.allocation,(unsigned long long)row->access.incarnation,
               (unsigned long long)row->access.offset,proof[row->proof],row->status,row->initialized,
               (unsigned long long)row->qualified.allocation,(unsigned long long)row->qualified.incarnation,
               (unsigned long long)row->qualified.offset);
    }
}

void resource_effects_image_trace_blocker(RcRegistryPosition read, const RcRegistryNativeImageFailure *failure) {
    if (!failure || !failure->blocker.id || !rc_registry_position_valid(failure->blocking_output) ||
        !resource_effects_image_trace_enabled()) return;
    const RcRegistryPosition output = failure->blocking_output;
    const RcRegistryWrite writer = failure->blocker;
    if (failure->writer_kind==RC_REGISTRY_BUFFER) {
        buffer_trace_blocker(read,failure);
        return;
    }
    plat_mutex_lock(&image_trace_lock);
    if (image_trace->reports == IMAGE_TRACE_REPORTS) {
        plat_mutex_unlock(&image_trace_lock);
        return;
    }
    image_trace->reports++;
    const ImageTraceEntry *entry = image_trace_entry_slot(output, writer, 0);
    const int found = entry != NULL;
    const ImageTraceEntry row = entry ? *entry : (ImageTraceEntry){0};
    plat_mutex_unlock(&image_trace_lock);

    RcRegistryImageProducerState state;
    const int registry_found = rc_registry_image_producer_state(rc_registry_runtime(), writer, output, &state);
    const RcInterval claim = rc_range_valid(failure->blocking_interval.range) ? failure->blocking_interval :
        found ? row.claim : state.interval;
    const ResourceImageTraceDetails details = found ? row.details : (ResourceImageTraceDetails){0};
    static const char *const routes[] = {"command", "draw", "draw-retained", "dispatch",
                                        "dispatch-compat-selected", "dispatch-retained"};
    static const char *const projection[] = {"unobserved", "legacy-write", "unsupported-plane", "placement",
                                             "meaningful-demand", "publication-refused", "published"};
    static const char *const baselines[] = {"unobserved", "full-overwrite", "current-plane", "alias-seed"};
    static const char *const placements[] = {"unobserved", "private", "pool-bound", "other"};
    static const char *const admissions[] = {"unobserved", "storage-layout", "final-binding", "storage-peer",
                                             "storage-target", "storage-plane", "retained"};
    rt_log("resource image producer read=%llu/%llu output=%llu/%llu writer=%llu/%u observer=%s registry=%s "
           "opcode=%u route=%s program=%llu mapping=%llu backing=%llu range=%#llx+%#llx guest=%#llx "
           "emitted=%d prepared=%u recorded=%u owned=%u missing-actual=%d actual-known=%d resolve=%d "
           "declared-image=%#llx actual-image=%#llx actual-hold=%llu/%u registry-image=%#llx registry-hold=%llu/%u "
           "active=%u unresolved=%u unknown=%d projected=%d incomplete=%d exact-effect=%d "
           "selected-view=%d format=%u aspect=%u mip=%u+%u layer=%u+%u independent=%d pinned=%d replacement=%d baseline=%s placement=%s plane=%s admission=%s",
           (unsigned long long)read.scene, (unsigned long long)read.ordinal, (unsigned long long)output.scene,
           (unsigned long long)output.ordinal, (unsigned long long)writer.id, writer.index,
           found ? "retained" : "unavailable", registry_found ? "retained" : "unavailable",
           found ? row.opcode : 0, found ? routes[row.route] : "unobserved", (unsigned long long)(found ? row.program : 0),
           (unsigned long long)claim.mapping, (unsigned long long)claim.range.backing,
           (unsigned long long)claim.range.offset, (unsigned long long)claim.range.size, (unsigned long long)claim.address,
           found ? row.emitted : -1, found ? row.prepared : 0, found ? row.recorded : 0, found ? row.owned : 0,
           found ? row.missing_actual : -1, found ? row.known : -1, found ? (int)row.resolved : -1,
           (unsigned long long)(found ? row.declared_image : 0), (unsigned long long)(found ? row.image.image : 0),
           (unsigned long long)(found ? row.image.hold.id : 0), found ? row.image.hold.index : 0,
           (unsigned long long)state.image.image, (unsigned long long)state.image.hold.id, state.image.hold.index,
           state.active, state.unresolved, registry_found ? state.unknown : -1, registry_found ? state.projected : -1,
           registry_found ? state.incomplete : -1, registry_found ? state.exact_effect : -1,
           details.selected_view, details.format, details.aspect, details.mip, details.mips, details.layer, details.layers,
           details.independent, details.pinned, details.replacement, baselines[details.baseline],
           placements[details.placement], projection[details.projection], admissions[details.admission]);
}
static RcInterval guest_interval(RcSourceRange range) {
    return (RcInterval){.address=range.offset,.range=range};
}
static int interval_valid(RcInterval interval) {
    return rc_range_valid(interval.range) && interval.range.size<=UINT64_MAX-interval.address;
}
static int contains(RcInterval outer,RcInterval inner) {
    return interval_valid(inner) && outer.mapping==inner.mapping && outer.range.backing==inner.range.backing &&
        inner.address>=outer.address && inner.range.offset>=outer.range.offset &&
        inner.address-outer.address==inner.range.offset-outer.range.offset &&
        inner.range.offset+inner.range.size<=outer.range.offset+outer.range.size;
}
static RcRegistryWrite image_ticket(const EffectClaim *claim) {
    if (claim->shared) return claim->shared->ticket;
    return claim->kind==RC_REGISTRY_IMAGE ? claim->ticket : (RcRegistryWrite){0};
}
static void invalidate_image_position(EffectClaim *claim) {
    if (claim->position_resolved) rc_registry_image_producer_invalidate(rc_registry_runtime(),image_ticket(claim));
}

static RcRegistryResult shared_image_take_mapped(EffectClaim *claims,RcRegistryWriteExtent *extents,
                                                RcRegistryWrite *tickets,RcRegistryPosition position) {
    SharedImageClaim *pending=NULL;
    uint32_t missing=0;
    RcRegistryResult result=RC_REGISTRY_OK;
    plat_mutex_lock(&images_lock);
    for (EffectClaim *c=claims;c;c=c->next) {
        SharedImageClaim *p=shared_image_find(shared_images,c->effect,position.scene);
        if (!p) p=shared_image_find(pending,c->effect,position.scene);
        if (!p) {
            p=calloc(1,sizeof *p);
            if (!p) {result=RC_REGISTRY_CAPACITY;break;}
            p->interval=c->effect;
            p->scene=position.scene;
            p->next=pending;
            pending=p;
            extents[missing++]=(RcRegistryWriteExtent){c->effect.range,RC_REGISTRY_IMAGE};
        }
        c->shared=p;
    }
    // Admit every new extent together. Reused groups get their additional positions only after admission succeeds.
    if (result==RC_REGISTRY_OK && missing)
        result=rc_registry_write_begin_extents(rc_registry_runtime(),extents,missing,position,tickets);
    while (pending) {
        SharedImageClaim *p=pending;
        pending=p->next;
        if (result==RC_REGISTRY_OK) {
            p->ticket=tickets[--missing];
            p->next=shared_images;
            shared_images=p;
            mapped_image_unique++;
        } else free(p);
    }
    if (result==RC_REGISTRY_OK) for (EffectClaim *c=claims;c;c=c->next) {
        // On position-capacity refusal the registry keeps the whole group conservative.
        rc_registry_image_producer_add(rc_registry_runtime(),c->shared->ticket,position);
        c->shared->refs++;
        mapped_image_requests++;
    }
    plat_mutex_unlock(&images_lock);
    return result;
}

ResourceCommandEffects *resource_effects_create(RcRegistryPosition position) {
    ResourceCommandEffects *e=calloc(1,sizeof *e);
    if (!e) rt_fatal("resource effects: allocation failed");
    e->position=position;
    return e;
}
void resource_effects_add(ResourceCommandEffects *e,RcSourceRange effect,RcSourceRange descriptor,
                          RcRegistryWriterKind kind) {
    EffectClaim *c=calloc(1,sizeof *c);
    if (!c) rt_fatal("resource effects: claim allocation failed");
    c->effect=guest_interval(effect);c->descriptor=guest_interval(descriptor);c->kind=kind;
    if (kind==RC_REGISTRY_IMAGE) c->role=e->declaring_role;
    if (effect.size) {
        if(kind==RC_REGISTRY_IMAGE)c->shared=shared_image_take(effect,e->position);
        else rc_registry_write_begin_at(rc_registry_runtime(),effect,kind,e->position,&c->ticket);
    }
    c->next=e->claims;e->claims=c;
}
void resource_effects_empty(ResourceCommandEffects *e,RcSourceRange descriptor) {
    resource_effects_add(e,(RcSourceRange){0},descriptor,RC_REGISTRY_BUFFER);
}
static RcRegistryResult add_intervals(ResourceCommandEffects *e,const RcInterval *effects,uint32_t effect_count,
                                      const RcInterval *descriptors,uint32_t descriptor_count,
                                      RcRegistryWriterKind kind,uint64_t pinned_image,RcRegistryBufferPhaseExtent *phase) {
    if (!e || !rc_registry_position_valid(e->position) || !descriptors || !descriptor_count || (effect_count && !effects) ||
        (kind!=RC_REGISTRY_IMAGE && kind!=RC_REGISTRY_BUFFER) || (!effect_count && kind!=RC_REGISTRY_BUFFER))
        return RC_REGISTRY_INVALID;
    for (uint32_t i=0;i<descriptor_count;i++)
        if (!descriptors[i].mapping || !interval_valid(descriptors[i]) ||
            descriptors[i].range.backing==RC_GUEST_COMPATIBILITY_BACKING) return RC_REGISTRY_INVALID;
    RcRegistryWriteExtent *extents=effect_count?malloc((size_t)effect_count*sizeof *extents):NULL;
    RcRegistryWrite *tickets=effect_count?malloc((size_t)effect_count*sizeof *tickets):NULL;
    RcRegistryResult result=effect_count && (!extents || !tickets)?RC_REGISTRY_CAPACITY:RC_REGISTRY_OK;
    EffectClaim *claims=NULL,**tail=&claims;
    for (uint32_t i=0;i<effect_count && result==RC_REGISTRY_OK;i++) {
        uint32_t descriptor=0;
        while (descriptor<descriptor_count && !contains(descriptors[descriptor],effects[i])) descriptor++;
        if (descriptor==descriptor_count) {result=RC_REGISTRY_INVALID;break;}
        EffectClaim *claim=calloc(1,sizeof *claim);
        if (!claim) {result=RC_REGISTRY_CAPACITY;break;}
        extents[i]=(RcRegistryWriteExtent){effects[i].range,kind};
        claim->effect=effects[i];claim->descriptor=descriptors[descriptor];claim->kind=kind;
        claim->pinned_image=pinned_image;
        if (kind==RC_REGISTRY_IMAGE) claim->role=e->declaring_role;
        *tail=claim;tail=&claim->next;
    }
    if (kind==RC_REGISTRY_BUFFER) for (uint32_t i=0;i<descriptor_count && result==RC_REGISTRY_OK;i++) {
        // A proven unwritten interval still needs to match the complete descriptor presented by recording.
        EffectClaim *claim=claims;
        while (claim && !contains(claim->descriptor,descriptors[i])) claim=claim->next;
        if (claim) continue;
        claim=calloc(1,sizeof *claim);
        if (!claim) {result=RC_REGISTRY_CAPACITY;break;}
        claim->descriptor=descriptors[i];claim->kind=kind;
        *tail=claim;tail=&claim->next;
    }
    if (result==RC_REGISTRY_OK && effect_count)
        result=kind==RC_REGISTRY_IMAGE ? shared_image_take_mapped(claims,extents,tickets,e->position) :
            rc_registry_write_begin_extents(rc_registry_runtime(),extents,effect_count,e->position,tickets);
    if (result==RC_REGISTRY_OK && phase) {
        for (uint32_t i=0;i<effect_count;i++) phase[i]=(RcRegistryBufferPhaseExtent){tickets[i],effects[i]};
        RcRegistryPosition read={e->position.scene,e->position.ordinal-1};
        result=rc_registry_buffer_phase_bind(rc_registry_runtime(),phase,effect_count,read);
        if (result!=RC_REGISTRY_OK) {
            rc_registry_write_cancel_extents(rc_registry_runtime(),tickets,effect_count);
            memset(phase,0,(size_t)effect_count*sizeof *phase);
        }
    }
    if (result==RC_REGISTRY_OK) {
        EffectClaim *claim=claims;
        for (uint32_t i=0;kind==RC_REGISTRY_BUFFER && i<effect_count;i++,claim=claim->next)
            claim->ticket=tickets[i];
        *tail=e->claims;e->claims=claims;
    } else while (claims) {
        EffectClaim *next=claims->next;
        free(claims);claims=next;
    }
    free(extents);free(tickets);
    return result;
}

RcRegistryResult resource_effects_add_intervals(ResourceCommandEffects *e,const RcInterval *effects,uint32_t effect_count,
                                               const RcInterval *descriptors,uint32_t descriptor_count,
                                               RcRegistryWriterKind kind) {
    return add_intervals(e,effects,effect_count,descriptors,descriptor_count,kind,0,NULL);
}

RcRegistryResult resource_effects_add_buffer_phase(ResourceCommandEffects *e,const RcInterval *intervals,
                                                   uint32_t count,RcRegistryBufferPhaseExtent *phase) {
    if (!phase || !count || !e || !(e->position.ordinal&1)) return RC_REGISTRY_INVALID;
    memset(phase,0,(size_t)count*sizeof *phase);
    return add_intervals(e,intervals,count,intervals,count,RC_REGISTRY_BUFFER,0,phase);
}

RcRegistryResult resource_effects_add_pinned_image(ResourceCommandEffects *e,uint64_t image,
                                                 const RcInterval *intervals,uint32_t count) {
    if(!image)return RC_REGISTRY_INVALID;
    return add_intervals(e,intervals,count,intervals,count,RC_REGISTRY_IMAGE,image,NULL);
}
void resource_effects_opaque(ResourceCommandEffects *e) {
    if (resource_effects_opaque_intervals(e, NULL) != RC_REGISTRY_OK) rc_registry_fail_closed(rc_registry_runtime());
    for(EffectClaim *c=e->claims;c;c=c->next)if(c->opaque)return;
    RcSourceRange domain={RC_GUEST_COMPATIBILITY_BACKING,0,UINT64_MAX};
    resource_effects_add(e,domain,domain,RC_REGISTRY_UNKNOWN);e->claims->opaque=1;
}
RcRegistryResult resource_effects_opaque_intervals(ResourceCommandEffects *e,RcRegistryWrite *out) {
    if (out) *out=(RcRegistryWrite){0};
    if (!e) return RC_REGISTRY_INVALID;
    RcRegistryResult result=RC_REGISTRY_OK;
    if (!e->opaque.ticket.id)
        result=rc_registry_write_guard_begin_reason_at(rc_registry_runtime(), RC_REGISTRY_UNKNOWN, e->position,
                                                       RC_REGISTRY_GUARD_COMMAND_OPAQUE, &e->opaque.ticket);
    if (out) *out=e->opaque.ticket;
    return result;
}
RcRegistryResult resource_effects_resolve_opaque_intervals(const RcSourceRange *written,uint32_t written_count,
                                                          const RcSourceRange *uncertain,uint32_t uncertain_count) {
    if (!current || !current->opaque.ticket.id || (written_count && !written) || (uncertain_count && !uncertain))
        return RC_REGISTRY_INVALID;
    size_t count=(size_t)written_count+uncertain_count;
    RcSourceRange *copy=count?malloc(count*sizeof *copy):NULL;
    if (count && !copy) return RC_REGISTRY_CAPACITY;
    for (uint32_t i=0;i<written_count;i++) copy[i]=written[i];
    for (uint32_t i=0;i<uncertain_count;i++) copy[(size_t)written_count+i]=uncertain[i];
    free(current->opaque_ranges);
    current->opaque_ranges=copy;
    current->opaque.written=copy;
    current->opaque.written_count=written_count;
    current->opaque.uncertain=copy?copy+written_count:NULL;
    current->opaque.uncertain_count=uncertain_count;
    current->opaque_resolved=1;
    return RC_REGISTRY_OK;
}
int resource_effects_read_allowed_range(const ResourceCommandEffects *e,RcSourceRange range) {
    if (!rc_range_valid(range)) return 0;
    if (e && e->opaque.ticket.id && range.backing!=RC_GUEST_COMPATIBILITY_BACKING) return 0;
    for (EffectClaim *c=e?e->claims:NULL;c;c=c->next)
        if (rc_range_overlap(c->effect.range,range)) return 0;
    return 1;
}
int resource_effects_read_allowed(const ResourceCommandEffects *e,uint64_t addr,uint64_t size) {
    return resource_effects_read_allowed_range(e,(RcSourceRange){RC_GUEST_COMPATIBILITY_BACKING,addr,size});
}

static int effect_shader_raw_has_mapping(const ResourceCommandEffects *effects,
    const ResourceImageEffectRole *attachment, const EffectClaim *raw) {
    uint64_t cursor = raw->effect.range.offset;
    const uint64_t end = cursor + raw->effect.range.size;
    while (cursor < end) {
        uint64_t through = cursor;
        for (const EffectClaim *claim = effects->claims; claim; claim = claim->next) {
            const RcInterval *effect = &claim->effect;
            if (claim->kind != raw->kind || claim->role == attachment || !effect->mapping ||
                effect->range.backing == RC_GUEST_COMPATIBILITY_BACKING || !interval_valid(*effect) ||
                effect->address > cursor) continue;
            const uint64_t edge = effect->address + effect->range.size;
            if (edge > through) through = edge;
        }
        if (through == cursor) return 0;
        cursor = through;
    }
    return 1;
}

RcRegistryResult resource_effects_shader_write_peer(const ResourceCommandEffects *effects,
    const ResourceImageEffectRole *attachment, RcInterval peer, RcInterval *overlap) {
    if (overlap) *overlap = (RcInterval){0};
    if (!effects || !attachment || attachment->owner != effects || effects->declaring_role ||
        !peer.mapping || peer.range.backing == RC_GUEST_COMPATIBILITY_BACKING || !interval_valid(peer))
        return RC_REGISTRY_INVALID;
    if (effects->opaque.ticket.id) return RC_REGISTRY_UNSUPPORTED;
    const RcSourceRange address = {RC_GUEST_COMPATIBILITY_BACKING, peer.address, peer.range.size};
    for (const EffectClaim *claim = effects->claims; claim; claim = claim->next) {
        const RcInterval *written = &claim->effect;
        if (claim->role == attachment || !written->range.size) continue;
        if (!interval_valid(*written)) return RC_REGISTRY_INVALID;
        if (claim->kind == RC_REGISTRY_UNKNOWN ||
            !(claim->kind == RC_REGISTRY_IMAGE ? image_ticket(claim) : claim->ticket).id)
            return RC_REGISTRY_UNSUPPORTED;
        const int raw = written->range.backing == RC_GUEST_COMPATIBILITY_BACKING;
        if (rc_range_overlap(written->range, raw ? address : peer.range)) {
            if (overlap) *overlap = *written;
            return RC_REGISTRY_BLOCKED;
        }
        if (raw ? !effect_shader_raw_has_mapping(effects, attachment, claim) : !written->mapping)
            return RC_REGISTRY_UNSUPPORTED;
    }
    return RC_REGISTRY_OK;
}

ResourceCommandEffects *resource_effects_enter(ResourceCommandEffects *e) {
    ResourceCommandEffects *old=current;current=e;return old;
}
void resource_effects_leave(ResourceCommandEffects *previous) {current=previous;}
void resource_effects_emitted(void) {if(current)current->emitted=1;}

void resource_effects_retain_part_disposition(void *part, int (*known_unhanded)(const void *),
    void *owner, void (*release)(void *part, void *owner, int known_unhanded)) {
    if (!current || !current->emitted || !part || !known_unhanded || !owner || !release ||
        (current->dispositions && current->dispositions->part != part))
        rt_fatal("resource effects: invalid recorded native part disposition");
    EffectPartDisposition *disposition = malloc(sizeof *disposition);
    if (!disposition) rt_fatal("resource effects: allocating recorded native part disposition");
    *disposition = (EffectPartDisposition){current->dispositions, part, owner, known_unhanded, release};
    current->dispositions = disposition;
}
int resource_effects_before_emission(const ResourceCommandEffects *e) {return e && !e->emitted;}
int resource_effects_unrecorded(const ResourceCommandEffects *e) {
    if (!resource_effects_before_emission(e)) return 0;
    for (const EffectClaim *claim = e->claims; claim; claim = claim->next)
        for (const EffectActual *actual = claim->actual; actual; actual = actual->next)
            if (actual->recorded) return 0;
    return 1;
}
static EffectActual *actual(EffectClaim *c,uint64_t image,RcInterval interval,int visible) {
    RcSourceRange input=interval.range, effect=c->effect.range;
    if (input.backing!=effect.backing || interval.mapping!=c->effect.mapping) return NULL;
    uint64_t lo=input.offset>effect.offset?input.offset:effect.offset;
    uint64_t hi=input.offset+input.size<effect.offset+effect.size?input.offset+input.size:effect.offset+effect.size;
    if (hi<=lo) return NULL;
    invalidate_image_position(c);
    for (EffectActual *a=c->actual;a;a=a->next)
        if (a->image==image && a->interval.range.offset==lo && a->interval.range.size==hi-lo) {
            a->visible &= visible;a->prepared=1;return a;
        }
    EffectActual *a=c->inline_used?calloc(1,sizeof *a):&c->inline_actual;
    if (!a) rt_fatal("resource effects: actual range allocation failed");
    c->inline_used=1;
    a->image=image;
    a->interval=(RcInterval){interval.address+(lo-input.offset),interval.mapping,{effect.backing,lo,hi-lo}};
    a->visible=visible;a->prepared=1;
    a->next=c->actual;c->actual=a;return a;
}
void resource_effects_opaque_commit(void) {
    for(EffectClaim *c=current?current->claims:NULL;c;c=c->next)if(c->opaque) {
        EffectActual *a=actual(c,0,c->effect,0);if(a)a->recorded=1;
    }
}
static int prepare_claim_matches(const EffectClaim *claim,uint64_t image,RcRegistryWriterKind kind,int pinned) {
    return claim->kind==kind && (claim->pinned_image==image || (!claim->pinned_image && !pinned));
}

static int prepare_intervals(uint64_t image,const RcInterval *intervals,uint32_t count,int visible,
                             RcRegistryWriterKind kind,int pinned) {
    if (!current || !intervals || !count) return 0;
    // Validate all legs before adopting any. A recycled mapping anywhere in the actual vector refuses the vector.
    for (uint32_t i=0;i<count;i++) {
        EffectClaim *claim=current->claims;
        while (claim && (!prepare_claim_matches(claim,image,kind,pinned) ||
                         !contains(claim->descriptor,intervals[i]))) claim=claim->next;
        if (!claim) return 0;
    }
    for (uint32_t i=0;i<count;i++)
        for (EffectClaim *c=current->claims;c;c=c->next)
            if (prepare_claim_matches(c,image,kind,pinned) && contains(c->descriptor,intervals[i]))
                actual(c,image,intervals[i],visible);
    return 1;
}
int resource_effects_buffer_prepare_intervals(const RcInterval *intervals,uint32_t count,int visible) {
    return prepare_intervals(0,intervals,count,visible,RC_REGISTRY_BUFFER,0);
}
int resource_effects_buffer_prepare(uint64_t addr,uint64_t size,int visible) {
    RcInterval interval=guest_interval((RcSourceRange){RC_GUEST_COMPATIBILITY_BACKING,addr,size});
    return prepare_intervals(0,&interval,1,visible,RC_REGISTRY_BUFFER,0);
}
void resource_effects_buffer_prepare_native(const RcInterval *intervals, uint32_t count, uint64_t address,
                                            RcRegistryBufferLocation location) {
    if (!current || !location.allocation) return;
    for (EffectClaim *c=current->claims;c;c=c->next) {
        if (c->kind!=RC_REGISTRY_BUFFER || c->opaque || !c->effect.mapping || !c->ticket.id) continue;
        for (uint32_t i=0;i<count;i++) {
            if (!contains(intervals[i],c->effect) || c->effect.address<address) continue;
            uint64_t delta=c->effect.address-address;
            if (delta>UINT64_MAX-location.offset) continue;
            for (EffectActual *a=c->actual;a;a=a->next) {
                if (a->prepared && a->interval.address==c->effect.address &&
                    rc_range_equal(a->interval.range,c->effect.range)) {
                    RcRegistryBufferLocation tagged=location;
                    tagged.offset+=delta;
                    if (a->native_buffer.allocation && (a->native_buffer.allocation!=tagged.allocation ||
                        a->native_buffer.incarnation!=tagged.incarnation || a->native_buffer.offset!=tagged.offset))
                        a->native_buffer_conflict=1;
                    a->native_buffer=tagged;
                }
            }
        }
    }
}
void resource_effects_buffer_access_native(const RcInterval *intervals, uint32_t count, uint64_t address,
                                           RcRegistryBufferLocation location) {
    if (!current || !intervals || !count || !location.allocation) return;
    for (EffectClaim *c=current->claims;c;c=c->next) {
        if (c->kind!=RC_REGISTRY_BUFFER || c->opaque || !c->ticket.id) continue;
        for (uint32_t i=0;i<count;i++) {
            RcInterval mapped;
            uint64_t target_address=c->effect.address, target_size=c->effect.range.size;
            if (c->effect.mapping) {
                if (!contains(intervals[i],c->effect) || target_address<address) continue;
                mapped=c->effect;
            } else if (c->effect.range.backing==RC_GUEST_COMPATIBILITY_BACKING) {
                if (target_address<address || target_address-address>intervals[i].range.size ||
                    target_size>intervals[i].range.size-(target_address-address) ||
                    target_address<intervals[i].address || target_size>UINT64_MAX-target_address ||
                    intervals[i].address>UINT64_MAX-intervals[i].range.size ||
                    target_address+target_size>intervals[i].address+intervals[i].range.size ||
                    target_address-intervals[i].address>UINT64_MAX-intervals[i].range.offset) continue;
                uint64_t delta_range=target_address-intervals[i].address;
                mapped=(RcInterval){target_address,intervals[i].mapping,
                    {intervals[i].range.backing,intervals[i].range.offset+delta_range,target_size}};
            } else continue;
            uint64_t delta=target_address-address;
            if (delta>UINT64_MAX-location.offset) continue;
            for (EffectActual *a=c->actual;a;a=a->next) {
                if (a->prepared && a->interval.address==c->effect.address &&
                    rc_range_equal(a->interval.range,c->effect.range)) {
                    RcRegistryBufferLocation tagged=location;
                    tagged.offset+=delta;
                    if (a->native_access.allocation && (a->native_access.allocation!=tagged.allocation ||
                        a->native_access.incarnation!=tagged.incarnation || a->native_access.offset!=tagged.offset ||
                        a->native_access_interval.mapping!=mapped.mapping ||
                        !rc_range_equal(a->native_access_interval.range,mapped.range)))
                        a->native_access_conflict=1;
                    a->native_access=tagged;
                    a->native_access_interval=mapped;
                }
            }
        }
    }
}
int resource_effects_buffer_preserve_native(const RcInterval *intervals, uint32_t count, uint64_t address,
    RcRegistryBufferLocation location, void *receipt, ResourceBufferPreserveCheck check, void (*release)(void *)) {
    if (!current || !intervals || !count || !location.allocation || !receipt || !check || !release) return 0;
    EffectBufferPreservation *owner = NULL;
    for (EffectClaim *c = current->claims; c; c = c->next) {
        if (c->kind != RC_REGISTRY_BUFFER || c->opaque || !c->effect.mapping || !c->ticket.id ||
            c->effect.address < address) continue;
        const uint64_t delta = c->effect.address - address;
        if (delta > UINT64_MAX - location.offset) continue;
        uint32_t i = 0;
        while (i < count && !contains(intervals[i], c->effect)) i++;
        if (i == count) continue;
        for (EffectActual *a = c->actual; a; a = a->next) {
            if (!a->prepared || a->interval.address != c->effect.address ||
                !rc_range_equal(a->interval.range, c->effect.range)) continue;
            RcRegistryBufferLocation actual = location;
            actual.offset += delta;
            if (a->preservation) {
                if (a->preserved_location.allocation != actual.allocation ||
                    a->preserved_location.incarnation != actual.incarnation || a->preserved_location.offset != actual.offset)
                    a->preservation_conflict = 1;
                continue;
            }
            if (!owner) {
                owner = malloc(sizeof *owner);
                if (!owner) return 0;
                *owner = (EffectBufferPreservation){current->preservations, receipt, check, release};
                current->preservations = owner;
            }
            a->preserved_location = actual;
            a->preservation = owner;
        }
    }
    return owner != NULL;
}

void resource_effects_trace_buffer_issued(ResourceCommandEffects *e, const RcInterval *mapped, uint32_t count,
    ResourceBufferIssueTrace binding) {
    if (!e || e!=current || !e->emitted || !count || !resource_effects_image_trace_enabled()) return;
    for (const EffectClaim *c=e->claims;c;c=c->next) {
        if (c->kind!=RC_REGISTRY_BUFFER || !c->ticket.id || !c->effect.mapping || c->opaque) continue;
        uint32_t i=0;
        while (i<count && !contains(mapped[i],c->effect)) i++;
        if (i==count) continue;
        for (const EffectActual *a=c->actual;a;a=a->next) {
            if (!a->prepared || a->interval.address!=c->effect.address ||
                !rc_range_equal(a->interval.range,c->effect.range)) continue;
            e->buffer_trace_issued=1;
            plat_mutex_lock(&image_trace_lock);
            BufferTraceEntry *row=buffer_trace_slot(e->position,c->ticket,0);
            if (row) {
                const ResourceBufferIssueTrace *previous=&row->binding;
                row->conflict |= previous->stage!=binding.stage || previous->set!=binding.set ||
                    previous->binding!=binding.binding || previous->buffer!=binding.buffer || previous->offset!=binding.offset;
            } else {
                row=buffer_trace_slot(e->position,c->ticket,1);
                *row=(BufferTraceEntry){.output=e->position,.writer=c->ticket,.claim=c->effect,.binding=binding,
                    .sequence=++image_trace->sequence,.access=a->native_access};
            }
            plat_mutex_unlock(&image_trace_lock);
        }
    }
}

void resource_effects_buffers_commit_payload(RcRegistryBufferPayload *payload) {
    for (EffectClaim *c=current?current->claims:NULL;c;c=c->next)
        if (c->kind==RC_REGISTRY_BUFFER)
            for (EffectActual *a=c->actual;a;a=a->next) if (a->prepared) {
                if (!a->recorded && a->native_buffer.allocation && !a->native_buffer_conflict) {
                    const RcRegistryBufferSpan span = {a->interval, a->native_buffer};
                    const RcRegistryResult result=payload && a->visible && c->descriptor.mapping ?
                        rc_registry_buffer_write_payload_record(rc_registry_runtime(), c->ticket, span,
                            current->position, payload) :
                        rc_registry_buffer_write_record(rc_registry_runtime(), c->ticket, span, current->position);
                    buffer_trace_proof(current,c,result==RC_REGISTRY_OK?span.native:(RcRegistryBufferLocation){0},1,result,0);
                } else if (!a->recorded && a->preservation && !a->preservation_conflict &&
                         !a->native_buffer_conflict && !a->native_access_conflict) {
                    const RcRegistryPosition read = {current->position.scene, current->position.ordinal - 1};
                    const RcRegistryBufferSpan span = {a->interval, a->preserved_location};
                    int initialized = 0;
                    if (a->preservation->check(a->preservation->receipt, span, read, &initialized)) {
                        const RcRegistryResult result=rc_registry_buffer_preserved_record(rc_registry_runtime(),c->ticket,
                            span,read,initialized);
                        buffer_trace_proof(current,c,result==RC_REGISTRY_OK?span.native:(RcRegistryBufferLocation){0},
                            2,result,initialized);
                    } else buffer_trace_proof(current,c,(RcRegistryBufferLocation){0},3,-1,0);
                }
                a->recorded=1;
            }
}

void resource_effects_buffers_commit(void) {
    resource_effects_buffers_commit_payload(NULL);
}

static EffectClaim *result_claim(RcInterval interval) {
    EffectClaim *match = NULL;
    for (EffectClaim *claim = current->claims; claim; claim = claim->next) {
        if (claim->kind != RC_REGISTRY_BUFFER || claim->opaque || !claim->ticket.id || claim->actual ||
            claim->effect.range.size != interval.range.size || !contains(claim->effect, interval)) continue;
        if (match) return NULL;
        match = claim;
    }
    return match;
}

int resource_effects_buffer_result_take(uint64_t address, uint64_t size, const RcInterval *intervals,
                                        uint32_t count, RcRegistryWrite *raw, RcRegistryWrite *mapped) {
    if (!current || !size || size > UINT64_MAX - address || !intervals || !count || !raw || !mapped) return 0;
    RcInterval compatibility_range = guest_interval((RcSourceRange){RC_GUEST_COMPATIBILITY_BACKING, address, size});
    EffectClaim *compatibility = result_claim(compatibility_range);
    if (!compatibility) return 0;
    uint64_t cursor = address, end = address + size;
    for (uint32_t i = 0; i < count; ++i) {
        const RcInterval *part = &intervals[i];
        if (!part->mapping || part->range.backing == RC_GUEST_COMPATIBILITY_BACKING || part->address != cursor ||
            !part->range.size || part->range.size > end - cursor || !result_claim(*part)) return 0;
        cursor += part->range.size;
    }
    if (cursor != end) return 0;
    *raw = compatibility->ticket;
    compatibility->ticket = (RcRegistryWrite){0};
    for (uint32_t i = 0; i < count; ++i) {
        EffectClaim *claim = result_claim(intervals[i]);
        mapped[i] = claim->ticket;
        claim->ticket = (RcRegistryWrite){0};
    }
    __atomic_fetch_add(&cpu_results_transferred, 1, __ATOMIC_RELAXED);
    return 1;
}

int resource_effects_image_prepare_intervals(uint64_t image,const RcInterval *intervals,uint32_t count) {
    return prepare_intervals(image,intervals,count,0,RC_REGISTRY_IMAGE,0);
}
int resource_effects_image_prepare(uint64_t image,uint64_t addr,uint64_t size) {
    RcInterval interval=guest_interval((RcSourceRange){RC_GUEST_COMPATIBILITY_BACKING,addr,size});
    return prepare_intervals(image,&interval,1,0,RC_REGISTRY_IMAGE,0);
}
int resource_effects_image_prepare_pinned(uint64_t image,const RcInterval *intervals,uint32_t count) {
    if (!image) return 0;
    for (EffectClaim *c=current?current->claims:NULL;c;c=c->next)
        if (c->pinned_image==image)
            return prepare_intervals(image,intervals,count,0,RC_REGISTRY_IMAGE,1) ? 1 : -1;
    return 0;
}
void resource_effects_image_written_interval(uint64_t image,RcInterval interval) {
    for (EffectClaim *c=current?current->claims:NULL;c;c=c->next)
        if (c->kind==RC_REGISTRY_IMAGE)
            for (EffectActual *a=c->actual;a;a=a->next)
                if (a->image==image && contains(interval,a->interval)) {
                    a->recorded=1;a->owned=0;
                    invalidate_image_position(c);
                }
}
void resource_effects_image_written(uint64_t image,uint64_t addr) {
    resource_effects_image_written_interval(image,guest_interval((RcSourceRange){RC_GUEST_COMPATIBILITY_BACKING,addr,UINT64_MAX-addr}));
}
static EffectActual *owned_image_effect(EffectClaim *claim, uint64_t image, RcRegistryImageWriteExtent extent) {
    if (claim->kind != RC_REGISTRY_IMAGE || !claim->shared || !claim->shared->ticket.id || claim->position_resolved ||
        !contains(claim->effect, extent.interval) || !rc_range_equal(claim->effect.range, extent.interval.range))
        return NULL;
    EffectActual *actual = NULL;
    for (EffectActual *candidate = claim->actual; candidate; candidate = candidate->next) {
        if (!candidate->recorded) continue;
        if (actual || !candidate->owned || candidate->image != image || candidate->hold.id != extent.hold.id ||
            candidate->hold.index != extent.hold.index || !contains(candidate->interval, extent.interval)) return NULL;
        actual = candidate;
    }
    return actual;
}

static int image_emitted_intervals(uint64_t image, const RcInterval *intervals, const RcRegistryWrite *holds,
    uint32_t count, RcRegistryPosition output, int first_write, int preserves_native, int exact_projection_layout) {
    if (!current || rc_registry_position_compare(output, current->position) || !count || !image) return 0;
    RcRegistry *registry = rc_registry_runtime();
    for (uint32_t i = 0; i < count; i++) {
        const RcRegistryImageWriteExtent extent = {intervals[i], holds[i]};
        int emitted = 0;
        for (EffectClaim *claim = current->claims; claim && !emitted; claim = claim->next) {
            if (!owned_image_effect(claim, image, extent)) continue;
            const RcRegistryImageBinding binding = {image, extent.hold};
            emitted = preserves_native
                ? rc_registry_image_hold_preserved(registry, claim->shared->ticket, binding, output,
                                                   exact_projection_layout)
                : rc_registry_image_hold_recorded(registry, claim->shared->ticket, binding, output, first_write);
        }
        if (!emitted) return 0;
    }
    return 1;
}

int resource_effects_image_emitted_intervals(uint64_t image, const RcInterval *intervals, const RcRegistryWrite *holds,
    uint32_t count, RcRegistryPosition output, int first_write) {
    return image_emitted_intervals(image, intervals, holds, count, output, first_write, 0, 0);
}

int resource_effects_image_preserved_intervals(uint64_t image, const RcInterval *intervals, const RcRegistryWrite *holds,
    uint32_t count, RcRegistryPosition output, int exact_projection_layout) {
    return image_emitted_intervals(image, intervals, holds, count, output, 0, 1, exact_projection_layout);
}

void resource_effects_image_unchanged_extent(uint64_t image, RcRegistryImageWriteExtent extent,
                                             RcRegistryPosition output) {
    if (!current || !image || !extent.hold.id || rc_registry_position_compare(output, current->position)) return;
    RcRegistry *registry = rc_registry_runtime();
    for (EffectClaim *claim = current->claims; claim; claim = claim->next) {
        EffectActual *actual = owned_image_effect(claim, image, extent);
        if (!actual) continue;
        RcRegistryWrite ticket = claim->shared->ticket;
        if (rc_registry_image_hold_recorded(registry, ticket, (RcRegistryImageBinding){image, extent.hold}, output, 0) &&
            rc_registry_image_producer_resolve(registry, ticket, output, NULL, 1) == RC_REGISTRY_OK) {
            // This mapping received no logical write even if the native command ran for its other mappings.
            actual->recorded = actual->owned = 0;
            claim->position_resolved = 1;
            return;
        }
    }
}

void resource_effects_image_owned_interval(uint64_t image,RcInterval interval,RcRegistryWrite hold) {
    for (EffectClaim *c=current?current->claims:NULL;c;c=c->next)
        if (c->kind==RC_REGISTRY_IMAGE)
            for (EffectActual *a=c->actual;a;a=a->next)
                if (a->image==image && contains(interval,a->interval) && a->recorded) {
                    a->owned=1;a->hold=hold;
                    invalidate_image_position(c);
                }
}
void resource_effects_image_owned(uint64_t image,uint64_t addr,RcRegistryWrite hold) {
    resource_effects_image_owned_interval(image,guest_interval((RcSourceRange){RC_GUEST_COMPATIBILITY_BACKING,addr,UINT64_MAX-addr}),hold);
}
void resource_effects_image_record_extents(uint64_t image,const RcRegistryImageWriteExtent *extents,uint32_t count) {
    if (!image || !extents) return;
    for (EffectClaim *c=current?current->claims:NULL;c;c=c->next) {
        if (c->kind!=RC_REGISTRY_IMAGE || !c->effect.mapping || (c->pinned_image && c->pinned_image!=image)) continue;
        for (uint32_t i=0;i<count;i++) {
            if (!contains(extents[i].interval,c->effect)) continue;
            EffectActual *a=actual(c,image,c->effect,0);
            if (a) {
                a->recorded=1;
                a->hold=extents[i].hold;
                a->owned=a->hold.id!=0;
            }
            break;
        }
    }
}
void resource_effects_image_role_begin(ResourceCommandEffects *e, ResourceImageEffectRole *role) {
    if (e->declaring_role || role->owner) rt_fatal("resource effects: overlapping image role declaration");
    e->declaring_role=role;
    role->owner=e;
}

void resource_effects_image_role_end(ResourceCommandEffects *e, ResourceImageEffectRole *role) {
    if (e->declaring_role!=role) rt_fatal("resource effects: mismatched image role declaration");
    e->declaring_role=NULL;
}

void resource_effects_image_role_finish(ResourceImageEffectRole *role, int issued) {
    ResourceCommandEffects *e=role->owner;
    if (!e) return;
    if (e->declaring_role) rt_fatal("resource effects: image role recorded while declaring");
    for (EffectClaim **at=&e->claims; *at;) {
        EffectClaim *c=*at;
        if (c->role!=role) {at=&c->next;continue;}
        c->role=NULL;
        int recorded=0;
        for (const EffectActual *a=c->actual; a; a=a->next) recorded |= a->recorded;
        const RcRegistryWrite ticket=image_ticket(c);
        if (issued || recorded || c->position_resolved || !c->shared || !ticket.id ||
            rc_registry_image_producer_resolve(rc_registry_runtime(),ticket,e->position,NULL,1)!=RC_REGISTRY_OK) {
            at=&c->next;
            continue;
        }
        // A canceled role must not accept a later sibling's preparation or recording.
        *at=c->next;
        shared_image_release(c->shared,NULL,e->position,0,0);
        while (c->actual) {
            EffectActual *a=c->actual;
            c->actual=a->next;
            if (a!=&c->inline_actual) free(a);
        }
        free(c);
    }
    role->owner=NULL;
}

void resource_effects_cancel_unissued_image(ResourceCommandEffects *e, uint64_t image) {
    if (!e || !image) return;
    for (EffectClaim *c = e->claims; c; c = c->next) {
        if (c->kind != RC_REGISTRY_IMAGE || c->pinned_image != image || c->position_resolved) continue;
        int recorded = 0;
        for (const EffectActual *a = c->actual; a; a = a->next) recorded |= a->recorded;
        if (recorded) continue; // An earlier issued phase can share this exact output.
        RcRegistryWrite writer = image_ticket(c);
        if (writer.id &&
            rc_registry_image_producer_resolve(rc_registry_runtime(), writer, e->position, NULL, 1) == RC_REGISTRY_OK)
            c->position_resolved = 1;
    }
}

static void effects_recorded(ResourceCommandEffects *e, int emitted, const ImageTraceCommand *trace) {
    if(!e)return;
    e->emitted |= emitted != 0;
    for(EffectClaim *c=e->claims;c;c=c->next) {
        if(c->kind==RC_REGISTRY_IMAGE)
            for(EffectActual *a=c->actual;a;a=a->next)if(a->recorded && a->owned &&
                !rc_registry_image_hold_covers(rc_registry_runtime(),a->hold,a->interval.range)) {
                a->owned=0;
                invalidate_image_position(c);
            }
        if(c->kind==RC_REGISTRY_BUFFER && c->ticket.id && e->emitted) {
            if(c->descriptor.mapping) {
                EffectActual *recorded=c->actual;
                while(recorded && !recorded->recorded)recorded=recorded->next;
                if(!recorded) {
                    EffectActual *unaccounted=actual(c,0,c->effect,0);
                    if(unaccounted)unaccounted->recorded=1;
                }
            }
            // This records only the native placement of a may-write that really emitted. It remains separate
            // from exact output/write publication and lets narrow RMW consumers match prior providers.
            for(EffectActual *a=c->actual;a;a=a->next)if(a->recorded && a->native_access.allocation &&
                !a->native_access_conflict)
                (void)rc_registry_buffer_access_record(rc_registry_runtime(),c->ticket,
                    (RcRegistryBufferSpan){a->native_access_interval,a->native_access},e->position);
        }
        RcRegistryWrite image_writer=image_ticket(c);
        if(!image_writer.id || c->position_resolved)continue;
        RcRegistryImageBinding image={0};int recorded=0,known=1;
        for(EffectActual *a=c->actual;a;a=a->next)if(a->recorded) {
            recorded++;
            if(!a->owned || !a->hold.id || !rc_range_equal(a->interval.range,c->effect.range) || recorded!=1)known=0;
            image=(RcRegistryImageBinding){a->image,a->hold};
        }
        // Actual command emission without a captured destination is not cancellation. Preserve the declared
        // range through real completion; this can conservatively include an unused output of that command.
        const int missing_actual = !recorded && e->emitted;
        if(missing_actual) {
            uint32_t prepared=0;for(EffectActual *a=c->actual;a;a=a->next)prepared+=a->prepared!=0;
            rc_registry_diagnose_missing_image(rc_registry_runtime(),image_writer,e->position,prepared);
            EffectActual *a=actual(c,0,c->effect,0);
            if(a)a->recorded=1;
            recorded=1;known=0;
        }
        if(!recorded)continue;
        const RcRegistryResult resolved = rc_registry_image_producer_resolve(rc_registry_runtime(),image_writer,
            e->position,known?&image:NULL,0);
        if (resource_effects_image_trace_enabled())
            image_trace_record(e, c, image_writer, resolved, known, missing_actual, trace);
        c->position_resolved=1;
    }
}

void resource_effects_recorded(ResourceCommandEffects *e, int emitted) {
    effects_recorded(e, emitted, NULL);
}

void resource_effects_recorded_image_trace(ResourceCommandEffects *e, int emitted, uint32_t opcode,
                                           uint64_t program, ResourceImageTraceRoute route) {
    const ImageTraceCommand trace = {.program = program, .opcode = opcode, .route = route};
    effects_recorded(e, emitted, &trace);
}

static void settle_normalized(ResourceCommandEffects *e,int completed,int maybe) {
    RcRegistry *r=rc_registry_runtime();
    uint32_t count=0;
    size_t range_count=0;
    for (EffectClaim *c=e->claims;c;c=c->next) if (c->descriptor.mapping && c->ticket.id) {
        count++;
        for (EffectActual *a=c->actual;a;a=a->next) if (a->recorded && !a->owned) range_count+=1+!a->visible;
    }
    if (!count) return;
    RcRegistryWrite *tickets=malloc((size_t)count*sizeof *tickets);
    if (!tickets) rt_fatal("resource effects: normalized ticket allocation failed");
    uint32_t i=0;
    for (EffectClaim *c=e->claims;c;c=c->next) if (c->descriptor.mapping && c->ticket.id) tickets[i++]=c->ticket;
    RcRegistryResult result=RC_REGISTRY_STALE;
    if (completed && rc_registry_write_submit_extents(r,tickets,count,e->position.scene)==RC_REGISTRY_OK) {
        RcRegistryWriteCompletion *completions=calloc(count,sizeof *completions);
        RcSourceRange *ranges=range_count?malloc(range_count*sizeof *ranges):NULL;
        if (completions && (!range_count || ranges)) {
            i=0;
            size_t cursor=0;
            for (EffectClaim *c=e->claims;c;c=c->next) if (c->descriptor.mapping && c->ticket.id) {
                RcRegistryWriteCompletion *completion=&completions[i++];
                completion->ticket=c->ticket;
                completion->written=ranges?ranges+cursor:NULL;
                for (EffectActual *a=c->actual;a;a=a->next) if (a->recorded && !a->owned) {
                    ranges[cursor++]=a->interval.range;
                    completion->written_count++;
                }
                completion->uncertain=ranges?ranges+cursor:NULL;
                for (EffectActual *a=c->actual;a;a=a->next) if (a->recorded && !a->owned && !a->visible) {
                    ranges[cursor++]=a->interval.range;
                    completion->uncertain_count++;
                }
            }
            result=rc_registry_write_complete_extents(r,completions,count);
        }
        free(ranges);free(completions);
    }
    if (result!=RC_REGISTRY_OK) {
        result=completed || maybe ? rc_registry_write_abandon_extents(r,tickets,count) :
                                  rc_registry_write_cancel_extents(r,tickets,count);
        if (result!=RC_REGISTRY_OK) rc_registry_fail_closed(r);
    }
    free(tickets);
}
static void settle_opaque(ResourceCommandEffects *e,int completed,int maybe) {
    RcRegistryWrite guard=e->opaque.ticket;
    if (!guard.id) return;
    RcRegistry *r=rc_registry_runtime();
    RcRegistryResult result=RC_REGISTRY_STALE;
    if (!e->emitted || (!completed && !maybe)) result=rc_registry_write_cancel(r,guard);
    else {
        if (completed && e->opaque_resolved &&
            rc_registry_write_submit_extents(r,&guard,1,e->position.scene)==RC_REGISTRY_OK)
            result=rc_registry_write_complete_extents(r,&e->opaque,1);
        if (result!=RC_REGISTRY_OK) result=rc_registry_write_guard_accumulate(r,guard);
    }
    if (result!=RC_REGISTRY_OK) rc_registry_fail_closed(r);
}
static void dispose(ResourceCommandEffects *e,int completed,int may_have_executed) {
    const int unhanded = e->dispositions && e->dispositions->known_unhanded(e->dispositions->part);
    // The scene may have handed an earlier prefix. Only an actual part witness can cancel this body.
    if (unhanded) completed = may_have_executed = 0;
    resource_effects_recorded(e,0);
    settle_opaque(e,completed,may_have_executed);
    settle_normalized(e,completed,may_have_executed);
    EffectClaim *c=e->claims;
    while(c) {
        EffectClaim *next=c->next;
        int recorded=0;uint32_t n=0,m=0;
        for (EffectActual *a=c->actual;a;a=a->next) {
            recorded |= a->recorded;
            n += a->recorded && !a->visible && !a->owned;
            m += a->recorded && !a->owned;
        }
        if(c->shared) {
            if(!c->position_resolved)
                rc_registry_image_producer_resolve(rc_registry_runtime(),c->shared->ticket,e->position,NULL,!e->emitted && !recorded);
            for (EffectActual *a=c->actual;a;a=a->next) {
                if (!a->recorded || !a->owned) continue;
                RcRegistryImageBinding image={a->image,a->hold};
                if (completed) rc_registry_image_producer_native_completed(rc_registry_runtime(),
                    c->shared->ticket,e->position,image);
                else if (rc_registry_image_producer_projection_abandon(rc_registry_runtime(),
                            c->shared->ticket,e->position,image,may_have_executed) &&
                         may_have_executed) a->owned=0;
            }
            shared_image_release(c->shared,c->actual,e->position,completed,may_have_executed);
        }
        if (c->ticket.id && !c->descriptor.mapping) {
            if (!recorded) rc_registry_write_cancel(rc_registry_runtime(),c->ticket);
            else if (!completed) rc_registry_write_abandon(rc_registry_runtime(),c->ticket,may_have_executed);
            else if (rc_registry_write_submit(rc_registry_runtime(),c->ticket,e->position.scene)==RC_REGISTRY_OK) {
                RcSourceRange *parts=(m||n)?malloc(((size_t)m+n)*sizeof *parts):NULL;
                if ((m||n) && !parts) rt_fatal("resource effects: completion allocation failed");
                RcSourceRange *written=parts,*uncertain=parts?parts+m:NULL;
                uint32_t i=0,j=0;
                for (EffectActual *a=c->actual;a;a=a->next)if(a->recorded && !a->owned) {
                    written[i++]=a->interval.range;
                    if(!a->visible)uncertain[j++]=a->interval.range;
                }
                if(rc_registry_write_complete_effects(rc_registry_runtime(),c->ticket,written,m,uncertain,n)!=RC_REGISTRY_OK)
                    rc_registry_write_complete(rc_registry_runtime(),c->ticket,0);
                free(parts);
            } else rc_registry_write_abandon(rc_registry_runtime(),c->ticket,1);
        }
        while(c->actual) {EffectActual *a=c->actual;c->actual=a->next;if(a!=&c->inline_actual)free(a);}
        free(c);c=next;
    }
    while (e->dispositions) {
        EffectPartDisposition *disposition = e->dispositions;
        e->dispositions = disposition->next;
        disposition->release(disposition->part, disposition->owner, unhanded);
        free(disposition);
    }
    free(e->opaque_ranges);free(e);
}
static void completed(void *arg) {dispose(arg,1,0);}
static void canceled(void *arg,int may_have_executed) {dispose(arg,0,may_have_executed);}
static int claim_needs_completion(const EffectClaim *claim,RcRegistryPosition position) {
    for (const EffectActual *a=claim->actual;a;a=a->next)
        if (a->recorded && (!a->owned || (claim->shared && rc_registry_image_producer_projection_pending(
                rc_registry_runtime(),claim->shared->ticket,position,(RcRegistryImageBinding){a->image,a->hold})))) return 1;
    return 0;
}
static void cancel_normalized_idle(ResourceCommandEffects *e,uint32_t count) {
    if (!count) return;
    RcRegistryWrite local[8];
    RcRegistryWrite *tickets=count<=8?local:malloc((size_t)count*sizeof *tickets);
    if (!tickets) return; // Leaving the declarations through completion remains conservative.
    uint32_t i=0;
    for (EffectClaim *c=e->claims;c;c=c->next)
        if (c->descriptor.mapping && c->ticket.id && !claim_needs_completion(c,e->position)) tickets[i++]=c->ticket;
    if (rc_registry_write_cancel_extents(rc_registry_runtime(),tickets,count)==RC_REGISTRY_OK)
        for (EffectClaim *c=e->claims;c;c=c->next)
            if (c->descriptor.mapping && c->ticket.id && !claim_needs_completion(c,e->position)) c->ticket=(RcRegistryWrite){0};
    if (tickets!=local) free(tickets);
}
void resource_effects_release(ResourceCommandEffects *e) {
    if (!e) return;
    while (e->preservations) {
        EffectBufferPreservation *owner = e->preservations;
        e->preservations = owner->next;
        owner->release(owner->receipt);
        free(owner);
    }
    resource_effects_recorded(e,0);
    int recorded=e->dispositions || (e->opaque.ticket.id && e->emitted);
    int opaque_recorded=e->emitted;
    uint32_t idle=0;
    for(EffectClaim *c=e->claims;c;c=c->next) {
        int needs_completion=claim_needs_completion(c,e->position);
        opaque_recorded |= c->opaque && needs_completion;
        // An image's persistent ownership hold was installed before the recording callback marked it owned.
        // Retiring the temporary declaration now avoids keeping duplicate per-draw reservations until the fence.
        if(!needs_completion && c->ticket.id) {
            if(c->descriptor.mapping) idle++;
            else {rc_registry_write_cancel(rc_registry_runtime(),c->ticket);c->ticket=(RcRegistryWrite){0};}
        }
        recorded |= needs_completion;
    }
    if(recorded) {
        // An issued typed write can keep the command alive after its opaque path was skipped.
        if (e->opaque.ticket.id && !opaque_recorded) {
            settle_opaque(e,0,0);
            e->opaque.ticket=(RcRegistryWrite){0};
        }
        cancel_normalized_idle(e,idle);
        vk_completion_add_owned(e->position.scene,completed,canceled,e);
    }
    else dispose(e,0,0);
}

void resource_effects_report(uint64_t frame) {
    plat_mutex_lock(&images_lock);
    rt_log("resource command effects: frame %llu image_declarations %llu unique_image_intervals %llu; "
           "mapped declarations %llu unique %llu; CPU results transferred %llu",
        (unsigned long long)frame,(unsigned long long)image_requests,(unsigned long long)image_unique,
        (unsigned long long)mapped_image_requests,(unsigned long long)mapped_image_unique,
        (unsigned long long)__atomic_exchange_n(&cpu_results_transferred, 0, __ATOMIC_RELAXED));
    image_requests=image_unique=mapped_image_requests=mapped_image_unique=0;
    plat_mutex_unlock(&images_lock);
}
