#include "resource_core_registry.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#ifndef RC_REGISTRY_SINGLE_THREADED
#include "platform/platform.h"
#endif

// Two interval treaps index registrations and writer authority independently. Generated identities determine
// priorities, not guest addresses. Fixed storage keeps records stable and makes capacity refusal explicit.
typedef struct {
    RcSourceRange range;
    uint64_t order, priority, max_end;
    uint32_t left, right;
} Interval;
typedef struct {
    Interval interval;
    RcSourceRange *source_spans;
    uint64_t id, key, object, interpretation, revision, scope, end;
    uint64_t content, representation, payload;
    uint64_t content_revision, ordered_revision, version, policy, published_revision, publication_scope;
    RcRegistryPosition publication;
    uint32_t older, newer, refs, owners, free_next, retired_prev, retired_next;
    uint32_t source_span_count;
    uint8_t used, source, current, finalized, retired_queued, provider_kind;
} Record;
enum { WRITE_PENDING = 1, WRITE_SUBMITTED, WRITE_FOREIGN, WRITE_HOLD };
enum { BUFFER_PHASE_BOUND = 1, BUFFER_PHASE_MAYBE, BUFFER_PHASE_DONE };
typedef struct {
    RcRegistryPosition position;
    RcRegistryImageBinding image, prefix_image;
    uint32_t refs, unresolved, active;
    uint8_t unknown, prefix_bad, prefix_active, projected, exact_effect, preserves_native, preserved_completed;
    uint8_t preserves_projection, exported_completed;
} ImageProducer;
typedef struct { uint64_t begin, end, producer_order; } ForeignSpan;
typedef struct {
    uint32_t refs, count;
    ForeignSpan spans[];
} ForeignCoverage;
typedef struct { uint64_t begin, end; } NativeImageSpan;
typedef struct NativeImageProjection {
    struct NativeImageProjection *next;
    RcRegistryPosition output;
    uint32_t count, capacity;
    uint8_t confirmed, physically_bound, completed;
    NativeImageSpan spans[];
} NativeImageProjection;
enum { NATIVE_BOUND_RECORDED = 1, NATIVE_BOUND_COMPLETED };
typedef struct {
    RcRegistryBufferPayload *owner;
    uint64_t address, mapping;
} BufferPayload;
typedef struct {
    Interval interval;
    uint64_t id, submission, producer_order, foreign_order;
    RcRegistryPosition position, visibility_position, visibility_latest;
    uint32_t free_next, guard_prev, guard_next;
    uint8_t state, kind, unordered, kinds;
    uint64_t image;
    uint64_t image_address, image_mapping;
    RcRegistryPosition native_output;
    int native_conflict;
    NativeImageProjection *native_projection;
    RcRegistryPosition native_unprojected, native_unprojected_latest, native_unprojected_opaque, native_retire_uncertain;
    RcRegistryPosition native_unprojected_uncertain, native_preserved_projection, native_resident_only;
    uint8_t native_projected, native_unordered, native_bound_whole, native_ordered_effect;
    ForeignCoverage *foreign_coverage, *image_possible;
    uint64_t foreign_address;
    uint8_t image_possible_known;
    ImageProducer *image_positions;
    uint32_t image_count, image_capacity, image_dirty;
    uint8_t image_incomplete, image_forgotten, guard, positioned_image_effect;
    uint8_t buffer_phase, native_buffer_conflict;
    uint64_t phase_address, phase_mapping;
    RcRegistryBufferLocation native_buffer;
    uint64_t native_buffer_address, native_buffer_mapping;
    uint8_t native_buffer_compacted;
    BufferPayload native_payload;
    uint8_t native_payload_conflict;
    RcRegistryBufferLocation native_access;
    RcRegistryPosition native_access_output;
    RcSourceRange native_access_range;
    uint64_t native_access_address;
    uint64_t native_access_mapping;
    uint8_t native_access_recorded, native_access_conflict;
    uint32_t diagnostic_flags;
    uintptr_t diagnostic_origin;
    RcRegistryPosition diagnostic_position;
    uint8_t diagnostic_guard_reason;
} Writer;
typedef struct {
    const RcSourceRange *coverage;
    uint64_t id, before_copy;
    uint32_t count, next, prev;
    uint8_t active, history_lost;
} GpuCapture;
typedef struct {
    RcSourceRange range;
    RcRegistryPosition producer, latest;
    RcRegistryBufferLocation native_buffer;
    BufferPayload payload;
    uint64_t payload_sequence;
    uint32_t payload_prev, payload_next;
} GpuVisibility;
typedef struct { uint64_t id, horizon; uint32_t free_next; } Scope;
typedef struct { uint64_t id; RcRegistryHandle record; uint32_t free_next; uint8_t source; } Capture;
typedef struct { void *items; size_t stride; uint32_t root; } Tree;

struct RcRegistry {
#ifndef RC_REGISTRY_SINGLE_THREADED
    PlatMutex lock;
    PlatCond gpu_capture_released;
#endif
    RcRegistryConfig capacity;
    Record *records;
    Writer *writers;
    Scope *scopes;
    Capture *captures;
    RcRegistryChange *changes;
    uint64_t change_sequence;
    RcRegistryScheduleChange *schedule_changes;
    uint64_t schedule_sequence;
    GpuCapture *gpu_captures;
    GpuVisibility *gpu_visibility;
    uint64_t gpu_visible_sequence, gpu_capture_release_epoch;
    uint32_t gpu_payload_head;
    uint64_t produced_scene, produced_begin;
    RcRegistryPosition produced_lost;
    int produced_lost_unknown;
    uint32_t active_gpu_captures;
    uint32_t native_image_outputs;
    uint32_t *keys, *scratch;
    uint32_t key_capacity, free_record, free_writer, free_scope, free_capture, retired_head, retired_scan;
    uint32_t guards, unresolved_guard;
    uint32_t source_span_count;
    uint32_t foreign_span_count;
    uint32_t native_projection_spans;
    uint64_t clock, oldest_scope, newest_scope, guard, authority_epoch, ordered_epoch;
    RcRegistryPosition declared, handed;
    Tree registrations, authority;
    RcRegistryStats stats;
    RcRegistryDiagnostics diagnostics;
    uint64_t diagnostic_sequence;
#ifndef RC_REGISTRY_SINGLE_THREADED
    int writer_census_logged;
#endif
};

static void buffer_payload_drop(BufferPayload *payload) {
    RcRegistryBufferPayload *owner = payload->owner;
    if (!owner) return;
    *payload = (BufferPayload){0};
    owner->release(owner);
}

static void gpu_visibility_payload_drop(RcRegistry *r, uint32_t slot) {
    GpuVisibility *entry = &r->gpu_visibility[slot - 1];
    if (!entry->payload.owner) return;
    if (entry->payload_prev) r->gpu_visibility[entry->payload_prev - 1].payload_next = entry->payload_next;
    else r->gpu_payload_head = entry->payload_next;
    if (entry->payload_next) r->gpu_visibility[entry->payload_next - 1].payload_prev = entry->payload_prev;
    entry->payload_prev = entry->payload_next = 0;
    buffer_payload_drop(&entry->payload);
}

#ifndef RC_REGISTRY_SINGLE_THREADED
static RcRegistry *runtime_registry;
static uint64_t native_final, native_part, native_uncertain;
void rt_log(const char *, ...) __attribute__((format(printf, 1, 2)));
static void writer_census(RcRegistry *r, const char *reason, uint32_t needed, uintptr_t origin);
#else
static void writer_census(RcRegistry *r, const char *reason, uint32_t needed, uintptr_t origin) {
    (void)r; (void)reason; (void)needed; (void)origin;
}
#endif

void rc_registry_runtime_native_handoff(uint64_t final, uint64_t part, uint64_t uncertain) {
#ifndef RC_REGISTRY_SINGLE_THREADED
    __atomic_store_n(&native_final, final, __ATOMIC_RELEASE);
    __atomic_store_n(&native_part, part, __ATOMIC_RELEASE);
    __atomic_store_n(&native_uncertain, uncertain, __ATOMIC_RELEASE);
#else
    (void)final; (void)part; (void)uncertain;
#endif
}

static void lock_registry(RcRegistry *r) {
#ifndef RC_REGISTRY_SINGLE_THREADED
    plat_mutex_lock(&r->lock);
#else
    (void)r;
#endif
}
static void unlock_registry(RcRegistry *r) {
#ifndef RC_REGISTRY_SINGLE_THREADED
    plat_mutex_unlock(&r->lock);
#else
    (void)r;
#endif
}
static uint64_t mixed(uint64_t x) {
    x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
static void poison(RcRegistry *r) {
    writer_census(r, "fail-closed", 0, 0);
    r->stats.fail_closed = 1;
    __atomic_store_n(&r->authority_epoch, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&r->ordered_epoch, 0, __ATOMIC_RELEASE);
}
static void epoch_advance(RcRegistry *r, uint64_t *epoch) {
    uint64_t old = __atomic_load_n(epoch, __ATOMIC_RELAXED);
    if (!old) return;
    if (old == UINT64_MAX) poison(r);
    else __atomic_store_n(epoch, old + 1, __ATOMIC_RELEASE);
}
static void authority_changed(RcRegistry *r) {
    epoch_advance(r, &r->authority_epoch);
    epoch_advance(r, &r->ordered_epoch);
}
static uint64_t identity(RcRegistry *r) {
    if (r->clock == UINT64_MAX) { poison(r); return 0; }
    return ++r->clock;
}
static Interval *node(Tree *tree, uint32_t i) {
    return i ? (Interval *)((char *)tree->items + (size_t)i * tree->stride) : NULL;
}
static void repair(Tree *tree, uint32_t i) {
    Interval *n = node(tree, i);
    n->max_end = n->range.offset + n->range.size;
    Interval *a = node(tree, n->left), *b = node(tree, n->right);
    if (a && a->max_end > n->max_end) n->max_end = a->max_end;
    if (b && b->max_end > n->max_end) n->max_end = b->max_end;
}
static int before(const Interval *a, const Interval *b) {
    if (a->range.backing != b->range.backing) return a->range.backing < b->range.backing;
    if (a->range.offset != b->range.offset) return a->range.offset < b->range.offset;
    return a->order < b->order;
}
static uint32_t rotate_left(Tree *tree, uint32_t i) {
    Interval *n = node(tree, i);
    uint32_t right = n->right;
    Interval *child = node(tree, right);
    n->right = child->left; child->left = i;
    repair(tree, i); repair(tree, right);
    return right;
}
static uint32_t rotate_right(Tree *tree, uint32_t i) {
    Interval *n = node(tree, i);
    uint32_t left = n->left;
    Interval *child = node(tree, left);
    n->left = child->right; child->right = i;
    repair(tree, i); repair(tree, left);
    return left;
}
static uint32_t insert_node(Tree *tree, uint32_t root, uint32_t i) {
    if (!root) { repair(tree, i); return i; }
    Interval *n = node(tree, root), *add = node(tree, i);
    if (before(add, n)) {
        n->left = insert_node(tree, n->left, i);
        if (node(tree, n->left)->priority < n->priority) root = rotate_right(tree, root);
    } else {
        n->right = insert_node(tree, n->right, i);
        if (node(tree, n->right)->priority < n->priority) root = rotate_left(tree, root);
    }
    repair(tree, root);
    return root;
}
static uint32_t erase_node(Tree *tree, uint32_t root, uint32_t i) {
    if (!root) return 0;
    Interval *n = node(tree, root), *target = node(tree, i);
    if (root == i) {
        if (!n->left) return n->right;
        if (!n->right) return n->left;
        if (node(tree, n->left)->priority < node(tree, n->right)->priority) {
            root = rotate_right(tree, root);
            node(tree, root)->right = erase_node(tree, node(tree, root)->right, i);
        } else {
            root = rotate_left(tree, root);
            node(tree, root)->left = erase_node(tree, node(tree, root)->left, i);
        }
    } else if (before(target, n)) n->left = erase_node(tree, n->left, i);
    else n->right = erase_node(tree, n->right, i);
    repair(tree, root);
    return root;
}
typedef int (*IntervalVisit)(RcRegistry *, uint32_t, void *);

static uint32_t foreign_span_first(const ForeignCoverage *coverage, uint64_t begin) {
    uint32_t lo = 0, hi = coverage->count;
    while (lo < hi) {
        uint32_t at = lo + (hi - lo) / 2;
        if (coverage->spans[at].end <= begin) lo = at + 1;
        else hi = at;
    }
    return lo;
}

static ForeignSpan writer_foreign_span(const Writer *writer, RcSourceRange range) {
    RcSourceRange whole = writer->interval.range;
    uint64_t begin = range.offset > whole.offset ? range.offset : whole.offset;
    uint64_t end = range.offset + range.size;
    uint64_t whole_end = whole.offset + whole.size;
    if (end > whole_end) end = whole_end;
    return (ForeignSpan){.begin = writer->foreign_address + (begin - whole.offset),
                         .end = writer->foreign_address + (end - whole.offset)};
}

static int writer_overlaps(const Writer *writer, RcSourceRange range) {
    if (!rc_range_overlap(writer->interval.range, range)) return 0;
    const ForeignCoverage *coverage = writer->foreign_coverage;
    if (!coverage) return 1;
    ForeignSpan span = writer_foreign_span(writer, range);
    uint32_t first = foreign_span_first(coverage, span.begin);
    return first < coverage->count && coverage->spans[first].begin < span.end;
}

static int overlaps(RcRegistry *r, Tree *tree, uint32_t root, RcSourceRange range,
                    IntervalVisit visit, void *data) {
    if (!root) return 0;
    Interval *n = node(tree, root);
    r->stats.overlap_visits++;
    if (n->range.backing < range.backing) return overlaps(r, tree, n->right, range, visit, data);
    if (n->range.backing > range.backing) return overlaps(r, tree, n->left, range, visit, data);
    Interval *left = node(tree, n->left);
    if (left && left->max_end > range.offset && overlaps(r, tree, n->left, range, visit, data)) return 1;
    if ((tree == &r->authority ? writer_overlaps(&r->writers[root], range) : rc_range_overlap(n->range, range)) &&
        visit(r, root, data)) return 1;
    if (n->range.offset < range.offset + range.size)
        return overlaps(r, tree, n->right, range, visit, data);
    return 0;
}
static int authority_overlaps(RcRegistry *r, RcSourceRange range, IntervalVisit visit, void *data) {
    if (range.backing != RC_GUEST_COMPATIBILITY_BACKING)
        for (uint32_t i = r->guards; i; i = r->writers[i].guard_next)
            if (visit(r, i, data)) return 1;
    return overlaps(r, &r->authority, r->authority.root, range, visit, data);
}

typedef struct { const Record *source; IntervalVisit visit; void *data; } SourceAuthorityVisit;

static int source_authority_visit(RcRegistry *r, uint32_t i, void *data) {
    const SourceAuthorityVisit *query = data;
    const Writer *writer = &r->writers[i];
    if (writer->guard) return query->visit(r, i, query->data);
    const RcSourceRange *spans = query->source->source_spans;
    RcSourceRange range = writer->interval.range;
    uint32_t lo = 0, hi = query->source->source_span_count;
    while (lo < hi) {
        uint32_t at = lo + (hi - lo) / 2;
        if (spans[at].offset + spans[at].size <= range.offset) lo = at + 1;
        else hi = at;
    }
    for (; lo < query->source->source_span_count && rc_range_overlap(range, spans[lo]); ++lo)
        if (writer_overlaps(writer, spans[lo])) return query->visit(r, i, query->data);
    return 0;
}

static int source_authority_overlaps(RcRegistry *r, const Record *e, IntervalVisit visit, void *data) {
    if (!e->source_span_count) return authority_overlaps(r, e->interval.range, visit, data);
    SourceAuthorityVisit query = {e, visit, data};
    return authority_overlaps(r, e->interval.range, source_authority_visit, &query);
}
static void diagnostic_sample(RcRegistryDiagnosticSample *s, uint32_t index, const Writer *w,
                              RcSourceRange requested, RcRegistryPosition read_position,
                              RcRegistryImageBinding selected, const ImageProducer *producer, uint32_t prepared) {
    *s = (RcRegistryDiagnosticSample){.requested = requested,
        .read_position = read_position, .selected = selected, .prepared_actuals = prepared};
    if (w) {
        s->writer = (RcRegistryWrite){w->id, index}; s->range = w->interval.range;
        s->submission = w->submission; s->position = w->position; s->writer_image = w->image;
        s->state = w->state; s->kind = w->kind; s->producer_count = w->image_count;
        s->flags = w->diagnostic_flags | (w->image_forgotten ? RC_REGISTRY_DIAG_FORGOTTEN : 0) |
            (w->image_incomplete ? RC_REGISTRY_DIAG_INCOMPLETE : 0) | (w->unordered ? RC_REGISTRY_DIAG_UNORDERED : 0);
    }
    if (producer) {
        s->producer_position = producer->position; s->producer_image = producer->image;
        s->producer_refs = producer->refs; s->producer_unresolved = producer->unresolved;
        s->producer_active = producer->active; s->producer_unknown = producer->unknown;
        s->prefix_bad = producer->prefix_bad;
    }
}
static void diagnose(RcRegistry *r, RcRegistryDiagnosticReason reason, uint32_t index,
                     RcSourceRange requested, RcRegistryPosition read_position,
                     RcRegistryImageBinding selected, const ImageProducer *producer, uint32_t prepared) {
    RcRegistryDiagnostics *d = &r->diagnostics;
    d->counts[reason]++;
    uint32_t n = d->sample_count[reason];
    if (n == RC_REGISTRY_DIAGNOSTIC_SAMPLES) return;
    const Writer *w = index ? &r->writers[index] : NULL;
    uint64_t id = w ? w->id : 0;
    for (uint32_t j = 0; j < n; ++j)
        if (d->samples[reason][j].writer.id == id && d->samples[reason][j].selected.image == selected.image) return;
    RcRegistryDiagnosticSample *s = &d->samples[reason][n];
    diagnostic_sample(s, index, w, requested, read_position, selected, producer, prepared);
    s->sequence = ++r->diagnostic_sequence;
    d->sample_count[reason]++;
}
typedef struct { RcSourceRange range; RcRegistryPosition position; } SourceDiagnostic;
static int observation_blocker(RcRegistry *r, uint32_t i, void *data) {
    Writer *w = &r->writers[i];
    if (w->state == WRITE_PENDING) r->stats.blocked_pending++;
    else if (w->state == WRITE_SUBMITTED) r->stats.blocked_submitted++;
    else if (w->state == WRITE_FOREIGN) r->stats.blocked_foreign++;
    else r->stats.blocked_held++;
    if (w->kind == RC_REGISTRY_IMAGE) r->stats.blocked_image++;
    else if (w->kind == RC_REGISTRY_BUFFER) r->stats.blocked_buffer++;
    else r->stats.blocked_unknown++;
    SourceDiagnostic *source = data;
    diagnose(r, (RcRegistryDiagnosticReason)((w->state - WRITE_PENDING) * 3 + w->kind), i,
        source ? source->range : (RcSourceRange){0}, source ? source->position : (RcRegistryPosition){0},
        (RcRegistryImageBinding){0}, NULL, 0);
    return 1;
}
static int pending_writer(RcRegistry *r, uint32_t i, void *data) {
    (void)data; return r->writers[i].state != WRITE_FOREIGN;
}
typedef struct {
    int flags;
    RcRegistryWrite writer;
    RcRegistryPosition position;
    uint32_t kind;
} Invalidation;
static int invalidate_record(RcRegistry *r, uint32_t i, void *data) {
    Record *e = &r->records[i];
    int flags = data ? *(const int *)data : 0;
    if (flags & 1) {
        if (e->content_revision == UINT64_MAX) poison(r);
        else e->content_revision++;
    }
    if (e->revision == UINT64_MAX) poison(r);
    else e->revision++;
    if (!(flags & 2)) {
        if (e->ordered_revision == UINT64_MAX) poison(r);
        else e->ordered_revision++;
    }
    if (e->source && !(flags & 2)) {
        if (r->change_sequence == UINT64_MAX) poison(r);
        else {
            r->changes[r->change_sequence % r->capacity.records] = (RcRegistryChange){
                .handle = {e->id, i}, .object_serial = e->object, .revision = e->ordered_revision,
                .range = e->interval.range, .association = e->interpretation};
            r->change_sequence++;
        }
    }
    if (e->source && (flags & 4)) {
        const Invalidation *change = data;
        if (r->schedule_sequence == UINT64_MAX) poison(r);
        else {
            r->schedule_changes[r->schedule_sequence % r->capacity.records] = (RcRegistryScheduleChange){
                .handle = {e->id, i}, .object_serial = e->object, .topology_revision = e->revision,
                .range = e->interval.range, .writer = change->writer, .position = change->position, .kind = change->kind,
                .association = e->interpretation};
            r->schedule_sequence++;
        }
    }
    r->stats.invalidations++;
    return 0;
}
static void invalidate_canonical(RcRegistry *r, uint32_t at, void *data) {
    if (!at) return;
    const Interval *entry = &r->records[at].interval;
    invalidate_canonical(r, entry->left, data);
    if (entry->range.backing != RC_GUEST_COMPATIBILITY_BACKING) invalidate_record(r, at, data);
    invalidate_canonical(r, entry->right, data);
}
static void invalidate_scope(RcRegistry *r, RcSourceRange range, void *data) {
    if (range.backing) overlaps(r, &r->registrations, r->registrations.root, range, invalidate_record, data);
    else invalidate_canonical(r, r->registrations.root, data);
}
static void invalidate_content(RcRegistry *r, RcSourceRange range, int content) {
    authority_changed(r);
    invalidate_scope(r, range, &content);
}
static void invalidate(RcRegistry *r, RcSourceRange range) { invalidate_content(r, range, 1); }
// Positioned additions cannot enter an already sealed prefix; removals only relax authority. Legacy observers
// have no prefix and still lose permission, but existing finite ordered proofs need no source notification.
static void invalidate_topology(RcRegistry *r, RcSourceRange range) {
    int flags = 2;
    epoch_advance(r, &r->authority_epoch);
    invalidate_scope(r, range, &flags);
}
static void invalidate_schedule(RcRegistry *r, RcSourceRange range, RcRegistryWrite writer,
                                RcRegistryPosition position, uint32_t kind) {
    Invalidation change = {.flags = kind == RC_REGISTRY_SCHEDULE_EXPAND ? 5 : 6,
        .writer = writer, .position = position, .kind = kind};
    if (kind == RC_REGISTRY_SCHEDULE_EXPAND) authority_changed(r);
    else epoch_advance(r, &r->authority_epoch);
    invalidate_scope(r, range, &change);
}
static uint64_t key_hash(int source, uint64_t key, RcSourceRange range, uint64_t association) {
    uint64_t h = mixed(key ^ (source ? UINT64_C(0x80e8cf71336d94c5) : UINT64_C(0x162cfab8e1f53a21)));
    if (source) h ^= mixed(range.backing) ^ mixed(range.offset) ^ mixed(range.size) ^ mixed(association);
    return h;
}
static uint32_t key_slot(RcRegistry *r, int source, uint64_t key, RcSourceRange range, uint64_t association, int *found) {
    uint32_t at = (uint32_t)key_hash(source, key, range, association) & (r->key_capacity - 1), tomb = UINT32_MAX;
    *found = 0;
    for (uint32_t n = 0; n < r->key_capacity; n++, at = (at + 1) & (r->key_capacity - 1)) {
        uint32_t i = r->keys[at];
        if (!i) return tomb != UINT32_MAX ? tomb : at;
        if (i == UINT32_MAX) { if (tomb == UINT32_MAX) tomb = at; continue; }
        Record *e = &r->records[i];
        if (e->source == source && e->key == key &&
            (!source || (e->interpretation==association && rc_range_equal(e->interval.range, range)))) {
            *found = 1; return at;
        }
    }
    return tomb;
}
// Backward-shift deletion keeps lookup cost dependent on occupied keys, not all registrations ever visited.
static void key_remove(RcRegistry *r, uint32_t hole) {
    uint32_t mask = r->key_capacity - 1;
    for (uint32_t at = (hole + 1) & mask; r->keys[at]; at = (at + 1) & mask) {
        Record *e = &r->records[r->keys[at]];
        uint32_t home = (uint32_t)key_hash(e->source, e->key, e->interval.range, e->interpretation) & mask;
        if (((at - home) & mask) < ((at - hole) & mask)) continue;
        r->keys[hole] = r->keys[at]; hole = at;
    }
    r->keys[hole] = 0;
}
static Record *record_of(RcRegistry *r, RcRegistryHandle handle) {
    if (!handle.id || !handle.index || handle.index > r->capacity.records) return NULL;
    Record *e = &r->records[handle.index];
    return e->used && e->id == handle.id ? e : NULL;
}
static Writer *writer_of(RcRegistry *r, RcRegistryWrite ticket) {
    if (!ticket.id || !ticket.index || ticket.index > r->capacity.writers) return NULL;
    Writer *w = &r->writers[ticket.index];
    return w->state && w->id == ticket.id ? w : NULL;
}
static int image_binding_equal(RcRegistryImageBinding a, RcRegistryImageBinding b) {
    return a.image == b.image && a.hold.id == b.hold.id && a.hold.index == b.hold.index;
}
static int image_binding_current_locked(RcRegistry *r, RcRegistryImageBinding image, RcSourceRange range) {
    Writer *w = writer_of(r, image.hold);
    return image.image && w && w->state == WRITE_HOLD && w->kind == RC_REGISTRY_IMAGE && !w->image_forgotten &&
        w->image == image.image && w->interval.range.backing == range.backing &&
        w->interval.range.offset <= range.offset &&
        range.size <= w->interval.range.size && range.offset - w->interval.range.offset <= w->interval.range.size - range.size;
}
static void native_image_output_clear(RcRegistry *r, Writer *w) {
    if (!rc_registry_position_valid(w->native_output)) return;
    r->native_image_outputs--;
    w->native_output = (RcRegistryPosition){0};
    w->native_bound_whole = 0;
}

static uint32_t image_span_room(const RcRegistry *r) {
    uint32_t limit = r->capacity.records * 32;
    if (limit > (1u << 20)) limit = 1u << 20;
    return limit - r->foreign_span_count - r->native_projection_spans;
}

static void native_projection_remove(RcRegistry *r, NativeImageProjection **at) {
    NativeImageProjection *projection = *at;
    *at = projection->next;
    r->native_projection_spans -= projection->capacity;
    free(projection);
}

static void native_projection_clear(RcRegistry *r, Writer *w) {
    while (w->native_projection) native_projection_remove(r, &w->native_projection);
}

static void native_projection_unproved(Writer *w, RcRegistryPosition output, int independent, int preserves_native) {
    if (!rc_registry_position_valid(output)) w->native_unordered = 1;
    else {
        if (!rc_registry_position_valid(w->native_unprojected) ||
            rc_registry_position_compare(output, w->native_unprojected) < 0) w->native_unprojected = output;
        if (rc_registry_position_compare(output, w->native_unprojected_latest) > 0) w->native_unprojected_latest = output;
        if (!preserves_native && rc_registry_position_compare(output, w->native_unprojected_uncertain) > 0)
            w->native_unprojected_uncertain = output;
        if (!independent && (!rc_registry_position_valid(w->native_unprojected_opaque) ||
            rc_registry_position_compare(output, w->native_unprojected_opaque) < 0)) w->native_unprojected_opaque = output;
    }
}

static uint32_t native_projection_first(const NativeImageProjection *projection, uint64_t begin) {
    uint32_t lo = 0, hi = projection->count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (projection->spans[mid].end <= begin) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static int native_projection_trim(RcRegistry *r, Writer *w, NativeImageProjection **at, RcSourceRange range) {
    NativeImageProjection *projection = *at;
    uint64_t end = range.offset + range.size;
    uint32_t first = native_projection_first(projection, range.offset);
    if (first == projection->count || projection->spans[first].begin >= end) return 1;
    NativeImageSpan first_span = projection->spans[first];
    if (first_span.begin < range.offset && end < first_span.end) {
        if (projection->count == projection->capacity) {
            NativeImageProjection *grown = NULL;
            if (image_span_room(r)) grown = realloc(projection, sizeof *projection +
                ((size_t)projection->capacity + 1) * sizeof *projection->spans);
            if (!grown) {
                // An older contributor cannot take over bytes lost to a metadata-capacity refusal.
                native_projection_unproved(w, projection->output, 0, 0);
                native_projection_remove(r, at);
                return 0;
            }
            *at = projection = grown;
            ++projection->capacity;
            ++r->native_projection_spans;
        }
        memmove(projection->spans + first + 2, projection->spans + first + 1,
                (projection->count - first - 1) * sizeof *projection->spans);
        projection->spans[first] = (NativeImageSpan){first_span.begin, range.offset};
        projection->spans[first + 1] = (NativeImageSpan){end, first_span.end};
        ++projection->count;
        return 1;
    }
    uint32_t keep = first;
    if (first_span.begin < range.offset)
        projection->spans[keep++] = (NativeImageSpan){first_span.begin, range.offset};
    uint32_t i = first;
    while (i < projection->count && projection->spans[i].begin < end) {
        const NativeImageSpan span = projection->spans[i++];
        if (span.end > end) projection->spans[keep++] = (NativeImageSpan){end, span.end};
    }
    memmove(projection->spans + keep, projection->spans + i, (projection->count - i) * sizeof *projection->spans);
    projection->count = keep + projection->count - i;
    if (projection->count) return 1;
    native_projection_remove(r, at);
    return 0;
}

typedef struct {
    RcSourceRange range;
    RcRegistryPosition through;
    const Writer *except;
    int uncertain;
} NativeProjectionChange;

static int native_projection_invalidate_visit(RcRegistry *r, uint32_t i, void *data) {
    const NativeProjectionChange *change = data;
    Writer *w = &r->writers[i];
    if (w == change->except) return 0;
    NativeImageProjection **at = &w->native_projection;
    while (*at) {
        NativeImageProjection *projection = *at;
        if (!rc_registry_position_valid(change->through) ||
            rc_registry_position_compare(projection->output, change->through) <= 0) {
            if (change->uncertain && rc_registry_position_compare(projection->output, w->native_retire_uncertain) > 0) {
                uint32_t first = native_projection_first(projection, change->range.offset);
                if (first < projection->count && projection->spans[first].begin < change->range.offset + change->range.size)
                    w->native_retire_uncertain = projection->output;
            }
            if (!native_projection_trim(r, w, at, change->range)) continue;
        }
        at = &(*at)->next;
    }
    return 0;
}

static void native_projection_invalidate(RcRegistry *r, RcSourceRange range, RcRegistryPosition through,
                                         const Writer *except, int uncertain) {
    if (r->native_projection_spans && range.backing != RC_GUEST_COMPATIBILITY_BACKING) {
        NativeProjectionChange change = {range, through, except, uncertain};
        overlaps(r, &r->authority, r->authority.root, range, native_projection_invalidate_visit, &change);
    }
}

static void native_projection_retire(RcRegistry *r, Writer *w) {
    for (const NativeImageProjection *p = w->native_projection; p; p = p->next) {
        if (!p->confirmed || rc_registry_position_compare(p->output, r->handed) > 0) continue;
        for (uint32_t i = 0; i < p->count; ++i) {
            NativeImageSpan span = p->spans[i];
            RcSourceRange written = {w->interval.range.backing, span.begin, span.end - span.begin};
            native_projection_invalidate(r, written, p->output, w, !p->completed);
        }
    }
    native_projection_clear(r, w);
}

static void native_projection_drop(RcRegistry *r, RcRegistryImageBinding binding, RcRegistryPosition output) {
    Writer *held = writer_of(r, binding.hold);
    if (!held || held->image != binding.image) return;
    for (NativeImageProjection **at = &held->native_projection; *at; at = &(*at)->next) {
        if (rc_registry_position_compare((*at)->output, output)) continue;
        native_projection_remove(r, at);
        return;
    }
}

static int native_projection_cancel_unconfirmed(RcRegistry *r, uint32_t i, void *data) {
    RcRegistryPosition output = *(const RcRegistryPosition *)data;
    Writer *w = &r->writers[i];
    NativeImageProjection **at = &w->native_projection;
    while (*at) {
        NativeImageProjection *projection = *at;
        if (!projection->confirmed && !rc_registry_position_compare(projection->output, output))
            native_projection_remove(r, at);
        else at = &projection->next;
    }
    return 0;
}

static int native_projection_confirm(RcRegistry *r, RcRegistryImageBinding binding, RcRegistryPosition output,
                                     int confirmed) {
    Writer *held = writer_of(r, binding.hold);
    if (!held || held->image != binding.image) return 0;
    for (NativeImageProjection *p = held->native_projection; p; p = p->next) {
        if (rc_registry_position_compare(p->output, output)) continue;
        p->confirmed = confirmed;
        return confirmed;
    }
    return 0;
}

static void image_position_incomplete(RcRegistry *r, Writer *w) {
    if (w->image_incomplete) return;
    w->image_incomplete = 1;
    for (uint32_t i = 0; i < w->image_count; ++i) {
        ImageProducer *p = &w->image_positions[i];
        if (p->active && p->image.image) p->projected = native_projection_confirm(r, p->image, p->position, 0);
    }
}

static void image_position_cancel_projection(RcRegistry *r, Writer *w, const ImageProducer *p) {
    if (p->image.image) native_projection_drop(r, p->image, p->position);
    if (p->unknown && r->native_projection_spans) {
        RcRegistryPosition output = p->position;
        overlaps(r, &r->authority, r->authority.root, w->interval.range,
                 native_projection_cancel_unconfirmed, &output);
    }
}

static void image_positions_clear(RcRegistry *r, Writer *w) {
    for (uint32_t i = 0; i < w->image_count; ++i) {
        const ImageProducer *p = &w->image_positions[i];
        if (!p->active) continue;
        if (rc_registry_position_compare(p->position, r->handed) > 0) {
            image_position_cancel_projection(r, w, p);
        } else if (!p->projected || p->unknown || w->image_incomplete) {
            // An issued image effect without exact projected coverage remains a possible overwrite even
            // after its command claim is consumed by the persistent native image hold.
            const Writer *preserved = p->preserved_completed && !p->unknown && !w->image_incomplete &&
                image_binding_current_locked(r, p->image, w->interval.range) ? writer_of(r, p->image.hold) : NULL;
            // A preserving export can reuse a still-selected prior span. Leaving other projections in the
            // sweep preserves their owner order and avoids trimming meaningful bytes in this image's padding.
            if (!preserved || !p->exported_completed)
                native_projection_invalidate(r, w->interval.range, p->position, preserved, 1);
        }
    }
    r->stats.image_positions -= w->image_count;
    free(w->image_positions); w->image_positions = NULL;
    w->image_count = w->image_capacity = w->image_dirty = 0;
    w->image_incomplete = 0;
}
static uint32_t image_position_lower(const Writer *w, RcRegistryPosition position) {
    uint32_t lo = 0, hi = w->image_count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (rc_registry_position_compare(w->image_positions[mid].position, position) < 0) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}
static void image_position_prefix(Writer *w, uint32_t end) {
    for (uint32_t i = w->image_dirty; i < end; ++i) {
        ImageProducer *p = &w->image_positions[i];
        p->prefix_bad = p->unknown || p->unresolved;
        p->prefix_active = p->active != 0;
        p->prefix_image = p->image;
        if (i) {
            const ImageProducer *before = &w->image_positions[i - 1];
            p->prefix_bad |= before->prefix_bad;
            if (before->prefix_active) {
                if (p->prefix_active && !image_binding_equal(before->prefix_image, p->prefix_image)) p->prefix_bad = 1;
                p->prefix_active = 1; p->prefix_image = before->prefix_image;
            }
        }
    }
    if (end > w->image_dirty) w->image_dirty = end;
}
RcRegistryResult rc_registry_image_hold_bind(RcRegistry *r, RcRegistryWrite hold, uint64_t image) {
    if (!r || !image) return RC_REGISTRY_INVALID;
    lock_registry(r); Writer *w = writer_of(r, hold);
    if (!w || w->state != WRITE_HOLD || w->kind != RC_REGISTRY_IMAGE || w->image_forgotten ||
        (w->image && w->image != image)) {
        unlock_registry(r); return RC_REGISTRY_STALE;
    }
    if (!w->image) {
        w->image = image;
        if (w->interval.range.backing == RC_GUEST_COMPATIBILITY_BACKING)
            w->image_address = w->interval.range.offset;
        invalidate_content(r, w->interval.range, 0);
    }
    unlock_registry(r); return RC_REGISTRY_OK;
}
void rc_registry_image_hold_forget(RcRegistry *r, RcRegistryWrite hold) {
    if (!r) return;
    lock_registry(r); Writer *w = writer_of(r, hold);
    if (w && w->state == WRITE_HOLD && w->kind == RC_REGISTRY_IMAGE && !w->image_forgotten) {
        w->image_forgotten = 1;
        invalidate_content(r, w->interval.range, 0);
    }
    unlock_registry(r);
}
int rc_registry_image_binding_current(RcRegistry *r, RcRegistryImageBinding image) {
    if (!r || !image.image) return 0;
    lock_registry(r); Writer *w = writer_of(r, image.hold);
    int current = !r->stats.fail_closed && w && w->state == WRITE_HOLD &&
        w->kind == RC_REGISTRY_IMAGE && !w->image_forgotten && w->image == image.image;
    unlock_registry(r); return current;
}
int rc_registry_image_hold_covers(RcRegistry *r, RcRegistryWrite hold, RcSourceRange range) {
    if (!r || !rc_range_valid(range)) return 0;
    lock_registry(r); Writer *w = writer_of(r, hold);
    int covered = w && w->state == WRITE_HOLD && w->kind == RC_REGISTRY_IMAGE &&
        w->interval.range.backing == range.backing && w->interval.range.offset <= range.offset &&
        range.size <= w->interval.range.size &&
        range.offset - w->interval.range.offset <= w->interval.range.size - range.size;
    unlock_registry(r); return covered;
}
RcRegistryResult rc_registry_image_producer_add(RcRegistry *r, RcRegistryWrite ticket, RcRegistryPosition position) {
    if (!r || !rc_registry_position_valid(position)) return RC_REGISTRY_INVALID;
    lock_registry(r); Writer *w = writer_of(r, ticket);
    if (!w || w->state != WRITE_PENDING || w->kind != RC_REGISTRY_IMAGE || position.scene != w->position.scene) {
        unlock_registry(r); return RC_REGISTRY_STALE;
    }
    if (rc_registry_position_compare(position, r->declared) <= 0 ||
        rc_registry_position_compare(position, w->position) < 0) {
        r->stats.late_declarations++; poison(r); unlock_registry(r); return RC_REGISTRY_STALE;
    }
    if (w->image_count) invalidate_schedule(r, w->interval.range, ticket, position, RC_REGISTRY_SCHEDULE_ADD);
    uint32_t at = image_position_lower(w, position);
    if (at < w->image_count && !rc_registry_position_compare(w->image_positions[at].position, position)) {
        ImageProducer *p = &w->image_positions[at];
        if (p->refs == UINT32_MAX || p->unresolved == UINT32_MAX) goto capacity;
        p->refs++; p->unresolved++;
        if (at < w->image_dirty) w->image_dirty = at;
        unlock_registry(r); return RC_REGISTRY_OK;
    }
    if (r->stats.image_positions == r->capacity.records) goto capacity;
    if (w->image_count == w->image_capacity) {
        uint32_t capacity = w->image_capacity ? w->image_capacity * 2 : 4;
        if (capacity > r->capacity.records) capacity = r->capacity.records;
        ImageProducer *grown = realloc(w->image_positions, (size_t)capacity * sizeof *grown);
        if (!grown) goto capacity;
        w->image_positions = grown; w->image_capacity = capacity;
    }
    memmove(w->image_positions + at + 1, w->image_positions + at, (w->image_count - at) * sizeof *w->image_positions);
    w->image_positions[at] = (ImageProducer){.position = position, .refs = 1, .unresolved = 1};
    w->image_count++; r->stats.image_positions++;
    if (r->stats.image_positions > r->stats.image_position_peak) r->stats.image_position_peak = r->stats.image_positions;
    if (at < w->image_dirty) w->image_dirty = at;
    unlock_registry(r); return RC_REGISTRY_OK;
capacity:
    image_position_incomplete(r, w); r->stats.image_position_refusals++;
    unlock_registry(r); return RC_REGISTRY_CAPACITY;
}
RcRegistryResult rc_registry_image_producer_resolve(RcRegistry *r, RcRegistryWrite ticket,
    RcRegistryPosition position, const RcRegistryImageBinding *image, int canceled) {
    if (!r || !rc_registry_position_valid(position) || (image && canceled)) return RC_REGISTRY_INVALID;
    lock_registry(r); Writer *w = writer_of(r, ticket);
    if (!w || w->state != WRITE_PENDING || w->kind != RC_REGISTRY_IMAGE) {
        unlock_registry(r); return RC_REGISTRY_STALE;
    }
    uint32_t at = image_position_lower(w, position);
    if (at == w->image_count || rc_registry_position_compare(w->image_positions[at].position, position) ||
        !w->image_positions[at].unresolved) {
        image_position_incomplete(r, w); invalidate_content(r, w->interval.range, 0);
        unlock_registry(r); return RC_REGISTRY_STALE;
    }
    ImageProducer *p = &w->image_positions[at]; p->unresolved--;
    if (!canceled) {
        if (!image || !image_binding_current_locked(r, *image, w->interval.range)) p->unknown = 1;
        else if ((p->active || p->preserves_native) && !image_binding_equal(p->image, *image)) p->unknown = 1;
        else p->image = *image;
        p->active++;
    }
    if (p->preserves_native && (canceled || p->unknown)) {
        p->preserves_native = p->preserves_projection = 0;
        if (!p->active) p->image = (RcRegistryImageBinding){0};
    }
    if (p->image.image) p->projected = native_projection_confirm(r, p->image, position,
        !w->image_incomplete && !p->unresolved && !p->unknown);
    if (at < w->image_dirty) w->image_dirty = at;
    // Resolving an already declared boundary does not extend any existing receipt past that boundary. A prior
    // unresolved producer could not issue a receipt at all. Avoid turning ordinary command progress into a new
    // source-wide invalidation; additions and contradictory/late updates still invalidate explicitly.
    unlock_registry(r); return RC_REGISTRY_OK;
}

int rc_registry_image_producer_state(RcRegistry *r, RcRegistryWrite ticket, RcRegistryPosition output,
                                     RcRegistryImageProducerState *state) {
    if (!state) return 0;
    *state = (RcRegistryImageProducerState){0};
    if (!r || !ticket.id || !rc_registry_position_valid(output)) return 0;
    int found = 0;
    lock_registry(r);
    Writer *writer = writer_of(r, ticket);
    if (writer && writer->kind == RC_REGISTRY_IMAGE) {
        const uint32_t at = image_position_lower(writer, output);
        if (at < writer->image_count && !rc_registry_position_compare(writer->image_positions[at].position, output)) {
            const ImageProducer *producer = &writer->image_positions[at];
            *state = (RcRegistryImageProducerState){
                .interval = {writer->image_address, writer->image_mapping, writer->interval.range},
                .image = producer->image, .unresolved = producer->unresolved, .active = producer->active,
                .unknown = producer->unknown, .projected = producer->projected, .incomplete = writer->image_incomplete,
                .exact_effect = producer->exact_effect,
            };
            found = 1;
        }
    }
    unlock_registry(r);
    return found;
}

int rc_registry_image_hold_recorded(RcRegistry *r, RcRegistryWrite claim, RcRegistryImageBinding image,
    RcRegistryPosition output, int first_write) {
    if (!r || !rc_registry_position_valid(output) || !(output.ordinal & 1)) return 0;
    lock_registry(r);
    Writer *declared = writer_of(r, claim), *held = writer_of(r, image.hold);
    int exact = !r->stats.fail_closed && declared && declared->state == WRITE_PENDING &&
        declared->kind == RC_REGISTRY_IMAGE && !declared->image_incomplete && held &&
        image_binding_current_locked(r, image, declared->interval.range) &&
        rc_range_equal(held->interval.range, declared->interval.range);
    if (exact) {
        const uint32_t at = image_position_lower(declared, output);
        exact = at < declared->image_count &&
            !rc_registry_position_compare(declared->image_positions[at].position, output) &&
            declared->image_positions[at].unresolved == 1;
    }
    if (exact && first_write) held->native_ordered_effect = 1;
    unlock_registry(r);
    return exact;
}

static int read_position_open(RcRegistry *, RcRegistryPosition);

int rc_registry_image_hold_preserved(RcRegistry *r, RcRegistryWrite claim, RcRegistryImageBinding image,
    RcRegistryPosition output, int exact_projection_layout) {
    if (!r || !rc_registry_position_valid(output) || !(output.ordinal & 1)) return 0;
    lock_registry(r);
    Writer *declared = writer_of(r, claim), *held = writer_of(r, image.hold);
    int exact = !r->stats.fail_closed && read_position_open(r, output) && declared &&
        declared->state == WRITE_PENDING && declared->kind == RC_REGISTRY_IMAGE && !declared->image_incomplete &&
        image_binding_current_locked(r, image, declared->interval.range) &&
        rc_range_equal(held->interval.range, declared->interval.range) &&
        held->image_mapping && held->native_ordered_effect && !held->native_unordered &&
        !held->native_conflict && !rc_registry_position_valid(held->native_output);
    if (exact) {
        const uint32_t at = image_position_lower(declared, output);
        ImageProducer *producer = at < declared->image_count ? &declared->image_positions[at] : NULL;
        exact = producer && !rc_registry_position_compare(producer->position, output) &&
            producer->unresolved == 1 && !producer->active && !producer->unknown && !producer->preserves_native;
        if (exact) {
            producer->preserves_native = 1;
            producer->preserves_projection = exact_projection_layout &&
                held->native_projected && held->image_possible_known && held->image_possible;
            producer->image = image;
        }
    }
    unlock_registry(r);
    return exact;
}

static int native_bound_output(const Writer *held, RcRegistryImageBinding binding, RcRegistryPosition output) {
    return held && held->state == WRITE_HOLD && held->image == binding.image &&
        !held->image_forgotten && held->native_bound_whole && !rc_registry_position_compare(held->native_output, output);
}

static ImageProducer *image_producer_affecting_projection(RcRegistry *r, Writer *writer,
    RcRegistryPosition output, RcRegistryImageBinding binding) {
    if (!writer || writer->state != WRITE_PENDING || writer->kind != RC_REGISTRY_IMAGE) return NULL;
    uint32_t at = image_position_lower(writer, output);
    if (at == writer->image_count) return NULL;
    ImageProducer *producer = &writer->image_positions[at];
    if (producer->unresolved || producer->unknown || !producer->active ||
        rc_registry_position_compare(producer->position, output) || !image_binding_equal(producer->image, binding)) return NULL;
    const Writer *held = writer_of(r, producer->image.hold);
    return producer->projected || (held && (held->native_projected || held->native_ordered_effect)) ||
        native_bound_output(held, binding, output) ? producer : NULL;
}

int rc_registry_image_producer_projection_pending(RcRegistry *r, RcRegistryWrite ticket,
    RcRegistryPosition output, RcRegistryImageBinding binding) {
    if (!r) return 0;
    lock_registry(r);
    int pending = image_producer_affecting_projection(r, writer_of(r, ticket), output, binding) != NULL;
    unlock_registry(r);
    return pending;
}

void rc_registry_image_producer_native_completed(RcRegistry *r, RcRegistryWrite ticket,
    RcRegistryPosition output, RcRegistryImageBinding binding) {
    if (!r) return;
    lock_registry(r);
    Writer *held = writer_of(r, binding.hold);
    Writer *claim = writer_of(r, ticket);
    if (held && claim && !claim->image_incomplete && rc_range_equal(held->interval.range, claim->interval.range) &&
        rc_registry_position_compare(output, r->handed) <= 0) {
        ImageProducer *producer = image_producer_affecting_projection(r, claim, output, binding);
        if (producer && native_bound_output(held, binding, output))
            held->native_bound_whole = NATIVE_BOUND_COMPLETED;
        if (producer && producer->projected) {
            for (NativeImageProjection *p = held->native_projection; p; p = p->next)
                if (p->confirmed && !rc_registry_position_compare(p->output, output)) {
                    p->completed = 1;
                    break;
                }
        } else if (producer && held->native_ordered_effect) {
            producer->preserved_completed = producer->preserves_native;
            producer->exported_completed = producer->preserved_completed && producer->preserves_projection &&
                held->native_projected && held->image_possible_known && held->image_possible &&
                !held->native_unordered && !held->native_conflict && !held->image_forgotten &&
                !rc_registry_position_valid(held->native_output);
            native_projection_unproved(held, output, held->image_possible_known && !held->native_unordered &&
                                       !rc_registry_position_valid(held->native_output), producer->preserved_completed);
            if (producer->preserved_completed) {
                RcRegistryPosition *latest = producer->exported_completed ? &held->native_preserved_projection :
                    &held->native_resident_only;
                if (rc_registry_position_compare(output, *latest) > 0) *latest = output;
            }
        }
    }
    unlock_registry(r);
}

int rc_registry_image_producer_projection_abandon(RcRegistry *r, RcRegistryWrite ticket,
    RcRegistryPosition output, RcRegistryImageBinding binding, int may_have_executed) {
    if (!r) return 0;
    lock_registry(r);
    Writer *writer = writer_of(r, ticket);
    ImageProducer *producer = image_producer_affecting_projection(r, writer, output, binding);
    int found = producer != NULL;
    if (producer) {
        Writer *held = writer_of(r, binding.hold);
        if (native_bound_output(held, binding, output)) native_image_output_clear(r, held);
        native_projection_drop(r, binding, output);
        producer->projected = 0;
        if (may_have_executed) {
            producer->unknown = 1;
            if (held) native_projection_unproved(held, output, 0, 0);
            native_projection_invalidate(r, writer->interval.range, output, NULL, 1);
        } else producer->active = 0;
        uint32_t at = (uint32_t)(producer - writer->image_positions);
        if (at < writer->image_dirty) writer->image_dirty = at;
        invalidate_content(r, writer->interval.range, 0);
    }
    unlock_registry(r);
    return found;
}

RcRegistryResult rc_registry_image_producer_settle(RcRegistry *r, RcRegistryWrite ticket,
    RcRegistryPosition output, int executed) {
    if (!r || !rc_registry_position_valid(output)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    Writer *w = writer_of(r, ticket);
    RcRegistryResult result = RC_REGISTRY_STALE;
    if (w && w->state == WRITE_PENDING && w->kind == RC_REGISTRY_IMAGE && (!executed || !w->image_incomplete)) {
        uint32_t at = image_position_lower(w, output);
        if (at < w->image_count) {
            ImageProducer *p = &w->image_positions[at];
            if (!rc_registry_position_compare(p->position, output) && p->refs == 1 && !p->unresolved &&
                p->active == 1 && p->unknown && !p->exact_effect) {
                if (!executed) {
                    image_position_cancel_projection(r, w, p);
                    p->active = 0;
                    p->unknown = 0;
                    if (at < w->image_dirty) w->image_dirty = at;
                    result = RC_REGISTRY_OK;
                } else if (rc_registry_position_compare(output, r->handed) <= 0) {
                    p->exact_effect = 1;
                    result = RC_REGISTRY_OK;
                }
            }
        }
    }
    unlock_registry(r);
    return result;
}
void rc_registry_image_producer_invalidate(RcRegistry *r, RcRegistryWrite ticket) {
    if (!r) return;
    lock_registry(r); Writer *w = writer_of(r, ticket);
    if (w && !w->image_incomplete) { image_position_incomplete(r, w); invalidate_content(r, w->interval.range, 0); }
    unlock_registry(r);
}
static RcRegistryRecord record_value(Record *e, uint32_t index) {
    return (RcRegistryRecord){.handle = {e->id, index}, .key = e->key, .object_serial = e->object,
        .interpretation = e->interpretation, .range = e->interval.range, .content = e->content,
        .representation = e->representation, .payload = e->payload,
        .finalized = e->finalized, .current = e->current};
}
static void retire_record(RcRegistry *r, uint32_t i, uint64_t end) {
    Record *e = &r->records[i];
    if (e->current) {
        if (e->source) r->stats.live_sources--; else r->stats.live_descriptors--;
    }
    e->current = 0;
    if (!e->end) e->end = end;
    if (e->retired_queued) return;
    e->retired_next = r->retired_head;
    if (e->retired_next) r->records[e->retired_next].retired_prev = i;
    r->retired_head = i;
    if (!r->retired_scan) r->retired_scan = i;
    e->retired_queued = 1; r->stats.retired_records++;
}
static void free_record(RcRegistry *r, uint32_t i) {
    Record *e = &r->records[i];
    if (e->current) {
        if (e->source) r->stats.live_sources--; else r->stats.live_descriptors--;
    }
    if (e->retired_queued) {
        if (r->retired_scan == i) r->retired_scan = e->retired_next;
        if (e->retired_prev) r->records[e->retired_prev].retired_next = e->retired_next;
        else r->retired_head = e->retired_next;
        if (e->retired_next) r->records[e->retired_next].retired_prev = e->retired_prev;
        r->stats.retired_records--;
    }
    if (e->source) r->stats.source_records--;
    r->registrations.root = erase_node(&r->registrations, r->registrations.root, i);
    if (e->newer) r->records[e->newer].older = e->older;
    else {
        int found;
        uint32_t slot = key_slot(r, e->source, e->key, e->interval.range, e->interpretation, &found);
        if (found && r->keys[slot] == i) {
            if (e->older) r->keys[slot] = e->older;
            else key_remove(r, slot);
        }
    }
    if (e->older) r->records[e->older].newer = e->newer;
    r->source_span_count -= e->source_span_count;
    free(e->source_spans);
    memset(e, 0, sizeof *e);
    e->free_next = r->free_record; r->free_record = i;
    r->stats.records--; r->stats.collected++;
}

static int record_scope_reachable_locked(const RcRegistry *r, const Record *e) {
    if (!r->stats.scopes || e->id > r->newest_scope || (e->end && e->end <= r->oldest_scope)) return 0;
    if (e->id <= r->oldest_scope || !e->end || r->newest_scope < e->end) return 1;
    for (uint32_t i = 1; i <= r->capacity.scopes; i++) {
        const Scope *scope = &r->scopes[i];
        if (scope->id && e->id <= scope->horizon && scope->horizon < e->end) return 1;
    }
    return 0;
}

static uint32_t collect_locked(RcRegistry *r, uint32_t budget) {
    uint32_t freed = 0;
    if (budget > r->stats.retired_records) budget = r->stats.retired_records;
    for (uint32_t n = 0; n < budget; n++) {
        uint32_t i = r->retired_scan ? r->retired_scan : r->retired_head;
        if (!i) break;
        Record *e = &r->records[i];
        r->retired_scan = e->retired_next;
        if (e->used && !e->current && !e->refs && e->end &&
            (e->source ? e->end <= r->oldest_scope : !record_scope_reachable_locked(r, e))) {
            free_record(r, i); freed++;
        }
    }
    return freed;
}

static ForeignCoverage *foreign_coverage_alloc(RcRegistry *r, uint32_t count) {
    // A readback shape has its own storage bound, independent of pending writer identities.
    if (!count || count > image_span_room(r)) return NULL;
    ForeignCoverage *coverage = malloc(sizeof *coverage + (size_t)count * sizeof *coverage->spans);
    if (coverage) {
        coverage->refs = 1;
        coverage->count = count;
        r->foreign_span_count += count;
    }
    return coverage;
}

static void foreign_coverage_release(RcRegistry *r, ForeignCoverage *coverage) {
    if (!coverage || --coverage->refs) return;
    r->foreign_span_count -= coverage->count;
    free(coverage);
}

RcRegistry *rc_registry_create(RcRegistryConfig c) {
    if (!c.records || !c.writers || !c.scopes || c.records > (1u << 26) ||
        c.writers > (1u << 26) || c.scopes > (1u << 20)) return NULL;
    RcRegistry *r = calloc(1, sizeof *r);
    if (!r) return NULL;
#ifndef RC_REGISTRY_SINGLE_THREADED
    plat_cond_init(&r->gpu_capture_released);
#endif
    r->capacity = c; r->key_capacity = 2;
    while (r->key_capacity < c.records * 2) r->key_capacity <<= 1;
    r->records = calloc((size_t)c.records + 1, sizeof *r->records);
    r->writers = calloc((size_t)c.writers + 1, sizeof *r->writers);
    r->scopes = calloc((size_t)c.scopes + 1, sizeof *r->scopes);
    r->captures = calloc((size_t)c.records + 1, sizeof *r->captures);
    r->changes = calloc(c.records, sizeof *r->changes);
    r->schedule_changes = calloc(c.records, sizeof *r->schedule_changes);
    r->keys = calloc(r->key_capacity, sizeof *r->keys);
    r->scratch = calloc(c.writers, sizeof *r->scratch);
    if (!r->records || !r->writers || !r->scopes || !r->captures || !r->changes || !r->schedule_changes || !r->keys || !r->scratch) {
        rc_registry_destroy(r); return NULL;
    }
    for (uint32_t i = 1; i <= c.records; i++) r->records[i].free_next = r->captures[i].free_next = i == c.records ? 0 : i + 1;
    for (uint32_t i = 1; i <= c.writers; i++) r->writers[i].free_next = i == c.writers ? 0 : i + 1;
    for (uint32_t i = 1; i <= c.scopes; i++) r->scopes[i].free_next = i == c.scopes ? 0 : i + 1;
    r->free_record = r->free_writer = r->free_scope = r->free_capture = 1;
    r->oldest_scope = UINT64_MAX; r->guard = 1; r->authority_epoch = r->ordered_epoch = 1;
    r->registrations = (Tree){r->records, sizeof *r->records, 0};
    r->authority = (Tree){r->writers, sizeof *r->writers, 0};
    return r;
}
void rc_registry_destroy(RcRegistry *r) {
    if (!r) return;
    if (r->writers) for (uint32_t i = 1; i <= r->capacity.writers; ++i) {
        native_projection_clear(r, &r->writers[i]);
        free(r->writers[i].image_positions);
        foreign_coverage_release(r, r->writers[i].foreign_coverage);
        foreign_coverage_release(r, r->writers[i].image_possible);
        buffer_payload_drop(&r->writers[i].native_payload);
    }
    while (r->gpu_payload_head) gpu_visibility_payload_drop(r, r->gpu_payload_head);
    if (r->records) for (uint32_t i = 1; i <= r->capacity.records; ++i) free(r->records[i].source_spans);
    free(r->records); free(r->writers); free(r->scopes); free(r->captures); free(r->changes); free(r->schedule_changes);
    free(r->gpu_captures); free(r->gpu_visibility);
    free(r->keys); free(r->scratch); free(r);
}
#ifndef RC_REGISTRY_SINGLE_THREADED
static PlatOnce runtime_registry_once = PLAT_ONCE_INIT;
static void runtime_registry_init(void *unused) {
    (void)unused;
    runtime_registry = rc_registry_create((RcRegistryConfig){32768, 8192, 256});
}
RcRegistry *rc_registry_runtime(void) {
    plat_once(&runtime_registry_once, runtime_registry_init, NULL);
    return runtime_registry;
}
#else
RcRegistry *rc_registry_runtime(void) { return NULL; }
#endif

typedef struct { const RcSourceRange *spans; uint32_t count; } SourceShape;

static int source_shape_same(const Record *e, SourceShape shape) {
    return shape.count == e->source_span_count &&
           (!shape.count || !memcmp(shape.spans, e->source_spans, (size_t)shape.count * sizeof *shape.spans));
}

static RcRegistryResult register_locked(RcRegistry *r, int source, uint64_t key, uint64_t object,
    RcSourceRange range, uint64_t interpretation, SourceShape shape, RcRegistryHandle *out) {
    int found;
    uint32_t slot = key_slot(r, source, key, range, interpretation, &found), old = found ? r->keys[slot] : 0;
    if (old) {
        Record *e = &r->records[old];
        if (e->current && !e->finalized && e->object == object && e->interpretation == interpretation &&
            rc_range_equal(e->interval.range, range)) {
            if (!source_shape_same(e, shape)) return RC_REGISTRY_STALE;
            *out = (RcRegistryHandle){e->id, old};
            return RC_REGISTRY_OK;
        }
        authority_changed(r);
    }
    // No scope or capture can resolve this old head after replacement. Reuse its slot immediately, including
    // when the pool is full; retaining invisible history until a periodic scan only amplifies descriptor churn.
    if (old && !r->records[old].refs &&
        (!r->stats.scopes || (!source && !record_scope_reachable_locked(r, &r->records[old])))) {
        if (r->clock == UINT64_MAX) { poison(r); return RC_REGISTRY_EXHAUSTED; }
        free_record(r, old);
    }
    // Shapes are metadata on the existing source records, with a shared bound over requested coverage.
    uint32_t span_limit = r->capacity.records * 32;
    if (!r->free_record || shape.count > span_limit - r->source_span_count) collect_locked(r, r->stats.retired_records);
    // Collection can remove this key's historical head; locate it again before linking its replacement.
    slot = key_slot(r, source, key, range, interpretation, &found); old = found ? r->keys[slot] : 0;
    if (!r->free_record || slot == UINT32_MAX || shape.count > span_limit - r->source_span_count) {
        r->stats.record_refusals++;
        return RC_REGISTRY_CAPACITY;
    }
    RcSourceRange *spans = NULL;
    if (shape.count) {
        spans = malloc((size_t)shape.count * sizeof *spans);
        if (!spans) { r->stats.record_refusals++; return RC_REGISTRY_CAPACITY; }
        memcpy(spans, shape.spans, (size_t)shape.count * sizeof *spans);
    }
    uint64_t id = identity(r);
    if (!id) { free(spans); return RC_REGISTRY_EXHAUSTED; }
    uint32_t i = r->free_record;
    Record *e = &r->records[i]; r->free_record = e->free_next;
    *e = (Record){.interval = {.range = range, .order = id, .priority = mixed(id)}, .id = id,
        .key = key, .object = object, .interpretation = interpretation, .revision = 1, .content_revision = 1,
        .ordered_revision = 1, .source_spans = spans, .source_span_count = shape.count,
        .older = old, .used = 1, .source = (uint8_t)source, .current = 1};
    if (old) {
        retire_record(r, old, id);
        r->records[old].newer = i;
        invalidate_record(r, old, NULL);
    }
    r->keys[slot] = i;
    r->source_span_count += shape.count;
    r->registrations.root = insert_node(&r->registrations, r->registrations.root, i);
    r->stats.records++; r->stats.registrations++;
    if (source) r->stats.source_records++, r->stats.live_sources++;
    else r->stats.live_descriptors++;
    *out = (RcRegistryHandle){id, i};
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_register(RcRegistry *r, uint64_t key, uint64_t object, RcSourceRange range,
                                      uint64_t interpretation, RcRegistryHandle *out) {
    if (out) *out = (RcRegistryHandle){0};
    if (!r || !out || !key || !object || !rc_range_valid(range)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = register_locked(r, 0, key, object, range, interpretation, (SourceShape){0}, out);
    unlock_registry(r); return result;
}
RcRegistryResult rc_registry_source(RcRegistry *r, uint64_t object, RcSourceRange range, RcRegistryHandle *out) {
    if (out) *out = (RcRegistryHandle){0};
    if (!r || !out || !object || !rc_range_valid(range)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = register_locked(r, 1, object, object, range, 0, (SourceShape){0}, out);
    unlock_registry(r); return result;
}
static RcRegistryResult source_acquire_associated(RcRegistry *r, uint64_t object, RcSourceRange range,
                                                  uint64_t association, SourceShape shape, RcRegistrySource *lease) {
    if (!r || !lease || !object || !rc_range_valid(range)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    if (lease->id) {
        Capture *owner = lease->index && lease->index <= r->capacity.records ? &r->captures[lease->index] : NULL;
        Record *e = owner && owner->id == lease->id && owner->source ? record_of(r, owner->record) : NULL;
        RcRegistryResult result = e && e->current && !e->finalized && e->object == object && e->interpretation==association &&
            rc_range_equal(e->interval.range, range) && source_shape_same(e, shape) && e->id == lease->handle.id &&
            owner->record.index == lease->handle.index ? RC_REGISTRY_OK : RC_REGISTRY_STALE;
        unlock_registry(r); return result;
    }
    if (lease->index || lease->handle.id || lease->handle.index) { unlock_registry(r); return RC_REGISTRY_INVALID; }
    if (!r->free_capture) { r->stats.record_refusals++; unlock_registry(r); return RC_REGISTRY_CAPACITY; }
    RcRegistryHandle handle;
    RcRegistryResult result = register_locked(r, 1, object, object, range, association, shape, &handle);
    if (result != RC_REGISTRY_OK) { unlock_registry(r); return result; }
    Record *e = record_of(r, handle);
    if (e->refs == UINT32_MAX || e->owners == UINT32_MAX) { unlock_registry(r); return RC_REGISTRY_CAPACITY; }
    uint64_t id = identity(r);
    if (!id) { unlock_registry(r); return RC_REGISTRY_EXHAUSTED; }
    uint32_t i = r->free_capture; r->free_capture = r->captures[i].free_next;
    r->captures[i] = (Capture){.id = id, .record = handle, .source = 1};
    e->refs++; e->owners++; r->stats.source_leases++;
    *lease = (RcRegistrySource){id, i, handle};
    unlock_registry(r); return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_source_acquire(RcRegistry *r, uint64_t object, RcSourceRange range, RcRegistrySource *lease) {
    return source_acquire_associated(r, object, range, 0, (SourceShape){0}, lease);
}
RcRegistryResult rc_registry_source_release(RcRegistry *r, RcRegistrySource *lease) {
    if (!r || !lease || !lease->id || !lease->index || lease->index > r->capacity.records) return RC_REGISTRY_INVALID;
    lock_registry(r); Capture *owner = &r->captures[lease->index];
    Record *e = owner->id == lease->id && owner->source ? record_of(r, owner->record) : NULL;
    if (!e || !e->refs || !e->owners || e->id != lease->handle.id || owner->record.index != lease->handle.index) {
        *lease = (RcRegistrySource){0};
        unlock_registry(r); return RC_REGISTRY_STALE;
    }
    e->refs--; e->owners--; r->stats.source_leases--;
    uint32_t record = owner->record.index;
    *owner = (Capture){.free_next = r->free_capture}; r->free_capture = lease->index;
    *lease = (RcRegistrySource){0};
    if (!e->owners && e->current) {
        uint64_t end = identity(r);
        if (end) { authority_changed(r); e->finalized = 1; invalidate_record(r, record, NULL); retire_record(r, record, end); }
    }
    if (!e->current && !e->refs && e->end && e->end <= r->oldest_scope) free_record(r, record);
    unlock_registry(r); return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_finalize(RcRegistry *r, uint64_t key, uint64_t object) {
    if (!r || !key || !object) return RC_REGISTRY_INVALID;
    lock_registry(r);
    int found; uint32_t slot = key_slot(r, 0, key, (RcSourceRange){0}, 0, &found);
    Record *e = found ? &r->records[r->keys[slot]] : NULL;
    RcRegistryResult result = RC_REGISTRY_STALE;
    if (e && e->current && e->object == object) {
        if (!e->finalized) { authority_changed(r); e->finalized = 1; invalidate_record(r, r->keys[slot], NULL); }
        result = RC_REGISTRY_OK;
    }
    unlock_registry(r); return result;
}
RcRegistryResult rc_registry_retire_registration(RcRegistry *r, uint64_t key, uint64_t object) {
    if (!r || !key || !object) return RC_REGISTRY_INVALID;
    lock_registry(r);
    int found;
    uint32_t slot = key_slot(r, 0, key, (RcSourceRange){0}, 0, &found);
    uint32_t index = found ? r->keys[slot] : 0;
    Record *record = index ? &r->records[index] : NULL;
    RcRegistryResult result = RC_REGISTRY_STALE;
    if (record && record->current && record->object == object) {
        uint64_t end = identity(r);
        result = RC_REGISTRY_EXHAUSTED;
        if (end) {
            record->finalized = 1;
            authority_changed(r);
            invalidate_record(r, index, NULL);
            retire_record(r, index, end);
            result = RC_REGISTRY_OK;
        }
    }
    unlock_registry(r);
    return result;
}
typedef struct { uint64_t object, end; int gone; } Finalize;
typedef struct { RcSourceRange range; uint64_t order; } MappingRetirement;

static int writer_mapping_gone(RcRegistry *r, uint32_t i, void *data) {
    // A flat guest-address range may now denote replacement backing. Retain the old writer's exclusion, but an
    // earlier copy proof cannot remove it even if its execution callback arrives after this mapping change.
    Writer *writer = &r->writers[i];
    const MappingRetirement *change = data;
    ForeignCoverage *coverage = writer->foreign_coverage;
    if (!coverage || (change->range.offset <= writer->interval.range.offset &&
        change->range.offset + change->range.size >= writer->interval.range.offset + writer->interval.range.size)) {
        writer->producer_order = change->order;
        return 0;
    }
    if (coverage->refs > 1) {
        uint64_t begin = writer->foreign_address, end = begin + writer->interval.range.size;
        uint32_t first = foreign_span_first(coverage, begin), last = first;
        while (last < coverage->count && coverage->spans[last].begin < end) last++;
        ForeignCoverage *owned = foreign_coverage_alloc(r, last - first);
        if (!owned) {
            // On bounded metadata pressure keep the old exclusion with a conservative whole-owner horizon.
            writer->producer_order = change->order;
            return 0;
        }
        memcpy(owned->spans, &coverage->spans[first], (size_t)owned->count * sizeof *owned->spans);
        foreign_coverage_release(r, coverage);
        writer->foreign_coverage = coverage = owned;
    }
    ForeignSpan affected = writer_foreign_span(writer, change->range);
    for (uint32_t j = foreign_span_first(coverage, affected.begin);
         j < coverage->count && coverage->spans[j].begin < affected.end; ++j)
        coverage->spans[j].producer_order = change->order;
    return 0;
}
static int finalize_record(RcRegistry *r, uint32_t i, void *data) {
    Record *e = &r->records[i]; Finalize *f = data;
    if (!f->gone && e->object != f->object) return 0;
    e->finalized = 1;
    if ((f->gone || e->source) && e->current) retire_record(r, i, f->end);
    invalidate_record(r, i, NULL);
    return 0;
}
RcRegistryResult rc_registry_finalize_object(RcRegistry *r, uint64_t object, RcSourceRange range) {
    if (!r || !object || !rc_range_valid(range)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    authority_changed(r);
    uint64_t end = identity(r);
    Finalize f = {object, end, 0};
    if (end) overlaps(r, &r->registrations, r->registrations.root, range, finalize_record, &f);
    unlock_registry(r); return end ? RC_REGISTRY_OK : RC_REGISTRY_EXHAUSTED;
}
RcRegistryResult rc_registry_range_gone(RcRegistry *r, RcSourceRange range) {
    if (!r || !rc_range_valid(range)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    authority_changed(r);
    uint64_t end = identity(r);
    Finalize f = {0, end, 1};
    if (end) overlaps(r, &r->registrations, r->registrations.root, range, finalize_record, &f);
    if (end) {
        MappingRetirement change = {range, end};
        overlaps(r, &r->authority, r->authority.root, range, writer_mapping_gone, &change);
    }
    // Pending writers retain their own completion identity. A flat virtual-address adapter conservatively keeps
    // their authority until real backing/visibility reconciliation; an old completion cannot certify replacement data.
    unlock_registry(r); return end ? RC_REGISTRY_OK : RC_REGISTRY_EXHAUSTED;
}
RcRegistryResult rc_registry_cpu_changed(RcRegistry *r, RcSourceRange range) {
    if (!r || !rc_range_valid(range)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    native_projection_invalidate(r, range, (RcRegistryPosition){0}, NULL, 1);
    invalidate(r, range);
    unlock_registry(r);
    return RC_REGISTRY_OK;
}

RcRegistryResult rc_registry_scope_begin(RcRegistry *r, RcRegistryScope *out) {
    if (out) *out = (RcRegistryScope){0};
    if (!r || !out) return RC_REGISTRY_INVALID;
    lock_registry(r);
    if (!r->free_scope) { unlock_registry(r); return RC_REGISTRY_CAPACITY; }
    uint64_t horizon = r->clock, id = identity(r);
    if (!id) { unlock_registry(r); return RC_REGISTRY_EXHAUSTED; }
    uint32_t i = r->free_scope; r->free_scope = r->scopes[i].free_next;
    r->scopes[i] = (Scope){id, horizon, 0};
    if (horizon < r->oldest_scope) r->oldest_scope = horizon;
    if (horizon > r->newest_scope) r->newest_scope = horizon;
    r->stats.scopes++; *out = (RcRegistryScope){id, i};
    unlock_registry(r); return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_scope_end(RcRegistry *r, RcRegistryScope scope) {
    if (!r || !scope.id || !scope.index || scope.index > r->capacity.scopes) return RC_REGISTRY_INVALID;
    lock_registry(r); Scope *s = &r->scopes[scope.index];
    if (s->id != scope.id) { unlock_registry(r); return RC_REGISTRY_STALE; }
    *s = (Scope){.free_next = r->free_scope}; r->free_scope = scope.index; r->stats.scopes--;
    r->oldest_scope = UINT64_MAX;
    r->newest_scope = 0;
    for (uint32_t i = 1; i <= r->capacity.scopes; i++) if (r->scopes[i].id) {
        uint64_t horizon = r->scopes[i].horizon;
        if (horizon < r->oldest_scope) r->oldest_scope = horizon;
        if (horizon > r->newest_scope) r->newest_scope = horizon;
    }
    unlock_registry(r); return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_resolve(RcRegistry *r, uint64_t key, RcRegistryScope scope, RcRegistryRecord *out) {
    if (out) *out = (RcRegistryRecord){0};
    if (!r || !key || !out) return RC_REGISTRY_INVALID;
    lock_registry(r); uint64_t horizon = UINT64_MAX;
    if (scope.id) {
        if (!scope.index || scope.index > r->capacity.scopes || r->scopes[scope.index].id != scope.id) {
            unlock_registry(r); return RC_REGISTRY_STALE;
        }
        horizon = r->scopes[scope.index].horizon;
    } else if (scope.index) { unlock_registry(r); return RC_REGISTRY_INVALID; }
    int found; uint32_t slot = key_slot(r, 0, key, (RcSourceRange){0}, 0, &found);
    for (uint32_t i = found ? r->keys[slot] : 0; i; i = r->records[i].older) {
        Record *e = &r->records[i]; r->stats.history_visits++;
        if ((!scope.id && e->current) || (scope.id && e->id <= horizon && (!e->end || horizon < e->end))) {
            *out = record_value(e, i); unlock_registry(r); return RC_REGISTRY_OK;
        }
        if (!scope.id) break;
    }
    unlock_registry(r); return RC_REGISTRY_STALE;
}
RcRegistryResult rc_registry_get(RcRegistry *r, RcRegistryHandle handle, RcRegistryRecord *out) {
    if (out) *out = (RcRegistryRecord){0};
    if (!r || !out) return RC_REGISTRY_INVALID;
    lock_registry(r); Record *e = record_of(r, handle);
    if (e) *out = record_value(e, handle.index);
    unlock_registry(r); return e ? RC_REGISTRY_OK : RC_REGISTRY_STALE;
}
int rc_registry_registration_reachable(RcRegistry *r, RcRegistryHandle handle) {
    if (!r) return 0;
    lock_registry(r);
    const Record *record=record_of(r,handle);
    int reachable=record && !record->source && record_scope_reachable_locked(r, record);
    unlock_registry(r);
    return reachable;
}
RcRegistryResult rc_registry_capture(RcRegistry *r, RcRegistryHandle handle, RcRegistryCapture *out) {
    if (!r || !out || out->id) return RC_REGISTRY_INVALID;
    lock_registry(r); Record *e = record_of(r, handle);
    if (!e) { unlock_registry(r); return RC_REGISTRY_STALE; }
    if (!r->free_capture || e->refs == UINT32_MAX) { unlock_registry(r); return RC_REGISTRY_CAPACITY; }
    uint64_t id = identity(r);
    if (!id) { unlock_registry(r); return RC_REGISTRY_EXHAUSTED; }
    uint32_t i = r->free_capture; r->free_capture = r->captures[i].free_next;
    r->captures[i] = (Capture){.id = id, .record = handle};
    e->refs++; r->stats.captures++;
    *out = (RcRegistryCapture){.id = id, .index = i, .record = record_value(e, handle.index)};
    unlock_registry(r); return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_capture_release(RcRegistry *r, RcRegistryCapture capture) {
    if (!r || !capture.id || !capture.index || capture.index > r->capacity.records) return RC_REGISTRY_INVALID;
    lock_registry(r); Capture *c = &r->captures[capture.index];
    if (c->id != capture.id || c->source) { unlock_registry(r); return RC_REGISTRY_STALE; }
    Record *e = record_of(r, c->record);
    if (!e || !e->refs) { poison(r); unlock_registry(r); return RC_REGISTRY_INVALID; }
    e->refs--; r->stats.captures--;
    *c = (Capture){.free_next = r->free_capture}; r->free_capture = capture.index;
    unlock_registry(r); return RC_REGISTRY_OK;
}

static int observation_current_locked(RcRegistry *r, const RcRegistryObservation *o) {
    Record *e = o ? record_of(r, o->handle) : NULL;
    return e && !r->stats.fail_closed && e->current && !e->finalized && e->revision == o->revision &&
        e->content_revision == o->content_revision && e->scope == o->scope && o->scope &&
        o->guard == r->guard && rc_range_equal(e->interval.range, o->range);
}
static RcRegistryResult observe_locked(RcRegistry *r, RcRegistryHandle handle, uint64_t scope, RcRegistryObservation *out) {
    if (out) *out = (RcRegistryObservation){0};
    if (!r || !out || !scope) return RC_REGISTRY_INVALID;
     Record *e = record_of(r, handle);
    if (!e || !e->current || e->finalized) { return RC_REGISTRY_STALE; }
    if (r->stats.fail_closed) {
        r->stats.blocked_fail_closed++;
        return RC_REGISTRY_BLOCKED;
    }
    SourceDiagnostic source = {.range = e->interval.range};
    if (source_authority_overlaps(r, e, observation_blocker, &source)) {
        return RC_REGISTRY_BLOCKED;
    }
    e->scope = scope; r->stats.observations++;
    *out = (RcRegistryObservation){.handle = handle, .revision = e->revision, .scope = scope,
        .guard = r->guard, .content_revision = e->content_revision, .range = e->interval.range};
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_observe(RcRegistry *r, RcRegistryHandle handle, uint64_t scope, RcRegistryObservation *out) {
    if (!r) return RC_REGISTRY_INVALID;
    lock_registry(r); RcRegistryResult result = observe_locked(r, handle, scope, out); unlock_registry(r); return result;
}
int rc_registry_observation_current(RcRegistry *r, const RcRegistryObservation *o) {
    if (!r || !o) return 0;
    lock_registry(r); int current = observation_current_locked(r, o); unlock_registry(r); return current;
}
static int content_current_locked(RcRegistry *r, const RcRegistryObservation *o) {
    Record *e = record_of(r, o->handle);
    return e && !r->stats.fail_closed && e->current && !e->finalized && o->content_revision &&
        o->guard && o->guard == r->guard && e->content_revision == o->content_revision &&
        rc_range_equal(e->interval.range, o->range);
}
int rc_registry_content_current(RcRegistry *r, const RcRegistryObservation *o) {
    if (!r || !o) return 0;
    lock_registry(r); int current = content_current_locked(r, o); unlock_registry(r); return current;
}
RcRegistryResult rc_registry_publish(RcRegistry *r, const RcRegistryObservation *o, uint64_t content,
                                     uint64_t representation, uint64_t payload) {
    if (!r || !o || !content || !representation || !payload) return RC_REGISTRY_INVALID;
    lock_registry(r);
    if (!observation_current_locked(r, o)) { unlock_registry(r); return RC_REGISTRY_STALE; }
    Record *e = &r->records[o->handle.index];
    e->content = content; e->representation = representation; e->payload = payload;
    e->version = 0;
    unlock_registry(r); return RC_REGISTRY_OK;
}

RcRegistryResult rc_registry_declare_through(RcRegistry *r, RcRegistryPosition position) {
    if (!r || !rc_registry_position_valid(position)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = RC_REGISTRY_OK;
    if (rc_registry_position_compare(position, r->declared) < 0) result = RC_REGISTRY_STALE;
    else r->declared = position;
    unlock_registry(r); return result;
}

RcRegistryResult rc_registry_reopen_unemitted(RcRegistry *r, RcRegistryPosition read) {
    if (!r || !rc_registry_position_valid(read) || (read.ordinal & 1)) return RC_REGISTRY_INVALID;
    RcRegistryPosition output = {read.scene, read.ordinal + 1};
    lock_registry(r);
    if (r->stats.fail_closed || rc_registry_position_compare(r->declared, output) ||
        rc_registry_position_compare(r->handed, read) >= 0) {
        unlock_registry(r);
        return RC_REGISTRY_STALE;
    }
    for (uint32_t i = 1; i <= r->capacity.writers; ++i) {
        const Writer *w = &r->writers[i];
        if (!w->state) continue;
        if (!rc_registry_position_compare(w->position, output) ||
            !rc_registry_position_compare(w->diagnostic_position, output)) {
            unlock_registry(r);
            return RC_REGISTRY_BUSY;
        }
        uint32_t j = image_position_lower(w, output);
        if (j < w->image_count && !rc_registry_position_compare(w->image_positions[j].position, output) &&
            (w->image_positions[j].unresolved || w->image_positions[j].active)) {
            unlock_registry(r);
            return RC_REGISTRY_BUSY;
        }
    }
    r->declared = read;
    unlock_registry(r);
    return RC_REGISTRY_OK;
}

int rc_registry_provisional_read_current(RcRegistry *r, RcRegistryPosition read) {
    if (!r || !rc_registry_position_valid(read) || (read.ordinal & 1)) return 0;
    lock_registry(r);
    int current = !r->stats.fail_closed && !rc_registry_position_compare(r->declared, read) &&
                  rc_registry_position_compare(r->handed, read) < 0;
    unlock_registry(r);
    return current;
}

RcRegistryResult rc_registry_handoff_through(RcRegistry *r, RcRegistryPosition position) {
    if (!r || !rc_registry_position_valid(position)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = RC_REGISTRY_OK;
    if (rc_registry_position_compare(position, r->handed) < 0 ||
        rc_registry_position_compare(position, r->declared) > 0) result = RC_REGISTRY_STALE;
    else r->handed = position;
    unlock_registry(r); return result;
}
int rc_registry_position_handed(RcRegistry *r, RcRegistryPosition position) {
    if (!r || !rc_registry_position_valid(position)) return 0;
    lock_registry(r);
    int handed = rc_registry_position_compare(position, r->handed) <= 0;
    unlock_registry(r);
    return handed;
}
RcRegistryPosition rc_registry_handed_cutoff(RcRegistry *r) {
    if (!r) return (RcRegistryPosition){0};
    lock_registry(r);
    const RcRegistryPosition cut = r->stats.fail_closed ? (RcRegistryPosition){0} : r->handed;
    unlock_registry(r);
    return cut;
}

static int read_position_open(RcRegistry *r, RcRegistryPosition position) {
    return rc_registry_position_valid(position) && rc_registry_position_compare(position, r->declared) <= 0 &&
           rc_registry_position_compare(position, r->handed) > 0;
}
int rc_registry_position_open(RcRegistry *r, RcRegistryPosition position) {
    if (!r) return 0;
    lock_registry(r); int open = !r->stats.fail_closed && read_position_open(r, position);
    unlock_registry(r); return open;
}
static int future_pending_writer(const Writer *w, RcRegistryPosition position) {
    return w->state == WRITE_PENDING && rc_registry_position_valid(w->position) &&
           rc_registry_position_compare(w->position, position) > 0;
}
typedef struct { RcRegistryPosition position, next_write; RcSourceRange range; } OrderedRead;
static int ordered_read_blocker(RcRegistry *r, uint32_t i, void *data) {
    OrderedRead *read = data;
    const RcRegistryPosition *position = &read->position;
    const Writer *w = &r->writers[i];
    if (future_pending_writer(w, *position)) {
        r->stats.future_writers_ignored++;
        if (!rc_registry_position_valid(read->next_write) ||
            rc_registry_position_compare(w->position, read->next_write) < 0) read->next_write = w->position;
        return 0;
    }
    SourceDiagnostic source = {.range = read->range, .position = read->position};
    return observation_blocker(r, i, &source);
}

static int diagnose_read_blocker(RcRegistry *r, uint32_t i, void *data) {
    RcRegistryReadBlocker *out = data;
    const Writer *w = &r->writers[i];
    RcSourceRange requested = out->sample.requested;
    RcRegistryPosition read_position = out->sample.read_position;
    if (rc_registry_position_valid(read_position) && future_pending_writer(w, read_position)) return 0;
    diagnostic_sample(&out->sample, i, w, requested, read_position, (RcRegistryImageBinding){0}, NULL, 0);
    out->reason = (RcRegistryDiagnosticReason)((w->state - WRITE_PENDING) * 3 + w->kind);
    out->writer_address = w->image_address;
    out->writer_mapping = w->image_mapping;
    out->origin_position = w->diagnostic_position;
    out->writer_origin = w->diagnostic_origin;
    out->guard_reason = w->diagnostic_guard_reason;
    return 1;
}

int rc_registry_diagnose_read_blocker(RcRegistry *r, RcSourceRange range, RcRegistryPosition position,
                                      RcRegistryReadBlocker *out) {
    if (!out) return 0;
    *out = (RcRegistryReadBlocker){.sample = {.requested = range, .read_position = position}};
    if (!r || !rc_range_valid(range)) return 0;
    lock_registry(r);
    int blocked = authority_overlaps(r, range, diagnose_read_blocker, out);
    unlock_registry(r);
    return blocked;
}
static int read_observation_current_locked(RcRegistry *r, const RcRegistryReadObservation *o) {
    Record *e = o ? record_of(r, o->handle) : NULL;
    // A sealed declaration prefix cannot gain an earlier positioned producer. Unpositioned changes, actual CPU
    // visibility and mapping expansion advance content_revision. Future declarations only change generic revision.
    return e && !r->stats.fail_closed && e->current && !e->finalized && o->scope && o->guard == r->guard &&
           e->content_revision == o->content_revision && e->ordered_revision == o->revision &&
           rc_registry_position_valid(o->sealed_through) &&
           rc_registry_position_compare(o->position, o->sealed_through) <= 0 &&
           rc_range_equal(e->interval.range, o->range) &&
           read_position_open(r, o->position);
}
static RcRegistryResult observe_at_locked(RcRegistry *r, RcRegistryHandle handle, uint64_t scope,
                                       RcRegistryPosition position, RcRegistryReadObservation *out) {
    if (out) *out = (RcRegistryReadObservation){0};
    if (!r || !out || !scope || !rc_registry_position_valid(position)) return RC_REGISTRY_INVALID;
    
    Record *e = record_of(r, handle);
    if (!e || !e->current || e->finalized) { return RC_REGISTRY_STALE; }
    if (r->stats.fail_closed || !read_position_open(r, position) ||
        !rc_registry_position_valid(rc_registry_sealed_deadline(r->declared))) {
        r->stats.ordered_blocked++;
        if (r->stats.fail_closed) r->stats.blocked_fail_closed++;
        return RC_REGISTRY_BLOCKED;
    }
    OrderedRead read = {.position = position, .range = e->interval.range};
    if (source_authority_overlaps(r, e, ordered_read_blocker, &read)) {
        r->stats.ordered_blocked++;
        return RC_REGISTRY_BLOCKED;
    }
    *out = (RcRegistryReadObservation){.handle = handle, .content_revision = e->content_revision,
        .scope = scope, .guard = r->guard, .range = e->interval.range, .position = position,
        .revision = e->ordered_revision, .topology_revision = e->revision,
        .next_write = read.next_write, .sealed_through = r->declared};
    r->stats.ordered_observations++;
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_observe_at(RcRegistry *r, RcRegistryHandle handle, uint64_t scope, RcRegistryPosition position, RcRegistryReadObservation *out) {
    if (!r) return RC_REGISTRY_INVALID;
    lock_registry(r); RcRegistryResult result = observe_at_locked(r, handle, scope, position, out); unlock_registry(r); return result;
}
int rc_registry_read_observation_current(RcRegistry *r, const RcRegistryReadObservation *o) {
    if (!r || !o) return 0;
    lock_registry(r); int result = read_observation_current_locked(r, o); unlock_registry(r); return result;
}
static RcRegistryResult read_revalidate_locked(RcRegistry *r, const RcRegistryReadObservation *prior,
    uint64_t scope, RcRegistryPosition position, RcRegistryReadObservation *out) {
    if (!r || !prior || !out || !scope) return RC_REGISTRY_INVALID;
    RcRegistryReadObservation old = *prior; // in-place update is allowed
    *out = (RcRegistryReadObservation){0};
    
    Record *e = record_of(r, old.handle);
    int valid = e && !r->stats.fail_closed && e->current && !e->finalized && old.scope && old.guard == r->guard &&
        e->content_revision == old.content_revision && e->ordered_revision == old.revision &&
        rc_range_equal(e->interval.range, old.range) && read_position_open(r, position) &&
        rc_registry_position_valid(old.sealed_through) &&
        rc_registry_position_compare(position, old.sealed_through) <= 0 &&
        rc_registry_position_compare(position, old.position) >= 0 &&
        (!rc_registry_position_valid(old.next_write) || rc_registry_position_compare(position, old.next_write) < 0);
    if (valid) { *out = old; out->scope = scope; out->position = position; }
    return valid ? RC_REGISTRY_OK : RC_REGISTRY_STALE;
}
RcRegistryResult rc_registry_read_revalidate(RcRegistry *r, const RcRegistryReadObservation *prior, uint64_t scope, RcRegistryPosition position, RcRegistryReadObservation *out) {
    if (!r) return RC_REGISTRY_INVALID;
    lock_registry(r); RcRegistryResult result = read_revalidate_locked(r, prior, scope, position, out); unlock_registry(r); return result;
}
static RcRegistryResult read_renew_locked(RcRegistry *r, const RcRegistryReadObservation *prior,
    uint64_t scope, RcRegistryPosition position, RcRegistryReadObservation *out) {
    if (!r || !prior || !out || !scope) return RC_REGISTRY_INVALID;
    RcRegistryReadObservation old = *prior; *out = (RcRegistryReadObservation){0};
     Record *e = record_of(r, old.handle);
    if (!e || !e->current || e->finalized || !old.content_revision || old.content_revision != e->content_revision ||
        !old.guard || old.guard != r->guard || !rc_range_equal(old.range, e->interval.range) ||
        old.scope != scope || !rc_registry_position_valid(old.position) || position.scene != old.position.scene ||
        rc_registry_position_compare(position, old.position) < 0 || !rc_registry_position_valid(old.sealed_through) ||
        rc_registry_position_compare(old.position, old.sealed_through) > 0) {
        return RC_REGISTRY_STALE;
    }
    if (r->stats.fail_closed || !read_position_open(r, position) ||
        !rc_registry_position_valid(rc_registry_sealed_deadline(r->declared))) {
        return RC_REGISTRY_BLOCKED;
    }
    if (old.topology_revision && old.topology_revision == e->revision && old.revision == e->ordered_revision &&
        (!rc_registry_position_valid(old.next_write) || rc_registry_position_compare(position, old.next_write) < 0)) {
        *out = old; out->position = position; out->sealed_through = r->declared;
        r->stats.read_renew_fast++;
        return RC_REGISTRY_OK;
    }
    r->stats.read_renew_queries++;
    OrderedRead read = {.position = position, .range = e->interval.range};
    if (source_authority_overlaps(r, e, ordered_read_blocker, &read)) {
        r->stats.ordered_blocked++; return RC_REGISTRY_BLOCKED;
    }
    *out = (RcRegistryReadObservation){.handle = old.handle, .content_revision = e->content_revision,
        .scope = scope, .guard = r->guard, .range = e->interval.range, .position = position,
        .revision = e->ordered_revision, .topology_revision = e->revision,
        .next_write = read.next_write, .sealed_through = r->declared};
    r->stats.ordered_observations++;
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_read_renew(RcRegistry *r, const RcRegistryReadObservation *prior, uint64_t scope, RcRegistryPosition position, RcRegistryReadObservation *out) {
    if (!r) return RC_REGISTRY_INVALID;
    lock_registry(r); RcRegistryResult result = read_renew_locked(r, prior, scope, position, out); unlock_registry(r); return result;
}
RcRegistryResult rc_registry_read_range_at(RcRegistry *r, RcSourceRange range, RcRegistryPosition position) {
    if (!r || !rc_range_valid(range) || !rc_registry_position_valid(position)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    if (r->stats.fail_closed || !read_position_open(r, position)) {
        r->stats.ordered_blocked++;
        unlock_registry(r); return RC_REGISTRY_BLOCKED;
    }
    OrderedRead read = {.position = position, .range = range};
    int blocked = authority_overlaps(r, range, ordered_read_blocker, &read);
    if (blocked) r->stats.ordered_blocked++;
    else r->stats.ordered_observations++;
    unlock_registry(r); return blocked ? RC_REGISTRY_BLOCKED : RC_REGISTRY_OK;
}

static int device_buffer_read_blocker(RcRegistry *r, uint32_t index, void *data) {
    const Writer *writer = &r->writers[index];
    const RcRegistryPosition *read = data;
    if (writer->state == WRITE_PENDING && rc_registry_position_valid(writer->position) &&
        rc_registry_position_compare(writer->position, *read) > 0) return 0;
    // A native copy sees ordered writes to this very allocation, including completed device-only writes.
    // Unknown ranges, image ownership and unpositioned/mixed effects do not identify that provider.
    return writer->guard || writer->unordered || writer->state == WRITE_HOLD ||
        writer->kinds != (1u << RC_REGISTRY_BUFFER) ||
        !rc_registry_position_valid(writer->visibility_latest) ||
        rc_registry_position_compare(writer->visibility_latest, *read) >= 0;
}

RcRegistryResult rc_registry_read_device_buffer_at(RcRegistry *r, const RcInterval *ranges, uint32_t count,
                                                   RcRegistryPosition read) {
    if (!r || !ranges || !count || !rc_registry_position_valid(read)) return RC_REGISTRY_INVALID;
    for (uint32_t i = 0; i < count; i++)
        if (!rc_range_valid(ranges[i].range) || !ranges[i].mapping || ranges[i].range.backing == RC_GUEST_COMPATIBILITY_BACKING)
            return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = RC_REGISTRY_OK;
    if (r->stats.fail_closed || !read_position_open(r, read)) result = RC_REGISTRY_BLOCKED;
    for (uint32_t i = 0; i < count && result == RC_REGISTRY_OK; i++)
        if (authority_overlaps(r, ranges[i].range, device_buffer_read_blocker, &read)) result = RC_REGISTRY_BLOCKED;
    unlock_registry(r);
    return result;
}

static int buffer_phase_interval_valid(RcInterval input) {
    return input.mapping && input.range.backing != RC_GUEST_COMPATIBILITY_BACKING && rc_range_valid(input.range) &&
        input.range.size <= UINT64_MAX - input.address;
}

static int buffer_phase_read_valid(RcRegistryPosition read) {
    return rc_registry_position_valid(read) && !(read.ordinal & 1);
}

static RcRegistryResult buffer_phase_validate(RcRegistry *r, const RcRegistryBufferPhaseExtent *extents, uint32_t count,
    RcRegistryPosition read, unsigned expected) {
    RcRegistryPosition output = {read.scene, read.ordinal + 1};
    RcRegistryPosition boundary = expected ? output : read;
    if (r->stats.fail_closed || !read_position_open(r, read) || rc_registry_position_compare(r->declared, boundary))
        return RC_REGISTRY_STALE;
    for (uint32_t i = 0; i < count; ++i) {
        const RcRegistryBufferPhaseExtent *extent = &extents[i];
        const RcInterval interval = extent->interval;
        const Writer *w = writer_of(r, extent->writer);
        if (!buffer_phase_interval_valid(interval)) return RC_REGISTRY_INVALID;
        if (!w || w->state != WRITE_PENDING || w->kind != RC_REGISTRY_BUFFER || w->guard ||
            w->buffer_phase != expected || rc_registry_position_compare(w->position, output) ||
            !rc_range_equal(w->interval.range, interval.range) ||
            (expected && (w->phase_address != interval.address || w->phase_mapping != interval.mapping)))
            return RC_REGISTRY_STALE;
        for (uint32_t j = 0; j < i; ++j)
            if (extents[j].writer.index == extent->writer.index) return RC_REGISTRY_INVALID;
    }
    return RC_REGISTRY_OK;
}

RcRegistryResult rc_registry_buffer_phase_bind(RcRegistry *r, const RcRegistryBufferPhaseExtent *extents, uint32_t count,
                                               RcRegistryPosition read) {
    if (!r || !extents || !count || count > r->capacity.writers || !buffer_phase_read_valid(read))
        return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = buffer_phase_validate(r, extents, count, read, 0);
    if (result == RC_REGISTRY_OK) for (uint32_t i = 0; i < count; ++i) {
        Writer *w = writer_of(r, extents[i].writer);
        w->buffer_phase = BUFFER_PHASE_BOUND;
        w->phase_address = extents[i].interval.address;
        w->phase_mapping = extents[i].interval.mapping;
    }
    unlock_registry(r);
    return result;
}

RcRegistryResult rc_registry_buffer_phase_note(RcRegistry *r, const RcRegistryBufferPhaseExtent *extents, uint32_t count,
                                               RcRegistryPosition read, RcRegistryBufferPhaseOutcome outcome) {
    if (!r || !extents || !count || count > r->capacity.writers || !buffer_phase_read_valid(read) ||
        (outcome != RC_REGISTRY_BUFFER_PHASE_POSSIBLY_ISSUED && outcome != RC_REGISTRY_BUFFER_PHASE_COMPLETED))
        return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = buffer_phase_validate(r, extents, count, read, BUFFER_PHASE_BOUND);
    if (result == RC_REGISTRY_OK) for (uint32_t i = 0; i < count; ++i) {
        Writer *w = writer_of(r, extents[i].writer);
        w->buffer_phase = outcome == RC_REGISTRY_BUFFER_PHASE_COMPLETED ? BUFFER_PHASE_DONE : BUFFER_PHASE_MAYBE;
    }
    unlock_registry(r);
    return result;
}

RcRegistryResult rc_registry_read_device_buffer_phase(RcRegistry *r, const RcInterval *input, uint32_t input_count,
    const RcRegistryBufferPhaseExtent *extents, uint32_t count, RcRegistryPosition read) {
    if (!r || !input || !input_count || !extents || !count || count > r->capacity.writers || !buffer_phase_read_valid(read))
        return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = buffer_phase_validate(r, extents, count, read, BUFFER_PHASE_DONE);
    for (uint32_t i = 0; i < input_count && result == RC_REGISTRY_OK; ++i) {
        const RcSourceRange requested = input[i].range;
        if (!buffer_phase_interval_valid(input[i])) { result = RC_REGISTRY_INVALID; break; }
        uint32_t j = 0;
        for (; j < count; ++j) {
            RcSourceRange written = extents[j].interval.range;
            if (written.backing == requested.backing && requested.offset >= written.offset &&
                requested.offset + requested.size <= written.offset + written.size) break;
        }
        if (j == count || authority_overlaps(r, requested, device_buffer_read_blocker, &read)) result = RC_REGISTRY_BLOCKED;
    }
    unlock_registry(r);
    return result;
}

static uint32_t gpu_capture_range(const GpuCapture *capture, RcSourceRange range) {
    uint32_t lo = 0, hi = capture->count;
    while (lo < hi) {
        uint32_t at = lo + (hi - lo) / 2;
        RcSourceRange span = capture->coverage[at];
        if (span.backing < range.backing ||
            (span.backing == range.backing && span.offset + span.size <= range.offset)) lo = at + 1;
        else hi = at;
    }
    return lo;
}

static int gpu_capture_overlaps(const GpuCapture *capture, RcSourceRange range) {
    uint32_t at = gpu_capture_range(capture, range);
    return at < capture->count && rc_range_overlap(capture->coverage[at], range);
}

static int gpu_capture_covers(const GpuCapture *capture, RcSourceRange range) {
    uint32_t at = gpu_capture_range(capture, range);
    if (at == capture->count) return 0;
    RcSourceRange span = capture->coverage[at];
    return span.backing == range.backing && span.offset <= range.offset &&
        range.offset + range.size <= span.offset + span.size;
}

// Keep only visibility transitions relevant to an outstanding copy. The live writer tree remains the
// authority for pending and foreign bytes, and finished history never consumes recording capacity.
typedef struct { RcRegistryPosition output; uint64_t image; } NativeImageOutput;

static int native_image_conflict(RcRegistry *r, uint32_t i, void *data) {
    const NativeImageOutput *output = data;
    Writer *held = &r->writers[i];
    if (held->state == WRITE_HOLD && held->image && held->image != output->image &&
        rc_registry_position_valid(held->native_output) &&
        (!rc_registry_position_valid(output->output) ||
         rc_registry_position_compare(output->output, held->native_output) >= 0))
        held->native_conflict = 1;
    return 0;
}

static int gpu_visibility_payload_needed(RcRegistry *r, RcSourceRange range, uint64_t sequence) {
    for (uint32_t i = r->active_gpu_captures; i; i = r->gpu_captures[i].next) {
        const GpuCapture *capture = &r->gpu_captures[i];
        if (!capture->history_lost && capture->before_copy <= sequence && gpu_capture_overlaps(capture, range))
            return 1;
    }
    return 0;
}

static void gpu_visibility_trim_payloads(RcRegistry *r) {
    // The ordinary visibility journal stays available to produced-scene readers. Its extra payload
    // references belong only to the GPU captures that started before the corresponding change.
    for (uint32_t i = r->gpu_payload_head; i;) {
        const GpuVisibility *entry = &r->gpu_visibility[i - 1];
        const uint32_t next = entry->payload_next;
        if (!gpu_visibility_payload_needed(r, entry->range, entry->payload_sequence)) gpu_visibility_payload_drop(r, i);
        i = next;
    }
}

static void gpu_visibility_note_payload(RcRegistry *r, RcSourceRange range, RcRegistryPosition producer,
    RcRegistryPosition latest, RcRegistryBufferLocation native, BufferPayload payload) {
    if (range.backing == RC_GUEST_COMPATIBILITY_BACKING) return;
    native_projection_invalidate(r, range, latest, NULL, !native.allocation);
    if (r->native_image_outputs) {
        // Completing an actual visible overwrite removes its writer from the tree. A live bound image keeps
        // the evidence of that overwrite until its next recorded write; a new read cannot resurrect it.
        NativeImageOutput output = {latest, 0};
        authority_overlaps(r, range, native_image_conflict, &output);
    }
    int produced = r->produced_scene && (!rc_registry_position_valid(latest) || latest.scene >= r->produced_scene) &&
        (!rc_registry_position_valid(producer) || producer.scene <= r->produced_scene);
    uint32_t i = r->active_gpu_captures;
    if (!produced) {
        for (; i; i = r->gpu_captures[i].next)
            if (gpu_capture_overlaps(&r->gpu_captures[i], range)) break;
        if (!i) return;
    }
    uint64_t sequence = r->gpu_visible_sequence;
    if (sequence == UINT64_MAX) {
        if (r->produced_scene) r->produced_lost_unknown = 1;
        for (i = r->active_gpu_captures; i; i = r->gpu_captures[i].next) r->gpu_captures[i].history_lost = 1;
        gpu_visibility_trim_payloads(r);
        return;
    }
    const uint32_t slot = (uint32_t)(sequence % r->capacity.records) + 1;
    GpuVisibility *entry = &r->gpu_visibility[slot - 1];
    int history_lost = 0;
    if (sequence >= r->capacity.records) {
        uint64_t overwritten = sequence - r->capacity.records;
        if (r->produced_scene && r->produced_begin <= overwritten) {
            if (!rc_registry_position_valid(entry->latest)) r->produced_lost_unknown = 1;
            else if (rc_registry_position_compare(entry->latest, r->produced_lost) > 0)
                r->produced_lost = entry->latest;
        }
        for (i = r->active_gpu_captures; i; i = r->gpu_captures[i].next) {
            GpuCapture *capture = &r->gpu_captures[i];
            if (!capture->history_lost && capture->before_copy <= overwritten && gpu_capture_overlaps(capture, entry->range)) {
                capture->history_lost = 1;
                history_lost = 1;
            }
        }
    }
    gpu_visibility_payload_drop(r, slot);
    *entry = (GpuVisibility){.range=range, .producer=producer, .latest=latest, .native_buffer=native};
    r->gpu_visible_sequence++;
    if (history_lost) gpu_visibility_trim_payloads(r);
    if (payload.owner && native.allocation && gpu_visibility_payload_needed(r, range, sequence)) {
        payload.owner->retain(payload.owner);
        entry->payload = payload;
        entry->payload_sequence = sequence;
        entry->payload_next = r->gpu_payload_head;
        if (r->gpu_payload_head) r->gpu_visibility[r->gpu_payload_head - 1].payload_prev = slot;
        r->gpu_payload_head = slot;
    }
}

static void gpu_visibility_note_native(RcRegistry *r, RcSourceRange range, RcRegistryPosition producer,
                                       RcRegistryPosition latest, RcRegistryBufferLocation native) {
    gpu_visibility_note_payload(r, range, producer, latest, native, (BufferPayload){0});
}

static RcRegistryPosition gpu_visibility_earliest(RcRegistryPosition a, RcRegistryPosition b) {
    if (!rc_registry_position_valid(a) || !rc_registry_position_valid(b)) return (RcRegistryPosition){0};
    return rc_registry_position_compare(a, b) <= 0 ? a : b;
}

static RcRegistryPosition gpu_visibility_latest(RcRegistryPosition a, RcRegistryPosition b) {
    if (!rc_registry_position_valid(a) || !rc_registry_position_valid(b)) return (RcRegistryPosition){0};
    return rc_registry_position_compare(a, b) >= 0 ? a : b;
}

static int gpu_visibility_allocate(RcRegistry *r) {
    if (r->gpu_visibility) return 1;
    r->gpu_visibility = calloc(r->capacity.records, sizeof *r->gpu_visibility);
    return r->gpu_visibility != NULL;
}

static GpuCapture *gpu_capture_of(RcRegistry *r, RcRegistryGpuCapture token) {
    if (!r->gpu_captures || !token.id || !token.index || token.index > r->capacity.scopes) return NULL;
    GpuCapture *capture = &r->gpu_captures[token.index];
    return capture->id == token.id ? capture : NULL;
}

static void gpu_visibility_note(RcRegistry *r, RcSourceRange range, RcRegistryPosition producer,
                                RcRegistryPosition latest) {
    gpu_visibility_note_native(r, range, producer, latest, (RcRegistryBufferLocation){0});
}

RcRegistryResult rc_registry_gpu_capture_reserve(RcRegistry *r, const RcSourceRange *coverage, uint32_t count,
                                                 uint32_t foreground_slots, RcRegistryGpuCapture *out,
                                                 uint64_t *release_epoch) {
    if (release_epoch) *release_epoch = 0;
    if (!r || !out || out->id || out->index || !coverage || !count) return RC_REGISTRY_INVALID;
    if (foreground_slots >= r->capacity.scopes) return RC_REGISTRY_CAPACITY;
    for (uint32_t i = 0; i < count; i++) {
        RcSourceRange range = coverage[i];
        if (!rc_range_valid(range) || range.backing == RC_GUEST_COMPATIBILITY_BACKING) return RC_REGISTRY_INVALID;
        if (i && (coverage[i - 1].backing > range.backing || (coverage[i - 1].backing == range.backing &&
            coverage[i - 1].offset + coverage[i - 1].size > range.offset))) return RC_REGISTRY_INVALID;
    }
    lock_registry(r);
    RcRegistryResult result = RC_REGISTRY_OK;
    if (r->stats.fail_closed) result = RC_REGISTRY_BLOCKED;
    else if (r->gpu_visible_sequence == UINT64_MAX) result = RC_REGISTRY_EXHAUSTED;
    else if (!r->gpu_captures) {
        GpuCapture *captures = calloc((size_t)r->capacity.scopes + 1, sizeof *captures);
        if (!captures || !gpu_visibility_allocate(r)) {
            free(captures); result = RC_REGISTRY_CAPACITY;
        } else r->gpu_captures = captures;
    }
    if (result == RC_REGISTRY_OK) {
        uint32_t i = r->capacity.scopes - foreground_slots;
        while (i && r->gpu_captures[i].id) i--;
        if (!i) {
            result = RC_REGISTRY_BUSY;
            if (release_epoch) *release_epoch = r->gpu_capture_release_epoch;
        } else {
            uint64_t id = identity(r);
            if (!id) result = RC_REGISTRY_EXHAUSTED;
            else {
                r->gpu_captures[i] = (GpuCapture){.coverage = coverage, .id = id, .count = count};
                *out = (RcRegistryGpuCapture){id, i};
            }
        }
    }
    unlock_registry(r);
    return result;
}

void rc_registry_gpu_capture_wait(RcRegistry *r, uint64_t release_epoch, uint32_t milliseconds) {
#ifndef RC_REGISTRY_SINGLE_THREADED
    if (!r || !milliseconds) return;
    lock_registry(r);
    if (release_epoch == r->gpu_capture_release_epoch)
        plat_cond_wait_ms(&r->gpu_capture_released, &r->lock, milliseconds);
    unlock_registry(r);
#else
    (void)r; (void)release_epoch; (void)milliseconds;
#endif
}

RcRegistryResult rc_registry_gpu_capture_start(RcRegistry *r, RcRegistryGpuCapture token) {
    if (!r) return RC_REGISTRY_INVALID;
    lock_registry(r);
    GpuCapture *capture = gpu_capture_of(r, token);
    RcRegistryResult result = RC_REGISTRY_OK;
    if (!capture) result = RC_REGISTRY_STALE;
    else if (capture->active) result = RC_REGISTRY_INVALID;
    else if (r->stats.fail_closed) result = RC_REGISTRY_BLOCKED;
    else if (r->gpu_visible_sequence == UINT64_MAX) result = RC_REGISTRY_EXHAUSTED;
    else {
        capture->active = 1;
        capture->before_copy = r->gpu_visible_sequence;
        capture->next = r->active_gpu_captures;
        if (capture->next) r->gpu_captures[capture->next].prev = token.index;
        r->active_gpu_captures = token.index;
    }
    unlock_registry(r);
    return result;
}

RcRegistryResult rc_registry_gpu_capture_begin(RcRegistry *r, const RcSourceRange *coverage, uint32_t count,
                                               RcRegistryGpuCapture *out) {
    RcRegistryResult result = rc_registry_gpu_capture_reserve(r, coverage, count, 0, out, NULL);
    if (result == RC_REGISTRY_BUSY) return RC_REGISTRY_CAPACITY;
    if (result != RC_REGISTRY_OK) return result;
    result = rc_registry_gpu_capture_start(r, *out);
    if (result != RC_REGISTRY_OK) rc_registry_gpu_capture_end(r, out);
    return result;
}

typedef struct {
    RcRegistryPosition read;
    int blocked, ordered, classify;
} GpuCaptureBlocker;

static int gpu_capture_live_blocker(RcRegistry *r, uint32_t i, void *data) {
    const Writer *writer = &r->writers[i];
    GpuCaptureBlocker *blocker = data;
    int positioned = rc_registry_position_valid(writer->position);
    int earlier = positioned && rc_registry_position_compare(writer->position, blocker->read) < 0;
    if (writer->state == WRITE_PENDING && positioned && !earlier) return 0;
    blocker->blocked = 1;
    blocker->ordered &= earlier && !writer->unordered && !writer->guard && writer->kind == RC_REGISTRY_BUFFER &&
        (writer->state == WRITE_PENDING || writer->state == WRITE_SUBMITTED);
    return !blocker->classify || !blocker->ordered;
}

static int buffer_native_span_valid(RcRegistryBufferSpan span) {
    return buffer_phase_interval_valid(span.interval) && span.native.allocation &&
        span.interval.range.size <= UINT64_MAX - span.native.offset;
}

static int buffer_native_equal(RcRegistryBufferLocation a, RcRegistryBufferLocation b) {
    return a.allocation == b.allocation && a.incarnation == b.incarnation && a.offset == b.offset;
}

static RcRegistryResult buffer_result_record_locked(RcRegistry *r, Writer *w, RcRegistryBufferSpan span,
                                                     RcRegistryPosition output) {
    RcRegistryResult result = RC_REGISTRY_STALE;
    if (!r->stats.fail_closed && read_position_open(r, output) && w && w->state == WRITE_PENDING && !w->guard &&
        w->kind == RC_REGISTRY_BUFFER && !rc_registry_position_compare(w->position, output) &&
        rc_range_equal(w->interval.range, span.interval.range)) {
        if (!w->native_buffer.allocation || buffer_native_equal(w->native_buffer, span.native)) {
            if (!w->native_buffer.allocation) {
                w->native_buffer_address = span.interval.address;
                w->native_buffer_mapping = span.interval.mapping;
            } else if (w->native_buffer_address != span.interval.address ||
                       w->native_buffer_mapping != span.interval.mapping) {
                // Multiple guest aliases can still identify one physical result, but cannot certify a
                // stable affine mapping for coalescing separate ordered writes.
                w->native_buffer_mapping = 0;
            }
            w->native_buffer = span.native;
            result = RC_REGISTRY_OK;
        } else {
            // A conflicting native destination is an unsupported projection. Preserve the actual writer exclusion.
            w->native_buffer = (RcRegistryBufferLocation){0};
            w->native_buffer_conflict = 1;
            result = RC_REGISTRY_BLOCKED;
        }
        if (w->native_buffer_conflict) {
            w->native_buffer = (RcRegistryBufferLocation){0};
            buffer_payload_drop(&w->native_payload);
            result = RC_REGISTRY_BLOCKED;
        }
    }
    return result;
}

RcRegistryResult rc_registry_buffer_write_record(RcRegistry *r, RcRegistryWrite ticket, RcRegistryBufferSpan span,
                                                  RcRegistryPosition output) {
    if (!r || !buffer_native_span_valid(span) || !rc_registry_position_valid(output) || !(output.ordinal & 1))
        return RC_REGISTRY_INVALID;
    lock_registry(r);
    Writer *w = writer_of(r, ticket);
    RcRegistryResult result = buffer_result_record_locked(r, w, span, output);
    unlock_registry(r);
    return result;
}

RcRegistryResult rc_registry_buffer_write_payload_record(RcRegistry *r, RcRegistryWrite ticket, RcRegistryBufferSpan span,
    RcRegistryPosition output, RcRegistryBufferPayload *payload) {
    if (!r || !payload || !payload->retain || !payload->release || !buffer_native_span_valid(span) ||
        !rc_registry_position_valid(output) || !(output.ordinal & 1)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    Writer *w = writer_of(r, ticket);
    RcRegistryResult result = buffer_result_record_locked(r, w, span, output);
    if (result == RC_REGISTRY_OK) {
        BufferPayload *current = &w->native_payload;
        if (w->native_payload_conflict || (current->owner &&
            (current->owner != payload || current->address != span.interval.address ||
             current->mapping != span.interval.mapping))) {
            buffer_payload_drop(current);
            w->native_payload_conflict = 1;
            result = RC_REGISTRY_BLOCKED;
        } else if (!current->owner) {
            payload->retain(payload);
            *current = (BufferPayload){payload, span.interval.address, span.interval.mapping};
        }
    }
    unlock_registry(r);
    return result;
}

RcRegistryResult rc_registry_buffer_access_record(RcRegistry *r, RcRegistryWrite ticket, RcRegistryBufferSpan span,
                                                   RcRegistryPosition output) {
    if (!r || !buffer_native_span_valid(span) || !rc_registry_position_valid(output) || !(output.ordinal & 1))
        return RC_REGISTRY_INVALID;
    lock_registry(r);
    Writer *w = writer_of(r, ticket);
    RcRegistryResult result = RC_REGISTRY_STALE;
    int interval_matches = w && rc_range_equal(w->interval.range, span.interval.range);
    if (w && w->interval.range.backing == RC_GUEST_COMPATIBILITY_BACKING)
        interval_matches = w->interval.range.offset == span.interval.address &&
            w->interval.range.size == span.interval.range.size;
    if (!r->stats.fail_closed && read_position_open(r, output) && w && w->state == WRITE_PENDING && !w->guard &&
        w->kind == RC_REGISTRY_BUFFER && !w->unordered && w->kinds == (1u << RC_REGISTRY_BUFFER) &&
        !rc_registry_position_compare(w->position, output) && interval_matches) {
        if (!w->native_access_recorded || (buffer_native_equal(w->native_access, span.native) &&
                                            w->native_access_mapping == span.interval.mapping &&
                                            rc_range_equal(w->native_access_range, span.interval.range) &&
                                            w->native_access_address == span.interval.address)) {
            w->native_access = span.native;
            w->native_access_output = output;
            w->native_access_range = span.interval.range;
            w->native_access_address = span.interval.address;
            w->native_access_mapping = span.interval.mapping;
            w->native_access_recorded = 1;
            result = RC_REGISTRY_OK;
        } else {
            w->native_access = (RcRegistryBufferLocation){0};
            w->native_access_conflict = 1;
            result = RC_REGISTRY_BLOCKED;
        }
        if (w->native_access_conflict) {
            w->native_access = (RcRegistryBufferLocation){0};
            result = RC_REGISTRY_BLOCKED;
        }
    }
    unlock_registry(r);
    return result;
}

// 1 matches, 0 is incomplete/foreign authority, -1 is a changed mapping or stale access receipt.
static int buffer_access_matches(const Writer *w, RcInterval wanted, RcRegistryBufferLocation expected) {
    if (w->guard || w->unordered || w->kind != RC_REGISTRY_BUFFER ||
        w->kinds != (1u << RC_REGISTRY_BUFFER) || w->state == WRITE_HOLD ||
        w->state == WRITE_FOREIGN || !w->native_access_recorded || w->native_access_conflict) return 0;
    if (rc_registry_position_compare(w->native_access_output, w->position)) return -1;
    if (w->native_access_range.backing != wanted.range.backing || w->native_access_mapping != wanted.mapping)
        return -1;
    if (w->interval.range.backing == RC_GUEST_COMPATIBILITY_BACKING &&
        (w->interval.range.offset != w->native_access_address ||
         w->interval.range.size != w->native_access_range.size)) return 0;
    uint64_t lo = w->native_access_range.offset > wanted.range.offset ?
        w->native_access_range.offset : wanted.range.offset;
    uint64_t we = w->native_access_range.offset + w->native_access_range.size;
    uint64_t qe = wanted.range.offset + wanted.range.size;
    uint64_t hi = we < qe ? we : qe;
    if (lo >= hi) return 1;
    if (w->native_access.offset > UINT64_MAX - (lo - w->native_access_range.offset)) return 0;
    RcRegistryBufferLocation actual = w->native_access;
    actual.offset += lo - w->native_access_range.offset;
    uint64_t delta = lo - wanted.range.offset;
    if (expected.offset > UINT64_MAX - delta) return 0;
    expected.offset += delta;
    return buffer_native_equal(actual, expected);
}

RcRegistryResult rc_registry_buffer_access_check(RcRegistry *r, RcRegistryBufferSpan span, RcRegistryPosition read) {
    if (!r || !buffer_native_span_valid(span) || !buffer_phase_read_valid(read)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = RC_REGISTRY_OK;
    if (r->stats.fail_closed) result = RC_REGISTRY_BLOCKED;
    else if (!read_position_open(r, read)) result = RC_REGISTRY_STALE;
    else {
        // Opaque guards do not necessarily have a range node in this canonical backing. Match the registry's
        // authority-overlap rule: any global guard can alias a non-compatibility GPU access.
        for (uint32_t i = r->guards; i && result == RC_REGISTRY_OK; i = r->writers[i].guard_next)
            result = RC_REGISTRY_BLOCKED;
        for (uint32_t i = 1; i <= r->capacity.writers && result == RC_REGISTRY_OK; i++) {
            const Writer *w = &r->writers[i];
            if (!w->state || !writer_overlaps(w, span.interval.range)) continue;
            if (w->state == WRITE_PENDING && rc_registry_position_valid(w->position) &&
                rc_registry_position_compare(w->position, read) >= 0) continue;
            if (!rc_registry_position_valid(w->position) || w->position.scene != read.scene ||
                rc_registry_position_compare(w->position, read) >= 0)
                result = RC_REGISTRY_BLOCKED;
            else {
                int matched = buffer_access_matches(w, span.interval, span.native);
                if (matched < 0) result = RC_REGISTRY_STALE;
                else if (!matched) result = RC_REGISTRY_BLOCKED;
            }
        }
        // Include legacy compatibility claims whose explicit guest address aliases the selected canonical word.
        for (uint32_t i = 1; i <= r->capacity.writers && result == RC_REGISTRY_OK; i++) {
            const Writer *w = &r->writers[i];
            if (!w->state || w->interval.range.backing != RC_GUEST_COMPATIBILITY_BACKING) continue;
            RcSourceRange guest = {RC_GUEST_COMPATIBILITY_BACKING, span.interval.address, span.interval.range.size};
            if (!writer_overlaps(w, guest)) continue;
            if (w->state == WRITE_PENDING && rc_registry_position_valid(w->position) &&
                rc_registry_position_compare(w->position, read) >= 0) continue;
            if (!rc_registry_position_valid(w->position) || w->position.scene != read.scene ||
                rc_registry_position_compare(w->position, read) >= 0)
                result = RC_REGISTRY_BLOCKED;
            else {
                int matched = buffer_access_matches(w, span.interval, span.native);
                if (matched < 0) result = RC_REGISTRY_STALE;
                else if (!matched) result = RC_REGISTRY_BLOCKED;
            }
        }
    }
    unlock_registry(r);
    return result;
}

typedef struct {
    uint64_t from, end;
    RcRegistryPosition read, output, unknown;
    RcRegistryBufferLocation expected;
    uint64_t address, mapping;
    RcRegistryBufferPayload *payload;
    int conflict, blocked, want_payload;
} NativeBufferSlice;

static void native_buffer_slice(NativeBufferSlice *slice, RcSourceRange range, RcRegistryPosition output,
                                RcRegistryBufferLocation location, const BufferPayload *payload) {
    uint64_t end = range.offset + range.size;
    if (range.offset >= slice->end || end <= slice->from) return;
    if (range.offset > slice->from) {
        slice->end = range.offset;
        return;
    }
    if (end < slice->end) slice->end = end;
    if (!rc_registry_position_valid(output) || rc_registry_position_compare(output, slice->read) >= 0) {
        slice->blocked = 1;
        return;
    }
    if (!location.allocation) {
        if (rc_registry_position_compare(output, slice->unknown) > 0) slice->unknown = output;
        return;
    }
    const uint64_t delta = slice->from - range.offset;
    location.offset += delta;
    int latest = rc_registry_position_compare(output, slice->output);
    if (latest < 0) return;
    const int exact = buffer_native_equal(location, slice->expected);
    RcRegistryBufferPayload *owner = NULL;
    if (slice->want_payload && exact && payload && payload->owner && payload->mapping == slice->mapping &&
        payload->address + delta == slice->address) owner = payload->owner;
    if (latest > 0) {
        slice->output = output;
        slice->conflict = !exact;
        slice->payload = owner;
    } else {
        if (!exact) slice->conflict = 1;
        if (slice->payload != owner) slice->payload = NULL;
    }
}

static int native_buffer_live(RcRegistry *r, uint32_t i, void *opaque) {
    NativeBufferSlice *slice = opaque;
    const Writer *w = &r->writers[i];
    if (w->state == WRITE_PENDING && rc_registry_position_valid(w->position) &&
        rc_registry_position_compare(w->position, slice->read) >= 0) return 0;
    if (w->guard) return slice->blocked = 1;
    RcRegistryBufferLocation location = w->native_buffer;
    if (w->unordered || w->kind != RC_REGISTRY_BUFFER || w->kinds != (1u << RC_REGISTRY_BUFFER) ||
        w->state == WRITE_HOLD) location = (RcRegistryBufferLocation){0};
    const BufferPayload *payload = (w->state == WRITE_PENDING || w->state == WRITE_SUBMITTED) ?
        &w->native_payload : NULL;
    native_buffer_slice(slice, w->interval.range, w->visibility_latest, location, payload);
    return slice->blocked;
}

static RcRegistryResult gpu_capture_buffer_fragments(RcRegistry *r, RcRegistryGpuCapture token,
    const RcRegistryBufferSpan *demands, uint32_t count, RcRegistryPosition read,
    RcRegistryBufferFragment *out, RcRegistryBufferPayload **payloads, uint32_t capacity, uint32_t *fragment_count) {
    if (fragment_count) *fragment_count = 0;
    if (!r || !demands || !count || (capacity && !out) || !fragment_count || !rc_registry_position_valid(read))
        return RC_REGISTRY_INVALID;
    lock_registry(r);
    GpuCapture *capture = gpu_capture_of(r, token);
    RcRegistryResult result = RC_REGISTRY_OK;
    if (!capture || !capture->active) result = RC_REGISTRY_STALE;
    else if (r->stats.fail_closed || !read_position_open(r, read)) result = RC_REGISTRY_BLOCKED;
    else if (capture->history_lost) result = RC_REGISTRY_CAPACITY;
    uint64_t sequence = r->gpu_visible_sequence;
    uint64_t floor = sequence > r->capacity.records ? sequence - r->capacity.records : 0;
    if (capture && floor < capture->before_copy) floor = capture->before_copy;
    uint32_t used = 0;
    for (uint32_t i = 0; i < count && result == RC_REGISTRY_OK; ++i) {
        const RcRegistryBufferSpan *demand = &demands[i];
        const RcInterval input = demand->interval;
        if (!buffer_native_span_valid(*demand) || !gpu_capture_covers(capture, input.range)) {
            result = RC_REGISTRY_INVALID;
            break;
        }
        uint64_t total_end = input.range.offset + input.range.size;
        for (uint64_t cursor = input.range.offset; cursor < total_end;) {
            uint64_t delta = cursor - input.range.offset;
            NativeBufferSlice slice = {.from = cursor, .end = total_end, .read = read, .expected = demand->native,
                .address = input.address + delta, .mapping = input.mapping, .want_payload = payloads != NULL};
            slice.expected.offset += delta;
            RcSourceRange remaining = {input.range.backing, cursor, total_end - cursor};
            authority_overlaps(r, remaining, native_buffer_live, &slice);
            for (uint64_t seq = floor; seq < sequence && !slice.blocked; ++seq) {
                const GpuVisibility *event = &r->gpu_visibility[seq % r->capacity.records];
                if (event->range.backing != remaining.backing ||
                    (rc_registry_position_valid(event->producer) &&
                     rc_registry_position_compare(event->producer, read) >= 0)) continue;
                native_buffer_slice(&slice, event->range, event->latest, event->native_buffer, &event->payload);
            }
            if (slice.blocked || slice.conflict ||
                (rc_registry_position_valid(slice.unknown) && rc_registry_position_compare(slice.unknown, slice.output) >= 0)) {
                result = RC_REGISTRY_BLOCKED;
                break;
            }
            if (rc_registry_position_valid(slice.output)) {
                uint64_t size = slice.end - cursor;
                RcRegistryBufferFragment *prior = used ? &out[used - 1] : NULL;
                if (prior && prior->demand == i && prior->interval.range.offset + prior->interval.range.size == cursor &&
                    !rc_registry_position_compare(prior->output, slice.output) &&
                    (!payloads || payloads[used - 1] == slice.payload)) prior->interval.range.size += size;
                else if (used == capacity) { result = RC_REGISTRY_CAPACITY; break; }
                else {
                    if (payloads) payloads[used] = slice.payload;
                    out[used++] = (RcRegistryBufferFragment){
                        {input.address + delta, input.mapping, {input.range.backing, cursor, size}}, slice.output, i};
                }
            }
            cursor = slice.end;
        }
    }
    if (result == RC_REGISTRY_OK) {
        if (payloads) for (uint32_t i = 0; i < used; ++i)
            if (payloads[i]) payloads[i]->retain(payloads[i]);
        *fragment_count = used;
    } else if (payloads && used) memset(payloads, 0, (size_t)used * sizeof *payloads);
    unlock_registry(r);
    return result;
}

RcRegistryResult rc_registry_gpu_capture_buffer_fragments(RcRegistry *r, RcRegistryGpuCapture token,
    const RcRegistryBufferSpan *demands, uint32_t count, RcRegistryPosition read,
    RcRegistryBufferFragment *out, uint32_t capacity, uint32_t *fragment_count) {
    return gpu_capture_buffer_fragments(r, token, demands, count, read, out, NULL, capacity, fragment_count);
}

RcRegistryResult rc_registry_gpu_capture_buffer_fragments_with_payloads(RcRegistry *r, RcRegistryGpuCapture token,
    const RcRegistryBufferSpan *demands, uint32_t count, RcRegistryPosition read,
    RcRegistryBufferFragment *out, RcRegistryBufferPayload **payloads, uint32_t capacity, uint32_t *fragment_count) {
    if (capacity && !payloads) {
        if (fragment_count) *fragment_count = 0;
        return RC_REGISTRY_INVALID;
    }
    return gpu_capture_buffer_fragments(r, token, demands, count, read, out, payloads, capacity, fragment_count);
}

RcRegistryResult rc_registry_gpu_capture_read_evidence(RcRegistry *r, RcRegistryGpuCapture token,
                                                       const RcSourceRange *ranges, uint32_t count, RcRegistryPosition read,
                                                       RcRegistryGpuReadEvidence *evidence) {
    if (evidence) *evidence = RC_GPU_CAPTURE_NO_EVIDENCE;
    if (!r || !count || !ranges || !rc_registry_position_valid(read)) return RC_REGISTRY_INVALID;
    for (uint32_t i = 0; i < count; i++)
        if (!rc_range_valid(ranges[i]) || ranges[i].backing == RC_GUEST_COMPATIBILITY_BACKING) return RC_REGISTRY_INVALID;
    lock_registry(r);
    GpuCapture *capture = gpu_capture_of(r, token);
    RcRegistryResult result = RC_REGISTRY_OK;
    if (!capture) result = RC_REGISTRY_STALE;
    else if (!capture->active) result = RC_REGISTRY_INVALID;
    else if (r->stats.fail_closed || !read_position_open(r, read)) result = RC_REGISTRY_BLOCKED;
    GpuCaptureBlocker blocker = {.read = read, .ordered = 1, .classify = evidence != NULL};
    for (uint32_t i = 0; i < count && result == RC_REGISTRY_OK; i++) {
        if (!gpu_capture_covers(capture, ranges[i])) result = RC_REGISTRY_INVALID;
        else if (authority_overlaps(r, ranges[i], gpu_capture_live_blocker, &blocker)) result = RC_REGISTRY_BLOCKED;
    }
    if (result == RC_REGISTRY_OK && blocker.blocked) {
        result = RC_REGISTRY_BLOCKED;
        if (evidence && !capture->history_lost) *evidence = RC_GPU_CAPTURE_EARLIER_BUFFER;
    }
    if (result == RC_REGISTRY_OK && capture->history_lost) result = RC_REGISTRY_CAPACITY;
    if (result == RC_REGISTRY_OK) {
        uint64_t sequence = r->gpu_visible_sequence;
        uint64_t floor = sequence > r->capacity.records ? sequence - r->capacity.records : 0;
        if (floor < capture->before_copy) floor = capture->before_copy;
        for (uint64_t j = floor; j < sequence && result == RC_REGISTRY_OK; j++) {
            const GpuVisibility *event = &r->gpu_visibility[j % r->capacity.records];
            if (rc_registry_position_valid(event->producer) && rc_registry_position_compare(event->producer, read) >= 0)
                continue;
            for (uint32_t i = 0; i < count; i++) if (rc_range_overlap(event->range, ranges[i])) {
                if (evidence && rc_registry_position_valid(event->producer))
                    *evidence = RC_GPU_CAPTURE_EARLIER_VISIBLE;
                result = RC_REGISTRY_STALE;
                break;
            }
        }
    }
    unlock_registry(r);
    return result;
}

RcRegistryResult rc_registry_gpu_capture_read_at(RcRegistry *r, RcRegistryGpuCapture token,
                                                 const RcSourceRange *ranges, uint32_t count, RcRegistryPosition read) {
    return rc_registry_gpu_capture_read_evidence(r, token, ranges, count, read, NULL);
}

RcRegistryResult rc_registry_gpu_capture_end(RcRegistry *r, RcRegistryGpuCapture *token) {
    if (!r || !token) return RC_REGISTRY_INVALID;
    lock_registry(r);
    GpuCapture *capture = gpu_capture_of(r, *token);
    if (!capture) { unlock_registry(r); return RC_REGISTRY_STALE; }
    if (capture->active) {
        if (capture->prev) r->gpu_captures[capture->prev].next = capture->next;
        else r->active_gpu_captures = capture->next;
        if (capture->next) r->gpu_captures[capture->next].prev = capture->prev;
        gpu_visibility_trim_payloads(r);
        if (!r->active_gpu_captures && !r->produced_scene) r->gpu_visible_sequence = 0;
    }
    *capture = (GpuCapture){0};
    *token = (RcRegistryGpuCapture){0};
    r->gpu_capture_release_epoch++;
#ifndef RC_REGISTRY_SINGLE_THREADED
    plat_cond_broadcast(&r->gpu_capture_released);
#endif
    unlock_registry(r);
    return RC_REGISTRY_OK;
}

RcRegistryResult rc_registry_produced_scene_begin(RcRegistry *r, uint64_t scene) {
    if (!r || !scene) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = RC_REGISTRY_OK;
    if (r->produced_scene) result = RC_REGISTRY_BUSY;
    else if (r->stats.fail_closed) result = RC_REGISTRY_BLOCKED;
    else if (r->gpu_visible_sequence == UINT64_MAX) result = RC_REGISTRY_EXHAUSTED;
    else if (!gpu_visibility_allocate(r)) result = RC_REGISTRY_CAPACITY;
    else {
        r->produced_scene = scene;
        r->produced_begin = r->gpu_visible_sequence;
    }
    unlock_registry(r);
    return result;
}

RcRegistryResult rc_registry_produced_scene_end(RcRegistry *r, uint64_t scene) {
    if (!r || !scene) return RC_REGISTRY_INVALID;
    lock_registry(r);
    if (r->produced_scene != scene) { unlock_registry(r); return RC_REGISTRY_STALE; }
    r->produced_scene = r->produced_begin = 0;
    r->produced_lost = (RcRegistryPosition){0};
    r->produced_lost_unknown = 0;
    if (!r->active_gpu_captures) {
        gpu_visibility_trim_payloads(r);
        r->gpu_visible_sequence = 0;
    }
    unlock_registry(r);
    return RC_REGISTRY_OK;
}

typedef struct { RcRegistryPosition output, read; } ProducedRead;

static int produced_live_blocker(RcRegistry *r, uint32_t i, void *data) {
    const Writer *w = &r->writers[i];
    const ProducedRead *p = data;
    if (w->state == WRITE_PENDING && rc_registry_position_valid(w->position) &&
        rc_registry_position_compare(w->position, p->read) >= 0) return 0;
    if (w->guard || w->state == WRITE_HOLD || !rc_registry_position_valid(w->visibility_latest)) return 1;
    int order = rc_registry_position_compare(w->visibility_latest, p->output);
    return order > 0 || (!order && (w->kinds & ~(1u << RC_REGISTRY_BUFFER)));
}

RcRegistryResult rc_registry_read_produced_at(RcRegistry *r, const RcInterval *ranges, uint32_t count,
                                              RcRegistryPosition output, RcRegistryPosition read) {
    if (!r || !ranges || !count || !rc_registry_position_valid(output) || output.scene != read.scene ||
        rc_registry_position_compare(output, read) >= 0) return RC_REGISTRY_INVALID;
    for (uint32_t i = 0; i < count; i++)
        if (!rc_range_valid(ranges[i].range) || ranges[i].range.backing == RC_GUEST_COMPATIBILITY_BACKING)
            return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = RC_REGISTRY_OK;
    ProducedRead position = {output, read};
    if (r->produced_scene != read.scene) result = RC_REGISTRY_STALE;
    else if (r->stats.fail_closed || !read_position_open(r, read)) result = RC_REGISTRY_BLOCKED;
    for (uint32_t i = 0; i < count && result == RC_REGISTRY_OK; i++)
        if (authority_overlaps(r, ranges[i].range, produced_live_blocker, &position)) result = RC_REGISTRY_BLOCKED;
    if (result == RC_REGISTRY_OK && (r->produced_lost_unknown || rc_registry_position_compare(r->produced_lost, output) > 0))
        result = RC_REGISTRY_CAPACITY;
    uint64_t sequence = r->gpu_visible_sequence;
    uint64_t floor = sequence > r->capacity.records ? sequence - r->capacity.records : 0;
    if (floor < r->produced_begin) floor = r->produced_begin;
    for (uint64_t i = floor; i < sequence && result == RC_REGISTRY_OK; i++) {
        const GpuVisibility *event = &r->gpu_visibility[i % r->capacity.records];
        if ((rc_registry_position_valid(event->latest) && rc_registry_position_compare(event->latest, output) <= 0) ||
            (rc_registry_position_valid(event->producer) && rc_registry_position_compare(event->producer, read) >= 0))
            continue;
        for (uint32_t j = 0; j < count; j++) if (rc_range_overlap(event->range, ranges[j].range)) {
            result = RC_REGISTRY_STALE;
            break;
        }
    }
    unlock_registry(r);
    return result;
}

static int binding_interval_visit(RcRegistry *r, uint32_t i, void *data) {
    RcRegistryBindingObservation *o = data;
    Writer *w = &r->writers[i];
    o->writer_kinds |= 1u << w->kind;
    if (w->guard && (w->state != WRITE_PENDING || rc_registry_position_compare(w->position, o->position) <= 0))
        return 1;
    if (o->image_binding.image && (w->kinds & (1u << RC_REGISTRY_IMAGE))) {
        RcRegistryDiagnosticReason reason = w->state == WRITE_FOREIGN ? RC_REGISTRY_DIAG_IMAGE_FOREIGN :
            w->image_incomplete ? RC_REGISTRY_DIAG_IMAGE_INCOMPLETE : RC_REGISTRY_DIAG_IMAGE_SCHEDULE;
        const ImageProducer *offender = NULL;
        if (w->state == WRITE_HOLD) {
            reason = RC_REGISTRY_DIAG_OTHER_HOLD;
            o->provenance |= RC_REGISTRY_BINDING_HELD;
            if (w->id == o->image_binding.hold.id && i == o->image_binding.hold.index &&
                w->image == o->image_binding.image) return 0;
        } else if ((w->state == WRITE_PENDING || w->state == WRITE_SUBMITTED) &&
                   !w->image_incomplete && w->image_count &&
                   !rc_registry_position_compare(w->image_positions[0].position, w->position)) {
            uint32_t at = image_position_lower(w, o->position);
            image_position_prefix(w, at);
            if (at) {
                const ImageProducer *prior = &w->image_positions[at - 1];
                offender = prior;
                if (prior->prefix_bad) { reason = RC_REGISTRY_DIAG_IMAGE_PREFIX; goto image_refused; }
                if (prior->prefix_active && !image_binding_equal(prior->prefix_image, o->image_binding)) {
                    reason = RC_REGISTRY_DIAG_IMAGE_IDENTITY; goto image_refused;
                }
                o->provenance |= w->state == WRITE_PENDING ? RC_REGISTRY_BINDING_PRIOR_PENDING :
                                                           RC_REGISTRY_BINDING_PRIOR_SUBMITTED;
            }
            for (; at < w->image_count; ++at) {
                const ImageProducer *next = &w->image_positions[at];
                if (!next->unresolved && !next->active) continue; // Explicitly canceled references only.
                offender = next; reason = RC_REGISTRY_DIAG_IMAGE_EDGE;
                if (w->state != WRITE_PENDING || next->position.ordinal <= 1) goto image_refused;
                RcRegistryPosition edge = {next->position.scene, next->position.ordinal - 1};
                if (rc_registry_position_compare(edge, o->position) <= 0) goto image_refused;
                if (!rc_registry_position_valid(o->next_read) || rc_registry_position_compare(edge, o->next_read) < 0)
                    o->next_read = edge;
                break;
            }
            return 0;
        }
image_refused:
        // Foreign/coalesced image coverage has no exact surviving provider identity. It cannot be promoted to
        // the selected image merely because an actual preparation happened at an overlapping address.
        diagnose(r, reason, i, o->range, o->position, o->image_binding, offender, 0);
        r->stats.binding_image_alias++; return 1;
    }
    // Unknown holds can stand for physical aliases whose future virtual-address producers do not overlap this
    // range. Their stable revision is not a producer boundary. Retirement without ordered completion retains
    // that uncertainty; an actually completed unknown-range command still has ordinary legacy provenance.
    if (w->state == WRITE_HOLD) {
        o->provenance |= RC_REGISTRY_BINDING_HELD;
        if (w->kinds & (1u << RC_REGISTRY_UNKNOWN)) { r->stats.binding_unknown_held++; return 1; }
        if (w->kinds & (1u << RC_REGISTRY_IMAGE)) {
            diagnose(r, RC_REGISTRY_DIAG_GENERIC_IMAGE, i, o->range, o->position, o->image_binding, NULL, 0);
            r->stats.binding_image_alias++; return 1;
        }
        return 0;
    }
    if (w->state == WRITE_FOREIGN) {
        o->provenance |= RC_REGISTRY_BINDING_FOREIGN;
        if (w->unordered && (w->kinds & (1u << RC_REGISTRY_UNKNOWN))) {
            r->stats.binding_unordered_foreign++; return 1;
        }
        return 0;
    }
    if (!rc_registry_position_valid(w->position)) { r->stats.binding_unpositioned++; return 1; }
    int order = rc_registry_position_compare(w->position, o->position);
    if (order < 0) {
        // Repeated image writes may share this first-position claim. It cannot certify that a CPU alias's
        // prepared image remains current until the next separately declared producer.
        if (w->kinds & (1u << RC_REGISTRY_IMAGE)) {
            diagnose(r, RC_REGISTRY_DIAG_GENERIC_IMAGE, i, o->range, o->position, o->image_binding, NULL, 0);
            r->stats.binding_image_alias++; return 1;
        }
        o->provenance |= w->state == WRITE_PENDING ? RC_REGISTRY_BINDING_PRIOR_PENDING :
                                                   RC_REGISTRY_BINDING_PRIOR_SUBMITTED;
        return 0;
    }
    if (!order || w->state != WRITE_PENDING || w->position.ordinal <= 1) { r->stats.binding_edge++; return 1; }
    RcRegistryPosition edge = {w->position.scene, w->position.ordinal - 1};
    if (rc_registry_position_compare(edge, o->position) <= 0) { r->stats.binding_edge++; return 1; }
    if (!rc_registry_position_valid(o->next_read) || rc_registry_position_compare(edge, o->next_read) < 0)
        o->next_read = edge;
    return 0;
}
static int binding_observation_current_locked(RcRegistry *r, const RcRegistryBindingObservation *o,
                                              uint64_t scope, RcRegistryPosition position) {
    Record *e = o ? record_of(r, o->handle) : NULL;
    return e && !r->stats.fail_closed && e->current && !e->finalized && o->scope && o->scope == scope &&
        (!o->image_binding.image || image_binding_current_locked(r, o->image_binding, o->range)) &&
        o->guard == r->guard && e->ordered_revision == o->revision && rc_range_equal(e->interval.range, o->range) &&
        rc_registry_position_valid(o->position) && position.scene == o->position.scene &&
        rc_registry_position_valid(o->sealed_through) && rc_registry_position_compare(position, o->sealed_through) <= 0 &&
        rc_registry_position_compare(position, o->position) >= 0 && read_position_open(r, position) &&
        (!rc_registry_position_valid(o->next_read) || rc_registry_position_compare(position, o->next_read) < 0);
}
static RcRegistryResult binding_begin(RcRegistry *r, RcRegistryHandle handle, uint64_t scope,
    RcRegistryPosition position, RcRegistryImageBinding image, RcRegistryBindingObservation *out) {
    if (out) *out = (RcRegistryBindingObservation){0};
    if (!r || !out || !scope || !rc_registry_position_valid(position)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    r->stats.binding_begins++;
    if (image.image) r->diagnostics.counts[RC_REGISTRY_DIAG_TYPED_BEGIN]++;
    Record *e = record_of(r, handle);
    if (!e || !e->current || e->finalized) {
        r->stats.binding_stale++; unlock_registry(r); return RC_REGISTRY_STALE;
    }
    if (image.image && !image_binding_current_locked(r, image, e->interval.range)) {
        Writer *w = writer_of(r, image.hold);
        RcRegistryDiagnosticReason reason = !w ? RC_REGISTRY_DIAG_HOLD_MISSING :
            w->state != WRITE_HOLD ? RC_REGISTRY_DIAG_HOLD_STATE :
            w->kind != RC_REGISTRY_IMAGE ? RC_REGISTRY_DIAG_HOLD_KIND :
            w->image_forgotten ? RC_REGISTRY_DIAG_HOLD_FORGOTTEN :
            !w->image ? RC_REGISTRY_DIAG_HOLD_UNBOUND :
            w->image != image.image ? RC_REGISTRY_DIAG_HOLD_IMAGE : RC_REGISTRY_DIAG_HOLD_RANGE;
        diagnose(r, reason, w ? image.hold.index : 0, e->interval.range, position, image, NULL, 0);
        r->stats.binding_image_alias++; unlock_registry(r); return RC_REGISTRY_BLOCKED;
    }
    if (r->stats.fail_closed || !read_position_open(r, position)) {
        if (r->stats.fail_closed) r->stats.binding_failed++;
        else r->stats.binding_closed++;
        unlock_registry(r); return RC_REGISTRY_BLOCKED;
    }
    RcRegistryBindingObservation observation = {.handle = handle, .revision = e->ordered_revision,
        .topology_revision = e->revision, .scope = scope,
        .guard = r->guard, .range = e->interval.range, .position = position, .image_binding = image,
        .sealed_through = r->declared};
    RcRegistryPosition sealed_end = rc_registry_sealed_deadline(r->declared);
    if (!rc_registry_position_valid(sealed_end)) {
        r->stats.binding_edge++; unlock_registry(r); return RC_REGISTRY_BLOCKED;
    }
    int blocked = source_authority_overlaps(r, e, binding_interval_visit, &observation);
    if (!blocked) {
        observation.next_producer = observation.next_read;
        if (!rc_registry_position_valid(observation.next_read) || rc_registry_position_compare(sealed_end, observation.next_read) < 0)
            observation.next_read = sealed_end;
        *out = observation; r->stats.binding_ready++;
        if (image.image) r->diagnostics.counts[RC_REGISTRY_DIAG_TYPED_READY]++;
    }
    unlock_registry(r); return blocked ? RC_REGISTRY_BLOCKED : RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_binding_begin(RcRegistry *r, RcRegistryHandle handle, uint64_t scope,
    RcRegistryPosition position, RcRegistryBindingObservation *out) {
    return binding_begin(r, handle, scope, position, (RcRegistryImageBinding){0}, out);
}
RcRegistryResult rc_registry_binding_begin_image(RcRegistry *r, RcRegistryHandle handle, uint64_t scope,
    RcRegistryPosition position, RcRegistryImageBinding image, RcRegistryBindingObservation *out) {
    if (!image.image || !image.hold.id) { if (out) *out = (RcRegistryBindingObservation){0}; return RC_REGISTRY_INVALID; }
    return binding_begin(r, handle, scope, position, image, out);
}
RcRegistryResult rc_registry_binding_commit(RcRegistry *r, const RcRegistryBindingObservation *observation,
                                            RcRegistryBindingReceipt *out) {
    if (!r || !observation || !out) return RC_REGISTRY_INVALID;
    RcRegistryBindingObservation saved = *observation;
    *out = (RcRegistryBindingReceipt){0};
    lock_registry(r);
    int current = binding_observation_current_locked(r, &saved, saved.scope, saved.position);
    if (current) out->observation = saved;
    unlock_registry(r); return current ? RC_REGISTRY_OK : RC_REGISTRY_STALE;
}
RcRegistryResult rc_registry_binding_revalidate(RcRegistry *r, const RcRegistryBindingReceipt *prior,
                                                uint64_t scope, RcRegistryPosition position,
                                                RcRegistryBindingReceipt *out) {
    if (!r || !prior || !out || !scope) return RC_REGISTRY_INVALID;
    RcRegistryBindingReceipt saved = *prior;
    *out = (RcRegistryBindingReceipt){0};
    lock_registry(r);
    int current = binding_observation_current_locked(r, &saved.observation, scope, position);
    if (current) *out = saved;
    unlock_registry(r); return current ? RC_REGISTRY_OK : RC_REGISTRY_STALE;
}
typedef struct {
    RcRegistryBindingObservation observation;
    RcRegistryPosition prior_prefix;
} BindingRenew;
static int binding_renew_crossed(RcRegistryPosition writer, RcRegistryPosition prefix, RcRegistryPosition read) {
    if (!rc_registry_position_valid(writer) || rc_registry_position_compare(writer, prefix) <= 0) return 0;
    if (writer.ordinal <= 1) return 1;
    RcRegistryPosition edge = {writer.scene, writer.ordinal - 1};
    return rc_registry_position_compare(edge, read) <= 0;
}
static int binding_renew_visit(RcRegistry *r, uint32_t i, void *data) {
    BindingRenew *renew = data; Writer *w = &r->writers[i];
    if (w->state == WRITE_PENDING || w->state == WRITE_SUBMITTED) {
        if (w->kind == RC_REGISTRY_IMAGE && w->image_count) {
            // The old receipt reached its sealed edge without crossing a real producer. Only newly exposed
            // positions can require another actual preparation; prior cancellations need no history scan.
            for (uint32_t j = image_position_lower(w, renew->prior_prefix); j < w->image_count; ++j) {
                const ImageProducer *p = &w->image_positions[j];
                if (rc_registry_position_compare(p->position, renew->prior_prefix) <= 0) continue;
                if (!binding_renew_crossed(p->position, renew->prior_prefix, renew->observation.position)) break;
                if (p->active || p->unresolved) return 1;
            }
        } else if (binding_renew_crossed(w->position, renew->prior_prefix, renew->observation.position)) return 1;
    }
    return binding_interval_visit(r, i, &renew->observation);
}
static RcRegistryResult binding_refresh(RcRegistry *r, const RcRegistryBindingReceipt *prior,
    uint64_t scope, RcRegistryPosition position, RcRegistryBindingReceipt *out, int schedule) {
    if (!r || !prior || !out || !scope) return RC_REGISTRY_INVALID;
    RcRegistryBindingObservation old = prior->observation; *out = (RcRegistryBindingReceipt){0};
    lock_registry(r); Record *e = record_of(r, old.handle);
    RcRegistryPosition old_end = rc_registry_sealed_deadline(old.sealed_through);
    if (!e || !e->current || e->finalized || !old.guard || old.guard != r->guard ||
        old.revision != e->ordered_revision || !rc_range_equal(old.range, e->interval.range) || old.scope != scope ||
        !rc_registry_position_valid(old.position) || position.scene != old.position.scene ||
        rc_registry_position_compare(position, old.position) < 0 || !rc_registry_position_valid(old_end) ||
        rc_registry_position_compare(old.position, old.sealed_through) > 0 ||
        (rc_registry_position_compare(old.next_read, old_end) &&
         !(schedule && rc_registry_position_valid(old.next_read) && rc_registry_position_compare(position, old.next_read) < 0)) ||
        (old.image_binding.image && !image_binding_current_locked(r, old.image_binding, old.range))) {
        unlock_registry(r); return RC_REGISTRY_STALE;
    }
    RcRegistryPosition next_end = rc_registry_sealed_deadline(r->declared);
    if (r->stats.fail_closed || !read_position_open(r, position) || !rc_registry_position_valid(next_end)) {
        unlock_registry(r); return RC_REGISTRY_BLOCKED;
    }
    if (old.topology_revision && old.topology_revision == e->revision &&
        (!rc_registry_position_valid(old.next_producer) || rc_registry_position_compare(position, old.next_producer) < 0)) {
        out->observation = old; out->observation.sealed_through = r->declared;
        out->observation.next_read = !rc_registry_position_valid(old.next_producer) ||
            rc_registry_position_compare(next_end, old.next_producer) < 0 ? next_end : old.next_producer;
        r->stats.binding_renew_fast++;
        unlock_registry(r); return RC_REGISTRY_OK;
    }
    r->stats.binding_renew_queries++;
    BindingRenew renew = {.observation = {.handle = old.handle, .revision = e->ordered_revision,
        .topology_revision = e->revision, .scope = scope,
        .guard = r->guard, .range = old.range, .position = position,
        .sealed_through = r->declared, .image_binding = old.image_binding}, .prior_prefix = old.sealed_through};
    if (authority_overlaps(r, old.range, binding_renew_visit, &renew)) {
        unlock_registry(r); return RC_REGISTRY_BLOCKED;
    }
    renew.observation.next_producer = renew.observation.next_read;
    if (!rc_registry_position_valid(renew.observation.next_read) ||
        rc_registry_position_compare(next_end, renew.observation.next_read) < 0) renew.observation.next_read = next_end;
    renew.observation.position = old.position; // Renewal is not a new preparation of the physical image.
    renew.observation.provenance |= old.provenance;
    renew.observation.writer_kinds |= old.writer_kinds;
    out->observation = renew.observation;
    unlock_registry(r); return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_binding_renew(RcRegistry *r, const RcRegistryBindingReceipt *prior,
    uint64_t scope, RcRegistryPosition position, RcRegistryBindingReceipt *out) {
    return binding_refresh(r, prior, scope, position, out, 0);
}
static RcRegistryPosition producer_read_deadline(RcRegistryPosition producer) {
    if (!rc_registry_position_valid(producer)) return (RcRegistryPosition){0};
    if (producer.ordinal > 1) producer.ordinal--;
    return producer;
}
RcRegistryResult rc_registry_read_schedule(RcRegistry *r, const RcRegistryReadObservation *prior,
    uint64_t scope, RcRegistryPosition position, RcRegistryReadSchedule *out) {
    if (!out) return RC_REGISTRY_INVALID;
    RcRegistryReadObservation saved = prior ? *prior : (RcRegistryReadObservation){0};
    *out = (RcRegistryReadSchedule){0};
    RcRegistryReadObservation fresh;
    RcRegistryResult result = rc_registry_read_renew(r, prior ? &saved : NULL, scope, position, &fresh);
    if (result != RC_REGISTRY_OK) return result;
    lock_registry(r); Record *e = record_of(r, fresh.handle);
    // The renewal and this snapshot may be separated by another caller. Never acknowledge that caller's
    // topology epoch with an observation that predates a relevant schedule mutation.
    if (!read_observation_current_locked(r, &fresh) || !e || e->revision != fresh.topology_revision) {
        unlock_registry(r); return RC_REGISTRY_STALE;
    }
    *out = (RcRegistryReadSchedule){.observation = fresh, .next_read = producer_read_deadline(fresh.next_write),
        .epoch = __atomic_load_n(&r->authority_epoch, __ATOMIC_RELAXED)};
    unlock_registry(r); return RC_REGISTRY_OK;
}
// Source-access storage belongs to one serialized consumer, while the registry owns the retained lease and
// all authority facts. This service is the sole writer of the consumer's provenance and permission decisions.
static Record *source_access_record_locked(RcRegistry *r, const RcRegistrySourceAccess *a) {
    RcRegistrySource lease = a->owner;
    Capture *owner = lease.index && lease.index <= r->capacity.records ? &r->captures[lease.index] : NULL;
    Record *e = owner && owner->id == lease.id && owner->source ? record_of(r, owner->record) : NULL;
    return e && e->current && !e->finalized && e->id == lease.handle.id &&
        owner->record.index == lease.handle.index ? e : NULL;
}
RcRegistryResult rc_registry_source_access_acquire(RcRegistry *r, uint64_t object, RcSourceRange range,
                                                  RcRegistrySourceAccess *a) {
    if (!a) return RC_REGISTRY_INVALID;
    return rc_registry_source_acquire(r, object, range, &a->owner);
}
RcRegistryResult rc_registry_source_access_acquire_associated(RcRegistry *r, uint64_t object, RcSourceRange range,
                                                             uint64_t association, RcRegistrySourceAccess *a) {
    if (!a || !association) return RC_REGISTRY_INVALID;
    return source_acquire_associated(r, object, range, association, (SourceShape){0}, &a->owner);
}

RcRegistryResult rc_registry_source_access_acquire_shaped(RcRegistry *r, uint64_t object, RcSourceRange hull,
    uint64_t association, const RcSourceRange *spans, uint32_t count, RcRegistrySourceAccess *a) {
    if (!a || !association || !spans || !count || count > RC_REGISTRY_SOURCE_SPAN_LIMIT || !rc_range_valid(hull))
        return RC_REGISTRY_INVALID;
    uint64_t end = hull.offset;
    for (uint32_t i = 0; i < count; i++) {
        RcSourceRange span = spans[i];
        if (span.backing != hull.backing || !rc_range_valid(span) || span.offset < end ||
            span.offset + span.size > hull.offset + hull.size) return RC_REGISTRY_INVALID;
        end = span.offset + span.size;
    }
    SourceShape shape = count == 1 && rc_range_equal(hull, spans[0]) ? (SourceShape){0} : (SourceShape){spans, count};
    return source_acquire_associated(r, object, hull, association, shape, &a->owner);
}
RcRegistryResult rc_registry_source_access_release(RcRegistry *r, RcRegistrySourceAccess *a) {
    if (!a) return RC_REGISTRY_INVALID;
    RcRegistryResult result = rc_registry_source_release(r, &a->owner);
    *a = (RcRegistrySourceAccess){0};
    return result;
}
static RcRegistryResult source_access_observe_locked(RcRegistry *r, const RcRegistrySourceAccess *a,
    uint64_t scope, RcRegistryPosition position, RcRegistrySourceProof *out) {
    return !source_access_record_locked(r, a) ? RC_REGISTRY_STALE :
        rc_registry_position_valid(position) ? observe_at_locked(r, a->owner.handle, scope, position, &out->ordered) :
        observe_locked(r, a->owner.handle, scope, &out->legacy);
}
RcRegistryResult rc_registry_source_access_observe_extents(RcRegistry *r, const RcRegistrySourceAccess *access,
    uint32_t count, uint64_t scope, RcRegistryPosition position, RcRegistrySourceProof *out) {
    if (!r || !access || !count || !out) return RC_REGISTRY_INVALID;
    memset(out, 0, (size_t)count * sizeof *out);
    lock_registry(r);
    RcRegistryResult result = RC_REGISTRY_OK;
    for (uint32_t i = 0; i < count && result == RC_REGISTRY_OK; ++i)
        result = source_access_observe_locked(r, &access[i], scope, position, &out[i]);
    unlock_registry(r);
    if (result != RC_REGISTRY_OK) memset(out, 0, (size_t)count * sizeof *out);
    return result;
}
RcRegistryResult rc_registry_source_access_observe(RcRegistry *r, const RcRegistrySourceAccess *a,
    uint64_t scope, RcRegistryPosition position, RcRegistrySourceProof *out) {
    return rc_registry_source_access_observe_extents(r, a, 1, scope, position, out);
}
int rc_registry_source_proofs_current(RcRegistry *r, const RcRegistrySourceProof *proofs, uint32_t count,
                                     RcRegistryPosition position) {
    if (!r || !proofs || !count) return 0;
    lock_registry(r);
    int current = 1;
    for (uint32_t i = 0; i < count && current; ++i) {
        const RcRegistrySourceProof *proof = &proofs[i];
        if (rc_registry_position_valid(proof->ordered.position))
            current = !rc_registry_position_compare(position, proof->ordered.position) &&
                read_observation_current_locked(r, &proof->ordered);
        else current = observation_current_locked(r, &proof->legacy);
    }
    unlock_registry(r);
    return current;
}
static int source_access_can_commit(RcRegistry *r, const RcRegistrySourceAccess *a, const RcRegistrySourceProof *proof) {
    RcRegistryObservation content = rc_registry_source_proof_content(proof);
    return source_access_record_locked(r, a) && content.handle.id == a->owner.handle.id &&
        content.handle.index == a->owner.handle.index && content_current_locked(r, &content);
}
static int source_access_commit_content(RcRegistrySourceAccess *a, RcRegistrySourceProof proof, int same_contents) {
    int retained = same_contents && rc_registry_source_proof_same(&proof, &a->_content);
    if (!retained) {
        a->_content = proof;
        a->_read = proof.ordered;
        a->_maintenance = (RcRegistryObservation){0};
    }
    if (proof.legacy.handle.id) a->_maintenance = proof.legacy;
    return !retained;
}
RcRegistryResult rc_registry_source_access_commit_extents(RcRegistry *r, RcRegistrySourceAccess *access,
    const RcRegistrySourceProof *proofs, uint32_t count, int same_contents, int *changed) {
    if (changed) *changed = 0;
    if (!r || !access || !proofs || !count) return RC_REGISTRY_INVALID;
    lock_registry(r);
    for (uint32_t i = 0; i < count; ++i) if (!source_access_can_commit(r, &access[i], &proofs[i])) {
        unlock_registry(r); return RC_REGISTRY_STALE;
    }
    int different = 0;
    for (uint32_t i = 0; i < count; ++i)
        different |= source_access_commit_content(&access[i], proofs[i], same_contents);
    if (changed) *changed = different;
    unlock_registry(r);
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_source_access_commit(RcRegistry *r, RcRegistrySourceAccess *a,
    const RcRegistrySourceProof *proof, int same_contents, int *changed) {
    return rc_registry_source_access_commit_extents(r, a, proof, 1, same_contents, changed);
}
static RcRegistryResult source_access_read_locked(RcRegistry *r, RcRegistrySourceAccess *a,
    uint64_t scope, RcRegistryPosition position, RcRegistryReadObservation *out) {
    if (out) *out = (RcRegistryReadObservation){0};
    if (!r || !a || !scope || !rc_registry_position_valid(position)) return RC_REGISTRY_INVALID;
    if (!source_access_record_locked(r, a)) return RC_REGISTRY_STALE;
    RcRegistryReadObservation prior = a->_read, fresh;
    RcRegistrySourceProof identity = {.ordered = prior};
    int paired = rc_registry_source_proof_same(&identity, &a->_content);
    if (paired && read_revalidate_locked(r, &prior, scope, position, &fresh) == RC_REGISTRY_OK) {
        a->_read.scope = fresh.scope;
        if (out) *out = fresh;
        return RC_REGISTRY_OK;
    }
    // Renewal starts at the retained permission's real origin. Never backdate a proof obtained after a producer.
    if (paired && prior.scope == scope && prior.position.scene == position.scene &&
        rc_registry_position_compare(position, prior.position) >= 0 &&
        read_renew_locked(r, &prior, scope, prior.position, &fresh) == RC_REGISTRY_OK) {
        RcRegistryReadObservation query;
        if (read_revalidate_locked(r, &fresh, scope, position, &query) == RC_REGISTRY_OK) {
            a->_read = fresh;
            if (out) *out = query;
            return RC_REGISTRY_OK;
        }
    }
    RcRegistryObservation content = rc_registry_source_proof_content(&a->_content);
    RcRegistryResult result = observe_at_locked(r, content.handle, scope, position, &fresh);
    if (result != RC_REGISTRY_OK) return result;
    identity = (RcRegistrySourceProof){.ordered = fresh};
    if (!rc_registry_source_proof_same(&a->_content, &identity)) return RC_REGISTRY_STALE;
    a->_read = fresh;
    if (out) *out = fresh;
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_source_access_read(RcRegistry *r, RcRegistrySourceAccess *a,
    uint64_t scope, RcRegistryPosition position, RcRegistryReadObservation *out) {
    return rc_registry_source_access_read_extents(r, a, 1, scope, position, out);
}
RcRegistryResult rc_registry_source_access_read_extents(RcRegistry *r, RcRegistrySourceAccess *access, uint32_t count,
    uint64_t scope, RcRegistryPosition position, RcRegistryReadObservation *out) {
    if (out) *out = (RcRegistryReadObservation){0};
    if (!r || !access || !count) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryReadObservation anchor = {0};
    RcRegistryResult result = RC_REGISTRY_OK;
    for (uint32_t i = 0; i < count && result == RC_REGISTRY_OK; ++i) {
        RcRegistryReadObservation read;
        result = source_access_read_locked(r, &access[i], scope, position, &read);
        if (!i) anchor = read;
        else if (rc_registry_position_valid(read.next_write) &&
                 (!rc_registry_position_valid(anchor.next_write) ||
                  rc_registry_position_compare(read.next_write, anchor.next_write) < 0)) anchor.next_write = read.next_write;
    }
    if (result == RC_REGISTRY_OK && out) *out = anchor;
    unlock_registry(r);
    return result;
}
int rc_registry_source_access_current(RcRegistry *r, RcRegistrySourceAccess *a, uint64_t scope,
                                      RcRegistryPosition position) {
    return rc_registry_source_access_current_extents(r, a, 1, scope, position);
}
int rc_registry_source_access_current_extents(RcRegistry *r, RcRegistrySourceAccess *access, uint32_t count,
    uint64_t scope, RcRegistryPosition position) {
    if (rc_registry_position_valid(position))
        return rc_registry_source_access_read_extents(r, access, count, scope, position, NULL) == RC_REGISTRY_OK;
    if (!r || !access || !count || !scope) return 0;
    lock_registry(r);
    int current = 1;
    for (uint32_t i = 0; i < count && current; ++i) {
        RcRegistrySourceAccess *a = &access[i];
        RcRegistrySourceProof permission = {.legacy = a->_maintenance};
        current = source_access_record_locked(r, a) && a->_maintenance.scope == scope &&
            rc_registry_source_proof_same(&permission, &a->_content) && observation_current_locked(r, &a->_maintenance);
    }
    unlock_registry(r); return current;
}
RcRegistryResult rc_registry_source_access_schedule(RcRegistry *r, RcRegistrySourceAccess *a,
    uint64_t scope, RcRegistryPosition position, RcRegistryReadSchedule *out) {
    return rc_registry_source_access_schedule_extents(r, a, 1, scope, position, out);
}
RcRegistryResult rc_registry_source_access_schedule_extents(RcRegistry *r, RcRegistrySourceAccess *access, uint32_t count,
    uint64_t scope, RcRegistryPosition position, RcRegistryReadSchedule *out) {
    if (!r || !access || !count || !out) return RC_REGISTRY_INVALID;
    *out = (RcRegistryReadSchedule){0};
    lock_registry(r);
    RcRegistryReadObservation anchor = {0};
    RcRegistryPosition deadline = {0};
    RcRegistryResult result = RC_REGISTRY_OK;
    for (uint32_t i = 0; i < count && result == RC_REGISTRY_OK; ++i) {
        RcRegistrySourceAccess *a = &access[i];
        RcRegistrySourceProof permission = {.ordered = a->_read};
        RcRegistryReadObservation fresh;
        result = !source_access_record_locked(r, a) || !rc_registry_source_proof_same(&permission, &a->_content) ?
            RC_REGISTRY_STALE : read_renew_locked(r, &a->_read, scope, position, &fresh);
        if (result != RC_REGISTRY_OK) break;
        if (!i) anchor = fresh;
        RcRegistryPosition next = producer_read_deadline(fresh.next_write);
        if (rc_registry_position_valid(next) && (!rc_registry_position_valid(deadline) ||
            rc_registry_position_compare(next, deadline) < 0)) deadline = next;
    }
    if (result == RC_REGISTRY_OK) *out = (RcRegistryReadSchedule){.observation = anchor, .next_read = deadline,
        .epoch = __atomic_load_n(&r->authority_epoch, __ATOMIC_RELAXED)};
    unlock_registry(r); return result;
}
RcRegistryResult rc_registry_binding_schedule(RcRegistry *r, const RcRegistryBindingReceipt *prior,
    uint64_t scope, RcRegistryPosition position, RcRegistryBindingSchedule *out) {
    if (!out) return RC_REGISTRY_INVALID;
    RcRegistryBindingReceipt saved = prior ? *prior : (RcRegistryBindingReceipt){0};
    *out = (RcRegistryBindingSchedule){0};
    RcRegistryBindingReceipt fresh;
    RcRegistryResult result = binding_refresh(r, prior ? &saved : NULL, scope, position, &fresh, 1);
    if (result != RC_REGISTRY_OK) return result;
    lock_registry(r); Record *e = record_of(r, fresh.observation.handle);
    if (!binding_observation_current_locked(r, &fresh.observation, scope, position) || !e ||
        e->revision != fresh.observation.topology_revision) {
        unlock_registry(r); return RC_REGISTRY_STALE;
    }
    *out = (RcRegistryBindingSchedule){.receipt = fresh, .next_read = fresh.observation.next_producer,
        .epoch = __atomic_load_n(&r->authority_epoch, __ATOMIC_RELAXED)};
    unlock_registry(r); return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_read_commit(RcRegistry *r, const RcRegistryReadObservation *o,
                                        const RcPreparedInput *input, RcRegistryReadVersion *out) {
    if (out) *out = (RcRegistryReadVersion){0};
    if (!r || !o || !input || !out || !rc_input_exact(input)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    if (!read_observation_current_locked(r, o)) { unlock_registry(r); return RC_REGISTRY_STALE; }
    Record *e = &r->records[o->handle.index];
    int reuse = e->version && e->published_revision == o->content_revision && e->content == input->content &&
                e->representation == input->representation && e->policy == input->policy && e->payload == input->payload &&
                e->provider_kind == input->kind;
    uint64_t version = reuse ? e->version : identity(r);
    if (!version) { unlock_registry(r); return RC_REGISTRY_EXHAUSTED; }
    *out = (RcRegistryReadVersion){.id = version, .origin = *o, .input = *input};
    if (reuse) {
        out->origin.position = e->publication;
        out->origin.scope = e->publication_scope;
    } else {
        r->stats.read_versions++;
        // An earlier chunk may finish capture after a later chunk. Its exact version remains valid, but it cannot
        // replace the source's newer published association. Provider ownership lives with the material/captures.
        if (!e->version || rc_registry_position_compare(o->position, e->publication) >= 0) {
            e->version = version; e->publication = o->position; e->publication_scope = o->scope;
            e->published_revision = o->content_revision;
            e->content = input->content; e->representation = input->representation;
            e->policy = input->policy; e->payload = input->payload;
            e->provider_kind = (uint8_t)input->kind;
        }
    }
    unlock_registry(r); return RC_REGISTRY_OK;
}

static uint32_t *writer_state_count(RcRegistry *r, uint8_t state) {
    switch (state) {
    case WRITE_PENDING: return &r->stats.writer_pending;
    case WRITE_SUBMITTED: return &r->stats.writer_submitted;
    case WRITE_FOREIGN: return &r->stats.writer_foreign;
    default: return &r->stats.writer_holds;
    }
}
static uint32_t *writer_kind_count(RcRegistry *r, uint8_t kind) {
    return kind == RC_REGISTRY_IMAGE ? &r->stats.writer_image :
           kind == RC_REGISTRY_BUFFER ? &r->stats.writer_buffer : &r->stats.writer_unknown;
}
static void writer_set_state(RcRegistry *r, Writer *writer, uint8_t state) {
    if (writer->state) --*writer_state_count(r, writer->state);
    writer->state = state;
    ++*writer_state_count(r, state);
}
static void free_writer(RcRegistry *r, uint32_t i) {
    Writer *w = &r->writers[i];
    if (w->native_projection) native_projection_retire(r, w);
    native_image_output_clear(r, w);
    image_positions_clear(r, w);
    foreign_coverage_release(r, w->foreign_coverage);
    foreign_coverage_release(r, w->image_possible);
    buffer_payload_drop(&w->native_payload);
    --*writer_state_count(r, w->state);
    --*writer_kind_count(r, w->kind);
    if (w->guard) {
        if (r->unresolved_guard==i) r->unresolved_guard=0;
        if (w->guard_prev) r->writers[w->guard_prev].guard_next = w->guard_next;
        else r->guards = w->guard_next;
        if (w->guard_next) r->writers[w->guard_next].guard_prev = w->guard_prev;
    } else r->authority.root = erase_node(&r->authority, r->authority.root, i);
    *w = (Writer){.free_next = r->free_writer};
    r->free_writer = i;
    r->stats.writers--;
}
static int kind_valid(RcRegistryWriterKind kind) { return (unsigned)kind <= RC_REGISTRY_UNKNOWN; }
static void writer_count_add(RcRegistry *r) {
    if (++r->stats.writers > r->stats.writer_peak) r->stats.writer_peak = r->stats.writers;
}
#ifndef RC_REGISTRY_SINGLE_THREADED
enum { CENSUS_GROUPS = 16 };
enum { CENSUS_UNPOSITIONED, CENSUS_UNHANDED, CENSUS_PART_SCENE, CENSUS_FINAL_SCENE, CENSUS_UNCERTAIN_SCENE };
enum { CENSUS_RAW, CENSUS_CANONICAL, CENSUS_GUARD, CENSUS_UNRESOLVED };
#ifdef _WIN32
extern char __ImageBase;
#endif
static unsigned long long census_caller(uintptr_t origin) {
#ifdef _WIN32
    return origin ? (unsigned long long)(origin - (uintptr_t)&__ImageBase) : 0;
#else
    return (unsigned long long)origin;
#endif
}
typedef struct {
    uintptr_t origin;
    RcSourceRange range;
    uint64_t scene, ordinal;
    uint8_t kind, state, domain, guard_reason, native;
} WriterCensusRow;
typedef struct {
    WriterCensusRow first;
    uint32_t count, ranges, range_scenes, range_positions;
} WriterCensusGroup;

static int census_group_cmp(const WriterCensusRow *a, const WriterCensusRow *b) {
    if (a->kind != b->kind) return a->kind < b->kind ? -1 : 1;
    if (a->state != b->state) return a->state < b->state ? -1 : 1;
    if (a->domain != b->domain) return a->domain < b->domain ? -1 : 1;
    if (a->guard_reason != b->guard_reason) return a->guard_reason < b->guard_reason ? -1 : 1;
    if (a->native != b->native) return a->native < b->native ? -1 : 1;
    return a->origin == b->origin ? 0 : a->origin < b->origin ? -1 : 1;
}
static int census_range_cmp(RcSourceRange a, RcSourceRange b) {
    if (a.backing != b.backing) return a.backing < b.backing ? -1 : 1;
    if (a.offset != b.offset) return a.offset < b.offset ? -1 : 1;
    return a.size == b.size ? 0 : a.size < b.size ? -1 : 1;
}
static int census_row_cmp(const void *left, const void *right) {
    const WriterCensusRow *a = left, *b = right;
    int group = census_group_cmp(a, b);
    if (group) return group;
    int range = census_range_cmp(a->range, b->range);
    if (range) return range;
    if (a->scene != b->scene) return a->scene < b->scene ? -1 : 1;
    return a->ordinal == b->ordinal ? 0 : a->ordinal < b->ordinal ? -1 : 1;
}
static WriterCensusRow census_row(const RcRegistry *r, uint32_t index, uint64_t final, uint64_t part,
                                  uint64_t uncertain) {
    const Writer *w = &r->writers[index];
    uint64_t scene = w->diagnostic_position.scene;
    uint8_t native = !scene ? CENSUS_UNPOSITIONED :
        scene <= final ? CENSUS_FINAL_SCENE :
        scene <= part ? CENSUS_PART_SCENE :
        scene <= uncertain ? CENSUS_UNCERTAIN_SCENE : CENSUS_UNHANDED;
    unsigned domain = index == r->unresolved_guard ? CENSUS_UNRESOLVED : w->guard ? CENSUS_GUARD :
        w->interval.range.backing == RC_GUEST_COMPATIBILITY_BACKING ? CENSUS_RAW : CENSUS_CANONICAL;
    return (WriterCensusRow){w->diagnostic_origin, w->interval.range, scene, w->diagnostic_position.ordinal,
                             w->kind, w->state, (uint8_t)domain, w->diagnostic_guard_reason, native};
}
static void census_show_groups(WriterCensusRow *rows, uint32_t count) {
    qsort(rows, count, sizeof *rows, census_row_cmp);
    WriterCensusGroup top[CENSUS_GROUPS] = {0};
    uint32_t groups = 0, represented = 0;
    for (uint32_t i = 0; i < count;) {
        WriterCensusGroup group = {.first = rows[i], .count = 1, .ranges = 1, .range_scenes = 1, .range_positions = 1};
        while (++i < count && !census_group_cmp(&group.first, &rows[i])) {
            int other_range = census_range_cmp(rows[i - 1].range, rows[i].range) != 0;
            int other_scene = other_range || rows[i - 1].scene != rows[i].scene;
            group.count++;
            group.ranges += other_range;
            group.range_scenes += other_scene;
            group.range_positions += other_scene || rows[i - 1].ordinal != rows[i].ordinal;
        }
        groups++;
        for (uint32_t j = 0; j < CENSUS_GROUPS; ++j)
            if (group.count > top[j].count) {
                for (uint32_t k = CENSUS_GROUPS - 1; k > j; --k) top[k] = top[k - 1];
                top[j] = group;
                break;
            }
    }
    static const char *const kinds[] = {"image", "buffer", "unknown"};
    static const char *const states[] = {"", "pending", "submitted", "foreign", "held"};
    static const char *const domains[] = {"raw", "canonical", "guard", "unresolved"};
    static const char *const native_states[] = {"unpositioned", "unhanded", "part-scene", "final-scene", "uncertain-scene"};
    static const char *const guard_reasons[] = {"unclassified", "mapping-unavailable", "reservation-refused",
                                               "command-opaque", "gpa-projection"};
    for (uint32_t i = 0; i < CENSUS_GROUPS && top[i].count; ++i) {
        const WriterCensusGroup *g = &top[i];
        const WriterCensusRow *row = &g->first;
        represented += g->count;
        rt_log("resource writer census %s %s %s %s guard=%s caller=%llx count=%u distinct range/range-scene/range-position "
               "%u/%u/%u example scene %llu range %llu:%llx+%llx",
               kinds[row->kind], states[row->state], domains[row->domain], native_states[row->native],
               guard_reasons[row->guard_reason], census_caller(row->origin), g->count, g->ranges, g->range_scenes,
               g->range_positions, (unsigned long long)row->scene, (unsigned long long)row->range.backing,
               (unsigned long long)row->range.offset, (unsigned long long)row->range.size);
    }
    rt_log("resource writer census groups %u listed %u records listed %u omitted %u", groups,
           groups < CENSUS_GROUPS ? groups : CENSUS_GROUPS, represented, count - represented);
}
static void writer_census(RcRegistry *r, const char *reason, uint32_t needed, uintptr_t origin) {
    if (r != runtime_registry || r->writer_census_logged) return;
    r->writer_census_logged = 1;
    const RcRegistryStats *s = &r->stats;
    uint64_t final = __atomic_load_n(&native_final, __ATOMIC_ACQUIRE);
    uint64_t part = __atomic_load_n(&native_part, __ATOMIC_ACQUIRE);
    uint64_t uncertain = __atomic_load_n(&native_uncertain, __ATOMIC_ACQUIRE);
    rt_log("resource writer census %s need %u caller %llx active %u/%u image/buffer/unknown %u/%u/%u "
           "pending/submitted/foreign/held %u/%u/%u/%u refusals %llu",
           reason, needed, census_caller(origin), s->writers, r->capacity.writers,
           s->writer_image, s->writer_buffer, s->writer_unknown,
           s->writer_pending, s->writer_submitted, s->writer_foreign, s->writer_holds,
           (unsigned long long)s->writer_refusals);
    rt_log("resource writer census registration records/sources/descriptors/scopes %u/%u/%u/%u "
           "declared %llu/%llu registry handed %llu/%llu native final/part/uncertain %llu/%llu/%llu",
           s->records, s->live_sources, s->live_descriptors, s->scopes,
           (unsigned long long)r->declared.scene, (unsigned long long)r->declared.ordinal,
           (unsigned long long)r->handed.scene, (unsigned long long)r->handed.ordinal,
           (unsigned long long)final, (unsigned long long)part, (unsigned long long)uncertain);
    if (!s->writers) return;
    WriterCensusRow *rows = malloc((size_t)r->capacity.writers * sizeof *rows);
    if (!rows) {
        rt_log("resource writer census detail allocation failed");
        return;
    }
    uint32_t count = 0;
    for (uint32_t i = 1; i <= r->capacity.writers; ++i)
        if (r->writers[i].state) rows[count++] = census_row(r, i, final, part, uncertain);
    census_show_groups(rows, count);
    free(rows);
}
#endif

static RcRegistryWrite insert_writer(RcRegistry *r, RcSourceRange range, RcRegistryWriterKind kind,
                                      RcRegistryPosition position, uint64_t id, int guard,
                                      RcRegistryGuardReason reason, uintptr_t origin) {
    uint32_t i = r->free_writer;
    Writer *w = &r->writers[i];
    r->free_writer = w->free_next;
    *w = (Writer){.interval = {.range = range, .order = id, .priority = mixed(id)}, .id = id,
                  .producer_order = id, .state = WRITE_PENDING, .kind = (uint8_t)kind,
                  .kinds = (uint8_t)(1u << kind), .position = position, .guard = guard,
                  .visibility_position = position, .visibility_latest = position,
                  .diagnostic_origin = origin, .diagnostic_position = position, .diagnostic_guard_reason = reason};
    if (guard) {
        w->guard_next = r->guards;
        if (r->guards) r->writers[r->guards].guard_prev = i;
        r->guards = i;
    } else r->authority.root = insert_node(&r->authority, r->authority.root, i);
    writer_count_add(r);
    r->stats.writer_pending++;
    ++*writer_kind_count(r, w->kind);
    RcRegistryWrite out = {id, i};
    if (rc_registry_position_valid(position)) invalidate_schedule(r, range, out, position, RC_REGISTRY_SCHEDULE_ADD);
    else invalidate(r, range);
    return out;
}
static RcRegistryResult write_begin_locked(RcRegistry *r, RcSourceRange range, RcRegistryWriterKind kind,
                                           RcRegistryPosition position, RcRegistryWrite *out, uintptr_t origin) {
    if (!rc_range_valid(range) || !kind_valid(kind)) {
        poison(r); return RC_REGISTRY_INVALID;
    }
    if (rc_registry_position_valid(position) && rc_registry_position_compare(position, r->declared) <= 0) {
        r->stats.late_declarations++;
        poison(r); return RC_REGISTRY_STALE;
    }
    if (!r->free_writer) {
        r->stats.writer_refusals++;
        writer_census(r, "writer-capacity", 1, origin);
        poison(r);
        return RC_REGISTRY_CAPACITY;
    }
    uint64_t id = identity(r);
    if (!id) return RC_REGISTRY_EXHAUSTED;
    *out = insert_writer(r, range, kind, position, id, 0, RC_REGISTRY_GUARD_UNCLASSIFIED, origin);
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_write_begin(RcRegistry *r, RcSourceRange range, RcRegistryWriterKind kind,
                                        RcRegistryWrite *out) {
    if (out) *out = (RcRegistryWrite){0};
    if (!r || !out) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = write_begin_locked(r, range, kind, (RcRegistryPosition){0}, out,
                                                  (uintptr_t)__builtin_return_address(0));
    unlock_registry(r); return result;
}
RcRegistryResult rc_registry_write_begin_at(RcRegistry *r, RcSourceRange range, RcRegistryWriterKind kind,
                                           RcRegistryPosition position, RcRegistryWrite *out) {
    if (out) *out = (RcRegistryWrite){0};
    if (!r || !out || !rc_registry_position_valid(position)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = write_begin_locked(r, range, kind, position, out, (uintptr_t)__builtin_return_address(0));
    unlock_registry(r); return result;
}
static RcRegistryResult write_group_space(RcRegistry *r, RcRegistryPosition position, uint32_t count, uintptr_t origin) {
    if (rc_registry_position_valid(position) && rc_registry_position_compare(position, r->declared) <= 0) {
        r->stats.late_declarations++;
        return RC_REGISTRY_STALE;
    }
    if (count > r->capacity.writers - r->stats.writers) {
        r->stats.writer_refusals++;
        writer_census(r, "writer-capacity", count, origin);
        return RC_REGISTRY_CAPACITY;
    }
    if (count > UINT64_MAX - r->clock) { poison(r); return RC_REGISTRY_EXHAUSTED; }
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_write_begin_extents(RcRegistry *r, const RcRegistryWriteExtent *extents,
                                                uint32_t count, RcRegistryPosition position, RcRegistryWrite *out) {
    if (out) memset(out, 0, (size_t)count * sizeof *out);
    if (!r || !out || !extents || !count || (!!position.scene != !!position.ordinal)) return RC_REGISTRY_INVALID;
    for (uint32_t i = 0; i < count; ++i)
        if (!rc_range_valid(extents[i].range) || !kind_valid(extents[i].kind)) return RC_REGISTRY_INVALID;
    uintptr_t origin = (uintptr_t)__builtin_return_address(0);
    lock_registry(r);
    RcRegistryResult result = write_group_space(r, position, count, origin);
    if (result == RC_REGISTRY_OK)
        for (uint32_t i = 0; i < count; ++i)
            out[i] = insert_writer(r, extents[i].range, extents[i].kind, position, identity(r), 0,
                                   RC_REGISTRY_GUARD_UNCLASSIFIED, origin);
    unlock_registry(r);
    return result;
}
static RcRegistryResult write_guard_begin_at(RcRegistry *r, RcRegistryWriterKind kind, RcRegistryPosition position,
                                              RcRegistryGuardReason reason, RcRegistryWrite *out, uintptr_t origin) {
    if (out) *out = (RcRegistryWrite){0};
    if (!r || !out || (!!position.scene != !!position.ordinal) || !kind_valid(kind) || kind == RC_REGISTRY_IMAGE ||
        (unsigned)reason > RC_REGISTRY_GUARD_GPA_PROJECTION)
        return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = write_group_space(r, position, 1, origin);
    if (result == RC_REGISTRY_OK)
        *out = insert_writer(r, (RcSourceRange){0}, kind, position, identity(r), 1, reason, origin);
    unlock_registry(r);
    return result;
}
RcRegistryResult rc_registry_write_guard_begin_at(RcRegistry *r, RcRegistryWriterKind kind, RcRegistryPosition position,
                                                  RcRegistryWrite *out) {
    return write_guard_begin_at(r, kind, position, RC_REGISTRY_GUARD_UNCLASSIFIED, out,
                                (uintptr_t)__builtin_return_address(0));
}
RcRegistryResult rc_registry_write_guard_begin_reason_at(RcRegistry *r, RcRegistryWriterKind kind, RcRegistryPosition position,
                                                         RcRegistryGuardReason reason, RcRegistryWrite *out) {
    return write_guard_begin_at(r, kind, position, reason, out, (uintptr_t)__builtin_return_address(0));
}
RcRegistryResult rc_registry_write_expand(RcRegistry *r, RcRegistryWrite ticket, RcSourceRange added) {
    if (!r || !rc_range_valid(added)) return RC_REGISTRY_INVALID;
    lock_registry(r); Writer *w = writer_of(r, ticket);
    if (!w || w->state != WRITE_PENDING) { unlock_registry(r); return RC_REGISTRY_STALE; }
    RcSourceRange old = w->interval.range;
    if (w->guard || added.backing != old.backing) { unlock_registry(r); return RC_REGISTRY_INVALID; }
    uint64_t lo = old.offset < added.offset ? old.offset : added.offset;
    uint64_t old_end = old.offset + old.size, added_end = added.offset + added.size;
    uint64_t hi = old_end > added_end ? old_end : added_end;
    if (lo != old.offset || hi != old_end) {
        r->authority.root = erase_node(&r->authority, r->authority.root, ticket.index);
        w->interval.range = (RcSourceRange){old.backing, lo, hi - lo};
        w->native_buffer = (RcRegistryBufferLocation){0};
        w->native_buffer_conflict = 1;
        buffer_payload_drop(&w->native_payload);
        w->interval.left = w->interval.right = 0;
        r->authority.root = insert_node(&r->authority, r->authority.root, ticket.index);
    }
    invalidate_schedule(r, w->interval.range, ticket, w->position, RC_REGISTRY_SCHEDULE_EXPAND);
    unlock_registry(r); return RC_REGISTRY_OK;
}
static int writer_handed(RcRegistry *r, const Writer *w, uint64_t submission) {
    RcRegistryPosition last = w->position;
    for (uint32_t i = w->image_count; i; --i)
        if (w->image_positions[i - 1].active || w->image_positions[i - 1].unresolved) {
            last = w->image_positions[i - 1].position; break;
        }
    return !rc_registry_position_valid(w->position) ||
        (w->position.scene == submission && rc_registry_position_compare(last, r->handed) <= 0);
}
RcRegistryResult rc_registry_write_submit(RcRegistry *r, RcRegistryWrite ticket, uint64_t submission) {
    if (!r || !submission) return RC_REGISTRY_INVALID;
    lock_registry(r); Writer *w = writer_of(r, ticket);
    if (!w || w->state != WRITE_PENDING) { unlock_registry(r); return RC_REGISTRY_STALE; }
    if (!writer_handed(r, w, submission)) {
        poison(r); unlock_registry(r); return RC_REGISTRY_STALE;
    }
    w->submission = submission; writer_set_state(r, w, WRITE_SUBMITTED);
    unlock_registry(r); return RC_REGISTRY_OK;
}
static void finish_writer_locked(RcRegistry *r, uint32_t index, int cpu_visible, int notify);

static void cancel_pending_writer(RcRegistry *r, uint32_t index, int may_have_executed) {
    Writer *w = &r->writers[index];
    if (!may_have_executed && w->buffer_phase < BUFFER_PHASE_MAYBE) {
        invalidate_topology(r, w->interval.range);
        free_writer(r, index);
        return;
    }
    // Canceling an unsubmitted tail cannot cancel an earlier native part of that same guest operation.
    int phase_completed = w->buffer_phase == BUFFER_PHASE_DONE;
    w->submission = phase_completed ? w->position.scene : 0;
    w->diagnostic_flags |= phase_completed ? RC_REGISTRY_DIAG_COMPLETED : RC_REGISTRY_DIAG_ABANDONED;
    w->buffer_phase = 0;
    w->phase_address = w->phase_mapping = 0;
    w->native_buffer = (RcRegistryBufferLocation){0};
    finish_writer_locked(r, index, 0, 1);
}

RcRegistryResult rc_registry_write_cancel(RcRegistry *r, RcRegistryWrite ticket) {
    if (!r) return RC_REGISTRY_INVALID;
    lock_registry(r); Writer *w = writer_of(r, ticket);
    if (!w || w->state != WRITE_PENDING) { unlock_registry(r); return RC_REGISTRY_STALE; }
    cancel_pending_writer(r, ticket.index, 0);
    unlock_registry(r); return RC_REGISTRY_OK;
}
static RcRegistryResult write_pending_extents(RcRegistry *r, const RcRegistryWrite *tickets, uint32_t count,
                                               uint64_t submission) {
    if (!r || !tickets || !count) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = RC_REGISTRY_OK;
    for (uint32_t i = 0; i < count && result == RC_REGISTRY_OK; ++i) {
        Writer *w = writer_of(r, tickets[i]);
        if (!w || w->state != WRITE_PENDING || (submission && !writer_handed(r, w, submission)))
            result = RC_REGISTRY_STALE;
        for (uint32_t j = 0; j < i; ++j)
            if (tickets[j].index == tickets[i].index) result = RC_REGISTRY_INVALID;
    }
    if (result == RC_REGISTRY_OK) for (uint32_t i = 0; i < count; ++i) {
        Writer *w = &r->writers[tickets[i].index];
        if (submission) {
            w->submission = submission;
            writer_set_state(r, w, WRITE_SUBMITTED);
        } else cancel_pending_writer(r, tickets[i].index, 0);
    }
    unlock_registry(r);
    return result;
}
RcRegistryResult rc_registry_write_submit_extents(RcRegistry *r, const RcRegistryWrite *tickets, uint32_t count,
                                                 uint64_t submission) {
    return submission ? write_pending_extents(r, tickets, count, submission) : RC_REGISTRY_INVALID;
}
RcRegistryResult rc_registry_write_cancel_extents(RcRegistry *r, const RcRegistryWrite *tickets, uint32_t count) {
    return write_pending_extents(r, tickets, count, 0);
}
RcRegistryResult rc_registry_hold_begin(RcRegistry *r, RcSourceRange range, RcRegistryWriterKind kind,
                                       RcRegistryWrite *out) {
    RcRegistryResult result = rc_registry_write_begin(r, range, kind, out);
    if (result != RC_REGISTRY_OK) return result;
    lock_registry(r);
    Writer *w = writer_of(r, *out);
    // The reservation has not been handed to another owner yet. It already blocks every intervening observation.
    if (w && w->state == WRITE_PENDING) writer_set_state(r, w, WRITE_HOLD);
    else result = RC_REGISTRY_STALE;
    unlock_registry(r); return result;
}
RcRegistryResult rc_registry_image_hold_begin_extents(RcRegistry *r, uint64_t image, const RcInterval *intervals,
                                                      uint32_t count, RcRegistryWrite *out) {
    if (out) memset(out, 0, (size_t)count * sizeof *out);
    if (!r || !out || !image || !intervals || !count) return RC_REGISTRY_INVALID;
    for (uint32_t i = 0; i < count; i++) {
        const RcInterval *interval = &intervals[i];
        if (!interval->mapping || interval->range.backing == RC_GUEST_COMPATIBILITY_BACKING ||
            !rc_range_valid(interval->range) || interval->range.size > UINT64_MAX - interval->address)
            return RC_REGISTRY_INVALID;
    }
    uintptr_t origin = (uintptr_t)__builtin_return_address(0);
    lock_registry(r);
    RcRegistryResult result = write_group_space(r, (RcRegistryPosition){0}, count, origin);
    if (result == RC_REGISTRY_OK) for (uint32_t i = 0; i < count; i++) {
        out[i] = insert_writer(r, intervals[i].range, RC_REGISTRY_IMAGE, (RcRegistryPosition){0}, identity(r), 0,
                               RC_REGISTRY_GUARD_UNCLASSIFIED, origin);
        Writer *writer = &r->writers[out[i].index];
        writer_set_state(r, writer, WRITE_HOLD);
        writer->image = image;
        writer->image_address = intervals[i].address;
        writer->image_mapping = intervals[i].mapping;
    }
    unlock_registry(r);
    return result;
}

void rc_registry_image_hold_unordered(RcRegistry *r, uint64_t image, const RcRegistryWrite *holds, uint32_t count) {
    if (!r || !holds || !image) return;
    lock_registry(r);
    for (uint32_t i = 0; i < count; i++) {
        Writer *w = writer_of(r, holds[i]);
        if (!w || w->state != WRITE_HOLD || w->image != image || !w->native_ordered_effect) continue;
        w->native_ordered_effect = 0;
        w->native_unordered = 1;
        invalidate_content(r, w->interval.range, 0);
    }
    unlock_registry(r);
}

static int native_image_hold_exact(RcRegistry *r, const Writer *w, RcRegistryWrite hold, uint64_t image,
                                   const RcInterval *interval) {
    return w && image_binding_current_locked(r, (RcRegistryImageBinding){image, hold}, interval->range) &&
        w->image_address == interval->address && w->image_mapping == interval->mapping &&
        rc_range_equal(w->interval.range, interval->range);
}

int rc_registry_native_image_written(RcRegistry *r, uint64_t image, const RcInterval *intervals,
    const RcRegistryWrite *holds, uint32_t count, RcRegistryPosition output) {
    if (!r || !image || !intervals || !holds || !count) return 0;
    lock_registry(r);
    NativeImageOutput written = {output, image};
    int exact = rc_registry_position_valid(output) && (output.ordinal & 1) && !r->stats.fail_closed &&
        read_position_open(r, output);
    for (uint32_t i = 0; i < count; ++i) {
        Writer *w = writer_of(r, holds[i]);
        if (!native_image_hold_exact(r, w, holds[i], image, &intervals[i])) {
            exact = 0;
            continue;
        }
        if (w->native_projected) native_projection_unproved(w, output, 0, 0);
        else if (rc_registry_position_valid(w->native_output) && !w->native_bound_whole &&
                 rc_registry_position_compare(w->native_output, output) < 0)
            native_projection_unproved(w, w->native_output, 0, 0);
        native_image_output_clear(r, w);
        authority_overlaps(r, intervals[i].range, native_image_conflict, &written);
    }
    if (exact) for (uint32_t i = 0; i < count; ++i) {
        Writer *w = writer_of(r, holds[i]);
        // An ordered write is an exact producer event, but may cover only a subresource or a
        // draw's surviving fragments. It cannot repair an image layout destroyed by another alias.
        w->native_output = output;
        r->native_image_outputs++;
    }
    unlock_registry(r);
    return exact;
}

typedef struct {
    RcRegistryPosition output, read;
    RcRegistryImageBinding binding;
} NativeImageRead;

static int native_image_read_blocker(RcRegistry *r, uint32_t i, void *data) {
    const NativeImageRead *state = data;
    const Writer *w = &r->writers[i];
    if (i == state->binding.hold.index && w->id == state->binding.hold.id) return 0;
    if (w->state == WRITE_HOLD)
        return !rc_registry_position_valid(w->native_output) ||
            rc_registry_position_compare(w->native_output, state->output) >= 0;
    RcRegistryPosition first = w->position;
    if (w->state == WRITE_PENDING && rc_registry_position_valid(first) &&
        rc_registry_position_compare(first, state->read) > 0) return 0;
    if (w->guard) return 1;
    if (w->state == WRITE_FOREIGN) {
        return !rc_registry_position_valid(w->visibility_latest) ||
            rc_registry_position_compare(w->visibility_latest, state->output) >= 0;
    }
    if (!rc_registry_position_valid(first)) return 1;
    if (w->kind != RC_REGISTRY_IMAGE)
        return rc_registry_position_compare(first, state->output) >= 0;
    if (w->image_incomplete || !w->image_count) return 1;
    for (uint32_t at = image_position_lower(w, state->output); at < w->image_count; ++at) {
        const ImageProducer *p = &w->image_positions[at];
        if (rc_registry_position_compare(p->position, state->read) >= 0) break;
        if (p->unresolved || p->unknown ||
            (p->active && !image_binding_equal(p->image, state->binding))) return 1;
    }
    return 0;
}

int rc_registry_native_image_read_current(RcRegistry *r, uint64_t image, const RcInterval *intervals,
    const RcRegistryWrite *holds, uint32_t count, RcRegistryPosition output, RcRegistryPosition read) {
    if (!r || !image || !intervals || !holds || !count || !rc_registry_position_valid(output) ||
        !rc_registry_position_valid(read) ||
        rc_registry_position_compare(output, read) >= 0) return 0;
    lock_registry(r);
    int exact = !r->stats.fail_closed && read_position_open(r, read) &&
        (output.scene == read.scene || rc_registry_position_compare(output, r->handed) <= 0);
    for (uint32_t i = 0; exact && i < count; ++i) {
        Writer *w = writer_of(r, holds[i]);
        NativeImageRead state = {output, read, {image, holds[i]}};
        exact = native_image_hold_exact(r, w, holds[i], image, &intervals[i]) && !w->native_conflict &&
            !rc_registry_position_compare(w->native_output, output) &&
            !authority_overlaps(r, intervals[i].range, native_image_read_blocker, &state);
    }
    unlock_registry(r);
    return exact;
}

static int native_projected_interval(RcInterval interval) {
    return interval.mapping && interval.range.backing != RC_GUEST_COMPATIBILITY_BACKING &&
           rc_range_valid(interval.range) && interval.range.size <= UINT64_MAX - interval.address;
}

static int native_possible_equal(const Writer *w, const RcRegistryNativeImageWrite *layout) {
    const ForeignCoverage *known = w->image_possible;
    if ((known ? known->count : 0) != layout->written_count) return 0;
    for (uint32_t i = 0; i < layout->written_count; ++i) {
        const RcSourceRange span = layout->written[i];
        uint64_t address = w->image_address + (span.offset - w->interval.range.offset);
        if (known->spans[i].begin != address || known->spans[i].end != address + span.size) return 0;
    }
    return 1;
}

RcRegistryResult rc_registry_native_image_possible(RcRegistry *r, uint64_t image,
    const RcRegistryNativeImageWrite *layouts, uint32_t count) {
    if (!r || !image || !layouts || !count || count > r->capacity.writers) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = r->stats.fail_closed ? RC_REGISTRY_STALE : RC_REGISTRY_OK;
    int install = 0;
    for (uint32_t i = 0; result == RC_REGISTRY_OK && i < count; ++i) {
        const RcRegistryNativeImageWrite *layout = &layouts[i];
        RcInterval interval = layout->owner.interval;
        Writer *w = writer_of(r, layout->owner.hold);
        if (!native_image_hold_exact(r, w, layout->owner.hold, image, &interval)) result = RC_REGISTRY_STALE;
        else if (w->native_unordered || rc_registry_position_valid(w->native_output)) result = RC_REGISTRY_UNSUPPORTED;
        else if (!native_projected_interval(interval) || (layout->written_count && !layout->written))
            result = RC_REGISTRY_INVALID;
        if (result != RC_REGISTRY_OK) break;
        for (uint32_t j = 0; j < i; ++j)
            if (layouts[j].owner.hold.index == layout->owner.hold.index) result = RC_REGISTRY_INVALID;
        uint64_t prior_end = interval.range.offset;
        uint64_t end = prior_end + interval.range.size;
        for (uint32_t j = 0; result == RC_REGISTRY_OK && j < layout->written_count; ++j) {
            RcSourceRange span = layout->written[j];
            if (!rc_range_valid(span) || span.backing != interval.range.backing ||
                span.offset < prior_end || span.offset + span.size > end) result = RC_REGISTRY_INVALID;
            else prior_end = span.offset + span.size;
        }
        if (result == RC_REGISTRY_OK && w->image_possible_known && !native_possible_equal(w, layout))
            result = RC_REGISTRY_STALE;
        for (const NativeImageProjection *p = w->native_projection; result == RC_REGISTRY_OK && p; p = p->next) {
            uint32_t next = 0;
            for (uint32_t j = 0; result == RC_REGISTRY_OK && j < p->count; ++j) {
                uint64_t begin = p->spans[j].begin, end = p->spans[j].end;
                while (next < layout->written_count &&
                       layout->written[next].offset + layout->written[next].size <= begin) next++;
                for (uint32_t k = next; k < layout->written_count && layout->written[k].offset <= begin; ++k) {
                    const RcSourceRange part = layout->written[k];
                    begin = part.offset + part.size;
                    if (begin >= end) break;
                }
                if (begin < end) result = RC_REGISTRY_STALE;
            }
        }
        install |= !w->image_possible_known;
    }
    ForeignCoverage *local[4] = {0}, **staged = local;
    if (result == RC_REGISTRY_OK && install) {
        if (count > sizeof local / sizeof *local) staged = calloc(count, sizeof *staged);
        if (!staged) result = RC_REGISTRY_CAPACITY;
        for (uint32_t i = 0; result == RC_REGISTRY_OK && i < count; ++i) {
            const RcRegistryNativeImageWrite *layout = &layouts[i];
            Writer *w = writer_of(r, layout->owner.hold);
            if (w->image_possible_known || !layout->written_count) continue;
            ForeignCoverage *shape = foreign_coverage_alloc(r, layout->written_count);
            if (!shape) { result = RC_REGISTRY_CAPACITY; break; }
            staged[i] = shape;
            for (uint32_t j = 0; j < shape->count; ++j) {
                const RcSourceRange span = layout->written[j];
                const uint64_t address = w->image_address + (span.offset - w->interval.range.offset);
                shape->spans[j] = (ForeignSpan){.begin = address, .end = address + span.size};
            }
        }
        for (uint32_t i = 0; staged && i < count; ++i) {
            Writer *w = writer_of(r, layouts[i].owner.hold);
            if (result == RC_REGISTRY_OK && !w->image_possible_known) {
                w->image_possible = staged[i];
                w->image_possible_known = 1;
            } else foreign_coverage_release(r, staged[i]);
        }
        if (staged != local) free(staged);
    }
    unlock_registry(r);
    return result;
}

static int native_possible_covers(const Writer *w, RcSourceRange range) {
    if (!w->image_possible_known) return 1;
    const ForeignCoverage *shape = w->image_possible;
    if (!shape) return 0;
    uint64_t at = w->image_address + (range.offset - w->interval.range.offset);
    const uint64_t end = at + range.size;
    for (uint32_t i = foreign_span_first(shape, at); i < shape->count; ++i) {
        ForeignSpan span = shape->spans[i];
        if (span.begin > at) return 0;
        if (span.end >= end) return 1;
        at = span.end;
    }
    return 0;
}

static void native_possible_revoke(RcRegistry *r, Writer *w) {
    foreign_coverage_release(r, w->image_possible);
    w->image_possible = NULL;
    w->image_possible_known = 0;
}

typedef struct { RcRegistryPosition output; RcRegistryImageBinding binding; int matched; } NativeImageAssociation;
static int native_projection_association(RcRegistry *r, uint32_t i, void *data) {
    NativeImageAssociation *state = data;
    Writer *w = &r->writers[i];
    if (w->kind != RC_REGISTRY_IMAGE || w->image_incomplete) return 0;
    uint32_t at = image_position_lower(w, state->output);
    if (at == w->image_count) return 0;
    ImageProducer *p = &w->image_positions[at];
    if (rc_registry_position_compare(p->position, state->output) || !p->active || p->unresolved ||
        p->unknown || !image_binding_equal(p->image, state->binding)) return 0;
    p->projected = 1;
    state->matched = 1;
    return 0;
}

typedef struct { RcRegistryPosition output; RcSourceRange range; } NativeImageDeclaration;

static int native_projection_declared(RcRegistry *r, uint32_t i, void *data) {
    const NativeImageDeclaration *declaration = data;
    const Writer *w = &r->writers[i];
    if (w->state != WRITE_PENDING || w->kind != RC_REGISTRY_IMAGE ||
        !rc_range_equal(w->interval.range, declaration->range)) return 0;
    uint32_t at = image_position_lower(w, declaration->output);
    return at < w->image_count && !rc_registry_position_compare(w->image_positions[at].position, declaration->output);
}

static void native_projection_compact(RcRegistry *r, Writer *w) {
    // Only a completed successor can discard an older part of the same physical provider.
    NativeImageProjection *latest = w->native_projection;
    while (latest && (!latest->confirmed || !latest->completed || rc_registry_position_compare(latest->output, r->handed) > 0))
        latest = latest->next;
    if (!latest) return;
    NativeImageProjection **at = &latest->next;
    while (*at) {
        int kept = 1;
        for (uint32_t i = 0; kept && i < latest->count; ++i) {
            NativeImageSpan span = latest->spans[i];
            RcSourceRange range = {w->interval.range.backing, span.begin, span.end - span.begin};
            kept = native_projection_trim(r, w, at, range);
        }
        if (kept) at = &(*at)->next;
    }
}

static RcRegistryResult native_projection_validate(RcRegistry *r, uint64_t image,
    const RcRegistryNativeImageWrite *writes, uint32_t count, RcRegistryPosition output, int bound) {
    if (r->stats.fail_closed || !read_position_open(r, output)) return RC_REGISTRY_STALE;
    for (uint32_t i = 0; i < count; ++i) {
        const RcRegistryNativeImageWrite *write = &writes[i];
        RcInterval interval = write->owner.interval;
        Writer *w = writer_of(r, write->owner.hold);
        if (!native_image_hold_exact(r, w, write->owner.hold, image, &interval) ||
            (w->native_projection && rc_registry_position_compare(output, w->native_projection->output) <= 0))
            return RC_REGISTRY_STALE;
        if (bound ? (w->native_projected || w->native_conflict ||
                     rc_registry_position_compare(w->native_output, output)) :
                    rc_registry_position_valid(w->native_output)) return RC_REGISTRY_STALE;
        if (!native_projected_interval(interval) || !write->written || !write->written_count)
            return RC_REGISTRY_INVALID;
        for (uint32_t j = 0; j < i; ++j)
            if (writes[j].owner.hold.index == write->owner.hold.index) return RC_REGISTRY_INVALID;
        uint64_t prior_end = interval.range.offset;
        uint64_t end = prior_end + interval.range.size;
        for (uint32_t j = 0; j < write->written_count; ++j) {
            RcSourceRange range = write->written[j];
            if (range.backing != interval.range.backing || range.offset < prior_end ||
                !rc_range_valid(range) || range.offset + range.size > end) return RC_REGISTRY_INVALID;
            if (!bound && !native_possible_covers(w, range)) {
                // The adapter already recorded the operation. A contradictory actual write may not
                // use the old geometry as an exclusion proof when this physical image retires.
                native_possible_revoke(r, w);
                return RC_REGISTRY_INVALID;
            }
            prior_end = range.offset + range.size;
        }
    }
    return RC_REGISTRY_OK;
}

static RcRegistryResult native_image_write_fragments(RcRegistry *r, uint64_t image,
    const RcRegistryNativeImageWrite *writes, uint32_t count, RcRegistryPosition output, int bound) {
    if (!r || !image || !writes || !count || count > r->capacity.writers ||
        !rc_registry_position_valid(output) || !(output.ordinal & 1)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = native_projection_validate(r, image, writes, count, output, bound);
    NativeImageProjection *staged = NULL;
    if (result == RC_REGISTRY_OK) for (uint32_t i = 0; i < count; ++i) {
        const RcRegistryNativeImageWrite *write = &writes[i];
        Writer *w = writer_of(r, write->owner.hold);
        native_projection_compact(r, w);
        uint32_t spans = write->written_count;
        NativeImageProjection *p = NULL;
        if (spans <= image_span_room(r)) p = malloc(sizeof *p + (size_t)spans * sizeof *p->spans);
        if (!p) { result = RC_REGISTRY_CAPACITY; break; }
        *p = (NativeImageProjection){.next = staged, .output = output, .capacity = spans, .physically_bound = (uint8_t)bound};
        for (uint32_t j = 0; j < spans; ++j) {
            RcSourceRange range = write->written[j];
            uint64_t end = range.offset + range.size;
            if (p->count && p->spans[p->count - 1].end == range.offset) p->spans[p->count - 1].end = end;
            else p->spans[p->count++] = (NativeImageSpan){range.offset, end};
        }
        r->native_projection_spans += spans;
        staged = p;
    }
    if (result == RC_REGISTRY_OK) for (uint32_t i = count; i; --i) {
        Writer *w = writer_of(r, writes[i - 1].owner.hold);
        NativeImageProjection *p = staged;
        staged = p->next;
        p->next = w->native_projection;
        w->native_projection = p;
        if (bound) w->native_bound_whole = NATIVE_BOUND_RECORDED;
        else w->native_projected = 1;
        NativeImageAssociation association = {.output = output, .binding = {image, writes[i - 1].owner.hold}};
        authority_overlaps(r, w->interval.range, native_projection_association, &association);
        p->confirmed = association.matched;
    }
    else {
        while (staged) native_projection_remove(r, &staged);
        // Native work was already recorded. On refusal it remains an unknown image writer, even if the
        // previous output of this same image had a successful projection.
        for (uint32_t i = 0; i < count; ++i) {
            Writer *w = writer_of(r, writes[i].owner.hold);
            if (!w || w->state != WRITE_HOLD || w->image != image) continue;
            NativeImageDeclaration declaration = {output, w->interval.range};
            // A live orderly claim already owns this uncertainty, including its possible cancellation.
            if (!authority_overlaps(r, declaration.range, native_projection_declared, &declaration))
                native_projection_unproved(w, output, 0, 0);
        }
    }
    unlock_registry(r);
    return result;
}

RcRegistryResult rc_registry_native_image_write_fragments(RcRegistry *r, uint64_t image,
    const RcRegistryNativeImageWrite *writes, uint32_t count, RcRegistryPosition output) {
    return native_image_write_fragments(r, image, writes, count, output, 0);
}

RcRegistryResult rc_registry_native_image_bound_written(RcRegistry *r, uint64_t image,
    const RcRegistryNativeImageWrite *writes, uint32_t count, RcRegistryPosition output) {
    return native_image_write_fragments(r, image, writes, count, output, 1);
}

typedef struct {
    RcSourceRange range;
    RcInterval demand;
    RcRegistryPosition read;
    uint64_t end;
    RcRegistryImageBinding binding, resident;
    const NativeImageProjection *winner;
    RcRegistryNativeImageFailure *failure;
    const RcRegistryBufferSpan *baseline;
    RcRegistryBufferLocation buffer;
    RcRegistryPosition buffer_output, preserved_output;
    int inclusive, ambiguous, bound_snapshot, buffer_conflict, use_buffer, completed_only;
} NativeImageSlice;

static int native_slice_includes(const NativeImageSlice *slice, RcRegistryPosition position) {
    int order = rc_registry_position_compare(position, slice->read);
    return order < 0 || (slice->inclusive && !order);
}

// Only a live, independently placed image with an exact recorded effect can give padding back to
// the older source. An equal normalized backing alone does not prove the same guest placement.
static uint64_t native_independent_span_end(const Writer *held, const NativeImageSlice *slice, int *padding) {
    *padding = 0;
    if (!held || !held->image_possible_known || !held->native_ordered_effect || held->image_forgotten ||
        held->native_unordered || held->native_conflict || rc_registry_position_valid(held->native_output) ||
        !slice->demand.mapping || slice->demand.mapping != held->image_mapping ||
        slice->range.backing != held->interval.range.backing) return slice->end;

    const RcSourceRange extent = held->interval.range;
    const uint64_t begin = slice->range.offset < extent.offset ? extent.offset : slice->range.offset;
    const uint64_t extent_end = extent.offset + extent.size;
    if (begin >= slice->end || begin >= extent_end) return slice->end;
    const uint64_t guest = held->image_address + (begin - extent.offset);
    if (guest != slice->demand.address + (begin - slice->demand.range.offset)) return slice->end;
    if (begin != slice->range.offset) return begin;

    const ForeignCoverage *possible = held->image_possible;
    uint64_t end = extent_end < slice->end ? extent_end : slice->end;
    *padding = 1;
    if (possible) {
        const uint32_t i = foreign_span_first(possible, guest);
        if (i < possible->count) {
            const ForeignSpan span = possible->spans[i];
            *padding = guest < span.begin;
            const uint64_t boundary = *padding ? span.begin : span.end;
            if (boundary - guest < end - begin) end = begin + (boundary - guest);
        }
    }
    return end;
}

static int native_independent_padding(const Writer *held, const NativeImageSlice *slice) {
    int padding;
    return native_independent_span_end(held, slice, &padding) == slice->end && padding;
}

static int native_independent_completed(const Writer *held, const NativeImageSlice *slice) {
    return rc_registry_position_valid(held->native_unprojected) &&
        native_slice_includes(slice, held->native_unprojected) &&
        !(rc_registry_position_valid(held->native_unprojected_opaque) &&
          native_slice_includes(slice, held->native_unprojected_opaque));
}

static const Writer *native_independent_pending(RcRegistry *r, const Writer *claim, const ImageProducer *producer) {
    if (!producer->active || producer->projected || producer->unknown || producer->unresolved ||
        !image_binding_current_locked(r, producer->image, claim->interval.range)) return NULL;
    const Writer *held = writer_of(r, producer->image.hold);
    return rc_range_equal(held->interval.range, claim->interval.range) ? held : NULL;
}

static RcRegistryPosition native_snapshot_buffer_output(const RcRegistry *r, const Writer *w, RcRegistryPosition read,
                                                          int inclusive) {
    if (w->kind != RC_REGISTRY_BUFFER || w->kinds != (1u << RC_REGISTRY_BUFFER) || w->guard || w->unordered ||
        w->state == WRITE_HOLD || w->foreign_coverage || w->native_buffer_conflict || !w->native_buffer.allocation ||
        (w->diagnostic_flags & RC_REGISTRY_DIAG_ABANDONED)) return (RcRegistryPosition){0};
    RcRegistryPosition output = w->state == WRITE_FOREIGN ? w->visibility_latest : w->position;
    int order = rc_registry_position_compare(output, read);
    if (!rc_registry_position_valid(output) || order > 0 || (!inclusive && !order) ||
        (w->state == WRITE_FOREIGN && !w->native_buffer_compacted &&
            rc_registry_position_compare(w->visibility_position, output)) ||
        (output.scene != read.scene && rc_registry_position_compare(output, r->handed) > 0))
        return (RcRegistryPosition){0};
    return output;
}

static void native_snapshot_select_buffer(RcRegistry *r, const Writer *w, NativeImageSlice *slice) {
    if (slice->completed_only) {
        if (w->state != WRITE_FOREIGN || !(w->diagnostic_flags & RC_REGISTRY_DIAG_COMPLETED)) return;
    } else if (!slice->baseline) return;
    RcRegistryPosition output = native_snapshot_buffer_output(r, w, slice->read, slice->inclusive);
    if (!rc_registry_position_valid(output)) return;
    const RcSourceRange range = w->interval.range;
    if (range.offset >= slice->end) return;
    if (range.offset > slice->range.offset) {
        slice->end = range.offset;
        return;
    }
    const uint64_t end = range.offset + range.size;
    if (end < slice->end) slice->end = end;
    RcRegistryBufferLocation actual = w->native_buffer;
    actual.offset += slice->range.offset - range.offset;
    const int mismatch = !slice->completed_only && !buffer_native_equal(slice->buffer, actual);
    int newer = rc_registry_position_compare(output, slice->buffer_output);
    if (newer > 0) {
        slice->buffer_output = output;
        slice->buffer_conflict = mismatch;
    } else if (!newer && mismatch) slice->buffer_conflict = 1;
}

static int native_projection_select(RcRegistry *r, uint32_t i, void *data) {
    NativeImageSlice *slice = data;
    const Writer *w = &r->writers[i];
    if (w->state != WRITE_HOLD) {
        native_snapshot_select_buffer(r, w, slice);
        if (slice->demand.mapping && w->state == WRITE_PENDING && w->kind == RC_REGISTRY_IMAGE &&
            !w->image_incomplete && !future_pending_writer(w, slice->read)) {
            for (uint32_t j = 0; j < w->image_count; ++j) {
                const ImageProducer *p = &w->image_positions[j];
                if (!native_slice_includes(slice, p->position)) break;
                const Writer *held = native_independent_pending(r, w, p);
                int padding;
                slice->end = native_independent_span_end(held, slice, &padding);
            }
        }
        return 0;
    }
    if (slice->demand.mapping && native_independent_completed(w, slice)) {
        int padding;
        slice->end = native_independent_span_end(w, slice, &padding);
    }
    for (const NativeImageProjection *p = w->native_projection; p; p = p->next) {
        if (!native_slice_includes(slice, p->output) || (slice->completed_only && (!p->completed || !p->confirmed))) continue;
        uint32_t at = native_projection_first(p, slice->range.offset);
        if (at == p->count || p->spans[at].begin >= slice->end) continue;
        const NativeImageSpan span = p->spans[at];
        if (span.begin > slice->range.offset) { slice->end = span.begin; continue; }
        if (span.end < slice->end) slice->end = span.end;
        int newer = slice->winner ? rc_registry_position_compare(p->output, slice->winner->output) : 1;
        if (newer > 0) {
            slice->winner = p;
            slice->binding = (RcRegistryImageBinding){w->image, {w->id, i}};
            slice->ambiguous = 0;
        } else if (!newer && !image_binding_equal(slice->binding, (RcRegistryImageBinding){w->image, {w->id, i}}))
            slice->ambiguous = 1;
    }
    return 0;
}

static int native_projection_blocked(const NativeImageSlice *slice, const Writer *writer, uint32_t index,
    RcRegistryNativeImageBlock cause, RcRegistryPosition output) {
    RcRegistryNativeImageFailure *failure = slice->failure;
    if (failure) {
        failure->blocker = (RcRegistryWrite){writer->id, index};
        failure->blocking_interval = (RcInterval){.address = writer->image_address, .mapping = writer->image_mapping,
                                                  .range = writer->interval.range};
        failure->blocking_output = output;
        failure->cause = cause;
        failure->writer_kind = writer->kind;
        failure->writer_state = writer->state;
        failure->guard_reason = writer->diagnostic_guard_reason;
    }
    return 1;
}

static int native_projection_blocker(RcRegistry *r, uint32_t i, void *data) {
    const NativeImageSlice *slice = data;
    const Writer *w = &r->writers[i];
    RcRegistryPosition winner = slice->use_buffer ? slice->buffer_output :
        slice->winner ? slice->winner->output : (RcRegistryPosition){0};
    if (w->state == WRITE_HOLD) {
        if (w->native_unordered)
            return native_projection_blocked(slice, w, i, RC_NATIVE_IMAGE_BLOCK_UNORDERED_HOLD, (RcRegistryPosition){0});
        if (rc_registry_position_valid(w->native_unprojected) &&
            native_slice_includes(slice, w->native_unprojected) &&
            rc_registry_position_compare(w->native_unprojected_latest, winner) >= 0 &&
            !(slice->resident.hold.id == w->id &&
              rc_registry_position_compare(w->native_unprojected_uncertain, winner) < 0) &&
            !(slice->binding.hold.id == w->id && rc_registry_position_valid(slice->preserved_output)) &&
            !(native_independent_completed(w, slice) && native_independent_padding(w, slice)))
            return native_projection_blocked(slice, w, i, RC_NATIVE_IMAGE_BLOCK_UNPROJECTED, w->native_unprojected_latest);
        if (rc_registry_position_valid(w->native_output)) {
            // A physically bound optimal image can clobber the raw buffer, including its padding.
            // Ignore that placement only under a later exact overwrite to the selected native buffer.
            if (slice->use_buffer)
                return rc_registry_position_compare(w->native_output, winner) >= 0 &&
                    native_projection_blocked(slice, w, i, RC_NATIVE_IMAGE_BLOCK_UNPROJECTED, w->native_output);
            if (slice->bound_snapshot && w->native_bound_whole && !w->native_conflict) {
                // The actual whole native write declared every meaningful texel. Its raw bound allocation also
                // contains padding, which cannot hide the earlier writer the projection sweep chose for this slice.
                for (const NativeImageProjection *p = w->native_projection; p; p = p->next)
                    if (p->physically_bound && p->confirmed &&
                        !rc_registry_position_compare(p->output, w->native_output) &&
                        native_slice_includes(slice, p->output)) return 0;
            }
            return !w->native_projected && rc_registry_position_compare(w->native_output, winner) >= 0 &&
                native_projection_blocked(slice, w, i, RC_NATIVE_IMAGE_BLOCK_UNPROJECTED, w->native_output);
        }
        // Positioned actual effects remain in the authority tree until completion or cancellation. A retained
        // image placement alone has no further writer order. Completed unprojected effects remain above.
        return !w->native_projected && !w->native_ordered_effect &&
            native_projection_blocked(slice, w, i, RC_NATIVE_IMAGE_BLOCK_UNQUALIFIED_HOLD, (RcRegistryPosition){0});
    }
    if (future_pending_writer(w, slice->read)) return 0;
    if (w->guard) return native_projection_blocked(slice, w, i, RC_NATIVE_IMAGE_BLOCK_GUARD, w->position);
    if (slice->use_buffer && w->kind == RC_REGISTRY_BUFFER) {
        RcRegistryPosition output = native_snapshot_buffer_output(r, w, slice->read, slice->inclusive);
        if (rc_registry_position_valid(output) && !rc_registry_position_compare(output, winner)) {
            RcRegistryBufferLocation actual = w->native_buffer;
            actual.offset += slice->range.offset - w->interval.range.offset;
            if (buffer_native_equal(actual, slice->buffer)) return 0;
        }
    }
    if (w->state == WRITE_FOREIGN)
        return (!rc_registry_position_valid(w->visibility_latest) ||
            rc_registry_position_compare(w->visibility_latest, winner) >= 0) &&
            native_projection_blocked(slice, w, i, RC_NATIVE_IMAGE_BLOCK_FOREIGN, w->visibility_latest);
    if (!rc_registry_position_valid(w->position))
        return native_projection_blocked(slice, w, i, RC_NATIVE_IMAGE_BLOCK_UNPOSITIONED, w->position);
    if (w->kind != RC_REGISTRY_IMAGE)
        return rc_registry_position_compare(w->position, winner) >= 0 &&
            native_projection_blocked(slice, w, i, RC_NATIVE_IMAGE_BLOCK_OTHER_WRITER, w->position);
    if (!w->image_count || w->image_incomplete)
        return native_projection_blocked(slice, w, i, RC_NATIVE_IMAGE_BLOCK_MISSING_OUTPUT, w->position);
    for (uint32_t at = image_position_lower(w, winner); at < w->image_count; ++at) {
        const ImageProducer *p = &w->image_positions[at];
        if (!native_slice_includes(slice, p->position)) break;
        if (p->unresolved)
            return native_projection_blocked(slice, w, i, RC_NATIVE_IMAGE_BLOCK_PENDING_OUTPUT, p->position);
        if (p->unknown || (p->active && !p->projected &&
            !(slice->resident.image && p->preserves_native && image_binding_equal(p->image, slice->resident) &&
              (p->position.scene == slice->read.scene || rc_registry_position_compare(p->position, r->handed) <= 0)) &&
            !(p->exported_completed && image_binding_equal(p->image, slice->binding) &&
              rc_registry_position_compare(p->position, slice->preserved_output) <= 0) &&
            !native_independent_padding(native_independent_pending(r, w, p), slice)))
            return native_projection_blocked(slice, w, i, RC_NATIVE_IMAGE_BLOCK_UNKNOWN_OUTPUT, p->position);
        // The joint sweep has already chosen the latest projected owner for every byte of this slice.
        // An exactly associated image producer whose footprint misses the slice contributes no write here.
    }
    return 0;
}

static RcRegistryPosition native_preserved_projection(const RcRegistry *r, const NativeImageSlice *slice,
                                                       const Writer *hold) {
    const NativeImageProjection *proof = slice->winner;
    const RcRegistryPosition output = hold->native_preserved_projection;
    if (!proof->completed || proof->physically_bound || !hold->image_possible_known ||
        !hold->native_ordered_effect || hold->native_unordered || hold->native_conflict ||
        rc_registry_position_valid(hold->native_output) ||
        rc_registry_position_compare(output, proof->output) <= 0 ||
        rc_registry_position_compare(output, hold->native_unprojected_latest) ||
        rc_registry_position_compare(output, r->handed) > 0 ||
        rc_registry_position_compare(hold->native_unprojected_uncertain, proof->output) >= 0 ||
        rc_registry_position_compare(hold->native_resident_only, proof->output) >= 0 ||
        !native_slice_includes(slice, output) || hold->image_mapping != slice->demand.mapping ||
        slice->range.offset < hold->interval.range.offset ||
        hold->image_address + (slice->range.offset - hold->interval.range.offset) !=
            slice->demand.address + (slice->range.offset - slice->demand.range.offset))
        return (RcRegistryPosition){0};
    return output;
}

static RcRegistryNativeImageGate native_projection_read_gate(RcRegistry *r, NativeImageSlice *slice, uint64_t image,
                                                              const Writer *hold) {
    if (slice->baseline) {
        RcRegistryPosition image_output = slice->winner ? slice->winner->output : (RcRegistryPosition){0};
        int newer = rc_registry_position_compare(slice->buffer_output, image_output);
        if (slice->winner && !newer) return RC_NATIVE_IMAGE_READ_AMBIGUOUS;
        slice->use_buffer = !slice->winner || newer > 0;
        if (slice->use_buffer) {
            int blocked = authority_overlaps(r, slice->range, native_projection_blocker, slice);
            return blocked || slice->buffer_conflict ? RC_NATIVE_IMAGE_READ_BLOCKED : RC_NATIVE_IMAGE_READ_OK;
        }
    }
    if (!slice->winner) {
        if (slice->failure) authority_overlaps(r, slice->range, native_projection_blocker, slice);
        return RC_NATIVE_IMAGE_READ_NO_PRODUCER;
    }
    if (slice->ambiguous) return RC_NATIVE_IMAGE_READ_AMBIGUOUS;
    if (!slice->winner->confirmed) return RC_NATIVE_IMAGE_READ_TENTATIVE;
    if (!hold || hold->image_forgotten) return RC_NATIVE_IMAGE_READ_LOST_OWNER;
    if (slice->winner->physically_bound && (!slice->bound_snapshot || hold->native_conflict ||
        !native_bound_output(hold, slice->binding, slice->winner->output))) return RC_NATIVE_IMAGE_READ_BOUND;
    if (image && (slice->binding.image != image ||
        (slice->resident.image && !image_binding_equal(slice->binding, slice->resident))))
        return RC_NATIVE_IMAGE_READ_DIFFERENT_IMAGE;
    if (slice->winner->output.scene != slice->read.scene && rc_registry_position_compare(slice->winner->output, r->handed) > 0)
        return RC_NATIVE_IMAGE_READ_UNHANDED;
    // Keep the original winner for comparisons with every other writer. The handed continuation
    // changes the bytes it supplies and their provenance only where that original proof still wins.
    slice->preserved_output = native_preserved_projection(r, slice, hold);
    return authority_overlaps(r, slice->range, native_projection_blocker, slice)
        ? RC_NATIVE_IMAGE_READ_BLOCKED : RC_NATIVE_IMAGE_READ_OK;
}

static RcRegistryResult buffer_preserve_input_locked(RcRegistry *r, RcRegistryBufferSpan span,
    RcRegistryPosition read, int initialized) {
    RcSourceRange remaining = span.interval.range;
    while (remaining.size) {
        NativeImageSlice slice = {.range = remaining, .demand = span.interval, .read = read,
            .end = remaining.offset + remaining.size, .baseline = &span, .buffer = span.native,
            .bound_snapshot = 1};
        slice.buffer.offset += remaining.offset - span.interval.range.offset;
        overlaps(r, &r->authority, r->authority.root, remaining, native_projection_select, &slice);
        slice.range.size = slice.end - slice.range.offset;
        const Writer *held = writer_of(r, slice.binding.hold);
        if (native_projection_read_gate(r, &slice, 0, held) != RC_NATIVE_IMAGE_READ_OK || !slice.use_buffer ||
            (!initialized && (!rc_registry_position_valid(slice.buffer_output) ||
                (span.native.incarnation && slice.buffer_output.scene != read.scene))))
            return RC_REGISTRY_BLOCKED;
        remaining.offset = slice.end;
        remaining.size -= slice.range.size;
    }
    return RC_REGISTRY_OK;
}

RcRegistryResult rc_registry_buffer_preserve_input_at(RcRegistry *r, RcRegistryBufferSpan span,
    RcRegistryPosition read, int initialized) {
    if (!r || !buffer_native_span_valid(span) || !buffer_phase_read_valid(read)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    const RcRegistryResult result = !r->stats.fail_closed && read_position_open(r, read) ?
        buffer_preserve_input_locked(r, span, read, initialized) : RC_REGISTRY_STALE;
    unlock_registry(r);
    return result;
}

RcRegistryResult rc_registry_buffer_preserved_record(RcRegistry *r, RcRegistryWrite ticket,
    RcRegistryBufferSpan span, RcRegistryPosition read, int initialized) {
    if (!r || !buffer_native_span_valid(span) || !buffer_phase_read_valid(read) || read.ordinal == UINT64_MAX)
        return RC_REGISTRY_INVALID;
    const RcRegistryPosition output = {read.scene, read.ordinal + 1};
    lock_registry(r);
    RcRegistryResult result = RC_REGISTRY_STALE;
    Writer *w = writer_of(r, ticket);
    if (!r->stats.fail_closed && read_position_open(r, read) && read_position_open(r, output) &&
        w && w->state == WRITE_PENDING && !w->guard && !w->unordered && w->kind == RC_REGISTRY_BUFFER &&
        w->kinds == (1u << RC_REGISTRY_BUFFER) && !rc_registry_position_compare(w->position, output) &&
        rc_range_equal(w->interval.range, span.interval.range)) {
        result = buffer_preserve_input_locked(r, span, read, initialized);
        if (result == RC_REGISTRY_OK) result = buffer_result_record_locked(r, w, span, output);
    }
    unlock_registry(r);
    return result;
}

static void native_fragment_failure(RcRegistryNativeImageFailure *failure, RcRegistryNativeImageGate gate,
    RcInterval demand, uint32_t index, const NativeImageSlice *slice) {
    if (!failure) return;
    failure->gate = gate;
    failure->demand = index;
    failure->interval = (RcInterval){.address = demand.address + (slice->range.offset - demand.range.offset),
                                     .mapping = demand.mapping, .range = slice->range};
    failure->selected = slice->binding;
    if (slice->winner) failure->selected_output = slice->winner->output;
}

static int native_fragment_adjacent(const RcRegistryNativeImageFragment *before,
                                    const RcRegistryNativeImageFragment *after) {
    return before->demand == after->demand && image_binding_equal(before->binding, after->binding) &&
           !rc_registry_position_compare(before->output, after->output) &&
           before->interval.address + before->interval.range.size == after->interval.address;
}

typedef struct {
    const RcRegistryBufferSpan *baselines;
    RcRegistryNativeImageSnapshotFragment *out;
    uint32_t count;
} NativeImageMixed;

static const RcRegistryBufferSpan *native_snapshot_baseline(const NativeImageMixed *mixed, RcInterval demand,
    NativeImageSlice *slice, uint32_t *index) {
    if (!mixed) return NULL;
    const RcRegistryBufferSpan *found = NULL;
    const uint64_t begin = slice->range.offset;
    int conflict = 0;
    for (uint32_t i = 0; i < mixed->count; ++i) {
        const RcRegistryBufferSpan *base = &mixed->baselines[i];
        const RcInterval span = base->interval;
        const uint64_t span_end = span.range.offset + span.range.size;
        if (span.mapping != demand.mapping || span.range.backing != demand.range.backing ||
            span.range.offset >= slice->end || span_end <= begin) continue;
        const uint64_t overlap = span.range.offset > begin ? span.range.offset : begin;
        if (demand.address + (overlap - demand.range.offset) != span.address + (overlap - span.range.offset))
            continue;
        if (span.range.offset > begin) {
            slice->end = span.range.offset;
            continue;
        }
        if (span_end < slice->end) slice->end = span_end;
        RcRegistryBufferLocation at = base->native;
        at.offset += begin - span.range.offset;
        if (found) conflict |= !buffer_native_equal(slice->buffer, at);
        else {
            found = base;
            slice->buffer = at;
            *index = i;
        }
    }
    // An independent image can still supply a slice whose raw certificates disagree.
    return conflict ? NULL : found;
}

typedef struct {
    const RcRegistryWrite *holds;
    uint32_t count;
} NativeImageResident;

static RcRegistryImageBinding native_image_resident_owner(RcRegistry *r, const NativeImageResident *resident,
    uint64_t image, RcInterval demand) {
    for (uint32_t i = 0; i < resident->count; ++i) {
        const RcRegistryImageBinding candidate = {image, resident->holds[i]};
        const Writer *held = writer_of(r, candidate.hold);
        if (image_binding_current_locked(r, candidate, demand.range) && held->native_projected &&
            !held->native_unordered && !held->native_conflict && !rc_registry_position_valid(held->native_output) &&
            held->image_mapping == demand.mapping &&
            held->image_address + (demand.range.offset - held->interval.range.offset) == demand.address)
            return candidate;
    }
    return (RcRegistryImageBinding){0};
}

typedef enum {
    NATIVE_IMAGE_OPEN_READ,
    NATIVE_IMAGE_LATEST_HANDOFF,
    NATIVE_IMAGE_FIXED_HANDOFF,
} NativeImageReadCut;

static RcRegistryResult native_image_fragments(RcRegistry *r, const RcInterval *demands, uint32_t count,
    RcRegistryPosition read, uint64_t image, RcRegistryNativeImageFragment *out, uint32_t capacity,
    uint32_t *fragment_count, RcRegistryNativeImageFailure *failure, int bound_snapshot, const NativeImageMixed *mixed,
    NativeImageReadCut cut, const NativeImageResident *resident) {
    const int handed_only = cut != NATIVE_IMAGE_OPEN_READ;
    if (failure) *failure = (RcRegistryNativeImageFailure){0};
    if (fragment_count) *fragment_count = 0;
    if (!r || !demands || !count || !fragment_count || (!handed_only && !rc_registry_position_valid(read)) ||
        (capacity && !(mixed ? mixed->out != NULL : out != NULL)) || (mixed && mixed->count && !mixed->baselines)) {
        if (failure) failure->gate = RC_NATIVE_IMAGE_READ_INVALID_DEMAND;
        return RC_REGISTRY_INVALID;
    }
    if (mixed) for (uint32_t j = 0; j < mixed->count; ++j) {
        if (!buffer_native_span_valid(mixed->baselines[j])) {
            if (failure) failure->gate = RC_NATIVE_IMAGE_READ_INVALID_DEMAND;
            return RC_REGISTRY_INVALID;
        }
    }
    lock_registry(r);
    RcRegistryResult result = RC_REGISTRY_OK;
    uint32_t used = 0;
    if (cut == NATIVE_IMAGE_LATEST_HANDOFF) read = r->handed;
    if (r->stats.fail_closed || (cut == NATIVE_IMAGE_FIXED_HANDOFF && rc_registry_position_compare(read, r->handed)) ||
        !(handed_only ? rc_registry_position_valid(read) : read_position_open(r, read))) {
        if (failure) failure->gate = RC_NATIVE_IMAGE_READ_CLOSED;
        result = RC_REGISTRY_STALE;
    }
    for (uint32_t i = 0; i < count && result == RC_REGISTRY_OK; ++i) {
        if (!native_projected_interval(demands[i])) {
            if (failure) *failure = (RcRegistryNativeImageFailure){
                .gate = RC_NATIVE_IMAGE_READ_INVALID_DEMAND, .demand = i, .interval = demands[i]};
            result = RC_REGISTRY_INVALID;
            break;
        }
        const RcRegistryImageBinding owner = resident ? native_image_resident_owner(r, resident, image, demands[i])
            : (RcRegistryImageBinding){0};
        if (resident && !owner.image) {
            result = RC_REGISTRY_STALE;
            break;
        }
        RcSourceRange remaining = demands[i].range;
        while (remaining.size) {
            NativeImageSlice slice = {.range = remaining, .demand = demands[i], .read = read, .resident = owner,
                .end = remaining.offset + remaining.size,
                .failure = failure, .bound_snapshot = bound_snapshot, .inclusive = handed_only};
            uint32_t baseline_index = UINT32_MAX;
            const RcRegistryBufferSpan *baseline = native_snapshot_baseline(mixed, demands[i], &slice, &baseline_index);
            slice.baseline = baseline;
            overlaps(r, &r->authority, r->authority.root, remaining, native_projection_select, &slice);
            slice.range.size = slice.end - slice.range.offset;
            const Writer *held = writer_of(r, slice.binding.hold);
            RcRegistryNativeImageGate gate = native_projection_read_gate(r, &slice, image, held);
            if (gate != RC_NATIVE_IMAGE_READ_OK) {
                native_fragment_failure(failure, gate, demands[i], i, &slice);
                result = RC_REGISTRY_BLOCKED;
                break;
            }
            if (!image) {
                RcRegistryNativeImageSnapshotFragment result_fragment = {.baseline = UINT32_MAX};
                RcRegistryNativeImageFragment *fragment = &result_fragment.fragment;
                fragment->demand = i;
                fragment->interval.range = slice.range;
                if (slice.use_buffer) {
                    fragment->interval.address =
                        baseline->interval.address + (slice.range.offset - baseline->interval.range.offset);
                    fragment->interval.mapping = baseline->interval.mapping;
                    fragment->output = slice.buffer_output;
                    result_fragment.baseline = baseline_index;
                    result_fragment.native_buffer = slice.buffer;
                } else {
                    fragment->interval.address = held->image_address + (slice.range.offset - held->interval.range.offset);
                    fragment->interval.mapping = held->image_mapping;
                    fragment->binding = slice.binding;
                    fragment->output = rc_registry_position_valid(slice.preserved_output) ?
                        slice.preserved_output : slice.winner->output;
                }
                RcRegistryNativeImageSnapshotFragment *prior_mixed = mixed && used ? &mixed->out[used - 1] : NULL;
                RcRegistryNativeImageFragment *prior = !used ? NULL : mixed ? &prior_mixed->fragment : &out[used - 1];
                if (prior && (!mixed || prior_mixed->baseline == result_fragment.baseline) &&
                    native_fragment_adjacent(prior, fragment))
                    prior->interval.range.size += slice.range.size;
                else if (used == capacity) {
                    native_fragment_failure(failure, RC_NATIVE_IMAGE_READ_CAPACITY, demands[i], i, &slice);
                    result = RC_REGISTRY_CAPACITY;
                    break;
                } else {
                    if (mixed) mixed->out[used] = result_fragment;
                    else out[used] = *fragment;
                    used++;
                }
            }
            remaining.offset = slice.end;
            remaining.size -= slice.range.size;
        }
    }
    if (result == RC_REGISTRY_OK) *fragment_count = used;
    unlock_registry(r);
    return result;
}

RcRegistryResult rc_registry_native_image_fragments(RcRegistry *r, const RcInterval *demands, uint32_t count,
    RcRegistryPosition read, RcRegistryNativeImageFragment *out, uint32_t capacity, uint32_t *fragment_count) {
    return native_image_fragments(r, demands, count, read, 0, out, capacity, fragment_count, NULL, 0, NULL, NATIVE_IMAGE_OPEN_READ, NULL);
}

RcRegistryResult rc_registry_native_image_fragments_checked(RcRegistry *r, const RcInterval *demands, uint32_t count,
    RcRegistryPosition read, RcRegistryNativeImageFragment *out, uint32_t capacity, uint32_t *fragment_count,
    RcRegistryNativeImageFailure *failure) {
    return native_image_fragments(r, demands, count, read, 0, out, capacity, fragment_count, failure, 0, NULL, NATIVE_IMAGE_OPEN_READ, NULL);
}

RcRegistryResult rc_registry_native_image_snapshot_fragments(RcRegistry *r, const RcInterval *demands, uint32_t count,
    RcRegistryPosition read, RcRegistryNativeImageFragment *out, uint32_t capacity, uint32_t *fragment_count,
    RcRegistryNativeImageFailure *failure) {
    return native_image_fragments(r, demands, count, read, 0, out, capacity, fragment_count, failure, 1, NULL, NATIVE_IMAGE_OPEN_READ, NULL);
}

RcRegistryResult rc_registry_native_image_snapshot_mixed(RcRegistry *r, const RcInterval *demands, uint32_t count,
    const RcRegistryBufferSpan *baselines, uint32_t baseline_count, RcRegistryPosition read,
    RcRegistryNativeImageSnapshotFragment *out, uint32_t capacity, uint32_t *fragment_count,
    RcRegistryNativeImageFailure *failure) {
    const NativeImageMixed mixed = {.baselines = baselines, .out = out, .count = baseline_count};
    return native_image_fragments(r, demands, count, read, 0, NULL, capacity, fragment_count, failure, 1, &mixed, NATIVE_IMAGE_OPEN_READ, NULL);
}

RcRegistryResult rc_registry_native_image_snapshot_handed(RcRegistry *r, const RcInterval *demands, uint32_t count,
    const RcRegistryBufferSpan *baselines, uint32_t baseline_count, RcRegistryPosition cut,
    RcRegistryNativeImageSnapshotFragment *out, uint32_t capacity, uint32_t *fragment_count,
    RcRegistryNativeImageFailure *failure) {
    const NativeImageMixed mixed = {.baselines = baselines, .out = out, .count = baseline_count};
    return native_image_fragments(r, demands, count, cut, 0, NULL, capacity, fragment_count, failure, 1, &mixed,
                                  NATIVE_IMAGE_FIXED_HANDOFF, NULL);
}

int rc_registry_native_image_covers(RcRegistry *r, uint64_t image, const RcInterval *demands, uint32_t count,
                                    RcRegistryPosition read) {
    uint32_t fragments;
    return image && native_image_fragments(r, demands, count, read, image, NULL, 0, &fragments, NULL, 0, NULL, NATIVE_IMAGE_OPEN_READ, NULL) == RC_REGISTRY_OK;
}

int rc_registry_native_image_resident_covers(RcRegistry *r, uint64_t image, const RcRegistryWrite *holds,
    uint32_t hold_count, const RcInterval *demands, uint32_t count, RcRegistryPosition read) {
    const NativeImageResident resident = {holds, hold_count};
    uint32_t fragments;
    return r && image && holds && hold_count && hold_count <= r->capacity.writers &&
        native_image_fragments(r, demands, count, read, image, NULL, 0, &fragments, NULL, 0, NULL,
                               NATIVE_IMAGE_OPEN_READ, &resident) == RC_REGISTRY_OK;
}

int rc_registry_native_image_covers_handed(RcRegistry *r, uint64_t image,
    const RcInterval *demands, uint32_t count) {
    uint32_t fragments;
    return image && native_image_fragments(r, demands, count, (RcRegistryPosition){0}, image,
        NULL, 0, &fragments, NULL, 1, NULL, NATIVE_IMAGE_LATEST_HANDOFF, NULL) == RC_REGISTRY_OK;
}

int rc_registry_native_image_whole_written(RcRegistry *r, uint64_t image,
    const RcRegistryImageWriteExtent *holders, uint32_t count, RcRegistryPosition output) {
    if (!r || !holders || !count || count > r->capacity.writers) return 0;
    RcRegistryNativeImageWrite *writes = calloc(count, sizeof *writes);
    if (!writes) return 0;
    for (uint32_t i = 0; i < count; ++i)
        writes[i] = (RcRegistryNativeImageWrite){holders[i], &holders[i].interval.range, 1};
    RcRegistryResult result = rc_registry_native_image_bound_written(r, image, writes, count, output);
    free(writes);
    return result == RC_REGISTRY_OK;
}

typedef struct {
    RcRegistryImageBinding old;
    uint64_t clone;
    RcRegistryPosition output;
} NativeImageRelocation;

static int native_image_relocate_producer(RcRegistry *r, uint32_t i, void *data) {
    const NativeImageRelocation *move = data;
    Writer *w = &r->writers[i];
    uint32_t at = image_position_lower(w, move->output);
    if (at == w->image_count) return 0;
    ImageProducer *p = &w->image_positions[at];
    if (rc_registry_position_compare(p->position, move->output) || !p->active || p->unresolved ||
        p->unknown || !image_binding_equal(p->image, move->old)) return 0;
    p->image.image = move->clone;
    p->projected = 1;
    if (at < w->image_dirty) w->image_dirty = at;
    return 0;
}

static int native_image_relocation_witness(RcRegistry *r, uint32_t i, void *data) {
    const NativeImageRead *source = data;
    Writer *claim = &r->writers[i];
    const Writer *held = writer_of(r, source->binding.hold);
    return !claim->image_incomplete && rc_range_equal(claim->interval.range, held->interval.range) &&
        image_producer_affecting_projection(r, claim, source->output, source->binding) != NULL;
}

static RcRegistryResult native_image_relocation_current(RcRegistry *r, uint64_t image,
    const RcRegistryImageWriteExtent *holders, uint32_t count, RcRegistryPosition output, RcRegistryPosition read) {
    if (r->stats.fail_closed || !read_position_open(r, read)) return RC_REGISTRY_STALE;
    if (rc_registry_position_compare(output, r->handed) > 0) return RC_REGISTRY_BLOCKED;
    for (uint32_t i = 0; i < count; ++i) {
        const RcRegistryImageWriteExtent *owner = &holders[i];
        RcInterval interval = owner->interval;
        Writer *w = writer_of(r, owner->hold);
        if (!native_projected_interval(interval)) return RC_REGISTRY_INVALID;
        if (!native_image_hold_exact(r, w, owner->hold, image, &interval) || !w->native_bound_whole || w->native_projected ||
            rc_registry_position_compare(w->native_output, output)) return RC_REGISTRY_STALE;
        const NativeImageProjection *proof = w->native_projection;
        if (!proof || !proof->physically_bound || !proof->confirmed ||
            rc_registry_position_compare(proof->output, output)) return RC_REGISTRY_STALE;
        for (uint32_t j = 0; j < i; ++j)
            if (rc_range_overlap(holders[j].interval.range, interval.range)) return RC_REGISTRY_INVALID;
        NativeImageRead state = {output, read, {image, owner->hold}};
        if (w->native_conflict || authority_overlaps(r, interval.range, native_image_read_blocker, &state) ||
            (w->native_bound_whole != NATIVE_BOUND_COMPLETED &&
             !authority_overlaps(r, interval.range, native_image_relocation_witness, &state)))
            return RC_REGISTRY_BLOCKED;
    }
    return RC_REGISTRY_OK;
}

RcRegistryResult rc_registry_native_image_relocate(RcRegistry *r, uint64_t image, uint64_t clone,
    const RcRegistryImageWriteExtent *holders, uint32_t count, RcRegistryPosition output, RcRegistryPosition read) {
    if (!r || !image || !clone || image == clone || !holders || !count || count > r->capacity.writers ||
        !rc_registry_position_valid(output) || !(output.ordinal & 1) ||
        !rc_registry_position_valid(read) || (read.ordinal & 1) ||
        rc_registry_position_compare(output, read) >= 0) return RC_REGISTRY_INVALID;

    lock_registry(r);
    RcRegistryResult result = native_image_relocation_current(r, image, holders, count, output, read);
    if (result == RC_REGISTRY_OK) for (uint32_t i = 0; i < count; ++i) {
        const RcRegistryImageWriteExtent *owner = &holders[i];
        Writer *w = writer_of(r, owner->hold);
        NativeImageProjection *proof = w->native_projection;
        // The native clone contains the latest meaningful texels; earlier outputs from this same
        // physical image have been overwritten. The completed original scope owns any older reader.
        while (proof->next) native_projection_remove(r, &proof->next);
        proof->physically_bound = 0;
        w->native_projected = 1;
        w->image = clone;
        native_image_output_clear(r, w);
        NativeImageRelocation move = {{image, owner->hold}, clone, output};
        authority_overlaps(r, owner->interval.range, native_image_relocate_producer, &move);
        // A recorded binding to the original physical image now has to retain it independently.
        // Guest bytes, mapping and producer order remained unchanged.
        invalidate_content(r, owner->interval.range, 0);
    }
    unlock_registry(r);
    return result;
}

typedef struct { RcSourceRange range; RcRegistryPosition read; int inclusive, bound; } NativeImageOverlap;

static int native_image_before_cut(RcRegistryPosition output, const NativeImageOverlap *query) {
    const int order = rc_registry_position_compare(output, query->read);
    return order < 0 || (order == 0 && query->inclusive);
}

static int native_projection_overlaps_read(RcRegistry *r, uint32_t i, void *data) {
    const NativeImageOverlap *query = data;
    const Writer *writer = &r->writers[i];
    const uint64_t end = query->range.offset + query->range.size;
    for (const NativeImageProjection *p = writer->native_projection; p; p = p->next) {
        if (p->physically_bound != query->bound || !native_image_before_cut(p->output, query)) continue;
        uint32_t first = native_projection_first(p, query->range.offset);
        if (first < p->count && p->spans[first].begin < end) return 1;
    }
    return 0;
}

static int native_image_has_prior_projection(RcRegistry *r, const RcInterval *ranges, uint32_t count,
                                             RcRegistryPosition read, int bound) {
    if (!r || !ranges || !count || !rc_registry_position_valid(read)) return 0;
    lock_registry(r);
    int prior = 0;
    for (uint32_t i = 0; i < count && !prior && r->native_projection_spans; ++i) {
        if (!native_projected_interval(ranges[i])) continue;
        NativeImageOverlap query = {.range = ranges[i].range, .read = read, .bound = bound};
        prior = overlaps(r, &r->authority, r->authority.root, query.range, native_projection_overlaps_read, &query);
    }
    unlock_registry(r);
    return prior;
}

int rc_registry_native_image_has_prior_projection(RcRegistry *r, const RcInterval *ranges, uint32_t count,
                                                   RcRegistryPosition read) {
    return native_image_has_prior_projection(r, ranges, count, read, 0);
}

int rc_registry_native_image_has_prior_bound_projection(RcRegistry *r, const RcInterval *ranges, uint32_t count,
                                                         RcRegistryPosition read) {
    return native_image_has_prior_projection(r, ranges, count, read, 1);
}

static int native_buffer_image_authority(RcRegistry *r, uint32_t i, void *opaque) {
    const NativeImageOverlap *query = opaque;
    const Writer *w = &r->writers[i];
    // native_projected is specific to independently placed images. A retired projection no longer
    // has a readable VkImage but its lost bytes still cannot fall through to the old physical buffer.
    if (!w->native_projected) return 0;
    if (w->state == WRITE_FOREIGN)
        return !rc_registry_position_valid(w->visibility_position) ||
            native_image_before_cut(w->visibility_position, query);
    if (w->state != WRITE_HOLD) return 0;
    if (w->native_unordered || (rc_registry_position_valid(w->native_unprojected) &&
        native_image_before_cut(w->native_unprojected, query))) return 1;
    return native_projection_overlaps_read(r, i, opaque);
}

static int native_image_buffer_requires_coverage(RcRegistry *r, const RcInterval *ranges, uint32_t count,
    RcRegistryPosition read, int handed) {
    if (!r || !ranges || !count || !rc_registry_position_valid(read)) return 1;
    lock_registry(r);
    int required = r->stats.fail_closed ||
        (handed ? rc_registry_position_compare(read, r->handed) != 0 : !read_position_open(r, read));
    for (uint32_t i = 0; i < count && !required; i++) {
        if (!native_projected_interval(ranges[i])) { required = 1; break; }
        NativeImageOverlap query = {.range = ranges[i].range, .read = read, .inclusive = handed};
        required = authority_overlaps(r, query.range, native_buffer_image_authority, &query);
    }
    unlock_registry(r);
    return required;
}

int rc_registry_native_image_buffer_requires_coverage(RcRegistry *r, const RcInterval *ranges, uint32_t count,
    RcRegistryPosition read) {
    return native_image_buffer_requires_coverage(r, ranges, count, read, 0);
}

int rc_registry_native_image_handed_requires_coverage(RcRegistry *r, const RcInterval *ranges, uint32_t count,
    RcRegistryPosition cut) {
    return native_image_buffer_requires_coverage(r, ranges, count, cut, 1);
}

typedef struct { RcRegistryImageBinding binding; } NativeImageDiscard;

static int native_discard_unsettled(RcRegistry *r, uint32_t i, void *opaque) {
    const NativeImageDiscard *discard = opaque;
    const Writer *claim = &r->writers[i];
    if (claim->state != WRITE_PENDING || claim->kind != RC_REGISTRY_IMAGE) return 0;
    if (claim->image_incomplete) return 1;
    for (uint32_t j = 0; j < claim->image_count; ++j) {
        const ImageProducer *producer = &claim->image_positions[j];
        if (producer->unresolved ||
            (producer->active && image_binding_equal(producer->image, discard->binding))) return 1;
    }
    return 0;
}

static int native_discard_replaced(RcRegistry *r, const Writer *w, uint64_t begin, uint64_t end,
                                    RcRegistryPosition through) {
    while (begin < end) {
        NativeImageSlice slice = {
            .range = {w->interval.range.backing, begin, end - begin}, .read = r->handed,
            .end = end, .inclusive = 1, .completed_only = 1
        };
        authority_overlaps(r, slice.range, native_projection_select, &slice);
        const NativeImageProjection *winner = slice.winner;
        const Writer *replacement = winner ? writer_of(r, slice.binding.hold) : NULL;
        int replacement_live = replacement && replacement->image != w->image && !replacement->image_forgotten &&
            !slice.ambiguous && (!winner->physically_bound || (!replacement->native_conflict &&
            native_bound_output(replacement, slice.binding, winner->output)));
        if (rc_registry_position_compare(slice.buffer_output, through) <= 0 &&
            (!replacement_live || rc_registry_position_compare(winner->output, through) <= 0)) return 0;
        begin = slice.end;
    }
    return 1;
}

static int native_discard_current(RcRegistry *r, const Writer *w, RcRegistryImageBinding binding) {
    NativeImageDiscard discard = {binding};
    if (authority_overlaps(r, w->interval.range, native_discard_unsettled, &discard)) return 0;
    RcRegistryPosition uncertain = w->native_unprojected_latest;
    if (rc_registry_position_compare(w->native_retire_uncertain, uncertain) > 0) uncertain = w->native_retire_uncertain;
    const ForeignCoverage *possible = w->image_possible;
    if (possible && !w->native_ordered_effect && !w->native_projected) return 0;
    // A known or possible producer stays in physical custody over the entire part that may need it.
    if (possible && rc_registry_position_valid(uncertain)) {
        for (uint32_t i = 0; i < possible->count; ++i) {
            const ForeignSpan span = possible->spans[i];
            uint64_t begin = w->interval.range.offset + (span.begin - w->image_address);
            if (!native_discard_replaced(r, w, begin, begin + span.end - span.begin, uncertain)) return 0;
        }
    }
    for (const NativeImageProjection *p = w->native_projection; p; p = p->next) {
        if (!p->confirmed || !p->completed || p->physically_bound ||
            rc_registry_position_compare(p->output, r->handed) > 0) return 0;
        for (uint32_t i = 0; i < p->count; ++i)
            if (!native_discard_replaced(r, w, p->spans[i].begin, p->spans[i].end, p->output)) return 0;
    }
    return 1;
}

static RcRegistryResult native_image_discard(RcRegistry *r, uint64_t image, uint64_t reclaimed_backing,
    const RcRegistryWrite *holds, uint32_t count) {
    if (!r || !image || !holds || !count || count > r->capacity.writers) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = r->stats.fail_closed && !reclaimed_backing ? RC_REGISTRY_STALE : RC_REGISTRY_OK;
    for (uint32_t i = 0; result == RC_REGISTRY_OK && i < count; ++i) {
        Writer *w = writer_of(r, holds[i]);
        if (!w || w->state != WRITE_HOLD || w->kind != RC_REGISTRY_IMAGE || w->image != image || w->image_forgotten)
            result = RC_REGISTRY_STALE;
        else if (reclaimed_backing) {
            if (w->interval.range.backing != reclaimed_backing || w->native_bound_whole) result = RC_REGISTRY_STALE;
        } else if (!w->image_possible_known || w->native_unordered || rc_registry_position_valid(w->native_output)) {
            result = RC_REGISTRY_UNSUPPORTED;
        }
        if (result != RC_REGISTRY_OK) break;
        for (uint32_t j = 0; j < i; ++j)
            if (holds[j].index == holds[i].index) result = RC_REGISTRY_INVALID;
        if (!reclaimed_backing && result == RC_REGISTRY_OK &&
            !native_discard_current(r, w, (RcRegistryImageBinding){image, holds[i]})) result = RC_REGISTRY_BLOCKED;
    }
    if (result == RC_REGISTRY_OK) for (uint32_t i = 0; i < count; ++i) {
        invalidate(r, r->writers[holds[i].index].interval.range);
        free_writer(r, holds[i].index);
    }
    unlock_registry(r);
    return result;
}

RcRegistryResult rc_registry_native_image_discard_superseded(RcRegistry *r, uint64_t image,
    const RcRegistryWrite *holds, uint32_t count) {
    return native_image_discard(r, image, 0, holds, count);
}

RcRegistryResult rc_registry_native_image_discard_reclaimed(RcRegistry *r, uint64_t image, uint64_t backing,
    const RcRegistryWrite *holds, uint32_t count) {
    if (backing <= RC_GUEST_COMPATIBILITY_BACKING) return RC_REGISTRY_INVALID;
    return native_image_discard(r, image, backing, holds, count);
}

RcRegistryResult rc_registry_image_hold_release_extents(RcRegistry *r, const RcRegistryWrite *holds, uint32_t count,
                                                        int cpu_visible) {
    if (!r || !holds || !count) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = RC_REGISTRY_OK;
    uint64_t image = 0;
    for (uint32_t i = 0; i < count && result == RC_REGISTRY_OK; i++) {
        Writer *writer = writer_of(r, holds[i]);
        if (!writer || writer->state != WRITE_HOLD || writer->kind != RC_REGISTRY_IMAGE ||
            !writer->image || (i && image != writer->image)) {
            result = RC_REGISTRY_STALE;
            break;
        }
        image = writer->image;
        for (uint32_t j = 0; j < i; j++)
            if (holds[j].index == holds[i].index) { result = RC_REGISTRY_INVALID; break; }
    }
    if (result == RC_REGISTRY_OK)
        for (uint32_t i = 0; i < count; i++) {
            if (!cpu_visible) r->writers[holds[i].index].diagnostic_flags |= RC_REGISTRY_DIAG_HOLD_RELEASE;
            finish_writer_locked(r, holds[i].index, cpu_visible, 1);
        }
    unlock_registry(r);
    return result;
}

RcRegistryResult rc_registry_hold_release(RcRegistry *r, RcRegistryWrite ticket, int cpu_visible) {
    if (!r) return RC_REGISTRY_INVALID;
    lock_registry(r); Writer *w = writer_of(r, ticket);
    if (!w || w->state != WRITE_HOLD) { unlock_registry(r); return RC_REGISTRY_STALE; }
    if (!cpu_visible) w->diagnostic_flags |= RC_REGISTRY_DIAG_HOLD_RELEASE;
    finish_writer_locked(r, ticket.index, cpu_visible, 1);
    unlock_registry(r); return RC_REGISTRY_OK;
}
typedef struct {
    uint32_t count, retained;
    const RcRegistryVisibility *proof;
    const Writer *merge;
    RcSourceRange range;
} Gather;

static int foreign_proof_reaches(const Writer *writer, RcSourceRange range, const RcRegistryVisibility *proof) {
    const ForeignCoverage *coverage = writer->foreign_coverage;
    if (!coverage) return 1;
    ForeignSpan overlap = writer_foreign_span(writer, range);
    for (uint32_t i = foreign_span_first(coverage, overlap.begin);
         i < coverage->count && coverage->spans[i].begin < overlap.end; ++i)
        if (coverage->spans[i].producer_order <= proof->horizon) return 1;
    return 0;
}

static int same_completed_buffer_placement(const Writer *latest, const Writer *prior) {
    return latest->kind == RC_REGISTRY_BUFFER && prior->kind == RC_REGISTRY_BUFFER &&
        latest->kinds == (1u << RC_REGISTRY_BUFFER) && prior->kinds == latest->kinds &&
        !latest->unordered && !prior->unordered && !latest->guard && !prior->guard &&
        !latest->native_buffer_conflict && !prior->native_buffer_conflict &&
        latest->native_buffer.allocation && latest->native_buffer_mapping &&
        latest->native_buffer_mapping == prior->native_buffer_mapping &&
        latest->native_buffer_address == prior->native_buffer_address &&
        buffer_native_equal(latest->native_buffer, prior->native_buffer) &&
        rc_range_equal(latest->interval.range, prior->interval.range) &&
        (latest->diagnostic_flags & RC_REGISTRY_DIAG_COMPLETED) &&
        (prior->diagnostic_flags & RC_REGISTRY_DIAG_COMPLETED) &&
        !(latest->diagnostic_flags & RC_REGISTRY_DIAG_ABANDONED) &&
        !(prior->diagnostic_flags & RC_REGISTRY_DIAG_ABANDONED) &&
        rc_registry_position_valid(latest->visibility_latest) &&
        rc_registry_position_valid(prior->visibility_latest) &&
        rc_registry_position_compare(latest->visibility_latest, prior->visibility_latest) >= 0;
}

static int gather_foreign(RcRegistry *r, uint32_t i, void *data) {
    Gather *g = data; Writer *w = &r->writers[i];
    if (w->state != WRITE_FOREIGN) return 0;
    // Different providers or unordered effects cannot spread their uncertainty into a buffer copy's coverage.
    // Keep those overlapping intervals independent, so a valid buffer proof can recover bytes outside an image.
    if (g->merge && (w->foreign_coverage || g->merge->kinds != w->kinds || g->merge->unordered != w->unordered ||
        g->merge->native_projected != w->native_projected)) return 0;
    // Repeated exact image writes to the same bytes can reuse one record. Joining different geometry
    // would spread the latest output across bytes that it did not touch.
    if (g->merge && (w->positioned_image_effect || g->merge->positioned_image_effect) &&
        (!w->positioned_image_effect || !g->merge->positioned_image_effect ||
         !rc_range_equal(w->interval.range, g->merge->interval.range))) return 0;
    // Keep other placements or partial/unknown writes independent. A completed exact rewrite of
    // the same full affine region can keep the latest native source and the earlier visibility bound.
    if (g->merge && (w->native_buffer.allocation || g->merge->native_buffer.allocation) &&
        !same_completed_buffer_placement(g->merge, w)) return 0;
    const RcRegistryVisibility *p = g->proof;
    if (p && ((w->kinds & ~p->writer_kinds) || w->producer_order > p->horizon ||
        !foreign_proof_reaches(w, g->range, p) ||
        (p->mode == RC_REGISTRY_VISIBILITY_COMPLETED ? w->foreign_order > p->horizon :
         w->unordered || !w->submission || w->submission > p->submission))) {
        g->retained++;
        return 0;
    }
    r->scratch[g->count++] = i;
    return 0;
}
static int native_projection_retirement_order(RcRegistry *r, Writer *w) {
    if (w->state != WRITE_HOLD || (!w->native_projected && !w->native_ordered_effect) || w->native_unordered) return 0;
    RcRegistryPosition first = w->native_unprojected, last = w->native_unprojected_latest;
    for (const NativeImageProjection *p = w->native_projection; p; p = p->next) {
        if (!p->confirmed || p->physically_bound || rc_registry_position_compare(p->output, r->handed) > 0) return 0;
        if (!rc_registry_position_valid(first) || rc_registry_position_compare(p->output, first) < 0) first = p->output;
        if (rc_registry_position_compare(p->output, last) > 0) last = p->output;
    }
    if (!rc_registry_position_valid(first) || !rc_registry_position_valid(last) ||
        rc_registry_position_compare(last, r->handed) > 0) return 0;
    // The retained physical image is going away, but its earlier actual GPU writes still bound
    // what can authorize a read. Keep that real order while retaining foreign (no CPU) authority.
    w->visibility_position = first;
    w->visibility_latest = last;
    w->submission = last.scene;
    return 1;
}

static int native_possible_retirement(const Writer *w) {
    return w->image_possible_known && !w->native_unordered && !rc_registry_position_valid(w->native_output);
}

static void finish_writer_locked(RcRegistry *r, uint32_t index, int cpu_visible, int notify) {
    Writer *w = &r->writers[index];
    const int ordered_image = !cpu_visible && (w->native_projection || w->native_ordered_effect) &&
        native_projection_retirement_order(r, w);
    if (ordered_image && native_possible_retirement(w) && !w->foreign_coverage) {
        // An independently placed image cannot write its guest-layout padding. Lose the physical
        // receipt without making those raw bytes foreign or publishing any new CPU-visible data.
        if (!w->image_possible) {
            if (notify) invalidate(r, w->interval.range);
            free_writer(r, index);
            return;
        }
        w->foreign_coverage = w->image_possible;
        w->foreign_address = w->image_address;
        w->image_possible = NULL;
    }
    native_possible_revoke(r, w);
    if (w->native_projection) native_projection_retire(r, w);
    native_image_output_clear(r, w);
    RcSourceRange range = w->interval.range;
    if (notify) invalidate(r, range);
    if (cpu_visible) {
        gpu_visibility_note_payload(r, range, w->visibility_position, w->visibility_latest, w->native_buffer,
                                    w->native_payload);
        free_writer(r, index);
        return;
    }
    buffer_payload_drop(&w->native_payload);
    if (w->guard) {
        // All still-reachable backings remain excluded until the adapter can supply exhaustive coverage.
        w->unordered = 1;
        w->visibility_position = w->visibility_latest = (RcRegistryPosition){0};
        writer_set_state(r, w, WRITE_FOREIGN);
    }
    else {
        if (w->state == WRITE_SUBMITTED) w->diagnostic_flags |= RC_REGISTRY_DIAG_COMPLETED;
        diagnose(r, (RcRegistryDiagnosticReason)(RC_REGISTRY_DIAG_FOREIGN_IMAGE + w->kind), index,
            range, (RcRegistryPosition){0}, (RcRegistryImageBinding){0}, NULL, 0);
        image_positions_clear(r, w); w->image = 0;
        w->foreign_order = identity(r);
        if (!w->foreign_order) w->foreign_order = UINT64_MAX;
        w->unordered = !w->submission;
        if (w->unordered) w->visibility_position = w->visibility_latest = (RcRegistryPosition){0};
        if (w->foreign_coverage) {
            writer_set_state(r, w, WRITE_FOREIGN);
            return;
        }
        // Remove before changing bounds. Adjacent foreign spans have identical unknown CPU authority, so their
        // union is exact too. Pending and held producers retain independent identities and cancellation rights.
        r->authority.root = erase_node(&r->authority, r->authority.root, index);
        for (;;) {
            Gather g = {.merge = w};
            RcSourceRange touching = range;
            if (touching.offset) touching.offset--, touching.size++;
            if (range.offset + range.size < UINT64_MAX) touching.size++;
            overlaps(r, &r->authority, r->authority.root, touching, gather_foreign, &g);
            if (!g.count) break;
            for (uint32_t j = 0; j < g.count; j++) {
                uint32_t i = r->scratch[j]; Writer *other = &r->writers[i];
                RcSourceRange prior = other->interval.range;
                uint64_t end = range.offset + range.size, prior_end = prior.offset + prior.size;
                if (prior.offset < range.offset) range.offset = prior.offset;
                if (prior_end > end) end = prior_end;
                range.size = end - range.offset;
                if (other->producer_order > w->producer_order) w->producer_order = other->producer_order;
                if (other->foreign_order > w->foreign_order) w->foreign_order = other->foreign_order;
                if (other->submission > w->submission) w->submission = other->submission;
                w->unordered |= other->unordered;
                if (w->native_buffer.allocation) w->native_buffer_compacted = 1;
                w->visibility_position = gpu_visibility_earliest(w->visibility_position, other->visibility_position);
                w->visibility_latest = gpu_visibility_latest(w->visibility_latest, other->visibility_latest);
                w->kinds |= other->kinds;
                w->diagnostic_flags |= other->diagnostic_flags;
                if (w->diagnostic_origin != other->diagnostic_origin) w->diagnostic_origin = 0;
                if (w->diagnostic_position.scene != other->diagnostic_position.scene)
                    w->diagnostic_position = (RcRegistryPosition){0};
                else if (w->diagnostic_position.ordinal != other->diagnostic_position.ordinal)
                    w->diagnostic_position.ordinal = 0;
                if (w->diagnostic_guard_reason != other->diagnostic_guard_reason)
                    w->diagnostic_guard_reason = RC_REGISTRY_GUARD_UNCLASSIFIED;
                if (w->kind != other->kind) {
                    --*writer_kind_count(r, w->kind);
                    w->kind = RC_REGISTRY_UNKNOWN;
                    ++*writer_kind_count(r, w->kind);
                }
                free_writer(r, i);
            }
        }
        writer_set_state(r, w, WRITE_FOREIGN);
        w->interval.range = range; w->interval.left = w->interval.right = 0;
        r->authority.root = insert_node(&r->authority, r->authority.root, index);
    }
}
RcRegistryResult rc_registry_write_complete(RcRegistry *r, RcRegistryWrite ticket, int cpu_visible) {
    if (!r) return RC_REGISTRY_INVALID;
    lock_registry(r); Writer *w = writer_of(r, ticket);
    if (!w || w->state != WRITE_SUBMITTED) { unlock_registry(r); return RC_REGISTRY_STALE; }
    if (w->guard && cpu_visible) { unlock_registry(r); return RC_REGISTRY_INVALID; }
    finish_writer_locked(r, ticket.index, cpu_visible, 1);
    unlock_registry(r); return RC_REGISTRY_OK;
}
static int range_contains(RcSourceRange outer, RcSourceRange inner) {
    return rc_range_valid(inner) && inner.backing == outer.backing && inner.offset >= outer.offset &&
        inner.size <= outer.size && inner.offset - outer.offset <= outer.size - inner.size;
}
static Writer completion_provenance(const Writer *w) {
    Writer original = *w;
    // Completion partitions retain authority provenance, never live declaration schedules or image ownership.
    original.image_positions = NULL; original.image_count = original.image_capacity = original.image_dirty = 0;
    original.native_projection = NULL;
    original.image_incomplete = 0; original.image = original.image_address = original.image_mapping = 0;
    original.guard = 0; original.guard_next = original.guard_prev = 0;
    original.foreign_coverage = NULL;
    original.image_possible = NULL;
    original.image_possible_known = 0;
    original.foreign_address = 0;
    return original;
}
static int completion_valid(const Writer *w, const RcRegistryWriteCompletion *completion, int precise) {
    if ((completion->written_count && !completion->written) || (completion->uncertain_count && !completion->uncertain))
        return 0;
    RcSourceRange whole = w->interval.range;
    for (uint32_t j = 0; j < completion->written_count; j++) {
        RcSourceRange part = completion->written[j];
        if (!(w->guard ? rc_range_valid(part) && part.backing != RC_GUEST_COMPATIBILITY_BACKING : range_contains(whole, part)))
            return 0;
    }
    for (uint32_t j = 0; j < completion->uncertain_count; j++) {
        RcSourceRange part = completion->uncertain[j];
        if (!(w->guard ? rc_range_valid(part) && part.backing != RC_GUEST_COMPATIBILITY_BACKING : range_contains(whole, part)))
            return 0;
        if (precise) {
            uint32_t k = 0;
            while (k < completion->written_count && !range_contains(completion->written[k], part)) k++;
            if (k == completion->written_count) return 0;
        }
    }
    return 1;
}
static void complete_uncertain(RcRegistry *r, const Writer *original, RcSourceRange range) {
    uint32_t i = r->free_writer;
    uint64_t id = identity(r);
    r->free_writer = r->writers[i].free_next;
    r->writers[i] = *original;
    r->writers[i].id = id;
    r->writers[i].interval = (Interval){.range = range, .order = id, .priority = mixed(id)};
    if (original->native_buffer.allocation) {
        const uint64_t delta = range.offset - original->interval.range.offset;
        r->writers[i].native_buffer.offset += delta;
        r->writers[i].native_buffer_address += delta;
    }
    r->writers[i].free_next = 0;
    writer_count_add(r);
    ++*writer_state_count(r, original->state);
    ++*writer_kind_count(r, original->kind);
    r->authority.root = insert_node(&r->authority, r->authority.root, i);
    // Authorization and content were invalidated before replacing the broad reservation.
    finish_writer_locked(r, i, 0, 0);
}

static int image_effect_position_exact(const Writer *w, RcRegistryPosition output) {
    uint32_t at = image_position_lower(w, output);
    return at < w->image_count && w->image_positions[at].exact_effect &&
        !rc_registry_position_compare(w->image_positions[at].position, output);
}

RcRegistryResult rc_registry_image_effects_settle(RcRegistry *r, RcRegistryWrite ticket,
    const RcRegistryImageEffectSpan *spans, uint32_t count) {
    if (!r || !spans || !count) return RC_REGISTRY_INVALID;
    lock_registry(r);
    Writer *w = writer_of(r, ticket);
    RcRegistryResult result = RC_REGISTRY_STALE;
    if (!w || w->state != WRITE_PENDING || w->kind != RC_REGISTRY_IMAGE || w->image_incomplete ||
        !writer_handed(r, w, w->position.scene)) goto done;
    for (uint32_t i = 0; i < count; ++i) {
        const RcRegistryImageEffectSpan *span = &spans[i];
        const int completed = rc_registry_position_valid(span->completed_latest);
        if (!range_contains(w->interval.range, span->range) ||
            !image_effect_position_exact(w, span->first) || !image_effect_position_exact(w, span->latest) ||
            rc_registry_position_compare(span->first, span->latest) > 0 ||
            completed != rc_registry_position_valid(span->completed_first) ||
            (completed && (!image_effect_position_exact(w, span->completed_first) ||
                           !image_effect_position_exact(w, span->completed_latest) ||
                           rc_registry_position_compare(span->first, span->completed_first) > 0 ||
                           rc_registry_position_compare(span->completed_first, span->completed_latest) > 0 ||
                           rc_registry_position_compare(span->completed_latest, span->latest) > 0))) {
            result = RC_REGISTRY_INVALID;
            goto done;
        }
    }
    if (count > (uint64_t)r->capacity.writers - r->stats.writers + 1) {
        r->stats.completion_range_refusals++;
        writer_census(r, "image-settlement-capacity", count, 0);
        result = RC_REGISTRY_CAPACITY;
        goto done;
    }
    if (count > (UINT64_MAX - r->clock) / 2) {
        poison(r);
        result = RC_REGISTRY_EXHAUSTED;
        goto done;
    }

    Writer provenance = completion_provenance(w);
    provenance.positioned_image_effect = 1;
    const uint32_t diagnostic_flags = provenance.diagnostic_flags;
    invalidate_topology(r, w->interval.range);
    for (uint32_t i = 0; i < count; ++i) {
        const RcRegistryImageEffectSpan *span = &spans[i];
        if (rc_registry_position_compare(span->completed_latest, span->latest))
            native_projection_invalidate(r, span->range, span->latest, NULL, 1);
        invalidate(r, span->range);
        if (rc_registry_position_valid(span->completed_latest))
            gpu_visibility_note(r, span->range, span->completed_first, span->completed_latest);
    }
    // The exact handed effects above replaced the broad fallback for these outputs only.
    for (uint32_t i = 0; i < w->image_count; ++i) {
        ImageProducer *p = &w->image_positions[i];
        if (!p->exact_effect) continue;
        image_position_cancel_projection(r, w, p);
        p->active = 0;
    }
    free_writer(r, ticket.index);
    for (uint32_t i = 0; i < count; ++i) {
        const RcRegistryImageEffectSpan *span = &spans[i];
        provenance.position = provenance.visibility_position = span->first;
        provenance.visibility_latest = span->latest;
        provenance.submission = span->latest.scene;
        provenance.diagnostic_position = span->latest;
        const int completed = !rc_registry_position_compare(span->completed_latest, span->latest);
        provenance.state = completed ? WRITE_SUBMITTED : WRITE_PENDING;
        provenance.diagnostic_flags = diagnostic_flags | (completed ? 0 : RC_REGISTRY_DIAG_ABANDONED);
        complete_uncertain(r, &provenance, span->range);
    }
    result = RC_REGISTRY_OK;
done:
    unlock_registry(r);
    return result;
}

static uint64_t readback_foreign_spans(uint64_t begin, uint64_t end, const RcSourceRange *visible,
                                      uint32_t count, ForeignSpan *out) {
    uint64_t cursor = begin, spans = 0;
    for (uint32_t i = 0; i < count && cursor < end; ++i) {
        uint64_t offset = visible[i].offset;
        if (offset >= end) break;
        if (offset > cursor) {
            if (out) out[spans] = (ForeignSpan){.begin = cursor, .end = offset};
            spans++;
        }
        uint64_t covered = offset + visible[i].size;
        if (covered > cursor) cursor = covered;
    }
    if (cursor < end) {
        if (out) out[spans] = (ForeignSpan){.begin = cursor, .end = end};
        spans++;
    }
    return spans;
}

static void readback_visibility(RcRegistry *r, RcInterval interval, const RcSourceRange *visible, uint32_t count,
                                RcRegistryPosition producer, RcRegistryPosition latest) {
    if (!r->active_gpu_captures && !r->produced_scene) return;
    uint64_t end = interval.address + interval.range.size;
    for (uint32_t i = 0; i < count && visible[i].offset < end; ++i) {
        uint64_t begin = visible[i].offset > interval.address ? visible[i].offset : interval.address;
        uint64_t finish = visible[i].offset + visible[i].size;
        if (finish > end) finish = end;
        if (finish > begin) gpu_visibility_note(r, (RcSourceRange){interval.range.backing,
            interval.range.offset + (begin - interval.address), finish - begin}, producer, latest);
    }
}

// Intersect the actual readback complement with the independently placed image's possible bytes.
static uint32_t readback_possible_gaps(const ForeignCoverage *possible, const ForeignCoverage *gaps,
                                      ForeignSpan *out) {
    uint32_t count = 0;
    if (!possible || !gaps) return count;
    uint32_t i = 0, j = foreign_span_first(gaps, possible->spans[0].begin);
    while (i < possible->count && j < gaps->count) {
        ForeignSpan a = possible->spans[i], b = gaps->spans[j];
        uint64_t lo = a.begin > b.begin ? a.begin : b.begin;
        uint64_t hi = a.end < b.end ? a.end : b.end;
        if (lo < hi) {
            if (out) out[count] = (ForeignSpan){.begin = lo, .end = hi};
            count++;
        }
        if (a.end <= b.end) i++;
        if (b.end <= a.end) j++;
    }
    return count;
}

RcRegistryResult rc_registry_image_hold_readback(RcRegistry *r, uint64_t image, const RcRegistryImageWriteExtent *targets,
                                                 uint32_t count, const RcSourceRange *visible, uint32_t visible_count) {
    if (!r || !image || !targets || !count || count > r->capacity.writers || !visible || !visible_count)
        return RC_REGISTRY_INVALID;
    uint64_t prior_end = 0;
    for (uint32_t i = 0; i < visible_count; ++i) {
        const RcSourceRange *range = &visible[i];
        if (!rc_range_valid(*range) || range->backing != RC_GUEST_COMPATIBILITY_BACKING || range->offset < prior_end)
            return RC_REGISTRY_INVALID;
        prior_end = range->offset + range->size;
    }
    lock_registry(r);
    RcRegistryResult result = RC_REGISTRY_OK;
    uint64_t begin = UINT64_MAX, end = 0;
    for (uint32_t i = 0; i < count && result == RC_REGISTRY_OK; ++i) {
        const RcRegistryImageWriteExtent *target = &targets[i];
        Writer *w = writer_of(r, target->hold);
        if (!w || w->state != WRITE_HOLD || w->kind != RC_REGISTRY_IMAGE || w->image != image || w->image_forgotten ||
            w->image_address != target->interval.address || w->image_mapping != target->interval.mapping ||
            !rc_range_equal(w->interval.range, target->interval.range) ||
            target->interval.range.size > UINT64_MAX - target->interval.address) {
            result = RC_REGISTRY_STALE;
            break;
        }
        for (uint32_t j = 0; j < i; ++j)
            if (targets[j].hold.index == target->hold.index) { result = RC_REGISTRY_INVALID; break; }
        if (result != RC_REGISTRY_OK) break;
        if (target->interval.address < begin) begin = target->interval.address;
        uint64_t finish = target->interval.address + target->interval.range.size;
        if (finish > end) end = finish;
    }
    ForeignCoverage *coverage = NULL;
    if (result == RC_REGISTRY_OK) {
        uint64_t gaps = readback_foreign_spans(begin, end, visible, visible_count, NULL);
        if (gaps > UINT32_MAX) result = RC_REGISTRY_CAPACITY;
        else if (gaps && count > UINT64_MAX - r->clock) result = RC_REGISTRY_EXHAUSTED;
        else if (gaps && !(coverage = foreign_coverage_alloc(r, (uint32_t)gaps))) result = RC_REGISTRY_CAPACITY;
        if (coverage) readback_foreign_spans(begin, end, visible, visible_count, coverage->spans);
    }
    if (result == RC_REGISTRY_OK) {
        for (uint32_t i = 0; i < count; ++i) {
            const RcRegistryImageWriteExtent *target = &targets[i];
            uint32_t index = target->hold.index;
            Writer *w = &r->writers[index];
            // The adapter reports actual changed bytes separately while the original mapping is held.
            invalidate_topology(r, w->interval.range);
            readback_visibility(r, target->interval, visible, visible_count, w->visibility_position, w->visibility_latest);
            uint32_t first = coverage ? foreign_span_first(coverage, target->interval.address) : 0;
            if (coverage && first < coverage->count &&
                coverage->spans[first].begin < target->interval.address + target->interval.range.size) {
                ForeignCoverage *remaining = coverage;
                if (native_possible_retirement(w) && native_projection_retirement_order(r, w)) {
                    uint32_t wanted = readback_possible_gaps(w->image_possible, coverage, NULL);
                    if (!wanted) { free_writer(r, index); continue; }
                    ForeignCoverage *clipped = foreign_coverage_alloc(r, wanted);
                    if (clipped) {
                        readback_possible_gaps(w->image_possible, coverage, clipped->spans);
                        remaining = clipped;
                    }
                }
                if (remaining == coverage) coverage->refs++;
                w->foreign_coverage = remaining;
                w->foreign_address = target->interval.address;
                finish_writer_locked(r, index, 0, 0);
            } else free_writer(r, index);
        }
    }
    foreign_coverage_release(r, coverage);
    unlock_registry(r);
    return result;
}

static void complete_buffer_visibility(RcRegistry *r, RcSourceRange range, const Writer *provenance,
    const RcSourceRange *uncertain, uint32_t uncertain_count) {
    RcRegistryBufferLocation native = provenance->native_buffer;
    if (native.allocation) native.offset += range.offset - provenance->interval.range.offset;
    BufferPayload payload = provenance->native_payload;
    if (!payload.owner || !native.allocation) {
        gpu_visibility_note_native(r, range, provenance->visibility_position, provenance->visibility_latest, native);
        return;
    }
    payload.address += range.offset - provenance->interval.range.offset;
    const uint64_t end = range.offset + range.size;
    for (uint64_t at = range.offset; at < end;) {
        uint64_t boundary = end;
        int maybe = 0;
        for (uint32_t i = 0; i < uncertain_count; ++i) {
            const RcSourceRange part = uncertain[i];
            const uint64_t part_end = part.offset + part.size;
            if (part.backing != range.backing || part_end <= at) continue;
            if (part.offset <= at) {
                maybe = 1;
                if (part_end < boundary) boundary = part_end;
            } else if (part.offset < boundary) boundary = part.offset;
        }
        const uint64_t delta = at - range.offset;
        RcRegistryBufferLocation source = native;
        source.offset += delta;
        const BufferPayload clipped = maybe ? (BufferPayload){0} :
            (BufferPayload){payload.owner, payload.address + delta, payload.mapping};
        gpu_visibility_note_payload(r, (RcSourceRange){range.backing, at, boundary - at},
            provenance->visibility_position, provenance->visibility_latest, source, clipped);
        at = boundary;
    }
}

static void complete_visibility(RcRegistry *r, RcSourceRange range, const Writer *provenance,
    const RcRegistryWriteCompletion *completion, const RcRegistryBufferSpan *definite, uint32_t count) {
    if (!count) {
        complete_buffer_visibility(r, range, provenance, completion->uncertain, completion->uncertain_count);
        return;
    }
    const uint64_t end = range.offset + range.size;
    for (uint64_t at = range.offset; at < end;) {
        uint64_t boundary = end;
        const RcRegistryBufferSpan *next = NULL;
        for (uint32_t i = 0; i < count; ++i) {
            RcSourceRange part = definite[i].interval.range;
            if (part.backing != range.backing || part.offset < at || part.offset >= boundary) continue;
            boundary = part.offset;
            next = &definite[i];
        }
        if (boundary > at) {
            gpu_visibility_note_native(r, (RcSourceRange){range.backing, at, boundary - at},
                provenance->visibility_position, provenance->visibility_latest, (RcRegistryBufferLocation){0});
        }
        if (next) {
            RcSourceRange part = next->interval.range;
            gpu_visibility_note_native(r, part, provenance->visibility_position, provenance->visibility_latest,
                                       next->native);
            boundary += part.size;
        }
        at = boundary;
    }
}

static RcRegistryResult complete_extents_locked(RcRegistry *r, const RcRegistryWriteCompletion *completions,
    Writer *provenance, uint32_t count, int precise, int allow_guard,
    const RcRegistryBufferSpan *definite, uint32_t definite_count) {
    uint64_t needed = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const RcRegistryWriteCompletion *completion = &completions[i];
        Writer *w = writer_of(r, completion->ticket);
        if (!w || (w->state != WRITE_SUBMITTED && !(w->guard && w->state == WRITE_FOREIGN))) return RC_REGISTRY_STALE;
        if ((w->guard && !allow_guard) || !completion_valid(w, completion, precise)) return RC_REGISTRY_INVALID;
        for (uint32_t j = 0; j < i; ++j)
            if (completions[j].ticket.index == completion->ticket.index) return RC_REGISTRY_INVALID;
        needed += completion->uncertain_count;
        provenance[i] = completion_provenance(w);
    }
    if (needed > (uint64_t)r->capacity.writers - r->stats.writers + count) {
        r->stats.completion_range_refusals++;
        writer_census(r, "completion-capacity", needed > UINT32_MAX ? UINT32_MAX : (uint32_t)needed, 0);
        return RC_REGISTRY_CAPACITY;
    }
    if (needed > (UINT64_MAX - r->clock) / 2) { poison(r); return RC_REGISTRY_EXHAUSTED; }
    for (uint32_t i = 0; i < count; ++i) {
        const RcRegistryWriteCompletion *completion = &completions[i];
        RcSourceRange whole = provenance[i].interval.range;
        if (precise) {
            invalidate_topology(r, whole);
            for (uint32_t j = 0; j < completion->written_count; ++j) {
                invalidate(r, completion->written[j]);
                complete_visibility(r, completion->written[j], &provenance[i], completion, definite, definite_count);
            }
        } else {
            invalidate(r, whole);
            complete_buffer_visibility(r, whole, &provenance[i], completion->uncertain, completion->uncertain_count);
        }
        // The local provenance borrowed the writer's reference only through the definite visibility events above.
        // Recreated uncertain intervals keep their prior exclusion, never that borrowed payload.
        provenance[i].native_payload = (BufferPayload){0};
        free_writer(r, completion->ticket.index);
    }
    for (uint32_t i = 0; i < count; ++i)
        for (uint32_t j = 0; j < completions[i].uncertain_count; ++j)
            complete_uncertain(r, &provenance[i], completions[i].uncertain[j]);
    return RC_REGISTRY_OK;
}
static RcRegistryResult complete_effects(RcRegistry *r, RcRegistryWrite ticket,
    const RcSourceRange *written, uint32_t written_count, const RcSourceRange *ranges, uint32_t count, int precise) {
    if (!r || (count && !ranges) || (written_count && !written)) return RC_REGISTRY_INVALID;
    RcRegistryWriteCompletion completion = {ticket, written, ranges, written_count, count};
    Writer provenance;
    lock_registry(r);
    RcRegistryResult result = complete_extents_locked(r, &completion, &provenance, 1, precise, 0, NULL, 0);
    unlock_registry(r);
    return result;
}
RcRegistryResult rc_registry_write_complete_extents(RcRegistry *r, const RcRegistryWriteCompletion *completions,
                                                    uint32_t count) {
    if (!r || !completions || !count || count > r->capacity.writers) return RC_REGISTRY_INVALID;
    Writer one;
    Writer *provenance = count == 1 ? &one : malloc((size_t)count * sizeof *provenance);
    if (!provenance) return RC_REGISTRY_CAPACITY;
    lock_registry(r);
    RcRegistryResult result = complete_extents_locked(r, completions, provenance, count, 1, 1, NULL, 0);
    unlock_registry(r);
    if (provenance != &one) free(provenance);
    return result;
}
RcRegistryResult rc_registry_write_complete_gpa_projection(RcRegistry *r, const RcRegistryWriteCompletion *completion,
    const RcRegistryBufferSpan *definite, uint32_t count) {
    if (!r || !completion || !definite || !count || count > 128 || !completion->written ||
        !completion->written_count || completion->written_count > 128 ||
        completion->uncertain_count) return RC_REGISTRY_INVALID;
    for (uint32_t i = 0; i < completion->written_count; ++i) {
        if (!rc_range_valid(completion->written[i])) return RC_REGISTRY_INVALID;
        for (uint32_t j = 0; j < i; ++j)
            if (rc_range_overlap(completion->written[j], completion->written[i])) return RC_REGISTRY_INVALID;
    }
    for (uint32_t i = 0; i < count; ++i) {
        RcRegistryBufferSpan part = definite[i];
        if (!buffer_native_span_valid(part)) return RC_REGISTRY_INVALID;
        int contained = 0;
        for (uint32_t j = 0; j < completion->written_count; ++j)
            contained |= range_contains(completion->written[j], part.interval.range);
        if (!contained) return RC_REGISTRY_INVALID;
        for (uint32_t j = 0; j < i; ++j)
            if (rc_range_overlap(definite[j].interval.range, part.interval.range)) return RC_REGISTRY_INVALID;
    }
    lock_registry(r);
    Writer *w = writer_of(r, completion->ticket);
    RcRegistryResult result = RC_REGISTRY_STALE;
    if (w && w->state == WRITE_SUBMITTED && w->guard && w->kind == RC_REGISTRY_BUFFER &&
        w->diagnostic_guard_reason == RC_REGISTRY_GUARD_GPA_PROJECTION &&
        rc_registry_position_valid(w->position) && rc_registry_position_compare(w->position, r->handed) <= 0) {
        Writer provenance;
        result = complete_extents_locked(r, completion, &provenance, 1, 1, 1, definite, count);
    }
    unlock_registry(r);
    return result;
}
RcRegistryResult rc_registry_write_complete_ranges(RcRegistry *r, RcRegistryWrite ticket,
                                                  const RcSourceRange *ranges, uint32_t count) {
    return complete_effects(r, ticket, NULL, 0, ranges, count, 0);
}
RcRegistryResult rc_registry_write_complete_effects(RcRegistry *r, RcRegistryWrite ticket,
    const RcSourceRange *written, uint32_t written_count, const RcSourceRange *uncertain, uint32_t uncertain_count) {
    return complete_effects(r, ticket, written, written_count, uncertain, uncertain_count, 1);
}
RcRegistryResult rc_registry_write_abandon(RcRegistry *r, RcRegistryWrite ticket, int may_have_executed) {
    if (!r) return RC_REGISTRY_INVALID;
    lock_registry(r); Writer *w = writer_of(r, ticket);
    if (!w || (w->state != WRITE_PENDING && (!may_have_executed || w->state != WRITE_SUBMITTED))) {
        unlock_registry(r); return RC_REGISTRY_STALE;
    }
    // The caller already drained possible execution. Keep the exclusion continuously present when its exact
    // effects are unknown; no spare writer slot, invented submission value or temporarily uncovered range.
    cancel_pending_writer(r, ticket.index, may_have_executed);
    unlock_registry(r); return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_write_abandon_extents(RcRegistry *r, const RcRegistryWrite *tickets, uint32_t count) {
    if (!r || !tickets || !count) return RC_REGISTRY_INVALID;
    lock_registry(r);
    RcRegistryResult result = RC_REGISTRY_OK;
    for (uint32_t i = 0; i < count && result == RC_REGISTRY_OK; ++i) {
        Writer *w = writer_of(r, tickets[i]);
        if (!w || (w->state != WRITE_PENDING && w->state != WRITE_SUBMITTED)) result = RC_REGISTRY_STALE;
        for (uint32_t j = 0; j < i; ++j)
            if (tickets[j].index == tickets[i].index) result = RC_REGISTRY_INVALID;
    }
    if (result == RC_REGISTRY_OK) for (uint32_t i = 0; i < count; ++i) {
        cancel_pending_writer(r, tickets[i].index, 1);
    }
    unlock_registry(r);
    return result;
}
RcRegistryResult rc_registry_write_guard_accumulate(RcRegistry *r, RcRegistryWrite ticket) {
    if (!r) return RC_REGISTRY_INVALID;
    lock_registry(r);
    Writer *w=writer_of(r,ticket);
    RcRegistryResult result=RC_REGISTRY_STALE;
    if (w && w->guard) {
        if (!r->unresolved_guard) {
            w->submission=0;
            w->diagnostic_flags|=RC_REGISTRY_DIAG_ABANDONED;
            finish_writer_locked(r,ticket.index,0,1);
            r->unresolved_guard=ticket.index;
        } else if (r->unresolved_guard!=ticket.index) {
            Writer *old=&r->writers[r->unresolved_guard];
            uint64_t incarnation=identity(r);
            if (!incarnation) {unlock_registry(r);return RC_REGISTRY_EXHAUSTED;}
            invalidate(r,(RcSourceRange){0});
            old->id=incarnation;
            old->producer_order=incarnation;
            old->visibility_position = old->visibility_latest = (RcRegistryPosition){0};
            old->kinds|=w->kinds;
            if (old->diagnostic_origin != w->diagnostic_origin) old->diagnostic_origin = 0;
            if (old->diagnostic_position.scene != w->diagnostic_position.scene)
                old->diagnostic_position = (RcRegistryPosition){0};
            else if (old->diagnostic_position.ordinal != w->diagnostic_position.ordinal)
                old->diagnostic_position.ordinal = 0;
            if (old->diagnostic_guard_reason != w->diagnostic_guard_reason)
                old->diagnostic_guard_reason = RC_REGISTRY_GUARD_UNCLASSIFIED;
            if (old->kind!=w->kind) {
                --*writer_kind_count(r,old->kind);
                old->kind=RC_REGISTRY_UNKNOWN;
                ++*writer_kind_count(r,old->kind);
            }
            free_writer(r,ticket.index);
        }
        result=RC_REGISTRY_OK;
    }
    unlock_registry(r);
    return result;
}
RcRegistryWrite rc_registry_unresolved_guard(RcRegistry *r) {
    if (!r) return (RcRegistryWrite){0};
    lock_registry(r);
    uint32_t index=r->unresolved_guard;
    RcRegistryWrite result={r->writers[index].id,index};
    unlock_registry(r);
    return result;
}

static uint32_t foreign_subtract(const Writer *writer, RcSourceRange range, const RcRegistryVisibility *proof,
                                 ForeignSpan *out) {
    const ForeignCoverage *coverage = writer->foreign_coverage;
    ForeignSpan clear = writer_foreign_span(writer, range);
    uint64_t lower = writer->foreign_address, upper = lower + writer->interval.range.size;
    uint32_t count = 0;
    for (uint32_t i = foreign_span_first(coverage, lower);
         i < coverage->count && coverage->spans[i].begin < upper; ++i) {
        ForeignSpan span = coverage->spans[i];
        if (span.begin < lower) span.begin = lower;
        if (span.end > upper) span.end = upper;
        if (proof && span.producer_order > proof->horizon) {
            if (out) out[count] = span;
            count++;
            continue;
        }
        if (span.begin < clear.begin) {
            uint64_t end = span.end < clear.begin ? span.end : clear.begin;
            if (out) out[count] = (ForeignSpan){span.begin, end, span.producer_order};
            count++;
        }
        if (span.end > clear.end) {
            uint64_t begin = span.begin > clear.end ? span.begin : clear.end;
            if (out) out[count] = (ForeignSpan){begin, span.end, span.producer_order};
            count++;
        }
    }
    return count;
}

static void reconcile_visibility(RcRegistry *r, const Writer *writer, RcSourceRange range,
                                 const RcRegistryVisibility *proof) {
    RcSourceRange held = writer->interval.range;
    const ForeignCoverage *coverage = writer->foreign_coverage;
    if (!coverage) {
        uint64_t begin = range.offset > held.offset ? range.offset : held.offset;
        uint64_t end = range.offset + range.size, held_end = held.offset + held.size;
        if (end > held_end) end = held_end;
        gpu_visibility_note(r, (RcSourceRange){range.backing, begin, end - begin}, writer->visibility_position, writer->visibility_latest);
        return;
    }
    ForeignSpan clear = writer_foreign_span(writer, range);
    for (uint32_t i = foreign_span_first(coverage, clear.begin);
         i < coverage->count && coverage->spans[i].begin < clear.end; ++i) {
        ForeignSpan span = coverage->spans[i];
        if (proof && span.producer_order > proof->horizon) continue;
        uint64_t begin = span.begin > clear.begin ? span.begin : clear.begin;
        uint64_t end = span.end < clear.end ? span.end : clear.end;
        gpu_visibility_note(r, (RcSourceRange){range.backing, held.offset + (begin - writer->foreign_address),
                                               end - begin}, writer->visibility_position, writer->visibility_latest);
    }
}

static RcRegistryResult reconcile_locked(RcRegistry *r, RcSourceRange range, const RcRegistryVisibility *proof) {
    Gather g = {.proof = proof, .range = range};
    overlaps(r, &r->authority, r->authority.root, range, gather_foreign, &g);
    if (proof) r->stats.visibility_retained += g.retained;
    uint32_t splits = 0, removed = 0;
    uint64_t end = range.offset + range.size;
    ForeignCoverage *local[4] = {0}, **updated = NULL;
    RcRegistryResult result = RC_REGISTRY_OK;
    for (uint32_t j = 0; j < g.count; j++) {
        const Writer *w = &r->writers[r->scratch[j]];
        RcSourceRange old = w->interval.range;
        uint64_t old_end = old.offset + old.size;
        if (w->foreign_coverage) {
            if (!updated) {
                updated = g.count <= sizeof local / sizeof *local ? local : calloc(g.count, sizeof *updated);
                if (!updated) { result = RC_REGISTRY_CAPACITY; goto done; }
            }
            uint32_t count = foreign_subtract(w, range, proof, NULL);
            if (!count) removed++;
            else {
                updated[j] = foreign_coverage_alloc(r, count);
                if (!updated[j]) { result = RC_REGISTRY_CAPACITY; goto done; }
                foreign_subtract(w, range, proof, updated[j]->spans);
            }
        } else if (old.offset < range.offset && old_end > end) splits++;
        else if (range.offset <= old.offset && end >= old_end) removed++;
    }
    if (splits > r->capacity.writers - r->stats.writers + removed) {
        result = RC_REGISTRY_CAPACITY;
        goto done;
    }
    if (UINT64_MAX - r->clock < splits) {
        poison(r);
        result = RC_REGISTRY_EXHAUSTED;
        goto done;
    }
    if (r->active_gpu_captures || r->produced_scene) for (uint32_t j = 0; j < g.count; j++)
        reconcile_visibility(r, &r->writers[r->scratch[j]], range, proof);
    // Release completely covered entries first, and erase their scratch indices before split allocation reuses them.
    for (uint32_t j = 0; j < g.count; j++) {
        uint32_t i = r->scratch[j];
        const Writer *w = &r->writers[i];
        RcSourceRange old = w->interval.range;
        if (w->foreign_coverage ? !updated[j] : range.offset <= old.offset && end >= old.offset + old.size) {
            free_writer(r, i); r->scratch[j] = 0;
        }
    }
    for (uint32_t j = 0; j < g.count; j++) {
        uint32_t i = r->scratch[j];
        if (!i) continue;
        Writer *w = &r->writers[i];
        if (w->foreign_coverage) {
            foreign_coverage_release(r, w->foreign_coverage);
            w->foreign_coverage = updated[j];
            updated[j] = NULL;
            continue;
        }
        RcSourceRange old = w->interval.range;
        uint64_t old_end = old.offset + old.size;
        RcRegistryBufferLocation location = w->native_buffer;
        if (location.allocation && end > old.offset && end < old_end) location.offset += end - old.offset;
        r->authority.root = erase_node(&r->authority, r->authority.root, i);
        if (old.offset < range.offset && old_end > end) {
            uint32_t right = r->free_writer; uint64_t id = identity(r);
            r->free_writer = r->writers[right].free_next;
            r->writers[right] = (Writer){.interval = {.range = {old.backing, end, old_end - end},
                .order = id, .priority = mixed(id)}, .id = id, .submission = w->submission,
                .producer_order = w->producer_order, .foreign_order = w->foreign_order,
                .position = w->positioned_image_effect ? w->position : (RcRegistryPosition){0},
                .visibility_position = w->visibility_position, .visibility_latest = w->visibility_latest,
                .positioned_image_effect = w->positioned_image_effect,
                .native_buffer = location,
                .native_buffer_address = w->native_buffer_address + (end - old.offset),
                .native_buffer_mapping = w->native_buffer_mapping,
                .native_buffer_compacted = w->native_buffer_compacted,
                .state = WRITE_FOREIGN, .kind = w->kind, .unordered = w->unordered, .kinds = w->kinds,
                .diagnostic_flags = w->diagnostic_flags, .diagnostic_origin = w->diagnostic_origin,
                .diagnostic_position = w->diagnostic_position, .diagnostic_guard_reason = w->diagnostic_guard_reason};
            writer_count_add(r);
            r->stats.writer_foreign++; ++*writer_kind_count(r, w->kind);
            r->authority.root = insert_node(&r->authority, r->authority.root, right);
            w->interval.range.size = range.offset - old.offset;
        } else if (old.offset < range.offset) w->interval.range.size = range.offset - old.offset;
        else {
            w->interval.range = (RcSourceRange){old.backing, end, old_end - end};
            w->native_buffer = location;
            w->native_buffer_address += end - old.offset;
        }
        w->interval.left = w->interval.right = 0;
        r->authority.root = insert_node(&r->authority, r->authority.root, i);
    }
    if (g.count) invalidate(r, range);
    if (proof) r->stats.visibility_removed += g.count;
done:
    if (result == RC_REGISTRY_CAPACITY) r->stats.reconcile_capacity++;
    if (updated) {
        for (uint32_t j = 0; j < g.count; ++j) foreign_coverage_release(r, updated[j]);
        if (updated != local) free(updated);
    }
    return result;
}
RcRegistryResult rc_registry_reconcile_cpu(RcRegistry *r, RcSourceRange range) {
    if (!r || !rc_range_valid(range)) return RC_REGISTRY_INVALID;
    lock_registry(r);
    if ((r->guards && range.backing != RC_GUEST_COMPATIBILITY_BACKING) ||
        overlaps(r, &r->authority, r->authority.root, range, pending_writer, NULL)) {
        r->stats.reconcile_busy++;
        unlock_registry(r); return RC_REGISTRY_BUSY;
    }
    RcRegistryResult result = reconcile_locked(r, range, NULL);
    unlock_registry(r); return result;
}
RcRegistryResult rc_registry_visibility_begin(RcRegistry *r, RcRegistryVisibilityMode mode,
                                             RcRegistryVisibility *proof) {
    if (proof) *proof = (RcRegistryVisibility){0};
    if (!r || !proof || (mode != RC_REGISTRY_VISIBILITY_ORDERED && mode != RC_REGISTRY_VISIBILITY_COMPLETED))
        return RC_REGISTRY_INVALID;
    lock_registry(r);
    uint64_t horizon = identity(r);
    if (horizon) {
        *proof = (RcRegistryVisibility){.horizon = horizon, .mode = mode,
                                       .writer_kinds = 1u << RC_REGISTRY_BUFFER};
        r->stats.visibility_proofs++;
    }
    unlock_registry(r); return horizon ? RC_REGISTRY_OK : RC_REGISTRY_EXHAUSTED;
}
RcRegistryResult rc_registry_visibility_apply(RcRegistry *r, const RcRegistryVisibility *proof,
                                             RcSourceRange range) {
    if (!r || !proof || !proof->horizon || !proof->writer_kinds ||
        (proof->writer_kinds & ~((1u << (RC_REGISTRY_UNKNOWN + 1)) - 1)) || !rc_range_valid(range) ||
        (proof->mode != RC_REGISTRY_VISIBILITY_COMPLETED &&
         (proof->mode != RC_REGISTRY_VISIBILITY_ORDERED || !proof->submission))) return RC_REGISTRY_INVALID;
    lock_registry(r);
    if (proof->horizon > r->clock) { unlock_registry(r); return RC_REGISTRY_STALE; }
    RcRegistryResult result = reconcile_locked(r, range, proof);
    if (result == RC_REGISTRY_OK) r->stats.visibility_applied++;
    unlock_registry(r); return result;
}
void rc_registry_fail_closed(RcRegistry *r) {
    if (!r) return;
    lock_registry(r); poison(r); unlock_registry(r);
}

uint64_t rc_registry_authority_epoch(RcRegistry *r) {
    return r ? __atomic_load_n(&r->authority_epoch, __ATOMIC_ACQUIRE) : 0;
}
uint64_t rc_registry_ordered_epoch(RcRegistry *r) {
    return r ? __atomic_load_n(&r->ordered_epoch, __ATOMIC_ACQUIRE) : 0;
}

uint32_t rc_registry_changes(RcRegistry *r, uint64_t *cursor, RcRegistryChange *out, uint32_t capacity, int *overflow) {
    if (overflow) *overflow = 0;
    if (!r || !cursor || !out || !capacity || !overflow) return 0;
    lock_registry(r);
    uint64_t oldest = r->change_sequence > r->capacity.records ? r->change_sequence - r->capacity.records : 0;
    if (*cursor < oldest || *cursor > r->change_sequence) {
        *overflow = 1;
        *cursor = oldest;
    }
    uint32_t n = 0;
    while (n < capacity && *cursor < r->change_sequence)
        out[n++] = r->changes[(*cursor)++ % r->capacity.records];
    unlock_registry(r);
    return n;
}
uint64_t rc_registry_change_position(RcRegistry *r) {
    if (!r) return 0;
    lock_registry(r);
    uint64_t position = r->change_sequence;
    unlock_registry(r);
    return position;
}
uint32_t rc_registry_schedule_changes(RcRegistry *r, uint64_t *cursor, RcRegistryScheduleChange *out,
                                      uint32_t capacity, int *overflow) {
    if (overflow) *overflow = 0;
    if (!r || !cursor || !out || !capacity || !overflow) return 0;
    lock_registry(r);
    uint64_t oldest = r->schedule_sequence > r->capacity.records ? r->schedule_sequence - r->capacity.records : 0;
    if (*cursor < oldest || *cursor > r->schedule_sequence) { *overflow = 1; *cursor = oldest; }
    if (r->stats.fail_closed) *overflow = 1;
    uint32_t n = 0;
    while (n < capacity && *cursor < r->schedule_sequence)
        out[n++] = r->schedule_changes[(*cursor)++ % r->capacity.records];
    unlock_registry(r); return n;
}
uint64_t rc_registry_schedule_change_position(RcRegistry *r) {
    if (!r) return 0;
    lock_registry(r); uint64_t position = r->schedule_sequence;
    unlock_registry(r); return position;
}
uint32_t rc_registry_collect(RcRegistry *r, uint32_t budget) {
    if (!r) return 0;
    lock_registry(r); uint32_t result = collect_locked(r, budget); unlock_registry(r); return result;
}
RcRegistryStats rc_registry_stats(RcRegistry *r) {
    if (!r) return (RcRegistryStats){.fail_closed = 1};
    lock_registry(r); RcRegistryStats result = r->stats; unlock_registry(r); return result;
}
void rc_registry_diagnostics(RcRegistry *r, RcRegistryDiagnostics *out) {
    if (!out) return;
    if (!r) { *out = (RcRegistryDiagnostics){0}; return; }
    lock_registry(r); *out = r->diagnostics; unlock_registry(r);
}
const char *rc_registry_diagnostic_name(RcRegistryDiagnosticReason reason) {
    static const char *const names[] = {
        "source_pending_image", "source_pending_buffer", "source_pending_unknown",
        "source_submitted_image", "source_submitted_buffer", "source_submitted_unknown",
        "source_foreign_image", "source_foreign_buffer", "source_foreign_unknown",
        "source_held_image", "source_held_buffer", "source_held_unknown", "typed_begin", "typed_ready",
        "hold_missing", "hold_state", "hold_kind", "hold_forgotten", "hold_unbound", "hold_image", "hold_range",
        "other_hold", "image_foreign", "image_incomplete", "image_schedule", "image_prefix", "image_identity",
        "image_edge", "generic_image", "foreign_image", "foreign_buffer", "foreign_unknown", "missing_image"
    };
    _Static_assert(sizeof names / sizeof names[0] == RC_REGISTRY_DIAG_COUNT, "diagnostic names");
    return reason < RC_REGISTRY_DIAG_COUNT ? names[reason] : "invalid";
}
void rc_registry_diagnose_missing_image(RcRegistry *r, RcRegistryWrite ticket,
                                      RcRegistryPosition position, uint32_t prepared_actuals) {
    if (!r) return;
    lock_registry(r); Writer *w = writer_of(r, ticket);
    if (w) {
        w->diagnostic_flags |= RC_REGISTRY_DIAG_SYNTHETIC;
        diagnose(r, RC_REGISTRY_DIAG_MISSING_IMAGE, ticket.index, w->interval.range, position,
            (RcRegistryImageBinding){0}, NULL, prepared_actuals);
    }
    unlock_registry(r);
}
