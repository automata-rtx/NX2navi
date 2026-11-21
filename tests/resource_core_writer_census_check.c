#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include "../runtime/resource_core_registry.c"

static char log_text[8192];
static size_t log_length;
static unsigned log_lines;

void plat_mutex_lock(PlatMutex *mutex) {
    assert(!mutex->p);
    mutex->p = mutex;
}
void plat_mutex_unlock(PlatMutex *mutex) {
    assert(mutex->p == mutex);
    mutex->p = NULL;
}
void plat_cond_init(PlatCond *cond) {
    assert(!cond->p);
    cond->p = cond;
}
int plat_cond_wait_ns(PlatCond *cond, PlatMutex *mutex, uint64_t ns) {
    (void)cond;
    (void)mutex;
    (void)ns;
    assert(0 && "the writer census fixture does not wait for GPU captures");
    return 0;
}
void plat_cond_broadcast(PlatCond *cond) {
    assert(cond->p == cond);
}
void plat_once(PlatOnce *once, void (*fn)(void *), void *arg) {
    if (!once->state) {
        once->state = 1;
        fn(arg);
    }
}
void rt_log(const char *format, ...) {
    va_list args;
    va_start(args, format);
    int n = vsnprintf(log_text + log_length, sizeof log_text - log_length, format, args);
    va_end(args);
    assert(n > 0 && (size_t)n + 1 < sizeof log_text - log_length);
    log_length += (size_t)n;
    log_text[log_length++] = '\n';
    log_text[log_length] = 0;
    log_lines++;
}

int main(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 7, 4});
    assert(r);
    runtime_registry = r;
    rc_registry_runtime_native_handoff(2, 3, 4);
    RcRegistryWrite ticket;
    for (uint64_t ordinal = 1; ordinal <= 2; ++ordinal)
        assert(rc_registry_write_begin_at(r, (RcSourceRange){2, 0, 8}, RC_REGISTRY_BUFFER,
                                         (RcRegistryPosition){2, ordinal}, &ticket) == RC_REGISTRY_OK);
    assert(rc_registry_write_guard_begin_reason_at(r, RC_REGISTRY_BUFFER, (RcRegistryPosition){3, 1},
           RC_REGISTRY_GUARD_GPA_PROJECTION, &ticket) == RC_REGISTRY_OK);
    assert(rc_registry_write_guard_begin_reason_at(r, RC_REGISTRY_UNKNOWN, (RcRegistryPosition){4, 1},
           RC_REGISTRY_GUARD_COMMAND_OPAQUE, &ticket) == RC_REGISTRY_OK);
    assert(rc_registry_write_begin_at(r, (RcSourceRange){1, 0, 8}, RC_REGISTRY_UNKNOWN,
                                     (RcRegistryPosition){5, 1}, &ticket) == RC_REGISTRY_OK);
    assert(rc_registry_hold_begin(r, (RcSourceRange){1, 8, 8}, RC_REGISTRY_IMAGE, &ticket) == RC_REGISTRY_OK);
    assert(rc_registry_write_begin(r, (RcSourceRange){1, 16, 8}, RC_REGISTRY_BUFFER, &ticket) == RC_REGISTRY_OK);
    assert(rc_registry_write_submit(r, ticket, 2) == RC_REGISTRY_OK);
    assert(rc_registry_write_complete(r, ticket, 0) == RC_REGISTRY_OK);
    RcRegistryReadBlocker blocker;
    RcSourceRange read = {2, 0, 4};
    assert(rc_registry_diagnose_read_blocker(r, read, (RcRegistryPosition){0}, &blocker));
    assert(blocker.reason == RC_REGISTRY_DIAG_SOURCE_PENDING_UNKNOWN &&
           blocker.guard_reason == RC_REGISTRY_GUARD_COMMAND_OPAQUE && blocker.sample.position.scene == 4);
    assert(rc_registry_diagnose_read_blocker(r, read, (RcRegistryPosition){3, 1}, &blocker));
    assert(blocker.reason == RC_REGISTRY_DIAG_SOURCE_PENDING_BUFFER &&
           blocker.guard_reason == RC_REGISTRY_GUARD_GPA_PROJECTION && blocker.sample.position.scene == 3);
    assert(rc_registry_diagnose_read_blocker(r, read, (RcRegistryPosition){2, 1}, &blocker));
    assert(blocker.reason == RC_REGISTRY_DIAG_SOURCE_PENDING_BUFFER && !blocker.guard_reason &&
           rc_range_equal(blocker.sample.range, (RcSourceRange){2, 0, 8}) && blocker.sample.position.ordinal == 1);
    assert(!rc_registry_diagnose_read_blocker(r, read, (RcRegistryPosition){1, 1}, &blocker));
    assert(rc_registry_diagnose_read_blocker(r, (RcSourceRange){1, 16, 8}, (RcRegistryPosition){0}, &blocker));
    assert(blocker.reason == RC_REGISTRY_DIAG_SOURCE_FOREIGN_BUFFER && blocker.sample.writer.id == ticket.id);
    assert(!r->diagnostics.counts[RC_REGISTRY_DIAG_SOURCE_PENDING_BUFFER]);
    assert(!r->diagnostics.counts[RC_REGISTRY_DIAG_SOURCE_PENDING_UNKNOWN]);
    assert(!r->diagnostics.counts[RC_REGISTRY_DIAG_SOURCE_FOREIGN_BUFFER]);
    assert(!log_lines);
    assert(rc_registry_write_guard_begin_at(r, RC_REGISTRY_BUFFER, (RcRegistryPosition){5, 1},
                                           &ticket) == RC_REGISTRY_CAPACITY);
    RcRegistryStats s = rc_registry_stats(r);
    assert(!s.fail_closed && s.writers == 7 && s.writer_peak == 7 && s.writer_refusals == 1);
    assert(strstr(log_text, "writer-capacity need 1"));
    assert(strstr(log_text, "active 7/7 image/buffer/unknown 1/4/2"));
    assert(strstr(log_text, "pending/submitted/foreign/held 5/0/1/1"));
    assert(strstr(log_text, "native final/part/uncertain 2/3/4"));
    assert(strstr(log_text, "buffer pending canonical final-scene"));
    assert(strstr(log_text, "count=2 distinct range/range-scene/range-position 1/1/2"));
    assert(strstr(log_text, "buffer pending guard part-scene guard=gpa-projection"));
    assert(strstr(log_text, "unknown pending guard uncertain-scene guard=command-opaque"));
    assert(strstr(log_text, "unknown pending raw unhanded"));
    assert(strstr(log_text, "image held raw unpositioned"));
    assert(strstr(log_text, "buffer foreign raw unpositioned"));
    assert(strstr(log_text, "groups 6 listed 6 records listed 7 omitted 0"));
    unsigned lines = log_lines;
    assert(rc_registry_write_begin(r, (RcSourceRange){1, 24, 8}, RC_REGISTRY_BUFFER,
                                  &ticket) == RC_REGISTRY_CAPACITY);
    assert(rc_registry_stats(r).fail_closed && log_lines == lines);
    runtime_registry = NULL;
    rc_registry_destroy(r);
    puts("writer census: first capacity refusal and read-only exact blocker preserve registry decisions");
    return 0;
}
