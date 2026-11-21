"""Check production pin ownership through copy retirement and event cancellation."""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CMD = (ROOT / 'runtime/nvn_api_cmd.c').read_text()
PLAN = (ROOT / 'runtime/nvn_parent_plan.h').read_text()
EXEC = (ROOT / 'runtime/vk_exec.c').read_text()
DRAW = (ROOT / 'runtime/vk_exec_draw.c').read_text()
TEXTURE = (ROOT / 'runtime/vk_texture.c').read_text()


def section(source, start, stop):
    first = source.index(start)
    return source[first:source.index(stop, first)]


PREFIX = r'''
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include "nvn_hmap.h"
typedef struct { uint64_t serial, address; } tex_t;
typedef tex_t NVNtexture;
typedef int PlatMutex;
#define PLAT_MUTEX_INIT 0
#define plat_mutex_lock_shared plat_mutex_lock
#define plat_mutex_unlock_shared plat_mutex_unlock
static void plat_mutex_lock(PlatMutex *m) { assert(!*m); *m = 1; }
static void plat_mutex_unlock(PlatMutex *m) { assert(*m); *m = 0; }
#define RP_SCOPE(...)
#define RP_VALUE(...)
#define NX2NAVI_REPLAY_PHASES 0
#define NVN_PIN_AGE 1200
static const char *cfg_get(const char *name) { (void)name; return NULL; }
static void rt_log(const char *message, ...) { (void)message; }
static void rt_fatal(const char *message, ...) { fputs(message, stderr); abort(); }
typedef struct { uint32_t op, size; } op_hdr;
enum { OP_BEGIN, OP_CLEAR_TEX, OP_END, OP_COUNT_ };
#define PIN_OPS (1ull << OP_CLEAR_TEX)
static const op_hdr *nvn_op_first(uint64_t handle) { return (const void *)(uintptr_t)handle; }
static const op_hdr *nvn_op_next(const op_hdr *op) { return (const void *)((const char *)op + op->size); }
static const void *nvn_op_payload(const op_hdr *op) { return op + 1; }
static int op_pins(unsigned op, const void *payload, const tex_t **out) {
    assert(op == OP_CLEAR_TEX);
    *out = *(const tex_t *const *)payload;
    return 1;
}
typedef int NvnRecordSpan;
typedef int NvnExecStep;
typedef int NvnFrontStep;
typedef int VkTextureRegistrationScope;
typedef int p_bind_buf;
typedef struct { int live; } RcBudgetTicket;
typedef int VkResult;
typedef int VkReplayCaptureWait;
enum { VK_SUCCESS, VK_ERROR_OUT_OF_HOST_MEMORY };
#define CONT_STATE_BYTES 32
typedef struct { RcBudgetTicket budget; unsigned refs; } ParkedUbo;
static int vk_exec_draw_update_history_defer(void *commands, uint64_t bytes, RcBudgetTicket *budget) {
    assert(commands && bytes && budget->live);
    return 0;
}
static VkResult snapshot_metadata_allocate(size_t bytes, int clear, RcBudgetTicket *ticket, void **allocation,
                                           VkReplayCaptureWait *wait) {
    (void)wait;
    assert(!clear && !ticket->live);
    *allocation = malloc(bytes);
    if (!*allocation) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ticket->live = 1;
    return VK_SUCCESS;
}
static void vk_replay_metadata_free(void *storage,RcBudgetTicket *ticket) {
    assert(storage && ticket->live);
    free(storage);
}
typedef struct { const tex_t *color[8], *depth; } p_set_rt;
static uint64_t replayed, vk_exec_submit_seq;
static uint64_t gfx_replayed_submits(void) { return replayed; }
static VkTextureRegistrationScope *vk_exec_texture_registration_scope(void) { return NULL; }
static VkTextureRegistrationScope *vk_texture_registration_scope_retain(VkTextureRegistrationScope *s) { return s; }
static void vk_texture_registration_scope_end(VkTextureRegistrationScope *s) { assert(!s); }
'''

TARGET_ADAPTER = r'''
typedef struct VkTextureWriteTarget VkTextureWriteTarget;
typedef struct {
    struct { uintptr_t image; int gone; } pub;
    VkTextureWriteTarget *pending_write;
    void *material;
    uint64_t content_revision;
    int pin_owned, pin_detached, freed;
} Tex;
typedef void TextureMaterial;
typedef struct { TextureMaterial *material; } HeldMaterial;
typedef struct { TextureMaterial *material; } VkTextureRestore;
#define VK_NULL_HANDLE 0
struct VkTextureWriteTarget {
    Tex *entry;
    const tex_t *pin;
    uintptr_t image, selected;
    void *allocation;
    uint64_t revision, bytes;
    int seeded, viewed;
    VkTextureRestore *seed;
    void *cpu_seed;
    struct { HeldMaterial *instance; } source_use;
    struct { uint64_t registration; } storage;
};
typedef struct { uintptr_t vma; } VkCtx;
static unsigned num_sources, native_destinations;
static int tex_lock_depth;
static void lock(void) { assert(!pin_lock && !copy_lock); tex_lock_depth++; }
static void unlock(void) { assert(tex_lock_depth); tex_lock_depth--; }
static void free_tex(Tex *e, int retained_output) { assert(!retained_output && tex_lock_depth && e->pin_detached && !e->pin_owned); e->freed++; }
static VkCtx *vk_ctx(void) { static VkCtx c={1}; return &c; }
static void resource_core_vma_destroy_image(uintptr_t vma,uintptr_t image,void *allocation) {
    assert(vma==1 && image==43 && allocation==(void *)11 && !tex_lock_depth && !pin_lock);
    native_destinations++;
}
static void vk_exec_image_gone(uintptr_t image) { (void)image;abort(); }
static void vk_defer_image(uintptr_t image,void *allocation) { (void)image;(void)allocation;abort(); }
static void vk_texture_restore_cancel(VkTextureRestore *restore) { assert(!restore); }
static int write_target_cpu_current(const VkTextureWriteTarget *target) { (void)target; abort(); }
static int write_target_storage_current_locked(const VkTextureWriteTarget *target) { (void)target; abort(); }
static void write_target_cpu_destroy(void **seed) { assert(!*seed); }
static void write_target_release_source(VkTextureWriteTarget *target) {
    assert(tex_lock_depth && !pin_lock);
    if (target->source_use.instance) { target->source_use.instance=NULL; num_sources++; }
}
'''

MAIN = r'''
typedef struct { op_hdr begin, clear; const tex_t *texture; op_hdr end; } Clear;
static int releases, last_release;
static void release_owner(void *arg) {
    assert(!pin_lock && !copy_lock);
    releases++;
    last_release = *(int *)arg;
}

static VkTextureWriteTarget *candidate(Tex *e, const tex_t *pin, HeldMaterial *source) {
    VkTextureWriteTarget *target=calloc(1,sizeof *target);
    assert(target);
    target->pin=nvn_texture_pin_retain_attachment(pin->serial,e);
    assert(target->pin==pin);
    target->entry=e;
    target->selected=e->pub.image;
    target->revision=e->content_revision;
    target->source_use.instance=source;
    return target;
}

static void canceled_first_write(void) {
    tex_t first={551,0x8000}, next={552,0x8000};
    const tex_t *old_pin=tex_pin(&first), *new_pin=tex_pin(&next);
    HeldMaterial ordinary={.material=&first};
    Tex old={.pub={.image=29},.material=ordinary.material,.content_revision=7,.pin_owned=1};
    Tex new={.pub={.image=61},.content_revision=9,.pin_owned=1};
    assert(nvn_texture_pin_attach(old_pin,&old,pin_owner_release));
    assert(nvn_texture_pin_attach(new_pin,&new,pin_owner_release));
    const tex_t *command_pin=nvn_texture_pin_retain_attachment(first.serial,&old);
    const tex_t *handed_pin=nvn_texture_pin_retain_attachment(next.serial,&new);
    VkTextureWriteTarget *owner=candidate(&old,old_pin,&ordinary);
    old.pending_write=owner;
    assert(write_target_current_locked(owner));
    // An independent canceled holder cannot unpublish the tentative writer's attachment.
    VkTextureWriteTarget *other=candidate(&old,old_pin,&ordinary);
    assert(!write_target_current_locked(other));
    vk_texture_write_target_cancel(other);
    assert(old.pending_write==owner && old.pin_owned && num_sources==1);
    assert(nvn_texture_pin_attachment(old_pin)==&old && nvn_texture_pin_attachment(handed_pin)==&new);
    assert(!nvn_texture_pin_unattach(old_pin,&new) && !nvn_texture_pin_unattach(new_pin,&old));
    owner->image=43;owner->allocation=(void *)11;
    vk_texture_write_target_cancel(owner);
    assert(!nvn_texture_pin_attachment(command_pin) && !old.pin_owned && !old.pending_write && !old.freed);
    assert(num_sources==2 && native_destinations==1 && !tex_lock_depth);
    assert(nvn_texture_pin_attachment(handed_pin)==&new);

    // Retry the same command pin, then cancel a later preparation while the committed owner is still held.
    assert(nvn_texture_pin_attach(command_pin,&old,pin_owner_release));
    old.pin_owned=1;old.material=NULL;old.pub.image=71;
    const tex_t *first_part=nvn_texture_pin_retain_attachment(first.serial,&old);
    assert(first_part==command_pin);
    VkTextureWriteTarget *later=candidate(&old,old_pin,NULL);
    assert(write_target_current_locked(later));
    vk_texture_write_target_cancel(later);
    assert(nvn_texture_pin_attachment(first_part)==&old && old.pin_owned && !old.freed && native_destinations==1);
    nvn_texture_pin_release(first_part);
    nvn_texture_pin_release(handed_pin);
    nvn_texture_pin_release(command_pin);
    nvn_texture_pin_attachments_clear();
    assert(!old.pin_owned && !new.pin_owned && !old.freed && !new.freed);

    assert(nvn_texture_pin_attach(old_pin,&old,pin_owner_release));
    old.pin_owned=old.pin_detached=1;old.material=ordinary.material;
    owner=candidate(&old,old_pin,&ordinary);
    old.pending_write=owner;
    vk_texture_write_target_cancel(owner);
    assert(old.freed==1 && !old.pin_owned && !nvn_texture_pin_attachment(old_pin) && num_sources==3);
}

int main(void) {
    tex_t old = {41, 0x4000}, replacement = {42, 0x4000};
    const tex_t *pin = tex_pin(&old);
    const tex_t *next = tex_pin(&replacement);
    int owner = 100, newer_owner = 200;
    assert(!nvn_texture_pin_attach(&old, &owner, release_owner));
    assert(nvn_texture_pin_attach(pin, &owner, release_owner));
    assert(!nvn_texture_pin_attach(pin, &newer_owner, release_owner));
    assert(nvn_texture_pin_attach(next, &newer_owner, release_owner));
    assert(!nvn_texture_pin_attachment(&old));
    assert(nvn_texture_pin_attachment(pin) == &owner && nvn_texture_pin_attachment(next) == &newer_owner);
    assert(!nvn_texture_pin_retain_attachment(old.serial, &newer_owner));
    const tex_t *producer_pin = nvn_texture_pin_retain_attachment(old.serial, &owner);
    assert(producer_pin == pin);

    copy_prefix *pf = calloc(1, sizeof *pf + sizeof(Clear));
    assert(pf);
    *pf = (copy_prefix){.bytes = sizeof(Clear), .magic = 1, .used_at = 17, .pins = 1};
    Clear *bytes = (void *)(pf + 1);
    *bytes = (Clear){{OP_BEGIN, sizeof(op_hdr)}, {OP_CLEAR_TEX, sizeof(op_hdr) + sizeof pin},
                    pin, {OP_END, sizeof(op_hdr)}};
    assert(copy_pin_refs(pf, 1) == 1);
    const uint64_t handle = (uintptr_t)bytes;
    nvn_hm_put(&copy_live, &handle, pf);
    Cont *c = NULL;
    VkReplayCaptureWait wait = 0;
    assert(cont_prepare(&handle, 1, 0, &c, &wait) == VK_SUCCESS);
    assert(cont_new(c, &handle, 1, 0, &bytes->clear) == c);
    cont_free(c); // Readiness released its reference after the submission parked.
    assert(c->pin_leases[0] == pf && pf->plan_readers == 1 && c->h[0] != handle);
    assert(*(const tex_t *const *)nvn_op_payload(c->at) == pin);

    // The game released the command and replay reached its original fence. A parked event keeps its pin counted.
    nvn_hm_del(&copy_live, &handle);
    pf->magic = 0;
    zombie_add(pf);
    replayed = 17;
    doom_free(zombie_sweep(replayed, NULL));
    assert(nzombies == 1);
    uint64_t now = NVN_PIN_AGE + 90;
    nsubmits = now;
    nvn_pins_sweep(now, &next, 1);
    assert(!releases && nvn_texture_pin_attachment(pin) == &owner);

    // Canceling a continuation drops the original copy exactly once. A later pin sweep retires only its owner.
    cont_free(c);
    assert(nzombies == 0);
    nvn_pins_sweep(now + NVN_PIN_AGE, &next, 1);
    assert(!releases && nvn_texture_pin_attachment(producer_pin) == &owner);
    nsubmits = now + NVN_PIN_AGE;
    nvn_texture_pin_release(producer_pin);
    nvn_pins_sweep(now + 2 * NVN_PIN_AGE, &next, 1);
    assert(releases == 1 && last_release == 100 && nvn_texture_pin_attachment(next) == &newer_owner);
    nvn_texture_pin_attachments_clear();
    nvn_texture_pin_attachments_clear();
    assert(releases == 2 && last_release == 200 && !nvn_texture_pin_attachment(next));
    nvn_pins_sweep(now + 3 * NVN_PIN_AGE, NULL, 0);
    assert(!pins_live && releases == 2);
    canceled_first_write();
    puts("texture pins: same-address serials, tentative cancel and retry, handed owner retention, event lease and shutdown passed");
}
'''

parts = [
    section(CMD, 'typedef struct {\n    tex_t t;', '// (flatten_recording, pin_refs)'),
    section(CMD, 'static void pin_ref(', '// ---------- recording copies'),
    section(CMD, 'typedef struct {\n    uint64_t magic, bytes,', 'int nvn_record_uniform_offsets('),
    section(CMD, 'static void zombie_add(', 'int nvn_handle_live('),
    section(CMD, 'static copy_prefix *copy_find_submitted(', 'int nvn_submission_ordered_scratch_max('),
    section(PLAN, 'static void plan_drop_readers(', 'static void parent_plan_release('),
    section(CMD, 'void *nvn_record_retain_pins(', '// Validate newly recorded bytes'),
    section(EXEC, 'typedef struct Cont {', 'static struct { uintptr_t q;'),
    section(EXEC, 'static size_t cont_command_bytes(', 'void vk_exec_discard_held_locked('),
    TARGET_ADAPTER,
    section(TEXTURE, 'static void pin_owner_release(', '\n}\n') + '\n}\n',
    section(TEXTURE, 'static int write_target_current_locked(', '\n}\n') + '\n}\n',
    section(TEXTURE, 'static void write_target_release_allocation(VkTextureWriteTarget *target)', '\n}\n') + '\n}\n',
    section(TEXTURE, 'void vk_texture_write_target_cancel(', '\n}\n') + '\n}\n',
]
begin = section(TEXTURE, 'const VkTexImage *vk_texture_write_target_begin(', 'static void explicit_written(')
assert begin.index('e->pub.image = target->image;') < begin.index('e->pending_write = NULL;')
with tempfile.TemporaryDirectory() as tmp:
    deferred = section(EXEC, 'static void cont_free(', 'void vk_exec_discard_held_locked(')
    retained = section(DRAW, 'int vk_exec_draw_update_history_defer(', 'static void upd_entry_release(')
    released = section(DRAW, 'static void upd_entry_release(InlineUpdate *entry) {', 'static void upd_note(')
    assert 'vk_exec_draw_update_history_defer(c, c->bytes, &c->budget)' in deferred
    assert 'entry->command_owner = owner' in retained
    assert '--owner->refs' in released and 'vk_replay_metadata_free(owner->commands, &owner->budget)' in released
    directory = Path(tmp)
    source = directory / 'texture_pin_attachment_check.c'
    binary = directory / 'texture_pin_attachment_check.exe'
    source.write_text(PREFIX + '\n'.join(parts) + MAIN)
    cc = Path(os.environ.get('CC', 'clang'))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get('PATH', ''))
    subprocess.run([str(cc), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
                    '-Wno-unused-variable', '-fsanitize=undefined', '-fsanitize-trap=all',
                    '-I', str(ROOT / 'runtime'), str(source), '-o', str(binary)], env=env, check=True)
    subprocess.run([str(binary)], env=env, check=True)
