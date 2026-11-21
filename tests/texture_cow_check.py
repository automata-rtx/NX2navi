"""Exercise COW source eviction and failed-write boundaries using production texture functions."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texture_material_integration_check as shared


def function(source, signature):
    start = source.index(signature)
    return source[start:source.index('\n}\n', start) + 3]


SUPPORT = r'''
static TextureMaterialInstance *cow_watch;
static int cow_fail_first, cow_hold_destruction;
static unsigned cow_allocation_checks, cow_checkpoints, cow_prepares, cow_writes;
static void (*cow_done)(void *);
static void *cow_done_arg;
static int cow_allocation_probe(void);
static void cow_checkpoint_probe(void);
'''

ADAPTERS = r'''
static int source_upload_retain_current(Tex *e){assert(!e->source_canonical);return 0;}
static int cow_allocation_probe(void) {
    if (!cow_watch) return 0;
    assert(cow_watch->lifetime.captures == 1);
    if (cow_fail_first) { cow_fail_first = 0; return 1; }
    // The first allocation failed; the real COW retry evicted the source before asking again.
    assert(!cow_watch->material->resident.image && cow_watch->queued && !cow_watch->deferred);
    content_collect(UINT64_MAX);
    assert(!cow_watch->destroyed && !cow_done && cow_watch->lifetime.captures == 1);
    cow_allocation_checks++;
    return 0;
}
static void cow_checkpoint_probe(void) {
    if (!cow_watch) return;
    // Simulate collection at an outermost upload-end checkpoint, before the COW CPU hold is handed off.
    content_collect(UINT64_MAX);
    assert(cow_watch->lifetime.captures == 1 && !cow_watch->deferred && !cow_done);
    cow_checkpoints++;
}
static Tex *cow_entry;
static void texture_registration_source_clear(Tex *e) { (void)e; }
static Tex *entry_of(const tex_t *t) { assert(t == &cow_entry->t); return cow_entry; }
static void make_fresh(Tex *e) { assert(e == cow_entry); }
static Tex *pinned_writable(const tex_t *t) { assert(t == &cow_entry->t); return NULL; }
static void touch(Tex *e) { (void)e; }
static void tex_image_prepare(const Tex *entry, uint64_t size) {
    assert(entry->pub.image && entry->pub.addr && size); cow_prepares++;
}
typedef int VkDirtyPages;
static int source_stamp_equal(const TextureSourceStamp *a, const TextureSourceStamp *b) {
    return !memcmp(a,b,sizeof *a);
}
typedef int Region;
typedef int NVNcopyRegion;
'''

CASES = r'''
// The shared ordinary promotion cases record their own image writes before these COW-specific checks.
cow_prepares=0;
// A retained source must survive destination allocation even when the real reclaim path detaches it.
uint8_t cow_bytes[64];memset(cow_bytes,103,sizeof cow_bytes);
Tex cow=make_tex(118784,cow_bytes);assert(!content_early_prepare(&cow));publish_snapshot(&cow,1800);
TextureMaterial *cow_owner=cow.material;cow_watch=cow_owner->resident.instance;
submit_epoch++;cow_fail_first=1;cow_hold_destruction=1;
unsigned cow_before=copies;uint64_t cow_id=cow_watch->value.id;
assert(content_write_begin(&cow,1));
assert(cow_allocation_checks==1 && cow_checkpoints==1 && copies==cow_before+1);
assert(!cow.material && cow.uploaded && !cow.up_retry && cow.pub.image!=1800);
assert(!memcmp(physical[cow.pub.image],cow_bytes,sizeof cow_bytes));
assert(cow_watch->value.id==cow_id && !cow_watch->lifetime.captures && cow_watch->deferred);
assert(cow_done && !cow_watch->destroyed && cow_owner->retired && material_ledger.pending_instances==1);
// Backend destruction, not release of the temporary capture, settles the final instance and provider owner.
cow_done(cow_done_arg);cow_done=NULL;cow_done_arg=NULL;cow_watch=NULL;cow_hold_destruction=0;
content_collect(UINT64_MAX);assert(!material_ledger.owners && !material_ledger.pending_instances && !material_snapshot_bytes);
// Selecting and loading the same prepared attachment owns no guest write. A real generic write still reserves it.
cow_entry=&cow;
assert(vk_texture_get_target(&cow.t)==&cow.pub && !cow_prepares);
assert(vk_texture_get_write(&cow.t)==&cow.pub && cow_prepares==1);
cow_prepares=0;

// A nonresident exact source that cannot be restored must produce no destination or copy.
Tex unavailable=make_tex(122880,cow_bytes);assert(!content_early_prepare(&unavailable));publish_snapshot(&unavailable,1801);
TextureMaterial *unavailable_owner=unavailable.material;submit_epoch++;assert(content_reclaim(64,1)==64);
content_collect(UINT64_MAX);assert(!unavailable.pub.image && unavailable_owner->provider);
allocation_failure=1;tex_deferred=0;unsigned copy_before=copies,purge_before=purged;
assert(!content_write_begin(&unavailable,1) && tex_deferred);
assert(unavailable.material==unavailable_owner && !unavailable.pub.image && !unavailable.source_copy);
assert(!unavailable_owner->instances && !unavailable_owner->resident.image && copies==copy_before && purged==purge_before);
tex_deferred=0;cow_upload_boundary(&unavailable,NULL,0,NULL);
assert(tex_deferred && !cow_writes && copies==copy_before && !unavailable.source_copy);
assert(unavailable.material==unavailable_owner && material_snapshot_bytes==64);

// Exercise both actual writable wrappers with a stale association image: restoration failure cannot publish it.
unavailable.pub.image=1801;cow_entry=&unavailable;tex_deferred=0;
assert(!vk_texture_writable_locked(&unavailable.pub) && tex_deferred && !cow_prepares);
tex_deferred=0;assert(!vk_texture_get_write(&unavailable.t) && tex_deferred && !cow_prepares);
// The synchronous WriteTexels boundary must report an unretryable required failure before issuing a write.
tex_deferred=0;expect_fatal=1;
if(!setjmp(fatal_jump)){cow_image_texels_boundary(&unavailable,NULL,NULL,NULL,1);assert(0);}
assert(tex_deferred && !cow_writes && copies==copy_before && !cow_prepares);
expect_fatal=0;allocation_failure=0;unavailable.pub.image=0;
content_release(&unavailable,1);content_collect(UINT64_MAX);
assert(!material_ledger.owners && !material_snapshot_bytes);
puts("texture COW: exact source survives reclaim/checkpoint; failed preparation cannot expose or write a destination");
'''


def fixture():
    source = (shared.ROOT / 'runtime/vk_texture.c').read_text()
    result = shared.content_fixture()
    result = result.replace('static void tex_image_prepare(const Tex *e,uint64_t size){assert(e->pub.image&&e->pub.addr&&size);}',
                            'static void tex_image_prepare(const Tex *,uint64_t);')
    # The writable API relies on the production Tex layout: its public image is the first member.
    result = result.replace('typedef struct Tex { TexKey key; VkTexImage pub;',
                            'typedef struct Tex { VkTexImage pub; TexKey key;')
    result = result.replace('TextureSourceStamp source_copy_stamp;',
                            'TextureSourceStamp source_copy_stamp,source_authority_stamp;')
    result = result.replace('static int vma_image(VkCtx *c,', SUPPORT + '\nstatic int vma_image(VkCtx *c,', 1)
    result = result.replace('allocations++;if(allocation_failure)',
                            'allocations++;if(cow_allocation_probe())return -1;if(allocation_failure)', 1)
    # The callback stub is earlier than SUPPORT; declarations remain before every use.
    result = result.replace(SUPPORT, '', 1)
    result = result.replace('static void vk_defer_image_notify(', SUPPORT + '\nstatic void vk_defer_image_notify(', 1)
    result = result.replace('vk_defer_image_reclaim(image,alloc,credit);done(arg);',
        'vk_defer_image_reclaim(image,alloc,credit);if(cow_hold_destruction){assert(!cow_done);cow_done=done;cow_done_arg=arg;}else done(arg);', 1)
    result = result.replace('static void vk_upload_end(void){}',
                            'static void vk_upload_end(void){cow_checkpoint_probe();}', 1)
    wrappers = function(source, 'static const VkTexImage *texture_writable(')
    wrappers += function(source, 'const VkTexImage *vk_texture_writable_locked(')
    wrappers += function(source, 'static const VkTexImage *get_texture_write(')
    wrappers += function(source, 'const VkTexImage *vk_texture_get_write(')
    wrappers += function(source, 'const VkTexImage *vk_texture_get_target(')
    # Execute the production upload prefix through its first destination mutation. The remaining conversion code
    # is unrelated to whether a failed COW transaction is allowed to cross this boundary.
    prepare_upload = function(source, 'static int source_upload_prepare(')
    upload = function(source, 'static void upload(Tex *e,')
    upload = upload[:upload.index('    static volatile int32_t n_bad,')]
    upload = upload.replace('static void upload(', 'static void cow_upload_boundary(', 1)
    upload += '    (void)dirty;(void)tracked;(void)cutoff;(void)material_ready;(void)material_digest;cow_writes++;\n}\n'
    texels = function(source, 'static int image_texels(')
    texels = texels[:texels.index('    static volatile int32_t n_astc,')]
    texels = texels.replace('static int image_texels(', 'static int cow_image_texels_boundary(', 1)
    texels += '    (void)g;(void)r;(void)mem;cow_writes++;return 1;\n}\n'
    result = result.replace('int main(void){\nuint8_t bytes[64];',
        ADAPTERS + wrappers + prepare_upload + upload + texels + '\nint main(void){\nuint8_t bytes[64];', 1)
    marker = 'content_shutdown();assert(!material_snapshot_bytes && !rel_bytes);'
    assert result.count(marker) == 1
    return result.replace(marker, marker + '\n' + CASES, 1)


if __name__ == '__main__':
    work = shared.ROOT / 'build/test'
    work.mkdir(parents=True, exist_ok=True)
    work.resolve().relative_to(shared.ROOT.resolve())
    with tempfile.TemporaryDirectory(prefix='texture-cow-', dir=work) as temporary:
        shared.OUT = Path(temporary)
        shared.compile_and_run('cow', fixture())
