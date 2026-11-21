"""Exercise real source observations through denied snapshot admission and upload cutoff publication."""
from pathlib import Path
import argparse
import ast
import os
import re
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "runtime/vk_texture.c").read_text()
CONTENT = (ROOT / "runtime/vk_texture_content.h").read_text()


def function(text, name):
    match = re.search(r"^(?:static )?(?:inline )?[^\n]*\b" + name + r"\([^;]*?\)\s*\{", text, re.M)
    if not match:
        raise RuntimeError(name)
    begin = text.index("{", match.start())
    depth, end = 1, begin + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[match.start():end]


# Reuse the bridge fixture's dependency stubs without importing its compile/run entry point. Production authority
# and memory-cutoff decisions remain extracted from the renderer and linked to the real registry in both fixtures.
bridge = ast.parse((ROOT / "tests/resource_source_registry_check.py").read_text())
PREFIX = next(ast.literal_eval(n.value) for n in bridge.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "prefix" for t in n.targets))
SOURCE_FUNCTIONS = next(ast.literal_eval(n.value) for n in bridge.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "source_functions" for t in n.targets))
SOURCE_FUNCTIONS += ("source_observe_prior", "source_observe", "source_commit")
# The shared source bridge also has unrelated GPA-wait mocks at the end.
PREFIX = PREFIX[:PREFIX.index("typedef struct { unsigned requests;")]
PREFIX = PREFIX.replace("    int released;", "    int released,uploaded,last_ch; void *material;")
footprint_end = SOURCE.index("} TextureSampleFootprint;") + len("} TextureSampleFootprint;")
PREFIX += SOURCE[SOURCE.rindex("typedef struct {", 0, footprint_end):footprint_end] + "\n"
refusal_at = SOURCE.index("    SOURCE_REFUSAL_NONE,")
PREFIX += SOURCE[SOURCE.rindex("enum {", 0, refusal_at):SOURCE.index("};", refusal_at) + 2] + "\n"
# The production source observer can record these typed details even when tracing is
# disabled by the fixture. Use the live declarations instead of a second C struct.
host_write = (ROOT / "runtime/resource_host_write.h").read_text()
qualification_at = host_write.index("typedef enum VkMemCanonicalSourceQualificationStage {")
PREFIX += host_write[qualification_at:host_write.index("// Qualifies an ordinary tracked import", qualification_at)]
mapping_report_end = SOURCE.index("} SampleMappingRefusal;")
PREFIX += SOURCE[SOURCE.rindex("typedef struct {", 0, mapping_report_end):
                 SOURCE.index("static RcRegistryResult source_mapping_qualify", mapping_report_end)]
PREFIX += r'''
static RcRegistryResult vk_mem_canonical_source_qualify_report(const MemBackingCapture *capture,
    uint64_t address, uint64_t size, VkMemCanonicalSourceQualification *out) {
    assert(out);
    *out = (VkMemCanonicalSourceQualification){0};
    return vk_mem_canonical_source_qualify(capture, address, size);
}
'''
PREFIX += "static void texture_registration_source_clear(Tex *e){(void)e;}\n"
ordered_effects = function(PREFIX, "vk_exec_effects_read_allowed_range")
PREFIX = PREFIX.replace(ordered_effects, r"""
static RcSourceRange same_operation_effect;
static int vk_exec_effects_read_allowed_range(RcSourceRange range, int private_sample) {
    return !same_operation_write && (!require_private_sample || private_sample) &&
        !rc_range_overlap(range, same_operation_effect);
}
""")



SUPPORT = r'''
#include "resource_core_budget.h"
static unsigned n_checks,snapshot_attempts,uploads,advisories;
static int tex_deferred, pool_ctx;
enum { MATERIAL_PREPARE_REQUIRED, MATERIAL_PREPARE_OPTIONAL };
static struct {unsigned snapshot_refused;} material_stats;
static int content_image_eligible(const Tex *e){return !e->source_blocked;}
static int content_recovery_due(Tex *e){(void)e;return 0;}
static void content_recover(Tex *e,const TextureSourceObservation *o){(void)e;(void)o;assert(0);}
static void content_version(Tex *e,void*m){(void)e;(void)m;assert(0);}
static int vk_exec_turn_due(void){return 0;}
static int up_wait(Tex *e){(void)e;return 0;}
static uint64_t content_drop_providers(uint64_t size){assert(size);return 0;}
static void content_recovery_wait(Tex *e,uint32_t domains){assert(domains);e->source_retry_domains=domains;advisories++;}
static void content_recovery_wait_source(Tex *e,const TextureSourceStamp*s,const TextureSourceAuthority*a){
 (void)s;(void)a;e->source_retry_source=1;
}
RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer,int required,
 const RcAdmissionVector*cost,RcBudgetTicket*ticket){
 assert(producer==RC_BUDGET_SNAPSHOT&&!required&&cost->amount[RC_BUDGET_RAM]&&cost->amount[RC_BUDGET_COMMIT]);
 (void)ticket;snapshot_attempts++;return RC_ADMISSION_REFUSED;
}
static int content_early_prepare(Tex *e){
 const int policy = MATERIAL_PREPARE_REQUIRED;
 TextureSourceObservation obs;
 /* Actual observation and denied-admission branch. No capture was admitted or bytes copied. */
 /* refusal branch */
 assert(0);return 0;
}
static void content_source_discard(Tex *e){(void)e;assert(0);}
/* production upload source preparation */
static void upload(Tex *e,const void*dirty,int tracked,const TextureSourceObservation*observed){
 (void)dirty;(void)tracked;
 /* actual upload observation */
 // GPU conversion is modeled as successful. The real upload invalidates the old stamp before recording bytes,
 // then commits its pre-copy cutoff; snapshot admission and content certification are separate responsibilities.
 e->source_stamp=(TextureSourceStamp){0};uploads++;e->uploaded=1;source_commit(e,&cutoff,0);
 assert(texture_source_stamp_valid(&e->source_stamp));
}
'''

TESTS = r'''
static void source_shaped_effects(void) {
 uint8_t guest[64]={0};
 TextureSourceSpan span={0,sizeof guest};
 Tex e={.pub={.addr=4096,.mirrored=1,.levels=1,.layers=1}, .key.addr=4096,
        .t={.serial=700,.L.size=sizeof guest,.d={.format=0x25,.target=1,.flags=0x10,.stride=16,.width=2,.height=4},
            .mem=guest}, .source_spans=&span,.source_count=1};
 registry=rc_registry_create((RcRegistryConfig){16,8,4});assert(registry);
 read_position=(RcRegistryPosition){1,2};
 assert(rc_registry_declare_through(registry,read_position)==RC_REGISTRY_OK);
 canonical_ready=1;native_generation++;
 TextureSourceObservation proof;
 assert(source_observe(&e,NULL,&proof)==TEX_SOURCE_UNKNOWN && texture_source_stamp_valid(&proof.stamp));
 assert(e.source_mapping->sample_count==4 && e.source_mapping->count==2);
 source_commit(&e,&proof.stamp,0);
 RcSourceRange first=e.source_mapping->intervals[0].range;
 same_operation_effect=(RcSourceRange){first.backing,first.offset+8,8}; // first row's exact padding
 assert(source_observe(&e,&e.source_stamp,&proof)==TEX_SOURCE_CLEAN && !e.source_blocked);
 same_operation_effect=(RcSourceRange){first.backing,first.offset+4,8}; // straddles a meaningful texel
 assert(source_observe(&e,&e.source_stamp,&proof)==TEX_SOURCE_UNKNOWN && e.source_blocked &&
        e.source_refusal==SOURCE_REFUSAL_EFFECTS);
 same_operation_effect=(RcSourceRange){first.backing,first.offset+8,8};
 e.material_writable=1; // a mutable physical image keeps the original feedback discipline
 assert(source_observe(&e,&e.source_stamp,&proof)==TEX_SOURCE_UNKNOWN && e.source_blocked);
 e.material_writable=0;
 source_registration_release(&e);
 e.f.ideal=VK_FORMAT_ASTC_4x4_UNORM_BLOCK;e.t.serial++;
 assert(source_observe(&e,NULL,&proof)==TEX_SOURCE_UNKNOWN && e.source_blocked && !e.source_mapping->sample_count);
 same_operation_effect=(RcSourceRange){0};
 assert(source_observe(&e,NULL,&proof)==TEX_SOURCE_UNKNOWN && texture_source_stamp_valid(&proof.stamp));
 source_registration_release(&e);
 rc_registry_destroy(registry);
 puts("source continuity: exact sample padding, meaningful feedback and full-footprint guard passed");
}

int main(void){
 enum{N=5623,EPOCHS=60};
 registry=rc_registry_create((RcRegistryConfig){6000,32,4});assert(registry);
 TextureSourceSpan span={0,64};Tex *entries=calloc(N,sizeof *entries);assert(entries);
 for(unsigned i=0;i<N;i++){
  entries[i]=(Tex){.pub={UINT64_C(0x100000)+256*i},.t={1000+i,{64},0},.source_spans=&span,.source_count=1};
  assert(make_fresh_source(&entries[i]));
 }
 assert(uploads==N&&snapshot_attempts==N&&advisories==N);
 for(unsigned epoch=1;epoch<=EPOCHS;epoch++){
  submit_epoch++;watermark++;
  assert(rc_registry_declare_through(registry,(RcRegistryPosition){epoch,8})==RC_REGISTRY_OK);
  read_position=(RcRegistryPosition){epoch,2};
  for(unsigned i=0;i<N;i++)assert(make_fresh_source(&entries[i]));
  assert(rc_registry_handoff_through(registry,(RcRegistryPosition){epoch,8})==RC_REGISTRY_OK);
  read_position=(RcRegistryPosition){0};
  for(unsigned i=0;i<N;i++)assert(make_fresh_source(&entries[i]));
 }
 assert(n_checks==N*(1+2*EPOCHS)&&material_stats.snapshot_refused==snapshot_attempts);
 assert(qualifiers>=N); // Compatibility is selected after the actual qualification branch.
#ifdef OLD_MODE_COMPARISON
 assert(uploads==N*(1+2*EPOCHS)&&snapshot_attempts==uploads&&!passed_prior);
 printf("negative control: %u sources, %u mode alternations, %u snapshot refusals and %u uploads\n",N,2*EPOCHS,snapshot_attempts,uploads);
#else
 assert(uploads==N&&snapshot_attempts==N&&advisories==N&&passed_prior);
 printf("source continuity: %u sources, %u mode alternations, %u initial refused captures/uploads, zero repeated captures/uploads\n",N,2*EPOCHS,uploads);
 Tex *e=&entries[0];RcSourceRange range={1,e->pub.addr,e->t.L.size};
 TextureSourceAuthority before=e->source_access._content;
 RcRegistryObservation forged=before.legacy;forged.content_revision++;
 assert(!rc_registry_observation_current(registry,&forged));
 TextureSourceAuthority different=before;different.legacy.handle.index++;
 assert(!texture_authority_same(&before,&different));different=before;different.legacy.range.offset++;
 assert(!texture_authority_same(&before,&different));different=before;different.legacy.guard++;
 assert(!texture_authority_same(&before,&different));different=before;different.legacy.content_revision=0;
 assert(!texture_authority_same(&before,&different));
 // Unrelated source writes cannot change this source's byte provenance.
 assert(rc_registry_cpu_changed(registry,(RcSourceRange){1,UINT64_C(0x900000),64})==RC_REGISTRY_OK);
 assert(make_fresh_source(e)&&uploads==N);
 // A declared future producer changes generic revision, not yet the bytes. A new ordered permission can reuse
 // the old cutoff before that producer; an unordered read and the later ordered read still refuse it.
 RcRegistryWrite future;
 assert(rc_registry_write_begin_at(registry,range,RC_REGISTRY_BUFFER,(RcRegistryPosition){61,5},&future)==RC_REGISTRY_OK);
 assert(rc_registry_declare_through(registry,(RcRegistryPosition){61,8})==RC_REGISTRY_OK);
 assert(!make_fresh_source(e)&&e->source_blocked&&uploads==N);
 read_position=(RcRegistryPosition){61,2};assert(make_fresh_source(e)&&uploads==N&&!e->source_blocked);
 assert(texture_authority_same(&before,&e->source_access._content));
 read_position.ordinal=6;assert(!make_fresh_source(e)&&e->source_blocked&&uploads==N);
 assert(rc_registry_write_cancel(registry,future)==RC_REGISTRY_OK);
 read_position=(RcRegistryPosition){0};assert(make_fresh_source(e)&&uploads==N);
 // Completion with true CPU visibility changes contents even when the memory write-watch watermark is unchanged.
 assert(rc_registry_write_begin(registry,range,RC_REGISTRY_BUFFER,&future)==RC_REGISTRY_OK);
 assert(rc_registry_write_submit(registry,future,61)==RC_REGISTRY_OK);
 read_position=(RcRegistryPosition){61,6};assert(!make_fresh_source(e)&&uploads==N);
 assert(rc_registry_write_complete(registry,future,1)==RC_REGISTRY_OK);
 assert(make_fresh_source(e)&&uploads==N+1&&snapshot_attempts==N+1);
 read_position=(RcRegistryPosition){0};assert(make_fresh_source(e)&&uploads==N+1);
 // Foreign bytes remain unreadable until an actual visibility proof, then need one refresh in either mode.
 assert(rc_registry_write_begin(registry,range,RC_REGISTRY_BUFFER,&future)==RC_REGISTRY_OK);
 assert(rc_registry_write_submit(registry,future,61)==RC_REGISTRY_OK);
 assert(rc_registry_write_complete(registry,future,0)==RC_REGISTRY_OK);
 RcRegistryVisibility visible;assert(rc_registry_visibility_begin(registry,RC_REGISTRY_VISIBILITY_COMPLETED,&visible)==RC_REGISTRY_OK);
 assert(!make_fresh_source(e)&&uploads==N+1);
 assert(rc_registry_visibility_apply(registry,&visible,range)==RC_REGISTRY_OK);
 assert(make_fresh_source(e)&&uploads==N+2);
 // Ordinary CPU writes seen by memory tracking still invalidate without a registry notification.
 host_dirty=1;read_position=(RcRegistryPosition){61,6};assert(make_fresh_source(e)&&uploads==N+3);host_dirty=0;
 assert(make_fresh_source(e)&&uploads==N+3);
 // Mapping/address reincarnation obtains a different record even at the same virtual range.
 before=e->source_access._content;source_registration_release(e);e->t.serial++;
 assert(make_fresh_source(e)&&uploads==N+4);
 assert(!texture_authority_same(&before,&e->source_access._content));
 assert(rc_registry_handoff_through(registry,(RcRegistryPosition){61,8})==RC_REGISTRY_OK);
 read_position=(RcRegistryPosition){62,2};assert(rc_registry_declare_through(registry,(RcRegistryPosition){62,4})==RC_REGISTRY_OK);
 assert(make_fresh_source(e)&&uploads==N+4); // Closed old read is provenance, never current permission.
 same_operation_write=1;assert(!make_fresh_source(e)&&uploads==N+4);same_operation_write=0;
 rc_registry_fail_closed(registry);assert(!make_fresh_source(e)&&uploads==N+4);
 puts("source continuity: future/canceled writers, completed/foreign visibility, CPU writes, identity/range/guard, reincarnation and closed reads passed");
#endif
 for(unsigned i=0;i<N;i++)source_registration_release(&entries[i]);
 free(entries);rc_registry_destroy(registry);assert(!tex_deferred);
 source_shaped_effects();
}
'''


def upload_support(refusal):
    upload = function(SOURCE, "upload")
    upload = upload[upload.index("    TextureSourceObservation observation = {0};"):upload.index("    uint8_t material_digest[32];")]
    prepare = function(SOURCE, "source_upload_retain_current") + "\n" + function(SOURCE, "source_upload_prepare")
    return SUPPORT.replace(" /* refusal branch */", refusal).replace(" /* actual upload observation */", upload).replace(
        "/* production upload source preparation */", prepare)


def compiler_sources(path, cc):
    return ["-Wno-missing-field-initializers", "-I" + str(ROOT / "runtime"),
        "-I" + str(cc.parent.parent / "include/vma"), str(path), str(ROOT / "runtime/tex_deswizzle.c"),
        str(ROOT / "runtime/nvn_tex.c")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-source", type=Path, help="Archived source ZIP supplying the prior equality helper")
    args = parser.parse_args()
    prefix = PREFIX
    if args.negative_source:
        with zipfile.ZipFile(args.negative_source) as archive:
            name = next(n for n in archive.namelist() if n.endswith("runtime/resource_texture_source.h"))
            old = function(archive.read(name).decode(), "texture_authority_same")
        prefix = prefix.replace('#include "resource_texture_source.h"',
            '#define texture_authority_same current_content_comparison\n#include "resource_texture_source.h"\n'
            '#undef texture_authority_same\n#define OLD_MODE_COMPARISON\n' + old)
    refusal = CONTENT[CONTENT.index("    TextureSourceStamp empty = {0};"):]
    refusal = refusal[:refusal.index("    uint8_t *copy = malloc((size_t)size);")]
    # No allocation succeeds in this fixture, so the pre-copy authority local is deliberately not consumed.
    refusal += "    (void)captured_authority;\n"
    support = upload_support(refusal)
    funcs = "\n".join(function(SOURCE, n) for n in SOURCE_FUNCTIONS)
    code = prefix + funcs + support + function(SOURCE, "make_fresh_source") + TESTS
    cc = Path(os.environ.get("CC", "C:/Users/Andrew/AppData/Local/Programs/2NX2Navi/toolchain/msys64/ucrt64/bin/clang.exe"))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    work = ROOT / "build/test"
    work.mkdir(parents=True, exist_ok=True)
    work = work.resolve()
    work.relative_to(ROOT.resolve())
    with tempfile.TemporaryDirectory(prefix="source-continuity-", dir=work) as temporary:
        path = Path(temporary) / "check.c"
        path.write_text(code)
        exe = path.with_suffix(".exe")
        subprocess.run([str(cc), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                        "-fsanitize-trap=all", *compiler_sources(path, cc), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()
