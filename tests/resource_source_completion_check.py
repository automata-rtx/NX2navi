"""Run completion between copied source bytes and the production cutoff commit."""
from pathlib import Path
import argparse
import importlib.util
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("source_continuity", ROOT / "tests/resource_source_continuity_check.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

CALLBACK = r'''
static int complete_after_copy;
static void copied(Tex *e) {
 if(!complete_after_copy)return;
 int cpu_write=complete_after_copy==2;uint64_t scene=cpu_write?3:1;
 complete_after_copy=0;
 RcRegistryWrite w;RcSourceRange whole={1,0,UINT64_MAX},gpu={1,UINT64_C(0x8000001000),128};
 if(cpu_write)gpu=(RcSourceRange){1,e->pub.addr,e->t.L.size};
 assert(rc_registry_write_begin_at(registry,whole,RC_REGISTRY_IMAGE,(RcRegistryPosition){scene,3},&w)==RC_REGISTRY_OK);
 assert(rc_registry_declare_through(registry,(RcRegistryPosition){scene,4})==RC_REGISTRY_OK);
 assert(rc_registry_handoff_through(registry,(RcRegistryPosition){scene,3})==RC_REGISTRY_OK);
 assert(rc_registry_write_submit(registry,w,scene)==RC_REGISTRY_OK);
 assert(rc_registry_write_complete_effects(registry,w,&gpu,1,cpu_write?NULL:&gpu,cpu_write?0:1)==RC_REGISTRY_OK);
}
'''
TESTS = r'''
static void split_source_progress(Tex *e) {
 RcSourceRange range={1,e->pub.addr,e->t.L.size};
 RcRegistryWrite prefix,tail;
 const unsigned before=uploads;
 assert(rc_registry_write_begin_at(registry,range,RC_REGISTRY_BUFFER,(RcRegistryPosition){4,3},&prefix)==RC_REGISTRY_OK);
 assert(rc_registry_write_begin_at(registry,range,RC_REGISTRY_BUFFER,(RcRegistryPosition){4,7},&tail)==RC_REGISTRY_OK);
 assert(rc_registry_declare_through(registry,(RcRegistryPosition){4,8})==RC_REGISTRY_OK);
 read_position=(RcRegistryPosition){4,2};
 assert(make_fresh_source(e)&&uploads==before&&!e->source_blocked);
 assert(rc_registry_handoff_through(registry,(RcRegistryPosition){4,3})==RC_REGISTRY_OK);
 assert(rc_registry_write_submit(registry,prefix,4)==RC_REGISTRY_OK);
 assert(!make_fresh_source(e)&&uploads==before&&e->source_blocked); // Closed cut cannot be revived.
 read_position.ordinal=4;
 assert(!make_fresh_source(e)&&uploads==before&&e->source_blocked); // Handed prefix is still pending.
 assert(rc_registry_write_cancel(registry,tail)==RC_REGISTRY_OK);
 assert(!make_fresh_source(e)&&uploads==before&&e->source_blocked);
 assert(rc_registry_write_complete(registry,prefix,1)==RC_REGISTRY_OK);
 assert(make_fresh_source(e)&&uploads==before+1&&!e->source_blocked);
 read_position.ordinal=8;
 assert(make_fresh_source(e)&&uploads==before+1&&!e->source_blocked); // Canceled tail wrote nothing.
 puts("source completion: split, closed cut, handed prefix and known-unhanded tail passed");
}

int main(void) {
 registry=rc_registry_create((RcRegistryConfig){32,8,4});assert(registry);
 TextureSourceSpan span={0,64};Tex e={.pub={4096},.t={7,{64},0},.source_spans=&span,.source_count=1};
 complete_after_copy=1;assert(make_fresh_source(&e));assert(uploads==1);
 TextureSourceAuthority current;
 assert(texture_authority_observe(registry,e.source_access.owner.handle,submit_epoch,(RcRegistryPosition){0},&current)==RC_REGISTRY_OK);
 assert(texture_authority_same(&e.source_authority,&current));
 assert(!texture_authority_current(registry,&e.source_authority,(RcRegistryPosition){0}));
 int retained=texture_source_stamp_valid(&e.source_stamp);
 assert(make_fresh_source(&e));
 printf("source completion: unchanged content, cutoff retained=%d, uploads=%u, prior supplied=%d\n",retained,uploads,passed_prior);
#ifdef EXPECT_CUTOFF_LOSS
 assert(!retained&&uploads==2&&!passed_prior);
#else
 assert(retained&&uploads==1&&passed_prior);
#endif
 unsigned before=uploads;
 // Keeping byte provenance cannot authorize a read while a new producer remains pending.
 RcRegistryWrite pending;RcSourceRange own={1,e.pub.addr,e.t.L.size};
 assert(rc_registry_write_begin_at(registry,own,RC_REGISTRY_BUFFER,(RcRegistryPosition){2,3},&pending)==RC_REGISTRY_OK);
 assert(!make_fresh_source(&e)&&e.source_blocked&&uploads==before);
 assert(rc_registry_write_cancel(registry,pending)==RC_REGISTRY_OK);
 assert(make_fresh_source(&e)&&uploads==before);
 // Actual changed bytes still require a fresh upload even when memory's write-watch cutoff is unchanged.
 assert(rc_registry_cpu_changed(registry,own)==RC_REGISTRY_OK);
 assert(make_fresh_source(&e)&&uploads==before+1);
 assert(make_fresh_source(&e)&&uploads==before+1);
 // A completion that actually changes these bytes between copy and commit must erase the old cutoff.
 Tex changed={.pub={8192},.t={8,{64},0},.source_spans=&span,.source_count=1};
 complete_after_copy=2;before=uploads;assert(make_fresh_source(&changed)&&uploads==before+1);
 assert(!texture_source_stamp_valid(&changed.source_stamp));
 assert(make_fresh_source(&changed)&&uploads==before+2);
 assert(snapshot_attempts==uploads&&advisories==uploads&&n_checks>=uploads);
 assert(qualifiers>=2);
 split_source_progress(&changed);
 source_registration_release(&changed);source_registration_release(&e);rc_registry_destroy(registry);assert(!tex_deferred);
 puts("source completion: pending permission and true content-change guards passed");
}
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-cutoff-loss", action="store_true", help="Reproduce the pre-fix production failure")
    args = parser.parse_args()
    refusal = base.CONTENT[base.CONTENT.index("    TextureSourceStamp empty = {0};"):]
    refusal = refusal[:refusal.index("    uint8_t *copy = malloc((size_t)size);")] + "    (void)captured_authority;\n"
    support = base.upload_support(refusal)
    support = support.replace("uploads++;e->uploaded=1;source_commit", "uploads++;e->uploaded=1;copied(e);source_commit")
    support = support.replace(" assert(texture_source_stamp_valid(&e->source_stamp));", "")
    funcs = "\n".join(base.function(base.SOURCE, n) for n in base.SOURCE_FUNCTIONS)
    code = base.PREFIX + funcs + CALLBACK + support + base.function(base.SOURCE, "make_fresh_source") + TESTS
    cc = Path(os.environ.get("CC", "C:/Users/Andrew/AppData/Local/Programs/2NX2Navi/toolchain/msys64/ucrt64/bin/clang.exe"))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    work = (ROOT / "build/test").resolve(); work.mkdir(parents=True, exist_ok=True); work.relative_to(ROOT.resolve())
    with tempfile.TemporaryDirectory(prefix="source-completion-", dir=work) as tmp:
        path = Path(tmp) / "check.c"; path.write_text(code); exe = path.with_suffix(".exe")
        flags = ["-DEXPECT_CUTOFF_LOSS"] if args.expect_cutoff_loss else []
        subprocess.run([str(cc), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
            "-fsanitize-trap=all", *flags, *base.compiler_sources(path, cc), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)

if __name__ == "__main__":
    main()
