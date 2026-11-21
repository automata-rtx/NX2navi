from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent
source = Path(sys.argv[1] if len(sys.argv) > 1 else root.parent / "runtime/vk_exec_draw.c").read_text(encoding="utf-8")
baseline = "--baseline" in sys.argv[2:]

def function(name):
    begin = re.search(r"^"+re.escape(name)+r"[^;{}]*\{", source, re.M).start()
    return source[begin:source.index("\n}", begin)+2]

parts = [
    function("static int ssbo_slot("), function("static void nvn2_cs_tables("),
    function("static uint32_t nvn2_cs_tables_at("), function("static int push_ptr("),
]
if not baseline:
    parts.append(function("static int ssbo_bound_recipe("))
parts.append(function("static const p_bind_buf *ssbo_direct_checked("))
if baseline:
    parts.append(function("static const p_bind_buf *effect_storage_fixed("))
else:
    fixture = (root / "ordinary_storage_binding_fixture.c").read_text(encoding="utf-8")
    parts.append(fixture.replace("/* production binding and issued functions */",
        "\n".join((function("static OrdinaryStorageInputs *ordinary_storage_effect_inputs("),
                   function("static const OrdinaryStorageInput *ordinary_storage_input_find("),
                   function("static int ordinary_storage_inputs_current("),
                   function("static int ordinary_storage_write_record("),
                   function("static int ordinary_storage_writes_record("),
                   function("static int effect_ordinary_descriptor("),
                   function("static void ordinary_storage_inputs_issued("),
                   function("static void ordinary_storage_writers_issued(")))))
    effects = function("static void effect_ordinary_storage(")
    assert "effect_storage_capture_span(" in effects and "effect_storage_move(" in effects
    native = function("static void *vk_exec_effects_declare_dispatch(") if "static void *vk_exec_effects_declare_dispatch(" in source else function("void *vk_exec_effects_declare_dispatch(")
    assert "ordinary_storage_info = receipt->ordinary_storage_info" in native
    assert "const OrdinaryStorageInput *input = ordinary_storage_input_find(ordinary_storage_effect_inputs(d), binding)" in effects
    assert "selected = input->storage" in effects
    assert "ordinary_storage_writes_prepare" in function("static VkExecGraphicsResult graphics_buffers_prepare(")
    assert "ordinary_storage_writes_prepare" in function("static VkExecGraphicsResult dispatch_buffers_prepare(")
    for prefix, command in (
        ("static void dispatch_direct_emit(", "vkCmdDispatch("),
        ("static void dispatch_indirect_emit(", "vkCmdDispatchIndirect("),
    ):
        path = function(prefix)
        assert path.index("ordinary_storage_writes_record(") < path.index(command)
    for prefix in ("static int graphics_draw_activate(", "static int prepare_draw_phase("):
        path = function(prefix)
        assert path.index("ordinary_storage_writes_record(") < path.index("graphics_record_state(")
    prepare = function("static VkExecGraphicsResult ordinary_storage_writes_prepare(")
    assert "vk_exec_prepare_alias_readback(" in prepare
    assert prepare.index("ordinary_storage_seed_destinations_disjoint(") < prepare.index("vk_exec_prepare_alias_readback(")
    assert "vk_texture_alias_buffer_write_input_record(" not in prepare
    assert "ordinary_storage_info = graphics_current->use.info" in function("void *vk_exec_effects_declare(")
    assert "ordinary_storage_info = dispatch->selection->ordinary_storage_info" in function("static int dispatch_compat_inputs_transfer(")
    commit = function("static void storage_images_written(")
    assert commit.index("ordinary_storage_writers_issued(effect_current)") < commit.index("commit_recorded_writes(ex)")
    assert "retained ? vk_mem_span_buffer_location(retained, span, &native_access)" in function("static int note_buffer_written_provider(")
    assert "buffer_preserve_selected(output, span, addr, mapped, count)" in function("static int note_buffer_written_provider(")
common = r"""
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define NVN_STAGE_CS 5
typedef struct { int a,b; uint64_t addr,size; } p_bind_buf;
typedef struct { int cbuf_slot, handle, ptr, count; } GfxShaderRes;
typedef struct { uint64_t serial; } Prog;
typedef struct { uint64_t cb; void *dc; Prog *prog; p_bind_buf ssbo[8][16],ubo[8][18]; uint8_t cs_push[1024]; } VkExec;
static uint8_t control[0x100];
static void rt_log(const char *text, ...) { (void)text; }
static int c0_ssbo_base(int stage) { return stage==5 ? 0x310 : stage==1 ? 0x510 : 0x110; }
static int find_stage(void *dc, void *prog, int stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    (void)dc; (void)prog; (void)stage; (void)code; (void)len; *ctl=control; return 1;
}
static int ctl_nvn2(const uint8_t *ctl) { return ctl==control; }
static int ubo_ptr_res(const GfxShaderRes *r) { return r->cbuf_slot>=3; }
static int ubo_ptr(const VkExec *ex, int st, const GfxShaderRes *r, uint64_t *out) {
    (void)ex; (void)st; (void)r; *out=0; return 1;
}
"""
main = r"""
#define CHECK(x) do { if (!(x)) { puts("failed: " #x); return 1; } } while (0)
int main(void) {
    const uint32_t fields[]={0x040a0000,0x00010000,0x01580000};
    memcpy(control+0xb0, fields, sizeof fields);
    Prog program = {33};
    VkExec ex = {.prog=&program};
    ex.ssbo[5][8] = (p_bind_buf){.addr=0x80017be600, .size=0x154230};
    ex.ssbo[5][9] = (p_bind_buf){.addr=0x8003256000, .size=32};
    p_bind_buf temporary;
    int ubo=0, refused=0;
    for (int i=0; i<2; ++i) {
        const GfxShaderRes r={.cbuf_slot=0,.handle=68+i*4,.count=1};
        const p_bind_buf *final=ssbo_direct_checked(&ex, 5, &r, &ubo, &temporary, &refused);
        CHECK(final && final->addr==ex.ssbo[5][8+i].addr && final->size==ex.ssbo[5][8+i].size && !ubo && !refused);
#ifndef BASELINE
        p_bind_buf original;
        int indirect_ubo=0;
        CHECK(ssbo_bound_recipe(&ex,5,&r,&original,&indirect_ubo));
        CHECK(!indirect_ubo && original.addr==final->addr && original.size==final->size);
#else
        const p_bind_buf *original=effect_storage_fixed(&ex,5,&r);
        CHECK(original && original->addr==final->addr && original->size==final->size);
#endif
    }
#ifndef BASELINE
    p_bind_buf original;
    const GfxShaderRes empty={.handle=64,.count=1};
    CHECK(ssbo_bound_recipe(&ex,5,&empty,&original,&ubo) && !original.addr && !original.size);
    const GfxShaderRes pushed={.handle=4,.count=1};
    memcpy(ex.cs_push, &ex.ssbo[5][8].addr, 8);
    CHECK(ssbo_bound_recipe(&ex,5,&pushed,&original,&ubo) && !ubo && original.addr==ex.ssbo[5][8].addr && !original.size);
    const GfxShaderRes crosses_table={.handle=(0x90-4)/4,.count=1};
    CHECK(!ssbo_bound_recipe(&ex,5,&crosses_table,&original,&ubo));
    ex.ssbo[1][1]=ex.ssbo[5][9];
    const GfxShaderRes graphics={.handle=0x520/4,.count=1};
    CHECK(ssbo_bound_recipe(&ex,1,&graphics,&original,&ubo) && original.addr==ex.ssbo[1][1].addr);
    const GfxShaderRes generic_ubo={.handle=0x210/4,.count=1};
    CHECK(ssbo_bound_recipe(&ex,5,&generic_ubo,&original,&ubo) && ubo);
    if (run_provider() || run_seed_cohort(&ex)) return 1;
#endif
    puts("original generic and NVN2 SSBO slot/extent and actual bind agree");
}
"""
compiler = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ["PATH"])
with tempfile.TemporaryDirectory(prefix="ordinary-storage-binding-") as temp:
    target = Path(temp) / "check.c"
    target.write_text(common + "\n".join(parts) + main, encoding="utf-8")
    exe = target.with_suffix(".exe")
    subprocess.run([str(compiler), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-fsanitize-trap=all", *(["-DBASELINE"] if baseline else []), str(target), "-o", str(exe)],
                   env=env, check=True)
    sys.exit(subprocess.run([str(exe)], check=False, env=env).returncode)
