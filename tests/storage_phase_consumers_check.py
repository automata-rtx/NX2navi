"""Test selected storage phases at their production draw and mesh recording boundaries."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(os.environ.get("NX2_STORAGE_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
DRAW = Path(os.environ.get("NX2_STORAGE_DRAW", ROOT / "runtime/vk_exec_draw.c"))


def extract(source, name):
    match = re.search(r"^(?:static )?[^\n]*\b" + name + r"\([^;{]*\)\s*\{", source, re.M)
    if not match:
        raise ValueError(name)
    return source[match.start():source.index("\n}", match.end()) + 2]


def main():
    source = DRAW.read_text()
    for name, emission in (
        ("op_draw", "vkCmdDraw("), ("op_draw_idx", "vkCmdDrawIndexed("),
        ("op_draw_idx_ind", "vkCmdDrawIndexedIndirect("), ("op_draw_mesh", "rec_draw_mesh("),
        ("op_draw_mesh_ind", "rec_draw_mesh_ind("), ("mfb_draw", "vkCmdDrawIndirect("),
    ):
        body = extract(source, name)
        assert body.index("graphics_storage_begin(") < body.index(emission)
        assert body.index(emission) < body.index("graphics_storage_recorded(") < body.index("storage_images_written(")

    binding = extract(source, "graphics_record_state")
    assert binding.index("vk_exec_storage_phase_enter(") < binding.index("bind_resources(")
    assert binding.index("bind_resources(") < binding.index("vk_exec_storage_phase_leave(")
    declare = extract(source, "vk_exec_effects_declare")
    assert declare.index("vk_exec_storage_phase_declare(") < declare.index("effect_stage(")
    assert "vk_exec_storage_phase_cancel(&draw->storage)" in extract(source, "vk_exec_graphics_draw_release_gpa_inputs")
    dependencies = extract(source, "mfb_dependencies_ready")
    assert dependencies.index("mfb_scratch_cohort_ready(") < dependencies.index("mfb_storage_phase_ready(")
    assert dependencies.index("mfb_storage_phase_ready(") < dependencies.index("graphics_phase_descriptors_ready(")
    assert all(part in dependencies for part in ("&mfb->task.descriptors", "&mfb->mesh.descriptors",
                                                 "graphics_descriptors_ready(ex, draw, 1)"))
    late_final = extract(source, "mfb_final_late")
    assert late_final.index("graphics_pipeline_ready(") < late_final.index("graphics_descriptors_ready(")
    final = extract(source, "mfb_draw")
    assert final.index("draw_descriptor_grant_current(ex, draw->use.info") < final.index("graphics_record_state(")
    plan = extract(source, "mfb_storage_phase_ready")
    assert "mfb->final_mode != MFB_FINAL_EXACT && !mfb->final_disabled ? 2 : 3" in plan
    assert "stage_override = NVN_STAGE_TASK" in plan and "stage_override = NVN_STAGE_MESH" in plan

    fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "vk_exec.h"

#undef vkCmdDispatchIndirect

enum { MFB_FINAL_EXACT, MFB_FINAL_PASS_COMPAT, NVN_STAGE_TASK = 6, NVN_STAGE_MESH = 7 };
typedef struct { int final_mode, final_disabled; } MfbDraw;
typedef struct {
    const GfxShader *shader;
    VkPipeline pipeline;
    const VkPipeInfo *info;
    VkExecGraphicsScratch *scratch;
    DrawDescriptorGrant descriptors;
} MfbPhaseUse;
struct VkExecGraphicsDraw { MfbDraw *fallback; VkExecStoragePhase *storage; uint32_t opcode; };
struct VkExecStoragePhase { unsigned issued; };
struct DrawCtx { int mfb_stage; };

static unsigned events[40], nevents;
enum { ENTER = 1, BIND, LEAVE, BEGIN, EMIT, RECORDED, LEGACY, END };
static unsigned selected_phase, checked_pool, rejected_enter, rejected_begin, failed_binding, unavailable;
static int scoped;
static VkExecStoragePhase *selected_storage;
static VkPassMark marker;
static VkExecGraphicsDraw *selected_draw;
static VkExecGraphicsScratch *selected_scratch;
static VkExecPoolBindScope selected_pool;
static const VkPipeInfo *selected_info;
static DrawDescriptorGrant *descriptor_binding_grant;
static struct DrawDescriptorCell *consumed[4];
static uint32_t nconsumed;
static int draw_descriptor_grant_current(const VkExec *ex, const VkPipeInfo *info, const DrawDescriptorGrant *grant,
                                         uint64_t ordinal) {
    assert(ex->fence == 5);
    return grant->info == info && grant->ordinal == ordinal && grant->set[0];
}

static void note(unsigned event) { assert(nevents < sizeof events / sizeof *events); events[nevents++] = event; }
void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t opcode, const char *why) {
    assert(ex && (opcode == OP_DRAW_MESH || opcode == OP_DRAW) && why);
    unavailable++;
}
int vk_exec_storage_phase_enter(VkExec *ex, VkExecStoragePhase *storage, uint32_t phase,
                                const VkPipeInfo *info, VkExecStorageScope *scope) {
    assert(storage && info && ex->dc->mfb_stage && scope && !scoped);
    note(ENTER);
    if (rejected_enter) return 0;
    selected_storage = storage;
    selected_phase = phase;
    selected_info = info;
    scoped = 1;
    return 1;
}
void vk_exec_storage_phase_leave(VkExecStorageScope *scope) {
    assert(scope && scoped);
    scoped = 0;
    note(LEAVE);
}
int vk_exec_storage_phase_begin(VkExec *ex, VkExecStoragePhase *storage, uint32_t phase) {
    assert(ex && storage && !scoped && !(storage->issued & (1u << phase)));
    selected_storage = storage;
    selected_phase = phase;
    note(BEGIN);
    return !rejected_begin;
}
void vk_exec_storage_phase_recorded(VkExec *ex, VkExecStoragePhase *storage, uint32_t phase) {
    assert(ex && storage && storage == selected_storage && phase == selected_phase && !scoped);
    storage->issued |= 1u << phase;
    note(RECORDED);
}
static int flat3d_preflight(VkExec *ex, const GfxShader **shader, const int *stages, int count) {
    assert(ex && *shader && (*stages == NVN_STAGE_TASK || *stages == NVN_STAGE_MESH) && count == 1);
    return 1;
}
int vk_exec_pool_grant_current(const VkExec *ex, const VkExecPoolGrant *grant) {
    assert(ex && grant);
    checked_pool++;
    return 1;
}
static void bind_pipeline(VkExec *ex, VkPipelineBindPoint at, VkPipeline pipeline) {
    assert(ex && at == VK_PIPELINE_BIND_POINT_COMPUTE && pipeline);
}
VkExecGraphicsDraw *vk_exec_graphics_draw_enter(VkExecGraphicsDraw *draw) {
    VkExecGraphicsDraw *prior = selected_draw;
    selected_draw = draw;
    return prior;
}
VkExecGraphicsScratch *vk_exec_draw_graphics_scratch_enter(VkExecGraphicsScratch *scratch) {
    VkExecGraphicsScratch *prior = selected_scratch;
    selected_scratch = scratch;
    return prior;
}
VkExecPoolBindScope vk_exec_pool_bind_enter(VkExecPoolBindScope scope) {
    VkExecPoolBindScope prior = selected_pool;
    selected_pool = scope;
    return prior;
}
static void bind_resources(VkExec *ex, const VkPipeInfo *info) {
    assert(ex->dc->mfb_stage && scoped && selected_info == info && !selected_draw);
    assert(descriptor_binding_grant && descriptor_binding_grant->info == info &&
           descriptor_binding_grant->ordinal == ex->resource_op && descriptor_binding_grant->set[0]);
    note(BIND);
    if (failed_binding) return;
    for (uint32_t i = 0; i < nconsumed; i++) assert(consumed[i] != descriptor_binding_grant->set[0]);
    assert(nconsumed < sizeof consumed / sizeof *consumed);
    consumed[nconsumed++] = descriptor_binding_grant->set[0];
    descriptor_binding_grant->set[0] = NULL;
}
int vk_texture_deferred(void) { return 0; }
int vk_exec_recording_failed(void) { return failed_binding; }
static VkPassMark *dispatch_mark(VkExec *ex, const GfxShader *shader, uint32_t x, uint32_t y, uint32_t z) {
    assert(ex && shader && !x && !y && !z);
    return &marker;
}
void vk_pass_mark_end(VkCommandBuffer cb, const VkPassMark *mark) {
    assert(cb && mark == &marker);
    note(END);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDispatchIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset) {
    assert(cb && buffer && offset % 16 == 0 && !scoped);
    note(EMIT);
}
static void storage_images_written(VkExec *ex) { assert(ex); note(LEGACY); }
'''
    fixture += "\n".join(extract(source, name) for name in
                         ("graphics_draw_storage", "graphics_storage_begin", "graphics_storage_recorded", "mfb_dispatch"))
    fixture += r'''
int main(void) {
    struct DrawCtx dc = {0};
    VkExec ex = {.dc = &dc, .cb = (VkCommandBuffer)(uintptr_t)1, .fence = 5, .resource_op = 12};
    VkExecStoragePhase owner = {0};
    VkExecGraphicsDraw draw = {.storage = &owner, .opcode = OP_DRAW_MESH};
    VkPipeInfo phase_info = {.pool = 1};
    MfbPhaseUse task = {.info = &phase_info, .pipeline = (VkPipeline)(uintptr_t)4,
        .shader = (const GfxShader *)(uintptr_t)3, .scratch = (VkExecGraphicsScratch *)(uintptr_t)2,
        .descriptors = {.info = &phase_info, .ordinal = ex.resource_op, .set = {(struct DrawDescriptorCell *)(uintptr_t)5}}};
    MfbPhaseUse mesh = task;
    mesh.descriptors.set[0] = (struct DrawDescriptorCell *)(uintptr_t)6;
    VkExecPoolGrant *grant = (VkExecPoolGrant *)(uintptr_t)8;
    VkBuffer indirect = (VkBuffer)(uintptr_t)9;
    selected_draw = &draw;
    assert(mfb_dispatch(&ex, OP_DRAW_MESH, &task, grant, &owner, 0, NVN_STAGE_TASK, indirect, 0, 0));
    assert(!memcmp(events, (unsigned[]){ENTER, BIND, LEAVE, BEGIN, EMIT, RECORDED, LEGACY, END}, 8 * sizeof(unsigned)));
    assert(owner.issued == 1 && selected_draw == &draw && !dc.mfb_stage && ex.recorded == 1 &&
           !descriptor_binding_grant && !task.descriptors.set[0] && mesh.descriptors.set[0] && nconsumed == 1);

    // The next actual native part has a different command buffer but the same frame and logical operation.
    ex.cb = (VkCommandBuffer)(uintptr_t)2;
    nevents = 0;
    mesh.descriptors.ordinal++;
    assert(!mfb_dispatch(&ex, OP_DRAW_MESH, &mesh, grant, &owner, 1, NVN_STAGE_MESH, indirect, 16, 1));
    assert(!nevents && nconsumed == 1 && owner.issued == 1);
    mesh.descriptors.ordinal--;

    failed_binding = 1;
    assert(!mfb_dispatch(&ex, OP_DRAW_MESH, &mesh, grant, &owner, 1, NVN_STAGE_MESH, indirect, 16, 1));
    assert(nevents == 3 && events[2] == LEAVE && owner.issued == 1 && checked_pool == 1 && !dc.mfb_stage);
    failed_binding = 0;
    nevents = 0;
    rejected_enter = 1;
    assert(!mfb_dispatch(&ex, OP_DRAW_MESH, &mesh, grant, &owner, 1, NVN_STAGE_MESH, indirect, 16, 1));
    assert(nevents == 1 && events[0] == ENTER && owner.issued == 1 && !dc.mfb_stage);
    rejected_enter = 0;
    rejected_begin = 1;
    nevents = 0;
    assert(!mfb_dispatch(&ex, OP_DRAW_MESH, &mesh, grant, &owner, 1, NVN_STAGE_MESH, indirect, 16, 1));
    assert(nevents == 5 && events[3] == BEGIN && events[4] == END && owner.issued == 1 && nconsumed == 2);
    assert(!mesh.descriptors.set[0] && !descriptor_binding_grant);
    // A distinct trial prepares a new set. A bind recorded before a failed native phase is not reusable.
    mesh.descriptors.set[0] = (struct DrawDescriptorCell *)(uintptr_t)7;
    rejected_begin = 0;
    assert(mfb_dispatch(&ex, OP_DRAW_MESH, &mesh, grant, &owner, 1, NVN_STAGE_MESH, indirect, 16, 1));
    assert(owner.issued == 3 && nconsumed == 3 && !descriptor_binding_grant);

    MfbDraw late = {.final_mode = MFB_FINAL_PASS_COMPAT};
    draw.fallback = &late;
    nevents = 0;
    assert(graphics_storage_begin(&ex, &draw));
    graphics_storage_recorded(&ex, &draw);
    assert(!nevents && owner.issued == 3);
    late.final_mode = MFB_FINAL_EXACT;
    rejected_begin = 1;
    assert(!graphics_storage_begin(&ex, &draw) && unavailable == 1 && owner.issued == 3);
    rejected_begin = 0;
    nevents = 0;
    assert(graphics_storage_begin(&ex, &draw));
    graphics_storage_recorded(&ex, &draw);
    assert(owner.issued == 7 && nevents == 2 && selected_phase == 2);
    return 0;
}
'''
    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Clang is required (set CC)")
    clang = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get("VMA_INCLUDE", clang.parent.parent / "include" / "vma"))
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="storage-phase-consumer-") as temporary:
        folder = Path(temporary)
        test = folder / "check.c"
        executable = folder / ("check.exe" if os.name == "nt" else "check")
        test.write_text(fixture)
        subprocess.run([str(clang), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined",
                        "-I" + str(ROOT / "runtime"), "-I" + str(vma), str(test), "-o", str(executable)],
                       env=env, check=True)
        subprocess.run([str(executable)], env=env, check=True)
    print("Storage phases: scoped binding, exact emission and retained earlier phases passed")


if __name__ == "__main__":
    main()
