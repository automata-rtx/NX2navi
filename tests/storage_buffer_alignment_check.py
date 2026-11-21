"""Exercise the production buffer binding against logical and native SSBO alignments."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(os.environ.get("NX2_SCRATCH_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
DRAW = Path(os.environ.get("NX2_SCRATCH_DRAW", ROOT / "runtime/vk_exec_draw.c"))


def extract(source, name):
    start = re.search(r"^static [^\n]*\b" + name + r"\([^;{]*\)\s*\{", source, re.M)
    if not start:
        raise ValueError(name)
    return source[start.start():source.index("\n}", start.end()) + 2]


def main():
    source = DRAW.read_text()
    limits = next(line for line in source.splitlines() if "ubo_align, sbo_max, sbo_align" in line)
    uniform_range = re.search(r"typedef struct \{[^{}]+\} UniformAddressRange;", source)
    if not uniform_range:
        raise ValueError("UniformAddressRange")
    bindings = "\n".join([limits, uniform_range.group()] + [extract(source, name) for name in (
        "limits_init", "bind_range", "storage_uniform_address_range", "readonly_storage_pointer", "ssbo_info")])
    fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gfx.h"
#include "host.h"
#include "mesh_fb.h"
#include "vk_exec.h"
#include "vk_gpa.h"
#include "vk_mesh_fb.h"

#undef VK_MEM_USE
#undef VK_MEM_BOUND

typedef struct DrawCtx { int pb_spoiled, mo_ovr_on; } DrawCtx;
typedef struct { VkGpaWriteScope *gpa_writes; } CbRes;
static struct { VkBuffer sink, zbuf; } Z = {(VkBuffer)(uintptr_t)3, (VkBuffer)(uintptr_t)4};
enum { ZERO_BYTES = 4096, NVN_STAGE_CS = 5, W_SSBO_ALIGN, W_SSBO_SLOT };
VkCaps vk_caps;
static VkCtx context;
static VkMemSpan selected;
static p_bind_buf guest_binding;
static uint64_t expected_address;
static int stopped;
static uint8_t backing[24576];
static unsigned native_queries, uses, bounds, writes, mirrors, drops, copies, stops;
static uint8_t copied[256];
#define VK_MEM_USE(span) do { assert((span).buffer == selected.buffer); uses++; } while (0)
#define VK_MEM_BOUND(span) do { assert((span).buffer == selected.buffer); bounds++; } while (0)

VkCtx *vk_ctx(void) { return &context; }
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(VkPhysicalDevice physical,
                                                         VkPhysicalDeviceProperties *properties) {
    assert(physical == context.phys);
    native_queries++;
    *properties = (VkPhysicalDeviceProperties){0};
    properties->limits.minStorageBufferOffsetAlignment = 256;
}
int vk_mem_resolve(uint64_t address, uint64_t size, VkMemSpan *out) {
    assert(address == expected_address && size == 1);
    *out = selected;
    return 1;
}
static void count_use(DrawCtx *draw, int cls, int use, uint64_t bytes) {
    assert(draw && cls == selected.cls && use == VK_MUSE_SSBO && bytes);
}
static int note_written(VkExec *ex, uint64_t addr, uint64_t size, const VkMemSpan *sp) {
    assert(ex && addr && size && sp->offset == selected.offset && sp->buffer == selected.buffer);
    writes++;
    return 1;
}
void vk_mem_gpu_wrote_shader(const VkMemSpan *sp, uint64_t size) {
    assert(sp->buffer == selected.buffer && size && writes == mirrors + 1);
    mirrors++;
}
void vk_usnap_drop(uint64_t addr, uint64_t size) {
    assert(addr && size && mirrors == drops + 1);
    drops++;
}
void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t opcode, const char *description) {
    assert(ex && !opcode && description);
    stopped = 1;
    stops++;
}
void vk_mem_count(int cls, int use, uint64_t bytes) {
    assert(bytes && (cls == VK_MCLS_SCRATCH ? use == VK_MUSE_SSBO : use == VK_MUSE_CPU_READ));
}
typedef enum { SCRATCH_OPTIONAL, SCRATCH_REQUIRED, SCRATCH_REQUIRED_PRIOR } ScratchUse;
static uint8_t *scratch(VkExec *ex, uint64_t size, VkBuffer *buf, VkDeviceSize *offset, ScratchUse use) {
    assert(use == SCRATCH_REQUIRED);
    assert(ex && size <= sizeof copied);
    copies++;
    *buf = (VkBuffer)(uintptr_t)2;
    *offset = 0;
    return copied;
}
int vk_exec_recording_failed(void) { return stopped; }
static int pool_index_bind_input(VkExec *ex, int st, int slot, const GfxShaderRes *r, VkDescriptorBufferInfo *out) {
    (void)ex; (void)st; (void)slot; (void)r; (void)out;
    return 0;
}
#define mfb_task_buffer_descriptor(...) 0
#define mfb_device_pointer_descriptor(...) 0
#define effect_ordinary_input(...) 0
static const struct { void *selected_writers; uint32_t opcode; } *effect_current;
#define selected_storage_writer(...) 0
typedef struct { uint64_t value; } DeviceStoragePointer;
#define device_storage_pointer(...) ((const DeviceStoragePointer *)0)
static int effect_storage_descriptor(VkExec *ex, int st, const GfxShaderRes *r, int written,
                                     VkDescriptorBufferInfo *out) {
    (void)ex; (void)st; (void)r; (void)written; (void)out;
    return 0;
}
static const p_bind_buf *ssbo_direct_checked(const VkExec *ex, int st, const GfxShaderRes *r, int *ubo,
                                            p_bind_buf *temporary, int *refused) {
    (void)temporary;
    assert(ex && st == NVN_STAGE_CS && !r->ptr && !*ubo && !*refused);
    return &guest_binding;
}
static VkDescriptorBufferInfo zero_info(uint64_t max) {
    return (VkDescriptorBufferInfo){Z.zbuf, 0, max};
}
#define UNEXPECTED(...) (abort(), 0)
#define vk_mem_shadow_table_read() abort()
#define vk_gpa_info() (abort(), (VkDescriptorBufferInfo){0})
#define effect_gpa_prepare(...) UNEXPECTED(__VA_ARGS__)
#define cb_res(...) (abort(), (CbRes *)NULL)
#define vk_gpa_write_begin() (abort(), (VkGpaWriteScope *)NULL)
#define vk_meshfb_buffer(...) (abort(), (VkDescriptorBufferInfo){0})
#define ptr_addr_checked(...) UNEXPECTED(__VA_ARGS__)
#define raw_storage_requires_image(...) UNEXPECTED(__VA_ARGS__)
#define uniform_input_refused(...) abort()
#define uniform_buffer_descriptor(ex, binding, st, slot, address, bytes, alignment, ...) ((void)(address), (void)(bytes), (void)(alignment), abort(), 0)
#define warn_once(...) abort()
#define find_stage(...) UNEXPECTED(__VA_ARGS__)
#define rt_log(...) abort()
''' + bindings + r'''
static void ordered_reads(const VkDescriptorBufferInfo *writer, const VkDescriptorBufferInfo *reader,
                          const VkMemSpan *transfer, int emit) {
    assert(writer->buffer == reader->buffer && writer->buffer == transfer->buffer);
    const uint64_t at = writer->offset + 8195;
    assert(at == reader->offset + 3 && at == transfer->offset);
    const uint8_t bytes[] = {0x31, 0x72, 0xb4};
    if (emit) memcpy(backing + at, bytes, sizeof bytes);
    uint8_t copied_bytes[sizeof bytes];
    memcpy(copied_bytes, backing + transfer->offset, sizeof copied_bytes);
    for (unsigned i = 0; i < sizeof bytes; ++i) {
        assert(backing[reader->offset + 3 + i] == (emit ? bytes[i] : 0xd5));
        assert(copied_bytes[i] == backing[reader->offset + 3 + i]);
    }
    for (uint64_t i = 0; i < sizeof backing; ++i)
        if (i < at || i >= at + sizeof bytes) assert(backing[i] == 0xd5);
}

int main(void) {
    DrawCtx draw = {0};
    VkExec ex = {.dc = &draw, .fence = 1};
    GfxShaderRes resource = {0};
    const VkPipeBinding binding = {.res = &resource};
    uint8_t input[32];
    memset(input, 0xd5, sizeof input);
    context.props.limits.minStorageBufferOffsetAlignment = 8192;
    context.props.limits.minUniformBufferOffsetAlignment = 1024;
    context.props.limits.maxUniformBufferRange = 65536;
    context.props.limits.maxStorageBufferRange = 16384;
    vk_caps.sbuf_align = 8192;

    for (unsigned mode = 0; mode < 2; ++mode) {
        memset(backing, 0xd5, sizeof backing);
        VmaAllocation allocation = mode ? (VmaAllocation)(uintptr_t)1 : NULL;
        selected = (VkMemSpan){.buffer = (VkBuffer)(uintptr_t)1, .offset = 4096,
                               .avail = 16384, .alloc = allocation};
        guest_binding = (p_bind_buf){.addr = 8195, .size = 12285};
        expected_address = 8192;
        VkDescriptorBufferInfo writer = ssbo_info(&ex, NVN_STAGE_CS, &binding, 1);
        assert(native_queries == 1 && LIM.sbo_align == 256 && LIM.ubo_align == 1024 &&
               vk_caps.sbuf_align == 8192 && context.props.limits.minStorageBufferOffsetAlignment == 8192);
        assert(writer.buffer == selected.buffer && writer.offset == 4096 && writer.range == 12288);
        assert(writes == mode + 1 && mirrors == writes && drops == writes && !copies && !stops);

        guest_binding = (p_bind_buf){.addr = 16384, .size = 8192};
        expected_address = 16384;
        selected.offset = 12288;
        VkDescriptorBufferInfo reader = ssbo_info(&ex, NVN_STAGE_CS, &binding, 0);
        assert(reader.buffer == writer.buffer && reader.offset == 12288 && reader.range == 8192);
        assert(reader.offset + reader.range > writer.offset + writer.range &&
               reader.offset < writer.offset + writer.range);
        assert(uses == 2 * mode + 2 && bounds == uses && !copies && !stops);
        // The scripted command batch discards the first recording and executes the second.
        // Both native consumers resolve the writer's exact buffer; neither reads a scratch copy.
        selected.offset += 3;
        expected_address += 3;
        VkMemSpan transfer;
        assert(vk_mem_resolve(expected_address, 1, &transfer));
        ordered_reads(&writer, &reader, &transfer, mode);
    }

    guest_binding = (p_bind_buf){.addr = 8195, .size = 29};
    expected_address = 8192;
    selected.offset = 4097;
    for (unsigned mode = 0; mode < 2; ++mode) {
        stopped = 0;
        selected.alloc = mode ? (VmaAllocation)(uintptr_t)1 : NULL;
        VkDescriptorBufferInfo refused = ssbo_info(&ex, NVN_STAGE_CS, &binding, 1);
        assert(stopped && stops == mode + 1 && !refused.buffer && !refused.range &&
               !copies && writes == 2 && uses == 4 && bounds == 4);
    }
    selected.alloc = NULL;
    expected_address = (uintptr_t)input;
    VkDescriptorBufferInfo read_copy;
    assert(bind_range(&ex, expected_address, sizeof input, LIM.sbo_align, LIM.sbo_max, VK_MUSE_SSBO, 0, 1, &read_copy) > 0);
    assert(copies == 1 && draw.pb_spoiled && read_copy.buffer != selected.buffer && read_copy.range == sizeof input);
    assert(!memcmp(copied, input, sizeof input) && stops == 2 && writes == 2);
    return 0;
}
'''
    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Clang is required (set CC)")
    clang = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get("VMA_INCLUDE", clang.parent.parent / "include" / "vma"))
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="storage-buffer-alignment-") as temporary:
        folder = Path(temporary)
        test = folder / "check.c"
        executable = folder / ("check.exe" if os.name == "nt" else "check")
        test.write_text(fixture)
        subprocess.run([str(clang), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined",
                        "-I" + str(ROOT / "runtime"), "-I" + str(vma), str(test), "-o", str(executable)],
                       env=env, check=True)
        subprocess.run([str(executable)], env=env, check=True)
    print("SSBO alignment: guest rounding, native alias identity and invalid writer refusal passed")


if __name__ == "__main__":
    main()
