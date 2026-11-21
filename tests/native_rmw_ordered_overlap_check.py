"""Exercise the production scalar-to-bitmap writeback tail without executing Vulkan."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent

def section(source, start, end):
    first = source.index(start)
    return source[first:source.index(end, first)]

def function(source, name):
    match = re.search(r"^(?:static )?[A-Za-z_][\w *]*\b" + name + r"\([^;{}]*\) \{", source, re.M)
    if not match:
        raise ValueError(name)
    return source[match.start():source.index("\n}", match.end()) + 2]

def shader(source, name, aliases):
    # Both tail kernels are scalar C-compatible after substituting the GLSL interfaces.
    source = source[source.index("void main() {"):]
    source = source.replace("void main()", "static void " + name + "(void)")
    source = source.replace("gl_GlobalInvocationID.x", "invocation")
    for glsl, c in aliases.items():
        source = source.replace(glsl, c)
    source, stores = re.subn(r"dst\[(.*?)\] = ([^;]+);", r"write_import(\1, \2);", source)
    if not stores:
        raise ValueError(name + " has no traced import stores")
    return source

def fixture_sources(memory, shaders):
    declarations = section(memory, "typedef struct { PoolRec *r; uint64_t off, size, soff;",
                           "typedef struct { VkBuffer buf; VmaAllocation alloc;")
    declarations += section(memory, "// The exact word is provenance", "static RcRegistryResult selected_mirror_access_locked")
    declarations += section(memory, "struct VkMemNativeRmwWord {", "static void native_rmw_pipeline_free")
    declarations += section(memory, "typedef struct {\n    PoolRec *pool;\n    uint64_t off;\n} NativeRmwBitmapEdge;",
                            "static void native_rmw_bitmap_free_shapes")
    declarations += section(memory, "typedef struct {\n    size_t count;\n    const SnapRange *ordinary;",
                            "static int native_rmw_spans_overlap")
    visibility = section(memory, "// Only the actual writeback spans", "// Records the ranges as one")
    selected = function(memory, "selected_mirror_access_locked")
    if "static const char *native_rmw_bitmap_word_refusal(" in memory:
        start = memory.index("static const char *native_rmw_bitmap_word_refusal(")
        selected += "\n" + memory[start:memory.index(";", start) + 1]
    admission = section(memory, "static VkResult native_rmw_bitmap_failure(", "static VkResult native_rmw_bitmap_source_failure(")
    admission += section(memory, "static int native_rmw_bitmap_not_current(", "int vk_mem_native_rmw_bitmap_order(")
    bitmap_receipts = section(memory, "static void native_rmw_bitmap_binding_canceled(", "static const char *native_rmw_bitmap_tail_refusal(")
    bitmap_receipts += section(memory, "int vk_mem_native_rmw_bitmap_bind(", "static int merging(")
    names = ("native_rmw_word_binding_canceled", "native_rmw_word_record_physical",
             "vk_mem_native_rmw_word_bind", "native_rmw_guest_range", "native_rmw_word_cancel_tickets",
             "vk_mem_native_rmw_word_cancel", "native_rmw_word_cancel_submitted", "native_rmw_word_completed",
             "vk_mem_native_rmw_word_declare", "vk_mem_native_rmw_word_issued",
             "native_rmw_bitmap_marker_bytes", "native_rmw_bitmap_allocation_bytes")
    receipts = "\n\n".join(function(memory, name) for name in names)
    tail = section(memory, "static int native_rmw_spans_overlap(", "// NX2NAVI_TRACE=perf, every 600 frames: the mirrored pools")
    bitmap_source = (shaders / "native_rmw_bitmap.comp").read_text(encoding="utf-8")
    word_source = (shaders / "native_rmw_word.comp").read_text(encoding="utf-8")
    kernels = shader(bitmap_source, "bitmap_kernel", {
        "source.words": "src", "mirror_data.words": "mirror_view", "conditions.words": "marks",
        "import_data.words": "dst",
    })
    kernels += shader(word_source, "word_kernel", {
        "source.data": "src", "mirror_word.data": "mirror_view", "status.executed": "marks[0]",
        "status.conditions": "marks[1]", "import_word.data": "dst",
    })
    return declarations, visibility, selected + "\n" + receipts + "\n" + bitmap_receipts + "\n" + admission, tail, kernels

def main():
    runtime = HERE.parent / "runtime"
    source = (runtime / "vk_memory.c").read_text(encoding="utf-8")
    test = Path(__file__).with_suffix(".c").read_text(encoding="utf-8")
    values = fixture_sources(source, runtime / "shaders")
    for tag, content in zip(("declarations", "visibility", "receipts", "tail", "kernels"), values, strict=True):
        marker = f"/* production {tag} */"
        assert test.count(marker) == 1
        test = test.replace(marker, content)
    test = test.replace("rc_registry_runtime(", "fixture_registry(")
    compiler = os.environ.get("CC") or shutil.which("clang")
    if not compiler:
        raise SystemExit("Clang is required (set CC)")
    cc = Path(shutil.which(compiler) or compiler).resolve()
    vma = Path(os.environ.get("VMA_INCLUDE", cc.parent.parent / "include/vma"))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="native-rmw-overlap-") as directory:
        work = Path(directory)
        generated = work / "extracted.c"
        executable = work / "check.exe"
        generated.write_text(test, encoding="utf-8", newline="\n")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                        "-Wno-unused-variable", "-DRC_REGISTRY_SINGLE_THREADED", "-fsanitize=undefined",
                        "-fsanitize-trap=undefined", "-I", str(runtime), "-I", str(vma), str(generated),
                        str(runtime / "resource_core_registry.c"), "-o", str(executable)], env=env, check=True)
        subprocess.run([str(executable)], env=env, check=True)

if __name__ == "__main__":
    main()
