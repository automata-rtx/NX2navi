"""Verify retained storage writer selection, exact issued stores and native import custody."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT / "runtime"
draw = (runtime / "vk_exec_draw.c").read_text(encoding="utf-8")
memory = (runtime / "vk_memory.c").read_text(encoding="utf-8")

def section(source, start, end):
    first = source.index(start)
    return source[first:source.index(end, first)]

def function(source, start):
    first = source.index(start)
    return source[first:source.index("\n}", first) + 2]

def main():
    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Clang is required (set CC)")
    clang = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get("VMA_INCLUDE", clang.parent.parent / "include" / "vma"))
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get("PATH", ""))

    with tempfile.TemporaryDirectory(prefix="selected-storage-") as temp:
        directory = Path(temp)
        includes = {
            "selected_types.inc": section(draw, "typedef struct {\n    uint32_t set,binding;\n    uint64_t addr,size,authority_size;",
                                           "} DeviceStoragePointer;") + "} DeviceStoragePointer;\n",
            "selected_functions.inc": function(draw, "static const DeviceStoragePointer *device_storage_pointer(") + "\n" +
                                      function(draw, "static int device_storage_pointer_current(") + "\n" +
                                      section(draw, "static EffectStorageCapture *effect_storage_find(",
                                               "static EffectStorageCapture *effect_storage_new(") +
                                      section(draw, "static void buffer_preserve_release(",
                                              "static void buffer_preserve_bound(") +
                                      function(draw, "static void buffer_preserve_import(") + "\n" +
                                      function(draw, "static int selected_storage_cold_descriptor(") + "\n" +
                                      function(draw, "static int selected_storage_rmw_descriptor(") + "\n" +
                                      function(draw, "static void selected_storage_writers_issued(") + "\n" +
                                      function(draw, "static void ordinary_storage_writers_issued("),
            "selected_provider.inc": '#include "resource_host_write.h"\n' +
                                     section(memory, "struct VkMemSpanCapture {", "// pool_lock held. Compare only the actual argument span") +
                                     function(memory, "static PoolRec *snapshot_import_window(") + "\n" +
                                     section(memory, "static int import_writer_candidate(",
                                             "RcRegistryResult vk_mem_span_capture_bound_input(") +
                                     section(memory, "static void mirror_input_release(", "// pool_lock held. A reused Vulkan handle") +
                                     section(memory, "static uint64_t mirror_scene(void) {", "static void mem_use_locked(") +
                                     section(memory, "static void mirror_retire_check(PoolRec *pool, int completed) {",
                                             "// Finalized pools") +
                                     function(memory, "static int mirror_transition_ready_locked(const PoolRec *pool,") + "\n" +
                                     function(memory, "static int host_interval_vectors_match(const MemBackingCapture *source, const MemBackingCapture *provider,\n"
                                                       "                                       uint64_t address, uint64_t size, VkMemCanonicalSourceQualification *report) {"),
        }
        for name, content in includes.items():
            (directory / name).write_text(content, encoding="utf-8")
        for name, sources in (
            ("selected_storage_writer_check", [runtime / "resource_core_effects.c"]),
            ("selected_storage_provider_check", []),
        ):
            exe = directory / (name + ".exe")
            subprocess.run([str(clang), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=undefined", "-fsanitize-trap=undefined",
                            "-I" + str(runtime), "-I" + str(vma), "-I" + str(directory),
                            str(ROOT / "tests" / (name + ".c")), *(str(s) for s in sources), "-o", str(exe)],
                           env=env, check=True)
            subprocess.run([str(exe)], env=env, check=True)

if __name__ == "__main__":
    main()
