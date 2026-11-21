"""Verify the actual fragment mirror receipt geometry and completion without a GPU."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / "runtime/vk_memory.c"

def extract(text, name):
    start = re.search(r"^(?:static )?[A-Za-z_][\w *]*\b" + name + r"\([^;{}]*\) \{", text, re.M)
    if not start:
        raise ValueError(name)
    start = start.start()
    return text[start:text.index("\n}", start) + 3]

def between(text, start, end):
    pos = text.index(start)
    return text[pos:text.index(end, pos)]

def main():
    source = MEMORY.read_text()
    definitions = between(source, "// The exact word is provenance", "static RcRegistryResult selected_mirror_access_locked")
    definitions += between(source, "typedef struct {\n    PoolRec *pool;\n    uint64_t off;\n} NativeRmwBitmapEdge;",
                            "static void native_rmw_bitmap_free_shapes")
    definitions += between(source, "typedef struct {\n    size_t count;\n    const SnapRange *ordinary;",
                            "static int native_rmw_spans_overlap")
    names = ("selected_mirror_access_locked",
        "native_rmw_bitmap_marker_bytes", "native_rmw_bitmap_cancel_tickets",
        "vk_mem_native_rmw_bitmap_cancel", "native_rmw_bitmap_cancel_submitted",
        "native_rmw_bitmap_test", "native_rmw_bitmap_extents", "native_rmw_bitmap_guest_range",
        "native_rmw_bitmap_completed", "vk_mem_native_rmw_bitmap_declare", "vk_mem_native_rmw_bitmap_issued",
        "native_rmw_spans_overlap", "native_rmw_bitmap_overlaps", "native_rmw_bitmap_tail_refusal", "native_rmw_bitmap_tail_supported",
        "native_rmw_bitmap_compatible", "native_rmw_bitmap_edge_compare", "native_rmw_bitmap_slices",
        "native_rmw_bitmap_first_clip", "native_rmw_bitmap_snapshot_clipped", "native_rmw_bitmap_next_complement",
        "native_rmw_bitmap_tail_plan")
    fixture = Path(__file__).with_suffix(".c").read_text()
    fixture = fixture.replace("/* production declarations */", definitions)
    fixture = fixture.replace("/* production functions */", "\n\n".join(extract(source, name) for name in names))
    compiler = os.environ.get("CC") or shutil.which("clang")
    if not compiler:
        raise SystemExit("Clang is required (set CC)")
    cc = Path(shutil.which(compiler) or compiler).resolve()
    vma = Path(os.environ.get("VMA_INCLUDE", cc.parent.parent / "include/vma"))
    includes = ["-I" + str(ROOT / "runtime"), "-I" + str(vma)]
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="fragment-rmw-receipt-") as tmp:
        destination = Path(tmp)
        check = destination/"check.c"
        output = destination/"check.exe"
        check.write_text(fixture)
        subprocess.run([cc,"-std=gnu11","-O1","-fno-strict-aliasing","-fsanitize=undefined",
            "-fsanitize-trap=undefined",*includes,str(check),"-o",str(output)],check=True,env=env)
        subprocess.run([output],check=True,env=env)

if __name__ == "__main__":
    main()
