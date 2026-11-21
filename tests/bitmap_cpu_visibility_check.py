
"""Check production bitmap CPU visibility behavior against the real resource registry."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
runtime = root / "runtime"
source_path = root / "runtime/vk_memory.c"
fixture = root / "tests/bitmap_cpu_visibility_check.c"
cc = Path(os.environ["LOCALAPPDATA"]) / "Programs/2NX2Navi/toolchain/msys64/ucrt64/bin/clang.exe"

source = source_path.read_text(encoding="utf-8")
test = fixture.read_text(encoding="utf-8")

def region(text, first, following):
    begin = text.index(first)
    return text[begin:text.index(following, begin)]

# Extract the production bodies rather than compiling a copied implementation.
implementation = region(source, "// Only the actual writeback spans", "// Records the ranges as one")
implementation += region(source, "static int native_rmw_bitmap_visibility_ranges(", "// The bitmap union shader")
test = test.replace(region(test, "// Only the actual writeback spans", "#undef realloc"), implementation)
test = test.replace('enum { POOL_IMPORT', 'typedef struct VkMemSpanCapture VkMemSpanCapture;\n'
                    'typedef struct VkMemNativeRmwBitmap VkMemNativeRmwBitmap;\n\nenum { POOL_IMPORT', 1)
# One fixture-supplied global pair was unused because each test has local range values.
test = test.replace("rc_registry_runtime(", "fixture_registry(")
test = test.replace("static RcSourceRange guest_range, canonical_range;\n", "")

# Demand actual registry authority, not an arbitrary non-OK observation.
test = test.replace("return rc_registry_observe(registry, h, 1, &o) != RC_REGISTRY_OK;",
                    "RcRegistryResult result = rc_registry_observe(registry, h, 1, &o);\n"
                    "    assert(result == RC_REGISTRY_OK || result == RC_REGISTRY_BLOCKED);\n"
                    "    return result == RC_REGISTRY_BLOCKED;")
test = test.replace("    assert(capture.refs == 1);\n    rc_registry_destroy(registry);",
                    "    assert(capture.refs == 1 && copy_interval_calls == 0);\n    rc_registry_destroy(registry);")

env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="bitmap-visibility-") as path:
    generated = Path(path) / "check.c"
    executable = Path(path) / "check.exe"
    generated.write_text(test, encoding="utf-8", newline="\n")
    subprocess.run([cc, "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-Wno-unused-function", "-DRC_REGISTRY_SINGLE_THREADED",
                    "-fsanitize=undefined", "-fsanitize-trap=undefined", "-I", runtime,
                    generated, runtime / "resource_core_registry.c", "-o", executable],
                   check=True, env=env)
    subprocess.run([executable], check=True, env=env)


