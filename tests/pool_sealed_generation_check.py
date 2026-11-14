"""Extract and check the production sealed-pool version boundary; CPU only."""
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from bind_recipe_check import function


def main():
    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Set CC to MSYS2 UCRT64 Clang")
    content_root = next(p for p in (ROOT, *ROOT.parents) if (p / "tools/check_content.py").is_file())
    sys.path.insert(0, str(content_root / "tools"))
    from check_content import GENERATED_MARKER
    work = ROOT / "build/test/pool_sealed_generation"
    work.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "runtime/vk_exec_draw.c").read_text()
    def definition(name):
        # Select a definition, not a prototype or a call inside an earlier function.
        match = re.search(r"^(?:static )?(?:void|int|uint64_t) " + name + r"\([^;{]*\)\s*\{", source, re.M)
        if not match:
            raise RuntimeError("Missing production definition " + name)
        return function(source[match.start():], name)
    selected = "\n".join(definition(name) for name in
                         ("vk_exec_pool_generation", "pool_sync_complete", "pool_current", "pool_take_set", "pool_sync"))
    (work / "pool_sealed_under_test.h").write_text("// " + GENERATED_MARKER + "\n" + selected)
    exe = work / ("pool_sealed_generation_check" + (".exe" if os.name == "nt" else ""))
    env = dict(os.environ)
    env["PATH"] = str(Path(cc).resolve().parent) + os.pathsep + env.get("PATH", "")
    subprocess.run([cc, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(ROOT / "runtime"),
                    "-I", str(work), str(ROOT / "tests/pool_sealed_generation_check.c"), "-o", str(exe)],
                   check=True, env=env)
    result = subprocess.run([str(exe)], capture_output=True, text=True, check=True, env=env)
    (work / "result.txt").write_text(result.stdout)
    print(result.stdout, end="")


if __name__ == "__main__":
    main()
