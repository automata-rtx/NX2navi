"""Compile/run the core contracts without game data; always remove temporary executables."""
from pathlib import Path
import os
import subprocess
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
work = ROOT / "build" / "test"
work.mkdir(parents=True, exist_ok=True)
work = work.resolve()
work.relative_to(ROOT)  # Temporary-directory cleanup must remain in this worktree.
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
sys.path.insert(0, str(ROOT / "tools"))
import make_game as kit
_, _, _, vma = kit.toolchain(CC.parent.parent)
with tempfile.TemporaryDirectory(prefix="resource-core-", dir=work) as temporary:
    for name in ("observation", "lifetime", "admission", "integration", "registry", "multibacking", "writer_census",
                 "source_access", "backing", "budget", "material", "scene", "vma", "texture_allocation", "effects", "inventory",
                 "gpu_capture", "native_image", "buffer_phase", "buffer_producer"):
        source = ROOT / "tests" / f"resource_core_{name}_check.c"
        if not source.is_file():
            raise FileNotFoundError(f"Required contract fixture missing: {source}")
        exe = Path(temporary) / (source.stem + ".exe")
        flags = [*kit.vulkan_headers(CC, env), "-I" + str(vma), "-I" + str(ROOT / "runtime")] if name == "vma" else []
        subprocess.run([str(CC), "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", *flags, str(source), "-o", str(exe)],
                       env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)
    for recipes in (0, 1):
        source = ROOT / "tests" / "resource_effect_recipe_adversarial.c"
        exe = Path(temporary) / f"resource_effect_recipe_adversarial_{recipes}.exe"
        subprocess.run([str(CC), "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", f"-DRC_EFFECT_RECIPES={recipes}",
                        str(source), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)
for fixture in ("texture_registration_history_check.py", "registration_record_cutoff_check.py"):
    subprocess.run([sys.executable, str(ROOT / "tests" / fixture)], env=env, check=True)
