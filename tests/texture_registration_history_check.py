"""Check immutable descriptor history and continuation horizon selection without Vulkan."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from bind_recipe_check import function

cc = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
source = (ROOT / "runtime/vk_exec.c").read_text()
definition = source[source.index("static void cont_registration_defer("):]
texture = (ROOT / "runtime/vk_texture.c").read_text()
pool = texture[texture.index("static void texture_pool_close_slot("):]
work = ROOT / "build/test"
work.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="registration-history-", dir=work) as directory:
    out = Path(directory)
    selected = [function(definition, "cont_registration_defer"), function(definition, "cont_registration_resume"),
                function(pool, "texture_pool_close_slot"),
                function(pool, "texture_pool_needs_replay"), function(pool, "vk_texture_pool_release_ordered"),
                function(pool, "vk_texture_pool_release"),
                function(texture, "texture_history_sample_locked"),
                function(texture, "vk_texture_history_sample_current_locked")]
    (out / "registration_resume_under_test.h").write_text("\n".join(selected))
    exe = out / "check.exe"
    subprocess.run([str(cc), "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined", "-fsanitize-trap=all",
                    "-I", str(out), str(ROOT / "tests/texture_registration_history_check.c"), "-o", str(exe)],
                   env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)
