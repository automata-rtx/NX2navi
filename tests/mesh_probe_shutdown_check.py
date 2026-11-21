"""Exercise actual probe teardown with retained graphics revisions."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ordered_dispatch_preparation_check import function


def main():
    runtime = HERE.parent / "runtime"
    pipeline = (runtime / "vk_pipeline.c").read_text(encoding="utf-8")
    sync = (runtime / "vk_sync.c").read_text(encoding="utf-8")
    probe = (runtime / "mesh_fb_probe.c").read_text(encoding="utf-8")
    owners = [function(pipeline, name) for name in (
        "graphics_take_retired_locked", "graphics_destroy_retired", "graphics_drop_revision",
        "vk_pipeline_graphics_release", "graphics_scene_done", "graphics_scene_cancel",
        "vk_pipeline_graphics_record",
    )]
    owners.extend(function(sync, name) for name in (
        "pending_disposable", "pending_dispose", "pending_dispose_queue",
    ))
    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Clang is required (set CC)")
    clang = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get("VMA_INCLUDE", clang.parent.parent / "include" / "vma"))
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="mesh-probe-shutdown-") as directory:
        work = Path(directory)
        (work / "meshfb-owners.inc").write_text("\n".join(owners), encoding="utf-8")
        (work / "meshfb-finish.inc").write_text(function(probe, "meshfb_finish_probe"), encoding="utf-8")
        exe = work / ("check.exe" if os.name == "nt" else "check")
        subprocess.run([str(clang), "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined", "-I", str(runtime),
                        "-I", str(vma), "-I", str(work), str(HERE / "mesh_probe_shutdown_check.c"),
                        "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True, timeout=20)


if __name__ == "__main__":
    main()
