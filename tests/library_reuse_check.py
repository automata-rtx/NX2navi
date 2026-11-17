"""Exercise library repair across fresh Vulkan processes using a small local graphics-only fixture.

The fixture holds shader-prepared.bin, shader-answers.bin and a shader-cache directory
with matching translations. Its contents stay outside Git. All outputs use a new directory.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import struct
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "fixture", "out", "xlate"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    exe, source, out, xlate = (p.resolve() for p in (args.exe, args.fixture, args.out, args.xlate))
    out.mkdir(parents=True, exist_ok=False)
    for name in ("shader-prepared.bin", "shader-answers.bin"):
        shutil.copyfile(source / name, out / name)
    shutil.copytree(source / "shader-cache", out / "shader-cache")
    env = {k: v for k, v in os.environ.items() if not k.startswith("NX2NAVI_")}
    env["PATH"] = str(xlate.parent) + os.pathsep + env["PATH"]
    env.update({"NX2NAVI_" + k: str(v) for k, v in {
        "PIPELINE_CACHE": out / "pipeline-cache.bin", "SHADER_CACHE": out / "shader-cache",
        "XLATE": xlate, "SHADER_CARRY": 0, "PIPE_PREBUILD": 0,
    }.items()})

    def run(name, action="--probe-prewarm", **settings):
        options = {**env, **{"NX2NAVI_" + k: str(v) for k, v in settings.items()}}
        log = out / (name + ".log")
        with log.open("wb") as output:
            subprocess.run([exe, "--profile", out / "profile", action], cwd=out, env=options,
                           stdout=output, stderr=subprocess.STDOUT, check=True, timeout=120)
        return log.read_text(errors="replace")

    # A complete cache is allowed to hit immediately. A cold cache must converge
    # without a full sweep or clearing any files between the fresh processes.
    later = 0
    for i in range(1, 21):
        log = run(f"boot-{i:02}")
        terminal = re.search(r"library reuse: terminal ([a-z-]+).*", log)
        assert terminal, log[-1000:]
        print(i, terminal[0], flush=True)
        warm = terminal[1] == "sample-reused"
        if warm:
            assert "independent hits 6/6" in terminal[0] and "compile-required 0, built 0" in terminal[0]
            assert "tail entries visited 0" in terminal[0]
        later = later + 1 if warm else 0
        if later == 5:
            break
    assert later == 5, "Did not reach five consecutive fresh-process cache hits"

    full = run("explicit", "--warm-shaders")
    checked = re.search(r"(\d+) entries visited, 0 reused from checked preparation, (\d+) cache hits, 0 built, 0 failed", full)
    assert checked and checked[1] == checked[2], "Manual repair did not verify all graphics entries as hits"
    update = run("update", "--refresh-shaders")
    assert f"{checked[1]} entries visited, {checked[1]} reused from checked preparation" in update

    # Interrupt an explicit pass after its first durable chunk. The next fresh
    # process must complete the remaining entries rather than restarting the pass.
    checkpoint, = out.glob("shader-warm-*.bin")
    if int(checked[1]) > 256:
        with (out / "interrupt.log").open("wb") as output:
            proc = subprocess.Popen([exe, "--profile", out / "profile", "--probe-prewarm"], cwd=out,
                                    env={**env, "NX2NAVI_LIB_WARM": "force"}, stdout=output, stderr=output)
            try:
                until = time.monotonic() + 30
                while time.monotonic() < until and proc.poll() is None:
                    try:
                        record = checkpoint.read_bytes()
                    except PermissionError:
                        continue
                    cursor, = struct.unpack_from("<Q", record, 64)
                    if 0 < cursor < int(checked[1]):
                        proc.terminate()
                        break
                    time.sleep(0.001)
                else:
                    raise AssertionError("Did not observe an in-progress checkpoint")
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
        cursor, = struct.unpack_from("<Q", checkpoint.read_bytes(), 64)
        resumed = run("resumed", LIB_WARM="force")
        assert "terminal repair-complete" in resumed
        assert f"tail entries visited {int(checked[1]) - cursor})" in resumed
        print(f"Resumed at {cursor} of {checked[1]} entries.")
    # An update with a damaged record must verify each entry. A disabled check must
    # leave the repair journal alone. A missing list must give the menu a terminal signal.
    contents = bytearray(checkpoint.read_bytes())
    contents[-1] ^= 1
    checkpoint.write_bytes(contents)
    log = run("corrupt", "--refresh-shaders")
    assert f"{checked[1]} entries visited, 0 reused from checked preparation, {checked[1]} cache hits" in log
    data = checkpoint.read_bytes()
    assert data[:32] == hashlib.sha256(data[32:]).digest()
    assert "terminal disabled" in run("disabled", LIB_WARM=0)
    assert checkpoint.read_bytes() == data
    assert "terminal partial" in run("pressure", BUILD_FREE_MB=99999999)
    (out / "shader-prepared.bin").rename(out / "prepared.saved")
    assert "terminal missing-list" in run("missing-list")
    print("Verified cache reuse, full repair, update, corrupt checkpoint, disabled, memory pressure and missing-list paths.")


if __name__ == "__main__":
    main()
