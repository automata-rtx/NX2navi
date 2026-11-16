"""Run the three standard Windows routes with the compact recorder. See docs/benchmarks.md."""

import argparse
import ctypes as ct
from ctypes import wintypes as wt
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import subprocess
import time

from PIL import Image, ImageDraw, ImageGrab
from benchmark_log import archive_log
from benchmark_menu import MenuGate
from benchmark_processes import competing
from frame_capture import read_frames
from run_game import take_gpu_lock, release_gpu_lock

SCENES = ("castle-town", "kokiri", "field")
env_settings = dict(
    DLAA="1",
    DLSS="performance",
    TAA="dlss",
    FRAMEGEN="off",
    FPS_LIMIT="unlimited",
    FULLSCREEN="1",
    REPLAY_ASYNC="0",
    DYN_RES="0",
    REALTIME="1",
    VK_PRESENT="immediate",
    PERF_OVERLAY="off",
    TRACE="",
    INPUT="none",
    ASSUME_FOCUS="1",
)


def digest(file):
    with file.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(p, data):
    p.write_text(json.dumps(data, indent=2))


u = ct.windll.user32
u.GetForegroundWindow.restype = wt.HWND
ct.windll.user32.SetProcessDPIAware()


def capture(pid, path):
    hwnd = u.GetForegroundWindow()
    active = wt.DWORD()
    u.GetWindowThreadProcessId(wt.HWND(hwnd), ct.byref(active))
    if active.value != pid:
        raise RuntimeError("game window is not in foreground")
    rect = wt.RECT()
    u.GetClientRect(wt.HWND(hwnd), ct.byref(rect))
    point = wt.POINT(0, 0)
    u.ClientToScreen(wt.HWND(hwnd), ct.byref(point))
    image = ImageGrab.grab(
        bbox=(point.x, point.y, point.x + rect.right, point.y + rect.bottom), all_screens=True
    )
    image.thumbnail((1200, 800))
    image.convert("RGB").save(path, quality=90)


def frame_metrics(frames):
    intervals = sorted(frame.interval_ns for frame in frames)

    def fps_at(p):
        index = (len(intervals) - 1) * p
        fraction = index % 1
        value = intervals[math.floor(index)] * (1 - fraction) + intervals[math.ceil(index)] * fraction
        return 1e9 / value

    return dict(
        frames=len(intervals),
        average_fps=1e9 / statistics.mean(intervals),
        median_fps=fps_at(0.5),
        p5_low_fps=fps_at(0.95),
        p1_low_fps=fps_at(0.99),
        p0_1_low_fps=fps_at(0.999),
    )


class CameraDeviation(ValueError):
    """A settled view missed the benchmark heading."""


def review_camera(folder, log, reference):
    chunks = re.split(r"input: script frame \d+ at [\d.]+ ms: (?:movement-start|settled-[1-6])\b", log)
    if len(chunks) != 8 or len(reference) != 6:
        raise ValueError("Missing Castle Town camera marker or reference view")
    headings = []
    for chunk in chunks[:7]:
        matches = re.findall(r"mousetrace: cam t=[\d.]+ fwd=([-\d.]+),([-\d.]+),([-\d.]+)", chunk)
        if not matches:
            raise ValueError("Missing camera samples")
        headings.append(tuple(map(float, matches[-1])))
    errors = []
    montage = Image.new("RGB", (900, 304))
    labels = ImageDraw.Draw(montage)
    for i, (direction, expected) in enumerate(zip(headings[1:], reference)):
        target = expected["direction"]
        cosine = sum(x * y for x, y in zip(target, direction)) / (
            math.hypot(*target) * math.hypot(*direction)
        )
        errors.append(math.degrees(math.acos(max(-1, min(1, cosine)))))
        with Image.open(folder / f"turn-{i + 1}.jpg") as shot:
            x, y = (i % 3) * 300, (i // 3) * 152
            montage.paste(shot.resize((300, 126), Image.Resampling.LANCZOS), (x, y))
            labels.text((x + 4, y + 128), f"Turn {i + 1}", fill="white")
    montage.save(folder / "review.jpg", quality=87)
    angles = [math.degrees(math.atan2(v[0], v[2])) for v in headings]
    turns = [abs((b - a + 180) % 360 - 180) for a, b in zip(angles, angles[1:])]
    if max(errors) > 8 or min(turns) < 150:
        raise CameraDeviation(f"Castle Town camera changed course: turns {turns}, heading errors {errors}")
    return errors


def score(folder, log, scene, version):
    frames = list(read_frames(folder / "frames.bin"))
    if any(b.frame != a.frame + 1 or b.world_us < a.world_us for a, b in zip(frames, frames[1:])):
        raise ValueError("Discontinuous recording")
    required = [
        f"frame capture: saved {len(frames)} frames",
        "game steps by the clock",
        "NX2NAVI_REPLAY_ASYNC=0",
        "1720x720 -> 3440x1440",
        "present mode immediate",
        f"version: 2NX2Navi {version}, game ",
    ]
    if (
        any(s not in log for s in required)
        or "DLSS Frame Generation on the swapchain" in log
        or "perf: slow frame " in log
    ):
        raise ValueError("Unexpected renderer settings, version or capture mode")

    def boundary(name):
        found = re.findall(r"segment: '" + name + r"' starts at frame (\d+)", log)
        if len(found) != 1:
            raise ValueError(f"Missing or repeated {name} marker")
        return int(found[0])

    begin, end = boundary("movement-start"), boundary("movement-end")
    route = [r for r in frames if begin < r.frame <= end]
    if not route or len(route) != end - begin:
        raise ValueError("Incomplete route capture")
    if scene == "castle-town":
        by_frame = {r.frame: r for r in frames}
        taps = [
            int(f)
            for f, t in re.findall(r"input: script frame (\d+) at ([\d.]+) ms: ZL", log)
            if 5000 < float(t) < 20000
        ]
        if len(taps) != 6:
            raise ValueError("Expected six scored L2 taps")
        stress = []
        for frame in taps:
            elapsed = 0
            while elapsed < 750000000:
                frame += 1
                stress.append(by_frame[frame])
                elapsed += by_frame[frame].interval_ns
    else:
        a, b = (1, 8) if scene == "kokiri" else (24, 32)
        stress = [r for r in frames if a * 1000000 < r.world_us <= b * 1000000]
    return frame_metrics(route), frame_metrics(stress)


def run_route(args, folder, scene, reference):
    folder.mkdir()
    route = args.saves / f"{scene}-route.txt"
    if not route.is_file():
        route = Path(__file__).resolve().parent / "input" / "benchmark" / f"{scene}-route.txt"
    profile = folder / "profile"
    profile.mkdir()
    uid = args.saves / ("castle-town-account.uid" if scene == "castle-town" else "account.uid")
    shutil.copyfile(uid, profile / "account.uid")
    low, high = struct.unpack("<QQ", uid.read_bytes())
    source = args.saves / scene
    shutil.copytree(source, profile / "save" / f"{high:016x}{low:016x}")
    settings = dict(env_settings, FRAME_CAPTURE=str(folder / "frames.bin"))
    if scene == "castle-town":
        settings["MOUSE_TRACE"] = "1"
    env = {k: v for k, v in os.environ.items() if not k.startswith("NX2NAVI_")}
    env.update({"NX2NAVI_" + k: v for k, v in settings.items()})
    log_path = folder / "game.log"
    while competing():
        time.sleep(2)
    gate = MenuGate(route, log_path)
    env["NX2NAVI_INPUT_SCRIPT"] = gate.path
    meta = dict(
        scene=scene,
        started=stamp(),
        settings=settings,
        route_sha256=digest(route),
        save_sha256={str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()},
    )
    write_json(folder / "run.json", meta)
    overlaps, shots = [], []
    try:
        with log_path.open("wb") as output, log_path.open("rb") as watch:
            command = [
                str(args.exe),
                "--profile",
                str(profile),
                "--seconds",
                "71" if scene == "field" else "21",
            ]
            proc = subprocess.Popen(
                command, cwd=args.exe.parent, env=env, stdout=output, stderr=subprocess.STDOUT
            )
            gate.pid = proc.pid
            start = time.monotonic()
            poll = 0
            pending = b""
            try:
                while proc.poll() is None:
                    now = time.monotonic()
                    if now - start > 1500:
                        raise TimeoutError("The scripted route did not finish")
                    if now >= poll:
                        found = competing(proc.pid)
                        if found and gate.note.get("reason"):
                            overlaps.append([stamp(), found])
                        poll = now + 1
                    if scene == "castle-town":
                        lines = (pending + watch.read()).split(b"\n")
                        pending = lines.pop()
                        for line in lines:
                            marker = re.search(
                                rb"input: script frame (\d+) at ([\d.]+) ms: settled-([1-6])\b", line
                            )
                            if marker:
                                turn = int(marker[3])
                                before = time.perf_counter()
                                capture(proc.pid, folder / f"turn-{turn}.jpg")
                                shots.append(
                                    dict(
                                        turn=turn,
                                        frame=int(marker[1]),
                                        capture_ms=1000 * (time.perf_counter() - before),
                                    )
                                )
                    time.sleep(0.018 if scene == "castle-town" else 0.15)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            if proc.returncode:
                raise RuntimeError(f"Game exited with {proc.returncode}")
        log = log_path.read_text(encoding="utf-8", errors="replace")
        meta["route"], meta["stress"] = score(folder, log, scene, args.version)
        if scene == "castle-town":
            meta["heading_errors"] = review_camera(folder, log, reference)
        before_route = log.split("segment: 'movement-start'")[0]
        warm = (
            "library warm: done," in before_route
            or "driver's cache has the prepared libraries" in before_route
        )
        meta["comparable_fully_warm"] = warm and not overlaps
    finally:
        gate.cancel.set()
        meta.update(ended=stamp(), menu_gate=gate.note, overlaps=overlaps, screenshots=shots)
        write_json(folder / "run.json", meta)
        if log_path.is_file():
            archive_log(log_path)
        shutil.rmtree(profile)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--saves", required=True, type=Path)
    parser.add_argument(
        "--reference", required=True, type=Path, help="approved six-view camera direction JSON"
    )
    parser.add_argument("--out", required=True, type=Path, help="new benchmark directory")
    parser.add_argument("--version", required=True, help="expected version printed at boot")
    parser.add_argument("--revision", required=True, help="source commit of the executable")
    parser.add_argument("--runs", type=int, default=3, help="repetitions per scene")
    args = parser.parse_args()
    if ct.windll.shell32.IsUserAnAdmin():
        parser.error("Use a standard-user Windows desktop")
    if args.runs < 1:
        parser.error("--runs must be positive")
    args.exe = args.exe.resolve(strict=True)
    args.saves = args.saves.resolve(strict=True)
    args.out = args.out.resolve()
    reference = json.loads(args.reference.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(
        args.out / "manifest.json",
        dict(
            executable=str(args.exe),
            sha256=digest(args.exe),
            revision=args.revision,
            version=args.version,
            settings=env_settings,
            reference_sha256=digest(args.reference),
        ),
    )
    lock = take_gpu_lock(args.exe.parent)
    try:
        skip_castle = False
        for n in range(1, args.runs + 1):
            for scene in SCENES:
                if scene == "castle-town" and skip_castle:
                    continue
                for attempt in range(1, 4):
                    suffix = "" if attempt == 1 else f"-attempt{attempt}"
                    folder = args.out / f"{n:02d}-{scene}{suffix}"
                    print(f"{stamp()} starting {folder.name}", flush=True)
                    try:
                        record = run_route(args, folder, scene, reference)
                    except CameraDeviation as error:
                        (folder/"excluded.txt").write_text(str(error))
                        print(f"{folder.name}: {error}", flush=True)
                        skip_castle = attempt == 3
                        continue
                    print(f"{stamp()} finished {folder.name}, warm and quiet: {record['comparable_fully_warm']}", flush=True)
                    break
    finally:
        release_gpu_lock(lock)


if __name__ == "__main__":
    main()
