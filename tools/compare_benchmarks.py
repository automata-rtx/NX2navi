"""Compare warm, visually accepted compact benchmark runs across two builds."""

import argparse
import csv
import json
import statistics
from pathlib import Path

METRICS = ("average_fps", "median_fps", "p5_low_fps", "p1_low_fps", "p0_1_low_fps")
SCENES = ("kokiri", "field", "castle-town")


def read_group(root):
    rows = []
    historical = {}
    for key, name in [("route", "runs.csv"), ("stress", "stress-windows.csv")]:
        path = root / name
        if path.is_file():
            with path.open(newline="", encoding="utf-8-sig") as stream:
                historical[key] = {(r["scene"], int(r["run"])): r for r in csv.DictReader(stream)}
    for p in sorted(root.glob("[0-9][0-9]-*/run.json")):
        folder = p.parent
        run = int(folder.name[:2])
        meta = json.loads(p.read_text())
        if "scene" not in meta or (folder / "excluded.txt").is_file():
            continue
        scene = meta["scene"]
        if "route" in meta and "stress" in meta:
            warm = meta.get("comparable_fully_warm", False)
            scores = {k: meta[k] for k in ("route", "stress")}
        elif historical and (scene, run) in historical.get("route", {}):
            scores = {k: historical[k][scene, run] for k in ("route", "stress")}
            warm = scores["route"]["comparable_fully_warm"] == "True"
        else:
            continue
        if not warm:
            continue
        if scene == "castle-town" and not (folder / "camera-review.txt").is_file():
            raise ValueError(f"Review the six images and record acceptance in {folder}/camera-review.txt")
        settings = {k: v for k, v in meta["settings"].items() if k not in ("INPUT_SCRIPT", "FRAME_CAPTURE")}
        rows.append((scene, run, meta["save_sha256"], meta["route_sha256"], settings, scores))
    return rows


def compare(baseline, candidate):
    result = []
    for scene in SCENES:
        old = [r for r in baseline if r[0] == scene]
        new = [r for r in candidate if r[0] == scene]
        if not old or not new:
            raise ValueError(f"{scene} has no comparable runs on one or both builds")
        for r in old + new:
            if r[2:5] != old[0][2:5]:
                raise ValueError(f"{scene}: save, script or benchmark settings differ")
        for window in ("route", "stress"):
            for metric in METRICS:
                a = [float(r[-1][window][metric]) for r in old]
                b = [float(r[-1][window][metric]) for r in new]
                previous, current = statistics.median(a), statistics.median(b)
                result.append(
                    dict(
                        scene=scene,
                        window=window,
                        metric=metric,
                        baseline_runs=len(a),
                        candidate_runs=len(b),
                        baseline=previous,
                        candidate=current,
                        delta_pct=100 * (current / previous - 1),
                        baseline_min=min(a),
                        baseline_max=max(a),
                        candidate_min=min(b),
                        candidate_max=max(b),
                    )
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    rows = compare(read_group(args.baseline), read_group(args.candidate))
    path = args.candidate / "comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    print(path)


if __name__ == "__main__":
    main()
