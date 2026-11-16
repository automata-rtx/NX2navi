# Windows performance benchmarks

Run the suite on the standard-user desktop with a build containing the compact recorder. Use Python 3.12 or newer with Pillow installed. The three routes use 3440 × 1440 fullscreen, DLSS Performance at 1720 × 720, immediate presentation, unlimited FPS, frame generation off and one-frame-ahead off. The runner checks the corresponding game log messages. Keep other branch-specific graphics and latency options at their defaults and record which ones were active. When staging a candidate, include its matching runtime DLLs and use the same installed game data, shader pack, mods and settings where possible. Keep the personal save profile separate.

On the benchmark PC the three original saves, their two account UID files and the scripts live in `build/benchmark-saves`. The approved Castle Town collage is in `build/beta1-baseline-20261116/castle-town-six-turns.jpg`. Its six reference directions are in the JSON below. The saves and reference images belong outside Git. Copies of the scripted inputs are in `tools/input/benchmark`. The suite prefers the copies alongside the saved profiles. Keep their hashes and save hashes matched across candidates.

```powershell
python tools/benchmark_suite.py --exe "PATH/TO/GAME.exe" --version "Beta 1" --revision "SOURCE_COMMIT" --saves build/benchmark-saves --reference build/beta1-baseline-20261116/01-castle-town/camera-vs-reference.json --out build/new-benchmark --runs 3
```

Use the build's actual version and commit. Supply a fresh output directory. The runner rotates Castle Town, Kokiri Forest and Hyrule Field. It waits on the driver library probe or warm-up before feeding the original menu inputs and measures the route between the movement markers. Each launch has a disposable profile. Other running games and compiler processes are checked before and during the route. Compare runs marked `comparable_fully_warm` after verifying visual acceptance. Keep any cold or interrupted runs labeled. Coordinate with other desktop users to avoid competing tests.

For Castle Town, review each six-turn collage against the approved reference using stationary landmarks. NPCs can move. Check all six reversals for unintended NPC lock-on. The runner checks the camera direction vectors, retries a missed heading and drops the Castle Town set after three consecutive misses. If visual review discovers a deviation the heading check missed, exclude that attempt from the results. Delete each attempt's screenshots after review and keep the reference collage.

Each run directory holds `run.json`, `frames.bin` and a verified lossless `game.log.gz`. Compression starts after game exit. Each frame carries its application present number, interval in nanoseconds and game clock in microseconds. Whole-route scores include presents strictly after the start marker through the end marker. The Kokiri stress window is world seconds 1–8, Field is 24–32, and Castle Town is the first 750 ms of present intervals after each of six scored L2 taps. Average FPS is the reciprocal of mean interval. Median, 5%, 1% and 0.1% lows are reciprocals of linearly interpolated P50, P95, P99 and P99.9 frame times. Summarize scenes by the median of the per-run metrics and retain their ranges. Very short windows can have just one or two frames deciding the 0.1% low.

The latest Beta 1 reference on the benchmark PC is `build/beta1-frame-recorder/runs.csv` with matching `stress-windows.csv`. Filter on `comparable_fully_warm=True` (four Kokiri, four Field and three Castle Town runs). These historical runs support a comparison with past performance. A small difference inside their run-to-run variation needs stronger evidence to attribute it to a branch.

After verifying a Castle Town collage, write the verdict in that run's `camera-review.txt`. Use `excluded.txt` for a rejected attempt. The comparison tool checks the save hashes, route hashes and capture settings, then writes `comparison.csv` in the candidate directory. It includes both builds' sample counts, medians, ranges and percentage changes for all five FPS metrics in both windows.

```powershell
python tools/compare_benchmarks.py build/beta1-frame-recorder build/new-benchmark
```
