# NX2navi

Static recompilation of *Ocarina of Time* (Switch 2, ARM64) into a native Windows x86-64 exe with a Vulkan renderer. Identify the exact game revision from the selected build's inputs and manifest, not an inherited version label. Current NEXT requirements: [docs/settings-removal-plan.md](docs/settings-removal-plan.md). Architecture, building and launcher docs describe existing behavior and workflows but may be outdated: verify them against code and build evidence. Preserve required correctness semantics; those descriptions do not mandate retaining their implementation.

## Game content boundary

The repository's additions are all rights reserved (`LICENSE-ADDITIONS`). It never holds game data: game data lives outside it and every derived file stays in ignored folders.

- The user's game and key paths are in `config.local.toml` (ignored). Pass the keys file path to tools; the tools read it themselves. Key values stay out of your output, logs, commands and files.
- Everything extracted or generated goes under `build/` or `generated/`.
- Every file the code generator emits carries the marker defined as `GENERATED_MARKER` in `tools/check_content.py`.
- Run `python tools/check_content.py --staged` before proposing a commit. It is the same check CI runs.

## Reusing code

`refs/` holds the pinned checkout the build reads (`tools/fetch_refs.sh` recreates it). Read-only.

- Copy only MIT, BSD, ISC, zlib or Apache-2.0 code, keep its header, and add a row to `THIRD_PARTY_NOTICES.md`.
- GPL code is never copied: write our own.
- `tools/vendor/` stays byte-identical to its source. Extraction compatibility belongs in `tools/extract.py`; other dependency adaptations belong in their responsible wrapper or checked-in patch set applied to an ignored build copy (for example `tools/xlate/patches` through `tools/build_xlate.py`). Do not change pinned source trees in place.

## Code

- Build generated C with the supported GNU-compatible Clang/UCRT toolchain; `cl.exe` cannot compile its GNU extensions. Discover the existing validated compiler, Ninja and matching dependencies through build configuration or manifests rather than assuming `C:\msys64` exists. `tools/build_game.py` supports explicit toolchain, dependency and Ninja paths. Do not mix incompatible objects, headers or libraries; reuse generated objects only when their input/generator/patch identity matches.
- Reuse validated Python environments/interpreters with the required dependencies. `.venv` (3.12) and `.venv-py314` are historical names, not required locations. Extraction/shader scripts using `compression.zstd` need Python 3.14 or the explicitly supported equivalent; do not recreate environments merely because those directory names are absent.
- Generated C is never edited by hand: change the generator (`recomp/`).
- The runtime's own shaders are checked in as source only (`runtime/shaders/`, `tools/make_dlaa_mask.py`,
  `tools/make_mvobj_fs.py`). `tools/build_shaders.py` builds `runtime/<name>_spv.h` from them, `build_game.py` runs it,
  git ignores the headers and `check_content.py` refuses them in a commit.
- Every lifter change passes the differential tests against Unicorn ([docs/building.md](docs/building.md), "Checks"). After changing `recomp/jumptables.py`, `tests\jumptables_check.py` reports 0 disagreements on 1.2.1 and 1.4.3.
- `recomp/layout.py` alone decides module bases.
- Graphics bugs: [docs/maxwell.md](docs/maxwell.md) lists the Maxwell behaviour the Vulkan path must match, what goes wrong when it does not, and what our runtime does.
- Comments explain behavior and rationale in words that stand on their own. Cite our code by file plus function (optional line hint). Relevant technical sources may be cited in design/research documentation; source attribution for copied code belongs in its license header and `THIRD_PARTY_NOTICES.md` row.

## Fork releases

The public name is `2NX2Navi`. Resource-path development uses branch `NEXT` and `VERSION` labels `NEXT - N`, starting at `NEXT - 0`. Increment N for each subsequent candidate iteration; the spaces and hyphen are part of the label. Legacy is the original reference; use the explicitly selected checkpoint in the maintained plan for each comparison. This convention supersedes Beta/WIP numbering for this development line.

Benchmark with the validated harness and route/save definitions, compact per-frame records, route/camera checks and exact source/build/settings identities. Confirm effective settings and desktop exclusivity; preserve player saves and share compatible immutable inputs/caches safely. Fully warmed acceptance must be established by the startup record; retain partial/cold observations separately without presenting them as warmed comparisons. Compress logs and remove temporary captures/staging artifacts after review. NEXT currently lacks the root workspace's standalone benchmark guide/harness: locate their validated manifests before launching rather than inventing equivalent commands or silently copying unrelated dirty tools.

## NEXT governing objective

Make the complete resource-handling chain deliver correct GPU work with substantially less processing and waiting. Reuse is a means, not the objective. Evaluate connected consequences: a 0.01 ms validity check that avoids 0.001 ms locally can still be worthwhile if its result eliminates 0.3 ms downstream. Conversely, fewer operations or more hits do not establish a net benefit. FPS alone neither accepts nor rejects an unfinished architectural replacement.

- Design authoritative facts, ownership, invalidation and all relevant consumers together. Change stage boundaries or remove stages where this simplifies the complete path; do not preserve the old sequence by default.
- Once per coherent implementation packet, and when evidence changes its design, name the connected chain, enabled downstream changes, introduced costs, dependencies and completion evidence. Delegated tasks must fit that contract; routine edits do not require repetitive planning. Distinguish demonstrated effects from hypotheses. Include contention, memory pressure and required-submission timing.
- Current scope is the ordinary CPU-sampled texture lifecycle through preparation, binding, release, reclamation and reacquisition. Preserve useful correctness contracts, tests and checkpoint recovery; reconsider existing algorithms, including recent NEXT additions.
- Do not divert to unrelated optimizations or leave this replacement unfinished to polish local wins. Necessary correctness fixes and diagnostics must serve this scope. Replace or abandon an implementation approach if evidence defeats its intended benefits; preserving the objective does not require preserving that approach. State why when changing direction. Update the maintained plan in place; do not accumulate review history.

## Writing

Code, comments, docs and commit messages are in English. Commit messages are plain sentences without type prefixes.
