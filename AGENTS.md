# 2NX2Navi

Native Windows recompilation of Ocarina of Time with a Vulkan renderer. Follow the user's current scope and decisions. The [maintained plan](docs/settings-removal-plan.md) records approved requirements, current work and evidence; historical documentation and existing code are references, not proof of the best design.

Existing project documentation is likely to contain substantially outdated or incorrect information. Until a document has been rewritten from the ground up and verified against the current implementation, use it only as a lead for investigation, not as a reliable description or basis for technical decisions. Read the relevant code to verify its claims; use build or runtime evidence where code alone cannot establish behavior. Even replacement documentation needs rechecking when the implementation changes. User-approved requirements describe intended behavior, not proof that it is implemented.

## Engineering judgment

Evaluate changes against the intended player experience and the complete affected system. For NEXT resource work, the objective is less processing and waiting across the connected path to correct GPU work, with reliable pacing and memory use. Reuse, caching, tracking and parallelism are tools, not objectives.

Consider downstream benefits and introduced costs together. A locally more expensive check can enable much larger savings elsewhere. Fewer operations, more cache hits or an isolated timing improvement do not establish a system benefit. Equally, every component of an unfinished replacement need not independently improve FPS.

Performance is not an unconditional veto. Useful features may have acceptable costs, and a coherent rework may temporarily regress. Explain the tradeoff, remaining dependencies and how the result will be evaluated against the user's priorities. A cleaner design alone does not demonstrate future gains.

Preserve required behavior, not existing implementation choices. Stage boundaries, algorithms, dependencies, address models and testing methods may change with evidence. Distinguish user requirements from hypotheses and provisional choices; do not silently promote the latter into permanent constraints. Resolve changes to agreed product requirements with the user, while handling routine choices within authorized scope autonomously.

Keep each work packet coherent: identify the connected change, affected consumers, dependencies and completion evidence. The current NEXT phase completes the agreed resource-management replacement across ownership, observation, preparation/admission, binding, reclamation and lifetime before performance optimization. Ordinary CPU-sampled textures are the first migration lane, not the completion boundary for the foundation. Complete the connected consumers and remove the old responsibilities they replace; do not add a shortcut while leaving the old authority underneath. Use correctness checks and fix discovered bugs during this phase. Existing performance captures inform context, but isolated FPS gains do not select or gate foundational work; resume performance optimization after the foundation is integrated.

## Evidence and coordination

Verify consequential claims against code and observations. Distinguish measured results, inference and expectations. Validate affected behavior with appropriate checks; existing tests and reference implementations are useful but fallible. Investigate disagreements rather than treating either side as automatically correct.

Keep recoverable checkpoints and reproducible build/settings identities. Distinguish experimental results from readiness for the player.

Use model capacity efficiently. Previous work used Astra too often for tasks Sol could likely complete, consuming too much of the user's AI allowance. Use Astra 6.2 Medium for high-level planning and coordination, and Astra 6.2 High for decisions that cross systems and are costly to reverse. Delegate every individual task, including research, implementation, debugging, testing and focused review, to Sol 6.2 High. Define the deliverable, relevant context and file ownership. Run independent tasks in parallel when it saves time.

Astra owns integration decisions, reviews Sol's evidence and checks compatibility across tasks. Never repeat a delegated investigation. If evidence is missing or conflicting, send Sol a specific follow-up question. Get the user's approval before substituting a model or reasoning effort when the specified one is unavailable.

For game runs, follow the user's current authorization and settings, use the validated harness/routes where applicable, and coordinate desktop access. Protect player saves and installs. Distinguish warm and cold results; retain inconvenient results rather than filtering them to support a conclusion. Share compatible inputs, use compact/compressed records and clean temporary artifacts after review while retaining useful evidence.

## Repository care

Keep keys private and game data out of tracked files. Tools read key files themselves; extracted/game-derived outputs belong in ignored storage. Preserve generated-content marking and run `python tools/check_content.py --staged` before commits. Checker heuristics may be corrected without weakening that boundary.

Make durable changes reproducible from maintained sources, generators or explicit transformations. Track dependency versions/patches and preserve licenses and attribution. The current copying policy permits MIT, BSD, ISC, zlib and Apache-2.0 code, not GPL code. Reproducibility does not require freezing a dependency or generator indefinitely.

Discover compatible toolchains from configuration/manifests. Reuse build products only with matching relevant inputs. Keep authoritative shader sources; current source-control policy excludes generated shader binaries but permits prepared distribution artifacts. Coordinate changes to shared ABI/layout contracts across their consumers.

## Maintaining guidance

`AGENTS.md` is the sole repository agent instruction file. Keep it concise. Replace superseded guidance in place; put detailed procedures, requirements and evidence in their responsible documents rather than accumulating history here. Revisit constraints when their rationale changes.

The public name is **2NX2Navi**. Current development uses branch `NEXT` and `NEXT - N` versions; comparison checkpoints are specified in the plan. Write code, documentation and plain-sentence commit messages in English.
