# Presentation handoff

`runtime/gfx_present.c` separates source-image consumption from display presentation. With the Streamline swapchain, the presenter first composes an immutable output on the graphics queue. `runtime/vk_backend.c::present_main` then publishes the game's completion timeline and releases its source images. Acquisition, the final output copy and display presentation use a separate queue.

The change does not enable early replay or advance game simulation by a frame. There remain two output slots and a bounded pending presentation job. Backpressure is applied when a slot is still in use. A device without the spare queue uses the original direct path.

For a native swapchain without frame generation, composition goes directly into the acquired swapchain image and presentation still uses the separate queue. The swapchain already protects that output until its image is reacquired, so making another immutable copy would be redundant. This path retains acquisition backpressure before composition; it separates presentation execution rather than fully isolating window-system image availability. An initial screen of the version that always copied the output measured lower no-FG throughput, motivating removal of that redundant copy. Fresh controls later established substantial run-to-run variation as well; the initial difference cannot all be attributed to the copy.

## Ownership and synchronization

- Graphics-family queue 0 handles game rendering and composition. Queue 3 handles display copies and presentation. Queues 1 and 2 remain reserved for the FSR presentation backend.
- Each output slot owns its image, copy command buffer and copy-completion fence. A graphics timeline signals that composition is ready; the display copy waits for that value and for swapchain acquisition.
- Swapchain render-done semaphores belong to swapchain images, not frame slots. Reacquiring an image protects reuse of its presentation semaphore.
- HUD-less color and UI alpha have two presentation snapshots. Depth and motion are copied by the frame-generation SDK on the composition command buffer. The game's sources may be reused after composition; the presentation snapshots may not.
- Streamline's input-processing completion semaphore and value are captured after each present. Reusing that slot waits for this retirement as well as the output copy. If a running SDK cannot provide retirement state, the presentation queue is drained before reuse. The SDK's default queue-blocking mode remains enabled on the presentation queue.
- A swapchain rebuild flushes its pending flip and uses the interposed device-idle function before destroying SDK state or clearing retirement handles. Queue locks are acquired graphics first, presentation second. Ordinary game drains deliberately stop at source consumption rather than waiting for display completion.
- The alternate NGX presenter copies each pending job before releasing its producer slot and uses the presentation queue for output copies and presents. Its copied-timeline publication happens only after successful submission.
- FSR retains its SDK-owned display and acquisition queues. Preparation consumes the current raw depth and motion on the composition command buffer; subsequent interpolation reads SDK intermediates. Its existing graphics-queue ordering protects input reuse.

Changing an upscaler or resolution must not free consumer-facing snapshots from the replay thread. Presentation-slot allocation changes occur on the presenter after retirement. Device idle alone does not protect a consumer that has not yet been submitted.

## Diagnostics

`NX2NAVI_PRESENT_SPLIT=0` selects the original direct path for a local diagnostic. The default enables the handoff whenever a separate queue is available. This is independent of `replay_async`, which remains off by default.

`NX2NAVI_PRESENT_TIMING=<path>` optionally writes buffered TSV records containing the game frame ID, queued time, flip-call start/end times and whether the host present succeeded. These are host delivery timings, not physical scanout or input-to-photon latency. A comparison should verify rendered-frame throughput, successful host delivery and bounded queue age together. SDK counters separately establish that frame generation is active.

Pacing continues to use present-call start times. Delivery accounting uses successful return times independently. Feeding driver-call duration back into the next pacing target changes the pacing policy and can add unintended delay.

The bridge ABI is 4. Ship the matching `nx2navi_dlssg.dll` with the executable; the host rejects a bridge missing the input-retirement interface.

## Validation scope

The unchanged WIP 9 control reproduces Vulkan validation errors inside the SDK's fake-swapchain/pacer path. The application continues to use the documented present image layout. A missing present-ID extension dependency exposed by validation is corrected during device creation. A full synchronization-validation attempt also crashed inside the validation-layer DLL; that attempt is excluded from performance results. These observations do not establish that the remaining SDK validation errors are harmless.

Use matched routes, settings and source saves for performance comparisons. Keep pipeline waits in the measurements and distinguish partial startup pipeline residency from a fully warm baseline. Review camera directions and images, preserve compact timing records and delete temporary captures after review.

Runtime trials cover native presentation and Streamline 2X on the test machine. The FSR and alternate NGX paths received a code/lifetime audit but were not exercised in those runtime trials.

## Measured results (2026-11-17)

The performance comparison used the unchanged WIP 9 control and candidate executable SHA256 36818a5947311f1d1a2bf0223571d6140f7f0aca15455229e0ae99d9c4525aa7. All runs used DLSS Quality, unlimited real-frame rate, early replay off, dynamic resolution off, immediate presentation, and the full overlay. Frame generation was either off or DLSS 2X. The actual game window stayed focused. Personal saves were isolated from the disposable benchmark profiles; game and cache files were shared. See the retained build manifest for executable and DLL identities.

The final comparison contains 24 accepted runs: two runs per build and mode in Kokiri Forest and Kakariko Village, and one per build and mode in Hyrule Field and Castle Town. Values below are medians of per-run results. Lows are reciprocal frame-time percentiles. They describe real rendered frames, not generated output frames.

| Scene / FG | Average FPS control -> candidate | Change | 5% low change | 1% low change |
|---|---:|---:|---:|---:|
| Kokiri / off | 86.83 -> 84.95 | -2.17% | -3.38% | -1.74% |
| Kokiri / 2X | 72.77 -> 78.68 | +8.11% | +4.61% | +3.96% |
| Field / off | 69.03 -> 69.49 | +0.66% | +4.53% | +1.21% |
| Field / 2X | 62.80 -> 69.86 | +11.23% | +14.52% | +6.31% |
| Castle Town / off | 93.49 -> 92.85 | -0.69% | -3.32% | +2.08% |
| Castle Town / 2X | 78.38 -> 83.58 | +6.62% | +7.74% | +10.97% |
| Kakariko / off | 77.32 -> 76.93 | -0.51% | +0.14% | -0.67% |
| Kakariko / 2X | 67.34 -> 72.96 | +8.34% | +7.91% | +8.91% |

The FG improvement appeared in all four scenes and all reported full-route low percentiles. Castle Town's six-turn stress window improved by 8.48% in average FPS and 3.21% in the 1% low. The no-FG results are close, but the small negative pooled Kokiri result remains in the record. The later adjacent Kokiri no-FG pair improved average FPS by 0.74%, while the earlier candidate sample was slower. No broad, repeatable no-FG slowdown was established; these small samples do not rule out small regressions.

The candidate recovered about 42% of the original average-FPS loss from enabling FG in Kokiri, 56% in Kakariko, and 34% in Castle Town. Field's FG average became indistinguishable from its no-FG average at this sample size. This does not establish zero FG work. Expressed as incremental average frame cost, the FG overhead fell approximately 58%, 63%, and 42% in Kokiri, Kakariko, and Castle Town respectively. These are different measures and must not be described interchangeably.

The independent raw-record audit passed all 24 accepted runs. It recomputed average/median/lows, checked contiguous route frame IDs, route-clock agreement (maximum discrepancy 10.6 ms), actual focus, process overlap, fixed settings, clean shutdown, and actual 2X output with status zero during scored FG samples. No fatal, device-lost, or explicit frame-generation SDK failures appeared in accepted logs. Earlier runs did not record the process exit code; their clean shutdown and logs remain available. The later instrumented runs exited with zero. Common SDK startup warnings occur in both builds and are retained separately from failures.

Candidate host-present return timings account for every scored application frame with successful, monotonically ordered presents. Their average agrees with the app average within 0.004 FPS. FG queue-to-return p99 ranged from 0.349 to 0.371 ms, with no growing queue age; the largest FG observation was 1.088 ms. No-FG has a longer tail: Castle Town p99 was 5.949 ms and maximum 14.118 ms, without increasing end-of-route queue age. These are CPU host-present return measurements, not physical scanout, generated-frame timing, or input latency. The untouched control has no matching delivery recorder, so there is no claimed control-versus-candidate host-latency comparison.

All six Castle Town directions matched the reference within 0.23 degrees and passed visual review without lock-on deviations. Kakariko completed its intended rotation. Route endpoint images were also reviewed. A separate DLAA, FG 2X, 60 FPS smoke check exercised the F1 pause/resume and fullscreen/windowed transitions; its results are excluded from the benchmark comparison. Pipeline waits stayed at or below 11.67 ms total per route. Startup reported partial pipeline-library residency, so these are descriptive matched-route comparisons rather than a strict fully-warm baseline. No pipeline cost was subtracted from scores. Superseded candidates, diagnostic runs, and the validation-layer crash are excluded explicitly and preserved separately. The validation-layer crash must not be reported as a successful validation pass.

Local evidence: build/presentation-tests/comparison.json, comparison.csv, scorecard.txt, independent-audit.json, per-run compressed frames/delivery/logs, run metadata and camera verdicts. report.py refreshes the scorecard; audit.py independently verifies accepted compact records. Temporary images were removed after visual review.
