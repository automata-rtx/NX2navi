# Switch 2 movie playback

## Verified in the fork

Commit `10efd8e` is the tested renderer baseline. To reproduce the creation movie in its original context, load the
save made when Link collects the heart container, enter the blue ring, confirm with Navi, then advance the Great
Deku Tree's dialogue. Playing the same clip from the title screen missed the original bug.

The decoder continued delivering pictures while the subtitles appeared over frozen pink and green bands.
The movie upscaling pass picked a larger texture still bound from the preceding 3D scene. The corrected fallback
uses the shader's declared storage writes to choose eligible destinations when reflection is available
(`runtime/vk_exec_draw.c`, `untranslated_dispatch_fallback`). Captures from the story sequence matched successive
pictures from an independent decode. At a 120 FPS cap the player delivered all 7,040 frames of the 60 FPS clip.

## Timing and workload

With frame generation off and the same 60 FPS game cap throughout, the RTX 5090 took 8.81 ms of GPU time per frame
during the Deku Tree dialogue and 0.82 ms during the movie (medians of the sampled windows). Whole-process CPU use
on the Ryzen 7 7800X3D rose from the equivalent of 2.14 cores to 3.25. Playback still averaged close to 60 FPS
during a 40-second restriction to four logical processors, with occasional late frames. These figures cover one
scene on one machine. The CPU measurement includes decoding, filtering, copying and the running game.

At a 30 rendered-FPS cap, the player delivered about 30 consecutive source frames per second and the 60 FPS
movie advanced at roughly half speed. The following changes are proposed for future work.

- Temporarily allow 60 FPS when the game commits to presenting a 60 FPS clip under a lower in-game cap. Keep that
  allowance across queued clips, including decoder teardown between them. Restore the gameplay cap when the
  sequence ends. Preserve the user's saved setting and higher caps.
- Schedule pictures by their presentation timestamps against the playback clock. When rendering falls behind,
  show the latest due picture and advance audio, subtitles and event cues on time. This also covers an external
  30 FPS limiter, which the game cannot raise. Test adjacent clips whose timestamps restart, pausing, skipping
  and returning to gameplay. Check elapsed time and cue ordering across the clip boundary.

## Hardware AV1 decoding

The current path decodes AV1 on the CPU, debands the picture and packs its luma and chroma planes into NV12
(`runtime/ov_movie2.c`, `make_picture`). The guest receives CPU-readable picture data through `ov_acquire`.
The tested RTX 5090 driver exposed the Vulkan AV1 Main 8-bit 4:2:0 decode profile used by the creation movie.
Hardware decoding and its end-to-end efficiency have yet to be tested in the port.

A candidate implementation would query the active adapter's video decode capabilities, keep decoded pictures
on the GPU, apply debanding there and feed the existing subtitle and overlay composition path. Account for the
guest's CPU-visible frame contract and surface lifetime. Retain software decoding for unsupported hardware.
Measure decoding, filtering, transfers and whole-process CPU use separately on the same clip. Compare image
quality, playback time and subtitles with the existing path. Hardware decoding alone does not correct the
30 FPS scheduling issue.

## Separate shutdown observation

One 120 FPS run logged a host access violation during shutdown after the movie finished and the test closed the
window. The trigger is unknown. Reproduce it independently before attributing it to the decoder or frame cap.
