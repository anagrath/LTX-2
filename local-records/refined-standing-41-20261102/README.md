# Refined standing JE, 41 seconds

Approved reference: Shared/Images/JE - Refined Standing Reference v2.png.
Output: Shared/Videos/JE All Ages - Refined Standing - 41 Seconds.mp4.
Opened and played on the Mac immediately after completion. No new ASR/full-media
decode review was run; the owner requested direct playback. Successful generation
and playback commands do not establish subjective appearance/speech approval.

Same102-word FAQ, native LTX2.5 A/V,985frames/24fps (41.0417s),576x1024,
six185-frame windows,25carry,seed42. Updated reference and matching appearance
caption retain slender thighs, rounded hips and subtle abdominal definition.

The persistent worker used compile, chunked_compile decoder and32GiB prepared
CPU weights. A cold prime encoded six old captions in33.257s, retaining
32,277,182,208bytes. Six fresh appearance captions then took6.493s, with three
prepared-weight hits and no evictions; no exact-caption hits during that step.
The render subsequently reused those new caption embeddings in0.010s.

Render-only221.412s; previous standing41s render221.555s (essentially unchanged).
Fresh-caption preparation plus render, including queue/CLI overhead:230.747s.
Worker startup, cold prime, fresh captions and render:272.352s. Previous standing
driver total273.273s. Thus a cold start showed no meaningful overall improvement;
the benefit is avoiding repeated weight preparation on later changed captions.
The extra cold prime in this experiment measures that reuse explicitly. Separate
earlier loader debugging and direct GPU correctness tests are not included in
these driver totals and made the owner's actual elapsed wait longer.

These creative versions differ in reference/appearance captions, so these are
observed sequential overall timings rather than a controlled video-quality or
per-feature throughput comparison. Direct encoder benchmark and bitwise cold/warm
conditioning equality are recorded in ../prepared-weights-20261102.

Worker1996568 remains alive with retained CPU weights/models. Queue:
/home/anagrath/ai/experiments/ltx-setup-20261102/refined-standing-41/queue.
Use ripper3_ltx_worker.py stop against that queue before restoring GLM RTX drafting.
GLM drafting and its CUDA worker remain off; GLM coordinator continues serving.
