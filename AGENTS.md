# Local LTX work on Ripper3

- Use this project's Python 3.14 `.venv` and pinned `uv.lock`. The CUDA runtime is
  isolated from GLM. Do not upgrade the system driver or alter the GLM target
  architecture/placement for LTX experiments.
- The user chose LTX for joint video and audio generation. Use native LTX audio
  first. Do not silently replace it with a separate TTS model. Qwen3-TTS was
  investigated but is not used in this test.
- Keep GLM serving. Before using the RTX, verify the draft model AND its dedicated
  worker are OFF using the existing draft CLI. Leave them OFF afterward.
- The current test is the "What ages does Elephant Learning cover?" FAQ from
  the Mac marketing records. On November 2 the user authorized rewriting its
  wording as a young adult influencer ("Hey, check this out, guys! I get asked
  ALL the time..."). Preserve factual claims, rather than the old exact words. The user
  selected the beach boardwalk scene, JE's coral top/navy leggings/red ball, and
  the supplied generated voice reference. The current test-case/scene/plan files
  are under `/home/anagrath/ai/outputs/ltx/elephant-reference-20261102/`.
- Source video: `/home/anagrath/Downloads/Generated Video October 21, 2026 - 10_49AM.mp4`.
  Its opening visual frames are faulty; use clean middle/end frames for identity.
  Do not overwrite source media or historical marketing records.
- `ripper3_baseline.py` runs the official distilled pipeline. `ripper3_je_dubit.py`
  experimentally composes the official Dub-It blocks with a separate image and
  audio reference, target length independent of the short reference, and optional
  per-window captions. This composition is not an upstream-supported CLI mode.
  Validate generated speech, appearance and joins; do not claim a perfect voice
  match or human approval from a successful render or ASR result alone.
- Keep inference mode around every custom generation path; otherwise autograd
  retains training buffers and can fill the GPU during text encoding. The CPU
  checks in `test_ripper3_je_dubit.py` cover this and the late-window voice reference.
- All media processing runs locally. Keep HF credentials outside Git. Commit
  code and useful measured results locally; do not push or publish advertising.

## Current rerender direction

The user reviewed Test 1: the reference voice did not match and delivery was too
calm/monotone. They explicitly requested a new version WITHOUT Dub-It, with
influencer-style acting and more playful ball interaction. Use
`ripper3_je_native.py` and `all-ages-native-playful-story-plan.json` for this pass.
Plain LTX-2.5 generates video AND audio; no LoRAs or reference-audio conditioning.
Keep Test 1 and source inputs intact. Deliver new media to the RAID5 share at
`/mnt/mongo-raid/shared/Videos` (Mac `/Volumes/Shared/Videos`), verify its hash,
and preserve GLM with drafting and RTX worker OFF. The user authorized deleting
the previous shared playful render after its validated replacement is ready,
and opening the replacement on the Mac using SSH. Use the
`all-ages-native-athletic-story-plan.json` and `je-boardwalk-athletic-anchor.png`
for the current revision: user requested an athletic, toned, trimmer adult build,
a light top and the same leggings.

## Local performance configuration

The native story runner encodes all window prompts in one encoder lifetime and
holds the CPU-offloaded streaming transformer through all denoising windows and
both stages. The upstream stage residency implementation was extended to retain
streaming wrappers; their bounded GPU pool still streams block weights, with
teardown/dispose at scope exit even on failure. Use `--compile` for upstream
block compilation. CUDA graph capture and reduce-overhead/max-autotune modes are
not enabled with CPU offload. Seven CPU lifecycle/signature checks passed.

The pinned optional `natten==0.21.7+torch2130cu132` wheel was installed into this
venv (no Torch/CUDA changes). Preserve it during environment synchronization:
`uv pip install --python .venv/bin/python --no-deps -f https://whl.natten.org
'natten==0.21.7+torch2130cu132'`. Do not assume NATTEN is faster than the Triton
fallback: measured results belong in `local-records/je-athletic-20261102`.
The original completed native run took 548.808s; the first complete compiled,
resident-streaming athletic render took 291.198s with the Triton decoder fallback.
Prompts/image and compiler-cache state differ, so this is an observed overall
comparison, not an isolated attribution of speedup to individual changes.

Owner next goal: 5-10 minute YouTube videos from the characters, potentially a
children's channel. See `local-records/je-athletic-20261102/optimization-audit.md`.
The NATTEN review render took 284.826s. Its opening has ASR-detected direction
leakage/repetition and a garbled question. The owner explicitly stopped a proposed
correction, asked to view this render, and requested discussion of optimizations;
do not start another render without further direction. Mac QuickTime playback
of the NATTEN file was verified.

## Progressive export for long videos (November 2)

The native runner now exposes `story_result` as a context-managed progressive
A/V stream. Its production path uses `split_decoded_chunks` and the existing
H.264 encoder's bounded one-slot CPU queue instead of collecting all decoded
RGB windows on the GPU. Inference mode and transformer residency surround actual
lazy consumption; generator closure on encoder/setup failure releases residency.
The splitter releases historical frame bodies and clones completed audio to CPU;
the encoder no longer pins its first full frame window for the whole video.
Decoder carry/overlap and crossfade algorithms are unchanged; pending decoded
video/audio tails already clone their small retained regions.

Completed audio is still collected in CPU RAM and concatenated before AAC muxing,
so host audio memory grows with duration (including concatenation peak). Video
export buffering is bounded by window/carry and the encoder's CPU queue; this
is not a claim that all pipeline memory is independent of duration (prompt
contexts/layout metadata still grow). No full GPU render or measured VRAM/speed
comparison has been run for this change. Do not claim a tested 5-10 minute render.

Validation: `.venv/bin/python -m unittest test_ripper3_streaming_output
 test_ripper3_je_dubit` passed 11 CPU checks, including weakref lifetime checks,
real synthetic H.264/AAC encode/decode, frame/audio ordering, existing overlap
blend/audio-crossfade equivalence, independent retained-tail storage, and
residency/inference/cleanup on consumer failure. `git diff --check` passed.

## Fork and November 2 optimized worker

The owner explicitly authorized a GitHub fork, pushing our work, delegated RAM
streaming, and all three subsequent optimizations: decoder compilation, auxiliary
model reuse, and a persistent worker. `origin` is now `anagrath/LTX-2`; `upstream`
is `Lightricks/LTX-2`. Work branch: `local/ripper3-python314-20261102`. Never push
our changes to upstream. Model weights, tokens, and generated media stay outside Git.

Use `RIPPER3-RUNNER.md` for commands and worker job JSON. Native defaults now use
`chunked_compile` decode and opt-in scopes retaining the video/audio decoder and
upsampler; `--no-keep-auxiliary-models` and `--decoder-mode chunked_eager` enable
comparison. `ripper3_ltx_worker.py` provides a local serial file queue, graceful
stop, retained models, and a bounded 512-MiB CPU cache of complete unenhanced prompt
batches. All jobs still check GLM drafting/RTX worker are off. No external service,
boot service, or architecture/placement change was installed.

All 24 relevant CPU tests pass. Two 145-frame GPU smoke jobs in one worker took
99.632s then 33.601s; exact prompt cache hit reduced prompt encoding from 38.301s
to 0.002s. Output decode checks passed. With all changes enabled, the 985-frame JE
benchmark took 231.450s (warm worker, fresh JE prompt encoding 32.466s), compared
with the previous 284.826s NATTEN run. Peak PyTorch allocation was 14,689,660,928
bytes versus 16,480,361,984 before. This is an overall sequential comparison, not
a controlled per-feature attribution. Automatic decode tile sizing changes with
resident models and allocator state; current warm worker chose 120-frame decode
tiles versus the prior 192, so reuse/compilation gains have memory tradeoffs.

## Current JE creative revision and budget A/B

The owner now authorized the next render after settling these changes:
excited rather than playful; slow confident speech; full chest resonance with a
higher speaking pitch, smooth rather than raspy; adult cropped coral top, very
fitted navy leggings, and a lean toned but curvy elephant silhouette. Remove
all stretching directions and quoted/example dialogue from acting descriptions.
Keep the 102-word FAQ's factual claims. The new prompt file is
`/home/anagrath/ai/outputs/ltx/elephant-reference-20261102/all-ages-native-confident-slow-story-plan.json`,
with `je-boardwalk-cropped-confident-anchor.png`. It uses 1369 frames (57.04s),
six 249-frame windows and 25-frame overlap, 576x1024/24fps/seed42. The added time
is intentional to slow speech; raw render time is not directly comparable to
previous 985-frame creative versions.

The owner explicitly requested timing the agent's next speedup. That change
makes single-GPU CUDA AUTO tiling count reusable allocator cache via
`cuda_allocatable_bytes`; live allocations, process cap, model allowance and
compiled safety reserve remain accounted for. Twenty-eight CPU checks passed
with CUDA hidden; four CUDA-budget tests use mocks. A matched short GPU A/B is
running: detached baseline worktree at16835ab imports old ltx-pipelines via
PYTHONPATH, while fixed uses this checkout. Same145-frame image/plan/seed, fresh
serial worker per condition, first job warmup plus two repeated jobs. Baseline
runs first; results are sequential, not randomized. Only after both succeed does
the fixed worker run the revised57-second creative. Driver/logs under
`/home/anagrath/ai/experiments/ltx-setup-20261102/decode-budget-ab/`.

The matched budget A/B completed: render-only final repeats25.594s baseline vs
24.586s fixed (~3.94% less elapsed), with more VRAM used for larger stable tiles.
Earlier repeated jobs32.950s vs24.536s include shape/compiler/allocator warmup;
do not advertise that average as steady-state speedup. Evidence is in
`local-records/decode-budget-ab-20261102/`. Both145-frame final clips passed full
A/V decode. Revised57.04s creative completed in296.479s and passed1369-frame decode
and contact-sheet review. ASR no longer detects opening direction leakage; it
omits one "we" and repeats "that" near the ending. The review video is
`/mnt/mongo-raid/shared/Videos/JE All Ages - Confident Slow Voice.mp4`.
A/B workers stopped after completion. The owner now delegated individual prompt
caching, safe queued extension prewarming and conservative checkpoint/reuse
implementation; no new GPU tests/renders authorized for that coding task yet.

## Latest owner review and CPU overlap experiment

The49.04s natural-cadence video rendered255.261s; startup+explicit prewarm+render
was303.773s. Delivered directly per owner: do not delay future requested review
playback for extra media checks. Owner then requested41s, no stretching, standing
talking instead of awkward walking, fuller covered bust and slimmer legs. New
reference/plan are `je-boardwalk-standing-41-anchor.png` and
`all-ages-native-standing-41-story-plan.json` in the same reference workspace.
Standing41.04s video: render221.555s, prewarm41.395s, complete pipeline273.273s.
It is in Shared/Videos as `JE All Ages - Standing - 41 Seconds.mp4` and was opened
on the Mac without new decode/ASR checks as requested. Do not claim subjective
quality approval until the owner reviews it. Both render drivers stop workers
after completion; GLM draft/RTX worker remain off.

User correctly observed serial prewarm only moves work earlier. Do not describe
it as true compute overlap or subtract preparation from end-to-end comparisons.
Per-caption CPU LRU, explicit and serial lookahead prewarm, and opt-in completed
scene artifact reuse are committed52d1c41. Whole-scene artifact reuse is not
partial-window continuation and reports zero model generation explicitly.

Actual CPU FP32 benchmark on3990X:16threads idleGPU50.59s/warm caption;32threads
with simultaneousGPUwork33.40s/warm caption (first27.62s),~71.9GiB peakRAM.
Not a controlled thread-count comparison because contention differs. GPU timing
logs show six-caption encoder forward2.262s and projection0.735s, while encoder
load26.155s and projection load4.692s dominate. No backgroundCPU production
encoding was enabled; benchmark outputs have not had GPU-conditioning/quality
equivalence verified. Next targeted proposal is immutable packed CPU source-weight
reuse and fresh per-build GPU buffers, avoiding repeated staging and preserving
GPU inference. Do not simply switch streaming ModelRegistry cache_weights on:
pinned-source construction mutates source entries toNone while consuming them.
Benchmarks/recommendation are in local-records/cpu-encoding-overlap-20261102.

## Prepared prompt weights and current 41-second reference

The owner authorized retained prepared weights, GPU timing and the next41s JE
video. `ripper3_ltx_worker.py serve --prepared-weights-cache-gib 32` now retains
immutable packed encoder/projection CPU weights while releasing per-call GPU
buffers. Keep this worker alive across jobs; prior experiment drivers stopped
workers and discarded their RAM state. Do not call this a disk/KV/video-state
cache or claim serial prewarm is parallel compute. Built-in CPU RAM streaming,
no LoRAs only. Existing CUDA block copy stream/events already overlap transfers
with queued compute; no new provider/synchronization architecture was introduced.

46 CPU tests passed. A corrected real safetensors tuple-path regression preceded
the GPU success: direct encoding bypassing caption LRU took33.272s cold,6.435s
for changed captions using retained weights,6.283s re-encoding originals; original
cold/warm outputs were bitwise equal. Retained footprint32,277,182,208bytes.
Evidence: local-records/prepared-weights-20261102. First cold preparation remains
necessary; report it separately and include it in full cold totals.

Current approved image is `je-refined-standing-reference-v2.png` and plan is
`all-ages-native-refined-standing-41-story-plan.json` in the reference workspace.
Preserve102-word FAQ, standing/planted feet, natural excited confident cadence,
smooth chest resonance with medium-high pitch,985frames/185chunk/25carry,
576x1024/24fps/seed42. Appearance changes retain slender thighs, rounded hips and
subtle abdominal definition. No extra speech/stretching or external TTS.
Timing driver: experiments/ltx-setup-20261102/refined-standing-41/run.py (workspace
root). It primes older captions once, times fresh changed captions and render,
and keeps the worker alive afterward. Deliver to Shared/Videos and open/play on
Mac immediately after completion, without delaying for extra media QA.

Refined standing41s completed and opened/played on Mac at
Shared/Videos/JE All Ages - Refined Standing - 41 Seconds.mp4. Render221.412s
(previous221.555s), fresh-caption prep6.493s, warm request including queue230.747s,
cold startup+prime+fresh prep+render272.352s (previous273.273s). Cold total did not
meaningfully improve; subsequent changed-caption preparation benefits from the
retained weights. Owner wait also included earlier debugging/GPU correctness
tests, not counted in render driver totals. Evidence local-records/refined-standing-41-20261102.
Persistent workerPID1996568 remains alive at that experiment's queue. Reuse it
for next jobs; stop explicitly before restoring RTX drafting. No extra ASR/decode
review was done before playback, and owner quality review remains pending.

## Owner's replacement FAQ wording and Fu Lu

Clean-ending41s render completed188.147s including preparation on the existing
warm worker, opened/played as Shared/Videos/JE All Ages - Clean Ending - 41 Seconds.mp4.
The owner explicitly said not to wait for the speech check. Respect immediate
review playback for subsequent media; do not add ASR/decode gates unasked.
After reviewing that video, the owner said everything else was perfect and
supplied a new exact76-word script. It is saved in
all-ages-native-owner-script-41-story-plan.json in the reference workspace.
This supersedes previous99/102-word dialogue. Keep the same approved reference,
voice direction, standing performance,41.04s,6x185/25carry,576x1024/24fps/seed42.
Emphasize ALL as a spoken word, not spelled letters. Do not rewrite the supplied
wording or add a recap. Current experiment: owner-script-41; media should be
Shared/Videos/JE All Ages - Updated Script.mp4 and opened/played on Mac promptly.

Separate dog task completed: female Chinese Crested Fu Lu, sourceIMG_4340.HEIC
in Shared/Images, converted to Fu Lu - Source.png with macOS
sips (original preserved). Native6.04s video greeting: Hello, My name is Fu Lu.
Shared/Videos/Fu Lu - Hello.mp4 opened/played on Mac. Submission32.595s including
5.708s prompt preparation. Same warm worker; no external TTS or image edit.
Evidence records: clean-ending-41-20261102 and fu-lu-hello-20261102.

Owner-script41s completed188.647s including6.298s fresh-caption preparation using
the warm worker. Opened/played on Mac: Shared/Videos/JE All Ages - Updated Script.mp4.
Exact prompt wording was checked; no ASR/decode gate was added. Do not claim
generated speech is word-perfect without owner review. Evidence:
local-records/owner-script-41-20261102. Worker1996568 stays alive at the same queue.

## Restored full opening after owner-script v2

The owner revised the FAQ body to all ages because counting through algebra,
success as young as2years old, and college algebra confidence; keep the existing
If your student ending and close: Do me a favor, go check it out okay?
V2 incorrectly removed the original hook/question. Owner corrected this: only
add this to the original opening, retaining Hey! Okay, check this out guys. I get
asked this ALL the time, what ages does Elephant Learning cover?
Owner also reported stuttering. ffmpeg decoded v2 without errors,985frames/24fps,
but this does not establish smooth perceptual speech/video. Whisper suggests
extra speech near the end and is not ground truth. Stutter modality remains
unconfirmed; owner said render restored opening first and decide afterward.

Current89-word exact script: all-ages-native-restored-opening-41-story-plan.json
in the reference workspace. Same approved reference,voice,41.04s,six185-frame
windows/25carry,seed42,576x1024/24fps. Keep young-student and college-algebra
sentences intact. Final long sentence splits after help them, with closing in
last caption. Experiment restored-opening-41 uses existing worker1996568; deliver
Shared/Videos/JE All Ages - Restored Opening.mp4 and open/play immediately.
No additional ASR/decode playback gate. Prior rejected v2 evidence:
local-records/owner-script-v2-41-20261102.

Restored-opening41s completed and opened/played on Mac at the path above without
additional media QA. Prompt preparation5.858s; exact driver timing and89-word
script in local-records/restored-opening-41-20261102. Do not claim stutter or
word-perfect speech fixed before owner reviews; requested outcome was preview
with opening restored. Same warm worker remains alive.

## Four-window FAQ trial

Owner reported drawn-out speech in windows3/4 and requested merging3+4 and5+6.
Current88-word script all-ages-native-four-windows-41-story-plan.json uses four
prompts: restored opening,all-ages explanation,combined young/college student
statement,full closing. Middle exact wording ends college algebra find confidence.
Use four265-frame windows/25carry to retain985frames/41.04s. Same reference,
voice,standing performance,seed42,576x1024/24fps. Unlike earlier six-window jobs,
time allocation is first11.04s then10s new body per window; owner review needed
for cadence. All sentences fit inside individual windows now.
Completed186.275s render including prep,187.635s submission total; opened/played
Shared/Videos/JE All Ages - Four Windows.mp4 directly on Mac. No ASR/decode gate
or quality approval implied. Evidence local-records/four-windows-41-20261102.
Worker1996568 remains alive with CPU weights and video models retained.

## Preferred output, second take and shared cleanup

Owner called first four-window JE close to perfect, noting funny audio on we cover
from counting through algebra. Requested a second take with same prompts and cleanup.
Take2 uses same input/settings but seed43 (first take42); all4 captions hit cached
embeddings in0.013s. Opened/played Shared/Videos/JE All Ages - Four Windows - Take 2.mp4.
Keep first JE All Ages - Four Windows.mp4 and Fu Lu - Hello.mp4. Eleven other
shared videos were explicitly authorized for deletion and removed, with stale
sidecars. Current visible videos are those three. No new audio-quality approval
or spoken-phrase correctness is implied for Take2 until owner reviews.

Current Images must keep BOTH JE - Refined Standing Reference v2.png AND Fu Lu -
Source.png. The latter was mistakenly included in an archive move, then immediately
restored on owner clarification; workspace alias repaired. Other images archived
at Shared/Archive/Images-20261102-214611 on the same RAID5, not destroyed. Original
IMG_4340.HEIC was already absent in cleanup listing, only its orphan sidecar present.
Evidence/timing/manifests local-records/four-windows-take2-20261102. Worker1996568
remains alive with retained models and weights.

## A/B edit with Google Drive B-roll

Owner requested editorial assembly: B base, B-roll at Really, return on have in
We have seen success (preserve B walking), and A's picture/audio for Do me a favor
closing. Original A/B/Fu Lu stay intact. The first edit mistakenly retained Really
and owner reported fast-forward appearance. Rebuilt from original sources,
removing B11.125–13.25s from picture/audio. Return sourceB21.5417s, cut to A38.25s
after B38.9583s; corrected length39.625s/951frames/24fps. No time scaling/tempo
filter. Explicit QuickTime playback1x was observed after reopening corrected file.
Do not claim the reported perceptual speed issue is resolved without owner review.

Current edited review: Shared/Videos/JE All Ages - Edited A+B.mp4 (replaced failed
export; failed file retained on scratch only). App-use B-roll from verified Drive
file1NvqtYBFnQ0inut33ER1EFhImCOHuiKsQ, Students Using Elephant Learning App.mp4,
two portrait crops from30s and300s. B-roll audio excluded; B voice under footage.
Drive plugin raw download failed413, rclone downloaded read-only. Marketing index
was read for discovery; no Drive write or upload. Full index and source media stay
outside Git. Repro scripts/timelines: local-records/je-ab-edit-20261102.
Shared Images must still keep JE reference AND Fu Lu - Source.png.
