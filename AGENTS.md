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
