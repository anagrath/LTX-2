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
