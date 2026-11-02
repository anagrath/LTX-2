# Local LTX work on Ripper3

- Use this project's Python 3.14 `.venv` and pinned `uv.lock`. The CUDA runtime is
  isolated from GLM. Do not upgrade the system driver or alter the GLM target
  architecture/placement for LTX experiments.
- The user chose LTX for joint video and audio generation. Use native LTX audio
  first. Do not silently replace it with a separate TTS model. Qwen3-TTS was
  investigated but is not used in this test.
- Keep GLM serving. Before using the RTX, verify the draft model AND its dedicated
  worker are OFF using the existing draft CLI. Leave them OFF afterward.
- The current test is the exact 94-word "What ages does Elephant Learning cover?"
  FAQ from the Mac's marketing records. Preserve the words and claims. The user
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
