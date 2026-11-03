# Ripper3 native LTX runner

Use this checkout's `.venv`. GLM stays serving; its draft model and dedicated RTX
worker must be off. The runner checks this before every job. No network service
or automatic boot service is installed by these scripts.

## Optimized one-shot run

```bash
.venv/bin/python ripper3_je_native.py \
  --output-dir /absolute/new/output-directory \
  --image /absolute/reference.png --story-plan /absolute/plan.json \
  --num-frames 985 --chunk-frames 185 --carry-frames 25 --seed 42 \
  --compile --decoder-mode chunked_compile
```

The plan contains `script`, `dialogue` and `prompts`. Joining the dialogue must
reproduce the script; there must be one caption per generated window. Frame
counts must be `8*k+1`. Current output is 576×1024 at 24 fps. Do not reuse an
output directory containing a video or result report.

The runner encodes all prompts in one encoder lifetime, retains the streaming
transformer and auxiliary models, and progressively encodes completed frames
through a bounded CPU queue. Only active windows/carry remain in VRAM. Completed
audio still accumulates in CPU RAM until final muxing; prompt context also grows
with the number of planned windows. This is not a claim of constant total RAM.

`--decoder-mode chunked_eager` selects the previous decoder path.
`--no-keep-auxiliary-models` selects per-call auxiliary model lifetimes. Omit
`--compile` to disable transformer compilation. CUDA graphs are not enabled for
the CPU-offloaded transformer. `combined_compile` is available for experiments
but has a larger memory budget and is not the default.

## Persistent scene worker

Start from the checkout on Ripper3:

```bash
.venv/bin/python ripper3_ltx_worker.py serve \
  --queue /home/anagrath/.local/share/ltx-worker/queue \
  --compile --decoder-mode chunked_compile
```

Keep that process alive in a terminal or managed session. In another terminal,
write a job JSON with absolute paths:

```json
{
  "output_dir": "/absolute/new/output-directory",
  "image": "/absolute/reference.png",
  "story_plan": "/absolute/plan.json",
  "num_frames": 985,
  "chunk_frames": 185,
  "carry_frames": 25,
  "seed": 42
}
```

```bash
.venv/bin/python ripper3_ltx_worker.py submit \
  --queue /home/anagrath/.local/share/ltx-worker/queue --job /absolute/job.json
.venv/bin/python ripper3_ltx_worker.py status \
  --queue /home/anagrath/.local/share/ltx-worker/queue
.venv/bin/python ripper3_ltx_worker.py stop \
  --queue /home/anagrath/.local/share/ltx-worker/queue
```

The worker handles one job at a time and holds its reusable models across jobs.
It caches up to 512 MiB of complete unenhanced prompt embeddings in CPU RAM;
identical text batches can skip the text encoder. Changed prompts still need
encoding. Model/decoder settings are fixed for the worker lifetime.

Queue files transition from `.job.json` to `.running.json`, then a preserved
`.request.json` plus a `.done.json` or `.failed.json` receipt. Requests left
`.running.json` after a crash need inspection and are never automatically
retried. A render failure stops the worker to avoid reusing suspect CUDA state.
Stop requests and SIGTERM/SIGINT finish the current scene, then dispose retained
models. Stop the worker before attempting to reload GLM drafting onto the RTX.

Final review media belongs in `/mnt/mongo-raid/shared/Videos`, visible on the Mac
as `/Volumes/Shared/Videos`. Working renders and caches stay on RAID0. Model
weights and generated media do not belong in Git.
