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
For repeated new captions, add `--prepared-weights-cache-gib 32` to `serve`.
This retains packed text-encoder and projection weights in CPU RAM, while each
encoding call still releases its temporary GPU buffers. The measured retained
weight footprint is about 30.1 GiB, separate from prompt embeddings and video
models. The flag defaults to zero (disabled). Keep the same worker alive between
jobs: restarting discards this RAM cache and requires another cold preparation.
Only the built-in CPU RAM streaming path without LoRAs is supported. Restart
after changing model files or code; source changes detected by the encoder fail
instead of silently using old prepared weights.

A six-caption GPU check on November 2 measured 33.272s cold versus 6.435s for
changed captions with prepared weights retained. Encoding the original captions
again took 6.283s and produced bitwise-identical conditioning tensors. These
measurements bypassed the exact-caption embedding cache entirely; they measure
prompt preparation, not video generation or total render throughput. Existing
CUDA copy-stream/event ordering continues to overlap block transfers with compute.

It caches up to 512 MiB of individual unenhanced caption embeddings in CPU RAM.
Changing one caption only encodes the changed caption; other captions, including
reordered/duplicated ones, reuse their exact cached contexts. Returned tensors
are independent copies. The LRU budget covers retained CPU entries; the current
encoder batch and requested output contexts have their usual transient storage.
Image/seed-dependent enhancement and unknown conditioning options bypass the
exact-text cache. Model/decoder settings are fixed for the worker lifetime; restart
the worker after modifying model files, encoder settings or installed code.

Queue files transition from `.job.json` to `.running.json`, then a preserved
`.request.json` plus a `.done.json` or `.failed.json` receipt. Requests left
`.running.json` after a crash need inspection and are never automatically
retried. A render failure stops the worker to avoid reusing suspect CUDA state.
Stop requests and SIGTERM/SIGINT finish the current scene, then dispose retained
models. Stop the worker before attempting to reload GLM drafting onto the RTX.

Final review media belongs in `/mnt/mongo-raid/shared/Videos`, visible on the Mac
as `/Volumes/Shared/Videos`. Working renders and caches stay on RAID0. Model
weights and generated media do not belong in Git.

## Prepare an extension's prompts ahead of generation

Explicit prewarm jobs run serially in the same queue, with a distinct receipt:

```json
{
  "action": "prewarm",
  "story_plan": "/absolute/next-scene-plan.json"
}
```

Submit that JSON using the same `submit --job` command. It needs no image or
output directory: unenhanced text encoding is independent of those inputs. The
worker caches the captions in CPU RAM, releases temporary GPU outputs, and
preserves default Torch CPU/CUDA RNG states around preparation. It rechecks the
GLM draft/RTX-worker state before prewarming. A later render of those exact
captions can use the cache even when other captions in its batch differ.

Optional `serve --lookahead 1` prepares the next queued job and one successor at
job boundaries, before generation starts. Values 0–4 are supported; default 0
leaves automatic lookahead disabled. The captions are deduplicated and encoded
together in one encoder lifetime. Malformed future input is skipped by lookahead;
its normal queue claim later records the failure. This runs on the GPU only when
no render is active; it does not promise simultaneous background encoding while
video denoising saturates the same GPU. Stop requests finish an active prewarm
or scene before disposing models.

Preparation time appears separately in `worker.json` under `last_prewarm` and
in explicit prewarm `.done.json` receipts. Include that time when reporting total
cold throughput; moving encoding ahead of a render does not make that work free.
The next scene's prompt can be ready early, but continuity state still depends
on the preceding scene's result. Queuing a new native job currently starts a new
image-conditioned scene; it does not automatically continue a prior latent state.

## Completed-scene checkpoints (opt-in)

`serve --scene-cache-dir /absolute/scratch/scene-cache` enables immutable completed
scene artifacts. Exact matching scenes can reuse their completed MP4 without
invoking model generation. Cache identity includes checkpoint content digests,
renderer source digests, installed dependency versions, reference-image content,
all prompts/dialogue/script, seed, frame/window/carry settings, sampler/precision,
compile/decoder settings and geometry. Model-file hashes are catalogued with
size/inode/mtime/ctime fingerprints so they are hashed once, not once per render.
The first opt-in initialization may take time to hash the large model files.

Successful generated scenes are stored with their result report and a video
SHA256. Reuse verifies the stored video digest, copies it to a fresh output
directory, and verifies the copied bytes. Corrupt/incomplete artifacts do not hit.
The cache directory is managed scratch storage and has no automatic quota or
cleanup policy; remove old scene-key directories there explicitly when needed.

Optional render-job `previous_scene_key` requires an intact prior checkpoint and
becomes part of the key. This records a sequence dependency: changing an earlier
scene or its image invalidates dependent keys. It does **not** supply hidden
state, choose a last reference frame, or establish seamless continuation. Provide
the intended scene reference image explicitly. Each scene must be a separate
worker render job for earlier completed scenes to be independently reusable.
Changing the ending of one six-window job still invalidates that whole job.

Reused results explicitly report `artifact_reuse: true`,
`model_generation_performed: false`, `generation_wall_s: 0`, and the original
`source_render_wall_s` separately. Artifact copying must never be reported as
video-model generation throughput or as a new quality review.

Partial internal-window resume is deliberately not implemented. It requires both
denoise stages' carries, ancestral sampler call indices, shared generator state,
decoder overlap/audio seam state and compatible tiling. CPU checks verify cache
invalidation and artifact semantics, but do not establish equivalence for that
more complex continuation protocol. GPU integration and timing of these new cache
features remain to be run separately.
