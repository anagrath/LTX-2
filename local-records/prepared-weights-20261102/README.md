# Prepared prompt weights on Ripper3

The optional CPU RAM cache retains packed immutable encoder blocks, non-block
weights and projection weights across encoding calls. Each call builds fresh
providers/GPU buffers, so provider teardown does not invalidate cache ownership.
The 32-GiB cache retained 32,277,182,208 bytes (about 30.1 GiB), with no evictions.
The cache is process-local; workers must remain alive to retain it. It is neither
a disk KV cache nor a cache of video denoising state.

The direct GPU encoder benchmark bypassed the caption embedding LRU. Six original
captions took 33.272s cold, six modified captions took 6.435s warm, and re-encoding
the original captions took 6.283s. All outputs were finite. Original cold/warm
conditioning tensors were bitwise equal (maximum absolute difference zero).
Changed captions intentionally differ; they are not expected to produce equal
conditioning. This is a sequential single-process preparation benchmark, not
an isolated video throughput comparison. CUDA peak allocation was 10,269,617,152
bytes for each encoder call. The existing CUDA block-copy stream remains in use.

The first GPU attempt exposed missing tuple/list path normalization in the new
non-block loader. It failed before producing outputs. The corrected implementation
normalizes paths before the actual safetensors loader; a real two-shard regression
covers equivalent tuple/list cache hits and changed-shard invalidation. All 46
CPU tests passed with CUDA hidden before the successful GPU retry.

Only built-in CPU RAM streaming without LoRAs is supported. Source fingerprints,
dtype and transformations participate in keys; the prompt encoder rejects model
file changes and requires restart. The budget covers retained entries, not active
borrowers or copies awaiting completion. Existing provider teardown and copy-event
ordering release GPU storage without freeing an in-flight CPU source prematurely.

Evidence: `benchmark.py`, `gpu-result.json`, and `cpu-tests.log`. Model weights,
credentials and generated media are not included.
