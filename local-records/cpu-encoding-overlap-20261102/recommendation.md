# CPU encoding overlap feasibility

Actual pinned LTX2.5 Gemma encoder and embeddings processor, CPU FP32, native
1024-token padding,316 valid tokens, one1660-character caption repeated twice.
Child processes were CUDA-hidden, nice+10, inter-op1, no model quantization.

16-thread idle-GPU run: load8.687s; first caption50.947s; warm caption50.593s.
32-thread run concurrent with GPUcreative: load7.484s; first caption27.620s;
warm caption33.404s. Both stages produced finite FP32 video[1,1024,4096] and
audio[1,1024,2048]. PeakRSS~71.9GiB. Different contention prevents attributing
the difference solely to thread count. CPU outputs are not verified equal to
GPU BF16 conditioning, so an optional CPU producer needs quality verification.

The root's current GPUstage profile changes priority: of41.395s for six captions,
26.155s encoder load +4.692s projection load dominate; forward math totals~3s.
CPU background next-job encoding is feasible, but reproduces33s of CPUwork per
caption to hide a source-loading problem. It must not be advertised as a faster
first-job path: GPUalready batches all six captions efficiently.

Recommended next change: preserve immutable packed CPU block weights between
encoder builds while continuing to release GPU provider buffers and non-block
weights. StreamingModelBuilder._build_pinned_source currently reallocates/copies
every block, then PinnedWeightSource.cleanup clears CPU blocks on teardown.
A shared packed-source owner with fresh per-build PinnedWeightSource dictionaries
can retain CPUstorage without retaining GPUweight pools across video generation.
Key by checkpoint content/stat identity, dtype, key mappings, policy/module ops
and LoRA configuration; isolate mutations and release on worker exit.

Do not simply enable ModelRegistry(cache_weights=True): _build_pinned_source
sets block_sd.sd[key]=None while filling buffers, so retained registry state
would be destructively emptied. Projection source staging may save additional
load time but must not keep GPUweights accidentally.

No production source changes, new GPU actions, worker actions or renders were
performed for this investigation. All code/results live in this experiments
directory; GPUthroughput impact and cache implementation remain unverified.
