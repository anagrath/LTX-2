# Next optimization candidates

Measured completed runs: original native 548.808s; compiled, single text-encoder
lifetime and persistent streaming transformer 291.198s; same athletic input with
NATTEN 284.826s. NATTEN's observed difference is only 6.372s (2.19%), not a
controlled repeated benchmark. Do not promise a fixed NATTEN gain.

1. **Bound output memory for long videos.** `pipeline_output_from_chunks` drains
   the stream into a list; `_decode_chunk_video` explicitly returns frames on
   the latent GPU device. Completed float RGB chunks accumulate until encoding.
   Move completed bodies to CPU or use `split_decoded_chunks` and a streaming
   encode sink. Retain the overlap/carry state needed for the next window and
   preserve audio crossfade and mux ordering. The encoder already provides a
   bounded background encoding path; reuse it before adding another one.
   This targets duration capacity, not an assumed rendering speedup.
2. **Decoder and upsampler reuse.** The video decoder, audio decoder and spatial
   upsampler still build per chunk. Profile their setup versus compute, then
   consider lifetimes compatible with transformer streaming and decoder peaks.
   CPU output offload can free room for retained auxiliary models.
3. **Decode compilation/tile sizing.** Transformer compilation is enabled, but
   DiffVAE is still CHUNKED_EAGER. Upstream offers compiled decoder modes with
   different memory budgets. Benchmark the actual resolution/window length;
   do not assume bigger tiles or compilation improve a short cold run.
4. **Prompt embedding cache and process reuse.** Identical captions could reuse
   embeddings; a persistent worker can avoid initial encoder/transformer setup
   across scene jobs. Unlike GLM KV prefixes these are LTX conditioning tensors.
5. **GPU weight-buffer sizing/overlap.** The transformer already uses asynchronous
   copy/compute coordination and bounded buffers. More GPU slots may help but
   compete with decode memory; profile before changing defaults.

Long-form direction: 5-10 minute YouTube episodes assembled from shorter scenes,
with stable character references, planned dialogue and shot boundaries. At the
current 41.04-second portrait settings, linear render extrapolation is roughly
35-70 minutes for 5-10 minutes. This is not measured long-form throughput or a
1080p landscape estimate. A continuous chained generation also risks identity,
voice, action and narrative drift; CPU offload does not solve those.

No further optimization or render was started after the owner asked to view the
NATTEN version and discuss options. The first priority remains speech control:
the current opening reads an acting instruction aloud, repeats the greeting and
garbles the FAQ question. Preserve the delivered review artifact as evidence.
