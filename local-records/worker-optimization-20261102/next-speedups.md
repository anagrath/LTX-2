# Next LTX optimization: allocator-aware AUTO tiling

The completed all-optimization JE benchmark remains 231.450 seconds. No new
render, GPU test, GPU memory probe, worker restart, job submission, prompt, image,
voice or script change was made in this investigation. The user paused GPU tests
while discussing the script and voice with the parent agent.

## Implemented, awaiting GPU verification

`tiling_config_for_vae` previously used `activation_budget_bytes` for CUDA.
That estimator subtracts **reserved** allocator memory and caps by driver-reported
free bytes. A persistent worker has inactive cached blocks that PyTorch can reuse,
so a warm worker could receive smaller tiles despite having the same live tensors.
The current implementation instead uses the existing `cuda_allocatable_bytes`
estimator for CUDA AUTO plans. The multi-GPU decode planner already uses this
estimator for its fallback budget.

The estimator counts driver-free bytes plus this process's inactive cache, capped
by the configured process memory fraction minus live allocations. It does not
count live model weights as free or assume other processes' memory is available.
CPU/MPS behavior and explicit caller `free_bytes` values are unchanged. Estimated
model weights and the compiled decoder's existing 2-GiB reserve remain charged.

Mocked regression example: on a 24-GiB device with 6 GiB live, both a cold cache
(6 GiB reserved / 18 GiB driver-free) and warm cache (12 GiB reserved / 12 GiB
driver-free) now plan from 18 GiB and choose the same tile. Separate tests cover
other-process usage, memory-fraction caps, explicit overrides, CPU/MPS/Conv paths,
and unchanged weight/safety charges. These are synthetic memory states, not live
GPU measurements or a throughput result.

CPU-only geometry exploration using the pinned decoder's metadata and a
185-frame, 1024×576 window produced the following example plans:

| Assumed allocatable budget | Temporal tile | Spatial tile (H×W) |
| --- | --- | --- |
| 12 GiB | 120 frames | 448×576 |
| 15.4 GiB | 120 frames | 608×576 |
| 18 GiB | 192 frames | 448×576 |
| 20 GiB | 192 frames | 448×576 |
| 22 GiB | 192 frames | 608×576 |

These illustrate how allocator accounting can change the overlap tradeoff; they
do not establish which tile fits the active worker or which is fastest. There
is no measured speedup for this change yet. The already-running worker retains
its imported old code until it is restarted under later user direction.

## Validation

```text
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest \
  test_ripper3_decode_budget test_ripper3_streaming_output \
  test_ripper3_je_dubit test_ripper3_ltx_worker test_ripper3_aux_residency
Ran 28 tests in 0.216s
OK
```

All CUDA calls in the new budget tests are mocked. The suite ran with GPUs
hidden, so validation did not create a CUDA context or compete with GLM/LTX.
Existing optional matplotlib/OpenImageIO warnings remain nonfatal.

## Deferred candidates

- **Resident decoder weight accounting:** the planner subtracts estimated decoder
  weights even if the worker already holds that decoder in live VRAM. Removing
  this conservative charge needs an explicit, accurate residency contract;
  a blanket subtraction would be unsafe for cold or independent decoders.
- **GPU weight-buffer tuning:** the streaming builder defaults to two GPU slots.
  More slots compete with decoder activations, and the retained CPU weight source
  already avoids disk I/O. Needs transfer-versus-compute profiling before changes.
- **Combined decoder compilation:** already exposed as `combined_compile` and
  uses a larger memory budget. Needs output/peak-memory/timing comparison;
  it is not enabled by this patch.
- **Exact prompt cache:** already implemented and measured. The next approved
  script creates a new prompt batch, so its first encoding will be a cache miss;
  identical later rerenders may hit. No cache claim extends to arbitrary text.
