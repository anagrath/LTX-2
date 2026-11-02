#!/usr/bin/env python3
"""Run a small official LTX pipeline baseline and record time and peak memory."""
import argparse
import json
import os
from pathlib import Path
import resource
import runpy
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--prompt-file', type=Path)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--num-frames', type=int, default=49)
    parser.add_argument('--chunked', action='store_true')
    parser.add_argument('--models', type=Path, default=Path('/home/anagrath/ai/models/LTX-2.5'))
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 64 or args.width % 64:
        parser.error('Two-stage dimensions must be positive multiples of 64')
    if args.num_frames < 1 or (args.num_frames - 1) % 8:
        parser.error('Frame count must be 8*k+1')
    if args.image and not args.image.is_file():
        parser.error('Reference image does not exist')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    video = output / 'baseline.mp4'
    if video.exists():
        parser.error('Choose an output directory without an existing baseline.mp4')
    switch = '/home/anagrath/ai/code/stripe-runtime-worktrees/prefill-critical-20261102/python/draft_model.py'
    state = json.loads(subprocess.check_output(['/usr/bin/python3', switch, 'status'], text=True))
    if state['draft_loaded'] or state['worker_running']:
        parser.error('Turn drafting and its RTX worker OFF before running LTX')
    files = {
        '--transformer-path': 'diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors',
        '--text-encoder-path': 'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors',
        '--video-vae-path': 'vae/ltx-2.5-video-vae-bf16.safetensors',
        '--audio-vae-path': 'vae/ltx-2.5-audio-vae-bf16.safetensors',
        '--spatial-upsampler-path': 'latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors',
        '--duration-head-path': 'model_patches/ltx-2.5-duration-head-bf16.safetensors',
    }
    command = ['ltx_pipelines.distilled']
    for flag, name in files.items():
        path = args.models / name
        if not path.is_file():
            parser.error('Missing model file: ' + str(path))
        command += [flag, str(path)]
    command += ['--output-path', str(video), '--height', str(args.height), '--width', str(args.width),
                '--num-frames', str(args.num_frames), '--frame-rate', '24', '--seed', '42',
                '--quantization', 'fp8-cast', '--offload', 'cpu', '--prompt',
                'A small red wooden toy boat floats slowly across a calm pond. '
                'Static camera, gentle ripples, natural daylight. Soft water sounds.']
    if args.prompt_file:
        command[-1] = args.prompt_file.read_text().strip()
    if args.image:
        command += ['--image', str(args.image.resolve()), '0', '1.0']
    if args.chunked:
        command += ['--chunked', '--chunk-pixel-frames', '97', '--chunk-carry-frames', '25']
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # The weights and embedded tokenizer must suffice for this local run.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    torch.cuda.reset_peak_memory_stats()
    samples = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                row = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,power.draw',
                    '--format=csv,noheader,nounits'], text=True, timeout=5).strip().split(',')
                samples.append({'elapsed_s': round(time.monotonic() - start, 3),
                                'gpu_used_mib': float(row[0]), 'gpu_util_percent': float(row[1]),
                                'power_w': float(row[2])})
            except (subprocess.SubprocessError, ValueError):
                pass
            stop.wait(1)

    report = {'status': 'running', 'argv': command, 'python': sys.version.split()[0],
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'glm_state_before': state,
              'gpu_sampling_interval_s': 1, 'compile_enabled': False}
    start = time.monotonic()
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        sys.argv = command
        runpy.run_module('ltx_pipelines.distilled', run_name='__main__')
        torch.cuda.synchronize()
        report['status'] = 'completed'
    except BaseException as error:
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        report['error'] = str(error)
        raise
    finally:
        stop.set()
        thread.join(timeout=6)
        report.update(wall_s=round(time.monotonic() - start, 3),
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            sampled_peak_gpu_used_mib=max((x['gpu_used_mib'] for x in samples), default=None),
            output_bytes=video.stat().st_size if video.exists() else 0)
        (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        (output / 'gpu-samples.json').write_text(json.dumps(samples, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ['status', 'wall_s', 'sampled_peak_gpu_used_mib', 'output_bytes']}), flush=True)


if __name__ == '__main__':
    main()
