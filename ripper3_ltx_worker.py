#!/usr/bin/env python3
"""Serial local LTX job queue; keep models warm without opening a network port."""
import argparse
from collections import OrderedDict
from contextlib import ExitStack
import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import time
import uuid


class PromptCache:
    """Bounded, CPU-only cache of complete unenhanced prompt batches."""

    def __init__(self, encoder, device, max_bytes=512 * 1024**2):
        self.encoder, self.device, self.max_bytes = encoder, device, max_bytes
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def __call__(self, prompts, **options):
        import torch
        # Enhanced prompts depend on images/seeds; bypass this exact-text cache.
        if options.get('enhance_first_prompt'):
            return self.encoder(prompts, **options)
        key = tuple(prompts)
        if key in self.entries:
            self.hits += 1
            saved, size = self.entries.pop(key)
            self.entries[key] = saved, size
            return [value._replace(**{name: tensor.to(self.device, copy=True)
                for name, tensor in zip(value._fields, value) if isinstance(tensor, torch.Tensor)})
                for value in saved]
        self.misses += 1
        result = self.encoder(prompts, **options)
        size = sum(t.numel()*t.element_size() for value in result for t in value if isinstance(t, torch.Tensor))
        if size <= self.max_bytes:
            saved = [value._replace(**{name: tensor.detach().to('cpu', copy=True)
                for name, tensor in zip(value._fields, value) if isinstance(tensor, torch.Tensor)})
                for value in result]
            while self.entries and self.bytes + size > self.max_bytes:
                _, (_, removed) = self.entries.popitem(last=False)
                self.bytes -= removed
            self.entries[key] = saved, size
            self.bytes += size
        return result


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def job_args(job, config):
    allowed = {'output_dir', 'image', 'story_plan', 'num_frames', 'chunk_frames', 'carry_frames', 'seed'}
    unknown = set(job) - allowed
    if unknown:
        raise ValueError(f'Unknown job fields: {sorted(unknown)}')
    for name in ('output_dir', 'image', 'story_plan'):
        if name not in job or not Path(job[name]).is_absolute():
            raise ValueError(f'{name} must be an absolute path')
    return argparse.Namespace(**{**{'num_frames':145, 'chunk_frames':145, 'carry_frames':0, 'seed':42},
        **job, **{name:Path(job[name]) for name in ('output_dir','image','story_plan')},
        'compile':config.compile, 'decoder_mode':config.decoder_mode, 'keep_auxiliary_models':True})


def process_pending(directory, config, run):
    """Claim and finish one job; errors are recorded, never silently retried."""
    queued = sorted(directory.glob('*.job.json'))
    if not queued:
        return False
    path = queued[0]
    running = path.with_name(path.name.removesuffix('.job.json') + '.running.json')
    path.rename(running)
    started = time.monotonic()
    try:
        args = job_args(json.loads(running.read_text()), config)
        run(args)
    except Exception as error:
        write_json(running.with_name(running.name.replace('.running.json','.failed.json')),
            {'status':'failed','error_type':type(error).__name__,'error':str(error),
             'wall_s':round(time.monotonic()-started,3)})
        logging.exception('LTX job failed: %s', running.name)
    else:
        write_json(running.with_name(running.name.replace('.running.json','.done.json')),
            {'status':'completed','output_dir':str(args.output_dir),
             'wall_s':round(time.monotonic()-started,3)})
    finally:
        running.rename(running.with_name(running.name.replace('.running.json','.request.json')))
    return True


def serve(config):
    # Keep status/submit/stop usable without importing Torch or touching CUDA.
    import torch
    from ripper3_je_native import build_pipeline, glm_state, run_job
    directory = config.queue.resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (directory/'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = directory/'stop'
    stop.unlink(missing_ok=True)
    stopping = False

    def stop_signal(*_):
        nonlocal stopping
        stopping = True  # Finish the current scene; release all models afterward.

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    status = directory/'worker.json'
    info = {'pid':os.getpid(),'state':'starting','decoder_mode':config.decoder_mode,'compile':config.compile}
    write_json(status, info)
    try:
        glm_state()
        with torch.inference_mode(), ExitStack() as lifetime:
            pipeline = build_pipeline(compile_transformer=config.compile, decoder_mode=config.decoder_mode)
            cache = PromptCache(pipeline.prompt_encoder, pipeline.device)
            pipeline.prompt_encoder = cache
            for component in (pipeline.stage,pipeline.video_decoder,pipeline.audio_decoder,pipeline.upsampler):
                lifetime.enter_context(component.resident())

            def run(args):
                info.update(state='running',output_dir=str(args.output_dir))
                write_json(status,info)
                # run_job checks the live draft/worker state before every request.
                try:
                    run_job(args,pipeline=pipeline)
                except Exception:
                    # Retained models may be unsafe after CUDA failure; stop and release.
                    nonlocal stopping
                    stopping = True
                    raise

            while not stopping and not stop.exists():
                info.update(state='idle',prompt_cache_hits=cache.hits,prompt_cache_misses=cache.misses,
                            prompt_cache_bytes=cache.bytes)
                write_json(status,info)
                if not process_pending(directory,config,run):
                    time.sleep(0.25)
    finally:
        info['state']='stopped'
        write_json(status,info)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('serve','submit','status','stop'):
        command = sub.add_parser(name)
        command.add_argument('--queue',type=Path,required=True)
        if name == 'serve':
            command.add_argument('--compile',action='store_true')
            command.add_argument('--decoder-mode',choices=['chunked_eager','chunked_compile','combined_compile'],default='chunked_compile')
        if name == 'submit':
            command.add_argument('--job',type=Path,required=True)
    config = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if config.command == 'serve':
        serve(config)
    elif config.command == 'submit':
        job = json.loads(config.job.read_text())
        job_args(job,argparse.Namespace(compile=True,decoder_mode='chunked_compile'))
        directory = config.queue.resolve()
        directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        path = directory/(f'{time.time_ns():020d}-'+uuid.uuid4().hex+'.job.json')
        write_json(path,job)
        print(path)
    elif config.command == 'stop':
        (config.queue/'stop').touch()
        print('Stop requested; the worker finishes its current scene before releasing models.')
    else:
        status = config.queue/'worker.json'
        info = json.loads(status.read_text()) if status.exists() else {'state':'not_started'}
        if 'pid' in info:
            try:os.kill(info['pid'],0)
            except ProcessLookupError:info['state']='not_running'
        print(json.dumps(info,indent=2))


if __name__ == '__main__':
    main()
