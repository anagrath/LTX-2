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
    """CPU-only LRU of individual unenhanced captions, scoped to one encoder/config."""

    def __init__(self, encoder, device, max_bytes=512 * 1024**2, identity=None):
        self.encoder, self.device, self.max_bytes = encoder, device, max_bytes
        self.identity = identity if identity is not None else (type(encoder).__module__, type(encoder).__qualname__, id(encoder))
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    @staticmethod
    def _copy(value, device):
        import torch
        return value._replace(**{name: tensor.detach().to(device, copy=True)
            for name, tensor in zip(value._fields, value) if isinstance(tensor, torch.Tensor)})

    def _lookup(self, prompts, options, materialize=True):
        import torch
        prompts = list(prompts)
        known = {'enhance_first_prompt', 'enhance_static_cache', 'enhance_prompt_image', 'enhance_prompt_seed'}
        # Enhancement depends on image/seed and changes only the first caption.
        # Unknown options may change conditioning, so they must not hit this cache.
        if options.get('enhance_first_prompt') or set(options) - known:
            return self.encoder(prompts, **options)
        found, missing = {}, []
        for prompt in prompts:
            key = (self.identity, prompt)
            if key in self.entries:
                self.hits += 1
                value, size = self.entries.pop(key)
                self.entries[key] = value, size
                found[prompt] = value
            else:
                self.misses += 1
                if prompt not in missing:
                    missing.append(prompt)
        if missing:
            encoded = self.encoder(missing, **options)
            if len(encoded) != len(missing):
                raise ValueError('Prompt encoder returned the wrong number of contexts')
            for prompt, value in zip(missing, encoded):
                found[prompt] = value
                size = sum(t.numel()*t.element_size() for t in value if isinstance(t, torch.Tensor))
                if size <= self.max_bytes:
                    saved = self._copy(value, 'cpu')
                    while self.entries and self.bytes + size > self.max_bytes:
                        _, (_, removed) = self.entries.popitem(last=False)
                        self.bytes -= removed
                    self.entries[(self.identity, prompt)] = saved, size
                    self.bytes += size
        # Return independent tensors even for duplicate captions in this request.
        return [self._copy(found[prompt], self.device) for prompt in prompts] if materialize else None

    def __call__(self, prompts, **options):
        return self._lookup(prompts, options)

    def prewarm(self, prompts, **options):
        """Prepare captions without retaining GPU copies; caller serializes with rendering."""
        self._lookup(prompts, options, materialize=False)
        return {'hits': self.hits, 'misses': self.misses, 'bytes': self.bytes}


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def job_args(job, config):
    allowed = {'action', 'output_dir', 'image', 'story_plan', 'num_frames', 'chunk_frames', 'carry_frames', 'seed', 'previous_scene_key'}
    unknown = set(job) - allowed
    if unknown:
        raise ValueError(f'Unknown job fields: {sorted(unknown)}')
    if job.get('previous_scene_key') is not None and not getattr(config, 'scene_cache_dir', None):
        raise ValueError('previous_scene_key requires --scene-cache-dir; it is an artifact dependency, not hidden-state continuation')
    action = job.get('action', 'render')
    if action not in ('render', 'prewarm'):
        raise ValueError('action must be render or prewarm')
    required = ('story_plan',) if action == 'prewarm' else ('output_dir', 'image', 'story_plan')
    for name in required:
        if name not in job or not Path(job[name]).is_absolute():
            raise ValueError(f'{name} must be an absolute path')
    return argparse.Namespace(**{**{'num_frames':145, 'chunk_frames':145, 'carry_frames':0, 'seed':42,
        'output_dir':None, 'image':None, 'action':action}, **job,
        **{name:Path(job[name]) for name in ('output_dir','image','story_plan') if job.get(name) is not None},
        'compile':config.compile, 'decoder_mode':config.decoder_mode, 'keep_auxiliary_models':True})


def plan_prompts(path):
    plan = json.loads(path.read_text())
    prompts = plan.get('prompts')
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) for p in prompts):
        raise ValueError('story_plan must contain a nonempty list of prompt strings')
    return prompts


def prewarm_prompts(cache, device, prompts):
    """Serial boundary-only preparation; preserve default RNG state used by other work."""
    import torch
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    start = time.monotonic()
    with torch.inference_mode(), torch.random.fork_rng(devices=devices):
        cache.prewarm(prompts, enhance_first_prompt=False)
    return {'prewarm_wall_s': round(time.monotonic()-start, 3), 'prompt_count':len(prompts),
            'cache_hits':cache.hits,'cache_misses':cache.misses,'cache_bytes':cache.bytes}


def prepare_lookahead(directory, config, prepare):
    """Prepare the next job plus N queued successors, before any render starts."""
    count = getattr(config, 'lookahead', 0)
    if count <= 0:
        return None
    prompts = []
    for path in sorted(directory.glob('*.job.json'))[:count+1]:
        try:
            args = job_args(json.loads(path.read_text()), config)
            prompts.extend(plan_prompts(args.story_plan))
        except (OSError, ValueError, TypeError):
            # The normal queue claim produces the failure receipt; bad future input
            # must not stop an otherwise valid current scene before it is claimed.
            logging.warning('Skipping invalid lookahead input: %s', path.name, exc_info=True)
    if prompts:
        # Deduplication here also prevents repeated queue entries growing the encoder batch.
        return prepare(list(dict.fromkeys(prompts)))
    return None


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
        details = run(args)
    except Exception as error:
        write_json(running.with_name(running.name.replace('.running.json','.failed.json')),
            {'status':'failed','error_type':type(error).__name__,'error':str(error),
             'wall_s':round(time.monotonic()-started,3)})
        logging.exception('LTX job failed: %s', running.name)
    else:
        write_json(running.with_name(running.name.replace('.running.json','.done.json')),
            {'status':'completed','output_dir':str(args.output_dir) if args.output_dir else None,
             'action':args.action,'details':details if isinstance(details, dict) else None,
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
            cache = PromptCache(pipeline.prompt_encoder, pipeline.device,
                identity=(str(Path(__file__).resolve()),config.compile,config.decoder_mode,'pinned-LTX-2.5-49b093a39c19169a8a81721dcc541e07e945232f'))
            pipeline.prompt_encoder = cache
            for component in (pipeline.stage,pipeline.video_decoder,pipeline.audio_decoder,pipeline.upsampler):
                lifetime.enter_context(component.resident())

            scene_cache = None

            def artifacts():
                nonlocal scene_cache
                if scene_cache is None:
                    from ripper3_scene_cache import SceneArtifacts, rendering_identity
                    cache_root = config.scene_cache_dir.resolve()
                    cache_root.mkdir(parents=True,exist_ok=True,mode=0o700)
                    catalog_path = cache_root/'model-fingerprints.json'
                    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else {}
                    identity = rendering_identity(Path(__file__).parent,
                        Path('/home/anagrath/ai/models/LTX-2.5').rglob('*.safetensors'),catalog)
                    write_json(catalog_path,catalog)
                    scene_cache = SceneArtifacts(cache_root,identity)
                return scene_cache

            def prepare(prompts):
                glm_state()
                info.update(state='prewarming')
                write_json(status,info)
                details = prewarm_prompts(cache, pipeline.device, prompts)
                info.update(last_prewarm=details)
                write_json(status,info)
                return details

            def run(args):
                if args.action == 'prewarm':
                    return prepare(plan_prompts(args.story_plan))
                info.update(state='running',output_dir=str(args.output_dir))
                write_json(status,info)
                # run_job checks the live draft/worker state before every request.
                try:
                    if getattr(config,'scene_cache_dir',None):
                        glm_state()
                        checkpoint = artifacts()
                        key = checkpoint.key(args,json.loads(args.story_plan.read_text()))
                        reused = checkpoint.reuse(key,args.output_dir)
                        if reused is not None:
                            return reused
                        report = run_job(args,pipeline=pipeline)
                        checkpoint.store(key,args.output_dir)
                        return {'operation':'model_generation','scene_cache_key':key,'generation_wall_s':report['wall_s']}
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
                if getattr(config, 'lookahead', 0) and next(directory.glob('*.job.json'), None) is not None:
                    prepare_lookahead(directory, config, prepare)
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
            command.add_argument('--scene-cache-dir',type=Path,help='Opt-in completed-scene artifact checkpoints; never counted as generation throughput')
            command.add_argument('--lookahead',type=int,choices=range(5),default=0,help='Serial prewarm of next job plus N queued successors at job boundaries')
            command.add_argument('--decoder-mode',choices=['chunked_eager','chunked_compile','combined_compile'],default='chunked_compile')
        if name == 'submit':
            command.add_argument('--job',type=Path,required=True)
    config = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if config.command == 'serve':
        serve(config)
    elif config.command == 'submit':
        job = json.loads(config.job.read_text())
        job_args(job,argparse.Namespace(compile=True,decoder_mode='chunked_compile',scene_cache_dir=True))
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
