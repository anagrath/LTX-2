#!/usr/bin/env python3
"""Plain LTX-2.5 joint video/audio story with per-window acting prompts."""
import argparse
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import resource
import subprocess
import time

os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

import torch
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_core.types import VideoPixelShape
from ltx_core.quantization.fp8_cast import build_policy
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_pipelines.chunks import (ChunkConfig, generate_uniform_chunks, denoise_chunks,
    spatially_upsample_chunks, decode_chunks, pipeline_output_from_chunks)
from ltx_pipelines.chunks.layout import uniform_chunk_layouts
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.chunks.conditionings import image_conditionings_for_chunk
from ltx_pipelines.distilled import (DistilledPipeline, ANCESTRAL_NOISE_SEED_OFFSET,
    ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import ImageConditioningInput, OffloadMode, VideoAudio


def prepare_story_run(pipeline, prompts, **kwargs):
    """Encode all windows in one encoder lifetime, reusing the opening context."""
    encoder = pipeline.prompt_encoder
    encoded = []

    def encode_all(requested, **options):
        if requested != [prompts[0]] or options.get('enhance_first_prompt'):
            raise ValueError('Story setup requires the unenhanced opening prompt')
        start = time.monotonic()
        encoded.extend(encoder(prompts, **options))
        logging.info('All %d story prompts encoded in %.3fs', len(encoded), time.monotonic()-start)
        return encoded[:1]

    pipeline.prompt_encoder = encode_all
    try:
        ctx = pipeline._prepare_run(prompt=prompts[0], **kwargs)
    finally:
        pipeline.prompt_encoder = encoder
    return ctx, [VideoAudio(video=e.video_encoding, audio=e.audio_encoding) for e in encoded]


@torch.inference_mode()
def story_result(pipeline, args, plan):
    config=ChunkConfig(chunk_pixel_frames=args.chunk_frames,next_video_carry_frames=args.carry_frames)
    prompts=plan['prompts']
    frame_rate=24
    ctx,contexts=prepare_story_run(pipeline,prompts,seed=args.seed,height=1024,width=576,
        frame_rate=24,num_frames=args.num_frames,
        images=[ImageConditioningInput.at(str(args.image),0,1.0)],color_space=None,
        enhance_prompt=False,enhance_static_cache=False,vae_dtype=None,tiling_config=AUTO_TILING,
        generated_keyframes=0,decode_with_keyframes=False,chunk_config=config)
    target=VideoPixelShape(batch=1,frames=ctx.num_frames,height=ctx.stage_1_height,
        width=ctx.stage_1_width,fps=frame_rate)
    layouts=uniform_chunk_layouts(target_pixel_shape=target,config=config)
    if len(layouts)!=len(contexts):
        raise ValueError(f'{len(layouts)} windows but {len(contexts)} script contexts')
    raw=generate_uniform_chunks(target=target,context=contexts[0],device=pipeline.device,
        dtype=ctx.dtype,config=config,video_scale_factors=pipeline.stage.video_scale_factors,
        make_video_conditionings=pipeline._video_conditionings(ctx.images,ctx.num_frames))
    def scripted():
        for i,chunk in enumerate(raw):
            logging.info('Story window %d/%d: %s',i+1,len(prompts),plan['dialogue'][i])
            yield replace(chunk,video_context=contexts[i].video,audio_context=contexts[i].audio)
    base=denoise_chunks(scripted(),pipeline.stage,sigmas=DISTILLED_SIGMAS.to(pipeline.device),
        noiser=ctx.noiser,fps=frame_rate,noise_scale=1.0,
        **pipeline._sampler_kwargs(args.seed,ANCESTRAL_NOISE_SEED_OFFSET))
    upscaled=spatially_upsample_chunks(base,pipeline.upsampler)
    refined=denoise_chunks(upscaled,pipeline.stage,sigmas=STAGE_2_DISTILLED_SIGMAS.to(pipeline.device),
        noiser=ctx.noiser,fps=frame_rate,
        **pipeline._sampler_kwargs(args.seed,ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET))
    decoded=decode_chunks(refined,pipeline.video_decoder,pipeline.audio_decoder,fps=frame_rate,
        tiling_config=ctx.tiling_config,generator=ctx.generator,dtype=ctx.vae_dtype)
    # Both lazy denoising iterators share one streaming transformer lifetime.
    with pipeline.stage.resident():
        return pipeline_output_from_chunks(decoded,num_frames=ctx.num_frames,tiling_config=ctx.tiling_config)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--num-frames',type=int,default=145)
    p.add_argument('--chunk-frames',type=int,default=145)
    p.add_argument('--carry-frames',type=int,default=0)
    p.add_argument('--story-plan',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--compile',action='store_true',help='Use upstream block compilation, without CUDA graphs for CPU offload')
    args=p.parse_args()
    if args.num_frames<1 or (args.num_frames-1)%8:p.error('Target frames must be 8*k+1')
    root=Path('/home/anagrath/ai/models/LTX-2.5')
    state=json.loads(subprocess.check_output(['/usr/bin/python3',
        '/home/anagrath/ai/code/stripe-runtime-worktrees/prefill-critical-20261102/python/draft_model.py','status'],text=True))
    if state['draft_loaded'] or state['worker_running']:p.error('GLM drafting must be OFF')
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'video.mp4').exists():p.error('Choose a fresh output directory')
    models=ModelPaths.from_split(
        transformer_path=str(root/'diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors'),
        text_encoder_path=str(root/'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'),
        video_vae_path=str(root/'vae/ltx-2.5-video-vae-bf16.safetensors'),
        audio_vae_path=str(root/'vae/ltx-2.5-audio-vae-bf16.safetensors'))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    report={'status':'running','native_audio':True,'external_tts':False,
            'dub_it_loaded':False,'audio_reference_used':False,'loras':[],
            'target_frames':args.num_frames,'image':str(args.image),
            'seed':args.seed,'glm_state':state,'sampler':'official LTX-2.5 Euler ancestral',
            'optimizations':{'single_encoder_lifetime':True,'persistent_streaming_transformer':True,
                             'compile':args.compile,'cuda_graphs':False}}
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    try:
        pipeline=DistilledPipeline(model_paths=models,
            spatial_upsampler_path=str(root/'latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors'),
            loras=[],quantization=build_policy(models.transformer()),offload_mode=OffloadMode.CPU,
            compilation_config=CompilationConfig() if args.compile else None)
        plan=json.loads(args.story_plan.read_text())
        report['story_plan']=plan
        if ' '.join(plan['dialogue'])!=plan['script']:
            raise ValueError('Dialogue windows do not reproduce the script exactly')
        result=story_result(pipeline,args,plan)
        encode_video(video=result.video,fps=24,audio=result.audio,output_path=str(output/'video.mp4'),
            video_chunks_number=get_video_chunks_number(result.num_frames,result.tiling_config))
        torch.cuda.synchronize();report['status']='completed'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report.update(wall_s=round(time.monotonic()-start,3),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ['status','wall_s','peak_allocated_bytes']}),flush=True)


if __name__=='__main__':main()
