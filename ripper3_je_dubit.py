#!/usr/bin/env python3
"""Experimental LTX-native JE speech with a separate identity frame and voice clip."""
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
from ltx_core.conditioning import AudioConditionByReferenceLatent
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_core.types import VideoPixelShape
from ltx_core.quantization.fp8_cast import build_policy
from ltx_pipelines.chunks import (ChunkConfig, generate_uniform_chunks, denoise_chunks,
    spatially_upsample_chunks, replace_audio_conditionings, decode_chunks, pipeline_output_from_chunks)
from ltx_pipelines.chunks.layout import uniform_chunk_layouts
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.chunks.conditionings import image_conditionings_for_chunk
from ltx_pipelines.dubit import DubItPipeline
from ltx_pipelines.iclora_utils import patchify_audio_reference_latent
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import ImageConditioningInput, OffloadMode, VideoAudio


class ReferenceVoicePipeline(DubItPipeline):
    # Keep the official denoisers, audio generation and vocoder. Change only reference selection.
    def __init__(self, *args, target_frames, **kwargs):
        self.target_frames=target_frames
        super().__init__(*args, **kwargs)

    def _prepare_run(self, **kwargs):
        context=super()._prepare_run(**kwargs)
        return replace(context,num_frames=self.target_frames)

    def _video_conditionings(self, images, num_frames, reference_video_path,
                            reference_strength, color_space, frame_source):
        # The voice clip is in a classroom; it must not supply the new scene's visuals.
        def make(chunk):
            return self.image_conditioner(lambda encoder: image_conditionings_for_chunk(
                chunk, images=images, video_encoder=encoder))
        return make

    def _audio_conditionings(self, *, latent, fps):
        if latent is None:
            return super()._audio_conditionings(latent=None,fps=fps)
        # Every target window sees the same complete speaker reference, independent of target time.
        patch,positions=patchify_audio_reference_latent(latent,negative_positions=True,device=latent.device)
        return lambda chunk: [AudioConditionByReferenceLatent(patch,positions,strength=1.0)]


@torch.inference_mode()
def story_result(pipeline, args, plan):
    config=ChunkConfig(chunk_pixel_frames=args.chunk_frames,next_video_carry_frames=args.carry_frames)
    prompts=plan['prompts']
    ctx=pipeline._prepare_run(prompt=prompts[0],seed=args.seed,height=1024,width=576,
        images=[ImageConditioningInput.at(str(args.image),0,1.0)],color_space=None,
        reference_video_path=str(args.reference_video),reference_strength=0,
        enhance_prompt=False,enhance_static_cache=False,vae_dtype=None,tiling_config=AUTO_TILING,
        chunk_config=config)
    contexts=[VideoAudio(video=ctx.video_context,audio=ctx.audio_context)]
    if len(prompts)>1:
        for encoded in pipeline.prompt_encoder(prompts[1:]):
            contexts.append(VideoAudio(video=encoded.video_encoding,audio=encoded.audio_encoding))
    target=VideoPixelShape(batch=1,frames=ctx.num_frames,height=ctx.stage_1_height,
        width=ctx.stage_1_width,fps=ctx.frame_rate)
    layouts=uniform_chunk_layouts(target_pixel_shape=target,config=config)
    if len(layouts)!=len(contexts):
        raise ValueError(f'{len(layouts)} windows but {len(contexts)} script contexts')
    raw=generate_uniform_chunks(target=target,context=contexts[0],device=pipeline.device,
        dtype=ctx.dtype,config=config,video_scale_factors=pipeline.stage.video_scale_factors,
        make_video_conditionings=pipeline._video_conditionings(ctx.images,ctx.num_frames,
            ctx.reference_video_path,0,None,None),
        make_audio_conditionings=pipeline._audio_conditionings(latent=ctx.ref_audio_latent,fps=ctx.frame_rate))
    def scripted():
        for i,chunk in enumerate(raw):
            logging.info('Story window %d/%d: %s',i+1,len(prompts),plan['dialogue'][i])
            yield replace(chunk,video_context=contexts[i].video,audio_context=contexts[i].audio)
    base=denoise_chunks(scripted(),pipeline.stage,sigmas=DISTILLED_SIGMAS.to(pipeline.device),
        noiser=ctx.noiser,fps=ctx.frame_rate)
    upscaled=spatially_upsample_chunks(base,pipeline.upsampler)
    prepared=replace_audio_conditionings(upscaled,pipeline._audio_conditionings(latent=None,fps=ctx.frame_rate))
    refined=denoise_chunks(prepared,pipeline.stage,sigmas=STAGE_2_DISTILLED_SIGMAS.to(pipeline.device),
        noiser=ctx.noiser,fps=ctx.frame_rate,freeze_audio=True)
    decoded=decode_chunks(refined,pipeline.video_decoder,pipeline.audio_decoder,fps=ctx.frame_rate,
        tiling_config=ctx.tiling_config,generator=ctx.generator,dtype=ctx.vae_dtype)
    return pipeline_output_from_chunks(decoded,num_frames=ctx.num_frames,tiling_config=ctx.tiling_config)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--prompt-file',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--reference-video',type=Path,required=True)
    p.add_argument('--num-frames',type=int,default=145)
    p.add_argument('--chunk-frames',type=int,default=145)
    p.add_argument('--carry-frames',type=int,default=0)
    p.add_argument('--story-plan',type=Path)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.num_frames<1 or (args.num_frames-1)%8:p.error('Target frames must be 8*k+1')
    root=Path('/home/anagrath/ai/models/LTX-2.5')
    lora=root/'dubit/ltx-2.3-22b-ic-lora-dubit-0.9.safetensors'
    if not lora.is_file():p.error('Dub-It adapter has not finished downloading')
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
            'experimental_reference_composition':True,'target_frames':args.num_frames,
            'reference_video':str(args.reference_video),'image':str(args.image),
            'prompt':args.prompt_file.read_text().strip(),'seed':args.seed,'glm_state':state}
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    try:
        pipeline=ReferenceVoicePipeline(model_paths=models,
            spatial_upsampler_path=str(root/'latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors'),
            ic_lora=LoraPathStrengthAndSDOps(str(lora),1.0,LTXV_LORA_COMFY_RENAMING_MAP),
            quantization=build_policy(models.transformer()),offload_mode=OffloadMode.CPU,
            target_frames=args.num_frames)
        if args.story_plan:
            plan=json.loads(args.story_plan.read_text())
            report['story_plan']=plan
            result=story_result(pipeline,args,plan)
        else:
            result=pipeline(prompt=report['prompt'],seed=args.seed,height=1024,width=576,
                images=[ImageConditioningInput.at(str(args.image),0,1.0)],
                reference_video_path=str(args.reference_video),reference_strength=0,
                chunk_config=ChunkConfig(chunk_pixel_frames=args.chunk_frames,next_video_carry_frames=args.carry_frames))
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
