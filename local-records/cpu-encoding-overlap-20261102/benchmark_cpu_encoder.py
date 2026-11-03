"""CPU-only actual-model benchmark; invoked by a phase-timeout supervisor."""
import argparse,json,os,resource,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['OMP_NUM_THREADS']='16'
os.environ['MKL_NUM_THREADS']='16'
os.environ['OPENBLAS_NUM_THREADS']='16'
import torch
from ltx_pipelines.utils.blocks.prompt_encoder import PromptEncoder
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import OffloadMode

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 torch.set_num_threads(16);torch.set_num_interop_threads(1)
 root=Path('/home/anagrath/ai/models/LTX-2.5')
 plan=Path('/home/anagrath/ai/outputs/ltx/elephant-reference-20261102/all-ages-native-natural-cadence-story-plan.json')
 prompt=json.loads(plan.read_text())['prompts'][0]
 report={'status':'running','device':'cpu','dtype':'float32','threads':torch.get_num_threads(),
         'interop_threads':torch.get_num_interop_threads(),'torch':torch.__version__,
         'prompt_source':str(plan),'prompt_index':0,'prompt_chars':len(prompt),'phases':{},'pid':os.getpid()}
 def save():
  report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.output)
 def phase(name,call):
  report.update(phase=name,phase_started_monotonic=time.monotonic());save()
  print('START',name,flush=True);start=time.monotonic()
  result=call()
  report['phases'][name]={'wall_s':round(time.monotonic()-start,3)};save()
  print('DONE',name,report['phases'][name],flush=True)
  return result
 try:
  models=ModelPaths.from_split(transformer_path=str(root/'diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors'),
      text_encoder_path=str(root/'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'),
      video_vae_path=str(root/'vae/ltx-2.5-video-vae-bf16.safetensors'),audio_vae_path=str(root/'vae/ltx-2.5-audio-vae-bf16.safetensors'))
  encoder=PromptEncoder(models,dtype=torch.float32,device=torch.device('cpu'),offload_mode=OffloadMode.NONE)
  with torch.inference_mode():
   text=phase('text_model_load',encoder._build_text_encoder)
   processor=phase('projection_model_load',encoder._build_embeddings_processor)
   for i in [1,2]:
    hidden,mask=phase(f'caption_{i}_text_encode',lambda:text.encode([prompt])[0])
    report['phases'][f'caption_{i}_text_encode'].update(mask_shape=list(mask.shape),valid_tokens=int(mask.sum()),
        hidden_state_count=len(hidden),hidden_shape=list(hidden[-1].shape))
    out=phase(f'caption_{i}_projection',lambda:processor.process_hidden_states(hidden,mask))
    report['phases'][f'caption_{i}_projection'].update(video_shape=list(out.video_encoding.shape),
        audio_shape=list(out.audio_encoding.shape) if out.audio_encoding is not None else None,
        finite=bool(torch.isfinite(out.video_encoding).all() and (out.audio_encoding is None or torch.isfinite(out.audio_encoding).all())),
        dtype=str(out.video_encoding.dtype))
    del hidden,mask,out;save()
   report.update(status='completed',phase='finished');save()
 except BaseException as error:
  report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise
if __name__=='__main__':main()
