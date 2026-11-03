import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import json,time,logging,hashlib
from pathlib import Path
import torch
from ripper3_je_native import build_pipeline,glm_state

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
root=Path(__file__).parent
state=glm_state()
pipeline=build_pipeline(compile_transformer=True,decoder_mode='chunked_compile')
encoder=pipeline.prompt_encoder.enable_prepared_weights(max_bytes=32*1024**3)
plan=json.loads(Path('/home/anagrath/ai/outputs/ltx/elephant-reference-20261102/all-ages-native-standing-41-story-plan.json').read_text())
prompts=plan['prompts']
changed=[p.replace('sunny wooden beach boardwalk','bright sunny wooden beach boardwalk') for p in prompts]
assert changed!=prompts
report={'status':'running','glm_state':state,'prompt_count':len(prompts),'prepared_budget_bytes':32*1024**3,'embedding_cache_used':False,'runs':[]}
reference=None
try:
 with torch.inference_mode():
  for name,captions in [('cold',prompts),('warm_changed_captions',changed),('warm_original_again',prompts)]:
   torch.cuda.reset_peak_memory_stats();start=time.monotonic()
   values=encoder(captions,enhance_first_prompt=False)
   torch.cuda.synchronize()
   elapsed=time.monotonic()-start
   cpu=[v._replace(**{field:t.detach().cpu().clone() for field,t in zip(v._fields,v) if isinstance(t,torch.Tensor)}) for v in values]
   del values
   finite=all(bool(torch.isfinite(t).all()) for v in cpu for t in v if isinstance(t,torch.Tensor))
   entry={'name':name,'wall_s':round(elapsed,3),'all_finite':finite,'prepared_stats':encoder.prepared_weight_stats(),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
   assert finite
   if reference is None:reference=cpu
   elif name=='warm_original_again':
    same=True;max_abs=0.
    for a,b in zip(reference,cpu):
     for x,y in zip(a,b):
      if isinstance(x,torch.Tensor):
       same=same and torch.equal(x,y)
       max_abs=max(max_abs,float((x.float()-y.float()).abs().max()))
       torch.testing.assert_close(x,y,atol=0,rtol=0)
    entry.update(bitwise_equal_to_cold=same,max_abs_difference=max_abs)
   report['runs'].append(entry)
   (root/'result.json').write_text(json.dumps(report,indent=2)+'\n')
   print(json.dumps(entry),flush=True)
  report['status']='completed'
except BaseException as error:
 report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
finally:
 encoder.clear_prepared_weights()
 (root/'result.json').write_text(json.dumps(report,indent=2)+'\n')
