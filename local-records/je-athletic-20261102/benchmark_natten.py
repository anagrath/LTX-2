"""Isolated NA kernel comparison; same BF16 inputs, warmed CUDA-event timing."""
import json
from pathlib import Path
import torch,natten
from ltx_core.model.video_vae.transformer.fallback_na.triton_na import na3d

torch.manual_seed(42)
shape=(1,13,32,18,8,64)
q,k,v=[torch.randn(shape,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
q=q*64**-0.5
calls={'triton_fallback':lambda:na3d(q,k,v,kernel_size=(3,3,3),scale=1.0),
       'natten':lambda:natten.na3d(q,k,v,kernel_size=(3,3,3),scale=1.0)}
results={}
with torch.inference_mode():
 for name,call in calls.items():
  for _ in range(3):out=call()
  torch.cuda.synchronize()
  start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
  start.record()
  for _ in range(20):out=call()
  end.record();end.synchronize()
  results[name]={'mean_ms':start.elapsed_time(end)/20,'finite':bool(torch.isfinite(out).all())}
 a,b=[call() for call in calls.values()]
 results.update(shape=shape,kernel=[3,3,3],dtype=str(q.dtype),natten_version=natten.__version__,torch=torch.__version__,
                max_abs_difference=float((a-b).abs().max()),mean_abs_difference=float((a-b).abs().mean()),
                note='Representative synthetic NA kernel benchmark, not an end-to-end decoder measurement')
 assert all(results[n]['finite'] for n in calls)
 assert torch.allclose(a,b,atol=0.02,rtol=0.02)
Path(__file__).with_name('natten-kernel-benchmark.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
