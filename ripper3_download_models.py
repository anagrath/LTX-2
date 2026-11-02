import os
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import json,time,hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download
D=Path('/home/anagrath/ai/experiments/ltx-setup-20261102')
m=json.loads((Path(__file__).parent/'local-records/model-manifest.json').read_text())
root=Path('/home/anagrath/ai/models/LTX-2.5');root.mkdir(parents=True,exist_ok=True)
results=[]
for item in sorted(m['files'],key=lambda x:x['size']):
 print(json.dumps({'event':'downloading','file':item['path'],'gib':round(item['size']/2**30,3)}),flush=True)
 start=time.monotonic();p=Path(hf_hub_download(m['repository'],item['path'],revision=m['revision'],local_dir=root))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
 assert p.stat().st_size==item['size'] and h.hexdigest()==item['sha256'],item['path']+' integrity failure'
 r={**item,'local_path':str(p),'verified':True,'elapsed_s':round(time.monotonic()-start,3)};results.append(r)
 (D/'download-results.json').write_text(json.dumps({'revision':m['revision'],'complete':len(results)==len(m['files']),'files':results},indent=2)+'\n')
 print(json.dumps({'event':'verified','file':item['path'],'elapsed_s':r['elapsed_s']}),flush=True)
