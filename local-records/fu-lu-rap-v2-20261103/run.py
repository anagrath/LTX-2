from pathlib import Path
import json, subprocess, time

root=Path(__file__).parent
repo=Path('/home/anagrath/ai/code/LTX-2')
queue=Path('/home/anagrath/ai/experiments/ltx-setup-20261102/refined-standing-41/queue')
python=repo/'.venv/bin/python'
start=time.monotonic()
status=json.loads((queue/'worker.json').read_text())
if status['state']!='idle':
    raise RuntimeError(f'Expected idle warm worker, got {status["state"]}')
(root/'worker-before.json').write_text(json.dumps(status,indent=2)+'\n')
submitted=subprocess.check_output([str(python),str(repo/'ripper3_ltx_worker.py'),'submit','--queue',str(queue),'--job',str(root/'render-job.json')],cwd=repo,text=True).strip()
prefix=Path(submitted).name.removesuffix('.job.json')
deadline=time.monotonic()+900
while not (queue/(prefix+'.done.json')).exists():
    failed=queue/(prefix+'.failed.json')
    if failed.exists():raise RuntimeError(failed.read_text())
    if time.monotonic()>deadline:raise TimeoutError('Render')
    time.sleep(.5)
receipt=json.loads((queue/(prefix+'.done.json')).read_text())
(root/'render-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
timing={'submit_through_complete_wall_s':round(time.monotonic()-start,3),'existing_worker_pid':status['pid'],'cold_prime_performed':False,'worker_kept_alive':True}
(root/'driver-timing.json').write_text(json.dumps(timing,indent=2)+'\n')
print(json.dumps(timing),flush=True)
