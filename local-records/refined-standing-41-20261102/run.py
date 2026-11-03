from pathlib import Path
import json, subprocess, time, os

repo = Path('/home/anagrath/ai/code/LTX-2')
python = repo/'.venv/bin/python'
cli = repo/'ripper3_ltx_worker.py'
root = Path(__file__).parent
queue = root/'queue'
queue.mkdir(mode=0o700, exist_ok=True)
env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
start = time.monotonic()
with (root/'worker.log').open('w') as log:
    worker = subprocess.Popen([str(python),str(cli),'serve','--queue',str(queue),'--compile','--decoder-mode','chunked_compile','--prepared-weights-cache-gib','32'],cwd=repo,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
timing = {'worker_pid':worker.pid, 'artifact_reuse_requested':False, 'worker_kept_alive':True}
try:
    for name in ['cold-prime','prewarm','render']:
        step_start = time.monotonic()
        if name == 'prewarm':
            warm_start = step_start
        submitted = subprocess.check_output([str(python),str(cli),'submit','--queue',str(queue),'--job',str(root/(name+'-job.json'))],cwd=repo,text=True).strip()
        prefix = Path(submitted).name.removesuffix('.job.json')
        deadline = time.monotonic()+1200
        while not (queue/(prefix+'.done.json')).exists():
            failed = queue/(prefix+'.failed.json')
            if failed.exists():
                raise RuntimeError(failed.read_text())
            if worker.poll() is not None:
                raise RuntimeError(f'Worker exited {worker.returncode}; see worker.log')
            if time.monotonic()>deadline:
                raise TimeoutError(name)
            time.sleep(.5)
        receipt = json.loads((queue/(prefix+'.done.json')).read_text())
        (root/(name+'-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        timing[name+'_submitted_to_complete_s'] = round(time.monotonic()-step_start,3)
        print(name,json.dumps(receipt),flush=True)
    timing.update(warm_request_prep_and_render_wall_s=round(time.monotonic()-warm_start,3),startup_cold_prime_prep_and_render_wall_s=round(time.monotonic()-start,3))
    (root/'driver-timing.json').write_text(json.dumps(timing,indent=2)+'\n')
    print(json.dumps(timing),flush=True)
except BaseException:
    (queue/'stop').touch()
    raise
