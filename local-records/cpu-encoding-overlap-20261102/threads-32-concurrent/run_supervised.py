from pathlib import Path
import json,os,subprocess,time
root=Path(__file__).parent
report=root/'result.json'
repo=Path('/home/anagrath/ai/code/LTX-2')
env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']='32';env['MKL_NUM_THREADS']='32'
with (root/'benchmark.log').open('w') as log:
 child=subprocess.Popen(['nice','-n','10',str(repo/'.venv/bin/python'),str(root/'benchmark_cpu_encoder.py'),'--output',str(report)],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
 print('CPU child',child.pid,flush=True)
 while child.poll() is None:
  if report.exists():
   state=json.loads(report.read_text());phase=state.get('phase','');elapsed=time.monotonic()-state.get('phase_started_monotonic',time.monotonic())
   limit=125 if 'encode' in phase or 'projection' in phase and phase!='projection_model_load' else 300
   if elapsed>limit:
    child.terminate()
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:child.kill();child.wait()
    state.update(status='timed_out',timeout_phase=phase,elapsed_in_phase_s=round(elapsed,3),timeout_limit_s=limit)
    report.write_text(json.dumps(state,indent=2)+'\n')
    print('TIMEOUT',phase,elapsed,flush=True);break
  time.sleep(2)
 print('CPU child exit',child.returncode,flush=True)
 if report.exists():print(report.read_text(),flush=True)
