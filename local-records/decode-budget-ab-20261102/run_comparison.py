from pathlib import Path
import json,os,subprocess,time
repo=Path('/home/anagrath/ai/code/LTX-2')
python=repo/'.venv/bin/python';cli=repo/'ripper3_ltx_worker.py'
base=Path(__file__).parent
for condition in ['baseline','fixed']:
 root=base/condition;queue=root/'queue';queue.mkdir(mode=0o700,exist_ok=True)
 env=dict(os.environ)
 if condition=='baseline':env['PYTHONPATH']='/home/anagrath/ai/code/LTX-2-budget-baseline-20261102/packages/ltx-pipelines/src'
 else:env.pop('PYTHONPATH',None)
 with (root/'worker.log').open('w') as log:
  check=subprocess.run([str(python),'-c','import ltx_pipelines.utils.helpers as h; print(h.__file__)'],cwd=repo,env=env,capture_output=True,text=True,check=True)
  print(condition,'helpers:',check.stdout.strip(),flush=True)
  worker=subprocess.Popen([str(python),str(cli),'serve','--queue',str(queue),'--compile','--decoder-mode','chunked_compile'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
  try:
   jobs=[root/f'job-{i}.json' for i in [1,2,3]]
   if condition=='fixed':jobs.append(base/'je-job.json')
   for i,jobfile in enumerate(jobs,1):
    submitted=subprocess.check_output([str(python),str(cli),'submit','--queue',str(queue),'--job',str(jobfile)],cwd=repo,text=True).strip()
    name=Path(submitted).name.removesuffix('.job.json')
    deadline=time.monotonic()+900
    while not (queue/(name+'.done.json')).exists():
     failed=queue/(name+'.failed.json')
     if failed.exists():raise RuntimeError(failed.read_text())
     if worker.poll() is not None:raise RuntimeError(f'{condition} worker exited {worker.returncode}; see {root}/worker.log')
     if time.monotonic()>deadline:raise TimeoutError(f'{condition} job {i}')
     time.sleep(.5)
    result=json.loads((queue/(name+'.done.json')).read_text())
    print(condition,i,result,flush=True)
  finally:
   (queue/'stop').touch()
   try:worker.wait(timeout=60)
   except subprocess.TimeoutExpired:
    worker.terminate();worker.wait(timeout=60)
