from pathlib import Path
import json, subprocess, time
root=Path(__file__).parent
shared=Path('/mnt/mongo-raid/shared/Videos/FuLu')
out=Path('/home/anagrath/ai/outputs/ltx/fu-lu-full-rap-20261103')
out.mkdir(exist_ok=True)
body=Path('/home/anagrath/ai/outputs/ltx/fu-lu-full-rap-body-20261103/video.mp4')
phrase=root/'hook-loop.wav'
final=shared/'Fu Lu - Full Rap - Clean Join.mp4'
if final.exists():raise FileExistsError(final)
start=time.monotonic()
subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(shared/'Fu Lu - Walking Hook.mp4'),'-af','aresample=48000,atrim=start_sample=182400:end_sample=420000,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.005,afade=t=out:st=4.945:d=0.005','-c:a','pcm_s16le','-y',str(phrase)],check=True)
# Retain the owner-selected complete melodic phrase and the approved 450ms overlap.
filters=['[0:a]aresample=48000,atrim=end_sample=420000,asetpts=PTS-STARTPTS[lead]',
 '[1:a]asplit=12'+''.join(f'[l{i}]' for i in range(12))]
prev='lead'
for i in range(12):
    dest=f'j{i}'
    filters.append(f'[{prev}][l{i}]acrossfade=d=0.45:c1=qsin:c2=qsin[{dest}]')
    prev=dest
filters.append(f'[{prev}]atrim=end_sample=2912000[bed]')
with (root/'bed.log').open('w') as log:
 subprocess.run(['ffmpeg','-hide_banner','-nostdin','-threads','4','-i',str(shared/'Fu Lu - Walking Hook.mp4'),'-i',str(phrase),'-filter_complex_threads','8','-filter_complex',';'.join(filters),'-map','[bed]','-c:a','pcm_s16le','-y',str(root/'continuous-bed.wav')],stdout=log,stderr=subprocess.STDOUT,check=True)
filters=[
 '[0:v]settb=AVTB,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=0.5[v0]',
 '[1:v]settb=AVTB,setpts=PTS-STARTPTS[v1]',
 '[v0][v1]concat=n=2:v=1:a=0[v]',
 '[2:a]aresample=48000,apad,atrim=end_sample=498000,asetpts=PTS-STARTPTS,adelay=420000S:all=1,apad,atrim=end_sample=2912000[rapintro]',
 '[1:a]aresample=48000,apad,atrim=end_sample=1970000,asetpts=PTS-STARTPTS,adelay=942000S:all=1,apad,atrim=end_sample=2912000[rapbody]',
 '[rapintro][rapbody]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,asplit=2[speech][control]',
 '[3:a][control]sidechaincompress=threshold=0.04:ratio=5:attack=12:release=180:makeup=1[ducked]',
 '[speech][ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95:level=0:latency=1[a]'
]
(root/'filtergraph.txt').write_text(';'.join(filters)+'\n')
cmd=['ffmpeg','-hide_banner','-nostdin','-threads','4','-i',str(shared/'Fu Lu - Walking Rap - Blended Loop.mp4'),'-i',str(body),'-i','/home/anagrath/ai/outputs/ltx/fu-lu-rap-section-v2-20261103/video.mp4','-i',str(root/'continuous-bed.wav'),'-filter_complex_threads','8','-filter_complex',';'.join(filters),'-map','[v]','-map','[a]','-t','60.6666666667','-r','24','-c:v','libx264','-threads','12','-preset','fast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-ar','48000','-movflags','+faststart',str(final)]
with (root/'compose.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
alias=out/'video.mp4'
if alias.is_symlink():alias.unlink()
alias.symlink_to(final)
report={'output':str(final),'duration_s':1456/24,'frames':1456,'join_pause_s':0.5,'vocal_overlap_s':0,'body_starts_s':942000/48000,'loop_source_s':[3.8,8.75],'music_overlap_s':.45,'music_repeat_period_s':4.5,'native_ltx_voice':True,'existing_opening_preserved':True,'external_tts':False,'time_scaling':False,'compose_wall_s':round(time.monotonic()-start,3)}
(root/'composition.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
