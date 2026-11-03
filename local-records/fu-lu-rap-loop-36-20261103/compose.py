from pathlib import Path
import json,subprocess,time

root=Path(__file__).parent
intro=Path('/mnt/mongo-raid/shared/Videos/Fu Lu - Walking Hook.mp4')
rap=Path('/home/anagrath/ai/outputs/ltx/fu-lu-rap-section-v2-20261103/video.mp4')
out=Path('/home/anagrath/ai/outputs/ltx/fu-lu-rap-loop-36-20261103')
out.mkdir(exist_ok=True);video=out/'video.mp4'
if video.exists():raise FileExistsError(video)
# The owner chose the complete melodic phrase, not a tempo-derived loop.
# Restart it at its selected end in the original, independently of the picture cut.
filters=';'.join([
 '[0:v]settb=AVTB,setpts=PTS-STARTPTS[v0]',
 '[1:v]settb=AVTB,setpts=PTS-STARTPTS[v1]',
 '[v0][v1]xfade=transition=fade:duration=0.2916666667:offset=8.75,format=yuv420p,trim=end_frame=459[v]',
 '[0:a]aresample=48000,atrim=end_sample=415200,asetpts=PTS-STARTPTS,afade=t=out:st=8.645:d=0.005[lead]',
 '[2:a]atrim=end_sample=502800,asetpts=PTS-STARTPTS[loops]',
 '[lead][loops]concat=n=2:v=0:a=1[bed]',
 '[1:a]aresample=48000,apad,atrim=end_sample=498000,asetpts=PTS-STARTPTS,adelay=420000S:all=1,apad,atrim=end_sample=918000,asplit=2[speech][control]',
 '[bed][control]sidechaincompress=threshold=0.04:ratio=5:attack=12:release=180:makeup=1[ducked]',
 '[speech][ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95:level=0:latency=1[a]'
])
(root/'filtergraph.txt').write_text(filters+'\n')
command=['ffmpeg','-hide_banner','-nostdin','-threads','4','-i',str(intro),'-threads','4','-i',str(rap),
 '-stream_loop','-1','-i',str(root/'hook-loop.wav'),'-filter_complex_threads','8',
 '-filter_complex',filters,'-map','[v]','-map','[a]','-t',str(459/24),
 '-c:v','libx264','-threads','12','-preset','medium','-crf','18','-pix_fmt','yuv420p',
 '-c:a','aac','-b:a','192k','-ar','48000','-movflags','+faststart',str(video)]
start=time.monotonic()
with (root/'compose.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
report={'output':str(video),'intro':str(intro),'rap_section':str(rap),
 'owner_selected_loop_start_s':3.6,'owner_selected_loop_end_s':8.65,'loop_duration_s':5.05,
 'loop_source_samples':[172800,415200],'sample_rate':48000,
 'music_restart_output_s':8.65,'subsequent_loop_restarts_s':[13.7,18.75],
 'voice_section_output_start_s':8.75,'video_bridge_frames':7,'video_bridge_s':7/24,
 'frames':459,'duration_s':459/24,'rap_regenerated':False,'music_ducked_under_rap':True,
 'loop_edge_declick_ms':5,'loop_period_shortened_by_crossfade':False,
 'compose_wall_s':round(time.monotonic()-start,3)}
(root/'composition.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
