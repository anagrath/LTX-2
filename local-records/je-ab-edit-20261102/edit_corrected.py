from pathlib import Path
import json, subprocess, time

root=Path(__file__).parent
output=Path('/home/anagrath/ai/outputs/ltx/je-ab-broll-corrected-20261102')
output.mkdir(exist_ok=True)
video=output/'video.mp4'
if video.exists():raise FileExistsError(video)
shared=Path('/mnt/mongo-raid/shared/Videos')
a=shared/'JE All Ages - Four Windows.mp4'
b=shared/'JE All Ages - Four Windows - Take 2.mp4'
roll=root/'students-app.mp4'
# All source video is 24 fps. Cut at frame boundaries; never scale time or audio tempo.
timeline={'take_a':str(a),'take_b':str(b),'fps':24,
 'removed_b_frames':[267,318],'removed_b_s':[267/24,318/24],
 'broll_output_frames':[267,466],'broll_output_s':[267/24,466/24],
 'return_b_source_frame':517,'return_b_source_s':517/24,
 'closing_b_out_frame':935,'closing_a_in_frame':918,
 'closing_output_frame':884,'closing_output_s':884/24,
 'expected_frames':951,'expected_duration_s':951/24,
 'broll_shots':[{'start_s':30,'frames':96,'crop':[608,1080,950,0]},
                {'start_s':300,'frames':103,'crop':[608,1080,850,0]}],
 'broll_drive_id':'1NvqtYBFnQ0inut33ER1EFhImCOHuiKsQ',
 'broll_audio_used':False,'speed_multiplier':1.0,
 'note':'Really removed from both picture and audio. Source B audio preserved under B-roll; A supplies picture and audio for closing. No tempo or playback-rate filter.'}
(root/'corrected-timeline.json').write_text(json.dumps(timeline,indent=2)+'\n')
filters=';'.join([
 '[0:v]split=2[b0][b1]',
 '[0:a]asplit=3[ba0][ba1][ba2]',
 '[b0]trim=end_frame=267,setpts=PTS-STARTPTS,setsar=1[v0]',
 '[ba0]atrim=end_sample=534000,asetpts=PTS-STARTPTS,afade=t=out:st=11.120:d=0.005[a0]',
 '[2:v]fps=24,crop=608:1080:950:0,scale=576:1024:flags=lanczos,trim=end_frame=96,setpts=PTS-STARTPTS,setsar=1,format=yuv420p[r0]',
 '[3:v]fps=24,crop=608:1080:850:0,scale=576:1024:flags=lanczos,trim=end_frame=103,setpts=PTS-STARTPTS,setsar=1,format=yuv420p[r1]',
 '[r0][r1]concat=n=2:v=1:a=0[vr]',
 '[ba1]atrim=start_sample=636000:end_sample=1034000,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.005[ar]',
 '[b1]trim=start_frame=517:end_frame=935,setpts=PTS-STARTPTS,setsar=1[vb]',
 '[ba2]atrim=start_sample=1034000:end_sample=1870000,asetpts=PTS-STARTPTS,afade=t=out:st=17.4116666667:d=0.005[ab]',
 '[1:v]trim=start_frame=918:end_frame=985,setpts=PTS-STARTPTS,setsar=1[va]',
 '[1:a]atrim=start_sample=1836000:end_sample=1970000,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.005[aa]',
 '[v0][a0][vr][ar][vb][ab][va][aa]concat=n=4:v=1:a=1[v][a]'
])
(root/'corrected-filtergraph.txt').write_text(filters+'\n')
command=['ffmpeg','-hide_banner','-nostdin','-threads','4','-i',str(b),'-threads','4','-i',str(a),
 '-threads','4','-ss','30','-t','4.1','-i',str(roll),
 '-threads','4','-ss','300','-t','4.5','-i',str(roll),
 '-filter_complex_threads','8','-filter_complex',filters,'-map','[v]','-map','[a]',
 '-c:v','libx264','-threads','12','-preset','medium','-crf','18','-pix_fmt','yuv420p',
 '-c:a','aac','-b:a','192k','-ar','48000','-movflags','+faststart',
 '-metadata','title=Elephant Learning - All Ages',str(video)]
start=time.monotonic()
with (root/'corrected-encode.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
timeline['encode_wall_s']=round(time.monotonic()-start,3)
(root/'corrected-timeline.json').write_text(json.dumps(timeline,indent=2)+'\n')
print(json.dumps({'output':str(video),'encode_wall_s':timeline['encode_wall_s']}),flush=True)
