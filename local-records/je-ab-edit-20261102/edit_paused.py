from pathlib import Path
import json, subprocess, time

root=Path(__file__).parent
output=Path('/home/anagrath/ai/outputs/ltx/je-ab-paused-20261102')
output.mkdir(exist_ok=True)
video=output/'video.mp4'
if video.exists():raise FileExistsError(video)
shared=Path('/mnt/mongo-raid/shared/Videos')
a=shared/'JE All Ages - Four Windows.mp4'
b=shared/'JE All Ages - Four Windows - Take 2.mp4'
roll=root/'students-app.mp4'
timeline={
 'take_a':str(a),'take_b':str(b),'fps':24,'speed_multiplier':1.0,
 'removed_b_intervals_frames':[[267,318],[456,508]],
 'removed_b_intervals_s':[[267/24,318/24],[456/24,508/24]],
 'silence_observed_previous_edit_s':[16.749396,19.203104],
 'broll_output_frames':[267,419],'broll_output_s':[267/24,419/24],
 'return_b_source_frame':522,'return_b_source_s':522/24,
 'return_cue':'seen in We have seen success; We have remains under B-roll',
 'take_b_out_frame':751,'take_b_out_s':751/24,
 'take_a_in_frame':732,'take_a_in_s':732/24,
 'take_a_output_frame':648,'take_a_output_s':27.0,
 'take_a_cue':'Quiet natural A lead-in before If your student; use A through the end', 'a_quiet_lead_in_s':31.156042-30.5, 'added_a_lead_in_frames':13,
 'expected_frames':901,'expected_duration_s':901/24,
 'broll_shots':[{'source_start_s':30,'frames':96,'crop':[608,1080,950,0]},
                {'source_start_s':300,'frames':56,'crop':[608,1080,850,0]}],
 'broll_drive_id':'1NvqtYBFnQ0inut33ER1EFhImCOHuiKsQ',
 'broll_audio_used':False,'audio_declick_fade_ms':5,
 'note':'Built from original A/B. Really removed; gap collapsed while retaining short natural pause. A begins at30.5s within its quiet lead-in; no time scaling or tempo filters.'
}
(root/'paused-timeline.json').write_text(json.dumps(timeline,indent=2)+'\n')
filters=';'.join([
 '[0:v]split=2[b0][b1]',
 '[0:a]asplit=4[ba0][ba1][ba2][ba3]',
 '[b0]trim=end_frame=267,setpts=PTS-STARTPTS,setsar=1[v0]',
 '[ba0]atrim=end_sample=534000,asetpts=PTS-STARTPTS,afade=t=out:st=11.120:d=0.005[a0]',
 '[2:v]fps=24,crop=608:1080:950:0,scale=576:1024:flags=lanczos,trim=end_frame=96,setpts=PTS-STARTPTS,setsar=1,format=yuv420p[r0]',
 '[3:v]fps=24,crop=608:1080:850:0,scale=576:1024:flags=lanczos,trim=end_frame=56,setpts=PTS-STARTPTS,setsar=1,format=yuv420p[r1]',
 '[r0][r1]concat=n=2:v=1:a=0[vr]',
 '[ba1]atrim=start_sample=636000:end_sample=912000,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.005,afade=t=out:st=5.745:d=0.005[ar0]',
 '[ba2]atrim=start_sample=1016000:end_sample=1044000,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.005[ar1]',
 '[ar0][ar1]concat=n=2:v=0:a=1[ar]',
 '[b1]trim=start_frame=522:end_frame=751,setpts=PTS-STARTPTS,setsar=1[vb]',
 '[ba3]atrim=start_sample=1044000:end_sample=1502000,asetpts=PTS-STARTPTS,afade=t=out:st=9.5366666667:d=0.005[ab]',
 '[1:v]trim=start_frame=732:end_frame=985,setpts=PTS-STARTPTS,setsar=1[va]',
 '[1:a]atrim=start_sample=1464000:end_sample=1970000,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.005[aa]',
 '[v0][a0][vr][ar][vb][ab][va][aa]concat=n=4:v=1:a=1[v][a]'
])
(root/'paused-filtergraph.txt').write_text(filters+'\n')
command=['ffmpeg','-hide_banner','-nostdin','-threads','4','-i',str(b),'-threads','4','-i',str(a),
 '-threads','4','-ss','30','-t','4.1','-i',str(roll),
 '-threads','4','-ss','300','-t','2.5','-i',str(roll),
 '-filter_complex_threads','8','-filter_complex',filters,'-map','[v]','-map','[a]',
 '-c:v','libx264','-threads','12','-preset','medium','-crf','18','-pix_fmt','yuv420p',
 '-c:a','aac','-b:a','192k','-ar','48000','-movflags','+faststart',
 '-metadata','title=Elephant Learning - All Ages',str(video)]
start=time.monotonic()
with (root/'paused-encode.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
timeline['encode_wall_s']=round(time.monotonic()-start,3)
(root/'paused-timeline.json').write_text(json.dumps(timeline,indent=2)+'\n')
print(json.dumps({'output':str(video),'encode_wall_s':timeline['encode_wall_s']}),flush=True)
