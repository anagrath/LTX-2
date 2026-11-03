from pathlib import Path
import json, subprocess, time

root=Path(__file__).parent
output=Path('/home/anagrath/ai/outputs/ltx/je-ab-broll-edit-20261102')
output.mkdir(exist_ok=True)
video=output/'video.mp4'
if video.exists():raise FileExistsError(video)
shared=Path('/mnt/mongo-raid/shared/Videos')
a=shared/'JE All Ages - Four Windows.mp4'
b=shared/'JE All Ages - Four Windows - Take 2.mp4'
broll=root/'students-app.mp4'
timeline={
 'take_a':str(a),'take_b':str(b),
 'broll_drive_id':'1NvqtYBFnQ0inut33ER1EFhImCOHuiKsQ',
 'broll_drive_url':'https://drive.google.com/file/d/1NvqtYBFnQ0inut33ER1EFhImCOHuiKsQ/view',
 'broll_title':'Students Using Elephant Learning App.mp4',
 'fps':24,'width':576,'height':1024,
 'broll_start_b_frame':267,'broll_start_b_s':267/24,
 'broll_end_b_frame':517,'broll_end_b_s':517/24,
 'return_cue':'have in We have seen success',
 'broll_shots':[{'source_start_s':30,'frames':120,'crop':[608,1080,950,0]},
                {'source_start_s':300,'frames':130,'crop':[608,1080,850,0]}],
 'take_b_out_frame':935,'take_b_out_s':935/24,
 'take_a_in_frame':918,'take_a_in_s':918/24,
 'closing_cue':'Do me a favor, go check it out okay?',
 'broll_audio_used':False,'audio_declick_fade_ms':5,
 'expected_frames':1002,'expected_duration_s':1002/24,
 'notes':'B audio retained over B-roll, including Really; source A and B are preserved. Word timestamps are ASR estimates, not manual phonetic alignment.'
}
(root/'timeline.json').write_text(json.dumps(timeline,indent=2)+'\n')
filters=';'.join([
 '[0:v]fps=24,setpts=PTS-STARTPTS,setsar=1[bbase]',
 '[2:v]fps=24,crop=608:1080:950:0,scale=576:1024:flags=lanczos,trim=end_frame=120,setpts=PTS-STARTPTS,setsar=1,format=yuv420p[r1]',
 '[3:v]fps=24,crop=608:1080:850:0,scale=576:1024:flags=lanczos,trim=end_frame=130,setpts=PTS-STARTPTS,setsar=1,format=yuv420p[r2]',
 '[r1][r2]concat=n=2:v=1:a=0,setpts=PTS+11.125/TB[roll]',
 "[bbase][roll]overlay=eof_action=pass:repeatlast=0:enable='gte(t,11.125)*lt(t,21.5416666667)',trim=end_frame=935,setpts=PTS-STARTPTS[bv]",
 '[1:v]fps=24,trim=start_frame=918:end_frame=985,setpts=PTS-STARTPTS,setsar=1[av]',
 '[0:a]aresample=48000,atrim=end_sample=1870000,asetpts=PTS-STARTPTS,afade=t=out:st=38.9533333333:d=0.005[ba]',
 '[1:a]aresample=48000,atrim=start_sample=1836000:end_sample=1970000,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.005[aa]',
 '[bv][ba][av][aa]concat=n=2:v=1:a=1[v][a]'
])
(root/'filtergraph.txt').write_text(filters+'\n')
command=['ffmpeg','-hide_banner','-nostdin','-threads','4','-i',str(b),'-threads','4','-i',str(a),
 '-threads','4','-ss','30','-t','5.1','-i',str(broll),
 '-threads','4','-ss','300','-t','5.6','-i',str(broll),
 '-filter_complex_threads','8','-filter_complex',filters,'-map','[v]','-map','[a]',
 '-c:v','libx264','-threads','12','-preset','medium','-crf','18','-pix_fmt','yuv420p',
 '-c:a','aac','-b:a','192k','-ar','48000','-movflags','+faststart',
 '-metadata','title=Elephant Learning - All Ages',str(video)]
start=time.monotonic()
with (root/'encode.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
timeline['encode_wall_s']=round(time.monotonic()-start,3)
(root/'timeline.json').write_text(json.dumps(timeline,indent=2)+'\n')
print(json.dumps({'output':str(video),'encode_wall_s':timeline['encode_wall_s']}),flush=True)
