from pathlib import Path
import json,subprocess,time

root=Path(__file__).parent
while not (root/'driver-timing.json').exists():
    if 'Traceback' in (root/'driver.log').read_text():raise RuntimeError((root/'driver.log').read_text())
    time.sleep(.5)
intro=Path('/mnt/mongo-raid/shared/Videos/Fu Lu - Walking Hook.mp4')
rap=Path('/home/anagrath/ai/outputs/ltx/fu-lu-rap-section-20261103/video.mp4')
out=Path('/home/anagrath/ai/outputs/ltx/fu-lu-walk-rap-20261103')
out.mkdir(exist_ok=True);video=out/'video.mp4'
if video.exists():raise FileExistsError(video)
phase=(8.875-.336)%1.12
filters=';'.join([
 '[0:v]settb=AVTB,setpts=PTS-STARTPTS[v0]',
 '[1:v]settb=AVTB,setpts=PTS-STARTPTS[v1]',
 '[v0][v1]xfade=transition=fade:duration=0.1666666667:offset=8.875,format=yuv420p,trim=end_frame=430[v]',
 '[0:a]aresample=48000,apad,atrim=end_sample=434000,asetpts=PTS-STARTPTS[orig]',
 '[1:a]aresample=48000,apad,atrim=end_sample=434000,asetpts=PTS-STARTPTS,asplit=2[speech][control]',
 f'[2:a]atrim=start={phase:.9f}:end={phase+217/24:.9f},asetpts=PTS-STARTPTS,volume=0.6[beat]',
 '[beat][control]sidechaincompress=threshold=0.04:ratio=5:attack=12:release=180:makeup=1[ducked]',
 '[speech][ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95:level=0:latency=1[rapmix]',
 '[orig][rapmix]acrossfade=d=0.1666666667:c1=tri:c2=tri[a]'
])
(root/'filtergraph.txt').write_text(filters+'\n')
command=['ffmpeg','-hide_banner','-nostdin','-threads','4','-i',str(intro),'-threads','4','-i',str(rap),
 '-stream_loop','-1','-i',str(root/'beat-loop.wav'),'-filter_complex_threads','8',
 '-filter_complex',filters,'-map','[v]','-map','[a]','-t',str(430/24),
 '-c:v','libx264','-threads','12','-preset','medium','-crf','18','-pix_fmt','yuv420p',
 '-c:a','aac','-b:a','192k','-ar','48000','-movflags','+faststart',str(video)]
start=time.monotonic()
with (root/'compose.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
report={'output':str(video),'intro':str(intro),'rap_section':str(rap),'seed_source_frame':216,
 'continuation_type':'last-frame image-seeded new native clip, editorial crossfade; not latent-state extension',
 'beat_source_start_s':.336,'beat_loop_s':1.12,'estimated_bpm':60/.56,'loop_phase_offset_s':phase,
 'intro_video_audio_preserved_before_s':8.875,'bridge_frames':4,'bridge_s':4/24,
 'expected_frames':430,'expected_duration_s':430/24,'music_ducked_under_rap':True,
 'new_vocal_source':'native LTX audio prompted dry a cappella rap; no separate TTS',
 'compose_wall_s':round(time.monotonic()-start,3)}
(root/'composition.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
