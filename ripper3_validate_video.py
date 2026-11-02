#!/usr/bin/env python3
"""Decode a local result, check its tracks, and create a diagnostic contact sheet."""
import argparse
import hashlib
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',type=Path)
    p.add_argument('--expected-frames',type=int)
    args=p.parse_args()
    root=args.video.parent
    with av.open(str(args.video)) as c:
        stream=c.streams.video[0]
        fps=float(stream.average_rate);estimated=stream.frames
        targets=set(np.linspace(0,max(0,estimated-1),12,dtype=int).tolist())
        hashes=set();dark=[];selected=[];count=0
        for i,frame in enumerate(c.decode(video=0)):
            pixels=frame.to_ndarray(format='rgb24');hashes.add(hashlib.sha256(pixels.tobytes()).hexdigest())
            if float(pixels.mean())<2:dark.append(i)
            if i in targets:
                image=frame.to_image();path=root/f'review-frame-{i:04d}.png';image.save(path)
                selected.append((i,image))
            count=i+1
        result={'video':str(args.video),'width':stream.width,'height':stream.height,'fps':fps,
            'frame_count':count,'video_duration_s':count/fps,'distinct_frames':len(hashes),
            'near_black_frames':dark,'sample_frames':[i for i,_ in selected]}
    with av.open(str(args.video)) as c:
        audio=c.streams.audio[0]
        blocks=[f.to_ndarray().astype(np.float64) for f in c.decode(audio=0)]
        samples=np.concatenate(blocks,axis=-1)
        result.update(audio_rate=audio.rate,audio_channels=audio.channels,
            audio_samples_per_channel=samples.shape[-1],audio_duration_s=samples.shape[-1]/audio.rate,
            audio_rms=float(np.sqrt(np.mean(samples*samples))),audio_peak=float(np.max(np.abs(samples))),
            audio_finite=bool(np.isfinite(samples).all()))
    result['audio_video_duration_delta_s']=result['audio_duration_s']-result['video_duration_s']
    result['sha256']=hashlib.sha256(args.video.read_bytes()).hexdigest()
    if args.expected_frames is not None:assert count==args.expected_frames,(count,args.expected_frames)
    assert result['audio_finite'] and result['audio_rms']>0 and not dark,result
    sheet=Image.new('RGB',(4*180,3*344),'#161616');draw=ImageDraw.Draw(sheet)
    for n,(i,image) in enumerate(selected):
        image.thumbnail((180,320));x=(n%4)*180;y=(n//4)*344
        sheet.paste(image,(x,y));draw.text((x+5,y+322),f'{i/fps:.2f}s',fill='white')
    sheet.save(root/'review-contact-sheet.jpg',quality=92)
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
