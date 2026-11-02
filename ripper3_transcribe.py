#!/usr/bin/env python3
"""Transcribe local media with cached Whisper and save a speech-content check."""
import argparse
import json
from pathlib import Path
import av
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = p.parse_args()
    samples=[]
    with av.open(str(args.input)) as c:
        resampler=av.AudioResampler(format='fltp',layout='mono',rate=16000)
        for f in c.decode(audio=0):
            for r in resampler.resample(f):samples.append(r.to_ndarray().reshape(-1))
        for r in resampler.resample(None):samples.append(r.to_ndarray().reshape(-1))
    audio=np.concatenate(samples).astype(np.float32)
    repo='openai/whisper-small.en';rev='e8727524f962ee844a7319d92be39ac1bd25655a'
    processor=AutoProcessor.from_pretrained(repo,revision=rev,local_files_only=True)
    dtype=torch.float16 if args.device=='cuda' else torch.float32
    model=AutoModelForSpeechSeq2Seq.from_pretrained(repo,revision=rev,dtype=dtype,local_files_only=True).to(args.device).eval()
    encoded=processor(audio,sampling_rate=16000,return_tensors='pt',truncation=False,
        padding='longest',return_attention_mask=True)
    features=encoded.input_features.to(args.device,dtype=dtype)
    with torch.inference_mode():
        tokens=model.generate(features,attention_mask=encoded.attention_mask.to(args.device),
            max_new_tokens=440,return_timestamps=True)
    text=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
    result={'source':str(args.input),'model':repo,'revision':rev,'text':text,
            'transcription_mode':'native timestamped long-form; no independent hard cuts'}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(result['text'],flush=True)


if __name__=='__main__':main()
