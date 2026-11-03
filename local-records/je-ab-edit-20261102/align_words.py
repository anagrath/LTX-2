from pathlib import Path
import json
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

torch.set_num_threads(16)
torch.set_num_interop_threads(1)
root=Path(__file__).parent
repo='openai/whisper-small.en'
rev='e8727524f962ee844a7319d92be39ac1bd25655a'
processor=AutoProcessor.from_pretrained(repo,revision=rev,local_files_only=True)
model=AutoModelForSpeechSeq2Seq.from_pretrained(repo,revision=rev,dtype=torch.float32,local_files_only=True,attn_implementation='eager').eval()
recognizer=pipeline('automatic-speech-recognition',model=model,tokenizer=processor.tokenizer,feature_extractor=processor.feature_extractor,device=-1)
for name in ['b','a']:
    with torch.inference_mode():
        result=recognizer(str(root/(name+'.wav')),return_timestamps='word',chunk_length_s=30,stride_length_s=5)
    (root/(name+'-words.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(name,json.dumps(result),flush=True)
