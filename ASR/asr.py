from transformers import pipeline
import torch
import os

import nvidia_smi


num_devices = torch.cuda.device_count()
device = -1

nvidia_smi.nvmlInit()
for i in range(0, num_devices):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    total_memory = info.free / 1073741824
    print("Total Memory (ASR): ", total_memory)
    if total_memory > 10:
        device = i
        break
    else:
        torch.cuda.empty_cache()
        continue
nvidia_smi.nvmlShutdown()

torch.cuda.empty_cache()

pipe_en = pipeline("automatic-speech-recognition", "ASR/models/english", device=device)

pipe_hi = pipeline("automatic-speech-recognition", "ASR/models/hindi", device=device)

print("ASR Models loaded on device ", device)

def asr(audio, lang):
    
    if lang == "en":
        output = pipe_en(audio, device=device)
    elif lang == "hi":
        output = pipe_hi(audio, device=device)
    
    output_text = output["text"]
    output_text = output_text.lower().replace("<s>" , "")

    return output_text