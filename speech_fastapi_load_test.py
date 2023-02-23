import warnings
warnings.filterwarnings('ignore')


from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

app = FastAPI()

import json
import base64
import torch
import tensorflow as tf

import time
import multiprocessing

from filelock import FileLock

lock = FileLock("lock.txt")
lock.acquire(timeout = -1)

# num_devices = 8
# device = -1
# for i in range(0, num_devices):
#     total_memory = torch.cuda.mem_get_info(i)[0] / 1073741824
#     print("Total Memory (ASR): ", total_memory)
#     if total_memory > 15:
#         device = i
#         r = torch.cuda.memory_reserved(i)
#         a = torch.cuda.memory_allocated(i)
#         # print("ASR ", i, r, a)
#         break
#     else:
#         torch.cuda.empty_cache()
#         continue

num_devices = 8
device = -1
for i in range(0, num_devices):
    a = len(torch.cuda.list_gpu_processes(i).split("\n"))
    a = a - 1
    if a < 13:
        print(i, a)
        device = i
        break
    elif a < 14 and (i == 0 or i == 3):
        print(i, a)
        device = i
        break
    else:
        torch.cuda.empty_cache()
        continue

with open("/raid/nlp/shivam/speech-api/gpu.txt", "w") as f:
    f.write(str(device))

try:
    # SSMT
    print("loading ASR...")
    import ASR.asr as asr
    print("ASR loaded")

    print("loading MT...")
    import MT.mt as mt
    print("MT loaded")

    print("loading TTS...")
    import TTS.tts as tts
    print("TTS loaded")

finally:
    lock.release()

@app.post("/speech")
async def text(request : Request):
    if request.method == 'POST':
        
        form = await request.form()
        # print([f for f in form])
        # Audio File
        audio_file = await form['files'].read()

        # Paramaters
        input_json = form
        src_lang, tgt_lang = input_json["sourceLanguage"], input_json["targetLanguage"]

        # SSMT pipeline
        ## ASR
        asr_output = asr.asr(audio_file, src_lang)
        input_sentence = asr_output

        ## MT
        if(input_sentence[-1]!='.'):
            input_sentence += "."
                
        if src_lang == "en":
            if tgt_lang == "hi":
                mt_output = mt.mt_en_hi(input_sentence)
            if tgt_lang == "mr":
                mt_output = mt.mt_en_mr(input_sentence)
        
        if src_lang == "hi":
            if tgt_lang == "mr":
                mt_output = mt.mt_hi_mr(input_sentence)

        if src_lang == "mr" or src_lang == "mr_2":
            if tgt_lang == "hi":
                mt_output = mt.mt_mr_hi(input_sentence)

        ## TTS
        if(mt_output[-1]!='.'):
            mt_output += "."
        tts_output = tts.tts(mt_output, tgt_lang)
        
        tts_output_np = tts_output.numpy()
        tts_output_np = tts_output_np.reshape(-1, 1)
        tts_output_tf = tf.convert_to_tensor(tts_output_np)
        tts_output_wav = tf.audio.encode_wav(tts_output_tf, 22050, name=None)
        tts_output_bytes = tts_output_wav.numpy()

        audio_data = base64.b64encode(tts_output_bytes)

        audio_data_string = audio_data.decode('utf-8')

        response_body = {
            "data": audio_data_string,
            "text": {
                "source_text": input_sentence,
                "target_text": mt_output
            }
        }
        json_data = jsonable_encoder(response_body)
        return JSONResponse(content=json_data)