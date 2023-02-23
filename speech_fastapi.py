from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

app = FastAPI()

import json
import base64
import torch
import tensorflow as tf
import torchaudio
import os

from filelock import FileLock

lock = FileLock("lock.txt")
lock.acquire(timeout = -1)

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

lock.release()

@app.post("///text")
async def text(request : Request):
    if request.method == 'POST':

        form = await request.form()
        # Audio File
        audio_file = await form['files'].read()

        # Paramaters
        inp = form["data"]
        input_json = json.loads(inp)
        src_lang = input_json["sourceLanguage"]

        # SSMT pipeline

        ## ASR
        asr_output = asr.asr(audio_file, src_lang)
        response_body = {
            "text": asr_output
        }
        json_data = jsonable_encoder(response_body)
        return JSONResponse(content=json_data)
        

@app.post("///speech")
async def speech(request : Request):
    if request.method == 'POST':
        form = await request.form()
        inp = form["data"]

        input_json = json.loads(inp)
        input_lang = input_json["language"]
        src_lang, tgt_lang = input_lang["sourceLanguage"], input_lang["targetLanguage"]
        input_sentence = input_json["text"]

        ## MT
        if len(input_sentence) > 0 and input_sentence[-1]!='.':
            input_sentence += "."
                
        if src_lang == "en":
            if tgt_lang == "hi":
                mt_output = mt.mt_en_hi(input_sentence)
            if tgt_lang == "mr":
                mt_output = mt.mt_en_mr(input_sentence)
        
        if src_lang == "hi":
            if tgt_lang == "mr":
                mt_output = mt.mt_hi_mr(input_sentence)

        ## TTS
        if len(mt_output) > 0 and mt_output[-1]!='.':
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
