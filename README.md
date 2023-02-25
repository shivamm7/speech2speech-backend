# SSMT: Speech-to-Speech Machine Translation System

# Current Deployment

URL: https://www.cfilt.iitb.ac.in/ssmt/speech2speech

# How to Run?

1. Set the port number on which to run the backend in `uvicorn_worker.py` file.
2. Set the number of workers in `uvicorn_worker.py` file. (Number of workers is how many instances of the SSMT pipelines to load)
3. Run the `uvicorn_worker.py` file with command `python3 uvicorn_worker.py`

## **Note:** The repo does not contain all the models.

# Working

1. The SSMT pipeline consists of 3 models, Automatic Speech Recognition (ASR), Machine Translation (MT) and Text-to-Speech (TTS) models.
2. The input speech is passed to the ASR model which transribes the speech and generated the text in source language.
3. The source language text is passed through the MT model which translated the source langauge text to target language text.
4. The target language text is passed to the TTS model which generates the speech in target language.

# Deployment

1. The code is written in such a way that the multiple SSMT pipelines on a single GPU and also across multiple GPUs.
2. The free memory on a GPU is first checked and if sufficient memory is available on a GPU then the models are loaded on that GPU.
3. If sufficient free space is not available on a GPU then the next GPU on the machine is checked.
4. Example: Consider a DGX A100 machine which consists of 8 Nvidia A100 GPUs and the SSMT pipeline occupies a space of 6GB. Then on a single GPU 13 SSMT pipelines can be run. So, across 8 GPUs a total of 13*8=104 SSMT pipelines can be run.
5. The code is written is such a way that it can dynamically load models on multi GPUs machines to utilize the entire GPU memory.

# Frontend Repositry
Link: https://github.com/shivamm7/speech2speech-frontend
