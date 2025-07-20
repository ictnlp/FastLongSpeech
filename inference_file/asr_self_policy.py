import os
from io import BytesIO
from urllib.request import urlopen
import librosa
import sys

from FastLongSpeech.models.qwen2audio_withCTC import LongSpeechQwen2AudioForCausalLM
from transformers import AutoProcessor
import json
import pdb
import time
import torch
import fire
from transformers import GenerationConfig
import sentencepiece as spm
# 指定目录路径

def generate_response(data, model_name, audio_air):
    processor = AutoProcessor.from_pretrained(model_name)
    model = LongSpeechQwen2AudioForCausalLM.from_pretrained(model_name, device_map="auto")
    # conversation = [
    #     {"role": "system", "content": 'You are a helpful assistant.'},
    #     {"role": "user", "content": [
    #         {"type": "audio", "audio_url": audio_air},
    #         {"type": "text", "text": instruction},
    #     ]},
    # ]
    ctc_decoder_tokenizer = spm.SentencePieceProcessor(model_file='../spm.model')
    response_list = []

    raw_audios = librosa.load(audio_url, sr=processor.feature_extractor.sampling_rate)[0]
    
    inputs = processor(
        text='',
        audios=[raw_audios],
        return_tensors="pt",
        padding=True,
        sampling_rate=processor.feature_extractor.sampling_rate,
    )
    input_features = inputs['input_features'].to(device='cuda:0')
    feature_attention_mask = inputs['feature_attention_mask'].to(device='cuda:0')
    with torch.no_grad():
        generate_ids = model.forward(input_features=input_features, feature_attention_mask=feature_attention_mask)
        
    asr_response = ctc_decoder_tokenizer.decode_ids(generate_ids[0].cpu().numpy().tolist())
    
    return asr_response
    
if __name__ == "__main__":
    fire.Fire(generate_response)