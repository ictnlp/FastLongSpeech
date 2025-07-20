
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
import fire
from transformers import GenerationConfig
from peft import PeftModel
import math
import os

def generate_response(predefined_length=750, model_name='', lora_model_path='', audio_config='', audio_path_root='', target_path=''):
    
    audio_file = json.load(open(audio_config, 'r', encoding='utf-8'))
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = LongSpeechQwen2AudioForCausalLM.from_pretrained(model_name, device_map="auto")
    
    if lora_model_path != '':
        model = PeftModel.from_pretrained(model, lora_model_path)
    conversation = [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"},
            {"type": "text", "text": "Answer the question: "},
        ]},
    ]
    
    response_list = []
    i = 0
    length = len(audio_file)
    for i in range(length):
        print('sample'+str(i+1))
        
        if audio_path_root == '':
            audio_url = audio_file[i]['path']
        else:
            audio_url = os.path.join(audio_path_root, audio_file[i]['path'])
            
        question = audio_file[i]['question']
        
        conversation[1]['content'][1]['text'] = question
        conversation[1]['content'][0]['audio_url'] = audio_url

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(librosa.load(
                            ele['audio_url'], 
                            sr=processor.feature_extractor.sampling_rate)[0]
                        )
        
        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda(model.device)
        inputs['attention_mask'] = inputs['attention_mask'].cuda(model.device)
        inputs['input_features'] = inputs['input_features'].cuda(model.device)
        inputs['feature_attention_mask'] = inputs['feature_attention_mask'].cuda(model.device)
        inputs['predefined_length'] = predefined_length

        generate_ids = model.generate(**inputs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        tmp_dict = {
            "speech": audio_url,
            "question": question,
            "conversations":
            {
                "from": "system",
                "value": response
            }
        }

        response_list.append(tmp_dict)

    with open(target_path, 'w', encoding='utf-8') as json_file:
        json.dump(response_list, json_file, indent=4)
    
if __name__ == "__main__":
    fire.Fire(generate_response)