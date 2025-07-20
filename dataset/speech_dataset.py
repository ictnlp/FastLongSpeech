import copy
import json
import torch
import transformers

from typing import Dict, Sequence
from dataclasses import dataclass
from torch.utils.data import Dataset

from FastLongSpeech.arguments import DataArguments
import librosa
import pdb

# For CTC training
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_args: DataArguments,
                 processor,
                 path_item,
                 whisper_tokenizer=None):
        super(LazySupervisedDataset, self).__init__()
        if path_item == 'train':
            list_data_dict = json.load(open(data_args.data_path, "r"))
        else:
            list_data_dict = json.load(open(data_args.validate_path, "r"))

        self.processor = processor
        self.whisper_tokenizer = whisper_tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        modal_lengths = self.modality_lengths()
        for sample in self.list_data_dict:
            ques_len = len(sample['questions']) if 'questions' in sample else 0
            resp_len = len(sample['response']) if 'response' in sample else 0
            length_list.append(ques_len + resp_len + modal_lengths.pop(0))

        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            duration = sample['duration'] if 'duration' in sample else 0
            speech_length = int(duration * 25)  # 25 is the number of embedding in 1 second

            length_list.append(speech_length)
        return length_list
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]
        if "audio" in sample:
            speech = librosa.load(sample["audio"], sr=self.processor.feature_extractor.sampling_rate)[0]
            inputs = self.processor(
                text=sample['response'],
                audios=[speech],
                return_tensors="pt",
                padding=True,
                sampling_rate=self.processor.feature_extractor.sampling_rate,
            )
            data_dict = dict(
                input_features=inputs['input_features'][0],
                feature_attention_mask=inputs['feature_attention_mask'][0],
            )
        
        if self.data_args.ctc_training:
            if self.whisper_tokenizer is None:
                data_dict['asr_labels'] = inputs['input_ids'][0]
            else:
                data_dict['asr_labels'] = torch.tensor(self.whisper_tokenizer.encode_as_ids(sample['response']))
        return data_dict
    
class SpeechQA_and_CTC_Dataset(LazySupervisedDataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_args: DataArguments,
                 processor,
                 path_item,
                 whisper_tokenizer=None):
        super().__init__(data_args, processor, path_item, whisper_tokenizer)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        modal_lengths = self.modality_lengths()
        for sample in self.list_data_dict:
            ques_len = 0
            resp_len = 0
            for sample_item in sample:
                ques_len += len(self.processor(text=sample['content'])['input_ids'])
            length_list.append(ques_len + resp_len + modal_lengths.pop(0))
            
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            duration = sample['duration'] if 'duration' in sample else 0
            speech_length = int(duration * 25)  # 25 is the number of embedding in 1 second

            length_list.append(speech_length)
        return length_list
    
    def apply_chat_template(self, audio_urls, conversations):
        begin_index = 0
        for i in range(len(conversations)):
            if conversations[i]['role'] == 'user':
                conversations[i]['content'] = [
                    {"type": "audio", "audio_url": audio_urls[begin_index]},
                    {"type": "text", "text": conversations[i]['content']},
                ]
        conversations = [{"role": "system", "content": 'You are a helpful assistant.'}] + conversations
        audios = []
        for message in conversations:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(librosa.load(
                            ele['audio_url'], 
                            sr=self.processor.feature_extractor.sampling_rate)[0]
                        )
        conversation = self.processor.apply_chat_template(conversations[:-1], add_generation_prompt=True, tokenize=False)
        response = conversations[-1]['content']

        return conversation, response, audios
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]

        conversation, response, audios = self.apply_chat_template([sample['speech']], sample['conversations'])
        
        inputs = self.processor(
            text=conversation+response+self.processor.tokenizer.eos_token,
            audios=audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
        )
        
        inputs_no_response_len = len(self.processor(text=conversation)['attention_mask'])
        labels = inputs['input_ids'][0].clone()
        labels[:inputs_no_response_len] = self.data_args.padding_idx
        
        data_dict = dict(
            input_ids=inputs['input_ids'][0],
            attention_mask=inputs['attention_mask'][0],
            input_features=inputs['input_features'][0],
            feature_attention_mask=inputs['feature_attention_mask'][0],
            labels=labels
        )
        
        if self.data_args.ctc_training:
            data_dict['asr_labels'] = torch.tensor(self.whisper_tokenizer.encode_as_ids(sample['transcription']))
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    padding_idx: int

    def __call__(self, instances, return_tensors="pt"):
        audio_feature, feature_mask, asr_targets = tuple([instance[key] for instance in instances]
                                  for key in ("input_features", "feature_attention_mask", "asr_labels"))
        audio_feature = torch.nn.utils.rnn.pad_sequence(
            audio_feature,
            batch_first=True,
            padding_value=self.padding_idx)
        feature_mask = torch.nn.utils.rnn.pad_sequence(
            feature_mask,
            batch_first=True,
            padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(
            asr_targets,
            batch_first=True,
            padding_value=self.padding_idx)
        
        batch = dict(
            input_features=audio_feature,
            feature_attention_mask=feature_mask,
            asr_labels=labels,
            return_loss=True
        )
        return batch

@dataclass
class DataCollatorLLMsTraining(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    padding_idx: int
    
    def __call__(self, instances, return_tensors="pt"):
        input_ids, attention_mask, audio_feature, feature_mask, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "attention_mask", "input_features", "feature_attention_mask", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0)
        audio_feature = torch.nn.utils.rnn.pad_sequence(
            audio_feature,
            batch_first=True,
            padding_value=self.padding_idx)
        feature_mask = torch.nn.utils.rnn.pad_sequence(
            feature_mask,
            batch_first=True,
            padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.padding_idx)
        
        batch = dict(
            input_ids=input_ids,
            input_features=audio_feature,
            attention_mask=attention_mask,
            feature_attention_mask=feature_mask,
            labels=labels,
            return_loss=True
        )

        if 'asr_labels' in instances[0].keys():
            asr_targets = tuple([instance['asr_labels'] for instance in instances])
            asr_labels = torch.nn.utils.rnn.pad_sequence(
                asr_targets,
                batch_first=True,
                padding_value=self.padding_idx
            )
            batch['asr_labels'] = asr_labels
        return batch

def make_supervised_data_module(processor,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor=processor,
                                          path_item='train',
                                            data_args=data_args)
    evaluate_dataset = LazySupervisedDataset(processor=processor,
                                             path_item='validate',
                                            data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=processor, padding_idx=data_args.padding_idx)
    return dict(train_dataset=train_dataset,
                eval_dataset=evaluate_dataset,
                data_collator=data_collator)
    
    
def make_supervised_ctc_decoder_data_module(processor,
                                ctc_decoder_processor,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor=processor,
                                          whisper_tokenizer=ctc_decoder_processor,
                                          path_item='train',
                                            data_args=data_args)
    evaluate_dataset = LazySupervisedDataset(processor=processor,
                                             whisper_tokenizer=ctc_decoder_processor,
                                             path_item='validate',
                                            data_args=data_args)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=processor, padding_idx=data_args.padding_idx)
    return dict(train_dataset=train_dataset,
                eval_dataset=evaluate_dataset,
                data_collator=data_collator)
    
    
def make_dialogueAnswer_ctc_module(processor,
                                ctc_decoder_processor,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SpeechQA_and_CTC_Dataset(processor=processor,
                                          whisper_tokenizer=ctc_decoder_processor,
                                          path_item='train',
                                            data_args=data_args)
    evaluate_dataset = SpeechQA_and_CTC_Dataset(processor=processor,
                                             whisper_tokenizer=ctc_decoder_processor,
                                             path_item='validate',
                                            data_args=data_args)
    
    data_collator = DataCollatorLLMsTraining(tokenizer=processor, padding_idx=data_args.padding_idx)
    return dict(train_dataset=train_dataset,
                eval_dataset=evaluate_dataset,
                data_collator=data_collator)
