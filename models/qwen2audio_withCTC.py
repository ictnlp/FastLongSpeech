  #    Copyright 2023 Haotian Liu
#    Copyright 2024 Qingkai Fang
#
#    This project is modified based on LLaVA by Haotian Liu, Qingkai Fang adds further supports for speech-to-text/speech tasks.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Tuple, Union
import random
import pdb
import torch
import math
import torch.nn as nn
from torch.nn import Embedding
from .ctc_decoder.builder import build_ctc_decoder, build_self_ctc_decoder
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache
import deepspeed
from transformers.generation.utils import GenerateOutput

from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioConfig
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

@dataclass
class LongSpeechQwen2AudioCausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None


class LongSpeechQwen2AudioConfig(Qwen2AudioConfig):
    model_type = "long_speech_qwen2audio"
    
    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_index=151646,
        ctc_config=None,
        **kwargs,
    ):
        self.ctc_config = ctc_config

        super().__init__(audio_config=audio_config, text_config=text_config, audio_token_index=audio_token_index, **kwargs)

class LongSpeechQwen2AudioForCausalLM(Qwen2AudioForConditionalGeneration):
    config_class = LongSpeechQwen2AudioConfig
    def __init__(self, config: LongSpeechQwen2AudioConfig):
        super().__init__(config)
        self.config.hidden_size = self.language_model.get_input_embeddings().embedding_dim
        self.ctc_decoder = None
        self.indicator = False
        self.current_step = 100
        
        if hasattr(config, "ctc_decoder"):
            config.ctc_decoder_input_dim = self.language_model.get_input_embeddings().embedding_dim
            config.ctc_decoder_num_embeddings = self.language_model.get_input_embeddings().num_embeddings
            if hasattr(config, "ctc_embed_num") and config.ctc_embed_num > 0:
                self.ctc_decoder = build_self_ctc_decoder(config)
            else:    
                self.ctc_decoder = build_ctc_decoder(config)
        
        self.post_init()

    def get_ctc_decoder(self):
        ctc_decoder = getattr(self, 'ctc_decoder', None)
        
        return ctc_decoder
    
    def initialize_ctc_decoder(self, model_args):
        self.config.ctc_decoder = getattr(model_args, "ctc_decoder", None)
        self.config.ctc_embed_num = getattr(model_args, "ctc_embed_num", None)
        if self.get_ctc_decoder() is None:
            self.config.ctc_decoder_input_dim = self.language_model.get_input_embeddings().embedding_dim
            self.config.ctc_decoder_num_embeddings = self.language_model.get_input_embeddings().num_embeddings
            if self.config.ctc_embed_num > 0:
                self.ctc_decoder = build_self_ctc_decoder(self.config)
            else:
                self.ctc_decoder = build_ctc_decoder(self.config)
    
    def initial_partial_parameters(self):
        with torch.no_grad():
            tmp_linear = nn.Linear(self.language_model.get_input_embeddings().embedding_dim, 1, bias=False).to(device=self.language_model.device)
            nn.init.xavier_uniform_(tmp_linear.weight)
            
            tmp_weight = torch.cat([self.language_model.get_input_embeddings().weight, tmp_linear.weight], dim=0)
            self.ctc_decoder.linear.weight.copy_(tmp_weight)
            print(self.ctc_decoder.projector.weight[0])
            if self.config.ctc_decoder == "projector_llm_embed":
                self.ctc_decoder.projector.weight.copy_(self.multi_modal_projector.linear.weight)
                self.ctc_decoder.projector.bias.copy_(self.multi_modal_projector.linear.bias)
            print(self.ctc_decoder.projector.weight[0])
            print(self.multi_modal_projector.linear.weight[0])
    
    def compute_ctc_loss(self, hidden_inputs, asr_labels, audio_lengths):
        hidden_inputs = self.ctc_decoder(hidden_inputs).to(dtype=torch.float32).transpose(0, 1)
        probs = torch.log_softmax(hidden_inputs, dim=-1)
        target_lengths = asr_labels.ne(self.config.ignore_index).sum(-1)
        ctc_loss = nn.functional.ctc_loss(probs, asr_labels, audio_lengths, target_lengths, blank=self.ctc_decoder.linear.weight.size(0)-1)
        
        return ctc_loss
# 
    def get_model(self):
        return self.model

    def audio_features_forward(self, input_features, feature_attention_mask, predefined_length):
        audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
            feature_attention_mask.sum(-1)
        )
        batch_size, _, max_mel_seq_len = input_features.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.audio_tower.conv1.weight.dtype, device=self.audio_tower.conv1.weight.device
        )
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        audio_outputs = self.audio_tower(input_features, attention_mask=audio_attention_mask, predefined_length=predefined_length, audio_output_lengths=audio_output_lengths)
        selected_audio_feature = audio_outputs.last_hidden_state

        #audio_features = self.multi_modal_projector(selected_audio_feature)
        if self.config.ctc_decoder == "projector_llm_embed":
            audio_features = self.ctc_decoder.projector(selected_audio_feature)
        else:
            audio_features = self.multi_modal_projector(selected_audio_feature)
            
        return audio_features, audio_output_lengths

    def remove_blank_consecutive_tokens(
        self,
        output_indices, 
        blank_ids
    ):
        output_list = []
        for line in output_indices:
            output_list.append(torch.unique_consecutive(line[line != blank_ids]))
            
        return output_list

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        input_features=None,  # Ignore copy
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Ignore copy
            # Here, we get the attention_mask, which was previously stored in the state after _merge_input_ids_with_audio_features.
            if input_features is not None and kwargs.get("attention_mask") is not None:
                attention_mask = kwargs["attention_mask"]
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.audio_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Ignore copy
        feature_attention_mask = kwargs.get("feature_attention_mask", None)
        predefined_length = kwargs.get("predefined_length", None)
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "input_features": input_features,
                "feature_attention_mask": feature_attention_mask,
                "predefined_length": predefined_length,
            }
        )
        return model_inputs
    # For LongSpeech_Inputs Inference
    def merge_audios_reforward_onlysimilar_iterate_inte_blank_logit_longSpeech(
        self,
        audio_features,
        selected_features,
        predefined_length,
        audio_output_lengths
    ):
        selected_features = selected_features[:, :audio_output_lengths[0], :]
        #origin_length = selected_features.size(1)
        
        audio_features = selected_features[0]
        
        # judge interval
        #candidate_pool = [400, 200, 100, 50]
        flag = False
        #candidate_pool =  [500, 400, 200, 100, 50]
        while audio_features.size(0) > predefined_length:
            
            ctc_audio_features = self.audio_tower.layer_norm(audio_features)
            bsz_length = ctc_audio_features.size(0)
            
            # Split the LongSpeech Embeddings
            chunk_length = 2000
            iterate_runs = math.ceil(float(bsz_length) / chunk_length)
            candidate_iter_runs = []
            for i in range(iterate_runs):
                tmp_ctc_audio_features = self.ctc_decoder.projector(ctc_audio_features[chunk_length*i:chunk_length*(i+1), :]).to(device=selected_features.device)
                nonblank_logits = 1 - torch.softmax(self.ctc_decoder(tmp_ctc_audio_features).to(dtype=torch.float32, device=selected_features.device), dim=-1)[:, -1]
                candidate_iter_runs.append(nonblank_logits)
            nonblank_logits = torch.cat(candidate_iter_runs, dim=0)
            
            pre_feature = audio_features[:-1, :]
            post_feature = audio_features[1:, :]
            
            similarity = (pre_feature * post_feature).sum(dim=-1) / (torch.norm(pre_feature, dim=-1) * torch.norm(post_feature, dim=-1))
            if audio_features.size(0) / 2 > predefined_length:
                res_length = audio_features.size(0) - int(audio_features.size(0) / 2)
            else:
                res_length = audio_features.size(0) - predefined_length
                flag = True
            
            topk_indices = similarity.topk(res_length).indices
            topk_indices = topk_indices.sort()[0] + 1
            
            indices = torch.zeros(audio_features.size(0), dtype=torch.long, device=selected_features.device)
            
            indices[topk_indices] = 1
            indices = 1 - indices
            indices = indices.cumsum(dim=0) - 1
            
            if flag == True:
                final_features = torch.zeros(predefined_length, selected_features.size(-1)).to(device=selected_features.device)
                # add logits
                summ_logits = torch.zeros(predefined_length, device=selected_features.device)
            else:
                final_features = torch.zeros(int(audio_features.size(0) / 2), selected_features.size(-1)).to(device=selected_features.device)
                # add logits
                summ_logits = torch.zeros(int(audio_features.size(0) / 2), device=selected_features.device)
            #counts = torch.bincount(indices)
            final_features.index_add_(0, indices, audio_features*nonblank_logits.unsqueeze(1))
            # compute_logits
            summ_logits.index_add_(0, indices, nonblank_logits)
            final_features /= (summ_logits.unsqueeze(1) + 1e-7)
            
            audio_features = final_features
        
        return final_features.unsqueeze(0)

    # merge the audio embeddings when inference
    def merge_audios_reforward_onlysimilar_iterate_inte_blank_logit(
        self,
        audio_features,
        selected_features,
        predefined_length,
        audio_output_lengths
    ):
        audio_features = audio_features[:, :audio_output_lengths[0], :]
        #blank_logits = torch.softmax(self.ctc_decoder(audio_features).to(dtype=torch.float32, device=selected_features.device), dim=-1)[0, :, -1]
        selected_features = selected_features[:, :audio_output_lengths[0], :]
        #origin_length = selected_features.size(1)
        
        audio_features = selected_features[0]
        
        # judge interval
        candidate_pool =  [400, 200, 100, 50, 25, 12]
        candidate_start_index = len(candidate_pool) - 1
        for i in range(len(candidate_pool)):
            if audio_output_lengths[0] > candidate_pool[i]:
                candidate_start_index = i
                break
        
        while candidate_start_index < len(candidate_pool) and candidate_pool[candidate_start_index] >= predefined_length:
            # compute blank logits
            ctc_audio_features = self.audio_tower.layer_norm(audio_features)
            if self.config.ctc_decoder == "projector_llm_embed":
                ctc_audio_features = self.ctc_decoder.projector(ctc_audio_features).to(device=selected_features.device)
            else:
                ctc_audio_features = self.multi_modal_projector(ctc_audio_features).to(device=selected_features.device)
            nonblank_logits = 1 - torch.softmax(self.ctc_decoder(ctc_audio_features).to(dtype=torch.float32, device=selected_features.device), dim=-1)[:, -1]
            
            pre_feature = audio_features[:-1, :]
            post_feature = audio_features[1:, :]
            
            similarity = (pre_feature * post_feature).sum(dim=-1) / (torch.norm(pre_feature, dim=-1) * torch.norm(post_feature, dim=-1))
            res_length = audio_features.size(0) - candidate_pool[candidate_start_index]
            
            topk_indices = similarity.topk(res_length).indices
            topk_indices = topk_indices.sort()[0] + 1
            
            indices = torch.zeros(audio_features.size(0), dtype=torch.long, device=selected_features.device)
            
            indices[topk_indices] = 1
            indices = 1 - indices
            indices = indices.cumsum(dim=0) - 1
            
            final_features = torch.zeros(candidate_pool[candidate_start_index], selected_features.size(-1)).to(device=selected_features.device)
            # add logits
            summ_logits = torch.zeros(candidate_pool[candidate_start_index], device=selected_features.device)
            #counts = torch.bincount(indices)
            final_features.index_add_(0, indices, audio_features*nonblank_logits.unsqueeze(1))
            # compute_logits
            summ_logits.index_add_(0, indices, nonblank_logits)
            final_features /= (summ_logits.unsqueeze(1) + 1e-7)
            
            audio_features = final_features
            
            candidate_start_index += 1
        
        return final_features.unsqueeze(0)

    # dynamic audio supressing
    def merge_audios_reforward_onlysimilar_iterate_inte_blank_logit_training(
        self,
        selected_features,
        predefined_length,
        audio_output_lengths
    ):
        #audio_features = audio_features[:, :audio_output_lengths[0], :]
        #blank_logits = torch.softmax(self.ctc_decoder(audio_features).to(dtype=torch.float32, device=selected_features.device), dim=-1)[0, :, -1]
        #selected_features = selected_features[:, :audio_output_lengths[0], :]
        #origin_length = selected_features.size(1)
        
        audio_features = selected_features
        bsz, max_length, _ = selected_features.size()
        padding_mask = torch.arange(max_length).to(device=selected_features.device).unsqueeze(0).expand(bsz, max_length) < audio_output_lengths.unsqueeze(-1)
        
        candidate_pool =  [400, 200, 100, 50, 25, 12]
        candidate_start_index = len(candidate_pool) - 1
        for i in range(len(candidate_pool)):
            if max_length > candidate_pool[i]:
                candidate_start_index = i
                break
            
        while candidate_start_index < len(candidate_pool) and candidate_pool[candidate_start_index] >= predefined_length:
            # compute blank logits
            ctc_audio_features = self.audio_tower.layer_norm(audio_features)
            if self.config.ctc_decoder == "projector_llm_embed":
                ctc_audio_features = self.ctc_decoder.projector(ctc_audio_features).to(device=selected_features.device)
            else:
                ctc_audio_features = self.multi_modal_projector(ctc_audio_features).to(device=selected_features.device)
            nonblank_logits = 1 - torch.softmax(self.ctc_decoder(ctc_audio_features).to(dtype=torch.float32, device=selected_features.device), dim=-1)[:, :, -1]
            
            pre_feature = audio_features[:, :-1, :]
            post_feature = audio_features[:, 1:, :]
            
            similarity = (pre_feature * post_feature).sum(dim=-1) / (torch.norm(pre_feature, dim=-1) * torch.norm(post_feature, dim=-1))
            
            #add_mask for parallel operation
            similarity[~padding_mask[:, 1:]] = 1.0
            res_length = audio_features.size(1) - candidate_pool[candidate_start_index]
            
            topk_indices = similarity.topk(res_length, dim=-1).indices
            topk_indices = topk_indices.sort()[0] + 1
            
            indices = torch.zeros(audio_features.size(0), audio_features.size(1), dtype=torch.long, device=selected_features.device)
            
            indices.scatter_(-1, topk_indices, 1)
            indices = 1 - indices
            indices = indices.cumsum(dim=-1) - 1
            
            final_features = torch.zeros(bsz, candidate_pool[candidate_start_index], selected_features.size(-1)).to(device=selected_features.device)
            # add logits
            summ_logits = torch.zeros(bsz, candidate_pool[candidate_start_index]).to(device=selected_features.device)
            
            nonblank_logits = nonblank_logits.masked_fill(~padding_mask, 0.0).detach()
            tmp_audio_features = (audio_features*nonblank_logits.unsqueeze(-1)).contiguous().view(-1, final_features.size(-1))
            #counts = torch.bincount(indices)
            bsz, true_length, dimension = final_features.size()
            final_features = final_features.view(bsz * true_length, -1)
            summ_logits = summ_logits.view(-1)
            tmp_indices = indices + torch.arange(bsz).to(device=selected_features.device).unsqueeze(-1) * true_length
            tmp_indices = tmp_indices.view(-1).detach()
            
            final_features = final_features.index_add(0, tmp_indices, tmp_audio_features)
            # compute_logits final_features.index_add_(1, indices, audio_features*nonblank_logits.unsqueeze(-1))
            summ_logits = summ_logits.index_add(0, tmp_indices, nonblank_logits.view(-1))
            final_features /= (summ_logits.unsqueeze(-1) + 1e-7)
            
            audio_features = final_features.contiguous().view(bsz, true_length, dimension).to(dtype=audio_features.dtype)
            
            padding_mask = padding_mask[:, :candidate_pool[candidate_start_index]]
            
            candidate_start_index += 1
        return audio_features

    
    def forward_logits(
        self,
        selected_features
    ):
        audio_features = self.audio_tower.layer_norm(selected_features)
        if self.config.ctc_decoder == "projector_llm_embed":
            audio_features = self.ctc_decoder.projector(audio_features)
        else:
            audio_features = self.multi_modal_projector(audio_features)
        
        audio_logits = torch.softmax(self.ctc_decoder(audio_features).to(dtype=torch.float32), dim=-1)[:, -1]
        
        return audio_logits

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        asr_labels: Optional[torch.LongTensor] = None,
        predefined_length: Optional[int] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_loss: Optional[bool] = None,
        long_speech_processing: Optional[bool] = False,
    ) -> Union[Tuple, LongSpeechQwen2AudioCausalLMOutputWithPast]:
        
        if self.training and self.indicator == False and self.config.ctc_embed_num <= 0:
            self.indicator = True
            self.initial_partial_parameters()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        target_device = self.audio_tower.device

        if input_features is not None:
            input_features = input_features.to(target_device)
            feature_attention_mask = feature_attention_mask.to(target_device)
        # asr_training
        if asr_labels is not None:
            audio_features, audio_output_lengths = self.audio_features_forward(input_features, feature_attention_mask, predefined_length)
            ctc_loss = self.compute_ctc_loss(audio_features, asr_labels, audio_output_lengths)
            if input_ids is None:
                return LongSpeechQwen2AudioCausalLMOutputWithPast(
                    loss=ctc_loss,
                )
        # ASR decoding
        if input_ids is None and self.training == False:
            audio_features, audio_output_lengths = self.audio_features_forward(input_features, feature_attention_mask, predefined_length)
            hidden_inputs = self.ctc_decoder(audio_features).to(dtype=torch.float32)
            _, output_indices = torch.max(hidden_inputs, dim=-1)
            output_indices = output_indices[:, :audio_output_lengths[0]]
            final_output_ids = self.remove_blank_consecutive_tokens(output_indices, blank_ids=self.ctc_decoder.linear.weight.size(0)-1)
            
            return final_output_ids
        
        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)
            # 2. Merge text and audios
            if input_features is not None and input_ids.shape[1] != 1:
                audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                    feature_attention_mask.sum(-1), None
                )
                
                batch_size, _, max_mel_seq_len = input_features.shape
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                # Create a sequence tensor of shape (batch_size, max_seq_len)
                seq_range = (
                    torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
                )
                lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
                # Create mask
                padding_mask = seq_range >= lengths_expand

                audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                    batch_size, 1, max_seq_len, max_seq_len
                )
                audio_attention_mask = audio_attention_mask_.to(
                    dtype=self.audio_tower.conv1.weight.dtype, device=self.audio_tower.conv1.weight.device
                )
                audio_attention_mask[audio_attention_mask_] = float("-inf")
                # LongSpeech Inference; In this way, we split the long speech inputs into multi-batch inputs
                if long_speech_processing:
                    bsz = input_features.size(0)
                    outputs_list = []
                    outputs_features_list = []
                    
                    for i in range(bsz):
                        audio_outputs = self.audio_tower(input_features[i].unsqueeze(0), attention_mask=audio_attention_mask[i].unsqueeze(0), audio_output_lengths=audio_output_lengths[i].unsqueeze(0))
                        outputs_features_list.append(self.multi_modal_projector(audio_outputs.last_hidden_state))
                        outputs_list.append(audio_outputs.hidden_states)
                    audio_output_hidden_states = torch.cat(outputs_list, dim=0)
                    audio_features = torch.cat(outputs_features_list, dim=0)
                    
                    # flatten the tensor
                    audio_output_hidden_states = audio_output_hidden_states.contiguous().view(1, -1, audio_output_hidden_states.size(-1))
                    audio_features = audio_features.contiguous().view(1, -1, audio_features.size(-1))
                    
                    audio_output_lengths[0] = audio_output_lengths.sum()
                    audio_output_lengths = audio_output_lengths[:1]
                    if self.training == False and labels is None and predefined_length < audio_output_lengths[0]:
                        audio_features = self.merge_audios_reforward_onlysimilar_iterate_inte_blank_logit_longSpeech(audio_features, audio_output_hidden_states, predefined_length, audio_output_lengths)
                        #audio_features = audio_features.half()
                        audio_features = self.audio_tower.layer_norm(audio_features)
                        audio_features = self.multi_modal_projector(audio_features).to(device=audio_output_lengths.device)
                        audio_output_lengths[audio_output_lengths > predefined_length] = predefined_length
                else:
                    audio_outputs = self.audio_tower(input_features, attention_mask=audio_attention_mask, audio_output_lengths=audio_output_lengths)
                    
                    audio_features = self.audio_tower.layer_norm(audio_outputs.hidden_states)
                    audio_features = self.multi_modal_projector(audio_features).to(device=audio_output_lengths.device)
                    # For short-duration processing
                    if self.training == False and labels is None and predefined_length < audio_output_lengths[0]:
                        audio_features = self.merge_audios_reforward_onlysimilar_iterate_inte_blank_logit(audio_features, audio_outputs.hidden_states, predefined_length, audio_output_lengths)
                        audio_features = self.audio_tower.layer_norm(audio_features)
                        audio_features = self.multi_modal_projector(audio_features).to(device=audio_output_lengths.device)
                        audio_output_lengths[0] = predefined_length

                    # For dynamic compression sampling
                    if labels is not None:
                        audio_features = self.merge_audios_reforward_onlysimilar_iterate_inte_blank_logit_training(audio_outputs.hidden_states, predefined_length, audio_output_lengths)
                        audio_features = self.audio_tower.layer_norm(audio_features)
                        audio_features = self.multi_modal_projector(audio_features).to(device=audio_output_lengths.device)
                        audio_output_lengths[audio_output_lengths > predefined_length] = predefined_length
                
                inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
                    audio_features, audio_output_lengths, inputs_embeds, input_ids, attention_mask, labels
                )
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:] & (labels[..., 1:] != self.config.ignore_index)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_attention_mask = (labels[..., 1:] != self.config.ignore_index)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if asr_labels is not None:
            loss = ctc_loss + loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LongSpeechQwen2AudioCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
        )

    # def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
    #                                   inputs_embeds=None, **kwargs):
    #     speech = kwargs.pop("speech", None)
    #     speech_lengths = kwargs.pop("speech_lengths", None)
    #     inputs = super().prepare_inputs_for_generation(
    #         input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
    #     )
    #     if speech is not None:
    #         inputs['speech'] = speech
    #         inputs['speech_lengths'] = speech_lengths
    #     return inputs

# AutoConfig.register("long_speech_qwen2audio", LongSpeechQwen2AudioConfig)
# AutoModelForCausalLM.register(LongSpeechQwen2AudioConfig, LongSpeechQwen2AudioForCausalLM)
