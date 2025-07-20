# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
import sys

from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state
import os
import pathlib
import logging

import torch
import transformers
import sys
# from omni_speech.train.omni_speech_trainer import OmniSpeechTrainer
# from omni_speech import conversation as conversation_lib
# from omni_speech.model import *
# from omni_speech.datasets.speech_dataset import make_supervised_data_module
# from omni_speech.arguments import ModelArguments, DataArguments, TrainingArguments
# from omni_speech.utils import (
#     safe_save_model_for_hf_trainer,
#     find_all_linear_names,
#     get_peft_state_maybe_zero_3,
#     get_peft_state_non_lora_maybe_zero_3,
# )
sys.path.append('/data/guoshoutao/LongSpeechLLMs/CTCLLMs')
from CTCLLMs.models.qwen2audio_withCTC import LongSpeechQwen2AudioForCausalLM
from transformers import AutoProcessor
from CTCLLMs.arguments import ModelArguments, DataArguments, TrainingArguments
from CTCLLMs.utils import find_all_linear_names, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
from CTCLLMs.dataset.speech_dataset import make_supervised_ctc_decoder_data_module
from CTCLLMs.training_file.trainer_ctc import LongSpeechTrainer
import sentencepiece as spm
import pdb

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def train(attn_implementation=None):
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["ctc_decoder", "multi_modal_projector", "audio_tower"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    
    if model_args.version == "qwen2_audio":
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
        model = LongSpeechQwen2AudioForCausalLM.from_pretrained(
            model_args.model_name_or_path, 
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
            **bnb_model_from_pretrained_args
        )
    #model.config.use_cache = False model.language_model.model.embed_tokens model.language_model.model.layers[0].self_attn.q_proj.weight
    if model_args.freeze_backbone or training_args.freeze_speechLLMs:
        model.language_model.requires_grad_(False)
    if training_args.freeze_speechLLMs:
        model.multi_modal_projector.requires_grad_(False)
        model.audio_tower.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    if model_args.ctc_decoder is not None:
        model.initialize_ctc_decoder(model_args)
        model.config.ctc_training = data_args.ctc_training
        model.config.ctc_decoder_lr = training_args.ctc_decoder_lr
        ctc_decoder_tokenizer = spm.SentencePieceProcessor(model_file='../spm.model')
    # if data_args.has_tgt_units:
    #     model.initialize_speech_generator(model_args)
    #     if model_args.tune_speech_generator_only:
    #         model.requires_grad_(False)
    #         for p in model.speech_generator.parameters():
    #             p.requires_grad = True
    logging.info("Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name)
    rank0_print(f"In total: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000:.2f}M trainable parameters.")

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    data_args.padding_idx = model.config.ignore_index
    if model_args.ctc_decoder is not None:
        data_module = make_supervised_ctc_decoder_data_module(processor=processor, ctc_decoder_processor=ctc_decoder_tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_ctc_decoder_data_module(processor=processor, ctc_decoder_processor=ctc_decoder_tokenizer, data_args=data_args)
        
    trainer = LongSpeechTrainer(model=model,
                    tokenizer=processor,
                    args=training_args,
                    **data_module)
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    import warnings
    print(sys.path)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.utils\.checkpoint")
    train()
