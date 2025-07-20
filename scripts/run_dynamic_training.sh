#!/bin/bash

export PYTHONPATH=/data/guoshoutao/LongSpeechLLMs/FastLongSpeech
EXP=exp_dir
TRAIN_DIR=train_dir
BASE_MODEL=base_model
TRAIN_CONFIG=train_path
VALID_CONFIG=valid_path


deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29602 FastLongSpeech/training_file/train_llms.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --ctc_decoder_lr 2e-5 \
    --deepspeed FastLongSpeech/configuration/zero2.json \
    --model_name_or_path $BASE_MODEL \
    --data_path $TRAIN_CONFIG \
    --validate_path $VALID_CONFIG \
    --ctc_decoder 'projector_llm_embed' \
    --ctc_training False \
    --ctc_embed_num 10000 \
    --freeze_backbone False \
    --freeze_speechLLMs True \
    --group_by_modality_length True \
    --fp16 True \
    --output_dir FastLongSpeech/checkpoints/$EXP \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --metric_for_best_model 'loss' \
    --greater_is_better False \
    --do_eval True \
    --logging_steps 20 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard 2>&1 | tee FastLongSpeech/logs/$EXP.txt
