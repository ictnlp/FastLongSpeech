export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=the_path_of_the_program

MODEL=model_dir
LORA=lora_dir
LEN=400
AUDIO_CONFIG=audio_config
AUDIO_ROOT=audio_root
TARGET=target

python inference_file/chat_infer_long_speech.py --predefined_length $LEN --model_name $MODEL --lora_model_path $LORA --audio_config $AUDIO_CONFIG --audio_path_root $AUDIO_ROOT --target_path $TARGET
