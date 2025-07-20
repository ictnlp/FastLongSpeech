export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=the_path_of_the_program

MODEL=model_dir
AUDIO_DIR=audio_dir

python inference_file/asr_self_policy.py --audio_air ${AUDIO_DIR} --model_name ${MODEL}
