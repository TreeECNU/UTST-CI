#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate readability_summ

VAL_FILE='../train/data/val_summary_prompt_parallel.json'
# MODEL_PATH='../train/rl/trlx/checkpoint-diverse/best_checkpoint/hf_model'
MODEL_PATH='../train/mnt/hd3/checkpoints/summary'

# 第一组任务：使用 GPU 0 和 1
OUTPUT_DIR='outputs_readability_sft/1/'
CUDA_VISIBLE_DEVICES=0 python -u run_readability_sft.py --model_name_or_path ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} --text_column input_text --summary_column output_text \
 --train_file ${VAL_FILE} \
 --validation_file ${VAL_FILE} \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 256 \
 --max_target_length 256 \
 --generation_max_length 256 \
 --num_beams 3 \
 --source_prefix "Rewrite the following text for primary school students:\n\n" \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 16 \
 --predict_with_generate \
 --do_predict &

P1=$!

OUTPUT_DIR='outputs_readability_sft/2/'
CUDA_VISIBLE_DEVICES=1 python -u run_readability_sft.py --model_name_or_path ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} --text_column input_text --summary_column output_text \
 --train_file ${VAL_FILE} \
 --validation_file ${VAL_FILE} \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 256 \
 --max_target_length 256 \
 --generation_max_length 256 \
 --num_beams 3 \
 --source_prefix "Rewrite the following text for middle school students:\n\n" \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 16 \
 --predict_with_generate \
 --do_predict &

P2=$!

wait $P1 $P2

# 第二组任务：复用 GPU 0 和 1
OUTPUT_DIR='outputs_readability_sft/3/'
CUDA_VISIBLE_DEVICES=0 python -u run_readability_sft.py --model_name_or_path ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} --text_column input_text --summary_column output_text \
 --train_file ${VAL_FILE} \
 --validation_file ${VAL_FILE} \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 256 \
 --max_target_length 256 \
 --generation_max_length 256 \
 --num_beams 3 \
 --source_prefix "Rewrite the following text for high school students\n\n" \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 16 \
 --predict_with_generate \
 --do_predict &

P3=$!

OUTPUT_DIR='outputs_readability_sft/4/'
CUDA_VISIBLE_DEVICES=1 python -u run_readability_sft.py --model_name_or_path ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} --text_column input_text --summary_column output_text \
 --train_file ${VAL_FILE} \
 --validation_file ${VAL_FILE} \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 256 \
 --max_target_length 256 \
 --generation_max_length 256 \
 --num_beams 3 \
 --source_prefix "Rewrite the following text for college students\n\n" \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 16 \
 --predict_with_generate \
 --do_predict &

P4=$!

wait $P3 $P4

wait $P5

conda deactivate