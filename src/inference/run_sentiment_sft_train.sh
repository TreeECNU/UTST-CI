#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate readability_summ

VAL_FILE='../train/data/train_yelp_parallel_data_5.json'
# MODEL_PATH='../train/rl/trlx/checkpoint-diverse/best_checkpoint/hf_model'
MODEL_PATH='../train/mnt/hd3/checkpoints/transition'

# 第一组任务：使用 GPU 0 和 1
OUTPUT_DIR='outputs_sentiment_data/'
CUDA_VISIBLE_DEVICES=0 python -u run_sentiment_sft_train.py --model_name_or_path ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} --text_column input --summary_column output_text \
 --train_file ${VAL_FILE} \
 --validation_file ${VAL_FILE} \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 256 \
 --max_target_length 256 \
 --generation_max_length 256 \
 --num_beams 3 \
 --source_prefix " " \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 16 \
 --per_device_eval_batch_size 16\
 --predict_with_generate \
 --do_predict

P1=$!

wait $P1
conda deactivate