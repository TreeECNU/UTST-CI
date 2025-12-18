#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate readability_summ

export TOKENIZERS_PARALLELISM=false  # 添加这一行

VAL_FILE='../train/data/val_summary_prompt_parallel.json'
MODEL_PATH='../train/rl/trlx/checkpoint-diverse-readability-2-phases/checkpoint_5120/hf_model'

# 任务 1：Flesch Kincaid 得分 90
OUTPUT_DIR='outputs_readability_ppo_5120/1/'
CUDA_VISIBLE_DEVICES=0 python -u run_readability_ppo.py \
 --ppo_checkpoint ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} \
 --text_column input_text \
 --summary_column output_text \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 128 \
 --source_prefix "rewrite the following text for elementary school students:\n\n" \
 --num_beams 4 \
 --overwrite_cache &

P1=$!

# 任务 2：Flesch Kincaid 得分 70
OUTPUT_DIR='outputs_readability_ppo_5120/2/'
CUDA_VISIBLE_DEVICES=1 python -u run_readability_ppo.py \
 --ppo_checkpoint ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} \
 --text_column input_text \
 --summary_column output_text \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 128 \
 --source_prefix "rewrite the following text for middle school students:\n\n" \
 --num_beams 4 \
 --overwrite_cache &

P2=$!

wait $P1 $P2

# 任务 3：Flesch Kincaid 得分 50
OUTPUT_DIR='outputs_readability_ppo_5120/3/'
CUDA_VISIBLE_DEVICES=0 python -u run_readability_ppo.py \
 --ppo_checkpoint ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} \
 --text_column input_text \
 --summary_column output_text \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 128 \
 --source_prefix "rewrite the following text for high school students:\n\n" \
 --num_beams 4 \
 --overwrite_cache &

P3=$!

# 任务 4：Flesch Kincaid 得分 20
OUTPUT_DIR='outputs_readability_ppo_5120/4/'
CUDA_VISIBLE_DEVICES=1 python -u run_readability_ppo.py \
 --ppo_checkpoint ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} \
 --text_column input_text \
 --summary_column output_text \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 128 \
 --source_prefix "rewrite the following text for college students:\n\n" \
 --num_beams 4 \
 --overwrite_cache &

P4=$!

wait $P1 $P2 $P3 $P4

conda deactivate