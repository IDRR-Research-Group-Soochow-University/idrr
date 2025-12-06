#!/bin/bash
set -euo pipefail

# 配置参数
export CUDA_VISIBLE_DEVICES=4

# 数据集
EXPERIMENT_NAME="rl_cold_start"
DEV_DATASET="pdtb2_dev_${EXPERIMENT_NAME}"
TEST_DATASET="pdtb2_test_${EXPERIMENT_NAME}"
EXPERIMENT_PATH="expt/$EXPERIMENT_NAME/qwen3-0.6B/epo1"
OUTPUT_DIR="./results/${EXPERIMENT_PATH}"


# 评估参数
EVAL_BATCH_SIZE=1

llamafactory-cli train \
    --stage sft \
    --template qwen3 \
    --fp16 \
    --do_predict \
    --overwrite_cache \
    --overwrite_output_dir \
    --predict_with_generate \
    --model_name_or_path "$EXPERIMENT_PATH" \
    --eval_dataset "${TEST_DATASET}" \
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
    --output_dir "$OUTPUT_DIR" \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16