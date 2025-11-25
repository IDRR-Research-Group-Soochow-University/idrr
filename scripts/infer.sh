#!/bin/bash

# 配置参数
export CUDA_VISIBLE_DEVICES=0,1
DATASET_DIR="./data"
WORK="rl_cold_start"
WORK_PATH="${WORK}"
DEV_DATASET="pdtb2_dev_${WORK}"
TEST_DATASET="pdtb2_test_${WORK}"
# MODEL_PATH="/data/whsun/pretrained_models/Meta-Llama-3.1-8B-Instruct"
MODEL_PATH="/data/whsun/pretrained_models/Qwen/Qwen3-0.6B"
CHECKPOINTS_DIR="./expt/${WORK_PATH}"
OUTPUT_ROOT="./results/${WORK_PATH}"
CKPT_PATH="qwen3-0.6B/epo1/checkpoint-790"
PER_DEVICE_TRAIN_BATCH_SIZE=1

llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path "${MODEL_PATH}" \
    --adapter_name_or_path "${CHECKPOINTS_DIR}/${CKPT_PATH}" \
    --eval_dataset "${TEST_DATASET}" \
    --dataset_dir "${DATASET_DIR}" \
    --template qwen3 \
    --finetuning_type lora \
    --output_dir "${OUTPUT_ROOT}/${CKPT_PATH}" \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --predict_with_generate

# python /data/whsun/LLaMA-Factory/scripts/vllm_infer.py \
#     --model_name_or_path /data/whsun/pretrained_models/Qwen/Qwen3-0.6B \
#     --adapter_name_or_path expt/rl_cold_start/qwen3-0.6B/epo1/checkpoint-790 \
#     --template qwen3 \
#     --dataset pdtb2_test_rl_cold_start \
#     --enable_thinking 
#     # --vllm_config {"enforce_eager":True} \
#     # --dataset_dir ./data \