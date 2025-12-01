export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/offline_infer.py \
    --data-format alpaca \
    --data-path /data/whsun/idrr/data/arg2def/pdtb2/aplaca/test.json \
    --ckpt /data/whsun/pretrained_models/Qwen/Qwen3-0.6B \
    --out /data/whsun/idrr/results/arg2def/pdtb2/llama3/epo5/merged.vllm.pred.jsonl \
    --gpu_memory_utilization 0.9
    # --ckpt /data/whsun/idrr/expt/arg2def/pdtb2/llama3/epo5/merged \