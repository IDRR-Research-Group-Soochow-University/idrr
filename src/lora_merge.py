#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 LoRA 微调后的权重合并回基础模型，输出一个可直接推理的标准 HuggingFace 模型。
"""

import argparse
import json
import os
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def infer_base_model_name(lora_path: str) -> Optional[str]:
    """
    尝试从 LoRA 目录中的 peft 配置文件推断 base_model_name_or_path。
    """
    candidates = [
        os.path.join(lora_path, "adapter_config.json"),
        os.path.join(lora_path, "config.json"),
        os.path.join(lora_path, "peft_config.json"),
    ]
    for f in candidates:
        if os.path.isfile(f):
            try:
                with open(f, "r", encoding="utf-8") as rf:
                    data = json.load(rf)
                for k in ["base_model_name_or_path", "base_model_name", "model_name"]:
                    if k in data:
                        return data[k]
            except Exception:
                pass
    # 另一种方式：使用 PeftConfig 直接解析
    try:
        cfg = PeftConfig.from_pretrained(lora_path)
        return getattr(cfg, "base_model_name_or_path", None)
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--base_model", type=str, default=None,
                        help="基础模型名称或路径(若不提供则尝试从 LoRA 路径推断)。")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="LoRA 微调后的 checkpoint 路径，如: /data/.../checkpoint-790")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="合并后模型保存目录。")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="加载及合并使用的精度。")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="传递给 from_pretrained 的 device_map (auto / cpu / 单GPU号等)。")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="部分模型(如 Qwen)可能需要。")
    parser.add_argument("--tokenizer_base", type=str, default=None,
                        help="若需覆盖 tokenizer 的加载来源。")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="合并后直接 push 到 Hub，需要提前登录。")
    parser.add_argument("--hub_repo", type=str, default=None,
                        help="push 到的 repo 名(与 --push_to_hub 配合)。")
    return parser.parse_args()


def str_to_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"不支持的 dtype: {dtype_str}")


def main():
    args = parse_args()

    if not os.path.isdir(args.lora_path):
        print(f"[错误] LoRA 路径不存在: {args.lora_path}", file=sys.stderr)
        sys.exit(1)

    base_model = args.base_model
    if base_model is None:
        base_model = infer_base_model_name(args.lora_path)
        if base_model is None:
            print("[错误] 未能自动推断 base_model，请使用 --base_model 指定。", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"[信息] 推断基础模型: {base_model}")

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = str_to_dtype(args.dtype)

    print("[信息] 加载基础模型...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    print("[信息] 加载 LoRA 适配器并附加到基础模型...")
    model = PeftModel.from_pretrained(base, args.lora_path)

    print("[信息] 合并 LoRA 权重到主权重 (merge_and_unload)...")
    merged = model.merge_and_unload()  # 返回普通模型 (nn.Module)

    print("[信息] 保存合并后的模型权重与配置...")
    merged.save_pretrained(args.output_dir)

    # Tokenizer
    tok_source = args.tokenizer_base or base_model
    print(f"[信息] 加载/保存 tokenizer 来源: {tok_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_source,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(args.output_dir)

    # 可选推送到 Hub
    if args.push_to_hub:
        if not args.hub_repo:
            print("[警告] 未提供 --hub_repo，跳过 push。")
        else:
            print("[信息] 正在推送到 Hub...")
            merged.push_to_hub(args.hub_repo)
            tokenizer.push_to_hub(args.hub_repo)

    print("[完成] 合并结束。输出目录:", args.output_dir)
    print("测试推理示例(可选):")
    print(f"python - <<'EOF'\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\nm='{args.output_dir}'\nmodel=AutoModelForCausalLM.from_pretrained(m, torch_dtype={dtype})\ntok=AutoTokenizer.from_pretrained(m)\nipt=tok('你好', return_tensors='pt')\nout=model.generate(**ipt, max_new_tokens=32)\nprint(tok.decode(out[0], skip_special_tokens=True))\nEOF")


if __name__ == "__main__":
    main()