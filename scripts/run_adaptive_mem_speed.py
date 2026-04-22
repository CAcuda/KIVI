import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig, MistralConfig

from utils.adaptive_kv import attach_adaptive_fields, build_adaptive_kv_config


def has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Run adaptive KV memory/speed benchmark.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cached_models")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--prompt_length", type=int, default=160)
    parser.add_argument("--output_length", type=int, default=338)
    parser.add_argument("--num_repeats", type=int, default=3)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--out_json", type=str, default="./outputs/adaptive_mem_speed.json")
    parser.add_argument("--k_bits", type=int, default=2)
    parser.add_argument("--v_bits", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--residual_length", type=int, default=128)
    parser.add_argument("--adaptive_kv", action="store_true")
    parser.add_argument("--adaptive_policy", type=str, default="static_distance")
    parser.add_argument("--adaptive_segment_lengths", type=str, default=None)
    parser.add_argument("--adaptive_k_bits", type=str, default=None)
    parser.add_argument("--adaptive_v_bits", type=str, default=None)
    parser.add_argument("--use_flash_attention_2", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--reuse_window", type=int, default=64)
    parser.add_argument("--reuse_topk", type=int, default=32)
    parser.add_argument("--hybrid_high_bits", type=int, default=None)
    parser.add_argument("--hybrid_low_bits", type=int, default=None)
    return parser.parse_args()


def build_model(args, dtype: torch.dtype):
    flash_available = has_flash_attn()
    flash_enabled = bool(args.use_flash_attention_2 and flash_available)
    lower = args.model_name_or_path.lower()
    if "llama" in lower or "longchat" in lower:
        from models.llama_kivi import LlamaForCausalLM_KIVI

        config = LlamaConfig.from_pretrained(args.model_name_or_path)
        config.k_bits = args.k_bits
        config.v_bits = args.v_bits
        config.group_size = args.group_size
        config.residual_length = args.residual_length
        config.use_flash = flash_enabled
        attach_adaptive_fields(config, args)
        model = LlamaForCausalLM_KIVI.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=False,
            trust_remote_code=True,
        )
    elif "mistral" in lower:
        from models.mistral_kivi import MistralForCausalLM_KIVI

        config = MistralConfig.from_pretrained(args.model_name_or_path)
        config.k_bits = args.k_bits
        config.v_bits = args.v_bits
        config.group_size = args.group_size
        config.residual_length = args.residual_length
        config.use_flash = flash_enabled
        attach_adaptive_fields(config, args)
        model = MistralForCausalLM_KIVI.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=False,
            trust_remote_code=True,
        )
    else:
        raise NotImplementedError(f"Unsupported model family for adaptive benchmark: {args.model_name_or_path}")
    return model, tokenizer, flash_enabled


def build_inputs(tokenizer, batch_size: int, prompt_length: int):
    context = []
    for _ in range(batch_size):
        text = "t," * (prompt_length // 2)
        context.append(text[:-1])
    return tokenizer(context, return_tensors="pt").to("cuda")


def run_generation_benchmark(model, inputs, num_repeats: int, output_length: int):
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(num_repeats):
            _ = model.generate(**inputs, max_new_tokens=output_length)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / num_repeats

    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return elapsed, peak_mem_gb


def main():
    args = parse_args()
    if args.adaptive_kv:
        build_adaptive_kv_config(args)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model, tokenizer, flash_enabled = build_model(args, dtype)
    model.cuda().eval()

    attempt_batch_size = args.batch_size
    oom_retries = []
    inputs = None
    input_ids = None
    elapsed = None
    peak_mem_gb = None

    while attempt_batch_size >= 1:
        try:
            inputs = build_inputs(tokenizer, attempt_batch_size, args.prompt_length)
            input_ids = inputs["input_ids"]
            elapsed, peak_mem_gb = run_generation_benchmark(
                model,
                inputs,
                args.num_repeats,
                args.output_length,
            )
            break
        except torch.OutOfMemoryError as exc:
            oom_retries.append(
                {
                    "batch_size": attempt_batch_size,
                    "error": str(exc),
                }
            )
            inputs = None
            input_ids = None
            torch.cuda.empty_cache()
            if attempt_batch_size == 1:
                raise
            attempt_batch_size = max(1, attempt_batch_size // 2)

    if input_ids is None or elapsed is None or peak_mem_gb is None:
        raise RuntimeError("Adaptive memory/speed benchmark failed before metrics were collected.")

    result = {
        "model_name_or_path": args.model_name_or_path,
        "mode": "adaptive_kv" if args.adaptive_kv else "fixed_kivi",
        "adaptive_policy": args.adaptive_policy if args.adaptive_kv else None,
        "adaptive_segment_lengths": args.adaptive_segment_lengths,
        "adaptive_k_bits": args.adaptive_k_bits,
        "adaptive_v_bits": args.adaptive_v_bits,
        "flash_attention_2_enabled": flash_enabled,
        "dtype": args.dtype,
        "batch_size": attempt_batch_size,
        "requested_batch_size": args.batch_size,
        "prompt_length": int(input_ids.shape[1]),
        "output_length": args.output_length,
        "num_repeats": args.num_repeats,
        "avg_time_ms": elapsed * 1000.0,
        "peak_mem_gb": peak_mem_gb,
        "k_bits": args.k_bits,
        "v_bits": args.v_bits,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
        "oom_retries": oom_retries,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
