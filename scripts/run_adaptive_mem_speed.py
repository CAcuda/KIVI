import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig, MistralConfig

from utils.adaptive_kv import (
    attach_adaptive_fields,
    build_adaptive_kv_config,
    bucket_token_counts,
    effective_kv_bits,
    parse_int_list,
)


def has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def gpu_supports_flash_attn() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Memory/speed benchmark for FP16, fixed-bit KIVI and adaptive (SP-KV) KV caches."
    )
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
    parser.add_argument(
        "--full_precision",
        action="store_true",
        help="Benchmark the unmodified Hugging Face model with an FP16 KV cache.",
    )
    parser.add_argument(
        "--warmup_tokens",
        type=int,
        default=8,
        help="Untimed warmup generation length (Triton JIT, CUDA init). 0 disables warmup.",
    )
    parser.add_argument("--adaptive_kv", action="store_true")
    parser.add_argument("--adaptive_policy", type=str, default="static_distance")
    parser.add_argument("--adaptive_segment_lengths", type=str, default=None)
    parser.add_argument("--adaptive_k_bits", type=str, default=None)
    parser.add_argument("--adaptive_v_bits", type=str, default=None)
    parser.add_argument("--use_flash_attention_2", type=lambda x: str(x).lower() == "true", default=False)
    args = parser.parse_args()
    if args.full_precision and args.adaptive_kv:
        parser.error("--full_precision and --adaptive_kv are mutually exclusive.")
    return args


def build_full_precision_model(args, dtype: torch.dtype, flash_enabled: bool):
    lower = args.model_name_or_path.lower()
    if "llama" in lower or "longchat" in lower:
        from transformers import LlamaForCausalLM as model_cls
    elif "mistral" in lower:
        from transformers import MistralForCausalLM as model_cls
    else:
        raise NotImplementedError(f"Unsupported model family for FP16 benchmark: {args.model_name_or_path}")
    # Eager attention matches the non-flash KIVI attention path, so timings are comparable.
    model = model_cls.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="flash_attention_2" if flash_enabled else "eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=True)
    return model, tokenizer


def build_model(args, dtype: torch.dtype):
    flash_enabled = bool(args.use_flash_attention_2 and has_flash_attn() and gpu_supports_flash_attn())
    if args.full_precision:
        model, tokenizer = build_full_precision_model(args, dtype, flash_enabled)
        return model, tokenizer, flash_enabled
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


def run_generation_benchmark(model, inputs, num_repeats: int, output_length: int, warmup_tokens: int):
    with torch.no_grad():
        if warmup_tokens > 0:
            _ = model.generate(**inputs, max_new_tokens=warmup_tokens, min_new_tokens=warmup_tokens)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        generated = None
        for _ in range(num_repeats):
            # min_new_tokens forces the full length so every config decodes the same number of tokens.
            generated = model.generate(**inputs, max_new_tokens=output_length, min_new_tokens=output_length)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / num_repeats

    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    new_tokens = int(generated.shape[1] - inputs["input_ids"].shape[1])
    return elapsed, peak_mem_gb, new_tokens


def main():
    args = parse_args()
    if args.adaptive_kv:
        build_adaptive_kv_config(args)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model, tokenizer, flash_enabled = build_model(args, dtype)
    model.cuda().eval()
    torch.cuda.synchronize()
    weights_mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)

    attempt_batch_size = args.batch_size
    oom_retries = []
    inputs = None
    input_ids = None
    elapsed = None
    peak_mem_gb = None
    generated_tokens = None

    while attempt_batch_size >= 1:
        try:
            inputs = build_inputs(tokenizer, attempt_batch_size, args.prompt_length)
            input_ids = inputs["input_ids"]
            elapsed, peak_mem_gb, generated_tokens = run_generation_benchmark(
                model,
                inputs,
                args.num_repeats,
                args.output_length,
                args.warmup_tokens,
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

    prompt_tokens = int(input_ids.shape[1])
    max_kv_len = prompt_tokens + generated_tokens
    if args.full_precision:
        mode = "fp16"
    elif args.adaptive_kv:
        mode = "adaptive_kv"
    else:
        mode = "fixed_kivi"

    result = {
        "model_name_or_path": args.model_name_or_path,
        "mode": mode,
        "gpu": torch.cuda.get_device_name(),
        "flash_attention_2_enabled": flash_enabled,
        "dtype": args.dtype,
        "batch_size": attempt_batch_size,
        "requested_batch_size": args.batch_size,
        "prompt_length": prompt_tokens,
        "output_length": args.output_length,
        "generated_tokens": generated_tokens,
        "max_kv_len": max_kv_len,
        "num_repeats": args.num_repeats,
        "warmup_tokens": args.warmup_tokens,
        "avg_time_ms": elapsed * 1000.0,
        "decode_tokens_per_s": attempt_batch_size * generated_tokens / elapsed,
        "peak_mem_gb": peak_mem_gb,
        "weights_mem_gb": weights_mem_gb,
        "peak_mem_minus_weights_gb": peak_mem_gb - weights_mem_gb,
        "oom_retries": oom_retries,
    }
    if mode == "fixed_kivi":
        result.update(
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            group_size=args.group_size,
            residual_length=args.residual_length,
        )
    if mode == "adaptive_kv":
        segments = parse_int_list(args.adaptive_segment_lengths)
        k_bits = parse_int_list(args.adaptive_k_bits)
        v_bits = parse_int_list(args.adaptive_v_bits)
        counts = bucket_token_counts(max_kv_len, segments, args.residual_length)
        result.update(
            adaptive_policy=args.adaptive_policy,
            adaptive_segment_lengths=args.adaptive_segment_lengths,
            adaptive_k_bits=args.adaptive_k_bits,
            adaptive_v_bits=args.adaptive_v_bits,
            group_size=args.group_size,
            residual_length=args.residual_length,
            bucket_token_counts_at_max_len=counts,
            effective_k_bits_at_max_len=effective_kv_bits(max_kv_len, segments, k_bits, args.residual_length),
            effective_v_bits_at_max_len=effective_kv_bits(max_kv_len, segments, v_bits, args.residual_length),
        )
        unused = [bits for n, bits in zip(counts["segments"], k_bits) if n == 0]
        if unused:
            result["warning"] = (
                f"max_kv_len={max_kv_len} never reaches the {unused}-bit segment(s); this run does not "
                "measure the low-precision path. Use a longer --prompt_length."
            )
            print(f"[warning] {result['warning']}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
