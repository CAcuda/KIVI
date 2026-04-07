import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig, MistralConfig


def has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Run full-precision baseline memory/speed benchmark.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cached_models")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--prompt_length", type=int, default=160)
    parser.add_argument("--output_length", type=int, default=338)
    parser.add_argument("--num_repeats", type=int, default=3)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--out_json", type=str, default="./outputs/baseline_mem_speed.json")
    return parser.parse_args()


def build_model(model_name_or_path: str, cache_dir: str, dtype: torch.dtype):
    flash_ok = has_flash_attn()
    lower = model_name_or_path.lower()
    if "llama" in lower or "longchat" in lower:
        from transformers import LlamaForCausalLM

        config = LlamaConfig.from_pretrained(model_name_or_path)
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_flash_attention_2=flash_ok,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            trust_remote_code=True,
        )
    elif "mistral" in lower:
        from transformers import MistralForCausalLM

        config = MistralConfig.from_pretrained(model_name_or_path)
        model = MistralForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_flash_attention_2=flash_ok,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            trust_remote_code=True,
        )
    else:
        raise NotImplementedError(f"Unsupported model family for baseline benchmark: {model_name_or_path}")
    return model, tokenizer, flash_ok


def main():
    args = parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    model, tokenizer, flash_ok = build_model(args.model_name_or_path, args.cache_dir, dtype)
    model.cuda().eval()

    context = []
    for _ in range(args.batch_size):
        text = "t," * (args.prompt_length // 2)
        context.append(text[:-1])

    inputs = tokenizer(context, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.num_repeats):
            _ = model.generate(**inputs, max_new_tokens=args.output_length)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / args.num_repeats

    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    result = {
        "model_name_or_path": args.model_name_or_path,
        "mode": "baseline_full_precision",
        "flash_attention_2_enabled": flash_ok,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "prompt_length": int(input_ids.shape[1]),
        "output_length": args.output_length,
        "num_repeats": args.num_repeats,
        "avg_time_ms": elapsed * 1000.0,
        "peak_mem_gb": peak_mem_gb,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
