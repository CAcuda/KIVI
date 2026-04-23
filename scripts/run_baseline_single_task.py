import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaConfig, MistralConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_long_bench import scorer

os.environ["WANDB_DISABLED"] = "true"


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline full-precision LongBench on a single task.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cached_models")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument(
        "--use_flash_attention_2",
        type=lambda x: str(x).lower() == "true",
        default=False,
    )
    parser.add_argument("--pred_root", type=str, default="./outputs/baseline_single/pred")
    return parser.parse_args()


def build_chat(tokenizer, prompt, model_name):
    if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "longchat" in model_name.lower():
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def build_model(model_name_or_path: str, cache_dir: str, dtype: torch.dtype, use_flash_attention_2: bool):
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
            use_flash_attention_2=use_flash_attention_2,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            trust_remote_code=True,
            tokenizer_type="llama",
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
            use_flash_attention_2=use_flash_attention_2,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            trust_remote_code=True,
        )
    else:
        raise NotImplementedError(f"Unsupported model family: {model_name_or_path}")
    return model, tokenizer


def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                tokenized_prompt[-half:], skip_special_tokens=True
            )
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        model_input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = model_input.input_ids.shape[-1]
        if dataset == "samsum":
            output = model.generate(
                **model_input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **model_input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return preds


def main():
    seed_everything(42)
    args = parse_args()

    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    model_name = args.model_name_or_path.split("/")[-1]
    model_tag = f"{model_name}_{model2maxlen[model_name]}_16bits_baseline"
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = build_model(args.model_name_or_path, args.cache_dir, dtype, args.use_flash_attention_2)
    model.eval()

    data = load_dataset("THUDM/LongBench", args.dataset, split="test")
    prompt_format = dataset2prompt[args.dataset]
    max_gen = dataset2maxlen[args.dataset]
    max_length = model2maxlen[model_name]

    pred_dir = Path(args.pred_root) / model_tag
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / f"{args.dataset}.jsonl"

    preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, args.dataset, device, model_name)
    with out_path.open("w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")

    predictions = [item["pred"] for item in preds]
    answers = [item["answers"] for item in preds]
    all_classes = preds[0]["all_classes"] if preds else None
    score = scorer(args.dataset, predictions, answers, all_classes)

    result = {
        "model_name_or_path": args.model_name_or_path,
        "model_tag": model_tag,
        "dataset": args.dataset,
        "mode": "baseline_full_precision",
        "flash_attention_2_enabled": bool(args.use_flash_attention_2),
        "dtype": args.dtype,
        "num_predictions": len(preds),
        "score": score,
        "prediction_path": str(out_path),
    }
    result_path = pred_dir / f"{args.dataset}.result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
