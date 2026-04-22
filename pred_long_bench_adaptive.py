import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaConfig, MistralConfig

from utils.adaptive_kv import (
    adaptive_experiment_tag,
    attach_adaptive_fields,
    build_adaptive_kv_config,
    resolve_model_max_length,
)
from utils.process_args_adaptive import process_args_adaptive

os.environ["WANDB_DISABLED"] = "true"


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


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model(model_args, training_args, dtype):
    lower_name = model_args.model_name_or_path.lower()
    model_name = model_args.model_name_or_path.split("/")[-1]
    if "llama" in lower_name or "longchat" in lower_name:
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer_kwargs = {
            "trust_remote_code": True,
        }
        if "llama-3" in lower_name:
            tokenizer_kwargs["use_fast"] = True
        else:
            tokenizer_kwargs["use_fast"] = False
            tokenizer_kwargs["tokenizer_type"] = "llama"
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            **tokenizer_kwargs,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if model_args.k_bits < 16 and model_args.v_bits < 16:
            from models.llama_kivi import LlamaForCausalLM_KIVI

            config.k_bits = model_args.k_bits
            config.v_bits = model_args.v_bits
            config.group_size = model_args.group_size
            config.residual_length = model_args.residual_length
            config.use_flash = bool(model_args.use_flash_attention_2)
            attach_adaptive_fields(config, model_args)
            model = LlamaForCausalLM_KIVI.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        else:
            from transformers import LlamaForCausalLM

            model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_flash_attention_2=bool(model_args.use_flash_attention_2),
            )
    elif "mistral" in lower_name:
        config = MistralConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=False,
            trust_remote_code=True,
        )
        if model_args.k_bits < 16 and model_args.v_bits < 16:
            from models.mistral_kivi import MistralForCausalLM_KIVI

            config.k_bits = model_args.k_bits
            config.v_bits = model_args.v_bits
            config.group_size = model_args.group_size
            config.residual_length = model_args.residual_length
            config.use_flash = bool(model_args.use_flash_attention_2)
            attach_adaptive_fields(config, model_args)
            model = MistralForCausalLM_KIVI.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        else:
            from transformers import MistralForCausalLM

            model = MistralForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_flash_attention_2=bool(model_args.use_flash_attention_2),
            )
    else:
        raise NotImplementedError(f"Unsupported model family: {model_args.model_name_or_path}")

    return model, tokenizer, model_name


def default_longbench_datasets(use_longbench_e):
    if use_longbench_e:
        return [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    return ["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "qmsum", "multi_news", "passage_retrieval_en"]


if __name__ == "__main__":
    seed_everything(42)
    model_args, data_args, training_args = process_args_adaptive()
    if model_args.adaptive_kv:
        build_adaptive_kv_config(model_args)

    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    model, tokenizer, model_name = load_model(model_args, training_args, dtype)
    model.eval()

    _, max_length = resolve_model_max_length(model_args.model_name_or_path, model2maxlen)
    if data_args.datasets:
        datasets = [dataset.strip() for dataset in data_args.datasets.split(",") if dataset.strip()]
    else:
        datasets = default_longbench_datasets(data_args.e)
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    pred_root = data_args.pred_root or ("pred_e" if data_args.e else "pred")
    os.makedirs(pred_root, exist_ok=True)
    exp_tag = adaptive_experiment_tag(model_args)
    exp_dir = os.path.join(pred_root, f"{model_name}_{max_length}_{exp_tag}")
    os.makedirs(exp_dir, exist_ok=True)

    for dataset in datasets:
        out_path = os.path.join(exp_dir, f"{dataset}.jsonl")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[skip] {dataset}: existing predictions found at {out_path}")
            continue

        print(f"[run] {dataset}")
        split_name = f"{dataset}_e" if data_args.e else dataset
        data = load_dataset("THUDM/LongBench", split_name, split="test", trust_remote_code=True)
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
        print(f"[done] {dataset}: wrote {len(preds)} predictions to {out_path}")
