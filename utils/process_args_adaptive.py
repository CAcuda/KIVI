# coding=utf-8

import os
from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Output model local path, do not set manually"}
    )
    k_bits: Optional[int] = field(
        default=2,
        metadata={"help": "KV cache quantization bits."},
    )
    v_bits: Optional[int] = field(
        default=2,
        metadata={"help": "KV cache quantization bits."},
    )
    k_quant_dim: Optional[str] = field(
        default="token",
        metadata={"help": "KV cache key quantization dimension."},
    )
    v_quant_dim: Optional[str] = field(
        default="token",
        metadata={"help": "KV cache value quantization dimension."},
    )
    group_size: Optional[int] = field(
        default=128,
        metadata={"help": "KV cache quantization group size."},
    )
    residual_length: Optional[int] = field(
        default=128,
        metadata={"help": "KV cache residual length."},
    )
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )
    adaptive_kv: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable adaptive precision across sequence length."},
    )
    adaptive_policy: Optional[str] = field(
        default="static_distance",
        metadata={"help": "Adaptive policy: static_distance, reuse_aware, hybrid."},
    )
    adaptive_segment_lengths: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated segment lengths along the sequence axis."},
    )
    adaptive_k_bits: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated key bits for each segment."},
    )
    adaptive_v_bits: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated value bits for each segment."},
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to enable FlashAttention-2 / flash KV path."},
    )
    reuse_window: Optional[int] = field(
        default=64,
        metadata={"help": "Window size for reuse-aware token importance updates."},
    )
    reuse_topk: Optional[int] = field(
        default=32,
        metadata={"help": "How many high-reuse tokens or groups to preserve at higher precision."},
    )
    hybrid_high_bits: Optional[int] = field(
        default=None,
        metadata={"help": "Optional promoted bit width in hybrid mode."},
    )
    hybrid_low_bits: Optional[int] = field(
        default=None,
        metadata={"help": "Optional demoted bit width in hybrid mode."},
    )


@dataclass
class DataArguments:
    dataset: Optional[str] = field(
        default="c4",
        metadata={"help": "The dataset used for fine-tuning the model."},
    )
    eval_tasks: Optional[str] = field(
        default="wikitext",
        metadata={"help": "The dataset used for evaluation."},
    )
    tasks: Optional[str] = field(
        default="wikitext",
        metadata={"help": "The dataset used for evaluation."},
    )
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "The batch size."},
    )
    num_fewshot: Optional[int] = field(
        default=0,
        metadata={"help": "The number of fewshot examples."},
    )
    output_path: Optional[str] = field(
        default="./outputs",
        metadata={"help": "The output path."},
    )
    pred_root: Optional[str] = field(
        default=None,
        metadata={"help": "Optional prediction root directory for LongBench outputs."},
    )
    datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Optional comma-separated LongBench datasets to run."},
    )
    e: Optional[bool] = field(
        default=False,
        metadata={"help": "Evaluate on LongBench-E."},
    )
    use_our_imp: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use our KV cache quantization implementation."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="./outputs")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    num_train_epochs: Optional[int] = field(default=1)
    n_train_samples: Optional[int] = field(default=None)
    n_eval_samples: Optional[int] = field(default=None)
    qat: Optional[bool] = field(default=False)
    exp_name: Optional[str] = field(default="test")


def process_args_adaptive():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)

    model_args.output_model_local_path = os.path.join(
        training_args.output_dir, "models", str(model_args.output_model_filename)
    )

    return model_args, data_args, training_args
