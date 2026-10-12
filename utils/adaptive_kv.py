from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

SUPPORTED_POLICIES = {"static_distance"}


def parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def normalize_segment_lengths(
    segment_lengths: Optional[Iterable[int]],
    max_length: Optional[int] = None,
) -> Optional[List[int]]:
    if segment_lengths is None:
        return None
    lengths = [int(x) for x in segment_lengths]
    if any(x <= 0 for x in lengths):
        raise ValueError("All adaptive segment lengths must be positive.")
    if max_length is not None and sum(lengths) > max_length:
        raise ValueError("Adaptive segment lengths exceed the configured maximum length.")
    return lengths


@dataclass
class AdaptiveKVConfig:
    enabled: bool
    policy: str = "static_distance"
    segment_lengths: Optional[List[int]] = None
    k_segment_bits: Optional[List[int]] = None
    v_segment_bits: Optional[List[int]] = None

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.policy not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Unsupported adaptive KV policy: {self.policy}. Implemented: {sorted(SUPPORTED_POLICIES)}."
            )
        if not self.segment_lengths:
            raise ValueError("Static distance policy requires --adaptive_segment_lengths.")
        if not self.k_segment_bits or not self.v_segment_bits:
            raise ValueError("Static distance policy requires per-segment K/V bits.")
        if len(self.segment_lengths) != len(self.k_segment_bits):
            raise ValueError("adaptive_segment_lengths and adaptive_k_bits must have the same length.")
        if len(self.segment_lengths) != len(self.v_segment_bits):
            raise ValueError("adaptive_segment_lengths and adaptive_v_bits must have the same length.")
        if any(bits not in (2, 4, 8) for bits in self.k_segment_bits + self.v_segment_bits):
            raise ValueError("Adaptive KV supports only 2/4/8-bit segments.")


def build_adaptive_kv_config(model_args) -> AdaptiveKVConfig:
    cfg = AdaptiveKVConfig(
        enabled=bool(getattr(model_args, "adaptive_kv", False)),
        policy=getattr(model_args, "adaptive_policy", "static_distance"),
        segment_lengths=normalize_segment_lengths(parse_int_list(getattr(model_args, "adaptive_segment_lengths", None))),
        k_segment_bits=parse_int_list(getattr(model_args, "adaptive_k_bits", None)),
        v_segment_bits=parse_int_list(getattr(model_args, "adaptive_v_bits", None)),
    )
    cfg.validate()
    return cfg


def adaptive_experiment_tag(model_args) -> str:
    if not getattr(model_args, "adaptive_kv", False):
        return f"{model_args.k_bits}bits_group{model_args.group_size}_residual{model_args.residual_length}"

    cfg = build_adaptive_kv_config(model_args)
    seg = "none" if not cfg.segment_lengths else "-".join(str(x) for x in cfg.segment_lengths)
    k_bits = "none" if not cfg.k_segment_bits else "-".join(str(x) for x in cfg.k_segment_bits)
    v_bits = "none" if not cfg.v_segment_bits else "-".join(str(x) for x in cfg.v_segment_bits)
    return (
        f"adaptive_{cfg.policy}_seg{seg}_k{k_bits}_v{v_bits}"
        f"_group{model_args.group_size}_residual{model_args.residual_length}"
    )


def attach_adaptive_fields(config, model_args):
    config.adaptive_kv = bool(getattr(model_args, "adaptive_kv", False))
    config.adaptive_policy = getattr(model_args, "adaptive_policy", "static_distance")
    config.adaptive_segment_lengths = parse_int_list(getattr(model_args, "adaptive_segment_lengths", None))
    config.adaptive_k_bits = parse_int_list(getattr(model_args, "adaptive_k_bits", None))
    config.adaptive_v_bits = parse_int_list(getattr(model_args, "adaptive_v_bits", None))
    return config


def bucket_token_counts(seq_len: int, segment_lengths: List[int], residual_length: int) -> dict:
    """How many tokens of a `seq_len` key cache sit in each precision bucket.

    Mirrors the static_distance cache in models/*_kivi.py: tokens are quantized in
    chunks of `residual_length`, the newest chunks enter bucket 0, and overflow is
    pushed to the next bucket. Bucket i holds at most segment_lengths[i] tokens,
    except the last bucket, which is unbounded (its configured length is unused).
    """
    quantized = (seq_len // residual_length) * residual_length
    counts = []
    remaining = quantized
    for idx, capacity in enumerate(segment_lengths):
        take = remaining if idx == len(segment_lengths) - 1 else min(remaining, capacity)
        counts.append(take)
        remaining -= take
    return {"fp16_residual": seq_len - quantized, "segments": counts}


def effective_kv_bits(seq_len: int, segment_lengths: List[int], segment_bits: List[int], residual_length: int) -> float:
    """Average stored bits per KV element, ignoring per-group scale/zero-point overhead."""
    if seq_len <= 0:
        return 0.0
    counts = bucket_token_counts(seq_len, segment_lengths, residual_length)
    total_bits = 16 * counts["fp16_residual"] + sum(n * b for n, b in zip(counts["segments"], segment_bits))
    return total_bits / seq_len


def resolve_model_max_length(model_name_or_path: str, model2maxlen: dict) -> tuple[str, int]:
    candidates = []
    if model_name_or_path:
        candidates.append(model_name_or_path)
        candidates.append(model_name_or_path.split("/")[-1])

    seen = set()
    deduped_candidates = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        if candidate in model2maxlen:
            return candidate, int(model2maxlen[candidate])

    lower_to_key = {key.lower(): key for key in model2maxlen}
    for candidate in deduped_candidates:
        key = lower_to_key.get(candidate.lower())
        if key is not None:
            return key, int(model2maxlen[key])

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    max_length = getattr(config, "max_position_embeddings", None)
    if max_length is None:
        raise KeyError(
            f"Model '{model_name_or_path}' is missing from config/model2maxlen.json and "
            "its config does not expose max_position_embeddings."
        )
    return model_name_or_path.split("/")[-1], int(max_length)
