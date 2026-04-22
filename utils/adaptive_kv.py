from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from transformers import AutoConfig


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
    reuse_window: int = 64
    reuse_topk: int = 32
    hybrid_high_bits: Optional[int] = None
    hybrid_low_bits: Optional[int] = None

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.policy not in {"static_distance", "reuse_aware", "hybrid"}:
            raise ValueError(f"Unsupported adaptive KV policy: {self.policy}")

        if self.policy == "static_distance":
            if not self.segment_lengths:
                raise ValueError("Static distance policy requires --adaptive_segment_lengths.")
            if not self.k_segment_bits or not self.v_segment_bits:
                raise ValueError("Static distance policy requires per-segment K/V bits.")
            if len(self.segment_lengths) != len(self.k_segment_bits):
                raise ValueError("adaptive_segment_lengths and adaptive_k_bits must have the same length.")
            if len(self.segment_lengths) != len(self.v_segment_bits):
                raise ValueError("adaptive_segment_lengths and adaptive_v_bits must have the same length.")


def build_adaptive_kv_config(model_args) -> AdaptiveKVConfig:
    cfg = AdaptiveKVConfig(
        enabled=bool(getattr(model_args, "adaptive_kv", False)),
        policy=getattr(model_args, "adaptive_policy", "static_distance"),
        segment_lengths=normalize_segment_lengths(parse_int_list(getattr(model_args, "adaptive_segment_lengths", None))),
        k_segment_bits=parse_int_list(getattr(model_args, "adaptive_k_bits", None)),
        v_segment_bits=parse_int_list(getattr(model_args, "adaptive_v_bits", None)),
        reuse_window=int(getattr(model_args, "reuse_window", 64)),
        reuse_topk=int(getattr(model_args, "reuse_topk", 32)),
        hybrid_high_bits=getattr(model_args, "hybrid_high_bits", None),
        hybrid_low_bits=getattr(model_args, "hybrid_low_bits", None),
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
    config.reuse_window = int(getattr(model_args, "reuse_window", 64))
    config.reuse_topk = int(getattr(model_args, "reuse_topk", 32))
    config.hybrid_high_bits = getattr(model_args, "hybrid_high_bits", None)
    config.hybrid_low_bits = getattr(model_args, "hybrid_low_bits", None)
    return config


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

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    max_length = getattr(config, "max_position_embeddings", None)
    if max_length is None:
        raise KeyError(
            f"Model '{model_name_or_path}' is missing from config/model2maxlen.json and "
            "its config does not expose max_position_embeddings."
        )
    return model_name_or_path.split("/")[-1], int(max_length)
