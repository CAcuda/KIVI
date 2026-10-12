from types import SimpleNamespace

import pytest

from utils.adaptive_kv import (
    adaptive_experiment_tag,
    bucket_token_counts,
    build_adaptive_kv_config,
    effective_kv_bits,
)

SEGMENTS = [2048, 4096, 8192]


def _args(**overrides):
    base = dict(
        adaptive_kv=True,
        adaptive_policy="static_distance",
        adaptive_segment_lengths="2048,4096,8192",
        adaptive_k_bits="8,4,2",
        adaptive_v_bits="8,4,2",
        group_size=64,
        residual_length=128,
        k_bits=2,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_short_stress_test_only_uses_8bit_segment():
    # The 160 + 338 token benchmark never leaves the first (8-bit) segment.
    counts = bucket_token_counts(498, SEGMENTS, 128)
    assert counts == {"fp16_residual": 114, "segments": [384, 0, 0]}


def test_long_context_fills_all_segments_and_last_is_unbounded():
    counts = bucket_token_counts(31500, SEGMENTS, 128)
    assert counts["segments"][:2] == [2048, 4096]
    assert counts["segments"][2] > 8192
    assert counts["fp16_residual"] + sum(counts["segments"]) == 31500


def test_effective_bits_drop_with_context_length():
    bits = [8, 4, 2]
    short = effective_kv_bits(4096, SEGMENTS, bits, 128)
    long = effective_kv_bits(31500, SEGMENTS, bits, 128)
    assert 2 < long < short < 8


@pytest.mark.parametrize("policy", ["reuse_aware", "hybrid"])
def test_unimplemented_policies_are_rejected(policy):
    with pytest.raises(ValueError):
        build_adaptive_kv_config(_args(adaptive_policy=policy))


def test_unsupported_bit_width_is_rejected():
    with pytest.raises(ValueError):
        build_adaptive_kv_config(_args(adaptive_k_bits="8,3,2"))


def test_mismatched_segment_and_bit_lists_are_rejected():
    with pytest.raises(ValueError):
        build_adaptive_kv_config(_args(adaptive_k_bits="8,4"))


def test_experiment_tag_encodes_config():
    assert adaptive_experiment_tag(_args()) == "adaptive_static_distance_seg2048-4096-8192_k8-4-2_v8-4-2_group64_residual128"
