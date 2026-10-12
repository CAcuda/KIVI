"""Crash-safe LongBench prediction files.

A task counts as finished only when both of these hold:
  * `<dataset>.jsonl` was written atomically (temp file + os.replace), and
  * `<dataset>.done.json` records the expected sample count and the number of
    predictions written, and the two agree with the lines in the jsonl file.

A non-empty jsonl alone is not trusted: it can come from an interrupted write,
from an older run with a different config, or from a partial copy between
volumes. Such files are re-run instead of being skipped.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional


def done_marker_path(prediction_path: str) -> str:
    base = prediction_path[: -len(".jsonl")] if prediction_path.endswith(".jsonl") else prediction_path
    return f"{base}.done.json"


def _atomic_write_text(path: str, text: str) -> None:
    # The temp name must not end in "jsonl": eval_long_bench.py scores every *jsonl file.
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def write_predictions(prediction_path: str, preds: List[dict], expected_count: int) -> None:
    """Write predictions and, only if the count matches, the completion marker."""
    if len(preds) != expected_count:
        raise ValueError(
            f"Refusing to mark {prediction_path} complete: got {len(preds)} predictions, "
            f"expected {expected_count}."
        )
    marker_path = done_marker_path(prediction_path)
    if os.path.exists(marker_path):
        os.remove(marker_path)
    lines = "".join(json.dumps(pred, ensure_ascii=False) + "\n" for pred in preds)
    _atomic_write_text(prediction_path, lines)
    marker = {"expected": expected_count, "num_predictions": len(preds)}
    _atomic_write_text(marker_path, json.dumps(marker, indent=2))


def _count_valid_lines(prediction_path: str) -> Optional[int]:
    count = 0
    try:
        with open(prediction_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "pred" not in record or "answers" not in record:
                    return None
                count += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return count


def is_complete(prediction_path: str) -> bool:
    marker_path = done_marker_path(prediction_path)
    if not (os.path.exists(prediction_path) and os.path.exists(marker_path)):
        return False
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)
        expected = int(marker["expected"])
        written = int(marker["num_predictions"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if expected <= 0 or written != expected:
        return False
    return _count_valid_lines(prediction_path) == expected


def missing_datasets(exp_dir: str, datasets: Iterable[str]) -> List[str]:
    return [d for d in datasets if not is_complete(os.path.join(exp_dir, f"{d}.jsonl"))]
