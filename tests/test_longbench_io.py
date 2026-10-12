import json

import pytest

from utils.longbench_io import done_marker_path, is_complete, missing_datasets, write_predictions


def _preds(n):
    return [{"pred": f"p{i}", "answers": [f"a{i}"], "all_classes": None, "length": i} for i in range(n)]


def test_complete_output_is_skipped(tmp_path):
    out = tmp_path / "qasper.jsonl"
    write_predictions(str(out), _preds(3), expected_count=3)
    assert is_complete(str(out))
    assert missing_datasets(str(tmp_path), ["qasper"]) == []


def test_count_mismatch_refuses_to_write(tmp_path):
    out = tmp_path / "qasper.jsonl"
    with pytest.raises(ValueError):
        write_predictions(str(out), _preds(2), expected_count=3)
    assert not out.exists()
    assert not is_complete(str(out))


def test_legacy_nonempty_file_without_marker_is_rerun(tmp_path):
    out = tmp_path / "qasper.jsonl"
    out.write_text("".join(json.dumps(p) + "\n" for p in _preds(3)))
    assert not is_complete(str(out))
    assert missing_datasets(str(tmp_path), ["qasper"]) == ["qasper"]


def test_truncated_file_is_rerun(tmp_path):
    out = tmp_path / "qasper.jsonl"
    write_predictions(str(out), _preds(3), expected_count=3)
    lines = out.read_text().splitlines(keepends=True)
    out.write_text("".join(lines[:2]))
    assert not is_complete(str(out))


def test_corrupted_line_is_rerun(tmp_path):
    out = tmp_path / "qasper.jsonl"
    write_predictions(str(out), _preds(3), expected_count=3)
    out.write_text(out.read_text()[:-10])
    assert not is_complete(str(out))


def test_corrupted_marker_is_rerun(tmp_path):
    out = tmp_path / "qasper.jsonl"
    write_predictions(str(out), _preds(3), expected_count=3)
    with open(done_marker_path(str(out)), "w") as f:
        f.write("{not json")
    assert not is_complete(str(out))


def test_rewrite_leaves_no_temp_files_for_eval_to_pick_up(tmp_path):
    out = tmp_path / "qasper.jsonl"
    write_predictions(str(out), _preds(3), expected_count=3)
    write_predictions(str(out), _preds(3), expected_count=3)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["qasper.done.json", "qasper.jsonl"]
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith("jsonl")] == ["qasper.jsonl"]
