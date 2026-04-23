# Adaptive KV Implementation and Results

This document records the adaptive KV-cache work completed in this repository, including the implemented method, code changes, experiment workflow, Modal execution setup, and the final benchmark and LongBench results.

## Objective

The target method is `Adaptive Precision Across Sequence Length` for KV-cache quantization.

The implemented policy in this project is:

- `adaptive_policy = static_distance`
- sequence segments: `2048, 4096, 8192`
- key precision by segment: `8, 4, 2`
- value precision by segment: `8, 4, 2`

The intuition is:

- tokens that remain closer to the active decoding region keep higher precision
- older tokens are stored with lower precision
- the method aims to improve the memory-quality tradeoff compared with one fixed KV bit-width

## Implemented Scope

The current implementation covers inference-time adaptive KV only. No model training or fine-tuning was added.

The repository now supports:

- adaptive KV argument parsing
- adaptive KV configuration helpers
- adaptive memory/speed benchmarking
- adaptive LongBench prediction
- adaptive LongBench evaluation
- Modal-based detached execution
- per-task checkpointing and resume for LongBench
- per-task score persistence
- 3-way parallel execution on Modal with:
  - `1 task = 1 container = 1 A100`

## Main Code Changes

### 1. Adaptive argument and config helpers

Added:

- `utils/process_args_adaptive.py`
- `utils/adaptive_kv.py`

These files provide:

- adaptive CLI arguments:
  - `--adaptive_kv`
  - `--adaptive_policy`
  - `--adaptive_segment_lengths`
  - `--adaptive_k_bits`
  - `--adaptive_v_bits`
  - `--use_flash_attention_2`
- config attachment helpers for adaptive fields
- adaptive experiment tag generation
- model max-length resolution

### 2. Adaptive prediction and benchmark entrypoints

Added:

- `pred_long_bench_adaptive.py`
- `scripts/run_adaptive_mem_speed.py`
- `scripts/run_adaptive_repro.sh`

These files provide a dedicated adaptive workflow without replacing the original baseline scripts.

`pred_long_bench_adaptive.py` now supports:

- adaptive model loading
- adaptive LongBench inference
- explicit `pred_root`
- explicit dataset subset selection via `--datasets`
- skipping already completed dataset outputs

`scripts/run_adaptive_mem_speed.py` provides:

- fixed-length generation benchmark compatible with the baseline-style throughput/latency test
- JSON output for memory and speed results

### 3. LLaMA adaptive KV support

Updated:

- `models/llama_kivi.py`

Key changes:

- added adaptive configuration fields to attention/config consumption
- introduced segment-aware adaptive cache state
- added bucket creation, quantization, dequantization, rollover, and rebalance helpers
- implemented adaptive decode-time attention over multiple KV precision buckets
- preserved residual full-precision tail behavior
- kept non-adaptive behavior available when `adaptive_kv=False`

### 4. Mistral adaptive KV support

Updated:

- `models/mistral_kivi.py`

Key changes:

- added adaptive KV support to `MistralAttention_KIVI`
- added adaptive bucket state, quantization, dequantization, and bucket rebalance logic
- added adaptive decode path and adaptive prefill state construction
- added compatibility handling for `DynamicCache` during generation
- fixed Mistral rotary embedding calls for the installed `transformers` version
- fixed flash-attention mask path so flash mode uses the correct 2D mask path instead of the standard 4D attention mask
- added adaptive support to the Mistral flash-attention class as well

These fixes were necessary to make adaptive KV work on:

- `mistralai/Mistral-7B-Instruct-v0.2`

### 5. Evaluation and result path support

Updated:

- `eval_long_bench.py`

Changes:

- added optional `--pred_root`
- allowed evaluation from persistent output directories rather than only `pred/` or `pred_e/`

This was required for Modal checkpointing and resume.

### 6. Modal execution and checkpointing

Added:

- `scripts/modal_adaptive_run.py`
- `scripts/run_modal_adaptive_detached.sh`

This Modal workflow now supports:

- detached execution via `modal run --detach`
- benchmark-only runs
- LongBench-only runs
- adaptive per-task execution
- persistent outputs on Modal volumes

The LongBench workflow was further upgraded to:

- launch each dataset as its own Modal function call
- use `1 task = 1 container = 1 A100`
- limit concurrency to `3` simultaneous containers
- save outputs immediately after each task finishes
- resume from existing `jsonl` files
- backfill missing per-task scores when `jsonl` exists but `score.json` is missing

## Modal Persistence Layout

During cloud execution, outputs are saved under the Modal volume `kivi-outputs`.

Important paths:

- benchmark result:
  - `/adaptive_benchmark.json`
- LongBench predictions:
  - `/pred/<model_tag>/*.jsonl`
- LongBench final result:
  - `/pred/<model_tag>/result.json`
- per-task scores:
  - `/task_scores/<model_tag>/*.score.json`
- per-task metadata:
  - `/task_meta/<model_tag>/*.json`

This layout ensures that if a run is interrupted, already finished tasks are not lost.

## Experiment Configuration

### Model

- `mistralai/Mistral-7B-Instruct-v0.2`

### Adaptive KV setting

- policy: `static_distance`
- segment lengths: `2048,4096,8192`
- key bits: `8,4,2`
- value bits: `8,4,2`
- group size: `64`
- residual length: `128`

### Memory/speed benchmark setting

This benchmark was intentionally aligned to the baseline-style fixed-length stress test:

- batch size: `96`
- prompt length: `160`
- generation length: `338`
- repeats: `3`
- dtype: `float16`
- FlashAttention-2: `false`

### LongBench task set

The final LongBench run used the requested 9-task subset:

- `triviaqa`
- `qasper`
- `trec`
- `samsum`
- `lcc`
- `repobench-p`
- `qmsum`
- `multi_news`
- `passage_retrieval_en`

FlashAttention-2 was enabled for the LongBench run.

## Final Results

### A. Memory and speed benchmark

Source:

- `outputs/adaptive_results_local/adaptive_benchmark.json`

Result:

- model: `mistralai/Mistral-7B-Instruct-v0.2`
- mode: `adaptive_kv`
- policy: `static_distance`
- segment lengths: `2048,4096,8192`
- KV bits: `8,4,2`
- batch size: `96`
- prompt length: `160`
- output length: `338`
- repeats: `3`
- FlashAttention-2: `false`
- average latency: `50999.389 ms`
- peak memory: `17.829 GB`

### B. LongBench results

Source:

- `outputs/adaptive_results_local/Mistral-7B-Instruct-v0.2_31500_adaptive_static_distance_seg2048-4096-8192_k8-4-2_v8-4-2_group64_residual128/result.json`

Scores:

| Task                     |     Score |
| ------------------------ | --------: |
| `triviaqa`             | `86.64` |
| `qasper`               | `33.18` |
| `trec`                 | `71.00` |
| `samsum`               | `43.18` |
| `lcc`                  | `56.08` |
| `repobench-p`          | `53.94` |
| `qmsum`                | `24.01` |
| `multi_news`           | `26.96` |
| `passage_retrieval_en` | `73.62` |

## Local Saved Artifacts

The cloud outputs were downloaded into the local repository under:

- `outputs/adaptive_results_local/adaptive_benchmark.json`
- `outputs/adaptive_results_local/Mistral-7B-Instruct-v0.2_31500_adaptive_static_distance_seg2048-4096-8192_k8-4-2_v8-4-2_group64_residual128/result.json`
- `outputs/adaptive_results_local/Mistral-7B-Instruct-v0.2_31500_adaptive_static_distance_seg2048-4096-8192_k8-4-2_v8-4-2_group64_residual128/task_scores/`
- `outputs/adaptive_results_local/Mistral-7B-Instruct-v0.2_31500_adaptive_static_distance_seg2048-4096-8192_k8-4-2_v8-4-2_group64_residual128/predictions/`

These local files now include:

- benchmark summary
- final aggregated LongBench result
- per-task score JSON files
- raw prediction `jsonl` files for all 9 tasks

## Notes on Reproducibility

The repository now contains both:

- the original baseline path
- the new adaptive KV path

The adaptive path is isolated through dedicated scripts and helper utilities. The most important operational additions are:

- checkpoint-safe output persistence
- dataset-level resume
- detached Modal execution
- per-task parallel cloud execution

## Current Status

The implemented `static_distance` adaptive KV workflow is complete enough to:

- run benchmark measurements
- run 9-task LongBench evaluation
- save artifacts locally
- save artifacts in Modal volumes
- resume interrupted evaluations safely

This is the current working adaptive inference pipeline in this repository.
