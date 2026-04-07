# KIVI Baseline Run Guide

This note summarizes a stable way to reproduce the KIVI baseline run (memory/speed + LongBench task metrics), based on actual runs in this repository.

## 1. What baseline means here

- Full-precision KV cache baseline in this repo scripts:
  - `k_bits = 16`
  - `v_bits = 16`
- Pipeline run by `scripts/run_baseline_repro.sh`:
  1. `scripts/run_baseline_mem_speed.py` -> time/memory JSON
  2. `pred_long_bench.py` -> task predictions
  3. `eval_long_bench.py` -> task metrics JSON

## 2. Recommended working directory

Use the Ocean project copy to avoid home quota issues:

- Recommended repo: `/ocean/projects/cis260009p/can2/KIVI`
- Home repo also exists: `/jet/home/can2/KIVI`

If you work in both copies, sync outputs manually at the end.

## 3. Environment setup (interactive)

```bash
cd /ocean/projects/cis260009p/can2/KIVI
module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmsys-hw5

export HF_HOME=/ocean/projects/cis260009p/can2/hf_home
export PYTHONDONTWRITEBYTECODE=1
export PYDEPS=/ocean/projects/cis260009p/can2/kivi_pydeps_compat
mkdir -p "$HF_HOME/transformers" "$PYDEPS" outputs /ocean/projects/cis260009p/can2/kivi_cached_models
export PYTHONPATH="$PYDEPS:$PYTHONPATH"
```

## 4. Run baseline in one command

```bash
bash scripts/run_baseline_repro.sh \
  0 \
  mistralai/Mistral-7B-Instruct-v0.2 \
  32 \
  128 \
  0 \
  /ocean/projects/cis260009p/can2/kivi_cached_models
```

Arguments:

- `0`: GPU id
- model name/path
- `group_size` (default `32`)
- `residual_length` (default `128`)
- `RUN_LONG_E` (`0` for LongBench, `1` for LongBench-E)
- cache dir

## 5. Submit with Slurm (recommended for full run)

```bash
cd /ocean/projects/cis260009p/can2/KIVI
sbatch --job-name=kivi-base-ms \
  --partition=GPU-shared --gres=gpu:1 \
  --time=12:00:00 --cpus-per-task=4 --mem=22G \
  --output=/ocean/projects/cis260009p/can2/KIVI/logs/kivi-base-ms-%j.out \
  --error=/ocean/projects/cis260009p/can2/KIVI/logs/kivi-base-ms-%j.err \
  --wrap='set -e; cd /ocean/projects/cis260009p/can2/KIVI; \
  export HF_HOME=/ocean/projects/cis260009p/can2/hf_home; \
  export PYTHONDONTWRITEBYTECODE=1; \
  export PYDEPS=/ocean/projects/cis260009p/can2/kivi_pydeps_compat; \
  mkdir -p "$HF_HOME/transformers" outputs /ocean/projects/cis260009p/can2/kivi_cached_models "$PYDEPS"; \
  module load anaconda3/2024.10-1; \
  source "$(conda info --base)/etc/profile.d/conda.sh"; \
  conda activate llmsys-hw5; \
  export PYTHONPATH="$PYDEPS:$PYTHONPATH"; \
  bash scripts/run_baseline_repro.sh 0 mistralai/Mistral-7B-Instruct-v0.2 32 128 0 /ocean/projects/cis260009p/can2/kivi_cached_models'
```

Monitor:

```bash
squeue -u can2
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed -n -P
```

## 6. Expected output files

- `outputs/baseline_mem_speed.json`
  - keys include `avg_time_ms`, `peak_mem_gb`, `batch_size`, `output_length`
- `outputs/baseline_longbench_result.json`
  - task-level scores

## 7. Evaluation-only rerun (skip expensive generation)

If `[2/3]` predictions already exist and only final eval failed, run:

```bash
cd /ocean/projects/cis260009p/can2/KIVI
module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmsys-hw5
export PYDEPS=/ocean/projects/cis260009p/can2/kivi_pydeps_compat
export PYTHONPATH="$PYDEPS:$PYTHONPATH"
python eval_long_bench.py --model Mistral-7B-Instruct-v0.2_31500_16bits_group32_residual128
cp pred/Mistral-7B-Instruct-v0.2_31500_16bits_group32_residual128/result.json outputs/baseline_longbench_result.json
```

## 8. Common failure cases and fixes

- Job hangs at dataset prompt:
  - Ensure `load_dataset(..., trust_remote_code=True)` in `pred_long_bench.py`.
- Tokenizer issues (`fast tokenizer` / protobuf path):
  - Use slow tokenizer path with `use_fast=False, legacy=True` (already adjusted in Ocean copy).
- `transformers` / `tokenizers` mismatch:
  - Keep `tokenizers` compatible with your transformers version.
- Missing eval dependencies (for some environments):
  - `jieba`, `fuzzywuzzy`, `rouge` may be required; use compatible env or fallback-enabled code.
- Home quota problems:
  - Run from `/ocean/.../KIVI` and keep cache/output under `/ocean/...`.

## 9. Sync outputs back to home repo (optional)

```bash
mkdir -p /jet/home/can2/KIVI/outputs
cp /ocean/projects/cis260009p/can2/KIVI/outputs/baseline_mem_speed.json /jet/home/can2/KIVI/outputs/
cp /ocean/projects/cis260009p/can2/KIVI/outputs/baseline_longbench_result.json /jet/home/can2/KIVI/outputs/
```
