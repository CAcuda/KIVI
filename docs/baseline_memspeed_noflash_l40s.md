# Baseline Mem-Speed (No FlashAttention) on L40S

本文档给出一套可复现流程：
- 仅跑 baseline 全精度 mem-speed
- 强制关闭 FlashAttention-2
- 与 KIVI-2bit/KIVI-4bit 保持一致设置（batch/prompt/output/repeats）
- 固定在 L40S 节点上运行

## 1. 目标设置

- Model: `mistralai/Mistral-7B-Instruct-v0.2`
- GPU: `L40S` (`--constraint=l40s`, `--gres=gpu:1`)
- DType: `float16`
- Batch size: `96`
- Prompt length: `160`
- Output length: `338`
- Repeats: `3`
- FlashAttention-2: `OFF`（强制关闭）

## 2. 前置条件

在 Bridges2 登录节点执行：

```bash
cd /jet/home/can2/KIVI
module load cuda/12.6.1 anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /ocean/projects/cis260009p/can2/conda_envs/kivi
```

可选检查：

```bash
python -V
python -m py_compile scripts/run_baseline_mem_speed.py
```

## 3. 关键参数说明

`scripts/run_baseline_mem_speed.py` 新增参数：

- `--disable_flash_attention_2`
  - 启用后强制将 `flash_attention_2_enabled` 置为 `false`
  - 即使机器支持 FA2，也不会启用

## 4. 提交作业（推荐）

```bash
cd /jet/home/can2/KIVI

sbatch \
  --job-name=kivi-base-mem-noflash \
  --partition=GPU-shared \
  --constraint=l40s \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=22G \
  --time=02:00:00 \
  --output=/jet/home/can2/KIVI/logs/kivi-base-mem-noflash-%j.out \
  --error=/jet/home/can2/KIVI/logs/kivi-base-mem-noflash-%j.err \
  --wrap='set -e; \
    cd /jet/home/can2/KIVI; \
    module load cuda/12.6.1 anaconda3/2024.10-1; \
    source "$(conda info --base)/etc/profile.d/conda.sh"; \
    conda activate /ocean/projects/cis260009p/can2/conda_envs/kivi; \
    export PYTHONPATH=/jet/home/can2/KIVI:$PYTHONPATH; \
    export HF_HOME=/ocean/projects/cis260009p/can2/hf_home; \
    export HUGGINGFACE_HUB_CACHE=/ocean/projects/cis260009p/can2/hf_home/hub; \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
    CUDA_VISIBLE_DEVICES=0 python scripts/run_baseline_mem_speed.py \
      --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
      --cache_dir /ocean/projects/cis260009p/can2/kivi_cached_models \
      --batch_size 96 \
      --prompt_length 160 \
      --output_length 338 \
      --num_repeats 3 \
      --disable_flash_attention_2 \
      --out_json outputs/baseline_mistral_l40s_mem_speed_bs96_noflash.json'
```

## 5. 监控作业

提交后查看状态：

```bash
squeue -u can2
squeue -j <JOB_ID> -O jobid,state,reason,timeused,starttime,nodelist,name
```

查看日志：

```bash
tail -n 120 logs/kivi-base-mem-noflash-<JOB_ID>.out
tail -n 120 logs/kivi-base-mem-noflash-<JOB_ID>.err
```

## 6. 结果校验

输出文件：

`outputs/baseline_mistral_l40s_mem_speed_bs96_noflash.json`

重点校验字段：

- `"flash_attention_2_enabled": false`
- `"batch_size": 96`（或因 OOM 自动缩小后显示有效 batch）
- `"prompt_length": 160`
- `"output_length": 338`

示例检查命令：

```bash
cat outputs/baseline_mistral_l40s_mem_speed_bs96_noflash.json
```

## 7. 与 KIVI-2/4 对比

如果已有以下文件：

- `outputs/kivi2_mistral_l40s_mem_speed_bs96.json`
- `outputs/kivi4_mistral_l40s_mem_speed_bs96.json`

可用下面命令快速对比：

```bash
python - <<'PY'
import json
from pathlib import Path

b = json.loads(Path('outputs/baseline_mistral_l40s_mem_speed_bs96_noflash.json').read_text())
k2 = json.loads(Path('outputs/kivi2_mistral_l40s_mem_speed_bs96.json').read_text())
k4 = json.loads(Path('outputs/kivi4_mistral_l40s_mem_speed_bs96.json').read_text())

print('metric\tbaseline_noflash\tkivi2\tdelta2\tkivi4\tdelta4')
print(f"peak_mem_gb\t{b['peak_mem_gb']:.6f}\t{k2['peak_mem_gb']:.6f}\t{k2['peak_mem_gb']-b['peak_mem_gb']:+.6f}\t{k4['peak_mem_gb']:.6f}\t{k4['peak_mem_gb']-b['peak_mem_gb']:+.6f}")
print(f"avg_time_ms\t{b['avg_time_ms']:.6f}\t{k2['avg_time_ms']:.6f}\t{k2['avg_time_ms']-b['avg_time_ms']:+.6f}\t{k4['avg_time_ms']:.6f}\t{k4['avg_time_ms']-b['avg_time_ms']:+.6f}")
PY
```

## 8. 常见问题

- 队列长时间 `PENDING`:
  - 使用 `squeue -j <JOB_ID> -O jobid,state,reason,starttime,name` 查看原因。
- 缺包报错:
  - 确保使用 `kivi` 环境激活后再执行。
- OOM:
  - 脚本已支持自动降 batch 重试，结果会记录有效 batch。
