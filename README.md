# SP-KV: Sequence-Position Mixed-Precision KV Cache (built on KIVI)

This is a course-project fork of [KIVI](https://github.com/jy-yuan/KIVI). It adds **SP-KV**, a KV cache that sets
quantization precision by how far a token is from the current decoding position, plus a tool flow that runs
LongBench and memory/latency benchmarks on cloud GPUs and resumes after interruptions.
The original KIVI README is kept unchanged [below](#upstream-kivi-readme).

## Method

KIVI quantizes every cached token to the same bit-width. SP-KV keeps recent tokens at higher precision:

| Position in the KV cache (newest → oldest) | Precision |
|---|---|
| Last `residual_length` (128) tokens | FP16, as in KIVI |
| Next 2,048 tokens | 8-bit |
| Next 4,096 tokens | 4-bit |
| Everything older | 2-bit |

Keys are quantized per channel and values per token, following KIVI. During decoding, each new 128-token chunk
is quantized into the 8-bit segment. When a segment overflows, its oldest tokens are dequantized and requantized
into the next lower-precision segment. Attention logits are computed per segment and concatenated before softmax.
The last segment has no size limit, so the third value in `--adaptive_segment_lengths 2048,4096,8192` is not used.

## What I added

| Area | Files |
|---|---|
| Segmented cache, rollover between segments, and the decode/prefill attention path (Mistral and Llama, eager and FlashAttention prefill) | `models/mistral_kivi.py`, `models/llama_kivi.py` |
| Config parsing and validation; per-segment token and effective-bit accounting | `utils/adaptive_kv.py`, `utils/process_args_adaptive.py` |
| LongBench prediction with crash-safe output and resume | `pred_long_bench_adaptive.py`, `utils/longbench_io.py` |
| One benchmark script for FP16, fixed-bit KIVI and SP-KV (warmup, fixed decode length, weight vs. KV memory split, OOM batch backoff) | `scripts/run_adaptive_mem_speed.py` |
| Modal jobs: one LongBench task per A100 container (max 3 concurrent), detached runs, persistent volumes, and a controlled FP16/KIVI/SP-KV comparison | `scripts/modal_adaptive_run.py`, `scripts/modal_baseline_run.py` |
| Unit tests for resume logic and config/bucket accounting | `tests/` |
| Fixes to fixed-bit KIVI decoding with FlashAttention (see below) | `models/*_kivi.py`, `quant/matmul.py` |

**Resume safety.** A LongBench task counts as finished only when its `.jsonl` file was written atomically and a
`.done.json` marker records a prediction count equal to the dataset size. Truncated files, corrupted files and
files without a marker are re-run instead of being skipped.

## Results

Model: Mistral-7B-Instruct-v0.2. Inputs longer than 31,500 tokens are truncated in the middle.
SP-KV runs used group size 64 and residual length 128. Raw SP-KV predictions and scores are in
`outputs/adaptive_results_local/`. The FP16 and KIVI columns come from the course project's shared baseline runs.
Only the FP16 PassageRetrieval output (`outputs/baseline_single/`) is stored in this repo.

| Task | FP16 | KIVI-2bit | KIVI-4bit | **SP-KV** | SP-KV est. avg bits* |
|---|---:|---:|---:|---:|---:|
| TriviaQA | 86.23 | 86.74 | 87.30 | 86.64 | 4.5 |
| Qasper | 32.54 | 30.48 | 33.02 | **33.18** | 6.2 |
| TREC | 71.00 | 70.50 | 70.50 | 71.00 | 5.4 |
| SAMSum | 42.89 | 43.48 | 43.48 | 43.18 | 5.1 |
| LCC | 57.79 | 57.33 | 57.60 | 56.08 | 8.2 |
| RepoBench-P | 55.66 | 51.86 | 52.90 | 53.94 | 6.1 |
| QMSum | 24.31 | 23.13 | 22.87 | 24.01 | 3.9 |
| MultiNews | 27.03 | 26.68 | 26.96 | 26.96 | 7.5 |
| PassageRetrieval-en | 86.98 | 80.83 | — | **73.62** | 3.8 |
| Macro avg, 8 tasks (excl. PassageRetrieval) | 49.68 | 48.78 | 49.33 | 49.37 | |
| Macro avg, 9 tasks | 53.83 | 52.34 | — | 52.07 | |

\* Mean stored bits per KV element over each task's samples, from `utils.adaptive_kv.effective_kv_bits`.
Token counts are estimated as 1.3 × LongBench's word count and exclude scale/zero-point overhead.
Fixed KIVI uses 2 or 4 bits regardless of length.

### What these results show, and what they don't

- **Quality vs. budget.** On the 8 tasks, SP-KV scores +0.59 over KIVI-2bit and matches KIVI-4bit.
  At LongBench lengths, however, SP-KV stores about 4–8 bits per element on average, more than KIVI-4bit on most
  tasks. The gains on Qasper and RepoBench-P come at about 6 bits. This is **not** a comparison at equal memory.
  With these segment sizes, SP-KV only falls to an average near 2–3 bits for contexts of roughly 16K tokens or more.
- **PassageRetrieval regression.** SP-KV scores 7.2 points below KIVI-2bit, even though it stores every token at
  2 bits or more. This is not explained yet. Things to check: the error that accumulates as tokens are requantized
  8→4→2-bit, and whether the KIVI-2bit baseline used a different group size.
- The Llama code path is implemented but has not been evaluated.

### Long-context memory and latency (controlled comparison)

All four configs ran in one container on one A100-SXM4-40GB, with batch 4, an 8,192-token prompt,
256 forced generated tokens, FlashAttention prefill, group size 64, residual length 128, one warmup run,
and the mean of 3 timed runs. Command: `modal run scripts/modal_adaptive_run.py --action compare`.
Raw JSON is in `outputs/compare_a100_p8192/`.

| Config | Time per run | vs FP16 | Decode tok/s | Peak mem − weights | vs FP16 |
|---|---:|---:|---:|---:|---:|
| FP16 | 23.1 s | 1.00× | 44.3 | 10.12 GB | 1.00× |
| KIVI-2bit | 19.0 s | 0.82× | 54.0 | 6.77 GB | 0.67× |
| KIVI-4bit | 18.8 s | 0.81× | 54.4 | 7.27 GB | 0.72× |
| SP-KV (8/4/2) | 41.8 s | 1.81× | 24.5 | 7.40 GB | 0.73× |

At 8,448 tokens, SP-KV holds 2,048 / 4,096 / 2,304 tokens in its 8 / 4 / 2-bit segments, an average of 4.4 bits.
Its memory is therefore close to KIVI-4bit.

- **SP-KV is about 2.2× slower than fixed KIVI.** KIVI's fused CUDA kernel supports only 2- and 4-bit, so the
  8-bit segment is dequantized to FP16 and multiplied in PyTorch at every decode step in every layer. Every
  128 tokens, segment rollover also dequantizes, requantizes and concatenates tensors. The obvious next steps
  are an 8-bit path in the fused kernel and preallocated segment buffers.
- **Peak memory understates the KV savings.** Peak memory includes prefill transients such as FP32 logits for
  the whole prompt (about 4.2 GB here), which are the same for every config. For reference, the FP16 KV cache
  alone is about 4.4 GB at this size.
- The earlier 160 + 338-token stress test never filled SP-KV's 4-bit or 2-bit segments, so its numbers
  (including the 1.84× latency on the poster) describe only the 8-bit path.

### Bugs found while building the benchmark

- The SP-KV refactor had dropped the cache write-back in the fixed-bit decode branch of the FlashAttention classes
  (`MistralFlashAttention_KIVI`, `LlamaFlashAttention_KIVI`). Fixed-bit KIVI with FlashAttention decoded against
  a stale cache and then hit an assertion. This is fixed. SP-KV's own branch was not affected.
- `quant/matmul.py` called `.view()` on a query tensor that is non-contiguous on the FlashAttention decode path.
  It now uses `.reshape()`.

## Reproduce

Set up the environment and the CUDA kernels as described in the upstream README below.

```bash
# LongBench predictions with SP-KV (re-running skips only verified, complete tasks)
python pred_long_bench_adaptive.py \
  --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
  --group_size 64 --residual_length 128 --use_flash_attention_2 true \
  --adaptive_kv --adaptive_policy static_distance \
  --adaptive_segment_lengths 2048,4096,8192 --adaptive_k_bits 8,4,2 --adaptive_v_bits 8,4,2 \
  --datasets qasper,trec
python eval_long_bench.py --pred_root pred \
  --model Mistral-7B-Instruct-v0.2_31500_adaptive_static_distance_seg2048-4096-8192_k8-4-2_v8-4-2_group64_residual128

# Memory/latency for one config (--full_precision, fixed --k_bits/--v_bits, or --adaptive_kv ...)
python scripts/run_adaptive_mem_speed.py --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
  --batch_size 4 --prompt_length 8192 --output_length 256 --group_size 64 --adaptive_kv \
  --adaptive_segment_lengths 2048,4096,8192 --adaptive_k_bits 8,4,2 --adaptive_v_bits 8,4,2

# On Modal: all 9 LongBench tasks in parallel, or the controlled comparison
modal run --detach scripts/modal_adaptive_run.py --action longbench
modal run scripts/modal_adaptive_run.py --action compare

# Unit tests (CPU only)
python -m pytest tests
```

Implementation notes and the full experiment log are in [docs/adaptive_kv_plan.md](docs/adaptive_kv_plan.md).

---

## Upstream KIVI README

Implementation of [ICML24] [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)

## Updates
- [2025.09.25]: 🔥🔥🔥 We released a toolkit that tests the impact of numerical precision and enables deterministic LLM inference. This helps eliminate the training–inference mismatch in reinforcement learning. Please 🌟 it if you find this work is interesting and useful for your research! <https://github.com/nanomaoli/llm_reproducibility>
- [2025.01.18]: We add KIVI implementation with GQA and compatible with transformers 4.43. Now it supports the LLama3 family. Please reinstall KIVI.
- [2024.06.07]: 🎉 KIVI largely inspires the [HuggingFace Transformers KV Cache quantization](https://huggingface.co/docs/transformers/main/en/kv_cache)
- [2024.06.06]: (Beta) We extensively optimize the codebase in [branch develop](https://github.com/jy-yuan/KIVI/tree/develop) to reduce the latency of KIVI. Note that **you need to reinstall our CUDA implementation** under the ```quant``` folder.
- [2024.05.01]: 🎉 KIVI has been accepted by ICML 2024! See you in Vienna!
- [2024.04.12]: We add support for the Mistral model family. The performance of LongChat-7b-v1.5-32K and Mistral-7B-Instruct-v0.2 on 15 tasks from LongBench can be found in [long_bench.md](./docs/long_bench.md).

- [2024.04.05]: We release the code for reproducing our CoQA/TruthfulQA/GSM8K results using LM-Eval. Please check the [README of branch lmeval](https://github.com/jy-yuan/KIVI/tree/lmeval).

- [2024.04.04]: 🔥🔥We add a new 5-digit [passkey example](./long_context_example.py) with 12k context length to show the performance of 2bit KIVI under the long context senario.

- [2024.04.04]: (Beta) We add the flash-attention support for KIVI during the prefill phase. 

- [2024.04.03]: We add a new [5-shot GSM8K example.py](./example.py) to show the performance of 2/4 bit KIVI with 32 full precision tokens.

- [2024.02.05]: KIVI ver. 2 is released on [arXiv](https://arxiv.org/abs/2402.02750).

- [2024.02.03]: KIVI code is released.

- [2023.12.29]: KIVI ver. 1 is released on [researchgate](https://www.researchgate.net/publication/376831635_KIVI_Plug-and-play_2bit_KV_Cache_Quantization_with_Streaming_Asymmetric_Quantization).

## Overview

KIVI is a new plug-and-play 2bit KV cache quantization algorithm without any fine-tuning. This algorithm optimizes memory usage by quantizing the key cache per-channel and the value cache per-token to 2bit. KIVI's hardware-friendly design allows LLMs like Llama-2, Falcon, and Mistral to maintain comparable quality levels while reducing peak memory usage by 2.6 times. This enables up to 4 times larger batch sizes and significantly increases throughput by 2.35 to 3.47 times in real LLM inference workloads, effectively addressing the bottleneck issues in speed and memory usage.

Illustration of KIVI quantization scheme: key cache per-channel and value cache per-token.
<p align="center">
<img width="300" src="./img/quant_scheme.png">
</p>

Illustration of KIVI algorithm during inference prefill and decoding phase:
<p align="center">
<img width="700" src="./img/algo.png">
</p>

## How to use KIVI

### Setup

To install the required packages:

```bash
conda create -n kivi python=3.10
conda activate kivi
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

Then install our CUDA implementation:

```bash
cd quant && pip install -e .
```

### Example

Load model with KIVI: (e.g., Llama-2-7b)

```python
# LLaMA model with KIVI
import torch
import os
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

config.k_bits = K_BITS # current support 2/4 bit for KV Cache
config.v_bits = V_BITS # current support 2/4 bit for KV Cache
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH # the number of recent fp16 tokens
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

model = LlamaForCausalLM_KIVI.from_pretrained(
    pretrained_model_name_or_path='meta-llama/Llama-2-7b-hf',
    config=config,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-hf', 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')

# Inference
# e.g., model.generate(...)
```

#### GSM8K example
We use GSM8K as an example to show how to use KIVI. You can check [example.py](./example.py):

```bash
python example.py
```

#### Passkey retrieval example

Passkey retrieval with KIVI. You can check [long_context_example.py](./long_context_example.py):

```bash
python long_context_example.py
```

#### Evaluate KIVI on LongBench

We currently support Llama and Mistral family of models. We recently test KIVI on Mistral-7B-Instruct-v0.2 and Longchat-7b-v1.5-32k. Please check [long_bench.md](./docs/long_bench.md) for more details.
```bash
bash scripts/long_test.sh {GPU_ID} {K_BITS} {V_BITS} {GROUP_LENGTH} {RESIDUAL_LENGTH} {MODEL_NAME}
python eval_long_bench.py --model {MODEL} # MODEL is the dir name under pred/ Currently it support Llama family model and Mistral model.
```

## Citation

If you find our method useful, please kindly cite our paper.

```bibtex
@article{liu2024kivi,
  title={KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache},
  author={Liu, Zirui and Yuan, Jiayi and Jin, Hongye and Zhong, Shaochen and Xu, Zhaozhuo and Braverman, Vladimir and Chen, Beidi and Hu, Xia},
  journal={arXiv preprint arXiv:2402.02750},
  year={2024}
}
```

## Contributing
We welcome contributions from the research community to improve KIVI. If you have any idea or would like to report a bug, please open an issue or submit a pull request.

## License
The code is released under the MIT License.
