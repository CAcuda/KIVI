import os
import json
import sys
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "kivi-adaptive"
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/root/KIVI"
REMOTE_CACHE = "/cache"
REMOTE_OUTPUTS = "/outputs"

app = modal.App(APP_NAME)

cache_volume = modal.Volume.from_name("kivi-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("kivi-outputs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-auth", required_keys=["HF_TOKEN"])

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "build-essential",
        "clang",
        "gcc",
        "g++",
        "python3-dev",
        "ninja-build",
    )
    .pip_install(
        "packaging==24.0",
        "sentencepiece",
        "tokenizers==0.19.1",
        "torch==2.4.1",
        "transformers==4.43.1",
        "accelerate",
        "protobuf",
        "datasets==2.19.2",
        "fastchat",
        "jieba",
        "fuzzywuzzy",
        "rouge",
        "ninja",
        "wheel",
        "setuptools",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "flash-attn==2.5.6",
        extra_options="--no-build-isolation",
        env={
            "CC": "gcc",
            "CXX": "g++",
        },
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path=REMOTE_REPO,
        copy=True,
        ignore=[
            ".git",
            "__pycache__",
            "logs",
            "outputs",
            "pred",
            "pred_e",
            "cached_models",
            ".pytest_cache",
            ".mypy_cache",
        ],
    )
    .env(
        {
            "HF_HOME": f"{REMOTE_CACHE}/hf_home",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "CC": "gcc",
            "CXX": "g++",
            "MAX_JOBS": "4",
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6",
        }
    )
    .run_commands(
        f"cd {REMOTE_REPO} && python -m pip install -e . --no-build-isolation --no-deps",
        f"cd {REMOTE_REPO}/quant && python -m pip install . --no-build-isolation --no-deps",
    )
)


def _run(cmd: list[str], cwd: str | None = None) -> str:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    return_code = process.wait()
    output = "".join(output_lines)
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}: {' '.join(cmd)}\n{output}"
        )
    return output


def _copy_if_exists(src: str, dst_dir: str) -> None:
    if os.path.exists(src):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))


def _compute_model_tag(model_name_or_path: str, group_size: int, residual_length: int, segment_lengths: str, k_bits_list: str, v_bits_list: str) -> str:
    cmd = [
        "python",
        "-c",
        (
            "import json; "
            "from types import SimpleNamespace; "
            "from utils.adaptive_kv import adaptive_experiment_tag, resolve_model_max_length; "
            f"model_name_or_path={model_name_or_path!r}; "
            f"group_size={group_size}; "
            f"residual_length={residual_length}; "
            f"segment_lengths={segment_lengths!r}; "
            f"k_bits_list={k_bits_list!r}; "
            f"v_bits_list={v_bits_list!r}; "
            "model_name = model_name_or_path.split('/')[-1]; "
            "model2maxlen = json.load(open('config/model2maxlen.json', 'r')); "
            "_, max_length = resolve_model_max_length(model_name_or_path, model2maxlen); "
            "args = SimpleNamespace("
            "adaptive_kv=True, adaptive_policy='static_distance', "
            "adaptive_segment_lengths=segment_lengths, adaptive_k_bits=k_bits_list, adaptive_v_bits=v_bits_list, "
            "group_size=group_size, residual_length=residual_length, k_bits=2, v_bits=2"
            "); "
            "print(f'{model_name}_{max_length}_{adaptive_experiment_tag(args)}')"
        ),
    ]
    return _run(cmd, cwd=REMOTE_REPO).strip()


def _adaptive_longbench_datasets() -> list[str]:
    return [
        "triviaqa",
        "qasper",
        "trec",
        "samsum",
        "lcc",
        "repobench-p",
        "qmsum",
        "multi_news",
        "passage_retrieval_en",
    ]


def _adaptive_pred_root() -> str:
    return f"{REMOTE_OUTPUTS}/pred"


def _adaptive_task_meta_root() -> str:
    return f"{REMOTE_OUTPUTS}/task_meta"


def _adaptive_task_score_root() -> str:
    return f"{REMOTE_OUTPUTS}/task_scores"


def _score_single_longbench_dataset(dataset: str, prediction_path: str):
    if REMOTE_REPO not in sys.path:
        sys.path.insert(0, REMOTE_REPO)
    from eval_long_bench import scorer

    predictions = []
    answers = []
    lengths = []
    all_classes = None

    with open(prediction_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data["pred"])
            answers.append(data["answers"])
            all_classes = data["all_classes"]
            if "length" in data:
                lengths.append(data["length"])

    score = scorer(dataset, predictions, answers, all_classes)
    return {
        "dataset": dataset,
        "num_predictions": len(predictions),
        "score": score,
        "has_lengths": bool(lengths),
    }


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=32768,
    timeout=60 * 60,
    volumes={
        REMOTE_CACHE: cache_volume,
        REMOTE_OUTPUTS: outputs_volume,
    },
    secrets=[hf_secret],
)
def smoke_test() -> str:
    lines = []
    lines.append(
        _run(
            [
                "python",
                "-c",
                (
                    "import torch, transformers, triton, flash_attn; "
                    "import models.llama_kivi; "
                    "print('torch', torch.__version__); "
                    "print('transformers', transformers.__version__); "
                    "print('cuda', torch.cuda.is_available()); "
                    "print('device_count', torch.cuda.device_count()); "
                    "print('flash_attn', flash_attn.__version__)"
                ),
            ],
            cwd=REMOTE_REPO,
        )
    )
    cache_volume.commit()
    outputs_volume.commit()
    return "\n".join(lines)


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    memory=65536,
    timeout=12 * 60 * 60,
    volumes={
        REMOTE_CACHE: cache_volume,
        REMOTE_OUTPUTS: outputs_volume,
    },
    secrets=[hf_secret],
)
def run_adaptive_repro(
    model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
    group_size: int = 64,
    residual_length: int = 128,
    segment_lengths: str = "2048,4096,8192",
    k_bits_list: str = "8,4,2",
    v_bits_list: str = "8,4,2",
    run_long_e: bool = False,
) -> str:
    gpu_id = 0
    cmd = [
        "bash",
        "scripts/run_adaptive_repro.sh",
        str(gpu_id),
        model_name_or_path,
        str(group_size),
        str(residual_length),
        segment_lengths,
        k_bits_list,
        v_bits_list,
        "1" if run_long_e else "0",
        f"{REMOTE_CACHE}/cached_models",
    ]
    output = _run(cmd, cwd=REMOTE_REPO)

    _copy_if_exists(f"{REMOTE_REPO}/outputs/adaptive_mem_speed.json", REMOTE_OUTPUTS)
    _copy_if_exists(f"{REMOTE_REPO}/outputs/adaptive_longbench_result.json", REMOTE_OUTPUTS)
    cache_volume.commit()
    outputs_volume.commit()
    return output


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    memory=65536,
    timeout=4 * 60 * 60,
    volumes={
        REMOTE_CACHE: cache_volume,
        REMOTE_OUTPUTS: outputs_volume,
    },
    secrets=[hf_secret],
)
def run_adaptive_benchmark(
    model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
    group_size: int = 64,
    residual_length: int = 128,
    segment_lengths: str = "2048,4096,8192",
    k_bits_list: str = "8,4,2",
    v_bits_list: str = "8,4,2",
    batch_size: int = 96,
    prompt_length: int = 160,
    output_length: int = 338,
    num_repeats: int = 3,
) -> str:
    cmd = [
        "python",
        "scripts/run_adaptive_mem_speed.py",
        "--model_name_or_path",
        model_name_or_path,
        "--cache_dir",
        f"{REMOTE_CACHE}/cached_models",
        "--batch_size",
        str(batch_size),
        "--prompt_length",
        str(prompt_length),
        "--output_length",
        str(output_length),
        "--num_repeats",
        str(num_repeats),
        "--group_size",
        str(group_size),
        "--residual_length",
        str(residual_length),
        "--use_flash_attention_2",
        "false",
        "--adaptive_kv",
        "--adaptive_policy",
        "static_distance",
        "--adaptive_segment_lengths",
        segment_lengths,
        "--adaptive_k_bits",
        k_bits_list,
        "--adaptive_v_bits",
        v_bits_list,
        "--out_json",
        "outputs/adaptive_benchmark.json",
    ]
    output = _run(cmd, cwd=REMOTE_REPO)

    _copy_if_exists(f"{REMOTE_REPO}/outputs/adaptive_benchmark.json", REMOTE_OUTPUTS)
    cache_volume.commit()
    outputs_volume.commit()
    return output


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    memory=65536,
    timeout=6 * 60 * 60,
    max_containers=3,
    single_use_containers=True,
    volumes={
        REMOTE_CACHE: cache_volume,
        REMOTE_OUTPUTS: outputs_volume,
    },
    secrets=[hf_secret],
)
def run_adaptive_longbench_task(
    dataset: str,
    model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
    group_size: int = 64,
    residual_length: int = 128,
    segment_lengths: str = "2048,4096,8192",
    k_bits_list: str = "8,4,2",
    v_bits_list: str = "8,4,2",
) -> str:
    model_tag = _compute_model_tag(
        model_name_or_path=model_name_or_path,
        group_size=group_size,
        residual_length=residual_length,
        segment_lengths=segment_lengths,
        k_bits_list=k_bits_list,
        v_bits_list=v_bits_list,
    )
    pred_root = _adaptive_pred_root()
    exp_dir = f"{pred_root}/{model_tag}"
    out_path = f"{exp_dir}/{dataset}.jsonl"
    meta_dir = f"{_adaptive_task_meta_root()}/{model_tag}"
    meta_path = f"{meta_dir}/{dataset}.json"
    score_dir = f"{_adaptive_task_score_root()}/{model_tag}"
    score_path = f"{score_dir}/{dataset}.score.json"

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        score_payload = None
        if os.path.exists(score_path):
            with open(score_path, "r", encoding="utf-8") as f:
                score_payload = json.load(f)
        else:
            score_payload = _score_single_longbench_dataset(dataset, out_path)
            with open(score_path, "w", encoding="utf-8") as f:
                json.dump(score_payload, f, ensure_ascii=False, indent=2)
        payload = {
            "dataset": dataset,
            "status": "skipped_existing",
            "model_tag": model_tag,
            "prediction_path": out_path,
            "score_path": score_path if os.path.exists(score_path) else None,
            "score": score_payload,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        outputs_volume.commit()
        return json.dumps(payload, ensure_ascii=False)

    pred_cmd = [
        "python",
        "pred_long_bench_adaptive.py",
        "--model_name_or_path",
        model_name_or_path,
        "--cache_dir",
        f"{REMOTE_CACHE}/cached_models",
        "--group_size",
        str(group_size),
        "--residual_length",
        str(residual_length),
        "--use_flash_attention_2",
        "true",
        "--adaptive_kv",
        "--adaptive_policy",
        "static_distance",
        "--adaptive_segment_lengths",
        segment_lengths,
        "--adaptive_k_bits",
        k_bits_list,
        "--adaptive_v_bits",
        v_bits_list,
        "--pred_root",
        pred_root,
        "--datasets",
        dataset,
    ]
    output = _run(pred_cmd, cwd=REMOTE_REPO)
    score_payload = _score_single_longbench_dataset(dataset, out_path)
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(score_payload, f, ensure_ascii=False, indent=2)

    payload = {
        "dataset": dataset,
        "status": "completed",
        "model_tag": model_tag,
        "prediction_path": out_path,
        "score_path": score_path,
        "score": score_payload,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    outputs_volume.commit()
    cache_volume.commit()
    return output


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=10 * 60,
    volumes={
        REMOTE_CACHE: cache_volume,
        REMOTE_OUTPUTS: outputs_volume,
    },
    secrets=[hf_secret],
)
def run_adaptive_longbench(
    model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
    group_size: int = 64,
    residual_length: int = 128,
    segment_lengths: str = "2048,4096,8192",
    k_bits_list: str = "8,4,2",
    v_bits_list: str = "8,4,2",
) -> str:
    model_tag = _compute_model_tag(
        model_name_or_path=model_name_or_path,
        group_size=group_size,
        residual_length=residual_length,
        segment_lengths=segment_lengths,
        k_bits_list=k_bits_list,
        v_bits_list=v_bits_list,
    )
    pred_root = _adaptive_pred_root()
    exp_dir = f"{pred_root}/{model_tag}"
    os.makedirs(exp_dir, exist_ok=True)
    datasets = _adaptive_longbench_datasets()
    missing_datasets = [
        dataset
        for dataset in datasets
        if not (os.path.exists(f"{exp_dir}/{dataset}.jsonl") and os.path.getsize(f"{exp_dir}/{dataset}.jsonl") > 0)
    ]

    launch_manifest = {
        "model_tag": model_tag,
        "pred_root": pred_root,
        "datasets": datasets,
        "missing_datasets": missing_datasets,
        "launched_calls": [],
    }

    if not missing_datasets:
        eval_cmd = [
            "python",
            "eval_long_bench.py",
            "--model",
            model_tag,
            "--pred_root",
            pred_root,
        ]
        output = _run(eval_cmd, cwd=REMOTE_REPO)
        _copy_if_exists(f"{exp_dir}/result.json", REMOTE_OUTPUTS)
        cache_volume.commit()
        outputs_volume.commit()
        return output

    for dataset in missing_datasets:
        call = run_adaptive_longbench_task.spawn(
            dataset=dataset,
            model_name_or_path=model_name_or_path,
            group_size=group_size,
            residual_length=residual_length,
            segment_lengths=segment_lengths,
            k_bits_list=k_bits_list,
            v_bits_list=v_bits_list,
        )
        launch_manifest["launched_calls"].append(
            {
                "dataset": dataset,
                "function_call_id": call.object_id,
                "dashboard_url": call.get_dashboard_url(),
            }
        )

    manifest_path = f"{REMOTE_OUTPUTS}/longbench_launch_{model_tag}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(launch_manifest, f, ensure_ascii=False, indent=2)
    outputs_volume.commit()

    return json.dumps(launch_manifest, ensure_ascii=False, indent=2)


@app.local_entrypoint()
def main(
    action: str = "smoke",
    model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
    group_size: int = 64,
    residual_length: int = 128,
    segment_lengths: str = "2048,4096,8192",
    k_bits_list: str = "8,4,2",
    v_bits_list: str = "8,4,2",
    run_long_e: bool = False,
    batch_size: int = 96,
    prompt_length: int = 160,
    output_length: int = 338,
    num_repeats: int = 3,
):
    if action == "smoke":
        print(smoke_test.remote())
        return
    if action == "benchmark":
        print(
            run_adaptive_benchmark.remote(
                model_name_or_path=model_name_or_path,
                group_size=group_size,
                residual_length=residual_length,
                segment_lengths=segment_lengths,
                k_bits_list=k_bits_list,
                v_bits_list=v_bits_list,
                batch_size=batch_size,
                prompt_length=prompt_length,
                output_length=output_length,
                num_repeats=num_repeats,
            )
        )
        return
    if action == "longbench":
        print(
            run_adaptive_longbench.remote(
                model_name_or_path=model_name_or_path,
                group_size=group_size,
                residual_length=residual_length,
                segment_lengths=segment_lengths,
                k_bits_list=k_bits_list,
                v_bits_list=v_bits_list,
            )
        )
        return
    if action == "repro":
        print(
            run_adaptive_repro.remote(
                model_name_or_path=model_name_or_path,
                group_size=group_size,
                residual_length=residual_length,
                segment_lengths=segment_lengths,
                k_bits_list=k_bits_list,
                v_bits_list=v_bits_list,
                run_long_e=run_long_e,
            )
        )
        return
    raise ValueError(f"Unsupported action: {action}")
