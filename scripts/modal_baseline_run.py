import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "kivi-baseline"
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
        "python3-dev",
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
        "wheel",
        "setuptools",
        extra_index_url="https://download.pytorch.org/whl/cu121",
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
        }
    )
    .run_commands(
        f"cd {REMOTE_REPO} && python -m pip install -e . --no-build-isolation --no-deps",
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
    src_path = Path(src)
    if src_path.exists():
        dst_dir_path = Path(dst_dir)
        dst_dir_path.mkdir(parents=True, exist_ok=True)
        if src_path.is_dir():
            target = dst_dir_path / src_path.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src_path, target)
        else:
            shutil.copy2(src_path, dst_dir_path / src_path.name)


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    memory=65536,
    timeout=6 * 60 * 60,
    volumes={
        REMOTE_CACHE: cache_volume,
        REMOTE_OUTPUTS: outputs_volume,
    },
    secrets=[hf_secret],
)
def run_baseline_passage_retrieval_en(
    model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> str:
    pred_root = f"{REMOTE_OUTPUTS}/baseline_single/pred"
    cmd = [
        "python",
        "scripts/run_baseline_single_task.py",
        "--model_name_or_path",
        model_name_or_path,
        "--dataset",
        "passage_retrieval_en",
        "--cache_dir",
        f"{REMOTE_CACHE}/cached_models",
        "--use_flash_attention_2",
        "false",
        "--pred_root",
        pred_root,
    ]
    output = _run(cmd, cwd=REMOTE_REPO)
    cache_volume.commit()
    outputs_volume.commit()
    return output
