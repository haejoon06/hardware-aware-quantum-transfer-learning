from __future__ import annotations

import csv
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# Keep direct module consumers deterministic as well as the main CLI. This
# assignment must happen before importing torch/CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import yaml

from .config import ROOT, canonical_json, sha256_bytes, sha256_file


DIRECTORIES = (
    "article",
    "configs/datasets",
    "configs/reducers",
    "configs/heads",
    "data/raw",
    "data/manifests",
    "data/audit",
    "data/embeddings",
    "environment",
    "frozen",
    "logs",
    "models",
    "notebooks",
    "results/figures",
    "results/metrics",
    "results/predictions",
    "results/statistics",
    "results/tables",
    "runs",
    "state/tasks",
)


def ensure_directories() -> None:
    for rel in DIRECTORIES:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", delete=False, dir=target.parent
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, target)


def atomic_write_json(path: str | Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))


def atomic_write_yaml(path: str | Path, data: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=True, allow_unicode=True))


def atomic_save_numpy(path: str | Path, array: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, array)
    os.replace(temp, target)


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, target)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def stable_subset_indices(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in np.unique(y):
        candidates = np.flatnonzero(y == label)
        if len(candidates) < per_class:
            raise ValueError(
                f"Class {label} has {len(candidates)} examples, fewer than required {per_class}."
            )
        selected.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def task_sentinel(task_id: str) -> Path:
    safe = task_id.replace("/", "__").replace("\\", "__")
    return ROOT / "state" / "tasks" / f"{safe}.json"


def task_complete(task_id: str, expected_hash: str | None = None) -> bool:
    path = task_sentinel(task_id)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("status") != "complete":
        return False
    return expected_hash is None or payload.get("input_hash") == expected_hash


def mark_task(task_id: str, status: str, **payload: Any) -> None:
    data = {"task_id": task_id, "status": status, "updated_at": utc_now(), **payload}
    atomic_write_json(task_sentinel(task_id), data)


@contextmanager
def task_context(task_id: str, input_hash: str | None = None) -> Iterator[None]:
    mark_task(task_id, "running", input_hash=input_hash, pid=os.getpid())
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        mark_task(
            task_id,
            "failed",
            input_hash=input_hash,
            error=repr(exc),
            elapsed_seconds=time.perf_counter() - started,
        )
        raise
    else:
        mark_task(
            task_id,
            "complete",
            input_hash=input_hash,
            elapsed_seconds=time.perf_counter() - started,
        )


def dataframe_hash(frame: Any, columns: list[str] | None = None) -> str:
    if columns:
        frame = frame[columns]
    csv_bytes = frame.sort_values(list(frame.columns)).to_csv(index=False).encode("utf-8")
    return sha256_bytes(csv_bytes)


def run_capture(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"command": command, "error": repr(exc)}


def capture_environment() -> dict[str, Any]:
    import importlib.metadata as metadata

    packages = {}
    for name in (
        "torch",
        "torchvision",
        "pennylane",
        "pennylane-lightning",
        "numpy",
        "scikit-learn",
        "scipy",
        "pandas",
        "medmnist",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "captured_at": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_names": gpu_names,
        "nvidia_smi": run_capture(["nvidia-smi"]),
        "nvcc": run_capture(["nvcc", "--version"]),
    }


def write_csv_rows(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, target)

