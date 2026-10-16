from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch import nn

A10 = Path(__file__).resolve().parents[2]
ROOT = A10
_default_upstream = A10.parent if (A10.parent / "data" / "embeddings").exists() else A10 / "upstream_article09"
UPSTREAM = Path(os.environ.get("ARTICLE09_ROOT", str(_default_upstream))).resolve()
if (UPSTREAM / "src").exists() and str(UPSTREAM / "src") not in sys.path:
    # Fitted SLP-6 joblib bundles retain their Article 9 class module names.
    # Loading them is therefore an explicit code-and-artifact dependency.
    sys.path.insert(0, str(UPSTREAM / "src"))
CONFIG_PATH = A10 / "configs" / "protocol.yaml"
STATE = A10 / "state"
RUNS = A10 / "runs"
RESULTS = A10 / "results"
LOGS = A10 / "logs"


def stable_subset_indices(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in np.unique(y):
        candidates = np.flatnonzero(y == label)
        if len(candidates) < per_class:
            raise ValueError(f"Class {label} has {len(candidates)} samples; {per_class} required")
        selected.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def load_embeddings(dataset: str, split: str, allow_test: bool = False):
    if split == "test" and not allow_test:
        raise PermissionError("Test embeddings require the post-freeze evaluation path")
    root = UPSTREAM / "data" / "embeddings" / dataset
    x = np.load(root / f"{split}_X.npy", mmap_mode="r")
    ids = pd.read_csv(root / f"{split}_ids.csv")
    y = None if split == "test" else np.load(root / f"{split}_y.npy")
    return x, y, ids


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences > left) & (confidences <= right)
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(value)


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, ece_bins: int = 15) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(probabilities.shape[1])
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[y_true]
    try:
        y_binary = label_binarize(y_true, classes=labels)
        auc = float(roc_auc_score(y_binary, probabilities, average="macro", multi_class="ovr"))
    except ValueError:
        auc = float("nan")
    return {
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "macro_precision": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_roc_auc_ovr": auc,
        "nll": float(log_loss(y_true, probabilities, labels=labels)),
        "ece": expected_calibration_error(y_true, probabilities, bins=ece_bins),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Parallel dataset workers can update independent status files at the same
    # time and also touch shared sentinels during setup. A process-specific
    # temporary name prevents Windows from raising PermissionError when two
    # workers otherwise contend for the same ``*.tmp`` path.
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_text(value, encoding="utf-8")
        for attempt in range(40):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.025 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["_hash"] = sha256_file(CONFIG_PATH)
    return config


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def split_hash(dataset: str) -> str:
    return sha256_file(UPSTREAM / "data" / "manifests" / f"{dataset}.csv")


def locked_test_hash(dataset: str) -> str:
    return sha256_file(UPSTREAM / "data" / "manifests" / "locked" / f"{dataset}_test.csv")


def upstream_shared_dir(dataset: str, seed: int) -> Path:
    return UPSTREAM / "runs" / "shared" / f"A09-{dataset.upper()}-SLP6-SHARED-S{seed}"


def upstream_hashes(dataset: str, seed: int) -> dict[str, str]:
    folder = upstream_shared_dir(dataset, seed)
    return {
        "manifest": split_hash(dataset),
        "reducer": sha256_file(folder / "reducer.joblib"),
        "angle_scaler": sha256_file(folder / "angle_scaler.joblib"),
    }


def _subset(x: np.ndarray, y: np.ndarray, n_per_class: int, seed: int):
    indices = stable_subset_indices(y, n_per_class, seed)
    return np.asarray(x)[indices], np.asarray(y)[indices], indices


def load_reduced(
    dataset: str,
    seed: int,
    config: dict[str, Any],
    include_test: bool = False,
) -> dict[str, np.ndarray]:
    if include_test and not (STATE / "FINAL_SELECTION_FROZEN.json").exists():
        raise RuntimeError("Test access refused: final selection has not been frozen.")
    train_x, train_y, _ = load_embeddings(dataset, "train")
    val_x, val_y, _ = load_embeddings(dataset, "val")
    train_x, train_y, _ = _subset(
        np.asarray(train_x),
        np.asarray(train_y),
        int(config["project"]["train_per_class"]),
        seed,
    )
    val_x, val_y, _ = _subset(
        np.asarray(val_x),
        np.asarray(val_y),
        int(config["project"]["val_per_class"]),
        int(config["project"]["validation_seed"]),
    )
    shared = upstream_shared_dir(dataset, seed)
    reducer = joblib.load(shared / "reducer.joblib")
    scaler = joblib.load(shared / "angle_scaler.joblib")

    def transform(values: np.ndarray) -> np.ndarray:
        return scaler.transform(reducer.transform(values)).astype(np.float32)

    arrays = {
        "train_x": transform(train_x),
        "train_y": train_y.astype(np.int64),
        "val_x": transform(val_x),
        "val_y": val_y.astype(np.int64),
    }
    if include_test:
        frozen_path = STATE / "FINAL_SELECTION_FROZEN.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        expected_test_hash = frozen["test_manifest_hashes"][dataset]
        actual_test_hash = locked_test_hash(dataset)
        if actual_test_hash != expected_test_hash:
            raise PermissionError(f"Locked test manifest changed for {dataset}")
        test_x, _, test_ids = load_embeddings(dataset, "test", allow_test=True)
        locked = pd.read_csv(
            UPSTREAM / "data" / "manifests" / "locked" / f"{dataset}_test.csv"
        ).set_index("sample_id")
        test_y = locked.loc[test_ids.sample_id, "label"].to_numpy(dtype=np.int64)
        arrays["test_x"] = transform(np.asarray(test_x))
        arrays["test_y"] = test_y
    return arrays


@dataclass(frozen=True)
class CircuitSpec:
    qubits: int
    depth: int
    encoding: str
    entanglement: str

    @property
    def slug(self) -> str:
        return f"q{self.qubits}-d{self.depth}-{self.encoding}-{self.entanglement}"


class QuantumHead(nn.Module):
    def __init__(self, classes: int, spec: CircuitSpec, backend: str = "default.qubit"):
        super().__init__()
        self.classes = classes
        self.spec = spec
        self.backend = backend
        self.theta = nn.Parameter(0.05 * torch.randn(spec.depth, spec.qubits, 3))
        self.output = nn.Linear(spec.qubits, classes, bias=True)
        self._build_qnode()

    def _encode(self, inputs):
        # All six SLP coordinates are retained. For q<6, the last coordinates are
        # sequential rotations on reused wires; for q>6, additional wires start at |0>.
        for feature in range(6):
            qml.RY(inputs[..., feature], wires=feature % self.spec.qubits)

    def _entangle(self):
        q = self.spec.qubits
        for wire in range(q - 1):
            qml.CNOT(wires=[wire, wire + 1])
        if self.spec.entanglement == "ring" and q > 2:
            qml.CNOT(wires=[q - 1, 0])

    def _build_qnode(self) -> None:
        dev = qml.device(self.backend, wires=self.spec.qubits, shots=None)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, theta: torch.Tensor):
            if self.spec.encoding == "once":
                self._encode(inputs)
            for layer in range(self.spec.depth):
                if self.spec.encoding == "reupload":
                    self._encode(inputs)
                for wire in range(self.spec.qubits):
                    qml.Rot(
                        theta[layer, wire, 0],
                        theta[layer, wire, 1],
                        theta[layer, wire, 2],
                        wires=wire,
                    )
                self._entangle()
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(self.spec.qubits))

        self.circuit = circuit

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        values = self.circuit(x, self.theta)
        if isinstance(values, (tuple, list)):
            values = torch.stack(list(values), dim=-1)
        elif values.ndim == 2 and values.shape[0] == self.spec.qubits:
            values = values.transpose(0, 1)
        return values.to(dtype=self.output.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.quantum_features(x))


def _predict(model: QuantumHead, x: np.ndarray, batch_size: int = 256):
    model.eval()
    all_features: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(np.asarray(x[start : start + batch_size], dtype=np.float32))
            features = model.quantum_features(batch)
            probabilities = torch.softmax(model.output(features), dim=1)
            all_features.append(features.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
    return np.concatenate(all_features), np.concatenate(all_probabilities)


def _run_hash(dataset: str, seed: int, spec: CircuitSpec, config: dict[str, Any]) -> str:
    return sha256_json(
        {
            "protocol": config["_hash"],
            "dataset": dataset,
            "seed": seed,
            "spec": asdict(spec),
            "upstream": upstream_hashes(dataset, seed),
        }
    )


def _run_dir(stage: str, dataset: str, seed: int, spec: CircuitSpec) -> Path:
    return RUNS / stage / dataset / spec.slug / f"seed_{seed}"


def fit_run(
    stage: str,
    dataset: str,
    seed: int,
    spec: CircuitSpec,
    config: dict[str, Any],
    resume: bool,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    folder = _run_dir(stage, dataset, seed, spec)
    folder.mkdir(parents=True, exist_ok=True)
    run_hash = _run_hash(dataset, seed, spec, config)
    success = folder / "SUCCESS.json"
    if resume and success.exists():
        record = json.loads(success.read_text(encoding="utf-8"))
        if record.get("run_hash") != run_hash:
            raise RuntimeError(f"Hash mismatch for completed run {folder}")
        return record
    if (folder / "run_hash.txt").exists():
        old = (folder / "run_hash.txt").read_text(encoding="utf-8").strip()
        if old != run_hash:
            raise RuntimeError(f"Refusing resume with changed inputs: {folder}")
    atomic_text(folder / "run_hash.txt", run_hash + "\n")
    atomic_json(folder / "config.json", {"dataset": dataset, "seed": seed, "spec": asdict(spec)})
    arrays = load_reduced(dataset, seed, config)
    classes = int(max(arrays["train_y"].max(), arrays["val_y"].max()) + 1)
    set_seed(seed)
    model = QuantumHead(classes, spec, str(config["training"]["backend"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    checkpoint = folder / "checkpoint_latest.pt"
    best_state = None
    best_f1 = -math.inf
    best_nll = math.inf
    start_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float]] = []
    elapsed_before = 0.0
    if resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("run_hash") != run_hash:
            raise RuntimeError(f"Checkpoint hash mismatch: {folder}")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        best_state = state["best_state"]
        best_f1 = float(state["best_f1"])
        best_nll = float(state["best_nll"])
        start_epoch = int(state["epoch"]) + 1
        bad_epochs = int(state["bad_epochs"])
        history = list(state["history"])
        elapsed_before = float(state.get("elapsed_seconds", 0.0))

    train_x = torch.from_numpy(arrays["train_x"])
    train_y = torch.from_numpy(arrays["train_y"])
    epochs = int(max_epochs or config["training"]["max_epochs"])
    batch_size = int(config["training"]["batch_size"])
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        model.train()
        order = np.random.default_rng(seed * 100_000 + epoch).permutation(len(train_y))
        losses, norms, variances = [], [], []
        for offset in range(0, len(order), batch_size):
            index = torch.as_tensor(order[offset : offset + batch_size])
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x[index])
            loss = nn.functional.cross_entropy(logits, train_y[index])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {folder} epoch {epoch}")
            loss.backward()
            gradients = torch.cat(
                [parameter.grad.detach().reshape(-1) for parameter in model.parameters() if parameter.grad is not None]
            )
            norms.append(float(torch.linalg.vector_norm(gradients)))
            variances.append(float(torch.var(gradients, unbiased=False)))
            nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        _, val_probabilities = _predict(model, arrays["val_x"])
        metrics = classification_metrics(
            arrays["val_y"], val_probabilities, int(config["statistics"]["ece_bins"])
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "gradient_norm_mean": float(np.mean(norms)),
            "gradient_variance_mean": float(np.mean(variances)),
            "val_macro_f1": float(metrics["macro_f1"]),
            "val_nll": float(metrics["nll"]),
        }
        history.append(row)
        improved = metrics["macro_f1"] > best_f1 + 1e-12 or (
            abs(metrics["macro_f1"] - best_f1) <= 1e-12 and metrics["nll"] < best_nll
        )
        if improved:
            best_f1 = float(metrics["macro_f1"])
            best_nll = float(metrics["nll"])
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        elapsed = elapsed_before + time.perf_counter() - started
        torch.save(
            {
                "run_hash": run_hash,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state,
                "best_f1": best_f1,
                "best_nll": best_nll,
                "bad_epochs": bad_epochs,
                "history": history,
                "elapsed_seconds": elapsed,
                "classes": classes,
                "spec": asdict(spec),
            },
            checkpoint,
        )
        pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
        if bad_epochs >= int(config["training"]["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"No valid state produced by {folder}")
    model.load_state_dict(best_state)
    features, probabilities = _predict(model, arrays["val_x"])
    metrics = classification_metrics(arrays["val_y"], probabilities, int(config["statistics"]["ece_bins"]))
    torch.save(
        {"model_state": best_state, "classes": classes, "spec": asdict(spec), "run_hash": run_hash},
        folder / "checkpoint_best.pt",
    )
    np.save(folder / "validation_quantum_features.npy", features)
    np.save(folder / "validation_probabilities.npy", probabilities)
    atomic_json(folder / "validation_metrics.json", metrics)
    record = {
        "status": "success",
        "completed_at": utc_now(),
        "run_hash": run_hash,
        "stage": stage,
        "dataset": dataset,
        "seed": seed,
        **asdict(spec),
        "epochs_completed": len(history),
        "fit_seconds": elapsed_before + time.perf_counter() - started,
        **{f"val_{key}": value for key, value in metrics.items()},
    }
    atomic_json(success, record)
    return record


def load_model(folder: Path, config: dict[str, Any]) -> QuantumHead:
    checkpoint = torch.load(folder / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    spec = CircuitSpec(**checkpoint["spec"])
    model = QuantumHead(int(checkpoint["classes"]), spec, str(config["training"]["backend"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def collect_stage(stage: str, dataset: str | None = None) -> pd.DataFrame:
    base = RUNS / stage
    records: list[dict[str, Any]] = []
    if not base.exists():
        return pd.DataFrame()
    paths = base.glob(f"{dataset}/**/SUCCESS.json") if dataset else base.glob("**/SUCCESS.json")
    for path in paths:
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(records)


def aggregate_stage(stage: str) -> pd.DataFrame:
    frame = collect_stage(stage)
    if frame.empty:
        return frame
    keys = ["dataset", "qubits", "depth", "encoding", "entanglement"]
    grouped = frame.groupby(keys, as_index=False).agg(
        n=("seed", "count"),
        macro_f1_mean=("val_macro_f1", "mean"),
        macro_f1_std=("val_macro_f1", "std"),
        nll_mean=("val_nll", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"),
        epochs_mean=("epochs_completed", "mean"),
    )
    output = RESULTS / "tables" / f"{stage}_validation_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output, index=False)
    return grouped


def _screen_specs(config: dict[str, Any]) -> list[CircuitSpec]:
    return [
        CircuitSpec(int(qubits), int(depth), str(config["architecture"]["screening_encoding"]), str(config["architecture"]["screening_entanglement"]))
        for qubits in config["architecture"]["qubits"]
        for depth in config["architecture"]["depths"]
    ]


def run_setup(config: dict[str, Any]) -> None:
    for folder in (STATE, RUNS, RESULTS / "tables", RESULTS / "figures", LOGS, A10 / "notebooks", A10 / "article"):
        folder.mkdir(parents=True, exist_ok=True)
    required = [UPSTREAM / "frozen" / "protocol_v1.json", UPSTREAM / "results" / "metrics" / "all_validation_metrics.csv"]
    for dataset in [*config["project"]["primary_datasets"], config["project"]["confirmatory_dataset"]]:
        required.extend(
            [
                UPSTREAM / "data" / "manifests" / f"{dataset}.csv",
                UPSTREAM / "data" / "manifests" / "locked" / f"{dataset}_test.csv",
            ]
        )
        for seed in config["project"]["final_seeds"]:
            required.extend(
                [
                    upstream_shared_dir(dataset, int(seed)) / "reducer.joblib",
                    upstream_shared_dir(dataset, int(seed)) / "angle_scaler.joblib",
                ]
            )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing upstream artifacts:\n" + "\n".join(missing))
    lock = {
        "created_at": utc_now(),
        "protocol_sha256": config["_hash"],
        "upstream_protocol_sha256": sha256_file(UPSTREAM / "frozen" / "protocol_v1.json"),
        "split_hashes": {name: split_hash(name) for name in [*config["project"]["primary_datasets"], config["project"]["confirmatory_dataset"]]},
        "locked_test_hashes": {name: locked_test_hash(name) for name in [*config["project"]["primary_datasets"], config["project"]["confirmatory_dataset"]]},
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "pennylane": qml.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "real_hardware_enabled": bool(config["execution"]["real_hardware_enabled"]),
    }
    frozen = STATE / "PROTOCOL_LOCK.json"
    if frozen.exists():
        old = json.loads(frozen.read_text(encoding="utf-8"))
        for key in ("protocol_sha256", "upstream_protocol_sha256", "split_hashes", "locked_test_hashes"):
            if old.get(key) != lock.get(key):
                raise RuntimeError(f"Protocol lock changed at {key}; use a new experiment directory.")
    else:
        atomic_json(frozen, lock)
    setup_marker = STATE / "SETUP_SUCCESS"
    if not setup_marker.exists():
        atomic_text(setup_marker, utc_now() + "\n")


def run_smoke(config: dict[str, Any], resume: bool) -> None:
    spec = CircuitSpec(4, 1, "once", "linear")
    record = fit_run("smoke", "cifar10", 11, spec, config, resume, max_epochs=1)
    folder = _run_dir("smoke", "cifar10", 11, spec)
    history = pd.read_csv(folder / "training_history.csv")
    if not np.isfinite(history[["train_loss", "gradient_norm_mean", "gradient_variance_mean"]].to_numpy()).all():
        raise AssertionError("Smoke run produced non-finite diagnostics")
    if not (folder / "checkpoint_latest.pt").exists():
        raise AssertionError("Smoke checkpoint missing")
    atomic_json(STATE / "SMOKE_SUCCESS.json", record)


def run_screen_dataset(dataset: str, config: dict[str, Any], resume: bool, max_epochs: int | None) -> None:
    total = len(_screen_specs(config)) * len(config["project"]["screening_seeds"])
    completed = 0
    for spec in _screen_specs(config):
        for seed in config["project"]["screening_seeds"]:
            fit_run("screen", dataset, int(seed), spec, config, resume, max_epochs)
            completed += 1
            atomic_json(STATE / f"screen_{dataset}.json", {"status": "running", "completed": completed, "total": total, "updated_at": utc_now()})
    aggregate_stage("screen")
    atomic_json(STATE / f"SCREEN_{dataset.upper()}_SUCCESS.json", {"completed": total, "completed_at": utc_now()})


def top_screen_specs(dataset: str, config: dict[str, Any]) -> list[CircuitSpec]:
    summary = aggregate_stage("screen")
    candidates = summary[(summary.dataset == dataset) & (summary.n == len(config["project"]["screening_seeds"]))].copy()
    if candidates.empty:
        raise RuntimeError(f"No complete screening candidates for {dataset}")
    candidates = candidates.sort_values(["macro_f1_mean", "nll_mean"], ascending=[False, True])
    return [
        CircuitSpec(int(row.qubits), int(row.depth), str(row.encoding), str(row.entanglement))
        for row in candidates.head(int(config["architecture"]["ablation_top_k"])).itertuples()
    ]


def run_ablate_dataset(dataset: str, config: dict[str, Any], resume: bool, max_epochs: int | None) -> None:
    bases = top_screen_specs(dataset, config)
    specs = {
        CircuitSpec(base.qubits, base.depth, str(encoding), str(entanglement))
        for base in bases
        for encoding in config["architecture"]["ablation_encodings"]
        for entanglement in config["architecture"]["ablation_entanglements"]
    }
    total = len(specs) * len(config["project"]["screening_seeds"])
    completed = 0
    for spec in sorted(specs, key=lambda value: value.slug):
        for seed in config["project"]["screening_seeds"]:
            fit_run("ablate", dataset, int(seed), spec, config, resume, max_epochs)
            completed += 1
            atomic_json(STATE / f"ablate_{dataset}.json", {"status": "running", "completed": completed, "total": total, "updated_at": utc_now()})
    aggregate_stage("ablate")
    atomic_json(STATE / f"ABLATE_{dataset.upper()}_SUCCESS.json", {"completed": total, "completed_at": utc_now()})


def _candidate_summary(config: dict[str, Any]) -> pd.DataFrame:
    frames = [frame for frame in (collect_stage("screen"), collect_stage("ablate")) if not frame.empty]
    all_runs = pd.concat(frames, ignore_index=True)
    keys = ["dataset", "qubits", "depth", "encoding", "entanglement"]
    return all_runs.groupby(keys, as_index=False).agg(
        n=("seed", "nunique"),
        macro_f1_mean=("val_macro_f1", "mean"),
        macro_f1_std=("val_macro_f1", "std"),
        nll_mean=("val_nll", "mean"),
    )


def freeze_selection(config: dict[str, Any]) -> dict[str, Any]:
    output = STATE / "FINAL_SELECTION_FROZEN.json"
    if output.exists():
        frozen = json.loads(output.read_text(encoding="utf-8"))
        if frozen.get("protocol_sha256") != config["_hash"]:
            raise RuntimeError("Frozen selection belongs to a different protocol")
        return frozen
    summary = _candidate_summary(config)
    needed = len(config["project"]["screening_seeds"])
    selections: dict[str, Any] = {}
    for dataset in config["project"]["primary_datasets"]:
        rows = summary[(summary.dataset == dataset) & (summary.n == needed)].sort_values(
            ["macro_f1_mean", "nll_mean"], ascending=[False, True]
        )
        if rows.empty:
            raise RuntimeError(f"No complete candidates to freeze for {dataset}")
        row = rows.iloc[0]
        selections[dataset] = {
            "spec": {
                "qubits": int(row.qubits),
                "depth": int(row.depth),
                "encoding": str(row.encoding),
                "entanglement": str(row.entanglement),
            },
            "validation_macro_f1_mean": float(row.macro_f1_mean),
            "validation_macro_f1_std": float(row.macro_f1_std),
            "validation_nll_mean": float(row.nll_mean),
        }
    # Robust common configuration for the confirmatory domain, selected by mean rank.
    common = summary[summary.n == needed].copy()
    common["rank"] = common.groupby("dataset")["macro_f1_mean"].rank(ascending=False, method="average")
    common_keys = ["qubits", "depth", "encoding", "entanglement"]
    common = common.groupby(common_keys, as_index=False).agg(mean_rank=("rank", "mean"), domains=("dataset", "nunique"))
    common = common[common.domains == len(config["project"]["primary_datasets"])].sort_values("mean_rank")
    robust = common.iloc[0]
    frozen = {
        "frozen_at": utc_now(),
        "protocol_sha256": config["_hash"],
        "selection_uses_test": False,
        "primary": selections,
        "confirmatory_robust_spec": {
            "qubits": int(robust.qubits),
            "depth": int(robust.depth),
            "encoding": str(robust.encoding),
            "entanglement": str(robust.entanglement),
        },
        "test_manifest_hashes": {name: locked_test_hash(name) for name in [*config["project"]["primary_datasets"], config["project"]["confirmatory_dataset"]]},
        "candidate_summary_sha256": sha256_json(summary.to_dict(orient="records")),
    }
    atomic_json(output, frozen)
    summary.to_csv(RESULTS / "tables" / "all_validation_candidates.csv", index=False)
    return frozen


def run_final_dataset(dataset: str, config: dict[str, Any], resume: bool, max_epochs: int | None) -> None:
    frozen = freeze_selection(config)
    spec = CircuitSpec(**frozen["primary"][dataset]["spec"])
    for index, seed in enumerate(config["project"]["final_seeds"], start=1):
        fit_run("final", dataset, int(seed), spec, config, resume, max_epochs)
        atomic_json(STATE / f"final_{dataset}.json", {"status": "running", "completed": index, "total": len(config["project"]["final_seeds"]), "updated_at": utc_now()})
    aggregate_stage("final")
    atomic_json(STATE / f"FINAL_{dataset.upper()}_SUCCESS.json", {"completed_at": utc_now(), "spec": asdict(spec)})


def _metric_record(y: np.ndarray, probabilities: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    return classification_metrics(y, probabilities, int(config["statistics"]["ece_bins"]))


def evaluate_shots_dataset(
    dataset: str,
    config: dict[str, Any],
    resume: bool,
    confirmatory: bool = False,
) -> None:
    frozen = freeze_selection(config)
    if confirmatory:
        spec = CircuitSpec(**frozen["confirmatory_robust_spec"])
        stage = "confirm"
        seeds = config["project"]["screening_seeds"]
        selection = frozen["confirmatory_robust_spec"]
    else:
        spec = CircuitSpec(**frozen["primary"][dataset]["spec"])
        stage = "final"
        seeds = config["project"]["final_seeds"]
        selection = frozen["primary"][dataset]
    output = RESULTS / "shots" / dataset
    output.mkdir(parents=True, exist_ok=True)
    success = output / "SUCCESS.json"
    stage_hash = sha256_json({"protocol": config["_hash"], "selection": selection, "test": locked_test_hash(dataset), "shots": config["finite_shots"], "confirmatory": confirmatory})
    if resume and success.exists() and json.loads(success.read_text())["stage_hash"] == stage_hash:
        return
    records: list[dict[str, Any]] = []
    for seed in seeds:
        arrays = load_reduced(dataset, int(seed), config, include_test=True)
        folder = _run_dir(stage, dataset, int(seed), spec)
        model = load_model(folder, config)
        features, analytic = _predict(model, arrays["test_x"])
        base_metrics = _metric_record(arrays["test_y"], analytic, config)
        records.append({"dataset": dataset, "seed": seed, "shots": "analytic", "repetition": 0, "probability_drift": 0.0, "label_flip_rate": 0.0, **base_metrics})
        weight = model.output.weight.detach().numpy()
        bias = model.output.bias.detach().numpy()
        analytic_labels = analytic.argmax(axis=1)
        for shots in config["finite_shots"]["levels"]:
            for repetition in range(int(config["finite_shots"]["repetitions"])):
                rng = np.random.default_rng(int(seed) * 1_000_000 + int(shots) * 100 + repetition)
                successes = rng.binomial(int(shots), np.clip((features + 1.0) / 2.0, 0.0, 1.0))
                noisy_features = 2.0 * successes / int(shots) - 1.0
                probabilities = softmax_numpy(noisy_features @ weight.T + bias)
                metrics = _metric_record(arrays["test_y"], probabilities, config)
                records.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "shots": int(shots),
                        "repetition": repetition,
                        "probability_drift": float(np.mean(np.abs(probabilities - analytic))),
                        "label_flip_rate": float(np.mean(probabilities.argmax(axis=1) != analytic_labels)),
                        **metrics,
                    }
                )
        pd.DataFrame(records).to_csv(output / "shot_metrics_partial.csv", index=False)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "shot_metrics.csv", index=False)
    atomic_json(success, {"status": "success", "stage_hash": stage_hash, "completed_at": utc_now(), "rows": len(frame)})


def _fixed_stratified(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    return stable_subset_indices(np.asarray(y), int(per_class), int(seed))


def _qiskit_circuit(spec: CircuitSpec, values: np.ndarray, theta: np.ndarray, measure: bool = True):
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(spec.qubits, spec.qubits if measure else 0)

    def encode():
        for feature in range(6):
            circuit.ry(float(values[feature]), feature % spec.qubits)

    if spec.encoding == "once":
        encode()
    for layer in range(spec.depth):
        if spec.encoding == "reupload":
            encode()
        for wire in range(spec.qubits):
            # PennyLane Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)
            circuit.rz(float(theta[layer, wire, 0]), wire)
            circuit.ry(float(theta[layer, wire, 1]), wire)
            circuit.rz(float(theta[layer, wire, 2]), wire)
        for wire in range(spec.qubits - 1):
            circuit.cx(wire, wire + 1)
        if spec.entanglement == "ring" and spec.qubits > 2:
            circuit.cx(spec.qubits - 1, 0)
    if measure:
        circuit.measure(range(spec.qubits), range(spec.qubits))
    return circuit


def _ring_coupling(qubits: int):
    from qiskit.transpiler import CouplingMap

    edges = []
    for left in range(qubits - 1):
        edges.extend([(left, left + 1), (left + 1, left)])
    if qubits > 2:
        edges.extend([(qubits - 1, 0), (0, qubits - 1)])
    return CouplingMap(edges)


def _aer_quantum_features(
    x: np.ndarray,
    spec: CircuitSpec,
    theta: np.ndarray,
    profile: dict[str, float],
    shots: int,
    seed: int,
) -> np.ndarray:
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    noise = NoiseModel()
    one = depolarizing_error(float(profile["one_qubit"]), 1)
    two = depolarizing_error(float(profile["two_qubit"]), 2)
    noise.add_all_qubit_quantum_error(one, ["ry", "rz", "sx", "x"])
    noise.add_all_qubit_quantum_error(two, ["cx"])
    readout = float(profile["readout"])
    noise.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    backend = AerSimulator(noise_model=noise)
    outputs = np.empty((len(x), spec.qubits), dtype=np.float32)
    for offset in range(0, len(x), 100):
        batch = [_qiskit_circuit(spec, row, theta, measure=True) for row in x[offset : offset + 100]]
        compiled = transpile(
            batch,
            backend=backend,
            coupling_map=_ring_coupling(spec.qubits),
            optimization_level=1,
            seed_transpiler=seed,
        )
        result = backend.run(compiled, shots=int(shots), seed_simulator=seed + offset).result()
        for local in range(len(batch)):
            counts = result.get_counts(local)
            expectations = np.zeros(spec.qubits, dtype=np.float64)
            for bitstring, count in counts.items():
                clean = bitstring.replace(" ", "")
                for wire in range(spec.qubits):
                    bit = int(clean[-1 - wire])
                    expectations[wire] += (1.0 if bit == 0 else -1.0) * count
            outputs[offset + local] = expectations / int(shots)
    return outputs


def evaluate_noise_dataset(dataset: str, config: dict[str, Any], resume: bool, confirmatory: bool = False) -> None:
    frozen = freeze_selection(config)
    if confirmatory:
        spec = CircuitSpec(**frozen["confirmatory_robust_spec"])
        stage = "confirm"
        seeds = config["project"]["screening_seeds"]
    else:
        spec = CircuitSpec(**frozen["primary"][dataset]["spec"])
        stage = "final"
        seeds = config["project"]["final_seeds"]
    # The median validation run is chosen deterministically without test access.
    values = []
    for seed in seeds:
        folder = _run_dir(stage, dataset, int(seed), spec)
        metrics = json.loads((folder / "validation_metrics.json").read_text(encoding="utf-8"))
        values.append((float(metrics["macro_f1"]), int(seed)))
    values.sort()
    seed = values[len(values) // 2][1]
    arrays = load_reduced(dataset, seed, config, include_test=True)
    folder = _run_dir(stage, dataset, seed, spec)
    model = load_model(folder, config)
    checkpoint = torch.load(folder / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    theta = checkpoint["model_state"]["theta"].numpy()
    weight = model.output.weight.detach().numpy()
    bias = model.output.bias.detach().numpy()
    output = RESULTS / "noise" / dataset
    output.mkdir(parents=True, exist_ok=True)
    success = output / "SUCCESS.json"
    stage_hash = sha256_json({"protocol": config["_hash"], "spec": asdict(spec), "seed": seed, "profiles": config["noise"]["profiles"], "test": locked_test_hash(dataset)})
    if resume and success.exists() and json.loads(success.read_text())["stage_hash"] == stage_hash:
        return
    records: list[dict[str, Any]] = []
    for split, per_class in (("val", int(config["noise"]["validation_per_class"])), ("test", int(config["noise"]["test_per_class"]))):
        x_all, y_all = arrays[f"{split}_x"], arrays[f"{split}_y"]
        indices = _fixed_stratified(y_all, per_class, 42484)
        x, y = x_all[indices], y_all[indices]
        _, analytic = _predict(model, x)
        analytic_labels = analytic.argmax(axis=1)
        records.append({"dataset": dataset, "split": split, "profile": "analytic", "seed": seed, "samples": len(y), "probability_drift": 0.0, "label_flip_rate": 0.0, **_metric_record(y, analytic, config)})
        for profile_name, profile in config["noise"]["profiles"].items():
            features = _aer_quantum_features(x, spec, theta, profile, int(config["noise"]["shots"]), seed)
            probabilities = softmax_numpy(features @ weight.T + bias)
            records.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "profile": profile_name,
                    "seed": seed,
                    "samples": len(y),
                    "probability_drift": float(np.mean(np.abs(probabilities - analytic))),
                    "label_flip_rate": float(np.mean(probabilities.argmax(axis=1) != analytic_labels)),
                    **_metric_record(y, probabilities, config),
                }
            )
            pd.DataFrame(records).to_csv(output / "noise_metrics_partial.csv", index=False)
    pd.DataFrame(records).to_csv(output / "noise_metrics.csv", index=False)
    atomic_json(success, {"status": "success", "stage_hash": stage_hash, "completed_at": utc_now(), "median_validation_seed": seed})


def run_transpile(config: dict[str, Any], resume: bool) -> None:
    output = RESULTS / "tables" / "transpilation_metrics.csv"
    success = STATE / "TRANSPILE_SUCCESS.json"
    stage_hash = sha256_json(
        {
            "protocol": config["_hash"],
            "architectures": config["architecture"],
            "logical_entanglements": ["linear", "ring"],
            "transpilation_table_version": 2,
        }
    )
    if resume and success.exists() and output.exists() and json.loads(success.read_text())["stage_hash"] == stage_hash:
        return
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    records = []
    specs = [
        CircuitSpec(int(qubits), int(depth), "once", entanglement)
        for qubits in config["architecture"]["qubits"]
        for depth in config["architecture"]["depths"]
        for entanglement in ("linear", "ring")
    ]
    for spec in specs:
        theta = np.zeros((spec.depth, spec.qubits, 3), dtype=np.float64)
        circuit = _qiskit_circuit(spec, np.zeros(6), theta, measure=False)
        topologies = {
            "all_to_all": None,
            "ring": _ring_coupling(spec.qubits),
            "linear": CouplingMap.from_line(spec.qubits, bidirectional=True),
        }
        for topology, coupling in topologies.items():
            compiled = transpile(circuit, coupling_map=coupling, optimization_level=1, seed_transpiler=2026)
            counts = {str(key): int(value) for key, value in compiled.count_ops().items()}
            records.append(
                {
                    **asdict(spec),
                    "topology": topology,
                    "logical_depth": circuit.depth(),
                    "transpiled_depth": compiled.depth(),
                    "two_qubit_gates": counts.get("cx", 0) + counts.get("cz", 0) + counts.get("ecr", 0),
                    "swap_count": counts.get("swap", 0),
                    "size": compiled.size(),
                    "ops_json": json.dumps(counts, sort_keys=True),
                }
            )
    pd.DataFrame(records).to_csv(output, index=False)
    atomic_json(success, {"status": "success", "stage_hash": stage_hash, "completed_at": utc_now(), "rows": len(records)})


def run_confirm(config: dict[str, Any], resume: bool, max_epochs: int | None) -> None:
    frozen = freeze_selection(config)
    dataset = str(config["project"]["confirmatory_dataset"])
    selected = CircuitSpec(**frozen["confirmatory_robust_spec"])
    reference = CircuitSpec(6, 2, "once", "ring")
    for spec in {selected, reference}:
        for seed in config["project"]["screening_seeds"]:
            fit_run("confirm", dataset, int(seed), spec, config, resume, max_epochs)
    aggregate_stage("confirm")
    evaluate_shots_dataset(dataset, config, resume, confirmatory=True)
    evaluate_noise_dataset(dataset, config, resume, confirmatory=True)
    atomic_json(STATE / "CONFIRM_SUCCESS.json", {"status": "success", "completed_at": utc_now(), "selected": asdict(selected), "reference": asdict(reference)})


def _plot_reports(config: dict[str, Any]) -> None:
    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, axis = plt.subplots(figsize=(7.2, 3.15), constrained_layout=True)
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 4)
    axis.axis("off")
    nodes = [
        (0.35, 2.92, 3.1, 0.72, "1  Locked partitions\nand test hashes", "#E8F1F8"),
        (4.45, 2.92, 3.1, 0.72, "2  Frozen DINOv2\nand SLP-6", "#E8F1F8"),
        (8.55, 2.92, 3.1, 0.72, "3  Validation-only\nmodel selection", "#FFF2CC"),
    ]
    for x, y, width, height, label, color in nodes:
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.035", facecolor=color, edgecolor="#355C7D", linewidth=1.15))
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9.2, fontweight="semibold")
    for left, right in [((3.45, 3.28), (4.45, 3.28)), ((7.55, 3.28), (8.55, 3.28))]:
        axis.add_patch(FancyArrowPatch(left, right, arrowstyle="-|>", mutation_scale=12, linewidth=1.15, color="#355C7D"))

    panel_x, panel_y, panel_w, panel_h = 0.35, 0.35, 11.3, 1.62
    axis.add_patch(FancyBboxPatch((panel_x, panel_y), panel_w, panel_h, boxstyle="round,pad=0.04", facecolor="#FAFAFA", edgecolor="#777777", linewidth=1.0))
    axis.text(6.0, 1.68, "Frozen finalist: independent post-selection evaluation", ha="center", va="center", fontsize=9.4, fontweight="bold")
    axis.add_patch(FancyArrowPatch((10.1, 2.92), (10.1, 1.97), arrowstyle="-|>", mutation_scale=12, linewidth=1.15, color="#355C7D"))

    evaluations = [
        (0.65, "Analytic test\n10 seeds"),
        (2.85, "Finite-shot test\n128-4096 shots"),
        (5.05, "Noise sensitivity\n3 fixed profiles"),
        (7.25, "Connectivity audit\n3 coupling maps"),
        (9.45, "EuroSAT\nconfirmation\n5 seeds"),
    ]
    for x, label in evaluations:
        axis.add_patch(FancyBboxPatch((x, 0.62), 1.9, 0.72, boxstyle="round,pad=0.03", facecolor="#E9E3F2", edgecolor="#5F4B8B", linewidth=1.0))
        axis.text(x + 0.95, 0.98, label, ha="center", va="center", fontsize=8.5)
    fig.savefig(figures / "figure_00_workflow.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "figure_00_workflow.svg", bbox_inches="tight")
    plt.close(fig)

    screen = aggregate_stage("screen")
    if not screen.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for axis, dataset in zip(axes, config["project"]["primary_datasets"]):
            subset = screen[screen.dataset == dataset]
            for qubits in sorted(subset.qubits.unique()):
                rows = subset[subset.qubits == qubits].sort_values("depth")
                axis.errorbar(rows.depth, rows.macro_f1_mean, yerr=rows.macro_f1_std.fillna(0), marker="o", label=f"{qubits} qubits")
            axis.set_title(dataset.upper())
            axis.set_xlabel("Circuit depth")
            axis.set_ylabel("Validation macro-F1")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        fig.savefig(figures / "figure_01_width_depth_scaling.png", dpi=300)
        fig.savefig(figures / "figure_01_width_depth_scaling.svg")
        plt.close(fig)
    shot_frames = []
    for dataset in [*config["project"]["primary_datasets"], config["project"]["confirmatory_dataset"]]:
        path = RESULTS / "shots" / dataset / "shot_metrics.csv"
        if path.exists():
            shot_frames.append(pd.read_csv(path))
    if shot_frames:
        shots = pd.concat(shot_frames, ignore_index=True)
        numeric = shots[shots.shots.astype(str) != "analytic"].copy()
        numeric["shots"] = pd.to_numeric(numeric.shots)
        summary = numeric.groupby(["dataset", "shots"], as_index=False).macro_f1.agg(["mean", "std"]).reset_index()
        fig, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        for dataset, rows in summary.groupby("dataset"):
            axis.errorbar(rows.shots, rows["mean"], yerr=rows["std"], marker="o", label=dataset.upper())
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Measurement shots")
        axis.set_ylabel("Test macro-F1")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.savefig(figures / "figure_02_finite_shots.png", dpi=300)
        fig.savefig(figures / "figure_02_finite_shots.svg")
        plt.close(fig)

    noise_path = RESULTS / "tables" / "all_noise_metrics.csv"
    if noise_path.exists():
        noise = pd.read_csv(noise_path)
        test = noise[noise.split == "test"].copy()
        order = ["analytic", "mild", "medium", "high"]
        test["profile"] = pd.Categorical(test.profile, categories=order, ordered=True)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
        for dataset, rows in test.groupby("dataset", observed=False):
            rows = rows.sort_values("profile")
            axes[0].plot(rows.profile.astype(str), rows.macro_f1, marker="o", label=dataset.upper())
            axes[1].plot(rows.profile.astype(str), rows.ece, marker="o", label=dataset.upper())
        axes[0].set_ylabel("Test macro-F1")
        axes[1].set_ylabel("Expected calibration error")
        for axis in axes:
            axis.set_xlabel("Noise profile")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        fig.savefig(figures / "figure_03_noise_sensitivity.png", dpi=300)
        fig.savefig(figures / "figure_03_noise_sensitivity.svg")
        plt.close(fig)

    transpilation_path = RESULTS / "tables" / "transpilation_metrics.csv"
    if transpilation_path.exists():
        transpilation = pd.read_csv(transpilation_path)
        subset = transpilation[(transpilation.qubits == 8) & (transpilation.topology == "linear")]
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
        for entanglement, rows in subset.groupby("entanglement"):
            rows = rows.sort_values("depth")
            axes[0].plot(rows.depth, rows.swap_count, marker="o", label=entanglement)
            axes[1].plot(rows.depth, rows.transpiled_depth, marker="o", label=entanglement)
        axes[0].set_ylabel("Inserted SWAP gates")
        axes[1].set_ylabel("Transpiled circuit depth")
        for axis in axes:
            axis.set_xlabel("Variational depth")
            axis.grid(alpha=0.25)
            axis.legend(title="Logical entanglement", fontsize=8)
        fig.savefig(figures / "figure_04_connectivity_overhead.png", dpi=300)
        fig.savefig(figures / "figure_04_connectivity_overhead.svg")
        plt.close(fig)

    trainability_path = RESULTS / "tables" / "trainability_summary.csv"
    if trainability_path.exists():
        trainability = pd.read_csv(trainability_path)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
        for axis, dataset in zip(axes, config["project"]["primary_datasets"]):
            subset = trainability[trainability.dataset == dataset]
            pivot = subset.pivot(index="qubits", columns="depth", values="grad_var_epoch1_mean")
            image = axis.imshow(np.log10(pivot.to_numpy()), aspect="auto", cmap="viridis")
            axis.set_xticks(range(len(pivot.columns)), [str(value) for value in pivot.columns])
            axis.set_yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
            axis.set_xlabel("Circuit depth")
            axis.set_ylabel("Qubits")
            axis.set_title(dataset.upper())
            for row in range(len(pivot.index)):
                for column in range(len(pivot.columns)):
                    axis.text(column, row, f"{pivot.iloc[row, column]:.1e}", ha="center", va="center", color="white", fontsize=7)
        fig.colorbar(image, ax=axes, label="log10 first-epoch gradient variance", shrink=0.85)
        fig.savefig(figures / "figure_05_trainability.png", dpi=300)
        fig.savefig(figures / "figure_05_trainability.svg")
        plt.close(fig)


def _bootstrap_mean_ci(values: np.ndarray, resamples: int, seed: int = 2026) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_statistical_tables(config: dict[str, Any]) -> None:
    shot_path = RESULTS / "tables" / "all_shot_metrics.csv"
    if shot_path.exists():
        shots = pd.read_csv(shot_path)
        metrics = ["macro_f1", "accuracy", "nll", "ece", "brier", "probability_drift", "label_flip_rate"]
        seed_means = shots.groupby(["dataset", "shots", "seed"], as_index=False)[metrics].mean()
        rows = []
        for (dataset, shot_level), group in seed_means.groupby(["dataset", "shots"], dropna=False):
            row: dict[str, Any] = {"dataset": dataset, "shots": shot_level, "seeds": int(group.seed.nunique())}
            for metric in metrics:
                values = group[metric].to_numpy(dtype=float)
                low, high = _bootstrap_mean_ci(values, int(config["statistics"]["bootstrap_resamples"]), seed=2026 + len(rows))
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
            rows.append(row)
        pd.DataFrame(rows).to_csv(RESULTS / "tables" / "finite_shot_summary_with_ci.csv", index=False)
    noise_path = RESULTS / "tables" / "all_noise_metrics.csv"
    if noise_path.exists():
        noise = pd.read_csv(noise_path)
        analytic = noise[noise.profile == "analytic"][
            ["dataset", "split", "macro_f1", "nll", "ece", "brier"]
        ].rename(columns={name: f"analytic_{name}" for name in ["macro_f1", "nll", "ece", "brier"]})
        comparison = noise[noise.profile != "analytic"].merge(analytic, on=["dataset", "split"], how="left")
        for metric in ["macro_f1", "nll", "ece", "brier"]:
            comparison[f"delta_{metric}"] = comparison[metric] - comparison[f"analytic_{metric}"]
        comparison.to_csv(RESULTS / "tables" / "noise_degradation_summary.csv", index=False)

    trainability_records: list[dict[str, Any]] = []
    for history_path in sorted((RUNS / "screen").glob("*/*/seed_*/training_history.csv")):
        dataset = history_path.parents[2].name
        slug = history_path.parents[1].name
        seed = int(history_path.parent.name.split("_", 1)[1])
        match = re.match(r"q(\d+)-d(\d+)-", slug)
        if match is None:
            continue
        history = pd.read_csv(history_path)
        trainability_records.append(
            {
                "dataset": dataset,
                "qubits": int(match.group(1)),
                "depth": int(match.group(2)),
                "seed": seed,
                "epochs": len(history),
                "grad_norm_epoch1": float(history.iloc[0].gradient_norm_mean),
                "grad_var_epoch1": float(history.iloc[0].gradient_variance_mean),
                "grad_norm_median": float(history.gradient_norm_mean.median()),
                "best_val": float(history.val_macro_f1.max()),
            }
        )
    if trainability_records:
        trainability = pd.DataFrame(trainability_records)
        summary = (
            trainability.groupby(["dataset", "qubits", "depth"], as_index=False)
            .agg(
                n=("seed", "count"),
                epochs_mean=("epochs", "mean"),
                grad_norm_epoch1_mean=("grad_norm_epoch1", "mean"),
                grad_var_epoch1_mean=("grad_var_epoch1", "mean"),
                grad_norm_median_mean=("grad_norm_median", "mean"),
                best_val_mean=("best_val", "mean"),
            )
        )
        summary.to_csv(RESULTS / "tables" / "trainability_summary.csv", index=False)


def build_notebook(config: dict[str, Any]) -> None:
    import nbformat as nbf
    from nbconvert.preprocessors import ExecutePreprocessor

    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {"display_name": "quantum_env", "language": "python", "name": "python3"}
    cells = [
        nbf.v4.new_markdown_cell("# Article 10: hardware-aware quantum transfer learning\n\nThis executed notebook summarizes immutable outputs from the checkpointed CLI. Architecture selection uses validation only; test evaluation begins only after the frozen-selection sentinel."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import Image, display\nA10 = Path.cwd().resolve()\njson.loads((A10/'state'/'FINAL_SELECTION_FROZEN.json').read_text())"),
        nbf.v4.new_markdown_cell("## Validation-locked experimental workflow"),
        nbf.v4.new_code_cell("display(Image(filename=str(A10/'results'/'figures'/'figure_00_workflow.png')))"),
        nbf.v4.new_markdown_cell("## Validation-only architecture screening"),
        nbf.v4.new_code_cell("pd.read_csv(A10/'results'/'tables'/'screen_validation_summary.csv').sort_values(['dataset','macro_f1_mean'], ascending=[True,False]).groupby('dataset').head(8)"),
        nbf.v4.new_code_cell("display(Image(filename=str(A10/'results'/'figures'/'figure_01_width_depth_scaling.png')))"),
        nbf.v4.new_markdown_cell("## Ten-seed finalists"),
        nbf.v4.new_code_cell("pd.read_csv(A10/'results'/'tables'/'final_validation_summary.csv')"),
        nbf.v4.new_markdown_cell("## Finite-shot robustness with seed-level bootstrap intervals"),
        nbf.v4.new_code_cell("pd.read_csv(A10/'results'/'tables'/'finite_shot_summary_with_ci.csv')"),
        nbf.v4.new_code_cell("display(Image(filename=str(A10/'results'/'figures'/'figure_02_finite_shots.png')))"),
        nbf.v4.new_markdown_cell("## Circuit-level depolarizing and readout noise"),
        nbf.v4.new_code_cell("pd.read_csv(A10/'results'/'tables'/'noise_degradation_summary.csv').query(\"split == 'test'\")"),
        nbf.v4.new_code_cell("display(Image(filename=str(A10/'results'/'figures'/'figure_03_noise_sensitivity.png')))"),
        nbf.v4.new_markdown_cell("## Connectivity-aware transpilation"),
        nbf.v4.new_code_cell("pd.read_csv(A10/'results'/'tables'/'transpilation_metrics.csv').query(\"qubits == 8 and topology == 'linear'\")"),
        nbf.v4.new_code_cell("display(Image(filename=str(A10/'results'/'figures'/'figure_04_connectivity_overhead.png')))"),
        nbf.v4.new_markdown_cell("## Trainability diagnostics"),
        nbf.v4.new_code_cell("pd.read_csv(A10/'results'/'tables'/'trainability_summary.csv')"),
        nbf.v4.new_code_cell("display(Image(filename=str(A10/'results'/'figures'/'figure_05_trainability.png')))"),
        nbf.v4.new_markdown_cell("## Re-run instructions\n\nUse `python run_article10.py --stage all --resume`. Set `ARTICLE09_ROOT` to an Article 9 workspace containing the immutable embeddings and fitted SLP-6 representations. Real-hardware execution remains disabled unless separately authorized."),
    ]
    notebook["cells"] = cells
    path = A10 / "notebooks" / "Article10_Hardware_Aware_QTL.ipynb"
    nbf.write(notebook, path)
    executor = ExecutePreprocessor(timeout=600, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(A10)}})
    nbf.write(notebook, path)


def run_report(config: dict[str, Any]) -> None:
    for stage in ("screen", "ablate", "final", "confirm"):
        aggregate_stage(stage)
    shot_frames, noise_frames = [], []
    for dataset in [*config["project"]["primary_datasets"], config["project"]["confirmatory_dataset"]]:
        shot_path = RESULTS / "shots" / dataset / "shot_metrics.csv"
        noise_path = RESULTS / "noise" / dataset / "noise_metrics.csv"
        if shot_path.exists():
            shot_frames.append(pd.read_csv(shot_path))
        if noise_path.exists():
            noise_frames.append(pd.read_csv(noise_path))
    if shot_frames:
        pd.concat(shot_frames, ignore_index=True).to_csv(RESULTS / "tables" / "all_shot_metrics.csv", index=False)
    if noise_frames:
        pd.concat(noise_frames, ignore_index=True).to_csv(RESULTS / "tables" / "all_noise_metrics.csv", index=False)
    build_statistical_tables(config)
    _plot_reports(config)
    build_notebook(config)
    frozen = freeze_selection(config)
    draft = f"""# {config['project']['title']}

## Status

This manuscript scaffold is generated from protocol `{config['_hash']}`. Numerical claims must be populated only from the machine-readable tables after the completion audit passes.

## Frozen validation selection

```json
{json.dumps(frozen, indent=2)}
```

## Intended contribution

The study separates ideal-circuit performance from trainability, finite-shot uncertainty, circuit-level noise, calibration degradation, and connectivity-aware transpilation while holding the visual representation and data splits fixed. It makes no claim of quantum advantage or speedup.
"""
    atomic_text(A10 / "article" / "ARTICLE10_DRAFT.md", draft)
    manifest = {
        "completed_at": utc_now(),
        "protocol_sha256": config["_hash"],
        "selection": frozen,
        "tables": {path.name: sha256_file(path) for path in sorted((RESULTS / "tables").glob("*.csv"))},
        "figures": {path.name: sha256_file(path) for path in sorted((RESULTS / "figures").glob("*.*"))},
        "notebook": sha256_file(A10 / "notebooks" / "Article10_Hardware_Aware_QTL.ipynb"),
    }
    atomic_json(RESULTS / "report_manifest.json", manifest)
    atomic_json(STATE / "REPORT_SUCCESS.json", {"status": "success", "completed_at": utc_now(), "manifest_sha256": sha256_file(RESULTS / "report_manifest.json")})


def _parallel(stage: str, datasets: Iterable[str], resume: bool, max_epochs: int | None) -> None:
    processes: list[tuple[str, subprocess.Popen, Any, Any]] = []
    for dataset in datasets:
        command = [sys.executable, str(A10 / "run_article10.py"), "--stage", stage, "--dataset", dataset]
        if resume:
            command.append("--resume")
        if max_epochs is not None:
            command.extend(["--max-epochs", str(max_epochs)])
        stdout_path = LOGS / f"{stage}_{dataset}.stdout.log"
        stderr_path = LOGS / f"{stage}_{dataset}.stderr.log"
        stdout_handle = stdout_path.open("a", encoding="utf-8")
        stderr_handle = stderr_path.open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(load_config()["execution"]["omp_threads_per_worker"])
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout_handle, stderr=stderr_handle, env=environment)
        processes.append((dataset, process, stdout_handle, stderr_handle))
    failures = []
    for dataset, process, stdout_handle, stderr_handle in processes:
        returncode = process.wait()
        stdout_handle.close()
        stderr_handle.close()
        if returncode != 0:
            failures.append((dataset, returncode, str(LOGS / f"{stage}_{dataset}.stderr.log")))
    if failures:
        raise RuntimeError(f"Parallel {stage} failures: {failures}")


def _assert_primary_complete(stage: str, config: dict[str, Any]) -> None:
    for dataset in config["project"]["primary_datasets"]:
        marker = STATE / f"{stage.upper()}_{dataset.upper()}_SUCCESS.json"
        if not marker.exists():
            raise RuntimeError(f"Required stage marker missing: {marker}")


def run_cli(args) -> None:
    config = load_config()
    run_setup(config)
    stage = args.stage
    resume = bool(args.resume)
    max_epochs = args.max_epochs
    status_path = (
        STATE / f"worker_status_{stage}_{args.dataset}.json"
        if args.dataset
        else STATE / "pipeline_status.json"
    )
    try:
        if stage == "all" and (STATE / "FAILED").exists():
            (STATE / "FAILED").unlink()
        atomic_json(status_path, {"status": "running", "stage": stage, "dataset": args.dataset, "pid": os.getpid(), "updated_at": utc_now()})
        if stage == "setup":
            pass
        elif stage == "smoke":
            run_smoke(config, resume)
        elif stage == "screen":
            if not args.dataset:
                raise ValueError("screen worker requires --dataset")
            run_screen_dataset(args.dataset, config, resume, max_epochs)
        elif stage == "ablate":
            if not args.dataset:
                raise ValueError("ablate worker requires --dataset")
            run_ablate_dataset(args.dataset, config, resume, max_epochs)
        elif stage == "final":
            if not args.dataset:
                raise ValueError("final worker requires --dataset")
            run_final_dataset(args.dataset, config, resume, max_epochs)
        elif stage == "shots":
            if not args.dataset:
                raise ValueError("shots worker requires --dataset")
            evaluate_shots_dataset(args.dataset, config, resume)
            atomic_json(STATE / f"SHOTS_{args.dataset.upper()}_SUCCESS.json", {"status": "success", "completed_at": utc_now()})
        elif stage == "noise":
            if not args.dataset:
                raise ValueError("noise worker requires --dataset")
            evaluate_noise_dataset(args.dataset, config, resume)
            atomic_json(STATE / f"NOISE_{args.dataset.upper()}_SUCCESS.json", {"status": "success", "completed_at": utc_now()})
        elif stage == "transpile":
            run_transpile(config, resume)
        elif stage == "confirm":
            run_confirm(config, resume, max_epochs)
        elif stage == "report":
            run_report(config)
        elif stage == "all":
            run_smoke(config, resume)
            _parallel("screen", config["project"]["primary_datasets"], resume, max_epochs)
            _assert_primary_complete("screen", config)
            _parallel("ablate", config["project"]["primary_datasets"], resume, max_epochs)
            _assert_primary_complete("ablate", config)
            freeze_selection(config)
            _parallel("final", config["project"]["primary_datasets"], resume, max_epochs)
            _assert_primary_complete("final", config)
            _parallel("shots", config["project"]["primary_datasets"], resume, None)
            _parallel("noise", config["project"]["primary_datasets"], resume, None)
            run_transpile(config, resume)
            run_confirm(config, resume, max_epochs)
            run_report(config)
            atomic_text(STATE / "SUCCESS", utc_now() + "\n")
        else:
            raise KeyError(stage)
        atomic_json(status_path, {"status": "success", "stage": stage, "dataset": args.dataset, "pid": os.getpid(), "updated_at": utc_now()})
    except Exception as error:
        atomic_json(
            status_path,
            {
                "status": "failed",
                "stage": stage,
                "dataset": args.dataset,
                "pid": os.getpid(),
                "updated_at": utc_now(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        atomic_text(STATE / "FAILED", traceback.format_exc())
        raise



