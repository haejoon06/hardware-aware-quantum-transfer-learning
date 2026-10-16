from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pennylane as qml
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from torch import nn

from .config import ROOT
from .metrics import classification_metrics
from .utils import atomic_torch_save, atomic_write_json, set_global_seed


class MatchedMLP(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.hidden = nn.Linear(6, 6, bias=False)
        self.output = nn.Linear(6, n_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.hidden(x)))


class VQCNet(nn.Module):
    def __init__(self, n_classes: int, backend: str = "default.qubit", layers: int = 2):
        super().__init__()
        self.n_classes = n_classes
        self.backend = backend
        self.layers = layers
        self.theta = nn.Parameter(0.05 * torch.randn(layers, 6, 3))
        self.output = nn.Linear(6, n_classes, bias=True)
        self._build_qnode()

    def _build_qnode(self) -> None:
        device = qml.device(self.backend, wires=6, shots=None)
        diff_method = "backprop" if self.backend == "default.qubit" else "adjoint"

        @qml.qnode(device, interface="torch", diff_method=diff_method)
        def circuit(inputs: torch.Tensor, theta: torch.Tensor):
            for layer in range(self.layers):
                qml.AngleEmbedding(inputs, wires=range(6), rotation="Y")
                for wire in range(6):
                    qml.Rot(
                        theta[layer, wire, 0],
                        theta[layer, wire, 1],
                        theta[layer, wire, 2],
                        wires=wire,
                    )
                for wire in range(5):
                    qml.CNOT(wires=[wire, wire + 1])
                qml.CNOT(wires=[5, 0])
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(6))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantum = self.circuit(x, self.theta)
        if isinstance(quantum, (tuple, list)):
            quantum = torch.stack(list(quantum), dim=-1)
        elif quantum.ndim == 2 and quantum.shape[0] == 6 and quantum.shape[1] != 6:
            quantum = quantum.transpose(0, 1)
        return self.output(quantum.to(dtype=self.output.weight.dtype))


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_classical(head: str, hparams: dict, seed: int):
    if head == "lr":
        return LogisticRegression(
            C=float(hparams["C"]),
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        )
    if head == "rbf":
        gamma: str | float = hparams["gamma"]
        if gamma != "scale":
            gamma = float(gamma)
        return SVC(
            C=float(hparams["C"]),
            gamma=gamma,
            kernel="rbf",
            probability=True,
            random_state=seed,
        )
    raise KeyError(head)


def fit_classical(
    head: str,
    hparams: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    work_dir: str | Path,
    seed: int,
) -> tuple[Any, dict]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    model = make_classical(head, hparams, seed)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    elapsed = time.perf_counter() - start
    joblib.dump(model, work_dir / "model.joblib")
    return model, {"fit_seconds": elapsed}


def _predict_neural(model: nn.Module, x: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(np.asarray(x[start : start + batch_size], dtype=np.float32)).to(device)
            outputs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def fit_neural(
    head: str,
    hparams: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    work_dir: str | Path,
    seed: int,
    config: dict,
    backend: str,
    device: str,
    resume: bool = True,
) -> tuple[nn.Module, dict]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    if head == "mlp":
        model: nn.Module = MatchedMLP(n_classes)
    elif head == "vqc":
        model = VQCNet(n_classes, backend=backend, layers=int(config["quantum"]["layers"]))
        device = "cpu"  # CPU simulators own their tensors; GPU is reserved for feature extraction.
    else:
        raise KeyError(head)
    model = model.to(device)
    expected = 36 + 7 * n_classes
    if trainable_parameters(model) != expected:
        raise AssertionError(
            f"Parameter parity failed for {head}: {trainable_parameters(model)} != {expected}"
        )
    cfg = config["heads"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    checkpoint_path = work_dir / "checkpoint_latest.pt"
    start_epoch = 0
    best_f1 = -np.inf
    best_nll = np.inf
    best_state = None
    bad_epochs = 0
    history: list[dict[str, float]] = []
    elapsed_before = 0.0
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint["best_f1"])
        best_nll = float(checkpoint["best_nll"])
        best_state = checkpoint["best_state"]
        bad_epochs = int(checkpoint["bad_epochs"])
        history = list(checkpoint["history"])
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))

    x_tensor = torch.from_numpy(np.asarray(x_train, dtype=np.float32)).to(device)
    y_tensor = torch.from_numpy(np.asarray(y_train, dtype=np.int64)).to(device)
    batch_size = int(cfg["batch_size"])
    started = time.perf_counter()
    for epoch in range(start_epoch, int(cfg["max_epochs"])):
        model.train()
        order = np.random.default_rng(seed * 100_000 + epoch).permutation(len(y_train))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start : start + batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor[idx])
            loss = nn.functional.cross_entropy(logits, y_tensor[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probabilities = _predict_neural(model, x_val, device)
        metrics = classification_metrics(
            y_val, probabilities, ece_bins=int(config["statistics"]["ece_bins"])
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_macro_f1": metrics["macro_f1"],
            "val_nll": metrics["nll"],
        }
        history.append(row)
        improved = metrics["macro_f1"] > best_f1 + 1e-12 or (
            abs(metrics["macro_f1"] - best_f1) <= 1e-12 and metrics["nll"] < best_nll
        )
        if improved:
            best_f1 = metrics["macro_f1"]
            best_nll = metrics["nll"]
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        elapsed = elapsed_before + time.perf_counter() - started
        atomic_torch_save(
            checkpoint_path,
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state,
                "best_f1": best_f1,
                "best_nll": best_nll,
                "bad_epochs": bad_epochs,
                "history": history,
                "elapsed_seconds": elapsed,
                "seed": seed,
                "head": head,
                "n_classes": n_classes,
                "backend": backend,
            },
        )
        pd.DataFrame(history).to_csv(work_dir / "training_history.csv", index=False)
        if bad_epochs >= int(cfg["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"{head} did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    atomic_torch_save(
        work_dir / "checkpoint_best.pt",
        {
            "model_state": best_state,
            "head": head,
            "n_classes": n_classes,
            "backend": backend,
            "hparams": hparams,
            "seed": seed,
            "parameter_count": trainable_parameters(model),
        },
    )
    return model, {
        "fit_seconds": elapsed_before + time.perf_counter() - started,
        "best_val_macro_f1": best_f1,
        "best_val_nll": best_nll,
        "epochs_completed": len(history),
        "parameter_count": trainable_parameters(model),
    }


def model_probabilities(model: Any, head: str, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    if head in {"lr", "rbf"}:
        return np.asarray(model.predict_proba(x), dtype=np.float64)
    return _predict_neural(model, x, device)


def load_neural_model(
    head: str, checkpoint_path: str | Path, backend: str, device: str = "cpu"
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_classes = int(checkpoint["n_classes"])
    if head == "mlp":
        model: nn.Module = MatchedMLP(n_classes)
    else:
        model = VQCNet(n_classes, backend=backend)
        device = "cpu"
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def benchmark_one_backend(config: dict, backend: str) -> dict:
    x = torch.linspace(-math.pi, math.pi, steps=32 * 6, dtype=torch.float32).reshape(32, 6)
    y = torch.arange(32, dtype=torch.long) % 10
    set_global_seed(2026)
    model = VQCNet(10, backend=backend)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    for _ in range(int(config["quantum"]["benchmark_forward"])):
        with torch.no_grad():
            model(x)
    forward_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(int(config["quantum"]["benchmark_backward"])):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    backward_seconds = time.perf_counter() - start
    return {
        "valid": True,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "score": forward_seconds + 5.0 * backward_seconds,
    }


def benchmark_backends(config: dict, output_path: str | Path) -> dict:
    results = {}
    probe_dir = ROOT / "state" / "backend_probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    for backend in config["quantum"]["candidate_backends"]:
        probe_path = probe_dir / f"{backend.replace('.', '_')}.json"
        if probe_path.exists():
            probe_path.unlink()
        command = [
            sys.executable,
            str(ROOT / "run_article09.py"),
            "_backend_probe",
            "--backend",
            backend,
            "--output",
            str(probe_path),
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        if completed.returncode == 0 and probe_path.exists():
            results[backend] = json.loads(probe_path.read_text(encoding="utf-8"))
        else:
            results[backend] = {
                "valid": False,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-2000:],
            }
    valid = {name: row for name, row in results.items() if row.get("valid")}
    if not valid:
        raise RuntimeError(f"No quantum backend passed benchmark: {results}")
    selected = min(valid, key=lambda name: valid[name]["score"])
    payload = {"selected_backend": selected, "results": results}
    atomic_write_json(output_path, payload)
    return payload

