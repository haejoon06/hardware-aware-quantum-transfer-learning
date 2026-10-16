from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from torch import nn

from .utils import atomic_torch_save, set_global_seed


@dataclass
class ReducerBundle:
    name: str
    input_scaler: StandardScaler
    reducer: Any = None
    projection_weight: np.ndarray | None = None
    projection_bias: np.ndarray | None = None

    def transform(self, x: np.ndarray) -> np.ndarray:
        standardized = self.input_scaler.transform(np.asarray(x))
        if self.name == "slp6":
            projected = standardized @ self.projection_weight.T + self.projection_bias
            return np.tanh(projected).astype(np.float32)
        return np.asarray(self.reducer.transform(standardized), dtype=np.float32)


@dataclass
class AngleScaler:
    scaler: StandardScaler

    @classmethod
    def fit(cls, x: np.ndarray) -> "AngleScaler":
        return cls(StandardScaler().fit(x))

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(x)
        return (math.pi * np.tanh(z)).astype(np.float32)


class _ProjectionNet(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.projection = nn.Linear(384, 6)
        self.classifier = nn.Linear(6, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.tanh(self.projection(x)))


def _fit_slp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    config: dict,
    work_dir: Path,
    device: str,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cfg = config["reducers"]["slp"]
    set_global_seed(seed)
    n_classes = int(max(y_train.max(), y_val.max()) + 1)
    model = _ProjectionNet(n_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    checkpoint_path = work_dir / "slp_latest.pt"
    start_epoch = 0
    best_f1 = -np.inf
    best_state = None
    bad_epochs = 0
    history: list[dict] = []
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint["best_f1"])
        best_state = checkpoint["best_state"]
        bad_epochs = int(checkpoint["bad_epochs"])
        history = list(checkpoint["history"])

    x_train_t = torch.from_numpy(np.asarray(x_train, dtype=np.float32)).to(device)
    y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.int64)).to(device)
    x_val_t = torch.from_numpy(np.asarray(x_val, dtype=np.float32)).to(device)
    batch_size = int(cfg["batch_size"])
    for epoch in range(start_epoch, int(cfg["max_epochs"])):
        model.train()
        order = np.random.default_rng(seed * 100_000 + epoch).permutation(len(y_train))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start : start + batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train_t[idx])
            loss = nn.functional.cross_entropy(logits, y_train_t[idx])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            pred = model(x_val_t).argmax(dim=1).cpu().numpy()
        score = float(f1_score(y_val, pred, average="macro"))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": score})
        if score > best_f1 + 1e-12:
            best_f1 = score
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        atomic_torch_save(
            checkpoint_path,
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state,
                "best_f1": best_f1,
                "bad_epochs": bad_epochs,
                "history": history,
                "seed": seed,
            },
        )
        if bad_epochs >= int(cfg["patience"]):
            break
    if best_state is None:
        raise RuntimeError("Supervised projection did not produce a valid state.")
    model.load_state_dict(best_state)
    weight = model.projection.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.projection.bias.detach().cpu().numpy().astype(np.float32)
    return weight, bias, {"best_val_macro_f1": best_f1, "history": history}


def fit_reducer(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    config: dict,
    work_dir: str | Path,
    device: str = "cpu",
    resume: bool = True,
) -> tuple[ReducerBundle, AngleScaler, dict]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler().fit(x_train)
    train_std = scaler.transform(x_train)
    val_std = scaler.transform(x_val)
    metadata: dict[str, Any] = {"name": name, "seed": seed}
    if name == "pca6":
        reducer = PCA(n_components=6, whiten=False, random_state=seed).fit(train_std)
        bundle = ReducerBundle(name=name, input_scaler=scaler, reducer=reducer)
        metadata["explained_variance_ratio"] = reducer.explained_variance_ratio_.tolist()
    elif name == "lda6":
        reducer = LinearDiscriminantAnalysis(n_components=6, solver="svd").fit(
            train_std, y_train
        )
        bundle = ReducerBundle(name=name, input_scaler=scaler, reducer=reducer)
    elif name == "nca6":
        reducer = NeighborhoodComponentsAnalysis(
            n_components=6,
            init="pca",
            max_iter=int(config["reducers"]["nca_max_iter"]),
            tol=float(config["reducers"]["nca_tol"]),
            random_state=seed,
        ).fit(train_std, y_train)
        bundle = ReducerBundle(name=name, input_scaler=scaler, reducer=reducer)
        metadata["n_iter"] = int(reducer.n_iter_)
    elif name == "slp6":
        weight, bias, slp_meta = _fit_slp(
            train_std,
            y_train,
            val_std,
            y_val,
            seed,
            config,
            work_dir,
            device,
            resume,
        )
        bundle = ReducerBundle(
            name=name,
            input_scaler=scaler,
            projection_weight=weight,
            projection_bias=bias,
        )
        metadata.update(slp_meta)
    else:
        raise KeyError(name)

    train_reduced = bundle.transform(x_train)
    angle_scaler = AngleScaler.fit(train_reduced)
    joblib.dump(bundle, work_dir / "reducer.joblib")
    joblib.dump(angle_scaler, work_dir / "angle_scaler.joblib")
    return bundle, angle_scaler, metadata

