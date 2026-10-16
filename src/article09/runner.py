from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from .config import ROOT, canonical_json, sha256_file, sha256_json
from .data import DATASETS
from .diagnostics import representation_diagnostics
from .features import load_embeddings
from .metrics import classification_metrics
from .models import (
    benchmark_backends,
    fit_classical,
    fit_neural,
    load_neural_model,
    model_probabilities,
)
from .reducers import fit_reducer
from .utils import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    capture_environment,
    stable_subset_indices,
    task_complete,
    task_context,
    task_sentinel,
    utc_now,
)


def _dataset_arrays(dataset: str, config: dict):
    train_x, train_y, train_ids = load_embeddings(dataset, "train")
    val_x, val_y, val_ids = load_embeddings(dataset, "val")
    val_idx = stable_subset_indices(
        np.asarray(val_y),
        int(config["project"]["val_per_class"]),
        int(config["project"]["validation_seed"]),
    )
    return (
        np.asarray(train_x),
        np.asarray(train_y),
        train_ids,
        np.asarray(val_x)[val_idx],
        np.asarray(val_y)[val_idx],
        val_ids.iloc[val_idx].reset_index(drop=True),
    )


def _training_subset(
    x: np.ndarray, y: np.ndarray, ids: pd.DataFrame, config: dict, seed: int
):
    idx = stable_subset_indices(
        y, int(config["project"]["train_per_class"]), seed
    )
    return x[idx], y[idx], ids.iloc[idx].reset_index(drop=True)


def _backend() -> str:
    path = ROOT / "frozen" / "backend_benchmark.json"
    return json.loads(path.read_text(encoding="utf-8"))["selected_backend"]


def _candidate_grid(head: str, config: dict) -> list[dict[str, Any]]:
    if head == "lr":
        return [{"C": value} for value in config["heads"]["lr_grid"]]
    if head == "rbf":
        return [
            {"C": c_value, "gamma": gamma}
            for c_value in config["heads"]["rbf_c_grid"]
            for gamma in config["heads"]["rbf_gamma_grid"]
        ]
    if head in {"mlp", "vqc"}:
        return [{"lr": value} for value in config["heads"]["neural_lr_grid"]]
    raise KeyError(head)


def _validation_score(y: np.ndarray, probabilities: np.ndarray, config: dict) -> dict:
    return classification_metrics(y, probabilities, int(config["statistics"]["ece_bins"]))


def _better(left: dict | None, right: dict) -> bool:
    if left is None:
        return True
    if right["metrics"]["macro_f1"] > left["metrics"]["macro_f1"] + 1e-12:
        return True
    return (
        abs(right["metrics"]["macro_f1"] - left["metrics"]["macro_f1"]) <= 1e-12
        and right["metrics"]["nll"] < left["metrics"]["nll"]
    )


def run_backend_benchmark(config: dict, resume: bool = True) -> dict:
    output = ROOT / "frozen" / "backend_benchmark.json"
    input_hash = sha256_json(config["quantum"])
    if resume and task_complete("backend-benchmark", input_hash) and output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    with task_context("backend-benchmark", input_hash):
        return benchmark_backends(config, output)


def run_pilot(config: dict, resume: bool = True, device: str = "cuda:0") -> dict:
    selected_path = ROOT / "configs" / "selected_hparams.yaml"
    pilot_hash = sha256_json(
        {
            "config": config["_config_hash"],
            "backend": _backend(),
            "manifests": {
                name: sha256_file(ROOT / "data" / "manifests" / f"{name}.csv")
                for name in DATASETS
            },
        }
    )
    if resume and task_complete("pilot", pilot_hash) and selected_path.exists():
        return yaml.safe_load(selected_path.read_text(encoding="utf-8"))
    selected: dict[str, Any] = {}
    with task_context("pilot", pilot_hash):
        for dataset in DATASETS:
            selected[dataset] = {}
            train_x_all, train_y_all, train_ids_all, val_x, val_y, val_ids = _dataset_arrays(
                dataset, config
            )
            train_x, train_y, _ = _training_subset(
                train_x_all,
                train_y_all,
                train_ids_all,
                config,
                int(config["project"]["pilot_seed"]),
            )
            n_classes = int(np.max(train_y) + 1)
            for reducer_name in config["reducers"]["names"]:
                shared_dir = ROOT / "runs" / "pilot" / dataset / reducer_name / "shared"
                bundle, angle_scaler, reducer_meta = fit_reducer(
                    reducer_name,
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    int(config["project"]["pilot_seed"]),
                    config,
                    shared_dir,
                    device=device,
                    resume=resume,
                )
                train_6 = angle_scaler.transform(bundle.transform(train_x))
                val_6 = angle_scaler.transform(bundle.transform(val_x))
                selected[dataset][reducer_name] = {}
                for head in config["heads"]["names"]:
                    best = None
                    for candidate_index, hparams in enumerate(_candidate_grid(head, config)):
                        work_dir = (
                            ROOT
                            / "runs"
                            / "pilot"
                            / dataset
                            / reducer_name
                            / head
                            / f"candidate_{candidate_index:02d}"
                        )
                        work_dir.mkdir(parents=True, exist_ok=True)
                        if head in {"lr", "rbf"}:
                            model, runtime = fit_classical(
                                head,
                                hparams,
                                train_6,
                                train_y,
                                work_dir,
                                int(config["project"]["pilot_seed"]),
                            )
                        else:
                            model, runtime = fit_neural(
                                head,
                                hparams,
                                train_6,
                                train_y,
                                val_6,
                                val_y,
                                n_classes,
                                work_dir,
                                int(config["project"]["pilot_seed"]),
                                config,
                                _backend(),
                                device,
                                resume,
                            )
                        probabilities = model_probabilities(
                            model, head, val_6, "cpu" if head == "vqc" else device
                        )
                        result = {
                            "hparams": hparams,
                            "metrics": _validation_score(val_y, probabilities, config),
                            "runtime": runtime,
                        }
                        atomic_write_json(work_dir / "validation_metrics.json", result)
                        if _better(best, result):
                            best = result
                    assert best is not None
                    selected[dataset][reducer_name][head] = best["hparams"]

            selected[dataset]["full384"] = {}
            scaler = joblib.load(
                ROOT / "runs" / "pilot" / dataset / config["reducers"]["names"][0] / "shared" / "reducer.joblib"
            ).input_scaler
            train_full = scaler.transform(train_x).astype(np.float32)
            val_full = scaler.transform(val_x).astype(np.float32)
            for head in ("lr", "rbf"):
                best = None
                for candidate_index, hparams in enumerate(_candidate_grid(head, config)):
                    work_dir = ROOT / "runs" / "pilot" / dataset / "full384" / head / f"candidate_{candidate_index:02d}"
                    model, runtime = fit_classical(
                        head,
                        hparams,
                        train_full,
                        train_y,
                        work_dir,
                        int(config["project"]["pilot_seed"]),
                    )
                    probabilities = model_probabilities(model, head, val_full)
                    result = {
                        "hparams": hparams,
                        "metrics": _validation_score(val_y, probabilities, config),
                        "runtime": runtime,
                    }
                    atomic_write_json(work_dir / "validation_metrics.json", result)
                    if _better(best, result):
                        best = result
                assert best is not None
                selected[dataset]["full384"][head] = best["hparams"]
        atomic_write_yaml(selected_path, selected)
    return selected


def freeze_protocol(config: dict, resume: bool = True) -> dict:
    selected_path = ROOT / "configs" / "selected_hparams.yaml"
    backend_path = ROOT / "frozen" / "backend_benchmark.json"
    if not selected_path.exists() or not backend_path.exists():
        raise FileNotFoundError("Pilot hyperparameters and backend benchmark must exist before freeze.")
    payload = {
        "protocol_version": 1,
        "frozen_at": utc_now(),
        "config_sha256": config["_config_hash"],
        "selected_hparams_sha256": sha256_file(selected_path),
        "backend_benchmark_sha256": sha256_file(backend_path),
        "selected_backend": _backend(),
        "datasets": {},
        "final_seeds": config["project"]["final_seeds"],
        "reducers": config["reducers"]["names"],
        "heads": config["heads"]["names"],
        "primary_endpoint": "macro_f1",
        "tie_breaker": "nll",
        "vqc": {
            "qubits": 6,
            "layers": 2,
            "encoding": "RY angle encoding with data re-uploading",
            "entanglement": "ring CNOT",
            "measurement": "six Pauli-Z expectations",
            "shots": None,
        },
    }
    for dataset in DATASETS:
        embed_meta_path = ROOT / "data" / "embeddings" / dataset / "metadata.json"
        audit_path = ROOT / "data" / "audit" / f"{dataset}_audit_report.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        payload["datasets"][dataset] = {
            "manifest_sha256": sha256_file(ROOT / "data" / "manifests" / f"{dataset}.csv"),
            "locked_test_sha256": audit["locked_test_sha256"],
            "embedding_metadata_sha256": sha256_file(embed_meta_path),
            "embedding_hashes": json.loads(embed_meta_path.read_text(encoding="utf-8"))["splits"],
        }
    output = ROOT / "frozen" / "protocol_v1.json"
    atomic_write_json(output, payload)
    digest = sha256_file(output)
    atomic_write_text(ROOT / "frozen" / "protocol_v1.sha256", digest + "\n")
    return payload


def _write_common_run_files(
    run_dir: Path,
    run_config: dict,
    train_ids: pd.DataFrame,
    val_ids: pd.DataFrame,
    dataset: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_yaml(run_dir / "config.yaml", run_config)
    atomic_write_text(run_dir / "config_hash.txt", sha256_json(run_config) + "\n")
    train_ids.to_csv(run_dir / "train_ids.csv", index=False)
    val_ids.to_csv(run_dir / "val_ids.csv", index=False)
    audit = json.loads(
        (ROOT / "data" / "audit" / f"{dataset}_audit_report.json").read_text(encoding="utf-8")
    )
    atomic_write_text(run_dir / "test_manifest_hash.txt", audit["locked_test_sha256"] + "\n")


def _write_validation_predictions(
    run_dir: Path,
    ids: pd.DataFrame,
    y: np.ndarray,
    probabilities: np.ndarray,
    run_config: dict,
) -> None:
    frame = ids[["sample_id"]].copy()
    frame["true_label"] = y
    frame["predicted_label"] = probabilities.argmax(axis=1)
    for index in range(probabilities.shape[1]):
        frame[f"prob_class_{index}"] = probabilities[:, index]
    for key in ("seed", "dataset", "reducer", "head"):
        frame[key] = run_config[key]
    frame.to_parquet(run_dir / "validation_predictions.parquet", index=False)


def _fit_final_run(
    dataset: str,
    reducer_name: str,
    head: str,
    seed: int,
    train_6: np.ndarray,
    train_y: np.ndarray,
    val_6: np.ndarray,
    val_y: np.ndarray,
    train_ids: pd.DataFrame,
    val_ids: pd.DataFrame,
    shared_dir: Path,
    config: dict,
    selected: dict,
    device: str,
    resume: bool,
) -> None:
    run_id = f"A09-{dataset.upper()}-{reducer_name.upper()}-{head.upper()}-S{seed}"
    run_dir = ROOT / "runs" / run_id
    hparams = selected[dataset][reducer_name][head]
    run_config = {
        "run_id": run_id,
        "dataset": dataset,
        "reducer": reducer_name,
        "head": head,
        "seed": seed,
        "hparams": hparams,
        "backend": _backend(),
        "protocol_config_sha256": config["_config_hash"],
    }
    input_hash = sha256_json(run_config)
    if resume and task_complete(run_id, input_hash):
        return
    with task_context(run_id, input_hash):
        _write_common_run_files(run_dir, run_config, train_ids, val_ids, dataset)
        for name in ("reducer.joblib", "angle_scaler.joblib"):
            shutil.copy2(shared_dir / name, run_dir / name)
        n_classes = int(np.max(train_y) + 1)
        if head in {"lr", "rbf"}:
            model, runtime = fit_classical(head, hparams, train_6, train_y, run_dir, seed)
        else:
            model, runtime = fit_neural(
                head,
                hparams,
                train_6,
                train_y,
                val_6,
                val_y,
                n_classes,
                run_dir,
                seed,
                config,
                _backend(),
                device,
                resume,
            )
        probabilities = model_probabilities(
            model, head, val_6, "cpu" if head == "vqc" else device
        )
        metrics = _validation_score(val_y, probabilities, config)
        atomic_write_json(run_dir / "validation_metrics.json", metrics)
        atomic_write_json(run_dir / "runtime.json", runtime)
        _write_validation_predictions(run_dir, val_ids, val_y, probabilities, run_config)


def _fit_full_reference(
    dataset: str,
    head: str,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    train_ids: pd.DataFrame,
    val_ids: pd.DataFrame,
    config: dict,
    selected: dict,
    resume: bool,
) -> None:
    from sklearn.preprocessing import StandardScaler

    run_id = f"A09-{dataset.upper()}-FULL384-{head.upper()}-S{seed}"
    run_dir = ROOT / "runs" / run_id
    hparams = selected[dataset]["full384"][head]
    run_config = {
        "run_id": run_id,
        "dataset": dataset,
        "reducer": "full384",
        "head": head,
        "seed": seed,
        "hparams": hparams,
        "protocol_config_sha256": config["_config_hash"],
    }
    input_hash = sha256_json(run_config)
    if resume and task_complete(run_id, input_hash):
        return
    with task_context(run_id, input_hash):
        _write_common_run_files(run_dir, run_config, train_ids, val_ids, dataset)
        scaler = StandardScaler().fit(train_x)
        joblib.dump(scaler, run_dir / "full_scaler.joblib")
        train_scaled = scaler.transform(train_x).astype(np.float32)
        val_scaled = scaler.transform(val_x).astype(np.float32)
        model, runtime = fit_classical(head, hparams, train_scaled, train_y, run_dir, seed)
        probabilities = model_probabilities(model, head, val_scaled)
        metrics = _validation_score(val_y, probabilities, config)
        atomic_write_json(run_dir / "validation_metrics.json", metrics)
        atomic_write_json(run_dir / "runtime.json", runtime)
        _write_validation_predictions(run_dir, val_ids, val_y, probabilities, run_config)


def run_final_worker(config: dict, seeds: Iterable[int], device: str, resume: bool = True) -> None:
    selected = yaml.safe_load((ROOT / "configs" / "selected_hparams.yaml").read_text(encoding="utf-8"))
    for seed in seeds:
        for dataset in DATASETS:
            train_x_all, train_y_all, train_ids_all, val_x, val_y, val_ids = _dataset_arrays(
                dataset, config
            )
            train_x, train_y, train_ids = _training_subset(
                train_x_all, train_y_all, train_ids_all, config, int(seed)
            )
            for reducer_name in config["reducers"]["names"]:
                shared_id = f"A09-{dataset.upper()}-{reducer_name.upper()}-SHARED-S{seed}"
                shared_dir = ROOT / "runs" / "shared" / shared_id
                shared_hash = sha256_json(
                    {
                        "dataset": dataset,
                        "reducer": reducer_name,
                        "seed": seed,
                        "train_ids": train_ids.sample_id.tolist(),
                        "val_ids": val_ids.sample_id.tolist(),
                        "config": config["_config_hash"],
                    }
                )
                if not (resume and task_complete(shared_id, shared_hash)):
                    with task_context(shared_id, shared_hash):
                        bundle, angle_scaler, reducer_meta = fit_reducer(
                            reducer_name,
                            train_x,
                            train_y,
                            val_x,
                            val_y,
                            int(seed),
                            config,
                            shared_dir,
                            device=device,
                            resume=resume,
                        )
                        train_6 = angle_scaler.transform(bundle.transform(train_x))
                        val_6 = angle_scaler.transform(bundle.transform(val_x))
                        diagnostics = representation_diagnostics(
                            val_x,
                            val_6,
                            val_y,
                            int(seed),
                            int(config["statistics"]["distance_pairs"]),
                            int(config["statistics"]["trustworthiness_k"]),
                        )
                        atomic_write_json(shared_dir / "reducer_metadata.json", reducer_meta)
                        atomic_write_json(shared_dir / "representation_diagnostics.json", diagnostics)
                bundle = joblib.load(shared_dir / "reducer.joblib")
                angle_scaler = joblib.load(shared_dir / "angle_scaler.joblib")
                train_6 = angle_scaler.transform(bundle.transform(train_x))
                val_6 = angle_scaler.transform(bundle.transform(val_x))
                for head in config["heads"]["names"]:
                    try:
                        _fit_final_run(
                            dataset,
                            reducer_name,
                            head,
                            int(seed),
                            train_6,
                            train_y,
                            val_6,
                            val_y,
                            train_ids,
                            val_ids,
                            shared_dir,
                            config,
                            selected,
                            device,
                            resume,
                        )
                    except Exception as exc:
                        run_id = f"A09-{dataset.upper()}-{reducer_name.upper()}-{head.upper()}-S{seed}"
                        atomic_write_json(
                            ROOT / "runs" / run_id / "optimization_failure.json",
                            {"status": "optimization_failure", "error": repr(exc), "time": utc_now()},
                        )
            for head in ("lr", "rbf"):
                try:
                    _fit_full_reference(
                        dataset,
                        head,
                        int(seed),
                        train_x,
                        train_y,
                        val_x,
                        val_y,
                        train_ids,
                        val_ids,
                        config,
                        selected,
                        resume,
                    )
                except Exception as exc:
                    run_id = f"A09-{dataset.upper()}-FULL384-{head.upper()}-S{seed}"
                    atomic_write_json(
                        ROOT / "runs" / run_id / "optimization_failure.json",
                        {"status": "optimization_failure", "error": repr(exc), "time": utc_now()},
                    )


def verify_freeze() -> dict:
    protocol_path = ROOT / "frozen" / "protocol_v1.json"
    hash_path = ROOT / "frozen" / "protocol_v1.sha256"
    if not protocol_path.exists() or not hash_path.exists():
        raise PermissionError("Locked test cannot be opened before protocol freeze.")
    expected = hash_path.read_text(encoding="utf-8").strip()
    observed = sha256_file(protocol_path)
    if observed != expected:
        raise PermissionError("Frozen protocol hash mismatch; test access refused.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256_file(ROOT / "configs" / "selected_hparams.yaml") != protocol["selected_hparams_sha256"]:
        raise PermissionError("Selected hyperparameters changed after freeze.")
    return protocol


def evaluate_test(config: dict, resume: bool = True) -> None:
    protocol = verify_freeze()
    run_dirs = sorted(path for path in (ROOT / "runs").glob("A09-*") if path.is_dir())
    for run_dir in run_dirs:
        if (run_dir / "optimization_failure.json").exists():
            continue
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            continue
        run_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        task_id = f"test-{run_config['run_id']}"
        input_hash = sha256_json(
            {"run": sha256_file(config_path), "freeze": sha256_file(ROOT / "frozen" / "protocol_v1.json")}
        )
        if resume and task_complete(task_id, input_hash):
            continue
        with task_context(task_id, input_hash):
            dataset = run_config["dataset"]
            x_test, _, test_ids = load_embeddings(dataset, "test", allow_test=True)
            locked_path = ROOT / "data" / "manifests" / "locked" / f"{dataset}_test.csv"
            if sha256_file(locked_path) != protocol["datasets"][dataset]["locked_test_sha256"]:
                raise PermissionError(f"Locked test manifest changed for {dataset}")
            locked = pd.read_csv(locked_path).set_index("sample_id")
            y_test = locked.loc[test_ids.sample_id, "label"].to_numpy(dtype=np.int64)
            reducer_name = run_config["reducer"]
            if reducer_name == "full384":
                scaler = joblib.load(run_dir / "full_scaler.joblib")
                x_model = scaler.transform(np.asarray(x_test)).astype(np.float32)
            else:
                reducer = joblib.load(run_dir / "reducer.joblib")
                angle_scaler = joblib.load(run_dir / "angle_scaler.joblib")
                x_model = angle_scaler.transform(reducer.transform(np.asarray(x_test)))
            head = run_config["head"]
            if head in {"lr", "rbf"}:
                model = joblib.load(run_dir / "model.joblib")
                device = "cpu"
            else:
                device = "cuda:0" if head == "mlp" and torch.cuda.is_available() else "cpu"
                model = load_neural_model(
                    head,
                    run_dir / "checkpoint_best.pt",
                    run_config.get("backend", protocol["selected_backend"]),
                    device,
                )
            start = time.perf_counter()
            probabilities = model_probabilities(model, head, x_model, device)
            inference_seconds = time.perf_counter() - start
            metrics = classification_metrics(
                y_test, probabilities, int(config["statistics"]["ece_bins"])
            )
            metrics["inference_seconds"] = inference_seconds
            atomic_write_json(run_dir / "metrics.json", metrics)
            predictions = test_ids[["sample_id"]].copy()
            predictions["true_label"] = y_test
            predictions["predicted_label"] = probabilities.argmax(axis=1)
            for index in range(probabilities.shape[1]):
                predictions[f"prob_class_{index}"] = probabilities[:, index]
            for key in ("seed", "dataset", "reducer", "head"):
                predictions[key] = run_config[key]
            predictions.to_parquet(run_dir / "predictions.parquet", index=False)
            np.save(run_dir / "probabilities.npy", probabilities.astype(np.float32))


def completion_audit(config: dict) -> dict:
    expected = []
    for seed in config["project"]["final_seeds"]:
        for dataset in DATASETS:
            for reducer in config["reducers"]["names"]:
                for head in config["heads"]["names"]:
                    expected.append(f"A09-{dataset.upper()}-{reducer.upper()}-{head.upper()}-S{seed}")
            for head in ("lr", "rbf"):
                expected.append(f"A09-{dataset.upper()}-FULL384-{head.upper()}-S{seed}")
    rows = []
    for run_id in expected:
        run_dir = ROOT / "runs" / run_id
        required = [
            "config.yaml",
            "config_hash.txt",
            "train_ids.csv",
            "val_ids.csv",
            "test_manifest_hash.txt",
            "validation_metrics.json",
            "runtime.json",
            "metrics.json",
            "predictions.parquet",
        ]
        failure = (run_dir / "optimization_failure.json").exists()
        missing = [name for name in required if not (run_dir / name).exists()]
        rows.append(
            {
                "run_id": run_id,
                "status": "failed_recorded" if failure else ("complete" if not missing else "incomplete"),
                "missing": missing,
            }
        )
    report = {
        "expected_runs": len(expected),
        "complete": sum(row["status"] == "complete" for row in rows),
        "failed_recorded": sum(row["status"] == "failed_recorded" for row in rows),
        "incomplete": sum(row["status"] == "incomplete" for row in rows),
        "runs": rows,
    }
    atomic_write_json(ROOT / "results" / "completion_audit.json", report)
    return report

