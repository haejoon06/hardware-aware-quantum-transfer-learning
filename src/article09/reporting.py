from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import classification_report, confusion_matrix

from .config import ROOT, sha256_file, sha256_json
from .data import DATASETS
from .runner import completion_audit, verify_freeze
from .utils import atomic_write_json, task_complete, task_context


def _holm(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _bootstrap_mean_ci(values: np.ndarray, resamples: int, seed: int = 2026) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty paired sample.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def collect_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    test_rows = []
    validation_rows = []
    for run_dir in sorted((ROOT / "runs").glob("A09-*")):
        config_path = run_dir / "config.yaml"
        if not config_path.exists() or (run_dir / "optimization_failure.json").exists():
            continue
        import yaml

        run_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        common = {key: run_config[key] for key in ("run_id", "dataset", "reducer", "head", "seed")}
        if (run_dir / "validation_metrics.json").exists():
            metrics = json.loads((run_dir / "validation_metrics.json").read_text(encoding="utf-8"))
            validation_rows.append({**common, **metrics})
        if (run_dir / "metrics.json").exists():
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            runtime = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
            test_rows.append({**common, **metrics, **runtime})
    test = pd.DataFrame(test_rows)
    validation = pd.DataFrame(validation_rows)
    test.to_csv(ROOT / "results" / "metrics" / "all_test_metrics.csv", index=False)
    validation.to_csv(ROOT / "results" / "metrics" / "all_validation_metrics.csv", index=False)
    return test, validation


def _summary_table(test: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "mcc",
        "macro_roc_auc_ovr",
        "nll",
        "ece",
        "brier",
        "fit_seconds",
        "inference_seconds",
    ]
    available = [name for name in metrics if name in test.columns]
    summary = (
        test.groupby(["dataset", "reducer", "head"])[available]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def _paired_analyses(test: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    reducers = list(config["reducers"]["names"])
    resamples = int(config["statistics"]["bootstrap_resamples"])
    compression_rows = []
    head_rows = []
    friedman_rows = []
    reducer_comparisons = []
    expected_seeds = len(config["project"]["final_seeds"])

    for dataset in DATASETS:
        dataset_frame = test[test.dataset == dataset]
        full = dataset_frame[
            (dataset_frame["reducer"] == "full384") & (dataset_frame["head"] == "rbf")
        ].set_index("seed")
        for reducer in reducers:
            reduced = dataset_frame[
                (dataset_frame["reducer"] == reducer) & (dataset_frame["head"] == "rbf")
            ].set_index("seed")
            paired = full[["macro_f1"]].join(
                reduced[["macro_f1"]], lsuffix="_384", rsuffix="_6", how="inner"
            )
            difference = paired.macro_f1_384 - paired.macro_f1_6
            if len(difference) != expected_seeds:
                raise RuntimeError(
                    f"{dataset}/{reducer}: expected {expected_seeds} paired RBF seeds, "
                    f"found {len(difference)}."
                )
            low, high = _bootstrap_mean_ci(difference.to_numpy(), resamples)
            compression_rows.append(
                {
                    "dataset": dataset,
                    "reducer": reducer,
                    "rbf384_mean": paired.macro_f1_384.mean(),
                    "rbf6_mean": paired.macro_f1_6.mean(),
                    "compression_penalty_mean": difference.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "n_seeds": len(difference),
                }
            )
            for comparison_head in ("mlp", "rbf"):
                vqc = dataset_frame[
                    (dataset_frame["reducer"] == reducer) & (dataset_frame["head"] == "vqc")
                ].set_index("seed")
                classical = dataset_frame[
                    (dataset_frame["reducer"] == reducer)
                    & (dataset_frame["head"] == comparison_head)
                ].set_index("seed")
                paired_head = vqc[["macro_f1"]].join(
                    classical[["macro_f1"]], lsuffix="_vqc", rsuffix="_classical", how="inner"
                )
                difference_head = paired_head.macro_f1_vqc - paired_head.macro_f1_classical
                if len(difference_head) != expected_seeds:
                    raise RuntimeError(
                        f"{dataset}/{reducer}/vqc-vs-{comparison_head}: expected "
                        f"{expected_seeds} paired seeds, found {len(difference_head)}."
                    )
                low, high = _bootstrap_mean_ci(difference_head.to_numpy(), resamples)
                try:
                    p_value = float(wilcoxon(difference_head).pvalue)
                except ValueError:
                    p_value = 1.0
                head_rows.append(
                    {
                        "dataset": dataset,
                        "reducer": reducer,
                        "comparison": f"vqc_minus_{comparison_head}",
                        "mean_difference": difference_head.mean(),
                        "sd_difference": difference_head.std(ddof=1),
                        "ci_low": low,
                        "ci_high": high,
                        "wilcoxon_p": p_value,
                        "n_seeds": len(difference_head),
                    }
                )

        vqc_pivot = dataset_frame[
            (dataset_frame["reducer"].isin(reducers)) & (dataset_frame["head"] == "vqc")
        ].pivot(index="seed", columns="reducer", values="macro_f1")
        vqc_pivot = vqc_pivot.dropna()
        if len(vqc_pivot) != expected_seeds:
            raise RuntimeError(
                f"{dataset}: expected {expected_seeds} complete VQC seed blocks, "
                f"found {len(vqc_pivot)}."
            )
        if len(vqc_pivot) >= 3:
            stat, p_value = friedmanchisquare(*[vqc_pivot[name] for name in reducers])
            friedman_rows.append(
                {"dataset": dataset, "statistic": stat, "p_value": p_value, "n_seeds": len(vqc_pivot)}
            )
            local_rows = []
            for name in ("lda6", "nca6", "slp6"):
                try:
                    result = wilcoxon(vqc_pivot[name] - vqc_pivot["pca6"])
                    p = float(result.pvalue)
                    statistic = float(result.statistic)
                except ValueError:
                    p, statistic = 1.0, 0.0
                local_rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{name}_vs_pca6",
                        "statistic": statistic,
                        "p_value": p,
                    }
                )
            adjusted = _holm([row["p_value"] for row in local_rows])
            for row, corrected in zip(local_rows, adjusted):
                row["holm_p"] = corrected
                reducer_comparisons.append(row)

    head_frame = pd.DataFrame(head_rows)
    if not head_frame.empty:
        head_frame["holm_p"] = np.nan
        for dataset, indices in head_frame.groupby("dataset").groups.items():
            adjusted = _holm(head_frame.loc[indices, "wilcoxon_p"].tolist())
            head_frame.loc[indices, "holm_p"] = adjusted
    return {
        "compression_penalty": pd.DataFrame(compression_rows),
        "head_gaps": head_frame,
        "friedman": pd.DataFrame(friedman_rows),
        "reducer_wilcoxon": pd.DataFrame(reducer_comparisons),
    }


def _representation_table() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "runs" / "shared").glob("A09-*-SHARED-S*")):
        diag_path = path / "representation_diagnostics.json"
        if not diag_path.exists():
            continue
        parts = path.name.split("-")
        dataset = parts[1].lower()
        reducer = parts[2].lower()
        seed = int(parts[-1][1:])
        rows.append(
            {"dataset": dataset, "reducer": reducer, "seed": seed, **json.loads(diag_path.read_text(encoding="utf-8"))}
        )
    return pd.DataFrame(rows)


def _per_class_tables(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    validation = pd.read_csv(ROOT / "results" / "metrics" / "all_validation_metrics.csv")
    candidates = validation[validation.reducer != "full384"].copy()
    ranked = (
        candidates.groupby(["dataset", "reducer", "head"], as_index=False).macro_f1.mean()
        .sort_values(["dataset", "macro_f1"], ascending=[True, False])
        .groupby("dataset")
        .head(1)
    )
    for _, winner in ranked.iterrows():
        subset = test[
            (test["dataset"] == winner["dataset"])
            & (test["reducer"] == winner["reducer"])
            & (test["head"] == winner["head"])
        ]
        for _, run in subset.iterrows():
            predictions = pd.read_parquet(ROOT / "runs" / run.run_id / "predictions.parquet")
            report = classification_report(
                predictions.true_label,
                predictions.predicted_label,
                output_dict=True,
                zero_division=0,
            )
            for label, values in report.items():
                if not str(label).isdigit():
                    continue
                rows.append(
                    {
                        "dataset": winner["dataset"],
                        "reducer": winner["reducer"],
                        "head": winner["head"],
                        "seed": run.seed,
                        "class_label": int(label),
                        "precision": values["precision"],
                        "recall": values["recall"],
                        "f1": values["f1-score"],
                        "support": values["support"],
                    }
                )
    return pd.DataFrame(rows)


def build_all_reports(config: dict, resume: bool = True) -> None:
    verify_freeze()
    audit = completion_audit(config)
    if audit["incomplete"]:
        raise RuntimeError("Cannot report while final runs are incomplete.")
    run_artifact_paths = sorted((ROOT / "runs").glob("A09-*/metrics.json"))
    run_artifact_paths += sorted((ROOT / "runs").glob("A09-*/validation_metrics.json"))
    run_artifact_paths += sorted((ROOT / "runs").glob("A09-*/predictions.parquet"))
    input_hash = sha256_json(
        {
            "code": sha256_file(Path(__file__)),
            "artifacts": {str(path): sha256_file(path) for path in run_artifact_paths},
        }
    )
    if resume and task_complete("report", input_hash):
        return
    with task_context("report", input_hash):
        test, validation = collect_metrics()
        summary = _summary_table(test)
        summary.to_csv(ROOT / "results" / "tables" / "main_performance_summary.csv", index=False)
        analyses = _paired_analyses(test, config)
        required_analysis_rows = {
            "compression_penalty": len(DATASETS) * len(config["reducers"]["names"]),
            "head_gaps": len(DATASETS) * len(config["reducers"]["names"]) * 2,
            "friedman": len(DATASETS),
            "reducer_wilcoxon": len(DATASETS) * (len(config["reducers"]["names"]) - 1),
        }
        for name, frame in analyses.items():
            if len(frame) != required_analysis_rows[name] or frame.isna().any().any():
                raise RuntimeError(
                    f"Invalid {name} table: expected {required_analysis_rows[name]} complete rows, "
                    f"found {len(frame)} rows (contains_na={frame.isna().any().any()})."
                )
            frame.to_csv(ROOT / "results" / "statistics" / f"{name}.csv", index=False)
        representation = _representation_table()
        representation.to_csv(ROOT / "results" / "tables" / "representation_diagnostics.csv", index=False)
        if not representation.empty:
            (
                representation.groupby(["dataset", "reducer"])
                .agg(["mean", "std"])
                .reset_index()
                .to_csv(ROOT / "results" / "tables" / "representation_diagnostics_summary.csv", index=False)
            )
        per_class = _per_class_tables(test)
        if per_class.empty or per_class[["precision", "recall", "f1", "support"]].isna().any().any():
            raise RuntimeError("Per-class report is empty or contains missing metric values.")
        per_class.to_csv(ROOT / "results" / "tables" / "per_class_metrics.csv", index=False)
        atomic_write_json(
            ROOT / "results" / "report_manifest.json",
            {
                "generated_from_runs": len(test),
                "validation_rows": len(validation),
                "completion_audit": audit,
                "files": [
                    str(path.relative_to(ROOT))
                    for path in sorted((ROOT / "results").rglob("*.csv"))
                ],
            },
        )

