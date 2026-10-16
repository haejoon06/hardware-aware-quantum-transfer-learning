from __future__ import annotations

import numpy as np
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


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences > left) & (confidences <= right)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, ece_bins: int = 15
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(probabilities.shape[1])
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
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_roc_auc_ovr": auc,
        "nll": float(log_loss(y_true, probabilities, labels=labels)),
        "ece": expected_calibration_error(y_true, probabilities, bins=ece_bins),
        "brier": multiclass_brier(y_true, probabilities),
    }

