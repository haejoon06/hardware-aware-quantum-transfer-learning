from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from article10.pipeline import CircuitSpec, QuantumHead, softmax_numpy


def test_quantum_head_shapes_and_gradients():
    for spec in (CircuitSpec(4, 1, "once", "linear"), CircuitSpec(6, 2, "reupload", "ring")):
        model = QuantumHead(3, spec)
        x = torch.zeros(5, 6)
        logits = model(x)
        assert logits.shape == (5, 3)
        logits.sum().backward()
        assert torch.isfinite(model.theta.grad).all()


def test_softmax_rows_sum_to_one():
    probabilities = softmax_numpy(np.array([[1.0, 2.0], [-3.0, 4.0]]))
    assert np.allclose(probabilities.sum(axis=1), 1.0)

