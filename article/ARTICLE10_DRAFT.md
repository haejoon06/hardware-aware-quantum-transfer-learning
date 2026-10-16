# From Statevector to Hardware-Aware Quantum Transfer Learning: Shot, Noise, Calibration, and Trainability Effects Across Visual Domains

## Status

This manuscript scaffold is generated from protocol `76f3d7c0330632b86c92d140f2ded5a67944176d07ad001e46992156dbcfb22b`. Numerical claims must be populated only from the machine-readable tables after the completion audit passes.

## Frozen validation selection

```json
{
  "candidate_summary_sha256": "f1e448a944013cb5f45742d24d6baeea643e76d6381a479a34da11e895bc8f05",
  "confirmatory_robust_spec": {
    "depth": 6,
    "encoding": "once",
    "entanglement": "linear",
    "qubits": 8
  },
  "frozen_at": "2026-10-15T15:25:23.076269+00:00",
  "primary": {
    "cifar10": {
      "spec": {
        "depth": 4,
        "encoding": "once",
        "entanglement": "linear",
        "qubits": 8
      },
      "validation_macro_f1_mean": 0.9259151566044714,
      "validation_macro_f1_std": 0.00917434897913264,
      "validation_nll_mean": 0.4348874823479848
    },
    "pathmnist": {
      "spec": {
        "depth": 6,
        "encoding": "once",
        "entanglement": "linear",
        "qubits": 8
      },
      "validation_macro_f1_mean": 0.8145223727084694,
      "validation_macro_f1_std": 0.012527595571906705,
      "validation_nll_mean": 0.8079546082160942
    }
  },
  "protocol_sha256": "76f3d7c0330632b86c92d140f2ded5a67944176d07ad001e46992156dbcfb22b",
  "selection_uses_test": false,
  "test_manifest_hashes": {
    "cifar10": "40393aa0e1302b8056d044995034019752937a009fd4a4259ea66b930ca432aa",
    "eurosat": "9803ce896325a3366cabf7fbf93afcea906a87e8781c0ef998f4edad42ebd9d0",
    "pathmnist": "bd14ecfb9726420b4c66f77d13da56a0c482e77c6f76010c1cc30b4bf98ef5e1"
  }
}
```

## Intended contribution

The study separates ideal-circuit performance from trainability, finite-shot uncertainty, circuit-level noise, calibration degradation, and connectivity-aware transpilation while holding the visual representation and data splits fixed. It makes no claim of quantum advantage or speedup.
