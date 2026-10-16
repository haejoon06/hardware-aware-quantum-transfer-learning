# Hardware-Aware Quantum Transfer Learning

Reproducible code and frozen results for **“From Statevector to Hardware-Aware Quantum Transfer Learning: Shot, Noise, Calibration, and Trainability Effects Across Visual Domains.”**

The study holds the audited image splits, frozen DINOv2 ViT-S/14 embeddings, and validation-selected SLP-6 representation fixed, then isolates the effects of circuit width, depth, data re-uploading, entanglement topology, finite-shot measurement, circuit-level noise, calibration, and hardware connectivity across CIFAR-10, PathMNIST, and EuroSAT.

## Main results

| Dataset | Validation macro-F1 | Analytic test macro-F1 | Selected circuit |
|---|---:|---:|---|
| CIFAR-10 | 0.9255 ± 0.0075 | 0.9145 ± 0.0069 | 8 qubits, depth 4, linear |
| PathMNIST | 0.8108 ± 0.0117 | 0.7422 ± 0.0283 | 8 qubits, depth 6, linear |
| EuroSAT (confirmatory) | 0.8501 ± 0.0078 | see result tables | 8 qubits, depth 6, linear |

At 128 shots, mean test macro-F1 was 0.9137 on CIFAR-10 and 0.7397 on PathMNIST. Circuit-level Aer experiments additionally quantify macro-F1, NLL, ECE, Brier score, probability drift, and prediction-flip rate under three preregistered depolarizing/readout-noise profiles. These results do **not** establish quantum advantage or speedup.

## Repository contents

- `configs/`: frozen protocol.
- `src/article10/`: checkpointed experimental pipeline.
- `notebooks/`: executed notebook with embedded outputs.
- `results/tables/`: machine-readable metrics and bootstrap intervals.
- `results/figures/`: manuscript-ready PNG and SVG figures generated from results.
- `models/`: final and confirmatory quantum-head checkpoints.
- `scripts/`: Windows launch and status helpers.
- `tests/`: shape, gradient, and probability tests.
- `EXPERIMENT_PROTOCOL.md`: validation/test separation and experimental stages.

## Prerequisite artifacts

The full rerun deliberately reuses immutable Article 9 artifacts instead of silently regenerating different splits. Clone and complete the Article 9 pipeline first:

<https://github.com/haejoon06/disentangling-representation-compression-quantum-head-effects>

Its workspace must contain `data/embeddings`, `data/manifests`, `runs/shared`, and `frozen/protocol_v1.json`. Point this repository to that workspace:

```powershell
$env:ARTICLE09_ROOT = 'D:\path\to\article09-workspace'
```

Raw public datasets and large DINOv2 embeddings are not duplicated in Git. Their hashes are recorded in `state/PROTOCOL_LOCK.json` and `state/FINAL_SELECTION_FROZEN.json`.

## Reproduce or resume

The environment used for the reported results was `quantum_env` with Python 3.11, PyTorch 2.5.1+cu124, PennyLane 0.44.1, Qiskit 2.4.1, and Qiskit Aer 0.17.2. Install the project dependencies, set `ARTICLE09_ROOT`, then run:

```powershell
python run_article10.py --stage all --resume
```

On the original Windows workstation, the background launcher is:

```powershell
.\scripts\launch_article10.ps1
```

Every run records an input hash, epoch checkpoint, training history, validation probabilities, and success sentinel. Selection code cannot read test embeddings until `FINAL_SELECTION_FROZEN.json` exists and the locked test-manifest hashes match.

## Hardware scope

The reported quantum experiments use analytic statevector simulation and Qiskit Aer finite-shot/noise simulation. Connectivity-aware transpilation is included. Physical-QPU execution is not claimed and remains disabled unless separately configured and authorized.

## Authors

- Fajri Yanuar Shiddiq Juanda — [ORCID 0009-0007-8894-0265](https://orcid.org/0009-0007-8894-0265)
- Kiki Ahmad Baihaqi — [ORCID 0000-0002-2430-9235](https://orcid.org/0000-0002-2430-9235)

Both authors are affiliated with the Informatics Engineering Study Program, Faculty of Computer Science, Universitas Buana Perjuangan Karawang, Indonesia.

## License

Code is released under the MIT License. Dataset licenses and terms remain with their original providers.

