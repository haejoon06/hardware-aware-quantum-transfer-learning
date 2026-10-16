# Article 10 experimental protocol

## Research question

This study asks how a validation-selected quantum transfer-learning head changes when it leaves ideal statevector conditions. The controlled factors are circuit width, circuit depth, data re-uploading, entanglement topology, finite-shot measurement, circuit-level noise, and hardware connectivity.

## Upstream lock

The image splits, duplicate audit, DINOv2 ViT-S/14 embeddings, training budget, and SLP-6 representation are inherited unchanged from Article 9. SLP-6 was selected using validation evidence only. Article 10 never refits the visual encoder and never uses test labels for architecture selection.

## Stages

1. **Smoke:** a tiny checkpoint/resume and gradient test.
2. **Screen:** 4, 6, 8, and 10 qubits crossed with depths 1, 2, 4, and 6 on CIFAR-10 and PathMNIST, five seeds, analytic statevector simulation.
3. **Ablate:** the two best width-depth pairs per dataset are crossed with encode-once/data-reupload and linear/ring entanglement.
4. **Final:** the validation-selected configuration is trained for ten paired seeds. Selection is frozen before any test array can be loaded.
5. **Shots:** analytic and 128/512/1024/4096-shot estimates are evaluated with 30 Monte Carlo repetitions. Reported outcomes include macro-F1, NLL, ECE, Brier score, probability drift, and label-flip rate.
6. **Noise:** circuit-level depolarizing and readout noise are evaluated with Qiskit Aer at three predeclared severities on fixed stratified samples.
7. **Transpile:** logical circuits are transpiled to all-to-all, ring, and linear coupling maps; depth, two-qubit count, and SWAP count are recorded.
8. **Confirm:** EuroSAT receives only the selected robust configuration and a preregistered 6-qubit, depth-2 reference, not the full search.
9. **Report:** machine-readable tables, figures, notebook, and manuscript-ready narrative are generated from saved results.

The test set is opened once, after `state/FINAL_SELECTION_FROZEN.json` exists and its input hashes match. Physical QPU execution is intentionally disabled unless credentials, backend choice, queue/cost implications, and explicit authorization are supplied later.

## Reproducibility and recovery

Every run writes a configuration hash, epoch checkpoint, training history, validation predictions, and a success sentinel. A resumed run rejects changed protocol, split, reducer, or scaler hashes. The launcher writes its PID and separate stdout/stderr logs. Power loss therefore requires only relaunching with `--resume`.

