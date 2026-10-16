# Final model artifacts

This directory contains only validation-selected final and confirmatory quantum-head checkpoints, not the hundreds of screening checkpoints. Each run directory includes:

- `checkpoint_best.pt`: best validation checkpoint;
- `config.json`: circuit and seed metadata;
- `validation_metrics.json`: selection-safe metrics;
- `training_history.csv`: convergence and gradient diagnostics;
- `run_hash.txt`: hash binding the run to its protocol, split, reducer, and scaler.

The complete directory is also distributed as `article10_final_models.zip`. SLP-6 reducers and DINOv2 embeddings remain versioned by hash in the prerequisite Article 9 workspace because duplicating them would create inconsistent upstream copies.

