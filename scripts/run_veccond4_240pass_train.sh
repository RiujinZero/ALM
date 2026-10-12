#!/usr/bin/env bash
# veccond4: dendritic predictor organ, 500MB capacity, 512d state vectors.
# Predictor dendrites replace far_w; each mode owns a K=8 dendritic predictor
# in 512d state space. Max capacity ~500MB (25000 modes x 17KB per mode).
# Run: screen -dmS v4train bash scripts/run_veccond4_240pass_train.sh
set -euo pipefail
REPO=/home/user/sonic-manifold-flow
PYTHON=/home/user/gtr_venv/bin/python
cd "$REPO"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PYTHON" -u \
    -m manifoldflow.experiments.dendritic_basin_shared_goal_convergence_veccond2 \
    --dataset-dir runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3 \
    --checkpoint runs/coadapt_probe_checkpoints/epoch_16_checkpoint.pkl \
    --snapshot-dir runs/shared_goal/veccond4_240pass_snapshots \
    --total-passes 240 \
    --k-inputs-per-branch 8 \
    --max-residual-modes 25000 \
    --max-context-assemblies 6250 \
    --content-dimensions 512 \
    --active-content-features 64 \
    --context-dimensions 128 \
    --ledger-event-dimensions 64 \
    --resume-from-mark 108560 \
    --report runs/shared_goal/convergence_veccond4_240pass.json
