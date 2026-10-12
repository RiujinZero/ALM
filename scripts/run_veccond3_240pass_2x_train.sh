#!/usr/bin/env bash
# Launch veccond3 128d training (dendritic context assemblies, K=8, 2x dims).
# Run from WSL in a screen session:  screen -dmS v3train2x bash scripts/run_veccond3_240pass_2x_train.sh
set -euo pipefail
REPO=/home/user/sonic-manifold-flow
PYTHON=/home/user/gtr_venv/bin/python
cd "$REPO"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PYTHON" -u \
    -m manifoldflow.experiments.dendritic_basin_shared_goal_convergence_veccond2 \
    --dataset-dir runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3 \
    --checkpoint runs/coadapt_probe_checkpoints/epoch_16_checkpoint.pkl \
    --snapshot-dir runs/shared_goal/veccond3_240pass_2x_snapshots \
    --total-passes 240 \
    --k-inputs-per-branch 8 \
    --content-dimensions 128 --active-content-features 32 \
    --context-dimensions 48 --ledger-event-dimensions 48 \
    --report runs/shared_goal/convergence_veccond3_240pass_2x.json
