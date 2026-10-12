#!/usr/bin/env bash
set -euo pipefail
REPO=/home/user/sonic-manifold-flow
PYTHON=/home/user/gtr_venv/bin/python
DATASET=runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3
CHECKPOINT=runs/coadapt_probe_checkpoints/epoch_16_checkpoint.pkl
SNAPSHOT_DIR=runs/shared_goal/veccond4_240pass_snapshots
TRAIN_REPORT=runs/shared_goal/convergence_veccond4_240pass.json
EVAL_DIR=runs/shared_goal/veccond4_240pass_evals
DIGEST_DIR=runs/shared_goal/veccond4_240pass_digests
ARM_NAMES=(A_k1_nooledger B_k1_ledger C_k4_nooledger D_k4_ledger)
cd "$REPO"

eval_and_rotate() {
    local MARK="$1"
    local PREV_MARK="$2"
    local REPORT="$EVAL_DIR/eval_mark_$(printf '%08d' "$MARK").json"
    [ -f "$REPORT" ] && return 0
    for ARM in "${ARM_NAMES[@]}"; do
        [ -f "$SNAPSHOT_DIR/${ARM}_mark_$(printf '%08d' "$MARK").pkl" ] || return 1
    done
    mkdir -p "$EVAL_DIR" "$DIGEST_DIR"
    echo "[eval-sequel] evaluating mark $MARK ..."
    PYTHONUNBUFFERED=1 PYTHONPATH=src "$PYTHON" -u \
        -m manifoldflow.experiments.dendritic_basin_shared_goal_convergence_veccond2_evaluate \
        --dataset-dir "$DATASET" --checkpoint "$CHECKPOINT" \
        --snapshot-dir "$SNAPSHOT_DIR" --mark "$MARK" \
        --report "$REPORT" \
        --digest "$DIGEST_DIR/digest_mark_$(printf '%08d' "$MARK").txt" \
        --content-dimensions 512 --active-content-features 64 \
        --context-dimensions 128 --ledger-event-dimensions 64
    echo "[eval-sequel] mark $MARK done"
    if [ -n "$PREV_MARK" ] && [ "$PREV_MARK" != "$MARK" ]; then
        for ARM in "${ARM_NAMES[@]}"; do
            rm -f "$SNAPSHOT_DIR/${ARM}_mark_$(printf '%08d' "$PREV_MARK").pkl"
        done
        echo "[eval-sequel] deleted arm bases for mark $PREV_MARK"
    fi
    return 0
}

echo "[eval-sequel] waiting for veccond4 240-pass training..."
LAST_EVALED=""
while ! [ -f "$TRAIN_REPORT" ]; do
    for ORGAN_PKL in "$SNAPSHOT_DIR"/organ_mark_*.pkl; do
        [ -f "$ORGAN_PKL" ] || continue
        MARK=$(echo "$ORGAN_PKL" | sed 's/.*organ_mark_\([0-9]*\)\.pkl/\1/' | sed 's/^0*//' | awk '{print ($1=="" ? 0 : $1)}')
        eval_and_rotate "$MARK" "$LAST_EVALED" && LAST_EVALED="$MARK" || true
    done
    sleep 60
done
echo "[eval-sequel] training done: $TRAIN_REPORT"
for ORGAN_PKL in "$SNAPSHOT_DIR"/organ_mark_*.pkl; do
    [ -f "$ORGAN_PKL" ] || continue
    MARK=$(echo "$ORGAN_PKL" | sed 's/.*organ_mark_\([0-9]*\)\.pkl/\1/' | sed 's/^0*//' | awk '{print ($1=="" ? 0 : $1)}')
    eval_and_rotate "$MARK" "$LAST_EVALED" && LAST_EVALED="$MARK" || true
done
echo "[eval-sequel] all veccond4 240-pass evals complete."
