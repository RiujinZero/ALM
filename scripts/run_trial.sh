#!/usr/bin/env bash
# Quick trial runner — tweak the variables below and run:
#   bash scripts/run_trial.sh
# Output lands in runs/shared_goal/trial_<NAME>_snapshots/ and
# runs/shared_goal/trial_<NAME>_report.json.
# Set SCREEN=1 to detach into a screen session named after the trial.
set -euo pipefail
REPO=/home/user/sonic-manifold-flow
PYTHON=/home/user/gtr_venv/bin/python

# ── trial identity ────────────────────────────────────────────────────────────
NAME=my_trial          # used for snapshot dir and report filename

# ── data / checkpoint ────────────────────────────────────────────────────────
DATASET=runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3
CHECKPOINT=runs/coadapt_probe_checkpoints/epoch_16_checkpoint.pkl

# ── training length ───────────────────────────────────────────────────────────
TOTAL_PASSES=8         # 8 = fast smoke-test (~2 min); 240 = full run (~8 h)

# ── state-space dimensions ───────────────────────────────────────────────────
CONTENT_DIM=512        # state vector size seen by the organ and arms
ACTIVE_FEATURES=64     # active content features passed to the base decoder
CONTEXT_DIM=128        # organ context vector dimension
LEDGER_DIM=64          # ledger event dimension

# ── organ capacity ────────────────────────────────────────────────────────────
K_INPUTS=8             # dendrite fan-in per predictor branch (0 = no predictor)
MAX_MODES=25000        # residual modes  (~14 KB each at 512d → 25k ≈ 350 MB)
MAX_ASSEMBLIES=6250    # context assemblies (≈ MAX_MODES / 4)

# ── launch ───────────────────────────────────────────────────────────────────
SNAPSHOT_DIR=runs/shared_goal/trial_${NAME}_snapshots
REPORT=runs/shared_goal/trial_${NAME}_report.json

cd "$REPO"

CMD=(
    PYTHONUNBUFFERED=1 PYTHONPATH=src "$PYTHON" -u
    -m manifoldflow.experiments.dendritic_basin_shared_goal_convergence_veccond2
    --dataset-dir "$DATASET"
    --checkpoint  "$CHECKPOINT"
    --snapshot-dir "$SNAPSHOT_DIR"
    --total-passes "$TOTAL_PASSES"
    --k-inputs-per-branch "$K_INPUTS"
    --max-residual-modes  "$MAX_MODES"
    --max-context-assemblies "$MAX_ASSEMBLIES"
    --content-dimensions "$CONTENT_DIM"
    --active-content-features "$ACTIVE_FEATURES"
    --context-dimensions "$CONTEXT_DIM"
    --ledger-event-dimensions "$LEDGER_DIM"
    --report "$REPORT"
)

if [ "${SCREEN:-0}" = "1" ]; then
    screen -dmS "$NAME" bash -c "${CMD[*]}"
    echo "launched in screen session '$NAME'"
    echo "attach: screen -r $NAME"
else
    echo "trial: $NAME"
    echo "passes: $TOTAL_PASSES  modes: $MAX_MODES  k: $K_INPUTS  dim: ${CONTENT_DIM}d"
    echo "report: $REPORT"
    echo "---"
    "${CMD[@]}"
    echo "---"
    python -c "
import json, sys
r = json.load(open('$REPORT'))
marks = r.get('marks', {})
if marks:
    last = sorted(marks, key=lambda m: int(m))[-1]
    arms = marks[last].get('arm_results', {})
    for arm, res in arms.items():
        g = res.get('long_interruption_sampled_mean_margin_delta', float('nan'))
        print(f'  {arm}: {g:+.4f}')
" 2>/dev/null || true
fi
