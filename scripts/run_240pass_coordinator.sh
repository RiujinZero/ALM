#!/usr/bin/env bash
# Waits for all four 240-pass eval runs to complete, then plots the final chart.
# Runs from WSL. Copies finished chart to the Windows-visible mount.

set -euo pipefail

REPO=/home/user/sonic-manifold-flow
PYTHON=/home/user/gtr_venv/bin/python
WIN_DEST="/mnt/c/Users/user/Documents/projects/sonic-manifold-flow/runs/shared_goal"
CHART="$REPO/runs/shared_goal/veccond_curves_all.png"

# Expected eval file counts after 240-pass runs (12 marks per variant)
# v1: 4 arms * 12 marks = 48 files; v2: 12 files
V1_TARGET=48
V2_TARGET=12

DIRS=(
    "veccond_240pass_evals:$V1_TARGET"
    "veccond2_240pass_evals:$V2_TARGET"
    "veccond_240pass_2x_evals:$V1_TARGET"
    "veccond2_240pass_2x_evals:$V2_TARGET"
)

echo "[coord] Waiting for all 240-pass evals to complete..."

while true; do
    all_done=true
    for entry in "${DIRS[@]}"; do
        dir="${entry%%:*}"
        target="${entry##*:}"
        count=$(find "$REPO/runs/shared_goal/$dir" -name "eval_*.json" 2>/dev/null | wc -l) || count=0
        if [ "$count" -lt "$target" ]; then
            echo "[coord] $dir: $count/$target done"
            all_done=false
        fi
    done

    if $all_done; then
        echo "[coord] All evals complete. Plotting..."
        break
    fi

    sleep 120
done

# Sync plot script from Windows mount in case it was updated
cp "/mnt/c/Users/user/Documents/projects/sonic-manifold-flow/scripts/plot_veccond_curves.py" \
   "$REPO/scripts/plot_veccond_curves.py"

PYTHONPATH=src "$PYTHON" "$REPO/scripts/plot_veccond_curves.py"

# Copy chart to Windows-visible location
cp "$CHART" "$WIN_DEST/veccond_curves_all.png"
echo "[coord] Chart written to $WIN_DEST/veccond_curves_all.png"
echo "[coord] DONE"
