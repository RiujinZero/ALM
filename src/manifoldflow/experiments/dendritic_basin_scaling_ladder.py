"""Run and aggregate the resumable Phase 13 ALM byte-class ladder."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    BYTE_CLASSES,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALID_PATH,
)


def _run_prefix(value: str) -> str:
    """Accept a single path component for isolated ladder artifacts."""
    if not value or value in {".", ".."} or Path(value).name != value:
        raise argparse.ArgumentTypeError(
            "run prefix must be one non-empty path component"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--train-bytes", type=int, default=10_000_000)
    parser.add_argument("--eval-bytes", type=int, default=1_000_000)
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=BYTE_CLASSES,
        default=["2MB", "16MB", "64MB"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-prefix", type=_run_prefix, default="alm_phase13")
    parser.add_argument("--contextual-residual", action="store_true")
    parser.add_argument("--utility-gated-persistence", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=250_000)
    parser.add_argument("--progress-every", type=int, default=250_000)
    parser.add_argument("--wait-for-report", action="append", type=Path, default=[])
    parser.add_argument("--include-report", action="append", type=Path, default=[])
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("runs/alm_phase13_10mb_summary.json"),
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _valid_report(path: Path, *, train_bytes: int, eval_bytes: int, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        int(report.get("train_bytes", -1)) == train_bytes
        and int(report.get("eval_bytes", -1)) == eval_bytes
        and int(report.get("config", {}).get("seed", -1)) == seed
    )


def _run_one(args: argparse.Namespace, alm_class: str, seed: int) -> Path:
    tag = alm_class.lower()
    run_dir = args.runs_dir / f"{args.run_prefix}_{tag}_10mb_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint.pkl"
    report = run_dir / "report.json"
    if _valid_report(
        report,
        train_bytes=args.train_bytes,
        eval_bytes=args.eval_bytes,
        seed=seed,
    ):
        return report

    command = [
        sys.executable,
        "-m",
        "manifoldflow.experiments.dendritic_basin_online_stream_smoke",
        "--train-path",
        args.train_path,
        "--valid-path",
        args.valid_path,
        "--train-bytes",
        str(args.train_bytes),
        "--eval-bytes",
        str(args.eval_bytes),
        "--alm-class",
        alm_class,
        "--seed",
        str(seed),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--progress-every",
        str(args.progress_every),
        "--report",
        str(report),
    ]
    if checkpoint.exists():
        command.extend(("--resume-from", str(checkpoint)))
    else:
        command.extend(("--checkpoint", str(checkpoint)))
    if getattr(args, "contextual_residual", False):
        command.append("--contextual-residual")
    if getattr(args, "utility_gated_persistence", False):
        command.append("--utility-gated-persistence")

    with (run_dir / "stdout.log").open("a", encoding="utf-8") as stdout:
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{alm_class} seed {seed} failed; inspect {run_dir / 'stderr.log'}"
        )
    if not _valid_report(
        report,
        train_bytes=args.train_bytes,
        eval_bytes=args.eval_bytes,
        seed=seed,
    ):
        raise RuntimeError(f"{alm_class} seed {seed} produced no valid report")
    return report


def summarize_reports(paths: list[Path]) -> dict[str, Any]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        population = report["population"]
        anatomy = report["anatomy"]
        live = int(report["live_relations"])
        births = int(population["births"])
        row = {
            "class": report["alm_class"],
            "seed": int(report["config"]["seed"]),
            "eval_bpb": float(report["eval_model_bits_per_byte"]),
            "train_bytes_per_second": float(report["train_bytes_per_second"]),
            "effective_byte_loop": report.get(
                "effective_byte_loop", "python_step"
            ),
            "adaptation_rng": report.get("config", {}).get(
                "adaptation_rng", "global"
            ),
            "contextual_residual": bool(
                report.get("config", {}).get("contextual_residual", False)
            ),
            "serialized_alm_bytes": int(report["serialized_alm_bytes"]),
            "births_per_kilobyte": float(report["births_per_kilobyte"]),
            "prunes_per_kilobyte": float(report["prunes_per_kilobyte"]),
            "live_birth_survival_fraction": live / max(births, 1),
            "mean_branch_age_bytes": float(anatomy["mean_branch_age_bytes"]),
            "depth_histogram": anatomy["depth_histogram"],
            "lifecycle_histogram": anatomy["lifecycle_histogram"],
            "corpus_bytes_per_alm_byte": float(
                report["corpus_bytes_per_alm_byte"]
            ),
        }
        rows.append(row)
        grouped.setdefault(row["class"], []).append(row)

    aggregates: dict[str, Any] = {}
    for alm_class, class_rows in grouped.items():
        bpbs = [row["eval_bpb"] for row in class_rows]
        aggregates[alm_class] = {
            "seed_count": len(class_rows),
            "eval_bpb_mean": mean(bpbs),
            "eval_bpb_sample_sd": stdev(bpbs) if len(bpbs) > 1 else None,
            "train_bytes_per_second_mean": mean(
                row["train_bytes_per_second"] for row in class_rows
            ),
            "serialized_alm_bytes_mean": mean(
                row["serialized_alm_bytes"] for row in class_rows
            ),
            "births_per_kilobyte_mean": mean(
                row["births_per_kilobyte"] for row in class_rows
            ),
            "prunes_per_kilobyte_mean": mean(
                row["prunes_per_kilobyte"] for row in class_rows
            ),
            "live_birth_survival_fraction_mean": mean(
                row["live_birth_survival_fraction"] for row in class_rows
            ),
        }
    return {"runs": rows, "classes": aggregates}


def main() -> None:
    args = parse_args()
    while any(not path.exists() for path in args.wait_for_report):
        time.sleep(30)

    reports = list(args.include_report)
    for seed in args.seeds:
        with ThreadPoolExecutor(max_workers=len(args.classes)) as executor:
            futures = [
                executor.submit(_run_one, args, alm_class, seed)
                for alm_class in args.classes
            ]
            reports.extend(future.result() for future in futures)
        _atomic_json(args.summary, summarize_reports(reports))


if __name__ == "__main__":
    main()
