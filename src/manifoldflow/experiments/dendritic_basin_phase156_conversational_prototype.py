"""Fresh-start native ALM prototype: mixed wiki/lmsys/smoltalk corpus,
targeting conversational coherence rather than BPB. Full memory stack
enabled from the first byte -- deep bands (rolling hash), combined
salience, Affective Intensity + replay, and tiered persistence -- per the
"go all in" launch decision. BPB is tracked but is a secondary signal; the
generation panel (including the multi-turn probes) is the primary one.

Unlike phase126 (which continues an existing Python-checkpoint lineage via
a one-time native conversion), this starts a brand-new NativeGrowableCortex
directly, so there is no epoch-1 ancestor to convert. "Epoch" here means
one `--epoch-bytes` slice of the corpus, not a full pass over it -- the
corpus is large enough (18.5GB) that waiting for a full pass before the
first generation-panel eval would take far too long to be useful. Epochs
walk the corpus sequentially and wrap back to the start if `target-epochs`
times `epoch-bytes` exceeds the corpus size.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.native_cortex import NativeGrowableCortex
from manifoldflow.experiments.dendritic_basin_phase122_full_corpus_4c import (
    _atomic_json,
    _atomic_pickle,
    _evaluate,
    _generation_panel,
    _partition,
)
from manifoldflow.experiments.dendritic_basin_phase126_three_epoch_generation import (
    _develop,
    _load_any_cortex,
)


def _new_native_cortex(
    long_range_band_count: int = 4,
    episodic_enabled: bool = False,
    trust_delta_clip: float = 2.0,
    episodic_k: int = 14,
    episodic_s: int = 6,
    episodic_strength: float = 4.0,
    long_range_bank: bool = False,
    long_range_context_bytes: int = 12,
    long_range_strength: float = 8.0,
) -> NativeGrowableCortex:
    return NativeGrowableCortex(
        max_context_bytes=6,
        # long_range_bank (not long_range_bands/deep bands): a SINGLE
        # additional confirmation-gated dendritic bank at
        # long_range_context_bytes, chained into the same
        # accumulate-evidence-before-trusting nursery/promotion contract
        # ordinary sequence banks already use (grow_sequence_nursery's own
        # "total >= confirmations" rule) -- computational substrate, not
        # deep bands' sparse/tiered memory. max_context_bytes=6 above is
        # the ordinary banks' own ceiling (packed_history_ can hold up to
        # 8, but this run uses 6); long_range_context_bytes=12 is double
        # that, keyed via LongRangeKey since packed_history_ can't hold
        # more than 8 bytes -- already validated (53%->95% recall across a
        # gap the ordinary banks can't see across at all) and already
        # fully serialized (unlike several other fields added this phase),
        # so no persistence-gap risk on resume. See docs/ALM_V1_Architecture.md.
        long_range_bank=long_range_bank,
        long_range_context_bytes=long_range_context_bytes,
        long_range_strength=long_range_strength,
        max_sequence_nursery_paths=50_000,
        connectivity_pruning=True,
        adaptive_competition=True,
        competition_exponent=1.0,
        trust_gamma=0.5,
        trust_learning_rate=0.2,
        attribution_gate_threshold=0.5,
        # Default (2.0) was found to saturate hard on real corpus text --
        # 49.8% of learned contexts sitting at the exact trust ceiling after
        # the first 500MB baseline. A standalone comparison (same corpus,
        # only this clip varied) showed tightening it de-saturates the
        # distribution substantially with no bpb cost (identical to
        # noise), confirming the saturation is a step-size artifact of the
        # update rule rather than a fundamental limit -- see
        # docs/ALM_V1_Architecture.md.
        trust_delta_clip=trust_delta_clip,
        burst_surprise_signal=True,
        long_range_bands=True,
        long_range_band_depths=(12, 16, 24, 32),
        long_range_band_count=long_range_band_count,
        long_range_rolling_hash=True,
        long_range_band_confirmations=3,
        long_range_importance_bits=4.0,
        long_range_importance_bonus=3,
        # 200,000/band (800,000 total across 4 bands) was sized before
        # deep bands existed as a byte-budget-tracked concept -- sleep's
        # consolidate_to_byte_budget only ever prunes the ordinary
        # sequence banks (deep bands have no budget-aware eviction at
        # all, only this fixed per-band entry count), and on real diverse
        # text deep bands alone hit 1.48GB from just 30MB of training with
        # no sleep in between. Shrunk so total deep-band memory has a
        # better chance of staying under max_body_bytes long enough for
        # ordinary-bank pruning to actually keep up.
        long_range_band_max_entries=50_000,
        long_range_use_combined_salience=True,
        long_range_salience_span_bytes=6,
        long_range_salience_degree_bank=0,
        long_range_use_affective_intensity=True,
        long_range_ai_threshold=2.0,
        long_range_ai_bypass_burst=0.5,
        long_range_replay_queue_capacity=64,
        long_range_consolidated_max_entries=8_000,
        long_range_permanent_max_entries=800,
        long_range_permanent_recurrence_threshold=100,
        # Defaults to False in the library (a privacy gate -- permanent
        # entries are never forgotten, so an operator has to opt in); this
        # is a private, single-operator research run, not the multi-tenant
        # deployment case that default protects against, and every
        # permanent-tier measurement taken this session (4/64/70 members
        # at various points) assumed this was on. Explicit True here so
        # resuming this run doesn't silently change its own behavior.
        long_range_allow_permanent_tier=True,
        # Sparse episodic layer (see docs/ALM_V1_Architecture.md's "A
        # proposed sparse episodic layer" section) -- off by default so
        # existing runs/configs are unaffected; k/s/bucket sizing/tier-3
        # threshold/strength all use the validated defaults from the
        # standalone prototype rather than being separately re-exposed
        # here, since none of that sweep has been re-run against real
        # generation quality yet.
        episodic_enabled=episodic_enabled,
        episodic_k=episodic_k,
        episodic_s=episodic_s,
        episodic_strength=episodic_strength,
    )


def _train_direct(
    ancestor: NativeGrowableCortex,
    data_path: Path,
    offset: int,
    length: int,
    chunk_bytes: int,
) -> dict[str, object]:
    """Train `ancestor` in place over [offset, offset+length), no subprocess,
    no ancestor/column pickling, no consolidate_delta merge.

    columns>1's whole point is N columns training in parallel, reconciled
    afterward -- but reconciliation cost (walking every touched context)
    turned out to scale with real text's novelty, not with column count, so
    at diverse-corpus scale it dominates wall time instead of being the
    cheap afterthought it was when column-parallelism was validated on
    repetitive synthetic text. Worse, deep bands have no lineage tracking
    or merge logic in consolidate.hpp at all, so column-sharded runs
    silently discard every column's deep-band learning at merge time. One
    column training directly has nothing to reconcile and nothing to lose:
    wake is the entire cost, at 1/columns the aggregate rate columns=N
    would give if merge were free -- which it isn't.
    """
    processed = 0
    scored = 0
    bit_sum = 0.0
    started = time.perf_counter()
    with data_path.open("rb") as handle:
        handle.seek(offset)
        while processed < length:
            raw = handle.read(min(chunk_bytes, length - processed))
            if not raw:
                raise ValueError("corpus ended inside an assigned territory")
            probability = ancestor.observe_chunk_probabilities(
                np.frombuffer(raw, dtype=np.uint8), learn=True
            )
            finite = probability[np.isfinite(probability)]
            bit_sum += float(np.sum(-np.log2(np.maximum(finite, 1e-300))))
            scored += len(finite)
            processed += len(raw)
    elapsed = time.perf_counter() - started
    return {
        "scored_bytes": scored,
        "train_bits": bit_sum,
        "train_bpb": bit_sum / max(scored, 1),
        "seconds": elapsed,
        "bytes_per_second": length / max(elapsed, 1e-12),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.target_epochs < 1:
        raise ValueError("target epochs must be positive")
    if args.sleep_bytes < args.columns:
        raise ValueError("sleep interval must provide work to every column")
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.train.resolve()
    total_bytes = train_path.stat().st_size

    epoch_rows: list[dict[str, object]] = []
    current_checkpoint: Path | None = None
    next_corpus_offset = 0
    latest_complete_epoch = 0
    # NOT a plain range(1, target_epochs) scan via _load_finished_epoch:
    # that function requires checkpoint.pkl to exist, but this run only
    # keeps the MOST RECENT epoch's checkpoint on disk (see the
    # `epoch_{epoch - 1}` unlink below, added to bound disk use) -- so a
    # strict from-1 scan stops at the first pruned intermediate epoch and
    # silently restarts the entire run from epoch 1, discarding real
    # progress. report.json is never pruned, so scan every epoch directory
    # that actually exists for one directly, and only require the
    # checkpoint to exist for whichever epoch turns out to be the latest.
    existing_epochs = sorted(
        int(name.removeprefix("epoch_"))
        for name in (p.name for p in work_dir.glob("epoch_*"))
        if name.removeprefix("epoch_").isdigit()
    )
    for epoch in existing_epochs:
        if epoch >= args.target_epochs:
            continue
        report_path = work_dir / f"epoch_{epoch}" / "report.json"
        if not report_path.exists():
            continue
        finished = json.loads(report_path.read_text(encoding="utf-8"))
        if finished.get("status") != "complete":
            continue
        epoch_rows.append(finished)
        latest_complete_epoch = max(latest_complete_epoch, epoch)
        checkpoint_path = Path(finished["checkpoint"])
        if checkpoint_path.exists():
            current_checkpoint = checkpoint_path
        # corpus_offset_after is the source of truth for where the next
        # epoch resumes -- NOT epoch * epoch_bytes, which only holds while
        # epoch_bytes stays constant across the whole run. A restart with a
        # different --epoch-bytes (e.g. shrinking it for faster feedback)
        # must still continue forward from here, not re-derive an offset
        # that silently re-reads or skips corpus bytes. A report written
        # before this field existed still recorded its own epoch_offset and
        # epoch_bytes, which is exactly enough to reconstruct it -- falling
        # back to next_corpus_offset (0 initially) would instead silently
        # re-read that epoch's bytes from the start.
        next_corpus_offset = int(
            finished.get(
                "corpus_offset_after",
                int(finished.get("epoch_offset", 0))
                + int(finished.get("epoch_bytes", 0)),
            )
        )

    run_started = time.perf_counter()
    start_epoch = latest_complete_epoch + 1

    for epoch in range(start_epoch, args.target_epochs + 1):
        epoch_dir = work_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        ancestor = (
            _new_native_cortex(
                args.long_range_band_count,
                args.episodic_enabled,
                args.trust_delta_clip,
                args.episodic_k,
                args.episodic_s,
                args.episodic_strength,
                args.long_range_bank,
                args.long_range_context_bytes,
                args.long_range_strength,
            )
            if current_checkpoint is None
            else _load_any_cortex(current_checkpoint)
        )
        # long_range_bank IS part of serialize.hpp (long_range_map_/
        # long_range_history_ round-trip fully), unlike the fields listed
        # below -- no reapply-every-epoch needed for it.
        # None of episodic_enabled/k/s, trust_delta_clip/floor/ceiling, or
        # long_range_allow_permanent_tier are part of serialize.hpp -- a
        # checkpoint round-trip silently reverts ALL of them to
        # CortexConfig's compiled defaults (episodic off/k=14/s=6,
        # trust_delta_clip=2.0/floor=0.1/ceiling=10.0,
        # long_range_allow_permanent_tier=false). _new_native_cortex's own
        # constructor arguments only take effect for a genuinely fresh
        # cortex above (current_checkpoint is None, i.e. epoch 1) -- every
        # resumed epoch must reapply all of them explicitly, every time,
        # or the run silently drifts back to defaults after its first
        # epoch. (Found this by tracing why an episodic-recall probe kept
        # coming up empty against a live checkpoint -- the same gap would
        # have silently reverted trust_delta_clip and the permanent-tier
        # flag on every prior multi-epoch run too.)
        ancestor.configure_episodic_layer(
            args.episodic_enabled, args.episodic_k, args.episodic_s
        )
        if args.episodic_enabled:
            ancestor.set_episodic_strength(args.episodic_strength)
        ancestor.set_trust_dynamics(
            args.trust_delta_clip, args.trust_floor, args.trust_ceiling
        )
        ancestor.set_long_range_allow_permanent_tier(True)

        # Ablation flag, not persisted by to_bytes()/from_bytes() (see
        # set_debug_deep_band_anchor_gate) -- must be reapplied every
        # epoch, same reasoning as above. Validated: 40/40 synthetic-repeat
        # trials still show clear recall at k=12/s=6 (mean signal ~61% of
        # ungated strength, since ~82% of bytes aren't anchors and get no
        # deep-band contribution that byte), eval bpb cost negligible
        # (+0.002, noise-level) across the whole k=6..12 sweep on a 40MB
        # real-corpus run, for up to 2.02x throughput at k=12 -- see
        # docs/ALM_V1_Architecture.md.
        if args.deep_band_anchor_gate:
            ancestor.set_debug_deep_band_anchor_gate(
                True, args.deep_band_anchor_gate_k, args.deep_band_anchor_gate_s
            )

        epoch_offset = next_corpus_offset % total_bytes
        epoch_length = min(args.epoch_bytes, total_bytes - epoch_offset)
        next_corpus_offset = epoch_offset + epoch_length

        wake_rows: list[dict[str, object]] = []
        epoch_started = time.perf_counter()
        pool = None if args.columns == 1 else ProcessPoolExecutor(max_workers=args.columns)
        try:
            for generation, round_offset in enumerate(
                range(0, epoch_length, args.sleep_bytes)
            ):
                round_bytes = min(args.sleep_bytes, epoch_length - round_offset)
                ancestor.reset_episode()

                if pool is None:
                    wake_started = time.perf_counter()
                    single = _train_direct(
                        ancestor,
                        train_path,
                        epoch_offset + round_offset,
                        round_bytes,
                        args.chunk_bytes,
                    )
                    wake_seconds = time.perf_counter() - wake_started
                    merge_seconds = 0.0
                    columns = [
                        {
                            **single,
                            "status": "complete",
                            "epoch": epoch,
                            "generation": generation,
                            "column": 0,
                            "offset": epoch_offset + round_offset,
                            "bytes": round_bytes,
                            "new_sequence_births": None,
                            "nursery_evictions": None,
                            "checkpoint": None,
                        }
                    ]
                    master = ancestor
                else:
                    ancestor_path = epoch_dir / "awake_ancestor.pkl"
                    _atomic_pickle(ancestor_path, {"learner": ancestor})
                    arguments = []
                    for column, (local_offset, length) in enumerate(
                        _partition(round_bytes, args.columns)
                    ):
                        arguments.append(
                            (
                                epoch,
                                generation,
                                column,
                                str(ancestor_path),
                                str(train_path),
                                epoch_offset + round_offset + local_offset,
                                length,
                                args.chunk_bytes,
                                str(epoch_dir / f"column_{column}.pkl"),
                            )
                        )
                    wake_started = time.perf_counter()
                    columns = list(pool.map(_develop, arguments))
                    wake_seconds = time.perf_counter() - wake_started
                    master = copy.deepcopy(ancestor)
                    merge_started = time.perf_counter()
                    for column in columns:
                        descendant = _load_any_cortex(column["checkpoint"])
                        master.consolidate_delta(descendant, ancestor)
                        del descendant
                    merge_seconds = time.perf_counter() - merge_started
                    # Each column checkpoint pickles that column's full
                    # local delta -- nursery churn from real diverse text
                    # made these run 350-540MB EACH (12 of them, every
                    # round, same filenames overwritten next round but
                    # never reclaimed meanwhile) and once filled the disk
                    # mid-epoch. Nothing here is needed once merged.
                    ancestor_path.unlink(missing_ok=True)
                    for column in columns:
                        checkpoint_path = column.get("checkpoint")
                        if checkpoint_path:
                            Path(checkpoint_path).unlink(missing_ok=True)
                            Path(checkpoint_path).with_suffix(".progress.json").unlink(
                                missing_ok=True
                            )

                before_sleep = master.embodied_bytes
                sleep_started = time.perf_counter()
                sleep = master.consolidate_to_byte_budget(args.max_body_bytes)
                sleep_seconds = time.perf_counter() - sleep_started
                wake_rows.append(
                    {
                        "generation": generation,
                        "offset": epoch_offset + round_offset,
                        "bytes": round_bytes,
                        "wake_seconds": wake_seconds,
                        "merge_seconds": merge_seconds,
                        "sleep_seconds": sleep_seconds,
                        "body_bytes_before_sleep": before_sleep,
                        "body_bytes_after_sleep": master.embodied_bytes,
                        "sleep": sleep,
                        "columns": columns,
                    }
                )
                ancestor = master
                current_checkpoint = epoch_dir / "latest_checkpoint.pkl"
                _atomic_pickle(current_checkpoint, {"learner": ancestor})
                _atomic_json(
                    epoch_dir / "progress.json",
                    {
                        "status": "training",
                        "epoch": epoch,
                        "completed_generations": generation + 1,
                        "processed_bytes": round_offset + round_bytes,
                        "epoch_bytes": epoch_length,
                        "body_bytes": ancestor.embodied_bytes,
                        "elapsed_seconds": time.perf_counter() - epoch_started,
                    },
                )
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        scored_bytes = sum(
            int(column["scored_bytes"])
            for wake in wake_rows
            for column in wake["columns"]
        )
        train_bits = sum(
            float(column["train_bits"])
            for wake in wake_rows
            for column in wake["columns"]
        )
        evaluation = {
            "mixed_conversational": _evaluate(
                ancestor, args.valid.resolve(), args.eval_bytes
            )
        }
        generations = _generation_panel(ancestor, max_new_bytes=args.generation_bytes)
        checkpoint = epoch_dir / "checkpoint.pkl"
        report: dict[str, object] = {
            "status": "complete",
            "epoch": epoch,
            "training_stream": str(train_path),
            "epoch_offset": epoch_offset,
            "epoch_bytes": epoch_length,
            "corpus_offset_after": next_corpus_offset,
            "sleep_bytes": args.sleep_bytes,
            "sleep_generations": len(wake_rows),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "training_bpb": train_bits / max(scored_bytes, 1),
            "body_bytes": ancestor.embodied_bytes,
            "mature_sequence_paths": ancestor.live_sequence_count,
            "deep_band_total_promotions": ancestor.deep_band_total_promotions,
            "deep_band_total_consolidations": ancestor.deep_band_total_consolidations,
            "deep_band_total_permanent_promotions": ancestor.deep_band_total_permanent_promotions,
            "evaluations": evaluation,
            "generations": generations,
            "wake_generations": wake_rows,
            "checkpoint": str(checkpoint.resolve()),
        }
        _atomic_pickle(checkpoint, {"learner": ancestor, "report": report})
        _atomic_json(epoch_dir / "report.json", report)
        # latest_checkpoint.pkl (the per-round rolling save) is now fully
        # redundant with checkpoint.pkl (the confirmed, final snapshot for
        # this epoch) -- keeping both doubles disk use for no benefit.
        (epoch_dir / "latest_checkpoint.pkl").unlink(missing_ok=True)
        # Only the most recent complete epoch's checkpoint is ever read on
        # resume (_load_finished_epoch); older ones are pure disk cost at
        # up to max_body_bytes each. Report/generation history stays (tiny
        # by comparison) so past epochs remain inspectable. Safe even on a
        # fresh start (epoch_0 never exists) or a resume (already pruned).
        (work_dir / f"epoch_{epoch - 1}" / "checkpoint.pkl").unlink(missing_ok=True)
        epoch_rows.append(report)
        current_checkpoint = checkpoint

    result = {
        "task": "phase156_conversational_prototype",
        "status": "complete",
        "target_epochs": args.target_epochs,
        "columns": args.columns,
        "max_body_bytes": args.max_body_bytes,
        "total_wall_seconds": time.perf_counter() - run_started,
        "epochs": epoch_rows,
        "final_checkpoint": str(current_checkpoint),
    }
    _atomic_json(work_dir / "report.json", result)
    # ensure_ascii=True here (unlike _atomic_json's UTF-8 file writes): the
    # console/redirected-file encoding Windows picks for stdout can't
    # represent arbitrary corpus Unicode (e.g. Cyrillic from mixed wiki
    # content), and this print is only a convenience echo -- the full
    # UTF-8 report already landed on disk via _atomic_json above.
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--target-epochs", type=int, default=200)
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--epoch-bytes", type=int, default=500_000_000)
    parser.add_argument("--sleep-bytes", type=int, default=64_000_000)
    parser.add_argument("--chunk-bytes", type=int, default=4_000_000)
    parser.add_argument("--eval-bytes", type=int, default=1_000_000)
    parser.add_argument("--generation-bytes", type=int, default=512)
    parser.add_argument("--max-body-bytes", type=int, default=500_000_000)
    parser.add_argument(
        "--long-range-band-count",
        type=int,
        default=4,
        help=(
            "How many of the 4 configured depths (12/16/24/32) to actually "
            "run. Tiered persistence (consolidated/permanent) stays on "
            "regardless -- this isolates whether multiple long-range "
            "depths are pulling their weight, or whether the tiering "
            "mechanism alone accounts for most of what deep bands buy."
        ),
    )
    parser.add_argument(
        "--episodic-enabled",
        action="store_true",
        help=(
            "Enable the sparse episodic layer (syncmer-anchored, rolling-"
            "window recall with LRU decay and tier-3 promotion -- see "
            "docs/ALM_V1_Architecture.md). Off by default. Writes are "
            "decoupled from learn=True/False and from deep-band "
            "liveness -- it tracks what was actually observed in this "
            "episode independent of whether the durable cortex is being "
            "rewired."
        ),
    )
    parser.add_argument(
        "--episodic-k",
        type=int,
        default=14,
        help=(
            "Episodic anchor k-mer width. A recall-rate sweep on a real "
            "trained checkpoint (probability-boost hit rate at planted "
            "markers) found hit rate tracks window w=k-s+1 as ~2/w almost "
            "exactly: w=5 -> 40%%, w=3 -> 67%%, w=2 -> 99%% (the trivial "
            "always-anchor case, since with only 2 sub-positions the "
            "minimum is always at a boundary). Throughput cost is "
            "negligible until w=2, which costs ~20%% vs the w>=3 range -- "
            "see docs/ALM_V1_Architecture.md."
        ),
    )
    parser.add_argument("--episodic-s", type=int, default=6)
    parser.add_argument(
        "--episodic-strength",
        type=float,
        default=4.0,
        help=(
            "Blend weight for episodic recall (out[t] = (prior*strength + "
            "count[t]) / (total+strength) -- same pseudo-count-prior shape "
            "as basal_strength/parent_strength elsewhere, so LOWER means "
            "the actual observation dominates MORE). A strength sweep at "
            "w=2 (k=6/s=5) found a sharp threshold, not a gradual "
            "tradeoff: the default 4.0 gave 0/20 exact-match generation "
            "recall of a planted code, while every value <=1.0 gave 20/20, "
            "with no throughput cost either way (pure blend arithmetic, "
            "not a control-flow change) -- see docs/ALM_V1_Architecture.md."
        ),
    )
    parser.add_argument(
        "--deep-band-anchor-gate",
        action="store_true",
        help=(
            "Ablation: gate deep-band lookup/grow to only fire at syncmer-"
            "anchor positions (see set_debug_deep_band_anchor_gate), "
            "leaving the O(1) rolling-hash history update dense every "
            "byte. Off by default. Validated on a 40MB real-corpus A/B: "
            "recall of a planted repeat survives 40/40 trials at k=12/s=6, "
            "eval bpb cost is noise-level (+0.002) across k=6..12, training "
            "bpb cost grows with k (+0.069 at k=6 to +0.104 at k=12) for "
            "1.53x-2.02x throughput -- see docs/ALM_V1_Architecture.md."
        ),
    )
    parser.add_argument("--deep-band-anchor-gate-k", type=int, default=12)
    parser.add_argument("--deep-band-anchor-gate-s", type=int, default=6)
    parser.add_argument(
        "--trust-delta-clip",
        type=float,
        default=2.0,
        help=(
            "Clip applied to the adaptive-trust TD(0) update before "
            "exponentiating (trust *= exp(clipped_delta)). Default (2.0) "
            "was measured to saturate hard (49.8%% of contexts at the "
            "ceiling) on the first 500MB baseline. Reapplied every epoch "
            "(not just to a fresh cortex) since it isn't part of the "
            "checkpoint format -- see set_trust_dynamics."
        ),
    )
    parser.add_argument("--trust-floor", type=float, default=0.1)
    parser.add_argument("--trust-ceiling", type=float, default=10.0)
    parser.add_argument(
        "--long-range-bank",
        action="store_true",
        help=(
            "Enable a single confirmation-gated dendritic bank at "
            "--long-range-context-bytes, beyond the ordinary sequence "
            "banks' own ceiling (max_context_bytes=6 here, hard-capped at "
            "8 by packed_history_'s bit-packing) -- chained via the same "
            "nursery/promotion contract as the ordinary banks (real "
            "confirmations before trust), not deep bands' sparse/tiered "
            "memory. Already validated (53%%->95%% recall across a gap the "
            "ordinary banks can't see across) and fully serialized. Off "
            "by default. See docs/ALM_V1_Architecture.md."
        ),
    )
    parser.add_argument("--long-range-context-bytes", type=int, default=12)
    parser.add_argument("--long-range-strength", type=float, default=8.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
