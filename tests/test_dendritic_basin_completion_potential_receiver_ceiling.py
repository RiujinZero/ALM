from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from manifoldflow.experiments import (
    dendritic_basin_completion_obligation_checker as checker,
)
from manifoldflow.experiments import (
    dendritic_basin_completion_potential_receiver_ceiling as assay,
)


class FakeCortex:
    """Deterministic positive-support model with transient episode storage."""

    learned_body_bytes = 500_000_000

    def __init__(self) -> None:
        self.history = bytearray()
        self.learn_flags: list[bool] = []
        self.reset_count = 0

    @property
    def embodied_bytes(self) -> int:
        return self.learned_body_bytes + len(self.history)

    def reset_episode(self) -> None:
        self.history.clear()
        self.reset_count += 1

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool
    ) -> np.ndarray:
        self.learn_flags.append(learn)
        values = np.asarray(byte_ids, dtype=np.uint8).reshape(-1)
        self.history.extend(int(value) for value in values)
        return np.full(len(values), 1.0 / 256.0)

    def predict_next_distribution(self) -> np.ndarray:
        tail = bytes(self.history[-32:])
        favored = int.from_bytes(
            hashlib.blake2b(tail or b"empty", digest_size=2).digest(), "little"
        ) % 256
        result = np.full(256, 0.4 / 255.0, dtype=np.float64)
        result[favored] = 0.6
        return result

    def advance_observation(self, byte_id: int, *, learn: bool) -> None:
        self.learn_flags.append(learn)
        self.history.append(int(byte_id))


def _compiled() -> tuple[
    checker.CompletionObligationPair,
    FakeCortex,
    assay.NativeBodyTracker,
    assay.CompiledWorld,
]:
    pair = checker.build_completion_obligation_pair(0, interruption_bytes=512)
    cortex = FakeCortex()
    tracker = assay.NativeBodyTracker(cortex)
    compiled = assay.compile_world(cortex, pair.world_a, pair.world_b, tracker)
    return pair, cortex, tracker, compiled


def test_compile_uses_actual_prompt_union_table_and_long_horizon_witness() -> None:
    pair, cortex, tracker, compiled = _compiled()

    assert compiled.prefix_table.prompt == pair.world_a.prompt
    assert compiled.prefix_table.trajectory_replays < sum(
        len(values) for values in compiled.accepted_sets.values()
    )
    assert len(compiled.prefix_table.prefixes) > 100
    assert set(compiled.potentials) == set(assay.POTENTIAL_SET_NAMES)
    assert all(
        np.isfinite(potential.accepted_log_mass)
        for potential in compiled.potentials.values()
    )
    witness = compiled.long_horizon_witness
    assert witness["found"] is True
    assert witness["base_distribution_exactly_equal"] is True
    assert witness["route_l1"] > 0.0
    assert witness["only_changed_input"] == "long_horizon_accepted_completion_set"
    assert tracker.maximum_transient > tracker.at_load
    assert cortex.learn_flags and not any(cortex.learn_flags)


@pytest.mark.parametrize("sampled", (False, True))
def test_correct_exact_h_reaches_only_independently_accepted_leaf(
    sampled: bool,
) -> None:
    pair, cortex, tracker, compiled = _compiled()
    draws = np.random.default_rng(19).random(128) if sampled else None

    trace = assay.generate(
        cortex,
        compiled,
        assay._arm_specs()["correct_exact_h"],
        horizon=128,
        uniform_draws=draws,
        tracker=tracker,
    )

    checked = checker.check_completion(pair.world_a, trace.data)
    assert checked.exact_accepted is True
    assert trace.exact_leaf_reached is True
    assert trace.terminated is True
    assert trace.off_trie_steps == 0
    assert trace.selected_zero_base_support == 0
    assert trace.fallback_count == 0
    assert trace.route_changed_steps > 0


def test_disconnected_and_zero_strength_are_bit_exact_native() -> None:
    _pair, cortex, tracker, compiled = _compiled()
    draws = np.random.default_rng(29).random(128)
    specs = assay._arm_specs()

    traces = {
        name: assay.generate(
            cortex,
            compiled,
            specs[name],
            horizon=128,
            uniform_draws=draws,
            tracker=tracker,
        )
        for name in ("native", "computed_but_disconnected", "zero_strength")
    }

    assert traces["native"] == traces["computed_but_disconnected"]
    assert traces["native"] == traces["zero_strength"]
    assert traces["native"].route_changed_steps == 0


def test_wrong_target_exact_h_is_accepted_only_on_crossed_world() -> None:
    pair, cortex, tracker, compiled = _compiled()
    trace = assay.generate(
        cortex,
        compiled,
        assay._arm_specs()["paired_wrong_target_h"],
        horizon=128,
        uniform_draws=np.random.default_rng(31).random(128),
        tracker=tracker,
    )

    assert checker.check_completion(pair.world_a, trace.data).exact_accepted is False
    assert checker.check_completion(pair.world_b, trace.data).exact_accepted is True
    assert trace.exact_leaf_reached is True

    # Corrupting compiler-owned accepted-set metadata cannot make the
    # independent semantic checker certify the route's wrong binding.
    forged = replace(pair.world_a, accepted_full_completions=(trace.data,))
    assert checker.check_completion(forged, trace.data).exact_accepted is False


def test_floor_stress_is_explicit_and_never_fabricates_native_support() -> None:
    _pair, cortex, tracker, compiled = _compiled()
    trace = assay.generate(
        cortex,
        compiled,
        assay._arm_specs()["false_positive_floor_1em3"],
        horizon=128,
        uniform_draws=np.random.default_rng(37).random(128),
        tracker=tracker,
    )

    assert trace.selected_zero_base_support == 0
    assert trace.fallback_count == 0
    assert trace.steps <= 128


def test_run_ablation_reports_primary_controls_and_native_body_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"frozen-checkpoint-fixture")
    cortex = FakeCortex()
    monkeypatch.setattr(assay, "_load_native", lambda _path: cortex)
    args = SimpleNamespace(
        checkpoint=checkpoint,
        report=tmp_path / "report.json",
        pairs=1,
        pair_start=3,
        interruption_bytes=512,
        sample_seeds=(43,),
        horizon=128,
        bootstrap_seed=47,
        bootstrap_resamples=100,
        wall_limit_seconds=60.0,
        smoke=True,
    )

    report = assay.run_ablation(args)

    assert report["claim_scope"][
        "exact_h_is_full_completion_subtree_mass_not_next_byte_supervision"
    ]
    assert report["claim_scope"][
        "does_not_distinguish_undertraining_from_objective_mismatch"
    ]
    assert report["claim_scope"][
        "offline_oracle_prefix_replay_used_only_to_compile_native_p0"
    ]
    assert report["claim_scope"]["no_teacher_forcing_during_scored_generation"]
    assert report["status"] == "complete_engineering_smoke"
    assert report["config"]["arms"] == list(assay.ARM_NAMES)
    assert report["config"]["pair_indices"] == [3]
    assert all(report["hard_invariants"].values())
    assert report["checkpoint"]["embodied_bytes_at_load"] == 500_000_000
    assert report["checkpoint"]["maximum_transient_embodied_bytes"] > 500_000_000
    assert report["checkpoint"]["embodied_bytes_after_final_reset"] == 500_000_000
    assert report["evaluation"]["summaries"]["correct_exact_h"]["greedy"][
        "independent_exact_completion_rate"
    ]["mean"] == 1.0
    assert report["evaluation"]["summaries"]["paired_wrong_target_h"][
        "greedy"
    ]["independent_crossed_acceptance_rate"]["mean"] == 1.0
    assert not any(cortex.learn_flags)
    json.dumps(report)


def test_cli_has_no_dataset_or_split_surface() -> None:
    args = assay._parser().parse_args(
        ["--checkpoint", "checkpoint.pkl", "--report", "report.json"]
    )

    assert args.pairs == 4
    assert args.horizon == 128
    assert args.smoke is False
    assert not hasattr(args, "dataset_dir")
    assert not hasattr(args, "split")


def test_invalid_budget_inputs_fail_before_checkpoint_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fail_if_loaded(_path: Path) -> FakeCortex:
        nonlocal called
        called = True
        return FakeCortex()

    monkeypatch.setattr(assay, "_load_native", fail_if_loaded)
    args = SimpleNamespace(
        checkpoint=tmp_path / "missing.pkl",
        report=tmp_path / "report.json",
        pairs=1,
        pair_start=0,
        interruption_bytes=512,
        sample_seeds=(1, 1),
        horizon=128,
        bootstrap_seed=2,
        bootstrap_resamples=100,
        wall_limit_seconds=60.0,
        smoke=False,
    )

    with pytest.raises(ValueError, match="unique"):
        assay.run_ablation(args)
    assert called is False
