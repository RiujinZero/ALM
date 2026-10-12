import pickle

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)
from manifoldflow.experiments.dendritic_basin_phase124_periodic_sleep import run


def test_periodic_sleep_runs_matched_wake_schedules(tmp_path) -> None:
    checkpoint = tmp_path / "start.pkl"
    with checkpoint.open("wb") as handle:
        pickle.dump(
            {
                "learner": SequenceCortexOpponentLearner(
                    confirmations=1,
                    context_confirmations=2,
                    sequence_confirmations=2,
                    max_sequence_nursery_paths=1_000,
                )
            },
            handle,
        )
    data = tmp_path / "train.bin"
    valid = tmp_path / "valid.bin"
    payload = (b"Mia found a red ball. She kept the red ball.\n" * 2_000)
    data.write_bytes(payload)
    valid.write_bytes(payload[:10_000])

    report = run(
        checkpoint,
        data,
        valid,
        tmp_path / "run",
        total_bytes=40_000,
        periodic_rounds=2,
        columns=2,
        workers=2,
        chunk_bytes=5_000,
        eval_bytes=5_000,
        max_body_bytes=2_000_000,
    )

    assert report["control"]["total_bytes"] == 40_000
    assert report["periodic"]["total_bytes"] == 40_000
    assert len(report["control"]["generations"]) == 1
    assert len(report["periodic"]["generations"]) == 2
    assert report["control"]["final_nursery_paths"] == 0
    assert report["periodic"]["final_nursery_paths"] == 0
    assert (tmp_path / "run" / "end_only_checkpoint.pkl").exists()
    assert (tmp_path / "run" / "periodic_checkpoint.pkl").exists()
