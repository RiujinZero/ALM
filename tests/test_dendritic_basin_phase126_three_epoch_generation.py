import argparse
import json
import pickle

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)
from manifoldflow.experiments.dendritic_basin_phase126_three_epoch_generation import (
    run,
)


def test_continued_epoch_emits_checkpoint_evaluation_and_generation(tmp_path) -> None:
    start = tmp_path / "epoch1.pkl"
    learner = SequenceCortexOpponentLearner(
        confirmations=1,
        context_confirmations=2,
        sequence_confirmations=2,
        max_sequence_nursery_paths=1_000,
    )
    with start.open("wb") as handle:
        pickle.dump({"learner": learner}, handle)
    epoch1_report = tmp_path / "epoch1.json"
    epoch1_report.write_text(
        json.dumps(
            {
                "training_bpb": 8.0,
                "consolidated_body_bytes": learner.embodied_bytes,
                "mature_sequence_paths": 0,
                "evaluations": {},
                "generations": [],
                "checkpoint": str(start),
            }
        ),
        encoding="utf-8",
    )
    payload = b"Once upon a time, Lily had a red ball. " * 2_000
    train = tmp_path / "train.txt"
    valid = tmp_path / "valid.txt"
    train.write_bytes(payload[:40_000])
    valid.write_bytes(payload[:10_000])
    work = tmp_path / "run"

    report = run(
        argparse.Namespace(
            epoch1_checkpoint=start,
            epoch1_report=epoch1_report,
            tinystories=train,
            tinystories_valid=valid,
            work_dir=work,
            target_epochs=2,
            columns=2,
            sleep_bytes=20_000,
            chunk_bytes=5_000,
            eval_bytes=5_000,
            generation_bytes=16,
            max_body_bytes=2_000_000,
        )
    )

    epoch2 = report["epochs"][1]
    assert report["status"] == "complete"
    assert epoch2["epoch"] == 2
    assert epoch2["training_bytes"] == 40_000
    assert epoch2["sleep_generations"] == 2
    assert epoch2["body_bytes"] <= 2_000_000
    assert epoch2["evaluations"]["tinystories"]["scored_bytes"] > 0
    assert len(epoch2["generations"]) == 8
    assert (work / "epoch_2" / "checkpoint.pkl").exists()
    assert (work / "report.json").exists()
