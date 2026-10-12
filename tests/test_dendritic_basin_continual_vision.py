from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from manifoldflow.dendritic_basin.continual_vision import (
    ALM64MB4CContinualVision,
    ALM64MB4CGrowingVision,
    ContinualVisionConfig,
    RawImageReceptor,
)
from manifoldflow.experiments.dendritic_basin_continual_vision_benchmark import (
    _clear_metrics,
    load_clear10,
)


def _image(value: int) -> Image.Image:
    return Image.fromarray(np.full((12, 12, 3), value, dtype=np.uint8))


def test_receptor_is_a_deterministic_raw_vector() -> None:
    receptor = RawImageReceptor(8)
    first = receptor(_image(100))
    second = receptor(_image(100))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (5 * 8 * 8,)


def test_prediction_is_made_before_external_update() -> None:
    config = ContinualVisionConfig(classes=2, image_size=8, sleep_interval=2)
    left = ALM64MB4CContinualVision(config)
    right = ALM64MB4CContinualVision(config)
    left_prediction, _, left_evidence = left.predict(_image(220))
    right_prediction, _, right_evidence = right.predict(_image(220))

    left.observe_external(left_evidence, 0)
    right.observe_external(right_evidence, 1)

    assert left_prediction == right_prediction
    torch.testing.assert_close(left_evidence, right_evidence)


def test_only_one_awake_column_learns_before_sleep() -> None:
    learner = ALM64MB4CContinualVision(
        ContinualVisionConfig(classes=2, image_size=8, sleep_interval=10)
    )
    _, _, evidence = learner.predict(_image(200))
    learner.observe_external(evidence, 1)

    assert learner.columns[0].counts[1] == 1
    assert sum(int(column.counts[1]) for column in learner.columns[1:]) == 0


def test_sleep_consolidates_and_stays_within_body_class() -> None:
    learner = ALM64MB4CContinualVision(
        ContinualVisionConfig(classes=2, image_size=8, sleep_interval=2)
    )
    for value, label in ((32, 0), (224, 1)):
        _, _, evidence = learner.predict(_image(value))
        learner.observe_external(evidence, label)

    torch.testing.assert_close(
        learner.columns[0].assemblies, learner.columns[3].assemblies
    )
    assert learner.sleep_cycles == 1
    assert learner.embodied_bytes <= 64_000_000


def test_growing_vision_births_sensory_assemblies_then_consolidates() -> None:
    config = ContinualVisionConfig(
        classes=2,
        image_size=8,
        sleep_interval=4,
        maximum_visual_assemblies=16,
        maximum_synapses=16,
    )
    learner = ALM64MB4CGrowingVision(config)
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[:, :6] = 255
    frame = Image.fromarray(image)

    assert learner.live_assemblies == 0
    for _ in range(4):
        _, _, evidence = learner.predict(frame)
        learner.observe_external(evidence, 1)

    assert learner.sleep_cycles == 1
    assert learner.live_assemblies == 4
    assert all(column.live == 4 for column in learner.columns)
    assert learner.embodied_bytes <= config.max_body_bytes

    prediction, _, _ = learner.predict(frame)
    assert prediction == 1


def test_growing_vision_labels_are_not_assembly_ids() -> None:
    learner = ALM64MB4CGrowingVision(
        ContinualVisionConfig(
            classes=2,
            image_size=8,
            sleep_interval=10,
            maximum_visual_assemblies=8,
            maximum_synapses=16,
        )
    )
    _, _, evidence = learner.predict(_image(200))
    learner.observe_external(evidence, 1)

    assert learner.columns[0].live == 1
    assert learner.columns[0].label_evidence[0, 1] == 1


def test_relation_turnover_reopens_capacity_at_sleep() -> None:
    learner = ALM64MB4CGrowingVision(
        ContinualVisionConfig(
            classes=2,
            image_size=8,
            sleep_interval=4,
            maximum_visual_assemblies=4,
            maximum_synapses=16,
            relation_turnover_fraction=0.25,
        )
    )
    for value, label in ((20, 0), (80, 1), (140, 0), (220, 1)):
        _, _, evidence = learner.predict(_image(value))
        learner.observe_external(evidence, label)

    assert learner.live_assemblies == 3
    assert learner.relation_turnover_releases == 1

    _, _, evidence = learner.predict(_image(110))
    learner.observe_external(evidence, 1)
    assert learner.columns[0].live == 4


def test_clear_loader_preserves_bucket_and_metadata_order(tmp_path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    (split / "class_names.txt").write_text("left\nright\n", encoding="utf-8")
    (split / "left.json").write_text(
        '{"b": {"IMG_PATH": "b.jpg"}, "a": {"IMG_PATH": "a.jpg"}}',
        encoding="utf-8",
    )
    (split / "right.json").write_text(
        '{"c": {"IMG_PATH": "c.jpg"}}', encoding="utf-8"
    )
    (split / "labeled_metadata.json").write_text(
        '{"2004": {"left": "left.json", "right": "right.json"}}',
        encoding="utf-8",
    )

    names, experiences = load_clear10(tmp_path, "train")

    assert names == ["left", "right"]
    assert [sample.path.name for sample in experiences[0][1]] == [
        "b.jpg",
        "a.jpg",
        "c.jpg",
    ]
    assert [sample.label for sample in experiences[0][1]] == [0, 0, 1]


def test_clear_metrics_follow_official_matrix_regions() -> None:
    metrics = _clear_metrics([[0.8, 0.4], [0.7, 0.9]])
    np.testing.assert_allclose(
        list(metrics.values()), [0.85, 0.4, 0.8, 0.7, 0.4]
    )
