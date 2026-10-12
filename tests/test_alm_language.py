"""ALMLanguageModel: the generic train/generate core, not the mechanism.

The learning math itself is covered exhaustively elsewhere (test_dendritic_
basin_opponent_*.py, test_native_cortex_matches_python.py, etc). This just
checks the harness wiring: config validation, that train() actually reduces
surprise, that generate() produces the right length/type, and that save/
load round-trips a live model.
"""

from __future__ import annotations

import pytest

from manifoldflow.dendritic_basin.alm_language import ALMLanguageConfig, ALMLanguageModel

native_cortex = pytest.importorskip("manifoldflow.dendritic_basin.native_cortex")


def test_columns_greater_than_one_requires_native() -> None:
    with pytest.raises(ValueError, match="requires native=True"):
        ALMLanguageConfig(columns=2, native=False)


def test_columns_must_be_positive() -> None:
    with pytest.raises(ValueError, match="columns must be positive"):
        ALMLanguageConfig(columns=0)


def test_sleep_bytes_must_cover_every_column() -> None:
    with pytest.raises(ValueError, match="sleep_bytes must provide work"):
        ALMLanguageConfig(sleep_bytes=1, columns=5)


def test_training_reduces_bits_per_byte_on_repetitive_text() -> None:
    config = ALMLanguageConfig(
        native=False, sleep_bytes=2000, max_body_bytes=200_000
    )
    model = ALMLanguageModel(config)
    text = b"the quick brown fox jumps over the lazy dog. " * 200
    first = model.train(text[:2000])["bits_per_byte"]
    model.train(text[2000:])
    later = model.evaluate_bits_per_byte(b"the quick brown fox jumps over the lazy dog.")
    assert later < first, "surprise on familiar text should drop after more training"


def test_generate_returns_requested_length_as_text_and_bytes() -> None:
    config = ALMLanguageConfig(native=False, sleep_bytes=2000, max_body_bytes=200_000)
    model = ALMLanguageModel(config)
    model.train(b"hello world. " * 100)

    text = model.generate("hello", max_new_bytes=30, seed=0)
    assert isinstance(text, str)
    assert len(text.encode("utf-8", errors="replace")) >= 30

    raw = model.generate("hello", max_new_bytes=30, seed=0, as_text=False)
    assert isinstance(raw, bytes)
    assert len(raw) == len(b"hello") + 30


def test_generate_is_deterministic_given_a_seed() -> None:
    config = ALMLanguageConfig(native=False, sleep_bytes=2000, max_body_bytes=200_000)
    model = ALMLanguageModel(config)
    model.train(b"hello world. " * 100)
    first = model.generate("hello", max_new_bytes=20, seed=42)
    second = model.generate("hello", max_new_bytes=20, seed=42)
    assert first == second


def test_save_and_load_round_trips_a_trained_model(tmp_path) -> None:
    config = ALMLanguageConfig(native=False, sleep_bytes=2000, max_body_bytes=200_000)
    model = ALMLanguageModel(config)
    model.train(b"hello world. " * 100)
    before = model.generate("hello", max_new_bytes=20, seed=1)

    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = ALMLanguageModel.load(path)

    assert loaded.config == config
    assert loaded.embodied_bytes == model.embodied_bytes
    assert loaded.generate("hello", max_new_bytes=20, seed=1) == before


@pytest.mark.skipif(
    not native_cortex.native_available(),
    reason="alm_native shared library is not built (see cpp/README.md)",
)
def test_native_backend_trains_and_generates() -> None:
    config = ALMLanguageConfig(native=True, sleep_bytes=2000, max_body_bytes=200_000)
    model = ALMLanguageModel(config)
    report = model.train(b"hello world. " * 100)
    assert report["bytes_trained"] == 1300
    text = model.generate("hello", max_new_bytes=20, seed=0)
    assert isinstance(text, str)
