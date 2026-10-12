from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.token_sequence_cortex import (
    LosslessLexicalCodec,
    TokenSequenceCortex,
)


def test_lexical_codec_round_trips_unknown_words_and_binary_bytes() -> None:
    training = b"the cat sat.\n" * 20
    codec = LosslessLexicalCodec.build(training, vocab_size=264)
    source = b"the unseen\xff cat!\r\n"

    encoded = codec.encode(source)

    assert codec.decode(encoded) == source
    assert np.sum(encoded == codec.RAW_BEGIN) >= 1


def test_lexical_codec_is_deterministic_and_counts_its_lexicon() -> None:
    source = b"beta alpha beta gamma alpha beta"
    left = LosslessLexicalCodec.build(source, vocab_size=262)
    right = LosslessLexicalCodec.build(source, vocab_size=262)

    assert left.id_to_unit == right.id_to_unit
    assert left.embodied_bytes > 0


def test_token_sequence_cortex_learns_repeated_continuation() -> None:
    learner = TokenSequenceCortex(
        300,
        confirmations=1,
        context_confirmations=1,
        sequence_confirmations=1,
        max_context_tokens=4,
    )
    stream = np.asarray(([258, 259, 260, 261] * 20), dtype=np.int32)
    learner.observe_chunk_probabilities(stream, learn=True)
    learner.reset_episode()
    learner.observe(258, learn=False)
    learner.observe(259, learn=False)

    distribution = learner.distribution()

    assert int(np.argmax(distribution)) == 260
    assert distribution[260] > distribution[262]
    assert learner.live_sequence_count > 0


def test_token_sequence_cortex_consolidates_to_physical_budget() -> None:
    learner = TokenSequenceCortex(
        300,
        confirmations=1,
        context_confirmations=1,
        sequence_confirmations=1,
        max_context_tokens=4,
    )
    stream = np.asarray(([258, 259, 260, 261, 262, 263] * 100), dtype=np.int32)
    learner.observe_chunk_probabilities(stream, learn=True)
    before = learner.embodied_bytes

    result = learner.consolidate_to_byte_budget(before - 100)

    assert learner.embodied_bytes <= before - 100
    assert result["pruned_mature_paths"] > 0
