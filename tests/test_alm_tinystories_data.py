"""Reproducible hashes, no split overlap, and shard round-trip tests.

Design doc section 17: data/tinystories.py must pass "reproducible hashes
and no split overlap." Uses small in-repo fixture text so these tests do
not depend on the external TinyStories corpus drive.
"""

from __future__ import annotations

from manifoldflow.alm.data.tinystories import (
    STORY_DELIM,
    TokenShard,
    assign_split,
    encode_shard,
    file_checksum,
    iter_stories,
    load_deduped_corpus,
    story_hash,
)
from manifoldflow.alm.data.tokenizer import ByteControlTokenizer


def _write_corpus(tmp_path, stories: list[str], name: str = "corpus.txt"):
    path = tmp_path / name
    text = f"\n{STORY_DELIM}\n".join(stories) + f"\n{STORY_DELIM}\n"
    path.write_text(text, encoding="utf-8")
    return path


def test_iter_stories_splits_on_delimiter(tmp_path) -> None:
    stories = ["First story here.", "Second story, with punctuation!", "Third."]
    path = _write_corpus(tmp_path, stories)
    assert list(iter_stories(path)) == stories


def test_file_checksum_is_reproducible(tmp_path) -> None:
    path = _write_corpus(tmp_path, ["A story."])
    assert file_checksum(path) == file_checksum(path)


def test_file_checksum_changes_with_content(tmp_path) -> None:
    path_a = _write_corpus(tmp_path, ["A story."], name="a.txt")
    path_b = _write_corpus(tmp_path, ["A different story."], name="b.txt")
    assert file_checksum(path_a) != file_checksum(path_b)


def test_story_hash_deterministic_and_content_sensitive() -> None:
    assert story_hash("same text") == story_hash("same text")
    assert story_hash("same text") != story_hash("different text")


def test_assign_split_is_deterministic_and_hash_only() -> None:
    story_id = story_hash("Once upon a time, a fox learned to read.")
    assert assign_split(story_id) == assign_split(story_id)


def test_assign_split_respects_val_fraction_on_average() -> None:
    # Not a statistical test of the hash function itself; just checks that a
    # tiny fraction yields far fewer validation stories than a large one for
    # the same set of hashes.
    ids = [story_hash(f"story number {i}") for i in range(2000)]
    small = sum(assign_split(i, val_fraction=0.01) == "validation" for i in ids)
    large = sum(assign_split(i, val_fraction=0.5) == "validation" for i in ids)
    assert small < large


def test_load_deduped_corpus_splits_have_no_overlap_and_dedups(tmp_path) -> None:
    base_stories = [f"Story number {i} about a fox and a hen." for i in range(500)]
    stories_with_dupes = base_stories + base_stories[:50]  # 50 exact duplicates
    path = _write_corpus(tmp_path, stories_with_dupes)

    corpus = load_deduped_corpus({"tinystories_original": path}, val_fraction=0.1)

    assert corpus.n_seen == len(stories_with_dupes)
    assert corpus.n_duplicates == 50
    assert set(corpus.train).isdisjoint(set(corpus.validation))
    assert len(corpus.train) + len(corpus.validation) == len(base_stories)
    assert corpus.source_checksums["tinystories_original"] == file_checksum(path)


def test_load_deduped_corpus_is_reproducible_across_runs(tmp_path) -> None:
    stories = [f"Story number {i} about a fox and a hen." for i in range(300)]
    path = _write_corpus(tmp_path, stories)

    first = load_deduped_corpus({"tinystories_original": path}, val_fraction=0.1)
    second = load_deduped_corpus({"tinystories_original": path}, val_fraction=0.1)

    assert first.train == second.train
    assert first.validation == second.validation


def test_load_deduped_corpus_limit_per_source_applies_before_dedup(tmp_path) -> None:
    stories = [f"Story number {i} about a fox and a hen." for i in range(100)]
    path = _write_corpus(tmp_path, stories)

    corpus = load_deduped_corpus({"tinystories_original": path}, limit_per_source=10)

    assert corpus.n_seen == 10


def test_encode_shard_offsets_lengths_and_crc_round_trip(tmp_path) -> None:
    stories = [
        "Once there was a small fox.",
        "The fox met a hen in the field.",
        "They became friends and played all day.",
    ]
    tokenizer = ByteControlTokenizer()
    shard = encode_shard(stories, tokenizer)

    assert shard.verify()
    assert len(shard.offsets) == len(stories) == len(shard.lengths)
    for i, story in enumerate(stories):
        recovered_ids = shard.story(i).tolist()
        assert recovered_ids == tokenizer.encode_story(story)
        assert tokenizer.decode(recovered_ids) == story

    path = tmp_path / "shard.npz"
    shard.save(path)
    reloaded = TokenShard.load(path)
    assert reloaded.verify()
    assert reloaded.tokens.tolist() == shard.tokens.tolist()
    assert reloaded.offsets.tolist() == shard.offsets.tolist()
    assert reloaded.lengths.tolist() == shard.lengths.tolist()


def test_encode_shard_crc_detects_corruption() -> None:
    tokenizer = ByteControlTokenizer()
    shard = encode_shard(["A short story about a fox."], tokenizer)
    shard.tokens[0] = shard.tokens[0] ^ 0xFFFF
    assert not shard.verify()
