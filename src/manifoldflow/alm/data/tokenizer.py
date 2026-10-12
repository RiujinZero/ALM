"""Tokenizer for MF-ALM-TS: a trained 4,096-entry BPE vocab plus a fixed,
untrained 260-symbol byte-level control vocabulary.

Design doc section 4.2:
  - Train a 4,096-entry BPE vocabulary on the training split only.
  - Reserve PAD, BOS, EOS, UNK, and optional STORY_SEP control IDs.
  - Preserve case and punctuation; normalize Unicode to NFC; do not lowercase.
  - Store IDs as uint16 with story offsets and checksums.
  - Run a deterministic 260-symbol byte vocabulary as the strict
    no-learned-tokenizer control.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PAD, BOS, EOS, UNK, STORY_SEP = "<pad>", "<bos>", "<eos>", "<unk>", "<story_sep>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK, STORY_SEP]

BPE_VOCAB_SIZE = 4096
# The byte control drops STORY_SEP (raw bytes need no learned separator) to
# land on exactly the design's 260 symbols: 256 byte values + PAD/BOS/EOS/UNK.
BYTE_SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
BYTE_VOCAB_SIZE = 256 + len(BYTE_SPECIAL_TOKENS)  # == 260

assert BYTE_VOCAB_SIZE == 260, "byte control vocab must match the design's 260 symbols"


@dataclass(frozen=True)
class TokenizerIds:
    """The reserved control IDs, identical across the BPE and byte tokenizers."""

    pad: int
    bos: int
    eos: int
    unk: int
    story_sep: int


def _reserved_ids() -> TokenizerIds:
    return TokenizerIds(*range(len(SPECIAL_TOKENS)))


class StoryBpeTokenizer:
    """A 4,096-entry byte-level BPE tokenizer trained on TinyStories text only.

    Wraps `tokenizers.Tokenizer` (byte-level BPE, NFC normalization, no
    lowercasing) so raw text is preprocessing, not learned neural semantics
    (design doc section 3).
    """

    def __init__(self, tokenizer: object) -> None:
        self._tok = tokenizer
        self.ids = _reserved_ids()

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @classmethod
    def train(
        cls, texts: list[str] | object, vocab_size: int = BPE_VOCAB_SIZE
    ) -> StoryBpeTokenizer:
        from tokenizers import Tokenizer, normalizers, pre_tokenizers, trainers
        from tokenizers.models import BPE

        tok = Tokenizer(BPE(unk_token=UNK))
        tok.normalizer = normalizers.NFC()
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = __import__("tokenizers").decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok)

    def save(self, path: str | Path) -> None:
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> StoryBpeTokenizer:
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def encode_story(self, text: str) -> list[int]:
        """BOS + tokens + EOS, matching the shard convention in tinystories.py."""
        return [self.ids.bos, *self.encode(text), self.ids.eos]

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)


class ByteControlTokenizer:
    """Deterministic 260-symbol byte-level tokenizer: no learned segmentation.

    IDs [0, 5) are the reserved specials (see SPECIAL_TOKENS); IDs [5, 261)
    map to raw byte values 0-255. This is the "strict no-learned-tokenizer
    control" from section 4.2 / ablation 19 ("Byte vocabulary").
    """

    def __init__(self) -> None:
        self.ids = TokenizerIds(pad=0, bos=1, eos=2, unk=3, story_sep=-1)
        self._byte_offset = len(BYTE_SPECIAL_TOKENS)

    @property
    def vocab_size(self) -> int:
        return BYTE_VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        return [b + self._byte_offset for b in text.encode("utf-8")]

    def encode_story(self, text: str) -> list[int]:
        return [self.ids.bos, *self.encode(text), self.ids.eos]

    def decode(self, ids: list[int]) -> str:
        byte_ids = [i - self._byte_offset for i in ids if i >= self._byte_offset]
        return bytes(byte_ids).decode("utf-8", errors="replace")


def ids_to_uint16(ids: list[int]) -> np.ndarray:
    """Store token IDs as uint16, per section 4.2 / 14.1 shard format."""
    array = np.asarray(ids, dtype=np.int64)
    if array.size and (array.min() < 0 or array.max() > np.iinfo(np.uint16).max):
        raise ValueError("token id out of uint16 range")
    return array.astype(np.uint16)


def tokenizer_manifest(vocab_size: int, kind: str, extra: dict | None = None) -> dict:
    """Small JSON-able record of tokenizer identity for reproducibility logs."""
    manifest = {
        "kind": kind,
        "vocab_size": vocab_size,
        "special_tokens": SPECIAL_TOKENS,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path: str | Path, manifest: dict) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True)
    Path(path).write_text(text, encoding="utf-8")
