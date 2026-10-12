#!/usr/bin/env python
"""Build an episode-aligned long-context continuation ablation corpus.

The existing mixed conversational byte stream cannot recover conversation
boundaries: both records and turns are separated by ``\n\n``.  This builder
therefore reads the source JSONL directly (one conversation per line), keeps
the raw UTF-8 payload in binary files, and writes all boundaries/provenance to
JSONL sidecars.

Evaluation pairs are symmetric.  Conversations A and B each contribute an
unanswered user request and its real, withheld assistant response.  The two
contexts receive the *same* third-conversation interruption and the same
resume cue.  A scorer evaluates A/A, A/B, B/A, and B/B; candidate-intrinsic
fluency cancels in the matched-versus-swapped interaction margin.

The natural-dialogue assay measures long-context compatibility.  It does not,
by itself, distinguish a durable goal from topic, entity, or episodic memory.
Use a counterbalanced latent-policy probe for that stronger claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

# Version 2 adds an independently ranged ``goal_target`` payload.  Version 3
# records explicit zero-based per-source scan windows for prospective corpora.
# Version 4 preserves the complete held-out assistant responses in addition to
# the equal-length score windows, so generation assays can be audited without
# reopening the source corpus after results are known.  It also records
# checksummed request subranges and prevents cross-script A/B pairs, which
# would make lexical reassociation artificially easy.
SCHEMA_VERSION = 4
MINIMUM_BRIDGE_BYTES = 512
NEAR_DUPLICATE_SHARED_SHINGLES = 4
GOAL_TARGET_KINDS = ("response", "interlocutor", "successor")
_TURN_PATTERN = re.compile(rb"(?:^|\n\n)(USER|ASSISTANT):[ \t]*")
_HOLD_PREFIX = (
    b"\n\nASSISTANT: I will keep that request pending while we handle a "
    b"brief interruption.\n\n"
)
_RESUME_SUFFIX = (
    b"\n\nUSER: Return to the unanswered request from before the interruption "
    b"and answer it now.\n\nASSISTANT: "
)
_PROVENANCE_KEYS = ("conversation_id", "thread_id", "session_id", "id")


@dataclass(frozen=True)
class SourceSpec:
    label: str
    path: Path
    scan_limit: int | None = None
    start_line: int = 0


@dataclass(frozen=True)
class Turn:
    index: int
    role: str
    separator_start: int
    marker_start: int
    content_start: int
    end: int


@dataclass(frozen=True)
class Conversation:
    episode_id: str
    split_key: str
    split_key_kind: str
    split: str
    source_label: str
    declared_source: str | None
    source_path: str
    source_line: int
    source_byte_offset: int
    raw: bytes
    turns: tuple[Turn, ...]
    target_turn_index: int
    long_turn_hashes: frozenset[str]

    @property
    def target(self) -> Turn:
        return self.turns[self.target_turn_index]

    @property
    def response(self) -> bytes:
        target = self.target
        return self.raw[target.content_start : target.end]

    @property
    def unanswered_prefix(self) -> bytes:
        return self.raw[: self.target.separator_start]

    @property
    def target_user(self) -> Turn:
        target_user = self.turns[self.target_turn_index - 1]
        if target_user.role != "USER" or target_user.end != self.target.separator_start:
            raise AssertionError(
                "target assistant is not preceded by a complete user turn"
            )
        return target_user

    @property
    def target_user_payload(self) -> bytes:
        target_user = self.target_user
        return self.raw[target_user.content_start : target_user.end]

    @property
    def target_user_script(self) -> str:
        return _dominant_script(self.target_user_payload)

    @property
    def target_user_normalized_hash(self) -> str:
        fingerprint = _normalized_fingerprint(self.target_user_payload)
        return str(fingerprint["normalized_sha256"])

    @property
    def response_normalized_hash(self) -> str:
        return str(_normalized_fingerprint(self.response)["normalized_sha256"])

    @property
    def target_user_shingle_quads(self) -> frozenset[tuple[int, ...]]:
        return _fingerprint_shingle_quads(self.target_user_payload)

    @property
    def response_shingle_quads(self) -> frozenset[tuple[int, ...]]:
        return _fingerprint_shingle_quads(self.response)

    @property
    def horizon_bytes(self) -> int:
        return self.target.content_start


@dataclass(frozen=True)
class BuildConfig:
    sources: tuple[SourceSpec, ...]
    output_dir: Path
    train_bytes: int = 96_000
    dev_pairs: int = 64
    test_pairs: int = 128
    seed: int = 82_972
    min_prefix_bytes: int = 512
    min_response_bytes: int = 64
    score_bytes: int = 128
    min_bridge_bytes: int = MINIMUM_BRIDGE_BYTES
    local_context_bytes: int = 8
    max_record_bytes: int = 16_384
    min_prior_exchanges: int = 2
    goal_target_kind: str = "response"
    overwrite: bool = False


@dataclass(frozen=True)
class Pair:
    a: Conversation
    b: Conversation
    bridge: Conversation
    bridge_payload: bytes
    bridge_turn_end: int
    common_tail: bytes
    context_a: bytes
    context_b: bytes
    response_a: bytes
    response_b: bytes
    full_response_a: bytes
    full_response_b: bytes
    goal_target_a: bytes
    goal_target_b: bytes
    score_length: int


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _normalized_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return unicodedata.normalize("NFKC", text)


def _hash_normalized_text(text: str) -> str:
    compact = " ".join(_normalized_text(text).casefold().split())
    return _sha256(compact.encode("utf-8"))


def _payload_hash(payload: bytes) -> str:
    compact = b" ".join(payload.strip().lower().split())
    return _sha256(compact)


def _script_name(character: str) -> str | None:
    """Return a coarse deterministic writing-system label for one character."""

    code = ord(character)
    if not character.isalpha():
        return None
    if (
        0x0041 <= code <= 0x024F
        or 0x1E00 <= code <= 0x1EFF
        or 0xAB30 <= code <= 0xAB6F
    ):
        return "latin"
    if 0x0400 <= code <= 0x052F or 0x2DE0 <= code <= 0x2DFF:
        return "cyrillic"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return "han"
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
        return "hangul"
    if 0x3040 <= code <= 0x309F:
        return "hiragana"
    if 0x30A0 <= code <= 0x30FF:
        return "katakana"
    if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
        return "arabic"
    if 0x0900 <= code <= 0x097F:
        return "devanagari"
    if 0x0590 <= code <= 0x05FF:
        return "hebrew"
    if 0x0E00 <= code <= 0x0E7F:
        return "thai"
    return "other"


def _dominant_script(payload: bytes) -> str:
    counts = Counter(
        name
        for character in payload.decode("utf-8", errors="replace")
        if (name := _script_name(character)) is not None
    )
    if not counts:
        return "none"
    # The lexical label is the deterministic tie break, independent of row order.
    return min(counts, key=lambda name: (-counts[name], name))


def _normalized_fingerprint(payload: bytes) -> dict[str, Any]:
    """Record exact-normalized and fixed bottom-k word-shingle fingerprints."""

    normalized = " ".join(
        _normalized_text(payload.decode("utf-8", errors="replace"))
        .casefold()
        .split()
    )
    tokens = re.findall(r"\w+", normalized, flags=re.UNICODE)
    shingles = (
        ["\x1f".join(tokens[index : index + 5]) for index in range(len(tokens) - 4)]
        if len(tokens) >= 5
        else ["\x1f".join(tokens)]
    )
    hashes = sorted(
        {
            int.from_bytes(
                hashlib.blake2s(
                    shingle.encode("utf-8"),
                    digest_size=8,
                    person=b"alm-shng",
                ).digest(),
                "little",
            )
            for shingle in shingles
            if shingle
        }
    )[:8]
    return {
        "normalized_sha256": _sha256(normalized.encode("utf-8")),
        "word_5_shingle_bottom8_u64": hashes,
    }


def _fingerprint_shingle_quads(payload: bytes) -> frozenset[tuple[int, ...]]:
    sketch = _normalized_fingerprint(payload)["word_5_shingle_bottom8_u64"]
    return frozenset(combinations(sketch, NEAR_DUPLICATE_SHARED_SHINGLES))


def _assign_split(split_key: str, seed: int) -> str:
    bucket = _stable_seed("split", seed, split_key) % 1_000
    if bucket < 800:
        return "train"
    if bucket < 900:
        return "dev"
    return "test"


def parse_turns(raw: bytes) -> tuple[Turn, ...]:
    """Parse the export's rendered USER/ASSISTANT chunks with byte offsets."""
    matches = list(_TURN_PATTERN.finditer(raw))
    turns: list[Turn] = []
    for index, match in enumerate(matches):
        separator_start = match.start()
        marker_start = separator_start + (
            2 if raw[separator_start : separator_start + 2] == b"\n\n" else 0
        )
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        turns.append(
            Turn(
                index=index,
                role=match.group(1).decode("ascii"),
                separator_start=separator_start,
                marker_start=marker_start,
                content_start=match.end(),
                end=end,
            )
        )
    return tuple(turns)


def _strict_alternation(turns: Sequence[Turn]) -> bool:
    if not turns or turns[0].role != "USER":
        return False
    return all(
        left.role != right.role for left, right in zip(turns, turns[1:], strict=False)
    )


def _contains_disallowed_control(text: str) -> bool:
    return any(ord(char) < 32 and char not in "\n\t" for char in text)


def _provenance_value(row: dict[str, Any], source_label: str) -> tuple[str, str] | None:
    for key in _PROVENANCE_KEYS:
        value = row.get(key)
        if value is not None and not isinstance(value, (dict, list)):
            return f"{source_label}:{key}:{value}", key
    return None


def _long_turn_hashes(raw: bytes, turns: Sequence[Turn]) -> frozenset[str]:
    hashes = {
        _payload_hash(raw[turn.content_start : turn.end])
        for turn in turns
        if len(b" ".join(raw[turn.content_start : turn.end].split())) >= 48
    }
    return frozenset(hashes)


def _goal_target_bounds(
    conversation: Conversation, kind: str
) -> tuple[int, int, int, int]:
    """Return byte and turn bounds for the self-supervised goal target."""

    target = conversation.target
    if kind == "response":
        return target.content_start, target.end, target.index, target.index
    future = conversation.turns[target.index + 1 :]
    if kind == "interlocutor":
        turn = next((item for item in future if item.role == "USER"), None)
        if turn is None:
            raise ValueError("interlocutor target has no later USER turn")
        return turn.content_start, turn.end, turn.index, turn.index
    if kind == "successor":
        if len(future) < 2:
            raise ValueError("successor target requires at least two later turns")
        # Retain role markers and separators so the downstream modality adapter
        # can recover an event trajectory rather than a flat text bag.
        return (
            future[0].separator_start,
            len(conversation.raw),
            future[0].index,
            future[-1].index,
        )
    raise ValueError(f"unknown goal target kind: {kind}")


def _goal_target_payload(conversation: Conversation, kind: str) -> bytes:
    start, end, _, _ = _goal_target_bounds(conversation, kind)
    return conversation.raw[start:end]


def _eligible_conversation(
    *,
    text: str,
    row: dict[str, Any],
    source: SourceSpec,
    source_line: int,
    source_byte_offset: int,
    config: BuildConfig,
) -> tuple[Conversation | None, str | None]:
    if not text.strip():
        return None, "empty_text"
    if _contains_disallowed_control(text):
        return None, "control_character"
    try:
        raw = text.encode("utf-8")
    except UnicodeEncodeError:
        return None, "invalid_unicode"
    if len(raw) > config.max_record_bytes:
        return None, "record_too_large"
    turns = parse_turns(raw)
    minimum_target_index = 2 * config.min_prior_exchanges + 1
    if len(turns) <= minimum_target_index or not _strict_alternation(turns):
        return None, "turn_structure"

    targets = [
        turn
        for turn in turns
        if turn.role == "ASSISTANT"
        and turn.index >= minimum_target_index
        and turn.content_start >= config.min_prefix_bytes
        and turn.end - turn.content_start >= config.min_response_bytes
        and (
            config.goal_target_kind == "response"
            or (
                config.goal_target_kind == "interlocutor"
                and any(item.role == "USER" for item in turns[turn.index + 1 :])
            )
            or (
                config.goal_target_kind == "successor"
                and len(turns) - turn.index - 1 >= 2
            )
        )
    ]
    if not targets:
        return None, "no_long_target"
    target = max(targets, key=lambda turn: turn.content_start)

    episode_id = _hash_normalized_text(text)
    provenance = _provenance_value(row, source.label)
    if provenance is None:
        split_key = episode_id
        split_key_kind = "episode_content_sha256"
    else:
        split_key, split_key_kind = provenance
    return (
        Conversation(
            episode_id=episode_id,
            split_key=split_key,
            split_key_kind=split_key_kind,
            split=_assign_split(split_key, config.seed),
            source_label=source.label,
            declared_source=(
                None if row.get("source") is None else str(row.get("source"))
            ),
            source_path=str(source.path),
            source_line=source_line,
            source_byte_offset=source_byte_offset,
            raw=raw,
            turns=turns,
            target_turn_index=target.index,
            long_turn_hashes=_long_turn_hashes(raw, turns),
        ),
        None,
    )


def _iter_source_rows(source: SourceSpec) -> Iterator[tuple[int, int, dict[str, Any]]]:
    with source.path.open("rb") as handle:
        line_number = 0
        while line_number < source.start_line:
            if not handle.readline():
                return
            line_number += 1
        emitted = 0
        while source.scan_limit is None or emitted < source.scan_limit:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            current_line = line_number
            line_number += 1
            emitted += 1
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                yield current_line, offset, {"_parse_error": True}
                continue
            if not isinstance(row, dict):
                yield current_line, offset, {"_parse_error": True}
                continue
            yield current_line, offset, row


def load_conversations(
    config: BuildConfig,
) -> tuple[list[Conversation], dict[str, dict[str, int]]]:
    """Load, validate, and content-deduplicate eligible conversations."""
    by_id: dict[str, Conversation] = {}
    diagnostics: dict[str, Counter[str]] = defaultdict(Counter)
    for source in config.sources:
        if not source.path.is_file():
            raise FileNotFoundError(source.path)
        for line_number, offset, row in _iter_source_rows(source):
            diagnostics[source.label]["scanned"] += 1
            if row.get("_parse_error"):
                diagnostics[source.label]["json_error"] += 1
                continue
            text = row.get("text")
            if not isinstance(text, str):
                diagnostics[source.label]["missing_text"] += 1
                continue
            conversation, rejection = _eligible_conversation(
                text=text,
                row=row,
                source=source,
                source_line=line_number,
                source_byte_offset=offset,
                config=config,
            )
            if conversation is None:
                diagnostics[source.label][str(rejection)] += 1
                continue
            if conversation.episode_id in by_id:
                diagnostics[source.label]["duplicate_episode"] += 1
                continue
            by_id[conversation.episode_id] = conversation
            diagnostics[source.label]["eligible"] += 1
    conversations = sorted(by_id.values(), key=lambda item: item.episode_id)
    return conversations, {
        source: dict(sorted(counts.items()))
        for source, counts in sorted(diagnostics.items())
    }


def _select_training(
    candidates: Sequence[Conversation], byte_budget: int, seed: int
) -> list[Conversation]:
    by_source: dict[str, list[Conversation]] = defaultdict(list)
    for conversation in candidates:
        by_source[conversation.source_label].append(conversation)
    for label, rows in by_source.items():
        random.Random(_stable_seed(seed, "train", label)).shuffle(rows)

    positions = {label: 0 for label in by_source}
    selected_bytes = {label: 0 for label in by_source}
    selected: list[Conversation] = []
    total = 0
    active = set(by_source)
    while active:
        label = min(active, key=lambda item: (selected_bytes[item], item))
        rows = by_source[label]
        position = positions[label]
        chosen = None
        while position < len(rows):
            candidate = rows[position]
            position += 1
            if total + len(candidate.raw) <= byte_budget:
                chosen = candidate
                break
        positions[label] = position
        if chosen is None:
            active.remove(label)
            continue
        selected.append(chosen)
        total += len(chosen.raw)
        selected_bytes[label] += len(chosen.raw)
        if total >= byte_budget:
            break
    return selected


def _horizon_bin(value: int) -> int:
    for edge in (4_096, 2_048, 1_024, 512):
        if value >= edge:
            return edge
    return 0


def _response_bin(value: int) -> int:
    for edge in (512, 256, 128, 64):
        if value >= edge:
            return edge
    return 0


def _bridge_payload(
    conversation: Conversation, minimum_bytes: int
) -> tuple[bytes, int] | None:
    for turn in conversation.turns:
        if turn.role == "ASSISTANT" and turn.end >= minimum_bytes:
            return conversation.raw[: turn.end], turn.index
    return None


def _build_pair(
    a: Conversation,
    b: Conversation,
    bridge: Conversation,
    bridge_payload: bytes,
    bridge_turn_end: int,
    config: BuildConfig,
) -> Pair:
    common_tail = _HOLD_PREFIX + bridge_payload + _RESUME_SUFFIX
    context_a = a.unanswered_prefix + common_tail
    context_b = b.unanswered_prefix + common_tail
    score_length = min(config.score_bytes, len(a.response), len(b.response))
    if score_length < config.min_response_bytes:
        raise AssertionError("pair response window fell below configured minimum")
    if len(bridge_payload) < config.min_bridge_bytes:
        raise AssertionError("bridge shorter than configured minimum")
    if context_a[-len(common_tail) :] != context_b[-len(common_tail) :]:
        raise AssertionError("pair contexts do not share their declared tail")
    if config.local_context_bytes > len(common_tail):
        raise AssertionError("common tail is shorter than local context")
    return Pair(
        a=a,
        b=b,
        bridge=bridge,
        bridge_payload=bridge_payload,
        bridge_turn_end=bridge_turn_end,
        common_tail=common_tail,
        context_a=context_a,
        context_b=context_b,
        response_a=a.response[:score_length],
        response_b=b.response[:score_length],
        full_response_a=a.response,
        full_response_b=b.response,
        goal_target_a=_goal_target_payload(a, config.goal_target_kind),
        goal_target_b=_goal_target_payload(b, config.goal_target_kind),
        score_length=score_length,
    )


def make_pairs(
    candidates: Sequence[Conversation],
    *,
    count: int,
    split: str,
    forbidden_turn_hashes: set[str],
    forbidden_request_hashes: set[str],
    forbidden_response_hashes: set[str],
    forbidden_request_quads: set[tuple[int, ...]],
    forbidden_response_quads: set[tuple[int, ...]],
    config: BuildConfig,
) -> list[Pair]:
    if count == 0:
        return []
    usable = [
        row
        for row in candidates
        if row.split == split
        and not (row.long_turn_hashes & forbidden_turn_hashes)
        and row.target_user_normalized_hash not in forbidden_request_hashes
        and row.response_normalized_hash not in forbidden_response_hashes
        and row.target_user_shingle_quads.isdisjoint(forbidden_request_quads)
        and row.response_shingle_quads.isdisjoint(forbidden_response_quads)
    ]
    groups: dict[tuple[str, str, int, int], list[Conversation]] = defaultdict(list)
    for row in usable:
        key = (
            row.source_label,
            row.target_user_script,
            _horizon_bin(row.horizon_bytes),
            _response_bin(len(row.response)),
        )
        groups[key].append(row)

    anchor_pairs: list[tuple[Conversation, Conversation]] = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda item: item.episode_id)
        random.Random(_stable_seed(config.seed, split, "group", *key)).shuffle(rows)
        anchor_pairs.extend(zip(rows[::2], rows[1::2], strict=False))
    random.Random(_stable_seed(config.seed, split, "anchor-pairs")).shuffle(
        anchor_pairs
    )

    bridge_rows: dict[str, list[tuple[Conversation, bytes, int]]] = defaultdict(list)
    for row in usable:
        result = _bridge_payload(row, config.min_bridge_bytes)
        if result is not None:
            payload, turn_end = result
            bridge_rows[row.source_label].append((row, payload, turn_end))
    for label, rows in bridge_rows.items():
        rows.sort(key=lambda item: item[0].episode_id)
        random.Random(_stable_seed(config.seed, split, "bridges", label)).shuffle(rows)

    used: set[str] = set()
    used_requests: set[str] = set()
    used_responses: set[str] = set()
    used_request_quads: set[tuple[int, ...]] = set()
    used_response_quads: set[tuple[int, ...]] = set()
    pairs: list[Pair] = []
    for a, b in anchor_pairs:
        if a.episode_id in used or b.episode_id in used:
            continue
        request_hashes = {
            a.target_user_normalized_hash,
            b.target_user_normalized_hash,
        }
        response_hashes = {a.response_normalized_hash, b.response_normalized_hash}
        request_quads = a.target_user_shingle_quads | b.target_user_shingle_quads
        response_quads = a.response_shingle_quads | b.response_shingle_quads
        if (
            len(request_hashes) != 2
            or len(response_hashes) != 2
            or request_hashes & used_requests
            or response_hashes & used_responses
            or not a.target_user_shingle_quads.isdisjoint(
                b.target_user_shingle_quads
            )
            or not a.response_shingle_quads.isdisjoint(b.response_shingle_quads)
            or request_quads & used_request_quads
            or response_quads & used_response_quads
        ):
            continue
        donor = next(
            (
                item
                for item in bridge_rows[a.source_label]
                if item[0].episode_id not in used
                and item[0].episode_id not in {a.episode_id, b.episode_id}
            ),
            None,
        )
        if donor is None:
            continue
        bridge, payload, turn_end = donor
        pair = _build_pair(a, b, bridge, payload, turn_end, config)
        pairs.append(pair)
        used.update((a.episode_id, b.episode_id, bridge.episode_id))
        used_requests.update(request_hashes)
        used_responses.update(response_hashes)
        used_request_quads.update(request_quads)
        used_response_quads.update(response_quads)
        if len(pairs) >= count:
            break
    if len(pairs) != count:
        raise ValueError(
            f"requested {count} {split} pairs but constructed {len(pairs)}; "
            f"eligible conversations after leakage filter: {len(usable)}"
        )
    return pairs


def _range_record(start: int, payload: bytes) -> dict[str, int | str]:
    return {
        "start": start,
        "end": start + len(payload),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _provenance_record(
    conversation: Conversation, goal_target_kind: str = "response"
) -> dict[str, Any]:
    target = conversation.target
    target_user = conversation.target_user
    goal_start, goal_end, goal_turn_start, goal_turn_end = _goal_target_bounds(
        conversation, goal_target_kind
    )
    return {
        "episode_id": conversation.episode_id,
        "split_key": conversation.split_key,
        "split_key_kind": conversation.split_key_kind,
        "assigned_split": conversation.split,
        "source_label": conversation.source_label,
        "declared_source": conversation.declared_source,
        "source_path": conversation.source_path,
        "source_line": conversation.source_line,
        "source_byte_offset": conversation.source_byte_offset,
        "episode_bytes": len(conversation.raw),
        "target_turn_index": target.index,
        "target_user_turn_index": target_user.index,
        "target_user_content_start": target_user.content_start,
        "target_user_content_end": target_user.end,
        "target_content_start": target.content_start,
        "target_content_end": target.end,
        "unanswered_prefix_bytes": len(conversation.unanswered_prefix),
        "unanswered_prefix_sha256": _sha256(conversation.unanswered_prefix),
        "target_full_response_bytes": len(conversation.response),
        "target_full_response_sha256": _sha256(conversation.response),
        "target_full_response_fingerprint": _normalized_fingerprint(
            conversation.response
        ),
        "target_user_script": conversation.target_user_script,
        "target_user_fingerprint": _normalized_fingerprint(
            conversation.target_user_payload
        ),
        "goal_target_kind": goal_target_kind,
        "goal_target_local_start": goal_start,
        "goal_target_local_end": goal_end,
        "goal_target_turn_start": goal_turn_start,
        "goal_target_turn_end": goal_turn_end,
        "goal_target_bytes": goal_end - goal_start,
        "goal_target_sha256": _sha256(conversation.raw[goal_start:goal_end]),
    }


def _write_training(
    conversations: Sequence[Conversation], output_dir: Path, config: BuildConfig
) -> tuple[int, list[dict[str, Any]]]:
    binary_path = output_dir / "train.bin"
    index_path = output_dir / "train.index.jsonl"
    rows: list[dict[str, Any]] = []
    offset = 0
    with (
        binary_path.open("wb") as binary,
        index_path.open("w", encoding="utf-8", newline="\n") as index,
    ):
        for conversation in conversations:
            start = offset
            binary.write(conversation.raw)
            offset += len(conversation.raw)
            target = conversation.target
            target_start = start + target.content_start
            target_end = start + target.end
            goal_start, goal_end, goal_turn_start, goal_turn_end = _goal_target_bounds(
                conversation, config.goal_target_kind
            )
            goal_payload = conversation.raw[goal_start:goal_end]
            record = {
                **_provenance_record(conversation, config.goal_target_kind),
                "start": start,
                "end": offset,
                "sha256": _sha256(conversation.raw),
                "target_offsets": {
                    "coordinate_space": "train.bin",
                    "turn_separator_start": start + target.separator_start,
                    "role_marker_start": start + target.marker_start,
                    "content_start": target_start,
                    "content_end": target_end,
                    "score_window_start": target_start,
                    "score_window_end": min(
                        target_end, target_start + config.score_bytes
                    ),
                },
                "goal_target": {
                    "kind": config.goal_target_kind,
                    "coordinate_space": "train.bin",
                    "start": start + goal_start,
                    "end": start + goal_end,
                    "bytes": len(goal_payload),
                    "sha256": _sha256(goal_payload),
                    "turn_start": goal_turn_start,
                    "turn_end": goal_turn_end,
                },
                "unanswered_prefix_range": _range_record(
                    start, conversation.unanswered_prefix
                ),
                "turns": [
                    {
                        **asdict(turn),
                        "separator_start": start + turn.separator_start,
                        "marker_start": start + turn.marker_start,
                        "content_start": start + turn.content_start,
                        "end": start + turn.end,
                    }
                    for turn in conversation.turns
                ],
            }
            index.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            rows.append(record)
    return offset, rows


def _append_blob(handle: Any, payload: bytes) -> dict[str, int | str]:
    start = handle.tell()
    handle.write(payload)
    return _range_record(start, payload)


def _write_pairs(
    pairs: Sequence[Pair], split: str, output_dir: Path, config: BuildConfig
) -> list[dict[str, Any]]:
    binary_name = f"{split}_pairs.bin"
    binary_path = output_dir / binary_name
    sidecar_path = output_dir / f"{split}_pairs.jsonl"
    rows: list[dict[str, Any]] = []
    with (
        binary_path.open("wb") as binary,
        sidecar_path.open("w", encoding="utf-8", newline="\n") as sidecar,
    ):
        for pair_index, pair in enumerate(pairs):
            context_a = _append_blob(binary, pair.context_a)
            context_b = _append_blob(binary, pair.context_b)
            response_a = _append_blob(binary, pair.response_a)
            response_b = _append_blob(binary, pair.response_b)
            full_response_a = _append_blob(binary, pair.full_response_a)
            full_response_b = _append_blob(binary, pair.full_response_b)
            goal_target_a = _append_blob(binary, pair.goal_target_a)
            goal_target_b = _append_blob(binary, pair.goal_target_b)
            local_suffix = pair.common_tail[-config.local_context_bytes :]
            original_prefix_a = _range_record(
                int(context_a["start"]), pair.a.unanswered_prefix
            )
            original_prefix_b = _range_record(
                int(context_b["start"]), pair.b.unanswered_prefix
            )
            target_user_a = _range_record(
                int(context_a["start"]) + pair.a.target_user.content_start,
                pair.a.target_user_payload,
            )
            target_user_b = _range_record(
                int(context_b["start"]) + pair.b.target_user.content_start,
                pair.b.target_user_payload,
            )
            pair_id = _sha256(
                (
                    f"{split}:{pair_index}:{pair.a.episode_id}:"
                    f"{pair.b.episode_id}:{pair.bridge.episode_id}:"
                    f"{config.goal_target_kind}"
                ).encode("ascii")
            )
            record = {
                "schema_version": SCHEMA_VERSION,
                "pair_id": pair_id,
                "split": split,
                "construction": "symmetric_interruption_resume_crossover",
                "blob_file": binary_name,
                "source_label": pair.a.source_label,
                "goal_target_kind": config.goal_target_kind,
                "matching": {
                    "target_user_script": pair.a.target_user_script,
                    "horizon_bin": _horizon_bin(pair.a.horizon_bytes),
                    "response_length_bin": _response_bin(len(pair.a.response)),
                },
                "episodes": {
                    "a": _provenance_record(pair.a, config.goal_target_kind),
                    "b": _provenance_record(pair.b, config.goal_target_kind),
                    "bridge": {
                        **_provenance_record(pair.bridge),
                        "bridge_payload_bytes": len(pair.bridge_payload),
                        "bridge_payload_sha256": _sha256(pair.bridge_payload),
                        "bridge_last_turn_index": pair.bridge_turn_end,
                    },
                },
                "blob_ranges": {
                    "context_a": context_a,
                    "context_b": context_b,
                    "response_a": response_a,
                    "response_b": response_b,
                    "full_response_a": full_response_a,
                    "full_response_b": full_response_b,
                    "goal_target_a": goal_target_a,
                    "goal_target_b": goal_target_b,
                },
                "audit_subranges": {
                    "original_prefix_a": original_prefix_a,
                    "original_prefix_b": original_prefix_b,
                    "target_user_a": target_user_a,
                    "target_user_b": target_user_b,
                },
                "matched_assignment": {
                    "context_a": "response_a",
                    "context_b": "response_b",
                },
                "response_score_bytes": pair.score_length,
                "common_suffix": {
                    "bytes": len(pair.common_tail),
                    "sha256": _sha256(pair.common_tail),
                    "context_a_start": context_a["end"] - len(pair.common_tail),
                    "context_b_start": context_b["end"] - len(pair.common_tail),
                },
                "local_suffix": {
                    "bytes": config.local_context_bytes,
                    "sha256": _sha256(local_suffix),
                    "hex": local_suffix.hex(),
                },
                "invariants": {
                    "three_distinct_episodes": len(
                        {pair.a.episode_id, pair.b.episode_id, pair.bridge.episode_id}
                    )
                    == 3,
                    "same_source_candidates": (
                        pair.a.source_label == pair.b.source_label
                    ),
                    "same_target_user_script": (
                        pair.a.target_user_script == pair.b.target_user_script
                    ),
                    "identical_common_suffix": pair.context_a.endswith(pair.common_tail)
                    and pair.context_b.endswith(pair.common_tail),
                    "equal_response_windows": len(pair.response_a)
                    == len(pair.response_b),
                    "full_responses_preserved": (
                        len(pair.full_response_a)
                        == pair.a.target.end - pair.a.target.content_start
                        and len(pair.full_response_b)
                        == pair.b.target.end - pair.b.target.content_start
                    ),
                    "nonempty_goal_targets": bool(pair.goal_target_a)
                    and bool(pair.goal_target_b),
                    "bridge_at_least_minimum": len(pair.bridge_payload)
                    >= config.min_bridge_bytes,
                    "unanswered_prefix_ends_at_target_user_end": (
                        len(pair.a.unanswered_prefix) == pair.a.target_user.end
                        and len(pair.b.unanswered_prefix) == pair.b.target_user.end
                    ),
                },
            }
            sidecar.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            rows.append(record)
    return rows


def _file_record(path: Path) -> dict[str, int | str]:
    payload = path.read_bytes()
    return {"path": path.name, "bytes": len(payload), "sha256": _sha256(payload)}


def _validate_config(config: BuildConfig) -> None:
    if not config.sources:
        raise ValueError("at least one source is required")
    labels = [source.label for source in config.sources]
    if len(labels) != len(set(labels)):
        raise ValueError("source labels must be unique")
    if any(source.start_line < 0 for source in config.sources):
        raise ValueError("source start lines must be non-negative")
    if any(
        source.scan_limit is not None and source.scan_limit < 0
        for source in config.sources
    ):
        raise ValueError("source scan limits must be non-negative or None")
    if config.train_bytes <= 0:
        raise ValueError("train_bytes must be positive")
    if config.dev_pairs < 0 or config.test_pairs < 0:
        raise ValueError("pair counts must be non-negative")
    if config.min_bridge_bytes < MINIMUM_BRIDGE_BYTES:
        raise ValueError(f"min_bridge_bytes must be at least {MINIMUM_BRIDGE_BYTES}")
    if config.min_response_bytes <= 0 or config.score_bytes <= 0:
        raise ValueError("response and score byte counts must be positive")
    if config.local_context_bytes <= 0:
        raise ValueError("local_context_bytes must be positive")
    if config.max_record_bytes < config.min_prefix_bytes:
        raise ValueError("max_record_bytes must cover min_prefix_bytes")
    if config.goal_target_kind not in GOAL_TARGET_KINDS:
        raise ValueError(
            "goal_target_kind must be one of " + ", ".join(GOAL_TARGET_KINDS)
        )


def _prepare_output(config: BuildConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "train.bin",
        "train.index.jsonl",
        "dev_pairs.bin",
        "dev_pairs.jsonl",
        "test_pairs.bin",
        "test_pairs.jsonl",
        "manifest.json",
    )
    existing = [
        config.output_dir / name
        for name in names
        if (config.output_dir / name).exists()
    ]
    if existing and not config.overwrite:
        raise FileExistsError(
            "refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
        )


def _turn_hash_union(conversations: Iterable[Conversation]) -> set[str]:
    result: set[str] = set()
    for conversation in conversations:
        result.update(conversation.long_turn_hashes)
    return result


def build_dataset(config: BuildConfig) -> dict[str, Any]:
    """Build the binary corpus, pair blobs, sidecars, and manifest."""
    _validate_config(config)
    _prepare_output(config)
    conversations, diagnostics = load_conversations(config)
    by_split = Counter(conversation.split for conversation in conversations)
    by_split_source = Counter(
        (conversation.split, conversation.source_label)
        for conversation in conversations
    )

    train = _select_training(
        [row for row in conversations if row.split == "train"],
        config.train_bytes,
        config.seed,
    )
    if not train:
        raise ValueError("no training episodes fit the configured byte budget")
    forbidden = _turn_hash_union(train)
    forbidden_requests = {row.target_user_normalized_hash for row in train}
    forbidden_responses = {row.response_normalized_hash for row in train}
    forbidden_request_quads = set().union(
        *(row.target_user_shingle_quads for row in train)
    )
    forbidden_response_quads = set().union(
        *(row.response_shingle_quads for row in train)
    )

    dev = make_pairs(
        conversations,
        count=config.dev_pairs,
        split="dev",
        forbidden_turn_hashes=forbidden,
        forbidden_request_hashes=forbidden_requests,
        forbidden_response_hashes=forbidden_responses,
        forbidden_request_quads=forbidden_request_quads,
        forbidden_response_quads=forbidden_response_quads,
        config=config,
    )
    dev_episodes = [item for pair in dev for item in (pair.a, pair.b, pair.bridge)]
    forbidden.update(_turn_hash_union(dev_episodes))
    forbidden_requests.update(
        row.target_user_normalized_hash for pair in dev for row in (pair.a, pair.b)
    )
    forbidden_responses.update(
        row.response_normalized_hash for pair in dev for row in (pair.a, pair.b)
    )
    forbidden_request_quads.update(
        quad
        for pair in dev
        for row in (pair.a, pair.b)
        for quad in row.target_user_shingle_quads
    )
    forbidden_response_quads.update(
        quad
        for pair in dev
        for row in (pair.a, pair.b)
        for quad in row.response_shingle_quads
    )
    test = make_pairs(
        conversations,
        count=config.test_pairs,
        split="test",
        forbidden_turn_hashes=forbidden,
        forbidden_request_hashes=forbidden_requests,
        forbidden_response_hashes=forbidden_responses,
        forbidden_request_quads=forbidden_request_quads,
        forbidden_response_quads=forbidden_response_quads,
        config=config,
    )
    test_episodes = [item for pair in test for item in (pair.a, pair.b, pair.bridge)]

    train_byte_count, train_rows = _write_training(train, config.output_dir, config)
    dev_rows = _write_pairs(dev, "dev", config.output_dir, config)
    test_rows = _write_pairs(test, "test", config.output_dir, config)

    source_records = []
    for source in config.sources:
        stat = source.path.stat()
        end_line = (
            None if source.scan_limit is None else source.start_line + source.scan_limit
        )
        eligible_lines = [
            row.source_line for row in conversations if row.source_label == source.label
        ]
        source_records.append(
            {
                "label": source.label,
                "path": str(source.path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "start_line_inclusive": source.start_line,
                "end_line_exclusive": end_line,
                "scan_limit": source.scan_limit,
                "scan_limit_semantics": "maximum rows read after start_line",
                "line_numbering": "zero_based",
                "eligible_source_line_min": (
                    None if not eligible_lines else min(eligible_lines)
                ),
                "eligible_source_line_max": (
                    None if not eligible_lines else max(eligible_lines)
                ),
            }
        )
    config_record = {
        key: value
        for key, value in asdict(config).items()
        if key not in {"sources", "output_dir", "overwrite"}
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": "episode_aligned_goal_trajectory_crossover",
        "claim_scope": (
            "natural pairs measure long-context compatibility, not goal identity alone"
        ),
        "source_scan_policy": {
            "line_numbering": "zero_based",
            "start_line": "inclusive",
            "scan_limit": "maximum rows read after start_line",
        },
        "split_policy": {
            "unit": (
                "first available declared conversation/thread/session/id within each "
                "source; otherwise exact normalized episode-content SHA-256"
            ),
            "declared_provenance_keys_in_priority_order": list(_PROVENANCE_KEYS),
            "split_key_kind_counts": dict(
                sorted(Counter(row.split_key_kind for row in conversations).items())
            ),
            "fuzzy_template_or_user_cluster_split": False,
            "fixed_target_request_response_near_duplicate_filter": {
                "sketch": "bottom-8 unsigned 64-bit word-5-shingle hashes",
                "shared_hash_threshold": NEAR_DUPLICATE_SHARED_SHINGLES,
            },
        },
        "known_leakage_limitations": [
            (
                "Rows without a declared conversation/thread/session/id are split by "
                "exact episode content only; latent user or template families are not "
                "clustered."
            ),
            (
                "Cross-split turn filtering uses normalized exact hashes for turns of "
                "at least 48 compacted bytes. Target requests and responses also use "
                "normalized exact hashes plus a fixed bottom-8 word-5-shingle screen "
                "at four shared hashes; paraphrases below that threshold remain."
            ),
        ],
        "config": config_record,
        "sources": source_records,
        "source_diagnostics": diagnostics,
        "eligible": {
            "total": len(conversations),
            "by_split": dict(sorted(by_split.items())),
            "by_split_and_source": {
                f"{split}:{source}": count
                for (split, source), count in sorted(by_split_source.items())
            },
        },
        "selected": {
            "train_episodes": len(train),
            "train_bytes": train_byte_count,
            "dev_pairs": len(dev_rows),
            "test_pairs": len(test_rows),
            "dev_distinct_episodes": len({row.episode_id for row in dev_episodes}),
            "test_distinct_episodes": len({row.episode_id for row in test_episodes}),
        },
        "invariants": {
            "episode_boundaries_are_sidecar_only": True,
            "transient_state_reset_required_at_train_index_boundaries": True,
            "future_targets_must_not_cross_train_index_boundaries": True,
            "pair_construction_uses_no_model_scores": True,
            "candidate_and_bridge_episodes_are_unique_within_each_split": True,
            "train_dev_test_episode_ids_disjoint": len(
                {row.episode_id for row in train}
                & {row.episode_id for row in dev_episodes + test_episodes}
            )
            == 0,
            "selected_long_turn_hashes_disjoint": not (
                _turn_hash_union(train) & _turn_hash_union(dev_episodes + test_episodes)
            )
            and not (_turn_hash_union(dev_episodes) & _turn_hash_union(test_episodes)),
            "selected_target_request_hashes_disjoint": not (
                {row.target_user_normalized_hash for row in train}
                & {
                    row.target_user_normalized_hash
                    for pair in dev + test
                    for row in (pair.a, pair.b)
                }
            )
            and not (
                {
                    row.target_user_normalized_hash
                    for pair in dev
                    for row in (pair.a, pair.b)
                }
                & {
                    row.target_user_normalized_hash
                    for pair in test
                    for row in (pair.a, pair.b)
                }
            ),
            "selected_target_response_hashes_disjoint": not (
                {row.response_normalized_hash for row in train}
                & {
                    row.response_normalized_hash
                    for pair in dev + test
                    for row in (pair.a, pair.b)
                }
            )
            and not (
                {
                    row.response_normalized_hash
                    for pair in dev
                    for row in (pair.a, pair.b)
                }
                & {
                    row.response_normalized_hash
                    for pair in test
                    for row in (pair.a, pair.b)
                }
            ),
            "selected_target_request_bottom8_shingle_quads_disjoint": not (
                set().union(*(row.target_user_shingle_quads for row in train))
                & set().union(
                    *(
                        row.target_user_shingle_quads
                        for pair in dev + test
                        for row in (pair.a, pair.b)
                    )
                )
            )
            and not (
                set().union(
                    *(
                        row.target_user_shingle_quads
                        for pair in dev
                        for row in (pair.a, pair.b)
                    )
                )
                & set().union(
                    *(
                        row.target_user_shingle_quads
                        for pair in test
                        for row in (pair.a, pair.b)
                    )
                )
            ),
            "selected_target_response_bottom8_shingle_quads_disjoint": not (
                set().union(*(row.response_shingle_quads for row in train))
                & set().union(
                    *(
                        row.response_shingle_quads
                        for pair in dev + test
                        for row in (pair.a, pair.b)
                    )
                )
            )
            and not (
                set().union(
                    *(
                        row.response_shingle_quads
                        for pair in dev
                        for row in (pair.a, pair.b)
                    )
                )
                & set().union(
                    *(
                        row.response_shingle_quads
                        for pair in test
                        for row in (pair.a, pair.b)
                    )
                )
            ),
            "selected_source_lines_within_declared_windows": all(
                row.source_line >= source.start_line
                and (
                    source.scan_limit is None
                    or row.source_line < source.start_line + source.scan_limit
                )
                for row in train + dev_episodes + test_episodes
                for source in config.sources
                if row.source_label == source.label
            ),
        },
        "outputs": [
            _file_record(config.output_dir / name)
            for name in (
                "train.bin",
                "train.index.jsonl",
                "dev_pairs.bin",
                "dev_pairs.jsonl",
                "test_pairs.bin",
                "test_pairs.jsonl",
            )
        ],
        "index_rows": len(train_rows),
    }
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmsys", type=Path, required=True)
    parser.add_argument(
        "--everyday",
        type=Path,
        help="Optional second source; omit it for a fresh LMSYS-only corpus.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--start-lmsys-line",
        type=int,
        default=0,
        help="Zero-based inclusive LMSYS source line at which scanning begins.",
    )
    parser.add_argument("--scan-lmsys", type=int, default=50_000)
    parser.add_argument(
        "--start-everyday-line",
        type=int,
        default=0,
        help="Zero-based inclusive Everyday source line at which scanning begins.",
    )
    parser.add_argument(
        "--scan-everyday",
        type=int,
        default=0,
        help="0 scans the complete (small) Everyday file.",
    )
    parser.add_argument("--train-bytes", type=int, default=96_000)
    parser.add_argument("--dev-pairs", type=int, default=64)
    parser.add_argument("--test-pairs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=82_972)
    parser.add_argument("--min-prefix-bytes", type=int, default=512)
    parser.add_argument("--min-response-bytes", type=int, default=64)
    parser.add_argument("--score-bytes", type=int, default=128)
    parser.add_argument("--min-bridge-bytes", type=int, default=512)
    parser.add_argument("--local-context-bytes", type=int, default=8)
    parser.add_argument("--max-record-bytes", type=int, default=16_384)
    parser.add_argument("--min-prior-exchanges", type=int, default=2)
    parser.add_argument(
        "--goal-target-kind", choices=GOAL_TARGET_KINDS, default="response"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    sources = []
    if args.everyday is not None:
        sources.append(
            SourceSpec(
                "everyday",
                args.everyday,
                None if args.scan_everyday == 0 else args.scan_everyday,
                args.start_everyday_line,
            )
        )
    sources.append(
        SourceSpec(
            "lmsys",
            args.lmsys,
            None if args.scan_lmsys == 0 else args.scan_lmsys,
            args.start_lmsys_line,
        )
    )
    config = BuildConfig(
        sources=tuple(sources),
        output_dir=args.output_dir,
        train_bytes=args.train_bytes,
        dev_pairs=args.dev_pairs,
        test_pairs=args.test_pairs,
        seed=args.seed,
        min_prefix_bytes=args.min_prefix_bytes,
        min_response_bytes=args.min_response_bytes,
        score_bytes=args.score_bytes,
        min_bridge_bytes=args.min_bridge_bytes,
        local_context_bytes=args.local_context_bytes,
        max_record_bytes=args.max_record_bytes,
        min_prior_exchanges=args.min_prior_exchanges,
        goal_target_kind=args.goal_target_kind,
        overwrite=args.overwrite,
    )
    manifest = build_dataset(config)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
