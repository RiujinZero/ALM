#!/usr/bin/env python
"""Audit goal-trajectory sidecars for exact and approximate split leakage.

The audit intentionally reads only ``manifest.json`` and JSONL sidecars.  It
never opens the binary payload files and never includes fingerprint values,
source paths, source lines, episode identifiers, or text in its report.

Schema-v4 datasets provide normalized request/response fingerprints and
bottom-8 word-5-shingle sketches.  Older datasets remain comparable wherever
their metadata permits: episode identity, logical source line, and exact raw
full-response SHA-256 are present in the earlier schemas used by this project.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SPLITS = ("train", "dev", "test")
_PAIR_ROLES = ("a", "b", "bridge")
_EXACT_FIELDS = (
    "episode",
    "source_line",
    "request_fingerprint",
    "response_fingerprint",
    "response_payload",
)
_SKETCH_FIELDS = ("request", "response")
_NEAR_DUPLICATE_SHARED_SHINGLES = 4


@dataclass(frozen=True)
class AuditRecord:
    """One episode represented exclusively by sidecar identifiers/hashes."""

    split: str
    episode: str | None
    source_line: tuple[str, int] | None
    request_fingerprint: str | None
    response_fingerprint: str | None
    response_payload: str | None
    request_sketch: frozenset[int]
    response_sketch: frozenset[int]


@dataclass(frozen=True)
class DatasetMetadata:
    """Sanitized dataset metadata and the records extracted from sidecars."""

    name: str
    schema_version: int | None
    records: tuple[AuditRecord, ...]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield payload


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _logical_source_line(record: dict[str, Any]) -> tuple[str, int] | None:
    """Return a source-qualified line without exposing it in the report."""

    line = record.get("source_line")
    if not isinstance(line, int) or isinstance(line, bool):
        return None
    namespace = _optional_string(record.get("declared_source"))
    if namespace is None:
        namespace = _optional_string(record.get("source_label"))
    if namespace is None:
        # A path is a last-resort namespace only.  It remains internal and is
        # never serialized into the aggregate report.
        namespace = _optional_string(record.get("source_path"))
    return (namespace, line) if namespace is not None else None


def _fingerprint(
    record: dict[str, Any], field: str
) -> tuple[str | None, frozenset[int]]:
    value = record.get(field)
    if not isinstance(value, dict):
        return None, frozenset()
    normalized = _optional_string(value.get("normalized_sha256"))
    sketch_value = value.get("word_5_shingle_bottom8_u64")
    if not isinstance(sketch_value, list):
        return normalized, frozenset()
    sketch: set[int] = set()
    for item in sketch_value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{field} contains a non-integer shingle hash")
        if not 0 <= item < 2**64:
            raise ValueError(f"{field} contains an out-of-range shingle hash")
        sketch.add(item)
    if len(sketch) > 8:
        raise ValueError(f"{field} contains more than eight shingle hashes")
    return normalized, frozenset(sketch)


def _audit_record(record: dict[str, Any], split: str) -> AuditRecord:
    request_fingerprint, request_sketch = _fingerprint(
        record, "target_user_fingerprint"
    )
    response_fingerprint, response_sketch = _fingerprint(
        record, "target_full_response_fingerprint"
    )
    return AuditRecord(
        split=split,
        episode=_optional_string(record.get("episode_id")),
        source_line=_logical_source_line(record),
        request_fingerprint=request_fingerprint,
        response_fingerprint=response_fingerprint,
        response_payload=_optional_string(record.get("target_full_response_sha256")),
        request_sketch=request_sketch,
        response_sketch=response_sketch,
    )


def load_dataset_metadata(dataset_dir: Path) -> DatasetMetadata:
    """Load one dataset using sidecars only; binary payloads are never read."""

    dataset_dir = Path(dataset_dir)
    manifest = _read_json(dataset_dir / "manifest.json")
    schema_value = manifest.get("schema_version")
    schema_version = (
        schema_value
        if isinstance(schema_value, int) and not isinstance(schema_value, bool)
        else None
    )
    records: list[AuditRecord] = []
    for row in _read_jsonl(dataset_dir / "train.index.jsonl"):
        records.append(_audit_record(row, "train"))
    for split in ("dev", "test"):
        for row in _read_jsonl(dataset_dir / f"{split}_pairs.jsonl"):
            episodes = row.get("episodes")
            if not isinstance(episodes, dict):
                raise ValueError(f"{split} pair is missing its episodes object")
            for role in _PAIR_ROLES:
                episode = episodes.get(role)
                if not isinstance(episode, dict):
                    raise ValueError(f"{split} pair is missing episode role {role}")
                records.append(_audit_record(episode, split))
    return DatasetMetadata(
        name=dataset_dir.name,
        schema_version=schema_version,
        records=tuple(records),
    )


def _exact_overlap(
    left: Sequence[AuditRecord],
    right: Sequence[AuditRecord],
    field: str,
) -> dict[str, Any]:
    left_values = [getattr(record, field) for record in left]
    right_values = [getattr(record, field) for record in right]
    left_counter = Counter(value for value in left_values if value is not None)
    right_counter = Counter(value for value in right_values if value is not None)
    shared = left_counter.keys() & right_counter.keys()
    result: dict[str, Any] = {
        "left_records_with_metadata": sum(left_counter.values()),
        "right_records_with_metadata": sum(right_counter.values()),
    }
    if not left_counter or not right_counter:
        result.update(
            {
                "status": "metadata_unavailable",
                "shared_values": None,
                "cross_record_pairs": None,
                "left_records_in_overlap": None,
                "right_records_in_overlap": None,
            }
        )
        return result
    result.update(
        {
            "status": "compared",
            "shared_values": len(shared),
            "cross_record_pairs": sum(
                left_counter[value] * right_counter[value] for value in shared
            ),
            "left_records_in_overlap": sum(left_counter[value] for value in shared),
            "right_records_in_overlap": sum(
                right_counter[value] for value in shared
            ),
        }
    )
    return result


def _sketch_overlap(
    left: Sequence[AuditRecord],
    right: Sequence[AuditRecord],
    field: str,
) -> dict[str, Any]:
    attribute = f"{field}_sketch"
    left_sketches = [getattr(record, attribute) for record in left]
    right_sketches = [getattr(record, attribute) for record in right]
    left_available = sum(bool(sketch) for sketch in left_sketches)
    right_available = sum(bool(sketch) for sketch in right_sketches)
    result: dict[str, Any] = {
        "left_records_with_metadata": left_available,
        "right_records_with_metadata": right_available,
    }
    if not left_available or not right_available:
        result.update(
            {
                "status": "metadata_unavailable",
                "record_pairs_sharing_at_least_one": None,
                "record_pairs_sharing_at_least_four": None,
                "maximum_shared_shingles": None,
                "maximum_sketch_jaccard": None,
                "shared_shingle_count_histogram": None,
            }
        )
        return result

    right_by_hash: dict[int, list[int]] = defaultdict(list)
    for right_index, sketch in enumerate(right_sketches):
        for shingle_hash in sketch:
            right_by_hash[shingle_hash].append(right_index)
    pair_shared_counts: Counter[tuple[int, int]] = Counter()
    for left_index, sketch in enumerate(left_sketches):
        for shingle_hash in sketch:
            for right_index in right_by_hash.get(shingle_hash, ()):
                pair_shared_counts[(left_index, right_index)] += 1

    histogram = Counter(pair_shared_counts.values())
    maximum_shared = max(histogram, default=0)
    maximum_jaccard = 0.0
    for (left_index, right_index), shared_count in pair_shared_counts.items():
        union_count = (
            len(left_sketches[left_index])
            + len(right_sketches[right_index])
            - shared_count
        )
        if union_count:
            maximum_jaccard = max(maximum_jaccard, shared_count / union_count)
    result.update(
        {
            "status": "compared",
            "record_pairs_sharing_at_least_one": len(pair_shared_counts),
            "record_pairs_sharing_at_least_four": sum(
                count
                for shared_count, count in histogram.items()
                if shared_count >= _NEAR_DUPLICATE_SHARED_SHINGLES
            ),
            "maximum_shared_shingles": maximum_shared,
            "maximum_sketch_jaccard": round(maximum_jaccard, 6),
            "shared_shingle_count_histogram": {
                str(shared_count): histogram[shared_count]
                for shared_count in sorted(histogram)
            },
        }
    )
    return result


def compare_records(
    left: Sequence[AuditRecord], right: Sequence[AuditRecord]
) -> dict[str, Any]:
    """Return aggregate overlap counts without serializing identifying values."""

    exact = {
        field: _exact_overlap(left, right, field) for field in _EXACT_FIELDS
    }
    approximate = {
        field: _sketch_overlap(left, right, field) for field in _SKETCH_FIELDS
    }
    exact_clear = all(
        result["status"] != "compared" or result["shared_values"] == 0
        for result in exact.values()
    )
    provenance_clear = all(
        exact[field]["status"] != "compared"
        or exact[field]["shared_values"] == 0
        for field in ("episode", "source_line")
    )
    content_clear = all(
        exact[field]["status"] != "compared"
        or exact[field]["shared_values"] == 0
        for field in (
            "request_fingerprint",
            "response_fingerprint",
            "response_payload",
        )
    )
    approximate_threshold_clear = all(
        result["status"] != "compared"
        or result["record_pairs_sharing_at_least_four"] == 0
        for result in approximate.values()
    )
    return {
        "left_records": len(left),
        "right_records": len(right),
        "exact": exact,
        "approximate_bottom8_word5": approximate,
        "exact_overlap_free_where_comparable": exact_clear,
        "provenance_overlap_free_where_comparable": provenance_clear,
        "content_fingerprint_overlap_free_where_comparable": content_clear,
        "no_pairs_sharing_four_or_more_shingles_where_comparable": (
            approximate_threshold_clear
        ),
    }


def _partition(records: Sequence[AuditRecord], split: str) -> list[AuditRecord]:
    return [record for record in records if record.split == split]


def audit_dataset(
    primary: DatasetMetadata,
    comparisons: Sequence[DatasetMetadata] = (),
) -> dict[str, Any]:
    """Audit primary split isolation and compare it with earlier datasets."""

    split_records = {
        split: _partition(primary.records, split) for split in _SPLITS
    }
    within = []
    for left_index, left_split in enumerate(_SPLITS):
        for right_split in _SPLITS[left_index + 1 :]:
            within.append(
                {
                    "left": left_split,
                    "right": right_split,
                    **compare_records(
                        split_records[left_split], split_records[right_split]
                    ),
                }
            )
    prior = [
        {
            "left": primary.name,
            "right": comparison.name,
            "right_schema_version": comparison.schema_version,
            **compare_records(primary.records, comparison.records),
        }
        for comparison in comparisons
    ]
    all_exact_clear = all(
        item["exact_overlap_free_where_comparable"] for item in [*within, *prior]
    )
    all_provenance_clear = all(
        item["provenance_overlap_free_where_comparable"]
        for item in [*within, *prior]
    )
    all_content_clear = all(
        item["content_fingerprint_overlap_free_where_comparable"]
        for item in [*within, *prior]
    )
    all_approximate_clear = all(
        item["no_pairs_sharing_four_or_more_shingles_where_comparable"]
        for item in [*within, *prior]
    )
    return {
        "audit_version": 1,
        "privacy": {
            "binary_payloads_read": False,
            "raw_text_in_report": False,
            "identifiers_or_fingerprint_values_in_report": False,
        },
        "approximate_method": {
            "stored_field": "word_5_shingle_bottom8_u64",
            "near_duplicate_alert_shared_shingles": (
                _NEAR_DUPLICATE_SHARED_SHINGLES
            ),
            "interpretation": (
                "Aggregate sketch collisions are a screening diagnostic, not an "
                "exact text-similarity measure."
            ),
        },
        "primary": {
            "name": primary.name,
            "schema_version": primary.schema_version,
            "records_by_split": {
                split: len(records) for split, records in split_records.items()
            },
        },
        "within_primary_split_comparisons": within,
        "prior_dataset_comparisons": prior,
        "summary": {
            "exact_overlap_free_where_comparable": all_exact_clear,
            "provenance_overlap_free_where_comparable": all_provenance_clear,
            "content_fingerprint_overlap_free_where_comparable": (
                all_content_clear
            ),
            "no_pairs_sharing_four_or_more_shingles_where_comparable": (
                all_approximate_clear
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="primary dataset directory")
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        type=Path,
        help="prior dataset directory; repeat for more than one",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON report path; stdout is always aggregate JSON",
    )
    parser.add_argument(
        "--fail-on-exact",
        action="store_true",
        help="exit 1 if any comparable exact identity/fingerprint overlaps",
    )
    parser.add_argument(
        "--fail-on-near-duplicate",
        action="store_true",
        help="exit 1 if a sketch pair shares at least four bottom-8 hashes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    primary = load_dataset_metadata(args.dataset)
    comparisons = [load_dataset_metadata(path) for path in args.compare]
    report = audit_dataset(primary, comparisons)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.fail_on_exact and not report["summary"][
        "exact_overlap_free_where_comparable"
    ]:
        return 1
    if args.fail_on_near_duplicate and not report["summary"][
        "no_pairs_sharing_four_or_more_shingles_where_comparable"
    ]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
