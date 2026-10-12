#!/usr/bin/env python
"""Round-trip retrieval test for gate-written content - the validation
gap flagged directly: nothing in the training corpus ever generates a
query against chat/filler content, so run_step's own eval_retrieval_accuracy
only ever measures the scripted-distractor path. This answers the actual
question: if the gate writes a fact into gate_bank, can a later, real
paraphrase-style query find it again?

Method: a set of hand-authored (write_text, query_text, correct_id) triples
- written in the same style real gate-selected content actually takes
(stated decisions/preferences/facts, not templated distractor statements),
matching the FreeGenPrompt "binding" category's existing convention in
manifoldflow_production_train.py rather than inventing a new one. Each
case's write_text is pooled and key-projected exactly like run_step's
Phase 1.5 does, written into one shared BasinMemory (novelty_threshold
matching --gate-novelty-threshold, same isolated-bank convention Lane.gate_bank
uses) alongside every OTHER case's write - so retrieval has to discriminate
among real competing content, not just recall a single fact from an
otherwise-empty bank. Then each case's query_text is pooled/keyed the same
way and read back; retrieval is correct if the returned basin's payload is
that case's own write, not another case's.

A second pass measures novelty-merge behavior directly: writes a
paraphrase of each case's write_text (same fact, different wording) and
checks whether it merges into the existing basin (a real dedup) or
creates a spurious new one - the direct test of whether
--gate-novelty-threshold is doing anything at the current value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from manifoldflow.continuous_memory import BasinMemory, Source
from manifoldflow.experiments.manifoldflow_production_train import (
    _pool,
    _tokenizer_identity,
    load_resume_state,
)
from manifoldflow.sonic_key_projection import (
    SonicKeyProjection,
    SonicKeyProjectionConfig,
    load_key_projection_state_dict,
)
from manifoldflow.sonic_ssm import SonicSSMConfig, SonicSSMLanguageModel

# Each case: (id, write_text, query_text, paraphrase_text). Deliberately
# styled like real gate-selected content spotted in
# scripts/inspect_gate_writes.py's output (a stated decision/preference/
# fact within a turn) rather than distractor's templated statements -
# that's exactly the content type this whole mechanism has to work for.
_CASES: list[tuple[str, str, str, str]] = [
    (
        "file_format",
        "I choose .png as the supported extension for the image export feature.",
        "What file extension did we decide to support for image export?",
        "The image export feature will support the .png extension, as decided earlier.",
    ),
    (
        "deploy_day",
        "Let's plan to deploy the new pricing page next Thursday afternoon.",
        "When are we deploying the new pricing page?",
        "We agreed the new pricing page goes out this coming Thursday, in the afternoon.",
    ),
    (
        "db_choice",
        "I've decided to use PostgreSQL instead of MongoDB for the new service, since we need strong transactional guarantees.",
        "Which database did we choose for the new service?",
        "For the new service we're going with PostgreSQL rather than MongoDB.",
    ),
    (
        "color_pref",
        "My favorite color for the dashboard theme is a deep teal, not the default blue.",
        "What color did I say I wanted for the dashboard theme?",
        "For the dashboard theme, deep teal is the color I prefer over the default blue.",
    ),
    (
        "budget_cap",
        "Let's cap the monthly cloud spend at $4,000 for this project.",
        "What's the monthly cloud spend cap we agreed on?",
        "We set a $4,000 per month ceiling on cloud costs for this project.",
    ),
    (
        "meeting_time",
        "The standup is moving from 9am to 10:30am starting next week.",
        "What time is the standup moving to?",
        "Starting next week, standup shifts from 9 in the morning to 10:30.",
    ),
    (
        "library_choice",
        "I'll use Zod for schema validation instead of Yup in this project.",
        "Which validation library are we using?",
        "For schema validation on this project, Zod was picked over Yup.",
    ),
    (
        "vacation_dates",
        "I'm taking vacation from the 12th through the 19th of next month.",
        "When is my vacation?",
        "My time off is scheduled for the 12th to the 19th of next month.",
    ),
    (
        "api_version",
        "We're pinning the payments API to v3, not upgrading to v4 yet.",
        "Which version of the payments API are we using?",
        "The payments integration stays on API v3 for now, v4 is on hold.",
    ),
    (
        "team_lead",
        "Priya is taking over as lead for the onboarding redesign project.",
        "Who's leading the onboarding redesign project?",
        "The onboarding redesign now has Priya as its project lead.",
    ),
]


def _encode(model, tokenizer, key_projection, text: str, device: str, window_tokens: int) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    padded = ids + [tokenizer.pad_token_id or tokenizer.eos_token_id] * (window_tokens - len(ids))
    ids_tensor = torch.tensor([padded[:window_tokens]], device=device, dtype=torch.long)
    with torch.no_grad():
        hidden, _ = model.forward_hidden(ids_tensor)
    hidden_float = hidden[0].float()
    pooled = _pool(hidden_float, (0, min(len(ids), window_tokens - 1)), window_tokens)
    with torch.no_grad():
        key = key_projection(pooled)[0]
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument(
        "--gate-novelty-threshold",
        type=float,
        nargs="+",
        default=[0.15],
        help="One or more values to test - each gets its own independent "
        "merge-check pass (retrieval accuracy is threshold-independent, "
        "computed once).",
    )
    parser.add_argument("--scan-backend", default="doubling")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window-tokens", type=int, default=8192)
    args = parser.parse_args()

    import transformers

    device = args.device
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    model_config = SonicSSMConfig(**checkpoint["model_config"])
    model = SonicSSMLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.configure_execution(scan_backend=args.scan_backend)

    tokenizer_dir = args.tokenizer_dir or (
        Path(__file__).resolve().parents[1] / "resources" / "tokenizers" / "gtr_arw_english_50k"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_dir))

    key_projection = SonicKeyProjection(SonicKeyProjectionConfig(dim=model_config.d_model)).to(device)
    tokenizer_identity = _tokenizer_identity(tokenizer)
    state = load_resume_state(
        args.checkpoint_path, model_config=model_config, tokenizer_identity=tokenizer_identity
    )
    load_key_projection_state_dict(key_projection, state["key_projection"])
    key_projection.eval()
    print(f"loaded checkpoint at update={state.get('update')}", flush=True)

    # Cache every embedding once - threshold sweeps below reuse these,
    # no reason to re-run the model per threshold value.
    write_keys = {}
    query_keys = {}
    paraphrase_keys = {}
    for case_id, write_text, query_text, paraphrase in _CASES:
        write_keys[case_id] = _encode(model, tokenizer, key_projection, write_text, device, args.window_tokens)
        query_keys[case_id] = _encode(model, tokenizer, key_projection, query_text, device, args.window_tokens)
        paraphrase_keys[case_id] = _encode(model, tokenizer, key_projection, paraphrase, device, args.window_tokens)

    # ---- Retrieval accuracy among real competing content - independent
    # of novelty threshold (read() doesn't consult it, only write() does).
    bank = BasinMemory(dimension=model_config.d_model, novelty_threshold=0.0)
    basin_to_case: dict[int, str] = {}
    for case_id, _write_text, _query_text, _paraphrase in _CASES:
        basin_id = int(bank.write_always_new(write_keys[case_id], "w", Source("user")))
        basin_to_case[basin_id] = case_id

    correct = 0
    print("\n=== retrieval among real competing content ===", flush=True)
    for case_id, _write_text, _query_text, _paraphrase in _CASES:
        hits = bank.read(query_keys[case_id], top_k=1)
        hit_case = basin_to_case.get(hits[0].basin_id) if hits else None
        is_correct = hit_case == case_id
        correct += int(is_correct)
        status = "OK  " if is_correct else "MISS"
        score_note = f"(score={hits[0].score:.3f})" if hits else "(no hits)"
        print(f"[{status}] query={case_id!r} -> retrieved={hit_case!r} {score_note}", flush=True)
    retrieval_acc = correct / len(_CASES)
    print(f"\nretrieval accuracy: {correct}/{len(_CASES)} = {retrieval_acc:.3f}", flush=True)

    # ---- Threshold sweep: for each candidate --gate-novelty-threshold,
    # measure BOTH directions - true-merge rate (a paraphrase of the SAME
    # fact should merge) and false-merge rate (a paraphrase of a
    # DIFFERENT fact should NOT merge into some unrelated existing
    # basin) - a threshold picked from only the first number risks being
    # too loose, corrupting unrelated facts (confirmed as a real failure
    # mode below, not hypothetical).
    print("\n=== novelty-threshold sweep ===", flush=True)

    def _fresh_case_bank(threshold: float) -> tuple[BasinMemory, dict[int, str]]:
        bank_ = BasinMemory(dimension=model_config.d_model, novelty_threshold=threshold)
        mapping: dict[int, str] = {}
        for case_id, _write_text, _query_text, _paraphrase in _CASES:
            basin_id = int(bank_.write(write_keys[case_id], "w", Source("user")))
            mapping[basin_id] = case_id
        return bank_, mapping

    for threshold in args.gate_novelty_threshold:
        # True-merge check: one shared bank, paraphrases applied in
        # sequence - each case's own paraphrase should merge into that
        # case's own basin (cases don't interact with each other here).
        sweep_bank, sweep_basin_to_case = _fresh_case_bank(threshold)
        true_merges = 0
        for case_id, _write_text, _query_text, _paraphrase in _CASES:
            basins_before = sweep_bank.basin_count
            basin_id = int(sweep_bank.write(paraphrase_keys[case_id], "p", Source("user")))
            merged = sweep_bank.basin_count == basins_before
            correct_merge = merged and sweep_basin_to_case.get(basin_id) == case_id
            true_merges += int(correct_merge)
            if merged and not correct_merge:
                print(
                    f"  [threshold={threshold}] FALSE MERGE: same-fact paraphrase of "
                    f"{case_id!r} landed on {sweep_basin_to_case.get(basin_id)!r}'s basin",
                    flush=True,
                )

        # False-merge probe: a fresh 10-basin bank per case (not an
        # accumulating one - a probe write must never become a target
        # for a later probe in this same sweep). Writes that case's own
        # QUERY key (not its paraphrase - a different text, so this is an
        # independent check) and asks: does it merge into some OTHER
        # case's basin, one that shouldn't apply to it at all?
        false_merges = 0
        cross_pairs = 0
        for probe_case, _wt, _qt, _pt in _CASES:
            fresh_bank, fresh_mapping = _fresh_case_bank(threshold)
            basins_before = fresh_bank.basin_count
            basin_id = int(fresh_bank.write(query_keys[probe_case], "probe", Source("user")))
            merged = fresh_bank.basin_count == basins_before
            landed_on = fresh_mapping.get(basin_id)
            cross_pairs += 1
            if merged and landed_on != probe_case:
                false_merges += 1

        print(
            f"threshold={threshold}: true_merge_rate={true_merges}/{len(_CASES)}="
            f"{true_merges / len(_CASES):.3f} false_merge_rate={false_merges}/{cross_pairs}="
            f"{false_merges / cross_pairs:.3f}",
            flush=True,
        )

    print(f"\nsummary: retrieval_acc={retrieval_acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
