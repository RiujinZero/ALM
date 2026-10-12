#!/usr/bin/env python
"""Sanity check: does a salience-gate-filtered write policy retrieve as well
as the current scripted (oracle, "write every distractor statement") policy?

Motivation: the salience gate currently only trains via a side-channel BCE
loss against Qwen labels - nothing reads its output to decide anything live.
Before investing further in wiring it into the real write path (candidate-
span extraction for non-distractor content, live inference plumbing, etc.),
this answers a cheaper, decision-relevant question first: on the one place
we *do* have ground-truth candidate spans structurally for free (distractor
statements, which are also the gate's "always salient" calibration anchor),
does gate-filtered writing lose anything real relative to the scripted
baseline? If retrieval holds up, converging + freezing the gate as-is may be
plenty; if it doesn't, that's a concrete, measured reason to keep training it
(or fix the underlying candidate/context mismatch) before relying on it.

Loads the live production checkpoint read-only (no training-state writes,
safe to run alongside the actual training process) and replays the same
real held-out session stream `run_eval` uses, computing ONE shared Sonic
forward pass per window and then two independent write/retrieve
bookkeeping tracks off the same hidden states:

  - "scripted": writes every valid statement span (today's actual behavior).
  - "gate":     writes only spans where sigmoid(salience_gate(pooled)) is
                above --gate-threshold.

Both tracks see the identical query set (the real due queries in these
windows), so eval_query_count matches exactly and retrieval accuracy is
directly comparable - the only thing that can differ is whether a given
query's target statement was ever actually written into that track's own
bank.

Simplifications versus the real training loop's run_step (acceptable for a
smoke-scale sanity check, not appropriate for a training objective):
  - No answer-decode/port-injection accuracy - retrieval accuracy (was the
    right statement the top-1 hit) is the direct signal for "did the write
    policy matter," and is far cheaper to get right than reproducing the
    batched decode path's quantity-value handling faithfully.
  - value_projection is applied to the same pooled span as key_projection
    (skips value_source="quantity"'s separate value-span pooling) - only
    the key/retrieval side is exercised by this comparison, so an inexact
    value doesn't affect the metric being measured.
  - Unbatched across lanes for write/query bookkeeping (a plain Python
    loop per lane) - fine at smoke scale (a handful of lanes, tens of
    windows), not the batched-for-throughput code run_step needs for a
    178k-update production run.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from manifoldflow.continuous_memory import BasinMemory, Source
from manifoldflow.experiments.manifoldflow_production_train import (
    PrefetchingSessionPool,
    SessionStream,
    Window,
    _pool,
    _tokenizer_identity,
    load_resume_state,
    new_lane,
)
from manifoldflow.salience_gate import SalienceGateConfig, SalienceGateHead
from manifoldflow.sonic_key_projection import (
    SonicKeyProjection,
    SonicKeyProjectionConfig,
    load_key_projection_state_dict,
)
from manifoldflow.sonic_memory_port import SonicMemoryPort, SonicMemoryPortConfig
from manifoldflow.sonic_ssm import SonicSSMConfig, SonicSSMLanguageModel


@dataclass
class ConditionState:
    bank: BasinMemory
    statement_keys: list[torch.Tensor] = field(default_factory=list)
    statement_values: list[torch.Tensor] = field(default_factory=list)
    unit_index_map: dict[int, list[int | None]] = field(default_factory=dict)
    candidate_spans: int = 0
    spans_written: int = 0
    retrieval_total: int = 0
    retrieval_correct: int = 0
    target_missing: int = 0  # query's target statement was never written this track


def _new_condition_state(d_model: int) -> ConditionState:
    return ConditionState(bank=BasinMemory(dimension=d_model, novelty_threshold=0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, default=Path("/path/to/manifoldflow_sessions"))
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--num-lanes", type=int, default=3)
    parser.add_argument("--eval-windows", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=31)
    parser.add_argument("--window-tokens", type=int, default=8192)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    # "hgrn" (what production actually runs) is a Triton kernel - GPU only.
    # "doubling" is the pure-PyTorch fallback, needed for a CPU smoke run.
    parser.add_argument("--scan-backend", default="doubling")
    parser.add_argument("--port-rank", type=int, default=64)
    parser.add_argument("--device", default="cpu")
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
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    tokenizer_identity = _tokenizer_identity(tokenizer)

    port = SonicMemoryPort(
        SonicMemoryPortConfig(
            d_model=model_config.d_model, memory_dim=model_config.d_model, rank=args.port_rank
        )
    ).to(device)
    key_projection = SonicKeyProjection(SonicKeyProjectionConfig(dim=model_config.d_model)).to(device)
    value_projection = SonicKeyProjection(SonicKeyProjectionConfig(dim=model_config.d_model)).to(device)
    salience_gate = SalienceGateHead(SalienceGateConfig(d_model=model_config.d_model)).to(device)

    state = load_resume_state(
        args.checkpoint_path, model_config=model_config, tokenizer_identity=tokenizer_identity
    )
    port.load_state_dict(state["port"])
    load_key_projection_state_dict(key_projection, state["key_projection"])
    load_key_projection_state_dict(value_projection, state["value_projection"])
    if state.get("salience_gate") is None:
        raise SystemExit("checkpoint has no trained salience_gate - nothing to compare")
    salience_gate.load_state_dict(state["salience_gate"])
    port.eval()
    key_projection.eval()
    value_projection.eval()
    salience_gate.eval()
    print(
        f"loaded checkpoint at update={state.get('update')} "
        f"(gate_threshold={args.gate_threshold})",
        flush=True,
    )

    eval_streams = [
        SessionStream(args.shard_dir, seed=args.eval_seed + lane_index)
        for lane_index in range(args.num_lanes)
    ]
    eval_pool = PrefetchingSessionPool(
        eval_streams, window_tokens=args.window_tokens, pad_token_id=pad_token_id, num_workers=2
    )
    lanes = [
        new_lane(eval_pool.handle(lane_index), model_config.d_model)
        for lane_index in range(args.num_lanes)
    ]

    scripted_states = [_new_condition_state(model_config.d_model) for _ in lanes]
    gate_states = [_new_condition_state(model_config.d_model) for _ in lanes]
    prev_session_id: list[int | None] = [None] * len(lanes)

    total_windows = 0
    with torch.no_grad():
        for _ in range(args.eval_windows):
            windows: list[Window] = [lane.advance_window() for lane in lanes]
            total_windows += 1
            for lane_index, lane in enumerate(lanes):
                current_id = id(lane.session)
                if current_id != prev_session_id[lane_index]:
                    scripted_states[lane_index] = _new_condition_state(model_config.d_model)
                    gate_states[lane_index] = _new_condition_state(model_config.d_model)
                    prev_session_id[lane_index] = current_id

            batch = torch.tensor([w.token_ids for w in windows], device=device, dtype=torch.long)
            input_ids = batch[:, :-1]
            hidden, _ = model.forward_hidden(input_ids)
            hidden_float = hidden.float()

            # ---- write phase: one shared pool/key/value/gate-score per
            # valid span, then split into the two tracks' own bookkeeping.
            for lane_index, window in enumerate(windows):
                s_state = scripted_states[lane_index]
                g_state = gate_states[lane_index]
                for write in window.writes:
                    valid = [
                        (local_i, span)
                        for local_i, span in enumerate(write.statement_spans)
                        if span[1] - span[0] > 0 and span[1] <= args.window_tokens - 1
                    ]
                    s_index_map: list[int | None] = [None] * len(write.statement_spans)
                    g_index_map: list[int | None] = [None] * len(write.statement_spans)
                    if valid:
                        pooled = torch.cat(
                            [_pool(hidden_float[lane_index], span, args.window_tokens) for _, span in valid],
                            dim=0,
                        )
                        keys = key_projection(pooled)
                        values = value_projection(pooled)
                        gate_scores = torch.sigmoid(salience_gate(pooled))
                        for row, (local_i, _span) in enumerate(valid):
                            s_state.candidate_spans += 1
                            g_state.candidate_spans += 1
                            key = keys[row]
                            value = values[row]

                            s_state.bank.write_always_new(key.detach(), "statement", Source("user"))
                            s_state.statement_keys.append(key)
                            s_state.statement_values.append(value)
                            s_index_map[local_i] = len(s_state.statement_keys) - 1
                            s_state.spans_written += 1

                            if gate_scores[row].item() > args.gate_threshold:
                                g_state.bank.write_always_new(key.detach(), "statement", Source("user"))
                                g_state.statement_keys.append(key)
                                g_state.statement_values.append(value)
                                g_index_map[local_i] = len(g_state.statement_keys) - 1
                                g_state.spans_written += 1
                    s_state.unit_index_map[write.unit_id] = s_index_map
                    g_state.unit_index_map[write.unit_id] = g_index_map

            # ---- query phase: same due queries feed both tracks.
            for lane_index, window in enumerate(windows):
                if not window.queries:
                    continue
                for query in window.queries:
                    q_pooled = _pool(hidden_float[lane_index], query.query_span, args.window_tokens)
                    q_key = key_projection(q_pooled)[0]
                    for cond_state in (scripted_states[lane_index], gate_states[lane_index]):
                        cond_state.retrieval_total += 1
                        index_map = cond_state.unit_index_map.get(query.unit_id)
                        target_bank_index = None
                        if index_map is not None and query.target_statement_index < len(index_map):
                            target_bank_index = index_map[query.target_statement_index]
                        if target_bank_index is None:
                            cond_state.target_missing += 1
                            continue
                        raw = cond_state.bank.read_raw(q_key.detach(), top_k=1)
                        best_hit_id = None
                        if raw is not None:
                            ids, _scores = raw
                            if ids.numel() > 0:
                                best_hit_id = int(ids[0].item())
                        if best_hit_id is not None and best_hit_id == target_bank_index:
                            cond_state.retrieval_correct += 1

    def summarize(label: str, states: list[ConditionState]) -> None:
        candidate = sum(s.candidate_spans for s in states)
        written = sum(s.spans_written for s in states)
        retrieval_total = sum(s.retrieval_total for s in states)
        retrieval_correct = sum(s.retrieval_correct for s in states)
        target_missing = sum(s.target_missing for s in states)
        write_rate = written / candidate if candidate else float("nan")
        retrieval_acc = retrieval_correct / retrieval_total if retrieval_total else float("nan")
        print(
            f"[{label}] candidate_spans={candidate} written={written} "
            f"write_rate={write_rate:.4f} queries={retrieval_total} "
            f"retrieval_correct={retrieval_correct} retrieval_acc={retrieval_acc:.4f} "
            f"target_missing={target_missing}",
            flush=True,
        )

    print(f"total windows processed: {total_windows}", flush=True)
    summarize("scripted", scripted_states)
    summarize("gate", gate_states)


if __name__ == "__main__":
    main()
