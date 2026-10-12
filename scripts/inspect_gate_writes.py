#!/usr/bin/env python
"""Qualitative spot-check for --gate-writes: decodes the actual text of
every chat/filler candidate turn a checkpoint's salience_gate would
write, alongside its score - answers "is this sensible content or noise
that happened to clear threshold," which eval_retrieval_accuracy alone
can't tell you (that number only reflects distractor content; nothing in
the corpus queries a gate-written chat/filler fact, so its only quality
signal is direct inspection).

Read-only: loads the checkpoint via torch.load and never writes anything
back - safe to run alongside a live training/eval process using the same
checkpoint path, including a checkpoint that process is still updating
(each save is atomic .tmp+rename, so a concurrent read gets a complete,
consistent file, not a partial write).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from manifoldflow.experiments.manifoldflow_production_train import (
    PrefetchingSessionPool,
    SessionStream,
    _fully_contained,
    _pool,
    _tokenizer_identity,
    load_resume_state,
    new_lane,
)
from manifoldflow.salience_gate import SalienceGateConfig, SalienceGateHead
from manifoldflow.sonic_ssm import SonicSSMConfig, SonicSSMLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, default=Path("/path/to/manifoldflow_sessions"))
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--num-lanes", type=int, default=3)
    parser.add_argument("--eval-windows", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=31)
    parser.add_argument("--window-tokens", type=int, default=8192)
    parser.add_argument("--gate-write-threshold", type=float, default=0.5)
    parser.add_argument("--scan-backend", default="doubling")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Also print candidates that scored BELOW threshold (to eyeball false negatives), "
        "not just the ones that would actually get written.",
    )
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

    salience_gate = SalienceGateHead(SalienceGateConfig(d_model=model_config.d_model)).to(device)
    state = load_resume_state(
        args.checkpoint_path, model_config=model_config, tokenizer_identity=tokenizer_identity
    )
    if state.get("salience_gate") is None:
        raise SystemExit("checkpoint has no trained salience_gate")
    salience_gate.load_state_dict(state["salience_gate"])
    salience_gate.eval()
    print(f"loaded checkpoint at update={state.get('update')}", flush=True)

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

    written = 0
    skipped = 0
    with torch.no_grad():
        for _ in range(args.eval_windows):
            windows = [lane.advance_window() for lane in lanes]
            batch = torch.tensor([w.token_ids for w in windows], device=device, dtype=torch.long)
            input_ids = batch[:, :-1]
            hidden, _ = model.forward_hidden(input_ids)
            hidden_float = hidden.float()

            for lane_index, window in enumerate(windows):
                for span in window.gate_candidates:
                    if span[1] - span[0] <= 0 or span[1] > args.window_tokens - 1:
                        continue
                    pooled = _pool(hidden_float[lane_index], span, args.window_tokens)
                    score = torch.sigmoid(salience_gate(pooled)).item()
                    would_write = score > args.gate_write_threshold
                    if not would_write and not args.show_all:
                        skipped += 1
                        continue
                    a, b = span
                    text = tokenizer.decode(
                        windows[lane_index].token_ids[a:b], skip_special_tokens=True
                    )
                    text = text.strip().replace("\n", " \\n ")
                    if len(text) > 300:
                        text = text[:300] + "..."
                    tag = "WRITE" if would_write else "skip "
                    print(f"[{tag} score={score:.3f}] {text}", flush=True)
                    if would_write:
                        written += 1
                    else:
                        skipped += 1

    print(f"\ntotal: {written} written, {skipped} skipped (threshold={args.gate_write_threshold})")


if __name__ == "__main__":
    main()
