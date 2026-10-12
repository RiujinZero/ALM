#!/usr/bin/env python
"""Trains `SonicKeyProjection` on real contrastive pairs
(`scripts/generate_contrastive_pairs.py`'s output) - the training signal
`round_trip_gate_retrieval_test.py` found missing: the main production
loop's own `info_nce_retrieval_loss` only ever sees templated distractor
statements against their own oracle query, so nothing ever taught
`key_projection` paraphrase-invariance (same fact, different wording ->
close) or hard-negative discrimination (similar wording, different fact
-> far). This is the fix aimed directly at that gap, not a bigger version
of the same training signal.

Loss, per batch of `batch_size` (anchor, paraphrase, hard_negative)
triples: treat each triple's `paraphrase` as the query, `anchor` as the
correct target, and everything else in the batch - every OTHER triple's
`anchor` (a real but different fact - a moderate negative) plus every
triple's own `hard_negative` (a deliberately hard, superficially-similar
negative) - as the wrong answers, via one softmax/cross-entropy per
query exactly like `info_nce_retrieval_loss`'s single-episode version,
just with the candidate set built from a full batch of real contrastive
material instead of one episode's own writes.

Sonic itself stays frozen by default (`torch.no_grad()` around
`model.forward_hidden`) - only `key_projection`'s own (small
Linear->ReLU->Linear, see sonic_key_projection.py) parameters get
gradient, matching this whole project's established pattern for training
a new, less-tested objective riding alongside (here: after) a
carefully-tuned main run, not touching Sonic's own weights.

`--unfreeze-sonic` opts into training Sonic's own weights too, split the
same way the production trainer already does
(`split_sonic_params_for_muon`: Muon for Sonic's 2D hidden weight
matrices, a separate small AdamW for everything else Sonic owns) - the
one candidate fix from docs/Salient_Gate_Dataset_Runbook.md's "Fourth
attempt" section not yet tried in isolation. Production's own Sonic has
been unfrozen (on the *scripted*-distractor retrieval signal) since
update 21801 without moving the round-trip natural-language number at
all - that's a different training distribution, so it doesn't test this
hypothesis. This flag lets Sonic's own representation adapt directly
against the paraphrase/hard-negative contrastive objective instead,
which is the actual untested case.

Saves updated checkpoints in the exact format
`round_trip_gate_retrieval_test.py`/the production trainer already load
(`model`, `model_config`, `key_projection`, `format`, `tokenizer` keys) -
a NEW file, never overwriting the real production checkpoint - so the
existing round-trip test can validate this directly with zero changes of
its own.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from manifoldflow.experiments.manifoldflow_production_train import (
    CHECKPOINT_FORMAT,
    _tokenizer_identity,
    load_resume_state,
    split_sonic_params_for_muon,
)
from manifoldflow.muon import Muon
from manifoldflow.sonic_key_projection import (
    SonicKeyProjection,
    SonicKeyProjectionConfig,
    load_key_projection_state_dict,
)
from manifoldflow.sonic_ssm import SonicSSMConfig, SonicSSMLanguageModel


def _load_triples(path: Path) -> list[tuple[str, str, str]]:
    triples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("parse_error"):
                continue
            triples.append((record["anchor"], record["paraphrase"], record["hard_negative"]))
    return triples


def _encode_batch(
    model: SonicSSMLanguageModel,
    tokenizer,
    key_projection: SonicKeyProjection,
    texts: list[str],
    device: str,
    *,
    requires_grad: bool,
    sonic_requires_grad: bool = False,
) -> torch.Tensor:
    """One key per text, batched by right-padding to the batch's own max
    length (not window_tokens - these are short, single-sentence texts,
    no need for the production trainer's fixed 8192-token canvas here;
    nothing here is under torch.compile, so a per-batch shape is fine).

    `sonic_requires_grad` (only ever True with `--unfreeze-sonic`) runs
    `model.forward_hidden` under grad instead of `torch.no_grad()`, so the
    contrastive loss's gradient reaches Sonic's own params too, not just
    key_projection's."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded = [tokenizer(t, add_special_tokens=True)["input_ids"] for t in texts]
    max_len = max(len(ids) for ids in encoded)
    padded = [ids + [pad_id] * (max_len - len(ids)) for ids in encoded]
    lengths = [len(ids) for ids in encoded]
    ids_tensor = torch.tensor(padded, device=device, dtype=torch.long)
    sonic_context = torch.enable_grad() if (requires_grad and sonic_requires_grad) else torch.no_grad()
    with sonic_context:
        hidden, _ = model.forward_hidden(ids_tensor)
    hidden = hidden.float()
    # Mean-pool each row's own real (non-pad) tokens only.
    pooled_rows = []
    for row, length in enumerate(lengths):
        pooled_rows.append(hidden[row, :length].mean(dim=0))
    pooled = torch.stack(pooled_rows)
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        keys = key_projection(pooled)
    return keys


def _contrastive_loss(
    anchor_keys: torch.Tensor, paraphrase_keys: torch.Tensor, hard_neg_keys: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    """Query = paraphrase, correct target = its own anchor, negatives =
    every other anchor in the batch + every hard_negative in the batch -
    see this module's docstring."""
    candidates = torch.cat([anchor_keys, hard_neg_keys], dim=0)  # [2*batch, dim]
    logits = paraphrase_keys @ candidates.T / temperature  # [batch, 2*batch]
    targets = torch.arange(anchor_keys.shape[0], device=anchor_keys.device)
    return F.cross_entropy(logits, targets)


def _accuracy(
    anchor_keys: torch.Tensor, paraphrase_keys: torch.Tensor, hard_neg_keys: torch.Tensor
) -> float:
    candidates = torch.cat([anchor_keys, hard_neg_keys], dim=0)
    logits = paraphrase_keys @ candidates.T
    targets = torch.arange(anchor_keys.shape[0], device=anchor_keys.device)
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--contrastive-pairs-path", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--scan-backend", default="doubling")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0, help="0 = only save at the end.")
    parser.add_argument(
        "--unfreeze-sonic",
        action="store_true",
        help="Also train Sonic's own weights against the contrastive loss "
        "(Muon for its 2D hidden matrices, a small AdamW for the rest) - "
        "see module docstring.",
    )
    parser.add_argument("--sonic-learning-rate", type=float, default=1e-5)
    parser.add_argument("--sonic-muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    args = parser.parse_args()

    import transformers

    device = args.device
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    model_config = SonicSSMConfig(**checkpoint["model_config"])
    model = SonicSSMLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    if args.unfreeze_sonic:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model.configure_execution(scan_backend=args.scan_backend)

    tokenizer_dir = args.tokenizer_dir or (
        Path(__file__).resolve().parents[1] / "resources" / "tokenizers" / "gtr_arw_english_50k"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_dir))
    tokenizer_identity = _tokenizer_identity(tokenizer)

    key_projection = SonicKeyProjection(SonicKeyProjectionConfig(dim=model_config.d_model)).to(device)
    state = load_resume_state(
        args.checkpoint_path, model_config=model_config, tokenizer_identity=tokenizer_identity
    )
    load_key_projection_state_dict(key_projection, state["key_projection"])
    print(f"loaded checkpoint at update={state.get('update')}", flush=True)

    triples = _load_triples(args.contrastive_pairs_path)
    random.Random(args.seed).shuffle(triples)
    val_size = max(1, int(len(triples) * args.val_fraction))
    val_triples = triples[:val_size]
    train_triples = triples[val_size:]
    print(f"{len(train_triples)} train / {len(val_triples)} val contrastive triples", flush=True)

    optimizer = torch.optim.AdamW(key_projection.parameters(), lr=args.learning_rate)

    sonic_muon_optimizer = None
    sonic_adamw_optimizer = None
    if args.unfreeze_sonic:
        sonic_muon_params, sonic_adamw_params = split_sonic_params_for_muon(model)
        sonic_adamw_optimizer = torch.optim.AdamW(
            sonic_adamw_params, lr=args.sonic_learning_rate, weight_decay=0.0
        )
        sonic_muon_optimizer = Muon(
            sonic_muon_params,
            lr=args.sonic_muon_lr,
            momentum=args.muon_momentum,
            nesterov=True,
            ns_steps=args.muon_ns_steps,
            weight_decay=args.muon_weight_decay,
        )
        print(
            f"--unfreeze-sonic: {len(sonic_muon_params)} Muon params, "
            f"{len(sonic_adamw_params)} Sonic-AdamW params, "
            f"sonic_lr={args.sonic_learning_rate} sonic_muon_lr={args.sonic_muon_lr}",
            flush=True,
        )

    def run_epoch(triple_list: list[tuple[str, str, str]], *, train: bool) -> tuple[float, float]:
        key_projection.train(train)
        model.train(train and args.unfreeze_sonic)
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        for start in range(0, len(triple_list), args.batch_size):
            batch = triple_list[start : start + args.batch_size]
            if len(batch) < 2:
                continue  # in-batch negatives need at least 2 examples
            anchors = [t[0] for t in batch]
            paraphrases = [t[1] for t in batch]
            hard_negs = [t[2] for t in batch]

            encode_kwargs = dict(requires_grad=train, sonic_requires_grad=args.unfreeze_sonic)
            anchor_keys = _encode_batch(model, tokenizer, key_projection, anchors, device, **encode_kwargs)
            paraphrase_keys = _encode_batch(model, tokenizer, key_projection, paraphrases, device, **encode_kwargs)
            hard_neg_keys = _encode_batch(model, tokenizer, key_projection, hard_negs, device, **encode_kwargs)

            loss = _contrastive_loss(anchor_keys, paraphrase_keys, hard_neg_keys, temperature=args.temperature)
            acc = _accuracy(anchor_keys.detach(), paraphrase_keys.detach(), hard_neg_keys.detach())

            if train:
                optimizer.zero_grad()
                if args.unfreeze_sonic:
                    sonic_adamw_optimizer.zero_grad()
                    sonic_muon_optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if args.unfreeze_sonic:
                    sonic_adamw_optimizer.step()
                    sonic_muon_optimizer.step()

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
        return total_loss / max(num_batches, 1), total_acc / max(num_batches, 1)

    def save_checkpoint(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "model_config": model_config.to_dict(),
                "tokenizer": tokenizer_identity,
                "model": model.state_dict(),
                "key_projection": key_projection.state_dict(),
                "port": state["port"],
                "value_projection": state["value_projection"],
                "salience_gate": state.get("salience_gate"),
                "update": state.get("update"),
            },
            tmp_path,
        )
        tmp_path.replace(path)
        print(f"checkpoint saved -> {path}", flush=True)

    val_loss, val_acc = run_epoch(val_triples, train=False)
    print(f"[epoch 0/before training] val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(train_triples, train=True)
        val_loss, val_acc = run_epoch(val_triples, train=False)
        print(
            f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )
        if args.checkpoint_every_steps and epoch % args.checkpoint_every_steps == 0:
            save_checkpoint(args.output_checkpoint)

    save_checkpoint(args.output_checkpoint)


if __name__ == "__main__":
    main()
