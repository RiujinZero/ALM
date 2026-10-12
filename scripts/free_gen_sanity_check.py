#!/usr/bin/env python
"""Standalone free-generation sanity check against an arbitrary checkpoint -
reuses production's own `run_free_generation_eval`/`FREE_GEN_PROMPTS`
directly so the output is directly comparable to what's already in
`/path/to/manifoldflow_production/train.log`, not a new eval built from
scratch.

Built after the "unfreeze Sonic during contrastive training" experiment
(see docs/Salient_Gate_Dataset_Runbook.md's "Fifth attempt") produced a
real round-trip retrieval improvement (0.300 -> 0.400) that turned out to
have completely destroyed Sonic's basic generative ability - a
catastrophic-forgetting result the retrieval metric alone never would
have surfaced. Run this on any checkpoint that touched Sonic's own
weights *before* trusting a retrieval or accuracy number from it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append("src")
sys.path.append("vendor/GTR-sonic-ssm")

import torch

from manifoldflow.experiments.manifoldflow_production_train import (
    _tokenizer_identity,
    load_resume_state,
    log_free_generation_eval,
    run_free_generation_eval,
)
from manifoldflow.sonic_key_projection import (
    SonicKeyProjection,
    SonicKeyProjectionConfig,
    load_key_projection_state_dict,
)
from manifoldflow.sonic_memory_port import SonicMemoryPort, SonicMemoryPortConfig
from manifoldflow.sonic_ssm import SonicSSMConfig, SonicSSMLanguageModel


class _PrintLogger:
    def __call__(self, msg: str) -> None:
        print(msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scan-backend", default="doubling")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--port-rank", type=int, default=64)
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
    tokenizer_identity = _tokenizer_identity(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    state = load_resume_state(
        args.checkpoint_path, model_config=model_config, tokenizer_identity=tokenizer_identity
    )
    print(f"loaded checkpoint at update={state.get('update')}", flush=True)

    port = SonicMemoryPort(
        SonicMemoryPortConfig(d_model=model_config.d_model, memory_dim=model_config.d_model, rank=args.port_rank)
    ).to(device)
    port.load_state_dict(state["port"])
    key_projection = SonicKeyProjection(SonicKeyProjectionConfig(dim=model_config.d_model)).to(device)
    load_key_projection_state_dict(key_projection, state["key_projection"])
    value_projection = SonicKeyProjection(SonicKeyProjectionConfig(dim=model_config.d_model)).to(device)
    load_key_projection_state_dict(value_projection, state["value_projection"])

    results = run_free_generation_eval(
        model=model,
        device=device,
        port=port,
        key_projection=key_projection,
        value_projection=value_projection,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        mixed_precision="bf16",
        max_new_tokens=args.max_new_tokens,
    )
    log_free_generation_eval(_PrintLogger(), update=state.get("update") or 0, results=results)


if __name__ == "__main__":
    main()
