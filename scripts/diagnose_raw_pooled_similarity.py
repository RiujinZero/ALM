#!/usr/bin/env python
"""Diagnostic: does the 'deploy_day' attractor pattern round_trip_gate_
retrieval_test.py keeps finding already exist in Sonic's own raw
mean-pooled hidden states, before key_projection touches them at all?
If yes, the ceiling is Sonic's frozen representation itself (no linear
key_projection can fix that, however much contrastive data trains it -
a linear map can only rotate/scale existing geometry, not introduce
discriminative power that isn't already latent in it). If the raw
vectors are NOT dominated by one attractor, the ceiling is more likely
key_projection's own limited capacity (a single linear layer) or the
training signal itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from manifoldflow.experiments.manifoldflow_production_train import _pool
from manifoldflow.sonic_ssm import SonicSSMConfig, SonicSSMLanguageModel

_CASES = [
    ("file_format", "I choose .png as the supported extension for the image export feature."),
    ("deploy_day", "Let's plan to deploy the new pricing page next Thursday afternoon."),
    ("db_choice", "I've decided to use PostgreSQL instead of MongoDB for the new service, since we need strong transactional guarantees."),
    ("color_pref", "My favorite color for the dashboard theme is a deep teal, not the default blue."),
    ("budget_cap", "Let's cap the monthly cloud spend at $4,000 for this project."),
    ("meeting_time", "The standup is moving from 9am to 10:30am starting next week."),
    ("library_choice", "I'll use Zod for schema validation instead of Yup in this project."),
    ("vacation_dates", "I'm taking vacation from the 12th through the 19th of next month."),
    ("api_version", "We're pinning the payments API to v3, not upgrading to v4 yet."),
    ("team_lead", "Priya is taking over as lead for the onboarding redesign project."),
]

_QUERIES = [
    ("file_format", "What file extension did we decide to support for image export?"),
    ("deploy_day", "When are we deploying the new pricing page?"),
    ("db_choice", "Which database did we choose for the new service?"),
    ("color_pref", "What color did I say I wanted for the dashboard theme?"),
    ("budget_cap", "What's the monthly cloud spend cap we agreed on?"),
    ("meeting_time", "What time is the standup moving to?"),
    ("library_choice", "Which validation library are we using?"),
    ("vacation_dates", "When is my vacation?"),
    ("api_version", "Which version of the payments API are we using?"),
    ("team_lead", "Who's leading the onboarding redesign project?"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
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

    def encode_raw(text: str) -> torch.Tensor:
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        ids_tensor = torch.tensor([ids], device=device, dtype=torch.long)
        with torch.no_grad():
            hidden, _ = model.forward_hidden(ids_tensor)
        hidden_float = hidden[0].float()
        pooled = _pool(hidden_float, (0, min(len(ids), args.window_tokens - 1)), args.window_tokens)
        return F.normalize(pooled, dim=-1)[0]

    anchor_vecs = {name: encode_raw(text) for name, text in _CASES}
    query_vecs = {name: encode_raw(text) for name, text in _QUERIES}

    names = [name for name, _ in _CASES]
    print("Raw pooled-hidden-state cosine similarity: each query vs every anchor\n", flush=True)
    header = "query".ljust(16) + "".join(n[:10].ljust(11) for n in names)
    print(header, flush=True)
    correct = 0
    attractor_counts = {name: 0 for name in names}
    for q_name in names:
        q = query_vecs[q_name]
        sims = {a_name: float((q @ anchor_vecs[a_name]).item()) for a_name in names}
        best = max(sims, key=sims.get)
        attractor_counts[best] += 1
        if best == q_name:
            correct += 1
        row = q_name.ljust(16) + "".join(f"{sims[a]:.3f}".ljust(11) for a in names)
        marker = "  <- best: " + best + (" OK" if best == q_name else " MISS")
        print(row + marker, flush=True)

    print(f"\nraw-vector nearest-neighbor accuracy (no key_projection at all): {correct}/{len(names)} = {correct/len(names):.3f}", flush=True)
    print("\nhow often each anchor was the 'nearest' pick across all 10 queries (an even spread ~1 each is healthy; one name dominating means it's a generic attractor):", flush=True)
    for name, count in sorted(attractor_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {count}", flush=True)


if __name__ == "__main__":
    main()
