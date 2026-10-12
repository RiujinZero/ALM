"""Vanilla Sonic SSM pretraining continuation, using Muon for Sonic's own
2D hidden weight matrices instead of AdamW - a direct, controlled
head-to-head comparison against the existing pure-AdamW checkpoint
lineage at `C:\\path\\to\\gdrive\\Sonic\\prod\\` (`latest_eval_30B.pt` ->
`latest_eval_40B.pt`, 13,000 updates / ~10.22B tokens apart, confirmed
directly from both checkpoints' own saved `recipe`/`tokens_seen` fields:
30B has recipe_update=38875/tokens_seen=30,572,544,000, 40B has
recipe_update=51875/tokens_seen=40,796,160,000 - exactly 13,000 updates
and 10,223,616,000 tokens, matching 13,000 * 8192 * 96 tokens/update
precisely).

No ManifoldFlow anywhere in this script - no port/key_projection/
salience_gate/memory bank, no session/window packing. This is Sonic in
complete isolation, trained by `gator.smollm3_recipe.SmolLM3BatchProvider`
exactly as `gator/sonic_ssm_train.py` already does (that file's own
`run()` is this script's direct model), with only the optimizer changed.

Warm start (`--resume-checkpoint latest_eval_30B.pt`, format
"sonic-ssm-smollm3-26b-v2"): loads the model weights, the data
provider's exact stream position, and the update/tokens_seen/micro_step
counters - continuing the identical corpus position the AdamW lineage
was at, not restarting the stream - but does NOT load its single-AdamW
optimizer state, since Muon's own state has no equivalent to inherit and
starting the new split fresh is the only sensible option anyway. Any
later resume of this script's OWN checkpoints (format
"sonic-ssm-vanilla-muon-v1") is a normal full resume: both optimizer
states restored, RNG state restored, no warm-start special-casing.

Constant learning rates throughout (`--adamw-peak-lr`, default 2e-4;
`--muon-peak-lr`, default 0.02, matching this project's own established
Muon-for-Sonic default from `manifoldflow_production_train.py`) - not
`gator.smollm3_recipe.learning_rate_for_update`'s multi-stage warmup/
decay schedule. Justified directly, not assumed: `grep -oE 'lr=[0-9.e+-]+'
train.log | sort -u` against the real AdamW lineage's own file log
returns exactly one distinct value, `lr=2.000e-04`, across its entire
recorded history - the baseline run's LR was already flat/constant over
this whole region (and past it), so reproducing that schedule exactly
would mean reconstructing an arbitrary large `--steps`/stage-boundary
value with no way to verify it lands in the same flat region; a literal
constant is the more faithful reproduction of that confirmed behavior,
not a simplification of it.

Parameter split for the two optimizers, mirroring
`manifoldflow_production_train.py`'s own `split_sonic_params_for_muon`
(reused directly, unmodified) plus `gator.smollm3_recipe.
adamw_parameter_groups`'s decay/no-decay convention (only
`token_embedding.weight` gets weight_decay=0.0, everything else gets
`--adamw-weight-decay`, default 0.1, matching the baseline exactly):
Muon takes every 2D hidden weight matrix except `token_embedding.weight`
and `lexical_projection.weight` (an embedding table and the layer
feeding straight off it - Muon is only defined for genuine hidden-layer
matrices); AdamW takes those two plus every 1D parameter (norms, biases,
the sparse gate's scalar temperature).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TOKENIZER_DIR = _PROJECT_ROOT / "resources" / "tokenizers" / "gtr_arw_english_50k"
sys.path.insert(0, str(_PROJECT_ROOT / "vendor" / "GTR-sonic-ssm"))

from gator.compute import CompileConfig, mark_compiled_iteration_begin  # noqa: E402
from gator.smollm3_recipe import STAGES, STAGES_DISTRACTOR_REASONING, SmolLM3BatchProvider  # noqa: E402

from manifoldflow.experiments.manifoldflow_production_train import split_sonic_params_for_muon  # noqa: E402
from manifoldflow.muon import Muon  # noqa: E402
from manifoldflow.sonic_ssm import SonicSSMConfig, SonicSSMLanguageModel  # noqa: E402

CHECKPOINT_FORMAT = "sonic-ssm-vanilla-muon-v1"
VANILLA_ADAMW_FORMAT = "sonic-ssm-smollm3-26b-v2"

# ----------------------------------------------------------------------------
# Small helpers mirrored directly from gator/smollm3_train.py and
# gator/sonic_ssm_train.py (not imported - importing gator.smollm3_train
# transitively pulls in gator/models/__init__.py's full model zoo, which
# imports gator_linear.core, which imports gator_convergent - a package
# that was never vendored into this repo. These functions are each small
# and self-contained; copied here rather than fighting that broken
# transitive dependency chain for code this project doesn't otherwise
# need. Kept byte-for-byte identical to the source so a future diff
# against the real files stays meaningful.
# ----------------------------------------------------------------------------


def _tensor_is_finite(tensor: torch.Tensor) -> bool:
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return True
    return bool(torch.isfinite(tensor.detach()).all().item())


def _first_nonfinite_optimizer_state(optimizer: torch.optim.Optimizer) -> str | None:
    for parameter_index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and not _tensor_is_finite(value):
                return f"parameter[{parameter_index}].{name}"
    return None


def _assert_training_state_finite(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, *, context: str
) -> None:
    for name, parameter in model.named_parameters():
        if not _tensor_is_finite(parameter):
            raise FloatingPointError(f"non-finite model parameter {name!r} detected {context}")
    optimizer_state = _first_nonfinite_optimizer_state(optimizer)
    if optimizer_state is not None:
        raise FloatingPointError(f"non-finite optimizer state {optimizer_state!r} detected {context}")


def _require_finite_loss(loss: torch.Tensor, *, update: int, micro_step: int, sources: tuple[str, ...]) -> None:
    if _tensor_is_finite(loss):
        return
    raise FloatingPointError(
        "non-finite loss before backward; "
        f"next_update={update + 1} micro_step={micro_step} sources={sources!r}; "
        "optimizer step and checkpoint were skipped"
    )


def _clip_gradients_or_raise(model: torch.nn.Module, *, update: int, micro_step: int, sources: tuple[str, ...]) -> float:
    try:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    except RuntimeError as error:
        raise FloatingPointError(
            "non-finite gradient before optimizer step; "
            f"next_update={update + 1} micro_step={micro_step} sources={sources!r}; "
            "optimizer step and checkpoint were skipped"
        ) from error
    return float(total_norm)


def _gradient_failure_diagnostic(model: torch.nn.Module, *, detail_limit: int = 8) -> str:
    nonfinite_parameter_count = 0
    details: list[str] = []
    largest_finite_abs = -1.0
    largest_finite_name = "none"
    gradient_parameter_count = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.detach()
        if values.is_sparse:
            values = values.coalesce().values()
        if not (values.is_floating_point() or values.is_complex()):
            continue
        gradient_parameter_count += 1
        finite_mask = torch.isfinite(values)
        all_finite = bool(finite_mask.all().item())
        if all_finite:
            finite_abs_max = float(values.abs().max().item()) if values.numel() else 0.0
        else:
            finite_values = values[finite_mask]
            finite_abs_max = float(finite_values.abs().max().item()) if finite_values.numel() else 0.0
            nonfinite_parameter_count += 1
            if len(details) < detail_limit:
                nan_count = int(torch.isnan(values).sum().item())
                inf_count = int(torch.isinf(values).sum().item())
                nonfinite_count = values.numel() - int(finite_mask.sum().item())
                details.append(
                    f"name={name!r} shape={tuple(values.shape)!r} dtype={str(values.dtype)!r} "
                    f"nonfinite={nonfinite_count} nan={nan_count} inf={inf_count} "
                    f"finite_abs_max={finite_abs_max:.6g}"
                )
        if finite_abs_max > largest_finite_abs:
            largest_finite_abs = finite_abs_max
            largest_finite_name = name
    largest = max(largest_finite_abs, 0.0)
    if nonfinite_parameter_count:
        omitted = max(nonfinite_parameter_count - len(details), 0)
        return (
            f"gradient_parameters={gradient_parameter_count} nonfinite_parameters={nonfinite_parameter_count} "
            f"largest_finite_name={largest_finite_name!r} largest_finite_abs={largest:.6g} "
            f"details={details!r} omitted={omitted}"
        )
    return (
        f"gradient_parameters={gradient_parameter_count} all_gradient_elements_finite=true "
        f"largest_finite_name={largest_finite_name!r} largest_finite_abs={largest:.6g}; "
        "aggregate_norm_overflow_suspected"
    )


def _discard_nonfinite_accumulation(
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    accumulation_steps: int,
    consecutive_failures: int,
    max_consecutive_recoveries: int,
    error: FloatingPointError,
) -> tuple[int, int]:
    optimizer.zero_grad(set_to_none=True)
    next_failures = consecutive_failures + 1
    if next_failures > max_consecutive_recoveries:
        raise FloatingPointError(
            f"{error}; recovery budget exhausted after {max_consecutive_recoveries} consecutive discarded windows"
        ) from error
    return update * accumulation_steps, next_failures


def _autoregressive_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.target_ids.shape[1] != 1:
        raise ValueError("autoregressive batches require one trailing token")
    tokens = torch.cat((batch.context_ids, batch.target_ids), dim=1)
    return tokens[:, :-1], tokens[:, 1:]


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    import hashlib

    serialized = tokenizer.backend_tokenizer.to_str().encode("utf-8")
    return {
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def _append_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + os.linesep)
        handle.flush()
        os.fsync(handle.fileno())


FIXED_EVAL_PROMPTS = (
    "The capital of France is",
    "In machine learning, a neural network is",
    "def add(a, b):\n    return",
    "The three primary colors are",
    "Once upon a time, there was a",
    "Water boils at a temperature of",
    "The main causes of climate change include",
    "To bake a simple loaf of bread, you need",
    "The speed of light in a vacuum is approximately",
    "The best way to learn a new programming language is",
    "David's father has three sons: Snap, Crackle, and what is the third son's name?",
    "Kevin has 4 cats. Maria has 8 parrots. How many parrots does Kevin have?",
    "Sarah had 3 books. She gave 1 book to Tom. How many books does Sarah have now?",
    "Kevin has 4 cats. What color are Kevin's cats?",
    "User: My dog's name is Max.\nAssistant: That's a great name!\nUser: What is my dog's name?\nAssistant:",
)


def _run_sonic_fixed_prompt_eval(
    model: SonicSSMLanguageModel,
    tokenizer: Any,
    *,
    device: torch.device,
    context_tokens: int,
    response_tokens: int,
    mixed_precision: str,
    update: int,
    eval_log_path: Path,
    compile_config: CompileConfig,
) -> None:
    was_training = model.training
    model.eval()
    lines = [f"=== fixed_prompt_eval update={update} ==="]
    try:
        with torch.no_grad():
            for prompt in FIXED_EVAL_PROMPTS:
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                prompt_ids = prompt_ids[-context_tokens:]
                pad_len = context_tokens - len(prompt_ids)
                padded = [tokenizer.eos_token_id] * pad_len + prompt_ids
                input_ids = torch.tensor([padded], dtype=torch.long, device=device)
                mark_compiled_iteration_begin(compile_config, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=mixed_precision == "bf16" and device.type in {"cuda", "cpu"},
                ):
                    logits, states = model(input_ids)
                    next_id = logits[:, -1, :].argmax(-1, keepdim=True)
                    generated = [int(next_id.item())]
                    for _ in range(response_tokens - 1):
                        logits, states = model(next_id, states)
                        next_id = logits[:, -1, :].argmax(-1, keepdim=True)
                        generated.append(int(next_id.item()))
                completion = tokenizer.decode(generated, skip_special_tokens=True)
                lines.append(f"prompt={prompt!r} completion={completion!r}")
    finally:
        if was_training:
            model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    text = "\n".join(lines) + "\n"
    print(text, end="", flush=True)
    with eval_log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _split_muon_adamw_decay(
    model: SonicSSMLanguageModel,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """(muon_params, adamw_decay_params, adamw_no_decay_params) - see
    module docstring's "Parameter split" paragraph."""
    muon_params, adamw_all = split_sonic_params_for_muon(model)
    muon_ids = {id(p) for p in muon_params}
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if id(parameter) in muon_ids:
            continue
        (no_decay if "token_embedding" in name else decay).append(parameter)
    assert len(decay) + len(no_decay) == len(adamw_all)
    return muon_params, decay, no_decay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Absolute recipe_update target, same convention as sonic_ssm_train.py's --steps "
        "(e.g. 51875 to match the 40B checkpoint's own recipe_update from a 30B warm start).",
    )
    parser.add_argument("--hf-tokenizer", type=Path, default=_DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--adamw-peak-lr", type=float, default=2e-4)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-peak-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=3)
    parser.add_argument("--effective-batch-sequences", type=int, default=96)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--response-tokens", type=int, default=128)
    parser.add_argument("--distractor-reasoning-stages", action="store_true")
    parser.add_argument("--distractor-reasoning-chat-data", action="append", default=[])
    parser.add_argument("--force-stage", type=int, default=None)
    parser.add_argument("--tokenized-data-root", type=Path, default=None)
    parser.add_argument("--stream-cache-dir", type=Path, default=None)
    parser.add_argument("--token-chunk-tokens", type=int, default=262144)
    parser.add_argument("--prefetch-tokens", type=int, default=196608)
    parser.add_argument("--prefetch-workers", type=int, default=16)
    parser.add_argument("--max-document-chars", type=int, default=100_000)
    parser.add_argument("--max-open-streams", type=int, default=10)
    parser.add_argument("--max-host-ram-gb", type=float, default=0.0)
    parser.add_argument("--memory-log-every-seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--scan-backend", default="hgrn")
    parser.add_argument("--packed-projections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ce-chunk-rows", type=int, default=1024)
    parser.add_argument("--max-consecutive-nonfinite-windows", type=int, default=5)
    parser.add_argument("--checkpoint-every-updates", type=int, default=1000)
    parser.add_argument("--fixed-eval-every-updates", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--log-every-seconds", type=float, default=300.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.batch_size * args.gradient_accumulation_steps != args.effective_batch_sequences:
        parser.error(
            "--batch-size * --gradient-accumulation-steps must equal "
            "--effective-batch-sequences"
        )
    if args.scan_backend == "hgrn" and args.device != "cuda":
        parser.error("--scan-backend hgrn requires --device cuda")

    device = torch.device(args.device)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_tokenizer), use_fast=True)
    if len(tokenizer) != 50_000:
        parser.error(f"expected the 50,000-token tokenizer, got {len(tokenizer)}")
    eval_tokenizer = AutoTokenizer.from_pretrained(str(args.hf_tokenizer), use_fast=True)
    distractor_reasoning_tokenizer = AutoTokenizer.from_pretrained(str(args.hf_tokenizer), use_fast=True)
    identity = _tokenizer_identity(tokenizer)

    state = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_format = state.get("format")
    if checkpoint_format not in (VANILLA_ADAMW_FORMAT, CHECKPOINT_FORMAT):
        parser.error(
            f"resume checkpoint format {checkpoint_format!r} is neither the vanilla-AdamW "
            f"format {VANILLA_ADAMW_FORMAT!r} (warm start) nor this script's own "
            f"{CHECKPOINT_FORMAT!r} (resume) - refusing to guess"
        )
    is_warm_start = checkpoint_format == VANILLA_ADAMW_FORMAT
    if state.get("tokenizer") != identity:
        parser.error("resume checkpoint's tokenizer identity does not match - refusing to resume into a mismatch")

    model_config = SonicSSMConfig(**state["model_config"])
    model = SonicSSMLanguageModel(model_config).to(device)
    model.load_state_dict(state["model"])
    model.configure_execution(scan_backend=args.scan_backend, packed_projections=args.packed_projections)
    if args.compile_layers:
        model.compile_layers(mode=args.compile_mode)
    compile_config = CompileConfig(enabled=args.compile_layers, mode=args.compile_mode)

    muon_params, adamw_decay, adamw_no_decay = _split_muon_adamw_decay(model)
    adamw_optimizer = torch.optim.AdamW(
        [
            {"params": adamw_decay, "weight_decay": args.adamw_weight_decay},
            {"params": adamw_no_decay, "weight_decay": 0.0},
        ],
        lr=args.adamw_peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=args.fused_adamw and device.type == "cuda",
    )
    muon_optimizer = Muon(
        muon_params,
        lr=args.muon_peak_lr,
        momentum=args.muon_momentum,
        nesterov=True,
        ns_steps=args.muon_ns_steps,
        weight_decay=args.muon_weight_decay,
    )
    print(
        f"sonic_vanilla_muon_optimizer_split muon_params={len(muon_params)} "
        f"adamw_decay_params={len(adamw_decay)} adamw_no_decay_params={len(adamw_no_decay)} "
        f"adamw_peak_lr={args.adamw_peak_lr} muon_peak_lr={args.muon_peak_lr}",
        flush=True,
    )

    provider_sources = STAGES_DISTRACTOR_REASONING if args.distractor_reasoning_stages else STAGES
    provider = SmolLM3BatchProvider(
        tokenizer,
        context_tokens=args.sequence_length - 1,
        output_slots=1,
        seed=args.seed,
        sources=provider_sources,
        distractor_reasoning_chat_data=args.distractor_reasoning_chat_data,
        distractor_reasoning_tokenizer=distractor_reasoning_tokenizer,
        force_stage=args.force_stage,
        tokenized_data_root=args.tokenized_data_root,
        stream_cache_dir=args.stream_cache_dir,
        chunk_tokens=args.token_chunk_tokens,
        prefetch_tokens=args.prefetch_tokens,
        prefetch_workers=args.prefetch_workers,
        max_document_chars=args.max_document_chars,
        max_open_streams=args.max_open_streams,
        max_host_ram_gb=args.max_host_ram_gb,
        memory_log_every_seconds=args.memory_log_every_seconds,
        effective_batch_sequences=args.effective_batch_sequences,
    )
    print(
        f"sonic_vanilla_muon_data_mode distractor_reasoning_stages={args.distractor_reasoning_stages} "
        f"force_stage={args.force_stage} objective=causal_next_token_shifted_all_positions",
        flush=True,
    )

    out = args.checkpoint_dir
    out.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or out
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    eval_log_path = log_dir / "eval_prompts.log"

    update = 0
    micro_step = 0
    recipe_update = 0
    tokens_seen = 0
    nonfinite_windows = 0
    best_loss = float("inf")

    if is_warm_start:
        if state.get("data") is not None:
            provider.load_state_dict(state["data"], allow_window_resize=False)
        update = int(state["update"])
        micro_step = int(state["micro_step"])
        tokens_seen = int(state["tokens_seen"])
        recipe_update = tokens_seen // (args.sequence_length * args.effective_batch_sequences)
        nonfinite_windows = int(state.get("nonfinite_windows", 0))
        best_loss = float(state.get("best_loss", float("inf")))
        print(
            f"warm_start_from={args.resume_checkpoint} update={update} recipe_update={recipe_update} "
            f"tokens_seen={tokens_seen} (fresh Muon+AdamW optimizer state, no momentum carried over)",
            flush=True,
        )
    else:
        adamw_optimizer.load_state_dict(state["adamw_optimizer"])
        muon_optimizer.load_state_dict(state["muon_optimizer"])
        provider.load_state_dict(state["data"], allow_window_resize=False)
        update = int(state["update"])
        micro_step = int(state["micro_step"])
        recipe_update = int(state["recipe_update"])
        tokens_seen = int(state["tokens_seen"])
        nonfinite_windows = int(state.get("nonfinite_windows", 0))
        best_loss = float(state.get("best_loss", float("inf")))
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        _assert_training_state_finite(model, adamw_optimizer, context="while loading checkpoint (adamw)")
        _assert_training_state_finite(model, muon_optimizer, context="while loading checkpoint (muon)")
        print(
            f"resumed from {args.resume_checkpoint} update={update} recipe_update={recipe_update} "
            f"tokens_seen={tokens_seen}",
            flush=True,
        )

    if recipe_update > args.steps:
        parser.error("--steps precedes the checkpoint's own recipe_update")
    final_update = update + (args.steps - recipe_update)
    starting_update = update
    accumulation_steps = args.gradient_accumulation_steps
    accumulation_loss = 0.0
    accumulation_sources: list[str] = []
    consecutive_nonfinite_windows = 0
    interval_loss = 0.0
    interval_updates = 0
    file_loss = 0.0
    file_updates = 0
    started = time.perf_counter()
    last_file_log = started
    adamw_optimizer.zero_grad(set_to_none=True)
    muon_optimizer.zero_grad(set_to_none=True)

    def save(path: Path) -> None:
        _assert_training_state_finite(model, adamw_optimizer, context=f"before checkpoint update={update} (adamw)")
        _assert_training_state_finite(model, muon_optimizer, context=f"before checkpoint update={update} (muon)")
        provider.set_progress(update=recipe_update, micro_step=micro_step)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "update": update,
            "micro_step": micro_step,
            "recipe_update": recipe_update,
            "tokens_seen": tokens_seen,
            "nonfinite_windows": nonfinite_windows,
            "best_loss": best_loss,
            "model": model.state_dict(),
            "model_config": model_config.to_dict(),
            "adamw_optimizer": adamw_optimizer.state_dict(),
            "muon_optimizer": muon_optimizer.state_dict(),
            "data": provider.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "tokenizer": identity,
            "recipe": {
                "sequence_length": args.sequence_length,
                "physical_batch_size": args.batch_size,
                "accumulation": accumulation_steps,
                "effective_batch_sequences": args.effective_batch_sequences,
                "adamw_peak_lr": args.adamw_peak_lr,
                "muon_peak_lr": args.muon_peak_lr,
                "objective": "causal_next_token_shifted_all_positions",
                "warm_started_from": str(args.resume_checkpoint) if is_warm_start else state.get("recipe", {}).get(
                    "warm_started_from"
                ),
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    while update < final_update:
        provider.set_progress(update=recipe_update, micro_step=micro_step)
        batch = provider.sample_cpu(args.batch_size).to(device)
        accumulation_sources.extend(provider.last_source_names)
        input_ids, labels = _autoregressive_batch(batch)
        mark_compiled_iteration_begin(compile_config, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in {"cuda", "cpu"}):
            hidden, _ = model.forward_hidden(input_ids)
            loss = model.chunked_cross_entropy(hidden, labels, chunk_rows=args.ce_chunk_rows)
        try:
            _require_finite_loss(loss, update=update, micro_step=micro_step + 1, sources=provider.last_source_names)
        except FloatingPointError as error:
            micro_step, consecutive_nonfinite_windows = _discard_nonfinite_accumulation(
                adamw_optimizer,
                update=update,
                accumulation_steps=accumulation_steps,
                consecutive_failures=consecutive_nonfinite_windows,
                max_consecutive_recoveries=args.max_consecutive_nonfinite_windows,
                error=error,
            )
            muon_optimizer.zero_grad(set_to_none=True)
            nonfinite_windows += 1
            accumulation_loss = 0.0
            accumulation_sources.clear()
            continue

        (loss / accumulation_steps).backward()
        accumulation_loss += float(loss.detach())
        micro_step += 1
        if micro_step % accumulation_steps:
            continue

        next_update = update + 1
        next_recipe_update = recipe_update + 1
        source_window = tuple(dict.fromkeys(accumulation_sources))
        try:
            grad = _clip_gradients_or_raise(model, update=update, micro_step=micro_step, sources=source_window)
        except FloatingPointError as error:
            diagnostic = _gradient_failure_diagnostic(model)
            micro_step, consecutive_nonfinite_windows = _discard_nonfinite_accumulation(
                adamw_optimizer,
                update=update,
                accumulation_steps=accumulation_steps,
                consecutive_failures=consecutive_nonfinite_windows,
                max_consecutive_recoveries=args.max_consecutive_nonfinite_windows,
                error=error,
            )
            muon_optimizer.zero_grad(set_to_none=True)
            nonfinite_windows += 1
            accumulation_loss = 0.0
            accumulation_sources.clear()
            event = f"[sonic_vanilla_muon:train] event=nonfinite_window_discarded diagnostic=({diagnostic})"
            print(event, flush=True)
            _append_log(log_path, event)
            continue

        adamw_optimizer.step()
        muon_optimizer.step()
        adamw_optimizer.zero_grad(set_to_none=True)
        muon_optimizer.zero_grad(set_to_none=True)
        mean_update_loss = accumulation_loss / accumulation_steps
        interval_loss += mean_update_loss
        file_loss += mean_update_loss
        interval_updates += 1
        file_updates += 1
        accumulation_loss = 0.0
        accumulation_sources.clear()
        consecutive_nonfinite_windows = 0
        update = next_update
        recipe_update = next_recipe_update
        tokens_seen += args.sequence_length * args.effective_batch_sequences
        now = time.perf_counter()
        elapsed = now - started
        updates_completed = update - starting_update
        tokens_per_second = (
            updates_completed * args.sequence_length * args.effective_batch_sequences / max(elapsed, 1e-9)
        )

        if update % args.log_every == 0 or update == final_update:
            mean_loss = interval_loss / max(interval_updates, 1)
            print(
                f"update={update}/{final_update} recipe_update={recipe_update}/{args.steps} "
                f"stage={provider._stage_for_update(recipe_update) + 1} "
                f"loss={mean_loss:.4f} ppl={math.exp(min(mean_loss, 20.0)):.3f} "
                f"grad={grad:.3f} updates_per_second={updates_completed / max(elapsed, 1e-9):.3f} "
                f"tokens_per_second={tokens_per_second:.1f} tokens={tokens_seen} "
                f"nonfinite_windows={nonfinite_windows}",
                flush=True,
            )
            interval_loss = 0.0
            interval_updates = 0

        if update % args.checkpoint_every_updates == 0 or update == final_update:
            if mean_loss < best_loss:
                best_loss = mean_loss
                save(out / "best.pt")
            save(out / "latest.pt")

        if update % args.fixed_eval_every_updates == 0 or update == final_update:
            _run_sonic_fixed_prompt_eval(
                model,
                eval_tokenizer,
                device=device,
                context_tokens=args.sequence_length - args.response_tokens,
                response_tokens=args.response_tokens,
                mixed_precision="bf16",
                update=update,
                eval_log_path=eval_log_path,
                compile_config=compile_config,
            )

        if (args.log_every_seconds > 0 and now - last_file_log >= args.log_every_seconds) or update == final_update:
            mean_file_loss = file_loss / max(file_updates, 1)
            _append_log(
                log_path,
                f"timestamp={datetime.now(timezone.utc).isoformat()} "
                f"update={update}/{final_update} recipe_update={recipe_update}/{args.steps} "
                f"stage={provider._stage_for_update(recipe_update) + 1} "
                f"loss={mean_file_loss:.4f} ppl={math.exp(min(mean_file_loss, 20.0)):.3f} "
                f"adamw_lr={args.adamw_peak_lr:.3e} muon_lr={args.muon_peak_lr:.3e} grad={grad:.3f} "
                f"tokens_per_second={tokens_per_second:.1f} tokens={tokens_seen} "
                f"nonfinite_windows={nonfinite_windows}",
            )
            last_file_log = now
            file_loss = 0.0
            file_updates = 0

        if args.max_runtime_seconds > 0 and elapsed >= args.max_runtime_seconds and update < final_update:
            save(out / "latest.pt")
            print(f"sonic_vanilla_muon_runtime_limit update={update} seconds={elapsed:.1f}", flush=True)
            break

    provider.close()


if __name__ == "__main__":
    main()
