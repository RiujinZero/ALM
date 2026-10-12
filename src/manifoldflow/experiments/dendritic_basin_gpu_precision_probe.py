"""Probe CUDA precision and resident cortical throughput on local hardware.

This is an execution-feasibility experiment, not a second learner. It keeps a
branch-by-byte causal loop resident on the GPU, measures representative
predictor-table reads, and quantizes ALM-like activity/count distributions
through CUDA's native FP64, FP32, BF16, and FP8 formats.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.online_stream import OnlineDendriticConfig

CUDA_SOURCE = r"""
#include <cuda_bf16.h>
#include <cuda_fp8.h>

extern "C" __global__ void pack_bf16(
    const float* source, unsigned short* target, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        reinterpret_cast<__nv_bfloat16*>(target)[i] =
            __float2bfloat16(source[i]);
    }
}

extern "C" __global__ void unpack_bf16(
    const unsigned short* source, float* target, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        target[i] = __bfloat162float(
            reinterpret_cast<const __nv_bfloat16*>(source)[i]
        );
    }
}

extern "C" __global__ void pack_e4m3(
    const float* source, unsigned char* target, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        reinterpret_cast<__nv_fp8_e4m3*>(target)[i] =
            __nv_fp8_e4m3(source[i]);
    }
}

extern "C" __global__ void unpack_e4m3(
    const unsigned char* source, float* target, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        target[i] = float(reinterpret_cast<const __nv_fp8_e4m3*>(source)[i]);
    }
}

extern "C" __global__ void pack_e5m2(
    const float* source, unsigned char* target, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        reinterpret_cast<__nv_fp8_e5m2*>(target)[i] =
            __nv_fp8_e5m2(source[i]);
    }
}

extern "C" __global__ void unpack_e5m2(
    const unsigned char* source, float* target, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        target[i] = float(reinterpret_cast<const __nv_fp8_e5m2*>(source)[i]);
    }
}

extern "C" __global__ void cortical_f64(
    const double* table, const unsigned char* tokens, double* output,
    int branches, int steps
) {
    int branch = blockDim.x * blockIdx.x + threadIdx.x;
    if (branch >= branches) return;
    double trace = 0.0;
    double total = 0.0;
    const double* row = table + branch * 256;
    for (int step = 0; step < steps; ++step) {
        double sensory = double(tokens[step]) * (1.0 / 255.0);
        trace = fma(0.97, trace, 0.03 * sensory);
        total = fma(row[tokens[step]], trace, total);
    }
    output[branch] = total;
}

extern "C" __global__ void cortical_f32(
    const float* table, const unsigned char* tokens, float* output,
    int branches, int steps
) {
    int branch = blockDim.x * blockIdx.x + threadIdx.x;
    if (branch >= branches) return;
    float trace = 0.0f;
    float total = 0.0f;
    const float* row = table + branch * 256;
    for (int step = 0; step < steps; ++step) {
        float sensory = float(tokens[step]) * (1.0f / 255.0f);
        trace = fmaf(0.97f, trace, 0.03f * sensory);
        total = fmaf(row[tokens[step]], trace, total);
    }
    output[branch] = total;
}

#define CORTICAL_PACKED(NAME, TYPE, STORAGE)                              \
extern "C" __global__ void NAME(                                       \
    const STORAGE* table, const unsigned char* tokens, float* output,    \
    int branches, int steps                                              \
) {                                                                      \
    int branch = blockDim.x * blockIdx.x + threadIdx.x;                  \
    if (branch >= branches) return;                                      \
    float trace = 0.0f;                                                  \
    float total = 0.0f;                                                  \
    const TYPE* row = reinterpret_cast<const TYPE*>(table) + branch*256; \
    for (int step = 0; step < steps; ++step) {                           \
        float sensory = float(tokens[step]) * (1.0f / 255.0f);           \
        trace = fmaf(0.97f, trace, 0.03f * sensory);                     \
        total = fmaf(float(row[tokens[step]]), trace, total);            \
    }                                                                    \
    output[branch] = total;                                              \
}

CORTICAL_PACKED(cortical_bf16, __nv_bfloat16, unsigned short)
CORTICAL_PACKED(cortical_e4m3, __nv_fp8_e4m3, unsigned char)
CORTICAL_PACKED(cortical_e5m2, __nv_fp8_e5m2, unsigned char)
"""

FORMATS = {
    "fp64": ("cortical_f64", None, None, 8),
    "fp32": ("cortical_f32", None, None, 4),
    "bf16": ("cortical_bf16", "pack_bf16", "unpack_bf16", 2),
    "fp8_e4m3": ("cortical_e4m3", "pack_e4m3", "unpack_e4m3", 1),
    "fp8_e5m2": ("cortical_e5m2", "pack_e5m2", "unpack_e5m2", 1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-branch-evaluations", type=int, default=200_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _module(cp: Any) -> Any:
    names = tuple(
        name for values in FORMATS.values() for name in values[:3] if name is not None
    )
    return cp.RawModule(
        code=CUDA_SOURCE,
        options=("-std=c++17",),
        name_expressions=names,
    )


def _pack(cp: Any, module: Any, values: Any, format_name: str) -> Any:
    _kernel, pack_name, _unpack_name, itemsize = FORMATS[format_name]
    if format_name == "fp64":
        return values.astype(cp.float64)
    if format_name == "fp32":
        return values.astype(cp.float32)
    storage = cp.empty(values.size * itemsize, dtype=cp.uint8)
    module.get_function(pack_name)(
        ((values.size + 255) // 256,),
        (256,),
        (values.astype(cp.float32, copy=False), storage, values.size),
    )
    return storage


def _roundtrip(
    cp: Any, module: Any, values: np.ndarray, format_name: str
) -> np.ndarray:
    device_values = cp.asarray(values, dtype=cp.float32)
    if format_name in {"fp64", "fp32"}:
        dtype = cp.float64 if format_name == "fp64" else cp.float32
        return device_values.astype(dtype).astype(cp.float32).get()
    storage = _pack(cp, module, device_values, format_name)
    target = cp.empty(values.size, dtype=cp.float32)
    unpack_name = FORMATS[format_name][2]
    module.get_function(unpack_name)(
        ((values.size + 255) // 256,),
        (256,),
        (storage, target, values.size),
    )
    return target.get().reshape(values.shape)


def _error(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def _precision_probe(cp: Any, module: Any, rng: np.random.Generator) -> dict[str, Any]:
    activities = rng.random(256 * 256, dtype=np.float32)
    residuals = np.clip(rng.normal(0.0, 2.0, activities.size), -8.0, 8.0).astype(
        np.float32
    )
    counts = np.clip(rng.exponential(64.0, activities.size), 0.0, 2_000.0).astype(
        np.float32
    )
    probability_rows = counts.reshape(256, 256) + 0.1
    probability_rows /= probability_rows.sum(axis=1, keepdims=True)
    result: dict[str, Any] = {}
    format_maximum = {"fp8_e4m3": 448.0, "fp8_e5m2": 57_344.0}
    for format_name in FORMATS:
        rounded_activity = _roundtrip(cp, module, activities, format_name)
        rounded_residual = _roundtrip(cp, module, residuals, format_name)
        rounded_counts = _roundtrip(cp, module, counts, format_name)
        rounded_probability = _roundtrip(cp, module, probability_rows, format_name)
        rounded_probability = np.maximum(rounded_probability, 1e-12)
        rounded_probability /= rounded_probability.sum(axis=1, keepdims=True)
        kl_bits = np.sum(
            probability_rows
            * np.log2(np.maximum(probability_rows, 1e-12) / rounded_probability),
            axis=1,
        )
        item = {
            "activity": _error(activities, rounded_activity),
            "log_residual": _error(residuals, rounded_residual),
            "raw_count": _error(counts, rounded_counts),
            "probability_kl_bits_mean": float(np.mean(kl_bits)),
            "probability_kl_bits_max": float(np.max(kl_bits)),
        }
        if format_name in format_maximum:
            scale = np.maximum(
                counts.reshape(256, 256).max(axis=1, keepdims=True)
                / format_maximum[format_name],
                1e-12,
            )
            scaled = counts.reshape(256, 256) / scale
            rounded_scaled = _roundtrip(cp, module, scaled, format_name) * scale
            item["row_scaled_count"] = _error(counts.reshape(256, 256), rounded_scaled)
            probability_scale = np.maximum(
                probability_rows.max(axis=1, keepdims=True)
                / format_maximum[format_name],
                1e-12,
            )
            scaled_probability = probability_rows / probability_scale
            rounded_scaled_probability = (
                _roundtrip(cp, module, scaled_probability, format_name)
                * probability_scale
            )
            rounded_scaled_probability = np.maximum(rounded_scaled_probability, 1e-12)
            rounded_scaled_probability /= rounded_scaled_probability.sum(
                axis=1, keepdims=True
            )
            scaled_kl = np.sum(
                probability_rows
                * np.log2(probability_rows / rounded_scaled_probability),
                axis=1,
            )
            item["row_scaled_probability_kl_bits_mean"] = float(np.mean(scaled_kl))
            item["row_scaled_probability_kl_bits_max"] = float(np.max(scaled_kl))
        result[format_name] = item
    return result


def _throughput_probe(
    cp: Any,
    module: Any,
    rng: np.random.Generator,
    target_evaluations: int,
    repeats: int,
) -> list[dict[str, Any]]:
    rows = []
    for byte_class in (2_000_000, 16_000_000, 64_000_000, 256_000_000):
        config = OnlineDendriticConfig(
            vocab_size=256,
            max_alm_bytes=byte_class,
            contextual_residual=True,
        )
        branches = config.max_relations
        steps = max(
            4_096,
            min(1_000_000, math.ceil(target_evaluations / branches)),
        )
        host_table = rng.random((branches, 256), dtype=np.float32)
        tokens = cp.asarray(rng.integers(0, 256, steps, dtype=np.uint8))
        device_float = cp.asarray(host_table)
        for format_name, (
            kernel_name,
            _pack_name,
            _unpack,
            itemsize,
        ) in FORMATS.items():
            table = _pack(cp, module, device_float, format_name)
            output_dtype = cp.float64 if format_name == "fp64" else cp.float32
            output = cp.empty(branches, dtype=output_dtype)
            kernel = module.get_function(kernel_name)
            grid = ((branches + 255) // 256,)
            arguments = (table, tokens, output, branches, steps)
            kernel(grid, (256,), arguments)
            cp.cuda.Stream.null.synchronize()
            elapsed = []
            for _ in range(repeats):
                start = cp.cuda.Event()
                stop = cp.cuda.Event()
                start.record()
                kernel(grid, (256,), arguments)
                stop.record()
                stop.synchronize()
                elapsed.append(cp.cuda.get_elapsed_time(start, stop) / 1000.0)
            seconds = min(elapsed)
            rows.append(
                {
                    "alm_class_bytes": byte_class,
                    "branches": branches,
                    "steps": steps,
                    "format": format_name,
                    "storage_bytes": branches * 256 * itemsize,
                    "seconds": seconds,
                    "branch_evaluations_per_second": branches * steps / seconds,
                    "equivalent_causal_bytes_per_second": steps / seconds,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - hardware environment
        raise SystemExit("install the 'alm-gpu' extra to run this probe") from exc

    module = _module(cp)
    rng = np.random.default_rng(args.seed)
    properties = cp.cuda.runtime.getDeviceProperties(0)
    payload = {
        "device": properties["name"].decode(),
        "compute_capability": cp.cuda.Device(0).compute_capability,
        "cupy": cp.__version__,
        "precision": _precision_probe(cp, module, rng),
        "throughput": _throughput_probe(
            cp,
            module,
            rng,
            args.target_branch_evaluations,
            args.repeats,
        ),
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
