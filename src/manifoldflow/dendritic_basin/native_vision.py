"""ctypes binding for the native growing-vision organism (``cpp/vision.hpp``).

Unlike ``native_cortex.py``, this is not a bit-exact port of its Python
reference (``ALM64MB4CGrowingVision`` in ``continual_vision.py``): PyTorch's
internal reduction order and ``topk`` tie-breaking under equal values are not
a portable target to reproduce bit-for-bit from a from-scratch
implementation.

The contract is narrower than "numerically close and functionally
identical decisions" might suggest, and it is worth being precise about
why (see ``tests/test_native_vision_matches_python.py`` and
``cpp/README.md`` for the full trace): growth and routing both select
their top-k candidates (128 synapses out of ~2,880 dimensions; 16
assemblies by activation) via discontinuous top-k operations. A ~1e-7
float32 rounding difference between this port and PyTorch's reduction
order can flip which candidate wins a near-tie, which changes what gets
*stored* -- not just one step's output -- so it compounds for the rest of
a run. Confirmed directly: evidence() matches the Python reference to the
float32 noise floor (~1e-7/1e-8) on every step, yet per-step grow/update
decisions can still diverge later in a stream, in either direction (not
a one-sided bug). So: evidence() is tight and deterministic; aggregate
quality (mean accuracy over many streams, growth staying in the same
ballpark) tracks the reference within a wide band. Per-step lockstep
matching is not claimed.

Ships as a separate DLL (``alm_vision``) from the byte-cortex's
``alm_native``, deliberately: the byte-cortex DLL stays loaded inside
long-running processes and Windows locks a DLL file while any process has it
loaded, so building both into one library would fail to link (or force
killing a live run) every time either module changes.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import POINTER, c_double, c_int32, c_int64, c_size_t, c_uint8
from pathlib import Path

import numpy as np

class _AlmVisionError(RuntimeError):
    pass


_STAT_STEPS = 0
_STAT_LIVE_ASSEMBLIES = 1
_STAT_SLEEP_CYCLES = 2
_STAT_VISUAL_BIRTHS = 3
_STAT_RELATION_TURNOVER_RELEASES = 4
_STAT_MEMORY_BYTES = 5


class _AlmVisionConfig(ctypes.Structure):
    _fields_ = [
        ("classes", c_int32),
        ("image_size", c_int32),
        ("columns", c_int32),
        ("context_decay", c_double),
        ("sleep_interval", c_int32),
        ("max_body_bytes", c_int64),
        ("learning_rate", c_double),
        ("maximum_synapses", c_int32),
        ("maximum_visual_assemblies", c_int32),
        ("visual_birth_activation", c_double),
        ("visual_routing_width", c_int32),
        ("relation_sketch_words", c_int32),
        ("relation_turnover_fraction", c_double),
        ("episodic_enabled", c_int32),
        ("episodic_lsh_bits", c_int32),
        ("episodic_probe_radius", c_int32),
        ("episodic_chain_slots", c_int32),
        ("episodic_tier3_refresh_threshold", c_int32),
        ("episodic_strength", c_double),
    ]


def _library_candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[3] / "cpp" / "build"
    names = ["alm_vision.dll"] if sys.platform == "win32" else ["libalm_vision.so"]
    candidates: list[Path] = []
    for name in names:
        candidates.append(root / "bin" / "Release" / name)
        candidates.append(root / "bin" / name)
        candidates.append(root / "Release" / name)
        candidates.append(root / name)
    return candidates


_LIBRARY = None


def _load() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    for candidate in _library_candidates():
        if candidate.exists():
            library = ctypes.CDLL(str(candidate))
            _bind(library)
            _LIBRARY = library
            return library
    raise FileNotFoundError(
        "alm_vision shared library not found; build it with:\n"
        "  cmake -S cpp -B cpp/build -G \"Visual Studio 17 2022\" -A x64\n"
        "  cmake --build cpp/build --config Release --target alm_vision"
    )


def _bind(library: ctypes.CDLL) -> None:
    library.alm_vision_create.argtypes = [POINTER(_AlmVisionConfig)]
    library.alm_vision_create.restype = ctypes.c_void_p
    library.alm_vision_destroy.argtypes = [ctypes.c_void_p]
    library.alm_vision_destroy.restype = None
    library.alm_vision_enable_episodic_layer.argtypes = [ctypes.c_void_p]
    library.alm_vision_enable_episodic_layer.restype = None
    library.alm_vision_dimensions.argtypes = [ctypes.c_void_p]
    library.alm_vision_dimensions.restype = c_int32
    library.alm_vision_classes.argtypes = [ctypes.c_void_p]
    library.alm_vision_classes.restype = c_int32
    library.alm_vision_evidence.argtypes = [
        ctypes.c_void_p,
        POINTER(ctypes.c_float),
        POINTER(ctypes.c_float),
    ]
    library.alm_vision_evidence.restype = None
    library.alm_vision_predict.argtypes = [
        ctypes.c_void_p,
        POINTER(ctypes.c_float),
        POINTER(ctypes.c_float),
    ]
    library.alm_vision_predict.restype = c_int32
    library.alm_vision_predict_single_column.argtypes = [
        ctypes.c_void_p,
        POINTER(ctypes.c_float),
        POINTER(ctypes.c_float),
    ]
    library.alm_vision_predict_single_column.restype = c_int32
    library.alm_vision_observe_external.argtypes = [
        ctypes.c_void_p,
        POINTER(ctypes.c_float),
        c_int32,
    ]
    library.alm_vision_observe_external.restype = None
    library.alm_vision_reset_episode.argtypes = [ctypes.c_void_p]
    library.alm_vision_reset_episode.restype = None
    library.alm_vision_sleep.argtypes = [ctypes.c_void_p]
    library.alm_vision_sleep.restype = None
    library.alm_vision_stat.argtypes = [ctypes.c_void_p, c_int32]
    library.alm_vision_stat.restype = c_int64
    library.alm_vision_save.argtypes = [ctypes.c_void_p, POINTER(c_size_t)]
    library.alm_vision_save.restype = POINTER(c_uint8)
    library.alm_vision_free_buffer.argtypes = [POINTER(c_uint8)]
    library.alm_vision_free_buffer.restype = None
    library.alm_vision_load.argtypes = [POINTER(c_uint8), c_size_t, POINTER(c_int32)]
    library.alm_vision_load.restype = ctypes.c_void_p


def native_vision_available() -> bool:
    try:
        _load()
    except (FileNotFoundError, OSError):
        return False
    return True


class NativeGrowingVision:
    """Native port of ``ALM64MB4CGrowingVision``. See module docstring for
    the (numerically-close, not bit-exact) correctness contract."""

    def __init__(
        self,
        *,
        classes: int,
        image_size: int = 24,
        columns: int = 4,
        context_decay: float = 0.94,
        sleep_interval: int = 4096,
        max_body_bytes: int = 64_000_000,
        learning_rate: float = 0.5,
        maximum_synapses: int = 128,
        maximum_visual_assemblies: int = 50_000,
        visual_birth_activation: float = 0.72,
        visual_routing_width: int = 16,
        relation_sketch_words: int = 4,
        relation_turnover_fraction: float = 0.0,
        episodic_enabled: bool = False,
        episodic_lsh_bits: int = 16,
        episodic_probe_radius: int = 2,
        episodic_chain_slots: int = 4,
        episodic_tier3_refresh_threshold: int = 3,
        episodic_strength: float = 0.1,
    ) -> None:
        self._library = _load()
        config = _AlmVisionConfig(
            classes=classes,
            image_size=image_size,
            columns=columns,
            context_decay=context_decay,
            sleep_interval=sleep_interval,
            max_body_bytes=max_body_bytes,
            learning_rate=learning_rate,
            maximum_synapses=maximum_synapses,
            maximum_visual_assemblies=maximum_visual_assemblies,
            visual_birth_activation=visual_birth_activation,
            visual_routing_width=visual_routing_width,
            relation_sketch_words=relation_sketch_words,
            relation_turnover_fraction=relation_turnover_fraction,
            episodic_enabled=1 if episodic_enabled else 0,
            episodic_lsh_bits=episodic_lsh_bits,
            episodic_probe_radius=episodic_probe_radius,
            episodic_chain_slots=episodic_chain_slots,
            episodic_tier3_refresh_threshold=episodic_tier3_refresh_threshold,
            episodic_strength=episodic_strength,
        )
        handle = self._library.alm_vision_create(ctypes.byref(config))
        if not handle:
            raise MemoryError("could not allocate a native vision organism")
        self._handle = handle
        self._classes = classes
        self._dimensions = int(
            self._library.alm_vision_dimensions(ctypes.c_void_p(handle))
        )

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._library.alm_vision_destroy(ctypes.c_void_p(handle))
            self._handle = None

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def evidence(self, rgb: np.ndarray) -> np.ndarray:
        """``rgb``: [image_size, image_size, 3] float32 in [0, 1] (already
        decoded and resized -- see the module docstring)."""
        rgb = np.ascontiguousarray(rgb, dtype=np.float32)
        out = np.empty(self._dimensions, dtype=np.float32)
        self._library.alm_vision_evidence(
            ctypes.c_void_p(self._handle),
            rgb.ctypes.data_as(POINTER(ctypes.c_float)),
            out.ctypes.data_as(POINTER(ctypes.c_float)),
        )
        return out

    def predict(self, evidence: np.ndarray) -> tuple[int, np.ndarray]:
        evidence = np.ascontiguousarray(evidence, dtype=np.float32)
        votes = np.empty(self._classes, dtype=np.float32)
        prediction = self._library.alm_vision_predict(
            ctypes.c_void_p(self._handle),
            evidence.ctypes.data_as(POINTER(ctypes.c_float)),
            votes.ctypes.data_as(POINTER(ctypes.c_float)),
        )
        return int(prediction), votes

    def predict_single_column(self, evidence: np.ndarray) -> tuple[int, np.ndarray]:
        """Diagnostic-only: votes from just the currently-awake column
        instead of ensembling all columns. See
        GrowingVisionOrganism::predict_evidence_single_column in vision.hpp."""
        evidence = np.ascontiguousarray(evidence, dtype=np.float32)
        votes = np.empty(self._classes, dtype=np.float32)
        prediction = self._library.alm_vision_predict_single_column(
            ctypes.c_void_p(self._handle),
            evidence.ctypes.data_as(POINTER(ctypes.c_float)),
            votes.ctypes.data_as(POINTER(ctypes.c_float)),
        )
        return int(prediction), votes

    def observe_external(self, evidence: np.ndarray, label: int) -> None:
        evidence = np.ascontiguousarray(evidence, dtype=np.float32)
        self._library.alm_vision_observe_external(
            ctypes.c_void_p(self._handle),
            evidence.ctypes.data_as(POINTER(ctypes.c_float)),
            c_int32(label),
        )

    def reset_episode(self) -> None:
        self._library.alm_vision_reset_episode(ctypes.c_void_p(self._handle))

    def enable_episodic_layer(self) -> None:
        """Turns episodic recall on for an already-constructed organism
        (e.g. resumed from a checkpoint saved before this was enabled).
        No-op if already enabled."""
        self._library.alm_vision_enable_episodic_layer(ctypes.c_void_p(self._handle))

    def sleep(self) -> None:
        self._library.alm_vision_sleep(ctypes.c_void_p(self._handle))

    def _stat(self, stat: int) -> int:
        return int(self._library.alm_vision_stat(ctypes.c_void_p(self._handle), stat))

    @property
    def steps(self) -> int:
        return self._stat(_STAT_STEPS)

    @property
    def live_assemblies(self) -> int:
        return self._stat(_STAT_LIVE_ASSEMBLIES)

    @property
    def sleep_cycles(self) -> int:
        return self._stat(_STAT_SLEEP_CYCLES)

    @property
    def visual_births(self) -> int:
        return self._stat(_STAT_VISUAL_BIRTHS)

    @property
    def relation_turnover_releases(self) -> int:
        return self._stat(_STAT_RELATION_TURNOVER_RELEASES)

    @property
    def memory_bytes(self) -> int:
        return self._stat(_STAT_MEMORY_BYTES)

    def to_bytes(self) -> bytes:
        """Native binary checkpoint (see cpp/include/alm/vision_serialize.hpp).

        Same-machine, same-build only -- a fast replacement for pickling
        the Python object graph, not a portable exchange format. Mirrors
        native_cortex.py's NativeGrowableCortex.to_bytes/from_bytes.
        """
        size = ctypes.c_size_t()
        buffer = self._library.alm_vision_save(
            ctypes.c_void_p(self._handle), ctypes.byref(size)
        )
        if not buffer:
            raise _AlmVisionError("failed to serialize native vision organism")
        try:
            return ctypes.string_at(buffer, size.value)
        finally:
            self._library.alm_vision_free_buffer(buffer)

    @classmethod
    def from_bytes(cls, data: bytes) -> NativeGrowingVision:
        library = _load()
        payload = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        error = c_int32()
        handle = library.alm_vision_load(
            payload, ctypes.c_size_t(len(data)), ctypes.byref(error)
        )
        if not handle or error.value:
            raise _AlmVisionError("failed to load native vision checkpoint")
        instance = cls.__new__(cls)
        instance._library = library
        instance._handle = handle
        instance._classes = int(library.alm_vision_classes(ctypes.c_void_p(handle)))
        instance._dimensions = int(
            library.alm_vision_dimensions(ctypes.c_void_p(handle))
        )
        return instance

    def __reduce__(self) -> tuple:
        return (NativeGrowingVision.from_bytes, (self.to_bytes(),))
