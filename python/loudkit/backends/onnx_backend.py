"""The ONNX backend: the three stages as ONNX graphs, driven from Python.

See ``docs/design/execution-config.md``.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ..checkpoint import TOKENIZER_FILENAME, Checkpoint
from ..config import AlgorithmConfig, ExecutionConfig, ONNXProvider
from ..contracts import MEL_BINS, TOKEN_MEL_RATIO, Mel, Sampler, SpeechTokens, Waveform
from ..engine import Engine
from ..errors import CancelledError
from ..frontend.text import GraphemeTextFrontend
from ..models.noise import gaussian_field, symmetric_uniforms
from ..models.windowing import (
    FLOW_NOISE_STREAM,
    START_TEXT_TOKEN,
    STOP_TEXT_TOKEN,
    UPSAMPLE_PER_FRAME,
    VOCODER_HARMONICS,
    VOCODER_NOISE_STREAM,
    VOCODER_PHASE_STREAM,
    eos_floor,
    frame_windows,
    time_grid,
)
from ..release import EXPORT_RECORD
from ..voice import EMOTION_NEUTRAL, VoiceProfile
from . import register_backend

__all__ = [
    "ONNXTokenGenerator",
    "ONNXMelDecoder",
    "ONNXVocoder",
    "build_onnx_engine",
    "resolve_provider",
]

ASSETS_ENV = "LOUDKIT_ONNX_ASSETS"
COND_GRAPH = "t3_cond.onnx"
PREFILL_GRAPH = "t3_prefill.onnx"
STEP_GRAPH = "t3_step.onnx"
PAIR_GRAPH = "t3_pair_step.onnx"
HEAD2_GRAPH = "t3_head2.onnx"
ENCODER_GRAPH = "flow_encoder.onnx"
ESTIMATOR_GRAPH = "flow_estimator.onnx"
HIFT_GRAPH = "vocoder.onnx"

_F32 = NDArray[np.float32]
_Tokens = NDArray[np.integer[Any]]
"""The renderer seams' two array shapes. Named because the CoreML subclass
overrides each seam and has to repeat the signature."""

_SESSIONS = (
    COND_GRAPH,
    PREFILL_GRAPH,
    STEP_GRAPH,
    ENCODER_GRAPH,
    ESTIMATOR_GRAPH,
    HIFT_GRAPH,
)


def graph_names(config: AlgorithmConfig) -> tuple[str, ...]:
    if config.decode == "single":
        return _SESSIONS
    if config.decode == "fusion_mtp2":
        return (
            COND_GRAPH,
            PREFILL_GRAPH,
            PAIR_GRAPH,
            HEAD2_GRAPH,
            ENCODER_GRAPH,
            ESTIMATOR_GRAPH,
            HIFT_GRAPH,
        )
    raise ValueError(f"unsupported decode mode: {config.decode!r}")


def _require_known_decode(config: AlgorithmConfig) -> None:
    """Refuse a decode mode no graph set covers, before the loader is reached.

    The name is the only place it appears in a message: past this point the
    failure is a missing `.onnx` file, which reads as a broken export rather
    than as a mode this backend does not have graphs for.
    """
    graph_names(config)


EXPORT_PROVENANCE = EXPORT_RECORD
"""What `tools/export_onnx.py` writes beside the graphs it exported."""


def _check_provenance(assets: Path, ckpt: Checkpoint, algorithm: AlgorithmConfig) -> None:
    """The export record beside the graphs must name this checkpoint."""
    from . import check_export_record

    check_export_record(
        assets,
        ckpt,
        algorithm,
        block="graphs",
        members=graph_names(algorithm),
        tool="tools/export_onnx.py",
    )


_ORT_DISABLE_TELEMETRY = "ORT_DISABLE_TELEMETRY"


def _ort_module() -> Any:
    """Import ONNX Runtime with its process telemetry disabled.

    Microsoft's official native builds enable telemetry by default. The
    environment switch is set before importing the binding so even its first
    initialization event and persistent device identifier are suppressed. The
    API call is the second belt for a host that imported ONNX Runtime earlier.
    """
    os.environ[_ORT_DISABLE_TELEMETRY] = "1"
    import onnxruntime as ort

    ort.disable_telemetry_events()
    return ort


PROVIDER_NAMES: Mapping[ONNXProvider, str] = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "directml": "DmlExecutionProvider",
}
"""loudkit's five names -> onnxruntime's. The right-hand column stops here."""

AUTO_ORDER: tuple[ONNXProvider, ...] = ("cuda", "cpu")
"""What ``auto`` will pick, best first. Same order in every port.

See ``docs/design/execution-config.md``.
"""

_INSTALL_HINT: Mapping[ONNXProvider, str] = {
    "cuda": "install the GPU build: pip install onnxruntime-gpu (CUDA 12, cuDNN 9)",
    "coreml": "the CoreML provider ships only in the macOS onnxruntime wheels",
    "directml": "install the Windows build: pip install onnxruntime-directml",
    "cpu": "reinstall onnxruntime; the CPU provider is in every build",
}


def resolve_provider(requested: ONNXProvider) -> ONNXProvider:
    """The provider that will actually run, or an error naming what is missing.

    ``auto`` takes the first of :data:`AUTO_ORDER` this build offers. An
    explicit provider that is absent raises instead of falling back: a run that
    says cuda and measures cpu is worse than a run that fails, because the
    number it produces looks publishable.
    """
    ort = _ort_module()

    available = list(ort.get_available_providers())
    offered = ", ".join(available) or "none"
    if requested == "auto":
        for name in AUTO_ORDER:
            if PROVIDER_NAMES[name] in available:
                return name
        raise RuntimeError(
            f"this onnxruntime build offers no provider loudkit knows: {offered}"
        )
    if PROVIDER_NAMES[requested] not in available:
        raise ValueError(
            f"onnx_provider={requested!r} needs {PROVIDER_NAMES[requested]}, which this "
            f"onnxruntime build does not have; it offers: {offered}. "
            f"To get it, {_INSTALL_HINT[requested]}. An explicit provider never falls "
            "back to cpu: ask for 'auto' if a fallback is what you want."
        )
    return requested


GENERATOR_GRAPHS: frozenset[str] = frozenset({COND_GRAPH, PREFILL_GRAPH, STEP_GRAPH})
"""The three T3 graphs: the autoregressive decode loop and what feeds it."""

RENDERER_GRAPHS: frozenset[str] = frozenset({ENCODER_GRAPH, ESTIMATOR_GRAPH, HIFT_GRAPH})
"""The three renderer graphs: the mel decoder pair and the vocoder."""

COREML_CACHE_ENV = "LOUDKIT_COREML_CACHE"
"""Where CoreML keeps its compiled models. See :func:`_coreml_cache_dir`."""

_COREML_OPTIONS: Mapping[str, str] = {"ModelFormat": "MLProgram"}
"""The one option that decides whether CoreML is worth using at all.

See ``docs/design/execution-config.md``.
"""


def _coreml_cache_dir() -> Path:
    """Where CoreML writes compiled models, and why it must be somewhere.

    See ``docs/design/execution-config.md``.
    """
    override = os.environ.get(COREML_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Caches" / "loudkit" / "coreml"


def _session_providers(
    provider: ONNXProvider, graph: str
) -> list[str | tuple[str, Mapping[str, str]]]:
    """The provider list handed to onnxruntime, per graph.

    See ``docs/design/execution-config.md``.
    """
    cpu = PROVIDER_NAMES["cpu"]
    if provider == "cpu":
        return [cpu]
    if provider == "coreml":
        # An allowlist, not a denylist: a graph this module gains later lands
        # on CPU until somebody measures it there, rather than inheriting an
        # accelerator by default. The enrollment graphs are the case that
        # makes this matter in the ports, the voice encoder decides what a
        # cloned voice sounds like, and CoreML has never been measured on it.
        if graph not in RENDERER_GRAPHS:
            return [cpu]
        options = dict(_COREML_OPTIONS)
        options["ModelCacheDirectory"] = str(_coreml_cache_dir())
        return [(PROVIDER_NAMES["coreml"], options), cpu]
    return [PROVIDER_NAMES[provider], cpu]


class _GraphSession(Protocol):
    """All the stages ask of a loaded graph: positional arrays in, arrays out.

    Two classes satisfy it, one per graph runtime, and each backend overrides
    ``_load_session`` to build its own. The loader is typed as this rather than
    as either class so the CoreML override is a substitution the checker can
    read, which is what it was returning ``Any`` to avoid.
    """

    def run_positional(self, values: Any) -> list[NDArray[Any]]: ...


class _Session:
    """A loaded ONNX graph, with its input/output names captured at load.

    See ``docs/design/execution-config.md``.
    """

    def __init__(
        self,
        path: Path,
        *,
        threads: int | None,
        deterministic: bool,
        providers: Sequence[str | tuple[str, Mapping[str, str]]],
    ) -> None:
        ort = _ort_module()

        so = ort.SessionOptions()
        if threads is not None:
            so.intra_op_num_threads = threads
        if deterministic:
            # ORT's deterministic-compute flag pins reductions the way
            # ``pin_determinism`` pins cuDNN: without it, two runs could pick
            # different parallel kernels and drift.
            so.use_deterministic_compute = True
        self._sess = ort.InferenceSession(str(path), so, providers=list(providers))
        self.in_names = [i.name for i in self._sess.get_inputs()]
        self.out_names = [o.name for o in self._sess.get_outputs()]

    def run(self, feed: Mapping[str, Any]) -> list[NDArray[np.generic]]:
        return cast(list[NDArray[np.generic]], self._sess.run(self.out_names, dict(feed)))

    def run_positional(
        self, values: Sequence[NDArray[np.generic]]
    ) -> list[NDArray[np.generic]]:
        return cast(
            list[NDArray[np.generic]],
            self._sess.run(self.out_names, dict(zip(self.in_names, values, strict=True))),
        )


def _assets_dir(ckpt: Checkpoint, algorithm: AlgorithmConfig | None = None) -> Path:
    required = graph_names(algorithm) if algorithm is not None else _SESSIONS
    env = os.environ.get(ASSETS_ENV)
    candidates = [Path(env)] if env else []
    candidates.append(ckpt.path.parent / "onnx")
    for cand in candidates:
        if all((cand / name).exists() for name in required):
            return cand
    raise FileNotFoundError(
        f"ONNX assets not found. Expected {', '.join(required)} in "
        f"{[str(c) for c in candidates]} (override with ${ASSETS_ENV}). "
        "Run tools/export_onnx.py to create them."
    )


def _load_session(directory: Path, name: str, execution: ExecutionConfig) -> _GraphSession:
    # Resolved per session rather than passed down: a component built directly
    # (the exporter does this) carries an unresolved "auto" and still has to
    # land on the same provider as one built through build_onnx_engine.
    return _Session(
        directory / name,
        threads=execution.num_threads,
        deterministic=execution.deterministic is not False,
        providers=_session_providers(resolve_provider(execution.onnx_provider or "auto"), name),
    )


def _as_f32(a: NDArray[np.generic]) -> NDArray[np.float32]:
    return np.asarray(a, dtype=np.float32)


class ONNXTokenGenerator:
    """``TokenGenerator`` over the exported T3 graphs.

    The sampler stays a loudkit object (counter-based, hardware-agnostic); this
    class owns only the framing, the cache and the loop, the numpy mirror of
    the torch generator's ``generate``.
    """

    def __init__(
        self,
        config: AlgorithmConfig,
        ckpt: Checkpoint,
        assets: Path,
        *,
        execution: ExecutionConfig,
    ) -> None:
        _require_known_decode(config)
        self.config = config
        self._cond = self._load_session(assets, COND_GRAPH, execution)
        self._prefill = self._load_session(assets, PREFILL_GRAPH, execution)
        self._step = self._load_session(
            assets, PAIR_GRAPH if config.decode == "fusion_mtp2" else STEP_GRAPH, execution
        )
        self._head2 = (
            self._load_session(assets, HEAD2_GRAPH, execution)
            if config.decode == "fusion_mtp2"
            else None
        )

        t3 = ckpt.tensors("t3.")
        self._fusion_weights = {k: _as_f32(v) for k, v in t3.items() if k.startswith("fuse.")}
        # fp16 storage upcasts exactly; the exported graphs carry the same
        # fp32 weights, so table and graph cannot drift.
        self._cond_cache: dict[tuple[bytes, bytes], NDArray[np.float32]] = {}
        # Serialises the check-evict-insert on `_cond_cache`. See the torch
        # generator's copy: the dict is only a cache, but the sequence is three
        # operations, and two threads both seeing it full evict two entries to
        # make room for one.
        self._cond_lock = threading.Lock()
        self._text_emb = _as_f32(t3["text_emb.weight"])
        self._speech_emb = _as_f32(t3["speech_emb.weight"])
        self._text_pos = _as_f32(t3["text_pos_emb.emb.weight"])
        self._speech_pos = _as_f32(t3["speech_pos_emb.emb.weight"])

    _load_session = staticmethod(_load_session)

    def _pair_rows(self, tokens: NDArray[np.int64]) -> NDArray[np.float32]:
        e = self._speech_emb[tokens]
        a, b = e[0::2], e[1::2]
        weights = self._fusion_weights
        x = (
            np.concatenate((a, b), axis=-1) @ weights["fuse.0.weight"].T
            + weights["fuse.0.bias"]
        )
        erf = np.asarray(
            [math.erf(float(v)) for v in (x / np.float32(np.sqrt(2))).flat], dtype=np.float32
        ).reshape(x.shape)
        x = np.float32(0.5) * x * (np.float32(1) + erf)
        return np.asarray(
            np.float32(0.5) * (a + b) + x @ weights["fuse.2.weight"].T + weights["fuse.2.bias"],
            dtype=np.float32,
        )

    # -- embedding construction (the numpy mirror of the torch module) -------

    def _cond_row(self, voice: VoiceProfile) -> NDArray[np.float32]:
        """``[1, 34, 1024]`` conditioning: speaker, perceiver prompt, emotion.

        Memoised by content (see :meth:`VoiceProfile.cond_key`): the row is a
        pure function of two profile tensors, and recomputing it per chunk was
        a full ``t3_cond`` session run each time.
        """
        key = voice.cond_key()
        with self._cond_lock:
            cached = self._cond_cache.get(key)
        if cached is not None:
            return cached
        values: list[NDArray[np.generic]] = [
            np.asarray(voice.speaker_embedding, dtype=np.float32)[None],
            np.asarray(voice.cond_prompt_tokens, dtype=np.int64)[None],
            np.asarray([EMOTION_NEUTRAL], dtype=np.float32)[None],
        ]
        # Outside the lock: this is a session run, and holding a lock across
        # one would serialise callers that share nothing but a cache.
        row = _as_f32(self._cond.run_positional(values)[0])
        with self._cond_lock:
            # Second lookup under the lock, see the torch generator's copy.
            # Without it, two threads missing on the same key each evict an
            # entry to make room for a key the other has already inserted.
            existing = self._cond_cache.get(key)
            if existing is not None:
                return existing
            if len(self._cond_cache) >= 8:
                self._cond_cache.pop(next(iter(self._cond_cache)))
            self._cond_cache[key] = row
        return row

    def _text_row(self, text_tokens: NDArray[np.int64]) -> NDArray[np.float32]:
        framed = np.concatenate(([START_TEXT_TOKEN], text_tokens, [STOP_TEXT_TOKEN]))
        row = self._text_emb[framed] + self._text_pos[np.arange(len(framed), dtype=np.int64)]
        return cast(NDArray[np.float32], row)

    def _speech_row(self, token: int, position: int) -> NDArray[np.float32]:
        row = self._speech_emb[token] + self._speech_pos[position]
        return np.asarray(row, dtype=np.float32)[None, None]

    def _prefill_embeds(
        self, text_tokens: NDArray[np.int64], voice: VoiceProfile, prefix: Sequence[int]
    ) -> NDArray[np.float32]:
        cond = self._cond_row(voice)[0]
        text = self._text_row(text_tokens)
        bos = self._speech_row(self.config.start_speech_token, 0)[0]
        rows: list[NDArray[np.float32]] = [cond, text, bos]
        if prefix:
            p = np.asarray(prefix, dtype=np.int64)
            if self.config.decode == "fusion_mtp2":
                p = p[: len(p) - len(p) % 2]
                rows.append(
                    self._pair_rows(p) + self._speech_pos[np.arange(1, len(p) // 2 + 1)]
                )
            else:
                rows.append(self._speech_emb[p] + self._speech_pos[np.arange(1, len(p) + 1)])
        return np.concatenate(rows, axis=0)[None]

    # -- contract ------------------------------------------------------------

    def generate(
        self,
        text_tokens: NDArray[np.int64],
        voice: VoiceProfile,
        *,
        sampler: Sampler,
        max_new_tokens: int | None = None,
        prefix: SpeechTokens = (),
        should_cancel: Callable[[], bool] | None = None,
    ) -> SpeechTokens:
        # `is None`, not `or`: see TorchTokenGenerator.generate.
        cap = self.config.sampling.max_new_tokens if max_new_tokens is None else max_new_tokens
        floor = eos_floor(len(text_tokens), self.config)
        stop = self.config.stop_speech_token

        prefix = [int(t) for t in prefix]
        if self.config.decode == "fusion_mtp2":
            prefix = prefix[: len(prefix) - len(prefix) % 2]
        embeds = self._prefill_embeds(text_tokens, voice, prefix)
        prefill_len = embeds.shape[1]
        positions = np.arange(prefill_len, dtype=np.int64)
        logits_all, *kv = self._prefill.run_positional([embeds, positions])
        hidden = kv.pop(0) if self._head2 is not None else None
        logits = _as_f32(logits_all)[0, -1].copy()

        seen = np.zeros(self.config.speech_vocab_size, dtype=bool)
        if prefix:
            seen[prefix] = True

        if self._head2 is not None:
            assert hidden is not None
            return self._generate_pairs(
                logits,
                hidden,
                kv,
                seen,
                cap,
                floor,
                stop,
                len(prefix),
                prefill_len,
                sampler,
                should_cancel,
            )

        out: list[int] = []
        for step in range(cap):
            # Token-level barge-in, same as the torch path: the partial row is
            # discarded, not returned. See `TorchTokenGenerator.generate`.
            if should_cancel is not None and should_cancel():
                raise CancelledError(f"at decode step {step}: cancelled")
            if len(out) < floor:
                logits[stop] = -np.inf
            token = sampler(logits, step=step, seen=seen)
            out.append(token)
            if token == stop:
                break
            seen[token] = True
            emb = self._speech_row(token, len(prefix) + step + 1)
            pos = np.asarray([prefill_len + step], dtype=np.int64)
            logits_all, *kv = self._step.run_positional([emb, pos, *kv])
            logits = _as_f32(logits_all)[0].copy()
        return out

    def _generate_pairs(
        self,
        logits: NDArray[np.float32],
        hidden: NDArray[np.generic],
        kv: list[NDArray[np.generic]],
        seen: NDArray[np.bool_],
        cap: int,
        floor: int,
        stop: int,
        prefix_len: int,
        prefill_len: int,
        sampler: Sampler,
        should_cancel: Callable[[], bool] | None,
    ) -> SpeechTokens:
        assert self._head2 is not None
        out: list[int] = []
        while len(out) < cap:
            if should_cancel is not None and should_cancel():
                break
            if len(out) < floor:
                logits[stop] = -np.inf
            first = sampler(logits, step=len(out), seen=seen)
            out.append(first)
            if first == stop or len(out) >= cap:
                break
            seen[first] = True
            logits = _as_f32(
                self._head2.run_positional([hidden, np.asarray([first], dtype=np.int64)])[0]
            )[0].copy()
            if len(out) < floor:
                logits[stop] = -np.inf
            second = sampler(logits, step=len(out), seen=seen)
            out.append(second)
            if second == stop:
                break
            seen[second] = True
            pair = len(out) // 2 - 1
            logits_all, hidden, *kv = self._step.run_positional(
                [
                    np.asarray([[first, second]], dtype=np.int64),
                    np.asarray([prefix_len // 2 + pair + 1], dtype=np.int64),
                    np.asarray([prefill_len + pair], dtype=np.int64),
                    *kv,
                ]
            )
            logits = _as_f32(logits_all)[0].copy()
        return out

    def teacher_forced_logits(
        self,
        text_tokens: NDArray[np.int64],
        voice: VoiceProfile,
        forced: SpeechTokens,
    ) -> NDArray[np.float32]:
        """Logits at every speech position with the stream pinned to ``forced``.

        One causal forward, exactly like the torch implementation: position
        ``k`` of the result is the distribution the model held before seeing
        ``forced[k]``.
        """
        if self._head2 is not None:
            embeds = self._prefill_embeds(text_tokens, voice, ())
            logits, hidden, *kv = self._prefill.run_positional(
                [embeds, np.arange(embeds.shape[1], dtype=np.int64)]
            )
            current = _as_f32(logits)[0, -1]
            fused_rows = [current.copy()]
            for index in range(0, len(forced), 2):
                first = int(forced[index])
                second_logits = self._head2.run_positional(
                    [hidden, np.asarray([first], dtype=np.int64)]
                )[0]
                fused_rows.append(_as_f32(second_logits)[0].copy())
                if index + 1 < len(forced):
                    logits, hidden, *kv = self._step.run_positional(
                        [
                            np.asarray([[first, forced[index + 1]]], dtype=np.int64),
                            np.asarray([index // 2 + 1], dtype=np.int64),
                            np.asarray([embeds.shape[1] + index // 2], dtype=np.int64),
                            *kv,
                        ]
                    )
                    fused_rows.append(_as_f32(logits)[0].copy())
            return np.asarray(fused_rows, dtype=np.float32)
        cond = self._cond_row(voice)[0]
        text = self._text_row(text_tokens)
        speech_start = cond.shape[0] + text.shape[0]  # index of the speech START
        bos = self._speech_row(self.config.start_speech_token, 0)[0]
        rows: list[NDArray[np.float32]] = [cond, text, bos]
        if len(forced) > 0:
            f = np.asarray(forced, dtype=np.int64)
            rows.append(self._speech_emb[f] + self._speech_pos[np.arange(1, len(f) + 1)])
        embeds = np.concatenate(rows, axis=0)[None]
        positions = np.arange(embeds.shape[1], dtype=np.int64)
        logits_all, *_ = self._prefill.run_positional([embeds, positions])
        return _as_f32(logits_all)[0, speech_start:].astype(np.float32)


class ONNXMelDecoder:
    """``MelDecoder`` over the exported encoder + estimator graphs.

    The CoreML renderer subclasses this and replaces the four seams below. The
    framing, the affine, the Euler loop and the crop are one implementation, so
    the two backends cannot drift apart.
    """

    _TOKEN_DTYPE: type[np.integer[Any]] = np.int64  # what the graphs declare
    _BUILDER = "build_onnx_engine"  # named in the "affine not attached" refusal

    def __init__(
        self, config: AlgorithmConfig, assets: Path, *, execution: ExecutionConfig
    ) -> None:
        if config.guidance != "single_path":
            # The decode loop below calls the estimator exactly once per step and never
            # forms (1+w)·v_cond − w·v_uncond.
            raise ValueError(
                f"the ONNX backend implements guidance 'single_path' only; this "
                f"algorithm declares {config.guidance!r}. The exported estimator is "
                "the guidance-distilled student, and running it once is not "
                "classifier-free guidance: export a dual-path graph, or load this "
                "checkpoint on the torch backend, which implements both."
            )
        self.config = config
        self._encoder = _load_session(assets, ENCODER_GRAPH, execution)
        self._estimator = _load_session(assets, ESTIMATOR_GRAPH, execution)
        self._spk_weight: NDArray[np.float32] | None = None
        self._spk_bias: NDArray[np.float32] | None = None

    def attach_speaker_affine(
        self, weight: NDArray[np.float32], bias: NDArray[np.float32]
    ) -> None:
        """The 192->80 speaker projection is part of the torch flow module and
        was baked into neither exported graph; the backend hands its weights
        over so the exported path computes the identical ``spks``."""
        self._spk_weight = weight
        self._spk_bias = bias

    def _prompt_length(self, row: NDArray[np.int64]) -> int:
        """How much of the framed row is reference prompt."""
        prompt = self.config.window.static_prompt_tokens
        # Same reading as the cap above, though `WindowConfig` already refuses a
        # non-positive value, so here the two spellings cannot differ.
        return row.shape[1] // 2 if prompt is None else prompt

    def _encode(self, prompt: _Tokens, query: _Tokens) -> _F32:
        return _as_f32(self._encoder.run_positional([prompt, query])[0])

    def _estimate(self, x: _F32, mu: _F32, t: _F32, spks: _F32, cond: _F32) -> _F32:
        return _as_f32(self._estimator.run_positional([x, mu, t, spks, cond])[0])

    def decode(self, tokens: SpeechTokens, voice: VoiceProfile, *, seed: int) -> Mel:
        if self._spk_weight is None or self._spk_bias is None:
            raise RuntimeError(f"speaker affine not attached; build via {self._BUILDER}")
        row, cond, prompt_frames, n = frame_windows(self.config, tokens, voice)
        t_mel = 2 * row.shape[1]
        prompt_len = self._prompt_length(row)
        prompt = row[:, :prompt_len].astype(self._TOKEN_DTYPE)
        query = row[:, prompt_len:].astype(self._TOKEN_DTYPE)

        mu = self._encode(prompt, query).reshape(1, MEL_BINS, t_mel)

        emb = np.asarray(voice.flow_embedding, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        spks = (self._spk_weight @ emb + self._spk_bias)[None].astype(np.float32)

        grid = time_grid(self.config)
        x = gaussian_field(seed, FLOW_NOISE_STREAM, MEL_BINS, t_mel)[None]
        for t0, t1 in zip(grid[:-1], grid[1:], strict=False):
            v = self._estimate(x, mu, np.asarray([t0], dtype=np.float32), spks, cond).reshape(
                1, MEL_BINS, t_mel
            )
            # np.float32 keeps the step in fp32 (identical arithmetic, NEP 50
            # rounds a weak python float to the array dtype anyway) and keeps
            # numpy's stubs from promoting the whole state to float64
            x = x + np.float32(t1 - t0) * v
        return x[0, :, prompt_frames : prompt_frames + TOKEN_MEL_RATIO * n].astype(np.float32)


class ONNXVocoder:
    """``Vocoder`` over the exported fp32 HiFT graph (static 510-frame mel).

    The CoreML vocoder subclasses this and replaces ``_vocode``.
    """

    def __init__(
        self, config: AlgorithmConfig, assets: Path, *, execution: ExecutionConfig
    ) -> None:
        self.config = config
        self._hift = _load_session(assets, HIFT_GRAPH, execution)

    def _vocode(self, padded: _F32, phase: _F32, noise: _F32) -> _F32:
        return _as_f32(self._hift.run_positional([padded, phase, noise])[0])

    def synthesize(self, mel: Mel, voice: VoiceProfile, *, seed: int) -> Waveform:
        del voice  # timbre already lives in the mel; see TorchVocoder
        frames = 2 * self.config.window.max_speech_tokens
        n_frames = min(int(mel.shape[1]), frames)
        padded = np.zeros((1, MEL_BINS, frames), dtype=np.float32)
        padded[0, :, :n_frames] = mel[:, :n_frames]

        n_samples = frames * UPSAMPLE_PER_FRAME
        phase = np.zeros((1, VOCODER_HARMONICS, 1), dtype=np.float32)
        phase[0, 1:, 0] = symmetric_uniforms(
            seed, VOCODER_PHASE_STREAM, VOCODER_HARMONICS - 1, math.pi
        )
        noise = gaussian_field(seed, VOCODER_NOISE_STREAM, VOCODER_HARMONICS, n_samples)[None]

        wav = self._vocode(padded, phase, noise).reshape(-1)
        return wav[: n_frames * UPSAMPLE_PER_FRAME].astype(np.float32)


def _require_static_window(config: AlgorithmConfig, kind: str = "ONNX") -> tuple[int, int]:
    """Refuse a window the exported graphs were not built for.

    ``kind`` names the export the caller loaded, the one word the CoreML
    backend needs changed. See ``docs/design/execution-config.md``.
    """
    w = config.window
    if w.static_length != 255 or w.static_prompt_tokens != 238:
        raise ValueError(
            f"the exported {kind} graphs are static at query 255 / prompt 238; "
            f"this AlgorithmConfig frames {w.static_length}/{w.static_prompt_tokens}. "
            "A different window is a different algorithm: re-export the graphs "
            "rather than silently reframing here."
        )
    return w.static_length, w.static_prompt_tokens


@register_backend("onnx")
def build_onnx_engine(
    ckpt: Checkpoint, execution: ExecutionConfig, algorithm: AlgorithmConfig
) -> Engine:
    """Torch-free: every stage is an ONNX graph, fp32, on one execution provider."""
    _require_static_window(algorithm)
    _require_known_decode(algorithm)
    for module, prec in execution.precision_map().items():
        if prec != "fp32":
            raise ValueError(
                f"the ONNX backend exports fp32 graphs only; "
                f"ExecutionConfig.precision[{module!r}] = {prec!r}. "
                "fp16 was measured not worth a second artifact (EXP-015) and "
                "int8 is blocked (EXP-017): re-export for fp32 instead."
            )

    # Resolved before the sessions are built and written back into the config
    # the Engine carries, so describe() names the provider that ran, what a
    # benchmark row and a bug report both need. Resolving "auto" again inside
    # _load_session costs nothing and reads the same build's provider list.
    execution = replace(
        execution, onnx_provider=resolve_provider(execution.onnx_provider or "auto")
    )

    assets = _assets_dir(ckpt, algorithm)
    _check_provenance(assets, ckpt, algorithm)
    frontend = GraphemeTextFrontend(
        ckpt.resolve_asset(TOKENIZER_FILENAME, manifest_key="tokenizer_sha256")
    )

    token_generator = ONNXTokenGenerator(algorithm, ckpt, assets, execution=execution)
    mel_decoder = ONNXMelDecoder(algorithm, assets, execution=execution)
    affine = ckpt.tensors("s3gen.flow.spk_embed_affine_layer.")
    mel_decoder.attach_speaker_affine(
        np.asarray(affine["weight"], dtype=np.float32),
        np.asarray(affine["bias"], dtype=np.float32),
    )
    vocoder = ONNXVocoder(algorithm, assets, execution=execution)

    return Engine(
        frontend=frontend,
        token_generator=token_generator,
        mel_decoder=mel_decoder,
        vocoder=vocoder,
        algorithm=algorithm,
        execution=execution,
        backend="onnx",
        checkpoint_sha256=ckpt.file_digest,
        checkpoint_path=str(ckpt.path),
    )
