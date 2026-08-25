"""The ONNX backend: the three stages as ONNX graphs, driven from Python.

The graphs are exported by ``tools/export_onnx.py`` from the packed checkpoint
and run in **fp32** on the execution provider named by
``ExecutionConfig.onnx_provider``. That precision choice is the gate, not a
default: the manifest stores the generator and flow estimator in fp16, but
EXP-015 measured fp32 ONNX as the only version worth a second artifact (fp32
parity max abs delta-logit 1.9e-05; int8 is 2.15x but its quality is
unmeasured, and int8 stays blocked per EXP-017). Upcasting fp16 storage to fp32
is exact, so nothing is lost and the comparison to the torch fp32 reference is
a pure export question.

The token generator runs entirely on the graphs: ``t3_cond`` builds the
34-slot conditioning row (speaker projection + perceiver + emotion), ``t3_prefill``
one causal forward over the whole framed sequence (returning every-position
logits for teacher forcing *and* the KV cache for the loop), and ``t3_step`` one
decode step against the cache. The surrounding logic — framing, embeddings,
RoPE positions, the sampler loop, the EOS floor — is replicated here in numpy
and matched bit-for-bit against the torch generator; the graph only ever sees
embedding rows and position ids.

This module imports no torch module. The helpers it needs (window framing, the
Euler grid, the Philox stream ids) live in :mod:`loudkit.models.windowing`,
which is torch-free by design, so a ``loudkit[onnx]`` install can synthesise
without torch in the process at all.

The renderer mirrors :mod:`.coreml_backend` exactly: ``flow_encoder`` +
``flow_estimator`` behind the :class:`~loudkit.contracts.MelDecoder` protocol and
``vocoder`` (HiFT, conv STFT/iSTFT) behind :class:`~loudkit.contracts.Vocoder`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..checkpoint import Checkpoint
from ..config import AlgorithmConfig, ExecutionConfig, ONNXProvider
from ..contracts import Mel, Sampler, SpeechTokens, Waveform
from ..engine import Engine
from ..frontend.text import GraphemeTextFrontend
from ..models.noise import gaussian_field, symmetric_uniforms
from ..models.windowing import (
    FLOW_NOISE_STREAM,
    START_TEXT_TOKEN,
    STOP_TEXT_TOKEN,
    VOCODER_NOISE_STREAM,
    VOCODER_PHASE_STREAM,
    eos_floor,
    frame_windows,
    time_grid,
)
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
ENCODER_GRAPH = "flow_encoder.onnx"
ESTIMATOR_GRAPH = "flow_estimator.onnx"
HIFT_GRAPH = "vocoder.onnx"

_MEL_BINS = 80
_N_HARMONICS = 9
_UPSAMPLE_PER_FRAME = 480

_SESSIONS = (
    COND_GRAPH,
    PREFILL_GRAPH,
    STEP_GRAPH,
    ENCODER_GRAPH,
    ESTIMATOR_GRAPH,
    HIFT_GRAPH,
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

auto prefers a provider only where a measurement says it is faster. CoreML
is faster: the split placement in :func:`_session_providers` measures RTF
1.35-1.70 on an M3 Pro against 0.85-1.02 for all-CPU. It is still not a
default, for a reason that is not speed. Compiling the renderer graphs costs
about 146 s the first time on a machine and leaves 1.6 GB of cache behind. A
default may not spend either without being asked, and a first call that
appears to hang for two minutes is a worse first impression than a slower one
that returns. Ask for it by name.

DirectML has never been run by this project. It stays selectable and is not
a default. CUDA leads until it is measured, and drops out the same way if it
loses.
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
            "back to cpu — ask for 'auto' if a fallback is what you want."
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

Left at its default (``NeuralNetwork``) the renderer graphs shatter into
hundreds of partitions — flow_estimator 342, flow_encoder 47, vocoder 51 —
and each boundary is a copy between CoreML and CPU. Under MLProgram the same
graphs take 2, 1 and 25, which is the difference between losing to CPU and
beating it. The default also *changes the numbers*: a NeuralNetwork vocoder
sums 217.70 where CPU sums 211.15, while MLProgram sums 211.149. So this is
not a speed knob with a quality cost; the fast setting is also the faithful
one.
"""


def _coreml_cache_dir() -> Path:
    """Where CoreML writes compiled models, and why it must be somewhere.

    Compiling the renderer graphs takes about 146 s. With a cache directory
    that is paid once per machine and later loads cost about 25 s; without one
    it is paid on *every* session, which no interactive use can absorb. The
    cache runs to roughly 1.6 GB.

    ``$LOUDKIT_COREML_CACHE`` overrides. The default follows the platform
    convention, and CoreML exists on exactly one platform.
    """
    override = os.environ.get(COREML_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Caches" / "loudkit" / "coreml"


def _session_providers(
    provider: ONNXProvider, graph: str
) -> list[str | tuple[str, Mapping[str, str]]]:
    """The provider list handed to onnxruntime, per graph.

    The CPU tail is ORT's *per-operator* placement fallback, not a provider
    fallback — :func:`resolve_provider` has already refused a provider this
    build lacks. Without CPU in the list a graph holding one op the accelerator
    cannot place fails to load at all, which is how every non-trivial CUDA and
    CoreML session behaves.

    CoreML is the one provider that is not applied to all six graphs, because
    it loses on three of them. ``t3_step`` runs one decode step and is called
    once per speech token, so it decides the whole synthesis: 9.8 ms on CPU
    against 17.6 ms for the best CoreML configuration found. ``t3_prefill``
    and ``t3_step`` also fail to compile under MLProgram outright (session
    creation dies in ``model.mil`` with error -7), and MLProgram is the only
    setting worth having. So loudkit's ``coreml`` is a *placement*: the
    generator on CPU, the renderer on CoreML. Measured on an M3 Pro over
    three synthesis repeats that is RTF 1.35-1.70 against 0.85-1.02 for
    all-CPU.

    The generator never touching CoreML is what keeps the token stream
    identical to the CPU run, index for index. The waveform is not
    bit-identical, which is what the identity contract already says about
    running the renderer somewhere else.
    """
    cpu = PROVIDER_NAMES["cpu"]
    if provider == "cpu":
        return [cpu]
    if provider == "coreml":
        # An allowlist, not a denylist: a graph this module gains later lands
        # on CPU until somebody measures it there, rather than inheriting an
        # accelerator by default. The enrollment graphs are the case that
        # makes this matter in the ports — the voice encoder decides what a
        # cloned voice sounds like, and CoreML has never been measured on it.
        if graph not in RENDERER_GRAPHS:
            return [cpu]
        options = dict(_COREML_OPTIONS)
        options["ModelCacheDirectory"] = str(_coreml_cache_dir())
        return [(PROVIDER_NAMES["coreml"], options), cpu]
    return [PROVIDER_NAMES[provider], cpu]


class _Session:
    """A loaded ONNX graph, with its input/output names captured at load.

    onnxruntime ships no type information (see the pyproject override), so the
    names are pulled from the model object itself rather than asserted from a
    constant the exporter and this module would have to keep in sync. The
    values entering and leaving the graph are numpy arrays (typed Any at the
    boundary), pinned down with ``np.asarray(..., dtype=...)`` at the call
    sites.
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


def _assets_dir(ckpt: Checkpoint) -> Path:
    env = os.environ.get(ASSETS_ENV)
    candidates = [Path(env)] if env else []
    candidates.append(ckpt.path.parent / "onnx")
    for cand in candidates:
        if all((cand / name).exists() for name in _SESSIONS):
            return cand
    raise FileNotFoundError(
        f"ONNX assets not found. Expected {', '.join(_SESSIONS)} in "
        f"{[str(c) for c in candidates]} (override with ${ASSETS_ENV}). "
        "Run tools/export_onnx.py to create them."
    )


def _load_session(directory: Path, name: str, execution: ExecutionConfig) -> _Session:
    # Resolved per session rather than passed down: a component built directly
    # (the exporter does this) carries an unresolved "auto" and still has to
    # land on the same provider as one built through build_onnx_engine.
    return _Session(
        directory / name,
        threads=execution.num_threads,
        deterministic=execution.deterministic,
        providers=_session_providers(resolve_provider(execution.onnx_provider), name),
    )


def _as_f32(a: NDArray[np.generic]) -> NDArray[np.float32]:
    return np.asarray(a, dtype=np.float32)


class ONNXTokenGenerator:
    """``TokenGenerator`` over the three T3 graphs.

    The sampler stays a loudkit object (counter-based, hardware-agnostic); this
    class owns only the framing, the cache and the loop — the numpy mirror of
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
        self.config = config
        self._cond = _load_session(assets, COND_GRAPH, execution)
        self._prefill = _load_session(assets, PREFILL_GRAPH, execution)
        self._step = _load_session(assets, STEP_GRAPH, execution)

        t3 = ckpt.tensors("t3.")
        # fp16 storage upcasts exactly; the exported graphs carry the same
        # fp32 weights, so table and graph cannot drift.
        self._cond_cache: dict[tuple[bytes, bytes], NDArray[np.float32]] = {}
        self._text_emb = _as_f32(t3["text_emb.weight"])
        self._speech_emb = _as_f32(t3["speech_emb.weight"])
        self._text_pos = _as_f32(t3["text_pos_emb.emb.weight"])
        self._speech_pos = _as_f32(t3["speech_pos_emb.emb.weight"])

    # -- embedding construction (the numpy mirror of the torch module) -------

    def _cond_row(self, voice: VoiceProfile) -> NDArray[np.float32]:
        """``[1, 34, 1024]`` conditioning: speaker, perceiver prompt, emotion.

        Memoised by content (see :meth:`VoiceProfile.cond_key`): the row is a
        pure function of two profile tensors, and recomputing it per chunk was
        a full ``t3_cond`` session run each time.
        """
        key = voice.cond_key()
        cached = self._cond_cache.get(key)
        if cached is not None:
            return cached
        values: list[NDArray[np.generic]] = [
            np.asarray(voice.speaker_embedding, dtype=np.float32)[None],
            np.asarray(voice.cond_prompt_tokens, dtype=np.int64)[None],
            np.asarray([EMOTION_NEUTRAL], dtype=np.float32)[None],
        ]
        row = _as_f32(self._cond.run_positional(values)[0])
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
        cap = max_new_tokens or self.config.sampling.max_new_tokens
        floor = eos_floor(len(text_tokens), self.config)
        stop = self.config.stop_speech_token

        prefix = [int(t) for t in prefix]
        embeds = self._prefill_embeds(text_tokens, voice, prefix)
        prefill_len = embeds.shape[1]
        positions = np.arange(prefill_len, dtype=np.int64)
        logits_all, *kv = self._prefill.run_positional([embeds, positions])
        logits = _as_f32(logits_all)[0, -1].copy()

        seen = np.zeros(self.config.speech_vocab_size, dtype=bool)
        if prefix:
            seen[prefix] = True

        out: list[int] = []
        for step in range(cap):
            # Token-level cancellation (barge-in), same as the torch path.
            if should_cancel is not None and should_cancel():
                break
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
    """``MelDecoder`` over the exported encoder + estimator graphs."""

    def __init__(
        self, config: AlgorithmConfig, assets: Path, *, execution: ExecutionConfig
    ) -> None:
        if config.guidance != "single_path":
            # The decode loop below calls the estimator exactly once per step
            # and never forms (1+w)·v_cond − w·v_uncond. Accepting a dual-path
            # config would run different maths under a fingerprint that says
            # otherwise — the founding defect, one layer down, where
            # `_assert_one_algorithm` cannot see it: the component carries the
            # very config object it is disobeying, so the fingerprints agree.
            # CoreML refuses the same mode for the same reason.
            raise ValueError(
                f"the ONNX backend implements guidance 'single_path' only; this "
                f"algorithm declares {config.guidance!r}. The exported estimator is "
                "the guidance-distilled student, and running it once is not "
                "classifier-free guidance — export a dual-path graph, or load this "
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
        self._spk_weight = weight
        self._spk_bias = bias

    def decode(self, tokens: SpeechTokens, voice: VoiceProfile, *, seed: int) -> Mel:
        if self._spk_weight is None or self._spk_bias is None:
            raise RuntimeError("speaker affine not attached; build via build_onnx_engine")
        row, cond, prompt_frames, n = frame_windows(self.config, tokens, voice)
        t_mel = 2 * row.shape[1]
        prompt_len = self.config.window.static_prompt_tokens or row.shape[1] // 2
        prompt = row[:, :prompt_len].astype(np.int64)
        query = row[:, prompt_len:].astype(np.int64)

        mu = _as_f32(self._encoder.run_positional([prompt, query])[0]).reshape(
            1, _MEL_BINS, t_mel
        )

        emb = np.asarray(voice.flow_embedding, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        spks = (self._spk_weight @ emb + self._spk_bias)[None].astype(np.float32)

        grid = time_grid(self.config)
        x = gaussian_field(seed, FLOW_NOISE_STREAM, _MEL_BINS, t_mel)[None]
        for t0, t1 in zip(grid[:-1], grid[1:], strict=False):
            v = _as_f32(
                self._estimator.run_positional(
                    [
                        x,
                        mu,
                        np.asarray([t0], dtype=np.float32),
                        spks,
                        cond.astype(np.float32),
                    ]
                )[0]
            ).reshape(1, _MEL_BINS, t_mel)
            x = x + np.float32(t1 - t0) * v
        return x[0, :, prompt_frames : prompt_frames + 2 * n].astype(np.float32)


class ONNXVocoder:
    """``Vocoder`` over the exported fp32 HiFT graph (static 510-frame mel)."""

    def __init__(
        self, config: AlgorithmConfig, assets: Path, *, execution: ExecutionConfig
    ) -> None:
        self.config = config
        self._hift = _load_session(assets, HIFT_GRAPH, execution)

    def synthesize(self, mel: Mel, voice: VoiceProfile, *, seed: int) -> Waveform:
        del voice
        frames = 2 * self.config.window.max_speech_tokens
        n_frames = min(int(mel.shape[1]), frames)
        padded = np.zeros((1, _MEL_BINS, frames), dtype=np.float32)
        padded[0, :, :n_frames] = mel[:, :n_frames]

        n_samples = frames * _UPSAMPLE_PER_FRAME
        phase = np.zeros((1, _N_HARMONICS, 1), dtype=np.float32)
        phase[0, 1:, 0] = symmetric_uniforms(
            seed, VOCODER_PHASE_STREAM, _N_HARMONICS - 1, np.pi
        )
        noise = gaussian_field(seed, VOCODER_NOISE_STREAM, _N_HARMONICS, n_samples)[None]

        wav = _as_f32(self._hift.run_positional([padded, phase, noise])[0]).reshape(-1)
        return wav[: n_frames * _UPSAMPLE_PER_FRAME].astype(np.float32)


def _require_static_window(config: AlgorithmConfig) -> tuple[int, int]:
    """Refuse a window the exported graphs were not built for.

    The CoreML backend has always checked this; ONNX did not, and the two are
    exported from the same recipe. A config framing anything other than 255/238
    reached `decode`, where `static_prompt_tokens or row.shape[1] // 2` silently
    picked a *different* prompt split from the one `frame_windows` used — so the
    graph read a prompt boundary the framing never put there and the mel came
    out subtly wrong, with no error on any layer.

    A different window is a different algorithm. Re-export the graphs.
    """
    w = config.window
    if w.static_length != 255 or w.static_prompt_tokens != 238:
        raise ValueError(
            "the exported ONNX graphs are static at query 255 / prompt 238; "
            f"this AlgorithmConfig frames {w.static_length}/{w.static_prompt_tokens}. "
            "A different window is a different algorithm — re-export the graphs "
            "rather than silently reframing here."
        )
    return w.static_length, w.static_prompt_tokens


@register_backend("onnx")
def build_onnx_engine(
    ckpt: Checkpoint, execution: ExecutionConfig, algorithm: AlgorithmConfig
) -> Engine:
    """Torch-free: every stage is an ONNX graph, fp32, on one execution provider."""
    _require_static_window(algorithm)
    for module, prec in execution.precision.items():
        if prec != "fp32":
            raise ValueError(
                f"the ONNX backend exports fp32 graphs only; "
                f"ExecutionConfig.precision[{module!r}] = {prec!r}. "
                "fp16 was measured not worth a second artifact (EXP-015) and "
                "int8 is blocked (EXP-017) — re-export for fp32 instead."
            )

    # Resolved before the sessions are built and written back into the config
    # the Engine carries, so describe() names the provider that ran — what a
    # benchmark row and a bug report both need. Resolving "auto" again inside
    # _load_session costs nothing and reads the same build's provider list.
    execution = replace(execution, onnx_provider=resolve_provider(execution.onnx_provider))

    assets = _assets_dir(ckpt)
    frontend = GraphemeTextFrontend(
        ckpt.resolve_asset("tokenizer.json", manifest_key="tokenizer_sha256")
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
    )
