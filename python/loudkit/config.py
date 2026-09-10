"""What the engine computes, and how fast: two layers, kept apart.

See ``docs/design/runtime-notes.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Final, Literal

from .execution import ONNX_PROVIDERS, Device, ExecutionConfig, ONNXProvider, Precision
from .frontend.textconfig import TextConfig
from .postprocess import PostprocessConfig

__all__ = [
    "AlgorithmConfig",
    "ChunkConfig",
    "DecodeMode",
    "Device",
    "ExecutionConfig",
    "GuidanceMode",
    "ONNXProvider",
    "Precision",
    "SamplingConfig",
    "WindowConfig",
]

_ = ONNX_PROVIDERS  # re-exported: config is the door to both layers, so it is read here

FINGERPRINT_SCHEMA = 1
"""Version of the canonical form. Bump only when the serialisation changes."""

RECIPE_VERSION = "loudkit-1"
"""The one recipe. There is no other."""


class _UnsetType:
    """The sentinel for an optional field never set; survives copy and deepcopy as itself."""

    _instance: _UnsetType | None = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __deepcopy__(self, memo: dict[int, object]) -> _UnsetType:
        return self

    def __copy__(self) -> _UnsetType:
        return self

    def __repr__(self) -> str:
        return "UNSET"


_UNSET: Final[_UnsetType] = _UnsetType()
"""The one instance, typed as itself so the three fields defaulting to it say so.

A field annotated `DecodeMode` that holds this sentinel until its accessor is
read is an annotation that is not true, and a caller type-checking against it
writes `AlgorithmConfig().decode_mode == "single"` and gets `False`. The three
fields are therefore annotated as unions with this type, and the effective
values are read through :attr:`AlgorithmConfig.decode`,
:attr:`AlgorithmConfig.edge_fade` and
:meth:`ChunkConfig.resolved_first_chunk_max_tokens`, which are narrow.
"""

GuidanceMode = Literal["single_path", "cfg_dual_path"]
"""``single_path`` is one estimator call per step and is what ships.
``cfg_dual_path`` is for a teacher that was never guidance-distilled."""

DecodeMode = Literal["single", "fusion_mtp2"]
"""Speech tokens per transformer forward: one, or a fused pair. A property of the weights."""

EDGE_FADE_SECONDS = 0.02
"""The raised-cosine edge ramp in seconds; see docs/design/postprocess.md."""


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """The sampling law, LR-SAMPLER-v1 (:mod:`loudkit.sampler`). Identical on every backend."""

    temperature: float = 0.8
    repetition_penalty: float = 1.2
    min_p: float = 0.05
    max_new_tokens: int = 255

    silence_token_ids: tuple[int, ...] = ()
    """Tokens exempt from the ``min_p`` floor, and from nothing else."""

    min_tokens_floor: int = 0
    min_tokens_text_ratio: float = 0.0
    """The stop token is masked until ``max(floor, n_text * ratio)`` speech tokens exist."""

    def __post_init__(self) -> None:
        if not 0.0 < self.temperature <= 4.0:
            raise ValueError(f"temperature out of range: {self.temperature}")
        if self.repetition_penalty < 1.0:
            raise ValueError(
                f"repetition_penalty below 1.0 rewards repetition: {self.repetition_penalty}"
            )
        if not 0.0 <= self.min_p < 1.0:
            raise ValueError(f"min_p out of range: {self.min_p}")
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive: {self.max_new_tokens}")
        if self.min_tokens_floor < 0:
            raise ValueError(f"min_tokens_floor must be >= 0: {self.min_tokens_floor}")
        if self.min_tokens_text_ratio < 0.0:
            raise ValueError(
                f"min_tokens_text_ratio must be >= 0: {self.min_tokens_text_ratio}"
            )


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """How text longer than one window is split. Where the reader breathes is audible."""

    enabled: bool = True

    max_tokens: int = 255
    """Longest run of speech tokens one chunk may produce; matches the window."""

    prefix_tokens: int = 6
    """Speech tokens of the previous chunk fed back as context. Zero restarts the pitch
    contour at every join."""

    split_on: tuple[str, ...] = (". ", "! ", "? ", "; ", ", ")
    """Split candidates, strongest first."""

    abbreviations: tuple[str, ...] = (
        "A",
        "B",
        "Cpn",
        "D",
        "Dr",
        "F",
        "H",
        "Hr",
        "I",
        "J",
        "K",
        "M",
        "Mr",
        "Mrs",
        "R",
        "S",
        "St",
        "T",
        "V",
        "Vors",
        "dr",
        "mrs",
        "prof",
        "św",
    )
    """Words whose following period does not end a sentence. One list for every language;
    hashed as written, so keep it sorted."""

    cap_resplit: Literal["word", "off"] = "word"
    """What happens to a chunk that filled its window: split at the middle word and
    generate both halves under the same chunk index, or ship the truncation."""

    mid_sentence_period: Literal["hold", "break"] = "hold"
    """``hold``: a period after an abbreviation or before a lower-case word is not a
    boundary. ``break``: every separator match is."""

    first_chunk_max_tokens: int | None | _UnsetType = _UNSET
    """Token budget for the first chunk only, for time to first audio. Unset leaves
    the fingerprint alone; set, it is an audible decision and hashes."""

    def resolved_first_chunk_max_tokens(self) -> int | None:
        value = self.first_chunk_max_tokens
        return None if isinstance(value, _UnsetType) else value

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive: {self.max_tokens}")
        first = self.resolved_first_chunk_max_tokens()
        if first is not None and not (0 < first <= self.max_tokens):
            raise ValueError(
                f"first_chunk_max_tokens must be in 1..max_tokens ({self.max_tokens}): {first}"
            )
        # A character budget of zero makes split_text loop forever.
        from .frontend.chunking import CHARS_PER_TOKEN

        if int(self.max_tokens * CHARS_PER_TOKEN) < 1:
            raise ValueError(
                f"max_tokens={self.max_tokens} leaves no character budget to split on "
                f"(int({self.max_tokens} * {CHARS_PER_TOKEN}) == 0); "
                f"needs at least {math.ceil(1 / CHARS_PER_TOKEN)}"
            )
        if not 0 <= self.prefix_tokens < self.max_tokens:
            raise ValueError(f"prefix_tokens must be in [0, max_tokens): {self.prefix_tokens}")
        if not self.split_on:
            raise ValueError("split_on cannot be empty: there would be nowhere to break")
        if self.cap_resplit not in ("word", "off"):
            raise ValueError(
                f"unknown cap_resplit {self.cap_resplit!r}: expected 'word' or 'off'"
            )
        if self.mid_sentence_period not in ("hold", "break"):
            raise ValueError(
                f"unknown mid_sentence_period {self.mid_sentence_period!r}: "
                'expected "hold" or "break"'
            )
        if any(not a for a in self.abbreviations):
            # An empty entry is a suffix of everything and holds every split.
            raise ValueError("abbreviations cannot contain an empty string")


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """How token sequences are framed for the mel decoder.

    Algorithm layer, because the pad-and-truncate recipe was the whole measured
    deviation between two backends' renders.
    """

    max_speech_tokens: int = 255
    """Longest token sequence one window carries (~10.2 s at 25 Hz)."""

    static_length: int | None = None
    """Pad every window to this many tokens, or ``None`` for ragged."""

    pad_token_id: int | None = None
    """Padding token. ``None`` means the silence token; token 0 bleeds into the tail."""

    static_prompt_tokens: int | None = None
    """Fixed length of the reference-prompt window, or ``None`` for ragged."""

    def __post_init__(self) -> None:
        if self.static_length is not None and self.static_length < self.max_speech_tokens:
            raise ValueError(
                f"static_length {self.static_length} cannot be shorter than "
                f"max_speech_tokens {self.max_speech_tokens}"
            )
        if self.static_prompt_tokens is not None and self.static_prompt_tokens <= 0:
            raise ValueError(
                f"static_prompt_tokens must be positive: {self.static_prompt_tokens}"
            )


@dataclass(frozen=True, slots=True)
class AlgorithmConfig:
    """Everything that determines what the engine produces. Hashable; printed on every run."""

    recipe_version: str = RECIPE_VERSION
    """Names the parts of the algorithm that are code: the sampling law, the grid
    formula, the framing recipe. Bump it when the recipe changes."""

    guidance: GuidanceMode = "single_path"
    guidance_rate: float = 0.0
    """Only read when ``guidance == "cfg_dual_path"``."""

    euler_steps: int = 2
    euler_grid: tuple[float, ...] | None = None
    """Explicit time grid of ``euler_steps + 1`` values in [0, 1]; ``None`` is the cosine
    schedule."""

    decode_mode: DecodeMode | _UnsetType = _UNSET
    """Unset rather than ``"single"`` so adding the field left every fingerprint alone.
    Read it through :attr:`decode`."""

    edge_fade_seconds: float | _UnsetType = _UNSET
    """The edge ramp, defaulting to 20 ms and included in the canonical form.
    Only historical 5 ms is omitted. Read the effective value through :attr:`edge_fade`.
    """

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    """The artifact detectors' preset. Read from the checkpoint; ``mode`` is the one knob."""

    text: TextConfig = field(default_factory=TextConfig)
    """What the text funnel is. Hashed, because the funnel decides what the model is handed."""

    sample_rate: int = 24_000
    token_rate_hz: float = 25.0
    speech_vocab_size: int = 8194
    start_speech_token: int = 6561
    stop_speech_token: int = 6562

    def __post_init__(self) -> None:
        if self.guidance == "single_path" and self.guidance_rate != 0.0:
            raise ValueError("guidance_rate must be 0.0 in single_path mode")
        if self.guidance == "cfg_dual_path" and self.guidance_rate <= 0.0:
            raise ValueError("cfg_dual_path with a zero rate does twice the work for nothing")
        if self.euler_steps < 1:
            raise ValueError(f"euler_steps must be >= 1: {self.euler_steps}")
        self._validate_numeric_core()
        if self.euler_grid is not None:
            grid = self.euler_grid
            if len(grid) != self.euler_steps + 1:
                raise ValueError(
                    f"euler_grid has {len(grid)} points, expected {self.euler_steps + 1}"
                )
            if not all(b > a for a, b in zip(grid, grid[1:], strict=False)):
                raise ValueError("euler_grid must be strictly increasing")
            if abs(grid[0]) > 1e-6 or abs(grid[-1] - 1.0) > 1e-6:
                raise ValueError("euler_grid must run from 0.0 to 1.0")
        # The three token budgets have to agree, or a chunk overruns the render
        # window mid-stream, after earlier chunks have already played.
        window = self.window.max_speech_tokens
        if self.chunking.enabled and self.chunking.max_tokens > window:
            raise ValueError(
                f"chunking.max_tokens {self.chunking.max_tokens} exceeds the render "
                f"window ({window}): every chunk would be sized past what the "
                "renderer accepts, and the refusal would land mid-stream, after "
                "audio had already been delivered"
            )
        if self.sampling.max_new_tokens > window:
            raise ValueError(
                f"sampling.max_new_tokens {self.sampling.max_new_tokens} exceeds the "
                f"render window ({window}): generation is allowed to produce more "
                "speech than the renderer will accept, so a long utterance fails "
                "after it has been generated rather than before"
            )

    def _validate_numeric_core(self) -> None:
        fade = self.edge_fade_seconds
        if not isinstance(fade, _UnsetType) and not 0.001 <= fade <= 0.05:
            raise ValueError(f"edge_fade_seconds must be in [0.001, 0.05]: {fade}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0: {self.sample_rate}")
        if self.token_rate_hz <= 0:
            raise ValueError(f"token_rate_hz must be > 0: {self.token_rate_hz}")
        if self.speech_vocab_size < 1:
            raise ValueError(f"speech_vocab_size must be >= 1: {self.speech_vocab_size}")
        for name in ("start_speech_token", "stop_speech_token"):
            value = getattr(self, name)
            if not 0 <= value < self.speech_vocab_size:
                raise ValueError(f"{name} must be in [0, {self.speech_vocab_size}): {value}")
        if self.start_speech_token == self.stop_speech_token:
            raise ValueError(
                "start_speech_token and stop_speech_token must differ: both are "
                f"{self.start_speech_token}"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def canonical_form(self) -> str:
        """The exact string that is hashed: floats as ``repr``, keys sorted, unset fields
        dropped."""

        def norm(value: object) -> object:
            if isinstance(value, float):
                return repr(float(value))
            if isinstance(value, (list, tuple)):
                return [norm(v) for v in value]
            if isinstance(value, dict):
                return {k: norm(v) for k, v in sorted(value.items()) if v is not _UNSET}
            if hasattr(value, "item") and not isinstance(value, (str, bytes)):
                return norm(value.item())  # numpy scalars
            return value

        values = self.to_dict()
        values["edge_fade_seconds"] = self.edge_fade if self.edge_fade != 0.005 else _UNSET
        body = {k: norm(v) for k, v in sorted(values.items()) if v is not _UNSET}
        return json.dumps(
            {"schema": FINGERPRINT_SCHEMA, "algorithm": body},
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def decode(self) -> DecodeMode:
        value = self.decode_mode
        return "single" if isinstance(value, _UnsetType) else value

    @property
    def edge_fade(self) -> float:
        value = self.edge_fade_seconds
        return EDGE_FADE_SECONDS if isinstance(value, _UnsetType) else float(value)

    def fingerprint(self) -> str:
        """Sixteen hex digits over every algorithm value. Two engines that differ here
        compute different things."""
        return hashlib.sha256(self.canonical_form().encode()).hexdigest()[:16]

    def describe(self) -> str:
        """One line for logs."""
        g = self.guidance if self.guidance == "single_path" else f"cfg@{self.guidance_rate}"
        grid = "explicit" if self.euler_grid else "cosine"
        decode = "" if self.decode == "single" else f" decode={self.decode}"
        fade = (
            "" if isinstance(self.edge_fade_seconds, _UnsetType) else f" fade={self.edge_fade}s"
        )
        return (
            f"algo[{self.fingerprint()}] {self.recipe_version} {g}{decode}{fade} "
            f"euler={self.euler_steps}({grid}) "
            f"temp={self.sampling.temperature} rep={self.sampling.repetition_penalty} "
            f"min_p={self.sampling.min_p} sil={len(self.sampling.silence_token_ids)} "
            f"win={self.window.static_length or 'ragged'}"
        )

    def with_(self, **changes: object) -> AlgorithmConfig:
        """Copy with overrides, re-validated."""
        return replace(self, **changes)  # type: ignore[arg-type]

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object]) -> AlgorithmConfig:
        """Build from a checkpoint manifest; see :mod:`loudkit.manifest`."""
        from .manifest import algorithm_from

        return algorithm_from(manifest)


DEFAULT_ALGORITHM = AlgorithmConfig()
