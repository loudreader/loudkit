"""A voice, as data rather than as a model.

See ``docs/design/runtime-notes.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["VoiceProfile", "VOICE_FORMAT_VERSION", "MIN_EMBEDDING_NORM"]

VOICE_FORMAT_VERSION = 1

EMOTION_NEUTRAL = 0.5
"""The constant fed to the generator's emotion conditioning slot.

See ``docs/design/runtime-notes.md``.
"""

MIN_EMBEDDING_NORM = 1e-6
"""Smallest speaker-vector norm a profile may carry.

Below this the renderers stop agreeing: ONNX and CoreML divide by the raw norm
and yield NaN, torch's ``F.normalize`` carries an epsilon and yields a finite
but arbitrary direction. Enrolled vectors are order-1; anything this small is
a corrupt or synthetic file, not a quiet voice.
"""

MAX_VOICE_BYTES = 8 * 1024 * 1024
"""Largest voice file this will open.

See ``docs/design/runtime-notes.md``.
"""

_MAX_NAME_CHARS = 200
"""Longest ``name`` carried in a profile's header.

The name is metadata a caller supplies at enrolment and the server echoes back
in responses; nothing downstream truncates it. A megabyte of it in a 165 KB file
is legal safetensors and pointless otherwise.
"""


KNOWN_ENROLMENTS = frozenset({"first-10s", "first-10s-pause"})
"""Strategies this implementation can honour.

See ``docs/design/runtime-notes.md``.
"""

ENROLMENT_FIRST_WINDOW = "first-10s"
"""The original strategy: the prompt is the first ten seconds of the clip.

Every voice enrolled before the field existed was made this way, so a profile
whose header does not name a strategy is read as this one. The speaker embedding
has always read the whole clip; this names what the *prompt* was cut from.
"""

ENROLMENT_PAUSE_CUT = "first-10s-pause"
"""The prompt is the clip cut at its last pause before ten seconds, then 0.4 s of silence.

What ``loudkit clone`` makes by default and ``enroll(end_in_silence=True)`` makes on
request; a different prompt from the same recording, so a different label.
"""


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    """Everything needed to speak as one voice, and nothing else.

    See ``docs/design/runtime-notes.md``.
    """

    name: str
    speaker_embedding: NDArray[np.float32]
    flow_embedding: NDArray[np.float32]
    prompt_tokens: NDArray[np.int64]
    prompt_mel: NDArray[np.float32]
    cond_prompt_tokens: NDArray[np.int64]
    source_sample_rate: int = 24_000
    language: str = "en"

    source_sha256: str = ""
    """SHA-256 of the file this profile was loaded from, or ``""``.

    Set by :meth:`load`, never stored in the file (a file cannot carry its own
    digest). Provenance manifests record it as ``voice_profile_sha256``, so a
    rendered WAV names the exact profile bytes that voiced it, a voice *name*
    is a label anyone can reuse. Empty on a profile that never touched disk,
    such as one freshly enrolled.
    """

    enrolment: str = ENROLMENT_FIRST_WINDOW
    """Which reference audio the prompt was built from.

    See ``docs/design/runtime-notes.md``.
    """

    def __post_init__(self) -> None:
        self._validate_shapes()
        self._validate_values()
        self._validate_enrolment()

    def _validate_shapes(self) -> None:
        """The dimensions the two speaker encoders actually produce."""
        if self.speaker_embedding.ndim != 1:
            raise ValueError(
                f"speaker_embedding must be 1-D, got shape {self.speaker_embedding.shape}"
            )
        if self.speaker_embedding.shape[0] != 256:
            raise ValueError(
                f"speaker_embedding must be 256-d (token generator's voice-encoder "
                f"vector), got {self.speaker_embedding.shape[0]}"
            )
        if self.flow_embedding.ndim != 1:
            raise ValueError(
                f"flow_embedding must be 1-D, got shape {self.flow_embedding.shape}"
            )
        if self.flow_embedding.shape[0] != 192:
            raise ValueError(
                f"flow_embedding must be 192-d (CAM++ x-vector), got "
                f"{self.flow_embedding.shape[0]}"
            )
        if self.prompt_mel.ndim != 2 or self.prompt_mel.shape[0] != 80:
            raise ValueError(f"prompt_mel must be (80, frames), got {self.prompt_mel.shape}")
        if self.prompt_tokens.ndim != 1 or self.cond_prompt_tokens.ndim != 1:
            raise ValueError("token prompts must be 1-D")

    def _validate_enrolment(self) -> None:
        if self.enrolment not in KNOWN_ENROLMENTS:
            raise ValueError(
                f"{self.name or 'voice'}: enrolment strategy {self.enrolment!r} is not "
                f"one this build implements ({', '.join(sorted(KNOWN_ENROLMENTS))}). "
                "The profile was made by a build that cuts its prompt differently, "
                "so loading it here would speak in a different voice under the same "
                "name."
            )

    def _validate_values(self) -> None:
        """Finiteness, usable norms, and ranges.

        See ``docs/design/runtime-notes.md``.
        """
        for name, vector in (
            ("speaker_embedding", self.speaker_embedding),
            ("flow_embedding", self.flow_embedding),
        ):
            if not np.isfinite(vector).all():
                raise ValueError(f"{name} contains NaN or infinity")
            norm = float(np.linalg.norm(vector.astype(np.float64)))
            if norm < MIN_EMBEDDING_NORM:
                raise ValueError(
                    f"{name} has norm {norm:g}, below {MIN_EMBEDDING_NORM:g}. "
                    "A zero or near-zero speaker vector normalises to NaN on the "
                    "ONNX and CoreML renderers and to a finite arbitrary direction "
                    "on torch, so the same file would speak differently per backend."
                )
        if not np.isfinite(self.prompt_mel).all():
            raise ValueError("prompt_mel contains NaN or infinity")
        # Both ends, not just the floor.
        from .config import DEFAULT_ALGORITHM

        limits = DEFAULT_ALGORITHM
        for name, tokens, ceiling in (
            ("prompt_tokens", self.prompt_tokens, limits.start_speech_token),
            ("cond_prompt_tokens", self.cond_prompt_tokens, limits.speech_vocab_size),
        ):
            # `AlgorithmConfig`'s defaults are the shipped weights' dimensions,
            # and `check_manifest_sizes` refuses a manifest that says otherwise,
            # so these two agree by construction rather than by coincidence.
            if not tokens.size:
                continue
            if int(tokens.min()) < 0:
                raise ValueError(f"{name} contains a negative id: {int(tokens.min())}")
            if int(tokens.max()) >= ceiling:
                raise ValueError(
                    f"{name} contains id {int(tokens.max())}, at or above the "
                    f"{ceiling} the model can embed"
                )
        if self.source_sample_rate <= 0:
            raise ValueError(f"source_sample_rate must be positive: {self.source_sample_rate}")

    @property
    def n_bytes(self) -> int:
        """Total payload size, for the "a voice is a file" claim."""
        return sum(
            int(a.nbytes)
            for a in (
                self.speaker_embedding,
                self.flow_embedding,
                self.prompt_tokens,
                self.prompt_mel,
                self.cond_prompt_tokens,
            )
        )

    def cond_key(self) -> tuple[bytes, bytes]:
        """A content key for caching this profile's conditioning row.

        See ``docs/design/runtime-notes.md``.
        """
        return (
            hashlib.sha256(np.ascontiguousarray(self.speaker_embedding).tobytes()).digest(),
            hashlib.sha256(np.ascontiguousarray(self.cond_prompt_tokens).tobytes()).digest(),
        )

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write to ``safetensors``. Returns the path written."""
        from safetensors.numpy import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "format_version": VOICE_FORMAT_VERSION,
            "name": self.name,
            "source_sample_rate": self.source_sample_rate,
            "language": self.language,
            "enrolment": self.enrolment,
        }
        save_file(
            # Enrollment crops the mel along its last axis, leaving a strided
            # view. safetensors requires dense C-order storage for every tensor.
            {
                "speaker_embedding": np.ascontiguousarray(self.speaker_embedding),
                "flow_embedding": np.ascontiguousarray(self.flow_embedding),
                "prompt_tokens": np.ascontiguousarray(self.prompt_tokens),
                "prompt_mel": np.ascontiguousarray(self.prompt_mel),
                "cond_prompt_tokens": np.ascontiguousarray(self.cond_prompt_tokens),
            },
            str(path),
            metadata={"voice": json.dumps(header)},
        )
        # POSIX can make this owner-only directly. Windows has no equivalent
        # group/world mode bits; keep the ACL inherited from the destination
        # directory instead of pretending chmod(0600) tightened it.
        if os.name == "posix":
            path.chmod(0o600)
        return path

    @staticmethod
    def _header_str(header: Mapping[str, object], name: str, key: str, default: str) -> str:
        """A JSON string under ``header``, or ``default`` for an absent key.

        Not ``str()``, which turns a number into its digits: ``language: 5``
        would be the language id ``"5"``, and language selects the text funnel,
        so the voice would be read through a funnel nobody chose. A header
        value of the wrong type was written by something that did not mean it,
        and the manifest reader refuses the same shape in the same sentence.
        """
        if key not in header:
            return default
        value = header[key]
        if not isinstance(value, str):
            raise ValueError(f"{name}: voice header {key!r} must be a string, got {value!r}")
        return value

    @staticmethod
    def _header_int(header: Mapping[str, object], name: str, key: str, default: int) -> int:
        """A whole JSON number under ``header``, or ``default`` for an absent key."""
        if key not in header:
            return default
        value = header[key]
        # bool first: it is an int subclass, and `true` would be version one.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}: voice header {key!r} must be a number, got {value!r}")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(
                f"{name}: voice header {key!r} must be a whole number, got {value!r}"
            )
        return int(value)

    @classmethod
    def load(cls, path: str | Path) -> VoiceProfile:
        """Read a profile written by :meth:`save`."""
        from safetensors import safe_open

        path = Path(path)
        size = path.stat().st_size
        if size > MAX_VOICE_BYTES:
            raise ValueError(
                f"{path.name}: {size} bytes, over the {MAX_VOICE_BYTES} byte "
                "limit for a voice: see MAX_VOICE_BYTES"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with safe_open(str(path), framework="numpy") as f:
            meta: Mapping[str, str] = f.metadata() or {}
            header = json.loads(meta.get("voice", "{}"))
            if not isinstance(header, dict):
                # External data: a JSON list or string here would otherwise
                # surface as an AttributeError from inside a getter, where
                # every other malformed header is a ValueError.
                raise ValueError(
                    f"{path.name}: voice header is a {type(header).__name__}, "
                    "expected a JSON object"
                )
            version = cls._header_int(header, path.name, "format_version", 0)
            if version != VOICE_FORMAT_VERSION:
                raise ValueError(
                    f"{path.name}: voice format version {version}, "
                    f"this build reads {VOICE_FORMAT_VERSION}"
                )
            return cls(
                name=cls._header_str(header, path.name, "name", path.stem)[:_MAX_NAME_CHARS],
                speaker_embedding=f.get_tensor("speaker_embedding"),
                flow_embedding=f.get_tensor("flow_embedding"),
                prompt_tokens=f.get_tensor("prompt_tokens"),
                prompt_mel=f.get_tensor("prompt_mel"),
                cond_prompt_tokens=f.get_tensor("cond_prompt_tokens"),
                # Profiles written before 0.1 carry an "emotion" key; it is
                # ignored. The axis is dead on these weights (distillation
                # collapsed it) and the conditioning slot is fed
                # EMOTION_NEUTRAL, the value every profile ever written had.
                source_sample_rate=cls._header_int(
                    header, path.name, "source_sample_rate", 24_000
                ),
                source_sha256=digest,
                language=cls._header_str(header, path.name, "language", "en"),
                # Absent means the profile predates the field, and every one of
                # those was cut from the first ten seconds.
                enrolment=cls._header_str(
                    header, path.name, "enrolment", ENROLMENT_FIRST_WINDOW
                ),
            )

    def __repr__(self) -> str:
        return (
            f"VoiceProfile({self.name!r}, {self.language}, "
            f"{len(self.prompt_tokens)} prompt tokens, "
            f"{self.prompt_mel.shape[1]} mel frames, {self.n_bytes / 1024:.0f} KB)"
        )
