"""A voice, as data rather than as a model.

A :class:`VoiceProfile` is the handful of tensors the two stages need in order
to speak as someone: a speaker embedding, a prompt of speech tokens, the mel of
the reference audio, and the conditioning the token generator was trained to
read. A few hundred kilobytes, no weights.

That framing is deliberate and it is what makes cloning cheap. An earlier
version baked the prompt into the graph, so every voice was a separate exported
model of several hundred megabytes; taking the prompt as an input instead turned
a voice into a file you can email. It also means voices can be enrolled once on
a fast machine and shipped, rather than re-derived on a phone, which matters
because enrollment needs a speaker encoder and a speech tokenizer that synthesis
otherwise never touches — together about 40% of the checkpoint.

Profiles are saved as ``safetensors`` with a small JSON header, so they are
inspectable, versioned, and safe to load from an untrusted source.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["VoiceProfile", "VOICE_FORMAT_VERSION", "MIN_EMBEDDING_NORM"]

VOICE_FORMAT_VERSION = 1

EMOTION_NEUTRAL = 0.5
"""The constant fed to the generator's emotion conditioning slot.

The checkpoint architecture reserves one of its 34 conditioning slots for an
emotion scalar (``t3.cond_enc.emotion_adv_fc``). On these weights the axis is
dead — distillation collapsed the response — so the slot is not a control and
is not part of the profile format. It still has to be fed *something*, and it
has to be the value the model was distilled with and every profile ever
written carried: 0.5. Every renderer in every port uses this constant.
"""

MIN_EMBEDDING_NORM = 1e-6
"""Smallest speaker-vector norm a profile may carry.

Below this the renderers stop agreeing: ONNX and CoreML divide by the raw norm
and yield NaN, torch's ``F.normalize`` carries an epsilon and yields a finite —
but arbitrary — direction. Enrolled vectors are order-1; anything this small is
a corrupt or synthetic file, not a quiet voice.
"""

MAX_VOICE_BYTES = 8 * 1024 * 1024
"""Largest voice file this will open.

Every voice the kit ships weighs about 165 KB, and the format has no field that
grows with anything a caller controls — so fifty times the real size is room for
a format change, not for a payload. Without a bound, ``prompt_mel`` declared as
``(80, 100_000_000)`` is 64 GB of allocation the moment it is read, from a file
the server loads by name on an unauthenticated request. Checked on the file
rather than per tensor because the file bounds every tensor in it at once, and
does so before anything is materialised.
"""

_MAX_NAME_CHARS = 200
"""Longest ``name`` carried in a profile's header.

The name is metadata a caller supplies at enrolment and the server echoes back
in responses; nothing downstream truncates it. A megabyte of it in a 165 KB file
is legal safetensors and pointless otherwise.
"""


KNOWN_ENROLMENTS = frozenset({"first-10s"})
"""Strategies this implementation can honour.

A profile naming anything else is refused at load. That is the point of the
field: a build without the strategy that produced a voice must say so, rather
than apply its own and hand back a different voice under the same name. Adding a
strategy means adding it here *and* in the four ports, in one go — the failure
this guards against is a profile that loads everywhere and means something
different in one of them.
"""

ENROLMENT_FIRST_WINDOW = "first-10s"
"""The original strategy: the prompt is the first ten seconds of the clip.

Every voice enrolled before the field existed was made this way, so a profile
whose header does not name a strategy is read as this one. The speaker embedding
has always read the whole clip; this names what the *prompt* was cut from.
"""


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    """Everything needed to speak as one voice, and nothing else.

    The two stages read two *different* speaker encoders' outputs, so a profile
    carries two embeddings: the token generator was trained against a 256-d
    utterance-level voice-encoder vector, while the flow decoder conditions on
    a 192-d CAM++ x-vector. They are not interchangeable and neither can be
    derived from the other, which is why both are enrolled once and stored.

    Attributes:
        name: human label. Carried for provenance and error messages only;
            nothing dispatches on it.
        speaker_embedding: ``(256,)`` speaker vector for the token generator's
            conditioning encoder.
        flow_embedding: ``(192,)`` x-vector for the mel decoder.
        prompt_tokens: speech tokens of the reference audio, the prosodic and
            timbral prompt the mel decoder continues from. Stored at natural
            length; the window recipe (``WindowConfig``) decides framing.
        prompt_mel: ``(80, frames)`` mel of the reference, conditioning the flow.
        cond_prompt_tokens: the token generator's own conditioning prompt, which
            may be a different length from ``prompt_tokens``.
        source_sample_rate: sample rate of the audio this was enrolled from.
            Provenance only — **nothing reads it**. It said "kept so a mismatch
            is detectable rather than silently resampled", which described a
            check no layer performs: `enroll` writes it and no renderer, loader
            or engine looks at it again. Recorded here rather than removed
            because the fact is worth carrying and a future check would want it;
            described honestly because a promise in a docstring is the kind of
            thing a caller builds on.
        language: language of the reference audio, for provenance.
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
    rendered WAV names the exact profile bytes that voiced it — a voice *name*
    is a label anyone can reuse. Empty on a profile that never touched disk,
    such as one freshly enrolled.
    """

    enrolment: str = ENROLMENT_FIRST_WINDOW
    """Which reference audio the prompt was built from.

    A profile is an artefact, and this says how it was made. Enrolment picks a
    window of the reference clip before any of the tested transform runs, so two
    strategies produce two different voices from one recording — with nothing in
    the tensors to tell them apart.

    Recorded rather than assumed, for the reason `TextConfig.recipe` exists: an
    implementation that does not have the strategy named here must refuse the
    profile instead of silently applying its own. Five implementations agreeing
    on the transform is worth nothing if they disagree about which ten seconds
    to feed it.
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

        Checked here rather than discovered per backend. A profile is a file
        that can be copied, mailed and loaded from an untrusted source, and the
        three renderers disagree about what a degenerate one means: torch's
        ``F.normalize`` carries an epsilon and returns finite values for a zero
        vector, while ONNX and CoreML divide by the raw norm and produce 192
        NaNs. One accepted profile, two behaviours, no error anywhere — the
        divergence class this library exists to make impossible, arriving
        through data instead of through code.
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
        # Both ends, not just the floor. A negative id was refused and an
        # oversized one was not, so a profile carrying `prompt_tokens = [9000]`
        # loaded cleanly and then indexed past the end of `nn.Embedding` — an
        # `IndexError` from inside the token generator, or on the ONNX path a
        # read of whatever follows the table. `load()` promises a profile is
        # safe to open from an untrusted source; a bound the renderer relies on
        # has to be checked where that promise is made.
        #
        # The ceilings are the model's, taken from the shipped algorithm rather
        # than repeated here: prompt tokens index the speech codebook below the
        # start-of-speech marker, conditioning tokens the full speech vocabulary.
        from .config import AlgorithmConfig

        limits = AlgorithmConfig()
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

        The generator's conditioning is a pure function of
        ``speaker_embedding`` and ``cond_prompt_tokens`` (the third slot is the
        constant :data:`EMOTION_NEUTRAL`), so two profiles that agree on these
        bytes get the same row. Keyed by content rather than by object
        identity: profiles are frozen but freely copied, and an ``id()`` key
        would silently miss on every copy. A few hundred bytes of hashing per
        call, against a perceiver pass per miss.
        """
        import hashlib

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
            {
                "speaker_embedding": self.speaker_embedding,
                "flow_embedding": self.flow_embedding,
                "prompt_tokens": self.prompt_tokens,
                "prompt_mel": self.prompt_mel,
                "cond_prompt_tokens": self.cond_prompt_tokens,
            },
            str(path),
            metadata={"voice": json.dumps(header)},
        )
        # Owner-only. A voice profile is derived from a recording of a person;
        # anything group- or world-readable has wider reach than the consent
        # that covered the recording.
        path.chmod(0o600)
        return path

    @classmethod
    def load(cls, path: str | Path) -> VoiceProfile:
        """Read a profile written by :meth:`save`."""
        from safetensors import safe_open

        path = Path(path)
        size = path.stat().st_size
        if size > MAX_VOICE_BYTES:
            raise ValueError(
                f"{path.name}: {size} bytes, over the {MAX_VOICE_BYTES} byte "
                "limit for a voice — see MAX_VOICE_BYTES"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with safe_open(str(path), framework="numpy") as f:
            meta: Mapping[str, str] = f.metadata() or {}
            header = json.loads(meta.get("voice", "{}"))
            version = int(header.get("format_version", 0))
            if version != VOICE_FORMAT_VERSION:
                raise ValueError(
                    f"{path.name}: voice format version {version}, "
                    f"this build reads {VOICE_FORMAT_VERSION}"
                )
            return cls(
                name=str(header.get("name", path.stem))[:_MAX_NAME_CHARS],
                speaker_embedding=f.get_tensor("speaker_embedding"),
                flow_embedding=f.get_tensor("flow_embedding"),
                prompt_tokens=f.get_tensor("prompt_tokens"),
                prompt_mel=f.get_tensor("prompt_mel"),
                cond_prompt_tokens=f.get_tensor("cond_prompt_tokens"),
                # Profiles written before 0.1 carry an "emotion" key; it is
                # ignored. The axis is dead on these weights (distillation
                # collapsed it) and the conditioning slot is fed
                # EMOTION_NEUTRAL, the value every profile ever written had.
                source_sample_rate=int(header.get("source_sample_rate", 24_000)),
                source_sha256=digest,
                language=str(header.get("language", "en")),
                # Absent means the profile predates the field, and every one of
                # those was cut from the first ten seconds.
                enrolment=str(header.get("enrolment", ENROLMENT_FIRST_WINDOW)),
            )

    def __repr__(self) -> str:
        return (
            f"VoiceProfile({self.name!r}, {self.language}, "
            f"{len(self.prompt_tokens)} prompt tokens, "
            f"{self.prompt_mel.shape[1]} mel frames, {self.n_bytes / 1024:.0f} KB)"
        )
