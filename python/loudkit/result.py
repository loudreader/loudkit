"""What a synthesis hands back: audio, tokens, mel, timings, and how to save it."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import Mel, SpeechTokens, Waveform
from .postprocess import Inspection
from .timing import ChunkTiming

__all__ = ["Provenance", "Result", "StageTimings"]


@dataclass(frozen=True, slots=True)
class StageTimings:
    """Wall time per stage, in seconds."""

    tokens: float
    mel: float
    audio: float

    @property
    def total(self) -> float:
        return self.tokens + self.mel + self.audio

    def rtf(self, audio_seconds: float) -> float:
        """Real-time factor: seconds of audio produced per second of work."""
        return audio_seconds / self.total if self.total > 0 else float("inf")

    def describe(self, audio_seconds: float) -> str:
        return (
            f"tokens {self.tokens:.3f}s  mel {self.mel:.3f}s  audio {self.audio:.3f}s  "
            f"-> {audio_seconds:.2f}s @ RTF {self.rtf(audio_seconds):.2f}x"
        )


@dataclass(frozen=True, slots=True)
class Provenance:
    """What names the render for the C2PA manifest. Empty strings mean "not known here"."""

    algorithm_fingerprint: str = ""
    recipe_version: str = ""
    voice: str = ""
    language: str = ""
    voice_sha256: str = ""
    checkpoint_sha256: str = ""
    backend: str = ""
    execution: str = ""


@dataclass(frozen=True, slots=True)
class Result:
    """A synthesis with its intermediates.

    The tokens and the mel are kept because they localise a disagreement
    between two backends to one stage.
    """

    audio: Waveform
    tokens: SpeechTokens
    mel: Mel
    seed: int
    sample_rate: int
    timings: StageTimings

    hit_token_cap: bool = False
    """Generation stopped at the cap rather than at a stop token."""

    inspections: tuple[Inspection, ...] = ()
    """What the postprocess detectors concluded, one entry per chunk."""

    speed: float = 1.0
    """The time-stretch applied. ``1.0`` is the vocoder's own bytes."""

    chunks: tuple[ChunkTiming, ...] = ()
    """Where each chunk lands in ``audio``, adjacent and exact; the word times
    inside are estimates (:mod:`loudkit.timing`)."""

    provenance: Provenance = Provenance()
    """What :meth:`save` writes into the manifest."""

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def suspect(self) -> bool:
        """A chunk was impossibly long for its text and nothing could say where to cut."""
        return any(i.suspect for i in self.inspections)

    def save(
        self,
        path: str,
        *,
        voice: str = "",
        language: str = "",
        include_provenance: bool = True,
    ) -> None:
        """Write a 16-bit WAV, with a C2PA claim-only manifest by default.

        ``voice`` and ``language`` override the labels the result carries.
        """
        from ._version import package_version
        from .provenance import write_wav

        version = package_version()
        p = self.provenance
        write_wav(
            path,
            self.audio,
            self.sample_rate,
            manifest=include_provenance,
            algorithm_fingerprint=p.algorithm_fingerprint,
            recipe_version=p.recipe_version,
            seed=self.seed,
            voice=voice or p.voice,
            language=language or p.language,
            text=" ".join(c.text for c in self.chunks),
            speed=self.speed,
            version=version,
            voice_sha256=p.voice_sha256,
            checkpoint_sha256=p.checkpoint_sha256,
            backend=p.backend,
            execution=p.execution,
        )

    def __repr__(self) -> str:
        cap = ", HIT CAP" if self.hit_token_cap else ""
        trimmed = sorted({i.reason for i in self.inspections if i.cut})
        cut = f", cut={'+'.join(trimmed)}" if trimmed else ""
        flag = ", SUSPECT" if self.suspect else ""
        rate = f", speed={self.speed:g}x" if self.speed != 1.0 else ""
        return (
            f"Result({self.duration:.2f}s, {len(self.tokens)} tokens, seed={self.seed}, "
            f"RTF {self.timings.rtf(self.duration):.2f}x{cap}{rate}{cut}{flag})"
        )
