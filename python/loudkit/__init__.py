"""loudkit: a small text-to-speech engine that behaves the same everywhere.

See ``docs/design/engine-pipeline.md``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .engine import Engine, Result
from .errors import (
    AudioNotFoundError,
    CancelledError,
    InvalidTokensError,
    LoudkitError,
    NothingToSpeakError,
    ProvenanceError,
    UnsupportedFormatError,
    UnsupportedLanguageError,
    VoiceNotFoundError,
    WindowOverflowError,
)
from .frontend.numbers import NumberGrammarError

# The prompt cut and its five tunables live with the other reference-audio DSP
# and are re-exported here, where `loudkit.enroll` is how a caller meets them.
# Self-aliased because they are not in `__all__` and never were.
from .models.enrollment_audio import CLIP_MIN_SECONDS as CLIP_MIN_SECONDS
from .models.enrollment_audio import END_SILENCE_SECONDS as END_SILENCE_SECONDS
from .models.enrollment_audio import PAUSE_FLOOR_DBFS as PAUSE_FLOOR_DBFS
from .models.enrollment_audio import PAUSE_MIN_SECONDS as PAUSE_MIN_SECONDS
from .models.enrollment_audio import PROMPT_SECONDS as PROMPT_SECONDS
from .models.enrollment_audio import (
    with_prompt_ending_in_silence as with_prompt_ending_in_silence,
)
from .models.timestretch import MAX_SPEED, MIN_SPEED
from .voice import VoiceProfile

if TYPE_CHECKING:
    from .config import AlgorithmConfig, ExecutionConfig
    from .contracts import Waveform

from ._version import __version__

__all__ = [
    "__version__",
    "load",
    "voice",
    "voices",
    "languages",
    "enroll",
    "best_device",
    "Engine",
    "Result",
    "VoiceProfile",
    "MIN_SPEED",
    "MAX_SPEED",
    "LoudkitError",
    "AudioNotFoundError",
    "CancelledError",
    "InvalidTokensError",
    "NothingToSpeakError",
    "NumberGrammarError",
    "ProvenanceError",
    "UnsupportedFormatError",
    "UnsupportedLanguageError",
    "VoiceNotFoundError",
    "WindowOverflowError",
]
# The public surface. ``loudkit.config``, ``loudkit.contracts``, ``loudkit.errors``,
# ``loudkit.postprocess``, ``loudkit.sampler``, ``loudkit.timing`` and
# ``loudkit.provenance`` are supported homes for the rest (COMPATIBILITY.md).


def best_device() -> str:
    """CUDA, then Apple silicon (a split engine: generator on CPU, renderer on GPU),
    then CPU."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is an extra; the dev environment has it
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_BARE_INSTALL = """loudkit is installed without a runtime, so it can read a checkpoint but not
run one. Pick the one for your machine:

  pip install "loudkit[torch,audio]"   # CPU, CUDA or Apple GPU: the usual choice
  pip install "loudkit[onnx,audio]"    # no torch; needs the exported graphs
  pip install "loudkit[coreml,audio]"  # Apple silicon; needs the compiled packages

`audio` is what writes a .wav. Add `hub` to load models by name, `enroll` to
clone a voice, `server` for `loudkit serve`."""

_BARE_ENROLL = """voice cloning needs the enrollment runtime:

  pip install "loudkit[enroll]"

That installs torch, torchaudio and the audio reader used by `loudkit.enroll`."""


def _require_runtime(device: str, *, operation: str = "synthesis") -> None:
    """Name the extra a bare install is missing before a backend fails oddly."""
    import importlib.util

    if operation == "enroll" and device not in ("onnx", "coreml"):
        if importlib.util.find_spec("torch") is None:
            raise ModuleNotFoundError(_BARE_ENROLL)
        return
    base = (device or "").split(":", 1)[0]
    wanted = {"onnx": "onnxruntime", "coreml": "coremltools"}.get(base, "torch")
    guarded = ("", "cpu", "cuda", "mps", "onnx", "coreml")
    if base in guarded and importlib.util.find_spec(wanted) is None:
        raise ModuleNotFoundError(_BARE_INSTALL)


def _require_device(device: str) -> None:
    """Refuse a torch device this machine does not have, before any download.

    Left to torch, the failure comes later and deeper, as an assertion from
    inside a ``.to()`` call; the CLI promises one diagnosis line.
    """
    base, sep, index = device.partition(":")
    if base not in ("cuda", "mps") or device == best_device():
        return  # the machine's own pick is available by construction

    if sep and not index.isdigit():
        # Read as ordinal 0 before, so "cuda:abc" walked through the one check
        # that exists to keep it away from torch. Cheap, so it goes first.
        seen = f"{index!r} is not a device ordinal"
    else:
        import torch

        if base == "mps":
            if torch.backends.mps.is_available():
                return
            seen = "this torch has no MPS device"
        else:
            count = torch.cuda.device_count()
            if (int(index) if index else 0) < count:
                return
            seen = (
                f"this torch sees {count} CUDA device(s)"
                if count
                else "this torch was built without CUDA, or no CUDA device is visible"
            )
    raise ValueError(
        f"device {device!r} is not available on this machine: {seen}. "
        f"The best device here is {best_device()!r}; `loudkit doctor` lists them."
    )


def voice(ref: str, *, repo: str | None = None, revision: str | None = None) -> VoiceProfile:
    """A voice by path, or by name from a release. Fetches the one file, not the release."""
    from .hub import resolve_voice

    return VoiceProfile.load(resolve_voice(ref, repo=repo, revision=revision))


def voices(*, repo: str | None = None, revision: str | None = None) -> tuple[str, ...]:
    """The names :func:`voice` accepts from a release, sorted. ``repo`` is required;
    :meth:`Engine.voices` answers for an engine's own release."""
    if repo is None:
        raise ValueError(
            "loudkit.voices() needs a repo: pass a Hugging Face repo id such as "
            "repo='loudreader/loudr-1', or a directory holding an unpacked "
            "release."
        )
    from .hub import list_voices

    return list_voices(repo=repo, revision=revision)


def languages() -> tuple[str, ...]:
    """The language ids this build can read text in, sorted.

    Reading is the text funnel, not the roster: numbers, dates and letters
    are folded into words for twelve languages, and this release ships
    voices for ten of them. Finnish and Norwegian text reads correctly and
    has no voice of its own; :func:`voices` is what can speak.
    """
    from .frontend.numbers import supported_languages

    return supported_languages()


def _read_audio(path: str) -> tuple[Waveform, int]:
    """Samples at their native rate, channels averaged. The enroller resamples."""
    from pathlib import Path

    import numpy as np

    if not Path(path).is_file():
        raise AudioNotFoundError(f"audio not found: {path}")
    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ModuleNotFoundError(
            "reading an audio file needs soundfile: "
            "pip install soundfile, or pass samples directly"
        ) from exc
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.mean(axis=1, dtype=np.float32)), int(rate)


def enroll(
    audio: str | Waveform,
    checkpoint: str,
    *,
    name: str = "",
    language: str = "en",
    device: str = "cpu",
    revision: str | None = None,
    voice_encoder_weights: str | None = None,
    sample_rate: int = 24_000,
    end_in_silence: bool = False,
) -> VoiceProfile:
    """Clone a voice from a recording.

    See ``docs/design/engine-pipeline.md``.

    ``end_in_silence`` cuts the clip at its last pause before the prompt limit and pads
    silence after it (:func:`with_prompt_ending_in_silence`); ``loudkit clone`` asks for it,
    the library default leaves the clip as given.
    """
    import numpy as np

    _require_runtime(device, operation="enroll")
    _require_device(device)

    from .hub import is_repo_id, resolve_enrollment_checkpoint, resolve_voice_encoder

    if isinstance(audio, str):
        samples, sample_rate = _read_audio(audio)
    else:
        samples = np.asarray(audio, dtype=np.float32)
    from .voice import ENROLMENT_FIRST_WINDOW, ENROLMENT_PAUSE_CUT

    if end_in_silence:
        from .models.enrollment_audio import validate_reference_audio

        # Validate the original: padding must not rescue a too-short clip, and
        # trimming must not hide invalid samples or an oversized recording.
        validate_reference_audio(samples, sample_rate)
        samples = with_prompt_ending_in_silence(samples, sample_rate)

    if device in ("onnx", "coreml"):
        if voice_encoder_weights is not None:
            raise ValueError(
                "graph enrollment uses its exported voice encoder; "
                "voice_encoder_weights requires torch"
            )
        from .backends.graph_enroll import build_graph_enroller

        graph_enroller = build_graph_enroller(checkpoint, device=device, revision=revision)
        profile = graph_enroller.enroll(samples, sample_rate, name=name)
    else:
        from .backends.torch_backend import build_torch_enroller

        # The enrollment tensors and the voice encoder are not in the synthesis
        # checkpoint; the resolver finds both for a repo id, a directory or a file.
        resolved = resolve_enrollment_checkpoint(checkpoint, revision=revision)
        ve = voice_encoder_weights or str(
            resolve_voice_encoder(
                checkpoint if is_repo_id(checkpoint) else str(resolved), revision=revision
            )
        )
        enroller = build_torch_enroller(str(resolved), device=device, voice_encoder_weights=ve)
        profile = enroller.enroll(samples, sample_rate, name=name)

    return replace(
        profile,
        language=language,
        enrolment=ENROLMENT_PAUSE_CUT if end_in_silence else ENROLMENT_FIRST_WINDOW,
    )


def load(
    checkpoint: str,
    *,
    device: str | None = None,
    execution: ExecutionConfig | None = None,
    algorithm: AlgorithmConfig | None = None,
    revision: str | None = None,
) -> Engine:
    """Load a checkpoint and return a ready engine.

    See ``docs/design/engine-pipeline.md``.
    """
    from .backends import _agreed_device, require_backend
    from .hub import backend_for_device, resolve_checkpoint

    # Device first: a typo should not cost a download.
    resolved_device = _agreed_device(device, execution) or best_device()
    _require_runtime(resolved_device)
    _require_device(resolved_device)
    require_backend(resolved_device)
    backend = backend_for_device(resolved_device)
    return Engine.from_checkpoint(
        str(resolve_checkpoint(checkpoint, revision=revision, backend=backend)),
        device=resolved_device,
        execution=execution,
        algorithm=algorithm,
    )
