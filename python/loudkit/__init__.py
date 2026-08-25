"""loudkit — a small text-to-speech engine that behaves the same everywhere.

Two lines to speech::

    import loudkit as lk
    engine = lk.load("loudr-1.safetensors")           # picks the best device
    engine.synthesize("Hello there.", voice, seed=7).save("out.wav")

The engine has two stages. A **token generator** writes discrete speech tokens
at 25 Hz, autoregressively. A **renderer** turns those tokens into a waveform, in
one parallel pass. They have opposite shapes, and every backend we have measured
treats them differently — on Apple silicon the generator is faster on the CPU
and the renderer is faster on the GPU — so the library lets them live on
different devices and defaults to whichever split is quickest.

What the library promises, and does not
---------------------------------------

*Deterministic by build.* The same text, voice and seed give a bit-identical
waveform every time, on a given build and device.

*One sampling law everywhere.* Every backend implements the same
counter-based sampler, so the same seed makes the same token decisions on CPU,
CUDA, MPS or ONNX. This is not free: ``torch.multinomial`` returns different
samples for an identical probability vector on x86 and arm64, so the library
ships its own.

*Not* bit-identical across backends or across releases. Different hardware sums
floating-point numbers in different orders, and engine changes re-base the
goldens under a versioned contract. What is held constant is the sampling law
and the voice.

How it is organised
-------------------

Every decision belongs to one of two layers, and the split is enforced rather
than documented:

- :class:`~loudkit.config.AlgorithmConfig` — what is computed. Guidance mode,
  integration grid, sampling law, windowing. Identical on every backend; the
  engine refuses to run components that disagree.
- :class:`~loudkit.config.ExecutionConfig` — how fast. Precision, kernels,
  device placement, graph capture. Free to differ.

The split is enforced because a misapplied execution setting — guidance applied
twice, for instance — changes the audio while both outputs stay plausible.
"""

from __future__ import annotations

from .config import (
    DEFAULT_ALGORITHM,
    AlgorithmConfig,
    ExecutionConfig,
    ExecutionOverrides,
    SamplingConfig,
    WindowConfig,
)
from .contracts import (
    Mel,
    MelDecoder,
    Sampler,
    SpeechTokens,
    TextFrontend,
    TokenGenerator,
    Vocoder,
    VoiceEnroller,
    Waveform,
)
from .engine import Engine, Result, StageTimings
from .errors import (
    InvalidTokensError,
    LoudkitError,
    UnsupportedLanguageError,
    VoiceNotFoundError,
    WindowOverflowError,
)
from .frontend.numbers import NumberGrammarError
from .models.timestretch import MAX_SPEED, MIN_SPEED
from .provenance import read_provenance, verify_provenance
from .sampler import SAMPLER_VERSION, LRSamplerV1
from .timing import ChunkTiming, WordTiming
from .voice import VoiceProfile

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load",
    "voice",
    "read_provenance",
    "verify_provenance",
    "voices",
    "languages",
    "enroll",
    "best_device",
    # configuration
    "AlgorithmConfig",
    "ExecutionConfig",
    "ExecutionOverrides",
    "SamplingConfig",
    "WindowConfig",
    "DEFAULT_ALGORITHM",
    # engine
    "Engine",
    "Result",
    "StageTimings",
    "ChunkTiming",
    "WordTiming",
    # the speed bounds, because a UI drawing a speed slider needs them and
    # should not have to import from a models submodule to find out. The
    # stretcher itself stays internal: it is how the engine renders, not
    # something a caller composes with.
    "MIN_SPEED",
    "MAX_SPEED",
    "VoiceProfile",
    # sampling
    "LRSamplerV1",
    "SAMPLER_VERSION",
    # errors — every one also inherits the builtin it replaces, so existing
    # `except ValueError` / `except FileNotFoundError` handlers keep working
    "LoudkitError",
    "InvalidTokensError",
    "UnsupportedLanguageError",
    "VoiceNotFoundError",
    "WindowOverflowError",
    "NumberGrammarError",
    # contracts, for anyone writing a backend
    "TextFrontend",
    "TokenGenerator",
    "MelDecoder",
    "Vocoder",
    "Sampler",
    "VoiceEnroller",
    "SpeechTokens",
    "Mel",
    "Waveform",
]


def best_device() -> str:
    """The fastest device available here.

    Prefers CUDA, then Apple silicon, then CPU.

    On Apple silicon this returns ``"mps"``, which selects a **split** engine:
    the token generator stays on the CPU, where it is 1.7x faster than on the
    GPU because an autoregressive step at batch one is a few hundred tiny
    dispatches, and the renderer moves to the GPU, where it is 2.6x faster
    because it is one large parallel pass. Measured 1.35x -> 1.54x end to end.

    Override either stage with ``ExecutionConfig.generator_device`` and
    ``renderer_device`` if your hardware disagrees.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_BARE_INSTALL = """loudkit is installed without a runtime, so it can read a checkpoint but not
run one. Pick the one for your machine:

  pip install "loudkit[torch,audio]"   # CPU, CUDA or Apple GPU — the usual choice
  pip install "loudkit[onnx,audio]"    # no torch; needs the exported graphs

`audio` is what writes a .wav. Add `hub` to load models by name, `enroll` to
clone a voice, `server` for `loudkit serve`."""

_BARE_ENROLL = """voice cloning needs the enrollment runtime:

  pip install "loudkit[enroll]"

That installs torch, torchaudio and the audio reader used by `loudkit.enroll`."""


def _require_runtime(device: str, *, operation: str = "synthesis") -> None:
    """Say what a bare `pip install loudkit` is missing, before it fails oddly.

    The core package deliberately has no runtime — a torch install is 2 GB and
    an ONNX-only user should not carry it. But that means `pip install loudkit`
    followed by `load()` would otherwise end in `No module named torch`,
    raised from inside a backend — which reads as a broken package rather than
    an incomplete install.
    """
    import importlib.util

    # Enrollment always uses the torch enroller. A caller spelling
    # ``device="onnx"`` or even an invalid device must not let a raw torch
    # import escape first. Device validation belongs to the installed backend;
    # the public seam owns the missing dependency for every enrollment call.
    if operation == "enroll":
        if importlib.util.find_spec("torch") is None:
            raise ModuleNotFoundError(_BARE_ENROLL)
        return

    base = (device or "").split(":", 1)[0]
    wanted = "onnxruntime" if base == "onnx" else "torch"
    if base in ("", "cpu", "cuda", "mps", "onnx") and importlib.util.find_spec(wanted) is None:
        raise ModuleNotFoundError(_BARE_INSTALL)


def voice(ref: str, *, repo: str | None = None, revision: str | None = None) -> VoiceProfile:
    """A voice, by path or by name from a released repo.

    The partner of :func:`load`, so the two lines that get you speech are the
    same shape::

        engine = lk.load("loudreader/loudr-1")
        narrator = lk.voice("kathleen", repo="loudreader/loudr-1")

    Fetches the one ~150 KB file rather than the whole release, because trying a
    second voice should not re-download a gigabyte.
    """
    from .hub import resolve_voice

    return VoiceProfile.load(resolve_voice(ref, repo=repo, revision=revision))


def voices(*, repo: str | None = None, revision: str | None = None) -> tuple[str, ...]:
    """The names :func:`voice` will accept from a release, sorted.

    The question every new user asks before the second line of code::

        >>> lk.voices(repo="loudreader/loudr-1")
        ('kathleen', 'joe', 'gosia')
        >>> narrator = lk.voice("kathleen", repo="loudreader/loudr-1")

    Reads the repo's file listing rather than downloading anything, so choosing
    between voices costs one request instead of one file each. ``repo`` may also
    be a directory holding an unpacked release, in which case nothing leaves the
    machine.

    There is no default repo, exactly as there is none for :func:`voice`: the
    only honest answer to "which voices exist" is "in which release", and a
    hardcoded repo id would be a promise this package cannot keep across
    versions of the model.

    Raises:
        ValueError: when ``repo`` is omitted.
    """
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

    A thin wrapper over :func:`loudkit.frontend.numbers.supported_languages`, which reads
    the grammars that actually ship rather than a list written beside them. One
    authority: a language whose grammar is added or removed changes this answer
    without anyone remembering to edit it.
    """
    from .frontend.numbers import supported_languages

    return supported_languages()


def enroll(
    audio: str | Waveform,
    checkpoint: str,
    *,
    name: str = "",
    language: str = "en",
    device: str = "cpu",
    revision: str | None = None,
    voice_encoder_weights: str | None = None,
) -> VoiceProfile:
    """Clone a voice from a recording. The third of the three lines.

    ::

        engine = lk.load("loudreader/loudr-1")
        mine = lk.enroll("me.wav", "loudreader/loudr-1", name="mine")
        engine.synthesize_long("Now in my own voice.", mine, seed=7)

    Cloning is the feature people come for; this call resolves the checkpoint
    and builds the enroller so the first clone needs one concept, next to one
    for :func:`load` and one for :func:`voice`.

    The enroller is built per call rather than cached. It loads about 40% of the
    checkpoint that synthesis never touches, so holding it alive for a caller
    who enrolled once would cost them memory for the rest of the process; a
    caller enrolling in bulk should build one with
    :func:`~loudkit.backends.torch_backend.build_torch_enroller` and keep it.

    **Consent is yours to obtain**, and it is not a technical question — see
    ``RESPONSIBLE_USE.md``.

    Args:
        audio: a path to a readable audio file, or mono samples in ``[-1, 1]``.
            Ten seconds is plenty; longer input is truncated.
        checkpoint: the same path or repo id :func:`load` takes.
        name: what to call the voice. Carried in the profile.
        device: torch device for the enrollment models.
        revision: branch, tag or commit, when ``checkpoint`` is a repo id.
        voice_encoder_weights: ``ve.safetensors``, the 256-d utterance voice
            encoder. Resolved from the release by default — it sits at the
            release root, beside the checkpoint — so this is only for pointing
            at one somewhere else.

    Returns:
        A :class:`~loudkit.voice.VoiceProfile` — a few hundred kilobytes of
        tensors. ``.save(path)`` writes it; :func:`voice` reads it back.

    Raises:
        FileNotFoundError: the release ships no voice encoder, so it can
            synthesize but not clone.
    """
    import numpy as np

    # Check before importing the torch backend. A bare install should explain
    # which extra enables cloning, not leak a ModuleNotFoundError raised from
    # an implementation module the caller never named.
    _require_runtime(device, operation="enroll")

    from .backends.torch_backend import build_torch_enroller
    from .hub import is_repo_id, resolve_enrollment_checkpoint, resolve_voice_encoder

    if isinstance(audio, str):
        # Read at the enroller's own working rate rather than resampling twice:
        # it wants 24 kHz for the prompt mel and derives 16 kHz itself.
        try:
            import librosa
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise ModuleNotFoundError(
                "reading an audio file needs the 'enroll' extra: "
                "pip install 'loudkit[enroll]' — or pass samples directly"
            ) from exc
        # The rate is asserted rather than read back: librosa types its
        # return as `int | float`, and the enroller's contract is an int.
        samples, _ = librosa.load(audio, sr=24_000, mono=True)
    else:
        samples = np.asarray(audio, dtype=np.float32)

    # Neither of the two files enrollment reads is the one `load()` reads.
    #
    # The speech tokenizer and the speaker encoder live in the release's
    # *enrollment* artefact, which a synthesis fetch does not download at all;
    # handing the synthesis checkpoint to the enroller, as this used to,
    # fails inside it about `s3gen.speaker_encoder` — a tensor name no caller
    # of the public API has heard of. And the utterance voice encoder is in
    # neither artefact: it ships beside them, and is resolved here because a
    # caller of the public API has no `voice_encoder_weights` argument to pass
    # and cannot resolve the file by hand.
    #
    # A pre-split checkpoint is one file that holds all of it, and the
    # enrollment resolver answers with that file; nothing here has to know
    # which kind of release it was given.
    resolved = resolve_enrollment_checkpoint(checkpoint, revision=revision)
    ve = voice_encoder_weights or str(
        resolve_voice_encoder(
            checkpoint if is_repo_id(checkpoint) else str(resolved), revision=revision
        )
    )
    enroller = build_torch_enroller(str(resolved), device=device, voice_encoder_weights=ve)
    profile = enroller.enroll(samples, 24_000, name=name)
    # The language the profile carries.
    #
    # `VoiceProfile.language` exists, is documented as the voice's own, and is
    # what `_resolve_language` falls back to when a caller omits one — and
    # `enroll` had no way to write it, so every cloned voice claimed English by
    # construction: every cloned non-English voice would read its text
    # through the English funnel.
    #
    # Exercised only when a profile loads, which requires the checkpoint.
    from dataclasses import replace as _replace

    return _replace(profile, language=language)


def load(
    checkpoint: str,
    *,
    device: str | None = None,
    execution: ExecutionConfig | ExecutionOverrides | None = None,
    algorithm: AlgorithmConfig | None = None,
    revision: str | None = None,
) -> Engine:
    """Load a checkpoint and return a ready engine.

    Takes a path or a name::

        engine = lk.load("loudreader/loudr-1")   # downloads, caches
        engine = lk.load("./loudr-1.safetensors")  # exactly this file

    A path that exists always wins, so nothing reaches for the network because a
    directory merely shares a repo's name.

    Args:
        checkpoint: a packed ``.safetensors``, a directory holding one, or a
            Hugging Face repo id such as ``loudreader/loudr-1``.
        device: ``"cpu"``, ``"cuda"``, ``"mps"``, or ``None`` to choose.
        execution: an :class:`~loudkit.config.ExecutionOverrides` to change
            named fields and inherit the manifest's defaults for the rest, or a
            full :class:`~loudkit.config.ExecutionConfig` to specify everything.
        algorithm: override the checkpoint's algorithm. Deliberate deviations
            only — the fingerprint will differ from the shipping one, and
            conformance will report it.
        revision: branch, tag or commit, when ``checkpoint`` is a repo id. Pin
            it for anything reproducible: a moving ``main`` is a moving model.

    Returns:
        An :class:`~loudkit.engine.Engine`.
    """
    from .backends import require_backend
    from .hub import backend_for_device, resolve_checkpoint

    _require_runtime(device or "")

    # Device first: it is a dictionary lookup, and resolving `checkpoint` may
    # mean downloading 747 MB. A typo should not cost that.
    resolved_device = device or best_device()
    require_backend(resolved_device)

    # The resolver fetches by backend, and the backend is decided by the
    # device: `onnx` and `coreml` name backends whose exported graphs live
    # beside the checkpoint, so a caller asking for them must get them —
    # a resolver that fetched the torch set answered `device="onnx"` with a
    # snapshot the ONNX backend cannot run. Everything else is torch.
    backend = backend_for_device(resolved_device)

    return Engine.from_checkpoint(
        str(resolve_checkpoint(checkpoint, revision=revision, backend=backend)),
        device=resolved_device,
        execution=execution,
        algorithm=algorithm,
    )
