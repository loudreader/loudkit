"""Fixtures every test in this suite gets, whether it asks or not, and the
helpers the whole suite shares: how a script outside the package is imported,
and the weight-free component set the engine is faked with.

The stand-ins were four copies of the voice and three of each engine component,
byte for byte, with three modules importing a fourth module's privates to avoid
a fifth copy. One definition here means a change to a fake is a change to every
suite that uses it, rather than to whichever copy happened to be in front of
the reader.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

# Set before numba is imported anywhere. librosa's mel/STFT helpers are
# numba-cached functions, and numba's default cache location is next to the
# *source* — inside site-packages. In any environment where site-packages is
# not writable (CI images, system installs, pip install --user with a read-only
# venv) the first enrollment test then fails with a cache write error that
# reads like a broken install. Pointing the cache at a temp directory costs a
# recompile per session and removes the failure class; honoured only when the
# caller has not already chosen a location.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba-cache"))

# Below that line, not with the imports above it: the guarantee it makes is
# "before numba is imported anywhere", and importing the library first would
# rest that guarantee on nothing in loudkit's import graph ever reaching
# librosa, which is not a promise this file can keep.
import numpy as np
from numpy.typing import NDArray

import loudkit as _loudkit
from loudkit.config import AlgorithmConfig
from loudkit.contracts import Mel, Sampler, SpeechTokens, Waveform
from loudkit.engine import Engine
from loudkit.synthesis import VoiceLibrary
from loudkit.voice import VoiceProfile

# The suite must read the tree it lives in. `loudkit` is normally installed
# editable, and an editable install points at whichever checkout was installed,
# so a worktree that runs `pytest` without `PYTHONPATH` measures a different
# tree and reports on code nobody is editing: green where the tree under test
# is red, and red where it is green.
#
# An installed distribution is not a checkout and is not the mistake: the
# packaging job installs the built wheel and runs this suite against it, which
# is the only way to find out whether the wheel works. Only another `python/`
# tree is the wrong answer this refuses.
_want = Path(__file__).resolve().parent.parent / "python" / "loudkit"
_got = Path(_loudkit.__file__).resolve().parent
if _want.is_dir() and _got != _want and _got.parent.name == "python":
    raise RuntimeError(
        f"this suite imports loudkit from {_got}, not from {_want}, so it would "
        f"report on a different checkout. Run it with PYTHONPATH={_want.parent}"
    )


def _voice() -> VoiceProfile:
    """One profile with the shapes the engine expects and no acoustics.

    The older name for `fake_voice`, kept because six suites import it. One
    body, not two: the two definitions here were identical field for field
    (`VoiceProfile.language` already defaults to "en"), which is one edit away
    from a fake that means different things in different suites, and this file
    exists to stop exactly that.
    """
    return fake_voice()


def _voices(tmp_path) -> VoiceLibrary:
    """A voice library with one voice named 'fake', written to disk."""
    _voice().save(tmp_path / "fake.safetensors")
    return VoiceLibrary(tmp_path)


def _stub_checkpoint(tmp_path) -> Path:
    """A checkpoint file that exists but is never read.

    These tests inject an engine, so nothing loads the weights -- but `serve`
    hands its checkpoint argument to the hub whatever shape it has, and the hub
    will not hand back a path to nothing. It sits in a directory of its own so a
    test that also passes ``voices=tmp_path`` does not find it offered as a
    voice.
    """
    holder = tmp_path / "checkpoint"
    holder.mkdir(exist_ok=True)
    path = holder / "ckpt.safetensors"
    path.write_bytes(b"x")
    return path


class _SplitFrontend:
    """Splits on full stops so streaming has more than one chunk."""

    def encode(self, text: str, language: str = "en") -> np.ndarray:
        words = text.replace(".", " .").split()
        return np.arange(len(words), dtype=np.int64)


@pytest.fixture(autouse=True)
def _hold_the_provenance_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze the manifest `when`, so byte-equality assertions mean what they say.

    The suite's central claim is that every transport returns byte for byte
    what calling the library returns, asserted against `render_bytes` for HTTP
    (`test_the_route_returns_the_bytes_the_library_returns`), gRPC
    (`test_grpc_returns_the_same_bytes_as_the_library`) and MCP
    (`test_the_tool_returns_the_bytes_the_library_returns`), and for the
    OpenAI-compatible route through the native one it is pinned to. A rendered
    WAV carries a provenance chunk, and the chunk carries a creation timestamp,
    which is the one value in the file that is not a function of the input.

    So those assertions were time-dependent. Two identical renders that
    straddled a second boundary differed by one digit at some offset deep in
    the trailer, and the test reported a parity failure in the synthesis path.
    Observed once in four full runs of the suite: `At index 11573 diff: b'9' !=
    b'8'`, on audio that was identical.

    Autouse, because the point is that no test has to remember. What is being
    checked is that two paths produce the same audio; the wall clock is not
    part of that and never was.
    """
    monkeypatch.setattr("loudkit.provenance._stamped_now", lambda: "2026-01-01T00:00:00+00:00")


REPO = Path(__file__).resolve().parents[1]


def assert_amplitude(
    name: str, band_db: float, audio: NDArray[np.float32], reference: NDArray[np.float32]
) -> None:
    """The two amplitude facts a correlation cannot carry.

    Pearson correlation subtracts the mean and divides by the deviation, so a
    render at half volume, at twenty times volume, or with a DC offset
    correlates perfectly with the reference it does not sound like. Level is
    what that normalisation throws away, so level is what gets its own gate:
    the RMS ratio against the reference, in dB, inside ``band_db``.

    Peak is the second fact and needs no band, because
    :data:`loudkit.contracts.Waveform` declares the range and the int16
    quantiser clips to it, which is a defect no correlation reports.
    """
    got = np.asarray(audio, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    assert got.shape == ref.shape, f"{name}: {got.shape} against {ref.shape}"
    rms = float(np.sqrt((got**2).mean()))
    rms_ref = float(np.sqrt((ref**2).mean()))
    assert rms > 0.0, f"{name}: rendered silence"
    level = 20.0 * np.log10(rms / rms_ref)
    assert abs(level) <= band_db, (
        f"{name}: level {level:+.4f} dB against the reference, outside the "
        f"{band_db} dB band; correlation cannot see this"
    )
    peak = float(np.abs(got).max())
    assert peak <= 1.0, f"{name}: peak {peak:.4f} is outside the declared [-1, 1] waveform"


_SCRIPT_ROOTS = ("tools", "research")
_scripts: dict[str, ModuleType] = {}


def tool(name: str) -> ModuleType:
    """Import ``tools/<name>.py`` or ``research/<name>.py`` and remember it.

    Neither directory is a package, so this is the one way the suite reaches
    them.

    The path is inserted for the length of the import and removed again,
    because it has to be: ``research/bench_batch.py`` and
    ``research/bench_render.py`` do a bare ``import _bench``, and
    ``tools/export_coreml.py`` imports three of its siblings the same way, so a
    by-path import that never touches ``sys.path`` cannot load them at all.
    What is removed is the entry this function added, by value: a ``pop(0)``
    would take the *script's* own insert instead and leave this one behind.

    Cached, because the same script is asked for by a dozen tests and executing
    a module twice gives two sets of classes that fail ``isinstance``.
    """
    if name in _scripts:
        return _scripts[name]
    home = next(
        (
            d
            for d in _SCRIPT_ROOTS
            # `tools/release/` is a package rather than a script; same problem,
            # same answer.
            if (REPO / d / f"{name}.py").exists() or (REPO / d / name / "__init__.py").exists()
        ),
        None,
    )
    assert home is not None, f"no {name} under tools/ or research/"
    root = str(REPO / home)
    sys.path.insert(0, root)
    try:
        module = importlib.import_module(name)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(root)
    _scripts[name] = module
    return module


# ---------------------------------------------------------------- the fakes
#
# The weight-free component set, in one place. It was written out verbatim in
# five test modules under two spellings, and eight more modules imported it
# *from another test module*, which made test_server.py and test_engine.py
# libraries that the gRPC, limits, resolver, timing and reader suites could not
# run without.
#
# The recording variants are the ones kept: `calls`, `seeds` and `languages`
# are what test_engine asserts on, and a list nobody reads costs nothing.


def fake_voice(language: str = "en") -> VoiceProfile:
    """The six-field profile every weight-free test speaks with."""
    return VoiceProfile(
        name="fake",
        speaker_embedding=np.full(256, 0.0625, np.float32),
        flow_embedding=np.full(192, 0.0625, np.float32),
        prompt_tokens=np.zeros(8, np.int64),
        prompt_mel=np.zeros((80, 16), np.float32),
        cond_prompt_tokens=np.zeros(8, np.int64),
        language=language,
    )


class FakeFrontend:
    """A token per word. Records the language it was asked for; that is the
    whole assertion in ``TestLanguageComesFromTheVoice``."""

    def __init__(self) -> None:
        self.languages: list[str] = []

    def encode(self, text: str, language: str = "en") -> NDArray[np.int64]:
        self.languages.append(language)
        return np.arange(len(text.split()), dtype=np.int64)


class FakeGenerator:
    """Emits a token per input word, then stops. Records what it was given."""

    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        text_tokens: NDArray[np.int64],
        voice: VoiceProfile,
        *,
        sampler: Sampler,
        max_new_tokens: int | None = None,
        prefix: SpeechTokens = (),
        should_cancel=None,
    ) -> SpeechTokens:
        self.calls.append({"n_text": len(text_tokens), "prefix": list(prefix)})
        n = max(1, len(text_tokens))
        return [*range(n), self.config.stop_speech_token]

    def teacher_forced_logits(
        self, text_tokens: NDArray[np.int64], voice: VoiceProfile, forced: SpeechTokens
    ) -> NDArray[np.float32]:
        return np.zeros((len(forced) + 1, self.config.speech_vocab_size), np.float32)


class FakeMelDecoder:
    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config
        self.seeds: list[int] = []

    def decode(self, tokens: SpeechTokens, voice: VoiceProfile, *, seed: int) -> Mel:
        self.seeds.append(seed)
        return np.full((80, max(1, len(tokens)) * 2), float(seed % 97), np.float32)


class FakeVocoder:
    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config
        self.seeds: list[int] = []

    def synthesize(self, mel: Mel, voice: VoiceProfile, *, seed: int) -> Waveform:
        self.seeds.append(seed)
        return np.zeros(mel.shape[1] * 256, np.float32)


def fake_engine(
    algorithm: AlgorithmConfig | None = None,
    *,
    frontend: type = FakeFrontend,
    generator: type = FakeGenerator,
) -> Engine:
    """An engine of fakes.

    ``frontend`` and ``generator`` are classes rather than instances, because a
    test that asserts on a component's records needs the one the engine holds,
    and the engine is what builds it.
    """
    algo = algorithm or AlgorithmConfig()
    return Engine(
        frontend=frontend(),
        token_generator=generator(algo),
        mel_decoder=FakeMelDecoder(algo),
        vocoder=FakeVocoder(algo),
        algorithm=algo,
    )


def chunking_engine() -> Engine:
    """The transport engine: fakes, plus a chunking config that streams.

    ``max_tokens=2`` makes ``_SplitFrontend`` cut "one. two. three." into
    several chunks, so a streaming route has more than one thing to deliver;
    ``prefix_tokens=0`` keeps that from tripping the ``prefix < max_tokens``
    check. Every door -- HTTP, gRPC, the request caps -- wants exactly this
    engine, and the two that did not define it reached into ``test_server``
    for it, which made a test module a library for two others.
    """
    from dataclasses import replace

    chunking = replace(AlgorithmConfig().chunking, max_tokens=2, prefix_tokens=0)
    algo = AlgorithmConfig().with_(chunking=chunking)
    return Engine(
        frontend=_SplitFrontend(),
        token_generator=FakeGenerator(algo),
        mel_decoder=FakeMelDecoder(algo),
        vocoder=FakeVocoder(algo),
        algorithm=algo,
    )
