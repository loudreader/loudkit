"""What ``VoiceLibrary`` keeps between requests, and when it stops trusting it.

Every transport resolves a voice by name on the way into a synthesis, so the
one thing a cache here must never do is answer for a file that has changed or
gone. These tests pin both halves: the parse happens once for an unchanged
file, and the stamp (mtime, size) is what makes that safe.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest

from loudkit import synthesis
from loudkit.config import AlgorithmConfig, ChunkConfig
from loudkit.errors import VoiceNotFoundError
from loudkit.synthesis import VoiceLibrary
from loudkit.voice import VoiceProfile
from loudkit.window import carry_from

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from loudkit.engine import Engine


def _voice(name: str = "fake", mel_frames: int = 16) -> VoiceProfile:
    """A profile small enough to write in a test, sized by ``mel_frames``."""
    return VoiceProfile(
        name=name,
        speaker_embedding=np.full(256, 0.0625, np.float32),
        flow_embedding=np.full(192, 0.0625, np.float32),
        prompt_tokens=np.zeros(8, np.int64),
        prompt_mel=np.zeros((80, mel_frames), np.float32),
        cond_prompt_tokens=np.zeros(8, np.int64),
    )


@pytest.fixture
def parses(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every path ``VoiceProfile.load`` is asked to read, in order."""
    seen: list[Path] = []
    real = VoiceProfile.load

    def counting(path: str | Path) -> VoiceProfile:
        seen.append(Path(path))
        return real(path)

    monkeypatch.setattr(VoiceProfile, "load", staticmethod(counting))
    return seen


def test_unchanged_voice_is_parsed_once(tmp_path: Path, parses: list[Path]) -> None:
    """The second request for the same file gets the profile the first parsed.

    A profile is a whole-file read, a SHA-256 over those bytes and a
    safetensors parse. Doing it per request is the defect; the identity check
    is what proves nothing re-read it.
    """
    _voice().save(tmp_path / "fake.safetensors")
    library = VoiceLibrary(tmp_path)

    first = library.load("fake")
    second = library.load("fake")

    assert first is second
    assert len(parses) == 1


def test_rewriting_a_voice_invalidates_it(tmp_path: Path, parses: list[Path]) -> None:
    """Re-enrolment under the same name must reach the next request.

    The stamp is (mtime_ns, size): the nanoseconds catch an overwrite inside
    one second, which is what an enrolment loop does, and the size catches a
    filesystem whose timestamps are coarser than the edit.
    """
    path = tmp_path / "fake.safetensors"
    _voice(mel_frames=16).save(path)
    library = VoiceLibrary(tmp_path)
    before = library.load("fake")

    _voice(mel_frames=64).save(path)
    after = library.load("fake")

    assert before.prompt_mel.shape[1] == 16
    assert after.prompt_mel.shape[1] == 64
    assert len(parses) == 2


def test_a_deleted_voice_is_not_served_from_the_cache(
    tmp_path: Path, parses: list[Path]
) -> None:
    """A name that no longer names a file is a 404, cached or not."""
    path = tmp_path / "fake.safetensors"
    _voice().save(path)
    library = VoiceLibrary(tmp_path)
    library.load("fake")

    path.unlink()

    with pytest.raises(VoiceNotFoundError):
        library.load("fake")


def test_the_cache_stays_inside_its_byte_budget(
    tmp_path: Path, parses: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the budget the least recently used profile goes, not the newest.

    Bounded in bytes rather than entries — see ``_VOICE_CACHE_BYTES``. With a
    budget below one profile, only the profile a request just asked for
    survives, so the next request for the other one parses again.
    """
    _voice(name="one").save(tmp_path / "one.safetensors")
    _voice(name="two").save(tmp_path / "two.safetensors")
    monkeypatch.setattr(synthesis, "_VOICE_CACHE_BYTES", 1)
    library = VoiceLibrary(tmp_path)

    library.load("one")
    library.load("two")
    assert library.load("two") is not None
    library.load("one")

    assert [p.stem for p in parses] == ["one", "two", "one"]
    assert len(library._cache) == 1


def test_two_libraries_over_one_directory_stay_equal(tmp_path: Path) -> None:
    """The cache is state, not identity: it must not leak into equality."""
    _voice().save(tmp_path / "fake.safetensors")
    warm, cold = VoiceLibrary(tmp_path), VoiceLibrary(tmp_path)
    warm.load("fake")

    assert warm == cold


def test_eviction_survives_a_concurrent_insert(
    tmp_path: Path, parses: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two request threads may be in the cache at once, and one is walking it.

    The LRU walk in ``_evict`` sums every entry's size, and an insert landing
    mid-walk raises ``RuntimeError: dictionary changed size during iteration``
    out of a request that had nothing wrong with it. Reproduced here rather
    than raced for: the size lookup on the first entry parks its thread until
    the other thread has inserted, so the interleaving that fails is the one
    the test always runs.

    Under the lock the second thread cannot insert while the walk is going, so
    the park times out and both requests finish -- which is why the wait has a
    deadline rather than being an event both threads must reach.
    """
    for name in ("one", "two", "three", "four"):
        _voice(name=name).save(tmp_path / f"{name}.safetensors")
    library = VoiceLibrary(tmp_path)
    library.load("one")
    library.load("two")

    walking = threading.Event()
    inserted = threading.Event()
    parked_once = threading.Event()
    # Through the class dict: the attribute itself resolves to the getter,
    # and what has to be restored is the property.
    real_n_bytes = VoiceProfile.__dict__["n_bytes"].fget
    assert real_n_bytes is not None

    def parked(profile: VoiceProfile) -> int:
        # Only the eviction walk parks, and only once: every other reader of
        # this property (the profile's own repr, the loads above) must not
        # wait on a thread that is not coming. The latch is what makes "once"
        # true. Gating on "has the insert landed yet" instead would never open,
        # because the lock that insert is waiting on is the one under test, so
        # every entry in the walk would pay the full timeout.
        if walking.is_set() and not parked_once.is_set():
            parked_once.set()
            inserted.wait(timeout=0.2)
        return int(real_n_bytes(profile))

    monkeypatch.setattr(VoiceProfile, "n_bytes", property(parked))

    failures: list[BaseException] = []

    def walker() -> None:
        walking.set()
        try:
            library.load("three")
        except BaseException as exc:  # the defect, reported below
            failures.append(exc)

    def inserter() -> None:
        walking.wait(timeout=2.0)
        try:
            library.load("four")
        except BaseException as exc:  # the defect, reported below
            failures.append(exc)
        finally:
            inserted.set()

    threads = [threading.Thread(target=walker), threading.Thread(target=inserter)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
        assert not thread.is_alive()

    assert not failures, failures
    assert library.load("three") is not None
    assert library.load("four") is not None


class _CarryEngine:
    """Only the attribute ``_continuation`` reads off an engine."""

    def __init__(self, algorithm: AlgorithmConfig) -> None:
        self.algorithm = algorithm


@pytest.mark.parametrize("n_tokens", [3, 5, 6, 7, 63, 64, 199, 200, 201])
@pytest.mark.parametrize("decode_mode", ["single", "fusion_mtp2"])
def test_the_continuation_is_the_carry_the_engine_would_take(
    decode_mode: Literal["single", "fusion_mtp2"], n_tokens: int
) -> None:
    """A client that hands the header back continues from the same context an
    in-process join would, which is the only reason the header exists.

    Under ``fusion_mtp2`` the generator re-pairs a prefix from its own start,
    so the tail has to begin where a pair does; a trailing slice can begin on
    the second half of one and fuse the right tokens with the wrong partners.
    """
    algorithm = replace(AlgorithmConfig(), decode_mode=decode_mode)
    tokens = list(range(100, 100 + n_tokens))
    header = synthesis._continuation(cast("Engine", _CarryEngine(algorithm)), tokens)
    assert header == tuple(carry_from(algorithm, tokens))


def test_a_zero_prefix_carries_nothing() -> None:
    """``prefix_tokens: 0`` is a checkpoint saying joins take no context, and
    an empty header is what says so on the wire."""
    algorithm = replace(AlgorithmConfig(), chunking=ChunkConfig(prefix_tokens=0))
    engine = cast("Engine", _CarryEngine(algorithm))
    assert synthesis._continuation(engine, list(range(50))) == ()
