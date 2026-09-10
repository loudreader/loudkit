"""Provenance: the unsigned loudkit marking that rides with every saved WAV.

The manifest is metadata — it carries the fingerprint, the recipe, the seed
and the audio hash, and nothing else pretends otherwise. These tests pin the
contract: a saved render carries a manifest that verifies against its own
bytes, a plain file carries nothing, and a file whose audio changed fails the
binding.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pytest

from loudkit.provenance import (
    DIGITAL_SOURCE_TYPE,
    JUMBF_UUID,
    MANIFEST_LABEL,
    build_manifest,
    manifest_bytes,
    read_provenance,
    verify_provenance,
    write_wav,
)


class _WavArgs(TypedDict):
    """What every `write_wav` call in this module carries. Spelled out so the
    splat is checked against the signature rather than widened to `object`."""

    algorithm_fingerprint: str
    recipe_version: str
    seed: int
    sample_rate: int
    voice: str
    language: str
    text: str


@pytest.fixture
def audio() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.05, size=24_000).astype(np.float32)


def _manifest_args() -> dict[str, Any]:
    return {
        "algorithm_fingerprint": "a72d03b3ed3ef9ac",
        "recipe_version": "loudkit-1",
        "seed": 7,
        "voice": "en_klett",
        "language": "en",
        "text": "The quick brown fox.",
    }


def _loudkit_assertion(manifest: dict[str, object] | None) -> dict[str, Any]:
    assert manifest is not None
    assertions = manifest["assertions"]
    assert isinstance(assertions, list)
    entry = assertions[1]
    assert isinstance(entry, dict)
    data = entry.get("data")
    assert isinstance(data, dict)
    return data


class TestRoundTrip:
    def test_saved_wav_reads_its_own_manifest(self, tmp_path, audio) -> None:
        path = write_wav(tmp_path / "out.wav", audio, 24_000, **_manifest_args())
        manifest = read_provenance(path)
        assert manifest is not None
        assertions = manifest["assertions"]
        assert isinstance(assertions, list)
        action_entry = assertions[0]
        assert isinstance(action_entry, dict)
        action_data = action_entry.get("data")
        assert isinstance(action_data, dict)
        actions = action_data["actions"]
        assert isinstance(actions, list)
        action = actions[0]
        assert isinstance(action, dict)
        assert action["action"] == "c2pa.created"
        assert action["digitalSourceType"] == DIGITAL_SOURCE_TYPE
        assert action["softwareAgent"] == "loudkit"
        meta = _loudkit_assertion(manifest)
        assert meta["algorithm_fingerprint"] == "a72d03b3ed3ef9ac"
        assert meta["seed"] == 7
        assert meta["voice"] == "en_klett"

    def test_verify_binds_manifest_to_the_file_bytes(self, tmp_path, audio) -> None:
        path = write_wav(tmp_path / "out.wav", audio, 24_000, **_manifest_args())
        manifest, ok = verify_provenance(path)
        assert ok, "the manifest must verify against its own file"

    def test_a_file_with_no_provenance_reads_none(self, tmp_path, audio) -> None:
        import soundfile as sf

        path = tmp_path / "plain.wav"
        sf.write(str(path), audio, 24_000)
        assert read_provenance(path) is None
        assert verify_provenance(path) == (None, False)

    def test_changed_audio_fails_the_binding(self, tmp_path, audio) -> None:
        path = write_wav(tmp_path / "out.wav", audio, 24_000, **_manifest_args())
        data = bytearray(Path(path).read_bytes())
        # Flip one byte in the middle of the declared data chunk; the trailer
        # and the RIFF header stay intact.
        data[len(data) // 2] ^= 0xFF
        Path(path).write_bytes(bytes(data))
        _, ok = verify_provenance(path)
        assert not ok, "the hash must catch a tampered audio payload"

    def test_an_odd_chunk_before_the_data_does_not_shift_the_walk(self) -> None:
        """RIFF pads every chunk to an even boundary and does not count the pad.

        Both readers here walk chunks by their declared size, so a file
        carrying an odd `LIST` ahead of `fmt ` lands one byte off and finds
        neither the payload nor the manifest. loudkit writes 16-bit PCM, whose
        chunks are always even, but `read_provenance` is a public door that
        takes any file. Go, JS and Rust already step over the pad.
        """
        from loudkit.provenance import (
            PROVENANCE_CHUNK_ID,
            _find_provenance_chunk,
            _riff_audio_payload,
        )

        pcm = b"\x01\x00\x02\x00"
        body = b"WAVE"
        for cid, payload in (
            (b"LIST", b"INFOxyz"),
            (b"fmt ", b"f" * 16),
            (b"data", pcm),
            (PROVENANCE_CHUNK_ID, b"manifest!"),
        ):
            body += cid + struct.pack("<I", len(payload)) + payload
            body += b"\x00" * (len(payload) % 2)
        blob = b"RIFF" + struct.pack("<I", len(body)) + body

        assert _riff_audio_payload(blob) == pcm
        assert _find_provenance_chunk(blob) == b"manifest!"

    def test_a_file_from_0_1_0_still_reads(self, tmp_path) -> None:
        """0.1.0 appended the boxes past the length the RIFF header declares.

        Every WAV that release saved carries them there. A patch release that
        stopped reading its predecessor's files would leave the audio playable
        and the marking that says what produced it unreadable, which is the
        one thing the marking exists for. Nothing writes this layout now.
        """
        from loudkit.provenance import (
            _riff_audio_payload,
            build_manifest,
            manifest_bytes,
            read_provenance,
            verify_provenance,
            write_wav,
        )

        path = tmp_path / "legacy.wav"
        write_wav(str(path), np.zeros(2400, dtype=np.float32), 24_000, manifest=False)
        claim = build_manifest(
            audio=_riff_audio_payload(path.read_bytes()),
            algorithm_fingerprint="e352f5d090776b54",
            recipe_version="funnel-5",
            seed=7,
            sample_rate=24_000,
            voice="joe",
        )
        with path.open("ab") as handle:
            handle.write(manifest_bytes(claim))

        manifest = read_provenance(str(path))
        assert manifest is not None, "a 0.1.0 file must still read"
        assertions = manifest["assertions"]
        assert isinstance(assertions, list)
        ours = next(a for a in assertions if a["label"] == "loudkit.provenance")
        assert ours["data"]["algorithm_fingerprint"] == "e352f5d090776b54"
        _, ok = verify_provenance(str(path))
        assert ok, "and its hash must still bind to the audio"

    def test_the_manifest_chunk_is_not_called_c2pa(self) -> None:
        """A reader that finds a `C2PA` chunk is entitled to a manifest store.

        These boxes are not one: pointed at them, `c2pa-python` reported a
        malformed manifest rather than passing over a file it has no business
        reading. A private identifier is ignored by everything that does not
        know it, which is what metadata only loudkit understands should do.
        """
        from loudkit.provenance import PROVENANCE_CHUNK_ID

        assert PROVENANCE_CHUNK_ID != b"C2PA"
        assert len(PROVENANCE_CHUNK_ID) == 4

    def test_manifest_bytes_are_a_jumbf_claim_json_box(self) -> None:
        manifest = build_manifest(audio=b"\x00" * 8, sample_rate=24_000, **_manifest_args())
        wire = manifest_bytes(manifest)
        assert wire[:4] == struct.pack(">I", len(wire)), "the outer length prefixes the box"
        assert wire[4:20] == JUMBF_UUID.bytes

    def test_the_default_version_is_the_running_package(self) -> None:
        """A manifest is a claim, so the version in it has to be true.

        The default was the literal "0.1.0" and stayed there through the 0.1.1
        bump, which made every manifest written without an explicit `version=`
        false.
        """
        from loudkit._version import package_version

        manifest = build_manifest(audio=b"\x00" * 8, sample_rate=24_000, **_manifest_args())
        running = package_version()
        assert manifest["claim_generator"] == f"loudkit {running}"
        assert manifest["claim_generator_info"] == [{"name": "loudkit", "version": running}]


class TestResultSave:
    def test_result_save_embeds_provenance_by_default(self, tmp_path) -> None:
        from loudkit.engine import Provenance, Result, StageTimings

        result = Result(
            audio=np.zeros(4800, dtype=np.float32),
            tokens=[],
            mel=np.zeros((80, 10), dtype=np.float32),
            seed=3,
            sample_rate=24_000,
            timings=StageTimings(0.0, 0.0, 0.0),
            provenance=Provenance(
                algorithm_fingerprint="a72d03b3ed3ef9ac", recipe_version="loudkit-1"
            ),
        )
        path = tmp_path / "r.wav"
        result.save(str(path), voice="en_klett", language="en")
        manifest = read_provenance(path)
        assert manifest is not None
        assertions = manifest["assertions"]
        assert isinstance(assertions, list)
        entry = assertions[1]
        assert isinstance(entry, dict)
        assert entry["label"] == MANIFEST_LABEL
        _, ok = verify_provenance(path)
        assert ok

    def test_identity_reaches_the_saved_manifest(self, tmp_path) -> None:
        """The digests and the datapath land in the file, and the voice label
        defaults to what the result already knows."""
        from loudkit.engine import Provenance, Result, StageTimings

        result = Result(
            audio=np.zeros(4800, dtype=np.float32),
            tokens=[],
            mel=np.zeros((80, 10), dtype=np.float32),
            seed=3,
            sample_rate=24_000,
            timings=StageTimings(0.0, 0.0, 0.0),
            provenance=Provenance(
                algorithm_fingerprint="a72d03b3ed3ef9ac",
                recipe_version="loudkit-1",
                voice="en_klett",
                language="pl",
                voice_sha256="aa" * 32,
                checkpoint_sha256="bb" * 32,
                backend="onnx",
                execution="cpu | fp32",
            ),
        )
        path = tmp_path / "id.wav"
        result.save(str(path))
        manifest = read_provenance(path)
        assert manifest is not None
        data = _loudkit_assertion(manifest)
        assert data["voice"] == "en_klett"
        assert data["language"] == "pl"
        assert data["voice_profile_sha256"] == "aa" * 32
        assert data["checkpoint_sha256"] == "bb" * 32
        assert data["backend"] == "onnx"
        assert data["execution"] == "cpu | fp32"

    def test_identity_defaults_are_empty_not_absent(self) -> None:
        """A manifest built without the identity says "not known here" — the
        keys are present and empty, so a reader can distinguish an old file
        from a render whose caller withheld them."""
        manifest = build_manifest(audio=b"\x00" * 8, sample_rate=24_000, **_manifest_args())
        data = _loudkit_assertion(manifest)
        for key in ("voice_profile_sha256", "checkpoint_sha256", "backend", "execution"):
            assert data[key] == ""

    def test_include_provenance_false_writes_a_plain_wav(self, tmp_path) -> None:
        from loudkit.engine import Provenance, Result, StageTimings

        result = Result(
            audio=np.zeros(4800, dtype=np.float32),
            tokens=[],
            mel=np.zeros((80, 10), dtype=np.float32),
            seed=3,
            sample_rate=24_000,
            timings=StageTimings(0.0, 0.0, 0.0),
            provenance=Provenance(
                algorithm_fingerprint="a72d03b3ed3ef9ac", recipe_version="loudkit-1"
            ),
        )
        path = tmp_path / "plain.wav"
        result.save(str(path), include_provenance=False)
        assert read_provenance(path) is None

    def test_the_readers_are_one_object(self) -> None:
        """Not exported from the package root: the public surface is the six
        verbs and what they return. A caller who wants to read a manifest
        back imports the module that owns it."""
        import loudkit
        import loudkit.provenance

        assert loudkit.provenance.read_provenance is read_provenance
        assert loudkit.provenance.verify_provenance is verify_provenance
        assert not hasattr(loudkit, "read_provenance")


class TestTheWriteIsAllOrNothing:
    """A render that dies halfway must not leave a file the caller will use.

    `write_wav` wrote the WAV to the caller's path, read it back, then appended
    the JUMBF box in a second open. Two windows: between the write and the
    append the file was a WAV with no manifest, and during the append it was a
    WAV with half a box — which `soundfile` opens happily and `read_provenance`
    cannot parse, so a crash surfaced much later as "this render has no
    provenance" rather than as the crash it was.
    """

    @staticmethod
    def _kwargs() -> _WavArgs:
        return {
            "algorithm_fingerprint": "f" * 16,
            "recipe_version": "loudkit-1",
            "seed": 7,
            "sample_rate": 24_000,
            "voice": "v",
            "language": "en",
            "text": "hi",
        }

    def test_a_failure_leaves_the_previous_file_untouched(self, tmp_path) -> None:
        from unittest.mock import patch

        import numpy as np

        from loudkit.provenance import write_wav

        target = tmp_path / "out.wav"
        audio = np.zeros(2400, dtype=np.float32)
        write_wav(target, audio, **self._kwargs())
        before = target.read_bytes()

        with (
            patch("loudkit.provenance.manifest_bytes", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError),
        ):
            write_wav(target, audio, **self._kwargs())

        assert target.read_bytes() == before, "a failed rewrite damaged the old file"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["out.wav"], (
            "a partial file was left behind"
        )

    def test_the_temp_name_keeps_the_extension(self, tmp_path) -> None:
        """`soundfile` picks its format from the suffix.

        A temp file ending in `.partial` is one it refuses to write at all, and
        `write_wav("x.flac", …)` has always produced FLAC by that same
        mechanism — so the scratch name has to carry the suffix rather than
        append to it.
        """
        import numpy as np

        from loudkit.provenance import write_wav

        target = tmp_path / "out.flac"
        write_wav(target, np.zeros(2400, dtype=np.float32), **self._kwargs())
        assert target.exists()
        assert target.read_bytes()[:4] == b"fLaC"
