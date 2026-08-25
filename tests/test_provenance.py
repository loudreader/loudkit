"""Provenance: the C2PA claim-only marking that rides with every saved WAV.

The manifest is metadata — it carries the fingerprint, the recipe, the seed
and the audio hash, and nothing else pretends otherwise. These tests pin the
contract: a saved render carries a manifest that verifies against its own
bytes, a plain file carries nothing, and a file whose audio changed fails the
binding.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import loudkit
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
    def test_saved_wav_reads_its_own_manifest(self, tmp_path, audio) -> None:  # type: ignore[no-untyped-def]
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

    def test_verify_binds_manifest_to_the_file_bytes(self, tmp_path, audio) -> None:  # type: ignore[no-untyped-def]
        path = write_wav(tmp_path / "out.wav", audio, 24_000, **_manifest_args())
        manifest, ok = verify_provenance(path)
        assert ok, "the manifest must verify against its own file"

    def test_a_file_with_no_provenance_reads_none(self, tmp_path, audio) -> None:  # type: ignore[no-untyped-def]
        import soundfile as sf

        path = tmp_path / "plain.wav"
        sf.write(str(path), audio, 24_000)
        assert read_provenance(path) is None
        assert verify_provenance(path) == (None, False)

    def test_changed_audio_fails_the_binding(self, tmp_path, audio) -> None:  # type: ignore[no-untyped-def]
        path = write_wav(tmp_path / "out.wav", audio, 24_000, **_manifest_args())
        data = bytearray(Path(path).read_bytes())
        # Flip one byte in the middle of the declared data chunk; the trailer
        # and the RIFF header stay intact.
        data[len(data) // 2] ^= 0xFF
        Path(path).write_bytes(bytes(data))
        _, ok = verify_provenance(path)
        assert not ok, "the hash must catch a tampered audio payload"

    def test_manifest_bytes_are_a_jumbf_c2pa_json_box(self) -> None:
        manifest = build_manifest(audio=b"\x00" * 8, sample_rate=24_000, **_manifest_args())
        wire = manifest_bytes(manifest)
        assert wire[:4] == struct.pack(">I", len(wire)), "the outer length prefixes the box"
        assert wire[4:20] == JUMBF_UUID.bytes


class TestResultSave:
    def test_result_save_embeds_provenance_by_default(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from loudkit.engine import Result, StageTimings

        result = Result(
            audio=np.zeros(4800, dtype=np.float32),
            tokens=[],
            mel=np.zeros((80, 10), dtype=np.float32),
            seed=3,
            sample_rate=24_000,
            timings=StageTimings(0.0, 0.0, 0.0),
            algorithm_fingerprint="a72d03b3ed3ef9ac",
            recipe_version="loudkit-1",
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

    def test_identity_reaches_the_saved_manifest(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The digests and the datapath land in the file, and the voice label
        defaults to what the result already knows."""
        from loudkit.engine import Result, StageTimings

        result = Result(
            audio=np.zeros(4800, dtype=np.float32),
            tokens=[],
            mel=np.zeros((80, 10), dtype=np.float32),
            seed=3,
            sample_rate=24_000,
            timings=StageTimings(0.0, 0.0, 0.0),
            algorithm_fingerprint="a72d03b3ed3ef9ac",
            recipe_version="loudkit-1",
            voice_name="en_klett",
            language="pl",
            voice_sha256="aa" * 32,
            checkpoint_sha256="bb" * 32,
            backend="onnx",
            execution="cpu | fp32",
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

    def test_include_provenance_false_writes_a_plain_wav(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from loudkit.engine import Result, StageTimings

        result = Result(
            audio=np.zeros(4800, dtype=np.float32),
            tokens=[],
            mel=np.zeros((80, 10), dtype=np.float32),
            seed=3,
            sample_rate=24_000,
            timings=StageTimings(0.0, 0.0, 0.0),
            algorithm_fingerprint="a72d03b3ed3ef9ac",
            recipe_version="loudkit-1",
        )
        path = tmp_path / "plain.wav"
        result.save(str(path), include_provenance=False)
        assert read_provenance(path) is None

    def test_top_level_exports(self) -> None:
        assert loudkit.read_provenance is read_provenance
        assert loudkit.verify_provenance is verify_provenance


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
    def _kwargs() -> dict[str, object]:
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
        write_wav(target, audio, **self._kwargs())  # type: ignore[arg-type]
        before = target.read_bytes()

        with (
            patch("loudkit.provenance.manifest_bytes", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError),
        ):
            write_wav(target, audio, **self._kwargs())  # type: ignore[arg-type]

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
        write_wav(target, np.zeros(2400, dtype=np.float32), **self._kwargs())  # type: ignore[arg-type]
        assert target.exists()
        assert target.read_bytes()[:4] == b"fLaC"
