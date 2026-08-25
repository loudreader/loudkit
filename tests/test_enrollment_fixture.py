"""The enrollment fixture is self-consistent, without any weights.

These tests read only the committed files in ``tests/data/enrollment`` — the
reference clip, the golden tensors, and the golden profile — and pin the
invariants that make the fixture a usable parity spec for the four ports:

* every file the manifest names exists with the byte-count its shape implies;
* the golden ``profile.safetensors`` round-trips to exactly the committed
  ``.f32`` / ``.i64`` tensors — so "enroll the reference clip and produce this
  profile, byte for byte" is a single checkable claim, not a tolerance;
* the prompt mel stays aligned at 2 frames per prompt token, the one invariant
  the enrollment pipeline itself enforces.

Why byte-for-byte and not allclose: a filterbank or resampler that is off by
the last ulp does not fail to build and does not throw — it returns numbers,
the model consumes them, a voice comes out, and it is quietly worse with
nothing to point at. A tolerance would turn that divergence into a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from loudkit.voice import VoiceProfile

FIXTURE = Path(__file__).parent / "data" / "enrollment"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))


def _read(name: str, dtype: np.dtype, manifest: dict) -> np.ndarray:
    shape = manifest["files"][name]["shape"]
    raw = (FIXTURE / name).read_bytes()
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    assert len(raw) == expected, f"{name}: {len(raw)} bytes != {expected} for shape {shape}"
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


class TestFixtureFilesExist:
    def test_every_manifest_file_is_present_with_the_right_size(self, manifest) -> None:  # type: ignore[no-untyped-def]
        for name, meta in manifest["files"].items():
            path = FIXTURE / name
            assert path.exists(), f"manifest names {name}, which is missing"
            expected = int(np.prod(meta["shape"])) * (8 if name.endswith(".i64") else 4)
            assert path.stat().st_size == expected, (
                f"{name}: {path.stat().st_size} bytes, expected {expected} for "
                f"shape {meta['shape']}"
            )


class TestGoldenProfile:
    def test_profile_tensors_match_the_committed_files(self, manifest) -> None:  # type: ignore[no-untyped-def]
        profile = VoiceProfile.load(FIXTURE / "profile.safetensors")
        assert profile.name == manifest["name"]

        np.testing.assert_array_equal(
            profile.speaker_embedding,
            _read("speaker_embedding.f32", np.float32, manifest),
        )
        np.testing.assert_array_equal(
            profile.flow_embedding, _read("flow_embedding.f32", np.float32, manifest)
        )
        np.testing.assert_array_equal(
            profile.prompt_tokens, _read("prompt_tokens.i64", np.int64, manifest)
        )
        np.testing.assert_array_equal(
            profile.cond_prompt_tokens,
            _read("cond_prompt_tokens.i64", np.int64, manifest),
        )
        np.testing.assert_array_equal(
            profile.prompt_mel, _read("matcha_mel.f32", np.float32, manifest)
        )

    def test_prompt_mel_stays_aligned_to_tokens(self, manifest) -> None:  # type: ignore[no-untyped-def]
        tokens = _read("prompt_tokens.i64", np.int64, manifest)
        mel = _read("matcha_mel.f32", np.float32, manifest)
        assert mel.shape[0] == 80
        assert mel.shape[1] == 2 * len(tokens), "2 mel frames per prompt token"

    def test_the_embeddings_have_the_shapes_the_renderers_read(self, manifest) -> None:  # type: ignore[no-untyped-def]
        assert _read("flow_embedding.f32", np.float32, manifest).shape == (192,)
        assert _read("speaker_embedding.f32", np.float32, manifest).shape == (256,)
