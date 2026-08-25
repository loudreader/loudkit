"""VoiceProfile persistence — the round-trip, without weights.

A voice is a file you can copy around, so save/load is the whole deal: every
field must survive exactly (the tensors bit-for-bit, the scalars exactly), a
profile saved with a newer format version must be refused loudly rather than
mis-read, and a file that is not a loudkit voice must fail with a message that
says so. These use safetensors directly, so no weights and no torch.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from loudkit.voice import (
    ENROLMENT_FIRST_WINDOW,
    VOICE_FORMAT_VERSION,
    VoiceProfile,
)


def _voice(**overrides: object) -> VoiceProfile:
    rng = np.random.default_rng(7)
    base: dict[str, object] = {
        "name": "voice",
        "speaker_embedding": rng.normal(size=256).astype(np.float32),
        "flow_embedding": rng.normal(size=192).astype(np.float32),
        "prompt_tokens": rng.integers(0, 6561, size=250).astype(np.int64),
        "prompt_mel": rng.normal(size=(80, 500)).astype(np.float32),
        "cond_prompt_tokens": rng.integers(0, 6561, size=150).astype(np.int64),
    }
    base.update(overrides)
    return VoiceProfile(**base)  # type: ignore[arg-type]


class TestRoundTrip:
    def test_all_fields_survive_exactly(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        original = _voice(name="alice", source_sample_rate=44_100, language="pl")
        path = original.save(tmp_path / "alice.safetensors")

        loaded = VoiceProfile.load(path)
        assert loaded.name == "alice"
        assert loaded.source_sample_rate == 44_100
        assert loaded.language == "pl"
        for field in (
            "speaker_embedding",
            "flow_embedding",
            "prompt_tokens",
            "prompt_mel",
            "cond_prompt_tokens",
        ):
            np.testing.assert_array_equal(getattr(loaded, field), getattr(original, field))

    def test_load_records_the_file_digest(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A loaded profile knows the SHA-256 of the file it came from, and a
        profile that never touched disk carries the empty string."""
        import hashlib

        path = _voice().save(tmp_path / "v.safetensors")
        loaded = VoiceProfile.load(path)
        assert loaded.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert _voice().source_sha256 == ""

    def test_load_returns_a_distinct_object(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        original = _voice()
        loaded = VoiceProfile.load(original.save(tmp_path / "v.safetensors"))
        # mutating the loaded profile must not touch the loaded-from file
        assert loaded is not original

    def test_defaults_round_trip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        loaded = VoiceProfile.load(_voice().save(tmp_path / "v.safetensors"))
        assert loaded.source_sample_rate == 24_000
        assert loaded.language == "en"

    def test_path_parent_is_created(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        nested = tmp_path / "voices" / "dir"
        path = _voice().save(nested / "v.safetensors")
        assert path.exists()


class TestValidation:
    def test_save_writes_a_recognisable_header(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from safetensors import safe_open

        path = _voice().save(tmp_path / "v.safetensors")
        with safe_open(str(path), framework="numpy") as f:
            meta = f.metadata() or {}
            assert "voice" in meta
        import json

        header = json.loads((safe_open(str(path), framework="numpy").metadata() or {})["voice"])
        assert header["format_version"] == VOICE_FORMAT_VERSION
        assert header["name"] == "voice"

    def test_future_format_version_is_refused(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import json

        from safetensors.numpy import save_file

        path = _voice().save(tmp_path / "v.safetensors")
        tensors = {}
        with _open_numpy(path) as f:
            for k in f.keys():  # noqa: SIM118 — safe_open is not iterable
                tensors[k] = f.get_tensor(k)
        header = {"format_version": VOICE_FORMAT_VERSION + 1, "name": "v"}
        save_file(tensors, str(path), metadata={"voice": json.dumps(header)})
        with pytest.raises(ValueError, match="voice format version"):
            VoiceProfile.load(path)

    def test_file_without_a_voice_header_is_refused(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from safetensors.numpy import save_file

        path = tmp_path / "not_a_voice.safetensors"
        save_file({"x": np.zeros(4, np.float32)}, str(path), metadata={"other": "1"})
        with pytest.raises(ValueError, match="voice format version"):
            VoiceProfile.load(path)


class TestDegenerateProfilesAreRefused:
    """A profile that the backends would disagree about must not load.

    Shape validation alone let a well-shaped but degenerate file through, and
    the three renderers then differed on what it meant: torch's ``F.normalize``
    carries an epsilon and returns a finite (arbitrary) direction for a zero
    speaker vector, while ONNX and CoreML divide by the raw norm and produce
    NaN. One file, two behaviours, and no error on either path — which is the
    divergence class the whole library is built to prevent, arriving through
    data instead of through code.
    """

    def test_zero_embedding(self) -> None:
        with pytest.raises(ValueError, match="norm"):
            _voice(flow_embedding=np.zeros(192, np.float32))
        with pytest.raises(ValueError, match="norm"):
            _voice(speaker_embedding=np.zeros(256, np.float32))

    def test_non_finite_embedding(self) -> None:
        bad = np.ones(192, np.float32)
        bad[3] = np.nan
        with pytest.raises(ValueError, match="NaN or infinity"):
            _voice(flow_embedding=bad)
        worse = np.ones(256, np.float32)
        worse[0] = np.inf
        with pytest.raises(ValueError, match="NaN or infinity"):
            _voice(speaker_embedding=worse)

    def test_non_finite_prompt_mel(self) -> None:
        mel = np.zeros((80, 8), np.float32)
        mel[2, 2] = np.nan
        with pytest.raises(ValueError, match="prompt_mel contains"):
            _voice(prompt_mel=mel)

    def test_negative_token_id(self) -> None:
        """Negative ids index an embedding table from the end — silently."""
        tokens = np.array([1, 2, -3], np.int64)
        with pytest.raises(ValueError, match="negative id"):
            _voice(prompt_tokens=tokens)

    def test_non_positive_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="source_sample_rate"):
            _voice(source_sample_rate=0)

    def test_a_degenerate_profile_cannot_be_loaded_either(self, tmp_path) -> None:
        """Validation lives in ``__post_init__``, which ``load`` goes through.

        Worth pinning: the attack surface is a *file*, not a constructor call,
        and a loader that built the dataclass by another route would bypass
        every check above.
        """
        import json

        from safetensors.numpy import save_file

        good = _voice()
        path = tmp_path / "v.safetensors"
        good.save(path)
        tensors = {}
        with _open_numpy(path) as f:
            for k in f.keys():  # noqa: SIM118 — safe_open is not iterable
                tensors[k] = f.get_tensor(k)
        tensors["flow_embedding"] = np.zeros(192, np.float32)
        header = {"format_version": VOICE_FORMAT_VERSION, "name": "v"}
        save_file(tensors, str(path), metadata={"voice": json.dumps(header)})

        with pytest.raises(ValueError, match="norm"):
            VoiceProfile.load(path)


def _open_numpy(path):  # type: ignore[no-untyped-def]
    from safetensors import safe_open

    return safe_open(str(path), framework="numpy")


class TestTheProfileCarriesItsOwnLanguage:
    """`VoiceProfile.language` is what an omitted `language=` falls back to.

    The field existed, was documented as the voice's own, and `_resolve_language`
    has always used it — and `enroll()` had no parameter for it, so every cloned
    voice claimed English by construction. All eighteen shipped profiles said
    `en`, the Dutch and Polish and Portuguese ones included, so
    `engine.synthesize(polish_text, tomasz)` read Polish through the English
    funnel: "Mam twenty-five lat i three point five kilograma."

    Three rounds of review missed it because seeing it means loading a profile,
    which means having the weights.
    """

    def test_enroll_takes_a_language(self) -> None:
        import inspect

        import loudkit

        assert "language" in inspect.signature(loudkit.enroll).parameters, (
            "enroll() cannot set the field the engine falls back to"
        )

    def test_the_language_survives_a_save_and_load(self, tmp_path) -> None:
        import dataclasses

        from loudkit import VoiceProfile

        path = tmp_path / "v.safetensors"
        dataclasses.replace(_voice(), language="pl").save(path)
        assert VoiceProfile.load(path).language == "pl"

    def test_every_provenance_entry_is_complete_and_licenced(self) -> None:
        """Attribution with a hole is attribution that cannot be checked.

        Each roster entry must name its donor or speaker, a source with a URL
        and a licence this project has reviewed, and the consent basis the
        recording rests on. A profile without those is exactly the "where did
        this voice come from?" gap the roster exists to close.
        """
        import json as _json
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        provenance = repo / "docs" / "voices" / "roster" / "provenance.json"
        voices = _json.loads(provenance.read_text(encoding="utf-8"))
        assert len(voices) == 20, f"roster holds {len(voices)} voices, expected 20"

        known_licences = {"CC0-1.0", "CC-BY-4.0"}
        incomplete = []
        for v in voices:
            src = v.get("source", {})
            if (
                not v.get("name")
                or not v.get("donor")
                or not v.get("language_id")
                or not src.get("url")
                or src.get("license") not in known_licences
                or not src.get("consent")
            ):
                incomplete.append(v.get("name") or "<unnamed>")
        assert not incomplete, (
            "roster entries missing donor/source/licence/consent: " + ", ".join(incomplete)
        )


class TestTheEnrolmentStrategyIsRecorded:
    """A profile is an artefact; this says how it was made.

    Enrolment picks a window of the reference clip *before* the transform the
    four ports are held to by `tests/data/enrollment/`. So two strategies give
    two different voices from one recording, with nothing in the tensors to tell
    them apart — the same shape of problem `TextConfig.recipe` exists for.
    """

    def test_it_defaults_to_what_every_existing_voice_was_made_with(self) -> None:
        assert _voice().enrolment == ENROLMENT_FIRST_WINDOW

    def test_it_survives_a_round_trip(self, tmp_path) -> None:
        path = _voice().save(tmp_path / "v.safetensors")
        assert VoiceProfile.load(path).enrolment == ENROLMENT_FIRST_WINDOW

    def test_a_profile_without_the_field_reads_as_the_original_strategy(self, tmp_path) -> None:
        """Every voice enrolled before the field existed was cut this way.

        Written by hand rather than by `save`, because what is being tested is a
        header this version never writes: the one every shipped profile carries.
        """
        import json

        from safetensors.numpy import save_file

        v = _voice()
        header = {
            "format_version": VOICE_FORMAT_VERSION,
            "name": v.name,
            # every profile written before 0.1 carried this key; it is ignored
            "emotion": 0.5,
            "source_sample_rate": v.source_sample_rate,
            "language": v.language,
        }
        assert "enrolment" not in header
        path = tmp_path / "old.safetensors"
        save_file(
            {
                "speaker_embedding": v.speaker_embedding,
                "flow_embedding": v.flow_embedding,
                "prompt_tokens": v.prompt_tokens,
                "prompt_mel": v.prompt_mel,
                "cond_prompt_tokens": v.cond_prompt_tokens,
            },
            str(path),
            metadata={"voice": json.dumps(header)},
        )
        assert VoiceProfile.load(path).enrolment == ENROLMENT_FIRST_WINDOW

    def test_a_strategy_this_build_does_not_implement_is_refused(self) -> None:
        """The whole point of the field.

        A build that cannot reproduce the window a profile was cut from has to
        refuse it. Applying its own instead would return a different voice under
        the same name, which is exactly what recording the strategy prevents.
        """
        with pytest.raises(ValueError, match="enrolment strategy"):
            dataclasses.replace(_voice(), enrolment="best-window")
