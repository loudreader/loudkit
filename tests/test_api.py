"""The three calls the README opens with, and their shape."""

from __future__ import annotations

import pytest

import loudkit


class TestEnrollIsTopLevel:
    """Cloning is the feature people arrive for; it should cost one concept.

    `load` and `voice` are one call each. Enrolling used to mean importing
    `build_torch_enroller` out of a backend module and resolving the checkpoint
    by hand — a caller reading the README met three new names before their first
    clone, and the shape of the API told them cloning was an internal.
    """

    def test_enroll_is_exported_beside_load_and_voice(self) -> None:
        assert "enroll" in loudkit.__all__
        assert callable(loudkit.enroll)

    def test_enroll_takes_samples_without_needing_an_audio_reader(self) -> None:
        # Passing an array must not require librosa: the file-reading path is
        # the only part that needs it, and a caller who already has samples
        # should not be told to install an audio stack to use them.
        import inspect

        signature = inspect.signature(loudkit.enroll)
        assert list(signature.parameters) == [
            "audio",
            "checkpoint",
            "name",
            # Beside `name`, because it is the same kind of fact about the
            # voice and is the field an omitted `language=` falls back to at
            # synthesis. It was absent, so every cloned voice claimed English
            # and thirteen of the eighteen shipped profiles read their own
            # language through the English funnel.
            "language",
            "device",
            "revision",
            # Last, and keyword-only like the rest: a release resolves its own
            # voice encoder, so this is the escape hatch for a tree that keeps
            # it somewhere else, not part of the ordinary call.
            "voice_encoder_weights",
        ]
        # audio first, checkpoint second: the recording is the subject.
        assert signature.parameters["name"].kind is inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.parametrize("device", ["cpu", "onnx", "coreml", "garbage"])
    def test_enroll_explains_the_missing_extra_before_importing_torch(
        self, monkeypatch: pytest.MonkeyPatch, device: str
    ) -> None:
        """A bare install fails at the public seam, with an actionable command.

        Marking the backend module unavailable makes ordering observable even
        in this development environment, where torch itself is installed. If
        the backend import moves above the runtime guard again, Python raises
        its raw import error before LoudKit gets to explain the missing extra.
        """
        import importlib.util
        import sys

        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: None if name == "torch" else real_find_spec(name),
        )
        monkeypatch.setitem(sys.modules, "loudkit.backends.torch_backend", None)

        with pytest.raises(ModuleNotFoundError, match=r"loudkit\[enroll\]"):
            loudkit.enroll([], "unused.safetensors", device=device)
