"""An absent manifest key means one thing, in one place.

``algorithm_from`` used to retype fifteen of ``config.py``'s dataclass defaults
as its own literals. They agreed, but nothing said they had to: changing a
default in one file and not the other would have moved every fingerprint read
from a manifest that omits the key, silently, in the direction of "the old
number is still right somewhere".

So this reads the defaults off the dataclasses and compares field by field,
which is also what catches a *new* field arriving with a literal beside it.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from loudkit.config import EDGE_FADE_SECONDS, AlgorithmConfig, SamplingConfig, WindowConfig
from loudkit.manifest import algorithm_from


class TestAnEmptyManifestIsTheDataclass:
    def test_every_field_matches_except_the_documented_one(self) -> None:
        parsed = algorithm_from({})
        default = AlgorithmConfig()
        differ = [
            f.name
            for f in fields(AlgorithmConfig)
            if getattr(parsed, f.name) != getattr(default, f.name)
        ]
        # An absent edge_fade_seconds means a manifest older than the field, so
        # it takes the 5 ms those releases shipped rather than today's 20 ms.
        assert differ == ["edge_fade_seconds"]
        assert parsed.edge_fade == 0.005
        assert default.edge_fade == EDGE_FADE_SECONDS

    def test_naming_todays_edge_fade_reproduces_the_dataclass_exactly(self) -> None:
        assert algorithm_from({"edge_fade_seconds": EDGE_FADE_SECONDS}) == AlgorithmConfig()

    def test_the_nested_blocks_are_their_own_defaults(self) -> None:
        parsed = algorithm_from({})
        assert parsed.sampling == SamplingConfig()
        assert parsed.window == WindowConfig()
        # `window: null` is the ragged window said out loud, same object.
        assert algorithm_from({"window": None}).window == WindowConfig()

    def test_a_present_window_block_still_defaults_its_length(self) -> None:
        """The branch the field-by-field case above cannot reach: `window` is a
        mapping, so `_window_from` fills the rest in rather than returning
        WindowConfig() whole."""
        parsed = algorithm_from({"window": {"pad_token_id": 4254}})
        assert parsed.window.max_speech_tokens == WindowConfig().max_speech_tokens
        assert parsed.window.pad_token_id == 4254


class TestPresentKeysStillWin:
    """Reading the default off the dataclass must not shadow a declared value."""

    def test_a_declared_value_is_taken(self) -> None:
        parsed = algorithm_from(
            {
                "n_cfm_timesteps": 4,
                "token_rate_hz": 12.5,
                "sample_rate": 16_000,
                "speech_vocab_size": 9000,
                "speech_tokens": {"start": 6000, "stop": 6001},
                "sampling_defaults": {
                    "temperature": 0.9,
                    "repetition_penalty": 1.5,
                    "min_p": 0.1,
                    "max_new_tokens": 200,
                },
                "eos_floor": {"min_tokens_floor": 10, "min_tokens_text_ratio": 1.2},
                "window": {"max_speech_tokens": 300},
            }
        )
        assert parsed.euler_steps == 4
        assert parsed.token_rate_hz == 12.5
        assert parsed.sample_rate == 16_000
        assert parsed.speech_vocab_size == 9000
        assert (parsed.start_speech_token, parsed.stop_speech_token) == (6000, 6001)
        assert parsed.sampling.temperature == 0.9
        assert parsed.sampling.repetition_penalty == 1.5
        assert parsed.sampling.min_p == 0.1
        assert parsed.sampling.max_new_tokens == 200
        assert parsed.sampling.min_tokens_floor == 10
        assert parsed.sampling.min_tokens_text_ratio == 1.2
        assert parsed.window.max_speech_tokens == 300

    def test_the_guidance_mode_is_still_a_closed_set(self) -> None:
        assert algorithm_from({"guidance": "single_path"}).guidance == "single_path"
        with pytest.raises(ValueError, match="unknown guidance mode"):
            algorithm_from({"guidance": "single_pat"})


class TestEveryScalarComesThroughTheCheckedReader:
    """A key of the wrong JSON shape is refused, not coerced.

    `int()` and `float()` take a JSON boolean and a string of digits, so
    `pad_token_id: true` loaded as token 1 and `max_new_tokens: "7"` as a
    budget of seven, each producing a working engine under a fingerprint that
    said the manifest had been read. The four ports refuse these; the
    reference is the file they mirror, so it refuses them first.
    """

    @pytest.mark.parametrize("bad", [True, False, "7", None, [], {}])
    @pytest.mark.parametrize(
        "path",
        [
            ("window", "max_speech_tokens"),
            ("window", "static_length"),
            ("window", "pad_token_id"),
            ("window", "static_prompt_tokens"),
            ("chunking", "max_tokens"),
            ("chunking", "prefix_tokens"),
            ("chunking", "first_chunk_max_tokens"),
            ("speech_tokens", "start"),
            ("speech_tokens", "stop"),
            ("sampling_defaults", "temperature"),
            ("sampling_defaults", "repetition_penalty"),
            ("sampling_defaults", "min_p"),
            ("sampling_defaults", "max_new_tokens"),
            ("eos_floor", "min_tokens_floor"),
            ("eos_floor", "min_tokens_text_ratio"),
        ],
    )
    def test_a_wrong_shape_is_refused(self, path: tuple[str, str], bad: object) -> None:
        block, key = path
        if bad is None and key in (
            "static_length",
            "pad_token_id",
            "static_prompt_tokens",
            "first_chunk_max_tokens",
        ):
            # `null` is the ragged length and the unset budget, said out loud.
            pytest.skip("null is a value this key accepts")
        with pytest.raises(ValueError, match=key):
            algorithm_from({block: {key: bad}})

    @pytest.mark.parametrize(
        "path",
        [
            ("window", "max_speech_tokens"),
            ("window", "pad_token_id"),
            ("chunking", "prefix_tokens"),
            ("speech_tokens", "start"),
            ("sampling_defaults", "max_new_tokens"),
            ("eos_floor", "min_tokens_floor"),
        ],
    )
    def test_a_count_must_be_whole(self, path: tuple[str, str]) -> None:
        """2.7 tokens is not a value that got rounded, it is one computed wrong."""
        block, key = path
        with pytest.raises(ValueError, match="must be a whole number"):
            algorithm_from({block: {key: 2.7}})

    @pytest.mark.parametrize("bad", [True, "7", None, {}])
    @pytest.mark.parametrize("key", ["n_cfm_timesteps", "sample_rate", "speech_vocab_size"])
    def test_a_top_level_count_is_refused_too(self, key: str, bad: object) -> None:
        with pytest.raises(ValueError, match=key):
            algorithm_from({key: bad})

    @pytest.mark.parametrize(
        "key", ["silence_token_ids", "silence_render_ids", "quiet_render_ids"]
    )
    @pytest.mark.parametrize("bad", [True, "7", 2.7, None])
    def test_a_census_entry_is_checked_like_one_id(self, key: str, bad: object) -> None:
        with pytest.raises(ValueError, match=key):
            algorithm_from({key: [bad]})

    def test_a_euler_grid_point_is_checked_like_one_number(self) -> None:
        with pytest.raises(ValueError, match="euler_grid"):
            algorithm_from({"euler_grid": [0.0, "0.5", 1.0]})

    def test_the_refusal_names_the_whole_path(self) -> None:
        """The reader that refuses is not the one the manifest author reads, so
        the message carries the block as well as the key."""
        with pytest.raises(ValueError, match=r"manifest\['window'\]\['pad_token_id'\]"):
            algorithm_from({"window": {"pad_token_id": True}})

    def test_the_shipped_shapes_still_load(self) -> None:
        """The refusals are for values nobody wrote; every value a real
        manifest carries still reads, ints for counts and floats for rates."""
        parsed = algorithm_from(
            {
                "n_cfm_timesteps": 2,
                "token_rate_hz": 25.0,
                "sample_rate": 24_000,
                "speech_tokens": {"start": 6561, "stop": 6562},
                "sampling_defaults": {"temperature": 0.8, "max_new_tokens": 255},
                "eos_floor": {"min_tokens_floor": 10, "min_tokens_text_ratio": 1.2},
                "window": {
                    "max_speech_tokens": 255,
                    "static_length": 255,
                    "pad_token_id": 4254,
                },
                "chunking": {"enabled": True, "max_tokens": 255, "prefix_tokens": 6},
                "silence_token_ids": [1731, 1821],
                "euler_grid": [0.0, 0.5, 1.0],
            }
        )
        assert parsed.window.pad_token_id == 4254
        assert parsed.sampling.max_new_tokens == 255
        assert parsed.euler_grid == (0.0, 0.5, 1.0)
        # A whole number written as a float is still a whole number.
        assert algorithm_from({"sample_rate": 24000.0}).sample_rate == 24_000
