"""Reading a checkpoint manifest into an :class:`~loudkit.config.AlgorithmConfig`.

The manifest is the authority on every value that is a property of the
weights. Absent keys default; present keys must be the right shape, so a
truncated or hand-edited manifest fails to load rather than loading as the
defaults under a fingerprint that says otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from typing import Any, Literal, TypeVar, cast

from .config import (
    _UNSET,
    EDGE_FADE_SECONDS,
    RECIPE_VERSION,
    AlgorithmConfig,
    ChunkConfig,
    DecodeMode,
    GuidanceMode,
    SamplingConfig,
    WindowConfig,
    _UnsetType,
)
from .postprocess import PostprocessConfig

__all__ = ["algorithm_from"]


_Closed = TypeVar("_Closed", bound=str)


def _flag(block: Mapping[str, object], where: str, key: str, default: bool) -> bool:
    """A JSON boolean under ``block``, or ``default`` for an absent key.

    Not ``bool()``. Python calls 0 false and 1 true and JSON does not, so a
    manifest carrying ``"enabled": 0`` was written by a tool that meant
    ``false`` and emitted a number. Reading it as false hides that mistake
    behind audio joined differently rather than audio missing, and this is
    the one key whose misreading changes every join in a passage at once.
    The four ports refuse it in this sentence.
    """
    if key not in block:
        return default
    raw = block[key]
    if not isinstance(raw, bool):
        raise ValueError(
            f"manifest[{where!r}][{key!r}] must be JSON true or false, got {raw!r}"
        )
    return raw


def _literal(value: str, options: Sequence[_Closed], key: str) -> _Closed:
    """``value`` as the member of ``options`` it equals, or a ``ValueError``.

    The closed sets a manifest declares are Literals on the config fields, and a
    membership check does not narrow a `str` to one. Returning the option that
    matched narrows it without a cast, so the check and the type are the same
    line rather than a check followed by a suppression that trusts it.
    """
    for option in options:
        if value == option:
            return option
    raise ValueError(f"manifest declares unknown {key} {value!r}")


def postprocess_from(pp: Mapping[str, Any]) -> PostprocessConfig:
    """The ``postprocess`` block as a preset; missing knobs take the loudr-1 values.

    Read off the dataclass so a knob added there is read here without being
    named here. The render censuses live at the manifest top level and are
    refused inside the block, so one value has one home.
    """
    default = PostprocessConfig()
    values: dict[str, Any] = {}
    for f in dataclass_fields(PostprocessConfig):
        if f.name not in pp:
            continue
        raw = pp[f.name]
        want = type(getattr(default, f.name))
        if want is tuple:
            raise ValueError(
                f"manifest['postprocess'][{f.name!r}] belongs at the manifest "
                "top level, beside 'silence_token_ids'"
            )
        if want is str:
            if not isinstance(raw, str):
                raise ValueError(
                    f"manifest['postprocess'][{f.name!r}] must be a string, "
                    f"got {type(raw).__name__}"
                )
            values[f.name] = raw
            continue
        # bool first: it is an int subclass, and `true` would set a count to one.
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"manifest['postprocess'][{f.name!r}] must be a number, got {raw!r}"
            )
        if want is int and isinstance(raw, float) and not raw.is_integer():
            raise ValueError(
                f"manifest['postprocess'][{f.name!r}] must be a whole number, got {raw!r}"
            )
        values[f.name] = want(raw)
    mode = values.get("mode", default.mode)
    if mode not in ("off", "report", "trim"):
        raise ValueError(f"manifest declares unknown postprocess mode {mode!r}")
    return replace(default, **values)


def decode_from(raw: object) -> DecodeMode | _UnsetType:
    """The ``decode`` block's mode, or the unset sentinel.

    Absent and an explicit ``"single"`` both stay unset: they are one audible
    decision and must hash the same.
    """
    if raw is None:
        return _UNSET
    if not isinstance(raw, Mapping):
        raise ValueError(f"manifest['decode'] must be an object, got {type(raw).__name__}")
    # Spelled out rather than `get_args(DecodeMode)`, the same reason
    # `algorithm_from` spells out the guidance modes: the literal tuple is
    # checked against the annotation, and get_args returns Any, which is not.
    options: tuple[DecodeMode, ...] = ("single", "fusion_mtp2")
    raw_mode = str(raw.get("mode", "single"))
    for option in options:
        if raw_mode == option:
            return _UNSET if option == "single" else option
    raise ValueError(
        f"manifest['decode']['mode'] is {raw_mode!r}; this build decodes "
        f"{' or '.join(repr(m) for m in options)}"
    )


def _block(manifest: Mapping[str, object], key: str, kind: type, default: object) -> object:
    """``manifest[key]`` checked to be ``kind``, or ``default`` when absent."""
    if key not in manifest:
        return default
    value = manifest[key]
    if not isinstance(value, kind):
        raise ValueError(
            f"manifest key {key!r} must be {kind.__name__}, got {type(value).__name__}"
        )
    # A string is a Sequence of characters; `silence_token_ids: "123"` must not
    # load as three tokens.
    if kind is Sequence and isinstance(value, (str, bytes)):
        raise ValueError(f"manifest key {key!r} must be a list, got a {type(value).__name__}")
    return value


def _where(*path: str) -> str:
    """``manifest['sampling_defaults']['temperature']``, so a refusal names the key."""
    return "manifest" + "".join(f"[{part!r}]" for part in path)


def _number(block: Mapping[str, object], default: float, *path: str) -> float:
    """A JSON number under ``block``, or ``default`` for an absent key.

    Not ``float()``. ``true`` is an ``int`` subclass in Python and is not a
    number in JSON, and a string of digits is a string; either one coerced
    rather than refused gives an engine that computes with a value nobody
    wrote, under a fingerprint that says the manifest was read. Every scalar
    the manifest declares comes through here or through :func:`_flag`.
    """
    key = path[-1]
    if key not in block:
        return default
    value = block[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{_where(*path)} must be a number, got {value!r}")
    return value


def _int(block: Mapping[str, object], default: int, *path: str) -> int:
    """A whole JSON number under ``block``, or ``default`` for an absent key.

    The same rule :func:`postprocess_from` applies to its counts: a token id or
    a token budget of ``2.7`` is not a value that got rounded, it is a value
    that was computed wrong, and truncating it hides the arithmetic that
    produced it.
    """
    value = _number(block, default, *path)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{_where(*path)} must be a whole number, got {value!r}")
    return int(value)


def _opt_int(block: Mapping[str, object], *path: str) -> int | None:
    """A whole JSON number under ``block``, or ``None`` for an absent or null key."""
    key = path[-1]
    if block.get(key) is None:
        return None
    return _int(block, 0, *path)


def _float_list(values: Sequence[object], key: str) -> tuple[float, ...]:
    """A list of JSON numbers, every entry checked the way one number is checked."""
    out: list[float] = []
    for index, raw in enumerate(values):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{_where(key)}[{index}] must be a number, got {raw!r}")
        out.append(float(raw))
    return tuple(out)


def _int_list(values: Sequence[object], key: str) -> tuple[int, ...]:
    """A token-id census, every entry checked the way one id is checked."""
    out: list[int] = []
    for index, raw in enumerate(_float_list(values, key)):
        if not raw.is_integer():
            raise ValueError(f"{_where(key)}[{index}] must be a whole number, got {raw!r}")
        out.append(int(raw))
    return tuple(out)


def _window_from(manifest: Mapping[str, object]) -> WindowConfig:
    """``window: null`` is the ragged window said explicitly; any other non-mapping is
    refused."""
    if "window" not in manifest or manifest["window"] is None:
        return WindowConfig()
    win = manifest["window"]
    if not isinstance(win, Mapping):
        raise ValueError(
            f"manifest['window'] must be a mapping or null (ragged), got {type(win).__name__}"
        )

    default = WindowConfig()
    return WindowConfig(
        max_speech_tokens=_int(win, default.max_speech_tokens, "window", "max_speech_tokens"),
        static_length=_opt_int(win, "window", "static_length"),
        pad_token_id=_opt_int(win, "window", "pad_token_id"),
        static_prompt_tokens=_opt_int(win, "window", "static_prompt_tokens"),
    )


def _chunking_from(manifest: Mapping[str, object]) -> ChunkConfig:
    chunk = cast(Mapping[str, Any], _block(manifest, "chunking", Mapping, {}))
    default = ChunkConfig()
    for key in ("split_on", "abbreviations"):
        if key in chunk and isinstance(chunk[key], (str, bytes)):
            raise ValueError(
                f"manifest key 'chunking.{key}' must be a list of strings, got a string"
            )
    cap_resplit: Literal["word", "off"] = _literal(
        str(chunk.get("cap_resplit", default.cap_resplit)),
        ("word", "off"),
        "chunking.cap_resplit",
    )
    mid: Literal["hold", "break"] = _literal(
        str(chunk.get("mid_sentence_period", default.mid_sentence_period)),
        ("hold", "break"),
        "chunking.mid_sentence_period",
    )
    first: int | None | _UnsetType = default.first_chunk_max_tokens
    if "first_chunk_max_tokens" in chunk:
        first = _opt_int(chunk, "chunking", "first_chunk_max_tokens")
    return ChunkConfig(
        enabled=_flag(chunk, "chunking", "enabled", default.enabled),
        max_tokens=_int(chunk, default.max_tokens, "chunking", "max_tokens"),
        prefix_tokens=_int(chunk, default.prefix_tokens, "chunking", "prefix_tokens"),
        first_chunk_max_tokens=first,
        split_on=(
            tuple(str(s) for s in chunk["split_on"])
            if "split_on" in chunk
            else default.split_on
        ),
        abbreviations=(
            tuple(str(a) for a in chunk["abbreviations"])
            if "abbreviations" in chunk
            else default.abbreviations
        ),
        cap_resplit=cap_resplit,
        mid_sentence_period=mid,
    )


def edge_fade_from(raw: object) -> float | _UnsetType:
    """``edge_fade_seconds``, or the unset sentinel.

    Legacy manifests without the key retain 5 ms. New releases explicitly name 0.02.
    The canonical form includes that duration; historical 0.005 retains its old identity.
    """
    if raw is None:
        return 0.005  # Legacy manifests predate the explicit 20 ms release default.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"manifest['edge_fade_seconds'] must be a number, got {type(raw).__name__}"
        )
    seconds = float(raw)
    return _UNSET if seconds == EDGE_FADE_SECONDS else seconds


def algorithm_from(manifest: Mapping[str, object]) -> AlgorithmConfig:
    """Build the algorithm a checkpoint declares. Raises ``ValueError`` on a malformed key.

    Every absent key falls back to the dataclass field, read off an instance
    rather than retyped here, so an absent key means one thing in one place.
    ``edge_fade_seconds`` is the one deliberate exception: a manifest that never
    names it is older than the field, and :func:`edge_fade_from` gives it the
    5 ms those releases shipped rather than today's default.
    """
    default = AlgorithmConfig()
    samp = default.sampling
    sampling_defaults = cast(
        Mapping[str, Any], _block(manifest, "sampling_defaults", Mapping, {})
    )
    sil = cast(Sequence[Any], _block(manifest, "silence_token_ids", Sequence, ()))
    sil_render = cast(Sequence[Any], _block(manifest, "silence_render_ids", Sequence, ()))
    quiet_render = cast(Sequence[Any], _block(manifest, "quiet_render_ids", Sequence, ()))
    speech = cast(Mapping[str, Any], _block(manifest, "speech_tokens", Mapping, {}))

    # Spelled out rather than `get_args(GuidanceMode)`: the literal tuple is
    # checked against the annotation, and get_args returns Any, which is not.
    guidance: GuidanceMode = _literal(
        str(manifest.get("guidance", default.guidance)),
        ("single_path", "cfg_dual_path"),
        "guidance mode",
    )

    eos = cast(Mapping[str, Any], _block(manifest, "eos_floor", Mapping, {}))

    recipe = str(manifest.get("recipe_version", RECIPE_VERSION))
    if recipe != RECIPE_VERSION:
        raise ValueError(
            f"manifest declares recipe_version {recipe!r}; "
            f"the only recipe is {RECIPE_VERSION!r}"
        )
    postprocess = postprocess_from(
        cast(Mapping[str, Any], _block(manifest, "postprocess", Mapping, {}))
    )
    postprocess = replace(
        postprocess,
        silence_render_ids=_int_list(sil_render, "silence_render_ids"),
        quiet_render_ids=_int_list(quiet_render, "quiet_render_ids"),
    )

    grid_raw = manifest.get("euler_grid")
    if grid_raw is None:
        euler_grid: tuple[float, ...] | None = None
    elif isinstance(grid_raw, Sequence) and not isinstance(grid_raw, (str, bytes)):
        euler_grid = _float_list(grid_raw, "euler_grid")
    else:
        raise ValueError(
            f"manifest['euler_grid'] must be a list of floats or null, got "
            f"{type(grid_raw).__name__}"
        )

    return AlgorithmConfig(
        recipe_version=recipe,
        guidance=guidance,
        guidance_rate=float(_number(manifest, default.guidance_rate, "guidance_rate")),
        decode_mode=decode_from(manifest.get("decode")),
        edge_fade_seconds=edge_fade_from(manifest.get("edge_fade_seconds")),
        euler_steps=_int(manifest, default.euler_steps, "n_cfm_timesteps"),
        euler_grid=euler_grid,
        token_rate_hz=float(_number(manifest, default.token_rate_hz, "token_rate_hz")),
        chunking=_chunking_from(manifest),
        sample_rate=_int(manifest, default.sample_rate, "sample_rate"),
        speech_vocab_size=_int(manifest, default.speech_vocab_size, "speech_vocab_size"),
        start_speech_token=_int(speech, default.start_speech_token, "speech_tokens", "start"),
        stop_speech_token=_int(speech, default.stop_speech_token, "speech_tokens", "stop"),
        sampling=SamplingConfig(
            temperature=float(
                _number(sampling_defaults, samp.temperature, "sampling_defaults", "temperature")
            ),
            repetition_penalty=float(
                _number(
                    sampling_defaults,
                    samp.repetition_penalty,
                    "sampling_defaults",
                    "repetition_penalty",
                )
            ),
            min_p=float(_number(sampling_defaults, samp.min_p, "sampling_defaults", "min_p")),
            max_new_tokens=_int(
                sampling_defaults, samp.max_new_tokens, "sampling_defaults", "max_new_tokens"
            ),
            silence_token_ids=_int_list(sil, "silence_token_ids"),
            min_tokens_floor=_int(eos, samp.min_tokens_floor, "eos_floor", "min_tokens_floor"),
            min_tokens_text_ratio=float(
                _number(eos, samp.min_tokens_text_ratio, "eos_floor", "min_tokens_text_ratio")
            ),
        ),
        window=_window_from(manifest),
        postprocess=postprocess,
    )
