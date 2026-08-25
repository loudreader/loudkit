"""Where the tests that need real weights look for them.

Two jobs, and the second is the important one.

**Locate.** Every large asset is resolved from an environment variable with a
developer-machine default, so the suite runs unmodified somewhere other than the
machine it was written on.

**Refuse to pass quietly.** A test that skips because an asset is missing looks
exactly like a test that passed, and the parity suite is the one that would be
most damaging to lose that way: it is the evidence that this engine renders the
same audio as the one that shipped. Setting ``LOUDKIT_REQUIRE_ASSETS=1`` turns
every such skip into a failure, and CI sets it. Without that switch, a green run
on a machine with no checkpoint means nothing at all — which is precisely how a
flagship suite quietly stops running.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import pytest

__all__ = ["asset", "requires", "needs_module", "skip_or_fail", "REQUIRE_ASSETS"]

REQUIRE_ASSETS = os.environ.get("LOUDKIT_REQUIRE_ASSETS", "").lower() in {"1", "true", "yes"}

_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "assets"
"""Where the large release assets are: the repository's own ``assets/``.

The default used to be a sibling ``chatterbox-apple`` checkout, which is gone;
its artefacts were copied into ``assets/`` here (gitignored, about 4 GB). The
layout is the release bundle's, flat at the root: the synthesis checkpoint,
``tokenizer.json`` and ``ve.safetensors`` beside ``onnx/`` and ``coreml/``.
Enrollment audio is the checked-in ``tests/data/enrollment/ref_audio.f32``
fixture, so a clean checkout does not depend on an untracked source WAV.
Anywhere else, set ``LOUDKIT_ASSET_ROOT``.
"""
_ROOT = Path(os.environ.get("LOUDKIT_ASSET_ROOT", str(_DEFAULT_ROOT)))

_ASSETS: dict[str, tuple[str, Path]] = {
    "checkpoint": ("LOUDKIT_CHECKPOINT", _ROOT / "loudr-1.safetensors"),
    "tokenizer": ("LOUDKIT_TOKENIZER", _ROOT / "tokenizer.json"),
    "voice_encoder": ("LOUDKIT_VOICE_ENCODER", _ROOT / "ve.safetensors"),
}


# With the switch on, a missing named asset is a broken runner even when a
# test module gates on the path itself (a plain ``skipif(not path.exists())``
# never consults the switch, and two enrollment tests skipped that way on a
# runner that claimed to require assets). Enforced here, at import, so any
# module that resolves an asset through this file trips it before a single
# path-gated mark can quietly evaluate to "skip".
if REQUIRE_ASSETS:
    _absent = [
        f"{name} ({env}={path})"
        for name, (env, default) in _ASSETS.items()
        for path in [Path(os.environ.get(env, str(default)))]
        if not path.exists()
    ]
    if _absent:
        raise RuntimeError(
            "LOUDKIT_REQUIRE_ASSETS is set but these assets are missing: "
            + ", ".join(_absent)
            + ". Set the named variables, or LOUDKIT_ASSET_ROOT."
        )


def asset(name: str) -> Path:
    """Resolve one asset. Does not check that it exists — see :func:`requires`."""
    try:
        env, default = _ASSETS[name]
    except KeyError:
        raise KeyError(f"unknown asset {name!r}; known: {sorted(_ASSETS)}") from None
    return Path(os.environ.get(env, str(default)))


def requires(*names: str) -> pytest.MarkDecorator:
    """Mark a test as needing named assets.

    Skips when they are absent — unless ``LOUDKIT_REQUIRE_ASSETS`` is set, in
    which case it fails, because in CI a missing asset is a broken environment
    and not a reason to report success.
    """
    missing = [n for n in names if not asset(n).exists()]
    if not missing:
        return pytest.mark.skipif(False, reason="")

    detail = ", ".join(f"{n} ({_ASSETS[n][0]}={asset(n)})" for n in missing)
    if REQUIRE_ASSETS:
        # Raised at collection, so the run stops with the reason on screen
        # rather than reporting a pass it did not earn.
        raise RuntimeError(f"LOUDKIT_REQUIRE_ASSETS is set but these are missing: {detail}")
    return pytest.mark.skipif(
        True,
        reason=(
            f"missing {detail}. Set the variable, or LOUDKIT_ASSET_ROOT, or "
            f"LOUDKIT_REQUIRE_ASSETS=1 to make this a failure instead."
        ),
    )


def skip_or_fail(reason: str) -> NoReturn:
    """Skip, unless ``LOUDKIT_REQUIRE_ASSETS`` says a skip is a broken runner.

    ``requires()`` covers the four named large assets and nothing else, so
    ``pytest.importorskip("onnxruntime")`` and ``pytest.skip("graphs missing")``
    stayed unconditional — and those are the two that hid the ONNX backend
    entirely. The switch is meant to mean "a skip here is a failure", not "a
    skip in one of four specific places is a failure".
    """
    if REQUIRE_ASSETS:
        raise AssertionError(f"LOUDKIT_REQUIRE_ASSETS is set but: {reason}")
    pytest.skip(reason)


def requires_modules(*names: str) -> pytest.MarkDecorator:
    """Mark a test as needing importable optional packages.

    The decorator form of :func:`needs_module`, for a whole class whose every
    test needs the same extra. Skips when one is absent, and fails instead when
    ``LOUDKIT_REQUIRE_ASSETS`` is set, on the same reasoning: a runner meant to
    exercise a backend and missing that backend's runtime is broken, not
    passing.

    Ordinary CI installs ``.[dev]`` only, so a test that reaches for torch or
    onnxruntime without this mark is red on every matrix cell.
    """
    missing = [n for n in names if importlib.util.find_spec(n) is None]
    if not missing:
        return pytest.mark.skipif(False, reason="")
    detail = ", ".join(missing)
    if REQUIRE_ASSETS:
        raise RuntimeError(
            f"LOUDKIT_REQUIRE_ASSETS is set but these packages are missing: {detail}"
        )
    return pytest.mark.skip(reason=f"not installed: {detail}")


def needs_module(name: str) -> ModuleType:
    """``importorskip`` that honours ``LOUDKIT_REQUIRE_ASSETS``.

    A runner meant to exercise a backend and missing that backend's runtime is
    a broken runner. Before this, ``tests/test_onnx.py`` said in its own
    docstring that the switch turned its skips into failures; it did not, and
    no CI job installed onnxruntime, so the hand-written numpy mirror of the
    torch decode loop in ``onnx_backend.py`` ran nowhere at all.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        skip_or_fail(f"{name} is not installed ({exc})")
