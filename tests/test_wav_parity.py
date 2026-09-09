"""The shared fixtures, held against the reference that wrote them.

``tests/data/conformance/{release_plan,wav_quantise,wav_header,glob,repo_id,
cache_path,edge_fade}.json`` are read by the Go, Rust, JS and Swift suites in
place of tables each port used to transcribe by hand. Python reads them too, so
a fixture that drifted from the code that generates it fails here rather than in
four ports at once.
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from loudkit import hub
from loudkit.synthesis import _encode, _quantise

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "data" / "conformance"
NAMES = (
    "release_plan.json",
    "wav_quantise.json",
    "wav_header.json",
    "glob.json",
    "repo_id.json",
    "cache_path.json",
    "edge_fade.json",
)


def _read(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _value(x: object) -> float:
    if x == "nan":
        return float("nan")
    assert isinstance(x, (int, float)), f"fixture value {x!r} is not a number"
    return float(x)


def test_quantise_is_the_shared_rule() -> None:
    """``floor(x * 32768)``, clipped, NaN to 0: the ten frames four ports pin."""
    cases = _read("wav_quantise.json")["cases"]
    assert len(cases) >= 10
    x = np.array([_value(c["x"]) for c in cases], dtype=np.float32)
    with pytest.warns(RuntimeWarning):  # NaN's cast to int is the probe
        got = _quantise(x).tolist()
    assert got == [c["pcm16"] for c in cases]
    assert got[:10] == [0, 16384, -16384, 29491, -29492, 32767, -32768, 32767, -32768, 0]


def test_result_save_writes_the_quantised_frames(tmp_path: Path) -> None:
    """`Result.save()` and the server's encoder are one writer.

    The fixture above pins `_encode(_quantise())`, the server path; the file a
    user gets from `save()` went through `soundfile` on its own. Same audio,
    same PCM payload, with and without the provenance box.
    """
    from loudkit.provenance import _riff_audio_payload
    from loudkit.result import Result, StageTimings

    rng = np.random.default_rng(3)
    edges = np.array([0.0, 0.5, -0.5, 0.9, -0.9, 1.0, -1.0, 1.5, -1.5, 1e-9], dtype=np.float32)
    audio = np.concatenate([rng.uniform(-1.2, 1.2, 50_000).astype(np.float32), edges])
    result = Result(
        audio=audio,
        tokens=[1, 2, 3],
        mel=np.zeros((80, 2), np.float32),
        seed=0,
        sample_rate=24_000,
        timings=StageTimings(tokens=0.0, mel=0.0, audio=0.0),
    )
    want = _quantise(audio).tobytes()
    for provenance in (False, True):
        path = tmp_path / f"provenance-{provenance}.wav"
        result.save(str(path), include_provenance=provenance)
        assert _riff_audio_payload(path.read_bytes()) == want, f"provenance={provenance}"


def test_the_wav_bytes_are_the_fixtures() -> None:
    for case in _read("wav_header.json")["cases"]:
        x = np.array([_value(v) for v in case["samples"]], dtype=np.float32)
        # A NaN sample warns on the cast; a case without one must not.
        expect = (
            pytest.warns(RuntimeWarning)
            if any(map(math.isnan, x))
            else contextlib.nullcontext()
        )
        with expect:
            wav, media_type = _encode(_quantise(x), case["sample_rate"], "wav")
        assert media_type == "audio/wav"
        assert wav.hex() == case["hex"], case["samples"]


def test_the_fetch_plan_is_the_fixture() -> None:
    fixture = _read("release_plan.json")
    listing = fixture["listing"]
    assert len(fixture["cases"]) == 6
    for case in fixture["cases"]:
        allow, ignore = hub.release_patterns(case["backend"], cloning=case["cloning"])
        assert list(allow) == case["allow"], case
        assert list(ignore) == case["ignore"], case
        wanted = sorted(
            p
            for p in listing
            if any(fnmatch.fnmatch(p, a) for a in allow)
            and not any(fnmatch.fnmatch(p, i) for i in ignore)
        )
        assert wanted == case["wanted"], case


# The four cases that separate fnmatch from a shell glob, answered by hand
# rather than by the module that wrote the fixture. A port matching on path
# segments passes the other twelve and fails these.
GLOB_TRAPS = (
    ("*.safetensors", "voices/en/nested.safetensors", True),
    ("*", "anything/at/all", True),
    ("a?c", "ac", False),
    ("manifest.json", "coreml/x.mlpackage/Manifest.json", False),
)


def test_glob_is_fnmatch() -> None:
    """Two halves, and only the second is about the ports.

    The loop is a stdlib pin: `hub` selects release files with `fnmatch`, the
    four ports hard-code the fixture, so fnmatch semantics moving under them
    would change which files Python downloads and nothing else. It cannot
    catch a wrong fixture, because `tools/make_conformance.py` writes the
    answers with the same call.

    `GLOB_TRAPS` is the half that can. `*` crossing a `/` and `?` refusing to
    match nothing are what a port author reimplements wrongly, and those
    answers are written here by hand.
    """
    cases = _read("glob.json")["cases"]
    assert len(cases) >= 10
    for case in cases:
        assert fnmatch.fnmatch(case["name"], case["pattern"]) is case["match"], case
    stated = {(case["pattern"], case["name"]): case["match"] for case in cases}
    for pattern, name, match in GLOB_TRAPS:
        assert stated.get((pattern, name)) is match, (
            f"the fixture no longer states {pattern!r} vs {name!r} as {match}, "
            "so a port matching on path segments would pass it"
        )


def test_repo_ids_are_the_fixture() -> None:
    cases = _read("repo_id.json")["cases"]
    assert len(cases) >= 10
    for case in cases:
        assert not Path(case["ref"]).exists() or case["ref"] == "", case["ref"]
        assert hub.is_repo_id(case["ref"]) is case["is_repo_id"], case


def test_the_fixtures_are_what_the_generator_writes() -> None:
    """Regenerate into a temporary directory and diff, so a rule that moved in
    Python without a regeneration fails here rather than in four ports."""
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "tools" / "make_conformance.py"),
                "--fixtures-only",
                "--out",
                tmp,
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO,
        )
        assert result.returncode == 0, result.stderr
        for name in NAMES:
            fresh = (Path(tmp) / name).read_text(encoding="utf-8")
            committed = (FIXTURES / name).read_text(encoding="utf-8")
            assert fresh == committed, (
                f"{name} is stale: run tools/make_conformance.py --fixtures-only"
            )
