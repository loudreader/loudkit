"""Documentation integrity: every relative link in the repo's Markdown
resolves, and every code snippet names a real public symbol.

The README and tutorials are the only instructions a stranger gets, so a
broken link or a code example that imports a class that does not exist is a
release defect, not a typo. This module is the mechanical guard: it walks the
Markdown, resolves every relative link from the file's own directory, and
checks the few things a link checker cannot — that `from x import y` names a
real symbol.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import re
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from .assets import asset, requires
from .conftest import tool

REPO = Path(__file__).resolve().parent.parent
MD_FILES = sorted((REPO / "docs").rglob("*.md")) + [
    REPO / "README.md",
    REPO / "RESPONSIBLE_USE.md",
    REPO / "NOTICE",
]
# Non-Markdown files a doc may link to.
_NON_MD = {
    "py",
    "json",
    "safetensors",
    "yml",
    "yaml",
    "toml",
    "ts",
    "go",
    "rs",
    "swift",
    "wav",
    "npy",
    "png",
    "html",
    "bin",
    "sh",
}


def test_all_relative_links_resolve() -> None:
    """Every `](path)` in a Markdown file resolves from that file's directory."""
    broken: list[str] = []
    link_re = re.compile(r"\]\(([^)#]+)(?:#[^)]*)?\)")
    for md in MD_FILES:
        text = md.read_text(encoding="utf-8")
        for m in link_re.finditer(text):
            target = m.group(1)
            if (
                target.startswith("http")
                or target.startswith("#")
                or target.startswith("mailto")
            ):
                continue
            resolved = (md.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{md.relative_to(REPO)} -> {target}")
    assert not broken, "broken relative links:\n  " + "\n  ".join(broken)


def test_github_and_registry_readme_use_the_tracked_wordmark() -> None:
    """The README hero works on GitHub and registries, from a tracked image."""
    image = "assets/logo-wordmark.png"
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    url = f"https://raw.githubusercontent.com/loudreader/loudkit/main/{image}"
    assert f'<img src="{url}"' in readme
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", image],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked.returncode == 0, f"README logo is not tracked: {image}"


_PY_BLOCK_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
# `[\w, ]+` and not `[\w,]+`: the imported-name list is separated by ", ", and a
# class that stopped at the first space checked only the first name of every
# multi-name import. `docs/reference/provenance.md` imports two, and the second
# was never looked at.
_IMPORT_RE = re.compile(r"^from\s+(loudkit\.[\w.]+)\s+import\s+([\w, ]+)", re.MULTILINE)


def _missing_imports(label: str, text: str) -> list[str]:
    """Every `from loudkit.x import Y` in `text` whose module or name is absent."""
    missing: list[str] = []
    for block in _PY_BLOCK_RE.findall(text):
        for m in _IMPORT_RE.finditer(block):
            module_name, names = m.group(1), m.group(2)
            try:
                module = __import__(module_name, fromlist=["*"])
            except ImportError:
                missing.append(f"{label}: cannot import {module_name}")
                continue
            for name in (n.strip() for n in names.split(",") if n.strip()):
                if not hasattr(module, name):
                    missing.append(f"{label}: {module_name} has no {name!r}")
    return missing


def test_code_snippets_import_real_symbols() -> None:
    """`from loudkit.x import Y` inside a ```python code block must name a
    symbol that actually exists — the fastest way a doc can lie about the API.
    Only code blocks are checked; prose that happens to contain the word
    "import" is not code."""

    # Import every module so the symbols are resolvable via getattr.

    missing = [p for md in MD_FILES for p in _missing_imports(md.name, md.read_text("utf-8"))]
    assert not missing, "snippets import missing symbols:\n  " + "\n  ".join(missing)


def test_the_import_gate_reads_past_the_first_name() -> None:
    """A two-name import must be checked in full.

    The name list is separated by ", ", and the pattern used to stop at the
    space: a snippet importing two symbols had its second one waved through,
    including the one at `docs/reference/provenance.md:53`. Both exist today,
    so the hole never showed as a red test — which is why it is pinned here.
    """
    block = "```python\nfrom loudkit.provenance import read_provenance, no_such_symbol\n```"
    assert _missing_imports("made-up.md", block) == [
        "made-up.md: loudkit.provenance has no 'no_such_symbol'"
    ]


def test_no_broken_local_md_targets() -> None:
    """Every relative link in every *tracked* markdown file resolves to a
    *tracked* file.

    Both halves of that sentence are the test. It used to read README.md only,
    and to ask the filesystem whether the target existed — which is the wrong
    question twice over, and it missed both bugs it should have caught:
    `CONTRIBUTING.md` sent a first-time contributor to `openspec/PLAN.md` and a
    tutorial linked `../../openspec/IDENTITY-CONTRACT.md`, while `openspec/` is
    gitignored. Locally the directory is right there, so the filesystem said
    yes; on GitHub, where only tracked files exist, both were 404s. Asking git
    what it ships is the only check that reproduces what a stranger sees.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    shipped = {p for p in tracked if p}
    docs = sorted(p for p in shipped if p.endswith(".md"))

    link_re = re.compile(r"\]\(([^)#]+?)(?:#[^)]*)?\)")
    broken: list[str] = []
    for rel in docs:
        text = (REPO / rel).read_text(encoding="utf-8")
        parent = PurePosixPath(rel).parent
        for m in link_re.finditer(text):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            # `posixpath.normpath`, not `os.path.normpath`: both sides of the
            # comparison below are git's own paths, and git speaks forward
            # slashes on every platform. The `os` variant normalises to the
            # host separator, so on Windows every link resolved to
            # `docs\reference\errors.md`, matched nothing in `shipped`, and the
            # test reported all 47 links in the repository as broken.
            resolved = posixpath.normpath(str(parent / target))
            if resolved not in shipped and f"{resolved}/" not in {
                f"{PurePosixPath(p).parent}/" for p in shipped
            }:
                broken.append(f"{rel} -> {target}")
    assert not broken, "links to files git does not ship:\n  " + "\n  ".join(broken)


def test_voices_table_matches_provenance() -> None:
    """VOICES.md is generated, and this is the pin that keeps it honest.

    The table is the attribution made readable: donor or source, licence and
    sample per voice. It is produced by `tools/build_voices_md.py` from
    `docs/voices/roster/provenance.json`, one step later — so editing either
    without regenerating is a build failure here rather than a quiet lie on
    the page someone reads before trusting the roster.

    Rendered in memory rather than via `main()`: a test that refreshes the
    file it checks would pass silently while rewriting the working tree.
    """
    import json as _json

    module = tool("build_voices_md")

    voices = _json.loads(
        (REPO / "docs" / "voices" / "roster" / "provenance.json").read_text(encoding="utf-8")
    )
    expected = module.render(voices)
    on_disk = (REPO / "VOICES.md").read_text(encoding="utf-8")
    assert on_disk == expected, (
        "VOICES.md does not match tools/build_voices_md.py output — "
        "run `python tools/build_voices_md.py` and commit the result"
    )


def test_preview_model_comparisons_match_downloaded_profiles() -> None:
    """Each model sample identifies the repaired profile users can download."""
    import hashlib
    import json

    root = REPO / "docs" / "voices" / "preview"
    voices = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    for voice in voices:
        profile_sha = None
        if "profile" in voice:
            profile = voice["profile"]
            profile_sha = hashlib.sha256((root / profile["path"]).read_bytes()).hexdigest()
            assert profile_sha == profile["sha256"], voice["name"]
        for case in ("short", "long"):
            base = voice["sample"][case]
            turbo = voice["sample_turbo"][case]
            assert base["text"] == turbo["text"]
            assert base["seed"] == turbo["seed"] == 7
            for sample, model in ((base, "loudr-1"), (turbo, "loudr-1-turbo")):
                assert sample["model"] == model
                assert (
                    hashlib.sha256((root / sample["audio"]).read_bytes()).hexdigest()
                    == sample["sha256"]
                )
                if profile_sha is not None:
                    assert sample["profile_sha256"] == profile_sha


def test_roster_audio_files_exist() -> None:
    """Every provenance entry names an audio file — the file must be there."""
    import json as _json

    provenance = REPO / "docs" / "voices" / "roster" / "provenance.json"
    voices = _json.loads(provenance.read_text(encoding="utf-8"))
    missing = [
        v["name"] for v in voices if not (provenance.parent / v["sample"]["audio"]).is_file()
    ]
    assert not missing, f"roster entries whose sample audio is missing: {missing}"


def test_roster_paths_and_committed_digests_are_complete() -> None:
    """Every public path is exact, and committed previews match their digests.

    The roster used to name ``voices/profiles/<name>`` although the model repo
    has always shipped ``voices/<name>``.  It also called an unpublished source
    WAV an ``hf_path``.  A provenance record may identify an unpublished input,
    but it cannot claim that input is available at a path where it is not.

    Voice profiles are model artefacts and deliberately not in git. Their byte
    check is a separate slow test below, run by the asset-backed parity lane.
    Keeping the static half here means a fresh source checkout still validates
    every claim it actually contains instead of failing because private local
    files happened not to be present.
    """
    import hashlib as _hashlib
    import json as _json

    provenance = REPO / "docs" / "voices" / "roster" / "provenance.json"
    voices = _json.loads(provenance.read_text(encoding="utf-8"))
    problems: list[str] = []

    def digest(path: Path) -> str:
        return _hashlib.sha256(path.read_bytes()).hexdigest()

    for voice in voices:
        name = voice["name"]
        profile = voice["profile"]
        expected_profile = f"voices/{name}.safetensors"
        if profile.get("hf_path") != expected_profile:
            problems.append(f"{name}: profile path is {profile.get('hf_path')!r}")
        profile_digest = profile.get("sha256", "")
        if len(profile_digest) != 64 or any(
            c not in "0123456789abcdef" for c in profile_digest
        ):
            problems.append(f"{name}: profile digest is not a SHA-256")

        sample = voice["sample"]
        sample_path = provenance.parent / sample["audio"]
        if not sample_path.is_file() or digest(sample_path) != sample.get("sha256"):
            problems.append(f"{name}: sample digest does not match {sample.get('audio')}")

        reference = voice["reference"]
        preview = provenance.parent / reference.get("public_preview", "")
        if "hf_path" in reference or reference.get("published_in_model_repo") is not False:
            problems.append(f"{name}: unpublished reference claims a model-repo path")
        if reference.get("source_filename") != f"{name}.wav":
            problems.append(f"{name}: reference does not name its source WAV")
        source_digest = reference.get("sha256", "")
        if len(source_digest) != 64 or any(c not in "0123456789abcdef" for c in source_digest):
            problems.append(f"{name}: source reference digest is not a SHA-256")
        if not preview.is_file():
            problems.append(f"{name}: public reference preview is missing")

    assert not problems, "roster provenance disagrees with published bytes:\n  " + "\n  ".join(
        problems
    )


def _voice_profiles() -> list[Path]:
    root = Path(os.environ.get("LOUDKIT_ASSET_ROOT", str(REPO / "assets")))
    return sorted((root / "voices").glob("*.safetensors"))


@pytest.mark.slow
def test_published_voice_profile_digests_match_the_roster() -> None:
    """The shipped voice profiles match the provenance committed beside them."""
    import hashlib as _hashlib
    import json as _json

    profiles = _voice_profiles()
    if not profiles:
        from .assets import skip_or_fail

        skip_or_fail("no voice profiles under LOUDKIT_ASSET_ROOT/voices")
    assert len(profiles) == 28, f"expected 28 voice profiles, found {len(profiles)}"

    provenance = REPO / "docs" / "voices" / "roster" / "provenance.json"
    voices = _json.loads(provenance.read_text(encoding="utf-8"))
    expected = {v["profile"]["hf_path"]: v["profile"]["sha256"] for v in voices}
    actual = {
        f"voices/{path.name}": _hashlib.sha256(path.read_bytes()).hexdigest()
        for path in profiles
    }
    assert actual == expected, "voice profile bytes disagree with the provenance roster"


@pytest.mark.slow
def test_public_docs_quote_the_measured_voice_profile_size() -> None:
    """The shipped profiles average about 150 KB, not the old 300 KB estimate."""
    profiles = _voice_profiles()
    if not profiles:
        from .assets import skip_or_fail

        skip_or_fail("no voice profiles under LOUDKIT_ASSET_ROOT/voices")
    assert len(profiles) == 28
    average = sum(path.stat().st_size for path in profiles) / len(profiles)
    assert 100_000 <= average <= 200_000

    states_the_size = (
        "README.md",
        "SUPPORTED.md",
        "docs/MODEL_CARD.md",
        "notebooks/loudkit_quickstart.ipynb",
        "python/loudkit/hub.py",
    )
    for rel in states_the_size:
        text = (REPO / rel).read_text(encoding="utf-8")
        assert "150 KB" in text, f"{rel} does not state the measured profile size"
    for rel in states_the_size + ("python/loudkit/__init__.py",):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert "300 KB" not in text, f"{rel} still carries the old size estimate"


def test_first_download_claims_match_the_split_release() -> None:
    """The first synthesis fetch is 747 MB; cloning adds the 523 MB half.

    These five surfaces all described the old packed checkpoint after the hub
    resolver had begun fetching only the synthesis half. Keep the exact list:
    each one is either a user's first run or an operator-facing explanation of
    why the model is mounted and kept warm.
    """
    surfaces = (
        "Dockerfile",
        "compose.yaml",
        "integrations/speech-dispatcher/loudkit.conf",
        "notebooks/loudkit_quickstart.ipynb",
        "pyproject.toml",
    )
    for rel in surfaces:
        text = (REPO / rel).read_text(encoding="utf-8")
        assert "1.27 GB" not in text, f"{rel} still describes the packed checkpoint"
        assert re.search(r"74[7-9] MB|750 MB", text), f"{rel} omits the synthesis size"

    notebook = (REPO / "notebooks/loudkit_quickstart.ipynb").read_text(encoding="utf-8")
    assert "523 MB" in notebook, "the notebook does not explain the cloning download"


def _split_args(text: str) -> list[str]:
    """Split on commas at depth zero, so a struct literal counts as one.

    Go groups names (`a, b string` is two parameters) and the front door takes
    an options struct (`Options{Seed: 7}`), so counting declarations or plain
    commas is wrong in opposite directions.
    """
    parts, depth, buf = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)
    return [p for p in parts if p.strip() and not p.strip().startswith("//")]


def _method_arity(source: str, receiver: str, method: str) -> int | None:
    """Count the parameters of ``method`` in ``source``, receiver excluded."""
    import re as _re

    for pattern in (
        # Go:   func (e *Engine) SynthesizeLong(a A, b B) (…)
        rf"func \(\w+ \*?{receiver}\) {method}\((.*?)\)\s*\(",
        # Rust: pub fn synthesize(&self, a: A, b: B) -> …
        rf"pub fn {method}\(\s*&(?:mut )?self,(.*?)\)\s*->",
    ):
        m = _re.search(pattern, source, _re.DOTALL)
        if m:
            return len(_split_args(m.group(1)))
    return None


def test_binding_snippets_call_the_real_signatures() -> None:
    """A tutorial's, and a package README's, Go and Rust snippets must match.

    Both engines once gained a `shouldCancel` parameter that the tutorials never
    picked up, so the Go and Rust guides shipped snippets that do not compile
    ("not enough arguments in call to eng.Synthesize"). Nothing noticed because
    `test_code_snippets_import_real_symbols` above only reads ```python blocks.

    This counts arguments at the documented call site against parameters at the
    definition of the front door, `Synthesize` in Go and `synthesize` in Rust,
    the any-length call every page shows. It is not a compiler; it is the check
    that would have caught the drift that actually happened.
    """
    import re as _re

    go_call = _re.compile(r"\w+\.Synthesize\(\s*(.*?)\)\s*$", _re.DOTALL | _re.MULTILINE)
    rust_call = _re.compile(r"\w+\.synthesize\((.*?)\)\?", _re.DOTALL)
    cases = [
        (
            REPO / "docs/guides/08-go.md",
            REPO / "go/engine/engine.go",
            "Engine",
            "Synthesize",
            go_call,
        ),
        (
            REPO / "docs/guides/09-rust.md",
            REPO / "rust/src/engine.rs",
            "Engine",
            "synthesize",
            rust_call,
        ),
        # The registry READMEs carry the same call, and RELEASING.md's
        # acceptance pass runs *those* examples against a published package:
        # a drifted README example fails in a stranger's scratch project.
        (REPO / "go/README.md", REPO / "go/engine/engine.go", "Engine", "Synthesize", go_call),
        (
            REPO / "rust/README.md",
            REPO / "rust/src/engine.rs",
            "Engine",
            "synthesize",
            rust_call,
        ),
    ]
    problems: list[str] = []
    for doc, impl, receiver, method, call_re in cases:
        want = _method_arity(impl.read_text(encoding="utf-8"), receiver, method)
        assert want is not None, f"cannot find {method} in {impl.name}"
        where = doc.relative_to(REPO)
        call = call_re.search(doc.read_text(encoding="utf-8"))
        if call is None:
            problems.append(f"{where}: no {method} call to check")
            continue
        args = _split_args(call.group(1))
        if len(args) != want:
            problems.append(
                f"{where}: {method} is called with {len(args)} arguments, "
                f"but {impl.name} declares {want}"
            )
    assert not problems, "documented call sites have drifted:\n  " + "\n  ".join(problems)


def test_the_readme_does_not_duplicate_the_parity_table() -> None:
    """The front page never carries a copy of the generated report's table."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "<!-- parity-table:" not in readme


def test_every_parity_row_carries_a_gate_and_a_measurement() -> None:
    """A row with an empty cell is a claim with the evidence left out.

    `not measured` is an acceptable measurement — it says the environment could
    not run that comparison. An *empty* cell is not: it reads as a filled-in
    table to anyone skimming.
    """
    generated = (REPO / "docs" / "parity-measured.md").read_text(encoding="utf-8")
    body = generated[generated.index("| stage |") :]
    rows = body[: body.index("\n\n")].strip().splitlines()[2:]  # drop header + rule

    problems = []
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        assert len(cells) == 4, f"malformed row: {row}"
        if not all(cells):
            problems.append(row)
        if cells[3].startswith("✗"):
            problems.append(f"{row}  (a failing gate is committed)")
    assert not problems, "parity rows with missing or failing values:\n  " + "\n  ".join(
        problems
    )


def test_the_conformance_fixture_is_what_its_generator_produces() -> None:
    """Regen-and-diff, so the fixture and its generator cannot drift apart.

    They had. `tools/make_speechtext_fixture.py` is documented as the way to
    regenerate `tests/data/conformance/speechtext.json`, and cases had been
    added to the JSON by hand instead: running the generator as documented
    deleted 23 parity cases — every acronym, date, ordinal and NFC pin — along
    with 35 explanations and the whole `divergent` block, and left a file that
    still said `generated_by` at the top.

    Nothing caught it, because nothing had reason to run the generator. This
    does, on every test run, and it is the assertion that makes the fixture
    safe to regenerate: a legitimate engine change updates the JSON and this
    test goes green again, while a hand edit to the JSON alone fails here with
    the file to re-run named in the message.

    Note what this does *not* claim. The fixture's expected values come from
    Python, so this checks that the five ports are held to a fixture Python can
    reproduce — not that Python is right. That second question is what the
    hand-written per-language tests are for.
    """
    gen = tool("make_speechtext_fixture")
    committed = (REPO / "tests" / "data" / "conformance" / "speechtext.json").read_text(
        encoding="utf-8"
    )
    assert committed == gen.rendered(gen.build_payload()), (
        "tests/data/conformance/speechtext.json is not what its generator writes. "
        "Re-run `.venv/bin/python tools/make_speechtext_fixture.py`, and if the "
        "diff is a case you added by hand, add it to CASES in the generator instead."
    )


def test_every_path_notice_names_exists() -> None:
    """NOTICE attributes upstream work to files — so the files must be there.

    Attribution that points at a path nobody can open is attribution nobody can
    check, and this had rotted quietly: after the move to one directory per
    language, NOTICE still credited `src/loudkit/`, `Sources/LoudKit/` and
    `integrations/rust/src/`, none of which exist. The four copies stayed
    byte-identical to each other the whole time, so the CI check that compares
    them was green — they were identically wrong.

    What counts as a path: backticked, on one line, no spaces, and containing a
    separator. A bare word in backticks is a symbol, and the quoted upstream
    licence texts contain backticks of their own — an unconstrained match hands
    `os.stat` a paragraph of the CMU disclaimer. A `*` glob has to match
    something.
    """
    import re

    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    named = {
        m
        for m in re.findall(r"`([^`\n]+)`", notice)
        if "/" in m and " " not in m and not m.startswith(("http://", "https://"))
    }

    # Paths that belong to a *named upstream repository*, not to this one.
    # NOTICE says which repository in the surrounding sentence; the check cannot
    # read that, so the exception is written down here with the same answer.
    upstream = {"export/export_enrollment.py": "chatterbox-apple"}

    missing = sorted(
        path
        for path in named
        if path not in upstream
        and not (list(REPO.glob(path)) if "*" in path else (REPO / path).exists())
    )
    assert not missing, "NOTICE names paths that do not exist: " + ", ".join(missing)


def test_every_port_reads_every_postprocess_field() -> None:
    """The four hand-written manifest parsers, against the one that is derived.

    Python reads `PostprocessConfig` off the dataclass, and its docstring says
    why: "a hand-written wall is a list that a new constant gets left out of,
    and a constant the loader silently ignores is a manifest declaring one
    recipe while the engine runs another." The four ports write that wall by
    hand because their languages give them no cheap equivalent.

    Every one of them had drifted the same six fields behind — `pacing_tolerance`,
    `retry_max_attempts`, `dropout_min_tokens` and the three `repetition_*` — and
    nothing noticed, because the defaults agree. A checkpoint that sets one is
    all it takes for the manifest to declare one recipe and four engines to run
    another, which is the founding defect of this project arriving through a
    parser rather than through the funnel.

    A source-text check, not a behavioural one: the ports cannot be asked what
    keys they read without a checkpoint, and the failure this guards against is
    a name that is never mentioned. Crude, and it fails the moment someone adds
    a field to the dataclass without adding it to four files, which is exactly
    when it should.
    """
    import re
    from dataclasses import fields as dataclass_fields

    from loudkit.postprocess import PostprocessConfig

    parsers = {
        "go": REPO / "go" / "config" / "config.go",
        "rust": REPO / "rust" / "src" / "checkpoint.rs",
        "js": REPO / "js" / "src" / "types.ts",
        "swift": REPO / "swift" / "LoudKit" / "Config.swift",
    }
    # The render censuses (`silence_render_ids`, `quiet_render_ids`) are not
    # postprocess-block fields — they live at the manifest top level beside
    # `silence_token_ids`, and every parser refuses them inside the block by
    # name — but the refusal and the top-level read both live in these files,
    # so the source-text check covers them anyway.
    names = [f.name for f in dataclass_fields(PostprocessConfig)]

    def reads(src: str, name: str) -> bool:
        # Quoted is the normal shape (`block["pacing_tolerance"]`). The word on
        # its own covers the one field nobody looks up by string: `mode` is
        # read positionally in every port, as `block.mode` in JS, because it is
        # the only non-numeric one and the only one that can be refused.
        return f'"{name}"' in src or re.search(rf"\b{name}\b", src) is not None

    missing = {
        port: [n for n in names if not reads(path.read_text(encoding="utf-8"), n)]
        for port, path in parsers.items()
    }
    assert not any(missing.values()), (
        "these ports do not read every postprocess field from the manifest: "
        + "; ".join(f"{port}: {', '.join(fs)}" for port, fs in missing.items() if fs)
    )


def test_every_port_reads_every_chunking_field() -> None:
    """The same wall as above, for the block that decides where the reader breathes.

    `postprocess` got this test after four ports had drifted six fields behind
    it. `chunking` is the same shape of hazard and had no guard at all: Python
    builds `ChunkConfig` from the manifest field by field, and the four ports
    write that read by hand. A field they do not name is a field a checkpoint
    can declare and five engines can split differently on, under a
    `recipe_version` that says they agree.

    Cheaper than the postprocess version because `ChunkConfig` is small, and
    the same crude source-text check for the same reason: the ports cannot be
    asked what keys they read without a checkpoint, and the failure this guards
    against is a name that is never mentioned.
    """
    import re
    from dataclasses import fields as dataclass_fields

    from loudkit.config import ChunkConfig

    parsers = {
        "go": REPO / "go" / "config" / "config.go",
        "rust": REPO / "rust" / "src" / "checkpoint.rs",
        "js": REPO / "js" / "src" / "types.ts",
        "swift": REPO / "swift" / "LoudKit" / "Config.swift",
    }
    # `first_chunk_max_tokens` is Python-only and unset by default, so it is
    # absent from the fingerprint and from the ports. It takes the usual route
    # when it ships: reference, then fixture, then ports.
    names = [
        f.name for f in dataclass_fields(ChunkConfig) if f.name != "first_chunk_max_tokens"
    ]

    def strip_comments(src: str) -> str:
        src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
        return "\n".join(re.sub(r"(//|#).*$", "", ln) for ln in src.splitlines())

    def camel(name: str, upper: bool) -> str:
        head, *rest = name.split("_")
        return (head.title() if upper else head) + "".join(w.title() for w in rest)

    def assigns(port: str, code: str, name: str) -> bool:
        """Does the parsed value actually reach the config field?

        Mentioning the key is not reading it. The first version of this test
        searched the whole file for the name, so a port could stop parsing a
        key and satisfy the wall from a comment -- and after comments were
        stripped, from the error message that names the key it no longer reads.
        Demonstrated both ways against Go. What cannot be faked by prose is the
        assignment: a field the parser never writes is a field the manifest
        cannot set, whatever the file says about it.
        """
        if port == "go":
            return re.search(rf"\.{camel(name, True)}\s*=", code) is not None
        if port == "rust":
            return re.search(rf"\.{name}\s*=", code) is not None
        if port == "js":
            return re.search(rf"\b{camel(name, False)}\s*:", code) is not None
        return re.search(rf"\.{camel(name, False)}\s*=", code) is not None  # swift

    def looks_up(port: str, code: str, name: str) -> bool:
        """Is the manifest key itself named, in the shape that reads it?

        Go, Rust and Swift index a map with the quoted key. JS may do either:
        it reads some blocks as properties and passes the key to a checked
        reader for others, and the second shape is the stronger one, being
        what the other three are held to here. The quoted form is required
        for the three that use it, because a port can rename the key it looks
        for and still mention the old name in the error message it raises
        about it -- which is exactly how a broken Go parser satisfied the
        looser check.
        """
        if port == "js":
            return (
                re.search(rf"\.{name}\b", code) is not None
                or re.search(rf'"{name}"\s*,', code) is not None
            )
        return f'"{name}"' in code

    def reads(port: str, src: str, name: str) -> bool:
        code = strip_comments(src)
        return looks_up(port, code, name) and assigns(port, code, name)

    missing = {
        port: [n for n in names if not reads(port, path.read_text(encoding="utf-8"), n)]
        for port, path in parsers.items()
    }
    assert not any(missing.values()), (
        "these ports do not read every chunking field from the manifest: "
        + "; ".join(f"{port}: {', '.join(fs)}" for port, fs in missing.items() if fs)
    )


def test_the_shipping_abbreviations_are_hand_written_the_same_in_five_places() -> None:
    """The list is data, and it is data five implementations each hold a copy of.

    A port whose copy has drifted splits the passage somewhere else while the
    fingerprint declares the list it is not applying. The fingerprint would
    catch a drift in the *shipping* copy, but only through a hash: this names
    the port and the entry.

    Matched as one ordered sequence of quoted literals rather than entry by
    entry, and that is the whole strength of the check. Half the list is one and
    two characters long, so `'"D"' in source` is satisfied by any stray `"D"`
    anywhere in the file. Requiring `"A", "B", "Cpn", ...` in order, with only
    whitespace and commas between, fails on a missing entry, on a reordering,
    and on an entry inserted *inside* the run.

    What it does not catch, said plainly because an earlier version of this
    docstring claimed otherwise: an entry appended after the run, or prepended
    before it, or the real list gutted while the canonical order survives in a
    comment. The search finds the sequence anywhere in the file and does not
    care what surrounds it. Each of those is caught instead by the port's own
    fingerprint-against-fixture test, which hashes the list the port actually
    holds -- so this wall names the entry where those name only a hash, and is
    redundant rather than load-bearing.

    `Dr` and `St` are in the list on a second measurement rather than the
    ten-language survey's, whose English sample contained neither: on 24 books
    of English prose the list without them still ended a chunk on `St.` 178
    times and on `Dr.` 80.
    """
    import re

    from loudkit.config import ChunkConfig

    entries = ChunkConfig().abbreviations
    assert {"Dr", "St"} <= set(entries), (
        "Dr and St were measured into this list; a regeneration must not drop them"
    )
    copies = {
        "go": REPO / "go" / "chunking" / "chunking.go",
        "rust": REPO / "rust" / "src" / "chunking.rs",
        "js": REPO / "js" / "src" / "types.ts",
        "swift": REPO / "swift" / "LoudKit" / "Config.swift",
    }
    # Each language spells a non-ASCII literal its own way — `\u015b` in Go and
    # TypeScript, `\u{15b}` in Rust, `\u{15B}` in Swift — so the escapes are
    # decoded before the comparison rather than enumerated. That decoding is
    # load-bearing since the `ś` divergence: the ports emitted raw UTF-8 into
    # the canonical form where the reference emits an escape.
    escape = re.compile(r"\\u\{?([0-9a-fA-F]{1,6})\}?")
    sequence = re.compile(r"[\s,]*".join(re.escape(f'"{entry}"') for entry in entries))

    for port, path in copies.items():
        src = escape.sub(lambda m: chr(int(m.group(1), 16)), path.read_text(encoding="utf-8"))
        assert sequence.search(src), (
            f"{port} does not carry the shipping abbreviations as one ordered list; "
            f"the five copies have drifted. Expected, in order: {list(entries)}"
        )


def test_the_production_fingerprint_is_pinned_in_one_value_everywhere() -> None:
    """Every copy of the shipped fingerprint, against the golden fixture.

    Places that pin the production fingerprint as a literal: this suite's
    weighted public-API test, `docs/reference/IDENTITY-CONTRACT.md`,
    `docs/reference/COMPATIBILITY.md` and `docs/platforms/apple.md`. They went
    stale together the moment the shared grammar file changed, and nothing
    noticed for a session's worth of commits, because the only test among them
    is `@pytest.mark.slow` and asset-gated: it does not run without a
    checkpoint, which is precisely when the value moves. No user page pins it:
    the server guide shows the header with a placeholder.

    This runs without one. The golden fixture carries the same fingerprint and
    is regenerated whenever the config does, so comparing the literals against
    it catches the drift at the commit that causes it rather than at whoever
    next has the weights.
    """
    import json
    import re

    golden = json.loads(
        (REPO / "tests" / "data" / "conformance" / "vectors.json").read_text(encoding="utf-8")
    )["algorithm"]["fingerprint"]

    pinned = {
        REPO
        / "docs"
        / "reference"
        / "IDENTITY-CONTRACT.md": r"Production fingerprint `([0-9a-f]{16})`",
        REPO
        / "docs"
        / "reference"
        / "COMPATIBILITY.md": r"moves `[0-9a-f]{16}` to\s+`([0-9a-f]{16})`",
        REPO / "docs" / "platforms" / "apple.md": r"independently and agree: `([0-9a-f]{16})`",
        # The release note that tells a reader their audio changed. It names
        # the value they are moving *to*, so only the second of the pair is
        # checked here; the first is 0.1.0's and is deliberately historical.
        REPO / "CHANGELOG.md": r"The fingerprint moves: `[0-9a-f]{16}` -> `([0-9a-f]{16})`",
    }
    stale = {}
    for path, pattern in pinned.items():
        found = re.findall(pattern, path.read_text(encoding="utf-8"))
        assert found, f"{path.relative_to(REPO)}: no pinned fingerprint matched {pattern!r}"
        bad = [f for f in found if f != golden]
        if bad:
            stale[str(path.relative_to(REPO))] = bad

    assert not stale, (
        f"these pin a fingerprint other than the golden {golden}: {stale}. "
        "Regenerate the fixture, then update every copy."
    )


def test_the_headline_speed_figures_are_quoted_from_one_page() -> None:
    """Every copy of a benchmark figure, against `docs/benchmarks.md`.

    Several places print the same real-time factors: the benchmark page, the
    README, the model card, the landing page's cards, the CUDA card's table and
    two of the port guides. Nothing checked that they agree, and by 0.1.1 they
    did not agree with the release either — the whole set is 0.1.0's, while the
    branch's own table in the changelog reports a faster engine, and one of the
    stale figures (`7.47x` on a 3090) happens to equal a *different* row of the
    newer table. A reader comparing the two concludes the release changed
    nothing.

    The fingerprint has a test like this because four copies of it went stale
    together and nothing noticed for a session's worth of commits. Speed
    figures are quoted in more places than the fingerprint is.

    This pins agreement, not currency: it cannot tell whether the numbers are
    today's. `docs/benchmarks.md` states the release its rows were taken on,
    and `RELEASING.md` makes re-measuring them a step, because the two machines
    that hold the record are not in CI.
    """
    import re

    page = (REPO / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    # One row per measured path: the loudr-1 cell always carries a figure, the
    # turbo cell carries one or says it is not measured yet.
    quick: dict[str, tuple[str, str | None]] = {}
    for path, base, turbo in re.findall(
        r"^\| [^|]+ \| ([^|]+?) \| ([0-9.]+)x[^|]* \| ([^|]+?) \|$", page, re.MULTILINE
    ):
        found = re.match(r"([0-9.]+)x", turbo)
        quick[path] = (base, found.group(1) if found else None)
    assert len(quick) >= 6, f"the quick-answer table did not parse: {quick}"

    def figure(fragment: str) -> tuple[str, str | None]:
        matched = [v for k, v in quick.items() if fragment in k]
        assert len(matched) == 1, f"{fragment!r} matched {matched} in the quick answer"
        return matched[0]

    # (row of benchmarks.md, files that re-quote it). Every row of the quick
    # answer is here, including the two the first version of this test left
    # out — the Swift figure and the batched A100 one, which are quoted in the
    # guides and the model card and were checked by nothing.
    # The model cards carry no speed figure at all: they point at the
    # benchmark page, so there is nothing on them to drift.
    quoted = {
        "RTX 3090": [
            "README.md",
            "site/src/components/CudaCard.astro",
        ],
        "Jetson Orin Nano": [
            "README.md",
            "site/src/handwritten/index.mdx",
        ],
        "split PyTorch engine": [
            "README.md",
            "site/src/handwritten/index.mdx",
        ],
        "ONNX Runtime CPU provider": [
            "README.md",
            "site/src/handwritten/index.mdx",
        ],
        "native generator plus CoreML renderer": [
            "README.md",
            "docs/platforms/apple.md",
            "docs/guides/10-swift.md",
        ],
        "token generator, A100": [
            "README.md",
        ],
    }
    missing = []
    for fragment, files in quoted.items():
        base, turbo = figure(fragment)
        for rel in files:
            text = (REPO / rel).read_text(encoding="utf-8")
            for model, want in (("loudr-1", base), ("loudr-1-turbo", turbo)):
                if want is not None and want not in text:
                    missing.append(f"{rel} does not quote {want}x for {fragment} ({model})")
    assert not missing, (
        "docs/benchmarks.md is the one page these are measured on, and these "
        "copies have drifted from it:\n  " + "\n  ".join(missing)
    )


# Every section of every page that publishes a real-time factor, and the
# passage that has to date it. An inventory rather than one regex per file: a
# page carries figures in more than one place — the README has a table *and* a
# batch-throughput paragraph — and matching from the sentence after the table
# left the table itself unchecked while the test's own docstring claimed
# otherwise.
_SPEED_SECTIONS: dict[str, dict[str, str]] = {
    "README.md": {
        "the end-to-end table": r"## Measured speed.*?\n\nFor batched workloads",
        "batch throughput": r"For batched workloads.*?\n\n",
    },
    "docs/benchmarks.md": {
        "the quick answer": r"## Measured on which release[\s\S]*?\n## End-to-end",
    },
    "docs/platforms/jetson.md": {
        "what to expect": r"## What to expect[\s\S]*?(?=\n## |\Z)",
    },
    "docs/platforms/apple.md": {
        "the Swift port against the Python engine": r"The port is measured.*?\n\n",
    },
    "docs/guides/10-swift.md": {
        "the Swift port against the Python engine": (
            r"whole pipeline runs the third benchmark passage.*?\n\n"
        ),
    },
    "docs/guides/08-go.md": {"the CUDA provider": r"CUDA measured.*?\n\n"},
    "docs/guides/09-rust.md": {"the CUDA provider": r"CUDA measured.*?\n\n"},
    "site/src/handwritten/index.mdx": {
        # The whole section: the cards carry the figures and the foot carries
        # the epoch, so matching only the foot checked a passage with no
        # numbers in it — which the figures assertion above now refuses.
        "the where-it-runs cards": r'<section id="where-it-runs">[\s\S]*?</section>',
    },
}


@pytest.mark.parametrize(
    ("rel", "section"),
    [(rel, section) for rel, sections in _SPEED_SECTIONS.items() for section in sections],
)
def test_a_published_speed_figure_says_which_release_it_is(rel: str, section: str) -> None:
    """A number with no epoch is a number a reader dates to today.

    Some real-time factors are 0.1.1's and some are still 0.1.0's. Both are
    allowed, since the two GPU boxes are not in CI and re-measuring them is a
    release step, but only if each copy says which, because the alternative
    is a release that is measurably faster and advertises the previous one's
    numbers.

    Two earlier versions of this were weaker than they read. The first looked
    for the marker anywhere in the file, which passed for four pages carrying
    figures in a section the marker was nowhere near. The second bound it to a
    passage but took one passage per page, so the README's table and its batch
    paragraph were represented by whichever the regex happened to reach. This
    takes an inventory: every section, named, and each one checked for figures
    before it is checked for the marker — so a section that stops carrying
    numbers fails loudly rather than passing vacuously.
    """
    import re

    text = (REPO / rel).read_text(encoding="utf-8")
    found = re.search(_SPEED_SECTIONS[rel][section], text, re.DOTALL)
    assert found, f"{rel}: the {section!r} section has moved or gone"
    passage = found.group(0)
    # `&times;` on the site, `x` in the markdown: the same figure, two spellings.
    assert re.search(r"[0-9]+\.[0-9]+(?:x|&times;)", passage), (
        f"{rel}: {section!r} matched a passage with no real-time factor in it, so "
        f"this case is checking nothing"
    )
    assert re.search(r"measured on 0\.1\.[01]\b", passage), (
        f"{rel} publishes real-time factors in {section!r} without saying which "
        f"release they were measured on:\n{passage.strip()[:400]}"
    )


def _fresh_parity_table(checkpoint: Path | None) -> str:
    """The table the generator writes now, into a temporary directory."""
    import subprocess
    import sys
    import tempfile

    argv = [sys.executable, str(REPO / "tools" / "parity_table.py")]
    if checkpoint is not None:
        argv += ["--checkpoint", str(checkpoint)]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "parity-measured.md"
        result = subprocess.run(
            [*argv, "--out", str(out)], capture_output=True, text=True, check=False, cwd=REPO
        )
        assert result.returncode == 0, f"generator failed:\n{result.stdout}\n{result.stderr}"
        return out.read_text(encoding="utf-8")


def _parity_rows(text: str) -> dict[str, str]:
    """The table's rows by stage. The `Environment:` line above it is not a row:
    it names the commit the page was made at, and every commit moves it."""
    body = text[text.index("| stage |") :]
    table = body[: body.index("\n\n")].strip()
    rows = {}
    for line in table.splitlines()[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows[cells[0]] = line.strip()
    return rows


def _committed_parity_rows() -> dict[str, str]:
    return _parity_rows((REPO / "docs" / "parity-measured.md").read_text(encoding="utf-8"))


def test_the_parity_table_is_what_the_generator_produces() -> None:
    """Regenerate the table and diff it, the way the fixture is already guarded.

    The parity table has gone stale in both directions during this project's
    life: it claimed 49/49 when the fixture held 85, and 85/85 when it held 88.
    Each time it was corrected by hand and each time it drifted again, because
    the number moves whenever a case is added and nothing connected the two.

    The shared fixture got a regen-and-diff test for exactly this reason and the
    table did not. This is that test. It runs the generator into a temporary
    directory and compares, so a case added to the fixture without regenerating
    the table fails here rather than in the next review.

    Weight-free: the seven weighted rows come back as *not measured* and are
    skipped here. The test below holds them where the checkpoint is.
    """
    fresh_rows, committed_rows = (
        _parity_rows(_fresh_parity_table(None)),
        _committed_parity_rows(),
    )
    # Subset, not equality: rows for backends this environment cannot even
    # attempt (no checkpoint → no ONNX/CoreML row at all) may exist in the
    # committed table and be absent from the fresh run.
    assert set(fresh_rows) <= set(committed_rows), (
        "docs/parity-measured.md is stale — run tools/parity_table.py"
    )
    for stage, line in fresh_rows.items():
        # A row this environment could not measure says "not measured" in the
        # fresh run; the committed table may carry the real measurement from
        # the machine that has the assets. Weight-free rows must match
        # verbatim — they are what this test guards against fixture drift.
        if "not measured" in line:
            continue
        assert committed_rows[stage] == line, (
            f"docs/parity-measured.md is stale for {stage!r} — run tools/parity_table.py"
        )


@pytest.mark.slow
@requires("checkpoint")
def test_the_weighted_parity_rows_are_what_the_generator_produces() -> None:
    """Every row, regenerated with the checkpoint: about ten minutes.

    The weight-free gate above compares seven of the fourteen rows. The rows a
    reader opens the page for (the token counts, the top-1, the mel and
    vocoder correlations, the two renderer rows) were held by nothing, and the
    page carried numbers from before the fixes it now describes. Run where the
    weights are: the `parity` job in CI, and a maintainer's machine.
    """
    fresh = _fresh_parity_table(asset("checkpoint"))
    fresh_rows, committed_rows = _parity_rows(fresh), _committed_parity_rows()
    unmeasured = [stage for stage, line in fresh_rows.items() if "not measured" in line]
    assert not unmeasured, (
        f"the checkpoint is present and these rows were not measured: {unmeasured}"
    )
    base_rows = {
        stage: row
        for stage, row in committed_rows.items()
        if not stage.startswith("loudr-1-turbo ")
    }
    assert fresh_rows.keys() == base_rows.keys()
    for stage, fresh_row in fresh_rows.items():
        fresh_cells = [cell.strip() for cell in fresh_row.strip("|").split("|")]
        saved_cells = [cell.strip() for cell in base_rows[stage].strip("|").split("|")]
        assert fresh_cells[:3] == saved_cells[:3], f"{stage}: parity contract changed"
        if "corr" in fresh_cells[2]:
            # The generator has already enforced the numerical gates. Correlations
            # vary within those bands across backend builds; their printed last
            # digits are a measurement, not a bit-identical contract.
            assert fresh_cells[3].startswith("✓ "), stage
            assert saved_cells[3].startswith("✓ "), stage
        else:
            assert fresh_row == base_rows[stage], f"{stage}: regenerate the parity table"


# Every file that shows a `--local-dir` download and then reads what it produced.
_LAYOUT_DOCS = (
    sorted((REPO / "docs").rglob("*.md"))
    + sorted((REPO / "site" / "src" / "handwritten").rglob("*.mdx"))
    + [
        REPO / "README.md",
        REPO / "js" / "README.md",
        REPO / "go" / "README.md",
        REPO / "rust" / "README.md",
    ]
)
_LOCAL_DIR = re.compile(r"--local-dir\s+([\w./-]+)")


def _fetched_dirs(text: str) -> set[str]:
    """Every release directory a page's own commands write, however spelled."""
    return {
        found.rstrip("/")
        for pattern in (_LOCAL_DIR, _PORT_DOWNLOAD, _SWIFT_DOWNLOAD)
        for found in pattern.findall(text)
    }


def _roster() -> frozenset[str]:
    """The voice names the release ships.

    Only these are the download's output. `voices/my-voice.safetensors` in the
    cloning example is a file the *reader* writes, and it belongs wherever they
    put it -- checking it against the release layout would be reading an
    instruction to save as an instruction to load.
    """
    import json as _json

    entries = _json.loads(
        (REPO / "docs" / "voices" / "roster" / "provenance.json").read_text(encoding="utf-8")
    )
    return frozenset(v["name"] for v in entries)


# The artefacts `loudkit download` writes *inside* the directory it was given.
# The basename only: the directories in front of it are read by `_prefix_of`,
# which has to see them even when they are wrong. A lookbehind that refused a
# leading `/` made `wrong/loudr-1.safetensors` invisible rather than incorrect.
_ARTEFACT = re.compile(r"(loudr-1[\w.-]*\.safetensors|voices/[\w.-]+\.safetensors)")
_SEGMENT = re.compile(r"(?:[\w.-]+/)+$")


# The two ways the guides elide a leading path. They mean the same thing and
# they land in different places: `…` is not a character a path segment can
# hold, so it stays outside the segment match, while `...` is made of ones that
# can, so it is swallowed into it. Reading only the first left every
# `…/loudr-1/…` in the JS, Go and Rust guides unchecked; reading only the
# second turned `.../loudr-1/` into a directory named `...`.
_ELLIPSIS = ("\u2026", "...")


def _prefix_of(block: str, start: int) -> str | None:
    """The directory path written in front of an artefact, or ``None``.

    ``None`` means the path is a placeholder rather than a claim: an absolute
    `/path/to/loudr-1.safetensors` names no particular layout, and checking it
    against the release's would be reading an example as an instruction.

    An elided `…/loudr-1/loudr-1.safetensors` is the opposite -- the author
    chose to show that last directory, so it is checked.
    """
    match = _SEGMENT.search(block[:start])
    prefix = match.group(0) if match else ""
    before = block[: start - len(prefix)]
    elided = any(before.endswith(m) or before.endswith(m + "/") for m in _ELLIPSIS)
    for mark in _ELLIPSIS:
        if prefix.startswith(mark + "/"):
            elided, prefix = True, prefix[len(mark) + 1 :]
    if elided:
        return prefix
    if before.endswith("/"):
        return None
    return prefix


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("loudr-1/loudr-1.safetensors", "loudr-1/"),
        ("loudr-1/voices/joe.safetensors", "loudr-1/"),
        ("wrong/loudr-1.safetensors", "wrong/"),
        ("loudr-1.safetensors", ""),
        # Both elisions show their last directory on purpose, so both are read.
        ("\u2026/loudr-1/loudr-1.safetensors", "loudr-1/"),
        ("\u2026/wrong/loudr-1.safetensors", "wrong/"),
        (".../loudr-1/loudr-1.safetensors", "loudr-1/"),
        (".../wrong/loudr-1.safetensors", "wrong/"),
        # An absolute path names no layout at all.
        ("/path/to/loudr-1.safetensors", None),
    ],
)
def test_the_prefix_reader_sees_every_way_a_path_is_written(
    written: str, expected: str | None
) -> None:
    """The gates are only as good as this, and it was wrong twice.

    `…/loudr-1/loudr-1.safetensors` read as a placeholder and went unchecked --
    which is the form the JS, Go and Rust guides use, so the gate skipped the
    very examples its docstring claimed. `.../loudr-1/` read as a directory
    literally named `...`, so it never matched a fetched root either.
    """
    match = _ARTEFACT.search(written)
    assert match is not None, written
    assert _prefix_of(written, match.start()) == expected


def _code_blocks(text: str) -> list[str]:
    """Fenced blocks that carry a language tag.

    A filename in prose is a name, not a path. And an *untagged* fence is a
    directory listing or a program's output -- `loudr-1.safetensors` indented
    under `checkpoints/loudr-1/` is that layout drawn correctly, not a path to
    load. Only a tagged fence holds something a reader copies and runs.
    """
    return [block for block, _ in _tagged_blocks(text)]


def _tagged_blocks(text: str) -> list[tuple[str, str]]:
    """Each tagged fence with the language it declared."""
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    tag: str | None = None
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            if tag is None:
                tag = line.lstrip()[3:].strip()
                buf = []
            else:
                if tag:
                    out.append(("\n".join(buf), tag))
                tag, buf = None, []
            continue
        buf.append(line)
    return out


def test_examples_read_the_directory_the_download_wrote() -> None:
    """A page that says ``--local-dir loudr-1`` must then read ``loudr-1/…``.

    ``loudkit download --local-dir loudr-1`` puts the checkpoint at
    ``loudr-1/loudr-1.safetensors`` and the voices at ``loudr-1/voices/``. Three
    port guides, the landing page and three package READMEs asked for exactly
    that directory and then loaded ``loudr-1.safetensors`` and
    ``voices/joe.safetensors`` from the working directory instead -- so every
    copied example failed on the first line that opened a file, and the mistake
    sat inside a single line next to ``loudr-1/onnx``, which was right.

    Nothing caught it: links resolve, symbols exist, signatures match. None of
    those read a path as a path.
    """
    problems: list[str] = []
    for path in _LAYOUT_DOCS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        dirs = _fetched_dirs(text) - {"."}
        if len(dirs) != 1:
            # No download shown, or more than one layout on the page: this test
            # would be guessing which one an example meant.
            continue
        root = dirs.pop().rstrip("/")
        problems.extend(
            f"{path.relative_to(REPO)}: {written} is not under '{root}/', which is "
            f"where the page's fetch put it"
            for written in _misread_artefacts(text, {root})
        )
    assert not problems, "\n".join(problems)


def _misread_artefacts(text: str, fetched: set[str]) -> list[str]:
    """Every release artefact a page's code reads from outside ``fetched``.

    A bare `loudr-1.safetensors` counts: its prefix is the working directory,
    which no fetch on the page wrote either. Only a placeholder path
    (`/path/to/...`) and a voice the reader made themselves are left alone.
    """
    roots = {f"{root}/" for root in fetched}
    misread: list[str] = []
    for block in _code_blocks(text):
        for hit in _ARTEFACT.finditer(block):
            token = hit.group(0)
            if token.startswith("voices/") and (
                token[len("voices/") :].removesuffix(".safetensors") not in _roster()
            ):
                continue
            prefix = _prefix_of(block, hit.start())
            if prefix is None or prefix in roots:
                continue
            misread.append(f"'{prefix}{token}'")
    return misread


# Every page that shows a stranger how to run a port. Each has to stand on its
# own: the reader arrives here from npm, from pkg.go.dev, from crates.io or from
# the front page, with no other page open. The front page and the landing page
# show all four ports, so they are first-mile pages for every one.
_FIRST_MILE = (
    "README.md",
    "docs/guides/07-js-ts.md",
    "docs/guides/08-go.md",
    "docs/guides/09-rust.md",
    "docs/guides/10-swift.md",
    "site/src/handwritten/index.mdx",
    "js/README.md",
    "go/README.md",
    "rust/README.md",
    "swift/README.md",
)
_DOWNLOAD = re.compile(r"loudkit download\s+\S+[^\n]*?--local-dir\s+([\w./-]+)")
# A port that fetches for itself shows its own call instead of the Python CLI's.
# Requiring `loudkit download` of such a page would be requiring a Python
# install of a reader who does not need one. One pattern for every port: a
# `download` or `Download` taking the repo id and the directory, whatever else
# a port puts in front of them.
_PORT_DOWNLOAD = re.compile(
    r"""[Dd]ownload\([^)\n]*?["'][\w\-.]+/[\w\-.]+["']\s*,\s*["']([\w./-]+)["']"""
)


# Swift's fetch names its directory inside a URL initialiser, so the generic
# pattern above cannot see it.
_SWIFT_DOWNLOAD = re.compile(
    r'LoudKit\.download\(\s*repo:\s*"[^"]+",\s*to:\s*URL\(fileURLWithPath:\s*"([\w./-]+)"'
)


# A port's load taking a repo id straight: `Engine::load_bundle("org/name")`,
# `Engine.load("org/name")`, `loudkit.Load("org/name")`, `lk.load("org/name")`.
_REPO_LOAD = re.compile(r"""[Ll]oad(?:_bundle|Bundle|Paths)?\(\s*["'][\w\-.]+/[\w\-.]+["']""")


def _names_a_layout(text: str) -> bool:
    """Whether any code block writes an artefact under a directory prefix.

    A page that loads by repo id skips the layout check only when there is no
    layout on it to check: a stale `loudr-1/voices/joe.safetensors` beside a
    repo-id load is still a claim about where a file is.
    """
    for block in _code_blocks(text):
        for hit in _ARTEFACT.finditer(block):
            if _prefix_of(block, hit.start()):
                return True
    return False


def test_every_port_quickstart_stands_on_its_own() -> None:
    """A port example names a directory, so the page has to create it.

    The paths were wrong first: examples read `loudr-1.safetensors` from the
    working directory when the download had put it in `loudr-1/`. Correcting
    them is only half of it -- four of these seven pages then read `loudr-1/…`
    without ever showing the command that writes it, and the landing page went
    further and said the fetch goes to the shared cache, which is a different
    place from the one its own next line reads.

    ``test_examples_read_the_directory_the_download_wrote`` cannot see any of
    this: it skips a file with no ``--local-dir`` at all, which is exactly the
    failure here. This is the gate for the other direction -- the fetch must be
    on the page, and the examples must read what it wrote.
    """
    problems = [
        p
        for rel in _FIRST_MILE
        for p in _quickstart_problems(rel, (REPO / rel).read_text(encoding="utf-8"))
    ]
    assert not problems, "\n".join(problems)


def _quickstart_problems(rel: str, text: str) -> list[str]:
    fetched = {d.rstrip("/") for d in _DOWNLOAD.findall(text)} | _fetched_dirs(text)
    misread = _misread_artefacts(text, fetched)
    if not fetched and not misread and _REPO_LOAD.search(text):
        # The page loads by repo id, the port fetches into its own cache, and
        # no example names a release path: nothing on the page can misread a
        # directory. A page that loads by repo id *and* reads a path falls
        # through, because that path is a layout claim the page never wrote.
        return []
    if not fetched:
        what = ", ".join(misread) or "a local release"
        return [
            f"{rel}: reads {what} but never shows the fetch that writes it "
            "(`loudkit download … --local-dir <dir>`, or the port's own download call)"
        ]
    return [
        f"{rel}: {written} is under none of the directories this page fetches "
        f"into ({', '.join(sorted(fetched))})"
        for written in misread
    ]


_AGREEING_PAGE = (
    "```bash\nloudkit download loudreader/loudr-1 --for onnx --local-dir loudr-1\n```\n"
    '```go\nv, _ := voice.Load("loudr-1/voices/joe.safetensors")\n```\n'
)
_DISAGREEING_PAGE = _AGREEING_PAGE.replace("loudr-1/voices", "models/voices")
_PORT_FETCH_PAGE = (
    '```typescript\nconst dir = await download("loudreader/loudr-1", "loudr-1");\n'
    'const v = loadVoice("release/voices/joe.safetensors");\n```\n'
)
_SWIFT_FETCH_PAGE = (
    '```swift\nlet dir = try await LoudKit.download(repo: "loudreader/loudr-1", '
    'to: URL(fileURLWithPath: "loudr-1"))\n'
    "let v = try VoiceProfile.load(url: "
    'URL(fileURLWithPath: "models/loudr-1.safetensors"))\n```\n'
)
_REPO_LOAD_PAGE = '```go\neng, _ := loudkit.Load("loudreader/loudr-1")\n```\n'
_REPO_LOAD_AND_PATH_PAGE = _REPO_LOAD_PAGE + _AGREEING_PAGE.split("```\n", 1)[1]
_BARE_PATH_PAGE = '```go\nv, _ := voice.Load("loudr-1.safetensors")\n```\n'


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        (_AGREEING_PAGE, []),
        (_DISAGREEING_PAGE, ["x.md: 'models/voices/joe.safetensors' is under none"]),
        (_PORT_FETCH_PAGE, ["x.md: 'release/voices/joe.safetensors' is under none"]),
        (_SWIFT_FETCH_PAGE, ["x.md: 'models/loudr-1.safetensors' is under none"]),
        (_REPO_LOAD_PAGE, []),
        (_REPO_LOAD_AND_PATH_PAGE, ["x.md: reads 'loudr-1/voices/joe.safetensors' but never"]),
        (_BARE_PATH_PAGE, ["x.md: reads 'loudr-1.safetensors' but never"]),
    ],
)
def test_the_first_mile_gate_sees_a_fetch_and_a_read_disagree(
    page: str, expected: list[str]
) -> None:
    """The gate, driven by hand-written pages, one per way it could go quiet.

    Every fetch spelling (`--local-dir`, a port's `download(repo, dir)`, Swift's
    URL form) has to be read against the path the same page opens, and a
    repo-id load excuses a page only when it names no path at all. The last two
    cases are the early-out that used to skip a whole page: a page that loads
    by repo id and still reads `loudr-1/…` is reporting a layout it never wrote.
    """
    found = _quickstart_problems("x.md", page)
    assert len(found) == len(expected), found
    assert [p[: len(e)] for p, e in zip(found, expected, strict=True)] == expected


# A Swift block that loads both an engine and a voice is the quickstart, not an
# excerpt from further down a page that established its imports earlier. A voice
# arrives two ways: `VoiceProfile.load` from a path, and `voice(named:)` from the
# release directory, which is what the quickstarts show now.
_SWIFT_IMPORTS = ("Foundation", "LoudKit")


def _is_swift_quickstart(block: str) -> bool:
    return "Engine.load" in block and ("VoiceProfile.load" in block or "voice(named:" in block)


def test_swift_quickstarts_import_what_they_name() -> None:
    """A Swift quickstart that does not compile is not a quickstart.

    `LoudKit` does not re-export `Foundation` -- nothing in the package is
    `@_exported` -- so a snippet that builds a `URL` and imports only `LoudKit`
    fails with `cannot find 'URL' in scope`, which is what the compiler says
    here. The path gate could not see it: those paths were right and the code
    still would not build.

    The requirement does not depend on the block already saying `import`. A
    first version of this skipped any block without one, so deleting *both*
    imports -- the whole failure, not half of it -- made the block invisible and
    the test passed. A block that loads an engine and a voice is a quickstart;
    a reader copies it into an empty file, so it carries its own imports.
    `Foundation` is required only by a block that builds a `URL`; the
    string-path front door needs `LoudKit` alone.
    """
    problems: list[str] = []
    for rel in _FIRST_MILE:
        path = REPO / rel
        for block, tag in _tagged_blocks(path.read_text(encoding="utf-8")):
            if tag != "swift" or not _is_swift_quickstart(block):
                continue
            needed = [m for m in _SWIFT_IMPORTS if m != "Foundation" or "URL(" in block]
            for module in needed:
                if f"import {module}" not in block:
                    problems.append(f"{rel}: a Swift quickstart without `import {module}`")
    assert problems == [], "\n".join(problems)


def test_the_swift_gate_is_looking_at_something() -> None:
    """The gate above is only worth its name if a quickstart reaches it.

    Every guard in this file that filters before it checks can pass by finding
    nothing, and one of them did.
    """
    seen = {
        rel
        for rel in _FIRST_MILE
        for block, tag in _tagged_blocks((REPO / rel).read_text(encoding="utf-8"))
        if tag == "swift" and _is_swift_quickstart(block)
    }
    assert sorted(seen) == [
        "README.md",
        "docs/guides/10-swift.md",
        "site/src/handwritten/index.mdx",
        "swift/README.md",
    ], seen


# The pages a user reads: the twelve the documentation index lists, the front
# page, the landing page and the four registry READMEs. Internal vocabulary
# stays off them (a word here is a concept the reader would have to learn to
# use the page), and so do em dashes.
_USER_PAGES = (
    "SUPPORTED.md",
    "docs/README.md",
    "docs/MODEL_CARD.md",
    "docs/MODEL_CARD-turbo.md",
    "docs/guides/01-getting-started.md",
    "docs/guides/02-streaming-and-long-form.md",
    "docs/guides/03-cloning-a-voice.md",
    "docs/guides/04-server-and-agents.md",
    "docs/guides/11-choosing-a-model.md",
    "docs/reference/troubleshooting.md",
) + _FIRST_MILE

# Words that name the engine's internals. Each has a home under docs/design/
# or docs/reference/; a user page that needs one links there instead.
_INTERNAL_WORDS = re.compile(
    r"estimator|\bK\s*=\s*\d|recipe|manifest|format_version|fusion_mtp2|"
    r"fingerprint|provenance|digest",
    re.IGNORECASE,
)

# Spellings a port user must never be handed: the Python CLI's directory flag
# (the ports fetch for themselves) and the long-form verbs the ports renamed to
# `synthesize`.
_PORT_SPELLINGS = re.compile(r"--local-dir|synthesize_long|synthesizeLong|SynthesizeLong")
_PORT_PAGES = _FIRST_MILE

# Names the server sends, shown on the page as the API they are. The word
# inside them is not a concept the reader has to learn, so they are removed
# before the check.
_API_NAMES = {
    "docs/guides/04-server-and-agents.md": ("X-Loudkit-Fingerprint", "`fingerprint`"),
}
_LINK_TARGET = re.compile(r"\]\([^)]*\)")


def _prose(rel: str, text: str) -> str:
    """The page with link targets and the API names above removed."""
    text = _LINK_TARGET.sub("]()", text)
    for name in _API_NAMES.get(rel, ()):
        text = text.replace(name, "")
    return text


@pytest.mark.parametrize("rel", _USER_PAGES)
def test_user_pages_carry_no_internal_vocabulary(rel: str) -> None:
    """A user page names what the reader does, not how the engine is built."""
    text = _prose(rel, (REPO / rel).read_text(encoding="utf-8"))
    patterns = [_INTERNAL_WORDS] + ([_PORT_SPELLINGS] if rel in _PORT_PAGES else [])
    hits = [
        f"{n}: {line.strip()[:100]}"
        for n, line in enumerate(text.splitlines(), 1)
        if any(p.search(line) for p in patterns)
    ]
    assert not hits, f"{rel} carries internal vocabulary:\n  " + "\n  ".join(hits)


@pytest.mark.parametrize("rel", _USER_PAGES)
def test_user_pages_use_plain_punctuation(rel: str) -> None:
    """An em dash is easy to overuse and made the landing documents sound generated."""
    text = (REPO / rel).read_text(encoding="utf-8")
    assert "—" not in text, f"{rel} contains an em dash"


def test_the_vocabulary_gate_reads_prose_and_code_alike() -> None:
    """Drift check: a banned word in a code span, a fence or a table is caught;
    one inside a link target is not, because a URL is an address."""
    assert _INTERNAL_WORDS.search(_prose("x.md", "the `fusion_mtp2` loop"))
    assert _INTERNAL_WORDS.search(_prose("x.md", "```bash\nloudkit --manifest x\n```"))
    assert not _INTERNAL_WORDS.search(_prose("x.md", "[the roster](docs/provenance.json)"))
    assert not _INTERNAL_WORDS.search(
        _prose("docs/guides/04-server-and-agents.md", "X-Loudkit-Fingerprint   <hash>")
    )
    assert _INTERNAL_WORDS.search(_prose("x.md", "X-Loudkit-Fingerprint   <hash>"))


# The shared front door, as each language spells it: load by repo id, a voice
# from the engine, synthesize, save. The first hello must not require an upload;
# separately downloaded preview profiles remain an optional later step.
_FRONT_DOOR = {
    "python": (
        r"lk\.load\(",
        r"engine\.voice\(",
        r"\.synthesize\(",
        r"\.save\(",
    ),
    "swift": (
        r"Engine\.load\(",
        r"engine\.voice\(named:",
        r"\.synthesize\(",
        r"\.saveWav\(",
    ),
    "go": (
        r"loudkit\.Load\(",
        r"eng\.Voice\(",
        r"\.Synthesize\(",
        r"\.SaveWav\(",
    ),
    "rust": (
        r"Engine::load\(",
        r"engine\.voice\(",
        r"\.synthesize\(",
        r"\.save_wav\(",
    ),
    "typescript": (
        r"Engine\.load\(",
        r"engine\.voice\(",
        r"\.synthesize\(",
        r"\.saveWav\(",
    ),
}


def test_the_readme_hellos_walk_through_the_shared_front_door() -> None:
    """Every README hello loads, names a voice, synthesizes and saves."""
    blocks = {
        tag: block
        for block, tag in _tagged_blocks((REPO / "README.md").read_text(encoding="utf-8"))
    }
    problems = [
        f"README.md: the {tag} hello does not call {step!r}"
        for tag, steps in _FRONT_DOOR.items()
        for step in steps
        if not re.search(step, blocks.get(tag, ""))
    ]
    assert not problems, "\n".join(problems)


def test_the_documentation_index_lists_exactly_the_user_pages() -> None:
    """docs/README.md's numbered list is the twelve pages, and every one exists."""
    index = (REPO / "docs" / "README.md").read_text(encoding="utf-8")
    section = index.split("## Using loudkit", 1)[1].split("\n## ", 1)[0]
    listed = re.findall(r"^\d+\. \[[^\]]+\]\(([^)]+)\)", section, re.MULTILINE)
    assert len(listed) == 12, listed
    for target in listed:
        assert (REPO / "docs" / target).exists(), target
    assert not (REPO / "docs" / "guides" / "README.md").exists(), (
        "docs/guides/README.md duplicated this index; the index is docs/README.md"
    )


# --- the commands on the pages are real ---------------------------------------

# The docs tree, the front page, the landing page, and the two root pages that
# state the CLI as a contract: SUPPORTED.md to users, RELEASING.md to the
# person cutting a release.
_COMMAND_PAGES = sorted((REPO / "docs").rglob("*.md")) + [
    REPO / "README.md",
    REPO / "SUPPORTED.md",
    REPO / "RELEASING.md",
    REPO / "site" / "src" / "handwritten" / "index.mdx",
]
# A backtick span, or the landing page's HTML spelling of one.
_INLINE_CODE = re.compile(r"`([^`\n]+)`|<code>([^<\n]+)</code>")
_SUBCOMMAND = re.compile(r"^[a-z][a-z-]*$")
_FLAG = re.compile(r"^--?[A-Za-z][\w-]*")


def _command_lines(text: str) -> list[str]:
    """Every ``loudkit ...`` a reader could copy: fenced lines and inline code.

    Prose is not read, a page may say "loudkit is" without claiming a
    subcommand. Inside a fence a command may follow a prompt, a pipe or
    ``&&``; a shell continuation is joined first.
    """
    found: list[str] = []
    in_fence = False
    for line in re.sub(r"\\\n\s*", " ", text).splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            parts = re.split(r"&&|\|\||;|\|", stripped)
            spans = [part.strip().removeprefix("$").strip() for part in parts]
        else:
            spans = [tick or html for tick, html in _INLINE_CODE.findall(line)]
        found.extend(span for span in spans if span.startswith("loudkit "))
    return found


def _command_problems(text: str) -> list[str]:
    """What the CLI parser refuses among the commands a page names."""
    import shlex

    from loudkit.cli import build_parser

    parser = build_parser()
    subparsers = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    ).choices
    problems: list[str] = []
    for command in _command_lines(text):
        try:
            tokens = shlex.split(command, comments=True)
        except ValueError:
            tokens = command.split()
        rest = tokens[1:]
        while rest and rest[0].startswith("-"):
            flag = rest.pop(0).split("=", 1)[0]
            if _FLAG.match(flag) and flag not in parser._option_string_actions:
                problems.append(f"{command!r}: loudkit has no {flag}")
        if not rest or not _SUBCOMMAND.match(rest[0]):
            continue
        sub = rest[0]
        if sub not in subparsers:
            problems.append(f"{command!r}: loudkit has no subcommand {sub!r}")
            continue
        for token in rest[1:]:
            flag = token.split("=", 1)[0]
            if _FLAG.match(flag) and flag not in subparsers[sub]._option_string_actions:
                problems.append(f"{command!r}: loudkit {sub} has no {flag}")
    return problems


def _help_exits_cleanly(sub: str) -> None:
    """``loudkit <sub> --help`` through the parser, in-process."""
    import contextlib
    import io

    from loudkit.cli import build_parser

    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(SystemExit) as info:
        build_parser().parse_args([sub, "--help"])
    assert info.value.code == 0, f"loudkit {sub} --help exited {info.value.code}"


def test_every_command_a_page_names_is_one_the_parser_accepts() -> None:
    """A `loudkit <subcommand> --flag` on a page is run through the CLI parser.

    An advertised command that does not exist is the fastest way a page can
    lie to a stranger, and a name is cheap to check: every subcommand named
    must answer `--help`, and every flag named on the line must be one that
    subcommand declares.
    """
    problems: list[str] = []
    named: set[str] = set()
    for page in _COMMAND_PAGES:
        rel = page.relative_to(REPO).as_posix()
        text = page.read_text(encoding="utf-8")
        problems.extend(f"{rel}: {p}" for p in _command_problems(text))
        for command in _command_lines(text):
            tokens = command.split()
            if len(tokens) > 1 and _SUBCOMMAND.match(tokens[1]):
                named.add(tokens[1])
    assert not problems, "pages name commands the CLI refuses:\n  " + "\n  ".join(problems)
    from loudkit.cli import COMMANDS

    for sub in sorted(named & set(COMMANDS)):
        _help_exits_cleanly(sub)
    assert named & set(COMMANDS), "no page names a subcommand; the extractor is blind"


_MADE_UP_PAGE = (
    "Run `loudkit frobnicate` first, then `loudkit serve --frob`, and read\n"
    "`loudkit --version`. loudkit is a library.\n"
    "```bash\n"
    "$ loudkit speak --checkpoint loudr-1.safetensors \\\n"
    "    --voice joe 'Hello.' -o hello.wav\n"
    "loudkit doctor | head && loudkit serve --grpc\n"
    "```\n"
    "<p>Every figure is one <code>loudkit bench</code> run.</p>\n"
)


def test_the_command_gate_sees_a_made_up_command() -> None:
    """The gate is looking: a made-up subcommand and a made-up flag are named,
    in a backtick span and in the landing page's HTML one; a continuation line
    is joined, and prose and real commands are not."""
    problems = _command_problems(_MADE_UP_PAGE)
    assert len(problems) == 3, problems
    assert "frobnicate" in problems[0]
    assert "--frob" in problems[1]
    assert "bench" in problems[2]
    assert sorted(_command_lines(_MADE_UP_PAGE))[-1].startswith("loudkit speak --checkpoint")


@pytest.mark.slow
@requires("turbo_checkpoint")
def test_turbo_parity_rows_match_the_release() -> None:
    rows = _parity_rows(_fresh_parity_table(asset("turbo_checkpoint")))
    names = {"loudr-1-turbo ONNX renderer", "loudr-1-turbo CoreML renderer"}
    committed = _committed_parity_rows()
    assert names <= rows.keys()
    for name in names:
        assert "not measured" not in rows[name]
        assert rows[name] == committed[name]


def _selective_fetch_bytes(root: Path, backend: str, *, cloning: bool) -> int:
    """What `loudkit download --for <backend> [--with-cloning]` would fetch, in bytes."""
    import fnmatch

    from loudkit.release import release_patterns

    allow, ignore = release_patterns(backend, cloning=cloning)
    files: set[Path] = set()
    for pattern in allow:
        for hit in root.glob(pattern):
            files.update(p for p in ([hit] if hit.is_file() else hit.rglob("*")) if p.is_file())
    kept = [
        f
        for f in files
        if not any(
            fnmatch.fnmatch(f.name, pat) or fnmatch.fnmatch(str(f.relative_to(root)), pat)
            for pat in ignore
        )
    ]
    return sum(f.stat().st_size for f in kept)


def _documented_download_sizes() -> dict[str, list[float]]:
    """The size table in the getting-started guide, GB per column, by model."""
    rows: dict[str, list[float]] = {}
    for line in (
        (REPO / "docs" / "guides" / "01-getting-started.md")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 7 and cells[0] in ("loudr-1", "loudr-1-turbo"):
            rows[cells[0]] = [float(c.split()[0]) for c in cells[1:]]
    assert set(rows) == {"loudr-1", "loudr-1-turbo"}, rows
    return rows


@requires("checkpoint")
@requires("turbo_checkpoint")
def test_the_documented_download_sizes_are_what_the_releases_weigh() -> None:
    """Six numbers a newcomer plans disk space by, measured from the release
    directories rather than typed. Within 5%: the table rounds to 10 MB and a
    checksum file or two sits outside the selective patterns."""
    roots = {
        "loudr-1": asset("checkpoint").parent,
        "loudr-1-turbo": asset("turbo_checkpoint").parent,
    }
    columns = [
        ("torch", False),
        ("torch", True),
        ("onnx", False),
        ("onnx", True),
        ("coreml", False),
        ("coreml", True),
    ]
    documented = _documented_download_sizes()
    for model, root in roots.items():
        for (backend, cloning), claimed in zip(columns, documented[model], strict=True):
            measured = _selective_fetch_bytes(root, backend, cloning=cloning) / 1e9
            assert abs(measured - claimed) <= max(0.05 * claimed, 0.02), (
                f"{model} --for {backend}{' --with-cloning' if cloning else ''}: "
                f"docs say {claimed:.2f} GB, the release weighs {measured:.2f} GB"
            )


def test_rust_extended_guide_is_the_compiled_example() -> None:
    """The options and ownership example must stay in the compiler's coverage."""
    import textwrap

    guide = (REPO / "docs/guides/09-rust.md").read_text(encoding="utf-8")
    section = guide.split("## Synthesize", 1)[1].split("## Streaming", 1)[0]
    block = section.split("```rust\n", 1)[1].split("```", 1)[0].strip()
    example = (REPO / "rust/examples/guide_options.rs").read_text(encoding="utf-8")
    compiled = example.split("// BEGIN GUIDE\n", 1)[1].split("// END GUIDE", 1)[0]
    assert block == textwrap.dedent(compiled).strip()


def test_notebook_first_speech_needs_no_uploaded_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the first speech cell with a tiny engine; a file load must fail."""
    import json
    from types import SimpleNamespace

    import loudkit

    notebook = json.loads(
        (REPO / "notebooks/loudkit_quickstart.ipynb").read_text(encoding="utf-8")
    )
    cell = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "narrator =" in "".join(c["source"])
    )
    profile = object()
    calls: list[str] = []

    def voice(name: str) -> object:
        calls.append(name)
        assert name == "joe"
        return profile

    def synthesize(text: str, given: object, *, seed: int) -> SimpleNamespace:
        assert given is profile
        return SimpleNamespace(audio=[0.0], sample_rate=24000)

    monkeypatch.setattr(
        loudkit, "load", lambda _: SimpleNamespace(voice=voice, synthesize=synthesize)
    )
    # IPython is a notebook dependency, not a required package/test dependency.
    import sys
    import types

    display = types.ModuleType("IPython.display")
    display.Audio = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    exec(compile(cell, "notebook first speech", "exec"), {})
    assert calls == ["joe"]
