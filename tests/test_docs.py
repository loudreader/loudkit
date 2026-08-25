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

import posixpath
import re
import subprocess
from pathlib import Path, PurePosixPath

import pytest

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


def test_code_snippets_import_real_symbols() -> None:
    """`from loudkit.x import Y` inside a ```python code block must name a
    symbol that actually exists — the fastest way a doc can lie about the API.
    Only code blocks are checked; prose that happens to contain the word
    "import" is not code."""

    # Import every module so the symbols are resolvable via getattr.

    missing: list[str] = []
    block_re = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    import_re = re.compile(r"^from\s+(loudkit\.[\w.]+)\s+import\s+([\w,]+)", re.MULTILINE)
    for md in MD_FILES:
        text = md.read_text(encoding="utf-8")
        for block in block_re.findall(text):
            for m in import_re.finditer(block):
                module_name, names = m.group(1), m.group(2)
                try:
                    module = __import__(module_name, fromlist=["*"])
                except ImportError:
                    missing.append(f"{md.name}: cannot import {module_name}")
                    continue
                for name in (n.strip() for n in names.split(",") if n.strip()):
                    if not hasattr(module, name):
                        missing.append(f"{md.name}: {module_name} has no {name!r}")
    assert not missing, "snippets import missing symbols:\n  " + "\n  ".join(missing)


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
    import importlib.util as _ilu
    import json as _json

    spec = _ilu.spec_from_file_location(
        "build_voices_md", REPO / "tools" / "build_voices_md.py"
    )
    module = _ilu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    voices = _json.loads(
        (REPO / "docs" / "voices" / "roster" / "provenance.json").read_text(encoding="utf-8")
    )
    expected = module.render(voices)
    on_disk = (REPO / "VOICES.md").read_text(encoding="utf-8")
    assert on_disk == expected, (
        "VOICES.md does not match tools/build_voices_md.py output — "
        "run `python tools/build_voices_md.py` and commit the result"
    )


def test_roster_audio_files_exist() -> None:
    """Every provenance entry names an audio file — the file must be there."""
    import json as _json

    provenance = REPO / "docs" / "voices" / "roster" / "provenance.json"
    voices = _json.loads(provenance.read_text(encoding="utf-8"))
    missing = [
        v["name"] for v in voices if not (provenance.parent / v["sample"]["audio"]).is_file()
    ]
    assert not missing, f"roster entries whose sample audio is missing: {missing}"


def test_roster_paths_and_digests_name_the_bytes_we_publish() -> None:
    """Every public path resolves, and every published digest matches its bytes.

    The roster used to name ``voices/profiles/<name>`` although the model repo
    has always shipped ``voices/<name>``.  It also called an unpublished source
    WAV an ``hf_path``.  A provenance record may identify an unpublished input,
    but it cannot claim that input is available at a path where it is not.
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
        profile_path = REPO / "assets" / expected_profile
        if profile.get("hf_path") != expected_profile:
            problems.append(f"{name}: profile path is {profile.get('hf_path')!r}")
        elif not profile_path.is_file() or digest(profile_path) != profile.get("sha256"):
            problems.append(f"{name}: profile digest does not match {expected_profile}")

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


def test_public_docs_quote_the_measured_voice_profile_size() -> None:
    """The shipped profiles average about 150 KB, not the old 300 KB estimate."""
    profiles = sorted((REPO / "assets" / "voices").glob("*.safetensors"))
    assert len(profiles) == 20
    average = sum(path.stat().st_size for path in profiles) / len(profiles)
    assert 100_000 <= average <= 200_000

    for rel in (
        "README.md",
        "SUPPORTED.md",
        "docs/MODEL_CARD.md",
        "notebooks/loudkit_quickstart.ipynb",
        "python/loudkit/__init__.py",
        "python/loudkit/hub.py",
    ):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert "150 KB" in text, f"{rel} does not state the measured profile size"
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


def _method_arity(source: str, receiver: str, method: str) -> int | None:
    """Count the parameters of ``method`` in ``source``, receiver excluded.

    Deliberately textual rather than a real parse: the point is to notice that
    a documented call site and its implementation have drifted apart, and a
    regex over one signature line is enough for that and needs no toolchain.
    """
    import re as _re

    for pattern in (
        # Go:   func (e *Engine) SynthesizeLong(a A, b B) (…)
        rf"func \(\w+ \*?{receiver}\) {method}\((.*?)\)\s*\(",
        # Rust: pub fn synthesize_long(&mut self, a: A, b: B) -> …
        rf"pub fn {method}\(\s*&mut self,(.*?)\)\s*->",
    ):
        m = _re.search(pattern, source, _re.DOTALL)
        if m:
            params = m.group(1)
            # Go groups names: `a, b string` is two parameters, so count by
            # commas at depth zero rather than by declarations.
            depth, count, seen = 0, 0, False
            for ch in params:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                elif ch == "," and depth == 0:
                    count += 1
                if not ch.isspace():
                    seen = True
            return count + 1 if seen else 0
    return None


def test_binding_snippets_call_the_real_signatures() -> None:
    """A tutorial's — and a package README's — Go and Rust snippets must match.

    Both engines gained a `shouldCancel` parameter that the tutorials never
    picked up, so `docs/guides/08-go.md` and `09-rust.md` shipped snippets
    that do not compile — "not enough arguments in call to eng.Synthesize" and
    "this method takes 5 arguments but 4 arguments were supplied". Nothing
    noticed because `test_code_snippets_import_real_symbols` above only reads
    ```python blocks.

    This counts arguments at the documented call site against parameters at the
    definition. It is not a compiler; it is the check that would have caught
    the drift that actually happened.
    """
    import re as _re

    cases = [
        (
            REPO / "docs/guides/08-go.md",
            REPO / "go/engine/engine.go",
            "Engine",
            "SynthesizeLong",
            _re.compile(r"eng\.SynthesizeLong\(\s*(.*?)\)\s*$", _re.DOTALL | _re.MULTILINE),
        ),
        (
            REPO / "docs/guides/09-rust.md",
            REPO / "rust/src/engine.rs",
            "Engine",
            "synthesize_long",
            _re.compile(r"eng\.synthesize_long\((.*?)\)\?", _re.DOTALL),
        ),
        # The registry READMEs carry the same call, and RELEASING.md's
        # acceptance pass runs *those* examples against a published package —
        # a drifted README example fails in a stranger's scratch project.
        (
            REPO / "go/README.md",
            REPO / "go/engine/engine.go",
            "Engine",
            "SynthesizeLong",
            _re.compile(r"eng\.SynthesizeLong\(\s*(.*?)\)\s*$", _re.DOTALL | _re.MULTILINE),
        ),
        (
            REPO / "rust/README.md",
            REPO / "rust/src/engine.rs",
            "Engine",
            "synthesize_long",
            _re.compile(r"eng\.synthesize_long\((.*?)\)\?", _re.DOTALL),
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
        args = [a for a in call.group(1).split(",") if a.strip() and "//" not in a]
        if len(args) != want:
            problems.append(
                f"{where}: {method} is called with {len(args)} arguments, "
                f"but {impl.name} declares {want}"
            )
    assert not problems, "documented call sites have drifted:\n  " + "\n  ".join(problems)


def test_the_readme_links_to_the_generated_parity_report() -> None:
    """The front page links to the evidence instead of duplicating its table."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert (
        "[Measured parity](https://github.com/loudreader/loudkit/blob/main/"
        "docs/parity-measured.md)"
    ) in readme
    assert "<!-- parity-table:" not in readme


@pytest.mark.parametrize(
    "rel", ["README.md", "docs/MODEL_CARD.md", "site/src/handwritten/index.mdx"]
)
def test_front_doors_use_plain_punctuation(rel: str) -> None:
    """An em dash is easy to overuse and made both landing documents sound generated."""
    text = (REPO / rel).read_text(encoding="utf-8")
    assert "—" not in text, f"{rel} contains an em dash"


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
    import sys

    sys.path.insert(0, str(REPO / "tools"))
    try:
        import make_speechtext_fixture as gen
    finally:
        sys.path.pop(0)

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


def test_the_production_fingerprint_is_pinned_in_one_value_everywhere() -> None:
    """Every copy of the shipped fingerprint, against the golden fixture.

    Four places pinned the production fingerprint as a literal: this suite's
    weighted public-API test, `docs/reference/IDENTITY-CONTRACT.md`,
    `docs/platforms/apple.md` and
    the server tutorial. All four went stale together the moment the shared
    grammar file changed, and nothing noticed for a session's worth of commits,
    because the only test among them is `@pytest.mark.slow` and asset-gated —
    it does not run without a checkpoint, which is precisely when the value
    moves.

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
        REPO / "tests" / "test_public_api.py": r'fingerprint\(\) == "([0-9a-f]{16})"',
        REPO
        / "docs"
        / "reference"
        / "IDENTITY-CONTRACT.md": r"Production fingerprint `([0-9a-f]{16})`",
        REPO / "docs" / "platforms" / "apple.md": r"independently and agree: `([0-9a-f]{16})`",
        REPO
        / "docs"
        / "guides"
        / "04-server-and-agents.md": r"X-Loudkit-Fingerprint\s+([0-9a-f]{16})",
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

    The README carries a copy of the same table between markers, and the
    generator cannot write it — so the copy is compared too, which is the half
    that actually faces a reader.
    """
    import subprocess
    import sys
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        out = _Path(tmp) / "parity-measured.md"
        result = subprocess.run(
            [sys.executable, str(REPO / "tools" / "parity_table.py"), "--out", str(out)],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO,
        )
        assert result.returncode == 0, f"generator failed:\n{result.stdout}\n{result.stderr}"
        fresh = out.read_text(encoding="utf-8")

    def _rows(text: str) -> dict[str, str]:
        body = text[text.index("| stage |") :]
        table = body[: body.index("\n\n")].strip()
        rows = {}
        for line in table.splitlines()[2:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows[cells[0]] = line.strip()
        return rows

    committed = (REPO / "docs" / "parity-measured.md").read_text(encoding="utf-8")
    fresh_rows, committed_rows = _rows(fresh), _rows(committed)
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


def _roster() -> frozenset[str]:
    """The twenty voice names the release ships.

    Only these are the download's output. `voices/my-voice.safetensors` in the
    cloning example is a file the *reader* writes, and it belongs wherever they
    put it -- checking it against the release layout would be reading an
    instruction to save as an instruction to load.
    """
    import json as _json

    data = _json.loads((REPO / "docs" / "voices" / "roster" / "provenance.json").read_text())
    entries = data["voices"] if isinstance(data, dict) else data
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
        text = path.read_text()
        dirs = set(_LOCAL_DIR.findall(text)) - {"."}
        if len(dirs) != 1:
            # No download shown, or more than one layout on the page: this test
            # would be guessing which one an example meant.
            continue
        root = dirs.pop().rstrip("/")
        for block in _code_blocks(text):
            for hit in _ARTEFACT.finditer(block):
                token = hit.group(0)
                if token.startswith("voices/") and (
                    token[len("voices/") :].removesuffix(".safetensors") not in _roster()
                ):
                    continue
                prefix = _prefix_of(block, hit.start())
                if prefix is None:
                    continue
                if prefix != f"{root}/":
                    line = block[: hit.start()].count("\n") + 1
                    problems.append(
                        f"{path.relative_to(REPO)}: '{prefix}{token}' is not under "
                        f"'{root}/', which is where --local-dir {root} put it "
                        f"(code block line {line})"
                    )
    assert not problems, "\n".join(problems)


# Every page that shows a stranger how to run a port. Each has to stand on its
# own: the reader arrives here from npm, from pkg.go.dev, from crates.io or from
# the front page, with no other page open.
_FIRST_MILE = (
    "docs/guides/07-js-ts.md",
    "docs/guides/08-go.md",
    "docs/guides/09-rust.md",
    "docs/guides/10-swift.md",
    "site/src/handwritten/index.mdx",
    "js/README.md",
    "go/README.md",
    "rust/README.md",
)
_DOWNLOAD = re.compile(r"loudkit download\s+\S+[^\n]*?--local-dir\s+([\w./-]+)")


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
    problems: list[str] = []
    for rel in _FIRST_MILE:
        path = REPO / rel
        text = path.read_text()
        fetched = {d.rstrip("/") for d in _DOWNLOAD.findall(text)}
        if not fetched:
            problems.append(
                f"{rel}: reads a local release but never shows "
                "`loudkit download … --local-dir <dir>` that writes one"
            )
            continue
        for block in _code_blocks(text):
            for hit in _ARTEFACT.finditer(block):
                token = hit.group(0)
                if token.startswith("voices/") and (
                    token[len("voices/") :].removesuffix(".safetensors") not in _roster()
                ):
                    continue
                prefix = _prefix_of(block, hit.start())
                if prefix is None:
                    continue
                if prefix not in {f"{root}/" for root in fetched}:
                    problems.append(
                        f"{rel}: '{prefix}{token}' is under none of the directories "
                        f"this page fetches into ({', '.join(sorted(fetched))})"
                    )
    assert not problems, "\n".join(problems)


# A Swift block that loads both an engine and a voice is the quickstart, not an
# excerpt from further down a page that established its imports earlier.
_SWIFT_QUICKSTART = ("Engine.load", "VoiceProfile.load")
_SWIFT_IMPORTS = ("Foundation", "LoudKit")


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
    """
    problems: list[str] = []
    for rel in _FIRST_MILE:
        path = REPO / rel
        for block, tag in _tagged_blocks(path.read_text()):
            if tag != "swift" or not all(sym in block for sym in _SWIFT_QUICKSTART):
                continue
            for module in _SWIFT_IMPORTS:
                if f"import {module}" not in block:
                    problems.append(f"{rel}: a Swift quickstart without `import {module}`")
    assert problems == [], "\n".join(problems)


def test_the_swift_gate_is_looking_at_something() -> None:
    """The gate above is only worth its name if a quickstart reaches it.

    Every guard in this file that filters before it checks can pass by finding
    nothing, and one of them did.
    """
    seen = [
        rel
        for rel in _FIRST_MILE
        for block, tag in _tagged_blocks((REPO / rel).read_text())
        if tag == "swift" and all(sym in block for sym in _SWIFT_QUICKSTART)
    ]
    assert sorted(seen) == ["docs/guides/10-swift.md", "site/src/handwritten/index.mdx"], seen
