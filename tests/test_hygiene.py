"""The repository's own house rules, held by a test rather than by a reviewer.

Runtime code says what it does and the one reason it does it that way; the
long form lives in docs/design. Each rule here is one a reviewer would
otherwise have to remember on every file, which is what makes it a test.
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# A commit cited in a comment is investigation history: the reader cannot
# follow it from a wheel, and the reason it carried belongs in the sentence.
CITED_COMMIT = re.compile(r"\b(?:at|in|commit|since|after|before|from|by)\s+[0-9a-f]{7,40}\b")
# Narrated change: what the code did before, how many reviews missed it, whose
# job it was not. A reader of the shipped package cannot check any of it and
# does not need to; the rule the code follows is what belongs in the sentence.
# "previously" and "no longer" are left alone, because they describe algorithms
# as often as history, and "is used to" is not this pattern at all.
NARRATED_HISTORY = re.compile(
    r"(?<!\bis )(?<!\bare )(?<!\bbe )(?<!\bwas )(?<!\bwere )(?<!\bbeen )(?<!\bbeing )"
    r"\bused to\b"
    r"|\b(?:historically|before this change|after this change)\b"
    r"|\bnobody's job\b"
    r"|\brounds of (?:review|human review)\b",
    re.I,
)
MEASUREMENT = re.compile(r"\d+(?:\.\d+)?\s?(?:ms|dB|dBFS|MB|GB|%|x)\b")
POINTER = re.compile(r"docs/design|See |see :")
# A real suppression, not a docstring or comment that talks about one: the
# form mypy reads is a trailing comment carrying a bracketed error code.
SUPPRESSION = re.compile(r"#\s*type:\s*ignore\[")
# The same marker in either form, bracketed or bare, for the directory where
# neither spelling is read.
UNREAD_SUPPRESSION = re.compile(r"^#\s*type:\s*ignore\b")
NUMBER_WORDS = {
    word: n
    for n, word in enumerate(
        (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        )
    )
}
# A cited docs page, however it is written: bare, in backticks or in double
# backticks. Trailing punctuation is not part of the path.
DOC_POINTER = re.compile(r"docs/[A-Za-z0-9_./-]+\.md")

PORT_SOURCES = (
    ("go", "*.go", ("_test.go",)),
    ("rust/src", "*.rs", ()),
    ("js/src", "*.ts", ("/test/",)),
    ("swift/LoudKit", "*.swift", ()),
    ("swift/LoudKitText", "*.swift", ()),
)

# Not in the house style: a sentence reaching for one wants a colon, a comma,
# a pair of parentheses or a full stop, and the choice is information the dash
# throws away.
EM_DASH = "\u2014"
# The one place the character is data rather than punctuation someone wrote:
# the footnote-marker character class, whose hyphen, en dash and em dash are
# the marks a range inside a marker is written with. Named by path, one line
# per port, so a sixth site has to be added here deliberately; prose in these
# files is still held to the rule.
FUNNEL_DATA = frozenset(
    {
        "python/loudkit/frontend/speechtext.py",
        "go/speechtext/speechtext.go",
        "rust/src/speechtext.rs",
        "js/src/speechText.ts",
        "swift/LoudKitText/SpeechText.swift",
    }
)
FOOTNOTE_MARKER_CLASS = "-\u2013\u2014"


PACKAGE = REPO / "python" / "loudkit"

SHIPPED_PYTHON = (
    PACKAGE,
    # Shipped Python outside the package: `justfile` lints it with ruff and a
    # user reads and edits it, so the prose rules reach it too. Not the
    # docstring-length rule below: this module documents itself because it runs
    # standalone, with no package on its path to point at.
    REPO / "integrations" / "speech-dispatcher",
)


def _python_prose(roots: tuple[Path, ...] = (PACKAGE,)) -> list[tuple[str, str]]:
    """Every docstring and comment in the runtime, with where it lives."""
    out: list[tuple[str, str]] = []
    for path in sorted(p for root in roots for p in root.rglob("*.py")):
        if "proto" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO)
        for node in ast.walk(ast.parse(source)):
            if isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                doc = ast.get_docstring(node)
                if doc:
                    out.append((f"{rel}:{getattr(node, 'name', '<module>')}", doc))
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                out.append((f"{rel}:{tok.start[0]}", tok.string))
    return out


def _port_comments() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for root, glob, skips in PORT_SOURCES:
        for path in sorted((REPO / root).rglob(glob)):
            # `as_posix`, not `str`: the skips are written with slashes, and on
            # Windows a path prints with backslashes, so `js/src/test` matched
            # nothing there. The gate then read the port's test files on one
            # platform and not on the others, and reported prose it was never
            # meant to see.
            text = path.as_posix()
            if any(skip in text for skip in skips):
                continue
            for number, line in enumerate(
                path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
            ):
                stripped = line.strip()
                if stripped.startswith(("//", "/*", "*")):
                    out.append((f"{path.relative_to(REPO)}:{number}", stripped))
    return out


def _shipped_source_lines() -> list[tuple[str, str]]:
    """Every line of every port's shipped sources, with where it lives.

    Whole lines rather than comments alone: an em dash in an error message is
    prose a caller reads, and the TypeScript port shipped seventeen of them.
    """
    # `encoding="utf-8"`, not the platform default. Without it Windows reads
    # these files as cp1252, and `→ ✓ ✗ ≈ ≥ × ◦` in the funnel's tables decode
    # into bytes this gate then reported as em dashes. A gate that reads a
    # different file on one platform is worse than no gate: it is loud where
    # there is nothing and silent where there is.
    out: list[tuple[str, str]] = []
    files = [
        p
        for root in SHIPPED_PYTHON
        for p in sorted(root.rglob("*.py"))
        if "proto" not in p.parts
    ]
    for root, glob, skips in PORT_SOURCES:
        files += [
            p for p in sorted((REPO / root).rglob(glob)) if not any(s in str(p) for s in skips)
        ]
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
        ):
            out.append((f"{rel}:{number}", line))
    return out


def test_no_shipped_source_carries_an_em_dash() -> None:
    """The house rule, held in one place for all five implementations.

    Swift holds its own line in `DocCommentTests` and `tools/` in
    `test_tools_cli`; this is the gate for everything those two do not reach,
    and the reason a new file cannot arrive without the rule.
    """
    offenders = [
        f"{where}: {line.strip()}"
        for where, line in _shipped_source_lines()
        if EM_DASH in line
        and not (where.rsplit(":", 1)[0] in FUNNEL_DATA and FOOTNOTE_MARKER_CLASS in line)
    ]
    assert not offenders, (
        "em dashes in shipped prose; use the punctuation the sentence wants:\n  "
        + "\n  ".join(offenders)
    )


def test_runtime_prose_carries_no_commit_hashes_or_narrated_history() -> None:
    hits = [
        f"{where}: {match.group(0)!r}"
        for where, text in _python_prose(SHIPPED_PYTHON) + _port_comments()
        for match in (CITED_COMMIT.search(text), NARRATED_HISTORY.search(text))
        if match
    ]
    assert not hits, "history in runtime prose; say the rule, not the story:\n  " + "\n  ".join(
        hits
    )


def test_long_measured_docstrings_point_at_the_design_notes() -> None:
    """A docstring that argues with numbers for more than a dozen lines is a
    design note that wandered into the code."""
    hits = [
        where
        for where, text in _python_prose()
        if len(text.splitlines()) > 12 and MEASUREMENT.search(text) and not POINTER.search(text)
    ]
    assert not hits, (
        "long measured docstrings without a docs/design pointer:\n  " + "\n  ".join(hits)
    )


def test_the_typing_page_counts_the_suppressions_the_package_carries() -> None:
    """`docs/design/typing.md` states a number, and a number goes stale silently.

    The page's whole claim is that every remaining suppression is listed with
    the reason it survives. It had drifted to eleven against fourteen, and
    four suppressions were absent from its table with nobody the wiser: a
    reader trusting the page would have concluded the package suppresses
    nothing in `execution.py`, `cli.py` or `models/enroll.py`. Counting here
    turns the next drift into a failure that names the new site.
    """
    page = (REPO / "docs" / "design" / "typing.md").read_text(encoding="utf-8")
    sites = [
        f"{path.relative_to(REPO)}:{n}"
        for path in sorted((REPO / "python" / "loudkit").rglob("*.py"))
        if "proto" not in path.parts
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if SUPPRESSION.search(line)
    ]
    stated = re.search(r"contains \*\*([a-z]+)\*\* `type: ignore` comments", page)
    assert stated, "typing.md no longer states how many suppressions the package carries"
    assert NUMBER_WORDS.get(stated.group(1)) == len(sites), (
        f"typing.md says {stated.group(1)}, the package carries {len(sites)}; "
        "update the page's table and its count:\n  " + "\n  ".join(sites)
    )


def test_the_suites_carry_no_suppression_no_checker_reads() -> None:
    """The suites carry no `# type: ignore` comments, checked or not.

    `tools/mypy-tests.ini` reads fourteen of them and nothing reads the rest.
    Outside a profile a suppression claims a check that does not run and stands
    ready to hide the error it names on the day one reaches it; inside a
    profile, the error it names should be fixed. Either way the honest comment
    is none, which is what `docs/design/typing.md` calls worse than nothing.
    """
    sites = [
        f"{path.relative_to(REPO)}:{tok.start[0]}"
        for path in sorted((REPO / "tests").rglob("*.py"))
        for tok in tokenize.generate_tokens(
            io.StringIO(path.read_text(encoding="utf-8")).readline
        )
        if tok.type == tokenize.COMMENT and UNREAD_SUPPRESSION.match(tok.string)
    ]
    assert not sites, (
        "suppressions in tests/, where no mypy profile reads them:\n  " + "\n  ".join(sites)
    )


def test_the_tree_holds_no_agent_diaries() -> None:
    """Changelogs belong in CHANGELOG.md; session records do not ship."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    diaries = [
        path
        for path in tracked
        if re.fullmatch(r"CHANGELOG-[^/]+\.md", path)
        or re.fullmatch(r"docs/design/(wave-[^/]+|release-review-[^/]+|[^/]*-status)\.md", path)
        or path.endswith(".ses")
    ]
    assert not diaries, "tracked session records:\n  " + "\n  ".join(diaries)


def test_every_docs_pointer_in_the_runtime_resolves() -> None:
    """A pointer to a page that is not there is worse than no pointer: the
    reader stops looking. The noise module cited ``docs/reference/typing.md``
    for a page that lives at ``docs/design/typing.md``."""
    dangling = sorted(
        {
            f"{where} -> {target}"
            for where, text in _python_prose()
            for target in DOC_POINTER.findall(text)
            if not (REPO / target).is_file()
        }
    )
    assert not dangling, "docs pointers that resolve to nothing:\n  " + "\n  ".join(dangling)


# Source a person or a tool can read: text. A file carrying a raw NUL is
# binary to grep, to `file` and to most review surfaces, which skip it in
# silence rather than reporting it.
TEXT_SUFFIXES = frozenset(
    [
        ".py",
        ".pyi",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".swift",
        ".sh",
        ".md",
        ".mdx",
        ".toml",
        ".yml",
        ".yaml",
        ".json",
        ".proto",
        ".txt",
        ".cfg",
        ".ini",
        ".html",
        ".css",
    ]
)


def test_no_tracked_source_carries_a_raw_nul() -> None:
    """A separator byte belongs in an escape, not in the file.

    ``site/scripts/sync-docs.mjs`` joined a path to its route id with a literal
    NUL to hash them, and that one byte made the whole script binary: grep
    matched nothing in it and reported nothing, so a search over ``site/``
    quietly missed the file that decides which pages the site publishes.
    ``\\0`` in the same template literal hashes the identical bytes.
    """
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    binary = [
        path
        for path in tracked
        if Path(path).suffix in TEXT_SUFFIXES
        and (REPO / path).is_file()
        and b"\0" in (REPO / path).read_bytes()
    ]
    assert not binary, "raw NUL in a text source; write the escape:\n  " + "\n  ".join(binary)


# Scripts under `tools/` whose only text output is a report of one run: a
# benchmark row, a probe dump, a divergence the fuzzer found. Nobody compares
# their bytes against anything, on any platform, so the rule below does not
# apply to them.
PER_RUN_REPORTS = frozenset(
    {
        "tools/bench.py",
        "tools/check_voice.py",
        "tools/decode_probe/probe.py",
        "tools/fuzz_parity.py",
        "tools/profile_stages.py",
    }
)


def test_generators_write_lf_into_the_files_other_implementations_read() -> None:
    """A generator under ``tools/`` writes LF, whatever platform it runs on.

    Python's text mode translates ``\\n`` to the platform's separator, so the
    same generator run on Windows writes a different file: `numerals.json` is
    hashed into the grammar digest and would move the fingerprint, the
    conformance fixtures are read by five implementations, and a bundle
    member's bytes are pinned in `SHA256SUMS`. Every one of those is a file
    whose bytes are a contract rather than a local matter.

    Checked at the call site because that is where the omission happens, and
    because the alternative is a test per artefact that only fires on the one
    platform nobody develops on.
    """
    offenders: list[str] = []
    for path in sorted((REPO / "tools").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel in PER_RUN_REPORTS:
            continue
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name == "open":
                mode = next(
                    (
                        arg.value
                        for arg in node.args[1:2]
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    ),
                    "r",
                )
                if "w" not in mode or "b" in mode:
                    continue
            elif name != "write_text":
                continue
            if "newline" not in {keyword.arg for keyword in node.keywords}:
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        'text write without newline="\\n"; these bytes are read elsewhere:\n  '
        + "\n  ".join(offenders)
    )
