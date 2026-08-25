"""Release assembly: layout, checksums, manifest — without the real weights.

The release is the only artefact a stranger receives, so its integrity is a
release gate, not a convenience. These tests build a fake release from tiny
files through the real ``tools/build_release.py`` and assert the properties
that a broken release would silently lose:

* ``SHA256SUMS`` paths are **relative to the release root** and verify with
  ``sha256sum -c`` run from that root — a checksum list that only records
  basenames silently skips every file in a subdirectory like ``voices/``.
* Every file ``release.json`` lists appears in ``SHA256SUMS`` and vice versa —
  no file ships without a checksum, no checksum names a missing file.
  ``release.json`` is one of them: it carries the profile and the verified
  flag, so it is written first and covered by the manifest.
* Voice encoder ``ve.safetensors`` (voice cloning, v0.1) is included when
  requested, with its own checksum.
* The model card's logo and listening samples live in the model bundle, not
  on another repository whose branch can drift or disappear.

These need no weights: the fake checkpoint is a few bytes. The load-and-speak
gate needs a real engine, so a ``lenient`` build turns it off with
``--skip-verify`` and a ``full-0.1`` build, which refuses that flag, replaces
it in the module through ``_build_strict``. Checksum integrity is what this
file pins.

The second half of the file attacks the builder rather than exercising it.
Every case there was found by pointing a reviewer at the tool and asking what
it would accept: two voices whose names collide into one bundle path, a roster
with nineteen of twenty, a stranger among the voices, a checkpoint under the
wrong name, an export directory carrying a file the profile does not name, and
a run that fails after it has started copying. Each must end in a refusal and
in nothing on disk, because the cost of the tool being wrong is a published
artefact.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parent.parent
BUILDER = REPO / "tools" / "build_release.py"


def _make_fake_release(tmp_path: Path) -> Path:
    """Assemble a release from tiny fake files; returns the release root."""
    src = tmp_path / "src"
    (src / "voices").mkdir(parents=True)
    (src / "loudr-1.safetensors").write_bytes(b"fake checkpoint")
    (src / "manifest.json").write_text('{"format": "loudkit-checkpoint"}', encoding="utf-8")
    (src / "tokenizer.json").write_text('{"vocab": {}}', encoding="utf-8")
    (src / "voices" / "testvoice.safetensors").write_bytes(b"fake voice")
    ve = src / "ve.safetensors"
    ve.write_bytes(b"fake voice encoder")

    out = tmp_path / "release"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_release.py"),
            "--checkpoint",
            str(src / "loudr-1.safetensors"),
            "--voice-encoder",
            str(ve),
            "--out",
            str(out),
            # A fake release of three tiny files is exactly what the
            # `full-0.1` profile refuses: no graphs, no packages. The
            # checksum properties below hold for either profile, so this
            # asks for the lenient one and leaves the strict one to
            # `tools/build_release.py`'s own refusal.
            "--profile",
            "lenient",
            "--skip-verify",
        ],
        check=True,
        cwd=REPO,
    )
    return out


def _tomllib():  # type: ignore[no-untyped-def]
    """`tomllib`, or `tomli` on the Python this project still promises.

    `requires-python` says >=3.10 and the CI matrix runs 3.10, where `tomllib`
    does not exist -- it landed in 3.11. These three tests therefore failed on
    the one interpreter the floor exists to prove, and the failure looked like a
    broken test rather than a broken promise.

    Fixed here rather than by raising the floor, because the floor is a claim
    made to users and this is a claim made by the test suite. `tomli` is in the
    dev extra under the same marker.
    """
    try:
        import tomllib

        return tomllib
    except ModuleNotFoundError:  # pragma: no cover - only on 3.10
        import tomli

        return tomli


def test_checksums_verify_from_release_root(tmp_path: Path) -> None:
    """``sha256sum -c SHA256SUMS`` from the release root must pass, including
    files in subdirectories — a basename-only list fails here."""
    out = _make_fake_release(tmp_path)
    # SHA256SUMS paths are relative to the release root; run the check there.
    result = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=out,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"checksum verification failed:\n{result.stdout}\n{result.stderr}"
    )
    # Every listed path must actually resolve.
    for line in result.stdout.splitlines():
        assert "OK" in line, f"unverified entry: {line}"


def test_checksum_paths_are_relative_and_match_manifest(tmp_path: Path) -> None:
    """The set of paths in SHA256SUMS equals the set in release.json, plus
    ``release.json`` itself — nothing ships without a checksum, no checksum
    names a missing file.

    ``release.json`` carries the profile and the verified flag, so it is
    written first and checksummed like everything else. It cannot list itself,
    which is why it is added on this side of the comparison.
    """
    out = _make_fake_release(tmp_path)
    manifest = json.loads((out / "release.json").read_text(encoding="utf-8"))

    def manifest_paths() -> set[str]:
        paths: set[str] = set()
        for entry in manifest.values():
            if isinstance(entry, list):
                paths.update(e["path"] for e in entry)
            elif isinstance(entry, dict):
                paths.add(entry["path"])
        paths.add("release.json")
        return paths

    sums_paths = {
        line.split("  ")[-1].strip()
        for line in (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }

    assert sums_paths == manifest_paths(), (
        f"SHA256SUMS and release.json disagree on files:\n"
        f"  in sums only: {sums_paths - manifest_paths()}\n"
        f"  in manifest only: {manifest_paths() - sums_paths}"
    )
    # Every checksum path must resolve relative to the release root.
    for p in sums_paths:
        assert (out / p).is_file(), f"checksum names a missing file: {p}"
    # No duplicate checksum entries.
    assert len(sums_paths) == len((out / "SHA256SUMS").read_text(encoding="utf-8").splitlines())


def test_listening_samples_are_release_artifacts_not_external_dependencies(
    tmp_path: Path,
) -> None:
    """The model card's players resolve to bytes this exact bundle vouches for.

    A raw GitHub URL made the card prettier but left a model release dependent
    on another repository, branch and deployment.  The samples belong beside
    the model, with manifest entries and checksum lines like every weight and
    graph.
    """
    out = _make_fake_release(tmp_path)
    module = _builder()
    manifest = json.loads((out / "release.json").read_text(encoding="utf-8"))
    sums = (out / "SHA256SUMS").read_text(encoding="utf-8")

    expected = {name for _source, name in module.SAMPLES}
    assert {entry["path"] for entry in manifest["samples"]} == expected
    for source, name in module.SAMPLES:
        assert (out / name).read_bytes() == (REPO / source).read_bytes()
        assert f"  {name}\n" in sums

    card = (out / "README.md").read_text(encoding="utf-8")
    assert "raw.githubusercontent.com" not in card
    for name in expected:
        url = f"https://huggingface.co/loudreader/loudr-1/resolve/main/{name}"
        assert f'<audio controls src="{url}"></audio>' in card


def test_brand_image_is_part_of_the_release(tmp_path: Path) -> None:
    """The model card logo resolves to bytes vouched for by this bundle."""
    out = _make_fake_release(tmp_path)
    module = _builder()
    manifest = json.loads((out / "release.json").read_text(encoding="utf-8"))
    source, name, key = module.BRANDING

    assert manifest[key]["path"] == name
    assert (out / name).read_bytes() == (REPO / source).read_bytes()
    assert f"  {name}\n" in (out / "SHA256SUMS").read_text(encoding="utf-8")

    card = (out / "README.md").read_text(encoding="utf-8")
    url = f"https://huggingface.co/loudreader/loudr-1/resolve/main/{name}"
    assert f'<img src="{url}"' in card


def test_coreml_enrollment_gate_uses_the_shipped_cpu_placement() -> None:
    """Release and export gates must exercise Swift's actual placement."""
    module = _builder()
    exporter = ast.parse(
        (REPO / "tools" / "export_enroll_coreml.py").read_text(encoding="utf-8")
    )
    assignment = next(
        node
        for node in exporter.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_GATE_SCRIPT"
            for target in node.targets
        )
    )
    export_gate = ast.literal_eval(assignment.value)
    placement = "compute_units=ct.ComputeUnit.CPU_ONLY"

    assert module._COREML_ENROLL.count(placement) == 1
    assert export_gate.count(placement) == 1


def test_voice_encoder_ships_with_checksum(tmp_path: Path) -> None:
    """ve.safetensors is part of the cloning-capable release and is checksummed."""
    out = _make_fake_release(tmp_path)
    assert (out / "ve.safetensors").is_file()
    manifest = json.loads((out / "release.json").read_text(encoding="utf-8"))
    assert manifest["voice_encoder"]["path"] == "ve.safetensors"
    sums = (out / "SHA256SUMS").read_text(encoding="utf-8")
    assert "ve.safetensors" in sums


def test_sdist_target_is_explicit_allowlist() -> None:
    """The sdist must be scoped to the package source, not the monorepo.

    Without an explicit ``[tool.hatch.build.targets.sdist]``, hatchling ships
    "whole tree minus VCS ignores" — which would put integrations/ (1.1 GB of
    node_modules, Go and Rust trees), Sources/, Examples/ and tb-science/ into
    the published source archive. The allowlist below is the contract; CI builds
    the sdist and asserts no file escapes it.
    """
    tomllib = _tomllib()

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    sdist = (
        pyproject.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("sdist")
    )
    assert sdist is not None, (
        "missing [tool.hatch.build.targets.sdist] — sdist would ship the monorepo"
    )
    only = sdist.get("only-include", [])
    allowed = {p.rstrip("/") for p in only}
    assert allowed == {
        "python/loudkit",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "NOTICE",
        "RESPONSIBLE_USE.md",
    }, f"sdist only-include drifted: {only}"
    # None of the monorepo's non-package directories may be allowed.
    for banned in (
        "integrations",
        "swift",
        "go",
        "rust",
        "js",
        "Examples",
        "openspec",
        "tests",
        "tools",
    ):
        assert banned not in allowed, f"sdist allowlist permits monorepo dir: {banned}"

    excluded = set(sdist.get("exclude", []))
    assert "python/loudkit/models/data/dsp/*.f32" in excluded


def _extras() -> dict[str, set[str]]:
    """Every extra's transitive distribution set, resolving ``loudkit[x]``.

    Extras may name the package itself (``loudkit[server]``), which is how
    ``[mcp]`` inherits the HTTP server's dependencies. Resolving that here means
    the assertions below describe what a user actually receives rather than what
    one table row happens to list.
    """
    import re

    tomllib = _tomllib()

    raw = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    table: dict[str, list[str]] = raw["project"]["optional-dependencies"]

    def resolve(name: str, seen: frozenset[str] = frozenset()) -> set[str]:
        if name in seen:  # a cycle would hang; extras are not supposed to have one
            raise AssertionError(f"extras cycle through {name!r}")
        out: set[str] = set()
        for spec in table[name]:
            self_ref = re.fullmatch(r"loudkit\[([^]]+)\]", spec.strip())
            if self_ref:
                for inner in self_ref.group(1).split(","):
                    out |= resolve(inner.strip(), seen | {name})
            else:
                out.add(re.split(r"[<>=!~\[ ]", spec.strip(), maxsplit=1)[0])
        return out

    return {name: resolve(name) for name in table}


def test_documented_extras_can_actually_synthesise() -> None:
    """Every extra the README tells a stranger to install must be sufficient.

    This is the packaging equivalent of the founding defect: an extra that
    installs everything except the thing that makes sound produces a documented
    command that fails, and no test of the library itself notices — the modules
    are fine, the install is not.

    ``[server]`` and ``[mcp]`` both end up calling ``loudkit.load()``, whose
    default device is the torch backend, so both need torch. ``[dev]`` needs it
    because ``tests/test_models.py`` imports torch at module scope and CI
    installs nothing else, so a missing torch fails at *collection* — every test
    in the run disappears rather than one failing loudly.
    """
    extras = _extras()

    for extra, required in {
        "server": {"fastapi", "uvicorn", "soundfile", "torch"},
        # [mcp] no longer inherits [server]: loudkit.mcp takes its synthesis
        # surface from loudkit.synthesis, so fastapi is not its dependency.
        # torch rides along explicitly because load() defaults to that backend.
        "mcp": {"mcp", "soundfile", "torch"},
        "dev": {"pytest", "mypy", "ruff", "torch"},
        "enroll": {"torch", "torchaudio", "librosa"},
    }.items():
        missing = required - extras[extra]
        assert not missing, f"loudkit[{extra}] is missing {sorted(missing)}"
    assert "fastapi" not in extras["mcp"], (
        "[mcp] must not carry fastapi — the MCP transport does not import the HTTP one"
    )


def test_mcp_imports_the_synthesis_surface_not_the_http_transport() -> None:
    """``loudkit.mcp`` takes render_bytes from ``synthesis``, not from ``server``.

    Pinned as an import edge because the dependency list follows from it: while
    mcp imports only synthesis (transport-agnostic, fastapi-free), the ``[mcp]``
    extra must not carry fastapi. If mcp ever imports ``loudkit.server`` again,
    that is the moment to reintroduce the inheritance — consciously, here.
    """
    import ast

    tree = ast.parse(
        (REPO / "python" / "loudkit" / "transports" / "mcp.py").read_text(encoding="utf-8")
    )
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "synthesis"
    }
    assert "synthesis" in modules, "loudkit.transports.mcp must import loudkit.synthesis"
    assert "server" not in modules, (
        "loudkit.transports.mcp imports loudkit.server — transports are peers; take the "
        "synthesis surface from loudkit.synthesis"
    )


# --------------------------------------------------------------- attribution

_PORT_PACKAGES = ("go", "rust", "js")


def test_every_published_package_carries_the_licence_and_the_notice() -> None:
    """Each port ships LICENSE and NOTICE, byte-identical to the root pair.

    npm, crates.io and the Go module proxy each publish their directory, not
    this repository, so a recipient of the npm tarball receives the terms only
    if the tarball contains them — and all three shipped without either until
    2026-08-17. Apache-2.0 §4(a) requires giving the licence to every recipient
    and §4(d) requires carrying the NOTICE forward, and for two of the three the
    obligation is not only Apache's: Go embeds ``pl_en_respell.json`` and npm
    ships it in ``data/``, and that file is a derived work of CMUdict, whose
    own terms require its notice to travel with binary distributions.

    Byte-identical rather than merely present, because a copy that drifts is
    how the shared grammar data ended up two features behind in four ports at
    once. A copy nobody compares is a copy nobody maintains.
    """
    root_licence = (REPO / "LICENSE").read_bytes()
    root_notice = (REPO / "NOTICE").read_bytes()
    for package in _PORT_PACKAGES:
        for name, expected in (("LICENSE", root_licence), ("NOTICE", root_notice)):
            path = REPO / package / name
            assert path.exists(), (
                f"{package}/{name} is missing — that package publishes without terms"
            )
            assert path.read_bytes() == expected, (
                f"{package}/{name} has drifted from the root copy; "
                f"re-copy it rather than editing it in place"
            )


def test_the_npm_tarball_lists_its_terms_and_dual_use_disclosure() -> None:
    """``files`` decides the tarball, so present-but-unlisted is absent.

    Voice enrollment is a dual-use capability under npm's current policy. The
    declaration is permanent once published, so both halves are pinned here:
    machine-readable metadata for the registry and an explanation for people.
    """
    package = json.loads((REPO / "js" / "package.json").read_text(encoding="utf-8"))
    listed = set(package["files"])
    assert {"LICENSE", "NOTICE", "DISCLOSURE"} <= listed, (
        f"package.json files omits terms or disclosure: {sorted(listed)}"
    )
    assert package["license"] == "Apache-2.0"
    assert package["contentPolicy"] == {"class": "dual-use"}
    disclosure = (REPO / "js" / "DISCLOSURE").read_text(encoding="utf-8").lower()
    assert "voice enrollment" in disclosure
    assert "impersonation" in disclosure
    assert "permission" in disclosure


def test_dual_use_npm_release_never_automates_a_direct_publish() -> None:
    """npm requires proof of presence for this declared dual-use package."""
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    publish = workflow.split("\n  publish-npm:\n", 1)[1].split("\n  publish-pypi:\n", 1)[0]
    commands = [
        line.strip() for line in publish.splitlines() if not line.lstrip().startswith("#")
    ]
    assert "NPM_BOOTSTRAP_TOKEN" not in workflow
    assert any(line.startswith('npm stage publish "$tarball"') for line in commands)
    assert not any(line.startswith("npm publish ") for line in commands)
    assert "attest-npm" in publish


def test_crates_release_bootstraps_once_then_uses_short_lived_oidc() -> None:
    """The first crate is manual; later tags must not need a stored token."""
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    publish = workflow.split("\n  publish-crates:\n", 1)[1].split("\n  # --- the record", 1)[0]
    assert "id-token: write" in publish
    assert "rust-lang/crates-io-auth-action@c6f97d42243bad5fab37ca0427f495c86d5b1a18" in publish
    assert "secrets.CARGO_REGISTRY_TOKEN" not in workflow
    assert "steps.crates-auth.outputs.token" in publish
    assert "version_checksum" in publish
    assert "cargo package --locked" in publish
    assert "name: crate-tarball" in workflow
    assert "the checked crate has checksum" in publish


def test_release_tags_name_the_isolated_public_branch_explicitly() -> None:
    """Running the release commands on a private branch must not expose it."""
    releasing = (REPO / "RELEASING.md").read_text(encoding="utf-8")
    assert "git tag v0.1.0 public-main" in releasing
    assert "git tag go/v0.1.0 public-main" in releasing


def test_parity_uses_a_pinned_public_release_on_an_isolated_runner() -> None:
    """Public workflow code must not need a maintainer's machine or hidden WAV."""
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    parity = workflow.split("\n  parity:\n", 1)[1].split("\n  packaging:\n", 1)[0]
    revision = re.search(r'LOUDKIT_HF_REVISION: "([0-9a-f]{40})"', parity)
    install = re.search(r'pip install -e "\.\[([^]]+)]"', parity)
    assert revision, "parity must pin an immutable Hugging Face commit"
    assert install, "parity must install the runtimes it measures"
    extras = {extra.strip() for extra in install.group(1).split(",")}
    assert "enroll" in extras, "enrollment parity needs the torchaudio runtime"
    assert "runs-on: macos-latest" in parity
    assert "runs-on: [self-hosted" not in parity
    assert "loudkit download loudreader/loudr-1" in parity
    assert '--revision "$LOUDKIT_HF_REVISION"' in parity
    assert "for backend in torch onnx coreml" in parity
    assert "LOUDKIT_REFERENCE_WAV" not in parity

    assets = (REPO / "tests" / "assets.py").read_text(encoding="utf-8")
    reference = (REPO / "tests" / "test_parity.py").read_text(encoding="utf-8")
    assert '"reference_wav":' not in assets
    assert 'ENROLLMENT / "ref_audio.f32"' in reference


def test_the_crate_declares_a_licence_and_packs_the_notice() -> None:
    """``cargo publish`` refuses a crate with no licence and accepts one with no
    NOTICE, so the include list is what makes the second true."""
    tomllib = _tomllib()

    cargo = tomllib.loads((REPO / "rust" / "Cargo.toml").read_text(encoding="utf-8"))
    package = cargo["package"]
    assert package.get("license") == "Apache-2.0", "crate has no licence to publish under"
    include = set(package.get("include", []))
    assert {"LICENSE", "NOTICE"} <= include, f"crate include omits the terms: {sorted(include)}"


def _normalised_version(raw: str) -> tuple[str, str]:
    """``(release, prerelease)`` with each ecosystem's punctuation removed.

    The same release is spelled three ways and all three are correct: PEP 440
    writes ``0.1.0.dev0``, npm's semver ``0.1.0-dev0``, Cargo's ``0.1.0-dev.0``.
    Comparing the strings would fail on every pre-release and comparing only
    ``0.1.0`` would pass while npm shipped ``dev`` and PyPI shipped the release.
    Split at the release triple and strip the separators from the tail, so what
    is compared is what the three files are trying to say.
    """
    import re

    match = re.fullmatch(r"(\d+\.\d+\.\d+)(.*)", raw.strip())
    assert match, f"version {raw!r} does not start with a major.minor.patch triple"
    return match.group(1), re.sub(r"[^0-9a-z]", "", match.group(2).lower())


def test_every_published_manifest_carries_the_same_version() -> None:
    """PyPI, npm and crates.io publish from three files; the tag checks one.

    ``release.yml`` compares the tag against ``pyproject.toml`` alone, so the
    npm tarball and the crate can carry ``dev`` into a release that everything
    else calls ``0.1.0`` — and each registry's publish is irreversible, which
    makes "we noticed afterwards" the expensive outcome. ``__init__.py`` is
    included because it holds a *fourth* hard-coded copy: ``loudkit.__version__``
    is what the RELEASING acceptance pass prints and compares against the tag,
    and nothing else compares it to ``pyproject.toml``.

    Go and Swift take their version from the git tag and have nothing to carry;
    ``test_the_ports_without_a_manifest_version_still_have_none`` keeps that
    true rather than assuming it.
    """
    import re

    tomllib = _tomllib()

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    cargo = tomllib.loads((REPO / "rust" / "Cargo.toml").read_text(encoding="utf-8"))
    package = json.loads((REPO / "js" / "package.json").read_text(encoding="utf-8"))
    init = (REPO / "python" / "loudkit" / "__init__.py").read_text(encoding="utf-8")
    dunder = re.search(r'^__version__ = "([^"]+)"', init, re.MULTILINE)
    assert dunder, "python/loudkit/__init__.py no longer defines __version__"

    expected = _normalised_version(pyproject["project"]["version"])
    for label, raw in (
        ("js/package.json", package["version"]),
        ("rust/Cargo.toml", cargo["package"]["version"]),
        ("python/loudkit/__init__.py __version__", dunder.group(1)),
    ):
        assert _normalised_version(raw) == expected, (
            f"{label} says {raw!r} and pyproject.toml says "
            f"{pyproject['project']['version']!r} — one registry would publish "
            f"the other's version"
        )


def test_go_and_swift_still_take_their_version_from_the_tag() -> None:
    """The reason the check above names three files and not five.

    ``go.mod`` and ``Package.swift`` declare no version: the module proxy and
    SwiftPM read it from the git tag, so there is nothing to keep in step and
    RELEASING.md §1 says so. That is an assumption the test above rests on, and
    the day a manifest grows a ``version`` field it becomes another thing to
    edit at release time that no check knows about.
    """
    import re

    # Whitespace-flattened: the sentence is wrapped in the source, and a test
    # that breaks on a reflow is a test about line lengths.
    releasing = " ".join((REPO / "RELEASING.md").read_text(encoding="utf-8").split())
    assert "Go and Swift take their version from the git tag" in releasing, (
        "RELEASING.md §1 no longer says where Go and Swift take their version; "
        "if that changed, the version-sync check has to grow with it"
    )
    for manifest in (REPO / "go" / "go.mod", REPO / "Package.swift"):
        text = manifest.read_text(encoding="utf-8")
        assert not re.search(r"^\s*version\s*[:=]", text, re.MULTILINE | re.IGNORECASE), (
            f"{manifest.name} now declares a version; add it to RELEASING.md's table "
            f"and to test_every_published_manifest_carries_the_same_version"
        )


def test_the_release_table_names_the_versions_the_files_carry() -> None:
    """`RELEASING.md` said Rust's pre-release version was `0.1.0`; Cargo says
    `0.1.0-dev.0`.

    The version-sync workflow checks `pyproject.toml` alone, so a table that
    disagrees with a manifest is not caught anywhere — and this table is what
    someone follows at release time, when the cost of being wrong is a
    published artefact.
    """
    tomllib = _tomllib()

    releasing = (REPO / "RELEASING.md").read_text(encoding="utf-8")
    cargo = tomllib.loads((REPO / "rust" / "Cargo.toml").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    package = json.loads((REPO / "js" / "package.json").read_text(encoding="utf-8"))

    for label, version in (
        ("`rust/Cargo.toml`", cargo["package"]["version"]),
        ("`pyproject.toml`", pyproject["project"]["version"]),
        ("`js/package.json`", package["version"]),
    ):
        row = next(
            (line for line in releasing.splitlines() if line.startswith(f"| {label} |")),
            None,
        )
        assert row is not None, f"{label} has no row in the version table"
        assert f"`{version}`" in row, (
            f"the release table says {row.strip()} and the file says {version!r}"
        )


# ------------------------------------------------------- attacking the builder


def _builder():  # type: ignore[no-untyped-def]
    """``tools/build_release.py`` as a module, for the unit-level checks.

    ``tools/`` is not a package and is not on the path, so the module is loaded
    from its file. The tests below that drive the command line use a
    subprocess instead; this import is only for the pure functions.
    """
    spec = importlib.util.spec_from_file_location("build_release", BUILDER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _roster() -> list[str]:
    entries = json.loads(
        (REPO / "docs" / "voices" / "roster" / "provenance.json").read_text(encoding="utf-8")
    )
    return [e["name"] for e in entries]


SYNTHESIS_TENSORS = ("t3.text_emb.weight", "s3gen.flow.encoder.w", "s3gen.mel2wav.conv.w")
ENROLLMENT_TENSORS = ("s3gen.tokenizer.enc.w", "s3gen.speaker_encoder.fc.w")


def _write_safetensors(path: Path, names: Sequence[str], manifest: dict[str, object]) -> None:
    """A safetensors file of one-element float32 tensors, by hand.

    The builder's split check reads headers and never a tensor, so the tests
    that attack it need headers and never a tensor. Writing the container here
    rather than through ``safetensors.torch`` keeps these cases in reach: a
    pair whose halves overlap, or whose roles are swapped, is not something the
    real tool will produce, and is exactly what the builder must refuse.
    """
    header: dict[str, object] = {}
    offset = 0
    for name in sorted(names):
        header[name] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    header["__metadata__"] = {"manifest": json.dumps(manifest, sort_keys=True)}
    blob = json.dumps(header).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))


def _fixture_payload_sha256(names: Sequence[str]) -> str:
    """The digest `_write_safetensors` output really has.

    Same recipe as `tools/split_checkpoint.py:payload_sha256`, over the tensors
    that writer produces: one float32 zero each, in sorted name order. The
    builder now opens the halves and compares, so a fixture carrying an invented
    digest would be testing a bundle no split can produce.
    """
    h = hashlib.sha256()
    for name in sorted(names):
        h.update(name.encode())
        h.update(b"torch.float32")
        h.update(b"(1,)")
        h.update(bytes(4))
    return h.hexdigest()


def _split_pair(
    src: Path,
    *,
    synthesis: Sequence[str] = SYNTHESIS_TENSORS,
    enrollment: Sequence[str] = ENROLLMENT_TENSORS,
    source_names: Sequence[str] | None = None,
    roles: tuple[str, str] = ("synthesis", "enrollment"),
    synthesis_name: str = "loudr-1.safetensors",
    enrollment_source_payload: str = "a" * 64,
) -> None:
    """Both halves, with the split provenance a real split would have written.

    ``source_names`` defaults to the union, which is what makes the pair
    complete; passing a longer list is how a test says "a tensor went missing
    between the split and here" without deleting a file.
    """
    module = _builder()
    whole = sorted(source_names if source_names is not None else [*synthesis, *enrollment])
    block = {
        "source_payload_sha256": "a" * 64,
        "source_tensor_names_sha256": hashlib.sha256("\n".join(whole).encode()).hexdigest(),
        "source_tensor_count": len(whole),
        "roles": module.ROLE_FILENAMES,
    }
    halves = (
        (roles[0], synthesis, synthesis_name, _fixture_payload_sha256(synthesis), block),
        (
            roles[1],
            enrollment,
            module.ENROLLMENT_CHECKPOINT_NAME,
            _fixture_payload_sha256(enrollment),
            {**block, "source_payload_sha256": enrollment_source_payload},
        ),
    )
    for role, names, filename, payload, split in halves:
        _write_safetensors(
            src / filename,
            names,
            {
                "format": "loudkit-checkpoint",
                "artifact_role": role,
                "tensor_payload_sha256": payload,
                "tensor_count": len(names),
                "split": split,
            },
        )


def _fake_sources(
    tmp_path: Path,
    *,
    voices: list[str] | None = None,
    checkpoint_name: str = "loudr-1.safetensors",
    extra_onnx: str | None = None,
) -> Path:
    """A complete-looking artefact set of tiny files.

    Enough for the ``full-0.1`` preflight to pass, so the checks after it are
    the ones under test. Nothing here can load, which is why every strict build
    in this file passes ``--skip-verify``: what is being attacked is the
    assembly, not the engine.

    The two halves of the checkpoint are real safetensors containers holding
    five one-element tensors between them, because the preflight now reads
    their headers: it checks each half's ``artifact_role`` and proves the pair
    disjoint and complete before a byte is copied.
    """
    src = tmp_path / "src"
    (src / "voices").mkdir(parents=True)
    _split_pair(src, synthesis_name=checkpoint_name)
    (src / "manifest.json").write_text('{"format": "loudkit-checkpoint"}', encoding="utf-8")
    (src / "tokenizer.json").write_text('{"vocab": {}}', encoding="utf-8")
    (src / "ve.safetensors").write_bytes(b"fake voice encoder")
    for name in _roster() if voices is None else voices:
        (src / "voices" / f"{name}.safetensors").write_bytes(f"fake {name}".encode())

    module = _builder()
    onnx = src / "onnx"
    onnx.mkdir()
    for name in module.SYNTHESIS_ONNX + module.ENROLL_ONNX:
        (onnx / name).write_bytes(b"fake graph")
    if extra_onnx:
        (onnx / extra_onnx).write_bytes(b"a passenger")
    coreml = src / "coreml"
    for name in module.SYNTHESIS_COREML + module.ENROLL_COREML:
        (coreml / name / "Data").mkdir(parents=True)
        (coreml / name / "Manifest.json").write_text("{}", encoding="utf-8")
        (coreml / name / "Data" / "model.mlmodel").write_bytes(b"fake package")
    return src


def _synthesis_half(src: Path) -> Path:
    """The file ``--checkpoint`` names: not the voice encoder, not the other half.

    Three ``.safetensors`` files now sit at the top of an artefact set, and
    ``loudr-1-enrollment.safetensors`` sorts *before* ``loudr-1.safetensors``:
    picking the first one that is not the voice encoder would hand the builder
    the enrollment half and test the wrong refusal.
    """
    module = _builder()
    excluded = {"ve.safetensors", module.ENROLLMENT_CHECKPOINT_NAME}
    return next(p for p in sorted(src.glob("*.safetensors")) if p.name not in excluded)


def _build(src: Path, out: Path, *args: str) -> subprocess.CompletedProcess[str]:
    checkpoint = _synthesis_half(src)
    return subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--checkpoint",
            str(checkpoint),
            "--voice-encoder",
            str(src / "ve.safetensors"),
            "--voices",
            str(src / "voices"),
            "--out",
            str(out),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )


def _build_strict(monkeypatch: pytest.MonkeyPatch, src: Path, out: Path, *args: str) -> int:
    """A ``full-0.1`` build with the load-and-speak gate stubbed out.

    ``full-0.1`` refuses ``--skip-verify``, and a checkpoint of fifteen bytes
    cannot load, so the gate is replaced in the module rather than turned off
    from the command line. Everything else is the real build: the preflight,
    the copy by name, the allowlist audit, both manifests and the verified
    flag. Only the engine, and the Apple platform the CoreML half of the gate
    requires, are faked.
    """
    module = _builder()
    monkeypatch.setattr(module, "verify", lambda _out, **_kw: 0)
    monkeypatch.setattr(sys, "platform", "darwin")
    checkpoint = _synthesis_half(src)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(BUILDER),
            "--checkpoint",
            str(checkpoint),
            "--voice-encoder",
            str(src / "ve.safetensors"),
            "--voices",
            str(src / "voices"),
            "--out",
            str(out),
            *args,
        ],
    )
    return int(module.main())


def _nothing_left(out: Path) -> None:
    """No release, and no staging or previous directory beside where it went.

    The whole point of the staging rename: a run that refuses or fails must
    leave nothing an operator could mistake for a bundle, and nothing a `du`
    would find later and nobody would dare delete.
    """
    assert not out.exists(), f"a failed build left {out}"
    strays = [p.name for p in out.parent.iterdir() if p.name.startswith(f".{out.name}.")]
    assert not strays, f"a failed build left {strays} beside {out}"


def test_two_voices_that_normalise_to_one_path_are_refused(tmp_path: Path) -> None:
    """``a.voice.safetensors`` and ``a.safetensors`` both want ``voices/a.safetensors``.

    The builder stripped the ``.voice`` suffix, so the second copy landed on
    the first, both were hashed, and ``SHA256SUMS`` carried two lines for one
    path with two different digests. The build reported success and
    ``shasum -c`` failed on the bundle — the worst shape of failure, because
    it is found by the downloader rather than by the operator.
    """
    src = tmp_path / "src"
    (src / "voices").mkdir(parents=True)
    (src / "loudr-1.safetensors").write_bytes(b"fake checkpoint")
    (src / "manifest.json").write_text("{}", encoding="utf-8")
    (src / "tokenizer.json").write_text("{}", encoding="utf-8")
    (src / "ve.safetensors").write_bytes(b"fake voice encoder")
    (src / "voices" / "clash.safetensors").write_bytes(b"one")
    (src / "voices" / "clash.voice.safetensors").write_bytes(b"another")

    out = tmp_path / "release"
    result = _build(src, out, "--profile", "lenient", "--skip-verify")

    assert result.returncode != 0, result.stdout
    assert "voices/clash.safetensors" in result.stderr
    # Both sources are named: the operator has to know which two files to fix.
    assert "clash.safetensors" in result.stderr
    assert "clash.voice.safetensors" in result.stderr
    _nothing_left(out)


def test_strict_requires_the_whole_roster_by_name(tmp_path: Path) -> None:
    """Nineteen of twenty is not the release, and neither is one arbitrary voice.

    ``full-0.1`` used to ask only that *some* voice was present, so a bundle
    carrying a single profile assembled, verified and was publishable. The
    roster is ``docs/voices/roster/provenance.json``; the profile requires
    every name in it.
    """
    roster = _roster()
    src = _fake_sources(tmp_path, voices=roster[:-1])
    out = tmp_path / "release"
    result = _build(src, out, "--skip-verify")

    assert result.returncode != 0, result.stdout
    assert f"voices/{roster[-1]}.safetensors" in result.stderr
    _nothing_left(out)


def test_strict_refuses_a_voice_that_is_not_on_the_roster(tmp_path: Path) -> None:
    """A stranger among the twenty is a refusal, not a bonus."""
    src = _fake_sources(tmp_path, voices=[*_roster(), "stranger"])
    out = tmp_path / "release"
    result = _build(src, out, "--skip-verify")

    assert result.returncode != 0, result.stdout
    assert "voices/stranger.safetensors" in result.stderr
    assert "not on the roster" in result.stderr
    _nothing_left(out)


def test_strict_requires_the_canonical_checkpoint_name(tmp_path: Path) -> None:
    """Every guide, every port and ``hub`` name ``loudr-1.safetensors``.

    A checkpoint under another name copies, checksums and loads on the build
    machine, and breaks for everyone who follows a document.
    """
    src = _fake_sources(tmp_path, checkpoint_name="model.safetensors")
    out = tmp_path / "release"
    result = _build(src, out, "--skip-verify")

    assert result.returncode != 0, result.stdout
    assert "checkpoint name" in result.stderr
    assert "loudr-1.safetensors" in result.stderr
    _nothing_left(out)


def test_strict_ships_only_what_the_profile_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file the export directory happened to hold does not become the release.

    The builder copied the ``onnx/`` and ``coreml/`` directories wholesale, so
    a stale graph, an interrupted export's ``.tmp.mlpackage`` or an editor's
    backup shipped and was checksummed as part of the release. ``full-0.1``
    now copies by name and then audits the assembled bundle against the
    allowlist.
    """
    src = _fake_sources(tmp_path, extra_onnx="leftover.tmp.onnx")
    out = tmp_path / "release"
    assert _build_strict(monkeypatch, src, out) == 0
    shipped = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert "onnx/leftover.tmp.onnx" not in shipped
    module = _builder()
    paths, prefixes = module._allowlist(roster=_roster(), ships_onnx=True, ships_coreml=True)
    unexpected = {p for p in shipped - paths if not p.startswith(prefixes)}
    assert not unexpected, f"the bundle carries files the profile does not name: {unexpected}"


def test_the_allowlist_audit_rejects_a_stray_file(tmp_path: Path) -> None:
    """The audit itself, on a directory nobody could have assembled by hand.

    The copy step is exact, so the audit is a second wall rather than the
    first. It is unit-tested because it must keep working even when no copy
    path can produce a stray.
    """
    module = _builder()
    root = tmp_path / "bundle"
    (root / "voices").mkdir(parents=True)
    (root / "loudr-1.safetensors").write_bytes(b"x")
    (root / "notes.txt").write_bytes(b"left behind")

    paths, prefixes = module._allowlist(roster=["joe"], ships_onnx=False, ships_coreml=False)
    drift = module._audit(root, paths, prefixes)
    assert any("notes.txt" in line for line in drift)
    assert any("voices/joe.safetensors" in line for line in drift)


def test_a_build_that_fails_after_copying_leaves_nothing(tmp_path: Path) -> None:
    """The closing gate runs inside the staging directory, before the rename.

    A failed run used to leave ``release.json``, ``SHA256SUMS`` and a
    directory that looks exactly like a release: nothing on disk said it was
    the wreckage of a build that refused. This one copies every file, writes
    both manifests, then fails to load a checkpoint that is fifteen bytes of
    text — and the target must not exist afterwards.
    """
    src = tmp_path / "src"
    (src / "voices").mkdir(parents=True)
    (src / "loudr-1.safetensors").write_bytes(b"fake checkpoint")
    (src / "manifest.json").write_text("{}", encoding="utf-8")
    (src / "tokenizer.json").write_text("{}", encoding="utf-8")
    (src / "ve.safetensors").write_bytes(b"fake voice encoder")
    (src / "voices" / "testvoice.safetensors").write_bytes(b"fake voice")

    out = tmp_path / "release"
    result = _build(src, out, "--profile", "lenient")

    assert result.returncode != 0, result.stdout
    assert "FAILED" in result.stdout + result.stderr
    _nothing_left(out)


def _staging_stub(out: Path, pid: int, *, suffix: str = "staging") -> Path:
    """A staging tree beside ``out``, stamped with ``pid``, holding one file."""
    stray = out.parent / f".{out.name}.{suffix}-{pid}"
    (stray / "voices").mkdir(parents=True)
    (stray / "voices" / "carmen.safetensors").write_bytes(b"4.6 GB, in spirit")
    return stray


def _dead_pid() -> int:
    """The pid of a process that has run and been reaped."""
    done = subprocess.Popen([sys.executable, "-c", ""])
    done.wait()
    return done.pid


def test_a_build_reclaims_the_staging_tree_of_a_build_that_was_killed(tmp_path: Path) -> None:
    """``main``'s ``finally`` cannot run when the process is killed outright.

    A SIGSEGV in the closing gate left the whole staging tree on disk, and
    three killed runs left three of them. The target was correctly absent
    every time, so nothing looked publishable, but 4.6 GB accumulated per
    crash with nothing on disk saying it could go.
    """
    src = _fake_sources(tmp_path)
    out = tmp_path / "release"
    out.parent.mkdir(parents=True, exist_ok=True)
    stray = _staging_stub(out, _dead_pid())

    result = _build(src, out, "--profile", "lenient", "--skip-verify")

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert not stray.exists(), "the staging tree of a dead build survived the next run"
    assert stray.name in result.stdout, "the sweep removed it without saying so"


def test_a_build_leaves_the_staging_tree_of_a_running_build_alone(tmp_path: Path) -> None:
    """Two builds of the same target must not eat each other.

    The pid in the name is what makes the sweep safe: a tree whose owner is
    still running is a build in progress, not wreckage. This one is stamped
    with the pid of the test process, which is running by definition.
    """
    src = _fake_sources(tmp_path)
    out = tmp_path / "release"
    out.parent.mkdir(parents=True, exist_ok=True)
    stray = _staging_stub(out, os.getpid())

    result = _build(src, out, "--profile", "lenient", "--skip-verify")

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert (stray / "voices" / "carmen.safetensors").is_file(), (
        "a live build's staging tree was reclaimed underneath it"
    )


def test_the_last_copy_of_the_previous_release_is_not_reclaimed(tmp_path: Path) -> None:
    """``.previous-`` holds the old bundle while the new one lands.

    A crash between the two renames leaves it holding the only copy of the
    last release, so it is swept only when a release is in place beside it
    and it is therefore a duplicate.
    """
    module = _builder()
    out = tmp_path / "release"
    out.parent.mkdir(parents=True, exist_ok=True)
    orphan = _staging_stub(out, _dead_pid(), suffix="previous")

    module._sweep_stale(out)
    assert orphan.is_dir(), "the only surviving copy of the last release was removed"

    out.mkdir()
    module._sweep_stale(out)
    assert not orphan.exists(), "a duplicate of a release that is in place was kept"


def test_a_successful_build_replaces_the_previous_one_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building over an existing bundle leaves the new one, whole."""
    src = _fake_sources(tmp_path)
    out = tmp_path / "release"
    out.mkdir()
    (out / "stale.txt").write_bytes(b"the previous release")

    assert _build_strict(monkeypatch, src, out) == 0
    assert not (out / "stale.txt").exists(), "the previous bundle was merged into the new one"
    assert (out / "release.json").is_file()
    strays = [p.name for p in out.parent.iterdir() if p.name.startswith(f".{out.name}.")]
    assert not strays, f"the commit left {strays} behind"


def test_release_json_records_the_profile_that_built_it(tmp_path: Path) -> None:
    """A lenient bundle and a release must be distinguishable by machine.

    Without this a consumer, or the CI job that checks a download, has no way
    to tell a development bundle from a releasable one except by counting
    files and hoping. The ``full-0.1`` half of this is
    ``test_release_json_records_verified_only_when_the_gate_ran``, which
    builds through the gate rather than around it.
    """
    src = _fake_sources(tmp_path)
    out = tmp_path / "release"
    result = _build(src, out, "--profile", "lenient", "--skip-verify")

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    manifest = json.loads((out / "release.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "lenient"
    assert manifest["verified"] is False, "a bundle that skipped the gate claims nothing"


def test_strict_refuses_skip_verify(tmp_path: Path) -> None:
    """``full-0.1 --skip-verify`` is a contradiction, and is refused.

    It assembled a bundle stamped ``"profile": "full-0.1"``, exited 0, and
    printed one line of warning that nothing read. The release checklist
    treats the profile string as proof that the bundle loads and speaks, so
    an unverified bundle was indistinguishable from a verified one. The help
    text said "do not use for a real release"; nothing enforced it.
    """
    src = _fake_sources(tmp_path)
    out = tmp_path / "release"
    result = _build(src, out, "--skip-verify")

    assert result.returncode != 0, result.stdout
    assert "--skip-verify" in result.stderr
    assert "full-0.1" in result.stderr
    assert "--profile lenient" in result.stderr
    _nothing_left(out)


def test_release_json_records_verified_only_when_the_gate_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``verified`` is the flag a release gate reads, and the gate sets it.

    The profile alone says what the bundle was asked to be. ``verified`` says
    the closing gate ran on the assembled bundle and passed, and it is written
    after the gate returns, so no path through the tool can stamp it early.
    """
    src = _fake_sources(tmp_path)
    out = tmp_path / "release"
    assert _build_strict(monkeypatch, src, out) == 0

    manifest = json.loads((out / "release.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "full-0.1"
    assert manifest["verified"] is True

    # And the flag is inside what SHA256SUMS covers, so a bundle cannot gain
    # it in transit without failing its own checksums.
    sums = (out / "SHA256SUMS").read_text(encoding="utf-8")
    assert "  release.json\n" in sums
    checked = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"], cwd=out, capture_output=True, text=True, check=False
    )
    assert checked.returncode == 0, f"{checked.stdout}\n{checked.stderr}"


def test_a_gate_that_touches_the_bundle_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``verified: true`` must describe the bytes the gate saw, or refuse.

    The gate imports the bundle's own code and runs it, which makes it the one
    step of the build that is not a copy under the builder's control. A
    ``verify()`` that mutates a file or drops one into the tree, say a buggy
    load path writing a cache beside the weights, must end in a
    refusal, not in a bundle whose manifests describe bytes nobody verified.
    This stub does both at once: flips the checkpoint's bytes and adds a file.
    """
    src = _fake_sources(tmp_path)
    out = tmp_path / "release"
    module = _builder()

    def hostile(staging: Path, **_kw: object) -> int:
        (staging / "loudr-1.safetensors").write_bytes(b"mutated checkpoint")
        (staging / "onnx" / "dropped.onnx").write_bytes(b"a passenger from the gate")
        return 0

    monkeypatch.setattr(module, "verify", hostile)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(BUILDER),
            "--checkpoint",
            str(src / "loudr-1.safetensors"),
            "--voice-encoder",
            str(src / "ve.safetensors"),
            "--voices",
            str(src / "voices"),
            "--out",
            str(out),
        ],
    )
    assert module.main() != 0, "a gate that rewrote the bundle still produced a release"
    _nothing_left(out)
    err = capsys.readouterr().err
    assert "loudr-1.safetensors" in err, "the mutated file is not named"
    assert "onnx/dropped.onnx" in err, "the added file is not named"


def test_verify_only_is_the_build_audit_run_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--verify-only`` holds an assembled bundle to the build's own audit.

    The same ``check_bundle`` the build runs after its gate, so a pre-upload
    check and the post-build check cannot drift: a bundle that just passed the
    build passes in place, and one tampered byte or one added file afterwards
    is a refusal that names the file.
    """
    src = _fake_sources(tmp_path)
    out = tmp_path / "release"
    assert _build_strict(monkeypatch, src, out) == 0

    module = _builder()
    monkeypatch.setattr(sys, "argv", [str(BUILDER), "--verify-only", str(out)])
    assert module.main() == 0, "a bundle that just passed the build fails in place"

    voice = out / "voices" / f"{_roster()[0]}.safetensors"
    original = voice.read_bytes()
    voice.write_bytes(b"tampered")
    assert module.main() != 0, "a tampered voice verified"

    voice.write_bytes(original)
    assert module.main() == 0
    (out / "notes.txt").write_bytes(b"left behind")
    assert module.main() != 0, "a file with no checksum line verified"


def test_verify_only_refuses_a_bundle_with_no_profile(tmp_path: Path) -> None:
    """The shape ``release-dir/`` has: entries, and no claim about what built
    them. A bundle that does not say what it is cannot be pre-upload checked
    into being the release."""
    module = _builder()
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "release.json").write_text('{"checkpoint": {}}', encoding="utf-8")
    (root / "SHA256SUMS").write_text("", encoding="utf-8")
    problems = module.check_bundle(root)
    assert any("profile" in p for p in problems)


def test_check_bundle_refuses_a_symlink(tmp_path: Path) -> None:
    """A bundle is bytes; a link is an address that can point outside them."""
    module = _builder()
    root = tmp_path / "bundle"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"elsewhere")
    (root / "release.json").write_text(
        '{"profile": "lenient", "verified": false}', encoding="utf-8"
    )
    (root / "linked.bin").symlink_to(outside)
    (root / "SHA256SUMS").write_text(
        f"{module.sha256(root / 'release.json')}  release.json\n", encoding="utf-8"
    )
    problems = module.check_bundle(root)
    assert any("symlink" in p and "linked.bin" in p for p in problems)


def test_the_checksum_count_is_the_file_count_less_the_one_written_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invariant RELEASING.md asks the operator to check.

    ``SHA256SUMS`` cannot contain its own digest, so it is the only file in
    the bundle with no checksum line. Everything else has one, ``release.json``
    included: the file that says whether a bundle is trustworthy is not the
    file nothing vouches for.
    """
    src = _fake_sources(tmp_path)
    out = tmp_path / "release"
    assert _build_strict(monkeypatch, src, out) == 0

    files = [p for p in out.rglob("*") if p.is_file()]
    lines = (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(files) - 1

    releasing = " ".join((REPO / "RELEASING.md").read_text(encoding="utf-8").split())
    assert "one fewer line than the bundle has files" in releasing, (
        "RELEASING.md does not state the real checksum invariant"
    )


def test_the_roster_is_twenty_voices_two_per_language(tmp_path: Path) -> None:
    """``roster_names`` refuses anything that is not the canonical roster.

    The builder reads the roster rather than hard-coding it, so a truncated
    or duplicated provenance file would otherwise shrink what ``full-0.1``
    requires without anyone editing the builder.
    """
    module = _builder()
    names = module.roster_names()
    assert len(names) == module.ROSTER_SIZE == 20
    assert len(set(names)) == len(names)

    entries = json.loads(module.ROSTER_PATH.read_text(encoding="utf-8"))
    module.ROSTER_PATH = tmp_path / "short.json"
    module.ROSTER_PATH.write_text(json.dumps(entries[:19]), encoding="utf-8")
    with pytest.raises(module.BuildRefusedError):
        module.roster_names()


# ------------------------------------------------- the two halves of the checkpoint


def test_strict_requires_the_enrollment_half(tmp_path: Path) -> None:
    """A release ships both files, whatever a given client downloads.

    The packed checkpoint carried the enrollment towers, so every caller who
    only ever loads a shipped voice paid 523 MB for weights they never open.
    Splitting it makes that download optional, and makes it possible to cut a
    release that cannot enroll a voice at all, in any of the five languages
    SUPPORTED.md declares, on any port. Which is why the absence is a refusal
    and not a note.
    """
    src = _fake_sources(tmp_path)
    (src / "loudr-1-enrollment.safetensors").unlink()
    out = tmp_path / "release"
    result = _build(src, out, "--skip-verify")

    assert result.returncode != 0, result.stdout
    assert "loudr-1-enrollment.safetensors" in result.stderr
    assert "tools/split_checkpoint.py" in result.stderr
    _nothing_left(out)


def test_strict_refuses_halves_whose_roles_are_swapped(tmp_path: Path) -> None:
    """``artifact_role`` is the only thing that tells the two files apart.

    Both are safetensors files of the same shape sitting in one directory
    under two names. Swapped, they copy, checksum, and satisfy every other
    check in the builder; what a consumer then gets is a synthesis loader
    pointed at the speech tokenizer.
    """
    src = _fake_sources(tmp_path)
    _split_pair(src, roles=("enrollment", "synthesis"))
    out = tmp_path / "release"
    result = _build(src, out, "--skip-verify")

    assert result.returncode != 0, result.stdout
    assert "artifact_role" in result.stderr
    _nothing_left(out)


def test_strict_refuses_a_split_that_is_not_disjoint(tmp_path: Path) -> None:
    """One tensor, one file. A tensor in both is two copies that can diverge."""
    src = _fake_sources(tmp_path)
    shared = "s3gen.tokenizer.enc.w"
    _split_pair(src, synthesis=[*SYNTHESIS_TENSORS, shared])
    out = tmp_path / "release"
    result = _build(src, out, "--skip-verify")

    assert result.returncode != 0, result.stdout
    assert "not disjoint" in result.stderr
    assert shared in result.stderr
    _nothing_left(out)


def test_strict_refuses_a_split_that_is_not_complete(tmp_path: Path) -> None:
    """A tensor in neither file is the failure a file listing cannot show.

    Drop the speaker encoder and the pair is still two correctly named,
    correctly rolled, perfectly disjoint files, and cannot enroll a voice.
    The digest of the source's full tensor-name list, carried in both
    manifests, is what catches it without the packed original being present.
    """
    src = _fake_sources(tmp_path)
    _split_pair(
        src,
        enrollment=["s3gen.tokenizer.enc.w"],
        source_names=[*SYNTHESIS_TENSORS, *ENROLLMENT_TENSORS],
    )
    out = tmp_path / "release"
    result = _build(src, out, "--skip-verify")

    assert result.returncode != 0, result.stdout
    assert "not complete" in result.stderr
    _nothing_left(out)


def test_strict_refuses_halves_of_two_different_checkpoints(tmp_path: Path) -> None:
    """Each half can be a valid half of a checkpoint that is not the other's.

    Two packing runs, split separately, then one file taken from each: the
    pair is disjoint, complete by name, correctly rolled, and describes a
    checkpoint that never existed. The source payload digest both manifests
    carry is the only thing that says so.
    """
    src = _fake_sources(tmp_path)
    _split_pair(src, enrollment_source_payload="b" * 64)
    out = tmp_path / "release"
    result = _build(src, out, "--skip-verify")

    assert result.returncode != 0, result.stdout
    assert "source_payload_sha256" in result.stderr
    _nothing_left(out)


def test_the_release_carries_both_halves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both files ship, both are checksummed, and the profile names both."""
    src = _fake_sources(tmp_path)
    out = tmp_path / "release"
    assert _build_strict(monkeypatch, src, out) == 0

    module = _builder()
    enrollment = module.ENROLLMENT_CHECKPOINT_NAME
    assert (out / enrollment).is_file()
    manifest = json.loads((out / "release.json").read_text(encoding="utf-8"))
    assert manifest["enrollment_checkpoint"]["path"] == enrollment
    assert enrollment in (out / "SHA256SUMS").read_text(encoding="utf-8")
    paths, _prefixes = module._allowlist(roster=_roster(), ships_onnx=True, ships_coreml=True)
    assert enrollment in paths


# ------------------------------------------------------- the splitting tool


SPLITTER = REPO / "tools" / "split_checkpoint.py"


def _packed_checkpoint(path: Path, extra: str | None = None) -> dict[str, object]:
    """A packed checkpoint of tiny tensors, one per group the split routes.

    Small enough to write in a test, shaped like the real one where the split
    can see it: the five top-level groups, and a ``dtype_map`` keyed by group
    so the per-half filtering has something to filter.
    """
    import torch
    from safetensors.torch import save_file

    names = [*SYNTHESIS_TENSORS, *ENROLLMENT_TENSORS] + ([extra] if extra else [])
    tensors = {
        name: torch.arange(4, dtype=torch.float32) + float(i) for i, name in enumerate(names)
    }
    manifest = {
        "format": "loudkit-checkpoint",
        "recipe_version": "loudkit-1",
        "dtype_map": {
            "t3": "float16",
            "s3gen.flow": "float32",
            "s3gen.tokenizer": "float32",
            "s3gen.speaker_encoder": "float32",
            "s3gen.mel2wav": "float32",
        },
    }
    # A real digest, by the tool's own recipe: the splitter refuses a source
    # whose manifest cannot vouch for its tensors, and a fixture that skipped
    # this would be testing a path no real checkpoint takes.
    sys.path.insert(0, str(REPO / "tools"))
    from split_checkpoint import payload_sha256

    manifest["tensor_payload_sha256"] = payload_sha256(tensors)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: tensors[k] for k in sorted(tensors)},
        str(path),
        metadata={"manifest": json.dumps(manifest, sort_keys=True)},
    )
    return {"names": names, "tensors": tensors}


def _split(src: Path, out_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SPLITTER),
            "--checkpoint",
            str(src),
            "--out-dir",
            str(out_dir),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )


@pytest.mark.requires_torch
def test_the_split_is_disjoint_complete_and_byte_identical(tmp_path: Path) -> None:
    """The whole contract of the tool, on a checkpoint small enough to check.

    Every tensor lands in exactly one half; every tensor that comes back out
    is bit-for-bit the tensor that went in; each half claims its role and
    carries only the ``dtype_map`` entries for the groups it holds; and the
    pair passes the builder's own split check, which is the thing that has to
    agree with this tool for a release to assemble at all.
    """
    from safetensors.torch import load_file

    packed = tmp_path / "packed" / "loudr-1.safetensors"
    source = _packed_checkpoint(packed)
    out_dir = tmp_path / "split"
    result = _split(packed, out_dir)
    assert result.returncode == 0, result.stdout + result.stderr

    module = _builder()
    halves = {
        "synthesis": out_dir / "loudr-1.safetensors",
        "enrollment": out_dir / module.ENROLLMENT_CHECKPOINT_NAME,
    }
    landed: dict[str, list[str]] = {}
    for role, half in halves.items():
        manifest, names = module._read_header(half)
        assert manifest["artifact_role"] == role
        assert manifest["recipe_version"] == "loudkit-1"
        landed[role] = names
        for group in manifest["dtype_map"]:
            assert any(n == group or n.startswith(group + ".") for n in names), group
        for name, tensor in load_file(str(half)).items():
            assert tensor.equal(source["tensors"][name]), name  # type: ignore[index]

    assert set(landed["synthesis"]) == set(SYNTHESIS_TENSORS)
    assert set(landed["enrollment"]) == set(ENROLLMENT_TENSORS)
    assert not set(landed["synthesis"]) & set(landed["enrollment"])
    assert set(landed["synthesis"]) | set(landed["enrollment"]) == set(source["names"])  # type: ignore[arg-type]

    sibling = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    synthesis_manifest, _names = module._read_header(halves["synthesis"])
    assert sibling == synthesis_manifest, (
        "the sibling manifest.json must describe the half it ships beside, or a "
        "loader checking the payload digest against it refuses a correct release"
    )

    assert (
        module._split_problems(
            {
                module.CHECKPOINT_NAME: halves["synthesis"],
                module.ENROLLMENT_CHECKPOINT_NAME: halves["enrollment"],
            }
        )
        == []
    )


@pytest.mark.requires_torch
def test_the_split_refuses_a_tensor_it_cannot_route(tmp_path: Path) -> None:
    """A group this tool has never seen is a decision, not a silent drop.

    Dropping it would produce two halves that pass every disjointness check
    and are missing a tower, and the failure would surface as a load error on
    somebody else's machine.
    """
    packed = tmp_path / "packed" / "loudr-1.safetensors"
    _packed_checkpoint(packed, extra="s3gen.prosody.head.w")
    result = _split(packed, tmp_path / "split")

    assert result.returncode != 0, result.stdout
    assert "s3gen.prosody.head.w" in result.stdout + result.stderr
    assert not (tmp_path / "split" / "loudr-1.safetensors").exists()


@pytest.mark.requires_torch
def test_the_split_will_not_write_over_the_packed_original(tmp_path: Path) -> None:
    """``--out-dir`` cannot be the checkpoint's own directory.

    The synthesis half ships under the same name as the packed file it comes
    out of, so the one mistake available here overwrites the input with a
    third of its tensors removed, and there is no second copy.
    """
    packed = tmp_path / "packed" / "loudr-1.safetensors"
    _packed_checkpoint(packed)
    before = packed.read_bytes()
    result = _split(packed, packed.parent)

    assert result.returncode != 0, result.stdout
    assert "--out-dir" in result.stderr
    assert packed.read_bytes() == before


class TestTheBytesAreCheckedNotJustTheHeaders:
    """The pair check reads headers. These read tensors.

    Without them a bit flipped after the split is copied into the bundle,
    receives a fresh and perfectly correct checksum line describing the flipped
    bytes, and passes every other gate: the file agrees with SHA256SUMS, the
    roles agree, the provenance agrees, the union is complete. The manifest's
    own payload digest is the only witness that predates the copy.
    """

    def _pair(self, tmp_path: Path) -> Path:
        out = tmp_path / "bundle"
        out.mkdir()
        _split_pair(out)
        return out

    def test_a_clean_pair_passes(self, tmp_path: Path) -> None:
        assert _builder()._payload_agreement(self._pair(tmp_path)) == []

    def test_a_flipped_tensor_byte_is_caught_and_named(self, tmp_path: Path) -> None:
        out = self._pair(tmp_path)
        module = _builder()
        target = out / module.ENROLLMENT_CHECKPOINT_NAME
        data = bytearray(target.read_bytes())
        data[-1] ^= 0x01
        target.write_bytes(bytes(data))
        problems = module._payload_agreement(out)
        assert problems, "a flipped tensor byte passed the payload check"
        assert module.ENROLLMENT_CHECKPOINT_NAME in problems[0]

    def test_an_unreadable_header_is_reported_not_raised(self, tmp_path: Path) -> None:
        """One broken file must not end the audit before it has read the rest."""
        out = self._pair(tmp_path)
        module = _builder()
        (out / module.CHECKPOINT_NAME).write_bytes(b"not a safetensors container")
        problems = module._payload_agreement(out)
        assert any("cannot read" in p for p in problems), problems

    def test_a_digest_that_is_not_a_sha256_is_refused(self, tmp_path: Path) -> None:
        out = tmp_path / "bundle"
        out.mkdir()
        module = _builder()
        _write_safetensors(
            out / module.CHECKPOINT_NAME,
            SYNTHESIS_TENSORS,
            {"artifact_role": "synthesis", "tensor_payload_sha256": 12345},
        )
        problems = module._payload_agreement(out)
        assert any("not a sha256" in p for p in problems), problems


@pytest.mark.requires_torch
class TestTheSplitterWontVouchForWhatItCannotCheck:
    """A source digest the tool never verified must not be stamped onto halves.

    `if recorded and ...` let a checkpoint whose manifest vouches for nothing
    straight through, and both halves then carried a source digest nobody had
    checked, which is worse than carrying none: it reads as provenance.
    """

    def _source(self, tmp_path: Path, digest: object) -> Path:
        src = tmp_path / "packed.safetensors"
        manifest: dict[str, object] = {
            "format": "loudkit-checkpoint",
            "recipe_version": "loudkit-1",
        }
        if digest is not None:
            manifest["tensor_payload_sha256"] = digest
        _write_safetensors(src, [*SYNTHESIS_TENSORS, *ENROLLMENT_TENSORS], manifest)
        return src

    @pytest.mark.parametrize(
        ("digest", "why"),
        [
            (None, "absent"),
            (12345, "a number"),
            ("deadbeef", "too short"),
            ("Z" * 64, "not hex"),
            ("A" * 64, "uppercase"),
        ],
    )
    def test_a_source_digest_that_is_not_a_sha256_is_refused(
        self, tmp_path: Path, digest: object, why: str
    ) -> None:
        run = _split(self._source(tmp_path, digest), tmp_path / "out")
        assert run.returncode != 0, f"{why} was accepted"
        assert "sha256" in run.stderr, run.stderr
        assert not (tmp_path / "out").exists(), "a refused split left a directory"
