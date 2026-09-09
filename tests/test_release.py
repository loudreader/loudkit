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
import importlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import pytest

from .conftest import tool

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parent.parent
BUILDER = REPO / "tools" / "build_release.py"


class _PackedSource(TypedDict):
    """What ``_packed_checkpoint`` wrote, for the split to be checked against."""

    names: list[str]
    tensors: dict[str, Any]


class _TurboPieces(TypedDict):
    """The artefact set ``preflight.turbo`` takes, spelled out so a fixture
    that drifts from its signature is caught here rather than by a keyword
    error inside the builder."""

    ckpt: Path
    tokenizer: Path
    voice_encoder: Path
    voices: dict[str, Path]
    roster: Sequence[str]
    enrollment: Path
    onnx_dir: Path
    coreml_dir: Path
    samples: Path


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


def _tomllib():
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

    assert module.audit._COREML_ENROLL.count(placement) == 1
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
    assert "python/loudkit/models/data/dsp/*.f32" not in excluded


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
    """``loudkit.transports.mcp`` takes render_bytes from ``synthesis``.

    Pinned as an import edge because the dependency list follows from it: while
    mcp imports only synthesis (transport-agnostic, fastapi-free), the ``[mcp]``
    extra must not carry fastapi. If mcp ever imports the HTTP transport again,
    that is the moment to reintroduce the inheritance — consciously, here.

    The half that watches for the bad edge could not fire. It collected module
    names through a filter that kept only ``"synthesis"``, then asserted
    ``"server" not in`` that set — true of every possible input. It also
    watched for a module named ``server``, which has not existed since the
    transports became peers; the edge worth refusing today is
    ``transports.http``.
    """
    import ast

    tree = ast.parse(
        (REPO / "python" / "loudkit" / "transports" / "mcp.py").read_text(encoding="utf-8")
    )
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "synthesis" in modules, "loudkit.transports.mcp must import loudkit.synthesis"
    forbidden = {"http", "transports.http", "loudkit.transports.http", "server"}
    assert not (modules & forbidden), (
        f"loudkit.transports.mcp imports {sorted(modules & forbidden)} — transports are "
        "peers; take the synthesis surface from loudkit.synthesis"
    )


def test_a_stray_inside_a_coreml_package_is_refused(tmp_path) -> None:
    """The audit allowlists a package by prefix, so anything inside one ships.

    An `.mlpackage` is a directory tree whose layout belongs to coremltools, so
    the profile names it by prefix rather than by leaf — and `copy_tree` copies
    every file under it and checksums each one. `RELEASING.md` mandates cutting
    on macOS, which makes a `.DS_Store` inside a package near-certain, and it
    would have been published as part of the release.
    """
    module = _builder()
    src = tmp_path / "flow_encoder.mlpackage"
    (src / "Data").mkdir(parents=True)
    (src / "Manifest.json").write_text("{}", encoding="utf-8")
    (src / "Data" / ".DS_Store").write_bytes(b"junk")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="dotfile inside an exported package"):
        module.assemble.copy_tree(src, out / "pkg", out, "coreml/flow_encoder.mlpackage")


def test_a_symlink_inside_a_coreml_package_is_refused(tmp_path) -> None:
    """`shutil.copy2` follows a link, so the bundle-level symlink check never
    sees one: the target's bytes arrive under the link's name."""
    module = _builder()
    outside = tmp_path / "elsewhere.bin"
    outside.write_bytes(b"not ours")
    src = tmp_path / "vocoder.mlpackage"
    src.mkdir()
    (src / "Manifest.json").write_text("{}", encoding="utf-8")
    (src / "sneaky").symlink_to(outside)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="is a symlink"):
        module.assemble.copy_tree(src, out / "pkg", out, "coreml/vocoder.mlpackage")


def test_the_cosine_gate_refuses_a_zero_vector_rather_than_passing_it() -> None:
    """`nan <= x` is `False`, and `False` is the branch where a gate succeeds.

    A graph returning zeros is the standard mis-export failure, so the one
    output most likely to be wrong produced `0/0` = NaN, and NaN passed both
    `if c <= 0.999` and `if c <= 0.9999`. The bundle was stamped
    `"verified": true` with `cosine nan` printed beside it.
    """
    import numpy as np

    module = _builder()
    good = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert module.audit._cos(good, good) == pytest.approx(1.0)

    for _label, a, b in (
        ("zero vector", np.zeros(3, dtype=np.float32), good),
        ("zero on the right", good, np.zeros(3, dtype=np.float32)),
        ("nan", np.array([float("nan"), 1.0, 1.0], dtype=np.float32), good),
        ("inf", np.array([float("inf"), 1.0, 1.0], dtype=np.float32), good),
    ):
        with pytest.raises(ValueError, match="cosine"):
            module.audit._cos(a, b)


def test_both_cosine_gates_read_the_passing_side() -> None:
    """The source, because the gates run only in a 4.6 GB strict cut.

    Every strict test in this file stubs `verify`, so neither gate is entered
    by anything in the repository — their first execution is the real release.
    A comparison written the other way round is exactly the shape that would
    survive that.
    """
    source = (REPO / "tools" / "release" / "audit.py").read_text(encoding="utf-8")
    assert "if not (c > 0.999):" in source
    assert "if not (c > 0.9999):" in source
    assert "if c <= 0.999" not in source
    assert "if c <= 0.9999" not in source


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
    # Read from pyproject rather than written out: the literal needs bumping
    # every release otherwise, and a test that fails only because the version
    # moved teaches people to edit tests during a release.
    version = (
        _tomllib()
        .loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
        .split(".dev")[0]
    )
    assert f"git tag v{version} public-main" in releasing
    assert f"git tag go/v{version} public-main" in releasing


def test_turbo_parity_revision_is_pinned_or_explicitly_blocked() -> None:
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    parity = workflow.split("\n  parity:\n", 1)[1].split("\n  packaging:\n", 1)[0]
    match = re.search(r'LOUDKIT_TURBO_HF_REVISION: "([^"]*)"', parity)
    assert match, "parity must declare the turbo revision"
    revision = match.group(1)
    assert not revision or re.fullmatch(r"[0-9a-f]{40}", revision)
    assert '--revision "$LOUDKIT_TURBO_HF_REVISION"' in parity
    if not revision:
        gate = parity.split("- name: Require published model revisions", 1)[1]
        gate = gate.split("- uses:", 1)[0]
        assert '"$LOUDKIT_TURBO_HF_REVISION" =~ ^[0-9a-f]{40}$' in gate
        assert "exit 1" in gate


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
    assert re.search(
        r"(?m)^\s+- uses: actions/setup-go@40f1582b2485089dde7abd97c1529aa768e1baff\s",
        parity,
    )
    assert re.search(r'(?m)^\s+go-version: "1\.25"$', parity)
    assert re.search(
        r"(?m)^\s+- uses: actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020\s",
        parity,
    )
    assert re.search(r'(?m)^\s+node-version: "22"$', parity)
    assert re.search(
        r"(?m)^\s+- uses: dtolnay/rust-toolchain@"
        r"4360b52568e2003a75bf9bc1d59f33a8e3fc893c\s",
        parity,
    )
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
    init = (REPO / "python" / "loudkit" / "_version.py").read_text(encoding="utf-8")
    dunder = re.search(r'^__version__ = "([^"]+)"', init, re.MULTILINE)
    assert dunder, "python/loudkit/_version.py no longer defines __version__"

    expected = _normalised_version(pyproject["project"]["version"])
    for label, raw in (
        ("js/package.json", package["version"]),
        ("rust/Cargo.toml", cargo["package"]["version"]),
        ("python/loudkit/_version.py __version__", dunder.group(1)),
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


def _builder():
    """The ``tools/release`` package, for the unit-level checks.

    The names live in the phase modules (``preflight``, ``assemble``,
    ``verify``, ``audit``); ``main`` is on the package. The tests below that
    drive the command line use a subprocess instead.
    """
    release = tool("release")
    importlib.import_module("release.__main__")  # `release.main` imports it lazily
    return release


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
    role_filenames: dict[str, str] | None = None,
) -> None:
    """Both halves, with the split provenance a real split would have written.

    ``source_names`` defaults to the union, which is what makes the pair
    complete; passing a longer list is how a test says "a tensor went missing
    between the split and here" without deleting a file.

    ``role_filenames`` is what the split recorded under ``split.roles``, which
    is a separate claim from ``synthesis_name``, the name the file is written
    under: the packer's ``--out`` decides the second and the model decides the
    first, and a turbo half is exactly where the two come apart.
    """
    module = _builder()
    whole = sorted(source_names if source_names is not None else [*synthesis, *enrollment])
    block = {
        "source_payload_sha256": "a" * 64,
        "source_tensor_names_sha256": hashlib.sha256("\n".join(whole).encode()).hexdigest(),
        "source_tensor_count": len(whole),
        "roles": role_filenames if role_filenames is not None else module.ROLE_FILENAMES,
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
    # The records that say each set came from one export. A release requires
    # both, because files in one folder are not files from one run and a
    # release is where re-exporting is possible.
    (coreml / module.EXPORT_RECORD).write_text(
        json.dumps(
            {
                "format": "loudkit-coreml-export",
                "packages": dict.fromkeys(module.SYNTHESIS_COREML, {}),
            }
        ),
        encoding="utf-8",
    )
    (onnx / module.EXPORT_RECORD).write_text(
        json.dumps(
            {
                "format": "loudkit-onnx-export",
                "graphs": dict.fromkeys(module.SYNTHESIS_ONNX, {}),
            }
        ),
        encoding="utf-8",
    )
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
    monkeypatch.setattr(module.audit, "verify", lambda _out, **_kw: 0)
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
    paths, prefixes = module.verify._allowlist(
        roster=_roster(), ships_onnx=True, ships_coreml=True
    )
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

    paths, prefixes = module.verify._allowlist(
        roster=["joe"], ships_onnx=False, ships_coreml=False
    )
    drift = module.verify._audit(root, paths, prefixes)
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

    module.assemble._sweep_stale(out)
    assert orphan.is_dir(), "the only surviving copy of the last release was removed"

    out.mkdir()
    module.assemble._sweep_stale(out)
    assert not orphan.exists(), "a duplicate of a release that is in place was kept"


def test_a_commit_does_not_reclaim_what_the_sweep_kept(tmp_path: Path) -> None:
    """The other end of the same rule, which ``_commit`` was not holding.

    ``_commit`` cleared ``.{out}.previous-{pid}`` before it had a bundle in
    place. Pids come round again after a reboot, so a new build can be handed
    the very name a crashed one left behind, and the tree the sweep kept
    because no release sat beside it was removed on the way in. The staging
    tree is left untouched by the refusal, so nothing publishable is lost
    either.
    """
    module = _builder()
    out = tmp_path / "release"
    out.parent.mkdir(parents=True, exist_ok=True)
    orphan = _staging_stub(out, os.getpid(), suffix="previous")
    staging = _staging_stub(out, os.getpid())

    with pytest.raises(FileExistsError, match="only surviving copy of the last release"):
        module.assemble._commit(staging, out)

    assert (orphan / "voices" / "carmen.safetensors").is_file(), (
        "the only surviving copy of the last release was removed by a new build"
    )
    assert staging.is_dir(), "the refusal took the staging tree with it"

    # With a release beside it the tree is a duplicate, and committing over it
    # is the ordinary path.
    out.mkdir()
    (out / "stale.txt").write_bytes(b"the previous release")
    module.assemble._commit(staging, out)
    assert (out / "voices" / "carmen.safetensors").is_file()
    assert not orphan.exists(), "a duplicate was kept after a successful commit"


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

    monkeypatch.setattr(module.audit, "verify", hostile)
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
    problems = module.verify.check_bundle(root)
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
    problems = module.verify.check_bundle(root)
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


def test_the_roster_is_28_voices_with_ten_english(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``roster_names`` refuses anything that is not the canonical roster.

    The builder reads the roster rather than hard-coding it, so a truncated
    or duplicated provenance file would otherwise shrink what ``full-0.1``
    requires without anyone editing the builder.
    """
    module = _builder()
    names = module.verify.roster_names()
    assert len(names) == module.ROSTER_SIZE == 28
    assert len(set(names)) == len(names)

    entries = json.loads(module.ROSTER_PATH.read_text(encoding="utf-8"))
    short = tmp_path / "short.json"
    short.write_text(json.dumps(entries[:27]), encoding="utf-8")
    monkeypatch.setattr(module.verify, "ROSTER_PATH", short)
    with pytest.raises(module.BuildRefusedError):
        module.verify.roster_names()


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
    paths, _prefixes = module.verify._allowlist(
        roster=_roster(), ships_onnx=True, ships_coreml=True
    )
    assert enrollment in paths


# ------------------------------------------------------- the splitting tool


SPLITTER = REPO / "tools" / "split_checkpoint.py"


def _packed_checkpoint(path: Path, extra: str | None = None) -> _PackedSource:
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
    manifest["tensor_payload_sha256"] = tool("split_checkpoint").payload_sha256(tensors)
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
        manifest, names = module.verify._read_header(half)
        assert manifest["artifact_role"] == role
        assert manifest["recipe_version"] == "loudkit-1"
        landed[role] = names
        for group in manifest["dtype_map"]:
            assert any(n == group or n.startswith(group + ".") for n in names), group
        for name, tensor in load_file(str(half)).items():
            assert tensor.equal(source["tensors"][name]), name

    assert set(landed["synthesis"]) == set(SYNTHESIS_TENSORS)
    assert set(landed["enrollment"]) == set(ENROLLMENT_TENSORS)
    assert not set(landed["synthesis"]) & set(landed["enrollment"])
    assert set(landed["synthesis"]) | set(landed["enrollment"]) == set(source["names"])

    sibling = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    synthesis_manifest, _names = module.verify._read_header(halves["synthesis"])
    assert sibling == synthesis_manifest, (
        "the sibling manifest.json must describe the half it ships beside, or a "
        "loader checking the payload digest against it refuses a correct release"
    )

    assert (
        module.verify._split_problems(
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
        assert _builder().verify._payload_agreement(self._pair(tmp_path)) == []

    def test_a_flipped_tensor_byte_is_caught_and_named(self, tmp_path: Path) -> None:
        out = self._pair(tmp_path)
        module = _builder()
        target = out / module.ENROLLMENT_CHECKPOINT_NAME
        data = bytearray(target.read_bytes())
        data[-1] ^= 0x01
        target.write_bytes(bytes(data))
        problems = module.verify._payload_agreement(out)
        assert problems, "a flipped tensor byte passed the payload check"
        assert module.ENROLLMENT_CHECKPOINT_NAME in problems[0]

    def test_an_unreadable_header_is_reported_not_raised(self, tmp_path: Path) -> None:
        """One broken file must not end the audit before it has read the rest."""
        out = self._pair(tmp_path)
        module = _builder()
        (out / module.CHECKPOINT_NAME).write_bytes(b"not a safetensors container")
        problems = module.verify._payload_agreement(out)
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
        problems = module.verify._payload_agreement(out)
        assert any("not a sha256" in p for p in problems), problems


class TestTheTurboPairIsKeyedByTheNameItShipsUnder:
    """The turbo preflight prints "shipping X as loudr-1-turbo.safetensors".

    It then keyed the pair check by the file's own name, and
    ``_provenance_problems`` refuses whenever ``split.roles.synthesis`` differs
    from that key. So a turbo checkpoint arriving under any other name was
    refused two checks later, after the note said it would not be. The note is
    the contract: the name the bundle ships under is the name the pair is
    judged by.
    """

    def _pair(self, tmp_path: Path, *, filename: str) -> tuple[Path, Path]:
        module = _builder()
        src = tmp_path / "src"
        src.mkdir()
        _split_pair(
            src,
            synthesis_name=filename,
            role_filenames={
                "synthesis": module.TURBO_CHECKPOINT_NAME,
                "enrollment": module.ENROLLMENT_CHECKPOINT_NAME,
            },
        )
        return src / filename, src / module.ENROLLMENT_CHECKPOINT_NAME

    def test_a_canonically_named_turbo_pair_passes(self, tmp_path: Path) -> None:
        module = _builder()
        ckpt, enrollment = self._pair(tmp_path, filename=module.TURBO_CHECKPOINT_NAME)
        assert (
            module.preflight._pair_refusal(
                ckpt, enrollment, synthesis_name=module.TURBO_CHECKPOINT_NAME
            )
            == []
        )

    def test_a_turbo_half_under_another_name_is_still_the_pair(self, tmp_path: Path) -> None:
        """What the "Not fatal" note promises, held to."""
        module = _builder()
        ckpt, enrollment = self._pair(tmp_path, filename="model.safetensors")
        assert (
            module.preflight._pair_refusal(
                ckpt, enrollment, synthesis_name=module.TURBO_CHECKPOINT_NAME
            )
            == []
        )

    def test_the_loudr1_default_is_the_canonical_name(self, tmp_path: Path) -> None:
        """The other side keeps its rule: `loudr1` refuses a non-canonical name
        outright, so keying by the canonical one changes no outcome there."""
        module = _builder()
        src = tmp_path / "src"
        src.mkdir()
        _split_pair(src)
        problems = module.preflight._pair_refusal(
            src / module.CHECKPOINT_NAME, src / module.ENROLLMENT_CHECKPOINT_NAME
        )
        assert problems == []


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


# ------------------------------------------------------ the second model


FUSION_TENSORS = ("t3.head2.weight", "t3.fuse.0.weight", "t3.fuse.2.weight")
"""What `tools/pack_turbo.py` adds and a one-token checkpoint cannot have."""


def _turbo_checkpoint(
    path: Path,
    *,
    mode: str = "fusion_mtp2",
    version: int = 2,
    tokenizer_sha256: str | None = None,
) -> Path:
    """A file that reads as the packed turbo checkpoint, in miniature.

    ``mode="single", version=1`` is the impostor: a loudr-1-shaped checkpoint
    sitting where the turbo one should be, which is what passing the wrong
    ``--checkpoint`` produces.
    """
    manifest: dict[str, object] = {
        "format": "loudkit-checkpoint",
        "format_version": version,
        "name": "loudkit-v0.1-turbo" if mode != "single" else "loudkit-v0.1",
        "recipe_version": "loudkit-1",
        "n_cfm_timesteps": 1,
    }
    names = list(SYNTHESIS_TENSORS)
    if mode != "single":
        manifest["decode"] = {"mode": mode}
        names += list(FUSION_TENSORS)
    if tokenizer_sha256 is not None:
        manifest["tokenizer_sha256"] = tokenizer_sha256
    whole = sorted([*names, *ENROLLMENT_TENSORS])
    split = {
        "source_payload_sha256": "a" * 64,
        "source_tensor_names_sha256": hashlib.sha256("\n".join(whole).encode()).hexdigest(),
        "source_tensor_count": len(whole),
        "roles": {"synthesis": path.name, "enrollment": "loudr-1-enrollment.safetensors"},
    }
    manifest.update(
        artifact_role="synthesis",
        split=split,
        tensor_payload_sha256=_fixture_payload_sha256(names),
    )
    _write_safetensors(path, names, manifest)
    _write_safetensors(
        path.parent / "loudr-1-enrollment.safetensors",
        ENROLLMENT_TENSORS,
        {
            **manifest,
            "artifact_role": "enrollment",
            "tensor_payload_sha256": _fixture_payload_sha256(ENROLLMENT_TENSORS),
        },
    )
    return path


class TestTheTurboBundleIsRefusedBeforeItIsBuilt:
    """The turbo builder ships 1.2 GB, so every refusal is worth having early.

    Two of them are the ones that would otherwise be discovered by whoever
    downloaded the result: a checkpoint that is not the turbo model at all
    (the way to build the wrong bundle is to pass the wrong ``--checkpoint``
    and have every other check pass), and a tokenizer that does not belong to
    these weights, which every ``load()`` refuses after the download.
    """

    def _pieces(self, tmp_path: Path) -> _TurboPieces:
        tokenizer = tmp_path / "tokenizer.json"
        tokenizer.write_text('{"model": "test"}', encoding="utf-8")
        # The manifest's claim about the tokenizer has to be true of the file
        # this set ships, or the honest case cannot be told from the broken
        # one: with no claim recorded the check is skipped entirely.
        ckpt = _turbo_checkpoint(
            tmp_path / "loudr-1-turbo.safetensors",
            tokenizer_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        )
        encoder = tmp_path / "ve.safetensors"
        encoder.write_bytes(b"ve")
        voices = {f"{name}.safetensors": tokenizer for name in _roster()}
        module = _builder()
        enrollment = tmp_path / "loudr-1-enrollment.safetensors"
        for kind, suffix, extra in (
            ("onnx", ".onnx", module.ENROLL_ONNX),
            ("coreml", ".mlpackage", module.ENROLL_COREML),
        ):
            directory = tmp_path / kind
            directory.mkdir()
            for name in module.synthesis_files("fusion_mtp2", suffix) + extra:
                target = directory / name
                if suffix == ".mlpackage":
                    target.mkdir()
                    target = target / "model.bin"
                target.write_bytes(b"graph")
            (directory / "export.json").write_text("{}")
        samples = tmp_path / "samples"
        samples.mkdir()
        for _, name in module.SAMPLES:
            (samples / Path(name).name).write_bytes(b"sample")
        return {
            "ckpt": ckpt,
            "tokenizer": tokenizer,
            "voice_encoder": encoder,
            "voices": voices,
            "roster": _roster(),
            "enrollment": enrollment,
            "onnx_dir": tmp_path / "onnx",
            "coreml_dir": tmp_path / "coreml",
            "samples": samples,
        }

    def test_a_complete_set_has_nothing_to_refuse(self, tmp_path: Path) -> None:
        p = self._pieces(tmp_path)
        builder = _builder().preflight
        assert builder.turbo(**p) == []

    @pytest.mark.parametrize(
        "relative",
        [
            "onnx/t3_pair_step.onnx",
            "onnx/t3_head2.onnx",
            "coreml/t3_head2.mlpackage",
            "onnx/voice_encoder.onnx",
            "samples/joe.opus",
            "loudr-1-enrollment.safetensors",
        ],
    )
    def test_missing_decode_or_cloning_assets_are_refused(
        self, tmp_path: Path, relative: str
    ) -> None:
        pieces = self._pieces(tmp_path)
        builder = _builder().preflight
        target = tmp_path / relative
        target.rename(target.with_name(target.name + ".missing"))
        problems = builder.turbo(**pieces)
        assert any(str(target) == where or relative == what for what, where, _ in problems), (
            problems
        )

    def test_unreadable_enrollment_is_a_structured_refusal(self, tmp_path: Path) -> None:
        pieces = self._pieces(tmp_path)
        builder = _builder().preflight
        enrollment = tmp_path / "loudr-1-enrollment.safetensors"
        enrollment.write_bytes(b"bad")
        problems = builder.turbo(**pieces)
        assert any(
            what == "enrollment" and "unreadable manifest" in how for what, _, how in problems
        )

    def test_a_one_token_checkpoint_is_not_the_turbo_model(self, tmp_path: Path) -> None:
        p = self._pieces(tmp_path)
        builder = _builder().preflight
        _turbo_checkpoint(p["ckpt"], mode="single", version=1)
        problems = builder.turbo(**p)
        assert any("decode.mode" in how for _what, _where, how in problems), problems
        assert any("build_release.py" in how for _what, _where, how in problems), (
            "say which builder ships a one-token model, rather than only that this one will not"
        )

    def test_a_foreign_tokenizer_is_refused(self, tmp_path: Path) -> None:
        """A different tokenizer reads the same text as different ids, and
        nothing downstream reports it. The checkpoint records which one these
        weights expect; this is a file that is not it."""
        p = self._pieces(tmp_path)
        builder = _builder().preflight
        _turbo_checkpoint(p["ckpt"], tokenizer_sha256="0" * 64)
        problems = builder.turbo(**p)
        assert [what for what, _where, _how in problems] == ["tokenizer"], problems

    def test_a_missing_voice_names_it(self, tmp_path: Path) -> None:
        p = self._pieces(tmp_path)
        builder = _builder().preflight
        voices = dict(p["voices"])
        dropped = f"{_roster()[3]}.safetensors"
        del voices[dropped]
        p["voices"] = voices
        problems = builder.turbo(**p)
        assert any(_roster()[3] in how for _what, _where, how in problems), problems

    def test_skip_verify_is_refused_before_a_byte_moves(self, tmp_path: Path) -> None:
        """``turbo-0.1`` has no lenient sibling, so the flag is a plain refusal,
        and it lands in the preflight with the others rather than after 1.2 GB
        has been copied."""
        p = self._pieces(tmp_path)
        voices = tmp_path / "voices"
        voices.mkdir()
        for name in _roster():
            (voices / f"{name}.safetensors").write_bytes(b"fake voice")
        out = tmp_path / "release"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "tools" / "build_turbo_release.py"),
                "--checkpoint",
                str(p["ckpt"]),
                "--tokenizer",
                str(p["tokenizer"]),
                "--voice-encoder",
                str(p["voice_encoder"]),
                "--voices",
                str(voices),
                "--out",
                str(out),
                "--skip-verify",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO,
        )
        assert result.returncode != 0, result.stdout
        assert "--skip-verify" in result.stderr
        assert "turbo-0.1" in result.stderr
        assert "--profile lenient" not in result.stderr, "turbo has no lenient profile"
        assert "assembling" not in result.stdout, "the refusal came after copying started"
        _nothing_left(out)


class TestTheShimsStandForOneModel:
    """`tools/build_release.py` is `--model loudr-1` and
    `tools/build_turbo_release.py` is `--model turbo`, by name. A `--model`
    the user adds is refused with one sentence before the parser runs: had
    the shim passed it through, the later flag would have won and the
    script's name would have lied.
    """

    @pytest.mark.parametrize(
        ("shim", "model", "other"),
        [
            ("build_release.py", "loudr-1", "turbo"),
            ("build_turbo_release.py", "turbo", "loudr-1"),
        ],
    )
    def test_a_user_model_is_refused(self, shim: str, model: str, other: str) -> None:
        for spelling in (["--model", other], [f"--model={other}"], ["--model", model]):
            result = subprocess.run(
                [sys.executable, str(REPO / "tools" / shim), *spelling],
                capture_output=True,
                text=True,
                check=False,
                cwd=REPO,
            )
            assert result.returncode != 0, spelling
            assert f"{shim} builds {model}" in result.stderr, result.stderr
            assert "--checkpoint" not in result.stderr, "the parser ran before the refusal"

    def test_without_a_model_the_shim_reaches_the_parser(self) -> None:
        result = subprocess.run(
            [sys.executable, str(REPO / "tools" / "build_turbo_release.py")],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO,
        )
        assert result.returncode != 0
        assert "--checkpoint is required" in result.stderr, result.stderr


class TestTheTurboProfileIsJudgedByTheSameChecker:
    """One definition of "assembled correctly", for two models.

    `release.verify.check_bundle` is what a `--model turbo` build runs before
    its rename and what `--verify-only` runs before an upload. A checker that
    did not know `turbo-0.1` would report the release it was asked to bless as
    nameless, and would not hold it to `verified: true` at all.
    """

    def test_the_profile_is_known_and_is_a_release(self) -> None:
        module = _builder()
        assert module.TURBO_PROFILE in module.KNOWN_PROFILES
        assert module.TURBO_PROFILE in module.RELEASE_PROFILES
        assert module.TURBO_PROFILE not in module.PROFILES, (
            "`--model loudr-1` cannot build it; turbo has its own synthesis half "
            "with its own decode-specific graphs"
        )

    def test_the_allowlist_requires_both_graph_families_and_shared_cloning(self) -> None:
        module = _builder()
        paths, prefixes = module.verify.turbo_allowlist(_roster())
        assert module.TURBO_CHECKPOINT_NAME in paths
        assert module.CHECKPOINT_NAME not in paths
        assert module.ENROLLMENT_CHECKPOINT_NAME in paths
        assert module.VOICE_ENCODER_NAME in paths
        assert "onnx/t3_pair_step.onnx" in paths
        assert "onnx/t3_head2.onnx" in paths
        assert "onnx/t3_step.onnx" not in paths
        assert "coreml/t3_pair_step.mlpackage/" in prefixes
        assert "coreml/voice_encoder.mlpackage/" in prefixes
        assert "samples/joe.opus" in paths
        assert len([p for p in paths if p.startswith("voices/")]) == len(_roster())

    def test_an_unverified_turbo_bundle_is_reported(self, tmp_path: Path) -> None:
        module = _builder()
        out = tmp_path / "bundle"
        out.mkdir()
        (out / "release.json").write_text(
            json.dumps({"profile": module.TURBO_PROFILE, "verified": False}), encoding="utf-8"
        )
        (out / "SHA256SUMS").write_text("", encoding="utf-8")
        problems = module.verify.check_bundle(out)
        assert any("verified: true" in p for p in problems), problems

    def test_the_gate_reads_the_weights_back(self, tmp_path: Path) -> None:
        """The turbo-only gate step: is this actually the two-token model?

        A loudr-1 checkpoint copied under the turbo name passes every other
        check in the builder, because every other check is about layout.
        """
        module = _builder()
        good = _turbo_checkpoint(tmp_path / "loudr-1-turbo.safetensors")
        assert module.audit._verify_turbo_identity(good) == 0
        wrong = _turbo_checkpoint(tmp_path / "impostor.safetensors", mode="single", version=1)
        assert module.audit._verify_turbo_identity(wrong) == 1

    def test_the_builder_and_the_resolver_name_the_same_files(self) -> None:
        """Both spell the canonical names out, so nothing enforces that they
        agree except this. A bundle built under a name `hub` does not resolve
        is a release nobody can load by its directory."""
        from loudkit import hub

        module = _builder()
        assert module.TURBO_CHECKPOINT_NAME == hub.TURBO_CHECKPOINT_NAME
        assert module.CHECKPOINT_NAME == hub.CHECKPOINT_NAME
        assert module.ENROLLMENT_CHECKPOINT_NAME == hub.ENROLLMENT_NAME
        assert module.VOICE_ENCODER_NAME == hub.VOICE_ENCODER_NAME
