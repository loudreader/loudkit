"""The import graph is architecture; this file makes it an assertion.

Every intra-package import edge in ``loudkit`` is listed in ALLOWED below,
derived from the tree as it stood when this test was written. A new edge —
including a lazy one inside a function — fails here until a human adds it to
the allowlist, which is the point: the dependency direction is the
architecture, and architecture that changes silently is not architecture.

The rules the allowlist encodes:

* Foundation modules (``errors``, ``rng``, ``timing``, ``contracts``,
  ``provenance``) import the narrowest possible set.
* The text-frontend modules (``frontend.numbers``, ``frontend.dates``,
  ``frontend.letters``, ``frontend.chunking``, ``frontend.polish``,
  ``frontend.text``, ``frontend.textconfig``, and ``postprocess``) never
  import the engine, a transport, or packaging — the funnel runs before and
  beneath all of those.
* ``engine`` never imports a transport or packaging.
* ``hub`` never imports the engine or a transport — it is reachable before
  any weights exist.
* Transports may not import each other, without exceptions.

Adding an edge: run this test, read the failure, decide whether the
dependency belongs, and either fix the code or extend the allowlist here —
in the same commit.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "python" / "loudkit"

# module (without the "loudkit." prefix; "" = the package __init__) ->
# allowed intra-package targets, each as "loudkit.x[.y]". Every set is
# exhaustive, the package __init__ included: there is no wildcard.
ALLOWED: dict[str, set[str]] = {
    # The public init: pinned explicitly, because it is the one module every
    # user imports first and an eager transport or cli edge here would drag
    # fastapi into every `import loudkit`. Lazy hub/backends edges included.
    "": {
        "loudkit.backends",
        "loudkit.backends.torch_backend",
        "loudkit.config",
        "loudkit.contracts",
        "loudkit.engine",
        "loudkit.errors",
        "loudkit.frontend.numbers",
        "loudkit.hub",
        "loudkit.models.timestretch",
        "loudkit.provenance",
        "loudkit.sampler",
        "loudkit.timing",
        "loudkit.voice",
    },
    # The registry package re-exports the backend constructors and, lazily,
    # the engine-facing pieces they register against.
    "backends": {"loudkit.backends", "loudkit.checkpoint", "loudkit.config", "loudkit.engine"},
    "checkpoint": set(),  # leaf: format parsing, no loudkit deps
    "errors": set(),
    "models": set(),  # namespace package of signal/network modules
    "models.resample": {"loudkit.config"},
    "models.timestretch": {"loudkit.config"},
    "postprocess": set(),
    "proto": {"loudkit"},
    "rng": {"loudkit.contracts"},
    "timing": {"loudkit.contracts"},
    "backends.coreml_backend": {
        "loudkit.backends",
        "loudkit.backends.torch_backend",
        "loudkit.checkpoint",
        "loudkit.config",
        "loudkit.contracts",
        "loudkit.engine",
        "loudkit.models.flow",
        "loudkit.models.noise",
        "loudkit.models.vocoder",
        "loudkit.voice",
    },
    "backends.onnx_backend": {
        "loudkit.backends",
        "loudkit.checkpoint",
        "loudkit.config",
        "loudkit.contracts",
        "loudkit.engine",
        "loudkit.models.noise",
        "loudkit.frontend.text",
        "loudkit.models.windowing",
        "loudkit.voice",
    },
    "backends.torch_backend": {
        "loudkit.backends",
        "loudkit.checkpoint",
        "loudkit.config",
        "loudkit.engine",
        "loudkit.models.enroll",
        "loudkit.models.flow",
        "loudkit.models.generator",
        "loudkit.frontend.text",
        "loudkit.models.vocoder",
    },
    "bench": {"loudkit"},
    "cli": {
        "loudkit",
        "loudkit.checkpoint",
        "loudkit.config",
        "loudkit.errors",
        "loudkit.hub",
        "loudkit.provenance",
        "loudkit.transports.grpc",
        "loudkit.transports.http",
        "loudkit.transports.mcp",
        "loudkit.voice",
    },
    "config": {
        "loudkit.frontend.chunking",
        "loudkit.postprocess",
        "loudkit.frontend.textconfig",
    },
    "contracts": {"loudkit.config", "loudkit.voice"},
    "engine": {
        "loudkit.backends",
        "loudkit.frontend.chunking",
        "loudkit.config",
        "loudkit.contracts",
        "loudkit.errors",
        "loudkit.frontend.polish",
        "loudkit.models.timestretch",
        "loudkit.models.windowing",
        "loudkit.postprocess",
        "loudkit.provenance",
        "loudkit.sampler",
        "loudkit.timing",
        "loudkit.voice",
    },
    "frontend": {"loudkit.frontend"},
    "frontend.chunking": {"loudkit.config"},
    "frontend.dates": {"loudkit.frontend.numbers"},
    "frontend.letters": set(),
    "frontend.numbers": {"loudkit.errors"},
    "frontend.polish": {
        "loudkit.frontend.dates",
        "loudkit.frontend.letters",
        "loudkit.frontend.numbers",
    },
    "frontend.text": {"loudkit.errors", "loudkit.frontend.numbers"},
    "frontend.textconfig": set(),
    "transports": set(),  # adapter package; init stays import-free
    "transports.grpc": {
        "loudkit",
        "loudkit.checkpoint",
        "loudkit.config",
        "loudkit.engine",
        "loudkit.errors",
        # MIN_SPEED/MAX_SPEED, so an out-of-range speed is refused at the
        # boundary rather than arriving as a bare ValueError after the caller
        # has waited for the engine. transports.http declares the same edge for
        # the same reason.
        "loudkit.models.timestretch",
        # CHARS_PER_TOKEN and estimate_tokens, to bound a unary reply from the
        # request before rendering it. gRPC is the only transport with a hard
        # per-message ceiling -- 4 MiB at every default client -- so it is the
        # only one that has to answer "how much audio is this text" before it
        # makes any. The alternative is a second copy of a measured constant
        # inside a transport, which is the drift this graph exists to catch.
        # The chunker owns that number and the estimate is its own.
        "loudkit.frontend.chunking",
        # speech_text, so the unary reply preflight measures the text the
        # engine will speak rather than the text the caller sent. The funnel
        # expands -- a thousand digits normalise to five thousand characters
        # of number words -- so a preflight over the raw text admits replies
        # several times the client's 4 MiB receive limit. Same direction as
        # the chunking edge above: the funnel runs before and beneath every
        # transport, and importing it is how the transport avoids owning a
        # second copy of it.
        "loudkit.frontend.polish",
        # is_repo_id/resolve_checkpoint, so `loudkit grpc --checkpoint org/repo`
        # resolves the snapshot (and its voices/) exactly as the HTTP server
        # does. hub is beneath every transport for the same reason it is
        # beneath the engine: it is reachable before any weights exist.
        "loudkit.hub",
        "loudkit.proto",
        "loudkit.synthesis",
    },
    "hub": {"loudkit.checkpoint", "loudkit.errors"},
    "transports.mcp": {
        "loudkit",
        "loudkit.errors",
        # Same edge and same reason as the other two transports: a repo id has
        # to resolve to the snapshot before the default voice directory is
        # computed beside it, or the server starts with no voices.
        "loudkit.hub",
        "loudkit.synthesis",
    },
    "models.enroll": {
        "loudkit.checkpoint",
        "loudkit.models.resample",
        "loudkit.voice",
    },
    "models.flow": {
        "loudkit.config",
        "loudkit.contracts",
        "loudkit.models.noise",
        "loudkit.models.windowing",
        "loudkit.voice",
    },
    "models.generator": {
        "loudkit.config",
        "loudkit.contracts",
        "loudkit.models.windowing",
        "loudkit.voice",
    },
    "models.noise": {"loudkit.rng"},
    "models.vocoder": {
        "loudkit.config",
        "loudkit.contracts",
        "loudkit.models.noise",
        "loudkit.models.windowing",
        "loudkit.voice",
    },
    "models.windowing": {"loudkit.config", "loudkit.errors", "loudkit.voice"},
    "profile": {"loudkit"},
    "proto.loudkit_pb2_grpc": {"loudkit.proto"},
    "provenance": {"loudkit.contracts", "loudkit.errors"},
    "sampler": {"loudkit.config", "loudkit.rng"},
    "synthesis": {
        "loudkit.config",
        "loudkit.engine",
        "loudkit.errors",
        "loudkit.provenance",
        "loudkit.voice",
    },
    "transports.http": {
        "loudkit",
        "loudkit.checkpoint",
        "loudkit.config",
        "loudkit.engine",
        "loudkit.errors",
        "loudkit.hub",
        "loudkit.models.timestretch",
        "loudkit.provenance",
        "loudkit.synthesis",
        "loudkit.voice",
    },
    "voice": {"loudkit.config"},
}


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _intra_package_deps(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    deps: set[str] = set()
    parts = module.split(".") if module else []
    # An ``__init__`` participates in its own package's level count: ``..``
    # from ``loudkit/backends/__init__.py`` is ``loudkit``, so the package
    # gets a sentinel leaf exactly like a plain module would.
    full_pkg = (
        ["loudkit"] + parts + ["<self>"] if path.name == "__init__.py" else ["loudkit"] + parts
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = full_pkg[: len(full_pkg) - node.level]
            target = ".".join(base + ([node.module] if node.module else []))
            deps.add(target)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "loudkit" or node.module.startswith("loudkit."):
                deps.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "loudkit" or alias.name.startswith("loudkit."):
                    deps.add(alias.name)
    return deps


def _iter_modules() -> list[tuple[Path, str]]:
    out = []
    for path in sorted(ROOT.rglob("*.py")):
        if "proto" in path.relative_to(ROOT).parts and path.name.startswith("loudkit_pb2"):
            continue  # generated; guarded by test_grpc's regeneration check
        out.append((path, _module_name(path)))
    return out


def test_every_module_is_in_the_allowlist() -> None:
    modules = {m for _, m in _iter_modules()}
    unlisted = sorted(modules - set(ALLOWED))
    assert not unlisted, (
        "modules with no declared dependency set — add them to ALLOWED with the "
        "edges they are allowed to have: " + ", ".join(unlisted)
    )


def test_no_undeclared_edges() -> None:
    problems = []
    for path, module in _iter_modules():
        if module not in ALLOWED:
            continue
        allowed = ALLOWED[module]
        for dep in sorted(_intra_package_deps(path, module)):
            if dep in allowed:
                continue
            problems.append(f"{module} -> {dep}")
    assert not problems, (
        "undeclared intra-package imports — fix the direction or declare the "
        "edge in ALLOWED (tests/test_import_graph.py docstring explains how to "
        "decide):\n  " + "\n  ".join(problems)
    )


def test_transports_never_import_each_other() -> None:
    """The one rule worth its own assertion.

    ``transports.http``, ``transports.mcp`` and ``transports.grpc`` are
    peers: three adapters over ``loudkit.synthesis``, none of them layered on
    another.
    """
    # A transport importing a peer (or the cli) fails here.
    peers = {"transports.http", "transports.mcp", "transports.grpc"}
    for path, module in _iter_modules():
        if module not in peers:
            continue
        peer = "loudkit." + module
        for dep in _intra_package_deps(path, module):
            if dep.startswith("loudkit.transports.") and dep != peer:
                pytest.fail(f"transport-to-transport import: {peer} -> {dep}")
            if dep == "loudkit.cli":
                pytest.fail(f"transport importing the cli: {peer} -> {dep}")
