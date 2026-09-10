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
  ``frontend.letters``, ``frontend.chunking``, ``frontend.speechtext``,
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
    # The version literal, a leaf so provenance can read it below the root.
    "_version": set(),
    "": {
        "loudkit._version",
        "loudkit.backends",
        "loudkit.backends.torch_backend",
        "loudkit.config",
        "loudkit.contracts",
        "loudkit.engine",
        "loudkit.errors",
        "loudkit.frontend.numbers",
        "loudkit.hub",
        # The two models submodules the root re-exports names from:
        # MIN_SPEED/MAX_SPEED, the bounds a UI slider needs, and the prompt cut
        # with its five tunables, which a caller meets through `loudkit.enroll`.
        "loudkit.models.timestretch",
        "loudkit.models.enrollment_audio",
        "loudkit.backends.graph_enroll",
        "loudkit.voice",
    },
    # The registry package re-exports the backend constructors and, lazily,
    # the engine-facing pieces they register against.
    "backends": {
        # The three registrations. `register_backend` is import-time, so the
        # package `__init__` imports each backend to populate the registry —
        # `from . import onnx_backend` and its two siblings. Declared rather
        # than invisible: the walker used to collapse `from . import X` to the
        # package name, which every allowlist already permits.
        "loudkit.backends.coreml_backend",
        "loudkit.backends.onnx_backend",
        "loudkit.backends.torch_backend",
        "loudkit.checkpoint",
        "loudkit.config",
        "loudkit.engine",
        "loudkit.postprocess",
        # EXPORT_RECORD: the export.json filename, written once there.
        "loudkit.release",
    },
    "checkpoint": set(),  # leaf: format parsing, no loudkit deps
    "errors": set(),
    "models": set(),  # namespace package of signal/network modules
    # Both are pure signal processing over numpy: they take a rate and a ratio
    # as arguments and import nothing from the package. Declaring an edge they
    # do not have would let them grow one without the gate noticing.
    "models.resample": set(),
    "models.timestretch": set(),
    "postprocess": set(),
    "proto": set(),  # namespace package for the generated stubs
    "rng": set(),  # leaf: Philox over numpy, no loudkit deps
    "timing": set(),  # leaf: arithmetic over sample counts
    "backends.coreml_backend": {
        # Lazy compatibility path for published renderer-only releases.
        "loudkit.backends.torch_backend",
        "loudkit.backends",
        # The renderers subclass the ONNX ones, so the framing, the noise and
        # the voice reach this module through that base class rather than here.
        "loudkit.backends.onnx_backend",
        "loudkit.checkpoint",
        "loudkit.config",
        "loudkit.contracts",
        "loudkit.engine",
        "loudkit.frontend.text",
        "loudkit.release",
    },
    "backends.onnx_backend": {
        "loudkit.backends",
        "loudkit.checkpoint",
        "loudkit.release",
        "loudkit.config",
        "loudkit.contracts",
        "loudkit.engine",
        # Same refusal as the torch loop, same class.
        "loudkit.errors",
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
    "cli": {
        "loudkit",
        # `--speed` is range-checked by argparse, before `load` can download.
        "loudkit.models.timestretch",
        "loudkit.checkpoint",
        "loudkit.config",
        "loudkit.errors",
        "loudkit.execution",
        "loudkit.frontend.speechtext",
        "loudkit.hub",
        "loudkit.provenance",
        "loudkit.transports.grpc",
        "loudkit.transports.http",
        "loudkit.transports.mcp",
        "loudkit.voice",
    },
    "config": {
        "loudkit.execution",
        "loudkit.frontend.chunking",
        # `from_manifest` delegates to the reader, lazily; the reader imports
        # the dataclasses it builds.
        "loudkit.manifest",
        "loudkit.postprocess",
        "loudkit.frontend.textconfig",
    },
    "execution": set(),
    "manifest": {"loudkit.config", "loudkit.postprocess"},
    "result": {
        "loudkit._version",
        "loudkit.contracts",
        "loudkit.postprocess",
        "loudkit.provenance",
        "loudkit.timing",
    },
    # One window: free functions over the engine's stages, beneath the engine.
    "window": {
        "loudkit.config",
        "loudkit.contracts",
        "loudkit.errors",
        "loudkit.frontend.chunking",
        "loudkit.frontend.speechtext",
        "loudkit.models.timestretch",
        "loudkit.models.windowing",
        "loudkit.postprocess",
        "loudkit.result",
        "loudkit.sampler",
        "loudkit.timing",
        "loudkit.voice",
    },
    # The pipeline drives an engine it is handed (typing only); it imports the
    # window functions and the result type.
    "stream": {
        "loudkit.engine",
        "loudkit.errors",
        "loudkit.result",
        "loudkit.timing",
        "loudkit.voice",
        "loudkit.window",
    },
    "contracts": {"loudkit.config", "loudkit.voice"},
    "engine": {
        "loudkit.backends",
        "loudkit.frontend.chunking",
        "loudkit.config",
        "loudkit.contracts",
        "loudkit.errors",
        "loudkit.frontend.speechtext",
        # `Engine.voice`/`Engine.voices`: the release an engine was loaded from
        # is a directory, and the resolver that reads one lives in `hub`. Lazy,
        # and one-way: `hub` still knows nothing about the engine.
        "loudkit.hub",
        "loudkit.models.timestretch",
        # The engine checks the rate it renders at against the geometry every
        # backend shares, and `UPSAMPLE_PER_FRAME` is where that geometry is
        # stated. Lazy, and one-way, as `models.timestretch` above.
        "loudkit.models.windowing",
        "loudkit.result",
        "loudkit.stream",
        "loudkit.timing",
        "loudkit.voice",
        "loudkit.window",
    },
    "frontend": set(),  # namespace package of funnel modules, imports nothing
    "frontend.chunking": {"loudkit.config"},
    # `grammar_languages`: the grammar file is located and parsed once for the
    # three passes that read different blocks out of it. Same reason as the
    # `speechtext` edge below -- `textconfig` owns where the funnel's data
    # lives because it is also what hashes it into the fingerprint.
    "frontend.dates": {"loudkit.frontend.numbers", "loudkit.frontend.textconfig"},
    "frontend.letters": {"loudkit.frontend.textconfig"},
    "frontend.numbers": {"loudkit.errors", "loudkit.frontend.textconfig"},
    "frontend.speechtext": {
        "loudkit.frontend.dates",
        "loudkit.frontend.letters",
        "loudkit.frontend.numbers",
        # `NUMERALS_PATH` and `RESPELL_PATH`: the numeral fold reads shared data
        # rather than the runtime's own Unicode tables, and `textconfig` owns
        # where that data lives because it is also what hashes it into the
        # fingerprint.
        "loudkit.frontend.textconfig",
    },
    "frontend.text": {"loudkit.errors", "loudkit.frontend.numbers"},
    "frontend.textconfig": set(),
    "transports": set(),  # adapter package; init stays import-free
    "transports.grpc": {
        "loudkit",
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
        "loudkit.frontend.speechtext",
        "loudkit.proto",
        "loudkit.synthesis",
        "loudkit.transports.limits",
        # `open_release`, so `loudkit serve --grpc --checkpoint org/repo`
        # resolves the snapshot (and its voices/) by the same call the other
        # two doors make. The hub edge lives there now, not here.
        "loudkit.transports.resolve",
        # `_STREAMABLE`, the formats a chunk can be delivered in on its own.
        # `proto/loudkit.proto` says this contract mirrors the HTTP one field
        # for field, so the streaming RPC has to refuse what
        # `/v1/synthesize/stream` refuses -- from the same set, not a second
        # copy of it that can drift.
        "loudkit.transports.schemas",
    },
    # The hub talks to the network and re-exports the two modules beneath it:
    # `release` (what a release is, read off disk) and `checksums` (the
    # SHA256SUMS rules). Neither of those knows the hub exists.
    "hub": {"loudkit.checkpoint", "loudkit.checksums", "loudkit.errors", "loudkit.release"},
    "release": {"loudkit.checkpoint", "loudkit.errors"},
    "checksums": {"loudkit.checkpoint", "loudkit.release"},
    # The two modules the three transports share: what every door refuses and
    # how long it may hold the engine, and the HTTP wire shapes. Neither is a
    # transport, so the peer rule below lets the peers import them.
    "transports.limits": {"loudkit.synthesis"},
    # The resolver the three doors share: a checkpoint reference in, the
    # release's checkpoint and the voice library beside it out. `hub` is a lazy
    # edge, so importing a transport still costs no `hub` extra. `checkpoint`
    # and `config` are the same, and sit here rather than in the two doors that
    # take `--first-chunk-tokens` because the override they build is one
    # function now.
    "transports.resolve": {
        "loudkit.checkpoint",
        "loudkit.config",
        "loudkit.hub",
        "loudkit.synthesis",
    },
    "transports.schemas": {
        "loudkit.errors",
        "loudkit.models.timestretch",
        "loudkit.synthesis",
        "loudkit.transports.limits",
    },
    "transports.mcp": {
        "loudkit",
        # Engine and VoiceProfile, so `build_server` and `_render` name the two
        # objects they take instead of typing them `Any`, as grpc and http
        # already do. `synthesis` imports both at module scope, so this declares
        # an edge the transport already had rather than adding one.
        "loudkit.engine",
        "loudkit.voice",
        "loudkit.errors",
        # MIN_SPEED/MAX_SPEED, the same edge and reason as grpc and http: an
        # out-of-range speed is refused at the door as bad_request, the shape
        # the tool description promises, not after the engine as a ValueError.
        "loudkit.models.timestretch",
        "loudkit.synthesis",
        "loudkit.transports.limits",
        # Same edge and same reason as the other two transports: a repo id has
        # to resolve to the snapshot before the default voice directory is
        # computed beside it, or the server starts with no voices.
        "loudkit.transports.resolve",
    },
    "models.enrollment_audio": set(),
    "backends.graph_enroll": {
        "loudkit.hub",
        "loudkit.config",
        # TOKEN_MEL_RATIO: the graph enroller cuts the prompt mel and its
        # tokens to the same ratio the torch enroller does.
        "loudkit.contracts",
        "loudkit.voice",
        "loudkit.models.resample",
        "loudkit.models.enrollment_audio",
        "loudkit.backends.onnx_backend",
        "loudkit.backends.coreml_backend",
    },
    "models.enroll": {
        "loudkit.models.enrollment_audio",
        "loudkit.checkpoint",
        # TOKEN_MEL_RATIO and UPSAMPLE_PER_FRAME: the prompt mel and its tokens
        # are cut to the same ratio the renderer frames with, and the mel hop is
        # the renderer's samples-per-frame. Restating either here is how the
        # enroller and the renderer come to disagree about one geometry.
        "loudkit.contracts",
        "loudkit.models.resample",
        "loudkit.models.windowing",
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
        # The decode loop refuses a cancel where it sees one, in the shared
        # error class the transports map by code.
        "loudkit.errors",
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
    "models.windowing": {
        "loudkit.config",
        "loudkit.contracts",
        "loudkit.errors",
        "loudkit.voice",
    },
    "proto.loudkit_pb2_grpc": {"loudkit.proto"},
    # _version: the manifest defaults claim the running version, and the leaf
    # is what a foundation module may read to get it.
    "provenance": {"loudkit._version", "loudkit.contracts", "loudkit.errors"},
    "sampler": {"loudkit.config", "loudkit.rng"},
    "synthesis": {
        "loudkit._version",
        "loudkit.engine",
        "loudkit.errors",
        # carry_pair_aligned: the continuation a transport hands a client has to
        # be the tail the engine itself would carry into the next window, and
        # that slice is a window rule, not a transport constant. Declared here
        # so the two streams fold through this module rather than each reaching
        # past it into window for the same rule. window is below engine, which
        # this module already imports.
        "loudkit.window",
        "loudkit.provenance",
        # release_confinement: the voice library confines a voice file to the
        # release it belongs to, and for a Hub-cached release that boundary is
        # the cache entry, which only the release module knows how to find.
        # release imports checkpoint and errors, nothing above it.
        "loudkit.release",
        "loudkit.voice",
        "loudkit.result",
    },
    "transports.http": {
        "loudkit",
        "loudkit.engine",
        "loudkit.errors",
        "loudkit.models.timestretch",
        "loudkit.synthesis",
        "loudkit.transports.limits",
        "loudkit.transports.resolve",
        "loudkit.transports.schemas",
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
            if node.module:
                deps.add(".".join(base + [node.module]))
                continue
            # `from . import x` — the module is in the *names*, not in
            # `node.module`, and this branch used to collapse the edge to the
            # package: `from .. import cli` inside a transport recorded
            # `loudkit`, which every allowlist already permits. The docstring
            # below and `docs/design/ARCHITECTURE.md` both claim a transport
            # importing the cli fails this suite. It did not — verified by
            # inserting that import and watching three tests pass.
            #
            # Each name is a module edge only if it names a module. `from .
            # import __version__` and `from .. import load` import values, and
            # recording those as edges would invent dependencies on modules
            # that do not exist.
            for alias in node.names:
                candidate = base + [alias.name]
                rel = Path(*candidate[1:])
                if (ROOT / rel).is_dir() or (ROOT / rel.with_suffix(".py")).is_file():
                    deps.add(".".join(candidate))
                else:
                    deps.add(".".join(base))
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


def test_no_declared_edge_is_unused() -> None:
    """The other direction, which nothing checked.

    `test_no_undeclared_edges` only asks that every real import be declared, so
    a declaration for an import that does not exist passed unnoticed and left
    the door open for exactly the edge it named. Nine modules carried one,
    including two leaves declared to read `loudkit.config` while importing
    nothing but numpy, and three self-edges that could never be violated.
    """
    stale = []
    for path, module in _iter_modules():
        if module not in ALLOWED:
            continue
        for dep in sorted(ALLOWED[module] - _intra_package_deps(path, module)):
            stale.append(f"{module} -> {dep}")
    assert not stale, (
        "declared but unused intra-package imports — delete the entry, which is "
        "what stops the module from gaining the edge silently:\n  " + "\n  ".join(stale)
    )


def test_transports_never_import_each_other() -> None:
    """The one rule worth its own assertion.

    ``transports.http``, ``transports.mcp`` and ``transports.grpc`` are
    peers: three adapters over ``loudkit.synthesis``, none of them layered on
    another.
    """
    # A transport importing a peer (or the cli) fails here. `limits`,
    # `schemas` and `resolve` are not peers: they are what the peers share.
    peers = {"transports.http", "transports.mcp", "transports.grpc"}
    shared = {
        "loudkit.transports.limits",
        "loudkit.transports.resolve",
        "loudkit.transports.schemas",
    }
    for path, module in _iter_modules():
        if module not in peers:
            continue
        peer = "loudkit." + module
        for dep in _intra_package_deps(path, module):
            if dep.startswith("loudkit.transports.") and dep != peer and dep not in shared:
                pytest.fail(f"transport-to-transport import: {peer} -> {dep}")
            if dep == "loudkit.cli":
                pytest.fail(f"transport importing the cli: {peer} -> {dep}")


def test_limits_declares_every_name_its_peers_import() -> None:
    """``limits.__all__`` is the seam, so it lists what crosses it.

    The underscore names are private to the package, not to the module: four
    transports read caps, the guard and the loopback test out of it. A list
    that omitted them would read as peers reaching past the seam.
    """
    from loudkit.transports import limits

    declared = set(limits.__all__)
    imported: set[str] = set()
    for path, module in _iter_modules():
        if not module.startswith("transports.") or module == "transports.limits":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in (
                "limits",
                "loudkit.transports.limits",
            ):
                imported.update(alias.name for alias in node.names)
    assert imported, "no transport imports from limits; the walk is wrong"
    assert imported <= declared, sorted(imported - declared)
