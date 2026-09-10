"""Assemble the directory that gets published, or nothing at all.

Two models, one release shape: ``loudr-1`` (profile ``full-0.1``) and
``loudr-1-turbo`` (``turbo-0.1``). A build is transactional (staged beside
``--out`` and renamed into place after every check), exact (a release profile
holds what its allowlist names and nothing else) and named (``release.json``
records the profile and whether the load-and-speak gate passed, and is itself
checksummed). ``lenient`` assembles whatever is present, for development, and
is not releasable.

The phases are the modules: ``preflight`` refuses before a byte moves,
``assemble`` copies and writes the manifests, ``verify`` judges a bundle from
disk alone, ``audit`` loads what shipped and speaks, ``__main__`` is the
command line. ``tools/build_release.py`` and ``tools/build_turbo_release.py``
call it. This module holds the names every phase agrees on.

Nothing here uploads. Publishing is a separate act; see RELEASING.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "python"))

from loudkit.checkpoint import file_sha256 as sha256  # noqa: E402
from loudkit.checkpoint import payload_sha256, read_header  # noqa: E402

# The names a release ships under are the runtime's, not this tool's copy of
# them: `loudkit.release` is what every port and every downloader reads, so a
# second spelling here is a second source of truth for the same string.
from loudkit.release import (  # noqa: E402
    CHECKPOINT_NAME,
    ENROLLMENT_ROLE,
    SYNTHESIS_ROLE,
    TURBO_CHECKPOINT_NAME,
    VOICE_ENCODER_NAME,
)
from loudkit.release import ENROLLMENT_NAME as ENROLLMENT_CHECKPOINT_NAME  # noqa: E402

# The whole public surface, so the list is a statement rather than a sample:
# every name the four phase modules, the two shim scripts and
# `tests/test_release.py` take from this package.
__all__ = [
    "ARTIFACT_ROLES",
    "BRANDING",
    "CHECKPOINT_NAME",
    "DOCUMENTS",
    "ENROLLMENT_CHECKPOINT_NAME",
    "ENROLLMENT_ROLE",
    "ENROLL_COREML",
    "ENROLL_ONNX",
    "EXPORT_RECORD",
    "KNOWN_PROFILES",
    "MODELS",
    "PROFILES",
    "RELEASE_PROFILES",
    "REPO",
    "ROLE_FILENAMES",
    "ROSTER_PATH",
    "ROSTER_PER_LANGUAGE",
    "ROSTER_SIZE",
    "SAMPLES",
    "STRICT",
    "SYNTHESIS_COREML",
    "SYNTHESIS_ONNX",
    "SYNTHESIS_ROLE",
    "TURBO_CHECKPOINT_NAME",
    "TURBO_DOCUMENTS",
    "TURBO_MODEL_CARD",
    "TURBO_PROFILE",
    "UNCHECKSUMMED",
    "VOICE_ENCODER_NAME",
    "BuildRefusedError",
    "main",
    "payload_sha256",
    "read_header",
    "sha256",
    "shim",
    "synthesis_files",
]

MODELS = ("loudr-1", "turbo")

PROFILES = ("full-0.1", "lenient")
STRICT = "full-0.1"

TURBO_PROFILE = "turbo-0.1"
"""What ``--model turbo`` stamps on a ``loudr-1-turbo`` bundle.

Not in :data:`PROFILES`, which are loudr-1's choices: turbo has a synthesis
checkpoint, a separate enrollment checkpoint and exported graphs. Both are in
:data:`KNOWN_PROFILES`, because ``verify.check_bundle`` judges both.
"""

KNOWN_PROFILES = (*PROFILES, TURBO_PROFILE)
RELEASE_PROFILES = frozenset({STRICT, TURBO_PROFILE})
"""Profiles that claim to be a release, and so must record ``verified: true``.
``lenient`` is the development bundle and claims nothing."""

# CHECKPOINT_NAME, TURBO_CHECKPOINT_NAME and ENROLLMENT_CHECKPOINT_NAME are
# imported above from `loudkit.release`. The last is spelled ENROLLMENT_NAME
# there; the longer name is kept here because twenty call sites in this package
# read better with it, and the alias makes the two provably the same string.
#
# What the second half is: `tools/split_checkpoint.py` writes the pair, the
# synthesis file carrying t3, the flow and the vocoder, and the enrollment file
# carrying the two towers synthesis never opens (the S3 speech tokenizer and
# the speaker encoder). A full release ships both. Whether a *download* takes
# the second one is a question for the client (enrolling from audio needs it,
# loading a shipped voice does not), but a release that does not carry it
# cannot answer the question at all.

# The role each name claims in its own manifest. Checked rather than assumed:
# the two files are interchangeable by name alone, and a bundle whose halves
# were swapped copies, checksums and passes every other check in this tool.
ARTIFACT_ROLES = {
    CHECKPOINT_NAME: SYNTHESIS_ROLE,
    ENROLLMENT_CHECKPOINT_NAME: ENROLLMENT_ROLE,
}
# The same table the other way round, which is the shape `split.roles` carries
# in both manifests: role -> the filename a release ships it under.
ROLE_FILENAMES = {role: name for name, role in ARTIFACT_ROLES.items()}

# The roster is data, not code: `docs/voices/roster/provenance.json` is the
# source of truth for which voices a release carries, and it also
# carries their licences and their provenance. Reading it here means the
# builder and the model card cannot disagree about what the release is.
ROSTER_PATH = REPO / "docs" / "voices" / "roster" / "provenance.json"
ROSTER_SIZE = 28
ROSTER_PER_LANGUAGE = {
    "en": 10,
    "es": 2,
    "fr": 2,
    "de": 2,
    "it": 2,
    "pl": 2,
    "pt": 2,
    "nl": 2,
    "sv": 2,
    "da": 2,
}


def synthesis_files(decode: str, suffix: str) -> tuple[str, ...]:
    """The complete graph family declared by a checkpoint's decoder."""
    if decode not in {"single", "fusion_mtp2"}:
        raise ValueError(f"unsupported decode mode {decode!r}")
    step = ("t3_step",) if decode == "single" else ("t3_pair_step", "t3_head2")
    return tuple(
        name + suffix
        for name in (
            "t3_cond",
            "t3_prefill",
            *step,
            "flow_encoder",
            "flow_estimator",
            "vocoder",
        )
    )


# Every graph and package a release ships, split by the tool that writes them
# so the refusal can name it. The enrollment triple is the half that went
# missing: SUPPORTED.md declares voice enrollment in five languages, and a
# bundle without these three cannot enroll in any of them, on any port.
SYNTHESIS_ONNX = synthesis_files("single", ".onnx")
ENROLL_ONNX = ("s3_tokenizer.onnx", "camp.onnx", "voice_encoder.onnx")
SYNTHESIS_COREML = synthesis_files("single", ".mlpackage")
ENROLL_COREML = ("s3_tokenizer.mlpackage", "camp.mlpackage", "voice_encoder.mlpackage")

EXPORT_RECORD = "export.json"
"""What each exporter writes beside its graphs, naming the checkpoint they came
from. Both backends read it back and refuse a set whose members disagree, so it
ships with them: a record left in the export directory checks nothing for the
person who downloaded the bundle."""

# (source in the repo, name in the release, key in release.json). Each is its
# own key rather than one list, so every document carries a checksum line the
# way the checkpoint does.
DOCUMENTS = (
    ("docs/MODEL_CARD.md", "README.md", "readme"),
    ("LICENSE", "LICENSE", "license"),
    ("NOTICE", "NOTICE", "notice"),
    ("RESPONSIBLE_USE.md", "RESPONSIBLE_USE.md", "responsible_use"),
)

# (source in the repo, name in the release, key in release.json). The model
# card resolves this from the model repository itself, so the image must ship
# and be checksummed with the release rather than depend on another site.
BRANDING = ("assets/logo-wordmark.png", "logo.png", "logo")

# The model card's first job is to let a visitor hear the model. Keep those
# players self-contained on Hugging Face instead of depending on GitHub Pages
# or raw.githubusercontent.com: the release carries the exact bytes it embeds,
# and both manifests vouch for them like every graph and weight.
SAMPLES = (
    ("docs/voices/roster/audio/joe.opus", "samples/joe.opus"),
    ("docs/voices/roster/audio/kathleen.opus", "samples/kathleen.opus"),
)

# `SHA256SUMS` cannot contain its own digest, so it is the one file with no
# checksum line. This is the invariant behind the count RELEASING.md asks
# for: a bundle of N files carries exactly N-1 checksum lines.
UNCHECKSUMMED = frozenset({"SHA256SUMS"})

# The same four documents a loudr-1 release carries, under the same four
# names, with one substitution: the model card is turbo's own. The names have
# to match `DOCUMENTS`, because `verify.turbo_allowlist` is written from that
# table. Deriving the entries below preserves names and keys by construction;
# only the model card's source changes.
TURBO_MODEL_CARD = "docs/MODEL_CARD-turbo.md"
TURBO_DOCUMENTS = tuple(
    (TURBO_MODEL_CARD if key == "readme" else source, name, key)
    for source, name, key in DOCUMENTS
)


class BuildRefusedError(Exception):
    """A refusal carrying ``(what, where, how)`` triples to print."""

    def __init__(self, problems: Sequence[tuple[str, str, str]]) -> None:
        super().__init__(f"{len(problems)} problem(s)")
        self.problems = list(problems)


def main(argv: Sequence[str] | None = None) -> int:
    """The command line; see ``__main__``. Imported there and not here, so
    that reading these names does not import the whole tool."""
    from .__main__ import main as run

    return run(argv)


def shim(name: str, model: str, argv: Sequence[str]) -> int:
    """:func:`main` for a script that stands for one model. A ``--model`` in
    ``argv`` is refused: the script's name already chose, and a second
    choice would either be ignored or win over it."""
    if any(arg == "--model" or arg.startswith("--model=") for arg in argv):
        print(
            f"{name} builds {model}; to choose a model, run "
            "python -m tools.release --model ...",
            file=sys.stderr,
        )
        return 2
    return main(["--model", model, *argv])
