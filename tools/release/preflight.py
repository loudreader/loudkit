"""What a release requires and this artefact set lacks, before a byte moves.

One function per model, each returning ``(what, where, how)`` triples; an
empty list is a set that may be assembled. ``loudr1`` proves the checkpoint
pair from the safetensors headers alone (roles, provenance, disjoint,
complete), so refusing a 4.3 GB set costs two short reads. ``turbo`` reads
its one checkpoint's manifest the same way.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from . import (
    BRANDING,
    CHECKPOINT_NAME,
    DOCUMENTS,
    ENROLL_COREML,
    ENROLL_ONNX,
    ENROLLMENT_CHECKPOINT_NAME,
    EXPORT_RECORD,
    PROFILES,
    REPO,
    SAMPLES,
    STRICT,
    SYNTHESIS_COREML,
    SYNTHESIS_ONNX,
    TURBO_CHECKPOINT_NAME,
    TURBO_DOCUMENTS,
    VOICE_ENCODER_NAME,
    sha256,
    synthesis_files,
)
from .verify import _split_problems

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from collections.abc import Sequence


def _pair_refusal(
    ckpt: Path, enrollment: Path | None = None, *, synthesis_name: str = CHECKPOINT_NAME
) -> list[tuple[str, str, str]]:
    """The enrollment half is beside the checkpoint, and the two are one split.

    Split out of :func:`loudr1` so the whole question, is the other file
    there and is it the other half of *this* file, is one call there and one
    place here.

    ``synthesis_name`` is the name the bundle will ship this half under, which
    is what ``split.roles.synthesis`` records and therefore what
    :func:`~.verify._split_problems` compares against. It is a parameter rather
    than ``ckpt.name`` because the packer writes whatever ``--out`` said: a
    turbo checkpoint arriving as ``model.safetensors`` still ships, and is
    still split, as ``loudr-1-turbo.safetensors``.
    """
    enrollment = enrollment or ckpt.parent / ENROLLMENT_CHECKPOINT_NAME
    if not enrollment.is_file():
        return [
            (
                f"enrollment checkpoint ({ENROLLMENT_CHECKPOINT_NAME})",
                str(enrollment),
                "split the packed checkpoint with tools/split_checkpoint.py",
            )
        ]
    if not ckpt.is_file():
        return []  # the missing synthesis half is already on the list
    return _split_problems({synthesis_name: ckpt, ENROLLMENT_CHECKPOINT_NAME: enrollment})


# -------------------------------------------------------------------- refusal


def loudr1(
    *,
    ckpt: Path,
    voice_encoder: Path | None,
    voices: dict[str, Path],
    roster: Sequence[str],
    onnx_dir: Path,
    coreml_dir: Path,
) -> list[tuple[str, str, str]]:
    """What ``full-0.1`` requires and this artefact set does not have.

    Runs before a single byte is copied. Everything is required, the three
    enrollment graphs included: a bundle without them assembles, verifies on
    the torch path and cannot enroll a voice on any port.

    Returns ``(what, where, how)`` triples, empty when the set is complete.
    """
    missing: list[tuple[str, str, str]] = []

    def need_file(label: str, path: Path, how: str) -> None:
        if not path.is_file():
            missing.append((label, str(path), how))

    def need_dir(label: str, path: Path, how: str) -> None:
        if not path.is_dir():
            missing.append((label, str(path), how))

    # The name is part of the artefact. A checkpoint under any other name
    # would ship, checksum and load on the build machine, and break every
    # guide and every port that names `loudr-1.safetensors`.
    if ckpt.name != CHECKPOINT_NAME:
        missing.append(
            (
                "checkpoint name",
                f"{ckpt.name} (at {ckpt})",
                f"a release ships {CHECKPOINT_NAME}; rename it or pack it again",
            )
        )
    need_file(
        "checkpoint",
        ckpt,
        "pack it from the research checkpoints, then split it with tools/split_checkpoint.py",
    )

    # The other half, and then whether the two are one checkpoint. A release
    # ships both files: which of them a given client downloads is a question
    # for the client, and a bundle carrying only the synthesis half cannot
    # enroll a voice on any port, in any of the five languages SUPPORTED.md
    # declares.
    missing += _pair_refusal(ckpt)

    need_file(
        "manifest.json", ckpt.parent / "manifest.json", "it belongs beside the checkpoint"
    )
    need_file(
        "tokenizer.json", ckpt.parent / "tokenizer.json", "it belongs beside the checkpoint"
    )

    if voice_encoder is None:
        missing.append(
            (
                VOICE_ENCODER_NAME,
                "(no --voice-encoder given)",
                f"pass --voice-encoder {VOICE_ENCODER_NAME}",
            )
        )
    else:
        need_file(
            VOICE_ENCODER_NAME, voice_encoder, f"pass --voice-encoder {VOICE_ENCODER_NAME}"
        )

    # The roster, by name. "At least one voice" accepted a bundle carrying a
    # single arbitrary profile, which loads, speaks and is not the release.
    for name in roster:
        if f"{name}.safetensors" not in voices:
            missing.append(
                (
                    f"voices/{name}.safetensors",
                    "(absent from the voice directory)",
                    "the roster is docs/voices/roster/provenance.json",
                )
            )
    wanted = {f"{name}.safetensors" for name in roster}
    for stranger in sorted(set(voices) - wanted):
        missing.append(
            (
                f"voices/{stranger}",
                str(voices[stranger]),
                "not on the roster; a release ships the roster and nothing else",
            )
        )

    for names, kind, tool in (
        (SYNTHESIS_ONNX, "onnx", "tools/export_onnx.py"),
        (ENROLL_ONNX, "onnx", "tools/export_enroll_onnx.py"),
    ):
        for name in names:
            need_file(f"{kind}/{name}", onnx_dir / name, f"export it: {tool}")
    for names, kind, tool in (
        (SYNTHESIS_COREML, "coreml", "tools/export_coreml.py"),
        (ENROLL_COREML, "coreml", "tools/export_enroll_coreml.py"),
    ):
        for name in names:
            need_dir(f"{kind}/{name}", coreml_dir / name, f"export it: {tool}")

    # Files in one folder are not files from one run. Both exporters take a
    # `--stages` list, so a run naming some stages leaves the rest as they
    # were, and the presence checks above are satisfied either way: a turbo
    # encoder and vocoder beside a stale K=2 estimator ships, loads and
    # integrates a one-step estimator twice, and a re-exported ONNX renderer
    # beside three stale T3 graphs speaks one checkpoint's tokens through
    # another's. Required here rather than merely warned about, which is what
    # the runtime does, because a release is exactly where re-exporting is
    # possible.
    need_file(
        f"coreml/{EXPORT_RECORD}",
        coreml_dir / EXPORT_RECORD,
        "re-export the whole set: tools/export_coreml.py --checkpoint ... (no --stages)",
    )
    need_file(
        f"onnx/{EXPORT_RECORD}",
        onnx_dir / EXPORT_RECORD,
        "re-export the whole set: tools/export_onnx.py --checkpoint ... (no --stages)",
    )

    for requirement in (*DOCUMENTS, BRANDING, *SAMPLES):
        source, name, *_metadata = requirement
        need_file(name, REPO / source, f"it is {source} in this repository")

    # A strict build is only strict if its closing gate can run, and half of
    # that gate is CoreML packages that open on Apple platforms and
    # nowhere else. Publishing from Linux would ship 589 MB of packages that
    # nothing ever opened, the exact defect this profile exists to prevent,
    # so the refusal is here rather than a skip at the end. `full-0.1` refuses
    # `--skip-verify`, so the gate always runs under this profile and the
    # platform is always its business.
    if sys.platform != "darwin":
        missing.append(
            (
                "Apple platform",
                f"sys.platform is {sys.platform!r}",
                "the CoreML packages only open on macOS; cut the release there",
            )
        )

    return missing


def _skip_verify_refusal(skip_verify: bool, profile: str) -> list[tuple[str, str, str]]:
    """``--skip-verify`` and a release profile are a contradiction, so it is refused.

    The flag turns off the load-and-speak gate, and the release checklist reads
    the profile string as proof that the gate passed, so an unverified bundle
    would be indistinguishable from a verified one by the one thing anybody
    checks. Under ``lenient`` the flag stays: that bundle is already marked
    unreleasable.
    """
    if not skip_verify:
        return []
    lenient = ", or build with --profile lenient" if profile == STRICT else ""
    return [
        (
            "--skip-verify",
            f"the load-and-speak gate does not run under {profile}",
            f"a {profile} bundle is the claim that it loads and speaks; drop the flag{lenient}",
        )
    ]


def _refuse(problems: Sequence[tuple[str, str, str]], profile: str) -> int:
    width = max(len(what) for what, _, _ in problems)
    print(
        f"\nREFUSING to build a {profile} release: {len(problems)} problem(s).\n",
        file=sys.stderr,
    )
    for what, where, how in problems:
        print(f"  {what:{width}s}  {where}", file=sys.stderr)
        print(f"  {'':{width}s}  -> {how}", file=sys.stderr)
    lenient = (
        ", or build a partial\nbundle with --profile lenient. "
        "A lenient bundle is not releasable"
        if profile in PROFILES
        else ""
    )
    print(f"\nNothing was assembled. Fix what is listed{lenient}.", file=sys.stderr)
    return 1


def turbo(
    *,
    ckpt: Path,
    tokenizer: Path,
    voice_encoder: Path,
    voices: dict[str, Path],
    roster: Sequence[str],
    enrollment: Path,
    onnx_dir: Path,
    coreml_dir: Path,
    samples: Path,
) -> list[tuple[str, str, str]]:
    """What ``turbo-0.1`` requires and this artefact set does not have.

    The order is what makes it cheap: the manifest and the tensor names come
    from the safetensors header, which is two short reads, so "this is not the
    turbo model" costs nothing to discover.
    """
    problems: list[tuple[str, str, str]] = []

    if ckpt.name != TURBO_CHECKPOINT_NAME:
        # Not fatal: the packer writes whatever `--out` said. The bundle
        # always ships the canonical name, and this says so out loud rather
        # than renaming silently. The pair check below is keyed by that same
        # canonical name, so the note holds: nothing two checks later refuses
        # the file for the name this line just said it would rename.
        print(f"  note: shipping {ckpt.name} as {TURBO_CHECKPOINT_NAME}")

    from loudkit.checkpoint import Checkpoint, decode_mode

    try:
        packed = Checkpoint.open(str(ckpt))
    except Exception as exc:  # every failure is the same refusal
        return [("checkpoint", str(ckpt), f"not a readable loudkit checkpoint: {exc}")]

    mode = decode_mode(packed.manifest)
    if mode != "fusion_mtp2":
        problems.append(
            (
                "checkpoint",
                str(ckpt),
                f"declares decode.mode {mode!r}; --model turbo ships the two-token "
                "model. Pack it with tools/pack_turbo.py, or build loudr-1 with "
                "--model loudr-1 (tools/build_release.py)",
            )
        )
    if packed.manifest.get("format_version") != 2:
        problems.append(
            (
                "checkpoint",
                str(ckpt),
                f"format_version {packed.manifest.get('format_version')!r}; a "
                "fusion checkpoint is version 2, and four engines gate on that "
                "number alone",
            )
        )
    problems += _pair_refusal(ckpt, enrollment, synthesis_name=TURBO_CHECKPOINT_NAME)

    # The tokenizer decides what the weights are asked to say, and the
    # manifest records which one these weights expect. A bundle whose
    # tokenizer does not match is one every `load()` will refuse after the
    # download, which is the worst place to find out.
    expected = packed.manifest.get("tokenizer_sha256")
    if not tokenizer.is_file():
        problems.append(("tokenizer", str(tokenizer), "not there; pass --tokenizer"))
    elif isinstance(expected, str) and (found := sha256(tokenizer)) != expected:
        problems.append(
            (
                "tokenizer",
                str(tokenizer),
                f"sha256 {found[:16]}… against the manifest's "
                f"{expected[:16]}…; a different tokenizer reads the same text as "
                "different ids and nothing downstream reports it",
            )
        )

    if not voice_encoder.is_file():
        problems.append(
            (
                "voice encoder",
                str(voice_encoder),
                "not there; pass --voice-encoder. Without it the release can "
                "speak but not clone",
            )
        )

    absent = [name for name in roster if f"{name}.safetensors" not in voices]
    if absent:
        problems.append(
            (
                "voices",
                f"{len(absent)} of {len(roster)} missing",
                "the roster is the release: " + ", ".join(absent),
            )
        )

    for source, _name, _key in TURBO_DOCUMENTS:
        if not (REPO / source).is_file():
            problems.append(("document", str(REPO / source), "not there"))
    if not (REPO / BRANDING[0]).is_file():
        problems.append(("branding", str(REPO / BRANDING[0]), "not there"))
    problems += _cloning_graph_refusals(enrollment, onnx_dir, coreml_dir, samples)
    return problems


def _cloning_graph_refusals(
    enrollment: Path, onnx_dir: Path, coreml_dir: Path, samples: Path
) -> list[tuple[str, str, str]]:
    problems: list[tuple[str, str, str]] = []
    if not enrollment.is_file():
        problems.append(
            ("enrollment", str(enrollment), "pass the shared enrollment half with --enrollment")
        )
    else:
        from safetensors import SafetensorError

        from loudkit.checkpoint import read_manifest

        try:
            role = read_manifest(str(enrollment)).get("artifact_role")
        except (ValueError, TypeError, OSError, SafetensorError) as exc:
            problems.append(("enrollment", str(enrollment), f"unreadable manifest: {exc}"))
        else:
            if role != "enrollment":
                problems.append(
                    ("enrollment", str(enrollment), "must declare artifact_role enrollment")
                )
    for kind, directory, suffix, extra in (
        ("onnx", onnx_dir, ".onnx", ENROLL_ONNX),
        ("coreml", coreml_dir, ".mlpackage", ENROLL_COREML),
    ):
        for name in (*synthesis_files("fusion_mtp2", suffix), *extra, EXPORT_RECORD):
            path = directory / name
            present = path.is_dir() if name.endswith(".mlpackage") else path.is_file()
            if not present:
                problems.append(
                    (f"{kind}/{name}", str(path), "export the complete matching graph set")
                )
    for _, name in SAMPLES:
        if not (samples / Path(name).name).is_file():
            problems.append(
                (name, str(samples), "render this model's voice samples before building")
            )
    if sys.platform != "darwin":
        problems.append(("Apple platform", sys.platform, "verify CoreML packages on macOS"))
    return problems
