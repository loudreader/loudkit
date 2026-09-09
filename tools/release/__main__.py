"""The command line: ``--model loudr-1|turbo``, ``--out``, ``--verify-only``.

    python -m tools.release --model loudr-1 --checkpoint dist/split/loudr-1.safetensors \\
        --voice-encoder assets/ve.safetensors --voices assets/voices --out dist/loudr-1
    python -m tools.release --model turbo --checkpoint assets/loudr-1-turbo.safetensors \\
        --out dist/loudr-1-turbo
    python -m tools.release --verify-only dist/loudr-1

One build for both models: preflight, assemble into staging, both manifests,
the allowlist audit, the load-and-speak gate, the manifests again with
``verified: true``, the closing audit, the rename into place. The models
differ only in what the preflight demands and what the assembly copies, so
each is a :class:`Plan` and the build is one function.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import (
    ENROLLMENT_CHECKPOINT_NAME,
    MODELS,
    PROFILES,
    REPO,
    STRICT,
    TURBO_PROFILE,
    UNCHECKSUMMED,
    VOICE_ENCODER_NAME,
    BuildRefusedError,
    assemble,
    audit,
    preflight,
    verify,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class Plan:
    """One model's build, decided before a byte moves."""

    profile: str
    out: Path
    assemble: Callable[[Path], dict[str, Any]]
    """Copies every piece into the staging directory; returns the ``release.json`` body."""
    allowlist: tuple[set[str], tuple[str, ...]] | None
    """Exactly what the bundle must hold; ``None`` under ``lenient``."""
    skip_verify: bool


def _loudr1_plan(args: argparse.Namespace) -> Plan:
    strict = args.profile == STRICT
    ckpt = args.checkpoint.resolve()
    voice_encoder = args.voice_encoder.resolve() if args.voice_encoder else None
    voice_dir = (args.voices or ckpt.parent / "voices").resolve()
    onnx_dir = (args.onnx or ckpt.parent / "onnx").resolve()
    coreml_dir = (args.coreml or ckpt.parent / "coreml").resolve()

    # A collision breaks `SHA256SUMS` under either profile, so it is checked
    # under either profile.
    voices = verify._voice_sources(voice_dir)
    roster = verify.roster_names() if strict else ()
    if strict:
        missing = preflight._skip_verify_refusal(args.skip_verify, STRICT)
        missing += preflight.loudr1(
            ckpt=ckpt,
            voice_encoder=voice_encoder,
            voices=voices,
            roster=roster,
            onnx_dir=onnx_dir,
            coreml_dir=coreml_dir,
        )
        if missing:
            raise BuildRefusedError(missing)

    def build(staging: Path) -> dict[str, Any]:
        return assemble.loudr1(
            staging,
            profile=args.profile,
            ckpt=ckpt,
            voice_encoder=voice_encoder,
            voices=voices,
            roster=roster,
            onnx_dir=onnx_dir,
            coreml_dir=coreml_dir,
            onnx_flag=args.onnx,
            coreml_flag=args.coreml,
        )

    return Plan(
        profile=args.profile,
        out=(args.out or REPO / "dist" / "loudr-1").resolve(),
        assemble=build,
        allowlist=(
            verify._allowlist(roster=roster, ships_onnx=True, ships_coreml=True)
            if strict
            else None
        ),
        skip_verify=args.skip_verify,
    )


def _turbo_plan(args: argparse.Namespace) -> Plan:
    ckpt = args.checkpoint.resolve()
    beside = ckpt.parent
    tokenizer = (args.tokenizer or beside / "tokenizer.json").resolve()
    voice_encoder = (args.voice_encoder or beside / VOICE_ENCODER_NAME).resolve()
    voice_dir = (args.voices or beside / "voices").resolve()

    enrollment = (args.enrollment or beside / ENROLLMENT_CHECKPOINT_NAME).resolve()
    onnx_dir = (args.onnx or beside / "onnx").resolve()
    coreml_dir = (args.coreml or beside / "coreml").resolve()
    samples = (args.samples or beside / "samples").resolve()

    roster = verify.roster_names()
    voices = verify._voice_sources(voice_dir)
    problems = preflight._skip_verify_refusal(args.skip_verify, TURBO_PROFILE)
    problems += preflight.turbo(
        ckpt=ckpt,
        tokenizer=tokenizer,
        voice_encoder=voice_encoder,
        voices=voices,
        roster=roster,
        enrollment=enrollment,
        onnx_dir=onnx_dir,
        coreml_dir=coreml_dir,
        samples=samples,
    )
    if problems:
        raise BuildRefusedError(problems)

    def build(staging: Path) -> dict[str, Any]:
        return assemble.turbo(
            staging,
            ckpt=ckpt,
            tokenizer=tokenizer,
            voice_encoder=voice_encoder,
            voices=voices,
            roster=roster,
            enrollment=enrollment,
            onnx_dir=onnx_dir,
            coreml_dir=coreml_dir,
            samples=samples,
        )

    return Plan(
        profile=TURBO_PROFILE,
        out=(args.out or REPO / "dist" / "loudr-1-turbo").resolve(),
        assemble=build,
        allowlist=verify.turbo_allowlist(roster),
        skip_verify=False,
    )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="tools/release",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", choices=MODELS, default="loudr-1")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        help="synthesis checkpoint, with its matching enrollment half beside it",
    )
    ap.add_argument("--voices", type=Path, help="directory of voice profiles")
    ap.add_argument(
        "--voice-encoder",
        type=Path,
        help=f"{VOICE_ENCODER_NAME} (voice cloning); turbo defaults to the one beside "
        "the checkpoint",
    )
    ap.add_argument(
        "--tokenizer",
        type=Path,
        help="turbo only: tokenizer.json; defaults to the one beside the checkpoint",
    )
    ap.add_argument(
        "--onnx",
        type=Path,
        help="exported ONNX graphs; defaults to onnx/ beside the checkpoint",
    )
    ap.add_argument(
        "--coreml",
        type=Path,
        help="exported CoreML packages; defaults to coreml/ beside the checkpoint",
    )
    ap.add_argument(
        "--enrollment", type=Path, help="shared enrollment checkpoint for the fusion bundle"
    )
    ap.add_argument(
        "--samples", type=Path, help="directory of measured model samples for the fusion bundle"
    )
    ap.add_argument("--out", type=Path, help="defaults to dist/<model> in this repository")
    ap.add_argument(
        "--profile",
        choices=PROFILES,
        help=(
            "loudr-1 only: full-0.1 (default) requires every piece a release ships and "
            "refuses without it; lenient assembles whatever is present and is not releasable"
        ),
    )
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="assemble without the load-and-speak check; refused under a release profile",
    )
    ap.add_argument(
        "--verify-only",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "judge an assembled bundle in place (re-hash every file, match the "
            "inventory against both manifests, re-audit the allowlist) and "
            "build nothing. The same audit a build runs before its rename."
        ),
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = _parser()
    args = ap.parse_args(argv)

    if args.verify_only is not None:
        return audit._verify_only(args.verify_only)
    if args.checkpoint is None:
        ap.error("--checkpoint is required (or pass --verify-only DIR)")

    turbo = args.model == "turbo"
    if turbo and args.profile is not None:
        ap.error(f"--profile is loudr-1's; the paired-decoder bundle uses {TURBO_PROFILE}")
    if not turbo:
        for flag in ("tokenizer", "enrollment", "samples"):
            if getattr(args, flag) is not None:
                ap.error(
                    f"--{flag} is for the paired-decoder bundle; loudr-1 uses its split layout"
                )
    if args.profile is None:
        args.profile = STRICT
    profile = TURBO_PROFILE if turbo else args.profile

    try:
        plan = (_turbo_plan if turbo else _loudr1_plan)(args)
    except BuildRefusedError as refused:
        return preflight._refuse(refused.problems, profile)

    out = plan.out
    out.parent.mkdir(parents=True, exist_ok=True)
    assemble._sweep_stale(out)
    staging = assemble._staging_dir(out)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    print(f"assembling {out}  (profile {plan.profile})")
    try:
        code = _build(staging, plan)
        if code == 0:
            assemble._commit(staging, out)
            print(f"\n  {out}")
        return code
    finally:
        # Whatever happened (a refusal, a failed check, an exception, a
        # Ctrl-C) the staging directory goes. A failed run must leave nothing
        # that looks publishable.
        shutil.rmtree(staging, ignore_errors=True)


def _build(staging: Path, plan: Plan) -> int:
    files = plan.assemble(staging)
    checksummed = assemble._write_manifests(staging, files, verified=False)
    total = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
    print(f"\n  total {total / 1e9:.2f} GB")

    # Every file in the bundle has a checksum line, or this is not a bundle
    # that vouches for itself: the downloader's checksum pass skips what the
    # manifest does not name.
    uncovered = verify._uncovered(staging, checksummed)
    if uncovered:
        print(
            "\nFAILED: SHA256SUMS does not cover every file in the bundle:\n  "
            + "\n  ".join(uncovered),
            file=sys.stderr,
        )
        return 1
    files_shipped = len(checksummed) + len(UNCHECKSUMMED)
    print(f"  {files_shipped} files, {len(checksummed)} checksummed")

    if plan.allowlist is not None:
        paths, prefixes = plan.allowlist
        drift = verify._audit(staging, paths, prefixes)
        if drift:
            print(
                f"\nFAILED: the bundle is not what the {plan.profile} profile names:\n  "
                + "\n  ".join(drift),
                file=sys.stderr,
            )
            return 1
        voices = sum(p.startswith("voices/") for p in paths)
        print(f"  matches the {plan.profile} allowlist ({voices} voices)")

    if plan.skip_verify:
        # A release profile refused this flag before anything was copied, so
        # this is a lenient bundle, and it records `"verified": false`.
        print("\nskipped the load-and-speak check: not a releasable build")
        return audit._closing_audit(staging)

    code = audit.verify(staging, profile=plan.profile)
    if code:
        return code

    # The gate ran and passed, so the claim goes into the bundle. Both
    # manifests are written again from the digests taken before the gate:
    # `release.json` gains the flag, and `SHA256SUMS` covers the file it is
    # now in. The digests are deliberately the pre-gate ones: writing them
    # from a fresh hash of the tree would launder whatever the gate did to it,
    # and the audit below exists to catch exactly that.
    assemble._write_manifests(staging, files, verified=True)
    print("  release.json records verified: true")

    # The gate imported the bundle's own code and ran it, which is the one
    # step of this build that is not a copy under this tool's control. So the
    # bundle is judged again, from disk alone.
    return audit._closing_audit(staging)


if __name__ == "__main__":
    raise SystemExit(main())
