"""The command line, from a bare machine to a cloned voice speaking.

Every synthesis command builds an :class:`~loudkit.engine.Engine` and calls
it; there is no synthesis path only the CLI can reach. Repo tooling
(``tools/bench.py``, ``tools/profile_stages.py``) is not here.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import UnsupportedLanguageError

if TYPE_CHECKING:
    from .config import ExecutionConfig, ONNXProvider
    from .voice import VoiceProfile

__all__ = ["main"]

DEFAULT_CHECKPOINT = os.environ.get("LOUDKIT_CHECKPOINT", "loudreader/loudr-1")
"""What ``--checkpoint`` means when omitted: ``$LOUDKIT_CHECKPOINT``, else the release."""


COMMANDS = ("speak", "text", "clone", "voices", "download", "serve", "verify", "doctor")
"""Commands in the order ``--help`` lists them."""


def _device_names() -> tuple[str, ...]:
    """The backends, read from the ``Device`` Literal that defines them.

    Imported lazily, as :func:`_provider_arg` reads ``ONNX_PROVIDERS``: a
    backend added to the Literal reaches the refusal and the help with no
    second edit here, and the CLI keeps its bare-import startup.
    """
    from typing import get_args

    from .execution import Device

    return get_args(Device)


def _device_arg(value: str) -> str:
    """A backend name, or ``cuda:<index>`` for one GPU on a multi-GPU box."""
    names = _device_names()
    if value in names:
        return value
    base, sep, suffix = value.partition(":")
    if base == "cuda" and sep and suffix.isdigit():
        return value
    raise argparse.ArgumentTypeError(
        f"device {value!r} must be one of {', '.join(names)} or cuda:<index> (e.g. cuda:1)"
    )


def _device_help(prefix: str, default: str) -> str:
    """One ``--device`` help line, spelled from the same tuple the refusal uses."""
    names = ", ".join(_device_names())
    return f"{prefix}: {names}, cuda:<index> (default: {default})"


def _provider_arg(value: str) -> str:
    from .execution import ONNX_PROVIDERS

    if value in ONNX_PROVIDERS:
        return value
    raise argparse.ArgumentTypeError(
        f"provider {value!r} must be one of {', '.join(ONNX_PROVIDERS)}"
    )


def _provider_help() -> str:
    """The ``--provider`` help, spelled from the same tuple the refusal uses."""
    from .execution import ONNX_PROVIDERS

    return (
        f"onnx execution provider: {', '.join(ONNX_PROVIDERS)}. Requires "
        "--device onnx. A provider this machine does not carry is refused, not "
        "demoted to cpu"
    )


def _checked_provider(args: argparse.Namespace) -> ONNXProvider | None:
    """The provider, or ``None``. Refused off the onnx device rather than dropped."""
    from typing import cast

    provider = getattr(args, "provider", None)
    if provider is None:
        return None
    if (args.device or "").split(":", 1)[0] != "onnx":
        raise ValueError(
            f"--provider {provider} needs --device onnx; "
            f"got {args.device or 'the default device'}"
        )
    return cast("ONNXProvider", provider)


def _speed_arg(value: str) -> float:
    """Range-checked by argparse, before ``load`` can download anything."""
    from .models.timestretch import MAX_SPEED, MIN_SPEED

    try:
        speed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise argparse.ArgumentTypeError(
            f"speed {speed} is outside [{MIN_SPEED}, {MAX_SPEED}]. Beyond that "
            f"range the time-stretch is audibly processed rather than merely "
            f"faster or slower, so it is refused rather than clamped."
        )
    return speed


def _execution(args: argparse.Namespace) -> ExecutionConfig | None:
    """The execution fields this command line names, or ``None`` for the build's own."""
    provider = _checked_provider(args)
    if provider is None:
        return None
    from .config import ExecutionConfig

    return ExecutionConfig(onnx_provider=provider)


_EXTRAS = {
    "torch": "torch",
    "soundfile": "audio",
    "librosa": "audio",
    "onnxruntime": "onnx",
    "coremltools": "coreml",
    "fastapi": "server",
    "uvicorn": "server",
    "mcp": "mcp",
    "grpc": "grpc",
    "torchaudio": "enroll",
    "huggingface_hub": "hub",
}


def _explain_missing(exc: ModuleNotFoundError) -> str:
    """A pip command for the interpreter's ``No module named``; a raiser's own message
    untouched."""
    written = str(exc).strip()
    if written and not written.startswith("No module named"):
        return written
    name = (exc.name or "").split(".")[0]
    if not name:
        return (
            "loudkit needs a package for this that is not installed, "
            "and the failed import did not say which."
        )
    extra = _EXTRAS.get(name)
    if extra:
        return (
            f"loudkit needs the '{name}' package for this, which is not installed.\n"
            f'  pip install "loudkit[{extra}]"'
        )
    return f"loudkit needs the '{name}' package for this, which is not installed."


_MAX_STDIN_CHARS = 1 << 20
"""How much ``loudkit speak -`` reads before deciding it was handed the wrong file.

Characters, not bytes: ``sys.stdin`` is a text stream, so ``read(n)`` counts
what the decoder produced. A passage in a language this library reads can cost
up to four bytes a character, so the byte bound is up to four times this.
"""


def _use_utf8_output() -> None:
    """Voice names and passages are in twelve languages; a cp1252 console must not kill them."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(OSError, ValueError):
            reconfigure(encoding="utf-8", errors="replace")


def _read_stdin() -> str:
    text = sys.stdin.read(_MAX_STDIN_CHARS + 1)
    if len(text) > _MAX_STDIN_CHARS:
        raise SystemExit(
            f"stdin is over {_MAX_STDIN_CHARS} characters; this is almost certainly the "
            "wrong file. For a book, use the library's stream() or the server."
        )
    return text


def _is_bare_voice_name(value: str) -> bool:
    """``kathleen``, not anything addressed as a path."""
    return bool(value) and not value.startswith(".") and "/" not in value and "\\" not in value


def _voice_release(args: argparse.Namespace) -> str | None:
    """The release a bare ``--voice`` resolves against: a repo id, a directory, or a
    file's parent."""
    from .hub import is_repo_id

    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is None:
        return None
    if is_repo_id(str(checkpoint)):
        return str(checkpoint)
    path = Path(checkpoint)
    if path.is_dir():
        return str(path)
    if path.is_file():
        return str(path.parent)
    return None


def _pinned_checkpoint(args: argparse.Namespace) -> str:
    """``--checkpoint`` resolved to a file when ``--revision`` pins it; a path passes
    through."""
    checkpoint = str(args.checkpoint)
    revision = getattr(args, "revision", None)
    if revision is None:
        return checkpoint
    from .hub import backend_for_device, is_repo_id, resolve_checkpoint

    if not is_repo_id(checkpoint):
        return checkpoint
    backend = backend_for_device(getattr(args, "device", None))
    return str(resolve_checkpoint(checkpoint, revision=revision, backend=backend))


def _play_hint(output: Path) -> str:
    """One command that plays ``output`` on this platform."""
    import platform

    system = platform.system()
    if system == "Darwin":
        return f"afplay {shlex.quote(str(output))}"
    if system == "Windows":
        # Quoted like the other two: a path with a space is otherwise two
        # arguments and the hint does not run. Double quotes because PowerShell
        # takes them and `"` is not legal in a Windows path, so nothing here
        # needs escaping.
        return f'Start-Process "{output}"'
    return f"aplay {shlex.quote(str(output))}"


def _play_audio(output: Path) -> None:
    """Play a saved WAV using the platform player, with paths passed as data."""
    import platform
    import shutil
    import subprocess

    system = platform.system()
    players, install = {
        "Darwin": (("afplay",), "restore the macOS afplay system utility"),
        "Windows": (("powershell", "pwsh"), "install PowerShell"),
    }.get(system, (("aplay", "paplay"), "install alsa-utils or pulseaudio-utils"))
    player = next((found for name in players if (found := shutil.which(name))), None)
    if player is None:
        raise RuntimeError(f"playback unavailable: {install}; WAV saved at {output}")
    path = str(output.resolve())
    argv = [player, path]
    env = None
    if system == "Windows":
        env = {**os.environ, "LOUDKIT_PLAY_WAV": path}
        argv = [
            player,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$ErrorActionPreference = 'Stop'; "
            "(New-Object System.Media.SoundPlayer $env:LOUDKIT_PLAY_WAV).PlaySync()",
        ]
    try:
        subprocess.run(argv, check=True, env=env)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"playback failed with {Path(player).name}; WAV saved at {output}"
        ) from exc


def _cmd_text(args: argparse.Namespace) -> int:
    """Print prepared speech and the changes made by the existing text funnel."""
    from difflib import SequenceMatcher

    import loudkit

    from .frontend.speechtext import speech_text

    language = args.language.lower()
    if language not in loudkit.languages():
        raise UnsupportedLanguageError(
            f"{language!r}; known: {', '.join(loudkit.languages())}",
            language=language,
            supported=loudkit.languages(),
        )
    original = args.text if args.text != "-" else _read_stdin()
    prepared = speech_text(original, language)
    print(prepared)
    # Compare complete words so expansions stay readable; this is a diff, not a stage trace.
    before, after = original.split(), prepared.split()
    for operation, i, j, a, b in SequenceMatcher(None, before, after).get_opcodes():
        old, new = " ".join(before[i:j]), " ".join(after[a:b])
        if operation == "replace":
            print(f"replaced {old!r} -> {new!r}", file=sys.stderr)
        elif operation == "delete":
            print(f"removed {old!r}", file=sys.stderr)
        elif operation == "insert":
            print(f"inserted {new!r}", file=sys.stderr)
    return 0


def _load_voice(args: argparse.Namespace) -> VoiceProfile:
    """A path, or a bare name against the checkpoint's own release, at the same revision."""
    import loudkit

    from .hub import resolve_voice

    release = _voice_release(args)
    if Path(args.voice).is_file():
        return loudkit.VoiceProfile.load(args.voice)
    if release is not None and _is_bare_voice_name(str(args.voice)):
        return loudkit.VoiceProfile.load(
            resolve_voice(str(args.voice), repo=release, revision=args.revision)
        )
    return loudkit.VoiceProfile.load(args.voice)


def _cmd_speak(args: argparse.Namespace) -> int:
    """Renders and exits, so it never warms: see ``docs/design/transports.md``."""
    import loudkit

    # The two cheap refusals first. Reading the checkpoint is 747 MB off disk,
    # or a download for a repo id; stdin over the cap and a voice that is not
    # there both cost nothing to discover, and a user told either one after
    # minutes of loading has waited for an answer the first line could give.
    text = args.text if args.text != "-" else _read_stdin()
    voice = _load_voice(args)
    engine = loudkit.load(
        args.checkpoint,
        device=args.device,
        execution=_execution(args),
        revision=args.revision,
    )
    if args.verbose:
        print(engine.describe(), file=sys.stderr)
    result = engine.synthesize(
        text, voice, seed=args.seed, language=args.language, speed=args.speed
    )
    result.save(args.output, include_provenance=not args.no_provenance)
    print(
        f"{result.duration:.2f}s of audio -> {args.output}  "
        f"({result.timings.describe(result.duration)})",
        file=sys.stderr,
    )
    if result.hit_token_cap:
        print(
            "warning: generation stopped at the token cap rather than at a stop "
            "token, so the reading is probably truncated",
            file=sys.stderr,
        )
    if args.play:
        _play_audio(args.output)
    else:
        print(f"hear it: {_play_hint(args.output)}", file=sys.stderr)
    return 0


_CLONE_AUDIO_SUFFIXES = (".wav", ".flac")
"""Lossless and local. A lossy recording makes a worse clone than its owner can see."""

_ENROLLMENT_TENSOR_PREFIXES = ("s3gen.speaker_encoder.", "s3gen.tokenizer.")
"""The two tensor groups a clone reads."""


def _clone_output(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return Path(args.output)
    return Path("voices") / f"{args.name}.safetensors"


def _require_clone_extras(checkpoint: str, device: str | None) -> None:
    """Every extra a clone needs, named up front rather than one download apart."""
    import importlib.util

    from .hub import is_repo_id

    if device in ("onnx", "coreml"):
        module = "onnxruntime" if device == "onnx" else "coremltools"
        needed = [module, "soundfile"]
        extras = f"{device},audio"
    else:
        needed = ["torch", "torchaudio", "librosa"]
        extras = "enroll"
    if is_repo_id(checkpoint):
        needed.append("huggingface_hub")
        extras += ",hub"
    missing = [module for module in needed if importlib.util.find_spec(module) is None]
    if missing:
        raise ModuleNotFoundError(
            f"cloning needs {', '.join(missing)}, which this environment does not have.\n"
            f'  pip install "loudkit[{extras}]"'
        )


def _save_voice_atomically(profile: VoiceProfile, output: Path) -> Path:
    """Write beside ``output`` and rename onto it, so a dying run leaves no half profile."""
    import tempfile

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.stem}.", suffix=f".partial{output.suffix}"
    )
    os.close(fd)
    tmp = Path(name)
    try:
        profile.save(tmp)
        os.replace(tmp, output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return output


def _cmd_clone(args: argparse.Namespace) -> int:
    """A front end over :func:`loudkit.enroll`. Consent is yours to obtain:
    RESPONSIBLE_USE.md."""
    import loudkit

    from .hub import resolve_enrollment_checkpoint

    audio = Path(args.audio)
    if audio.suffix.lower() not in _CLONE_AUDIO_SUFFIXES:
        raise ValueError(
            f"{audio.name}: clone reads a WAV or a FLAC file. Convert it first, "
            "or pass samples to loudkit.enroll() from Python."
        )
    if not _is_bare_voice_name(args.name):
        raise ValueError(
            f"--name {args.name!r} becomes a filename and a voice key, so it may "
            "not hold a path separator or start with a dot."
        )
    # Lower-cased before the check, as `text` does it: a language is a tag, not
    # a spelling, and `--language EN` must not be refused here while the same
    # tag is accepted there.
    language = args.language.lower()
    supported = loudkit.languages()
    if language not in supported:
        raise UnsupportedLanguageError(
            f"{language!r}; known: {', '.join(supported)}",
            language=language,
            supported=supported,
        )
    checkpoint = str(args.checkpoint)
    _require_clone_extras(checkpoint, args.device)
    output = _clone_output(args)
    # The one deliberate early `return 1` here: a question about the disk, not
    # about the input, so it carries no exception class and nothing above needs
    # to classify it.
    if output.exists() and not args.force:
        print(
            f"{output} exists. Pass --force to overwrite it, or -o to write elsewhere.",
            file=sys.stderr,
        )
        return 1
    # Before the model loads: a synthesis-only release cannot clone at all.
    if args.device not in ("onnx", "coreml"):
        enrollment = resolve_enrollment_checkpoint(checkpoint, revision=args.revision)
        print(f"enrollment tensors: {enrollment}", file=sys.stderr)
    print(f"enrolling {audio} ...", file=sys.stderr)
    profile = loudkit.enroll(
        str(audio),
        checkpoint,
        end_in_silence=not args.no_end_in_silence,
        name=args.name,
        language=language,
        device=args.device or "cpu",
        revision=args.revision,
    )
    _save_voice_atomically(profile, output)
    permissions = "mode 0600" if os.name == "posix" else "directory ACL inherited"
    print(
        f"{output}  ({output.stat().st_size} bytes, language {profile.language}, {permissions})"
    )
    print(
        f"speak with:\n  loudkit speak --checkpoint {checkpoint} "
        f'--voice {output} "hello"\n'
        "consent is yours to obtain: see RESPONSIBLE_USE.md",
        file=sys.stderr,
    )
    return 0


_DOCTOR_LISTING_CAP = 8
"""How many artefacts ``doctor`` lists per section; what it drops, it counts."""


def _report_truncation(total: int) -> None:
    if total > _DOCTOR_LISTING_CAP:
        print(
            f"  ... and {total - _DOCTOR_LISTING_CAP} more (listing stops at "
            f"{_DOCTOR_LISTING_CAP})"
        )


def _header(path: Path) -> tuple[dict[str, str], list[str]] | None:
    """The safetensors metadata and tensor names, or ``None`` for a header that will
    not parse."""
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            return (f.metadata() or {}), list(f.keys())
    except Exception:  # a header that will not parse, whatever the cause
        return None


def _kind(path: Path) -> str | None:
    """``checkpoint``, ``voice``, ``unreadable``, or ``None`` for somebody else's file.

    A truncated download is the one file ``doctor`` most needs to name.
    """
    header = _header(path)
    if header is None:
        return "unreadable"
    meta, _ = header
    if "manifest" in meta:
        return "checkpoint"
    if "voice" in meta:
        return "voice"
    return None


def _carries_enrollment_tensors(path: Path) -> bool:
    header = _header(path)
    if header is None:
        return False
    _, names = header
    return all(
        any(name.startswith(prefix) for name in names) for prefix in _ENROLLMENT_TENSOR_PREFIXES
    )


def _enrollment_tensor_file(checkpoint: Path) -> Path | None:
    """The file a clone would read beside ``checkpoint``, by the hub's rules, with its
    tensors checked."""
    from .hub import resolve_enrollment_checkpoint

    try:
        holder = resolve_enrollment_checkpoint(str(checkpoint))
    except FileNotFoundError:
        return None
    return holder if _carries_enrollment_tensors(holder) else None


def _doctor_root(checkpoint: str | None) -> Path:
    """Where ``doctor`` looks for assets: ``--checkpoint`` when it names a
    release directory or a file in one, else the current directory. A repo id
    names nothing on disk and falls through."""
    if checkpoint:
        path = Path(checkpoint)
        if path.is_file():
            return path.parent
        if path.is_dir():
            return path
    return Path.cwd()


def _report_cloning(found: list[tuple[Path, str]], where: str) -> None:
    """Three questions with three remedies: the extra, the encoder, the enrollment tensors."""
    import importlib.util

    from . import hub

    missing = [
        module
        for module in ("torch", "torchaudio", "librosa")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        print(f'  extra      missing {", ".join(missing)}: pip install "loudkit[enroll]"')
    else:
        print("  extra      installed   (torch, torchaudio, librosa)")

    checkpoints = [
        path
        for path, kind in found
        if kind == "checkpoint"
        and path.name != hub.ENROLLMENT_NAME
        and hub._artifact_role(path) != hub.ENROLLMENT_ROLE
    ]
    for path in checkpoints[:_DOCTOR_LISTING_CAP]:
        encoder = path.parent / hub.VOICE_ENCODER_NAME
        holder = _enrollment_tensor_file(path)
        # Its own name: `where` is the directory searched, which the last line
        # of this function still prints, and the loop's answer is a file.
        if holder is None:
            tensors_where = f"none (looked in this file and in {hub.ENROLLMENT_NAME})"
        elif holder == path:
            tensors_where = "in this file"
        else:
            tensors_where = holder.name
        state = str(encoder) if encoder.is_file() else f"missing ({encoder})"
        print(f"  {path}  enrollment tensors: {tensors_where}  encoder: {state}")
    _report_truncation(len(checkpoints))
    if not checkpoints:
        print(f"  no checkpoint under {where} to read for enrollment tensors")


def _cached_releases() -> list[str]:
    """loudkit releases in the Hugging Face cache, by repo id. Nobody else's models."""
    from . import hub

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ModuleNotFoundError:
        return []
    cache = Path(HF_HUB_CACHE)
    if not cache.exists():
        return []
    ours = []
    for entry in sorted(cache.glob("models--*")):
        held = any(
            hub._manifest_of(path) is not None
            for snapshot in (entry / "snapshots").glob("*")
            for path in hub._root_checkpoints(snapshot)
        )
        if held:
            ours.append(entry.name.removeprefix("models--").replace("--", "/"))
    return ours


def _cmd_doctor(  # noqa: PLR0912, PLR0915 - a checklist reads as a checklist
    args: argparse.Namespace,
) -> int:
    """What this machine can run and the command that fixes each gap.

    Exits 0 whatever it finds on disk. Two things outside the checklist can
    still exit 1: a ``--checkpoint`` that is not there, refused before dispatch
    like every other command's, and ``--describe``, which loads the engine and
    can fail the way ``speak`` fails.
    """
    import importlib.util
    import platform

    from . import __version__

    def have(module: str) -> bool:
        return importlib.util.find_spec(module) is not None

    print(f"loudkit {__version__}")
    print(f"python  {platform.python_version()} on {platform.system()} {platform.machine()}")
    print()
    print("backends:")
    if have("torch"):
        import torch

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
        print(f"  torch {torch.__version__:<12} devices: {', '.join(devices)}")
    else:
        print('  torch        missing: pip install "loudkit[torch]"')
    for module, extra in (("onnxruntime", "onnx"), ("coremltools", "coreml")):
        if have(module):
            print(f"  {module} installed")
        else:
            print(f'  {module:<12} missing: pip install "loudkit[{extra}]"')
    print()
    print("extras:")
    for module, extra, what in (
        ("soundfile", "audio", "writing WAVs"),
        ("librosa", "audio", "loading audio for enrollment"),
        ("huggingface_hub", "hub", "downloading by repo id"),
        ("fastapi", "server", "loudkit serve"),
        ("grpc", "grpc", "loudkit serve --grpc"),
        ("mcp", "mcp", "loudkit serve --mcp"),
    ):
        state = "installed" if have(module) else f'missing: pip install "loudkit[{extra}]"'
        print(f"  {module:<16} {state}   ({what})")
    print()
    print("assets:")
    root = _doctor_root(args.checkpoint)
    where = "the current directory" if root == Path.cwd() else str(root)
    found = [
        (path, kind)
        for path in sorted(root.glob("*.safetensors")) + sorted(root.glob("*/*.safetensors"))
        if (kind := _kind(path)) is not None
    ]
    for path, kind in found[:_DOCTOR_LISTING_CAP]:
        print(f"  {path}  ({kind})")
    _report_truncation(len(found))
    if not found:
        print(f"  no loudkit checkpoint or voice under {where}")
    if have("huggingface_hub"):
        from huggingface_hub.constants import HF_HUB_CACHE

        cached = _cached_releases()
        for name in cached:
            print(f"  cached: {name}")
        if not cached:
            print(f"  no loudkit release in the hub cache ({HF_HUB_CACHE})")
    print()
    print("cloning:")
    _report_cloning(found, where)
    print()
    if args.describe:
        import loudkit

        engine = loudkit.load(
            args.checkpoint,
            device=args.device,
            execution=_execution(args),
            revision=args.revision,
        )
        print(f"engine:  {engine.describe()}")
        print()
    print("to fetch a release:  loudkit download loudreader/loudr-1")
    print('to hear a voice:     loudkit speak --voice <name> "hello"')
    print("to clone a voice:    loudkit clone me.wav --name mine --language en")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    """Fetch what one backend needs from a release, verified, and nothing more."""
    from . import hub

    print(f"resolving {args.repo} ...", file=sys.stderr)
    root = hub.download(
        args.repo,
        revision=args.revision,
        backend=args.backend,
        cloning=args.with_cloning,
        local_dir=args.local_dir,
    )
    checkpoint = hub.resolve_checkpoint(str(root))
    print(f"checkpoint  {checkpoint}")
    if args.with_cloning and args.backend == "torch":
        print(f"encoder     {root / hub.VOICE_ENCODER_NAME}")
        print(f"enrollment  {root / hub.ENROLLMENT_NAME}")
    elif args.with_cloning:
        print(f"enrollment  {args.backend} graphs in {root / args.backend}")
    if args.backend != "torch":
        print(f"{args.backend:<7}     {root / args.backend}")
    voice_count = sum(1 for p in (root / "voices").glob("*.safetensors") if p.is_file())
    print(f"voices      {voice_count} in {root / 'voices'}")
    ref = str(args.local_dir) if args.local_dir else args.repo
    print(
        f"\ndone. speak with:\n  loudkit speak --checkpoint {ref} "
        f'--voice <name from `loudkit voices {ref}`> "hello"',
        file=sys.stderr,
    )
    return 0


def _cmd_voices(args: argparse.Namespace) -> int:
    from .hub import VOICE_DIR, VOICE_SUFFIX, list_voices

    names = list_voices(repo=args.repo, revision=args.revision)
    for name in names:
        print(name)
    if not names:
        where = f"{args.repo}@{args.revision}" if args.revision else args.repo
        print(
            f"{where}: nothing under {VOICE_DIR}/ ends in {VOICE_SUFFIX}.\n"
            "Check the repo id and, if you pinned one, the revision. A release "
            "built without its voice directory lists nothing here too.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Say what a file is and whether its own claims hold."""
    path = Path(args.path)
    with path.open("rb") as handle:
        head = handle.read(4)
    if head == b"RIFF":
        return _verify_wav(path)
    return _verify_safetensors(path)


def _verify_wav(path: Path) -> int:
    from .provenance import verify_provenance

    # One call, not two: `verify_provenance` reads the manifest itself and
    # answers `(None, False)` when there is none, so asking first would read
    # the whole file twice to learn the same thing.
    manifest, ok = verify_provenance(path)
    if manifest is None:
        print(f"{path}: a WAV with no provenance manifest")
        return 1
    assertions = manifest.get("assertions") if isinstance(manifest, dict) else None
    for assertion in assertions if isinstance(assertions, list) else []:
        if not isinstance(assertion, dict) or assertion.get("label") != "loudkit.provenance":
            continue
        data = assertion.get("data")
        if not isinstance(data, dict):
            continue
        for key in (
            "algorithm_fingerprint",
            "checkpoint_sha256",
            "voice",
            "voice_profile_sha256",
            "backend",
            "execution",
            "seed",
        ):
            value = data.get(key, "")
            if value != "":
                print(f"  {key}: {value}")
    if ok:
        print(f"{path}: provenance verified; the audio is the audio the manifest signs for")
        print(
            "  (integrity, not authenticity: the manifest is unsigned, so this "
            "proves the file is intact, not who made it)"
        )
        return 0
    print(f"{path}: provenance FAILED; the audio does not match its manifest")
    return 1


def _verify_safetensors(path: Path) -> int:
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata() or {}

    if "voice" in meta:
        from .checkpoint import file_sha256
        from .voice import VoiceProfile

        profile = VoiceProfile.load(path)
        print(f"{path}: a voice profile, and a valid one")
        print(f"  name: {profile.name}")
        print(f"  language: {profile.language}")
        print(f"  enrolment: {profile.enrolment}")
        print(f"  sha256: {file_sha256(path)}")
        return 0

    if "manifest" in meta:
        from .checkpoint import file_sha256, payload_sha256, read_manifest

        manifest = read_manifest(path)
        print(f"{path}: a loudkit checkpoint")
        print(f"  name: {manifest.get('name', path.stem)}")
        print(f"  recipe: {manifest.get('recipe_version', '?')}")
        print(f"  sha256: {file_sha256(path)}  (compare against the release's SHA256SUMS)")
        recorded = manifest.get("tensor_payload_sha256")
        if not recorded:
            print("  payload digest: not recorded by this checkpoint")
            return 0
        print("  hashing the tensor payload (streams every tensor, takes a moment) ...")
        actual = payload_sha256(path)
        if actual == recorded:
            print("  payload digest: verified; the tensors are the tensors it names")
            return 0
        print(f"  payload digest: MISMATCH\n    recorded: {recorded}\n    actual:   {actual}")
        return 1

    print(f"{path}: a safetensors file, but neither a loudkit checkpoint nor a voice profile")
    print(f"  metadata keys: {sorted(meta) or 'none'}")
    return 1


_SERVE_DROPS: dict[str, dict[str, str]] = {
    "mcp": {
        "--host": "MCP speaks over stdio; it binds no address",
        "--port": "MCP speaks over stdio; it binds no port",
        "--allow-public": "there is no bind here to make public",
        "--token": "there is no request to authenticate; the host owns the pipe",
        "--first-chunk-tokens": "the tool returns the whole reply, so it has no first chunk",
    },
    "grpc": {
        "--allow-public": "the gRPC transport has no authentication to opt out of; "
        "serve HTTP for a public bind",
        "--token": "the gRPC transport reads no bearer token; serve HTTP for one",
    },
}
"""Per transport, the ``serve`` flags it cannot act on and why.

Each was parsed and dropped, which is the worst of the three possible
answers: the server came up, said nothing, and ran with a setting the
operator believes is in force. ``--first-chunk-tokens`` is the sharp one, it
re-fingerprints the algorithm on the other two doors and did nothing here.
HTTP drops none of them and so is absent."""


def _dropped_serve_flags(args: argparse.Namespace, door: str) -> list[tuple[str, str]]:
    """Every flag on this command line that ``door`` would ignore, with the reason.

    ``--allow-public`` defaults to ``False`` and the rest to ``None``, so
    holding anything else is "the operator typed it". The namespace attribute
    is derived from the flag the way argparse itself derives it, rather than
    written down a second time beside it.
    """
    return [
        (flag, why)
        for flag, why in _SERVE_DROPS.get(door, {}).items()
        if getattr(args, flag.lstrip("-").replace("-", "_")) not in (None, False)
    ]


def _cmd_serve(args: argparse.Namespace) -> int:
    """HTTP by default; ``--grpc`` and ``--mcp`` are the other two transports over the
    one engine."""
    door = "mcp" if args.mcp else "grpc" if args.grpc else "http"
    dropped = _dropped_serve_flags(args, door)
    if dropped:
        # Before `_pinned_checkpoint`, which can download a release: a command
        # line that will be refused should not first fetch a gigabyte.
        named = ", ".join(flag for flag, _ in dropped)
        reasons = "\n".join(f"  {flag}: {why}" for flag, why in dropped)
        print(
            f"serve --{door} cannot honour {named}.\n{reasons}\n"
            "Drop the flag, or serve a transport that reads it.",
            file=sys.stderr,
        )
        return 2
    checkpoint = _pinned_checkpoint(args)
    if args.mcp:
        from .transports.mcp import run_stdio

        run_stdio(checkpoint, args.voices, device=args.device)
        return 0
    # Passed only when given, so each transport's own signature default is the
    # one statement of its host and its port. The `--host` and `--port` help
    # strings keep the values for a reader.
    bind = {
        **({"host": args.host} if args.host is not None else {}),
        **({"port": args.port} if args.port is not None else {}),
    }
    if args.grpc:
        from .transports.grpc import serve as serve_grpc

        serve_grpc(
            checkpoint,
            args.voices,
            device=args.device,
            **bind,
            first_chunk_tokens=args.first_chunk_tokens,
        )
        return 0
    from .transports.http import serve

    serve(
        checkpoint,
        voices=args.voices,
        **bind,
        device=args.device,
        allow_public=args.allow_public,
        # The environment keeps the token out of `ps` and shell history.
        token=args.token or os.environ.get("LOUDKIT_TOKEN"),
        first_chunk_tokens=args.first_chunk_tokens,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915 - flat argparse declarations
    """The full grammar, exposed so tests can assert on it without running a command."""
    parser = argparse.ArgumentParser(
        prog="loudkit", description="Text to speech that behaves the same everywhere."
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="re-raise errors with their traceback instead of one diagnosis line",
    )
    sub = parser.add_subparsers(dest="command", metavar=f"{{{','.join(COMMANDS)}}}")

    common = argparse.ArgumentParser(add_help=False)
    # str, not Path: a repo id is not a path.
    common.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"the synthesis .safetensors, a release directory, or a repo id "
        f"(default: $LOUDKIT_CHECKPOINT, else {DEFAULT_CHECKPOINT})",
    )
    common.add_argument(
        "--revision",
        default=None,
        metavar="REF",
        help="commit, tag or branch to pin --checkpoint to; without it a repo "
        "id follows the default branch, which moves. Ignored for a path.",
    )
    common.add_argument(
        "--device",
        default=None,
        type=_device_arg,
        metavar="DEVICE",
        help=_device_help("backend", "the fastest available"),
    )
    common.add_argument(
        "--provider", default=None, type=_provider_arg, metavar="NAME", help=_provider_help()
    )

    speak = sub.add_parser("speak", parents=[common], help="synthesise to a WAV")
    speak.add_argument("text", help="text to speak, or '-' to read stdin")
    speak.add_argument(
        "--voice", required=True, type=Path, help="a voice name, or a profile file"
    )
    speak.add_argument("-o", "--output", default="out.wav", type=Path)
    speak.add_argument("--seed", type=int, default=0, help="same seed, same audio")
    speak.add_argument(
        "--language",
        default=None,
        help="language id for the text frontend (default: the voice's own)",
    )
    speak.add_argument(
        "--speed",
        type=_speed_arg,
        default=1.0,
        metavar="X",
        help="playback speed, 0.5 to 2.0, pitch preserved (default: 1.0, exact)",
    )
    speak.add_argument(
        "--no-provenance",
        action="store_true",
        help="write a plain WAV, without the C2PA claim-only manifest",
    )
    speak.add_argument(
        "--verbose", action="store_true", help="print the engine's algorithm and execution line"
    )
    speak.add_argument("--play", action="store_true", help="play the WAV after saving it")
    speak.set_defaults(func=_cmd_speak)

    text = sub.add_parser("text", help="preview spoken text without loading a model")
    text.add_argument("text", help="text to prepare, or '-' to read stdin")
    text.add_argument("--language", default="en", help="text language (default: en)")
    text.set_defaults(func=_cmd_text)

    # No `parents=[common]`, deliberately: cloning's
    # `--device` is the enrollment backend, whose default is cpu and whose help
    # names a different order of backends, and `--provider` belongs to synthesis
    # and must not appear here. `--checkpoint` and `--revision` are re-declared
    # for the same reason, so the three that differ stay together.
    clone = sub.add_parser("clone", help="clone a voice from a recording and write the profile")
    clone.add_argument(
        "audio",
        help="a local WAV or FLAC recording: 5 to 10 seconds of clean, "
        "single-speaker audio. More than 30 seconds is refused",
    )
    clone.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="a cloning-capable release: the enrollment artefact and ve.safetensors "
        f"ride with it (default: $LOUDKIT_CHECKPOINT, else {DEFAULT_CHECKPOINT})",
    )
    clone.add_argument(
        "--name", required=True, help="what to call the voice; also the filename"
    )
    clone.add_argument(
        "--language",
        required=True,
        help="the language this voice speaks, e.g. en, pl; what the engine reads "
        "text as when a call names none",
    )
    clone.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        metavar="OUTPUT",
        help="where to write the profile (default: voices/<name>.safetensors)",
    )
    clone.add_argument("--revision", default=None, metavar="REF", help="pin --checkpoint")
    clone.add_argument(
        "--device",
        default=None,
        type=_device_arg,
        metavar="DEVICE",
        help=_device_help("enrollment backend", "cpu"),
    )
    clone.add_argument("--force", action="store_true", help="overwrite an existing output")
    clone.add_argument(
        "--no-end-in-silence",
        action="store_true",
        help="enroll the clip as given; by default it is cut at its last pause before the "
        "10 s prompt limit (or truncated if no pause fits) and padded with silence",
    )
    clone.set_defaults(func=_cmd_clone)

    voices = sub.add_parser("voices", help="list the voices a release holds, one name per line")
    voices.add_argument(
        "repo",
        nargs="?",
        default=DEFAULT_CHECKPOINT,
        help=f"repo id, or a local release directory (default: {DEFAULT_CHECKPOINT})",
    )
    voices.add_argument("--revision", default=None, help="commit, tag or branch")
    voices.set_defaults(func=_cmd_voices)

    download = sub.add_parser(
        "download", help="fetch what one backend needs: checkpoint, graphs, voices"
    )
    download.add_argument("repo", help="Hugging Face repo id, e.g. loudreader/loudr-1")
    download.add_argument(
        "--for",
        dest="backend",
        choices=("torch", "onnx", "coreml"),
        default="torch",
        help="the backend the files must run on (default: torch). onnx is also "
        "what the Rust, Go and JS ports read; coreml is also what Swift reads",
    )
    download.add_argument(
        "--with-cloning",
        action="store_true",
        help="also fetch what this backend enrols with",
    )
    download.add_argument(
        "--revision", default=None, metavar="REF", help="commit, tag or branch"
    )
    download.add_argument(
        "--local-dir",
        default=None,
        type=Path,
        metavar="DIR",
        help="materialise the files in this directory instead of the shared cache",
    )
    download.set_defaults(func=_cmd_download)

    serve = sub.add_parser("serve", parents=[common], help="run a local synthesis server")
    transport = serve.add_mutually_exclusive_group()
    transport.add_argument(
        "--grpc", action="store_true", help="serve over gRPC instead of HTTP (port 50051)"
    )
    transport.add_argument(
        "--mcp", action="store_true", help="serve over MCP on stdio instead of HTTP"
    )
    serve.add_argument(
        "--host", default=None, help="127.0.0.1 by default; HTTP and gRPC, MCP is stdio"
    )
    serve.add_argument(
        "--port",
        type=int,
        default=None,
        help="8765 for HTTP, 50051 for gRPC; MCP is stdio",
    )
    serve.add_argument(
        "--allow-public",
        action="store_true",
        help="bind a non-loopback host; HTTP only, and requires a bearer token",
    )
    serve.add_argument(
        "--token",
        default=None,
        help="bearer token every request must carry, HTTP only (falls back to $LOUDKIT_TOKEN)",
    )
    serve.add_argument("--voices", type=Path, help="directory of voice profiles")
    serve.add_argument(
        "--first-chunk-tokens",
        type=int,
        default=None,
        metavar="N",
        help="cap the FIRST streamed chunk at N tokens so audio starts sooner; "
        "HTTP and gRPC (opt-in; re-fingerprints)",
    )
    serve.set_defaults(func=_cmd_serve)

    verify = sub.add_parser(
        "verify",
        help="check a checkpoint, voice profile or rendered WAV against its own claims",
    )
    verify.add_argument("path", type=Path, help=".safetensors or .wav")
    verify.set_defaults(func=_cmd_verify)

    doctor = sub.add_parser(
        "doctor",
        parents=[common],
        help="report what this machine can run, and how to fix what it cannot",
    )
    doctor.add_argument(
        "--describe",
        action="store_true",
        help="also load --checkpoint and print its algorithm and execution line",
    )
    doctor.set_defaults(func=_cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    _use_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from . import __version__

        print(__version__)
        return 0
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    missing = _missing_path(args)
    if missing is not None:
        print(f"{missing[0]} not found: {missing[1]}", file=sys.stderr)
        return 1
    return _run(args)


def _missing_path(args: argparse.Namespace) -> tuple[str, str] | None:
    """The first argument naming a file that is not there, so the error carries a filename."""
    from .hub import is_repo_id

    for label in ("voices",):
        path = getattr(args, label, None)
        if path is not None and not Path(path).is_dir():
            return label, str(path)
    audio = getattr(args, "audio", None)
    if audio is not None and not Path(audio).is_file():
        return "audio", str(audio)
    for label in ("checkpoint", "voice"):
        path = getattr(args, label, None)
        if not isinstance(path, str | Path):
            continue
        if Path(path).exists() or is_repo_id(str(path)):
            continue
        # A bare voice name is not a missing file when --checkpoint names its release.
        if label == "voice" and _is_bare_voice_name(str(path)) and _voice_release(args):
            continue
        return label, str(path)
    return None


try:  # pragma: no cover - the import guard, not the behaviour
    from safetensors import SafetensorError as _SafetensorError
except ImportError:  # safetensors is a hard dependency; this keeps --help alive

    class _SafetensorError(Exception):  # type: ignore[no-redef]
        """Stands in when safetensors is absent, so the ladder still parses."""


def _run(args: argparse.Namespace) -> int:
    """Dispatch, turning each failure into one line. ``--debug`` keeps the traceback."""
    if getattr(args, "debug", False):
        # Above the ladder rather than in each clause: the flag means "do not
        # catch", and a guard repeated six times is one a seventh clause can be
        # written without.
        return int(args.func(args))
    try:
        return int(args.func(args))
    except ModuleNotFoundError as exc:
        print(_explain_missing(exc), file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"not found: {exc.filename or exc}", file=sys.stderr)
        return 1
    # Before RuntimeError: it is a subclass, and it is a question about the input.
    except UnsupportedLanguageError as exc:
        print(f"unsupported: {exc}", file=sys.stderr)
        return 1
    except NotImplementedError as exc:
        print(
            f"internal error: {type(exc).__name__}: {exc}\n"
            "This is a bug in loudkit, not in your command. Re-run with --debug "
            "for the traceback.",
            file=sys.stderr,
        )
        return 1
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, _SafetensorError) as exc:
        print(
            f"unreadable: {exc}\n"
            "The file is present but could not be read. If it was downloaded, "
            "the copy may be truncated. Check its size against the release, or "
            "fetch it again.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
