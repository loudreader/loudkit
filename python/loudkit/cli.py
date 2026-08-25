"""The command line: speak, and the commands around speaking.

``doctor``, ``download``, ``voices`` and ``verify`` exist so the path from a
bare machine to a first WAV is four commands with no reading: what can this
machine run, fetch the release, pick a voice, check what arrived.

Small on purpose. Every synthesis command constructs an
:class:`~loudkit.engine.Engine` and calls it — there is no synthesis path that
only the CLI can reach, because a second path is a second thing to keep in
agreement, and this library exists because two paths drifted.

Missing optional dependencies are reported by name. A stranger who runs
``loudkit speak`` and gets ``ModuleNotFoundError: torch`` has been told nothing
useful; being told to install ``loudkit[torch]`` costs one line here and saves
them a search.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import UnsupportedLanguageError

if TYPE_CHECKING:
    from .config import ExecutionOverrides, ONNXProvider
    from .voice import VoiceProfile

__all__ = ["main"]

_DEVICE_NAMES = ("cpu", "cuda", "mps", "coreml", "onnx")


def _device_arg(value: str) -> str:
    """Validate ``--device``: a backend name, or ``cuda`` with an index.

    The registry accepts ``cuda:N`` for a specific GPU on a multi-GPU box
    (``build_engine`` splits on ``:``); the argparse ``choices`` list could
    not, which made ``loudkit bench --device cuda:1`` an error a user had to
    route around through the API. Accept ``cuda``, ``cuda:0``, ``cuda:1``, …
    and the other named backends; reject everything else with a usable message.
    """
    if value in _DEVICE_NAMES:
        return value
    base, sep, suffix = value.partition(":")
    if base == "cuda" and sep and suffix.isdigit():
        return value
    raise argparse.ArgumentTypeError(
        f"device {value!r} must be one of {', '.join(_DEVICE_NAMES)} "
        "or cuda:<index> (e.g. cuda:1)"
    )


_CLONE_DEVICE_NAMES = ("cpu", "cuda", "mps")


def _clone_device_arg(value: str) -> str:
    """Validate ``clone --device``: the torch devices, and nothing else.

    Enrollment runs the speaker encoder and the speech tokenizer through the
    torch backend. There is no ONNX or CoreML enrollment graph, so accepting
    those two names here would take a flag and answer it somewhere else.
    """
    if value in _CLONE_DEVICE_NAMES:
        return value
    base, sep, suffix = value.partition(":")
    if base == "cuda" and sep and suffix.isdigit():
        return value
    raise argparse.ArgumentTypeError(
        f"device {value!r} must be one of {', '.join(_CLONE_DEVICE_NAMES)} "
        "or cuda:<index> — enrollment runs on the torch backend"
    )


def _provider_arg(value: str) -> str:
    """Validate ``--provider`` against the five cross-port spellings.

    The list is imported here rather than at module scope because importing
    ``loudkit.config`` imports the package, which costs more than every other
    thing ``loudkit --help`` does. ``type=`` runs only when the flag is passed.
    """
    from .config import ONNX_PROVIDERS

    if value in ONNX_PROVIDERS:
        return value
    raise argparse.ArgumentTypeError(
        f"provider {value!r} must be one of {', '.join(ONNX_PROVIDERS)}"
    )


_PROVIDER_HELP = (
    "onnx execution provider: auto, cpu, cuda, coreml, directml. Requires "
    "--device onnx. A provider this machine does not carry is refused, not "
    "demoted to cpu"
)


def _checked_provider(args: argparse.Namespace) -> ONNXProvider | None:
    """The requested provider, or ``None`` when the flag was not passed.

    A provider names an onnxruntime backend, so it means nothing on the torch
    or coreml backends. Refused rather than dropped: a run that was asked for
    ``cuda`` and quietly answered on the torch CPU backend is a wrong number
    with a right label. Availability is the ONNX backend's question, and it
    raises for an explicit provider it cannot honour.
    """
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


def _execution_overrides(args: argparse.Namespace) -> ExecutionOverrides | None:
    """The execution knobs this command line names, or ``None`` for the build's own.

    Overrides, not a config: ``--cuda-graphs`` means "the shipping engine plus
    graphs". Passing a full ExecutionConfig here would reset the manifest's
    fp16 precision map to the dataclass default and silently benchmark a
    different engine than the one that ships.
    """
    provider = _checked_provider(args)
    graphs = bool(getattr(args, "cuda_graphs", False))
    compiled = bool(getattr(args, "compile", False))
    if provider is None and not graphs and not compiled:
        return None

    from typing import cast

    import loudkit

    from .config import Device, ExecutionOverrides

    if graphs or compiled:
        # These two are torch-side and name the device outright, so that
        # `--cuda-graphs` on a build whose default device is CPU is a request,
        # not a no-op. A bare `--provider` leaves the device to `load`.
        return ExecutionOverrides(
            device=cast("Device", args.device or loudkit.best_device()),
            cuda_graphs=graphs,
            compile_model=compiled,
            onnx_provider=provider,
        )
    return ExecutionOverrides(onnx_provider=provider)


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
    """Turn a bare import failure into an instruction.

    Two kinds of exception arrive here. Python's own — ``No module named
    'torch'`` — names the module and nothing a user can act on, so the extras
    table above turns it into a pip command. The other kind is raised by this
    package (`hub`, the transports, `enroll`) and already carries a written
    message naming the pip command *and* the alternative; that message is
    better than anything reconstructable from a module name, so it is passed
    through untouched rather than replaced with the generic sentence.

    ``exc.name`` is optional and is ``None`` on every raise that passes only a
    message, so nothing here may assume it: interpolating it produced ``the ''
    package``, which named nothing and sent the reader looking for a package
    with no name.
    """
    written = str(exc).strip()
    # `No module named 'x'` is the interpreter's, not a raiser's. Matched on
    # the prefix because that text is stable across every version this
    # supports, and a false match only costs the generic sentence.
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
            f"  pip install 'loudkit[{extra}]'"
        )
    return f"loudkit needs the '{name}' package for this, which is not installed."


_MAX_STDIN_BYTES = 1 << 20
"""How much `loudkit speak -` will read before giving up.

Not a security bound -- this is the user's own shell and their own pipe. It is
there because `sys.stdin.read()` on a file that was not meant for it consumes
the file before anything looks at it, and `speak` synthesises a *single window*
of about a sentence: piping a gigabyte into `loudkit speak -` reads it fully into memory before
the window check refuses it. A
megabyte is far past anything one window can hold and far short of anything
that hurts.
"""


def _use_utf8_output() -> None:
    """Make the streams this CLI prints to speak UTF-8, whatever the locale.

    Almost everything here is a name someone else chose: a voice key, a language
    tag, the passage being spoken back. Python encodes stdout with the locale's
    encoding, which is cp1252 on a Windows console and ASCII under ``LC_ALL=C``,
    and neither can represent the languages this ships voices for. Printing a
    Polish voice name then raises ``UnicodeEncodeError`` from inside `print` —
    the synthesis having already succeeded — and the traceback is about the
    report, not about the work.

    Reconfiguring is the same choice ``PYTHONUTF8=1`` makes, and the one Python
    itself is moving to; ``errors="replace"`` is the belt to its braces, because
    a listing that renders one name imperfectly still beats one that dies on it.

    Stdin is deliberately untouched: `speak` reads a passage from it, and
    re-encoding a stream someone is piping into is not this function's business.
    """
    for stream in (sys.stdout, sys.stderr):
        # Not every stdout is a TextIOWrapper. Test harnesses and embedders
        # substitute objects with no `reconfigure` at all, and a CLI that
        # insisted on one would fail on the way to doing its job.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # A detached or already-closed stream raises rather than reconfigures,
        # and that is not worth failing a command over: the worst case is the
        # encoding that cannot change at that point anyway.
        with contextlib.suppress(OSError, ValueError):
            reconfigure(encoding="utf-8", errors="replace")


def _read_stdin() -> str:
    text = sys.stdin.read(_MAX_STDIN_BYTES + 1)
    if len(text) > _MAX_STDIN_BYTES:
        raise SystemExit(
            f"stdin is over {_MAX_STDIN_BYTES} bytes. `speak` renders one window "
            "— a sentence or so — so this is almost certainly the wrong file. "
            "For a passage, use the library's long-form path or the server."
        )
    return text


def _is_bare_voice_name(value: str) -> bool:
    """True for ``kathleen``, false for anything addressed as a path.

    The same rule :func:`loudkit.hub.resolve_voice` applies to the names it
    accepts, restated here because two callers below have to agree with it: a
    name with a separator in it is a path the resolver would refuse, so it must
    keep reaching the path error that names the file.
    """
    return bool(value) and not value.startswith(".") and "/" not in value and "\\" not in value


def _voice_release(args: argparse.Namespace) -> str | None:
    """The release a bare ``--voice`` name resolves against, or ``None``.

    ``doctor`` and ``download`` both end by printing ``speak --checkpoint <ref>
    --voice <name>``, so that exact shape has to work whatever ``<ref>`` is: a
    repo id, an unpacked release directory, or the checkpoint file inside one —
    a release is a checkpoint beside ``voices/``, so the file's parent is the
    release. Anything else has no release to name and gets ``None``.
    """
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
    """``--checkpoint``, resolved to a file when ``--revision`` pins it.

    The three server subcommands hand their checkpoint to a transport, and no
    transport's ``serve`` takes a revision — so a repo id reaches
    :func:`loudkit.load` inside them with nothing pinning it, and the guide's
    "pin the revision in production" had no CLI spelling at all. Resolving here
    keeps the pin in the one place that parsed it.

    Without ``--revision`` the value is passed through untouched, so the
    unpinned path stays byte-for-byte what it was: the transport still receives
    the repo id and still resolves it itself.

    The fetch is asked for by the backend ``--device`` needs. Pinning used to
    imply the torch set, so ``--device onnx --revision`` downloaded a snapshot
    holding no graphs and handed the transport a torch checkpoint -- the pin
    turned a working unpinned command into a broken pinned one.
    """
    checkpoint = str(args.checkpoint)
    revision = getattr(args, "revision", None)
    if revision is None:
        return checkpoint
    from .hub import backend_for_device, is_repo_id, resolve_checkpoint

    if not is_repo_id(checkpoint):
        # A path names its bytes already, so there is nothing to pin. Passed
        # through rather than refused, because `loudkit.load` answers the same
        # pairing the same way and one flag must not mean two things.
        return checkpoint
    backend = backend_for_device(getattr(args, "device", None))
    return str(resolve_checkpoint(checkpoint, revision=revision, backend=backend))


def _play_hint(output: Path) -> str:
    """One command that plays ``output`` on this platform.

    A hint, not a dependency and not a subprocess: the first run ended at a
    filename, which is a poor place to leave someone who has just installed a
    speech synthesiser and cannot yet hear it. Named per platform because the
    answer differs and a command that is not there is worse than none —
    ``afplay`` ships with macOS, ``aplay`` with alsa-utils on every desktop
    Linux, and PowerShell hands the file to whatever is registered for it.
    """
    import platform

    system = platform.system()
    if system == "Darwin":
        return f"afplay {shlex.quote(str(output))}"
    if system == "Windows":
        return f"Start-Process {output}"
    return f"aplay {shlex.quote(str(output))}"


def _cmd_speak(args: argparse.Namespace) -> int:
    import loudkit

    from .hub import resolve_voice

    engine = loudkit.load(
        args.checkpoint,
        device=args.device,
        execution=_execution_overrides(args),
        revision=args.revision,
    )
    # A bare voice name resolves against the checkpoint's own release — the
    # exact form doctor and download print as the next step. A path keeps
    # working unchanged. The revision reaches the voice too: a pinned
    # checkpoint read by a voice from the moving default branch is pinned in
    # name only.
    release = _voice_release(args)
    if Path(args.voice).is_file():
        voice = loudkit.VoiceProfile.load(args.voice)
    elif release is not None and _is_bare_voice_name(str(args.voice)):
        voice = loudkit.VoiceProfile.load(
            resolve_voice(str(args.voice), repo=release, revision=args.revision)
        )
    else:
        voice = loudkit.VoiceProfile.load(args.voice)
    text = args.text if args.text != "-" else _read_stdin()

    print(engine.describe(), file=sys.stderr)
    from .errors import WindowOverflowError

    try:
        result = engine.synthesize(
            text, voice, seed=args.seed, language=args.language, speed=args.speed
        )
    except WindowOverflowError:
        # The one interface most people try first should not refuse a
        # paragraph. Split at sentence boundaries and say so — the long-form
        # path derives per-chunk seeds, so the same seed reads a long text
        # differently than it reads its first sentence alone, and that is
        # worth a line rather than a surprise.
        print(
            "text is longer than one window; splitting at sentence boundaries "
            "(synthesize_long)",
            file=sys.stderr,
        )
        result = engine.synthesize_long(
            text, voice, seed=args.seed, language=args.language, speed=args.speed
        )
    # The library has taken `include_provenance=False` since the manifest was
    # added; the CLI could not say it, so the one interface most people use had
    # no way to write a plain WAV. Provenance stays the default — a marked file
    # is the point — but "I am piping this into something that chokes on a
    # trailing box" is a real answer and now expressible.
    result.save(args.output, include_provenance=not args.no_provenance)

    print(
        f"{result.duration:.2f}s of audio -> {args.output}  "
        f"({result.timings.describe(result.duration)})\n"
        f"hear it: {_play_hint(args.output)}",
        file=sys.stderr,
    )
    if result.hit_token_cap:
        print(
            "warning: generation stopped at the token cap rather than at a stop "
            "token, so the reading is probably truncated",
            file=sys.stderr,
        )
    return 0


_CLONE_AUDIO_SUFFIXES = (".wav", ".flac")
"""What ``clone`` reads. Two containers, both lossless, both local.

Narrow on purpose for 0.1. A recording is the one input a clone cannot be
better than, and a command that accepts a lossy file quietly makes the
enrollment worse than the person who ran it can see. Anything else converts in
one ``ffmpeg`` call, or reaches :func:`loudkit.enroll` as samples from Python.
"""

_ENROLLMENT_TENSOR_PREFIXES = ("s3gen.speaker_encoder.", "s3gen.tokenizer.")
"""The two tensor groups a clone reads: the speaker encoder and the speech
tokenizer.

They are about 40% of the packed file and synthesis never touches them, so a
release ships them as their own artefact, :data:`loudkit.hub.ENROLLMENT_NAME`
beside the synthesis file, and a synthesis-only release publishes the synthesis
file alone. A checkpoint packed before the split carries both. `doctor` reads
the names from the safetensors header of whichever artefact the hub resolves,
to say whether a local release can clone at all.
"""


def _clone_output(args: argparse.Namespace) -> Path:
    """Where ``clone`` writes.

    ``voices/<name>.safetensors`` by default: the layout ``serve --voices``
    and ``mcp --voices`` already read, so a first clone lands where the rest of
    the toolkit looks for it without a second flag.
    """
    if args.output is not None:
        return Path(args.output)
    return Path("voices") / f"{args.name}.safetensors"


def _require_clone_extras(checkpoint: str) -> None:
    """Refuse before any work, naming every extra this clone needs.

    Enrollment needs ``enroll``; a repo id needs ``hub`` on top of it to
    resolve the checkpoint and the voice encoder. Asked together and up front
    because discovering the second one after the first install costs a download
    of both halves to find out.

    The message is written rather than reconstructed from a module name, so
    :func:`_explain_missing` passes it through: one command that satisfies the
    whole invocation beats two that each satisfy half of it.
    """
    import importlib.util

    from .hub import is_repo_id

    needed = [("torch", "enroll"), ("torchaudio", "enroll"), ("librosa", "enroll")]
    extras = "enroll"
    if is_repo_id(checkpoint):
        needed.append(("huggingface_hub", "hub"))
        extras = "enroll,hub"
    missing = [module for module, _ in needed if importlib.util.find_spec(module) is None]
    if missing:
        raise ModuleNotFoundError(
            f"cloning needs {', '.join(missing)}, which this environment does not have.\n"
            f"  pip install 'loudkit[{extras}]'"
        )


def _save_voice_atomically(profile: VoiceProfile, output: Path) -> Path:
    """Write ``profile`` beside ``output`` and move it onto it.

    The same shape :func:`loudkit.provenance.write_wav` uses, for the same
    reason: a run that dies halfway leaves the previous file or no file, never
    a truncated profile that ``VoiceProfile.load`` refuses much later with a
    complaint about shapes. Same directory, so ``os.replace`` stays inside one
    filesystem and is the atomic rename it promises.

    On POSIX, ``VoiceProfile.save`` chmods the file it writes to ``0600`` and
    ``os.replace`` carries the mode across, so the profile is owner-only from
    the first moment it exists under its own name. On Windows the temporary and
    final files inherit the destination directory's ACL; Unix mode bits cannot
    express that policy there.

    ``mkstemp`` creates the temp file, so the name comes from the OS and is
    unique against every other caller, in this process and any other. A name
    built here cannot be: a pid separates two processes but not two threads of
    one, and two threads racing to the same output would then share a name,
    delete each other's half-written file in the cleanup below, and let the
    later ``os.replace`` publish the other's bytes as its own. ``mkstemp`` also
    opens at ``0600``, so the profile is never briefly world-readable under its
    temp name either.
    """
    import os
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


def _obtain_enrollment(checkpoint: str, revision: str | None) -> Path:
    """The artefact holding the tensors a clone reads, fetched if it is remote.

    A release is two files: ``loudr-1.safetensors`` carries synthesis, and
    ``loudr-1-enrollment.safetensors`` carries the speaker encoder and the
    speech tokenizer. Which file that is, and where it comes from, is the hub's
    question, so the CLI asks
    :func:`loudkit.hub.resolve_enrollment_checkpoint` rather than joining a
    name onto a directory: a repo id, an unpacked release, a bare checkpoint
    path and a pre-split file that answers for itself all resolve the one way.

    Asked before the enrollment rather than during it, for the reason the
    output check above is: a synthesis-only release cannot clone at all, and
    finding that out after ~10 s of model loading, or after a download, wastes
    all of it.

    Raises:
        FileNotFoundError: the release is synthesis-only. The hub writes that
            sentence; the CLI prints it and stops.
    """
    from .hub import resolve_enrollment_checkpoint

    return resolve_enrollment_checkpoint(checkpoint, revision=revision)


def _cmd_clone(args: argparse.Namespace) -> int:
    """Enroll a voice from a recording and write the profile.

    A front end over :func:`loudkit.enroll`, not a second enrollment path. The
    command validates its own arguments, decides where to write and writes
    safely; everything between the recording and the profile is the library's.

    **Consent is yours to obtain**, and it is not a technical question — see
    ``RESPONSIBLE_USE.md``.
    """
    import loudkit

    audio = Path(args.audio)
    if audio.suffix.lower() not in _CLONE_AUDIO_SUFFIXES:
        raise ValueError(
            f"{audio.name}: clone reads a WAV or a FLAC file. Convert it first, "
            "or pass samples to loudkit.enroll() from Python."
        )
    # The name is written into a filename and read back as a voice key, so it
    # has to satisfy both. Refused rather than sanitised: a name silently
    # rewritten is a voice the user cannot find again by the name they chose.
    if not _is_bare_voice_name(args.name):
        raise ValueError(
            f"--name {args.name!r} becomes a filename and a voice key, so it may "
            "not hold a path separator or start with a dot."
        )

    checkpoint = str(args.checkpoint)
    _require_clone_extras(checkpoint)

    output = _clone_output(args)
    # Checked before the enrollment, not after: the work is ~10 s of model
    # loading and inference, and finding out at the end that the answer has
    # nowhere to go wastes all of it.
    if output.exists() and not args.force:
        print(
            f"{output} exists. Pass --force to overwrite it, or -o to write elsewhere.",
            file=sys.stderr,
        )
        return 1

    enrollment = _obtain_enrollment(checkpoint, args.revision)
    print(f"enrollment tensors: {enrollment}", file=sys.stderr)

    print(f"enrolling {audio} ...", file=sys.stderr)
    profile = loudkit.enroll(
        str(audio),
        checkpoint,
        name=args.name,
        language=args.language,
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
        "consent is yours to obtain — see RESPONSIBLE_USE.md",
        file=sys.stderr,
    )
    return 0


def _loudkit_kind(path: Path) -> str | None:
    """``"checkpoint"``, ``"voice"``, or ``None`` for a file that is neither.

    Read from the safetensors header, which is the only thing that can answer
    it: both artefacts are ``.safetensors``, and a checkpoint declares itself
    with a ``manifest`` metadata key while a profile declares itself with
    ``voice``. Header only — no tensor is read, so this costs the same on a
    747 MB checkpoint as on a 300 kB voice.

    Any failure means "not ours": a truncated file, a directory, a file
    someone else's tool wrote. ``doctor`` diagnoses, so a file it cannot read
    is a file it does not mention.
    """
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            meta = f.metadata() or {}
    except Exception:  # noqa: BLE001 — every failure means the same thing here
        return None
    if "manifest" in meta:
        return "checkpoint"
    if "voice" in meta:
        return "voice"
    return None


def _is_enrollment(path: Path) -> bool:
    """Whether ``path`` declares itself a release's enrollment artefact.

    Read from the embedded manifest, so a renamed file still answers. `doctor`
    asks because the enrollment artefact is not a checkpoint to report cloning
    for: it is the *answer* for the checkpoint beside it, and a line of its own
    would read as a second release that cannot speak.
    """
    try:
        import json

        from safetensors import safe_open

        from .hub import ENROLLMENT_ROLE

        with safe_open(str(path), framework="numpy") as f:
            meta = f.metadata() or {}
        manifest = json.loads(meta.get("manifest", "{}"))
    except Exception:  # noqa: BLE001 — every failure means the same thing here
        return False
    return bool(isinstance(manifest, dict) and manifest.get("artifact_role") == ENROLLMENT_ROLE)


def _holds_loudkit_release(entry: Path) -> bool:
    """Whether a hub cache entry holds a loudkit checkpoint.

    The cache is shared with every other library on the machine, so listing it
    wholesale answered "which models has this user downloaded" — and `doctor`
    output is what people paste into bug reports. Each entry is judged by its
    own snapshots: a repo qualifies when a top-level ``.safetensors`` in one of
    them carries a loudkit manifest, which is the same question `load` asks.
    """
    from .hub import CHECKPOINT_GLOB

    return any(
        _loudkit_kind(path) == "checkpoint"
        for snapshot in (entry / "snapshots").glob("*")
        for path in snapshot.glob(CHECKPOINT_GLOB)
    )


def _carries_enrollment_tensors(path: Path) -> bool:
    """Whether one file holds both tensor groups a clone reads.

    Read from the safetensors header, like :func:`_loudkit_kind`: the tensor
    *names* answer it, and no tensor is loaded, so this costs the same on a
    747 MB checkpoint as on a stub. Any failure means "cannot enroll from
    this", which is what `doctor` prints.
    """
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            names = list(f.keys())
    except Exception:  # noqa: BLE001 — every failure means the same thing here
        return False
    return all(
        any(name.startswith(prefix) for name in names) for prefix in _ENROLLMENT_TENSOR_PREFIXES
    )


def _enrollment_tensor_file(checkpoint: Path) -> Path | None:
    """Which file beside ``checkpoint`` a clone would read, or ``None``.

    Two answers are a yes, and `doctor` prints which one it got: the enrollment
    artefact beside the checkpoint for a release built after the split, or the
    checkpoint itself for one packed before it, which is what most people have
    on disk. Asked of the hub, so `doctor` answers by the same rules a clone
    will follow rather than by a name it made up.

    The tensor names are then checked rather than the resolved file trusted,
    because `doctor` reads a directory it was pointed at, not a verified
    release: a file of the right name holding the wrong tensors is exactly the
    case that would otherwise be found by the enroller, ten seconds in.
    """
    from .hub import resolve_enrollment_checkpoint

    try:
        holder = resolve_enrollment_checkpoint(str(checkpoint))
    except FileNotFoundError:
        return None
    return holder if _carries_enrollment_tensors(holder) else None


def _report_cloning(found: list[tuple[Path, str]]) -> None:
    """The `doctor` section that answers "can this machine clone a voice".

    Three separate questions, reported separately because each has its own
    remedy: the extra is a pip command, the encoder is a file the release
    either ships or does not, and the enrollment tensors are ~40% of the
    weights that a synthesis-only release does not publish. Answering them as
    one "yes/no" sent people reinstalling extras to fix a checkpoint.

    The tensors live in one of two files, so the line names the one it found:
    the enrollment artefact beside the checkpoint, or the checkpoint itself for
    a release packed before the two were split.
    """
    import importlib.util

    from .hub import ENROLLMENT_NAME, VOICE_ENCODER_NAME

    missing = [
        module
        for module in ("torch", "torchaudio", "librosa")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        print(f"  extra      missing {', '.join(missing)} — pip install 'loudkit[enroll]'")
    else:
        print("  extra      installed   (torch, torchaudio, librosa)")

    # The enrollment artefact is not a checkpoint to report on, it is the
    # answer for the checkpoint beside it, so it is not a line of its own.
    checkpoints = [
        path
        for path, kind in found
        if kind == "checkpoint" and path.name != ENROLLMENT_NAME and not _is_enrollment(path)
    ]
    for path in checkpoints[:8]:
        encoder = path.parent / VOICE_ENCODER_NAME
        holder = _enrollment_tensor_file(path)
        if holder is None:
            where = f"none (looked in this file and in {ENROLLMENT_NAME})"
        elif holder == path:
            where = "in this file"
        else:
            where = holder.name
        state = str(encoder) if encoder.is_file() else f"missing ({encoder})"
        print(f"  {path}  enrollment tensors: {where}  encoder: {state}")
    if not checkpoints:
        print("  no checkpoint under the current directory to read for enrollment tensors")


def _cmd_doctor(  # noqa: PLR0912 — a checklist reads as a checklist, not a dispatch
    _args: argparse.Namespace,
) -> int:
    """What this machine can run, and the one command that fixes each gap.

    Exists because every question it answers otherwise arrives as an issue:
    "ModuleNotFoundError", "no CUDA", "where did the download go". Reads state,
    changes nothing, always exits 0 — a diagnosis is not a failure.
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
        print("  torch        missing — pip install 'loudkit[torch]'")
    for module, extra in (("onnxruntime", "onnx"), ("coremltools", "coreml")):
        if have(module):
            print(f"  {module} installed")
        else:
            print(f"  {module:<12} missing — pip install 'loudkit[{extra}]'")
    print()

    print("extras:")
    for module, extra, what in (
        ("soundfile", "audio", "writing WAVs"),
        ("librosa", "audio", "loading audio for enrollment"),
        ("huggingface_hub", "hub", "downloading by repo id"),
        ("fastapi", "server", "loudkit serve"),
        ("grpc", "grpc", "loudkit grpc"),
        ("mcp", "mcp", "loudkit mcp"),
    ):
        state = "installed" if have(module) else f"missing — pip install 'loudkit[{extra}]'"
        print(f"  {module:<16} {state}   ({what})")
    print()

    print("assets:")
    found = [
        (path, kind)
        for path in sorted(Path.cwd().glob("*.safetensors"))
        + sorted(Path.cwd().glob("*/*.safetensors"))
        if (kind := _loudkit_kind(path)) is not None
    ]
    for path, kind in found[:8]:
        print(f"  {path}  ({kind})")
    if not found:
        print("  no loudkit checkpoint or voice under the current directory")
    if have("huggingface_hub"):
        from huggingface_hub.constants import HF_HUB_CACHE

        hub_cache = Path(HF_HUB_CACHE)
        cached = sorted(hub_cache.glob("models--*")) if hub_cache.exists() else []
        ours = [entry for entry in cached if _holds_loudkit_release(entry)]
        for entry in ours:
            name = entry.name.removeprefix("models--").replace("--", "/")
            print(f"  cached: {name}")
        if not ours:
            print(f"  no loudkit release in the hub cache ({HF_HUB_CACHE})")
    print()

    print("cloning:")
    _report_cloning(found)
    print()
    print("to fetch a release:  loudkit download loudreader/loudr-1")
    print('to hear a voice:     loudkit speak --checkpoint <ref> --voice <name> "hello"')
    print(
        "to clone a voice:    loudkit clone me.wav --checkpoint <ref> --name mine --language en"
    )
    return 0


def _release_patterns(
    backend: str, *, cloning: bool = False
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(allow_patterns, ignore_patterns)`` for a selective release fetch.

    A delegation, not a table: the table lives in :func:`loudkit.hub.release_patterns`,
    the capability plan the resolver ``load()`` uses, so the CLI and the API
    cannot drift apart about what one backend needs.
    """
    from . import hub

    return hub.release_patterns(backend, cloning=cloning)


def _cmd_download(args: argparse.Namespace) -> int:
    """Fetch what one backend needs — and nothing more — from a release.

    ``--for torch`` (the default) is synthesis on the reference backend;
    ``--for onnx`` adds the exported graphs the ONNX backend and the Rust, Go
    and JS ports read; ``--for coreml`` adds the CoreML packages Python's
    coreml backend and Swift read. ``--with-cloning`` adds the enrollment
    pieces. Voices are never an axis: all twenty always come.
    """
    from . import hub

    allow, ignore = _release_patterns(args.backend, cloning=args.with_cloning)
    client = hub._hub()
    print(f"resolving {args.repo} ...", file=sys.stderr)
    try:
        root = Path(
            client.snapshot_download(
                repo_id=args.repo,
                revision=args.revision,
                allow_patterns=list(allow),
                ignore_patterns=list(ignore) or None,
                local_dir=str(args.local_dir) if args.local_dir else None,
            )
        )
    except Exception as exc:  # noqa: BLE001 — mapped by name below
        friendly = hub._friendly_hub_error(exc, args.repo, args.revision)
        if friendly is None:
            raise
        raise friendly from exc
    # The same verification `load()` runs: fetched files are hashed against the
    # release's own manifest before any path is printed as usable.
    hub._verify_sha256sums(root, repo=args.repo)
    # And the plan must have been *fulfilled*: `allow_patterns` fetches
    # whatever subset the repo holds, so a missing encoder, graph set or
    # voices directory is a failed download here — an error naming what is
    # absent, never a printed path to a directory that does not exist.
    hub.verify_release_inventory(
        root, args.backend, cloning=args.with_cloning, require_voices=True
    )

    checkpoint = hub.resolve_checkpoint(str(root))
    print(f"checkpoint  {checkpoint}")
    if args.with_cloning and args.backend == "torch":
        # Only torch fetches these two, because only the torch enroller reads
        # them. Printing them for a graph backend named files the plan
        # deliberately left behind, which is a path to nothing.
        print(f"encoder     {root / hub.VOICE_ENCODER_NAME}")
        print(f"enrollment  {root / hub.ENROLLMENT_NAME}")
    elif args.with_cloning:
        print(f"enrollment  {args.backend} graphs in {root / args.backend}")
    if args.backend != "torch":
        assets = root / args.backend
        print(f"{args.backend:<7}     {assets}")
    voice_count = sum(1 for p in (root / "voices").glob("*.safetensors") if p.is_file())
    print(f"voices      {voice_count} in {root / 'voices'}")
    # The hint names what was just fetched: the local directory when one was
    # asked for, the repo id otherwise (which resolves to the same cache copy).
    ref = str(args.local_dir) if args.local_dir else args.repo
    print(
        f"\ndone. speak with:\n  loudkit speak --checkpoint {ref} "
        f'--voice <name from `loudkit voices {ref}`> "hello"',
        file=sys.stderr,
    )
    return 0


def _cmd_voices(args: argparse.Namespace) -> int:
    """List the voices a release holds, without downloading anything."""
    from .hub import list_voices

    names = list_voices(repo=args.repo, revision=args.revision)
    for name in names:
        print(name)
    if not names:
        print(f"{args.repo}: no voices", file=sys.stderr)
        return 1
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Say what a file is and whether its own claims hold."""
    path = Path(args.path)
    data = path.open("rb").read(12)
    if data[:4] == b"RIFF":
        return _verify_wav(path)
    return _verify_safetensors(path)


def _verify_wav(path: Path) -> int:
    from .provenance import read_provenance, verify_provenance

    manifest = read_provenance(path)
    if manifest is None:
        print(f"{path}: a WAV with no provenance manifest")
        return 1
    manifest, ok = verify_provenance(path)
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
        print(f"{path}: provenance verified — the audio is the audio the manifest signs for")
        print(
            "  (integrity, not authenticity: the manifest is unsigned, so this "
            "proves the file is intact, not who made it)"
        )
        return 0
    print(f"{path}: provenance FAILED — the audio does not match its manifest")
    return 1


def _verify_safetensors(path: Path) -> int:
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata() or {}

    if "voice" in meta:
        from .checkpoint import file_sha256
        from .voice import VoiceProfile

        profile = VoiceProfile.load(path)  # runs every shape and range check
        print(f"{path}: a voice profile, and a valid one")
        print(f"  name: {profile.name}")
        print(f"  language: {profile.language}")
        print(f"  enrolment: {profile.enrolment}")
        print(f"  sha256: {file_sha256(path)}")
        return 0

    if "manifest" in meta:
        from .checkpoint import file_sha256, read_manifest

        manifest = read_manifest(path)  # refuses junk and unknown versions
        print(f"{path}: a loudkit checkpoint")
        print(f"  name: {manifest.get('name', path.stem)}")
        print(f"  recipe: {manifest.get('recipe_version', '?')}")
        print(f"  sha256: {file_sha256(path)}  (compare against the release's SHA256SUMS)")
        recorded = manifest.get("tensor_payload_sha256")
        if not recorded:
            print("  payload digest: not recorded by this checkpoint")
            return 0
        print("  hashing the tensor payload (streams every tensor, takes a moment) ...")
        actual = _payload_sha256(path)
        if actual == recorded:
            print("  payload digest: verified — the tensors are the tensors it names")
            return 0
        print(f"  payload digest: MISMATCH\n    recorded: {recorded}\n    actual:   {actual}")
        return 1

    print(f"{path}: a safetensors file, but neither a loudkit checkpoint nor a voice profile")
    print(f"  metadata keys: {sorted(meta) or 'none'}")
    return 1


def _payload_sha256(path: Path) -> str:
    """sha256 over (name, dtype, shape, raw bytes) in sorted key order.

    The same recipe that writes ``tensor_payload_sha256`` in a packed
    checkpoint, so the two digests are comparable. The reader maps the stored
    safetensors dtype to the recipe's PyTorch spelling without importing torch.
    """
    from .checkpoint import _tensor_payload_sha256

    return _tensor_payload_sha256(path)


def _cmd_describe(args: argparse.Namespace) -> int:
    """Print the resolved configuration and exit.

    This subcommand exists because the defect that shaped this library survived
    an entire optimisation campaign for want of exactly this output: nothing
    ever said which mode was active.
    """
    import loudkit

    engine = loudkit.load(
        args.checkpoint,
        device=args.device,
        execution=_execution_overrides(args),
        revision=args.revision,
    )
    print(engine.describe())
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import os

    from .transports.http import serve

    serve(
        _pinned_checkpoint(args),
        voices=args.voices,
        host=args.host,
        port=args.port,
        device=args.device,
        allow_public=args.allow_public,
        # The environment fallback keeps the token out of `ps` and out of
        # shell history; the flag wins when both are given.
        token=args.token or os.environ.get("LOUDKIT_TOKEN"),
        first_chunk_tokens=getattr(args, "first_chunk_tokens", None),
    )
    return 0


def _load_engine_and_voice(args: argparse.Namespace) -> tuple[object, object]:
    """Load the engine and voice named by ``--checkpoint`` / ``--voice``.

    Load time is measured here and stashed on the namespace, so ``bench`` and
    ``profile`` can report it without re-measuring a second load.
    """
    import loudkit

    execution = _execution_overrides(args)
    t0 = time.perf_counter()
    engine = loudkit.load(
        args.checkpoint, device=args.device, execution=execution, revision=args.revision
    )
    args._load_s = time.perf_counter() - t0
    voice = loudkit.VoiceProfile.load(args.voice)
    return engine, voice


def _cmd_bench(args: argparse.Namespace) -> int:
    from . import bench

    engine, voice = _load_engine_and_voice(args)
    texts = [t.strip() for t in (args.texts or list(bench.DEFAULT_TEXTS))]
    result = bench.bench(
        engine,
        voice,
        texts=texts,
        seed=args.seed,
        load_s=getattr(args, "_load_s", 0.0),
        command=_bench_command(args),
    )
    print(bench.render_table(result))
    if args.json:
        # Explicit UTF-8: a benchmark carries its passages, and a passage is in
        # whatever language it was written in. Text mode defaults to the
        # locale's encoding, which on a Windows console is cp1252 and cannot
        # represent most of the nine languages this ships voices for.
        Path(args.json).write_text(bench.to_json(result) + "\n", encoding="utf-8")
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


def _bench_command(args: argparse.Namespace) -> str:
    """The exact command a reader can re-run to reproduce this benchmark.

    Loudkit's leaderboard rule: every number carries the command that
    reproduced it, or it is marketing.
    """
    base = ["loudkit", "bench", "--checkpoint", "<checkpoint>", "--voice", "<voice>"]
    if args.device:
        base += ["--device", args.device]
    if getattr(args, "provider", None):
        base += ["--provider", args.provider]
    if getattr(args, "cuda_graphs", False):
        base += ["--cuda-graphs"]
    if getattr(args, "compile", False):
        base += ["--compile"]
    if args.texts:
        # Each text as its own argument, quoted. Joined with ", " this printed a
        # command that re-parsed as ONE passage, so the "reproducible" line
        # benchmarked something other than the run it was printed for.
        base += ["--texts", *(shlex.quote(t) for t in args.texts)]
    base += ["--seed", str(args.seed)]
    return " ".join(base)


def _cmd_profile(args: argparse.Namespace) -> int:
    from . import profile

    engine, voice = _load_engine_and_voice(args)
    result = profile.profile_passage(
        engine,
        voice,
        args.text,
        seed=args.seed,
        runs=args.runs,
        load_s=getattr(args, "_load_s", 0.0),
        command=_profile_command(args),
    )
    print(profile.render_table(result))
    if args.json:
        Path(args.json).write_text(profile.to_json(result) + "\n", encoding="utf-8")
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


def _profile_command(args: argparse.Namespace) -> str:
    base = ["loudkit", "profile", "--checkpoint", "<checkpoint>", "--voice", "<voice>"]
    if args.device:
        base += ["--device", args.device]
    if getattr(args, "provider", None):
        base += ["--provider", args.provider]
    if getattr(args, "cuda_graphs", False):
        base += ["--cuda-graphs"]
    if getattr(args, "compile", False):
        base += ["--compile"]
    if args.runs != 5:
        base += ["--runs", str(args.runs)]
    base += ["--seed", str(args.seed), "--", args.text]
    return " ".join(base)


def _cmd_grpc(args: argparse.Namespace) -> int:
    from .transports.grpc import serve

    serve(
        _pinned_checkpoint(args),
        args.voices,
        device=args.device,
        host=args.host,
        port=args.port,
        first_chunk_tokens=getattr(args, "first_chunk_tokens", None),
    )
    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    from .transports.mcp import run_stdio

    # Handed over the way the other two transports get it. The MCP transport
    # used to keep a "checkpoint is a file" contract, so a helper here resolved
    # a repo id on its behalf; it resolves every shape itself now, and the
    # helper had become a second local resolution of an already-resolved path.
    run_stdio(_pinned_checkpoint(args), args.voices, device=args.device)
    return 0


_ADVERTISED = (
    "speak",
    "clone",
    "voices",
    "download",
    "serve",
    "verify",
    "doctor",
    "grpc",
)
"""The top-level commands, and the order `loudkit --help` lists them in.

Eight, and eight is the budget. Everything a stranger needs between a bare
machine and a cloned voice speaking is here; a ninth has to displace one of
these rather than join them.
"""


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915 — flat argparse declarations
    """The full CLI grammar, exposed so tests can assert on it without running
    any command. ``main`` is a thin shell over this."""
    parser = argparse.ArgumentParser(
        prog="loudkit", description="Text to speech that behaves the same everywhere."
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="re-raise errors with their traceback instead of one diagnosis line",
    )
    # The advertised surface, in the order `--help` should read them.
    #
    # `describe`, `bench`, `profile` and `mcp` stay registered and stay
    # runnable — they are repo tools and a preview transport, and anyone who
    # knows their names keeps them — but they are not peers of these eight and
    # do not appear beside them. Two mechanisms are needed, because argparse
    # prints the subcommand set twice: a subparser added *without* `help=` is
    # left out of the listing (`add_parser` only builds a pseudo-action when
    # `help` is in its kwargs), while the usage line comes from this metavar,
    # which otherwise joins every registered choice.
    sub = parser.add_subparsers(dest="command", metavar=f"{{{','.join(_ADVERTISED)}}}")

    common = argparse.ArgumentParser(add_help=False)
    # str, not Path: this value may be a repo id, and `hub.is_repo_id` asks a
    # string questions a `PosixPath` cannot answer — `--checkpoint
    # loudreader/loudr-1` reached it as a Path and died with
    # "'PosixPath' object has no attribute 'startswith'", which is the README's
    # own first command. Every consumer below takes a str or a str | Path.
    common.add_argument(
        "--checkpoint", required=True, help="the synthesis .safetensors, or a repo id"
    )
    # On `common`, not on one subcommand at a time: `--checkpoint` is what
    # takes a repo id, so every subcommand that carries it can be handed a
    # moving default branch, and the guide tells production to pin. Declared
    # beside the flag it qualifies so the two cannot drift apart.
    common.add_argument(
        "--revision",
        default=None,
        metavar="REF",
        help="commit, tag or branch to pin --checkpoint to; without it a repo "
        "id follows the default branch, which moves. Ignored for a path, "
        "which names its bytes already.",
    )
    common.add_argument(
        "--device",
        default=None,
        type=_device_arg,
        metavar="DEVICE",
        help="backend: cpu, cuda, cuda:<index>, mps, coreml, onnx "
        "(default: the fastest available)",
    )

    speak = sub.add_parser("speak", parents=[common], help="synthesise to a WAV")
    speak.add_argument("text", help="text to speak, or '-' to read stdin")
    speak.add_argument("--voice", required=True, type=Path, help="voice profile")
    speak.add_argument("-o", "--output", default="out.wav", type=Path)
    speak.add_argument("--seed", type=int, default=0, help="same seed, same audio")
    speak.add_argument(
        "--language",
        default=None,
        help="language id for the text frontend; omitted means the voice's own "
        "language (a profile that carries none reads as 'en'). Pass one to read "
        "text in a language the voice was not enrolled in.",
    )
    speak.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="X",
        help="playback speed, 0.5 to 2.0, pitch preserved (default: 1.0, which "
        "is an exact bypass)",
    )
    speak.add_argument(
        "--no-provenance",
        action="store_true",
        help="write a plain WAV, without the C2PA claim-only manifest",
    )
    speak.add_argument(
        "--provider",
        default=None,
        type=_provider_arg,
        metavar="NAME",
        help=_PROVIDER_HELP,
    )
    speak.set_defaults(func=_cmd_speak)

    clone = sub.add_parser(
        "clone",
        help="clone a voice from a recording and write the profile",
    )
    clone.add_argument(
        "audio",
        help="a local WAV or FLAC recording: 5 to 10 seconds of clean, "
        "single-speaker audio. More than 30 seconds is refused",
    )
    clone.add_argument(
        "--checkpoint",
        required=True,
        help="the synthesis .safetensors, or a repo id. It must be a cloning-capable "
        "release: the enrollment artefact and ve.safetensors ride with it",
    )
    clone.add_argument(
        "--name",
        required=True,
        help="what to call the voice. Carried in the profile, and the default filename",
    )
    clone.add_argument(
        "--language",
        required=True,
        help="the language this voice speaks, e.g. en, pl. Written into the "
        "profile, and what the engine reads text as when a call names none",
    )
    clone.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        metavar="OUTPUT",
        help="where to write the profile (default: voices/<name>.safetensors)",
    )
    clone.add_argument(
        "--revision",
        default=None,
        metavar="REF",
        help="commit, tag or branch to pin --checkpoint to; without it a repo "
        "id follows the default branch, which moves",
    )
    clone.add_argument(
        "--device",
        default=None,
        type=_clone_device_arg,
        metavar="DEVICE",
        help="torch device for the enrollment models: cpu, cuda, cuda:<index>, "
        "mps (default: cpu, which is enough)",
    )
    clone.add_argument(
        "--force",
        action="store_true",
        help="overwrite the output if it is already there",
    )
    clone.set_defaults(func=_cmd_clone)

    voices = sub.add_parser("voices", help="list the voices a release holds, one name per line")
    voices.add_argument("repo", help="repo id, or a local release directory")
    voices.add_argument("--revision", default=None, help="commit, tag or branch")
    voices.set_defaults(func=_cmd_voices)

    download = sub.add_parser(
        "download",
        help="fetch what one backend needs — checkpoint, graphs, voices",
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
        help="also fetch what this backend enrols with: for torch the "
        "enrollment checkpoint and ve.safetensors, for onnx and coreml their "
        "three enrollment graphs",
    )
    download.add_argument(
        "--revision",
        default=None,
        metavar="REF",
        help="commit, tag or branch to pin; without it the default branch moves",
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
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument(
        "--allow-public",
        action="store_true",
        help="bind a non-loopback host; requires a bearer token, generated if not given",
    )
    serve.add_argument(
        "--token",
        default=None,
        help="bearer token every request must carry (required for a public bind; "
        "falls back to $LOUDKIT_TOKEN so the secret stays out of the process list)",
    )
    serve.add_argument("--voices", type=Path, help="directory of voice profiles")
    serve.add_argument(
        "--first-chunk-tokens",
        type=int,
        default=None,
        metavar="N",
        help="cap the FIRST streamed chunk at N tokens so audio starts sooner "
        "(opt-in; re-fingerprints — see serve() docs)",
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
        help="report what this machine can run, and how to fix what it cannot",
    )
    doctor.set_defaults(func=_cmd_doctor)

    grpc = sub.add_parser(
        "grpc",
        parents=[common],
        help="serve loudkit over gRPC (typed schema, streaming backpressure)",
    )
    grpc.add_argument("--host", default="127.0.0.1")
    grpc.add_argument("--port", type=int, default=50051)
    grpc.add_argument("--voices", type=Path, help="directory of voice profiles")
    grpc.add_argument(
        "--first-chunk-tokens",
        type=int,
        default=None,
        metavar="N",
        help="cap the FIRST streamed chunk at N tokens so audio starts sooner "
        "(opt-in; re-fingerprints — see loudkit.transports.http.serve)",
    )
    # No `--token`: unlike `serve`, this transport has no authentication, so a
    # non-loopback bind is refused rather than made defensible. See
    # `grpc_server.serve`.
    grpc.set_defaults(func=_cmd_grpc)

    # Registered and runnable, not advertised: a repo tool, not one of the
    # eight. See _ADVERTISED.
    describe = sub.add_parser("describe", parents=[common])
    describe.add_argument(
        "--provider",
        default=None,
        type=_provider_arg,
        metavar="NAME",
        help=_PROVIDER_HELP,
    )
    describe.set_defaults(func=_cmd_describe)

    # Registered and runnable, not advertised: preview, not part of the 0.1
    # surface. See _ADVERTISED.
    mcp = sub.add_parser("mcp", parents=[common])
    mcp.add_argument("--voices", type=Path, help="directory of voice profiles")
    mcp.set_defaults(func=_cmd_mcp)

    # Registered and runnable, not advertised: a repo tool, not one of the
    # eight. See _ADVERTISED.
    bench = sub.add_parser("bench", parents=[common])
    bench.add_argument("--voice", required=True, type=Path, help="voice profile")
    bench.add_argument(
        "--texts",
        nargs="*",
        default=None,
        help="passages to synthesise; default: the shipped benchmark set",
    )
    bench.add_argument("--seed", type=int, default=7, help="seed for every sample")
    bench.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="capture the token-generator decode as a CUDA graph (static KV cache; CUDA only)",
    )
    bench.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the token-generator decode step",
    )
    bench.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the machine-readable row to this file",
    )
    bench.add_argument(
        "--provider",
        default=None,
        type=_provider_arg,
        metavar="NAME",
        help=_PROVIDER_HELP,
    )
    bench.set_defaults(func=_cmd_bench)

    # Registered and runnable, not advertised: a repo tool, not one of the
    # eight. See _ADVERTISED.
    profile = sub.add_parser("profile", parents=[common])
    profile.add_argument("text", help="the passage to profile")
    profile.add_argument("--voice", required=True, type=Path, help="voice profile")
    profile.add_argument("--runs", type=int, default=5, help="timed runs after warm-up")
    profile.add_argument("--seed", type=int, default=7)
    profile.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="capture the token-generator decode as a CUDA graph (static KV cache; CUDA only)",
    )
    profile.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the token-generator decode step",
    )
    profile.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the machine-readable profile to this file",
    )
    profile.add_argument(
        "--provider",
        default=None,
        type=_provider_arg,
        metavar="NAME",
        help=_PROVIDER_HELP,
    )
    profile.set_defaults(func=_cmd_profile)

    return parser


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 — argparse builders read better flat
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
    """The first argument naming a file that is not there, or ``None``.

    Checked here rather than left to whatever opens them first: a missing
    checkpoint surfacing as a ``FileNotFoundError`` with no filename attached
    tells the reader nothing at all.
    """
    from .hub import is_repo_id

    # A directory, checked the same way: a mistyped `--voices /typo` would
    # otherwise answer "voices: none", reading as a broken install rather than
    # a mistyped path.
    for label in ("voices",):
        path = getattr(args, label, None)
        if path is not None and not Path(path).is_dir():
            return label, str(path)

    # `clone` reads one recording, and it must exist before ten seconds of
    # model loading, not after: a FileNotFoundError from inside librosa names
    # the file but arrives at the end of the work.
    audio = getattr(args, "audio", None)
    if audio is not None and not Path(audio).is_file():
        return "audio", str(audio)

    for label in ("checkpoint", "voice"):
        path = getattr(args, label, None)
        # A repeatable flag holds a list, not a path (`download --voice` once
        # did, and crashed here with a TypeError traceback); the checks
        # below are for the flags that hold one path. Handing that list to
        # `Path()` crashed the command with a TypeError traceback.
        if not isinstance(path, str | Path):
            continue
        if Path(path).exists():
            continue
        # A repo id is not a missing file. `loudkit.load` and `loudkit.voice`
        # both resolve `org/name` against the hub, so this pre-check — which
        # runs before either of them is called — was refusing the exact spelling
        # the README puts in front of a new user:
        # `loudkit serve --checkpoint loudreader/loudr-1` answered "checkpoint
        # not found". The resolver owns that question; this check exists only to
        # give a *path* a filename in its error, so it steps aside for anything
        # the resolver would recognise.
        if is_repo_id(str(path)):
            continue
        # And a bare voice name is not a missing file either when `--checkpoint`
        # names the release it belongs to: `speak --checkpoint <ref> --voice
        # kathleen` is what doctor and download print as the next step, and
        # this check answering "voice not found: kathleen" before the resolver
        # ever ran made that printed command a dead end.
        if label == "voice" and _is_bare_voice_name(str(path)) and _voice_release(args):
            continue
        return label, str(path)
    return None


def _run(args: argparse.Namespace) -> int:
    """Dispatch to the chosen subcommand, turning each failure into a message.

    Split out of :func:`main` only so the argument checks above and this
    exception ladder are two readable pieces rather than one long function.
    """
    try:
        return int(args.func(args))
    except ModuleNotFoundError as exc:
        print(_explain_missing(exc), file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"not found: {exc.filename or exc}", file=sys.stderr)
        return 1
    # UnsupportedLanguageError first: it is a *subclass* of RuntimeError (via
    # NotImplementedError), so the broader clause below would swallow it and
    # label a supported-input question an error. The frontend raises it for a
    # language off the twelve-id roster, and the message names the twelve.
    except UnsupportedLanguageError as exc:
        if getattr(args, "debug", False):
            raise
        print(f"unsupported: {exc}", file=sys.stderr)
        return 1
    # Every *other* NotImplementedError is a defect in this build, not a
    # question about the input. Catching the builtin here printed "unsupported:
    # ..." for a stub method in a backend, which tells the user to change a
    # command that was never wrong and hides the bug from whoever could fix it.
    # Reported as what it is, with the traceback one flag away.
    except NotImplementedError as exc:
        if getattr(args, "debug", False):
            raise
        print(
            f"internal error: {type(exc).__name__}: {exc}\n"
            "This is a bug in loudkit, not in your command. Re-run with --debug "
            "for the traceback.",
            file=sys.stderr,
        )
        return 1
    # RuntimeError alongside ValueError: the backends raise both, and they are
    # the same kind of event to a user — "this configuration or these assets
    # will not run". Catching only ValueError let a backend's RuntimeError
    # escape as a multi-line interpreter traceback, which is noise in a
    # terminal and worse in one being read aloud by a screen reader. The
    # traceback is still available under --debug, because a diagnosis that is
    # unreachable is not a diagnosis.
    except (ValueError, RuntimeError) as exc:
        if getattr(args, "debug", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
