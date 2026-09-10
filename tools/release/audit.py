"""The closing gate: load what the bundle ships, from its own paths, and speak.

``verify`` runs the gate for a profile. ``_closing_audit`` judges the staged
bundle again afterwards, because the gate ran the bundle's own code and is
the one step of a build that is not a copy. ``_verify_only`` is that audit
run in place on a finished bundle.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from . import CHECKPOINT_NAME, REPO, STRICT, TURBO_PROFILE, VOICE_ENCODER_NAME
from .verify import _sums_entries, check_bundle

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from collections.abc import Sequence
    from types import ModuleType

    from numpy.typing import NDArray

    # One enrollment backend: the fixture's inputs in, the three graph outputs
    # out. Here rather than at module scope because `np` is only imported for
    # type checking, so building this at runtime meant forward-reference
    # strings inside a `Callable`.
    _EnrollRunner: TypeAlias = Callable[
        [Path, dict[str, dict[str, NDArray[Any]]]], dict[str, NDArray[Any]]
    ]


def _verify_only(out: Path) -> int:
    """``--verify-only``: judge an assembled bundle in place, building nothing.

    The same :func:`check_bundle` the build runs after its closing gate, so an
    operator, or a pre-upload CI step, holds a bundle to exactly the standard
    the builder held it to before the rename. It does not load the model; the
    load-and-speak gate belongs to the build; this is about whether the bytes
    on disk are the bytes the manifests vouch for.
    """
    out = out.resolve()
    if not out.is_dir():
        print(f"FAILED: {out} is not a directory", file=sys.stderr)
        return 1
    print(f"checking {out}")
    problems = check_bundle(out)
    if problems:
        print(
            f"\nFAILED: {len(problems)} problem(s):\n  " + "\n  ".join(problems),
            file=sys.stderr,
        )
        return 1
    manifest = json.loads((out / "release.json").read_text(encoding="utf-8"))
    entries, _ = _sums_entries(out / "SHA256SUMS")
    print(
        f"  profile {manifest['profile']}, verified {manifest['verified']}, "
        f"{len(entries)} files checksummed and re-hashed from disk"
    )
    return 0


def _closing_audit(staging: Path) -> int:
    problems = check_bundle(staging)
    if problems:
        print(
            "\nFAILED: the assembled bundle does not survive its own audit:\n  "
            + "\n  ".join(problems),
            file=sys.stderr,
        )
        return 1
    print("  re-hashed from disk: the bundle is what the manifests say")
    return 0


# --------------------------------------------------------------- the closing gate


def verify(out: Path, *, profile: str = STRICT) -> int:
    """Load the assembled release from its own paths and speak.

    Deliberately uses only what is inside ``out``: a release that quietly
    depends on a file left over on the build machine passes every other check
    and fails for everyone else.

    ``full-0.1`` loads what it shipped, and nothing it did not: the
    checkpoint, ``ve.safetensors``, every voice on the roster, the ONNX
    graphs and CoreML packages. Checking torch alone is what let a
    bundle with no enrollment graphs pass: the torch path never opens them,
    so the piece most users receive was the piece nothing exercised.
    """
    print("\nverifying: loading the release as a stranger would")
    # A build machine with an export directory on either assets variable would
    # otherwise verify somebody else's graphs.
    os.environ.pop("LOUDKIT_ONNX_ASSETS", None)
    os.environ.pop("LOUDKIT_COREML_ASSETS", None)
    # `python/` is already on the path: the package puts it there at import.
    try:
        import loudkit
    except ImportError as exc:
        print(f"  cannot import loudkit: {exc}", file=sys.stderr)
        return 1

    strict = profile == STRICT
    turbo = profile == TURBO_PROFILE
    ckpt = _release_checkpoint(out, strict=strict)
    voices = sorted((out / "voices").glob("*.safetensors"))
    if ckpt is None:
        return 1
    if not voices:
        print("  no voice to speak with: cannot verify", file=sys.stderr)
        return 1

    # Each step returns 0 or 1 and prints its own reason. Listed rather than
    # chained so the strict half is one visible block: what a release is
    # checked for, beyond what any bundle is checked for.
    steps: list[Callable[[], int]] = [lambda: _verify_torch(loudkit, ckpt, voices[0])]
    if strict or turbo:
        steps += [
            lambda: _verify_voices(loudkit, voices),
            lambda: _verify_voice_encoder(loudkit, out, ckpt),
            lambda: _verify_onnx(loudkit, ckpt, voices[0]),
            lambda: _verify_coreml(ckpt, voices[0]),
        ]
    if turbo:
        # A turbo bundle is loudr-1's layout with one file swapped, so the way
        # to build the wrong one is to pass the wrong `--checkpoint` and have
        # every other check pass. This is the step that reads the weights back
        # and asks which model they are.
        steps.append(lambda: _verify_turbo_identity(ckpt))
    for step in steps:
        code = step()
        if code:
            return code
    return 0


def _release_checkpoint(out: Path, *, strict: bool) -> Path | None:
    """The bundle's checkpoint, by its canonical name or by ``release.json``.

    Under ``full-0.1`` the name is fixed, so the lookup is a constant. Under
    ``lenient`` and under ``turbo-0.1``, whose checkpoint is named
    :data:`TURBO_CHECKPOINT_NAME`, it is whatever ``release.json`` recorded,
    not "any .safetensors at the root", since ``ve.safetensors`` is one too.
    """
    if strict:
        return out / CHECKPOINT_NAME
    manifest = json.loads((out / "release.json").read_text(encoding="utf-8"))
    entry = manifest.get("checkpoint")
    if not isinstance(entry, dict):
        print("  release.json names no checkpoint", file=sys.stderr)
        return None
    return out / str(entry["path"])


def _verify_torch(loudkit: ModuleType, ckpt: Path, voice_path: Path) -> int:
    """Speak on the torch path, twice on one seed.

    Determinism is checked here rather than anywhere else because it is the
    property a caller is most likely to build on and least likely to test:
    the same seed and the same text give byte-identical audio, or the release
    does not keep the promise its documents make.
    """
    try:
        engine = loudkit.load(str(ckpt), device="cpu")
        voice = loudkit.VoiceProfile.load(voice_path)
        result = engine.synthesize("The release is assembled.", voice, seed=7)
        again = engine.synthesize("The release is assembled.", voice, seed=7)
    except Exception as exc:  # any failure here fails the release
        print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"  {engine.describe()}")
    print(f"  spoke {result.duration:.2f}s with {voice_path.stem}, {len(result.tokens)} tokens")
    if not (again.audio == result.audio).all():
        print("  FAILED: same seed produced different audio: determinism is broken")
        return 1
    print("  same seed, byte-identical audio")
    return 0


SPEAKER_EMBEDDING_WIDTH = 256
"""The utterance voice encoder's vector, what the token generator conditions on."""

FLOW_EMBEDDING_WIDTH = 192
"""The CAM++ x-vector, what the flow conditions on."""


def _verify_voices(loudkit: ModuleType, voices: Sequence[Path]) -> int:
    """Every shipped profile opens and carries the tensors a voice needs.

    Speaking with one voice opens one profile out of the whole roster. A truncated or mis-packed
    profile is a per-file defect, and a per-file defect needs a per-file check.
    """
    print(f"  voices: {len(voices)}")
    for path in voices:
        try:
            profile = loudkit.VoiceProfile.load(path)
        except Exception as exc:  # any failure here fails the release
            print(f"  FAILED to load {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if profile.speaker_embedding.shape != (SPEAKER_EMBEDDING_WIDTH,) or (
            profile.flow_embedding.shape != (FLOW_EMBEDDING_WIDTH,)
        ):
            print(
                f"  FAILED: {path.name} carries "
                f"speaker {profile.speaker_embedding.shape}, "
                f"flow {profile.flow_embedding.shape}",
                file=sys.stderr,
            )
            return 1
    print(f"    all {len(voices)} load, embeddings the right shape")
    return 0


def _verify_turbo_identity(ckpt: Path) -> int:
    """The bundle really carries the two-token model, and says so at every layer.

    Three statements that must agree: the manifest's ``decode.mode``, the
    ``format_version`` four other engines gate on, and the second head the
    loop needs. A loudr-1 checkpoint copied under the turbo name satisfies
    none of them and passes every other check in this file.
    """
    from loudkit.checkpoint import Checkpoint, decode_mode

    try:
        packed = Checkpoint.open(str(ckpt))
    except Exception as exc:  # any failure here fails the release
        print(f"  FAILED to read {ckpt.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    mode = decode_mode(packed.manifest)
    version = packed.manifest.get("format_version")
    heads = [n for n in packed.iter_names("t3.") if n.startswith(("t3.head2", "t3.fuse"))]
    if mode != "fusion_mtp2" or version != 2 or not heads:
        print(
            f"  FAILED: {ckpt.name} is not the turbo model: decode.mode {mode!r}, "
            f"format_version {version!r}, {len(heads)} fusion tensors. Pack it "
            f"with tools/pack_turbo.py, or build loudr-1 with --model loudr-1.",
            file=sys.stderr,
        )
        return 1
    print(
        f"  turbo: decode.mode {mode}, format_version {version}, "
        f"{len(heads)} fusion tensors, K={packed.manifest.get('n_cfm_timesteps')}"
    )
    return 0


def _verify_voice_encoder(loudkit: ModuleType, out: Path, ckpt: Path) -> int:
    """Clone a voice with the shipped ``ve.safetensors``.

    ``ve.safetensors`` was copied and checksummed and never opened, so a
    release could ship voice cloning that does not clone. This enrolls the
    fixture's reference recording through the bundle's own voice encoder and
    checks the two embeddings against the fixture, which is the same spec
    every port is held to.
    """
    import numpy as np

    fixture = _fixture()
    if fixture is None:
        return 1
    print("  voice encoder:")
    audio = np.frombuffer((fixture / "ref_audio.f32").read_bytes(), dtype=np.float32)
    try:
        cloned = loudkit.enroll(
            audio,
            str(ckpt),
            name="release-check",
            voice_encoder_weights=str(out / VOICE_ENCODER_NAME),
        )
    except Exception as exc:  # any failure here fails the release
        print(f"  FAILED to enroll: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for label, got, want_name in (
        ("speaker", cloned.speaker_embedding, "speaker_embedding.f32"),
        ("flow", cloned.flow_embedding, "flow_embedding.f32"),
    ):
        want = np.frombuffer((fixture / want_name).read_bytes(), dtype=np.float32)
        c = _cos(np.asarray(got, dtype=np.float32).ravel(), want)
        # `not (c > x)`, not `c <= x`: NaN compares False to everything, so
        # `c <= x` is the branch where a NaN *passes*. `_cos` refuses a NaN at
        # the source now; this is the second wall, because a gate that reads
        # backwards is the kind of line that gets copied.
        if not (c > 0.999):
            print(f"  FAILED: cloned {label} cosine {c:.6f} <= 0.999", file=sys.stderr)
            return 1
        print(f"    {label:8s} cosine {c:.6f}")
    return 0


def _verify_onnx(loudkit: ModuleType, ckpt: Path, voice_path: Path) -> int:
    """Speak on the synthesis graphs, then enroll on the three shared graphs."""
    print("  onnx path:")
    try:
        engine = loudkit.load(str(ckpt), device="onnx")
        voice = loudkit.VoiceProfile.load(voice_path)
        result = engine.synthesize("The release is assembled.", voice, seed=7)
    except Exception as exc:  # any failure here fails the release
        print(f"  FAILED on the onnx path: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"    {engine.describe()}")
    print(f"    spoke {result.duration:.2f}s, {len(result.tokens)} tokens")
    return _enrollment_gate("onnx", ckpt.parent / "onnx", _run_onnx_enrollment)


def _verify_coreml(ckpt: Path, voice_path: Path) -> int:
    """Speak and enroll using the shipped CoreML packages on macOS."""
    if sys.platform != "darwin":
        print("  coreml path: SKIPPED, not an Apple platform (sys.platform != 'darwin')")
        return 0
    print("  coreml path:")
    spoke = _speak_coreml(ckpt, voice_path)
    if spoke is None:
        return 1
    print(f"    {spoke['describe']}")
    print(f"    spoke {spoke['duration']:.2f}s, {spoke['tokens']} tokens")
    return _enrollment_gate("coreml", ckpt.parent / "coreml", _run_coreml_enrollment)


# Speaking on the coreml device can kill the process that did it, about a
# second after the last predict and from a thread that process does not own:
# CoreML lets its execution stream linger, then resets it on a dispatch queue
# (`-[MLE5ExecutionStream resetAfterLingering:]`), and the `MLFeatureValue`
# deallocs underneath freed the Python objects coremltools handed them,
# calling `_PyObject_Free` on a background thread without the GIL. The
# backend's `_PinnedInputs` fixes that by keeping one interpreter-owned buffer
# per input, so CoreML never holds the last reference (see
# `loudkit.backends.coreml_backend` and apple/coremltools#2827).
#
# The child below is the regression detector for that fix, so it must not
# escape through `os._exit(0)` after the predict: a clean, ordinary exit
# after the linger window is the property under test. So the child
#
#   - runs several create / synthesize / destroy cycles, because a single
#     render never reuses the pinned buffers and a single teardown never
#     follows a reuse;
#   - outlives the measured one-second linger by a five-second margin after
#     **every** render, with the engine still alive, because that is when the
#     reset queue frees the inputs of a finished predict. Both halves of that
#     were measured to matter on this fault: tearing the engine down first
#     deallocs the model on the main thread and hides it, and a second render
#     issued inside the first one's linger also hides it, so back-to-back
#     renders with one wait at the end detect nothing. The wait churns small
#     allocations, because the fault needs pymalloc in use when an off-thread
#     free corrupts it, and sleeping is a weaker detector;
#   - then drops the engine and collects, and churns a moment longer, because
#     releasing the model must tear the stream down here, on the main thread,
#     not on the reset queue;
#   - exits through the interpreter's normal shutdown.
_COREML_LINGER_SECONDS = 5.0
_COREML_CYCLES = 3
_COREML_TIMEOUT_SECONDS = 1800.0
"""Hard ceiling on each CoreML child process.

A first CoreML load compiles the packages, which can take minutes; a healthy
gate run is comfortably inside half an hour. Without a ceiling a child hung
inside a predict hangs the whole build with no verdict, which is worse than a
failure: nothing says the gate did not pass. A timeout is a refusal like any
other.
"""
_COREML_SPEAK = """
import gc, json, sys, time
sys.path.insert(0, sys.argv[1])
import loudkit
voice = loudkit.VoiceProfile.load(sys.argv[3])
cycles, linger = int(sys.argv[4]), float(sys.argv[5])

def churn(seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        [object() for _ in range(2000)]
        time.sleep(0.01)

for _ in range(cycles):
    engine = loudkit.load(sys.argv[2], device="coreml")
    for _ in range(2):  # the second render is the one that reuses the buffers
        result = engine.synthesize("The release is assembled.", voice, seed=7)
        print("SPOKE " + json.dumps({
            "describe": engine.describe(),
            "duration": result.duration,
            "tokens": len(result.tokens),
        }), flush=True)
        churn(linger)   # the engine is alive; the lingering stream resets now
    del engine, result  # teardown must happen here, on the main thread
    gc.collect()
    churn(1.0)
print("SURVIVED", flush=True)
"""


def _speak_coreml(ckpt: Path, voice_path: Path) -> dict[str, Any] | None:
    """Load the bundle on the coreml device and speak, in a child process.

    Same load and same call a caller makes, on the packages the bundle ships.
    The child process is not an escape hatch any more; it is the measurement.
    The pass condition is threefold, and the exit status comes first: a child
    that printed ``SPOKE`` and then took a signal is precisely the regression
    the backend's ``_PinnedInputs`` exists to prevent, and its stdout looks
    like success up to the last line. Correct audio from a process that then
    dies is a failing release, not a passing one.
    """
    import subprocess

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _COREML_SPEAK,
                str(REPO / "python"),
                str(ckpt),
                str(voice_path),
                str(_COREML_CYCLES),
                str(_COREML_LINGER_SECONDS),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=_COREML_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        # A hung child is a failing gate, not a stalled build: kill it (the
        # exception handler has already done so) and refuse with a verdict.
        print(
            f"  FAILED on the coreml path: the speak process was still running "
            f"after {_COREML_TIMEOUT_SECONDS:.0f}s and was killed",
            file=sys.stderr,
        )
        return None

    def fail(reason: str) -> None:
        print(
            f"  FAILED on the coreml path: {reason}\n"
            + "\n".join(f"    {line}" for line in proc.stderr.strip().splitlines()[-20:]),
            file=sys.stderr,
        )

    if proc.returncode != 0:
        # `returncode` is named because a signal shows up here as a negative
        # number and nothing else in the output would say so.
        signal = f" (signal {-proc.returncode})" if proc.returncode < 0 else ""
        fail(
            f"the speak process exited {proc.returncode}{signal} instead of "
            f"surviving {_COREML_LINGER_SECONDS:.0f}s past its last predict"
        )
        return None
    spoken = next(
        (line for line in reversed(proc.stdout.splitlines()) if line.startswith("SPOKE ")),
        None,
    )
    if spoken is None:
        fail("the speak process exited 0 without speaking")
        return None
    lines = proc.stdout.splitlines()
    if not lines or lines[-1] != "SURVIVED":
        fail("the speak process never reported SURVIVED after the linger window")
        return None
    result: dict[str, Any] = json.loads(spoken[len("SPOKE ") :])
    return result


# ----------------------------------------------------------- shared enrollment


def _fixture() -> Path | None:
    fixture = REPO / "tests" / "data" / "enrollment"
    if not (fixture / "manifest.json").is_file():
        print(
            f"  FAILED: the enrollment fixture is missing: {fixture}\n"
            "    rebuild it with tools/make_enrollment.py",
            file=sys.stderr,
        )
        return None
    return fixture


def _cos(a: NDArray[Any], b: NDArray[Any]) -> float:
    """Cosine similarity, refusing the inputs that make it meaningless.

    A zero vector gives `0/0` = NaN, and NaN silently passes every gate that
    reads `if c <= threshold`: `nan <= x` is `False`, and `False` is the
    branch where the check succeeds. A graph returning zeros is the standard
    mis-export failure, so the one input most likely to be wrong was the one
    the gate could not refuse: the bundle was stamped `"verified": true` with
    `cosine nan` printed beside it.

    Raises:
        ValueError: for a zero or non-finite vector, which is a broken graph
            rather than a low score, and has to be told apart from one.
    """
    import numpy as np

    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if not (np.isfinite(na) and np.isfinite(nb)) or na == 0.0 or nb == 0.0:
        raise ValueError(
            f"cosine is undefined for these vectors (norms {na!r}, {nb!r}). A "
            f"zero or non-finite output is a broken export, not a low score."
        )
    c = float(np.dot(a, b) / (na * nb))
    if not np.isfinite(c):
        raise ValueError(f"cosine came out {c!r}; the inputs are not comparable.")
    return c


def _partials(mel: NDArray[Any]) -> NDArray[Any]:
    """``_VoiceEncoder.embed``'s partial windows: stride 77, window 160.

    Zero-padded so the last one is full. Host orchestration in every port,
    which is why it is not inside the graph.
    """
    import numpy as np

    step, window = 77, 160
    n_wins, remainder = divmod(max(len(mel) - window + step, 0), step)
    if n_wins == 0 or (remainder + (window - step)) / window >= 0.8:
        n_wins += 1
    target = window + step * (n_wins - 1)
    if target > len(mel):
        mel = np.concatenate([mel, np.zeros((target - len(mel), mel.shape[1]), np.float32)])
    return np.stack([mel[i * step : i * step + window] for i in range(n_wins)]).astype(
        np.float32
    )


def _enrollment_feeds(fixture: Path) -> dict[str, dict[str, NDArray[Any]]]:
    """The three graphs' inputs, read from the fixture once.

    One enrollment, three graphs, one read: the three graphs are stages of one
    operation, and running them as one is the closer imitation of a caller.
    """
    import numpy as np

    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))

    def read(name: str, dtype: type) -> NDArray[Any]:
        shape = manifest["files"][name]["shape"]
        flat: NDArray[Any] = np.frombuffer((fixture / name).read_bytes(), dtype=dtype)
        return np.asarray(flat.reshape(shape))

    return {
        "s3_tokenizer": {"mel": read("tokenizer_mel.f32", np.float32)[None].astype(np.float32)},
        "camp": {"fbank": read("kaldi_fbank.f32", np.float32).T[None].astype(np.float32)},
        "voice_encoder": {"partials": _partials(read("voiceenc_mel.f32", np.float32))},
    }


def _enrollment_gate(label: str, assets: Path, runner: _EnrollRunner) -> int:
    """Run one enrollment through the three graphs and check it against the fixture.

    The fixture (``tests/data/enrollment``) is the spec every port is held to,
    so the graphs are checked against real DSP inputs and the shipped outputs:
    the tokenizer exactly, the two encoders to cosine > 0.9999. Present but
    broken is the failure this profile exists to prevent, and a graph nobody
    runs is indistinguishable from a graph that is not there.
    """
    import numpy as np

    fixture = _fixture()
    if fixture is None:
        return 1
    feeds = _enrollment_feeds(fixture)
    print(f"    enrollment graphs ({label}):")
    try:
        got = runner(assets, feeds)
    except Exception as exc:  # any failure here fails the release
        print(f"  FAILED ({label}): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))

    def read(name: str, dtype: type) -> NDArray[Any]:
        shape = manifest["files"][name]["shape"]
        flat: NDArray[Any] = np.frombuffer((fixture / name).read_bytes(), dtype=dtype)
        return np.asarray(flat.reshape(shape))

    want_tokens = read("prompt_tokens.i64", np.int64)
    if not np.array_equal(got["s3_tokenizer"].astype(np.int64).ravel(), want_tokens):
        print(
            f"  FAILED ({label}): s3_tokenizer does not reproduce the fixture tokens",
            file=sys.stderr,
        )
        return 1
    print(f"      s3_tokenizer   {len(want_tokens)} tokens, exact")

    camp = got["camp"].astype(np.float32).ravel()
    pooled = got["voice_encoder"].astype(np.float32).mean(0)
    pooled = pooled / np.linalg.norm(pooled)
    for graph, vector, want in (
        ("camp", camp, read("flow_embedding.f32", np.float32)),
        ("voice_encoder", pooled, read("speaker_embedding.f32", np.float32)),
    ):
        c = _cos(vector, want)
        if not (c > 0.9999):  # NaN-safe; see the cloning gate above
            print(f"  FAILED ({label}): {graph} cosine {c:.6f} <= 0.9999", file=sys.stderr)
            return 1
        print(f"      {graph:14s} cosine {c:.6f}")
    return 0


def _run_onnx_enrollment(
    assets: Path, feeds: dict[str, dict[str, NDArray[Any]]]
) -> dict[str, NDArray[Any]]:
    import numpy as np
    import onnxruntime as ort

    out: dict[str, NDArray[Any]] = {}
    for stem, feed in feeds.items():
        session = ort.InferenceSession(
            str(assets / f"{stem}.onnx"), providers=["CPUExecutionProvider"]
        )
        out[stem] = np.asarray(session.run(None, feed)[0])
    return out


# coremltools' in-process ``MLModel.predict`` segfaults on these graphs: the
# same crash ``docs/platforms/apple.md`` records for the T3 decode loop and
# ``tools/export_enroll_coreml.py`` works around the same way. One subprocess
# runs the whole enrollment, so the three packages still cost one process.
_COREML_ENROLL = """
import sys, numpy as np, coremltools as ct
assets, in_npz, out_npz = sys.argv[1:4]
feed = dict(np.load(in_npz, allow_pickle=False))
out = {}
for stem in ("s3_tokenizer", "camp", "voice_encoder"):
    # Swift's Enroller and the exporter both declare these graphs CPU-only.
    # The release gate must validate that shipped placement, not Core ML's
    # default ALL placement, which may try and fail to build an ANE plan.
    model = ct.models.MLModel(
        f"{assets}/{stem}.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY
    )
    inputs = {k.split(".", 1)[1]: v for k, v in feed.items() if k.startswith(stem + ".")}
    out[stem] = np.asarray(next(iter(model.predict(inputs).values())))
np.savez(out_npz, **out)
"""


def _run_coreml_enrollment(
    assets: Path, feeds: dict[str, dict[str, NDArray[Any]]]
) -> dict[str, NDArray[Any]]:
    import subprocess
    import tempfile

    import numpy as np

    # `dict[str, Any]`: numpy types `savez`'s second positional as `allow_pickle`,
    # so a `**` splat of concrete arrays does not type-check against it.
    flat: dict[str, Any] = {
        f"{stem}.{k}": v for stem, feed in feeds.items() for k, v in feed.items()
    }
    with tempfile.TemporaryDirectory() as tmp:
        in_npz, out_npz = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        np.savez(in_npz, **flat)
        # Same ceiling as the speak gate: a child hung inside a predict must
        # end the build with a verdict, not hold it open.
        subprocess.run(
            [sys.executable, "-c", _COREML_ENROLL, str(assets), str(in_npz), str(out_npz)],
            check=True,
            timeout=_COREML_TIMEOUT_SECONDS,
        )
        with np.load(out_npz) as z:
            got = {k: np.asarray(z[k]) for k in z.files}
    # The base-3 FSQ fold is on the host for the CoreML export: coremltools'
    # int64 output segfaults in-process, so `s3_tokenizer.mlpackage` returns
    # the 8 float dims and the port folds them. `tools/export_enroll_coreml.py`
    # does the same, and this mirrors it rather than inventing a second rule.
    powers: NDArray[Any] = 3 ** np.arange(8, dtype=np.float32)
    got["s3_tokenizer"] = (np.squeeze(got["s3_tokenizer"], 0) * powers).sum(-1).astype(np.int64)
    return got
