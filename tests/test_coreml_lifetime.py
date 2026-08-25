"""The CoreML backend must not kill the process it succeeded in.

CoreML holds the input feature values of a finished predict until its MLE5
execution stream stops lingering, then resets the stream on
``com.apple.coreml.MLE5ExecutionStream.resetQueue`` about a second later.
coremltools wraps input arrays without copying and keeps the ``py::array`` as an
Objective-C ivar, so that reset is where the Python references are released, on
a dispatch thread holding no GIL. When the interpreter's own reference is gone
by then, the release reaches ``_PyObject_Free`` and the host segfaults roughly a
second after a synthesis that returned correct audio. See
``loudkit.backends.coreml_backend._PinnedInputs`` for the mechanism and the fix,
and apple/coremltools#2827 for the upstream report.

**A test that only checks the synthesis succeeds cannot see this.** The audio is
right; the process dies afterwards, and inside pytest that death takes the whole
run with it, reported as a crashed worker rather than as a failing test. So the
synthesis runs in a child process and this asserts on the child's *exit status*
after it has outlived the linger window. The child churns small allocations
while it waits, because the fault needs pymalloc to be in use when the arena is
corrupted; sleeping alone is a weaker detector.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from .assets import asset, needs_module, requires, skip_or_fail

CKPT = asset("checkpoint")
REFERENCE = Path(__file__).parent / "data" / "reference"

LINGER_WAIT_SECONDS = 5.0
"""How long the child must outlive its last predict.

The reset was measured at about one second after the predict on macOS 26.1;
five is that with room for a loaded machine, and still short enough that this
test is a few seconds rather than a minute.
"""

CHILD_TIMEOUT_SECONDS = 600.0
"""Generous: the child loads the checkpoint and compiles nothing, but a cold
package open on a busy machine is not fast."""

_CHILD = textwrap.dedent(
    '''
    """Renders one reference window over CoreML, then proves it is still here."""
    import json
    import sys
    import time
    from pathlib import Path

    import loudkit
    from loudkit.voice import VoiceProfile

    checkpoint, reference, wait = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3])
    with open(reference / "meta.json", encoding="utf-8") as f:
        record = json.load(f)["0"]
    voice = VoiceProfile.load(reference / "testvoice.voice.safetensors")

    engine = loudkit.load(checkpoint, device="coreml")
    # synthesize_tokens rather than synthesize: it drives all three CoreML
    # packages (encoder, estimator, vocoder) without spending a token-generation
    # pass on a question that is entirely about the renderer's predict calls.
    result = engine.synthesize_tokens(record["speech_tokens"], voice, seed=record["seed"])
    print(f"SPOKE {len(result.audio)}", flush=True)

    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        # Allocation churn is the detector: a free performed without the GIL
        # corrupts the arenas an allocating main thread is walking.
        [object() for _ in range(2000)]
        time.sleep(0.01)
    print("SURVIVED", flush=True)
    '''
).strip()

pytestmark = [
    pytest.mark.slow,
    requires("checkpoint"),
]


def test_process_survives_the_coreml_linger_window(tmp_path: Path) -> None:
    needs_module("coremltools")
    if not REFERENCE.exists():
        # skip_or_fail rather than a plain skipif marker: under
        # LOUDKIT_REQUIRE_ASSETS a missing reference dump is a broken runner,
        # and a skip on the asset-backed job must not look like a pass.
        skip_or_fail(f"reference dumps not present at {REFERENCE}")
    from loudkit.backends.coreml_backend import _assets_dir
    from loudkit.checkpoint import Checkpoint

    try:
        _assets_dir(Checkpoint.open(str(CKPT)))
    except FileNotFoundError as exc:
        skip_or_fail(str(exc))

    script = tmp_path / "linger_probe.py"
    script.write_text(_CHILD, encoding="utf-8")
    child = subprocess.run(
        [
            sys.executable,
            "-u",
            str(script),
            str(CKPT),
            str(REFERENCE),
            str(LINGER_WAIT_SECONDS),
        ],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
        check=False,  # the exit status is the assertion, not an error to raise
    )

    # The order matters: a child that spoke and then died is the regression, and
    # its stdout looks like success up to the last line. Assert the status first
    # so the failure names the signal.
    assert child.returncode == 0, (
        f"the coreml backend killed its host {LINGER_WAIT_SECONDS}s after a "
        f"successful render (exit {child.returncode}"
        + (f", signal {-child.returncode}" if child.returncode < 0 else "")
        + "). CoreML released input feature values off-thread without the GIL; "
        "see loudkit.backends.coreml_backend._PinnedInputs.\n"
        f"stdout: {child.stdout}\nstderr: {child.stderr[-2000:]}"
    )
    assert child.stdout.splitlines()[-1] == "SURVIVED", child.stdout
