"""Benchmark the engine on this machine, reproducibly.

The leaderboard's engine half. ``loudkit bench`` measures what this build and
this hardware actually do — load time, per-stage wall time, real-time factor,
peak resident memory, and whether the determinism promise holds on this device —
and prints a table plus an optional JSON dump. The other half of the leaderboard
is the machine it ran on, which is a matter of running this on each box and
collecting the JSON, not of anything clever in here.

Two rules, both learned the hard way elsewhere in this project:

* **Every number carries the command that reproduced it.** A benchmark without
  its invocation is marketing. The JSON records the full ``loudkit bench``
  command, the loudkit version, the device, the resolved algorithm and execution
  configs, and the fingerprint — enough that a row can be re-run verbatim.
* **Determinism is measured, not assumed.** The whole selling point is "same
  seed, same build, bit-identical waveform". A benchmark that did not check that
  on a given machine would advertise a promise not verified on that machine.

Peak resident memory needs no extra dependency and answers the "will this fit"
question directly. Three platforms report it three ways — ``ru_maxrss`` in bytes
on macOS and kilobytes on Linux, the psapi peak working set on Windows — so the
unit travels with the number in the JSON rather than pretending a cross-OS
figure is comparable.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

__all__ = ["bench", "run_bench", "RESULT", "BenchResult"]

RESULT = "loudkit bench"
"""The exact command shape a result was produced by, minus live paths.

Used as the reproducible-command anchor in the JSON: the caller substitutes the
checkpoint path. See :data:`RUN_CMD` in this module's use at the CLI.
"""

DEFAULT_TEXTS = (
    "Hello from my own machine.",
    "The quick brown fox jumps over the lazy dog and the reader keeps its composure.",
    "This is a longer passage, written to exercise more than a single window of "
    "speech tokens. It should run through several chunks and a couple of joins, "
    "so the streaming path and the long-form path are both measured, not just the "
    "shortest sentence that is fastest to type.",
)


_CANCEL_AFTER_POLLS = 4
"""Which ``should_cancel`` poll arms the barge-in being measured.

``should_cancel`` is consulted once before each chunk and then once per decode
step, so anything above one puts the interrupt inside the decode loop rather
than at the boundary before it — which is where a real barge-in lands, and the
only place the measurement means anything. Kept small so that even the shortest
benchmark passage reaches it; a passage that ends first reports no latency at
all rather than a flattering zero.

Arming and firing are deliberately one poll apart — see the measurement itself
for why the gap is the whole point.
"""


def _peak_rss() -> tuple[int, str]:
    """Peak resident memory so far, and the unit the OS reported it in.

    Three platforms, three answers, and the unit travels with the number
    because a cross-OS comparison of the bare figure would be nonsense:
    ``ru_maxrss`` is bytes on macOS and kilobytes on Linux, and Windows has no
    ``resource`` module at all.

    Windows matters here beyond tidiness: it is in the CI matrix, this module
    is imported at test collection, and a top-level ``import resource`` made
    every test on that runner disappear rather than one fail. The peak working
    set comes from psapi, which is where Windows keeps the same fact.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # Both signatures are declared, and on 64-bit Windows that is load
        # bearing rather than good manners. A HANDLE is pointer-sized; ctypes
        # defaults every unannotated function to `restype = c_int`, so
        # `GetCurrentProcess()` came back as a 32-bit -1 and was passed on as
        # 32 bits. psapi read a pointer-sized argument, found the top half
        # unset, and refused the call with ERROR_INVALID_HANDLE — every
        # `loudkit bench` on Windows x64 died in the last line of the run,
        # after the whole measurement had already been paid for.
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError(ctypes.GetLastError(), "GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize), "bytes"

    # Imported here rather than at module scope: `resource` does not exist on
    # Windows, and a top-level import made *every* test on that runner vanish
    # at collection instead of one failing.
    import resource

    unit = "bytes" if sys.platform == "darwin" else "kB"
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, unit


def _cpu_brand() -> str:
    """The CPU's marketing name, or the architecture if it will not say.

    ``platform.processor()`` returns "arm" on Apple silicon and "" on most
    Linux, neither of which distinguishes an M1 from an M3 Pro or an i7-6850K
    from anything else — which is the whole reason this exists.
    """
    import platform
    import subprocess

    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.machine() or "unknown"


def _host(device: str) -> str:
    """What this row was measured on, in one line.

    Recorded because a benchmark that does not say which machine produced it
    cannot be reproduced or trusted, and the omission bit: ``out/rows/`` and
    ``out/rows_s1/`` both held a row labelled ``device: cpu``, one an M3 Pro
    and one a 2016 i7, distinguishable only by which directory someone had
    filed it in. The two CUDA rows were worse — a 3090 and a 1080 Ti, telling
    themselves apart by ``cuda`` versus ``cuda:1``.

    The GPU name comes from torch when the row is a CUDA row, because that is
    the only place it exists; a CPU/MPS/ONNX row does not pay for the import.
    """
    import platform

    parts = [platform.node() or "unknown-host", platform.system(), _cpu_brand()]
    if device.split(":", 1)[0] == "cuda":
        try:
            import torch

            index = int(device.split(":", 1)[1]) if ":" in device else 0
            parts.append(torch.cuda.get_device_name(index))
        except Exception:  # noqa: BLE001 - a missing GPU name must not fail a run
            parts.append("cuda device name unavailable")
    return " | ".join(parts)


@dataclass(frozen=True, slots=True)
class Sample:
    """One measured synthesis of one text."""

    text: str
    duration_s: float
    n_tokens: int
    tokens_s: float
    mel_s: float
    audio_s: float
    total_s: float
    rtf: float
    hit_token_cap: bool
    ttfa_s: float = 0.0
    """Time to first audio: seconds from the start of synthesis to the first
    audio chunk becoming available (measured via the streaming path). This is
    the number an agent or robot actually feels — RTF says how fast the whole
    passage is, TTFA says how alive the speaker feels."""

    ttfa_warm_s: float | None = None
    """Median TTFA over repeated warm runs (after the first sample warms the
    kernels and caches). The first ``ttfa_s`` is the cold number and depends on
    order; the warm median is the number a long-lived server actually serves."""

    cancel_latency_s: float | None = None
    """How long a barge-in takes: from the interrupt arriving to the generator
    stopping.

    The interrupt is armed between polls, so this includes waiting for the
    decode loop to come round again — one forward pass, which is the worst case
    and the one an agent budgets for. It does **not** include the audio already
    handed to the caller: cancelling stops generation, it does not un-deliver
    chunks that were already yielded. See ``docs/design/barge-in.md`` for the half of
    the contract that belongs to the consumer."""


@dataclass(frozen=True, slots=True)
class BenchResult:
    """Everything a benchmark produced, in plain data for the JSON dump."""

    loudkit_version: str
    command: str
    device: str
    host: str
    """Machine the row was measured on — hostname, OS, CPU, and the GPU name
    for a CUDA row. Without it ``device: cpu`` names an abstraction, not a
    computer, and two rows from different machines look interchangeable."""
    execution: str
    algorithm: str
    fingerprint: str
    load_s: float
    sample_rate: int
    peak_rss: int
    rss_unit: str
    deterministic: bool
    samples: list[Sample] = field(default_factory=list)


def run_bench(
    engine: Any,
    voice: Any,
    *,
    texts: list[str],
    seed: int = 7,
    command: str = RESULT,
) -> BenchResult:
    """Run the benchmark against a warm engine and voice.

    Args:
        engine: a loaded :class:`~loudkit.engine.Engine`.
        voice: a :class:`~loudkit.voice.VoiceProfile`.
        texts: passages to synthesise, shortest first.
        seed: the seed for every sample. Same seed for all of them on purpose:
            determinism is checked by re-rendering the first sample and comparing
            bytes, which only means something when the bytes are expected to be
            equal.
        command: the exact command this run stands for, recorded in the JSON.
    """
    samples: list[Sample] = []
    for text in texts:
        # One streaming pass for both TTFA and the total: streaming is what a
        # voice agent uses, and a record that mixed TTFA from stream() with RTF
        # from a separate synthesize() would be joining two different runs
        # (different seeds, different chunking, maybe a cap). Here TTFA is the
        # time to the first chunk and RTF is the whole stream, from the same
        # pass over the same text.
        t0 = time.perf_counter()
        ttfa = 0.0
        parts: list[Any] = []
        for result in engine.stream(text, voice, seed=seed):
            if not parts:
                ttfa = time.perf_counter() - t0
            parts.append(result)
        total_s = time.perf_counter() - t0
        if not parts:  # pragma: no cover - nothing to speak
            ttfa = total_s
        duration = sum(p.duration for p in parts)
        n_tokens = sum(len(p.tokens) for p in parts)
        tokens_s = sum(p.timings.tokens for p in parts)
        mel_s = sum(p.timings.mel for p in parts)
        audio_s = sum(p.timings.audio for p in parts)
        # Warm TTFA: a second pass measures the steady-state first-chunk time
        # (kernels and caches are warm after the cold pass). Median of a few
        # runs, because a single warm sample is still order-dependent.
        warm_ttfas: list[float] = []
        for _ in range(3):
            tw = time.perf_counter()
            for i, _r in enumerate(engine.stream(text, voice, seed=seed)):
                if i == 0:
                    warm_ttfas.append(time.perf_counter() - tw)
                    break
        ttfa_warm = float(sorted(warm_ttfas)[len(warm_ttfas) // 2]) if warm_ttfas else None
        # Cancel latency (barge-in): how long from the interrupt landing to the
        # engine actually stopping. That is the number a voice agent lives by,
        # and it has to be measured *mid-generation*: a callback that returns
        # true on the very first poll is caught by the pre-chunk check before
        # any decoding starts, so it times an empty loop and always reports
        # ~0 s. Here the flag stays false for a while, so the interrupt arrives
        # while the decode loop is running, exactly as a real barge-in does.
        # Armed on one poll, honoured on the *next*: an interrupt does not
        # arrive at a polling instant, it arrives while a forward pass is
        # running, and the agent waits for that pass to finish before the loop
        # looks again. Flipping the flag inside the poll that then returns true
        # measured only the cost of breaking the loop — ~100 us, some 30x under
        # the real figure, and flattering in exactly the direction that matters.
        # Arming and returning false leaves one full poll interval inside the
        # measurement, which is the worst case a barge-in actually pays.
        armed_at: float | None = None
        polls = 0

        def cancel_mid_generation() -> bool:
            nonlocal armed_at, polls
            polls += 1
            if polls < _CANCEL_AFTER_POLLS:
                return False
            if armed_at is None:
                armed_at = time.perf_counter()
                return False
            return True

        # Consumed to the end rather than broken out of: the interrupt is what
        # stops this stream, and leaving early would stop it first and time
        # nothing. Cancellation guarantees termination, so this cannot hang.
        for _part in engine.stream(text, voice, seed=seed, should_cancel=cancel_mid_generation):
            pass
        # If the passage finished before the interrupt was armed, there was no
        # interrupt to time. Reported as absent rather than as a very small
        # number, which would read as an excellent result for a measurement
        # that never ran.
        cancel_latency = None if armed_at is None else time.perf_counter() - armed_at
        samples.append(
            Sample(
                text=text,
                duration_s=duration,
                n_tokens=n_tokens,
                tokens_s=tokens_s,
                mel_s=mel_s,
                audio_s=audio_s,
                total_s=total_s,
                rtf=duration / total_s if total_s > 0 else 0.0,
                hit_token_cap=any(p.hit_token_cap for p in parts),
                ttfa_s=ttfa,
                ttfa_warm_s=ttfa_warm,
                cancel_latency_s=cancel_latency,
            )
        )

    # Determinism: same seed again must give the same bytes. Compare audio via
    # the result rather than a file, so soundfile is not required for the check.
    first = samples[0]
    reference_audio = engine.synthesize(first.text, voice, seed=seed).audio
    again = engine.synthesize(first.text, voice, seed=seed).audio
    deterministic = bool((reference_audio == again).all())

    from . import __version__

    peak_rss, rss_unit = _peak_rss()
    return BenchResult(
        loudkit_version=__version__,
        command=command,
        device=engine.execution.device,
        host=_host(engine.execution.device),
        execution=engine.execution.describe(),
        algorithm=engine.algorithm.describe(),
        fingerprint=engine.algorithm.fingerprint(),
        load_s=0.0,  # set by bench(); loading is measured separately
        sample_rate=engine.algorithm.sample_rate,
        peak_rss=peak_rss,
        rss_unit=rss_unit,
        deterministic=deterministic,
        samples=samples,
    )


def bench(
    engine: Any,
    voice: Any,
    *,
    texts: list[str],
    seed: int = 7,
    load_s: float = 0.0,
    command: str = RESULT,
) -> BenchResult:
    """Like :func:`run_bench`, but records the model load time separately."""
    result = run_bench(engine, voice, texts=texts, seed=seed, command=command)
    # replace(), not a field-by-field rebuild: the rebuild had to name every
    # field, so any field added to BenchResult silently reverted to its default
    # on this path. `host` was the one that caught it.
    return replace(result, load_s=load_s)


def to_json(result: BenchResult) -> str:
    """Serialise a benchmark to the JSON that becomes the leaderboard row."""
    return json.dumps(asdict(result), indent=2, sort_keys=True)


def render_table(result: BenchResult) -> str:
    """Human-readable table from a :class:`BenchResult`."""
    header = (
        f"loudkit {result.loudkit_version}  {result.device}\n"
        f"  {result.algorithm}\n"
        f"  {result.execution}\n"
        f"  load {result.load_s:.2f}s  peak RSS {result.peak_rss:,} {result.rss_unit}  "
        f"deterministic: {result.deterministic}\n"
    )
    cols = (
        "sample",
        "audio(s)",
        "tokens",
        "tok(s)",
        "mel(s)",
        "aud(s)",
        "RTF",
        "TTFA",
        "TTFAwarm",
        "CancelLat",
    )
    lines = [header, "  ".join(f"{c:>10}" for c in cols)]
    for i, s in enumerate(result.samples, 1):
        cap = "  <cap>" if s.hit_token_cap else ""
        warm = f"{s.ttfa_warm_s:6.3f}" if s.ttfa_warm_s is not None else "     -"
        clat = f"{s.cancel_latency_s:7.4f}" if s.cancel_latency_s is not None else "       -"
        lines.append(
            f"{i:>10}  {s.duration_s:8.2f}  {s.n_tokens:6d}  {s.tokens_s:7.3f}  "
            f"{s.mel_s:7.3f}  {s.audio_s:7.3f}  {s.rtf:6.2f}x{cap}  "
            f"{s.ttfa_s:6.3f}  {warm}  {clat}"
        )
    lines.append("")
    lines.append(f"reproduce: {result.command}")
    return "\n".join(lines)
