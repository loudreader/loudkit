"""Render the reading set with loudkit and with each comparator system.

One WAV per passage per system, then a pairs manifest for
``research/judge_pairwise.py``. Resumable: a passage whose WAV already exists is
skipped, so an interrupted render continues where it stopped and a single
system can be re-rendered without touching the others.

Every backend writes 24 kHz mono. Loudness is matched later, by the judge, on
the samples it is about to send, so a backend here should write its natural
output and not normalise.

The comparator set is the league loudkit actually plays in: open systems a
reader could run on the same laptop, on a CPU, with no account. Kokoro, Piper,
Kyutai's Pocket TTS and KittenTTS all qualify. ElevenLabs is a paid API rather
than a peer; its backend is kept here for anyone who wants the ceiling drawn in,
and it stays off unless a key and a voice are both configured.

Each system reads the passage by whatever long-text path its own authors
provide. Where a system has none, the passage is split at sentence boundaries
and the pieces are joined, because that is what its user would have to do. The
point is to measure each system at its best, not to score it on an interface it
never claimed to have.

**Voice choice is a confound, and there is no way to remove it.** A comparator
using a fixed provider voice against loudkit using a cloned reference is not a
clean comparison, because prosody is not fully separable from the voice
carrying it. Pick the most neutral available voice on both sides, keep the
choice in the output metadata, and say plainly in the write-up which side had
which.

Usage::

    python research/render_comparators.py --system loudkit --voice joe
    python research/render_comparators.py --system kokoro --voice af_heart
    python research/render_comparators.py --system pockettts --voice alba
    python research/render_comparators.py --system kitten --voice expr-voice-2-m
    python research/render_comparators.py --system piper --voice en_US-lessac-medium
    python research/render_comparators.py --manifest --system-b kokoro
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
READING_SET = REPO / "tests/data/judge/reading-en.json"
SAMPLE_RATE = 24000


def load_passages(path: Path, limit: int) -> list[dict]:
    passages = json.loads(path.read_text())["passages"]
    return passages[:limit] if limit else passages


def write_wav(path: Path, samples, rate: int) -> None:
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(".partial.wav")
    sf.write(str(scratch), np.asarray(samples, dtype="float32"), rate)
    # Atomic rename, so an interrupted render never leaves a half-written WAV
    # that the resume logic would then skip.
    scratch.replace(path)


def sentence_chunks(text: str, max_chars: int = 200) -> list[str]:
    """Split at sentence ends, for a backend with no long-text path of its own.

    Only used where the system cannot take a paragraph. Breaking at a full stop
    is the least audible seam available, and it is what the system's own user
    would have to do by hand.
    """
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces or [text.strip()]


def join(chunks: list, silence_seconds: float, rate: int):
    """Concatenate rendered pieces with a short gap where the seam falls."""
    import numpy as np

    gap = np.zeros(int(rate * silence_seconds), dtype="float32")
    parts: list = []
    for index, chunk in enumerate(chunks):
        if index:
            parts.append(gap)
        parts.append(np.asarray(chunk, dtype="float32").reshape(-1))
    return np.concatenate(parts)


# --- backends ----------------------------------------------------------------


def render_loudkit(passages: list[dict], out: Path, voice: str, seed: int, **_) -> None:
    import loudkit as lk

    engine = lk.load("loudreader/loudr-1")
    profile = lk.voice(voice, repo="loudreader/loudr-1")
    truncated = []
    for index, passage in enumerate(passages, start=1):
        target = out / f"{passage['id']}.wav"
        if target.is_file():
            continue
        result = engine.synthesize(passage["text"], profile, seed=seed)
        if result.hit_token_cap:
            # A capped render is cut off mid-passage. It is a real correctness
            # failure and belongs in the judge's input, not quietly re-rolled,
            # but it must be visible in the render log either way.
            truncated.append(passage["id"])
        write_wav(target, result.audio, result.sample_rate)
        if index % 25 == 0:
            print(f"  {index}/{len(passages)}", flush=True)
    if truncated:
        print(
            f"  ! {len(truncated)} passages hit the token cap: {', '.join(truncated[:10])}",
            file=sys.stderr,
            flush=True,
        )


def render_kokoro(passages: list[dict], out: Path, voice: str, **_) -> None:
    try:
        from kokoro import KPipeline
    except ModuleNotFoundError:
        raise SystemExit("kokoro is not installed: pip install kokoro soundfile") from None

    import numpy as np

    pipeline = KPipeline(lang_code="a")
    for index, passage in enumerate(passages, start=1):
        target = out / f"{passage['id']}.wav"
        if target.is_file():
            continue
        # Kokoro chunks internally and yields per chunk; the passage is one
        # utterance for our purposes, so the chunks are concatenated.
        chunks = [
            np.asarray(audio, dtype="float32")
            for _, _, audio in pipeline(passage["text"], voice=voice)
        ]
        write_wav(target, np.concatenate(chunks), 24000)
        if index % 25 == 0:
            print(f"  {index}/{len(passages)}", flush=True)


def render_pockettts(passages: list[dict], out: Path, voice: str, language: str, **_) -> None:
    try:
        from pocket_tts import TTSModel
    except ModuleNotFoundError:
        raise SystemExit("pocket-tts is not installed: pip install pocket-tts") from None

    # Kyutai's 100M CPU model. It takes a whole paragraph on its own, so no
    # sentence splitting here. `voice` is a catalogue name, a local wav, or an
    # hf:// path, which means it can also be given loudkit's own reference.
    model = TTSModel.load_model(language=language) if language else TTSModel.load_model()
    state = model.get_state_for_audio_prompt(voice)
    for index, passage in enumerate(passages, start=1):
        target = out / f"{passage['id']}.wav"
        if target.is_file():
            continue
        audio = model.generate_audio(state, passage["text"])
        samples = audio.numpy() if hasattr(audio, "numpy") else audio
        write_wav(target, samples, model.sample_rate)
        if index % 25 == 0:
            print(f"  {index}/{len(passages)}", flush=True)


def render_kitten(passages: list[dict], out: Path, voice: str, **_) -> None:
    try:
        from kittentts import KittenTTS
    except ModuleNotFoundError:
        raise SystemExit(
            "kittentts is not installed: pip install kittentts\n"
            "it also needs espeak-ng (brew install espeak-ng)"
        ) from None

    # A 25M ONNX model with no long-text path: it phonemises whatever it is
    # given in one pass. Paragraphs go in sentence by sentence.
    model = KittenTTS()
    for index, passage in enumerate(passages, start=1):
        target = out / f"{passage['id']}.wav"
        if target.is_file():
            continue
        rendered = [
            model.generate(text=piece, voice=voice, speed=1.0)
            for piece in sentence_chunks(passage["text"])
        ]
        write_wav(target, join(rendered, 0.12, 24000), 24000)
        if index % 25 == 0:
            print(f"  {index}/{len(passages)}", flush=True)


def render_piper(passages: list[dict], out: Path, voice: str, **_) -> None:
    try:
        from piper import PiperVoice
    except ModuleNotFoundError:
        raise SystemExit("piper is not installed: pip install piper-tts") from None

    import numpy as np

    model = Path(voice)
    if not model.is_file():
        raise SystemExit(
            f"piper voice model not found: {model}\n"
            "download one from https://huggingface.co/rhasspy/piper-voices "
            "and pass the .onnx path to --voice"
        )
    piper = PiperVoice.load(str(model))
    for index, passage in enumerate(passages, start=1):
        target = out / f"{passage['id']}.wav"
        if target.is_file():
            continue
        pcm = b"".join(chunk.audio_int16_bytes for chunk in piper.synthesize(passage["text"]))
        samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        write_wav(target, samples, piper.config.sample_rate)
        if index % 25 == 0:
            print(f"  {index}/{len(passages)}", flush=True)


def render_elevenlabs(passages: list[dict], out: Path, voice: str, **_) -> None:
    import os
    import urllib.request

    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise SystemExit("set ELEVENLABS_API_KEY to render the ElevenLabs comparator")

    import numpy as np

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_24000"
    for index, passage in enumerate(passages, start=1):
        target = out / f"{passage['id']}.wav"
        if target.is_file():
            continue
        body = json.dumps({"text": passage["text"], "model_id": "eleven_flash_v2_5"}).encode()
        request = urllib.request.Request(
            url, data=body, headers={"xi-api-key": key, "Content-Type": "application/json"}
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    pcm = response.read()
                break
            except Exception:  # rate limits are routine
                if attempt == 4:
                    raise
                time.sleep(5 * (2**attempt))
        samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        write_wav(target, samples, 24000)
        if index % 25 == 0:
            print(f"  {index}/{len(passages)}", flush=True)


Renderer = Callable[..., None]
"""What every entry in BACKENDS is.

Declared rather than inferred: the table's values are functions with different
keyword signatures, each system takes what it needs, so mypy widens the dict
to `object` and the call site below becomes "cannot call function of unknown
type". `...` is the honest parameter list here: the dispatch passes a fixed set
and every renderer absorbs the rest with `**_`.
"""

BACKENDS: dict[str, Renderer] = {
    "loudkit": render_loudkit,
    "kokoro": render_kokoro,
    "pockettts": render_pockettts,
    "kitten": render_kitten,
    "piper": render_piper,
    "elevenlabs": render_elevenlabs,
}


# --- manifest ----------------------------------------------------------------


def build_manifest(
    passages: list[dict], audio_root: Path, system_a: str, system_b: str, out: Path
) -> None:
    rows, missing = [], 0
    for passage in passages:
        a = audio_root / system_a / f"{passage['id']}.wav"
        b = audio_root / system_b / f"{passage['id']}.wav"
        if not (a.is_file() and b.is_file()):
            missing += 1
            continue
        rows.append({"id": passage["id"], "text": passage["text"], "a": str(a), "b": str(b)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"wrote {out}: {len(rows)} pairs" + (f", {missing} incomplete" if missing else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reading-set", type=Path, default=READING_SET)
    parser.add_argument("--audio-root", type=Path, default=Path("out/judge/audio"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--system", choices=sorted(BACKENDS), help="render this system")
    parser.add_argument(
        "--label",
        default="",
        help="name this render is filed under; defaults to --system. "
        "Use it to compare two voices or two builds of one backend",
    )
    parser.add_argument("--voice", default="", help="voice name, id or model path")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--language",
        default="",
        help="backend language pack, where the backend has them "
        "(pockettts: english, italian_24l, ...)",
    )
    parser.add_argument("--manifest", action="store_true", help="emit a pairs manifest instead")
    parser.add_argument("--system-a", default="loudkit")
    parser.add_argument("--system-b", default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    passages = load_passages(args.reading_set, args.limit)

    if args.manifest:
        if not args.system_b:
            raise SystemExit("--manifest needs --system-b")
        out = args.out or Path(f"out/judge/{args.system_a}-vs-{args.system_b}.manifest.jsonl")
        build_manifest(passages, args.audio_root, args.system_a, args.system_b, out)
        return 0

    if not args.system:
        raise SystemExit("pass --system to render, or --manifest to pair up existing audio")
    if not args.voice:
        raise SystemExit(f"--voice is required for {args.system}")

    label = args.label or args.system
    out = args.audio_root / label
    out.mkdir(parents=True, exist_ok=True)
    done = sum(1 for p in passages if (out / f"{p['id']}.wav").is_file())
    print(f"{label}: {len(passages)} passages, {done} already rendered", flush=True)

    started = time.time()
    BACKENDS[args.system](
        passages, out, voice=args.voice, seed=args.seed, language=args.language
    )
    (out / "render.json").write_text(
        json.dumps(
            {
                "label": label,
                "system": args.system,
                "voice": args.voice,
                "seed": args.seed,
                "language": args.language,
                "reading_set": str(args.reading_set),
                "passages": len(passages),
            },
            indent=1,
        )
        + "\n"
    )
    print(f"{label}: done in {time.time() - started:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
