"""Generate the enrollment conformance fixture.

One directory (``tests/data/enrollment``) that every port's enrollment path is
held to. Enrollment is the one place in loudkit where *two* independent things
must both match the reference bit for bit: the DSP (resamplers, mels,
filterbanks) and the three models (S3 tokenizer, CAM++ x-vector, utterance
voice encoder). A wrong filterbank does not fail to build and does not throw —
it returns numbers, the model consumes them, a voice comes out, and it is
quietly worse with nothing to point at. So the fixture captures the boundary
of every stage, not just the final profile:

  ref_audio.f32          the input clip, float32 LE, 24 kHz
  wav16_flow.f32         torchaudio polyphase 24k->16k (the flow side's rate)
  wav16_t3.f32           librosa 24k->16k (the token-generator side's rate)
  tokenizer_mel.f32      (128, frames) log-mel the S3 tokenizer reads
  matcha_mel.f32         (80, frames)  the flow's conditioning mel
  kaldi_fbank.f32        (frames, 80)  CAM++'s input, mean-removed
  voiceenc_trimmed.f32   wav16_t3 after librosa.effects.trim(top_db=20)
  voiceenc_mel.f32       (frames, 40)  the utterance voice encoder's input
  prompt_tokens.i64      S3 tokenizer output, raw (before the prompt cut)
  cond_prompt_tokens.i64 S3 tokenizer output capped at 150 (the conditioning)
  flow_embedding.f32     (192,) CAM++ x-vector
  speaker_embedding.f32  (256,) utterance voice-encoder vector
  profile.safetensors    the assembled VoiceProfile — the thing the ports
                         must reproduce by enrolling ref_audio.f32

The two resamplers are captured as separate files on purpose. The reference
enrollment downsamples 24->16 through *two different* resamplers — torchaudio's
polyphase on the flow side and librosa's on the token-generator side — and the
shipped voices were built through exactly that split; unifying them flips ~8%
of prompt tokens (see ``models/enroll.py``). A port that resamples once and
reuses the result cannot be byte-parity with the shipped voices.

Numbers that do not fit a JSON double exactly are stored as raw little-endian
files with shapes in the manifest, so no port needs an npy parser — the same
rule as ``make_conformance.py``.

Usage (regenerate when the enrollment pipeline legitimately changes):
  .venv/bin/python tools/make_enrollment.py \
      --checkpoint /path/to/loudr-1.safetensors \
      --voice-encoder /path/to/ve.safetensors \
      --audio tests/data/enrollment/ref_audio.f32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from loudkit.backends.torch_backend import build_torch_enroller  # noqa: E402
from loudkit.models.enroll import _S3_SR, _matcha_mel  # noqa: E402
from loudkit.models.resample import resample  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "tests" / "data" / "enrollment"

_MEL_SR = 24_000
_MAX_REF_SECONDS = 10.0
_COND_SECONDS = 6.0


def _write_f32(path: Path, values: np.ndarray) -> None:
    np.asarray(values, dtype=np.float32).tofile(path)


def _write_i64(path: Path, values: np.ndarray) -> None:
    np.asarray(values, dtype=np.int64).tofile(path)


def _resample_flow(wav24: np.ndarray) -> np.ndarray:
    """The flow side's 24k->16k downsample — the one portable law."""
    return resample(wav24, _MEL_SR, _S3_SR)


def _resample_t3(wav24_full: np.ndarray) -> np.ndarray:
    """The token-generator side's 24k->16k downsample — the same law."""
    return resample(wav24_full, _MEL_SR, _S3_SR)


def _resample_to_24k(wav: np.ndarray, sr: int) -> np.ndarray:
    """A clip at another rate, brought to 24 kHz the way enroll() does."""
    return resample(wav, sr, _MEL_SR)


def main() -> None:  # noqa: PLR0915 — one stage per block, linear and logged
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--voice-encoder", required=True, help="ve.safetensors")
    ap.add_argument("--audio", required=True, help="reference clip (float32 LE or wav)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--name", default="fixture", help="voice name in the golden profile")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    audio_path = Path(args.audio)
    if audio_path.suffix == ".wav":
        import librosa

        # librosa.load normalises int16 to float32 [-1,1] and resamples to the
        # target rate — the same call the parity suite uses, so the fixture's
        # ref_audio matches the shipped voices' enrollment path exactly.
        wav, loaded_sr = librosa.load(audio_path, sr=_MEL_SR, mono=True)
        wav = np.asarray(wav, dtype=np.float32)
        # librosa reports the rate as a float; every consumer below wants Hz.
        sr = int(loaded_sr)
    else:
        wav = np.fromfile(audio_path, dtype=np.float32)
        sr = _MEL_SR
    if wav.ndim != 1:
        raise SystemExit(f"--audio must be mono 1-D, got shape {wav.shape}")

    enroller = build_torch_enroller(
        args.checkpoint, device="cpu", voice_encoder_weights=args.voice_encoder
    )

    # The golden profile, produced by the public entry point. Everything below
    # re-derives it stage by stage and must agree with it byte for byte —
    # that agreement is what makes the captured intermediates trustworthy.
    golden = enroller.enroll(wav, sr, name=args.name)

    wav24_full = wav if sr == _MEL_SR else _resample_to_24k(wav, sr)
    wav24 = wav24_full[: int(_MAX_REF_SECONDS * _MEL_SR)]
    wav16_flow = _resample_flow(wav24)
    wav16_t3 = _resample_t3(wav24_full)

    tok = enroller._tokenizer  # noqa: SLF001 — the fixture owns the pipeline
    spk = enroller._speaker_encoder  # noqa: SLF001
    ve = enroller._voice_encoder  # noqa: SLF001
    assert ve is not None

    import torch
    import torchaudio

    # DSP inputs, captured where they are produced so the ports can test each
    # filterbank in isolation rather than only the final voice.
    tokenizer_mel = tok._log_mel(torch.from_numpy(wav16_flow)).numpy()  # noqa: SLF001
    matcha_mel = _matcha_mel(torch.from_numpy(wav24)).numpy()

    kaldi = torchaudio.compliance.kaldi.fbank(
        torch.from_numpy(wav16_flow)[None], num_mel_bins=80
    )
    kaldi_fbank = (kaldi - kaldi.mean(dim=0, keepdim=True)).numpy()

    import librosa

    voiceenc_trimmed, _ = librosa.effects.trim(  # type: ignore[attr-defined]
        wav16_t3, top_db=20
    )
    voiceenc_mel = ve._mel(voiceenc_trimmed.astype(np.float32))  # noqa: SLF001

    # The tokenizer runs twice on *different* resamples — the fixture must
    # carry both mels, or the cond path has nothing to be checked against:
    # prompt_tokens come from the torchaudio-resampled clip, cond_tokens from
    # the librosa-resampled clip truncated to 6 s.
    tokenizer_mel_cond = tok._log_mel(  # noqa: SLF001
        torch.from_numpy(wav16_t3[: int(_COND_SECONDS * _S3_SR)])
    ).numpy()

    # Model outputs, raw.
    prompt_tokens_raw = tok.tokenize(torch.from_numpy(wav16_flow)).numpy()
    cond_tokens = tok.tokenize(
        torch.from_numpy(wav16_t3[: int(_COND_SECONDS * _S3_SR)]), max_tokens=150
    ).numpy()
    flow_embedding = spk.embed(torch.from_numpy(wav16_flow)).numpy()
    speaker_embedding = ve.embed(wav16_t3).numpy()

    # Re-assemble the profile exactly as enroll() does, and prove the two agree.
    prompt_mel = matcha_mel.copy()
    prompt_tokens = prompt_tokens_raw.copy()
    n_tok = min(len(prompt_tokens), prompt_mel.shape[1] // 2)
    prompt_tokens = prompt_tokens[:n_tok]
    prompt_mel = prompt_mel[:, : 2 * n_tok]

    for name, a, b in (
        ("speaker_embedding", golden.speaker_embedding, speaker_embedding),
        ("flow_embedding", golden.flow_embedding, flow_embedding),
        ("prompt_tokens", golden.prompt_tokens, prompt_tokens),
        ("prompt_mel", golden.prompt_mel, prompt_mel),
        ("cond_prompt_tokens", golden.cond_prompt_tokens, cond_tokens),
    ):
        if not np.array_equal(a, b):
            raise SystemExit(
                f"stage-by-stage re-derivation disagrees with enroll() on {name} "
                f"({a.shape} vs {b.shape}) — the captured intermediates would lie"
            )

    # Determinism: a second run must agree bit for bit, or the fixture cannot
    # be the arbiter of "same audio, same voice" across five implementations.
    again = enroller.enroll(wav, sr, name=args.name)
    for field in (
        "speaker_embedding",
        "flow_embedding",
        "prompt_tokens",
        "prompt_mel",
        "cond_prompt_tokens",
    ):
        if not np.array_equal(getattr(golden, field), getattr(again, field)):
            raise SystemExit(f"enrollment is not deterministic in {field}")

    _write_f32(out / "ref_audio.f32", wav)
    _write_f32(out / "wav16_flow.f32", wav16_flow)
    _write_f32(out / "wav16_t3.f32", wav16_t3)
    _write_f32(out / "tokenizer_mel.f32", tokenizer_mel)
    _write_f32(out / "tokenizer_mel_cond.f32", tokenizer_mel_cond)
    _write_f32(out / "matcha_mel.f32", matcha_mel)
    _write_f32(out / "kaldi_fbank.f32", kaldi_fbank)
    _write_f32(out / "voiceenc_trimmed.f32", voiceenc_trimmed)
    _write_f32(out / "voiceenc_mel.f32", voiceenc_mel)
    _write_i64(out / "prompt_tokens.i64", prompt_tokens)
    _write_i64(out / "cond_prompt_tokens.i64", cond_tokens)
    _write_f32(out / "flow_embedding.f32", flow_embedding)
    _write_f32(out / "speaker_embedding.f32", speaker_embedding)
    golden.save(out / "profile.safetensors")

    manifest = {
        "source_sample_rate": int(sr),
        "name": args.name,
        "files": {
            "ref_audio.f32": {"shape": list(wav.shape), "role": "input clip"},
            "wav16_flow.f32": {"shape": list(wav16_flow.shape), "role": "torchaudio 24k->16k"},
            "wav16_t3.f32": {"shape": list(wav16_t3.shape), "role": "librosa 24k->16k"},
            "tokenizer_mel.f32": {
                "shape": list(tokenizer_mel.shape),
                "role": "S3 tokenizer input",
            },
            "tokenizer_mel_cond.f32": {
                "shape": list(tokenizer_mel_cond.shape),
                "role": "S3 tokenizer input (cond path)",
            },
            "matcha_mel.f32": {
                "shape": list(matcha_mel.shape),
                "role": "flow conditioning mel",
            },
            "kaldi_fbank.f32": {"shape": list(kaldi_fbank.shape), "role": "CAM++ input"},
            "voiceenc_trimmed.f32": {
                "shape": list(voiceenc_trimmed.shape),
                "role": "trimmed 16k",
            },
            "voiceenc_mel.f32": {
                "shape": list(voiceenc_mel.shape),
                "role": "voice-encoder input",
            },
            "prompt_tokens.i64": {"shape": list(prompt_tokens.shape), "role": "prompt tokens"},
            "cond_prompt_tokens.i64": {
                "shape": list(cond_tokens.shape),
                "role": "cond tokens",
            },
            "flow_embedding.f32": {
                "shape": list(flow_embedding.shape),
                "role": "CAM++ x-vector",
            },
            "speaker_embedding.f32": {
                "shape": list(speaker_embedding.shape),
                "role": "utterance vec",
            },
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"enrollment fixture written to {out}")
    print(f"  profile: {golden}")


if __name__ == "__main__":
    main()
