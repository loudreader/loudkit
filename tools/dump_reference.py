#!/usr/bin/env python3
"""Dump reference-side parity data from chatterbox-apple.

Runs INSIDE the chatterbox-apple venv (it needs the upstream `chatterbox`
package and the training artifacts), never inside loudkit's:

  cd ../chatterbox-apple && \
    .venv/bin/python ../loudkit/tools/dump_reference.py \
    --out /path/to/dumps --stage all

The two repositories are expected to be siblings; set LOUDKIT_REFERENCE_ROOT if
chatterbox-apple lives elsewhere. The checkpoint and the reference clip are
read from loudkit's own `assets/` directory (override with LOUDKIT_CHECKPOINT
and LOUDKIT_REFERENCE_WAV); only the upstream code comes from the sibling.

What it produces, per sentence:
  - text token ids (loudkit framing: bare ids, no START/STOP)
  - free-run speech tokens sampled with LR-SAMPLER-v1 + the production EOS
    floor — the same law loudkit runs, so token comparisons measure logits
  - teacher-forced speech logits on those tokens (packed checkpoint, T3 fp16)
  - mu / mel / waveform through the PRODUCTION algorithm: static windows
    (255 query / 238 prompt, silence-token 4254 padding), K=2 cosine Euler,
    single-path estimator (fp16), StaticHiFT with injected phase/noise

and once: the en_reader1 VoiceProfile in loudkit's format.

The randomness is loudkit's Philox noise (imported from loudkit's source
tree), injected as data — so the torch reference here and every loudkit
backend consume the identical prior/excitation and a waveform difference is
arithmetic, not RNG plumbing.

Provenance note: the *algorithm* implemented here is the shipped Swift
engine's (ChatterboxMelSynthesizer.swift), not `S3Token2Wav.flow_inference` —
the latter applies CFG on top of the CFG-distilled estimator (EXP-016) and a
linear time grid in its meanflow branch, neither of which ships.
"""

import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOUDKIT = os.path.dirname(_HERE)

# Both were absolute paths under one author's home directory, which is why this
# script ran on one computer and told everyone else where that computer keeps its
# files. `chatterbox-apple` is expected as a sibling of the loudkit checkout;
# override with LOUDKIT_REFERENCE_ROOT when it is somewhere else.
CB = os.environ.get(
    "LOUDKIT_REFERENCE_ROOT",
    os.path.join(os.path.dirname(_LOUDKIT), "chatterbox-apple"),
)
LOUDKIT_SRC = os.path.join(_LOUDKIT, "python")
sys.path.insert(0, os.path.join(CB, "tools"))
sys.path.insert(0, os.path.join(CB, "export"))
sys.path.insert(0, os.path.join(CB, "distill"))
sys.path.insert(0, LOUDKIT_SRC)

# The weights and the reference clip live in this repository's own assets/
# now (gitignored; same flat layout as the release bundle). Only the upstream
# *code* still needs a chatterbox-apple checkout, via the sys.path lines above.
_ASSETS = os.path.join(_LOUDKIT, "assets")
CKPT = os.environ.get("LOUDKIT_CHECKPOINT", os.path.join(_ASSETS, "loudr-1.safetensors"))
REF_VOICE = os.environ.get("LOUDKIT_REFERENCE_WAV", os.path.join(_ASSETS, "en_reader1.wav"))
SR = 24_000
UPS = 480
H = 9  # harmonic channels

# production static geometry (ChatterboxMelSynthesizer.swift)
P_TOK = 238
T_QUERY = 255
T_MEL = 2 * (P_TOK + T_QUERY)  # 986
HIFT_FRAMES = 510
PAD_SIL = 4254

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The lighthouse keeper climbed the narrow stairs slowly, pausing at each "
    "window to watch the storm gather over the bay.",
    "Could anyone really have known, before the results came in, that the "
    "entire experiment would need to be repeated?",
]
SEED_BASE = 4242


def derive(seed, stream):
    """loudkit's per-stage seed derivation (engine._derive)."""
    return (seed * 0x9E3779B97F4A7C15 + stream * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF


def build_voice(out_dir):
    """Enroll en_reader1 exactly as the reference does, save as a loudkit profile."""
    import librosa
    from distill_datagen import load_support

    from loudkit.voice import VoiceProfile

    ve, s3, _tok = load_support("cpu")
    wav, _ = librosa.load(REF_VOICE, sr=SR)
    r16 = librosa.resample(wav, orig_sr=SR, target_sr=16_000)
    with torch.inference_mode():
        cond_tokens, _ = s3.tokenizer.forward([r16[: 6 * 16_000]], max_len=150)
        spk = torch.from_numpy(ve.embeds_from_wavs([r16], sample_rate=16_000)).mean(
            0, keepdim=True
        )
        gref = s3.embed_ref(wav[: 10 * SR], SR, device="cpu")
    profile = VoiceProfile(
        name="en_reader1",
        speaker_embedding=spk[0].numpy().astype(np.float32),
        flow_embedding=gref["embedding"][0].numpy().astype(np.float32),
        prompt_tokens=gref["prompt_token"][0].numpy().astype(np.int64),
        prompt_mel=gref["prompt_feat"][0].numpy().astype(np.float32).T,
        cond_prompt_tokens=cond_tokens[0].numpy().astype(np.int64),
    )
    path = os.path.join(out_dir, "testvoice.voice.safetensors")
    profile.save(path)
    print(f"voice -> {path}  ({profile})")
    return profile


def text_ids(tok, text):
    """loudkit frontend semantics: no punc_norm, bare ids."""
    return tok.encode(text, language_id="en")


def free_run(t3, embeds, n_text, sil_ids, seed, max_new=255):
    """LR-SAMPLER-v1 free run with the production EOS floor, on the reference model."""
    from loudkit.config import SamplingConfig
    from loudkit.sampler import LRSamplerV1

    cfg = SamplingConfig(silence_token_ids=tuple(sil_ids))
    sampler = LRSamplerV1(cfg, seed=seed)
    floor = max(10, int(n_text * 1.2))
    stop = 6562
    dtype = next(t3.parameters()).dtype

    with torch.inference_mode():
        o = t3.tfmr(inputs_embeds=embeds, use_cache=True, past_key_values=None)
        past = o.past_key_values
        logits = t3.speech_head(o.last_hidden_state[:, -1]).float()[0].cpu().numpy()
        seen = np.zeros(8194, dtype=bool)
        out: list[int] = []
        for step in range(max_new):
            if len(out) < floor:
                logits[stop] = -np.inf
            token = sampler(logits, step=step, seen=seen)
            out.append(int(token))
            if token == stop:
                break
            seen[token] = True
            e = t3.speech_emb(torch.tensor([[token]])) + t3.speech_pos_emb.get_fixed_embedding(
                step + 1
            )
            o = t3.tfmr(inputs_embeds=e.to(dtype), use_cache=True, past_key_values=past)
            past = o.past_key_values
            logits = t3.speech_head(o.last_hidden_state[:, -1]).float()[0].cpu().numpy()
    return out


def prefill_embeds(t3, t3c, ids):
    import torch.nn.functional as F

    hp = t3.hp
    dtype = next(t3.parameters()).dtype
    tt = torch.tensor(ids, dtype=torch.long)[None]
    tt = F.pad(F.pad(tt, (1, 0), value=hp.start_text_token), (0, 1), value=hp.stop_text_token)
    emb, _ = t3.prepare_input_embeds(
        t3_cond=t3c,
        text_tokens=tt,
        speech_tokens=hp.start_speech_token * torch.ones_like(tt[:, :1]),
        cfg_weight=0.0,
    )
    return emb.to(dtype)


def teacher_forced(t3, embeds, forced):
    """Incremental teacher forcing (the reference implementation's shape)."""
    dtype = next(t3.parameters()).dtype
    with torch.inference_mode():
        o = t3.tfmr(inputs_embeds=embeds, use_cache=True, past_key_values=None)
        past = o.past_key_values
        rows = [t3.speech_head(o.last_hidden_state[:, -1]).float()[0].cpu().numpy()]
        for i, token in enumerate(forced):
            e = t3.speech_emb(
                torch.tensor([[int(token)]])
            ) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
            o = t3.tfmr(inputs_embeds=e.to(dtype), use_cache=True, past_key_values=past)
            past = o.past_key_values
            rows.append(t3.speech_head(o.last_hidden_state[:, -1]).float()[0].cpu().numpy())
    return np.stack(rows)


def loudkit_noise(seed, n_tokens):  # noqa: ARG001 - signature kept for symmetry with engine
    """The exact noise loudkit's renderer will draw for this seed."""
    from loudkit.models.noise import gaussian_field, symmetric_uniforms

    flow_seed = derive(seed, 1)
    voc_seed = derive(seed, 2)
    z = gaussian_field(flow_seed, 0, 80, T_MEL)[None]  # (1,80,986)
    phase = np.zeros((1, H, 1), dtype=np.float32)
    phase[0, 1:, 0] = symmetric_uniforms(voc_seed, 0, H - 1, math.pi)
    noise = gaussian_field(voc_seed, 1, H, HIFT_FRAMES * UPS)[None]  # (1,9,244800)
    return z, phase, noise


def render_production(s3, profile, tokens, z, phase, noise):
    """The shipped algorithm on the reference torch weights.

    Static windows with silence padding, K=2 cosine single-path Euler
    (estimator fp16 per the packed dtype map), StaticHiFT on the 510-frame
    window, output trimmed to 2N frames of audio.
    """
    import torch.nn.functional as F
    from hift_static import StaticHiFT

    n = min(len(tokens), T_QUERY)
    query = torch.full((1, T_QUERY), PAD_SIL, dtype=torch.long)
    query[0, :n] = torch.tensor(tokens[:n], dtype=torch.long)
    prompt = torch.full((1, P_TOK), PAD_SIL, dtype=torch.long)
    p_real = min(len(profile.prompt_tokens), P_TOK)
    prompt[0, :p_real] = torch.from_numpy(profile.prompt_tokens[:p_real])

    with torch.inference_mode():
        emb = F.normalize(torch.from_numpy(profile.flow_embedding)[None], dim=1)
        spks = s3.flow.spk_embed_affine_layer(emb)
        tok_all = torch.cat([prompt, query], dim=1)
        x = s3.flow.input_embedding(tok_all)
        h, _ = s3.flow.encoder(x, torch.tensor([tok_all.shape[1]]))
        mu = s3.flow.encoder_proj(h).transpose(1, 2).contiguous()

        cond = torch.zeros(1, 80, T_MEL)
        pm = torch.from_numpy(profile.prompt_mel)
        pf = min(pm.shape[1], 2 * P_TOK)
        cond[0, :, :pf] = pm[:, :pf]

        est = s3.flow.decoder.estimator
        est_dtype = next(est.parameters()).dtype
        ts = [1 - math.cos(i / 2 * math.pi / 2) for i in range(3)]
        xx = torch.from_numpy(z).to(est_dtype)
        mask = torch.ones(1, 1, T_MEL, dtype=est_dtype)
        for i in range(2):
            v = est.forward(
                x=xx,
                mask=mask,
                mu=mu.to(est_dtype),
                t=torch.tensor([ts[i]], dtype=est_dtype),
                spks=spks.to(est_dtype),
                cond=cond.to(est_dtype),
                r=None,
            )
            xx = xx + (ts[i + 1] - ts[i]) * v
        xx = xx.float()

        mel = torch.zeros(1, 80, HIFT_FRAMES)
        mel[0, :, : 2 * n] = xx[0, :, 2 * P_TOK : 2 * P_TOK + 2 * n]
        hift = StaticHiFT(s3.mel2wav, HIFT_FRAMES).eval()
        wav = hift(mel, torch.from_numpy(phase), torch.from_numpy(noise))
        wav = wav.squeeze(0).numpy()[: 2 * n * UPS].copy()
    return mu.float().numpy()[0], xx.numpy()[0, :, 2 * P_TOK : 2 * P_TOK + 2 * n], wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", choices=["voice", "t3", "s3gen", "all"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from chatterbox.models.tokenizers import MTLTokenizer
    from load_checkpoint import load_checkpoint, pin_backend_flags

    from loudkit.voice import VoiceProfile

    pin_backend_flags()
    voice_path = os.path.join(args.out, "testvoice.voice.safetensors")
    if args.stage in ("voice", "all") or not os.path.exists(voice_path):
        profile = build_voice(args.out)
    else:
        profile = VoiceProfile.load(voice_path)

    if args.stage == "voice":
        return

    t3, s3, manifest = load_checkpoint(CKPT, "cpu")
    # The loudkit token generator runs fp32 (ExecutionConfig default; the
    # conformance fixture is fp32, and every port matches it). The reference
    # must run at the same precision or the free-run tokens drift by the fp16
    # band and flip a token at a sampling boundary.
    t3 = t3.float().eval()
    tok = MTLTokenizer(os.path.join(CB, "models", "grapheme_mtl_merged_expanded_v1.json"))
    sil_ids = manifest["silence_token_ids"]

    from chatterbox.models.t3.modules.cond_enc import T3Cond

    from loudkit.voice import EMOTION_NEUTRAL

    t3c = T3Cond(
        speaker_emb=torch.from_numpy(profile.speaker_embedding)[None],
        cond_prompt_speech_tokens=torch.from_numpy(profile.cond_prompt_tokens)[None],
        emotion_adv=torch.tensor(EMOTION_NEUTRAL).view(1, 1, 1),
    )

    meta = {}
    for i, text in enumerate(SENTENCES):
        seed = SEED_BASE + i
        ids = text_ids(tok, text)
        embeds = prefill_embeds(t3, t3c, ids)
        rec: dict[str, Any] = {"text": text, "seed": seed, "text_ids": [int(x) for x in ids]}

        if args.stage in ("t3", "all"):
            tokens = free_run(t3, embeds, len(ids), sil_ids, seed)
            speech = [t for t in tokens if t < 6561][:T_QUERY]
            logits = teacher_forced(t3, embeds, speech[:64])
            np.save(os.path.join(args.out, f"s{i}_tf_logits.npy"), logits)
            rec["tokens"] = tokens
            rec["speech_tokens"] = speech
        else:
            with open(os.path.join(args.out, "meta.json"), encoding="utf-8") as f:
                rec.update(json.load(f)[str(i)])

        if args.stage in ("s3gen", "all"):
            speech = rec["speech_tokens"]
            z, phase, noise = loudkit_noise(seed, len(speech))
            mu, mel, wav = render_production(s3, profile, speech, z, phase, noise)
            np.save(os.path.join(args.out, f"s{i}_mu.npy"), mu)
            np.save(os.path.join(args.out, f"s{i}_mel.npy"), mel)
            np.save(os.path.join(args.out, f"s{i}_wav.npy"), wav)

        meta[str(i)] = rec
        print(f"s{i}: {len(rec.get('speech_tokens', []))} speech tokens")

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print("DONE ->", args.out)


if __name__ == "__main__":
    main()
