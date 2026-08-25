"""Model implementations of the loudkit contracts.

One module per pipeline stage, mirroring :mod:`loudkit.contracts`:

- :mod:`.text` — text normalisation + the multilingual grapheme tokenizer
- :mod:`.generator` — the Llama-architecture token generator (T3 student)
- :mod:`.flow` — conditional flow matching, tokens to mel
- :mod:`.vocoder` — HiFT (HiFiGAN + NSF source), mel to waveform
- :mod:`.enroll` — reference audio to a :class:`~loudkit.voice.VoiceProfile`
- :mod:`.noise` — render randomness as *data*, addressed by the Philox counter

The torch modules in here mirror the packed checkpoint's tensor names exactly
(``t3.*`` / ``s3gen.*`` with the prefix stripped), so weights load with
``strict=True`` and a naming drift fails at load time instead of rendering
plausible-but-wrong audio.
"""
