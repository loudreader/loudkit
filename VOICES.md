# Voices

**20 voices, 10 languages.** Every profile is enrolled by this project's own pipeline from a recording made or released for speech-technology use: personal donations recorded for TTS, and CC0 / CC-BY corpora whose terms allow it. The donor or source, the licence and a sample are named for every voice. The full chain lives in [docs/voices/roster/provenance.json](docs/voices/roster/provenance.json): consent basis, reference construction, SHA-256 of profile, reference and sample, and seed.

[Listen to all twenty voices](https://loudreader.github.io/loudkit/demo/) and compare each generated sample with its enrollment reference.

Profiles ship on the Hugging Face repository under `voices/`, versioned next to the checkpoint they enrol against.

The reference SHA-256 identifies the original WAV used for enrollment. Those source WAVs are not redistributed in the model repository; `reference.public_preview` names the Opus derivative played on the demo page, not the bytes that were enrolled.

We have evaluated **English** by ear. We do not speak the other nine languages well enough to judge their naturalness reliably. Feedback from native speakers is very welcome.

| voice | language | gender | source | licence |
|---|---|---|---|---|
| `joe` | English | M | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `kathleen` | English | F | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `carmen` | Spanish | F | [CML-TTS](https://huggingface.co/datasets/ylacombe/cml-tts) | CC-BY-4.0 |
| `dave` | Spanish | M | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `colette` | French | F | [Kyutai tts-voices](https://huggingface.co/kyutai/tts-voices) | CC-BY-4.0 |
| `henri` | French | M | [Kyutai tts-voices](https://huggingface.co/kyutai/tts-voices) | CC-BY-4.0 |
| `kerstin` | German | F | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `thorsten` | German | M | [Thorsten-Voice](https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full) | CC0 |
| `dante` | Italian | M | [MLS](https://huggingface.co/datasets/facebook/multilingual_librispeech) | CC-BY-4.0 |
| `paola` | Italian | F | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `darkman` | Polish | M | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `gosia` | Polish | F | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `ines` | Portuguese (European) | F | [CML-TTS](https://huggingface.co/datasets/ylacombe/cml-tts) | CC-BY-4.0 |
| `tugao` | Portuguese (European) | M | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `nathalie` | Dutch | F | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `pim` | Dutch | M | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `nils` | Swedish | M | [NST Swedish](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-17/) | CC0 |
| `selma` | Swedish | F | [NST Swedish](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-17/) | CC0 |
| `freja` | Danish | F | [NST Danish](https://huggingface.co/datasets/alexandrainst/nst-da) | CC0 |
| `soren` | Danish | M | [NST Danish](https://huggingface.co/datasets/alexandrainst/nst-da) | CC0 |


## Enrol your own

Ten seconds of clean audio is enough:

```python
import loudkit as lk

mine = lk.enroll("my-recording.wav", "loudreader/loudr-1", name="my-voice")
mine.save("voices/my-voice.safetensors")
```

Consent is yours to obtain. See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).
