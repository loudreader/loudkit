# Voices

loudkit ships 28 voices in 10 languages. Each profile is enrolled with loudkit's own pipeline from a recording made or released for speech-technology use: personal donations recorded for TTS, and CC0 or CC-BY corpora whose terms allow it. The table below names the source and licence of each voice. [docs/voices/roster/provenance.json](docs/voices/roster/provenance.json) records the rest: the donor, the consent basis, the reference construction, the SHA-256 of each profile, reference and sample, and the seed.

[Open the voice gallery](https://loudreader.github.io/loudkit/demo/) and compare each generated sample with its enrollment reference.

Profiles ship under `voices/` in each Hugging Face model repository, versioned with the checkpoint.

The reference SHA-256 identifies the original WAV used for enrollment. Those source WAVs are not redistributed in the model repository. `reference.public_preview` names the Opus derivative that the gallery plays; it is not the enrolled audio.

Only English has been evaluated by ear. The other nine languages have no native-speaker review of their naturalness. Feedback from native speakers is welcome.

The presentation column describes how a voice sounds, not the donor's gender.

| voice | language | presentation | source | licence |
|---|---|---|---|---|
| `clara` | English | feminine | [Kyutai tts-voices](https://huggingface.co/kyutai/tts-voices) | CC0 |
| `emma` | English | feminine | [Kyutai tts-voices](https://huggingface.co/kyutai/tts-voices) | CC0 |
| `henry` | English | masculine | [Kyutai tts-voices](https://huggingface.co/kyutai/tts-voices) | CC0 |
| `joe` | English | masculine | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `kathleen` | English | feminine | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `lucy` | English | feminine | [Kyutai tts-voices](https://huggingface.co/kyutai/tts-voices) | CC0 |
| `miles` | English | masculine | [Kyutai tts-voices](https://huggingface.co/kyutai/tts-voices) | CC0 |
| `oliver` | English | masculine | [Kyutai tts-voices](https://huggingface.co/kyutai/tts-voices) | CC0 |
| `oscar` | English | masculine | [Kyutai tts-voices](https://huggingface.co/kyutai/tts-voices) | CC0 |
| `sophie` | English | feminine | [Kyutai tts-voices](https://huggingface.co/kyutai/tts-voices) | CC0 |
| `carmen` | Spanish | feminine | [CML-TTS](https://huggingface.co/datasets/ylacombe/cml-tts) | CC-BY-4.0 |
| `dave` | Spanish | masculine | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `colette` | French | feminine | [Kyutai tts-voices](https://huggingface.co/kyutai/tts-voices) | CC-BY-4.0 |
| `henri` | French | masculine | [Kyutai tts-voices](https://huggingface.co/kyutai/tts-voices) | CC-BY-4.0 |
| `kerstin` | German | feminine | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `thorsten` | German | masculine | [Thorsten-Voice](https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full) | CC0 |
| `dante` | Italian | masculine | [MLS](https://huggingface.co/datasets/facebook/multilingual_librispeech) | CC-BY-4.0 |
| `paola` | Italian | feminine | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `darkman` | Polish | masculine | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `gosia` | Polish | feminine | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `tugao` | Portuguese (European) | masculine | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `ines` | Portuguese (Brazilian) | feminine | [CML-TTS](https://huggingface.co/datasets/ylacombe/cml-tts) | CC-BY-4.0 |
| `nathalie` | Dutch | feminine | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `pim` | Dutch | masculine | [OHF-Voice donations](https://github.com/NabuCasa/voice-datasets) | CC0 |
| `nils` | Swedish | masculine | [NST Swedish](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-17/) | CC0 |
| `selma` | Swedish | feminine | [NST Swedish](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-17/) | CC0 |
| `freja` | Danish | feminine | [NST Danish](https://huggingface.co/datasets/alexandrainst/nst-da) | CC0 |
| `soren` | Danish | masculine | [NST Danish](https://huggingface.co/datasets/alexandrainst/nst-da) | CC0 |

All 28 voices are included in both loudr-1 and loudr-1-turbo and load by name:

```python
voice = engine.voice("henry")
```

The gallery plays the 10 English voices on the same story in both models, with seed 7 and matched loudness.


## Enrol your own

Use five to ten seconds of clean audio from one speaker:

```python
import loudkit as lk

mine = lk.enroll("my-recording.wav", "loudreader/loudr-1", name="my-voice")
mine.save("voices/my-voice.safetensors")
```

Get the speaker's consent before you clone a voice. See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).
