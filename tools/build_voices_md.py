"""Write `VOICES.md` from the roster's own provenance record.

The voice table is the first thing most people will look at, and it is exactly
the kind of page that goes stale: a voice gets added, a sample gets
regenerated, an attribution changes, and the hand-written table quietly stops
matching the files. So it is generated from
`docs/voices/roster/provenance.json`, the same record that ships beside the
profiles on Hugging Face and carries donor, source, licence, consent basis and
hashes for every profile, reference and sample.

    python tools/build_voices_md.py

Every voice in the provenance record appears here: each one was cleared for
shipment, source by source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROVENANCE = REPO / "docs" / "voices" / "roster" / "provenance.json"
OUT = REPO / "VOICES.md"

# The roster's display order: English, Spanish, French, German, Italian,
# then the rest. The two Portuguese labels sit together.
LANGUAGE_ORDER = [
    "English",
    "Spanish",
    "French",
    "German",
    "Italian",
    "Polish",
    "Portuguese (European)",
    "Portuguese (Brazilian)",
    "Dutch",
    "Swedish",
    "Danish",
]

SOURCE_LABELS = [
    (
        "NabuCasa/voice-datasets",
        ("OHF-Voice donations", "https://github.com/NabuCasa/voice-datasets"),
    ),
    (
        "TV-44kHz-Full",
        ("Thorsten-Voice", "https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full"),
    ),
    ("kyutai/tts-voices", ("Kyutai tts-voices", "https://huggingface.co/kyutai/tts-voices")),
    ("cml-tts", ("CML-TTS", "https://huggingface.co/datasets/ylacombe/cml-tts")),
    (
        "multilingual_librispeech",
        ("MLS", "https://huggingface.co/datasets/facebook/multilingual_librispeech"),
    ),
    ("sbr-17", ("NST Swedish", None)),
    ("nst-da", ("NST Danish", None)),
]

LICENCE_LABELS = {"CC0-1.0": "CC0", "CC-BY-4.0": "CC-BY-4.0"}

# The roster's `gender` field is the perceived presentation of the voice, the
# same label the gallery shows. It is not the donor's gender.
PRESENTATION_LABELS = {"F": "feminine", "M": "masculine"}


def source_cell(source: dict) -> str:
    label, override = next(
        (
            (label, url)
            for needle, (label, url) in SOURCE_LABELS
            if needle in source["url"] or needle in source["name"]
        ),
        (source["name"], source["url"]),
    )
    return f"[{label}]({override or source['url']})"


def rows(voices: list[dict]) -> list[dict]:
    order = {lang: i for i, lang in enumerate(LANGUAGE_ORDER)}
    return sorted(
        voices, key=lambda v: (order.get(v["language"], len(order)), v["language"], v["name"])
    )


def render(voices: list[dict]) -> str:
    ordered = rows(voices)
    # Counted by `language_id`, not by the display label. The two Portuguese
    # voices carry two labels, European and Brazilian, and one `pt`, because
    # there is one Portuguese grammar and one language tag the frontend reads.
    # Ten is the number of languages the library can be asked to read, and the
    # number README, SUPPORTED, CHANGELOG and the model card state.
    language_count = len({v["language_id"] for v in voices})
    english_count = sum(1 for v in voices if v["language_id"] == "en")

    out = [
        "# Voices",
        "",
        f"loudkit ships {len(voices)} voices in {language_count} languages. "
        "Each profile is enrolled with loudkit's own pipeline from a recording "
        "made or released for speech-technology use: personal donations "
        "recorded for TTS, and CC0 or CC-BY corpora whose terms allow it. The "
        "table below names the source and licence of each voice. "
        "[docs/voices/roster/provenance.json](docs/voices/roster/provenance.json) "
        "records the rest: the donor, the consent basis, the reference "
        "construction, the SHA-256 of each profile, reference and sample, and "
        "the seed.",
        "",
        "[Open the voice gallery](https://loudreader.github.io/loudkit/demo/) "
        "and compare each generated sample with its enrollment reference.",
        "",
        "Profiles ship under `voices/` in each Hugging Face model repository, "
        "versioned with the checkpoint.",
        "",
        "The reference SHA-256 identifies the original WAV used for enrollment. "
        "Those source WAVs are not redistributed in the model repository. "
        "`reference.public_preview` names the Opus derivative that the gallery "
        "plays; it is not the enrolled audio.",
        "",
        "Only English has been evaluated by ear. The other nine languages have "
        "no native-speaker review of their naturalness. Feedback from native "
        "speakers is welcome.",
        "",
        "The presentation column describes how a voice sounds, not the donor's gender.",
        "",
        "| voice | language | presentation | source | licence |",
        "|---|---|---|---|---|",
    ]
    for v in ordered:
        licence = LICENCE_LABELS.get(v["source"]["license"], v["source"]["license"])
        presentation = PRESENTATION_LABELS.get(v["gender"], v["gender"])
        out.append(
            f"| `{v['name']}` | {v['language']} | {presentation} "
            f"| {source_cell(v['source'])} | {licence} |"
        )

    out += [
        "",
        f"All {len(voices)} voices are included in both loudr-1 and "
        "loudr-1-turbo and load by name:",
        "",
        "```python",
        'voice = engine.voice("henry")',
        "```",
        "",
        f"The gallery plays the {english_count} English voices on the same story "
        "in both models, with seed 7 and matched loudness.",
    ]
    out += [
        "",
        "",
        "## Enrol your own",
        "",
        "Use five to ten seconds of clean audio from one speaker:",
        "",
        "```python",
        "import loudkit as lk",
        "",
        'mine = lk.enroll("my-recording.wav", "loudreader/loudr-1", name="my-voice")',
        'mine.save("voices/my-voice.safetensors")',
        "```",
        "",
        "Get the speaker's consent before you clone a voice. See "
        "[RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    voices = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    # `newline="\n"`: VOICES.md is tracked, and a regeneration under a
    # translating text mode would show every line as changed.
    OUT.write_text(render(voices), encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(voices)} voices)")


if __name__ == "__main__":
    # In the entry point, not in `main`, which the suite calls as a function.
    # Source and destination are both fixed, so any argument is a misreading;
    # parsing is what makes `--help` print help instead of rewriting VOICES.md.
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    main()
