---
name: choose-voice
description: Help the user pick a loudkit voice by language or presentation, and let them hear it. Use when the user asks which voices or languages loudkit has, wants a voice for a given language, or wants to compare voices before generating audio.
---

# Choose a voice

loudkit ships 28 voices in 10 languages. Every voice is in both models,
`loudreader/loudr-1` and `loudreader/loudr-1-turbo`, and loads by name.
The list with source and licence per voice is in
[references/voices.md](references/voices.md).

## Steps

1. Ask for the language if the user did not name one. A voice carries its
   language, so the language decides the shortlist.
2. Offer the voices for that language from the list, with their presentation
   (feminine or masculine). Do not
   describe how a voice sounds: the list does not say, and you have not heard
   it.
3. Point the user to the voice gallery, which plays a sample of every voice:
   https://loudkit.loudreader.io/demo/
4. If you can run shell commands and loudkit is installed, offer to render the
   same sentence in each shortlisted voice, one file per voice:

   ```bash
   loudkit speak "A short sentence in the right language." --voice gosia -o gosia.wav
   ```

5. When the user picks a voice, say how to use it:
   `--voice <name>` on the command line, `engine.voice("<name>")` in code.

## Be honest about quality

- The project evaluated **English** by ear. For the other nine languages it
  has no native-speaker evaluation. Say so when the user picks one of them.
- Text handling (numbers, dates, abbreviations) covers 12 languages, more than
  the 10 with voices.
- No voice exists for a language not in the list. Do not offer a voice from
  another language as if it spoke this one.

## Licences

Every shipped voice is CC0 or CC-BY-4.0. A CC-BY-4.0 voice needs attribution
when the user publishes audio made with it: name the source from the list.

For a voice that is not in the list, such as the user's own, use the
`clone-voice` skill.
