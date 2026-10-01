---
name: clone-voice
description: Make a loudkit voice profile from a short recording of a voice the user has the right to use, such as their own, locally and offline. Use when the user wants speech in their own voice, wants to clone or enroll a voice from a recording, or asks how to make a custom voice for loudkit.
---

# Clone a voice

loudkit turns 5 to 10 seconds of clean speech into a voice profile: a file of
about 150 KB that works with both models. Cloning runs on the user's
computer; the recording is not uploaded anywhere.

## Consent comes first

Before anything else, ask whose voice is in the recording, unless the user has
already said. Continue only when it is one of these:

- the user's own voice;
- a voice whose owner gave the user written permission to clone it. Ask the
  user to confirm that the permission is in writing. Do not ask them to send
  you the document;
- a recording whose licence allows speech synthesis (for example CC0).

Refuse, and say why, when the request is to:

- impersonate a real person to deceive, defraud or harass;
- make a public figure say something, unless it is clearly labelled as
  synthetic;
- get past voice authentication (bank or phone voice ID, for example);
- clone a voice from a podcast, lecture, video or call without the speaker's
  consent. Public is not consenting.

The full policy is in the project's
[RESPONSIBLE_USE.md](https://github.com/loudreader/loudkit/blob/main/RESPONSIBLE_USE.md).
loudkit cannot check who owns a voice; the user is responsible for that.

## Where you are running

- **Your shell runs on the user's own computer** (a coding agent in a
  terminal, an IDE or a desktop app, working on a local project): follow the
  steps below.
- **Your shell runs in a cloud environment** (a cloud coding task or another
  remote container): clone there only if the recording is already in that
  environment and it can reach the internet for the models. Say first that
  cloning runs in that cloud environment, not on the user's computer. Use
  `"loudkit[onnx,audio,hub]"`, fetch the cloning graphs with `loudkit download
  loudreader/loudr-1 --for onnx --with-cloning`, and add `--device onnx` to
  `loudkit clone`. Never commit the profile or the recording to a repository.
- **You have no shell** (a chat app on the web or a phone): give the user the
  commands from steps 1 to 4 and say they run on their own computer. Do not
  ask them to upload the recording to you for cloning.

## Steps

1. **Install with permission.** Cloning needs the `enroll` extra and fetches
   about 0.5 GB of enrollment models once, in addition to the 0.75 GB
   synthesis model. Ask before you install:

   ```bash
   pip install "loudkit[torch,audio,enroll,hub]"
   ```

2. **Check the recording.** It must be one local WAV or FLAC file with one
   speaker. 5 to 10 seconds of clean speech without music or noise is right.
   Over 30 seconds is refused, and so are silence and clips under one second.
   - Convert other formats first: `afconvert -f WAVE -d LEI16 in.m4a out.wav`
     on macOS, or `ffmpeg -i in.m4a out.wav`.
   - For a longer recording, ask the user which 5 to 10 seconds to use, or
     cut the cleanest stretch of continuous speech.

3. **Name the language.** Ask which language the voice will speak. It is
   required, and a wrong language makes the voice read text with the wrong
   rules. Use a code such as `en`, `pl`, `de`.

4. **Clone.**

   ```bash
   loudkit clone recording.wav --name my-voice --language en
   ```

   This writes `voices/my-voice.safetensors` and prints the command that
   speaks with it. Add `--force` only if the user agrees to overwrite an
   existing profile.

5. **Try it.** Render one sentence and let the user listen:

   ```bash
   loudkit speak "This is my voice, made by loudkit." --voice voices/my-voice.safetensors -o test.wav --play
   ```

   If it sounds wrong, the recording is the usual cause. Suggest another
   5 to 10 seconds: one speaker, steady level, no background sound.

## After cloning

- The profile reproduces the voice on demand. Treat it like a private key:
  keep it out of public repositories, and do not share a profile of a person
  who has not agreed.
- Label audio made with it as synthetic when it is published. Do not remove
  the provenance note with `--no-provenance` to hide that.
- In code, load it with `lk.voice("voices/my-voice.safetensors")` in Python,
  or the `enroll` and `load` calls of the other ports (see the
  `add-offline-tts` skill).
