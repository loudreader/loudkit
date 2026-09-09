# Command line

Run `loudkit --help` or `loudkit <command> --help` for the full options.

## Speak and listen

```bash
loudkit speak "Hello from loudkit." --voice joe --play -o hello.wav
```

`speak` saves a WAV. `--play` then waits for the system player to finish:
`afplay` on macOS, `aplay` or `paplay` on Linux, and PowerShell on Windows.
Without the flag, it prints a playback command. The Python `synthesize`
method never plays audio.

If no player is available, the command names what to install and exits
with an error. The saved WAV remains available, including when playback fails.
Linux packages are `alsa-utils` or `pulseaudio-utils`; Windows needs PowerShell.

Use `--checkpoint` to select the model, `--language` to override the voice's
language, and `--seed` for repeatable generation. A positional `-` reads stdin.

## Preview speech text

```bash
loudkit text 'Dr. Smith paid $12 on 2024-01-02. [12]' --language en
```

`text` runs the same text funnel as synthesis without loading or downloading
a model. It prints prepared text to stdout, so it can be redirected or piped.
The language defaults to `en`; a positional `-` reads stdin.

Stderr shows a word diff: `replaced`, `removed`, or `inserted` spans. These
include number and date expansions and dropped annotations.
The diff compares input with output; it does not attribute changes to
individual processing stages. Whitespace-only changes are omitted.
Empty or entirely dropped input prints an empty line.

## Clone a voice

`loudkit clone recording.wav --name mine --language en` writes a portable
voice profile. `--device onnx` and `--device coreml` use the corresponding
enrollment graphs without torch. Install the matching backend and `audio`
extras, and prepare cloning assets with `download --with-cloning --for onnx`
or `download --with-cloning --for coreml`. See [cloning](../guides/03-cloning-a-voice.md).
