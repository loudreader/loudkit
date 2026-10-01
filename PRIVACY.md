# Privacy

This policy covers loudkit: the library, the `loudkit` command, the voice
profiles, and the loudkit plugin for ChatGPT and Codex.

## What we collect

Nothing. loudkit runs on your own computer. We operate no server that
receives your text, your audio, your recordings or your voice profiles, and
loudkit sends no usage data, analytics or crash reports to us.

The ONNX Runtime that the ONNX backends load has telemetry of its own.
loudkit turns it off before the runtime starts.

## What stays on your computer

- The text you synthesise and the audio loudkit makes.
- Recordings you clone a voice from, and the voice profiles that cloning
  writes. `loudkit clone` saves a profile readable by its owner only on macOS
  and Linux.

## Network connections

loudkit connects to other services only to fetch software and models:

- **Hugging Face** (`huggingface.co`), to download the model, the voices and
  the cloning models. Loading a model by name can also ask Hugging Face
  whether a newer revision exists. Load a local release directory instead
  and no request is made.
- **Package registries** (PyPI, npm, crates.io, the Go module proxy, GitHub
  for Swift packages), when you install loudkit.

These services see the request, including your IP address, as with any
download. Their own privacy policies apply. loudkit sends them no text,
audio or recordings.

## The plugin for ChatGPT and Codex

The plugin contains instructions only. It has no server, and we receive
nothing when you use it.

- In a coding agent with a shell, the plugin runs loudkit where that shell
  runs: on your computer, or in a cloud environment you chose, such as a
  Codex cloud task. Either way, nothing is sent to us.
- In ChatGPT, the plugin gives you code and commands to run locally.

Your conversation, including any text or file you share in it, is processed
by OpenAI under its own privacy policy, not by us.

## Children

loudkit is a developer tool and is not directed at children.

## Changes and contact

Changes to this policy are made in this file, and its history is public in
the repository. Questions: open an issue at
https://github.com/loudreader/loudkit/issues.
