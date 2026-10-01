---
name: add-offline-tts
description: Add offline, on-device text to speech to a software project with loudkit, an open-source engine with open weights, in Python, JavaScript or TypeScript (Node), Rust, Go or Swift (macOS, iOS). Use when the user wants speech output in their app without a cloud TTS API, needs local or private speech synthesis, or asks how to integrate loudkit.
---

# Add offline text to speech to a project

loudkit runs the same engine in five languages. Audio is made on the device
that runs the code. There is no API key, no account and no per-request cost.
The code is Apache-2.0; the voice licences are in the `choose-voice` skill.

## Steps

1. **Find the target.** Read the project to find its language and platform:

   | Project file | Port | Reference |
   |---|---|---|
   | `pyproject.toml`, `requirements.txt` | Python | [references/python.md](references/python.md) |
   | `package.json` (Node 20+) | JavaScript / TypeScript | [references/js.md](references/js.md) |
   | `Cargo.toml` | Rust | [references/rust.md](references/rust.md) |
   | `go.mod` | Go | [references/go.md](references/go.md) |
   | `Package.swift`, `.xcodeproj` (macOS 14+, iOS 17+) | Swift | [references/swift.md](references/swift.md) |

   Read only the reference for that port. If you cannot tell, ask.

2. **Say what does not fit.** There is no port for Android or for the browser
   (WebAssembly). The JavaScript port is for Node, not the browser.
   Say so instead of forcing a port onto a platform it does not support.
   For a desktop program in another language (C#, Java), `loudkit serve` runs
   a local HTTP server on the same machine that the program can call; see https://github.com/loudreader/loudkit/blob/main/docs/guides/04-server-and-agents.md

3. **Pick a model.** Start with `loudreader/loudr-1` (reference quality).
   Offer `loudreader/loudr-1-turbo` when latency matters more. Switching is a
   change of name; the code and voices stay the same. Tell the user to compare
   both on their own text.

4. **State the download.** The first load fetches the model once, then runs
   offline:

   | Runtime | loudr-1 | loudr-1-turbo |
   |---|---:|---:|
   | Python, PyTorch | 0.75 GB | 0.72 GB |
   | ONNX (JS, Rust, Go, Python without torch) | 2.60 GB | 2.44 GB |
   | CoreML (Swift) | 2.46 GB | 2.44 GB |

   For a mobile or desktop app, download at first launch or on demand, not
   inside the app bundle, unless the user decides otherwise.

5. **Write the integration.** Follow the reference for the port:
   - Load the engine once and keep it. Loading takes seconds.
   - Use `stream` when playback should start before the whole passage is
     rendered, and `synthesize` for a file.
   - Pin the release revision (`v0.1.1`) for reproducible builds.
   - Do not invent API names. Use only the calls in the reference. If the user
     needs something the reference does not show, link the full guide.

6. **Test it.** Add a test that synthesises one short sentence and checks that
   the audio is not empty. Ask before running it: it downloads the model.
   In a cloud environment, run it only if it can reach the internet, and say
   it ran there. Without a shell, give the command to run.

## Rules

- WAVs saved from Python carry a provenance note that marks them as
  synthetic; keep it unless the user asks otherwise. The Go, Rust, JS and
  Swift ports write plain PCM with no note: tell the user to label the audio
  as synthetic in their app or its metadata.
- The project measured speed on specific hardware only. Do not promise a speed
  on the user's device; tell them to measure it.
- For a custom voice, use the `clone-voice` skill.
