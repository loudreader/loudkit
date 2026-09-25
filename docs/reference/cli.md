# Command line

The `loudkit` command has eight subcommands: `speak`, `text`, `clone`,
`voices`, `download`, `serve`, `verify` and `doctor`. Run `loudkit --help` or
`loudkit <command> --help` for the full options. On an error, a command prints
a short diagnosis. `loudkit --debug <command>` prints the traceback instead.

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
`--speed` sets the playback speed, from 0.5 to 2.0 ([speed](speed.md)).
`--no-provenance` writes a plain WAV without the provenance manifest.

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
or `download --with-cloning --for coreml`. A repo id as `--checkpoint` also
needs the `hub` extra. See [cloning](../guides/03-cloning-a-voice.md).

## List the voices of a release

```bash
loudkit voices loudreader/loudr-1
```

`voices` prints the voice names in a release, one per line. The argument is a
repo id or a local release directory, and defaults to `loudreader/loudr-1`.
`--revision` pins a commit, tag or branch. The command exits 1 when the release
holds no voices.

## Download a release

```bash
loudkit download loudreader/loudr-1 --for onnx --with-cloning
```

`download` fetches what one backend needs: the checkpoint, the voices and, for
`onnx` or `coreml`, the exported graphs. It then checks the files against the
release's `SHA256SUMS`.

- `--for` takes `torch` (the default), `onnx` or `coreml`. The Rust, Go and JS
  ports read the `onnx` set. Swift reads the `coreml` set.
- `--with-cloning` adds what that backend enrolls with.
- `--revision` pins a commit, tag or branch.
- `--local-dir DIR` writes the files into `DIR` instead of the shared cache. A
  second run into the same directory fetches nothing while the release has
  not moved.

The argument must be a Hub repo id. Pass a release that is already on disk to
the other commands by its path.

## Run a server

```bash
loudkit serve --checkpoint loudreader/loudr-1
```

`serve` runs a local synthesis server. It serves HTTP on `127.0.0.1:8765` by
default.

- `--grpc` serves gRPC on port 50051. `--mcp` serves MCP on stdio.
- `--host` and `--port` change the bind for HTTP and gRPC.
- A non-loopback host needs `--allow-public` and a bearer token (`--token`, or
  `$LOUDKIT_TOKEN`). Both are HTTP only.
- `--voices DIR` names a directory of voice profiles.
- `--first-chunk-tokens N` caps the first streamed chunk at `N` tokens, for
  HTTP and gRPC. This changes where the first chunk ends, so it changes the
  audio.

A flag that the chosen transport cannot use is refused, and the command exits
2. HTTP needs the `server` extra, gRPC the `grpc` extra and MCP the `mcp`
extra. See [Server and agents](../guides/04-server-and-agents.md).

## Verify a file

```bash
loudkit verify hello.wav
```

`verify` checks a WAV, a voice profile or a checkpoint against its own claims.

- A WAV: the audio must match its provenance manifest. The manifest is
  unsigned, so a pass shows that the file is intact, not who made it.
- A voice profile: the command loads it and prints its name, language and
  SHA-256.
- A checkpoint: the command prints its SHA-256, to compare with the release's
  `SHA256SUMS`. When the checkpoint records a tensor payload hash, the command
  also checks it.

The command exits 1 when a check fails, when a WAV has no manifest, or when a
`.safetensors` file is neither a checkpoint nor a voice profile.

## Check the machine

```bash
loudkit doctor
```

`doctor` lists the installed backends and devices, the extras, the checkpoints
and voices in the current directory (or under `--checkpoint`), the releases in
the Hub cache, and what cloning needs. Each missing piece comes with its
`pip install` command. `--describe` also loads the engine and prints its
settings line. The command exits 0 whatever it finds. It exits 1 only when
`--checkpoint` does not exist or when `--describe` cannot load the engine.
