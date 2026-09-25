# research

These scripts are research notes. They are not part of the library: loudkit
does not import them, no package ships them, and nothing here is needed to
install, run or release it. `tools/` holds the scripts that build the shipped
artifacts.

The documentation cites these scripts as the way to reproduce the numbers it
publishes: the batch tables in `docs/benchmarks.md`, the method in
`docs/design/evaluation.md` and the judge in `docs/design/pairwise-judge.md`.

The scripts pass the repository's lint and type gates. Their command lines can
change at any time, and they carry no version.

Run each script from the repository root with the repository's own
interpreter, for example `.venv/bin/python research/eval_roundtrip.py --help`.
Most need a checkpoint, an API key or a GPU machine, and several cost money to
run.

## Benchmarks

- `bench_batch.py`: batch scaling of the token generator on CUDA.
- `bench_render.py`: batch scaling of the mel decoder and the vocoder, on any device.
- `_bench.py`: the command line and the device sync that the two scripts above share. Not run directly.
- `bench_cuda_box.py`: a full benchmark run on one device, JSON plus every sample.
- `compare_backends.py`: the same text through cpu/mps/coreml/onnx, with determinism and correlation.

## Evaluation

- `eval_roundtrip.py`: ASR round-trip CER per language, read against `eval_floors.json`.
- `eval_floors.json`: published Whisper-large-v3 CER on human speech (FLEURS) for nine of the ten shipped languages, the reference each language's round-trip number is read against.
- `chunk_silence_stats.py`: head, seam and interior silence per chunk, over hundreds of chunks.
- `run_silence_scale.sh`: runs `chunk_silence_stats.py` for one voice, comparing the shipped profile with a repaired one.
- `dump_reference.py`: dumps the upstream reference data `tests/test_parity.py` compares against. It needs the upstream `chatterbox` package and its training artifacts, so it runs in that environment, not in loudkit's.

## Pairwise judge

See `docs/design/pairwise-judge.md` for what the numbers mean.

- `build_reading_set.py`: freezes the public-domain English reading set the judge reads from.
- `render_comparators.py`: renders that set with loudkit and each comparator, then pairs them.
- `judge_pairwise.py`: order-balanced LLM-as-judge over an OpenAI-compatible API.
- `judge_report.py`: aggregates verdicts into a table and a dot-and-whisker chart.
- `judge_contact_sheet.py`: a listening page with the audio and the verdicts, for a person to check the judge.
- `run_judge_chain.sh`: the five above, end to end and resumable.

## Lexicons

- `fetch_nst_lexicons.py`: downloads the CC0 Swedish, Danish and Norwegian NST lexicons. Nothing of theirs is vendored here.
