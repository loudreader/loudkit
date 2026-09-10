# research

**A laboratory notebook, not part of the library.** Nothing here is imported by
loudkit, ships in any package, or is needed to install, run or release it:
`tools/` holds what builds the shipped artifacts, and this directory holds what
argues about them. It is in the repository for one reason, which is that the
documentation cites these scripts by name as the way to reproduce the numbers
it publishes: the batch tables in `docs/benchmarks.md`, the method in
`docs/design/evaluation.md`, the judge in `docs/design/pairwise-judge.md`. A
published measurement nobody can re-run is a claim rather than a measurement.

Read it as working notes. The scripts are held to the repository's lint and
type gates, and to nothing else: no stable command line, no compatibility
promise, no version.

They are not a package. Run each from the repository root with the repo's own
interpreter, for example `.venv/bin/python research/eval_roundtrip.py --help`.
Most need a checkpoint, an API key or a GPU box, and several cost money to run.

## Benchmarks

- `bench_batch.py` — batch scaling of the token generator on CUDA.
- `bench_render.py` — batch scaling of the mel decoder and the vocoder, on any device.
- `_bench.py` — the command line and the device sync those two share. Not run directly; imported by both because the interpreter puts a script's own directory first on `sys.path`.
- `bench_cuda_box.py` — a full benchmark run on one device, JSON plus every sample.
- `compare_backends.py` — the same text through cpu/mps/coreml/onnx, with determinism and correlation.

## Evaluation

- `eval_roundtrip.py` — ASR round-trip CER per language, read against `eval_floors.json`.
- `eval_floors.json` — human-speech CER floors, nine of the ten shipped languages, so a round-trip number means something.
- `chunk_silence_stats.py` — head, seam and interior silence per chunk, at a scale where rates mean something.
- `run_silence_scale.sh` — drives `chunk_silence_stats.py` over one voice, shipped against repaired.
- `dump_reference.py` — dumps the upstream reference data `tests/test_parity.py` compares against. Runs inside the chatterbox-apple venv, not this one.

## Pairwise judge

See `docs/design/pairwise-judge.md` for what the numbers mean.

- `build_reading_set.py` — freezes the public-domain English reading set the judge reads from.
- `render_comparators.py` — renders that set with loudkit and each comparator, then pairs them.
- `judge_pairwise.py` — order-balanced LLM-as-judge over an OpenAI-compatible API.
- `judge_report.py` — aggregates verdicts into a table and a dot-and-whisker chart.
- `judge_contact_sheet.py` — a listening page, so a person can overrule the judge.
- `run_judge_chain.sh` — the five above, end to end and resumable.

## Lexicons

- `fetch_nst_lexicons.py` — downloads the CC0 Swedish, Danish and Norwegian NST lexicons. Nothing of theirs is vendored here.
