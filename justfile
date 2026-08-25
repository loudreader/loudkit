# loudkit development commands
# Install: brew install just (or cargo install just)
# Usage: just --list

set shell := ["bash", "-uc"]

venv := ".venv"
python := venv / "bin" / "python"
pip := venv / "bin" / "pip"
ruff := venv / "bin" / "ruff"
mypy := venv / "bin" / "mypy"

# ─── Setup ────────────────────────────────────────────────────────────

# Create the venv and install the package with every extra
setup:
    uv sync --extra dev --extra server --extra mcp --extra torch
    @echo "Setup complete. Run: just test"

# Install extras you may be missing after a fresh clone
setup-full:
    uv sync --extra dev --extra server --extra mcp --extra torch --extra audio --extra coreml --extra onnx
    @echo "Setup complete (all extras)."

# ─── Test ─────────────────────────────────────────────────────────────

# Run the suite that needs no weights (fast, everywhere)
test:
    {{ python }} -m pytest tests/ -q -m "not slow"

# Run the whole suite, including parity (needs LOUDKIT_CHECKPOINT set)
test-all:
    {{ python }} -m pytest tests/ -q

# Run the Swift package's tests (same conformance fixture)
test-swift:
    swift test

# ─── JavaScript/TypeScript binding ─────────────────────────────────────────

# Build the JS/TS package (js)
js-build:
    npm --prefix js run build

# Run the JS weight-free conformance vectors (no assets needed)
js-test:
    LOUDKIT_FIXTURE=$(pwd)/tests/data/conformance/vectors.json \
    LOUDKIT_TOKENIZER=$(pwd)/tests/data/conformance/tokenizer.json \
    npm --prefix js test

# ─── Go binding ────────────────────────────────────────────────────────────

# Run the Go weight-free conformance vectors (no assets needed)
go-test:
    cd go && go test ./conformance/ -run 'TestPhilox|TestUniform|TestGumbel|TestSampler|TestFrontend|TestSeed'

# Full Go suite: weight-free + engine conformance (needs LOUDKIT_* assets +
# a libonnxruntime shared library)
go-test-all:
    cd go && go test ./...

# ─── Rust binding ──────────────────────────────────────────────────────────

# Run the Rust weight-free conformance vectors (no assets needed)
rust-test:
    cd rust && cargo test --test weightfree

# Full Rust suite: weight-free + engine conformance (needs LOUDKIT_* assets +
# ORT_DYLIB_PATH pointing at a libonnxruntime shared library)
rust-test-all:
    cd rust && cargo test && cargo test -- --ignored

# ─── Quality gates ────────────────────────────────────────────────────

# Lint + typecheck + format check, as CI runs them
check: check-python check-types check-format

check-python:
    {{ ruff }} check python/ tests/ tools/ integrations/speech-dispatcher/

check-types:
    {{ mypy }} python/loudkit/
    {{ mypy }} --config-file tools/mypy.ini tools/

check-format:
    {{ ruff }} format --check python/ tests/ tools/ integrations/speech-dispatcher/

# Auto-fix lint and format
fix:
    {{ ruff }} check python/ tests/ tools/ integrations/speech-dispatcher/ --fix
    {{ ruff }} format python/ tests/ tools/ integrations/speech-dispatcher/

# ─── Benchmark ────────────────────────────────────────────────────────

# Benchmark this machine (needs --checkpoint; set LOUDKIT_CHECKPOINT)
bench checkpoint_path voice_path device:
    {{ python }} -m loudkit.cli bench --checkpoint {{ checkpoint_path }} --voice {{ voice_path }} --device {{ device }} --json out/bench.json

# Profile one passage stage by stage
profile checkpoint_path voice_path passage:
    {{ python }} -m loudkit.cli profile --checkpoint {{ checkpoint_path }} --voice {{ voice_path }} -- "{{ passage }}"

# Cross-backend output comparison (cpu_fp32/cpu/mps/mps_fp32/onnx, determinism,
# mel/wave correlation vs the fp32 reference) — writes out/compare/backends_compare.json.
# Checkpoint from LOUDKIT_CHECKPOINT or LOUDKIT_ASSET_ROOT, as the test suite resolves it.
compare:
    {{ python }} tools/compare_backends.py

# ─── Voices table ─────────────────────────────────────────────────────

# Regenerate VOICES.md from the roster's provenance record
voices:
    {{ python }} tools/build_voices_md.py

# ─── Serve ────────────────────────────────────────────────────────────

# Run the synthesis server on localhost
serve checkpoint_path voice_dir port:
    {{ python }} -m loudkit.cli serve --checkpoint {{ checkpoint_path }} --voices {{ voice_dir }} --port {{ port }}

# Serve over MCP (stdio)
mcp checkpoint_path voice_dir:
    {{ python }} -m loudkit.cli mcp --checkpoint {{ checkpoint_path }} --voices {{ voice_dir }}

# ─── Release prep ─────────────────────────────────────────────────────

# Assemble a local release directory and verify it speaks (no upload)
release checkpoint_path:
    {{ python }} tools/build_release.py --checkpoint {{ checkpoint_path }} --out dist/loudr-1
