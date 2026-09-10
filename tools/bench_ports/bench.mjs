// node tools/bench_ports/bench.mjs BUNDLE VOICE TEXT (after npm --prefix js run build)
import { Engine, loadVoice } from '../../js/dist/index.js';
import { performance } from 'node:perf_hooks';
const [bundle, voicePath, text] = process.argv.slice(2);
if (!text) {
  console.error('usage: bench BUNDLE VOICE TEXT');
  process.exit(2);
}
let start = performance.now();
const engine = await Engine.load(bundle, {onnxProvider: 'cpu'});
const load = (performance.now() - start) / 1000;
// The checkpoint's rate, not a literal 24000: at any other rate a hardcoded
// divisor reports the wrong audio duration, and so the wrong RTF, for every run.
const sampleRate = engine.config.sampleRate;
try {
  const voice = loadVoice(voicePath);
  const runs = [];
  for (let run = 0; run < 4; run++) {
    start = performance.now();
    let first = 0, samples = 0, tokens = 0, chunks = 0;
    for await (const c of engine.stream(text, voice, {seed: 7})) {
      if (!chunks) first = (performance.now() - start) / 1000;
      chunks++;
      samples += c.audio.length;
      tokens += c.tokens.length;
    }
    runs.push({
      run,
      seconds: (performance.now() - start) / 1000,
      ttfa_s: first,
      audio_s: samples / sampleRate,
      tokens,
      chunks,
    });
  }
  console.log(JSON.stringify({runtime: 'js', bundle, load_s: load, text, seed: 7, runs}));
} finally {
  await engine.close();
}
