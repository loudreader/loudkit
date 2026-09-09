import { Engine } from "loudkit";

const engine = await Engine.load("loudreader/loudr-1");
const result = await engine.synthesize("Hello from loudkit.", engine.voice("joe"), { seed: 7 });
result.saveWav("hello.wav");
console.log(`hello.wav, ${(result.audio.length / result.sampleRate).toFixed(2)}s`);
await engine.close();
