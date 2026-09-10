import { existsSync, readFileSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import test from "node:test";
import assert from "node:assert/strict";
import { Engine } from "../engine.js";
import { Enroller, profileFrom } from "../enroll.js";
import { saveVoice } from "../voice.js";
import { refuseIfAssetsRequired } from "./assets.js";

const {LOUDKIT_CKPT: first, LOUDKIT_ONNX_DIR: graphs, LOUDKIT_SECOND_CKPT: second,
  LOUDKIT_SECOND_ONNX_DIR: secondGraphs, LOUDKIT_TOKENIZER: tokenizer} = process.env;
const available = [first,graphs,second,secondGraphs,tokenizer].every(p => p && existsSync(p));
refuseIfAssetsRequired(available, "LOUDKIT_CKPT/SECOND_CKPT/ONNX_DIR/SECOND_ONNX_DIR/TOKENIZER are not all present");

test("enroll once and reuse the unchanged profile through both models", {skip:!available && "set both model paths and tokenizer"}, async () => {
  const bytes = readFileSync(join(process.env.LOUDKIT_ENROLL_FIXTURE ?? "../tests/data/enrollment", "ref_audio.f32"));
  const view = new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  const samples = Float32Array.from({length:bytes.length/4},(_,i) => view.getFloat32(i*4,true));
  const enr = await Enroller.load(graphs!, {onnxProvider:"cpu"});
  let profile;
  try { profile = profileFrom(await enr.enroll(samples,24000),{name:"shared",sampleRate:24000,language:"en"}); }
  finally { await enr.close(); }
  const dir = mkdtempSync(join(tmpdir(),"loudkit-shared-voice-"));
  try {
    const path = join(dir,"shared.voice.safetensors");
    saveVoice(profile,path);
    const before = readFileSync(path);
    const modes = new Set<string>();
    for (const [checkpoint,onnx] of [[first,graphs],[second,secondGraphs]]) {
      const engine = await Engine.loadPaths(checkpoint!,onnx!,tokenizer!,{onnxProvider:"cpu"});
      try {
        modes.add(engine.config.decode ?? "single");
        const result = await engine.synthesize("Hello, this is a shared voice.",profile,{seed:7,language:"en"});
        assert.ok(result.audio.length > 0);
        assert.ok(result.audio.every(Number.isFinite));
      } finally { await engine.close(); }
    }
    assert.deepEqual(modes,new Set(["single","fusion_mtp2"]));
    saveVoice(profile,path);
    assert.deepEqual(readFileSync(path),before);
  } finally { rmSync(dir,{recursive:true,force:true}); }
});
