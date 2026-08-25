/**
 * The execution provider: its vocabulary, its resolution, and the error a
 * caller who asks for hardware this build does not have reads.
 *
 * Everything here runs on any machine. What cannot run here is the provider
 * itself — this laptop has no CUDA and no DirectML — so the backend list is
 * passed in rather than read from the binding, and the cases that matter most
 * are the ones for absent hardware.
 */

import test from "node:test";
import assert from "node:assert";

import { Engine } from "../engine.js";
import {
  ONNX_PROVIDERS,
  availableProviders,
  describeExecution,
  executionProviders,
  resolveOnnxProvider,
} from "../execution.js";
import { fingerprint } from "../fingerprint.js";
import { algorithmFromManifest } from "../types.js";

test("the vocabulary is the five words every port accepts", () => {
  // Pinned as a literal, not derived: this list is a cross-language contract,
  // and a port that quietly grows a sixth spelling is the defect the contract
  // exists to catch.
  assert.deepEqual([...ONNX_PROVIDERS], ["auto", "cpu", "cuda", "coreml", "directml"]);
});

test("onnxruntime's own backend names stay inside the module", () => {
  assert.deepEqual(executionProviders("directml"), ["dml"]);
  assert.deepEqual(executionProviders("cuda"), ["cuda"]);
  assert.deepEqual(executionProviders("cpu"), ["cpu"]);
});

// CoreML is worth having only when the compiled models are cached, and Node
// cannot name a cache directory: the addon reads only `coreMlFlags`, and
// session-config entries are ignored, so EPContext is unreachable too. Every
// process would pay the compile again -- 116.7 s to open the graphs against
// 2.1 s on cpu, measured on an M3 Pro. The word stays in the vocabulary
// because the five spellings are a cross-port contract; the port refuses it.
test("coreml is refused by name, and the refusal says why", () => {
  assert.throws(
    () => resolveOnnxProvider("coreml", ["cpu", "coreml"]),
    (e: Error) => {
      assert.match(e.message, /cannot cache compiled CoreML models/);
      assert.match(e.message, /116\.7 s/);
      assert.match(e.message, /Swift package/);
      return true;
    }
  );
});

test("coreml is not reported as available even where the binary carries it", () => {
  // A list naming a provider the next call rejects sends the reader looking
  // for the wrong fault.
  assert.deepEqual(availableProviders([{ name: "coreml" }, { name: "cpu" }]), ["cpu"]);
});

test("the build's backend list becomes loudkit's spellings, in preference order", () => {
  // `webgpu` and `tensorrt` are reported by real builds and have no loudkit
  // name; a name nobody can ask for would only make the error longer.
  assert.deepEqual(
    availableProviders([
      { name: "cpu" },
      { name: "webgpu" },
      { name: "coreml" },
      { name: "tensorrt" },
    ]),
    // coreml is dropped too, for a different reason: this port refuses it.
    ["cpu"]
  );
  assert.deepEqual(availableProviders([{ name: "cpu" }, { name: "dml" }]), ["directml", "cpu"]);
});

test("auto takes only a provider a measurement backs", () => {
  // auto never reaches coreml: this port refuses it outright, and
  // availableProviders does not report it either.
  assert.deepEqual(resolveOnnxProvider("auto", ["cpu"]), ["cpu"]);
  assert.deepEqual(resolveOnnxProvider("auto", ["cpu", "cuda", "directml"]), [
    "cuda",
    "cpu",
  ]);
  // Omitted means auto: the default has to be the same value the string names.
  assert.deepEqual(resolveOnnxProvider(undefined, ["cpu", "cuda"]), ["cuda", "cpu"]);
});

test("an explicit provider gets one candidate, so there is nothing to fall back to", () => {
  assert.deepEqual(resolveOnnxProvider("cpu", ["cuda", "cpu"]), ["cpu"]);
  assert.deepEqual(resolveOnnxProvider("cuda", ["cuda", "cpu"]), ["cuda"]);
});

test("an explicit provider this build lacks is an error that says how to get it", () => {
  assert.throws(
    () => resolveOnnxProvider("cuda", ["coreml", "cpu"]),
    (err: Error) => {
      // The three things the message owes the reader: what was asked for, what
      // this build has, and where the missing one comes from. A message with
      // only the first two sends the reader looking for an npm package that
      // does not exist.
      assert.match(err.message, /onnxProvider "cuda" is not available/);
      assert.match(err.message, /offers: coreml, cpu/);
      assert.match(err.message, /linux\/x64 prebuilt binary/);
      assert.match(err.message, /--onnxruntime-node-install=skip/);
      assert.match(err.message, new RegExp(`${process.platform}/${process.arch}`));
      return true;
    }
  );
  assert.throws(() => resolveOnnxProvider("directml", ["cpu"]), /win32\/x64 and win32\/arm64/);
  // coreml does not reach this message: the port refuses it before asking
  // whether the build carries it, because the answer would not help.
  assert.throws(() => resolveOnnxProvider("coreml", ["cpu"]), /cannot cache compiled CoreML/);
});

test("a value outside the vocabulary names the vocabulary", () => {
  // JS callers reach this: TypeScript rejects the string at compile time, and
  // a `--provider gpu` flag read from argv does not go through TypeScript.
  assert.throws(
    () => resolveOnnxProvider("gpu" as never, ["cpu"]),
    /must be one of "auto", "cpu", "cuda", "coreml", "directml", got "gpu"/
  );
  assert.throws(() => resolveOnnxProvider("CUDA" as never, ["cpu"]), /must be one of/);
});

test("auto on a build offering nothing loudkit can use is an error, not a guess", () => {
  assert.throws(() => resolveOnnxProvider("auto", []), /reports no execution provider/);
});

test("this onnxruntime-node build offers the CPU provider", () => {
  // The one thing that is true of every build, and the floor `auto` lands on.
  const available = availableProviders();
  assert.ok(available.includes("cpu"), `build offers ${available.join(", ") || "nothing"}`);
  assert.equal(resolveOnnxProvider("auto").at(-1), "cpu");
});

test("the provider that ran reaches describe()", () => {
  // Assembled from two fields; every other route to the string wants six ONNX
  // graphs and a checkpoint this suite may not have.
  const config = algorithmFromManifest({});
  const engine = Object.create(Engine.prototype) as Engine;
  Object.assign(engine, { config, onnxProvider: "coreml" });
  const line = engine.describe();
  assert.match(line, /provider=coreml/);
  assert.ok(line.startsWith(`algo[${fingerprint(config)}] ${config.recipeVersion} |`), line);
});

test("describe never prints auto", () => {
  // `auto` is a request, not an answer. A benchmark row that says `auto` names
  // no provider at all, which is the reporting hole this option closes.
  for (const provider of ["cpu", "cuda", "coreml", "directml"] as const) {
    assert.equal(describeExecution(provider), `exec[onnx provider=${provider} prec[all=fp32]]`);
  }
});
