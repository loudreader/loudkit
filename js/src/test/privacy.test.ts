import assert from "node:assert";
import { spawnSync } from "node:child_process";
import test from "node:test";

test("the package disables ONNX Runtime telemetry before loading its native addon", () => {
  const entry = new URL("../index.js", import.meta.url).href;
  const env = { ...process.env };
  delete env.ORT_DISABLE_TELEMETRY;
  const probe = spawnSync(
    process.execPath,
    [
      "--input-type=module",
      "-e",
      `await import(${JSON.stringify(entry)}); process.stdout.write(process.env.ORT_DISABLE_TELEMETRY ?? "missing")`,
    ],
    { encoding: "utf8", env }
  );
  assert.equal(probe.status, 0, probe.stderr);
  assert.equal(probe.stdout, "1");
});
