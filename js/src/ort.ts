/** Load ONNX Runtime only after disabling its built-in process telemetry. */

import { createRequire } from "node:module";

// Official ONNX Runtime native builds enable telemetry by default. This must
// happen before the native addon is loaded, which is why a runtime `require`
// lives here instead of a static import in each graph module.
process.env.ORT_DISABLE_TELEMETRY = "1";

const require = createRequire(import.meta.url);

export const ort = require("onnxruntime-node") as typeof import("onnxruntime-node");
export type OrtInferenceSession = import("onnxruntime-node").InferenceSession;
export type OrtTensor = import("onnxruntime-node").Tensor;
