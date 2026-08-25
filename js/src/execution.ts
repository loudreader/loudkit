/**
 * Which onnxruntime execution provider runs the exported graphs.
 *
 * The execution layer, mirroring `loudkit.config.ExecutionConfig`: it decides
 * how fast, and — unlike the algorithm layer — a GPU provider is allowed to
 * move the last bits. Nothing here is part of the fingerprint.
 *
 * Before this, every port asked onnxruntime for nothing and got the CPU
 * provider, so the published benchmark figures described the torch path while
 * a JS caller silently ran at about real time.
 */

import { ort } from "./ort.js";

/**
 * The accepted spellings. These five words are the cross-language contract:
 * the Python, Rust and Go engines take the same names for the same concept,
 * because a port that spells one of them differently is indistinguishable
 * from a port that does something different. onnxruntime's own backend names
 * (`dml`, `CUDAExecutionProvider`) never leave this module.
 */
export const ONNX_PROVIDERS = ["auto", "cpu", "cuda", "coreml", "directml"] as const;

export type ONNXProvider = (typeof ONNX_PROVIDERS)[number];

/** A provider that names hardware — what `"auto"` resolves to. */
export type ResolvedONNXProvider = Exclude<ONNXProvider, "auto">;

/**
 * How the graphs run. Not the text, the voice or the seed, and not the
 * algorithm: two engines that disagree here are still computing the same
 * thing, up to the provider's own arithmetic.
 */
export interface ExecutionOptions {
  /**
   * Execution provider for every ONNX graph. `"auto"` — the default — takes
   * the best one this build and machine offer. An explicit value that is not
   * available is an error, never a quiet downgrade to CPU: a benchmark row
   * that says `cuda` and ran on CPU is the failure this option exists to
   * prevent.
   */
  onnxProvider?: ONNXProvider;
}

/**
 * The order `"auto"` searches. Not the same list as {@link ONNX_PROVIDERS},
 * which is a vocabulary and carries no preference.
 */
// auto prefers a provider only where a measurement says it is faster.
// CoreML EP measured 0.66x against the CPU provider's 1.36x on an M3 Pro,
// cost 25s of session load against 2s, and moved the token stream at index
// 41 (docs/benchmarks.md). DirectML has never been run by this project.
// Both stay selectable by name; neither is a default. CUDA leads until it
// is measured, and drops out the same way if it loses.
const REPORT_ORDER: readonly ResolvedONNXProvider[] = ["cuda", "coreml", "directml", "cpu"];

// What `auto` is willing to choose, which is narrower than what the build
// offers. Availability answers "can this run", preference answers "should it
// be the default", and they are not the same question.
const AUTO_PREFERENCE: readonly ResolvedONNXProvider[] = ["cuda", "cpu"];

/** loudkit's spelling -> the backend name onnxruntime-node reports and accepts. */
const ORT_BACKEND: Record<ResolvedONNXProvider, string> = {
  cpu: "cpu",
  cuda: "cuda",
  coreml: "coreml",
  directml: "dml",
};

const BY_ORT_BACKEND = new Map<string, ResolvedONNXProvider>(
  (Object.keys(ORT_BACKEND) as ResolvedONNXProvider[]).map((p) => [ORT_BACKEND[p], p])
);

/**
 * Where each provider comes from, for the error a caller who asked for a
 * missing one reads.
 *
 * onnxruntime-node ships one prebuilt binary per platform and arch, and which
 * providers that binary carries is fixed at build time — there is no separate
 * npm package to install for CUDA or DirectML, so the honest advice is the
 * platform matrix plus "build from source". CUDA is the one exception worth
 * naming: the binding carries it on linux/x64, but the EP's own shared
 * libraries are downloaded by the package's postinstall step, which
 * `--onnxruntime-node-install=skip` turns off.
 */
const PROVENANCE: Record<ResolvedONNXProvider, string> = {
  cpu:
    "every onnxruntime-node build carries the CPU provider, so a build that does not " +
    "is damaged — reinstall onnxruntime-node",
  cuda:
    "CUDA is in the linux/x64 prebuilt binary only (CUDA 12), and the EP's libraries are " +
    "fetched by the package's postinstall step, which `--onnxruntime-node-install=skip` " +
    "disables",
  coreml: "CoreML is in the darwin/x64 and darwin/arm64 prebuilt binaries only",
  directml: "DirectML is in the win32/x64 and win32/arm64 prebuilt binaries only",
};

/**
 * The providers this onnxruntime-node build offers, in loudkit's spelling.
 *
 * The build is asked rather than guessed from `process.platform`: the platform
 * says which prebuilt binary was downloaded, not what a locally built binding
 * was compiled with. Backends onnxruntime reports that loudkit has no name for
 * (`webgpu`, `tensorrt`) are dropped — a name outside the shared vocabulary
 * cannot be asked for, so listing it would only make the error longer.
 *
 * Takes the backend list as an argument so the resolution and its error can be
 * tested for hardware this machine does not have.
 */
export function availableProviders(
  backends: ReadonlyArray<{ name: string }> = ort.listSupportedBackends()
): ResolvedONNXProvider[] {
  const seen = new Set<ResolvedONNXProvider>();
  for (const b of backends) {
    const provider = BY_ORT_BACKEND.get(b.name);
    if (provider !== undefined) seen.add(provider);
  }
  // CoreML is dropped even where the binary carries it. This port refuses it
  // (see COREML_REFUSED), and a list that offered a provider the next call
  // rejects would send the reader looking for the wrong fault.
  return REPORT_ORDER.filter((p) => p !== "coreml" && seen.has(p));
}

/**
 * Why this port refuses CoreML, in the words the caller sees.
 *
 * CoreML is worth having only when the compiled models are cached: compiling
 * the renderer graphs takes about two minutes, and the other four ports pay
 * that once per machine by naming a cache directory. Node cannot name one. The
 * native addon reads `coreMlFlags` and nothing else -- its symbol table holds
 * no `ModelFormat` and no `ModelCacheDirectory` -- and `extra` session-config
 * entries are ignored, so ONNX Runtime's own EPContext caching is unreachable
 * too (both the nested and the flat key form were tried: no context file is
 * written and no error is raised).
 *
 * So every process would pay the compile again. Measured here on an M3 Pro:
 * 116.7 s to open the graphs against 2.1 s on CPU, for a renderer that is
 * faster only after those two minutes are spent. Offering that as a choice
 * would be offering a trap.
 */
const COREML_REFUSED =
  'onnxProvider "coreml" is not offered by this port. onnxruntime-node cannot ' +
  "cache compiled CoreML models -- its addon reads only coreMlFlags, and " +
  "session-config entries are ignored -- so every process pays the roughly " +
  "two-minute compile again: 116.7 s to open the graphs against 2.1 s on cpu. " +
  'Use "cpu" here. For CoreML on Apple hardware use the Swift package, or the ' +
  "Python, Rust or Go ports, which can name a model cache directory.";

/**
 * The providers to try, in order, for a requested value.
 *
 * One element for an explicit request, because an explicit request has no
 * fallback. For `"auto"` the whole available preference order, so that a
 * provider the build offers but the machine cannot actually run — a CUDA
 * binding whose EP libraries were never downloaded — costs a warning rather
 * than a failed load. Falling back is `"auto"`'s job and only `"auto"`'s; the
 * fallback itself lives in `openSessions`, which is where a session actually
 * gets opened.
 *
 * Throws for a value outside the vocabulary and for an explicit value this
 * build does not offer.
 */
export function resolveOnnxProvider(
  requested: ONNXProvider | undefined = "auto",
  available: readonly ResolvedONNXProvider[] = availableProviders()
): ResolvedONNXProvider[] {
  if (!(ONNX_PROVIDERS as readonly string[]).includes(requested)) {
    throw new Error(
      `onnxProvider must be one of ${ONNX_PROVIDERS.map((p) => `"${p}"`).join(", ")}, ` +
        `got ${JSON.stringify(requested)}`
    );
  }
  if (requested === "auto") {
    if (available.length === 0) {
      throw new Error(
        "onnxruntime-node reports no execution provider loudkit can use. Every build " +
          "carries the CPU provider, so this one is damaged — reinstall onnxruntime-node."
      );
    }
    return AUTO_PREFERENCE.filter((p) => available.includes(p));
  }
  if (requested === "coreml") {
    throw new Error(COREML_REFUSED);
  }
  if (!available.includes(requested)) {
    throw new Error(unavailable(requested, available));
  }
  return [requested];
}

/**
 * Names the provider asked for, the ones this build offers, and how to get the
 * missing one. All three, because each one alone sends the reader somewhere
 * else: without the offered list they cannot tell a typo from a missing build,
 * and without the provenance they go looking for an npm package that does not
 * exist.
 */
function unavailable(
  requested: ResolvedONNXProvider,
  available: readonly ResolvedONNXProvider[]
): string {
  return (
    `onnxProvider "${requested}" is not available. This onnxruntime-node build offers: ` +
    `${available.join(", ")}. ${PROVENANCE[requested]}; this process is ` +
    `${process.platform}/${process.arch}. For a provider your platform's prebuilt binary ` +
    "does not carry, build onnxruntime-node from source. Pass \"auto\" to take the best " +
    "provider this build offers."
  );
}

/** The session option onnxruntime-node wants for a resolved provider. */
export function executionProviders(provider: ResolvedONNXProvider): string[] {
  return [ORT_BACKEND[provider]];
}

/**
 * The execution half of `Engine.describe()`, in the `exec[...]` shape Python
 * and Swift print.
 *
 * `provider=` is the resolved name, never `"auto"`: a benchmark row and a bug
 * report both have to say which provider ran, and "auto" says nothing.
 */
export function describeExecution(provider: ResolvedONNXProvider): string {
  return `exec[onnx provider=${provider} prec[all=fp32]]`;
}
