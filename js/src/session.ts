/**
 * ONNX graph sessions, loaded lazily per graph. Mirrors the Python backend's
 * `_Session`: names are pulled from the model object itself so the exporter
 * and this module never have to keep a name list in sync.
 */

import {
  type ONNXProvider,
  type ResolvedONNXProvider,
  executionProviders,
  resolveOnnxProvider,
} from "./execution.js";
import { ort, type OrtInferenceSession, type OrtTensor } from "./ort.js";

export class Session {
  private sess: OrtInferenceSession;
  inNames: string[];
  outNames: string[];

  private constructor(sess: OrtInferenceSession) {
    this.sess = sess;
    this.inNames = [...sess.inputNames];
    this.outNames = [...sess.outputNames];
  }

  /**
   * Open one graph on a named provider.
   *
   * The provider is required rather than defaulted, because the default before
   * this argument existed was "whatever onnxruntime picks", which is the CPU
   * provider on every build — the silent 1.2x this option exists to end.
   */
  static async create(path: string, provider: ResolvedONNXProvider): Promise<Session> {
    return new Session(
      await ort.InferenceSession.create(path, { executionProviders: executionProviders(provider) })
    );
  }

  /** Run with named feeds; returns a map of output name -> Tensor. */
  async run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>> {
    return this.sess.run(feeds);
  }

  /**
   * Release the native session.
   *
   * `InferenceSession` holds an onnxruntime handle outside the JS heap, so the
   * garbage collector cannot reclaim it: dropping the last reference frees the
   * wrapper and leaks the graph. Go's binding has `Engine.Close`
   * and Rust's `Session` drops with its owner; without a release path here, a
   * caller who built a second engine — a worker pool, a
   * model reload, a test per case — could not get the first one's six graphs
   * back.
   *
   * Idempotent, because the interesting call sites are error paths that cannot
   * easily know what has already been closed.
   */
  async close(): Promise<void> {
    if (this.released) return;
    this.released = true;
    await this.sess.release();
  }

  private released = false;
}

/**
 * Open a set of graphs on one provider, and say which provider that was.
 *
 * Two failure paths, deliberately different:
 *
 * * A graph that fails to open unwinds the ones already open. Written as a
 *   loop with an explicit unwind rather than `Promise.all` over an object
 *   literal, because there every session already created is abandoned when a
 *   later one throws — a missing or corrupt graph file leaves up to five
 *   native sessions with no reference and no way to release them.
 * * An `"auto"` request that fails on its first choice tries the next one and
 *   warns. This is the case onnxruntime cannot answer by inspection: the
 *   linux/x64 binding reports `cuda` because it was compiled with it, while
 *   the EP's shared libraries are a separate download the postinstall step may
 *   have skipped, so "does this build offer cuda" and "can this machine run
 *   cuda" are different questions and only opening a session asks the second.
 *   The fallback is announced, not silent, and an explicit request never gets
 *   one — `resolveOnnxProvider` hands back a single candidate for it.
 */
export async function openSessions(
  graphs: ReadonlyArray<readonly [name: string, path: string]>,
  requested: ONNXProvider | undefined
): Promise<{ provider: ResolvedONNXProvider; sessions: Record<string, Session> }> {
  const candidates = resolveOnnxProvider(requested);
  for (let i = 0; i < candidates.length; i++) {
    const provider = candidates[i];
    const sessions: Record<string, Session> = {};
    try {
      for (const [name, path] of graphs) {
        sessions[name] = await Session.create(path, provider);
      }
      return { provider, sessions };
    } catch (err) {
      await Promise.all(Object.values(sessions).map((s) => s.close()));
      if (i === candidates.length - 1) throw err;
      process.emitWarning(
        `opening the ONNX graphs on the ${provider} provider failed, so onnxProvider ` +
          `"auto" is falling back to ${candidates[i + 1]}. Pass onnxProvider: ` +
          `"${provider}" to get this as an error instead. The failure was: ` +
          `${err instanceof Error ? err.message : String(err)}`,
        "LoudkitProviderWarning"
      );
    }
  }
  // resolveOnnxProvider never returns an empty list, and the last candidate
  // above either returns or throws.
  throw new Error("no execution provider candidates");
}
