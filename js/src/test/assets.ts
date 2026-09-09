/**
 * The switch that turns a missing asset into a failure.
 *
 * A skipped test and a passing one look identical in a summary line, and the
 * asset-backed tests are the ones it would be most damaging to lose that way:
 * they are the evidence that this binding renders what the reference does. On
 * a runner that is supposed to have the assets, a missing one is a broken
 * environment, so `LOUDKIT_REQUIRE_ASSETS=1` refuses instead of skipping.
 *
 * Same switch and same meaning as the Python suite's `requires()` and the Go,
 * Rust and Swift conformance tests. Here rather than in each test file so that
 * a file which forgets it is a file that does not compile the guard at all,
 * rather than one that skips in silence.
 */

/** True when the runner has declared that the assets must be present. */
export function requireAssets(): boolean {
  const flag = process.env.LOUDKIT_REQUIRE_ASSETS;
  return Boolean(flag) && flag !== "0";
}

/**
 * Refuse at import time when the runner requires assets and `available` is
 * false. `what` names the variables to set, and is also the skip reason the
 * caller passes to `node:test`.
 */
export function refuseIfAssetsRequired(available: boolean, what: string): void {
  if (!available && requireAssets()) {
    throw new Error(`LOUDKIT_REQUIRE_ASSETS is set but ${what}`);
  }
}
