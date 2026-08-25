/**
 * Cross-call prosody context: the tail of the previous call, and nothing else.
 *
 * `carryFrom` is tested rather than `Engine.synthesize(..., { previousTokens })`
 * because the engine cannot run without the checkpoint and the exported ONNX
 * graphs, and this is the practical unit anyway: the feature is a slice and a
 * range check, and the engine's part of it is passing the result to the same `prefix`
 * argument the streaming loop uses. There is deliberately no second
 * conditioning mechanism to test — a request boundary and a chunk boundary are
 * the same join.
 *
 * What this therefore does NOT cover is the wiring: if the helper's result
 * stopped being handed to the generator's prefix, every assertion here would
 * still pass. That half is pinned in Python, by
 * tests/test_engine.py::TestCrossRequestContext, against a fake generator that
 * records the context it was given — building an equivalent seam in four more
 * languages would cost four engine refactors to re-assert one fact.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { carryFrom } from "../engine.js";

/** The shipping values, so the numbers here mean what they mean in production. */
const PREFIX = 6;
const START_SPEECH = 6561;

test("the carry is the tail of the previous call", () => {
  const previous = [10, 11, 12, 13, 14, 15, 16, 17];
  assert.deepEqual(carryFrom(previous, PREFIX, START_SPEECH), [12, 13, 14, 15, 16, 17]);
  // Any length is accepted because only the tail is used: `previousTokens:
  // result.tokens` is the intended call, and a caller should never have to know
  // the prefix length to make it.
  assert.deepEqual(carryFrom([1, 2], PREFIX, START_SPEECH), [1, 2]);
  assert.deepEqual(carryFrom(previous, 1, START_SPEECH), [17]);
});

test("no previous call is today's behaviour, byte for byte", () => {
  assert.deepEqual(carryFrom(undefined, PREFIX, START_SPEECH), []);
  assert.deepEqual(carryFrom([], PREFIX, START_SPEECH), []);
});

test("a zero prefix carries nothing, not everything", () => {
  // `slice(-0)` is the whole array in JavaScript, exactly as `tokens[-0:]` is in
  // Python. Zero prefix tokens is the setting that means "chunks are
  // independent", so the bug it hides is conditioning on the entire previous
  // utterance at the one setting that asked for no conditioning at all.
  assert.deepEqual(carryFrom([1, 2, 3], 0, START_SPEECH), []);
  assert.deepEqual(carryFrom([1, 2, 3], -1, START_SPEECH), []);
});

test("an id outside the acoustic codebook is refused, and named", () => {
  assert.throws(
    () => carryFrom([1, 2, START_SPEECH], PREFIX, START_SPEECH),
    new RegExp(`${START_SPEECH}.*0 <= id < ${START_SPEECH}`, "s")
  );
  assert.throws(() => carryFrom([-1], PREFIX, START_SPEECH), /-1/);
  assert.throws(() => carryFrom([NaN], PREFIX, START_SPEECH), /NaN/);
});

test("the whole history is checked, not only the slice that will be used", () => {
  // An id out of range means the sequence was built wrong. Reporting that only
  // when the bad id happens to land in the last six tokens would make the
  // failure depend on how long the caller's text was — the kind of bug that is
  // reproducible on one paragraph and not on the next.
  const history = [99_999, 1, 2, 3, 4, 5, 6, 7];
  assert.throws(() => carryFrom(history, PREFIX, START_SPEECH), /99999/);
});

test("the carry is a copy, so the caller's tokens are not aliased", () => {
  const previous = [1, 2, 3, 4, 5, 6, 7, 8];
  const carry = carryFrom(previous, PREFIX, START_SPEECH);
  carry[0] = 4242;
  assert.deepEqual(previous, [1, 2, 3, 4, 5, 6, 7, 8]);
});
