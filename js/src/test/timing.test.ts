/**
 * The two timing tiers: exact chunk spans, estimated word spans.
 *
 * No fixture and no weights — the arithmetic is the whole feature, and it is
 * pure. What is pinned here is what a highlight in a reading app depends on:
 * that the joins between chunks are the *same float* on both sides (not merely
 * close), that the words tile their chunk without drifting, and that a word's
 * weight is its code-point count so the five ports weight the same text the same
 * way.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { estimateWords, timeline } from "../timing.js";

const RATE = 24_000;

test("chunk joins are the same float on both sides, not merely close", () => {
  // Sample counts chosen so none of the offsets is exactly representable as a
  // short decimal: a timeline that accumulated seconds would differ here in the
  // last bit, which is a nanosecond gap — invisible to an assertion with a
  // tolerance, visible as a flicker in a highlight that switches on
  // `time >= start`.
  const spans = [
    { text: "One.", samples: 12_345, tokens: 41 },
    { text: "Two.", samples: 7, tokens: 1 },
    { text: "Three.", samples: 98_765, tokens: 190 },
  ];
  const chunks = timeline(spans, RATE);

  assert.equal(chunks.length, 3);
  assert.equal(chunks[0].start, 0);
  for (let i = 0; i + 1 < chunks.length; i++) {
    assert.equal(chunks[i].end, chunks[i + 1].start, `join ${i} is not exact`);
  }
  const total = spans.reduce((n, s) => n + s.samples, 0);
  assert.equal(chunks[chunks.length - 1].end, total / RATE);
  // Text and token counts travel with the span rather than being recomputed.
  assert.deepEqual(
    chunks.map((c) => [c.text, c.tokens]),
    [
      ["One.", 41],
      ["Two.", 1],
      ["Three.", 190],
    ]
  );
});

test("one span covers the whole render", () => {
  const [only] = timeline([{ text: "Hello there.", samples: RATE * 2, tokens: 50 }], RATE);
  assert.equal(only.start, 0);
  assert.equal(only.end, 2);
  assert.equal(only.words.length, 2);
});

test("word spans tile their chunk, monotonic and bounded", () => {
  const [chunk] = timeline([{ text: "one two three four", samples: RATE, tokens: 25 }], RATE);
  const words = chunk.words;
  assert.deepEqual(
    words.map((w) => w.text),
    ["one", "two", "three", "four"]
  );
  // The first start and the last end are the chunk's own, exactly: the
  // boundaries come from a running character total rather than from summing
  // per-word durations, so they cannot drift.
  assert.equal(words[0].start, chunk.start);
  assert.equal(words[words.length - 1].end, chunk.end);
  for (let i = 0; i < words.length; i++) {
    assert.ok(words[i].end >= words[i].start, `word ${i} runs backwards`);
    assert.ok(words[i].start >= chunk.start && words[i].end <= chunk.end, `word ${i} escapes`);
    if (i + 1 < words.length) {
      assert.equal(words[i].end, words[i + 1].start, `word join ${i} is not shared`);
    }
  }
});

test("a word's weight is its code-point count, not its UTF-16 length", () => {
  // "𝐚𝐛" is two code points and four UTF-16 units. Weighing it by `.length`
  // would give it two thirds of the span here while Python, Go, Rust and Swift
  // gave it half — the same text read as two different timings, with nothing to
  // show for it in any output anyone looks at.
  const [chunk] = timeline([{ text: "\u{1D41A}\u{1D41B} cd", samples: RATE, tokens: 25 }], RATE);
  assert.equal(chunk.words.length, 2);
  assert.equal(chunk.words[0].end, 0.5);
  assert.equal(chunk.words[1].start, 0.5);
});

test("punctuation stays attached to the word it follows", () => {
  // The split is on whitespace on purpose: a caller lighting up the word wants
  // the full stop lit with it, and a caller matching back against their own
  // string needs the word to be a substring of it.
  const words = estimateWords("Hello, world!", 0, 1);
  assert.deepEqual(
    words.map((w) => w.text),
    ["Hello,", "world!"]
  );
});

test("text with no words yields no words", () => {
  assert.deepEqual(estimateWords("", 0, 1), []);
  assert.deepEqual(estimateWords("   \n\t ", 0, 1), []);
  // Leading and trailing whitespace is dropped rather than becoming an empty
  // word with a zero-length span, which is what a naive split on " " gives.
  assert.deepEqual(
    estimateWords("  spaced   out  ", 0, 1).map((w) => w.text),
    ["spaced", "out"]
  );
});

test("an empty timeline is empty, not one zero-length chunk", () => {
  assert.deepEqual(timeline([], RATE), []);
});
