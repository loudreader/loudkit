import assert from "node:assert/strict";
import test from "node:test";

import { readFileSync } from "node:fs";

import { splitInHalf, splitText } from "../chunking.js";
import { PRODUCTION_CHUNKING, validateChunkConfig } from "../types.js";

// The repair for a chunk the window could not hold. The Python reference's
// TestSplitInHalf, case for case.

test("it halves at a word boundary", () => {
  const halves = splitInHalf("one two three four five six");
  assert.ok(halves);
  assert.equal(halves.join(" "), "one two three four five six");
});

test("it does not seek punctuation", () => {
  // A weaker separator sits nearer the middle than the comma does, and the
  // comma has no pull of its own: of 27 re-splits taken at the separator
  // nearest the middle, every seam over a second fell on a comma.
  assert.deepEqual(splitInHalf("aa bb, cc dddddddddddd ee"), ["aa bb, cc", "dddddddddddd ee"]);
});

test("the word-boundary fallback breaks on a non-breaking space", () => {
  // `splitText`'s last resort, when a window holds no punctuation. The boundary
  // table was introduced for `splitInHalf` and this fallback, ten lines away,
  // was left matching U+0020 alone, so text whose every space is non-breaking
  // was cut mid-word.
  const words = Array.from({ length: 30 }, (_, i) => `ord${String(i).padStart(2, "0")}`);
  const chunks = splitText(words.join("\u00a0"), PRODUCTION_CHUNKING);
  assert.ok(chunks.length > 1, "the case needs to cross a window");
  for (const c of chunks) {
    const parts = c.split("\u00a0");
    assert.ok(words.includes(parts[parts.length - 1]), `cut mid-word: ${c.slice(-12)}`);
  }
});

test("it counts scalars, not grapheme clusters", () => {
  // CRLF is one grapheme cluster and two Unicode scalars; Swift's Array(text)
  // yields clusters and halved this one word later until it was switched to
  // unicodeScalars. The funnel passes CRLF through verbatim.
  assert.deepEqual(splitInHalf("xx\r\nxx xx xxxxx"), ["xx\r\nxx", "xx xxxxx"]);
});

test("it cuts on boundaries the funnel keeps", () => {
  // NBSP and tab survive the funnel and are word boundaries. Matching only
  // U+0020 made a capped chunk whose separators were all non-breaking come back
  // unsplittable, so it shipped its truncation.
  for (const sep of ["\u00a0", "\t", "\u202f"]) {
    assert.deepEqual(splitInHalf(`alpha${sep}beta${sep}gamma`), [`alpha${sep}beta`, "gamma"]);
  }
});

test("a single unbroken run is refused", () => {
  // Splitting it would have to cut a word, which is worse than the truncation
  // it would be repairing.
  assert.equal(splitInHalf("omringden."), null);
  assert.equal(splitInHalf(""), null);
  // Trimming can empty a half the scan thought was interior. Unreachable
  // through the engine, which trims first, but this is exported.
  assert.equal(splitInHalf("x \t"), null);
});

test("neither half is empty", () => {
  for (const text of ["a bb", "aaaaaaaa b", "a bbbbbbbb"]) {
    const halves = splitInHalf(text);
    assert.ok(halves, text);
    assert.ok(halves[0] && halves[1], text);
  }
});

test("the default is the law and unknown spellings are refused by name", () => {
  assert.equal(PRODUCTION_CHUNKING.capResplit, "word");
  assert.throws(
    () => validateChunkConfig({ ...PRODUCTION_CHUNKING, capResplit: "halve" as never }),
    /cap_resplit/,
  );
});

function deriveSeed(seed: bigint, stream: bigint): bigint {
  const mask = (1n << 64n) - 1n;
  return (seed * 0x9e3779b97f4a7c15n + stream * 0xbf58476d1ce4e5b9n) & mask;
}

/**
 * Holds this port to the `cap_resplit` law without weights.
 *
 * The token streams need a model; the plan does not. Which window carries which
 * index, where the split falls and which seed each window draws are computable
 * from the fixture alone, and they are exactly the three things that went wrong
 * while this law was written: a second half seeded from the next chunk's
 * stream, a Swift split counting grapheme clusters, and a queue that did not
 * advance. A port that gets any of them wrong produces different audio while
 * every other test still passes.
 *
 * Whether a chunk overran needs the model, so that one bit is read from the
 * fixture; everything the port decides given it is recomputed and compared.
 */
test("the re-split plan matches the shared fixture", () => {
  const path = process.env.LOUDKIT_FIXTURE ?? "../tests/data/conformance/vectors.json";
  // Not a silent `return`: a suite that reports a pass because it could not
  // find its input is the failure this whole file exists to prevent. Verified
  // with LOUDKIT_FIXTURE=/nonexistent: every other fixture-backed test in
  // this package failed loudly and this one reported a tick.
  const vectors = JSON.parse(readFileSync(path, "utf8"));
  const section = vectors.resplit;
  assert.ok(section, "the fixture has no resplit section; nothing was compared");
  assert.ok(section.cases?.length, "resplit.cases is empty; nothing was compared");
  const resplitStream = BigInt(section.resplit_stream);
  const chunkBase = BigInt(section.chunk_stream_base);
  const prefixTokens: number = section.prefix_tokens;

  for (const c of section.cases) {
    const seed = BigInt(c.seed);
    // The case moves the window, so the chunk budget moves with it: the config
    // refuses a budget larger than the window, and the engine gates the
    // re-split on the window rather than on any cap.
    const cfg = { ...PRODUCTION_CHUNKING, maxTokens: c.window as number };
    assert.equal(cfg.capResplit, "word", "the fixture pins the law");
    const texts = splitText(c.prepared, cfg);
    const windows = c.windows;

    const tail = (w: any): number[] =>
      w.tokens.length <= prefixTokens ? w.tokens : w.tokens.slice(-prefixTokens);
    const check = (at: number, text: string, sd: bigint, pre: number[],
                   split: boolean, index: number) => {
      const w = windows[at];
      assert.equal(w.index, index,
        `${c.name} window ${at}: a moved index moves every later chunk's seed`);
      assert.equal(w.text, text, `${c.name} window ${at}: text`);
      assert.equal(w.split, split, `${c.name} window ${at}: split`);
      assert.equal(w.seed, "0x" + sd.toString(16),
        `${c.name} window ${at}: the second half must draw from its own stream`);
      assert.deepEqual(w.prefix, pre, `${c.name} window ${at}: carry`);
    };

    let wi = 0;
    let carry: number[] = [];
    texts.forEach((chunkText, index) => {
      // Chunk 0 draws the caller's seed itself; the base applies from chunk 1 up.
      const chunkSeed = index === 0 ? seed : deriveSeed(seed, chunkBase + BigInt(index));
      const wasSplit = windows[wi].split as boolean;
      const halves = splitInHalf(chunkText);
      if (wasSplit) {
        assert.ok(halves, `${c.name} chunk ${index}: fixture split it, this port cannot`);
        check(wi, halves[0], chunkSeed, carry, true, index);
        const pre = tail(windows[wi]);
        wi += 1;
        check(wi, halves[1], deriveSeed(chunkSeed, resplitStream), pre, true, index);
        carry = tail(windows[wi]);
        wi += 1;
        return;
      }
      check(wi, chunkText, chunkSeed, carry, false, index);
      carry = tail(windows[wi]);
      wi += 1;
    });
    assert.equal(wi, windows.length, `${c.name}: window count`);
  }
});

test("a NEL at either edge is stripped, as the other four ports strip it", () => {
  // `trim()` and `/\s/` both omit U+0085 NEL, which is why `trimWhitespace`
  // exists ten lines away. A NEL left on an edge rides into the chunk text and
  // is charged against the next chunk's budget, so the boundaries drift;
  // Python's `strip()` removes it and the other three ports strip it too.
  assert.deepEqual(splitText("\u0085hello world\u0085", PRODUCTION_CHUNKING), ["hello world"]);
  assert.deepEqual(splitInHalf("\u0085aaaa bbbb\u0085"), ["aaaa", "bbbb"]);
});
