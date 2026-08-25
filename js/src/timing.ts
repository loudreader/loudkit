/**
 * Where each chunk — and, approximately, each word — lands in the waveform.
 *
 * Port of `loudkit.timing`. A reading app highlights the sentence it is
 * speaking, and that needs two different kinds of answer. This module keeps them
 * apart on purpose, because conflating them is how a feature like this becomes a
 * lie.
 *
 * **Chunk times are exact.** The engine renders each chunk to its own waveform
 * and concatenates them, so it knows every chunk's sample offset and sample
 * length without estimating anything. {@link ChunkTiming} reports those,
 * converted to seconds. Chunk *k*'s `end` is bit-identical to chunk *k+1*'s
 * `start` — both are the same integer sample offset divided by the same sample
 * rate — so a highlight driven by them can neither gap nor overlap.
 *
 * **Word times are estimated.** The model emits speech tokens, not an alignment;
 * nothing in this pipeline knows where a word begins. {@link WordTiming}
 * distributes a chunk's real duration across its words in proportion to how long
 * each word is in characters, and that is all it is. It is right often enough to
 * be useful for a highlight at sentence scale and wrong in the ways you would
 * expect: a long word said fast, a short word held, a pause before a clause. The
 * error grows with the length of the chunk, because a single bad guess early
 * shifts everything after it — one sentence is usually fine, a long paragraph
 * read as one chunk is not. If you need real alignment you need a forced
 * aligner; this is not one, and pretending otherwise would be worse than the
 * estimate.
 *
 * Both are computed *after* any time-stretch, on the waveform the caller
 * actually receives, so a result rendered at `speed = 1.5` needs no `1/speed`
 * correction applied to them — applying one would double-count.
 */

/**
 * What one rendered chunk contributes to a timeline.
 *
 * The three facts the engine has at concatenation time and nothing else: the
 * text it was asked to speak (post-funnel, which is what was tokenised), how
 * many samples it rendered to, and how many speech tokens it took. Kept as an
 * input type rather than assembling {@link ChunkTiming} per chunk, because the
 * offsets are only knowable once the order is known.
 */
export interface ChunkSpan {
  text: string;
  samples: number;
  tokens: number;
}

/**
 * One word's estimated span, in seconds from the start of the synthesis.
 *
 * **Estimated, by proportional allocation.** The chunk's real duration is
 * divided among its words in proportion to their length in characters. There is
 * no alignment model here and no per-word measurement — see the module comment
 * for what that costs you.
 */
export interface WordTiming {
  /**
   * The word as it appears in the chunk, punctuation included.
   *
   * Punctuation stays attached because the split is on whitespace: a caller
   * highlighting `"end."` wants the full stop lit with the word, and a caller
   * matching back against their own text needs the substring to be a substring.
   */
  text: string;
  start: number;
  end: number;
}

/**
 * One chunk's exact span, and its words' estimated ones.
 *
 * The two tiers in one object on purpose: a caller that trusts only the exact
 * tier reads `start`/`end` and ignores `words`, and the field names make it
 * impossible to reach the estimate by accident.
 */
export interface ChunkTiming {
  /**
   * The chunk's text after the speech funnel — what was tokenised, which is not
   * always what the caller passed in (Polish respells embedded English, and
   * numbers are read as words).
   */
  text: string;

  /**
   * Seconds from the start of this result's audio.
   *
   * Zero for the first chunk, and for every chunk of a streamed result: a
   * streamed chunk is its own result and does not know what preceded it, so the
   * caller stitching the stream adds the offsets.
   */
  start: number;

  end: number;

  /**
   * Speech tokens this chunk generated. Duration over tokens is the pacing the
   * postprocess detectors measure against, which is the other reason to carry
   * it.
   */
  tokens: number;

  words: WordTiming[];
}

/**
 * Lay rendered chunks end to end and time them.
 *
 * Offsets accumulate in **samples**, not seconds, and are divided by the rate
 * once at the end. Accumulating seconds instead would make chunk *k*'s `end` and
 * chunk *k+1*'s `start` two different sums of the same floats, differing in the
 * last bit — a gap or an overlap of a few nanoseconds, invisible in a test that
 * compares with a tolerance and visible as a flicker in a highlight that
 * switches on `time >= start`.
 */
export function timeline(spans: ChunkSpan[], sampleRate: number): ChunkTiming[] {
  const out: ChunkTiming[] = [];
  let at = 0;
  for (const span of spans) {
    const start = at / sampleRate;
    at += span.samples;
    const end = at / sampleRate;
    out.push({
      text: span.text,
      start,
      end,
      tokens: span.tokens,
      words: estimateWords(span.text, start, end),
    });
  }
  return out;
}

/**
 * Split `text` on whitespace and share `[start, end]` out by length.
 *
 * The allocation is by **character count**, not by token count or by any
 * acoustic measure: a word's characters are the only thing known here, and they
 * correlate with duration well enough at sentence scale to drive a highlight.
 * Whitespace itself is not charged for — the gap between two words belongs to
 * whichever side of the boundary the caller's player is on, and splitting it
 * would only invent a third kind of span.
 *
 * Boundaries are computed from a running character total rather than by adding
 * per-word durations, so the spans cannot drift: the first `start` is exactly
 * `start`, the last `end` is exactly `end`, and every interior boundary is
 * shared by the two words that meet at it.
 *
 * Length is counted in **code points** (`[...w].length`), not in UTF-16 code
 * units, so that Python, Go, Rust, Swift and this port weight the same text the
 * same way. `.length` would count an astral character twice here and once
 * everywhere else, which is a silently different reading of the same string
 * rather than an error anyone would notice.
 */
export function estimateWords(text: string, start: number, end: number): WordTiming[] {
  const words = text.split(/\s+/u).filter((w) => w.length > 0);
  const lengths = words.map((w) => [...w].length);
  const total = lengths.reduce((a, b) => a + b, 0);
  if (total === 0) return [];
  const span = end - start;
  const out: WordTiming[] = [];
  let seen = 0;
  for (let i = 0; i < words.length; i++) {
    const at = start + span * (seen / total);
    seen += lengths[i];
    out.push({ text: words[i], start: at, end: start + span * (seen / total) });
  }
  return out;
}
