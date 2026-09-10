/**
 * Splitting text that is longer than one window: the port of
 * `loudkit.frontend.chunking`.
 *
 * A window carries about 255 speech tokens, roughly ten seconds. Anything
 * longer has to be split, generated in pieces and joined, and *where* the
 * splits fall is audible: a break at a full stop is inaudible, a break
 * mid-clause is not. That makes it an algorithm-layer decision rather than a
 * caller's convenience, which is why it lives in `AlgorithmConfig` and has to
 * be identical in every port. The rule is simple: break at the strongest
 * punctuation available, as late as possible.
 * Python reference: `loudkit/frontend/chunking.py`.
 */
import type { ChunkConfig } from "./types.js";

/** Unicode White_Space, the class the four other ports strip. */
const WS = "\u0009\u000a\u000b\u000c\u000d\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000";
const LEADING_WS_RE = new RegExp(`^[${WS}]+`, "u");
const TRIM_WS_RE = new RegExp(`^[${WS}]+|[${WS}]+$`, "gu");

function trimWhitespace(text: string): string {
  return text.replace(TRIM_WS_RE, "");
}

/**
 * Characters of prepared text per speech token.
 *
 * Measured on the reference voice across English, Polish (after the respelling
 * funnel) and German: 0.53–0.64. The constant is the **low end with margin**
 * (0.5 < the 0.53 measured minimum) because it is used to *stay under* a limit,
 * never to predict a length: picking the middle of the range would let the
 * worst case overflow the window more often still.
 *
 * It is a budget, not a guarantee. Measured over 9920 rendered chunks in
 * ten languages, 54 overran the window anyway: a mean cannot bound a
 * variance, and most of those are chunks that should have fitted and did
 * not because the model emitted no stop token in time. An overflow is not
 * an error either: the generator stops at the cap mid-word and the
 * remainder is never spoken, which is what `cap_resplit` and `split_in_half`
 * exist for.
 *
 * It must equal `loudkit.frontend.chunking.CHARS_PER_TOKEN`: a different value is a
 * different set of joins and therefore a different reading.
 */
export const CHARS_PER_TOKEN = 0.5;

/** Conservative upper estimate of the speech tokens `text` will produce.
 *
 * Counts code points, not UTF-16 units. `text.length` charges an emoji two
 * characters and a CJK Extension B ideograph two, so the estimate ran up to
 * 1.8x high, 105 tokens in Python against 185 here for the same string. Over-
 * estimating is not "conservative" in the safe direction: it makes the
 * splitter cut a passage into twice as many chunks as Python does, and every
 * chunk boundary is an audible join with its own derived seed.
 */
export function estimateTokens(text: string): number {
  return Math.floor([...text].length / CHARS_PER_TOKEN) + 1;
}

/**
 * Whether `code` is the char code of an ASCII letter or digit.
 *
 * A code unit, not a code point, and no `\\w` regex: the five implementations
 * have to answer this identically, and every language's idea of "letter" is
 * its own. ASCII is the part they cannot disagree on, and a surrogate or a
 * Latin-1 letter is never mistaken for one.
 */
function isAsciiWord(code: number): boolean {
  return (
    (code >= 48 && code <= 57) || (code >= 65 && code <= 90) || (code >= 97 && code <= 122)
  );
}

/**
 * Whether the candidate at UTF-16 offset `at` is a period inside a sentence
 * rather than the end of one.
 *
 * `look` is the search window plus one code point, because the test below reads
 * the character *after* the separator and the latest candidate can end the
 * window exactly.
 *
 * Gated on the period: `"! "` and `"? "` end sentences and `"; "` and `", "` do
 * not end them at all, so neither is ever in doubt. The whole question is about
 * the one mark that is written for two jobs.
 */
function holds(look: string, at: number, sep: string, config: ChunkConfig): boolean {
  if (config.midSentencePeriod !== "hold" || !sep.startsWith(".")) return false;
  // A sentence does not resume in lower case. This is what catches the ellipsis
  // the funnel folds to a single period, which no abbreviation list reaches.
  // ASCII only, and measured rather than assumed: over 2253 periods in ten
  // languages, four are followed by a word starting with a non-ASCII lowercase
  // letter, and reading the whole Unicode Lowercase property instead moves one
  // passage in 1200.
  const after = at + sep.length;
  if (after < look.length) {
    const code = look.charCodeAt(after);
    if (code >= 97 && code <= 122) return true;
  }
  // Or the token in front of the period is a listed abbreviation. Entries carry
  // no period of their own: the period belongs to the separator.
  const boundary = look.slice(0, at);
  for (const abbreviation of config.abbreviations) {
    if (!boundary.endsWith(abbreviation)) continue;
    const before = boundary.length - abbreviation.length;
    // The tail of a longer word, not a word of its own.
    if (before > 0 && isAsciiWord(boundary.charCodeAt(before - 1))) continue;
    return true;
  }
  return false;
}

/**
 * Split `text` into pieces that each fit one window, in order, together
 * covering the input. Never empty for non-empty input.
 */
export function splitText(text: string, config: ChunkConfig): string[] {
  const trimmed = trimWhitespace(text);
  if (!trimmed) return [];
  if (!config.enabled || estimateTokens(trimmed) <= config.maxTokens) return [trimmed];

  const budget = Math.floor(config.maxTokens * CHARS_PER_TOKEN);
  const chunks: string[] = [];
  // Indexed by code point, not by UTF-16 unit, as Rust indexes by `char` and
  // Go by rune, both with a comment saying a byte-indexed cut "produces
  // invalid UTF-8, the shape of bug the ports have had before". JS never got
  // that fix: `slice` on UTF-16 units cuts a surrogate pair in half, and the
  // lone surrogate goes straight to `frontend.encode()`. Measured over 218
  // shared cases: 31 produced chunks containing lone surrogates.
  let rest = [...trimmed];

  while (rest.length > 0) {
    if (rest.length <= budget) {
      chunks.push(trimWhitespace(rest.join("")));
      break;
    }
    const head = rest.slice(0, budget + 1).join("");
    // One code point past the window, and used only by `holds`: the latest
    // candidate can end the window exactly, and the test reads the character
    // after it. The search itself stays inside the budget.
    const look = rest.slice(0, budget + 2).join("");
    let cut = -1;
    // Strongest separator first, and within a separator the LATEST break, so
    // chunks run as long as they may rather than as short as they can.
    // `lastIndexOf` returns a UTF-16 offset, so it is converted back to a code
    // point count before it is used as an index into `rest`.
    for (const sep of config.splitOn) {
      let at = head.lastIndexOf(sep);
      // A period inside a sentence is not a boundary, so the search keeps
      // walking back through this separator's own occurrences before it gives
      // up and tries a weaker one. Searching `head.slice(0, at)` skips an
      // occurrence overlapping the held one, which no separator here can have.
      while (at > 0 && holds(look, at, sep, config)) {
        at = head.slice(0, at).lastIndexOf(sep);
      }
      if (at > 0) {
        cut = [...head.slice(0, at)].length + [...sep].length;
        break;
      }
    }
    if (cut <= 0) {
      // No punctuation in a whole window's worth of text. Break at the last
      // word boundary; it will be heard, and that is the point.
      // WORD_BOUNDARIES, not U+0020: NBSP survives the funnel and is ordinary
      // in real prose, so text whose every space is non-breaking would find
      // no boundary here and be cut mid-word.
      let at = -1;
      for (const b of WORD_BOUNDARIES) {
        const found = head.lastIndexOf(b);
        if (found > at) at = found;
      }
      if (at > 0) cut = [...head.slice(0, at)].length;
    }
    if (cut <= 0) {
      cut = budget; // one unbroken token longer than a window: mid-word
    }
    // Never zero: a cut of 0 leaves `rest` unchanged and the loop spins
    // forever.
    cut = Math.max(cut, 1);

    // `WS_RE`, not `trim()` and `/\s/`: ECMAScript excludes U+0085 NEL from
    // both, and the other four ports strip it. A NEL left on the front of the
    // remainder is charged against the *next* chunk's budget, so the boundaries
    // drift and words are cut mid-word -- measured, 51 of 500 fuzz cases split
    // differently here than in Python, Rust, Go and Swift, every one of them a
    // NEL.
    chunks.push(trimWhitespace(rest.slice(0, cut).join("")));
    rest = [...rest.slice(cut).join("").replace(LEADING_WS_RE, "")];
  }
  return chunks.filter((c) => c.length > 0);
}

/**
 * Characters `splitInHalf` may cut on, written out rather than tested for.
 *
 *
 * A predicate would be shorter and the five ports do not agree on one:
 * measured, Python's `str.isspace()` treats U+001C-U+001F as whitespace where
 * the other four do not, and JS alone KEEPS U+0085 where the other four strip
 * it. Swift alone strips U+200B, JS alone strips U+FEFF; the funnel removes
 * both before the splitter sees them, but U+0085 and U+001C-U+001F survive
 * it. A disagreement there is a different split point, which is different
 * audio for the same text and seed. A hand-written set cannot drift.
 *
 * The funnel does not remove these. NBSP in particular is ordinary in real
 * prose ("10 000", French punctuation, typeset copy), and a capped chunk whose
 * only boundaries are NBSP is unsplittable without it: it ships its truncation.
 */
export const WORD_BOUNDARIES: ReadonlySet<string> = new Set([
  "\u0020", "\u0009", "\u000a", "\u000d", "\u00a0", "\u2007", "\u202f",
]);

/**
 * Halve a chunk the window could not hold, at a word boundary.
 *
 * `splitText`'s estimate is conservative but not a guarantee: it budgets
 * characters against a constant, and a speaker slower than that constant fills
 * the window before the text runs out. The generator then stops at the cap
 * mid-word, and the words that did not fit are *lost* rather than deferred,
 * because chunk texts are fixed before any of them is rendered. A cap is rare
 * and not rare enough to ignore; the rendered census is in
 * `docs/design/text-funnel.md`.
 *
 * The boundary is the nearest WORD break, and punctuation is not sought. The
 * reason is mechanical rather than comparative: a comma is an instruction to
 * pause, this model has no pause-duration prior, and fed one it overshoots.
 * Not sought is not avoided: the nearest word break can follow a comma.
 *
 * `null` for a single unbroken run. Splitting it would have to cut a word,
 * which is worse than the truncation it would be repairing.
 */
export function splitInHalf(text: string): [string, string] | null {
  // Code points, not UTF-16 units, so a surrogate pair counts once and the
  // midpoint is the same one every port computes.
  const chars = [...text];
  const middle = chars.length / 2;
  let best: [number, number] | null = null;
  for (let i = 0; i < chars.length; i++) {
    // Interior only: a boundary at either end yields an empty half.
    if (!WORD_BOUNDARIES.has(chars[i]) || i === 0 || i + 1 >= chars.length) continue;
    const distance = Math.abs(i - middle);
    if (best === null || distance < best[0]) best = [distance, i + 1];
  }
  if (best === null) return null;
  const at = best[1];
  const first = trimWhitespace(chars.slice(0, at).join(""));
  const second = trimWhitespace(chars.slice(at).join(""));
  // Trimming can empty a half the scan thought was interior, on input whose
  // boundary run is all whitespace. `splitText` trims before this is ever
  // called, so the engine cannot reach it, but this is exported and a caller
  // handed an empty half would render silence and call it speech.
  if (!first || !second) return null;
  return [first, second];
}

