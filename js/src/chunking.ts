/**
 * Splitting text that is longer than one window: the port of
 * `loudkit.frontend.chunking`.
 *
 * A window carries about 255 speech tokens, roughly ten seconds. Anything
 * longer has to be split, generated in pieces and joined, and *where* the
 * splits fall is audible: a break at a full stop is inaudible, a break
 * mid-clause is not. That makes it an algorithm-layer decision rather than a
 * caller's convenience, which is why it lives in `AlgorithmConfig` and has to
 * be identical in every port. Prose keeps sentence endings with their closing
 * punctuation. Custom separator lists retain their literal order.
 * Python reference: `loudkit/frontend/chunking.py`.
 */
import type { ChunkConfig } from "./types.js";

/** Whitespace stripped by the reference (Python str.isspace). */
function isSpace(char: string): boolean {
  return /\p{White_Space}/u.test(char) || (char >= "\u001c" && char <= "\u001f");
}

function trimWhitespace(text: string): string {
  let start = 0;
  let end = text.length;
  while (start < end && isSpace(text[start])) start++;
  while (end > start && isSpace(text[end - 1])) end--;
  return text.slice(start, end);
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

export const PROSE_SEPARATORS = [". ", "! ", "? ", "; ", ", "] as const;
const CLOSERS = "\"'”’»›“‘«‹)]}";

function nextWord(chars: string[], at: number): number {
  while (at < chars.length && isSpace(chars[at])) at++;
  return at;
}

/** The full lookahead distinguishes a name or a lowercase continuation from a new sentence. */
function heldPeriod(
  chars: string[], start: number, dot: number, following: number,
  config: ChunkConfig, abbreviations: string[][]
): boolean {
  if (config.midSentencePeriod !== "hold" || following === chars.length) return false;
  const next = chars[following];
  if (next >= "a" && next <= "z") return true;
  if (dot > start && (dot === start + 1 || !isAsciiWord(chars[dot - 2].charCodeAt(0)))) {
    const initial = chars[dot - 1];
    if (initial >= "A" && initial <= "Z" && initial !== "I" && next >= "A" && next <= "Z") {
      return true;
    }
  }
  return abbreviations.some((abbr) => {
    const first = dot - abbr.length;
    return first >= start && (first === start || !isAsciiWord(chars[first - 1].charCodeAt(0)))
      && abbr.every((char, offset) => chars[first + offset] === char);
  });
}

function proseCut(
  chars: string[], start: number, budget: number, config: ChunkConfig, abbreviations: string[][],
  language: string
): number {
  const limit = start + budget;
  // The production window can attempt the remainder of a nearly complete sentence.
  // If generation actually fills the window, the engine retries it in halves.
  const overrun = config.maxTokens === 255 && config.capResplit === "word" ? 24 : 0;
  const stop = Math.min(chars.length, limit + overrun);
  let sentence = 0;
  let extension = 0;
  let pauses = 0;
  let clause = 0;
  let comma = 0;
  for (let i = start; i < stop; i++) {
    const mark = chars[i];
    if (!".!?;:,–\u2014".includes(mark)) continue;
    let end = i + 1;
    while (end < stop && CLOSERS.includes(chars[end])) end++;
    if (end < chars.length && !isSpace(chars[end])) continue;
    if (".!?".includes(mark)) {
      if (mark === "." && heldPeriod(chars, start, i, nextWord(chars, end), config, abbreviations)) {
        continue;
      }
      if (end <= limit) sentence = end;
      else if (!extension) extension = end;
    } else if (end <= limit && (budget < 64 || (end - start) * 4 >= budget)) {
      if (budget >= 64 && chars.length - end <= 24 &&
          chars.slice(end).filter((c, i) => !isSpace(c) && (i === 0 || isSpace(chars[end + i - 1]))).length === 1) continue;
      const spacedDash = i > start && isSpace(chars[i - 1]) && i + 1 < chars.length && isSpace(chars[i + 1]);
      if (mark === ",") {
        comma = end;
        pauses++;
      }
      else if (mark === ";" || mark === ":" || spacedDash) {
        clause = end;
        pauses++;
      }
    }
  }
  return sentence || (pauses < 2 && extension) || clause || comma ||
    proseWordCut(chars, start, budget, config, abbreviations, language);
}

function proseWordCut(
  chars: string[], start: number, budget: number, config: ChunkConfig, abbreviations: string[][],
  language: string
): number {
  const limit = start + budget;
  // Balance two windows if the greedy word cut would leave only a tiny final utterance.
  const remaining = chars.length - start;
  const balance = budget >= 64 && (remaining - budget) * 4 < budget;
  let best = 0;
  let latest = 0;
  let bestDistance = Infinity;
  for (let i = start + 1; i <= limit; i++) {
    if (!WORD_BOUNDARIES.has(chars[i]) || WORD_BOUNDARIES.has(chars[i - 1])) continue;
    latest = i;
    if (chars[i - 1] === "." && heldPeriod(chars, start, i - 1, nextWord(chars, i), config, abbreviations)) {
      continue;
    }
    const distance = Math.abs(2 * (i - start) - remaining);
    if (!balance || distance < bestDistance) {
      best = i;
      bestDistance = distance;
    }
  }
  if (best && budget >= 64 && language === "en") {
    return englishClauseCut(chars, start, best, budget);
  }
  return best || latest || limit;
}

const CONJUNCTIONS = ["although", "because", "unless", "whereas", "while"];
const WHILE_SUBJECTS = ["the", "they", "we", "she", "he", "it", "you"];

/** All literal words in the clause rules are ASCII. Offsets in chars count Unicode scalars. */
function startsWithPhrase(chars: string[], at: number, phrase: string): boolean {
  const end = at + phrase.length;
  return (end === chars.length || WORD_BOUNDARIES.has(chars[end])) &&
    chars.slice(at, end).join("") === phrase;
}

function englishClauseCut(chars: string[], start: number, cut: number, budget: number): number {
  // "through the valley while | the children followed" breaks the clause after its first word.
  for (const word of CONJUNCTIONS) {
    const before = cut - word.length;
    if (before <= start || !WORD_BOUNDARIES.has(chars[before - 1])) continue;
    if (chars.slice(before, cut).join("") !== word) continue;
    const subject = nextWord(chars, cut);
    const wouldOverflow = chars.length - subject <= budget && chars.length - before > budget;
    if ((before - start - 1) * 4 >= budget && !wouldOverflow &&
        (word !== "while" || WHILE_SUBJECTS.some((w) => startsWithPhrase(chars, subject, w)))) {
      return before;
    }
  }

  return cut;
}

function literalCut(chars: string[], start: number, budget: number, config: ChunkConfig): number {
  const head = chars.slice(start, start + budget + 1).join("");
  // One code point past the window, and used only by `holds`: the latest
  // candidate can end the window exactly, and the test reads the character
  // after it. The search itself stays inside the budget.
  const look = head + (chars[start + budget + 1] ?? "");
  let cut = -1;
  // Strongest separator first, and within a separator the LATEST break, so
  // chunks run as long as they may rather than as short as they can.
  // `lastIndexOf` returns a UTF-16 offset, so it is converted back to a code
  // point count before advancing through `chars`.
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
    // in real prose, so text whose every space is non-breaking found no
    // boundary here and got cut mid-word.
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
  return start + Math.max(cut, 1);
}

/** Plan windows in Unicode code points, matching every other SDK. */
export function splitText(text: string, config: ChunkConfig, language = "en"): string[] {
  const trimmed = trimWhitespace(text);
  if (!trimmed) return [];
  const budget = Math.max(1, Math.floor(config.maxTokens * CHARS_PER_TOKEN));
  if (!config.enabled || trimmed.length <= budget) return [trimmed];
  const chars = [...trimmed];
  const prose = config.splitOn.length === PROSE_SEPARATORS.length &&
    config.splitOn.every((sep, i) => sep === PROSE_SEPARATORS[i]);
  const abbreviations = config.abbreviations.map((abbr) => [...abbr]);
  const lang = language.toLowerCase();
  const chunks: string[] = [];
  for (let start = 0; start < chars.length;) {
    if (chars.length - start <= budget) {
      chunks.push(chars.slice(start).join(""));
      break;
    }
    const end = prose
      ? proseCut(chars, start, budget, config, abbreviations, lang)
      : literalCut(chars, start, budget, config);
    chunks.push(trimWhitespace(chars.slice(start, end).join("")));
    start = nextWord(chars, end);
  }
  return chunks.filter((chunk) => chunk.length > 0);
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

