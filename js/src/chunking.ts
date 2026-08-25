/**
 * Splitting text that is longer than one window — the port of
 * `loudkit.frontend.chunking`.
 *
 * A window carries about 255 speech tokens, roughly ten seconds. Anything
 * longer has to be split, generated in pieces and joined, and *where* the
 * splits fall is audible: a break at a full stop is inaudible, a break
 * mid-clause is not. That makes it an algorithm-layer decision rather than a
 * caller's convenience, which is why it lives in `AlgorithmConfig` and has to
 * be identical in every port. The rule is simple — break at the strongest
 * punctuation available, as late as possible.
 * Python reference: `loudkit/frontend/chunking.py`.
 */
import type { ChunkConfig } from "./types.js";

/**
 * Characters of prepared text per speech token.
 *
 * Measured on the reference voice across English, Polish (after the respelling
 * funnel) and German: 0.53–0.64. The constant is the **low end with margin**
 * (0.5 < the 0.53 measured minimum) because it is used to *stay under* a limit,
 * never to predict a length — picking the middle of the range would let the
 * worst case overflow the window, and an overflow is a hard failure.
 *
 * It must equal `loudkit.frontend.chunking.CHARS_PER_TOKEN`: a different value is a
 * different set of joins and therefore a different reading.
 */
export const CHARS_PER_TOKEN = 0.5;

/** Conservative upper estimate of the speech tokens `text` will produce.
 *
 * Counts code points, not UTF-16 units. `text.length` charges an emoji two
 * characters and a CJK Extension B ideograph two, so the estimate ran up to
 * 1.8x high — 105 tokens in Python against 185 here for the same string. Over-
 * estimating is not "conservative" in the safe direction: it makes the
 * splitter cut a passage into twice as many chunks as Python does, and every
 * chunk boundary is an audible join with its own derived seed.
 */
export function estimateTokens(text: string): number {
  return Math.floor([...text].length / CHARS_PER_TOKEN) + 1;
}

/**
 * Split `text` into pieces that each fit one window, in order, together
 * covering the input. Never empty for non-empty input.
 */
export function splitText(text: string, config: ChunkConfig): string[] {
  const trimmed = text.trim();
  if (!trimmed) return [];
  if (!config.enabled || estimateTokens(trimmed) <= config.maxTokens) return [trimmed];

  const budget = Math.floor(config.maxTokens * CHARS_PER_TOKEN);
  const chunks: string[] = [];
  // Indexed by code point, not by UTF-16 unit — as Rust indexes by `char` and
  // Go by rune, both with a comment saying a byte-indexed cut "produces
  // invalid UTF-8, the shape of bug the ports have had before". JS never got
  // that fix: `slice` on UTF-16 units cuts a surrogate pair in half, and the
  // lone surrogate goes straight to `frontend.encode()`. Measured over 218
  // shared cases: 31 produced chunks containing lone surrogates.
  let rest = [...trimmed];

  while (rest.length > 0) {
    if (rest.length <= budget) {
      chunks.push(rest.join("").trim());
      break;
    }
    const head = rest.slice(0, budget + 1).join("");
    let cut = -1;
    // Strongest separator first, and within a separator the LATEST break, so
    // chunks run as long as they may rather than as short as they can.
    // `lastIndexOf` returns a UTF-16 offset, so it is converted back to a code
    // point count before it is used as an index into `rest`.
    for (const sep of config.splitOn) {
      const at = head.lastIndexOf(sep);
      if (at > 0) {
        cut = [...head.slice(0, at)].length + [...sep].length;
        break;
      }
    }
    if (cut <= 0) {
      // No punctuation in a whole window's worth of text. Break at the last
      // word boundary; it will be heard, and that is the point.
      const at = head.lastIndexOf(" ");
      if (at > 0) cut = [...head.slice(0, at)].length;
    }
    if (cut <= 0) {
      cut = budget; // one unbroken token longer than a window: mid-word
    }
    // Never zero: a cut of 0 leaves `rest` unchanged and the loop spins
    // forever.
    cut = Math.max(cut, 1);

    chunks.push(rest.slice(0, cut).join("").trim());
    rest = [...rest.slice(cut).join("").replace(/^\s+/, "")];
  }
  return chunks.filter((c) => c.length > 0);
}
