/**
 * Where the funnel's data lives, and the grammar file parsed once.
 *
 * Four passes read `numbers.json` for different blocks: number grammars, date
 * and ordinal rules, letter names, unit words. Each spelling its own path to it
 * is how the parsed table and the hashed bytes drift apart, and this module is
 * where the paths already have to agree because it is also what hashes them.
 * Python reference: `loudkit/frontend/textconfig.py`.
 */

import grammarData from "../data/numbers.json" with { type: "json" };

/** Language rules, cardinals, months, ordinals, letter names, unit words. */
export const GRAMMAR_URL = new URL("../data/numbers.json", import.meta.url);

/**
 * The Polish English-respelling lexicon: 110k entries, 6.5 MB.
 *
 * Hashed alongside the grammar because it is a funnel input exactly as the
 * grammar is: it changes the spoken tokens, and an unhashed input would let a
 * build say different words under the same sixteen hex digits.
 */
export const RESPELL_URL = new URL("../data/pl_en_respell.json", import.meta.url);

/**
 * What every number character folds to, see `tools/make_numerals.py`.
 *
 * Hashed with the grammar and the lexicon for the same reason, and it also
 * pins the Unicode version the fold uses.
 */
export const NUMERALS_URL = new URL("../data/numerals.json", import.meta.url);

/**
 * The per-language blocks of the grammar file, located once and parsed once.
 *
 * `any` because the value shapes differ per key by design: a scale is an
 * object, `ones` an array, `word_join` a string, and a union describing all of
 * them would be longer than the four readers put together.
 */
export function grammarLanguages(): Record<string, any> {
  return (grammarData as { languages: Record<string, any> }).languages ?? {};
}
