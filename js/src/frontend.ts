/**
 * Text to text-tokens, a bit-parity port of `loudkit.frontend.text`.
 *
 * The pipeline is deliberately thin (lowercase, NFKD, a language tag, spaces
 * to `[SPACE]`, then plain BPE over Unicode scalars) and it is exactly what
 * the conformance fixture pins. The tokenizer JSON is the standard HF
 * `tokenizers` format; the JS port (`@huggingface/tokenizers`) is asserted
 * against the fixture's frontend vectors, so a drift in either side fails the
 * conformance run rather than producing plausible-but-wrong ids.
 */

import { Tokenizer } from "@huggingface/tokenizers";
import { readFileSync } from "node:fs";

import { UnsupportedLanguageError } from "./errors.js";
import { supportedNumberLanguages } from "./numbers.js";

const SPACE = "[SPACE]";

/**
 * The two characters the tokenizer's word split reads differently here.
 *
 * The vocabulary's pre-tokenizer is `Whitespace`, which keeps runs of word
 * characters and runs of symbols and drops whitespace between them. The
 * reference reads that rule through the Rust `tokenizers` crate, whose
 * whitespace is the Unicode `White_Space` property; this port reads it through
 * a JavaScript regex, whose `\s` is ECMAScript WhiteSpace plus LineTerminator.
 * Those two classes differ in exactly two characters, in opposite directions:
 * U+FEFF is whitespace only here, and U+0085 NEL is whitespace only there.
 *
 * So `encode("a\uFEFFb")` dropped the character Python, Go, Rust and Swift all
 * answer `[UNK]` for, and `encode("a\u0085b")` answered an `[UNK]` those four
 * all drop. Swapping the pair hands the tokenizer the class the reference would
 * have seen. The swap is a bijection between the two disputed classes, so a run
 * of symbols groups into the same word it groups into there; a one-way rewrite
 * would have fixed one direction and left the other.
 *
 * `speechText.ts` removes U+FEFF and folds U+0085 to a space before
 * `synthesize` ever reaches here, so this is the public `encode` alone.
 */
const DISPUTED_WHITESPACE = /[\uFEFF\u0085]/g;

function swapDisputedWhitespace(ch: string): string {
  return ch === "\uFEFF" ? "\u0085" : "\uFEFF";
}

/**
 * Refused languages whose refusal has a *specific* reason worth stating: their
 * upstream pipeline wants Cangjie codes, kana conversion, diacritisation, jamo
 * decomposition or stress marks, none of which this frontend carries. A subset
 * of "not on the roster", kept so the message can say why rather than just no.
 */
const NEEDS_MODEL_PREPROCESSING = new Set(["zh", "ja", "he", "ko", "ru"]);

/**
 * The allowlist: the twelve ids in `numbers.json`, the same roster Python's
 * `loudkit.frontend.numbers.supported_languages` reports.
 *
 * The roster is closed: the tokenizer's vocabulary carries tags for more
 * languages than have a frontend, and a tag outside this list still encodes.
 * `encode(text, "bg")` would NFKD-mangle Cyrillic into ids the model reads as
 * sounds it was never trained to make: no error, plausible audio, wrong
 * language.
 *
 * Read from the number grammars rather than restated here: one authority, so a
 * new grammar reaches the frontend without a second edit.
 */
export function supportedLanguages(): string[] {
  return supportedNumberLanguages();
}

export class GraphemeTextFrontend {
  private tokenizer: Tokenizer;

  constructor(tokenizerPath: string) {
    const blob = JSON.parse(readFileSync(tokenizerPath, "utf8"));
    this.tokenizer = new Tokenizer(blob, {});
    const vocab = new Set(Object.keys(blob.model.vocab));
    for (const required of ["[START]", "[STOP]", SPACE]) {
      if (!vocab.has(required)) {
        throw new Error(`${tokenizerPath}: vocabulary is missing '${required}'`);
      }
    }
  }

  encode(text: string, language = "en"): number[] {
    const lang = language.toLowerCase();
    const roster = supportedLanguages();
    if (!roster.includes(lang)) {
      const why = NEEDS_MODEL_PREPROCESSING.has(lang)
        ? "needs model-based text preprocessing " +
          "(Cangjie/kana/diacritics/jamo/stress) that this frontend does not carry"
        : "is not one of the languages this build's text layer is written for";
      throw new UnsupportedLanguageError(
        `language '${lang}' ${why}. Supported: ${roster.join(", ")}`
      );
    }
    let normalised = text.toLowerCase().normalize("NFKD");
    // Square brackets never reach the tokenizer from user text: the vocabulary
    // holds 117 bracket control tokens ([sigh], [gasp], the language tags) and
    // matches them greedily, so "he [sigh]ed" would make the model sigh. The
    // language tag added below is the one bracket that belongs.
    normalised = normalised.replace(/[[\]]/g, " ");
    normalised = normalised.replace(DISPUTED_WHITESPACE, swapDisputedWhitespace);
    const tagged = `[${lang}]${normalised}`.replace(/ /g, SPACE);
    return this.tokenizer.encode(tagged).ids;
  }
}
