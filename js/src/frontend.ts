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

// A tab or line break together with its indentation contributes one word
// boundary. Plain ASCII spaces each have their own token in the trained input.
const WORD_SPACES = /[ \t-\r\u0085\u2028\u2029]+/g;

function spaceTokens(run: string): string {
  return run.length === 1 || /[^ ]/.test(run) ? SPACE : SPACE.repeat(run.length);
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
    normalised = normalised.replace(WORD_SPACES, spaceTokens);
    // In a direct encode, Python treats U+FEFF as an unknown symbol. The JS
    // tokenizer's whitespace class drops it, so use its unknown U+0085 proxy.
    normalised = normalised.replace(/\uFEFF/g, "\u0085");
    const tagged = `[${lang}]${normalised}`;
    return this.tokenizer.encode(tagged).ids;
  }
}
