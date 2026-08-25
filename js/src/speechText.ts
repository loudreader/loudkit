/**
 * The speech funnel — a bit-parity port of the Swift engine's `SpeechText`.
 *
 * Before tokenising, the shipped engine scrubs the raw text: invisible
 * characters, symbols that carry meaning, footnote markers, and punctuation
 * (prosodic marks stay exactly where they are — the model is a language model
 * trained on punctuated text — everything else becomes a space). For Polish it
 * then respells embedded English the way a Polish reader says it (the
 * 110k-word lexicon in `respell.ts`). This is the JS half of that contract;
 * the Python engine runs the same funnel in `Engine._synthesize_one` via
 * `loudkit.frontend.polish.speech_text`.
 * Python reference: `loudkit/frontend/polish.py`.
 */

import { expandDates, expandOrdinals } from "./dates.js";
import { spellAcronyms } from "./letters.js";
import { lexicalRespelling } from "./respell.js";

const INVISIBLES = new Set(
  "\u200B\u200C\u200D\u2060\uFEFF\u00AD\u180E\u200E\u200F"
);

// Symbols the model cannot voice, as words: (en, pl). The first family
// (→ ✓ ✗ ≈ ≥) is literally outside the vocabulary; ¢ ° % $ do tokenize and
// are read at the ear's discretion. Both get words.
const SYMBOL_WORDS: Record<string, [string, string]> = {
  "%": ["percent", "procent"],
  "°": ["degrees", "stopni"],
  "¢": ["cents", "centów"],
  "€": ["euro", "euro"],
  "£": ["pounds", "funtów"],
  "¥": ["yen", "jenów"],
  "₹": ["rupees", "rupii"],
  "×": ["times", "razy"],
  "÷": ["divided by", "podzielone przez"],
  "≈": ["about", "około"],
  "≥": ["at least", "co najmniej"],
  "≤": ["at most", "najwyżej"],
  "≠": ["not equal to", "różne od"],
  "±": ["plus minus", "plus minus"],
  "→": [",", ","],
  "←": [",", ","],
  "⇒": [",", ","],
  "✓": ["yes", "tak"],
  "✔": ["yes", "tak"],
  "✗": ["no", "nie"],
  "✘": ["no", "nie"],
  "•": [",", ","],
  "·": [",", ","],
  "▪": [",", ","],
  "◦": [",", ","],
  "…": ["...", "..."],
  "&": ["and", "i"],
  "@": ["at", "małpa"],
};

// `$` and `£` before a number read as a prefix in writing and a SUFFIX in
// speech: "$5" is "five dollars", not "dollars five".
// Symbol -> word per language, from the shared grammar file (numbers.json).
// The old table was an (en, pl) pair with `pl if polish else en`, which meant
// seven of the nine languages heard English: "$5" in a German render said
// "5 dollars".
import unitWordData from "../data/numbers.json" with { type: "json" };
import {
  decimalSeparator,
  expandAbbreviations,
  expandNumbers,
  expandTimes,
  foldForeignDigits,
} from "./numbers.js";

const UNIT_WORDS: Record<string, Record<string, string>> = Object.fromEntries(
  Object.entries(
    (unitWordData as { languages: Record<string, { unit_words?: Record<string, string> }> })
      .languages
  )
    .filter(([, entry]) => entry.unit_words)
    .map(([lang, entry]) => [lang, entry.unit_words as Record<string, string>])
);

/** The word `symbol` takes in `language`, falling back to English so a symbol
 * is at least said, if with an accent. */
function unitWord(symbol: string, language: string): string | undefined {
  return UNIT_WORDS[language]?.[symbol] ?? UNIT_WORDS.en?.[symbol];
}

const CURRENCY_PREFIX_SYMBOLS = ["$", "£", "€", "¥", "₹"];

/**
 * Also `¢`, which nobody writes in front of a number — it is a suffix in every
 * convention, which is why the prefix pass never saw it and `0.49¢` reached the
 * clock reader with its dot intact.
 */
const CURRENCY_SYMBOLS = [...CURRENCY_PREFIX_SYMBOLS, "¢"];

// Punctuation that carries prosody stays; the rest becomes a space.
const PROSODIC = new Set(".,!?;:\u2014\u2013\u2026\"\u201C\u201D\u201E«»()'\u2019\u00BF\u00A1");

function stripInvisibles(text: string): string {
  let seen = false;
  for (const ch of text) {
    if (INVISIBLES.has(ch)) {
      seen = true;
      break;
    }
  }
  if (!seen) return text;
  let out = "";
  for (const ch of text) {
    if (!INVISIBLES.has(ch)) out += ch;
  }
  return out;
}

/**
 * A currency amount, with its decimal mark spelled the way `language` does.
 *
 * The one place a dot between digits is known not to be a clock time, and the
 * last place that knows it: by the time pass the symbol has become a trailing
 * word and `$0.49` is indistinguishable from `14.30`, which in the eleven
 * comma-decimal languages is how a time is written. German answered "null Uhr
 * neunundvierzig Dollar". Only a lone dot with a plain fraction is touched —
 * `$1,234.56` carries a grouping mark this cannot safely reinterpret.
 */
function priced(amount: string, language: string): string {
  const sep = decimalSeparator(language);
  if (sep === ".") return amount;
  if (/^\d+\.\d+$/.test(amount)) return amount.replace(".", sep);
  return amount;
}

function speakSymbols(text: string, languageId: string): string {
  // A language without a wording table hears English rather than silence.
  const language = UNIT_WORDS[languageId] ? languageId : "en";
  const polish = language === "pl";
  let out = text;
  // Prefix currencies first, while the digits still follow the symbol.
  for (const symbol of CURRENCY_PREFIX_SYMBOLS) {
    const word = unitWord(symbol, language);
    if (word === undefined) continue;
    const escaped = symbol.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    // A letter in front means a multi-character currency mark: `R$` is the
    // Brazilian real, `HK$` the Hong Kong dollar, and this table has a
    // wording for neither. Matching the `$` alone read `R$3,14` as "R3,14
    // Dollar" — the wrong currency, said confidently.
    const re = new RegExp("(?<!\\p{L})" + escaped + "\\s?(\\d+(?:[.,]\\d+)*)", "gu");
    out = out.replace(re, (_m, amount: string) => `${priced(amount, languageId ?? "en")} ${word}`);
  }
  // The same amount with the symbol behind it. `2.50 €` and `0.49¢` are prices by
  // exactly the evidence `€2.50` is, and reached the time pass with the dot intact:
  // German answered "zwei Uhr fünfzig Euro". Currency written as a *word* — `5.50
  // zł` — is not covered; telling those from a unit needs a per-language lexicon.
  for (const symbol of CURRENCY_SYMBOLS) {
    const word = unitWord(symbol, language);
    if (!word || !out.includes(symbol)) continue;
    const escaped = symbol.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp("(\\d+(?:[.,]\\d+)*)\\s?" + escaped, "g");
    out = out.replace(re, (_m, amount: string) => `${priced(amount, languageId ?? "en")} ${word}`);
  }
  for (const [symbol, [en, pl]] of Object.entries(SYMBOL_WORDS)) {
    if (!out.includes(symbol)) continue;
    // Not every symbol is a per-language word (arrows, ticks): the old pair
    // table still carries those.
    const replacement = unitWord(symbol, language) ?? (polish ? pl : en);
    // A word replacement needs spaces around it; a punctuation one must not
    // gain a space BEFORE it or the comma floats.
    const spaced =
      replacement.length === 1 && ",.".includes(replacement)
        ? replacement + " "
        : " " + replacement + " ";
    out = out.split(symbol).join(spaced);
  }
  return out;
}

function dropFootnoteMarkers(text: string): string {
  if (!text.includes("[")) return text;
  return text.replace(/\[[\d\s,;\-–—]{1,20}\]/g, "");
}

function punctuationForSpeech(text: string): string {
  const scalars = [...text];
  let out = "";
  for (let i = 0; i < scalars.length; i++) {
    const sc = scalars[i];
    if (
      isLetter(sc) ||
      isDecimal(sc) ||
      /\s/.test(sc) ||
      PROSODIC.has(sc)
    ) {
      out += sc;
      continue;
    }
    const prev = i > 0 ? scalars[i - 1] : null;
    const next = i + 1 < scalars.length ? scalars[i + 1] : null;
    // Between digits, "." and "," are numeric separators and "-" and "/" are
    // ranges and fractions — meaning, not decoration.
    const betweenDigits =
      prev !== null && isDecimal(prev) && next !== null && isDecimal(next);
    if (betweenDigits && "-/:.".includes(sc)) {
      out += sc;
      continue;
    }
    // A hyphen inside a token is part of the token ("well-known", "1e-3").
    // Either end alphanumeric, not both letters: a both-letters test leaves the
    // exponent in "1e-3" to become a space, so the model is handed "1e 3" after
    // the number pass has already declined to read it.
    // `+` alongside `-`: the number pass declines "1e+3" as a token with a
    // letter in it, and punctuation would otherwise take it apart into "1e 3".
    if (
      (sc === "-" || sc === "+") &&
      prev !== null &&
      (isLetter(prev) || isDigit(prev)) &&
      next !== null &&
      (isLetter(next) || isDigit(next))
    ) {
      out += sc;
      continue;
    }
    out += " ";
  }
  return out;
}

function isLetter(ch: string): boolean {
  return /[\p{L}]/u.test(ch);
}

function isDigit(ch: string): boolean {
  return /[\p{Nd}]/u.test(ch);
}

function isDecimal(ch: string): boolean {
  return /[\p{Nd}]/u.test(ch);
}

/**
 * Prepare `text` to be spoken in `languageId` — the same funnel the shipped
 * Swift engine runs as `SpeechText.prepared`. Order matters and is deliberate:
 * invisible characters first, symbols while digits are intact, footnote
 * markers before punctuation, punctuation last.
 */
export function speechText(text: string, languageId?: string | null): string {
  // The language id is lowercased once, here, and again in the respeller.
  // `GraphemeTextFrontend` lowercases its own tag, so "PL" produced Polish
  // *tokens* while silently skipping the Polish respelling — the same utterance
  // read half one way and half the other, with nothing to indicate it. Python
  // fixed this in `loudkit.frontend.polish.speech_text`, and Swift's
  // `LexicalRespelling.applied` carries the same `.lowercased()` with a
  // comment explaining why.
  languageId = languageId?.toLowerCase() ?? null;
  // NFC first, before anything inspects a character — the same opening pass the
  // Python funnel runs, and the one this funnel did not have.
  //
  // Unicode lets the same character arrive two ways: Polish ą as U+0105 or as
  // a + U+0328, Danish å as U+00E5 or a + U+030A. The tokenizer's vocabulary
  // holds one of them, so a decomposed spelling reaches it as a base letter
  // followed by an unknown combining mark — and every rule below, every pattern
  // and lexicon lookup and character class, is matching a string nobody wrote a
  // rule for.
  //
  // Ahead of `stripInvisibles`, which removes format characters: normalisation
  // can compose a sequence into a single character, and running it afterwards
  // would leave that composition unexamined.
  // Beside NFC, and before the symbol pass so the folded percent sign reaches
  // the table that turns it into a word.
  let out = stripInvisibles(foldForeignDigits(text.normalize("NFC"), languageId ?? "en"));
  out = speakSymbols(out, languageId ?? "en");
  out = dropFootnoteMarkers(out);
  // Acronyms while the capitals are still capitals: every later pass lowercases
  // or rewrites, and a spelled acronym has to be decided while the only evidence
  // — that the word stands alone in caps — still exists. The pass belongs here
  // rather than in `respell.ts`: a Polish-only table there spells `FBI`
  // *ef-be-i* in a Polish render and leaves the model raw graphemes in the
  // other eleven.
  out = spellAcronyms(out, languageId ?? "en");
  // Dates before times and numbers, and this ordering is the whole reason the
  // pass exists: `12.03.2026` is the ordinary written date of five of these
  // languages, and both passes below want a piece of it. The clock pattern
  // matches `12.03` and the digit run matches the lot, so a date recognised any
  // later has already been eaten and read as a time with a stray year.
  out = expandDates(out, languageId ?? "en");
  // Ordinals before numbers, for the same reason: the number pass expands the
  // digits and leaves the suffix stuck to them, so `1st` arrived as *onest*.
  out = expandOrdinals(out, languageId ?? "en");
  // Numbers after footnotes and before punctuation — see the Python funnel
  // for the ordering argument; the fixture pins it.
  out = expandAbbreviations(out, languageId ?? "en");
  out = expandTimes(out, languageId ?? "en");
  out = expandNumbers(out, languageId ?? "en");
  out = punctuationForSpeech(out);
  // Polish: respell embedded English the way a Polish reader says it. This is
  // the shipped engine's LexicalRespelling; see respell.ts.
  out = lexicalRespelling(out, languageId);
  out = out.replace(/[ \t]{2,}/g, " ");
  // A symbol that became a comma inherits the space that sat in
  // front of it ("0.49 → 0.24" would read "zero point four nine ,").
  out = out.replace(/\s+([.,;:!?])/g, "$1");
  // Two clause marks in a row is one clause mark.
  // A run, not a pair: `replace` does not overlap its matches, so a pair rule
  // turns "..." into ".." on one pass and "." on the next.
  out = out.replace(/([.,;:])(?:[\s]*[.,;:])+/g, "$1");
  return out.trim();
}
