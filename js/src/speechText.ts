/**
 * The speech funnel: a bit-parity port of the Swift engine's `SpeechText`.
 *
 * Before tokenising, the shipped engine scrubs the raw text: invisible
 * characters, symbols that carry meaning, footnote markers, and punctuation
 * (prosodic marks stay exactly where they are, because the model is a language
 * model trained on punctuated text; everything else becomes a space). For Polish it
 * then respells embedded English the way a Polish reader says it (the
 * 110k-word lexicon in `respell.ts`). This is the JS half of that contract;
 * the Python engine runs the same funnel in `Engine._synthesize_one` via
 * `loudkit.frontend.speechtext.speech_text`.
 * Python reference: `loudkit/frontend/speechtext.py`.
 */

import { readFileSync } from "node:fs";

import { expandDates, expandOrdinals } from "./dates.js";
import { spellAcronyms } from "./letters.js";
import {
  SCALE_SUFFIX_PATTERN,
  decimalSeparator,
  expandAbbreviations,
  expandNumbers,
  expandRomanNumerals,
  expandTimes,
  foldForeignDigits,
  scaleNouns,
  scaleSuffixWord,
} from "./numbers.js";
import { lexicalRespelling } from "./respell.js";
import { NUMERALS_URL, grammarLanguages } from "./textconfig.js";

const INVISIBLES = new Set(
  "\u200B\u200C\u200D\u2060\uFEFF\u00AD\u180E\u200E\u200F"
);

// Symbols the model cannot voice, in the order the pass replaces them. The
// first family (→ ✓ ✗ ≈ ≥) is literally outside the vocabulary; ¢ ° % $ do
// tokenize and are read at the ear's discretion. Both get words.
//
// Which word each takes is a per-language fact and comes from `unit_words` in
// the shared grammar; `SYMBOL_MARKS` carries the rest.
const SPOKEN_SYMBOLS: readonly string[] = [
  "%", "°", "¢", "€", "£", "¥", "₹",
  "×", "÷", "≈", "≥", "≤", "≠", "±",
  "→", "←", "⇒", "✓", "✔", "✗", "✘",
  "•", "·", "▪", "◦", "…", "&", "@",
];

// An arrow, a bullet and an ellipsis are the same pause in every language, so
// they are a rule here rather than a row in twelve grammars.
const SYMBOL_MARKS: Record<string, string> = {
  "→": ",",
  "←": ",",
  "⇒": ",",
  "•": ",",
  "·": ",",
  "▪": ",",
  "◦": ",",
  "…": "...",
};

/**
 * The ASCII spellings of the comparison operators, longest first, each named by
 * the mathematical symbol whose word it shares. Only these six: `-`, `/`, `.`
 * and `+` are ranges, paths, decimals and hyphens far more often than operators,
 * and a word put on one of them changes prose that reads correctly today.
 */
const ASCII_OPERATORS: readonly (readonly [string, string])[] = [
  ["<=", "≤"],
  [">=", "≥"],
  ["!=", "≠"],
  ["==", "="],
  ["<", "<"],
  [">", ">"],
];

/**
 * A markup tag, comment or declaration, which is not text anyone reads aloud.
 *
 * The name inside the angle brackets otherwise reaches the model as a word, and
 * `<!-- ... -->` additionally leaves its `!` behind as a sentence-final
 * exclamation. A tag is replaced by a space rather than by nothing, because two
 * block tags meeting back to back are two paragraphs and not one glued word.
 */
const MARKUP_TAG = /<\/?[A-Za-z!][^<>]*>/g;

// Symbol -> word per language, from the shared grammar file (numbers.json).
// One row per language on the roster, because a symbol is spoken in the
// language being read: "$5" in a German render says "5 Dollar", and `≈` says
// "ungefähr".
const UNIT_WORDS: Record<string, Record<string, string>> = Object.fromEntries(
  Object.entries(
    grammarLanguages() as Record<string, { unit_words?: Record<string, string> }>
  )
    .filter(([, entry]) => entry.unit_words)
    .map(([lang, entry]) => [lang, entry.unit_words as Record<string, string>])
);

/** The word `symbol` takes in `language`, or undefined when this language has
 * no wording for it. No fall back to English: a symbol is spoken in the
 * language being read or it is left written, which is what the funnel does
 * everywhere the evidence runs out. */
function unitWord(symbol: string, language: string): string | undefined {
  return UNIT_WORDS[language]?.[symbol];
}

// `$` and `£` before a number read as a prefix in writing and a SUFFIX in
// speech: "$5" is "five dollars", not "dollars five".
const CURRENCY_PREFIX_SYMBOLS = ["$", "£", "€", "¥", "₹"];

/**
 * Also `¢`, which nobody writes in front of a number: it is a suffix in every
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
 * neunundvierzig Dollar". Only a lone dot with a plain fraction is touched;
 * `$1,234.56` carries a grouping mark this cannot safely reinterpret.
 */
function priced(amount: string, language: string): string {
  const sep = decimalSeparator(language);
  if (sep === ".") return amount;
  if (/^\d+\.\d+$/.test(amount)) return amount.replace(".", sep);
  return amount;
}

function dropMarkupTags(text: string): string {
  if (!text.includes("<")) return text;
  return text.replace(MARKUP_TAG, " ");
}

/** The ASCII comparison operators, as words in this language. */
function speakOperators(text: string, language: string): string {
  return text.replace(OPERATOR_RUN, (written: string) => {
    for (const [spelling, symbol] of ASCII_OPERATORS) {
      if (spelling !== written) continue;
      // An operator no grammar covers stays written, like every other symbol
      // this module has no word for.
      return unitWord(symbol, language) ?? written;
    }
    return written;
  });
}

const scalePatterns = new Map<string, string>();

/**
 * The optional magnitude that may follow a currency amount, as a regex.
 *
 * Two alternatives and two groups: the abbreviating letter glued to the digits,
 * and the scale noun written beside them. Both cases of each noun are spelled
 * out rather than asked of a case-insensitive flag, because JavaScript has no
 * inline flag group and a pattern that needs one is a pattern the five
 * implementations cannot share.
 */
function scalePattern(language: string): string {
  const cached = scalePatterns.get(language);
  if (cached !== undefined) return cached;
  const written: string[] = [];
  for (const noun of scaleNouns(language)) {
    const cases = new Set([noun, noun.slice(0, 1).toUpperCase() + noun.slice(1)]);
    for (const form of cases) written.push(escapeRegex(form));
  }
  const pattern = written.length
    ? `(?:(${SCALE_SUFFIX_PATTERN})|${SPACE}(${written.join("|")}))(?![A-Za-z])`
    : "";
  scalePatterns.set(language, pattern);
  return pattern;
}

/**
 * The magnitude word standing between a price and its currency, or `""`.
 *
 * A written scale reaches speech in two shapes and both belong before the
 * currency word: the letter glued to the digits (`$2.5M`) and the noun beside
 * them (`$5 million`). The noun is already this language's own word and is kept
 * as written; the letter is a number, so the grammar's scale noun is asked for
 * the form this count takes.
 */
function scaleAfter(
  amount: string,
  suffix: string,
  spelled: string,
  language: string
): string {
  if (spelled) return spelled;
  if (!suffix) return "";
  const whole = amount.split(".")[0].split(",")[0].replace(/[^0-9]/g, "");
  return scaleSuffixWord(suffix, whole ? Number(whole) : 0, language) ?? "";
}

function speakSymbols(text: string, languageId: string): string {
  // A language without a wording table hears English rather than silence.
  const language = UNIT_WORDS[languageId] ? languageId : "en";
  let out = speakOperators(text, language);
  const scale = scalePattern(language);
  // Prefix currencies first, while the digits still follow the symbol.
  for (const symbol of CURRENCY_PREFIX_SYMBOLS) {
    const word = unitWord(symbol, language);
    if (word === undefined) continue;
    // The magnitude binds two groups, so a language with no scale noun at all
    // is handed the pattern without it and reads neither.
    const tail = scale ? `(?:${scale})?` : "";
    // A letter in front means a multi-character currency mark: `R$` is the
    // Brazilian real, `HK$` the Hong Kong dollar, and this table has a
    // wording for neither. Matching the `$` alone read `R$3,14` as "R3,14
    // Dollar", the wrong currency, said confidently.
    const re = new RegExp(
      `(?<!\\p{L})${escapeRegex(symbol)}${SPACE}?(${DIGIT}+(?:[.,]${DIGIT}+)*)${tail}`,
      "gu"
    );
    out = out.replace(re, (_m: string, ...rest: unknown[]) => {
      const amount = rest[0] as string;
      const suffix = tail ? (rest[1] as string | undefined) ?? "" : "";
      const spelled = tail ? (rest[2] as string | undefined) ?? "" : "";
      const said = scaleAfter(amount, suffix, spelled, language);
      const price = priced(amount, languageId);
      // A scale this language has no noun for. The suffix stays written, which
      // is what it did before the amount was moved.
      if (suffix && !said) return `${price} ${word}${suffix}`;
      return said ? `${price} ${said} ${word}` : `${price} ${word}`;
    });
  }
  // The same amount with the symbol behind it. `2.50 €` and `0.49¢` are prices by
  // exactly the evidence `€2.50` is, and reached the time pass with the dot intact:
  // German answered "zwei Uhr fünfzig Euro". Currency written as a *word*, such
  // as `5.50 zł`, is not covered: telling those from a unit needs a per-language
  // lexicon.
  for (const symbol of CURRENCY_SYMBOLS) {
    const word = unitWord(symbol, language);
    if (!word || !out.includes(symbol)) continue;
    const escaped = symbol.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp("(\\d+(?:[.,]\\d+)*)\\s?" + escaped, "g");
    out = out.replace(re, (_m, amount: string) => `${priced(amount, languageId)} ${word}`);
  }
  for (const symbol of SPOKEN_SYMBOLS) {
    if (!out.includes(symbol)) continue;
    const replacement = unitWord(symbol, language) ?? SYMBOL_MARKS[symbol];
    if (replacement === undefined) continue;
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

/**
 * Every number character becomes something a reader can say aloud.
 *
 * Python reference: `loudkit.frontend.speechtext.fold_numerals`, which carries
 * the reasoning. A non-ASCII decimal digit becomes the ASCII digit of the same
 * value and joins the run it was in; every other number character becomes the
 * text `numerals.json` names for it, separated from an adjacent alphanumeric so
 * `\u00b29` does not fold into `29` and read as twenty-nine.
 *
 * The table decides *whether* a character is a numeral too; see
 * `foldedNumeral`. Not `\p{N}`: that is V8's Unicode version rather than the
 * one the table was cut from.
 */
function foldNumerals(text: string): string {
  const chars = [...text];
  if (!chars.some((c) => foldedNumeral(c) !== null)) return text;
  const out: string[] = [];
  for (let i = 0; i < chars.length; i += 1) {
    const c = chars[i];
    const folded = foldedNumeral(c);
    if (folded === null) {
      out.push(c);
      continue;
    }
    const [raw, isDigit] = folded;
    // A slash inside a spelled numeral is a fraction bar, asserted by the
    // character itself: `½` is a half wherever it stands, where a typed `1/2` is
    // a fraction, a date or the `24/7` of ordinary prose. The division sign is
    // the mark the symbol table already has a word for in every language, so the
    // reading comes from the grammar and not from here.
    const spelled = raw.replaceAll("/", "÷");
    if (isDigit) {
      // A digit replacing a digit joins the run it was already in.
      out.push(spelled);
      continue;
    }
    // Inspect the last Unicode code point already emitted, not the last UTF-16
    // code unit: above the BMP that unit is a lone surrogate, which `\p{L}`
    // does not match.
    const previous = out.length ? [...out[out.length - 1]].pop() ?? "" : "";
    if (previous && isLetterOrDigit(previous)) out.push(" ");
    out.push(spelled);
    if (i + 1 < chars.length && isLetterOrDigit(chars[i + 1])) out.push(" ");
  }
  return out.join("");
}

/** `\p{L}` or `\p{Nd}`: the word class all five ports test. */
const LETTER_OR_DIGIT = /[\p{L}\p{Nd}]/u;

function isLetterOrDigit(c: string): boolean {
  return LETTER_OR_DIGIT.test(c);
}

/**
 * What `foldNumerals` puts in a numeral's place and whether it is a decimal
 * digit, or `null` for a character the table does not name.
 *
 * Both halves come from the shared `numerals.json`: which characters are
 * numerals, and what each one folds to. Neither is computed from the runtime's
 * own Unicode tables, because V8's move with the runtime and Python's with the
 * interpreter, and the five ports report one fingerprint. The table is cut from
 * one pinned UCD and hashed into the grammar digest.
 *
 * `null` for an ASCII digit, a letter or an ideographic numeral: all left
 * exactly as written.
 */
function foldedNumeral(c: string): [string, boolean] | null {
  const code = c.codePointAt(0) as number;
  if (code >= 0x30 && code <= 0x39) return null;
  const table = numeralTable();
  const spelled = table.spelled.get(code);
  if (spelled !== undefined) return [spelled, false];
  const zeros = table.decimalZeros;
  let low = 0;
  let high = zeros.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (zeros[mid] <= code) low = mid + 1;
    else high = mid;
  }
  if (low > 0) {
    const offset = code - zeros[low - 1];
    if (offset >= 0 && offset <= 9) return [String(offset), true];
  }
  return null;
}

interface NumeralTable {
  decimalZeros: number[];
  spelled: Map<number, string>;
}

let cachedNumerals: NumeralTable | null = null;

/** `numerals.json`, parsed once. */
function numeralTable(): NumeralTable {
  if (cachedNumerals === null) {
    const raw = JSON.parse(
      readFileSync(NUMERALS_URL, "utf8")
    ) as { decimal_zeros: number[]; spelled: Record<string, string> };
    cachedNumerals = {
      decimalZeros: raw.decimal_zeros,
      spelled: new Map(Object.entries(raw.spelled).map(([k, v]) => [Number(k), v])),
    };
  }
  return cachedNumerals;
}

/**
 * Unicode White_Space, written out, for every regex in this funnel.
 *
 * Python reference: `loudkit.frontend.speechtext.WHITE_SPACE`. ECMAScript
 * `\\s` excludes U+0085 NEL, which is ordinary in scraped and epub text, so a
 * separator that survives here and nowhere else changes where this port's
 * reader breathes.
 */
const WHITE_SPACE = "\u0009\u000a\u000b\u000c\u000d\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000";

/** `\s` as this funnel means it: the characters above and nothing else. */
const SPACE = `[${WHITE_SPACE}]`;

/**
 * `\d` as this funnel means it. Only ASCII digits reach the number grammars, so
 * the class is spelled out and the five implementations agree by construction
 * rather than by coincidence.
 */
const DIGIT = "[0-9]";

/**
 * An ASCII operator with whitespace on both sides.
 *
 * The spacing is the evidence that the mark is an operator and not markup or an
 * emoticon: `<p>`, `</div>`, `<3` and `a<b` all keep the mark written, and the
 * funnel leaves written what it cannot read.
 */
const OPERATOR_RUN = new RegExp(`(?<=${SPACE})(<=|>=|!=|==|<|>)(?=${SPACE})`, "gu");

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function dropFootnoteMarkers(text: string): string {
  if (!text.includes("[")) return text;
  // `[0-9]` and the explicit space class, not `\\d` and `\\s`: ECMAScript `\\s`
  // excludes U+0085 NEL, so a marker separated by one survived here and was
  // then read aloud by the number pass.
  return text.replace(new RegExp(`\\[[0-9${WHITE_SPACE},;\\-–—]{1,20}\\]`, "gu"), "");
}

/**
 * Whitespace, as the other four ports see it.
 *
 * ECMAScript's `\s` is WhiteSpace plus LineTerminator, and U+0085 NEXT LINE is
 * in neither, so JS alone turned it into a space where Python, Go, Rust and
 * Swift keep it. U+0085 is what CP1252 byte 0x85 becomes when text is decoded
 * as Latin-1, which is ordinary in scraped and epub sources.
 */
function isSpace(sc: string): boolean {
  return /\s/.test(sc) || sc === "\u0085";
}

function punctuationForSpeech(text: string): string {
  const scalars = [...text];
  let out = "";
  for (let i = 0; i < scalars.length; i++) {
    const sc = scalars[i];
    if (
      isLetter(sc) ||
      isDecimal(sc) ||
      isSpace(sc) ||
      PROSODIC.has(sc)
    ) {
      out += sc;
      continue;
    }
    const prev = i > 0 ? scalars[i - 1] : null;
    const next = i + 1 < scalars.length ? scalars[i + 1] : null;
    // Between digits, "." and "," are numeric separators and "-" and "/" are
    // ranges and fractions: meaning, not decoration.
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
 * Prepare `text` to be spoken in `languageId`, the same funnel the shipped
 * Swift engine runs as `SpeechText.prepared`. Order matters and is deliberate:
 * invisible characters first, symbols while digits are intact, footnote
 * markers before punctuation, punctuation last.
 */
export function speechText(text: string, languageId?: string | null): string {
  // The language id is lowercased once, here, and again in the respeller.
  // `GraphemeTextFrontend` lowercases its own tag, so "PL" produced Polish
  // *tokens* while silently skipping the Polish respelling: the same utterance
  // read half one way and half the other, with nothing to indicate it. Python
  // fixed this in `loudkit.frontend.speechtext.speech_text`, and Swift's
  // `LexicalRespelling.applied` carries the same `.lowercased()` with a
  // comment explaining why.
  languageId = languageId?.toLowerCase() ?? null;
  // Normalize to NFC before any symbol or pronunciation rule inspects a
  // character, the same opening pass the Python funnel runs. Unicode lets one
  // character arrive two ways, Polish ą as U+0105 or as a + U+0328, and the
  // tokenizer's vocabulary holds one of them, so every pattern, lexicon lookup
  // and character class below matches that one.
  //
  // Ahead of `stripInvisibles`, which removes format characters: normalisation
  // can compose a sequence into a single character, and running it afterwards
  // would leave that composition unexamined.
  // `foldForeignDigits` sits beside NFC and before the symbol pass, so the
  // folded percent sign reaches the table that turns it into a word.
  let out = stripInvisibles(foldForeignDigits(text.normalize("NFC"), languageId ?? "en"));
  // Before the symbol pass, which would otherwise read a tag's angle brackets as
  // comparison operators and its attributes as text.
  out = dropMarkupTags(out);
  // Before the symbol table: see the Python reference. Every pass downstream
  // asks "is this a digit" and the five ports spell it four ways, so folding
  // first means all of them see ASCII.
  out = foldNumerals(out);
  out = speakSymbols(out, languageId ?? "en");
  out = dropFootnoteMarkers(out);
  // Before the acronym pass, which spells a Roman numeral letter by letter, and
  // after the numeral fold, which is what turns `Ⅳ` into the `IV` this pass
  // reads.
  out = expandRomanNumerals(out, languageId ?? "en");
  // Acronyms while the capitals are still capitals: every later pass lowercases
  // or rewrites, and a spelled acronym has to be decided while the only evidence
  // (that the word stands alone in caps) still exists. The pass belongs here
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
  // Numbers after footnotes and before punctuation: see the Python funnel
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
  out = out.replace(new RegExp(`[${WHITE_SPACE}]+([.,;:!?])`, "gu"), "$1");
  // Two clause marks in a row is one clause mark.
  // A run, not a pair: `replace` does not overlap its matches, so a pair rule
  // turns "..." into ".." on one pass and "." on the next.
  out = out.replace(new RegExp(`([.,;:])(?:[${WHITE_SPACE}]*[.,;:])+`, "gu"), "$1");
  // `trim()` excludes U+0085 NEL, the same gap as everywhere else here.
  return out.replace(new RegExp(`^[${WHITE_SPACE}]+|[${WHITE_SPACE}]+$`, "gu"), "");
}
