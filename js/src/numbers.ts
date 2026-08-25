/**
 * Numbers, said out loud — the TypeScript half of `loudkit.frontend.numbers`.
 *
 * The grammar is data and only the interpreter is code: this module reads the
 * same numbers.json every other implementation reads, so a rule lives once.
 * The composition mirrors loudkit/frontend/numbers.py function for function; the
 * reasons behind the odd-looking behaviours (joiners carrying their own
 * spacing, per-value agreement scopes, a scale noun with its own gender) live
 * in the Python docstrings and docs/reference/preprocess.md, and the hand-written
 * fixture plus the 1300-row CLDR differential pin them.
 * Python reference: `loudkit/frontend/numbers.py`.
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import grammarData from "../data/numbers.json" with { type: "json" };

interface Scale {
  value: number;
  forms: string[];
  /** "~" composes the multiplier; "" uses the bare scale word; anything else
   * is the literal one-word (German "eine", Italian "un"). */
  oneWord: string;
  separate: boolean;
  link: string;
  smallJoiner: string;
  multiplierAgrees: boolean;
  multiplierGender: string;
}

interface Grammar {
  ones: string[];
  teens: string[];
  tens: string[];
  hundred: string;
  hundreds: string[];
  hundredsGendered: Record<string, string[]>;
  hundredPluralFinal: string;
  scales: Scale[];
  unitsBeforeTens: boolean;
  unitTensJoiner: string;
  timeInfix: string;
  abbreviations: [string, string][];
  tensJoinerExceptions: Record<number, string>;
  hundredJoiner: string;
  scaleJoinerOnRoundHundreds: boolean;
  scaleLargeJoiner: string;
  oneBeforeHundred: boolean;
  wordJoin: string;
  minusWord: string;
  decimalSeparator: string;
  decimalWord: string;
  exceptions: Record<number, string>;
  genders: Record<string, Record<number, string>>;
  genderScopes: Record<number, string>;
  combiningOnes: Record<number, string>;
}

 
function parseGrammar(e: any): Grammar {
  return {
    ones: e.ones ?? [],
    teens: e.teens ?? [],
    tens: e.tens ?? [],
    hundred: e.hundred ?? "",
    hundreds: e.hundreds ?? [],
    hundredsGendered: e.hundreds_gendered ?? {},
    hundredPluralFinal: e.hundred_plural_final ?? "",
    scales: (e.scales ?? []).map((sc: any) => ({
      value: sc.value,
      forms: sc.forms,
      oneWord: sc.one ?? "~",
      separate: sc.separate ?? false,
      link: sc.link ?? "",
      smallJoiner: sc.small_joiner ?? "",
      multiplierAgrees: sc.multiplier_agrees ?? false,
      multiplierGender: sc.multiplier_gender ?? "",
    })),
    unitsBeforeTens: e.units_before_tens ?? false,
    unitTensJoiner: e.unit_tens_joiner ?? "",
    timeInfix: e.time_infix ?? "",
    abbreviations: Object.entries((e.abbreviations ?? {}) as Record<string, string>).sort(
      (a, b) => b[0].length - a[0].length // longest first: fr.o.m. stays whole
    ),
    tensJoinerExceptions: e.tens_joiner_exceptions ?? {},
    hundredJoiner: e.hundred_joiner ?? "",
    scaleJoinerOnRoundHundreds: e.scale_joiner_on_round_hundreds ?? false,
    scaleLargeJoiner: e.scale_large_joiner ?? "",
    oneBeforeHundred: e.one_before_hundred ?? false,
    wordJoin: e.word_join ?? "",
    minusWord: e.minus_word ?? "",
    decimalSeparator: e.decimal_separator ?? ",",
    decimalWord: e.decimal_word ?? "",
    exceptions: e.exceptions ?? {},
    genders: e.genders ?? {},
    genderScopes: e.gender_scopes ?? {},
    combiningOnes: e.combining_ones ?? {},
  };
}
 

const GRAMMARS: Record<string, Grammar> = Object.fromEntries(
  Object.entries(
    (grammarData as { languages: Record<string, unknown> }).languages
  ).map(([lang, entry]) => [lang, parseGrammar(entry)])
);

/** Language ids `cardinal` can verbalize, sorted. */
export function supportedNumberLanguages(): string[] {
  return Object.keys(GRAMMARS).sort();
}

/**
 * The form `value` takes in `gender` at `position`, or undefined when it does
 * not inflect. Position is "standalone" (the whole number), "tail" (ends a
 * larger number) or "tens_pair" (inside the solid compound).
 */
function gendered(
  g: Grammar,
  value: number,
  gender: string,
  position: string
): string | undefined {
  if (!gender) return undefined;
  const scope = g.genderScopes[value];
  if (scope === "standalone" && position !== "standalone") return undefined;
  if (scope === "outside_tens" && position === "tens_pair") return undefined;
  return g.genders[gender]?.[value];
}

/**
 * `value` as words. An empty gender gives the citation form. An unknown
 * language or a value past the grammar's largest scale throws — silently
 * reading digits back would be indistinguishable from success.
 */
export function cardinal(value: number, language: string, gender = ""): string {
  const g = GRAMMARS[language];
  if (g === undefined) {
    throw new Error(`no number grammar for ${language}`);
  }
  const ceiling = g.scales.length > 0 ? g.scales[0].value * 1000 : 1000;
  if (Math.abs(value) >= ceiling) {
    throw new Error(
      `${value} is past the largest scale ${language} has a word for`
    );
  }
  if (value < 0) {
    // Always a spaced word, even in solid-writing languages: minus eins.
    return `${g.minusWord} ${cardinal(-value, language, gender)}`;
  }
  // Standalone agreement applies to the whole number only: Polish jedna
  // alone, but sto jeden.
  const standalone = gendered(g, value, gender, "standalone");
  if (standalone !== undefined) return standalone;
  return compose(value, g, gender, false);
}

function compose(
  value: number,
  g: Grammar,
  gender: string,
  asMultiplier: boolean
): string {
  const listed = g.exceptions[value];
  if (listed !== undefined) return listed;
  if (value < 100) return belowHundred(value, g, gender, asMultiplier);
  for (const sc of g.scales) {
    if (value >= sc.value) return scaleGroup(value, sc, g, gender);
  }
  return hundredsGroup(value, g, gender);
}

function scaleGroup(
  value: number,
  sc: Scale,
  g: Grammar,
  gender: string
): string {
  const count = Math.floor(value / sc.value);
  const rest = value % sc.value;
  const join = sc.separate ? " " : g.wordJoin;
  const linkDefault = sc.link || join;

  let head: string;
  if (count === 1 && sc.oneWord !== "~") {
    head = sc.oneWord
      ? `${sc.oneWord}${join}${scaleWord(1, sc.forms)}`
      : scaleWord(1, sc.forms);
  } else {
    // Whether the counted noun's gender reaches the multiplier is a fact
    // about the scale noun: Portuguese "duas mil", Polish "dwa tysiące".
    const mg = sc.multiplierGender || (sc.multiplierAgrees ? gender : "");
    head = `${compose(count, g, mg, true)}${join}${scaleWord(count, sc.forms)}`;
  }
  if (rest === 0) return head;

  const roundHundreds =
    g.scaleJoinerOnRoundHundreds && rest >= 100 && rest % 100 === 0;
  let link: string;
  if (sc.smallJoiner && (rest < 100 || roundHundreds)) {
    link = ` ${sc.smallJoiner} `;
  } else if (rest >= 100 && count >= 100 && g.scaleLargeJoiner) {
    link = g.scaleLargeJoiner;
  } else {
    link = linkDefault;
  }
  return `${head}${link}${compose(rest, g, gender, false)}`;
}

function scaleWord(count: number, forms: string[]): string {
  if (forms.length === 1 || count === 1) return forms[0];
  // singular / plural: Million / Millionen
  if (forms.length === 2) return forms[1];
  const lastTwo = count % 100;
  const last = count % 10;
  if (last >= 2 && last <= 4 && !(lastTwo >= 12 && lastTwo <= 14)) {
    return forms[1];
  }
  return forms[2];
}

function hundredsGroup(value: number, g: Grammar, gender: string): string {
  const count = Math.floor(value / 100);
  const rest = value % 100;
  const parts: string[] = [];
  const hundreds = (gender && g.hundredsGendered[gender]) || g.hundreds;
  if (hundreds.length > 0) {
    parts.push(hundreds[count - 1]);
  } else if (count === 1 && !g.oneBeforeHundred) {
    parts.push(g.hundred);
  } else {
    parts.push(compose(count, g, "", true));
    // French deux cents / deux cent un: the plural mark appears only when the
    // multiplied hundred ends the number.
    if (count > 1 && rest === 0 && g.hundredPluralFinal) {
      parts.push(g.hundredPluralFinal);
    } else {
      parts.push(g.hundred);
    }
  }
  if (rest !== 0) {
    if (g.hundredJoiner) parts.push(g.hundredJoiner);
    parts.push(belowHundred(rest, g, gender, false));
  }
  return parts.filter((p) => p).join(g.wordJoin);
}

function unitWord(
  value: number,
  g: Grammar,
  gender: string,
  asMultiplier: boolean
): string {
  const agreed = gendered(g, value, gender, asMultiplier ? "tens_pair" : "tail");
  if (agreed !== undefined) return agreed;
  if (asMultiplier) {
    const combining = g.combiningOnes[value];
    if (combining !== undefined) return combining;
  }
  return g.ones[value];
}

function belowHundred(
  value: number,
  g: Grammar,
  gender: string,
  asMultiplier: boolean
): string {
  const fixed = gendered(g, value, gender, "tail") ?? g.exceptions[value];
  if (fixed !== undefined) return fixed;
  if (value < 10) return unitWord(value, g, gender, asMultiplier);
  if (value < 20) return g.teens[value - 10];

  const ten = Math.floor(value / 10);
  const unit = value % 10;
  const tenWord = gendered(g, ten * 10, gender, "tail") ?? g.tens[ten - 2];
  if (unit === 0) return tenWord;

  // A unit inside a tens pair is always in composition: einundzwanzig holds
  // even when the pair ends the number.
  const unitW = unitWord(unit, g, gender, true);
  const joiner = g.tensJoinerExceptions[value] ?? g.unitTensJoiner;
  if (g.unitsBeforeTens) return `${unitW}${joiner}${tenWord}`;
  return `${tenWord}${joiner}${unitW}`;
}

// ASCII digits only, explicitly — see the Python module for why.
/**
 * The funnel's code version, bumped when the passes change what they emit for
 * text they already handled. A new language or table moves the digest instead.
 */
export const TEXT_RECIPE = "funnel-2";

/**
 * First 16 hex characters of the SHA-256 of the grammar file followed by the
 * respelling lexicon, as raw bytes — like every other implementation, so the
 * five agree only when they ship the same files.
 *
 * The lexicon is hashed alongside the grammar because it is a funnel input
 * exactly as the grammar is and it changes the spoken tokens, so both files
 * hash into the fingerprint. Leaving the lexicon out covers 55 KB of rules but
 * not 6.5 MB of vocabulary.
 */
export function grammarDigest(): string {
  if (cachedDigest === null) {
    const grammar = readFileSync(new URL("../data/numbers.json", import.meta.url));
    const respell = readFileSync(new URL("../data/pl_en_respell.json", import.meta.url));
    cachedDigest = createHash("sha256")
      .update(grammar)
      .update(respell)
      .digest("hex")
      .slice(0, 16);
  }
  return cachedDigest;
}

let cachedDigest: string | null = null;

/**
 * Python's `_DIGIT_RUN` — JS has lookbehind, so the shape ports over; only the
 * word class is spelled out rather than abbreviated.
 *
 * The three parts of the pattern are each audible: a run glued to a
 * word is part of that word (`iOS18` reads as *iOSeighteen*), a minus in front of
 * digits belongs to the number (`-5` reads as *five*), and space-grouped
 * thousands are one number (`1 000` reads as *one zero zero zero*).
 *
 * The guards spell the word class out because JS `\w` is `[A-Za-z0-9_]` and
 * stays ASCII even under the `u` flag, while Python's is every alphanumeric
 * character there is. That difference was the whole of a parity break: `é2`
 * read as *étwo* here and stayed written in the other four, and `zł200 000`
 * read as *złdoscientos mil*.
 */
const DIGIT_RUN =
  /(?<![\p{L}\p{N}_])(-(?=[0-9]))?([0-9]{1,3}(?: [0-9]{3})+(?! ?[0-9])|[0-9]+)((?:[.,][0-9]+)*)(?![\p{L}\p{N}_])/gu;

/** A letter in any script: Python's `str.isalpha`, which is not `[a-zA-Z]`. */
const WALK_LETTER = /\p{L}/u;

/**
 * What the walks step over: Python's `str.isalnum() or c in "_.,-+"` — the
 * characters an identifier puts between its letters and its digits.
 */
const WALK_CHAR = /[\p{L}\p{N}_.,\-+]/u;

/**
 * Whether three digits start at `i` — the shape `DIGIT_RUN` binds as a group
 * after the first, and so the shape a space in front of them may be grouping.
 */
function startsAGroup(text: string, i: number): boolean {
  return /^[0-9]{3}$/.test(text.slice(i, i + GROUP_DIGITS));
}

/**
 * ...and no fourth digit behind them, so the group is one the pattern could
 * have bound rather than a ragged run that only looks like one.
 */
function continuesAGroup(text: string, i: number): boolean {
  return startsAGroup(text, i) && !/[0-9]/.test(text[i + GROUP_DIGITS] ?? "");
}

/**
 * Whether the token continues past the match into a letter.
 *
 * The mirror of `gluedToAWord`, needed for the same reason: `200 000x` matches
 * `200` alone, because the grouped alternative reaches the `x` and the right-hand
 * guard refuses it, so the regex backtracks to the first group and reads "two
 * hundred 000x". Go and Rust, which do not backtrack, leave the whole token. A
 * grouping space is crossed so `200 000x` is one token; the ordinary space in
 * `2024 200 people` is not, because what follows it is a word.
 */
function gluedForward(text: string, end: number): boolean {
  for (let i = end; i < text.length; i += 1) {
    const c = text[i];
    if (WALK_LETTER.test(c)) return true;
    if (WALK_CHAR.test(c)) continue;
    // Three digits after the space and the walk crosses it, a fourth digit
    // notwithstanding: `startsAGroup` where the backward walk asks
    // `continuesAGroup`. The asymmetry is the measurement. Forwards the walk
    // finishes the run the pattern *refused* to bind, and a ragged group is why
    // it refused — `1 0023R` matched the `1` alone and read "en 0023R", half a
    // run spoken and the rest welded to a letter, which is the class the
    // right-hand guard exists to stop. Backwards the group is the match itself,
    // whose width the pattern already fixed, and the same looseness there
    // swallows the `1000` of `e3 1000`.
    //
    // Three digits and not fewer, so the walk stops where the run stops:
    // `1000 5.1e+3` keeps its `1000` rather than crossing into the exponent two
    // tokens away, and the `5` of `R2 5 iOS` is its own number.
    if (c === " " && /[0-9]/.test(text[i - 1] ?? "") && startsAGroup(text, i + 1)) {
      continue;
    }
    return false;
  }
  return false;
}

/**
 * Whether a decimal point with digits behind it follows the match.
 *
 * A decimal point with digits behind the match means the fraction group shrank
 * to zero so the right-hand guard could land on the dot instead of a letter:
 * `1.5e3` matched just the `1` and read "one.5e3".
 */
function truncatedByAFraction(text: string, end: number): boolean {
  return end + 1 < text.length && ".,".includes(text[end]) && /[0-9]/.test(text[end + 1]);
}

/**
 * Whether the digit run at `start` sits inside a token containing a letter —
 * Python's backward walk over word characters and dots, which is the question
 * its one-character lookbehind could not ask. In `v1.2.3` the scan starts at
 * the `2`, because a dot precedes it, and the version came out
 * "v1.two point three".
 */
function gluedToAWord(text: string, start: number): boolean {
  for (let i = start - 1; i >= 0; i -= 1) {
    const c = text[i];
    // `-` and `+` are in the walk because an exponent puts one between the
    // letter and the digits: in `1e-3` the scan starts at the `3`, walks back
    // over `-` to `e`, and stops calling it a number.
    //
    // A *grouping* space is crossed too, and only under the same strictness as
    // the non-backtracking ports. `x200 000` binds as a single match in Go and
    // Rust, whose engines do not backtrack, so their lookbehind refuses the
    // whole run; a backtracking engine that matched the standalone `000` reads
    // "x200 zero zero zero" — half a token spoken, which is the class the
    // right-hand guard exists to stop.
    //
    // Exactly three digits behind the space and no fourth — the only shape the
    // pattern binds across one, judged by the group the walk steps *out of*,
    // plus a digit behind the space. The looser shapes each break on a real
    // input: "a digit on each side" crosses into the `R` of `R2 5`, which is not
    // a grouped number; "exactly three digits behind" alone breaks `a1 000 000`,
    // whose first group is legitimately one digit; dropping the digit-behind
    // test lets the walk cross space after space, so `Sold 200 000` reaches
    // "Sold" and refuses a number nothing was glued to; and admitting a fourth
    // digit — which is what the forward walk does — reaches the `e` of
    // `e3 1000` and welds two tokens into one.
    const groupingSpace =
      c === " " &&
      /[0-9]/.test(text[i - 1] ?? "") &&
      continuesAGroup(text, i + 1);
    // The letter test is first because the walk's own class contains every
    // letter: asked second, and written in ASCII as it was, it never saw an `é`.
    if (WALK_LETTER.test(c)) return true;
    if (!groupingSpace && !WALK_CHAR.test(c)) return false;
  }
  return false;
}

/**
 * The mark `language` writes between a whole number and its fraction.
 *
 * Exported because the speech funnel needs it outside the number pass: a
 * currency amount is the one place a dot between digits is known not to be a
 * clock time, and the funnel must say so while the symbol is still in hand.
 */
export function decimalSeparator(language: string): string {
  return GRAMMARS[language]?.decimalSeparator ?? ".";
}

/**
 * Foreign digit systems and their separators, as this language spells them.
 *
 * Beside NFC because it is the same kind of pass: one spelling for every pass
 * that follows, and early enough that the symbol table still sees the folded
 * percent sign.
 *
 * Language-dependent for the separators, and that is not a detail. U+066B is a
 * *decimal* separator, so folding it to a dot everywhere turned `٣٫١٤` into
 * `3.14` — which in the eleven languages that write decimals with a comma is
 * the written form of a clock time, read out as *drei Uhr vierzehn*.
 */
export function foldForeignDigits(text: string, language: string): string {
  const decimal = GRAMMARS[language]?.decimalSeparator ?? ".";
  const grouping = decimal === "." ? "," : ".";
  let out = "";
  for (const c of text) {
    const code = c.codePointAt(0) ?? 0;
    if (code >= 0x0660 && code <= 0x0669) out += String(code - 0x0660);
    else if (code >= 0x06f0 && code <= 0x06f9) out += String(code - 0x06f0);
    else if (code === 0x066b) out += decimal;
    else if (code === 0x066c) out += grouping;
    else if (code === 0x066a) out += "%";
    else out += c;
  }
  return out;
}

/**
 * An E.164 telephone number, read digit by digit and taken before the digit
 * run, which cannot decline it: `+48 123 456 789` is a valid
 * one-to-three-then-threes grouping and read as a cardinal it is forty-eight
 * billion. The plus is the evidence — E.164 requires one and a grouped thousand
 * never carries one.
 */
const PHONE_RUN = /\+[0-9][0-9 ]*[0-9]/g;

/** ISO 8601's 24:00. Admitted as an hour, and only with a zero minute. */
/** Digits in a thousands group: every group after the first is exactly this. */
const GROUP_DIGITS = 3;

const END_OF_DAY_HOUR = 24;

/** Below this a plus-signed run is a delta, not a telephone number. */
const MIN_E164_DIGITS = 8;

/**
 * U+2212 MINUS SIGN and U+2010 HYPHEN, folded to ASCII where a digit follows.
 * Everything downstream reads the sign as `-`, so a typographic minus was not a
 * sign at all: it reached the punctuation pass, became a space, and `−5` was
 * read as *five*. Not U+2013, which writes a range.
 */
const UNICODE_MINUS = /[\u2212\u2010](?=[0-9])/g;

/**
 * Every run of digits in `text`, said as words — the seam between the
 * verbalizer and the funnel. Never throws and never leaves digits behind: a
 * number past every scale is read digit by digit (it is almost always an
 * identifier), and only the language's own decimal mark is a decimal mark —
 * the other one is grouping, and is dropped the way a reader drops it.
 */
export function expandNumbers(text: string, language: string): string {
  const g = GRAMMARS[language];
  if (g === undefined) return text;
  // Both before anything looks for a digit run: the sign has to be ASCII by the
  // time the pattern matches one, and a phone number has to be gone before the
  // grouping rule meets a shape it cannot decline.
  const folded = text.replace(UNICODE_MINUS, "-").replace(PHONE_RUN, (whole) => {
    const digits = [...whole].filter((c) => c >= "0" && c <= "9");
    if (digits.length < MIN_E164_DIGITS) return whole;
    return digits.map((d) => cardinal(Number(d), language)).join(" ");
  });
  return folded.replace(
    DIGIT_RUN,
    (
      whole: string,
      sign: string | undefined,
      digits: string,
      fraction: string,
      offset: number,
      source: string,
    ) => {
      const end = offset + whole.length;
      if (gluedToAWord(source, offset) || gluedForward(source, end) || truncatedByAFraction(source, end))
        return whole;
      // Normalised once, here, so everything downstream sees one shape: a sign
      // kept apart from the digits, and thousands spaces gone. Mirrors Python's
      // `say` in `expand_numbers`.
      const literal = digits.replaceAll(" ", "") + (fraction ?? "");
      if (!isQuantity(literal, g)) return whole;
      const said = sayNumber(literal, g, language);
      return sign && g.minusWord ? `${g.minusWord} ${said}` : said;
    },
  );
}

/**
 * Whether a digit run is a number rather than a version, an address or a date.
 *
 * `1.2.3`, `192.168.0.1` and `12.03.2026` all match the digit-run pattern and
 * none is a quantity. Reading one as a quantity says "nineteen million two
 * hundred sixteen thousand eight hundred one" for an IP address — and in the
 * Python reference is a hard crash.
 *
 * A run is a quantity when it has at most one separator, or when its separators
 * genuinely group: every segment after the first exactly three digits, the
 * first one to three. Anything else is left as written.
 */
function isQuantity(literal: string, g: Grammar): boolean {
  const grouping = g.decimalSeparator === "." ? "," : ".";
  const cut = literal.indexOf(g.decimalSeparator);
  const whole = cut === -1 ? literal : literal.slice(0, cut);
  const fraction = cut === -1 ? "" : literal.slice(cut + g.decimalSeparator.length);
  // A second mark in what should be the fraction: the split happens once, so
  // this is where "1.2.3" left "2.3" behind and the reference crashed on it.
  if (fraction.includes(grouping) || fraction.includes(g.decimalSeparator)) return false;
  const segments = whole.split(grouping);
  if (segments.length === 1) return true;
  const grouped =
    segments[0].length >= 1 &&
    segments[0].length <= 3 &&
    segments.slice(1).every((seg) => seg.length === 3);
  if (grouped) return true;
  // Two segments and no fraction is the "2.5 GB" shape: the mark that is not
  // this language's decimal separator, used as one anyway.
  return segments.length === 2 && cut === -1;
}

function sayNumber(literal: string, g: Grammar, language: string): string {
  // The non-decimal mark is only grouping when it groups: every following
  // segment exactly three digits. Polish "1.000" is a thousand; Polish "2.5"
  // is a de-facto decimal, and 2.5 read as 25 is a changed meaning.
  const grouping = g.decimalSeparator === "." ? "," : ".";
  const sep = literal.indexOf(g.decimalSeparator);
  let whole = sep < 0 ? literal : literal.slice(0, sep);
  let fraction = sep < 0 ? "" : literal.slice(sep + 1);
  const segments = whole.split(grouping);
  if (segments.length > 1) {
    if (segments.slice(1).every((s) => s.length === 3)) {
      whole = segments.join("");
    } else if (!fraction && segments.length === 2) {
      [whole, fraction] = segments;
    } else {
      whole = segments.join("");
    }
  }
  fraction = fraction.split(grouping).join("");

  const parts = [sayInteger(whole, language)];
  if (fraction) {
    parts.push(g.decimalWord);
    // Digit by digit — "point four nine", never "point forty-nine": leading
    // zeros carry meaning there that a cardinal would eat.
    parts.push(...digitByDigit(fraction, language));
  }
  return parts.join(" ");
}

function sayInteger(digits: string, language: string): string {
  // Leading zeros mean a code, not a quantity: 0042 is zero zero four two.
  if (digits.length > 1 && digits.startsWith("0")) {
    return digitByDigit(digits, language).join(" ");
  }
  const n = Number(digits);
  if (Number.isSafeInteger(n)) {
    try {
      return cardinal(n, language);
    } catch {
      // past the largest scale: fall through to digit-by-digit
    }
  }
  return digitByDigit(digits, language).join(" ");
}

function digitByDigit(digits: string, language: string): string[] {
  return [...digits].map((ch) => cardinal(ch.charCodeAt(0) - 48, language));
}


// The lookarounds are the point: `\b` alone let this match *inside* a longer
// dotted run, so `12.03.2026` — the ordinary written date of German, Polish,
// Danish, Finnish and Norwegian — matched `12.03` and was read as twelve
// o'clock three with the year trailing behind it. A time is a time only when
// nothing is attached to either end; `14:30.` at the end of a sentence still
// matches, because what follows the dot is not a digit.
const TIME_RUN = /(?<![\d.,:])([01]?[0-9]|2[0-4]):([0-5][0-9])(?![.,:]?\d)/g;

/**
 * `14.30`, which is a clock time in some of these languages and a decimal in
 * others — applied only where the language says it is a time.
 *
 * A language that writes clock times with a dot does not use the dot as its
 * decimal separator. German writes `14.30 Uhr` and `2,50 €`; English writes
 * `2:30` and `$2.50`. So this applies exactly where `decimalSeparator` is not
 * `.`, which today means everywhere but English — before which every English
 * decimal with two fraction digits read as the clock (`$0.49` as *zero
 * forty-nine*, `3.14` as *three fourteen*), and the shared fixture pinned one
 * of them, so all five implementations agreed on it.
 */
const DOTTED_TIME_RUN = /(?<![\d.,:])([01]?[0-9]|2[0-4])\.([0-5][0-9])(?![.,:]?\d)/g;

/**
 * The two clock-time patterns, extended to consume a written infix word.
 *
 * German writes the time *with* the word the spoken form also carries:
 * `um 14.30 Uhr`. The reading puts the infix where it belongs — between hour
 * and minutes, *vierzehn Uhr dreißig* — so the written `Uhr` is that same
 * spoken token, not an additional one, and leaving it standing said it twice.
 * When the source carries the infix immediately after the time, the match
 * swallows it and the normal reading supplies the one copy.
 *
 * Every piece is spelled out because five implementations must match
 * identically: the whitespace run is ASCII space and tab (regex engines
 * disagree on what `\s` covers), the guard refuses an ASCII letter or digit
 * so *Uhrzeit* keeps its word whole, and case matters — the grammar data says
 * `Uhr` and this rule does not reach past that.
 */
function timePatterns(timeInfix: string): { timeRun: RegExp; dottedTimeRun: RegExp } {
  const suffix = timeInfix
    ? `(?:[ \\t]+${escapeRegex(timeInfix)}(?![0-9A-Za-z]))?`
    : "";
  return {
    timeRun: new RegExp(TIME_RUN.source + suffix, "g"),
    dottedTimeRun: new RegExp(DOTTED_TIME_RUN.source + suffix, "g"),
  };
}

/** Clock times as words — see the Python module for the shape. */
export function expandTimes(text: string, language: string): string {
  const g = GRAMMARS[language];
  if (g === undefined) return text;
  const { timeRun, dottedTimeRun } = g.timeInfix
    ? timePatterns(g.timeInfix)
    : { timeRun: TIME_RUN, dottedTimeRun: DOTTED_TIME_RUN };
  const say = (whole: string, h: string, m: string): string => {
    // 24 is admitted only with a zero minute: ISO 8601 writes end-of-day as
    // 24:00, and without it the two halves were read as unrelated numbers with
    // the colon left standing between them. 24:30 is not a time in any
    // convention and stays as written.
    if (Number(h) === END_OF_DAY_HOUR && Number(m) !== 0) return whole;
    const words = [cardinal(Number(h), language)];
    if (g.timeInfix) words.push(g.timeInfix);
    if (Number(m) !== 0) words.push(cardinal(Number(m), language));
    return words.join(" ");
  };
  const out = text.replace(timeRun, say);
  // A dot means a time only where it does not already mean a decimal point.
  return g.decimalSeparator === "." ? out : out.replace(dottedTimeRun, say);
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** The authority-listed abbreviations, written out — see the Python module. */
export function expandAbbreviations(text: string, language: string): string {
  const g = GRAMMARS[language];
  if (g === undefined || g.abbreviations.length === 0) return text;
  let out = text;
  for (const [written, spoken] of g.abbreviations) {
    const re = new RegExp(`(^|[^\\w.])${escapeRegex(written)}($|[^\\w.])`, "g");
    out = out.replace(re, `$1${spoken}$2`);
  }
  return out;
}
