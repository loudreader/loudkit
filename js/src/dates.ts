/**
 * Dates and ordinals, said the way each language says them.
 *
 * A port of `loudkit.frontend.dates`: `12.03.2026` is the ordinary
 * written date of five of these twelve languages, and without this funnel it
 * reads as a clock time with a stray year, or as one eight-digit number. `1st`
 * arrives as *onest*, because the number pass expands the digits and leaves the
 * suffix stuck to them.
 *
 * Every rule is data from the shared numbers.json — month names, day forms, the
 * infixes Spanish and Portuguese speak between the parts, the German oblique
 * triggers, the ordinal tables. What is code here is the *shape*: which written
 * forms are dates at all, and how each language reads a year.
 *
 * Two refusals are as deliberate as anything it does. A yearless `12.3.` is
 * never matched — its closing period is indistinguishable from a sentence's, so
 * `Die Zahl ist 3.5.` would otherwise come out as *dritte Mai*. And `3/12/2026` is left alone in
 * English, where it is March twelfth to half the world and the third of December
 * to the other half: a listener recovers from hearing digits, not from a
 * confident wrong month.
 * Python reference: `loudkit/frontend/dates.py`.
 */

import grammarData from "../data/numbers.json" with { type: "json" };
import { cardinal } from "./numbers.js";

/** Above this a four-digit run is an identifier, not a year. */
const MAX_YEAR = 2999;
/** A three-digit year exists; a three-digit *anything* is far more often a
 * quantity, and nothing in the string separates them. */
const MIN_YEAR = 1000;
/** February is 29 on purpose: a plausibility bound, not a calendar. Refusing 29
 * February in a common year would reject a date a human wrote deliberately. */
const DAYS_IN_MONTH = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

interface Rules {
  dayWords: Record<number, string>;
  dayWordsOblique: Record<number, string>;
  obliqueTriggers: string[];
  dayOneWord: string;
  months: string[];
  dayMonthInfix: string;
  monthYearInfix: string;
  dayFirstPrefix: string;
  dayFirstInfix: string;
  yearRule: string;
  yearUnits: Record<number, string>;
  yearTeens: Record<number, string>;
  yearTens: Record<number, string>;
  yearTwoThousand: string;
  dottedIsAmbiguous: boolean;
  noDottedDates: boolean;
  ordSuffixes: string[];
  ordUnits: Record<number, string>;
  ordTeens: Record<number, string>;
  ordTens: Record<number, string>;
  ordJoiner: string;
}

const RULES: Record<string, Rules> = (() => {
  const intKeys = (raw: unknown): Record<number, string> => {
    const out: Record<number, string> = {};
    for (const [k, v] of Object.entries((raw as Record<string, string>) ?? {})) {
      const n = Number.parseInt(k, 10);
      if (Number.isFinite(n) && v) out[n] = v;
    }
    return out;
  };
  const out: Record<string, Rules> = {};
  const languages =
    (grammarData as { languages: Record<string, any> }).languages ?? {};
  for (const [lang, entry] of Object.entries(languages)) {
    const d = entry?.dates;
    if (!d) continue;
    const o = entry?.ordinals ?? {};
    out[lang] = {
      dayWords: intKeys(d.day_words),
      dayWordsOblique: intKeys(d.day_words_oblique),
      obliqueTriggers: (d.oblique_triggers as string[]) ?? [],
      dayOneWord: d.day_one_word ?? "",
      months: (d.months as string[]) ?? [],
      dayMonthInfix: d.day_month_infix ?? "",
      monthYearInfix: d.month_year_infix ?? "",
      dayFirstPrefix: d.day_first_prefix ?? "",
      dayFirstInfix: d.day_first_infix ?? "",
      yearRule: d.year_rule ?? "",
      yearUnits: intKeys(d.year_units),
      yearTeens: intKeys(d.year_teens),
      yearTens: intKeys(d.year_tens),
      yearTwoThousand: d.year_two_thousand ?? "",
      dottedIsAmbiguous: Boolean(d.dotted_is_ambiguous),
      noDottedDates: Boolean(d.no_dotted_dates),
      ordSuffixes: (o.suffixes as string[]) ?? [],
      ordUnits: intKeys(o.units),
      ordTeens: intKeys(o.teens),
      ordTens: intKeys(o.tens),
      ordJoiner: o.tens_joiner || "-",
    };
  }
  return out;
})();

const card = (n: number, lang: string): string => cardinal(n, lang) ?? "";

/** The month's name in this language, or `null` when it has no table. */
export function monthName(month: number, language: string): string | null {
  const r = RULES[language];
  if (!r || month < 1 || month > 12 || r.months.length !== 12) return null;
  return r.months[month - 1];
}

/**
 * The day-of-month word, in whatever form this language's dates take.
 *
 * `oblique` is German only — the `-en` ending that `am`/`den`/`vom` select.
 */
export function ordinalDay(
  day: number,
  language: string,
  oblique = false
): string | null {
  const r = RULES[language];
  if (!r || day < 1 || day > 31) return null;
  if (oblique && r.dayWordsOblique[day]) return r.dayWordsOblique[day];
  if (r.dayWords[day]) return r.dayWords[day];
  // Cardinal languages: the day is just a number, except where the first of the
  // month is lexicalised.
  if (day === 1 && r.dayOneWord) return r.dayOneWord;
  return card(day, language);
}

/**
 * A year, read the way this language reads years.
 *
 * English and Norwegian split it; German, Dutch and Swedish group it in
 * hundreds; the rest say one plain cardinal. Spanish is the explicit case — the
 * RAE writes that a year is read as its cardinal and *not* in two-figure blocks
 * as in English, so 2021 is *dos mil veintiuno*.
 */
export function sayYear(year: number, language: string): string {
  const r = RULES[language];
  if (!r) return card(year, language);
  switch (r.yearRule) {
    case "en_split": return yearEnglish(year);
    case "de_hundreds": return yearHundreds(year, "de", "hundert", 1100, 1999);
    case "nl_hundreds": return yearHundreds(year, "nl", "honderd", 1100, 1999);
    case "sv_hundreds": return yearHundreds(year, "sv", "hundra", 1100, 2099);
    case "no_split": return yearNorwegian(year);
    case "da_long": return yearDanish(year);
    case "pl_ordinal_genitive": return yearPolish(year, r);
    default: return card(year, language);
  }
}

function yearEnglish(year: number): string {
  if (year === 1000 || year === 2000 || (year >= 2001 && year <= 2009)) {
    return card(year, "en");
  }
  if ((year > 1000 && year < 2000) || year >= 2100) {
    const century = Math.floor(year / 100), rest = year % 100;
    if (rest === 0) return `${card(century, "en")} hundred`;
    // "nineteen oh five" — never "nineteen five", which nobody says.
    if (rest < 10) return `${card(century, "en")} oh ${card(rest, "en")}`;
    return `${card(century, "en")} ${card(rest, "en")}`;
  }
  if (year >= 2010 && year <= 2099) return `twenty ${card(year % 100, "en")}`;
  return card(year, "en");
}

/**
 * German, Dutch and Swedish all write `<century><joiner><rest>` solid; only the
 * joiner and the range differ. German stops at 1999 because the GfdS explicitly
 * rejects `zwanzighundert…`; Swedish runs to 2099 because Isof has recommended
 * the `tjugohundra…` series for decades.
 */
function yearHundreds(
  year: number, lang: string, joiner: string, lo: number, hi: number
): string {
  if (year < lo || year > hi) return card(year, lang);
  const century = Math.floor(year / 100), rest = year % 100;
  const head = `${card(century, lang)}${joiner}`;
  return rest === 0 ? head : `${head}${card(rest, lang)}`;
}

/** Norwegian splits 1100–1999 and drops `hundre`: 1972 is `nittensyttito`. */
function yearNorwegian(year: number): string {
  if (year < 1100 || year > 1999) return card(year, "no");
  const century = Math.floor(year / 100), rest = year % 100;
  if (rest === 0) return `${card(century, "no")}hundre`;
  return `${card(century, "no")}${card(rest, "no")}`;
}

/**
 * Dansk Sprognævn: the long form works for every year, and the short
 * "telephone-number" form is explicitly poor for a century's first decade.
 */
function yearDanish(year: number): string {
  if (year < 1100 || year > 1999) return card(year, "da");
  const century = Math.floor(year / 100), rest = year % 100;
  const head = `${card(century, "da")} hundrede`;
  return rest === 0 ? head : `${head} og ${card(rest, "da")}`;
}

/**
 * Only the tens and units of a Polish year decline. PWN's worked example is
 * *tysiąc dziewięćset dziewięćdziesiątego drugiego*: the thousands and hundreds
 * keep their cardinal form and the ordinal genitive lands on the last two
 * digits. Where those are zero the declension moves left, which is why 2000 has
 * its own word.
 */
function yearPolish(year: number, r: Rules): string {
  if (year === 2000 && r.yearTwoThousand) return r.yearTwoThousand;
  const head = Math.floor(year / 100), rest = year % 100;
  const lead = head !== 0 ? card(head * 100, "pl") : "";
  if (rest === 0) return lead;
  let tail: string;
  if (r.yearTeens[rest]) {
    tail = r.yearTeens[rest]!;
  } else {
    tail = [r.yearTens[Math.floor(rest / 10) * 10] ?? "", r.yearUnits[rest % 10] ?? ""]
      .filter(Boolean)
      .join(" ");
  }
  return `${lead} ${tail}`.trim();
}

function valid(day: number, month: number, year: number | null): boolean {
  if (month < 1 || month > 12) return false;
  if (day < 1 || day > DAYS_IN_MONTH[month - 1]) return false;
  return year === null || (year >= MIN_YEAR && year <= MAX_YEAR);
}

function spoken(
  day: number, month: number, year: number | null, language: string, oblique: boolean
): string | null {
  const r = RULES[language];
  if (!r) return null;
  const dayWord = ordinalDay(day, language, oblique);
  const monthWord = monthName(month, language);
  if (dayWord === null || monthWord === null) return null;
  const parts = [dayWord];
  if (r.dayMonthInfix) parts.push(r.dayMonthInfix);
  parts.push(monthWord);
  if (year !== null) {
    if (r.monthYearInfix) parts.push(r.monthYearInfix);
    parts.push(sayYear(year, language));
  }
  return parts.join(" ");
}

const ISO = /(?<![\d.,:/-])([12][0-9]{3})-([01][0-9])-([0-3][0-9])(?![\d-])/g;
/** With the year, which is what makes it a date rather than a guess. The
 * yearless `12.3.` is deliberately not matched — see the module note. */
const DOTTED = /(?<![\d.,:/-])([0-3]?[0-9])\.([01]?[0-9])\.([12][0-9]{3})\b/g;
/** Day-first in every language here; English is handled in the callback, where
 * the field order is genuinely ambiguous. */
const SLASHED = /(?<![\d.,:/-])([0-3]?[0-9])\/([01]?[0-9])\/([12][0-9]{3})(?![\d/])/g;

/**
 * Every written date in `text`, said the way `language` says it.
 *
 * Never throws and never invents: a run failing the bounds check, or whose field
 * order cannot be resolved, comes back exactly as it was written.
 */
export function expandDates(text: string, language: string): string {
  const r = RULES[language];
  if (!r) return text;

  let out = replace(text, ISO, (g, at, whole) => {
    const [y, m, d] = [Number(g[1]), Number(g[2]), Number(g[3])];
    if (!valid(d, m, y)) return null;
    return spoken(d, m, y, language, isOblique(whole, at, r));
  });
  out = replace(out, DOTTED, (g, at, whole) => {
    // Swedish marks an ordinal with a colon (`1:a`), never a trailing period, so
    // `12.` there is a list number or a sentence end. English writes dotted
    // dates almost never, and when it does the field order is as unresolvable as
    // in the slashed form.
    if (r.noDottedDates || r.dottedIsAmbiguous) return null;
    const [d, m, y] = [Number(g[1]), Number(g[2]), Number(g[3])];
    if (!valid(d, m, y)) return null;
    return spoken(d, m, y, language, isOblique(whole, at, r));
  });
  out = replace(out, SLASHED, (g, at, whole) => {
    const [d, m, y] = [Number(g[1]), Number(g[2]), Number(g[3])];
    // `3/12/2026` is March twelfth to half the English-speaking world and the
    // third of December to the other half, and nothing says which.
    if (language === "en" && d <= 12) return null;
    if (!valid(d, m, y)) return null;
    return spoken(d, m, y, language, isOblique(whole, at, r));
  });
  return textual(out, language, r);
}

/**
 * `12 marca 2026`, `12. März 2026`, `March 12, 2026` — a written month name
 * beside a bare day. The name is the disambiguator, so this runs for every
 * language including English.
 */
function textual(text: string, language: string, r: Rules): string {
  if (r.months.length !== 12) return text;
  const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const names = r.months.map(esc).join("|");
  // Spanish and Portuguese speak a preposition between every part, so the
  // written form carries it too: "12 de marzo de 2026".
  const infix = r.dayMonthInfix ? `(?:\\s+${esc(r.dayMonthInfix)})?` : "";
  const yinfix = r.monthYearInfix ? `(?:\\s+${esc(r.monthYearInfix)})?` : "";

  const dayFirst = new RegExp(
    `(?<![\\w])([0-3]?[0-9])\\.?${infix}\\s+(${names})(?:${yinfix}\\s+([12][0-9]{3}))?(?!\\w)`,
    "gi"
  );
  let out = replace(text, dayFirst, (g, at, whole) => {
    const d = Number(g[1]);
    const m = monthIndex(g[2] ?? "", r);
    const y = g[3] ? Number(g[3]) : null;
    if (m === null || !valid(d, m, y)) return null;
    if (r.dayFirstPrefix || r.dayFirstInfix) {
      // English written day-first reads "the twelfth of March": both dialects
      // say it that way, so no locale flag is needed.
      const head = ordinalDay(d, language);
      const monthWord = monthName(m, language);
      if (head === null || monthWord === null) return null;
      const rest = [monthWord];
      if (y !== null) rest.push(sayYear(y, language));
      const prefix = r.dayFirstPrefix ? `${r.dayFirstPrefix} ` : "";
      const join = r.dayFirstInfix ? ` ${r.dayFirstInfix} ` : " ";
      return `${prefix}${head}${join}${rest.join(" ")}`;
    }
    return spoken(d, m, y, language, isOblique(whole, at, r));
  });

  // Month-first is an English shape. Reading it in a language that never writes
  // it would be inventing a construction nobody used.
  if (!r.dayFirstInfix) return out;
  const monthFirst = new RegExp(
    `(?<![\\w])(${names})\\s+([0-3]?[0-9])(?:(?:st|nd|rd|th)\\b)?,?(?:\\s+([12][0-9]{3}))?(?!\\w)`,
    "gi"
  );
  out = replace(out, monthFirst, (g) => {
    const m = monthIndex(g[1] ?? "", r);
    const d = Number(g[2]);
    const y = g[3] ? Number(g[3]) : null;
    if (m === null || !valid(d, m, y)) return null;
    const monthWord = monthName(m, language);
    const dayWord = ordinalDay(d, language);
    if (monthWord === null || dayWord === null) return null;
    const parts = [monthWord, dayWord];
    if (y !== null) parts.push(sayYear(y, language));
    return parts.join(" ");
  });
  return out;
}

function monthIndex(name: string, r: Rules): number | null {
  const lowered = name.toLowerCase();
  for (let i = 0; i < r.months.length; i++) {
    if (r.months[i].toLowerCase() === lowered) return i + 1;
  }
  return null;
}

/** German only: `am`/`den`/`vom` before the day select the `-en` ending. */
function isOblique(whole: string, at: number, r: Rules): boolean {
  if (r.obliqueTriggers.length === 0) return false;
  const before = whole.slice(0, at).replace(/\s+$/, "");
  const fields = before.split(/\s+/).filter(Boolean);
  const last = fields[fields.length - 1];
  if (last === undefined) return false;
  const tail = last.toLowerCase().replace(/^[,;:]+|[,;:]+$/g, "");
  return r.obliqueTriggers.some((w) => w.toLowerCase() === tail);
}

/**
 * `value` as a written-out ordinal, or `null` when this language has no table.
 *
 * Composed rather than enumerated past ninety-nine: the hundreds and above stay
 * cardinal and only the last two digits become an ordinal, so *101st* is "one
 * hundred and first".
 */
export function ordinal(value: number, language: string): string | null {
  const r = RULES[language];
  if (!r || Object.keys(r.ordUnits).length === 0 || value < 0) return null;
  const head = Math.floor(value / 100), rest = value % 100;
  const tail = twoDigitOrdinal(rest, r);
  if (tail === null) return null;
  if (head === 0) return tail;
  const lead = card(head * 100, language);
  return rest !== 0 ? `${lead} ${tail}` : lead;
}

function twoDigitOrdinal(value: number, r: Rules): string | null {
  if (r.ordTeens[value]) return r.ordTeens[value];
  const tens = Math.floor(value / 10), units = value % 10;
  if (units === 0) return r.ordTens[tens * 10] ?? null;
  if (tens === 0) return r.ordUnits[units] ?? null;
  const unitWord = r.ordUnits[units];
  if (unitWord === undefined) return null;
  // The tens word is English's, because English is the only language of the
  // twelve writing an ordinal as digits plus a suffix.
  return `${card(tens * 10, "en")}${r.ordJoiner}${unitWord}`;
}

/**
 * `1st` and `22nd` as words.
 *
 * English is the only one of the twelve writing an ordinal as digits plus a
 * letter suffix, so for every other language this is a no-op. It runs before the
 * number pass, which would otherwise expand the digits and leave the suffix
 * stuck to them: *onest*, *fiveth place*, *twenty-twond*.
 */
export function expandOrdinals(text: string, language: string): string {
  const r = RULES[language];
  if (!r || r.ordSuffixes.length === 0) return text;
  const re = new RegExp(`\\b([0-9]+)(${r.ordSuffixes.join("|")})\\b`, "gi");
  return replace(text, re, (g) => ordinal(Number(g[1]), language));
}

/**
 * Rewrite every match, right to left so earlier offsets stay valid.
 *
 * The callback gets the capture groups (index 0 is the whole match), the match
 * offset, and the string being scanned — the last two because the German oblique
 * test reads the word *before* the date. Returning `null` leaves that match
 * exactly as written, which is this module's answer whenever evidence runs out.
 */
function replace(
  text: string,
  re: RegExp,
  body: (groups: (string | undefined)[], at: number, whole: string) => string | null
): string {
  const matches = [...text.matchAll(re)];
  if (matches.length === 0) return text;
  let out = text;
  for (const m of matches.reverse()) {
    const said = body([...m], m.index, text);
    if (said === null) continue;
    out = out.slice(0, m.index) + said + out.slice(m.index + m[0].length);
  }
  return out;
}
