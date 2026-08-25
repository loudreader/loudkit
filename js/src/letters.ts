/**
 * Acronyms, spelled in the language being read.
 *
 * `CIA` is *see-eye-ay* in an English render and *ce-i-a* in a Polish one, and
 * those are not two spellings of one thing — they are what the two languages
 * actually say. The engine is grapheme-based with a single language tag per
 * utterance, so the letter name has to be written in the target language's own
 * orthography: English `see` reads as /siː/ under English letter-to-sound rules,
 * Polish `ce` reads as /t͡sɛ/ under Polish ones, and putting either into the
 * other's render produces a word nobody says.
 *
 * Without this module, acronyms are spelled only in Polish, inside `respell.ts`,
 * with a Polish letter table: `FBI` becomes *ef-be-i* in a Polish render and
 * reaches the model as the raw graphemes `FBI` in the other eleven, where a
 * grapheme engine reads them as a word-shaped thing rather than as letters. The
 * tables are per
 * language in the shared grammar file; this reads them for all twelve, out of
 * the same numbers.json every other implementation reads.
 *
 * What is not spelled: an acronym that is a word in its language stays a word —
 * `NASA` and `NATO` everywhere, `SIDA` and `OVNI` in the Romance three, `PESEL`
 * and `ZUS` in Polish, `TUTKA` in Finnish. Those lists are per language because
 * the fact is: `LOT` is an airline in Poland and a common noun in English, and
 * only one of them should be spelled out.
 * Python reference: `loudkit/frontend/letters.py`.
 */

import grammarData from "../data/numbers.json" with { type: "json" };

const MIN_LETTERS = 2;

/**
 * Above five letters an all-caps run is far more often a shout, a product name
 * or a heading than an initialism, and spelling one out is a worse error than
 * leaving it — the listener can read `SIGGRAPH`; they cannot un-hear
 * *ess-eye-gee-gee-ar-ay-pee-aitch*.
 */
const MAX_LETTERS = 5;

interface Table {
  names: Record<string, string>;
  words: Set<string>;
}

const TABLES: Record<string, Table> = (() => {
  const out: Record<string, Table> = {};
  const languages = (grammarData as { languages: Record<string, any> }).languages ?? {};
  for (const [lang, entry] of Object.entries(languages)) {
    const names = entry?.letter_names as Record<string, string> | undefined;
    if (!names || Object.keys(names).length === 0) continue;
    out[lang] = {
      names,
      words: new Set<string>((entry?.word_acronyms as string[]) ?? []),
    };
  }
  return out;
})();

/** Whether this language has a letter table at all. */
export function spellsAcronyms(language: string): boolean {
  return language in TABLES;
}

/**
 * What `language` calls one letter, or `null` if it has no name for it.
 *
 * `null` rather than a guess: a letter with no entry means the acronym is left
 * alone entirely, because half-spelling one (*ef-be-**q***) is worse than not
 * spelling it at all.
 */
export function letterName(letter: string, language: string): string | null {
  return TABLES[language]?.names[letter.toLowerCase()] ?? null;
}

/**
 * `word` as spelled-out letters, or `null` if it should be left alone.
 *
 * `null` — "not an acronym, or not one I can spell" — for a word that is not
 * all-caps, is too short or too long, is a word in this language, or contains a
 * letter this language has no name for.
 */
export function spellAcronym(word: string, language: string): string | null {
  const chars = [...word];
  if (chars.length < MIN_LETTERS || !isAllCapsWord(word)) return null;
  const table = TABLES[language];
  if (!table) return null;
  const lowered = word.toLowerCase();
  if (table.words.has(lowered)) {
    // A word, not an initialism: read as itself, lowercased so no later pass
    // mistakes it for an acronym again.
    //
    // Checked *before* the length cap, and the order matters: the cap is about
    // how long a thing may be before spelling it
    // becomes worse than leaving it, and it has nothing to say about a word.
    // With the cap first, every entry over five letters is dead — UNESCO,
    // UNICEF and INTERPOL never reach this branch.
    return lowered;
  }
  if (chars.length > MAX_LETTERS) return null;
  const names: string[] = [];
  for (const ch of lowered) {
    const name = table.names[ch];
    if (name === undefined) return null;
    names.push(name);
  }
  // Hyphens rather than spaces: they keep the letters one prosodic unit, so the
  // model reads a run of names instead of a list of tiny words.
  return names.join("-");
}

/**
 * Every lone acronym in `text`, spelled the way `language` spells it.
 *
 * **Shouting is left alone**, and the rule for telling it from an initialism is
 * context rather than anything inside the word. An initialism appears as a
 * single capitalised island in ordinary text — "the CIA said" — while emphasis
 * comes in runs. That distinction is not available from the word itself: `IT` is
 * a word, an initialism and a shout depending only on what sits beside it, and
 * no table can separate those. So a capitalised word spells out only when
 * neither neighbour is also capitalised, and a text that is *entirely* capitals
 * is passed through whole, because someone pasted a headline and spelling all of
 * it would be the loudest possible wrong answer.
 */
export function spellAcronyms(text: string, language: string): string {
  if (!spellsAcronyms(language) || ![...text].some((c) => c !== c.toLowerCase()))
    return text;

  const tokens = splitOnNonWord(text);
  const words = tokens.filter(isWordToken);
  if (words.length > 1 && words.every(isAllCapsWord)) {
    // The whole text is capitals: someone pasted a shout, or a headline.
    //
    // More than one word, though. A text that is a single capitalised token —
    // `speechText("GPT")` — is an acronym on its own, not a shout: there is no
    // run to read emphasis from, and refusing it would mean the one call shaped
    // exactly like "say this acronym" was the one that did not.
    return text;
  }

  const isCaps = (i: number): boolean => {
    const t = tokens[i];
    return t !== undefined && isWordToken(t) && isAllCapsWord(t);
  };

  const out = [...tokens];
  for (let i = 0; i < tokens.length; i++) {
    if (!isCaps(i)) continue;
    // Neighbours, skipping the separator token between words.
    if ((i >= 2 && isCaps(i - 2)) || isCaps(i + 2)) continue;
    const said = spellAcronym(tokens[i], language);
    if (said !== null) out[i] = said;
  }
  return out.join("");
}

/**
 * Python's `token.isalpha() and token.isupper()`, over a whole token.
 *
 * `isupper()` is true when there is at least one cased character and no
 * lowercase one, so a token is judged as a unit rather than character by
 * character.
 */
function isAllCapsWord(token: string): boolean {
  if (token.length === 0) return false;
  let sawCased = false;
  for (const ch of token) {
    if (!isLetter(ch)) return false;
    if (ch !== ch.toUpperCase()) return false;
    if (ch !== ch.toLowerCase()) sawCased = true;
  }
  return sawCased;
}

function isLetter(ch: string): boolean {
  return /\p{L}/u.test(ch);
}

function isWordToken(token: string): boolean {
  return [...token].length > 1 && [...token].every(isLetter);
}

/**
 * Python's `re.split(r"(\W+)", text)`: separators are kept, so the pieces rejoin
 * exactly. Word characters are letters, digits and underscore, which is what
 * `\w` means under Python's default Unicode rules.
 */
function splitOnNonWord(text: string): string[] {
  const isWord = (ch: string) => /[\p{L}\p{N}_]/u.test(ch);
  const out: string[] = [];
  let current = "";
  let currentIsWord: boolean | null = null;
  for (const ch of text) {
    const w = isWord(ch);
    if (currentIsWord === null) {
      // Python's split starts on a word field, even an empty one.
      if (!w) out.push("");
      current = ch;
      currentIsWord = w;
    } else if (w === currentIsWord) {
      current += ch;
    } else {
      out.push(current);
      current = ch;
      currentIsWord = w;
    }
  }
  if (current.length > 0) out.push(current);
  return out;
}
