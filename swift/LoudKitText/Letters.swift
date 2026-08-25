/// Python reference: `loudkit/frontend/letters.py`.
import Foundation

/// Acronyms, spelled in the language being read.
///
/// `CIA` is *see-eye-ay* in an English render and *ce-i-a* in a Polish one, and
/// those are not two spellings of one thing — they are what the two languages
/// actually say. The engine is grapheme-based with a single language tag per
/// utterance, so the letter name has to be written in the target language's own
/// orthography: English `see` reads as /siː/ under English letter-to-sound
/// rules, Polish `ce` reads as /t͡sɛ/ under Polish ones, and putting either into
/// the other's render produces a word nobody says.
///
/// Without this module, acronyms are spelled only in Polish, inside
/// `LexicalRespelling`, with a Polish letter table: `FBI` becomes *ef-be-i* in a
/// Polish render and reaches the model as the raw graphemes `FBI` in the other
/// eleven, where a grapheme engine reads them as a word-shaped thing rather than
/// as letters. The tables live in the shared grammar file;
/// these are the same tables, read by this port, so all five spell the same
/// acronym the same way in all twelve languages.
///
/// **What is not spelled.** An acronym that is a word in its language stays a
/// word: `NASA` and `NATO` everywhere, `SIDA` and `OVNI` in the Romance three,
/// `PESEL` and `ZUS` in Polish, `TUTKA` in Finnish. Those lists are per language
/// because the fact is: `LOT` is an airline in Poland and a common noun in
/// English, and only one of them should be spelled out.
public enum Letters {

    private static let minLetters = 2

    /// Above five letters an all-caps run is far more often a shout, a product
    /// name or a heading than an initialism, and spelling one out is a worse
    /// error than leaving it — the listener can read `SIGGRAPH`; they cannot
    /// un-hear *ess-eye-gee-gee-ar-ay-pee-aitch*.
    private static let maxLetters = 5

    private struct Table {
        let names: [String: String]
        let words: Set<String>
    }

    private static let tables: [String: Table] = {
        guard let url = Bundle.module.url(forResource: "numbers", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let doc = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let languages = doc["languages"] as? [String: [String: Any]]
        else { return [:] }
        var out: [String: Table] = [:]
        for (lang, entry) in languages {
            guard let names = entry["letter_names"] as? [String: String], !names.isEmpty
            else { continue }
            let words = Set((entry["word_acronyms"] as? [String]) ?? [])
            out[lang] = Table(names: names, words: words)
        }
        return out
    }()

    /// Whether this language has a letter table at all.
    public static func spellsAcronyms(_ language: String) -> Bool {
        tables[language] != nil
    }

    /// What `language` calls one letter, or `nil` if it has no name for it.
    ///
    /// `nil` rather than a guess: a letter with no entry means the acronym is
    /// left alone entirely, because half-spelling one (*ef-be-**q***) is worse
    /// than not spelling it.
    public static func letterName(_ letter: String, language: String) -> String? {
        tables[language]?.names[letter.lowercased()]
    }

    /// `word` as spelled-out letters, or `nil` if it should be left alone.
    ///
    /// `nil` — "not an acronym, or not one I can spell" — for a word that is not
    /// all-caps, is too short or too long, is a word in this language, or
    /// contains a letter this language has no name for.
    public static func spellAcronym(_ word: String, language: String) -> String? {
        guard word.count >= minLetters, isAllCapsWord(word) else { return nil }
        guard let table = tables[language] else { return nil }
        let lowered = word.lowercased()
        if table.words.contains(lowered) {
            // A word, not an initialism: read as itself, lowercased so no later
            // pass mistakes it for an acronym again.
            //
            // Checked *before* the length cap, and the order matters: the cap
            // is about how long a thing may be
            // before spelling it becomes worse than leaving it, and it has
            // nothing to say about a word. With the cap first, every entry over
            // five letters is dead — UNESCO, UNICEF and INTERPOL never reach
            // this branch, so the table could grow entries that do nothing.
            return lowered
        }
        if word.count > maxLetters { return nil }
        var names: [String] = []
        for ch in lowered {
            guard let name = table.names[String(ch)] else { return nil }
            names.append(name)
        }
        // Hyphens rather than spaces: they keep the letters one prosodic unit,
        // so the model reads a run of names instead of a list of tiny words.
        return names.joined(separator: "-")
    }

    /// Every lone acronym in `text`, spelled the way `language` spells it.
    ///
    /// **Shouting is left alone**, and the rule for telling it from an
    /// initialism is context rather than anything inside the word. An initialism
    /// appears as a single capitalised island in ordinary text — "the CIA said"
    /// — while emphasis comes in runs. That distinction is not available from
    /// the word itself: `IT` is a word, an initialism and a shout depending only
    /// on what sits beside it, and no table can separate those. So a capitalised
    /// word spells out only when neither neighbour is also capitalised, and a
    /// text that is *entirely* capitals is passed through whole, because someone
    /// pasted a headline and spelling all of it would be the loudest possible
    /// wrong answer.
    public static func applied(to text: String, language: String) -> String {
        guard spellsAcronyms(language), text.contains(where: { $0.isUppercase })
        else { return text }

        let tokens = splitOnNonWord(text)
        let words = tokens.filter { isWordToken($0) }
        if words.count > 1 && words.allSatisfy({ isAllCapsWord($0) }) {
            // The whole text is capitals: someone pasted a shout, or a headline.
            //
            // More than one word, though. A text that is a single capitalised
            // token — `prepared("GPT")` — is an acronym on its own, not a shout:
            // there is no run to read emphasis from, and refusing it would mean
            // the one call shaped exactly like "say this acronym" was the one
            // that did not.
            return text
        }

        func isCaps(_ index: Int) -> Bool {
            guard index >= 0, index < tokens.count else { return false }
            return isWordToken(tokens[index]) && isAllCapsWord(tokens[index])
        }

        var out = tokens
        for (i, token) in tokens.enumerated() where isCaps(i) {
            // Neighbours, skipping the separator token between words.
            let before = i >= 2 ? isCaps(i - 2) : false
            let after = i + 2 < tokens.count ? isCaps(i + 2) : false
            if before || after { continue }  // part of a run: emphasis
            if let said = spellAcronym(token, language: language) { out[i] = said }
        }
        return out.joined()
    }

    // MARK: mirroring Python's character classes

    /// `str.isalpha()` and `str.isupper()` together, over a whole token.
    ///
    /// Python's `isupper()` is true when there is at least one cased character
    /// and no lowercase one, so a token has to be checked as a unit rather than
    /// character by character.
    private static func isAllCapsWord(_ token: String) -> Bool {
        guard !token.isEmpty, token.allSatisfy({ $0.isLetter }) else { return false }
        var sawCased = false
        for ch in token {
            if ch.isLowercase { return false }
            if ch.isUppercase { sawCased = true }
        }
        return sawCased
    }

    private static func isWordToken(_ token: String) -> Bool {
        token.count > 1 && token.allSatisfy { $0.isLetter }
    }

    /// `re.split(r"(\W+)", text)` — separators kept, so the pieces rejoin
    /// exactly. Word characters are letters, digits and underscore, which is
    /// what Python's `\w` means under its default Unicode rules.
    ///
    /// Walked by *unicode scalar* and not by `Character`, because Python's
    /// regex splits on code points and Swift's `Character` is a grapheme
    /// cluster. A base letter and a combining mark are two code points to
    /// Python — `a̬CIA` splits into `a`, the mark, and `CIA`, so the acronym
    /// stands alone and is spelled — and one `Character` here, which made
    /// `a̬CIA` a single mixed-case token that no longer looked like an acronym
    /// at all. NFC has already run, so the marks that survive to this point are
    /// the ones with no precomposed form, exactly the ones Python treats as
    /// separators.
    private static func splitOnNonWord(_ text: String) -> [String] {
        // Python's `\w` is `str.isalnum()` plus underscore, which is the general
        // categories L* and N* — not the Alphabetic *property*, whose
        // Other_Alphabetic half pulls in combining marks and would undo the
        // scalar walk above.
        func isWordScalar(_ s: Unicode.Scalar) -> Bool {
            switch s.properties.generalCategory {
            case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter,
                .modifierLetter, .otherLetter,
                .decimalNumber, .letterNumber, .otherNumber:
                return true
            default:
                return s == "_"
            }
        }
        var out: [String] = []
        var current = ""
        var currentIsWord: Bool? = nil
        for scalar in text.unicodeScalars {
            let isWord = isWordScalar(scalar)
            if currentIsWord == nil || isWord == currentIsWord {
                current.unicodeScalars.append(scalar)
                currentIsWord = isWord
            } else {
                out.append(current)
                current = String(scalar)
                currentIsWord = isWord
            }
        }
        if !current.isEmpty { out.append(current) }
        // Python's split starts and ends on a word field, even an empty one.
        if let first = out.first, let head = first.unicodeScalars.first, !isWordScalar(head) {
            out.insert("", at: 0)
        }
        return out
    }
}
