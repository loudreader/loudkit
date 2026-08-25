import Foundation

/// The one place text becomes something the engine is handed.
///
/// The reader app learned this the hard way and wrote it down: its
/// `SpeechSanitizer` header records that the same transform once existed in
/// two callers "kept in step by a comment", and that any drift between them
/// turned every prefetch into a silent cache miss. Here the drift risk is
/// different but the lesson is the same — superloud had a sanitizer for agent
/// text and nothing at all for selections and the clipboard, so the same
/// arrow, the same footnote marker, the same invisible character behaved
/// differently depending on which key the user pressed.
///
/// Order matters and is deliberate:
///
/// 1. **Invisible characters** first. They are not whitespace by Unicode's
///    rules, so every later rule that looks at neighbours would see them as
///    letters.
/// 2. **Symbols** that carry meaning become words while the digits around
///    them are still intact ("15%" needs its digit).
/// 3. **Footnote markers** before punctuation rules, so `[12]` disappears as
///    a unit rather than leaving a stray twelve.
/// 4. **Punctuation** last: prosodic marks stay exactly where they are — the
///    model is a language model trained on punctuated text, and the period is
///    its strongest stop cue — everything else becomes a space.
public enum SpeechText {

    /// Prepare `text` to be spoken in `languageId`.
    ///
    /// The language id is matched case-insensitively. The tokenizer lowercases
    /// its own tag, so `"PL"` produced Polish *tokens* while skipping the
    /// Polish respelling here — half the utterance read one way and half the
    /// other, with nothing to indicate it.
    public static func prepared(_ text: String, languageId: String?) -> String {
        let language = languageId?.lowercased()
        // NFC first, before anything inspects a character — the same opening
        // pass the Python funnel runs, and the one this funnel did not have.
        //
        // Unicode lets the same character arrive two ways: Polish ą as U+0105 or
        // as a + U+0328, Danish å as U+00E5 or a + U+030A. The tokenizer's
        // vocabulary holds one of them, so a decomposed spelling reaches it as a
        // base letter followed by an unknown combining mark — and every rule
        // below, every pattern and lexicon lookup and character class, is
        // matching a string nobody wrote a rule for.
        //
        // Ahead of `stripInvisibles`, which removes format characters:
        // normalisation can compose a sequence into a single character, and
        // running it afterwards would leave that composition unexamined.
        // Beside NFC, and before the symbol pass so the folded percent sign
        // reaches the table that turns it into a word.
        var out = stripInvisibles(
            Numbers.foldForeignDigits(
                text.precomposedStringWithCanonicalMapping, language: language ?? "en"))
        out = speakSymbols(out, languageId: language)
        out = dropFootnoteMarkers(out)
        // Acronyms while the capitals are still capitals: every later pass
        // lowercases or rewrites, and a spelled acronym has to be decided while
        // the only evidence — that the word stands alone in caps — still exists.
        //
        // The pass belongs here rather than inside `LexicalRespelling`, whose
        // Polish letter table spells `FBI` *ef-be-i* in a Polish render and
        // leaves the model raw graphemes in the other eleven. The tables are per
        // language in the shared grammar file; this reads them for all twelve.
        out = Letters.applied(to: out, language: language ?? "en")
        // Dates before times and numbers, and this ordering is the whole reason
        // the pass exists: `12.03.2026` is the ordinary written date of five of
        // these languages, and both passes below want a piece of it. The clock
        // pattern matches `12.03` and the digit run matches the lot, so a date
        // recognised any later has already been eaten and read as a time with a
        // stray year, or as one eight-digit number.
        out = Dates.expandDates(out, language: language ?? "en")
        // Ordinals before numbers, for the same reason: the number pass expands
        // the digits and leaves the suffix stuck to them, so `1st` would read as
        // *onest*.
        out = Dates.expandOrdinals(out, language: language ?? "en")
        // Numbers after footnotes and before punctuation — see the Python
        // funnel for the ordering argument; the fixture pins it.
        out = Numbers.expandAbbreviations(out, language: language ?? "en")
        out = Numbers.expandTimes(out, language: language ?? "en")
        out = Numbers.expandNumbers(out, language: language ?? "en")
        out = punctuationForSpeech(out)
        out = LexicalRespelling.applied(to: out, languageId: language)
        out = out.replacingOccurrences(of: #"[ \t]{2,}"#, with: " ",
                                       options: .regularExpression)
        // A symbol that became a comma inherits the space that sat in
        // front of it ("0.49 → 0.24" would read "zero point four nine ,").
        out = out.replacingOccurrences(of: #"\s+([.,;:!?])"#, with: "$1",
                                       options: .regularExpression)
        // Two clause marks in a row is one clause mark.
        // A run, not a pair: substitution does not overlap its matches, so a
        // pair rule turns "..." into ".." on one pass and "." on the next.
        out = out.replacingOccurrences(of: #"([.,;:])(?:[\s]*[.,;:])+"#, with: "$1",
                                       options: .regularExpression)
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: invisible characters

    /// Zero-width and formatting characters, and the soft hyphen.
    ///
    /// Straight from the reader app's audit, which found them in 21 of 25
    /// bundled books, 539 occurrences, invisible in every editor: a
    /// phonemizer looks words up exactly, so "me\u{FEFF}" misses the
    /// dictionary and comes back as a hum. This engine is grapheme-based, so
    /// the failure is milder but the same shape — the model sees a word that
    /// does not exist in any text it was trained on. Costless to remove and
    /// impossible to see, which is exactly why it belongs in the funnel.
    static let invisibles: Set<Unicode.Scalar> = [
        "\u{200B}", "\u{200C}", "\u{200D}", "\u{2060}", "\u{FEFF}",
        "\u{00AD}", "\u{180E}", "\u{200E}", "\u{200F}",
    ]

    static func stripInvisibles(_ text: String) -> String {
        guard text.unicodeScalars.contains(where: { invisibles.contains($0) }) else { return text }
        return String(String.UnicodeScalarView(text.unicodeScalars.filter { !invisibles.contains($0) }))
    }

    // MARK: symbols

    /// Symbols the model cannot voice, as words.
    ///
    /// Two families, and the token audit told them apart: `→ ✓ ✗ ≈ ≥` are
    /// literally outside the vocabulary — the tokenizer emits [UNK] and the
    /// model receives nothing at all — while `¢ ° % $` do tokenize, so they
    /// are read at the ear's discretion rather than dropped. Both get words;
    /// only the first family is a proven silent deletion.
    private static let symbolWords: [String: (en: String, pl: String)] = [
        "%": ("percent", "procent"),
        "°": ("degrees", "stopni"),
        "¢": ("cents", "centów"),
        "€": ("euro", "euro"),
        "£": ("pounds", "funtów"),
        "¥": ("yen", "jenów"),
        "₹": ("rupees", "rupii"),
        "×": ("times", "razy"),
        "÷": ("divided by", "podzielone przez"),
        "≈": ("about", "około"),
        "≥": ("at least", "co najmniej"),
        "≤": ("at most", "najwyżej"),
        "≠": ("not equal to", "różne od"),
        "±": ("plus minus", "plus minus"),
        "→": (",", ","), "←": (",", ","), "⇒": (",", ","),
        "✓": ("yes", "tak"), "✔": ("yes", "tak"),
        "✗": ("no", "nie"), "✘": ("no", "nie"),
        "•": (",", ","), "·": (",", ","), "▪": (",", ","), "◦": (",", ","),
        "…": ("...", "..."),
        "&": ("and", "i"),
        "@": ("at", "małpa"),
    ]

    /// `$` and `£` before a number read as a prefix in writing and a SUFFIX in
    /// speech: "$5" is "five dollars", not "dollars five".
    /// Symbol -> word per language. Mirrors `unit_words` in the shared grammar
    /// file (numbers.json, copied into this target's Resources); regenerate
    /// with tools/sync_port_data.py rather than editing. The old table was an
    /// (en, pl) pair, which meant seven of the nine languages heard English.
    static let unitWords: [String: [String: String]] = [
        "da": ["$": "dollar", "%": "procent", "kr": "kroner", "zł": "zloty", "¢": "cent", "£": "pund", "¥": "yen", "°": "grader", "€": "euro", "₹": "rupier"],
        "de": ["$": "Dollar", "%": "Prozent", "kr": "Kronen", "zł": "Zloty", "¢": "Cent", "£": "Pfund", "¥": "Yen", "°": "Grad", "€": "Euro", "₹": "Rupien"],
        "en": ["$": "dollars", "%": "percent", "kr": "kroner", "zł": "zlotys", "¢": "cents", "£": "pounds", "¥": "yen", "°": "degrees", "€": "euros", "₹": "rupees"],
        "es": ["$": "dólares", "%": "por ciento", "kr": "coronas", "zł": "eslotis", "¢": "centavos", "£": "libras", "¥": "yenes", "°": "grados", "€": "euros", "₹": "rupias"],
        "fi": ["$": "dollaria", "%": "prosenttia", "kr": "kruunua", "zł": "złotya", "¢": "senttiä", "£": "puntaa", "¥": "jeniä", "°": "astetta", "€": "euroa", "₹": "rupiaa"],
        "fr": ["$": "dollars", "%": "pour cent", "kr": "couronnes", "zł": "zlotys", "¢": "centimes", "£": "livres", "¥": "yens", "°": "degrés", "€": "euros", "₹": "roupies"],
        "it": ["$": "dollari", "%": "per cento", "kr": "corone", "zł": "zloty", "¢": "centesimi", "£": "sterline", "¥": "yen", "°": "gradi", "€": "euro", "₹": "rupie"],
        "nl": ["$": "dollar", "%": "procent", "kr": "kronen", "zł": "zloty", "¢": "cent", "£": "pond", "¥": "yen", "°": "graden", "€": "euro", "₹": "roepies"],
        "no": ["$": "dollar", "%": "prosent", "kr": "kroner", "zł": "zloty", "¢": "cent", "£": "pund", "¥": "yen", "°": "grader", "€": "euro", "₹": "rupier"],
        "pl": ["$": "dolarów", "%": "procent", "kr": "koron", "zł": "złotych", "¢": "centów", "£": "funtów", "¥": "jenów", "°": "stopni", "€": "euro", "₹": "rupii"],
        "pt": ["$": "dólares", "%": "por cento", "kr": "coroas", "zł": "zlótis", "¢": "cêntimos", "£": "libras", "¥": "ienes", "°": "graus", "€": "euros", "₹": "rupias"],
        "sv": ["$": "dollar", "%": "procent", "kr": "kronor", "zł": "zloty", "¢": "cent", "£": "pund", "¥": "yen", "°": "grader", "€": "euro", "₹": "rupier"],
    ]

    /// The word `symbol` takes in `language`, falling back to English so a
    /// symbol is at least said, if with an accent.
    static func unitWord(_ symbol: String, _ language: String) -> String? {
        unitWords[language]?[symbol] ?? unitWords["en"]?[symbol]
    }

    private static let currencyPrefixSymbols = ["$", "£", "€", "¥", "₹"]

    /// A currency amount, with its decimal mark spelled the way `language`
    /// does. Only a lone dot with a plain fraction is touched — "$1,234.56"
    /// carries a grouping mark this cannot safely reinterpret.
    private static func priced(_ amount: String, language: String) -> String {
        let separator = Numbers.decimalSeparator(language)
        if separator == "." { return amount }
        guard amount.range(of: #"^\d+\.\d+$"#, options: .regularExpression) != nil else {
            return amount
        }
        return amount.replacingOccurrences(of: ".", with: separator)
    }

    /// Marks that make the number beside them a price, whichever side they sit
    /// on. `¢` is here and not in `currencyPrefixSymbols` because nobody writes
    /// `¢49` — it is a suffix in every convention, which is precisely why the
    /// prefix pass never saw it and `0.49¢` reached the clock reader intact.
    ///
    /// (What stood here was a `currencyPrefixes` table nothing read: the
    /// wording comes from `unitWords`, and this was its predecessor.)
    private static let currencySymbols = currencyPrefixSymbols + ["¢"]

    static func speakSymbols(_ text: String, languageId: String?) -> String {
        // A language without a wording table hears English rather than silence.
        let language = unitWords[languageId ?? ""] != nil ? (languageId ?? "en") : "en"
        let polish = language == "pl"
        var out = text
        // Prefix currencies first, while the digits still follow the symbol.
        for symbol in currencyPrefixSymbols {
            guard let word = unitWord(symbol, language) else { continue }
            // The number, and NOT the sentence punctuation behind it: a
            // greedy [\d.,]* swallowed the comma in "£250," and produced
            // "250, pounds" — the currency word ended up after the clause it
            // belonged inside.
            // A letter in front means a multi-character currency mark: `R$` is
            // the Brazilian real, `HK$` the Hong Kong dollar, and this table
            // has a wording for neither. Matching the `$` alone read `R$3,14`
            // as "R3,14 Dollar" — the wrong currency, said confidently.
            let pattern = #"(?<![\p{L}])"#
                + NSRegularExpression.escapedPattern(for: symbol)
                + #"\s?(\d+(?:[.,]\d+)*)"#
            // `priced`, not a bare "$1": the one place a dot between digits is
            // known not to be a clock time, and the last place that knows it.
            // By `expandTimes` the symbol has become a trailing word and
            // "$0.49" is indistinguishable from "14.30", which in the eleven
            // comma-decimal languages is how a time is written — German
            // answered "null Uhr neunundvierzig Dollar".
            let re = try? NSRegularExpression(pattern: pattern)
            let ns = out as NSString
            var rebuilt = ""
            var cursor = 0
            for m in re?.matches(in: out, range: NSRange(location: 0, length: ns.length)) ?? [] {
                rebuilt += ns.substring(
                    with: NSRange(location: cursor, length: m.range.location - cursor))
                let amount = ns.substring(with: m.range(at: 1))
                rebuilt += Self.priced(amount, language: language) + " " + word
                cursor = m.range.location + m.range.length
            }
            rebuilt += ns.substring(from: cursor)
            out = rebuilt
        }
        // The same amount with the symbol behind it. `2.50 €` and `0.49¢` are
        // prices by exactly the evidence `€2.50` is, and reached the time pass
        // with the dot intact: German answered "zwei Uhr fünfzig Euro".
        // Currency written as a *word* — `5.50 zł` — is not covered; telling
        // those from a unit needs a per-language lexicon.
        for symbol in Self.currencySymbols {
            guard let word = unitWord(symbol, language), out.contains(symbol) else { continue }
            let pattern = #"(\d+(?:[.,]\d+)*)\s?"# + NSRegularExpression.escapedPattern(for: symbol)
            guard let re = try? NSRegularExpression(pattern: pattern) else { continue }
            let ns = out as NSString
            var rebuilt = ""
            var cursor = 0
            for m in re.matches(in: out, range: NSRange(location: 0, length: ns.length)) {
                rebuilt += ns.substring(
                    with: NSRange(location: cursor, length: m.range.location - cursor))
                rebuilt += Self.priced(ns.substring(with: m.range(at: 1)), language: language)
                    + " " + word
                cursor = m.range.location + m.range.length
            }
            rebuilt += ns.substring(from: cursor)
            out = rebuilt
        }
        for (symbol, words) in symbolWords where out.contains(symbol) {
            // Not every symbol is a per-language word (arrows, ticks): the
            // old pair table still carries those.
            let replacement = unitWord(symbol, language) ?? (polish ? words.pl : words.en)
            // A word replacement needs spaces around it; a punctuation one
            // must not gain a space BEFORE it or the comma floats.
            let spaced = replacement.count == 1 && ",.".contains(replacement)
                ? replacement + " "
                : " " + replacement + " "
            out = out.replacingOccurrences(of: symbol, with: spaced)
        }
        return out
    }

    // MARK: footnote markers

    /// `[12]`, `[3, 4]`, `[1-5]` — a reference marker, not a number to read.
    /// Bounded at 20 characters so a real bracketed phrase survives.
    static func dropFootnoteMarkers(_ text: String) -> String {
        guard text.contains("[") else { return text }
        return text.replacingOccurrences(
            of: #"\[[\d\s,;\-–—]{1,20}\]"#, with: "", options: .regularExpression)
    }

    /// Letters as Python's `str.isalpha()` means them: Unicode general
    /// categories L*, and *not* M*.
    ///
    /// Foundation's `CharacterSet.letters` is documented as L* **and M***, so
    /// it answers true for a combining mark. Python, Go, Rust and JS all answer
    /// false. A mark that composed into a base character never reached this
    /// pass — NFC handled it — but one that cannot compose survived, and this
    /// port kept it glued to the word while the other four turned it into a
    /// space: `"Az\u{032C}b"` read as `Az̬b` here and `Az b` everywhere else.
    /// Different tokens, different audio, from one funnel reporting one
    /// fingerprint.
    static let letterScalars = CharacterSet.letters.subtracting(.nonBaseCharacters)

    // MARK: punctuation

    /// Punctuation that carries prosody stays exactly where it is; the rest
    /// becomes a space.
    ///
    /// The list is the reader app's, and its reasoning transfers verbatim:
    /// these engines are language models trained on punctuated text, so the
    /// final period is the strongest stop cue, the comma the continuation
    /// cue, and the question mark the only route to interrogative intonation.
    /// Blanking them makes every sentence end on a guess.
    private static let prosodic: Set<Unicode.Scalar> = [
        ".", ",", "!", "?", ";", ":",
        "\u{2014}", "\u{2013}", "\u{2026}",
        "\"", "\u{201C}", "\u{201D}", "\u{201E}", "\u{AB}", "\u{BB}",
        "(", ")", "'", "\u{2019}",
        "\u{00BF}", "\u{00A1}",
    ]

    static func punctuationForSpeech(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        var out = String.UnicodeScalarView()
        out.reserveCapacity(scalars.count)
        for (i, scalar) in scalars.enumerated() {
            if Self.letterScalars.contains(scalar) || CharacterSet.decimalDigits.contains(scalar)
                || CharacterSet.whitespacesAndNewlines.contains(scalar)
                || prosodic.contains(scalar) {
                out.append(scalar)
                continue
            }
            // Between digits, "." and "," are numeric separators and "-" and
            // "/" are ranges and fractions — meaning, not decoration. They
            // survive; the number normalizer downstream reads them.
            let prev = i > 0 ? scalars[i - 1] : nil
            let next = i + 1 < scalars.count ? scalars[i + 1] : nil
            let betweenDigits = prev.map { CharacterSet.decimalDigits.contains($0) } == true
                && next.map { CharacterSet.decimalDigits.contains($0) } == true
            if betweenDigits, "-/:.".unicodeScalars.contains(scalar) {
                out.append(scalar)
                continue
            }
            // A hyphen inside a token is part of the token ("well-known",
            // "1e-3"). Either end alphanumeric, not both letters: the old test
            // left the exponent in "1e-3" to become a space, so the model was
            // handed "1e 3" after the number pass had already declined to read
            // it.
            // `+` alongside `-`: the number pass declines "1e+3" as a token
            // with a letter in it, and punctuation then took it apart into
            // "1e 3".
            if scalar == "-" || scalar == "+",
               prev.map({ Self.letterScalars.contains($0)
                   || CharacterSet.decimalDigits.contains($0) }) == true,
               next.map({ Self.letterScalars.contains($0)
                   || CharacterSet.decimalDigits.contains($0) }) == true {
                out.append(scalar)
                continue
            }
            out.append(" ")
        }
        return String(String.UnicodeScalarView(out))
    }
}
