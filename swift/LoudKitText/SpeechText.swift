import Foundation

/// The one place text becomes something the engine is handed.
///
/// One transform in one place. Split across two callers "kept in step by a
/// comment", it drifts, and then the same arrow, the same footnote marker and
/// the same invisible character are read differently depending on which entry
/// point the text came in through, with nothing anywhere saying so.
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
/// 4. **Punctuation** last: prosodic marks stay exactly where they are, the
///    model is a language model trained on punctuated text, and the period is
///    its strongest stop cue, everything else becomes a space.
public enum SpeechText {

    /// Prepare `text` to be spoken in `languageId`.
    ///
    /// The language id is matched case-insensitively. The tokenizer lowercases
    /// its own tag, so `"PL"` produced Polish *tokens* while skipping the
    /// Polish respelling here, half the utterance read one way and half the
    /// other, with nothing to indicate it.
    public static func prepared(_ text: String, languageId: String?) -> String {
        let language = languageId?.lowercased()
        // NFC first, before anything inspects a character, the same opening
        // pass the Python funnel runs, and the one this funnel did not have.
        //
        // Unicode lets the same character arrive two ways: Polish ą as U+0105 or
        // as a + U+0328, Danish å as U+00E5 or a + U+030A. The tokenizer's
        // vocabulary holds one of them, so a decomposed spelling reaches it as a
        // base letter followed by an unknown combining mark, and every rule
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
        // Before the symbol pass, which would otherwise read a tag's angle
        // brackets as comparison operators and its attributes as text.
        out = dropMarkupTags(out)
        // Before the symbol table: see the Python reference. Every pass
        // downstream asks "is this a digit" and the five ports spell it four
        // ways, so folding first means all of them see ASCII.
        out = foldNumerals(out)
        out = speakSymbols(out, languageId: language)
        out = dropFootnoteMarkers(out)
        // Before the acronym pass, which spells a Roman numeral letter by
        // letter, and after the numeral fold, which is what turns `Ⅳ` into the
        // `IV` this pass reads.
        out = Numbers.expandRomanNumerals(out, language: language ?? "en")
        // Acronyms while the capitals are still capitals: every later pass
        // lowercases or rewrites, and a spelled acronym has to be decided while
        // the only evidence, that the word stands alone in caps, still exists.
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
        // Numbers after footnotes and before punctuation, see the Python
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
        out = out.replacingOccurrences(of: "[\(whiteSpaceClass)]+([.,;:!?])", with: "$1",
                                       options: .regularExpression)
        // Two clause marks in a row is one clause mark.
        // A run, not a pair: substitution does not overlap its matches, so a
        // pair rule turns "..." into ".." on one pass and "." on the next.
        out = out.replacingOccurrences(of: "([.,;:])(?:[\(whiteSpaceClass)]*[.,;:])+", with: "$1",
                                       options: .regularExpression)
        return out.trimmingCharacters(in: whiteSpaceCharacters)
    }

    // MARK: invisible characters

    /// Zero-width and formatting characters, and the soft hyphen.
    ///
    /// Measured over 25 books: present in 21 of them, 539 occurrences,
    /// invisible in every editor. A phonemizer looks words up exactly, so
    /// "me\u{FEFF}" misses the dictionary and comes back as a hum. This engine
    /// is grapheme-based, so the failure is milder and the same shape: the
    /// model sees a word that does not exist in any text it was trained on.
    /// Costless to remove and impossible to see, which is exactly why it
    /// belongs in the funnel.
    static let invisibles: Set<Unicode.Scalar> = [
        "\u{200B}", "\u{200C}", "\u{200D}", "\u{2060}", "\u{FEFF}",
        "\u{00AD}", "\u{180E}", "\u{200E}", "\u{200F}",
    ]

    static func stripInvisibles(_ text: String) -> String {
        guard text.unicodeScalars.contains(where: { invisibles.contains($0) }) else { return text }
        return String(String.UnicodeScalarView(text.unicodeScalars.filter { !invisibles.contains($0) }))
    }

    // MARK: markup

    /// A markup tag, comment or declaration, which is not text anyone reads
    /// aloud.
    ///
    /// The name inside the angle brackets otherwise reaches the model as a
    /// word, `<p>Hello</p>` read *p Hello p*, and `<!-- ... -->` additionally
    /// leaves its `!` behind as a sentence-final exclamation. A tag is replaced
    /// by a space rather than by nothing, because two block tags meeting back
    /// to back are two paragraphs and not one glued word.
    private static let markupTag = try! NSRegularExpression(
        pattern: "</?[A-Za-z!][^<>]*>")

    static func dropMarkupTags(_ text: String) -> String {
        guard text.contains("<") else { return text }
        let ns = text as NSString
        return markupTag.stringByReplacingMatches(
            in: text, range: NSRange(location: 0, length: ns.length), withTemplate: " ")
    }

    // MARK: symbols

    /// Symbols the model cannot voice, in the order the pass replaces them.
    ///
    /// Two families, and the token audit told them apart: `→ ✓ ✗ ≈ ≥` are
    /// literally outside the vocabulary, the tokenizer emits [UNK] and the
    /// model receives nothing at all, while `¢ ° % $` do tokenize, so they
    /// are read at the ear's discretion rather than dropped. Both get words;
    /// only the first family is a proven silent deletion.
    ///
    /// Which word each takes is a per-language fact and comes from `unitWords`;
    /// `symbolMarks` carries the rest.
    private static let spokenSymbols: [String] = [
        "%", "°", "¢", "€", "£", "¥", "₹",
        "×", "÷", "≈", "≥", "≤", "≠", "±",
        "→", "←", "⇒", "✓", "✔", "✗", "✘",
        "•", "·", "▪", "◦", "…", "&", "@",
    ]

    /// An arrow, a bullet and an ellipsis are the same pause in every language,
    /// so they are a rule here rather than a row in twelve grammars.
    private static let symbolMarks: [String: String] = [
        "→": ",", "←": ",", "⇒": ",",
        "•": ",", "·": ",", "▪": ",", "◦": ",",
        "…": "...",
    ]

    /// The ASCII spellings of the comparison operators, longest first, each
    /// named by the mathematical symbol whose word it shares.
    ///
    /// Only these six: `-`, `/`, `.` and `+` are ranges, paths, decimals and
    /// hyphens far more often than operators, and a word put on one of them
    /// changes prose that reads correctly today.
    private static let asciiOperators: [(String, String)] = [
        ("<=", "≤"), (">=", "≥"), ("!=", "≠"), ("==", "="), ("<", "<"), (">", ">"),
    ]

    /// An ASCII operator with whitespace on both sides, which is not consumed.
    ///
    /// The spacing is the evidence that the mark is an operator and not markup
    /// or an emoticon: `<p>`, `</div>`, `<3` and `a<b` all keep the mark
    /// written, and the funnel leaves written what it cannot read. The class is
    /// this funnel's own written-out White_Space and never `\s`, for the reason
    /// `whiteSpaceScalars` gives.
    private static let operatorRun = try! NSRegularExpression(
        pattern: "(?<=[\(whiteSpaceClass)])(<=|>=|!=|==|<|>)(?=[\(whiteSpaceClass)])")

    /// The ASCII comparison operators, as words in this language.
    ///
    /// `if latency > 200 ms` read *if latency two hundred ms*, a condition with
    /// its relation removed, which states the opposite as readily as the one
    /// written.
    static func speakOperators(_ text: String, language: String) -> String {
        let ns = text as NSString
        var out = ""
        var cursor = 0
        for m in operatorRun.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let written = ns.substring(with: m.range(at: 1))
            out += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
            // An operator no grammar covers stays written, like every other
            // symbol this module has no word for.
            let symbol = asciiOperators.first { $0.0 == written }?.1
            out += symbol.flatMap { unitWord($0, language) } ?? written
            cursor = m.range.location + m.range.length
        }
        out += ns.substring(from: cursor)
        return out
    }

    /// Symbol to word, per language, read from the shared grammar file
    /// (numbers.json, copied into this target's Resources) exactly as the other
    /// four implementations read it. One row per language on the roster,
    /// because a symbol is spoken in the language being read: "$5" in a German
    /// render says "5 Dollar", and `≈` says "ungefähr".
    static let unitWords: [String: [String: String]] = {
        var out: [String: [String: String]] = [:]
        for (lang, entry) in Numbers.grammarLanguages {
            guard let words = entry["unit_words"] as? [String: String] else { continue }
            out[lang] = words
        }
        return out
    }()

    /// The word `symbol` takes in `language`, or nil when this language has no
    /// wording for it. No fall back to English: a symbol is spoken in the
    /// language being read or it is left written, which is what the funnel does
    /// everywhere the evidence runs out.
    static func unitWord(_ symbol: String, _ language: String) -> String? {
        unitWords[language]?[symbol]
    }

    /// `$` and `£` before a number read as a prefix in writing and a suffix in
    /// speech: "$5" is "five dollars", not "dollars five".
    private static let currencyPrefixSymbols = ["$", "£", "€", "¥", "₹"]

    /// A currency amount, with its decimal mark spelled the way `language`
    /// does. Only a lone dot with a plain fraction is touched, "$1,234.56"
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
    /// `¢49`, it is a suffix in every convention, which is precisely why the
    /// prefix pass never saw it and `0.49¢` reached the clock reader intact.
    ///
    /// (What stood here was a `currencyPrefixes` table nothing read: the
    /// wording comes from `unitWords`, and this was its predecessor.)
    private static let currencySymbols = currencyPrefixSymbols + ["¢"]

    /// The optional magnitude that may follow a currency amount, as a regex.
    ///
    /// Two alternatives and two groups: the abbreviating letter glued to the
    /// digits (`$2.5M`), and the scale noun written beside them (`$5 million`).
    /// Both cases of each noun are spelled out rather than asked of a
    /// case-insensitive flag, because JavaScript has no inline flag group and a
    /// pattern that needs one is a pattern the five implementations cannot
    /// share.
    private static func scalePattern(_ language: String) -> String {
        let nouns = Numbers.scaleNouns(language)
        if nouns.isEmpty { return "" }
        var written: [String] = []
        for noun in nouns {
            let capitalized = noun.prefix(1).uppercased() + noun.dropFirst()
            for form in noun == capitalized ? [noun] : [noun, capitalized] {
                written.append(NSRegularExpression.escapedPattern(for: form))
            }
        }
        return "(?:(\(Numbers.scaleSuffixPattern))"
            + "|[\(whiteSpaceClass)](\(written.joined(separator: "|"))))(?![A-Za-z])"
    }

    /// The magnitude word standing between a price and its currency, or `""`.
    ///
    /// A written scale reaches speech in two shapes and both belong before the
    /// currency word. The noun is already this language's own word and is kept
    /// as written; the letter is a number, so the grammar's scale noun is asked
    /// for the form this count takes.
    private static func scaleAfter(
        _ amount: String, suffix: String, spelled: String, language: String
    ) -> String {
        if !spelled.isEmpty { return spelled }
        if suffix.isEmpty { return "" }
        // The integer part, whichever mark ends it, and nothing but its digits:
        // `2.5` counts as two, `20` as twenty.
        let whole = amount.prefix { $0 != "." && $0 != "," }.filter { ("0"..."9").contains($0) }
        return Numbers.scaleSuffixWord(suffix, Int64(whole) ?? 0, language) ?? ""
    }

    /// Capture group `index`, or `""` where the pattern has no such group or
    /// the group took part in no match. Both are the optional scale's ordinary
    /// answer: a price with no magnitude written after it.
    private static func group(_ m: NSTextCheckingResult, _ index: Int, _ ns: NSString) -> String {
        guard index < m.numberOfRanges, m.range(at: index).location != NSNotFound else {
            return ""
        }
        return ns.substring(with: m.range(at: index))
    }

    static func speakSymbols(_ text: String, languageId: String?) -> String {
        // A language without a wording table hears English rather than silence.
        let language = unitWords[languageId ?? ""] != nil ? (languageId ?? "en") : "en"
        var out = speakOperators(text, language: language)
        let scale = scalePattern(language)
        // Prefix currencies first, while the digits still follow the symbol.
        for symbol in currencyPrefixSymbols {
            guard let word = unitWord(symbol, language) else { continue }
            // The number, and NOT the sentence punctuation behind it: a
            // greedy [\d.,]* swallowed the comma in "£250," and produced
            // "250, pounds", the currency word ended up after the clause it
            // belonged inside.
            // A letter in front means a multi-character currency mark: `R$` is
            // the Brazilian real, `HK$` the Hong Kong dollar, and this table
            // has a wording for neither. Matching the `$` alone read `R$3,14`
            // as "R3,14 Dollar", the wrong currency, said confidently.
            // The magnitude comes along with the amount, because it is spoken
            // between the amount and the currency word and the mark that moves
            // is written in front of both: `$2.5M` read *two point five
            // dollarsM*.
            let pattern = #"(?<![\p{L}])"#
                + NSRegularExpression.escapedPattern(for: symbol)
                + #"\s?(\d+(?:[.,]\d+)*)"#
                + (scale.isEmpty ? "" : "(?:\(scale))?")
            // `priced`, not a bare "$1": the one place a dot between digits is
            // known not to be a clock time, and the last place that knows it.
            // By `expandTimes` the symbol has become a trailing word and
            // "$0.49" is indistinguishable from "14.30", which in the eleven
            // comma-decimal languages is how a time is written, German
            // answered "null Uhr neunundvierzig Dollar".
            // `try!`: the pattern is a literal around an escaped symbol, so a
            // throw means this source or the currency table is broken, never
            // that the caller's text is. Skipping the pass on a `try?` dropped
            // a whole currency silently.
            let re = try! NSRegularExpression(pattern: pattern)
            let ns = out as NSString
            var rebuilt = ""
            var cursor = 0
            for m in re.matches(in: out, range: NSRange(location: 0, length: ns.length)) {
                rebuilt += ns.substring(
                    with: NSRange(location: cursor, length: m.range.location - cursor))
                let amount = ns.substring(with: m.range(at: 1))
                let suffix = Self.group(m, 2, ns)
                let said = Self.scaleAfter(
                    amount, suffix: suffix, spelled: Self.group(m, 3, ns), language: language)
                let priced = Self.priced(amount, language: language)
                if !suffix.isEmpty, said.isEmpty {
                    // A scale this language has no noun for. The suffix stays
                    // written, which is what it did before the amount moved.
                    rebuilt += priced + " " + word + suffix
                } else if said.isEmpty {
                    rebuilt += priced + " " + word
                } else {
                    rebuilt += priced + " " + said + " " + word
                }
                cursor = m.range.location + m.range.length
            }
            rebuilt += ns.substring(from: cursor)
            out = rebuilt
        }
        // The same amount with the symbol behind it. `2.50 €` and `0.49¢` are
        // prices by exactly the evidence `€2.50` is, and reached the time pass
        // with the dot intact: German answered "zwei Uhr fünfzig Euro".
        // Currency written as a *word*, `5.50 zł`, is not covered; telling
        // those from a unit needs a per-language lexicon.
        // `.literal`, here and at every symbol comparison below: see the note
        // on the symbol pass at the end of this function.
        for symbol in Self.currencySymbols {
            guard let word = unitWord(symbol, language),
                  out.range(of: symbol, options: .literal) != nil else { continue }
            let pattern = #"(\d+(?:[.,]\d+)*)\s?"# + NSRegularExpression.escapedPattern(for: symbol)
            // `try!` for the same reason as the prefix pass above.
            let re = try! NSRegularExpression(pattern: pattern)
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
        // `.literal` is what makes this table fire at all.
        //
        // Without it `contains` and `replacingOccurrences` compare extended
        // grapheme clusters, so `"✔️"` (U+2714 U+FE0F) is not `"✔"` and
        // `"50%\u{FE0F}"` does not contain `"%"`. The table never matched, the
        // punctuation pass then turned the unrecognised cluster into a space,
        // and the word was deleted from the utterance: "Done ✔️ and not done"
        // was spoken as "Done and not done". U+FE0F is what a chat client or a
        // Markdown document emits for a tick, a cross or an arrow, so this was
        // ordinary pasted text, and every one of the 29 rows was affected.
        //
        // The reference compares code points (`symbol not in out`,
        // `out.replace(symbol, spaced)`), and `.literal` is the option that
        // means code points here. Canonical folding goes with it, which is the
        // reference's behaviour too, and `prepared` has already run NFC.
        for symbol in spokenSymbols
        where out.range(of: symbol, options: .literal) != nil {
            guard let replacement = unitWord(symbol, language) ?? symbolMarks[symbol] else {
                continue
            }
            // A word replacement needs spaces around it; a punctuation one
            // must not gain a space BEFORE it or the comma floats.
            let spaced = replacement.count == 1 && ",.".contains(replacement)
                ? replacement + " "
                : " " + replacement + " "
            out = out.replacingOccurrences(of: symbol, with: spaced, options: .literal)
        }
        return out
    }

    // MARK: numerals

    /// Every number character becomes something a reader can say aloud.
    ///
    /// Python reference: `loudkit.frontend.speechtext.fold_numerals`, which
    /// carries the reasoning. A non-ASCII decimal digit becomes the ASCII digit
    /// of the same value and joins the run it was in; every other number
    /// character becomes the text `numerals.json` names for it, separated from
    /// an adjacent alphanumeric so `\u{b2}9` does not fold into `29` and read as
    /// twenty-nine. The table decides *whether* a character is a numeral too --
    /// see `foldedNumeral`.
    static func foldNumerals(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        guard scalars.contains(where: { foldedNumeral($0) != nil }) else { return text }
        var out = String.UnicodeScalarView()
        for (i, sc) in scalars.enumerated() {
            guard let folded = foldedNumeral(sc) else {
                out.append(sc)
                continue
            }
            // A slash inside a spelled numeral is a fraction bar, asserted by
            // the character itself: `½` is a half wherever it stands, where a
            // typed `1/2` is a fraction, a date or the `24/7` of ordinary
            // prose. The division sign is the mark the symbol table already has
            // a word for in every language, so the reading comes from the
            // grammar and not from here.
            let spelled = folded.spelled.replacingOccurrences(of: "/", with: "÷")
            if folded.isDigit {
                // A digit replacing a digit joins the run it was already in.
                out.append(contentsOf: spelled.unicodeScalars)
                continue
            }
            if let previous = out.last, isLetterOrDigit(previous) { out.append(" ") }
            out.append(contentsOf: spelled.unicodeScalars)
            if i + 1 < scalars.count, isLetterOrDigit(scalars[i + 1]) { out.append(" ") }
        }
        return String(out)
    }

    /// `\p{Nd}`, by general category rather than by `CharacterSet`.
    static func isDecimalDigit(_ sc: Unicode.Scalar) -> Bool {
        sc.properties.generalCategory == .decimalNumber
    }

    /// `\p{L}`, which is what Python's `str.isalpha`, Go's `unicode.IsLetter`,
    /// Rust's general category and JS's `\p{L}` all mean.
    ///
    /// Not `Character.isLetter`, which is the **Alphabetic** property and is
    /// wider by the circled letters (`So`), the Other_Alphabetic marks and the
    /// letter numbers; and not `CharacterSet.letters`, which Foundation
    /// answers wrongly above the BMP, measured, it omits all 6,145 Tangut
    /// ideographs and `subtracting(.nonBaseCharacters)` silently fails for
    /// astral scalars, so 1,162 astral combining marks leaked through.
    ///
    /// L* and *not* M*, which is the same measurement from the other side:
    /// `CharacterSet.letters` is documented as L* **and M***, so it calls a
    /// combining mark a letter where Python, Go, Rust and JS call it none. A
    /// mark that composed into its base never reached this pass -- NFC handled
    /// it -- but one that cannot compose survived, and this port kept it glued
    /// to the word while the other four spaced it: `"Az\u{032C}b"` read as
    /// `Az̬b` here and `Az b` everywhere else. Different tokens, different
    /// audio, from one funnel reporting one fingerprint.
    static func isLetter(_ sc: Unicode.Scalar) -> Bool {
        switch sc.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter,
             .modifierLetter, .otherLetter:
            return true
        default:
            return false
        }
    }

    /// Unicode White_Space, written out, for every space test in this funnel.
    ///
    /// `\s` is four different sets across the five implementations and the
    /// difference is audible. Measured: ECMAScript `\s` excludes U+0085 NEL,
    /// Go's hand-expanded class omitted U+000B, CPython's `\s` uniquely admits
    /// U+001C-U+001F. NEL in particular is ordinary in scraped and epub text,
    /// and a separator that survives in one port changes where that port's
    /// reader breathes.
    ///
    /// Swift was **not** one of the wrong ones: ICU's `\s` and Foundation's
    /// `.whitespacesAndNewlines` both answer White_Space exactly today, and
    /// swapping them for this table changed no output -- all twenty-five were
    /// probed. It is written out anyway because the other four ports name the
    /// set they mean, and a port whose answer depends on which ICU it links
    /// against is a port whose agreement is a coincidence rather than a
    /// contract. Python reference: `loudkit.frontend.speechtext.WHITE_SPACE`.
    static let whiteSpaceScalars: [Unicode.Scalar] = [
        "\u{0009}", "\u{000A}", "\u{000B}", "\u{000C}", "\u{000D}", "\u{0020}",
        "\u{0085}", "\u{00A0}", "\u{1680}",
        "\u{2000}", "\u{2001}", "\u{2002}", "\u{2003}", "\u{2004}", "\u{2005}",
        "\u{2006}", "\u{2007}", "\u{2008}", "\u{2009}", "\u{200A}",
        "\u{2028}", "\u{2029}", "\u{202F}", "\u{205F}", "\u{3000}",
    ]

    /// The same set, for a single scalar.
    static let whiteSpace = Set(whiteSpaceScalars)

    /// The same set as the body of a regex character class, escaped so a
    /// literal newline never reaches the pattern.
    static let whiteSpaceClass = whiteSpaceScalars
        .map { String(format: "\\u%04X", $0.value) }
        .joined()

    /// The same set, for trimming.
    static let whiteSpaceCharacters =
        CharacterSet(charactersIn: String(String.UnicodeScalarView(whiteSpaceScalars)))

    /// `\p{L}` or `\p{Nd}`: the word class all five ports test.
    static func isLetterOrDigit(_ sc: Unicode.Scalar) -> Bool {
        isLetter(sc) || isDecimalDigit(sc)
    }

    /// What `foldNumerals` puts in a numeral's place and whether it is a
    /// decimal digit, or `nil` for a character the table does not name.
    ///
    /// Read from the shared `numerals.json` rather than computed, detection
    /// included. Computing either half was wrong: walking down to a decimal
    /// block's start walks out of the block where two blocks touch,
    /// `MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO` follows `MATHEMATICAL SANS-SERIF
    /// DIGIT NINE` with nothing between, NFKC does not reach `ETHIOPIC NUMBER
    /// TEN`, the Aegean numbers, the Kaktovik digits or the Meroitic numerals,
    /// which then vanished; and asking `generalCategory` *whether* to fold put
    /// a Unicode version back into an answer the table had taken it out of.
    /// Swift's tables move with the OS, Python's with the interpreter, and the
    /// five ports report one fingerprint. The table is cut from one pinned UCD
    /// and hashed into the grammar digest.
    ///
    /// `nil` for an ASCII digit, a letter or an ideographic numeral: all left
    /// exactly as written.
    static func foldedNumeral(_ sc: Unicode.Scalar) -> (spelled: String, isDigit: Bool)? {
        if "0"..."9" ~= sc { return nil }
        let table = Numerals.shared
        if let text = table.spelled[sc.value] { return (text, false) }
        var low = 0
        var high = table.decimalZeros.count
        while low < high {
            let mid = (low + high) / 2
            if table.decimalZeros[mid] <= sc.value { low = mid + 1 } else { high = mid }
        }
        if low > 0 {
            let offset = sc.value - table.decimalZeros[low - 1]
            if offset <= 9 { return (String(offset), true) }
        }
        return nil
    }

    // MARK: footnote markers

    /// `[12]`, `[3, 4]`, `[1-5]`, a reference marker, not a number to read.
    /// Bounded at 20 characters so a real bracketed phrase survives.
    ///
    /// `[0-9]` and the written-out space class, not `\d` and `\s`: ICU reads
    /// both as Unicode and RE2 reads both as ASCII, so the same marker was a
    /// marker in one port and a number to read aloud in another.
    static func dropFootnoteMarkers(_ text: String) -> String {
        guard text.contains("[") else { return text }
        return text.replacingOccurrences(
            of: "\\[[0-9\(whiteSpaceClass),;\\-–—]{1,20}\\]",
            with: "", options: .regularExpression)
    }

    // MARK: punctuation

    /// Punctuation that carries prosody stays exactly where it is; the rest
    /// becomes a space.
    ///
    /// These engines are language models trained on punctuated text, so the
    /// final period is the strongest stop cue, the comma the continuation cue,
    /// and the question mark the only route to interrogative intonation.
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
            if Self.isLetter(scalar) || Self.isDecimalDigit(scalar)
                || whiteSpace.contains(scalar)
                || prosodic.contains(scalar) {
                out.append(scalar)
                continue
            }
            // Between digits, "." and "," are numeric separators and "-" and
            // "/" are ranges and fractions, meaning, not decoration. They
            // survive; the number normalizer downstream reads them.
            let prev = i > 0 ? scalars[i - 1] : nil
            let next = i + 1 < scalars.count ? scalars[i + 1] : nil
            let betweenDigits = prev.map { Self.isDecimalDigit($0) } == true
                && next.map { Self.isDecimalDigit($0) } == true
            if betweenDigits, "-/:.".unicodeScalars.contains(scalar) {
                out.append(scalar)
                continue
            }
            // A hyphen inside a token is part of the token ("well-known",
            // "1e-3"). Either end alphanumeric, not both letters: a test for
            // two letters leaves the exponent in "1e-3" to become a space, and
            // the model is handed "1e 3" after the number pass has already
            // declined to read it.
            // `+` alongside `-`: the number pass declines "1e+3" as a token
            // with a letter in it, and punctuation then took it apart into
            // "1e 3".
            if scalar == "-" || scalar == "+",
               prev.map({ Self.isLetter($0) || Self.isDecimalDigit($0) }) == true,
               next.map({ Self.isLetter($0) || Self.isDecimalDigit($0) }) == true {
                out.append(scalar)
                continue
            }
            out.append(" ")
        }
        return String(out)
    }
}
