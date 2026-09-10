/// Python reference: `loudkit/frontend/numbers.py`.
import CryptoKit
import Foundation

/// Numbers, said out loud, the Swift half of `loudkit.frontend.numbers`.
///
/// The grammar is data and only the interpreter is code: this file reads the
/// same `numbers.json` every other implementation reads (copied into this
/// target's Resources), so a rule lives once. The composition mirrors
/// `loudkit/frontend/numbers.py` function for function; the hand-written fixture plus
/// the 1300-row CLDR differential pin the behaviours.
public enum Numbers {
    struct Scale {
        let value: Int64
        let forms: [String]
        /// "~" composes the multiplier; "" uses the bare scale word; anything
        /// else is the literal one-word (German "eine", Italian "un").
        let oneWord: String
        let separate: Bool
        let link: String
        let smallJoiner: String
        let multiplierAgrees: Bool
        let multiplierGender: String
    }

    struct Grammar {
        let ones: [String]
        let teens: [String]
        let tens: [String]
        let hundred: String
        let hundreds: [String]
        let hundredsGendered: [String: [String]]
        let hundredPluralFinal: String
        let scales: [Scale]
        let unitsBeforeTens: Bool
        let unitTensJoiner: String
        let timeInfix: String
        let abbreviations: [(String, String)]
        let tensJoinerExceptions: [Int64: String]
        let hundredJoiner: String
        let scaleJoinerOnRoundHundreds: Bool
        let scaleLargeJoiner: String
        let oneBeforeHundred: Bool
        let wordJoin: String
        let minusWord: String
        let decimalSeparator: String
        let decimalWord: String
        let exceptions: [Int64: String]
        let genders: [String: [Int64: String]]
        let genderScopes: [Int64: String]
        let combiningOnes: [Int64: String]

        /// The form `value` takes in `gender` at `position`, or nil when it
        /// does not inflect. Position is "standalone" (the whole number),
        /// "tail" (ends a larger number) or "tens_pair" (inside the compound).
        func gendered(_ value: Int64, _ gender: String, position: String) -> String? {
            if gender.isEmpty { return nil }
            switch genderScopes[value] {
            case "standalone" where position != "standalone": return nil
            case "outside_tens" where position == "tens_pair": return nil
            default: break
            }
            return genders[gender]?[value]
        }
    }

    static let grammars: [String: Grammar] = loadGrammars()

    /// A bundled JSON resource, resolved the way every pass that reads one
    /// resolves it: an application's own copy through `ChatterboxAssets` wins,
    /// the packaged copy is the fallback.
    static func resourceBytes(_ name: String) -> Data? {
        let url = ChatterboxAssets.url(forResource: name, withExtension: "json")
            ?? Bundle.module.url(forResource: name, withExtension: "json")
        guard let url else { return nil }
        return try? Data(contentsOf: url)
    }

    /// The per-language blocks of the grammar file, located once and parsed
    /// once.
    ///
    /// Three passes read this file for different blocks: number grammars here,
    /// date and ordinal rules in `Dates`, letter names in `Letters`. Each
    /// spelling its own path to it is how the parsed table and the hashed bytes
    /// drift apart, and this type is where the path already lives because it is
    /// also what hashes it.
    ///
    /// Mirrors `loudkit.frontend.textconfig.grammar_languages`. Empty when the
    /// file is missing: it ships in the bundle, so its absence is a build
    /// defect, and an empty table makes every call fail loudly rather than
    /// wrongly.
    static let grammarLanguages: [String: [String: Any]] = {
        guard let data = resourceBytes("numbers"),
              let doc = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let languages = doc["languages"] as? [String: [String: Any]]
        else { return [:] }
        return languages
    }()

    /// First 16 hex characters of the SHA-256 over the three funnel data files
    /// this bundle carries, concatenated in the order `numbers.json`,
    /// `pl_en_respell.json`, `numerals.json`. Hashed as raw bytes, like every
    /// other implementation, so the five agree only when they ship the same
    /// files.
    ///
    /// All three are funnel inputs and all three move the spoken tokens, so all
    /// three are in the fingerprint. The grammar alone covers 55 KB of rules
    /// and not 6.5 MB of vocabulary, and a build whose lexicon has drifted says
    /// different words under the same sixteen hex digits.
    ///
    /// Every byte in these files is data the funnel reads, so prose inside one
    /// of them moves this digest and the fingerprint above it while every
    /// sample renders identically. Descriptions live beside the data instead,
    /// in `numbers.about.md` and `numerals.provenance.json`, neither hashed nor
    /// embedded here.
    ///
    /// The files are resolved the way the passes that read them resolve them:
    /// `ChatterboxAssets` first, `Bundle.module` second. Hashing
    /// `Bundle.module` unconditionally while `LexicalRespelling` prefers the
    /// asset channel makes an application shipping its own `pl_en_respell`
    /// speak from one lexicon and report the digest of another: a fingerprint
    /// describing a file that is not in use, which is the one thing this digest
    /// exists to make impossible.
    public static let grammarDigest: String = {
        guard let grammar = resourceBytes("numbers"),
              let respell = resourceBytes("pl_en_respell"),
              // And the numeral table, for the same reason: it decides the
              // words a numeral becomes, and it pins the Unicode version the
              // fold uses.
              let numerals = resourceBytes("numerals")
        else { return "" }
        return SHA256.hash(data: grammar + respell + numerals).map { String(format: "%02x", $0) }
            .joined().prefix(16).description
    }()

    /// The language ids ``cardinal(_:language:gender:)`` can verbalize, sorted,
    /// the roster in `numbers.json`, and the allowlist `TextFrontend`
    /// enforces.
    ///
    /// Mirrors `loudkit.frontend.numbers.supported_languages`. One authority for both
    /// questions: a port that keeps a second copy of the roster is a port that
    /// disagrees with Python the next time a grammar is added.
    public static var supportedLanguages: [String] { grammars.keys.sorted() }

    private static func loadGrammars() -> [String: Grammar] {
        let languages = grammarLanguages
        var out: [String: Grammar] = [:]
        func intKeys(_ raw: Any?) -> [Int64: String] {
            var table: [Int64: String] = [:]
            for (k, v) in (raw as? [String: String]) ?? [:] {
                if let n = Int64(k) { table[n] = v }
            }
            return table
        }
        for (lang, e) in languages {
            var scales: [Scale] = []
            for sc in (e["scales"] as? [[String: Any]]) ?? [] {
                scales.append(
                    Scale(
                        value: (sc["value"] as? NSNumber)?.int64Value ?? 0,
                        forms: sc["forms"] as? [String] ?? [],
                        oneWord: sc["one"] as? String ?? "~",
                        separate: sc["separate"] as? Bool ?? false,
                        link: sc["link"] as? String ?? "",
                        smallJoiner: sc["small_joiner"] as? String ?? "",
                        multiplierAgrees: sc["multiplier_agrees"] as? Bool ?? false,
                        multiplierGender: sc["multiplier_gender"] as? String ?? ""))
            }
            var genders: [String: [Int64: String]] = [:]
            for (name, forms) in (e["genders"] as? [String: Any]) ?? [:] {
                genders[name] = intKeys(forms)
            }
            out[lang] = Grammar(
                ones: e["ones"] as? [String] ?? [],
                teens: e["teens"] as? [String] ?? [],
                tens: e["tens"] as? [String] ?? [],
                hundred: e["hundred"] as? String ?? "",
                hundreds: e["hundreds"] as? [String] ?? [],
                hundredsGendered: e["hundreds_gendered"] as? [String: [String]] ?? [:],
                hundredPluralFinal: e["hundred_plural_final"] as? String ?? "",
                scales: scales,
                unitsBeforeTens: e["units_before_tens"] as? Bool ?? false,
                unitTensJoiner: e["unit_tens_joiner"] as? String ?? "",
                timeInfix: e["time_infix"] as? String ?? "",
                abbreviations: ((e["abbreviations"] as? [String: String]) ?? [:])
                    .sorted { $0.key.count > $1.key.count }  // longest first
                    .map { ($0.key, $0.value) },
                tensJoinerExceptions: intKeys(e["tens_joiner_exceptions"]),
                hundredJoiner: e["hundred_joiner"] as? String ?? "",
                scaleJoinerOnRoundHundreds: e["scale_joiner_on_round_hundreds"] as? Bool ?? false,
                scaleLargeJoiner: e["scale_large_joiner"] as? String ?? "",
                oneBeforeHundred: e["one_before_hundred"] as? Bool ?? false,
                wordJoin: e["word_join"] as? String ?? "",
                minusWord: e["minus_word"] as? String ?? "",
                decimalSeparator: e["decimal_separator"] as? String ?? ",",
                decimalWord: e["decimal_word"] as? String ?? "",
                exceptions: intKeys(e["exceptions"]),
                genders: genders,
                genderScopes: intKeys(e["gender_scopes"]),
                combiningOnes: intKeys(e["combining_ones"]))
        }
        return out
    }

    /// `value` as words. An empty gender gives the citation form. An unknown
    /// language or a value past the grammar's largest scale returns nil,
    /// silently reading digits back would be indistinguishable from success.
    public static func cardinal(_ value: Int64, language: String, gender: String = "") -> String? {
        guard let g = grammars[language] else { return nil }
        let ceiling = (g.scales.first?.value).map { $0 * 1000 } ?? 1000
        guard abs(value) < ceiling else { return nil }
        if value < 0 {
            // Always a spaced word, even in solid-writing languages: minus eins.
            guard let rest = cardinal(-value, language: language, gender: gender) else {
                return nil
            }
            return "\(g.minusWord) \(rest)"
        }
        // Standalone agreement applies to the whole number only: Polish jedna
        // alone, but sto jeden.
        if let word = g.gendered(value, gender, position: "standalone") { return word }
        return compose(value, g, gender, asMultiplier: false)
    }

    private static func compose(
        _ value: Int64, _ g: Grammar, _ gender: String, asMultiplier: Bool
    ) -> String {
        if let listed = g.exceptions[value] { return listed }
        if value < 100 { return belowHundred(value, g, gender, asMultiplier: asMultiplier) }
        for sc in g.scales where value >= sc.value {
            return scaleGroup(value, sc, g, gender)
        }
        return hundredsGroup(value, g, gender)
    }

    private static func scaleGroup(
        _ value: Int64, _ sc: Scale, _ g: Grammar, _ gender: String
    ) -> String {
        let count = value / sc.value
        let rest = value % sc.value
        let join = sc.separate ? " " : g.wordJoin
        let linkDefault = sc.link.isEmpty ? join : sc.link

        let head: String
        if count == 1 && sc.oneWord != "~" {
            head = sc.oneWord.isEmpty
                ? scaleWord(1, sc.forms)
                : "\(sc.oneWord)\(join)\(scaleWord(1, sc.forms))"
        } else {
            // Whether the counted noun's gender reaches the multiplier is a
            // fact about the scale noun: Portuguese "duas mil", Polish "dwa
            // tysiące".
            let mg = !sc.multiplierGender.isEmpty
                ? sc.multiplierGender : (sc.multiplierAgrees ? gender : "")
            head = "\(compose(count, g, mg, asMultiplier: true))\(join)\(scaleWord(count, sc.forms))"
        }
        if rest == 0 { return head }

        let roundHundreds = g.scaleJoinerOnRoundHundreds && rest >= 100 && rest % 100 == 0
        let link: String
        if !sc.smallJoiner.isEmpty && (rest < 100 || roundHundreds) {
            link = " \(sc.smallJoiner) "
        } else if rest >= 100 && count >= 100 && !g.scaleLargeJoiner.isEmpty {
            link = g.scaleLargeJoiner
        } else {
            link = linkDefault
        }
        return "\(head)\(link)\(compose(rest, g, gender, asMultiplier: false))"
    }

    private static func scaleWord(_ count: Int64, _ forms: [String]) -> String {
        if forms.count == 1 || count == 1 { return forms[0] }
        // singular / plural: Million / Millionen
        if forms.count == 2 { return forms[1] }
        let lastTwo = count % 100
        let last = count % 10
        if (2...4).contains(last) && !(12...14).contains(lastTwo) { return forms[1] }
        return forms[2]
    }

    private static func hundredsGroup(_ value: Int64, _ g: Grammar, _ gender: String) -> String {
        let count = value / 100
        let rest = value % 100
        var parts: [String] = []
        let hundreds = (gender.isEmpty ? nil : g.hundredsGendered[gender]) ?? g.hundreds
        if !hundreds.isEmpty {
            parts.append(hundreds[Int(count) - 1])
        } else if count == 1 && !g.oneBeforeHundred {
            parts.append(g.hundred)
        } else {
            parts.append(compose(count, g, "", asMultiplier: true))
            // French deux cents / deux cent un: the plural mark appears only
            // when the multiplied hundred ends the number.
            if count > 1 && rest == 0 && !g.hundredPluralFinal.isEmpty {
                parts.append(g.hundredPluralFinal)
            } else {
                parts.append(g.hundred)
            }
        }
        if rest != 0 {
            if !g.hundredJoiner.isEmpty { parts.append(g.hundredJoiner) }
            parts.append(belowHundred(rest, g, gender, asMultiplier: false))
        }
        return parts.filter { !$0.isEmpty }.joined(separator: g.wordJoin)
    }

    private static func unitWord(
        _ value: Int64, _ g: Grammar, _ gender: String, asMultiplier: Bool
    ) -> String {
        if let agreed = g.gendered(value, gender, position: asMultiplier ? "tens_pair" : "tail") {
            return agreed
        }
        if asMultiplier, let combining = g.combiningOnes[value] { return combining }
        return g.ones[Int(value)]
    }

    private static func belowHundred(
        _ value: Int64, _ g: Grammar, _ gender: String, asMultiplier: Bool
    ) -> String {
        if let fixed = g.gendered(value, gender, position: "tail") ?? g.exceptions[value] {
            return fixed
        }
        if value < 10 { return unitWord(value, g, gender, asMultiplier: asMultiplier) }
        if value < 20 { return g.teens[Int(value) - 10] }

        let ten = value / 10
        let unit = value % 10
        let tenWord = g.gendered(ten * 10, gender, position: "tail") ?? g.tens[Int(ten) - 2]
        if unit == 0 { return tenWord }

        // A unit inside a tens pair is always in composition: einundzwanzig
        // holds even when the pair ends the number.
        let unitW = unitWord(unit, g, gender, asMultiplier: true)
        let joiner = g.tensJoinerExceptions[value] ?? g.unitTensJoiner
        return g.unitsBeforeTens ? "\(unitW)\(joiner)\(tenWord)" : "\(tenWord)\(joiner)\(unitW)"
    }

    /// A Roman numeral written with I, V and X, and nothing else.
    ///
    /// L, C, D and M are left out, and that is the whole rule rather than an
    /// optimisation of it. Every two-letter initialism that is also a valid
    /// Roman numeral needs one of them -- CD, CV, DC, MC, MD, XL, CM -- and so
    /// does the only common English word that is one, MIX. What remains is 2 to
    /// 39, which is where chapter, act, volume, war and regnal numbers live.
    private static let romanRun = try! NSRegularExpression(
        pattern: "(?<![0-9A-Za-z])([IVX]{2,})(?![0-9A-Za-z])")

    /// The only spellings 2 to 39 has. Matched whole, so `IIX` and `VV` are
    /// refused: they are letters that happen to be in the alphabet rather than
    /// numbers, and a token this refuses is a token the acronym pass still sees.
    ///
    /// `\A` and `\z` rather than `^` and `$`, which ICU also matches around a
    /// trailing line terminator, and a whole-string test that admits a newline
    /// is not the test Python's `fullmatch` runs.
    private static let romanCanonical = try! NSRegularExpression(
        pattern: "\\A(X{0,3})(IX|IV|V?I{0,3})\\z")

    /// What one X is worth, which is the only place the tens digit comes from.
    private static let romanTen: Int64 = 10

    /// `numeral` as a number, or nil when it is not one.
    private static func romanValue(_ numeral: String) -> Int64? {
        let ns = numeral as NSString
        guard let m = romanCanonical.firstMatch(
            in: numeral, range: NSRange(location: 0, length: ns.length))
        else { return nil }
        let tail = ns.substring(with: m.range(at: 2))
        let units: Int64
        switch tail {
        case "IX": units = 9
        case "IV": units = 4
        default: units = (tail.hasPrefix("V") ? 5 : 0) + Int64(tail.filter { $0 == "I" }.count)
        }
        // The X run is ASCII, so its UTF-16 length is how many X there are.
        return Int64(m.range(at: 1).length) * romanTen + units
    }

    /// `Chapter IV` and `World War II`, said as numbers.
    ///
    /// Before the acronym pass, which spells a Roman numeral letter by letter
    /// (*eye-vee*), and after the numeral fold, which is what turns `Ⅳ` into the
    /// `IV` this pass reads. A run this cannot value is left written.
    public static func expandRomanNumerals(_ text: String, language: String) -> String {
        guard grammars[language] != nil else { return text }
        let ns = text as NSString
        var out = ""
        var cursor = 0
        for m in romanRun.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            guard let value = romanValue(ns.substring(with: m.range(at: 1))),
                  let said = cardinal(value, language: language)
            else { continue }
            out += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
            out += said
            cursor = m.range.location + m.range.length
        }
        out += ns.substring(from: cursor)
        return out
    }

    /// The letters a price abbreviates its magnitude with, longest first.
    ///
    /// Only beside a currency mark, which is what makes them unambiguous: a
    /// bare `5m` is five metres as readily as five million, and `20k` is a race
    /// distance. `$5m` is a sum of money in every convention that writes it.
    private static let scaleSuffixes: [(String, Int64)] = [
        ("bn", 1_000_000_000),
        ("tn", 1_000_000_000_000),
        ("k", 1_000),
        ("m", 1_000_000),
        ("b", 1_000_000_000),
        ("t", 1_000_000_000_000),
    ]

    /// The suffixes above as one alternation, either case, for the funnel's
    /// pattern. Each letter is spelled out in both cases rather than asked of a
    /// case-insensitive flag, because JavaScript has no inline flag group and a
    /// pattern that needs one is a pattern the five implementations cannot
    /// share.
    static let scaleSuffixPattern: String = scaleSuffixes
        .map { spelling, _ in
            spelling.map { "[\(String($0).lowercased())\(String($0).uppercased())]" }.joined()
        }
        .joined(separator: "|")

    /// The scale noun `suffix` abbreviates, in the form `count` of them takes.
    ///
    /// Nil where the language has no noun for that magnitude, which leaves the
    /// letter written rather than guessing at a word for it.
    static func scaleSuffixWord(_ suffix: String, _ count: Int64, _ language: String) -> String? {
        guard let g = grammars[language] else { return nil }
        let wanted = suffix.lowercased()
        for (spelling, value) in scaleSuffixes where spelling == wanted {
            for scale in g.scales where scale.value == value {
                return scaleWord(count, scale.forms)
            }
        }
        return nil
    }

    /// Every form of every scale noun this language has, longest first.
    ///
    /// Longest first because they go into an alternation, where a shorter form
    /// that prefixes a longer one would match first and leave the rest of the
    /// word behind.
    static func scaleNouns(_ language: String) -> [String] {
        guard let g = grammars[language] else { return [] }
        var forms = Set<String>()
        for scale in g.scales {
            for form in scale.forms where !form.isEmpty { forms.insert(form) }
        }
        return forms.sorted {
            let (a, b) = ($0.unicodeScalars.count, $1.unicodeScalars.count)
            return a == b ? $0 < $1 : a > b
        }
    }

    // ASCII digits only, explicitly, see the Python module for why.
    //
    // Python's `_TIME_RUN` guards both ends with lookaround:
    // `(?<![\d.,:]) ... (?![.,:]?\d)`. ICU rejects that exact combination, each
    // piece parsing alone and the whole not, so the guard lives in
    // `attachedToDigits` below, called on every match.
    //
    // No `\b` on the left either. Python's guard rejects a digit or a
    // separator on either side and says nothing about letters, while `\b` fires
    // between a letter and a digit as well, so a pattern with one read `a14:30`
    // differently from the reference under a matching fingerprint. Both guards
    // live in `attachedToDigits`, which was already doing exactly this job for
    // the right-hand side.
    //
    // The seconds are written with a colon and read only from the colon form,
    // which the guard in `expandTimes` enforces: `10:30:45` is a timestamp in
    // every convention, where `10.30.45` is a version string as readily as one.
    private static let timeRun = try! NSRegularExpression(
        pattern: "([01]?[0-9]|2[0-4])[:.]([0-5][0-9])(?::([0-5][0-9]))?")

    /// Whether the match at `range` has a digit or a separator touching either
    /// end, what tells `14:30` from the `12.03` inside a date. `12.03` matches
    /// inside `12.03.2026`, the ordinary written date of German, Polish,
    /// Danish, Finnish and Norwegian, which must not be read as twelve o'clock
    /// three with the year trailing behind it.
    ///
    /// Checked here rather than with lookarounds in the pattern: ICU rejects the
    /// combination this needs, and Go and Rust already do it by index because
    /// their engines have no lookaround at all. Three of the five agreeing on
    /// one shape beats two spellings of the same rule.
    ///
    /// A trailing sentence period is fine, what follows it is not a digit.
    private static func attachedToDigits(_ ns: NSString, _ range: NSRange) -> Bool {
        let separators: Set<Character> = [".", ",", ":"]
        let start = range.location
        let end = range.location + range.length
        if start > 0 {
            let before = scalarAt(ns, start - 1)
            if isDigit(ns, start - 1) { return true }
            if let before, separators.contains(Character(before)) { return true }
        }
        if end < ns.length {
            if isDigit(ns, end) { return true }
            if let after = scalarAt(ns, end), separators.contains(Character(after)),
                isDigit(ns, end + 1) {
                return true
            }
        }
        return false
    }

    /// Clock times as words, see the Python module for the shape.
    public static func expandTimes(_ text: String, language: String) -> String {
        guard let g = grammars[language] else { return text }
        let ns = text as NSString
        var out = ""
        var cursor = 0
        for m in timeRun.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            if attachedToDigits(ns, m.range) { continue }
            // A dot between an hour and two minutes is a clock time in some of
            // these languages and a decimal point in others, and the grammar
            // file already says which: a language that writes 14.30 for half
            // past two does not use the dot as its decimal mark. German writes
            // "14.30 Uhr" and "2,50 €"; English writes "2:30" and "$2.50".
            // Without this every English decimal with two fraction digits read
            // as the clock, "$0.49" as *zero forty-nine*, "3.14" as *three
            // fourteen*, and the shared fixture pinned one of them, so all
            // five implementations agreed on it.
            let separator = ns.substring(
                with: NSRange(location: m.range(at: 1).location + m.range(at: 1).length, length: 1))
            if separator == ".", g.decimalSeparator == "." { continue }
            // Seconds belong to the colon form alone, so a dotted run that
            // reached one is left exactly as written: `10.30:45` is a version
            // string as readily as a timestamp, and the funnel leaves written
            // what it cannot read.
            if separator == ".", m.range(at: 3).location != NSNotFound { continue }
            let hour = Int64(ns.substring(with: m.range(at: 1))) ?? 0
            let minute = Int64(ns.substring(with: m.range(at: 2))) ?? 0
            // A zero seconds field says nothing the hour and minute have not
            // already said, so `10:30:00` reads exactly as `10:30` does.
            let seconds = m.range(at: 3).location == NSNotFound
                ? 0 : Int64(ns.substring(with: m.range(at: 3))) ?? 0
            // 24 is admitted only with a zero minute: ISO 8601 writes
            // end-of-day as 24:00, while 24:30 is not a time in any convention
            // and stays as written.
            if hour == endOfDayHour, minute != 0 || seconds != 0 { continue }
            out += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
            var words: [String] = []
            if let said = cardinal(hour, language: language) { words.append(said) }
            if !g.timeInfix.isEmpty { words.append(g.timeInfix) }
            // A zero minute is dropped from `14:00` and kept in `14:00:45`,
            // where dropping it would move the seconds into the minutes' place.
            if minute != 0 || seconds != 0, let said = cardinal(minute, language: language) {
                words.append(said)
            }
            if seconds != 0, let said = cardinal(seconds, language: language) {
                words.append(said)
            }
            out += words.joined(separator: " ")
            var end = m.range.location + m.range.length
            if !g.timeInfix.isEmpty {
                end = consumeWrittenInfix(ns, end, g.timeInfix)
            }
            // The reading is words, and a word is not written against the
            // letters that followed the digits: `3:45pm` is "three forty-five
            // pm", the reading the spaced form already got.
            if asciiLetterAt(ns, end) { out += " " }
            cursor = end
        }
        out += ns.substring(from: cursor)
        return out
    }

    /// Whether an ASCII letter stands at `at`. The class the written infix is
    /// already guarded against, so the two rules that decide where a spoken
    /// time ends answer to one alphabet in all five implementations rather
    /// than to five spellings of `\w`.
    private static func asciiLetterAt(_ ns: NSString, _ at: Int) -> Bool {
        guard at < ns.length, let scalar = scalarAt(ns, at) else { return false }
        let value = scalar.value
        return (0x41...0x5A).contains(value) || (0x61...0x7A).contains(value)
    }

    /// Extends `end` past a written infix word, German writes "um 14.30 Uhr",
    /// and the spoken reading already puts the infix where it belongs, between
    /// hour and minutes (*vierzehn Uhr dreißig*). Leaving the written word
    /// standing said it twice. Consumed only when it is a whole word
    /// immediately after the time; *Uhrzeit* keeps its head.
    ///
    /// The whitespace in front of it may be absent: a word is the same word
    /// whether or not a space was typed before it. An empty infix is refused
    /// instead, which is what the zero-width match this scan would otherwise
    /// accept means.
    ///
    /// ASCII checks throughout, space and tab are single UTF-16 units and the
    /// trailing guard tests ASCII alphanumerics only, so this matches the
    /// other four implementations exactly.
    private static func consumeWrittenInfix(_ ns: NSString, _ end: Int, _ infix: String) -> Int {
        if infix.isEmpty { return end }
        var i = end
        while i < ns.length {
            let ch = ns.substring(with: NSRange(location: i, length: 1))
            if ch == " " || ch == "\t" { i += 1 } else { break }
        }
        let infixNS = infix as NSString
        guard i + infixNS.length <= ns.length else { return end }
        let candidate = ns.substring(with: NSRange(location: i, length: infixNS.length))
        guard candidate == infix else { return end }
        let after = i + infixNS.length
        if after < ns.length {
            let next = Character(ns.substring(with: NSRange(location: after, length: 1)))
            if next.isASCII, next.isLetter || next.isNumber { return end }
        }
        return after
    }

    /// The authority-listed abbreviations, written out, see the Python module.
    public static func expandAbbreviations(_ text: String, language: String) -> String {
        guard let g = grammars[language], !g.abbreviations.isEmpty else { return text }
        var out = text
        for (written, spoken) in g.abbreviations {
            // The boundaries are looked at, not consumed. A pattern that eats
            // the character on either side leaves the next occurrence with no
            // boundary to start from, because the matches are non-overlapping:
            // "etc. etc." expanded its first abbreviation and left the second
            // one written, in every language that has any.
            //
            // The word class is spelled out for the reason `digitRun` gives:
            // ICU's `\w` counts a combining mark and no other port does.
            let pattern = "(?<![\\p{L}\\p{N}_.])" + NSRegularExpression.escapedPattern(for: written)
                + "(?![\\p{L}\\p{N}_.])"
            out = out.replacingOccurrences(
                of: pattern, with: spoken, options: .regularExpression)
        }
        return out
    }

    /// Python's `_DIGIT_RUN`, character for character.
    ///
    /// Each part of the pattern is audible when missing:
    ///
    /// * `(?<![\p{L}\p{N}_])`, a run glued to a word is part of that word.
    ///   Without it `iOS18` comes out *iOSeighteen* here and stays `iOS18` in
    ///   Python.
    /// * `(-(?=[0-9]))?`, a minus in front of digits belongs to the number.
    ///   Without it `-5` reads as *five*, dropping the sign entirely.
    /// * `[0-9]{1,3}(?: [0-9]{3})+`, space-grouped thousands are one number.
    ///   Without it `1 000` reads as *one zero zero zero*.
    /// * `(?! ?[0-9])`, a grouped run must reach a boundary. Without it the
    ///   engine takes the longest prefix that fits and abandons the rest, so
    ///   `1 202 555 0199` matches `1 202 555 019` and reads as a ten-digit
    ///   cardinal with a bare `9` trailing behind it.
    ///
    /// The guards spell the word class out rather than writing `\w`, for the
    /// same reason JS does and the opposite problem. ICU's `\w` is
    /// `[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}]` plus the two joiners, it counts a *combining
    /// mark* as a word character, and no other port does: Python's `\w` is
    /// `str.isalnum()` plus underscore, JS spells the class out, and Go and Rust
    /// apply the guard by hand. So `a̬123`, a base letter, a combining caron
    /// below, three digits, read as *a hundre og tjuetre* in four ports and
    /// stayed `a 123` here, because the mark behind the digits looked like the
    /// tail of a word. The walks below were already right: they ask
    /// `SpeechText.isLetter` and `SpeechText.isDecimalDigit`, which a mark is
    /// neither -- and which are also the reason `\u{24D0}-1` reads as *minus one*
    /// here rather than as a bare `1`: `Character.isLetter` is the Alphabetic
    /// property and counts the circled letters, and `Character.isNumber` is
    /// Numeric_Type and counts `\u{4e00}`.
    private static let digitRun = try! NSRegularExpression(
        pattern: "(?<![\\p{L}\\p{N}_])(-(?=[0-9]))?"
            + "([0-9]{1,3}(?: [0-9]{3})+(?! ?[0-9])|[0-9]+)"
            + "((?:[.,][0-9]+)*)(?![\\p{L}\\p{N}_])")

    /// The Unicode scalar covering UTF-16 offset `at`, surrogate pair included.
    ///
    /// `NSString.character(at:)` hands back one UTF-16 unit, and a walk that
    /// wraps it in `UnicodeScalar(...) ?? " "` turns half an astral character
    /// into a space: `1 000.\u{17000}`, a grouped number, a dot and a Tangut
    /// ideograph, then looks like a number followed by whitespace here and like
    /// a number glued to a letter in the other four ports, and Swift alone
    /// reads it aloud. Both surrogate halves answer with the whole scalar, so a
    /// walk may land on either end of the pair and still ask about the
    /// character that is actually there.
    static func scalarAt(_ ns: NSString, _ at: Int) -> Unicode.Scalar? {
        guard at >= 0, at < ns.length else { return nil }
        let unit = ns.character(at: at)
        if UTF16.isLeadSurrogate(unit), at + 1 < ns.length {
            return combined(unit, ns.character(at: at + 1))
        }
        if UTF16.isTrailSurrogate(unit), at > 0 {
            return combined(ns.character(at: at - 1), unit)
        }
        return Unicode.Scalar(unit)
    }

    private static func combined(_ lead: unichar, _ trail: unichar) -> Unicode.Scalar? {
        guard UTF16.isLeadSurrogate(lead), UTF16.isTrailSurrogate(trail) else { return nil }
        let value = 0x10000 + (UInt32(lead - 0xD800) << 10) + UInt32(trail - 0xDC00)
        return Unicode.Scalar(value)
    }

    /// How many UTF-16 units a scalar occupies, so a walk steps over a pair
    /// rather than into the middle of it.
    private static func width(_ sc: Unicode.Scalar) -> Int { sc.value > 0xFFFF ? 2 : 1 }

    /// Whether the token continues past the match into a letter.
    ///
    /// The mirror of `gluedToAWord`: `200 000x` matches `200` alone, because
    /// the grouped alternative reaches the `x` and the right-hand guard refuses
    /// it, so the regex backtracks and reads "two hundred 000x". Go and Rust,
    /// which do not backtrack, leave the whole token. A grouping space is
    /// crossed so `200 000x` is one token; the ordinary space in `2024 200
    /// people` is not, because what follows it is a word.
    private static func gluedForward(_ ns: NSString, _ end: Int) -> Bool {
        var i = end
        while i < ns.length {
            guard let c = scalarAt(ns, i) else { return false }
            if SpeechText.isLetter(c) { return true }
            if SpeechText.isDecimalDigit(c) || c == "_" || c == "." || c == ","
                || c == "-" || c == "+" {
                i += width(c)
                continue
            }
            // A thousands group after the space, `gluedToAWord`'s check, one
            // end further on, and see `reachesAGroup` for why this end is the
            // looser of the two. Without any width test the walk crosses out of
            // one number and into the next, so `1000 5.1e+3` refuses the `1000`:
            // it finds the `e` of an exponent two tokens away and calls the
            // whole thing one glued token.
            if c == " ", isDigit(ns, i - 1), reachesAGroup(ns, i + 1) {
                i += 1
                continue
            }
            return false
        }
        return false
    }

    /// Whether a decimal point with digits behind it follows the match.
    ///
    /// The fraction group can match zero times, and the regex will shrink it to
    /// zero so the right-hand guard lands on the dot instead of a letter:
    /// `1.5e3` matched just the `1` and read "one.5e3".
    private static func truncatedByAFraction(_ ns: NSString, _ end: Int) -> Bool {
        guard end + 1 < ns.length else { return false }
        guard let sep = scalarAt(ns, end) else { return false }
        return (sep == "." || sep == ",") && isDigit(ns, end + 1)
    }

    /// Whether the digit run at `start` sits inside a token containing a
    /// letter, Python's backward walk over word characters and dots, which is
    /// the question its one-character lookbehind could not ask. In `v1.2.3` the
    /// scan starts at the `2`, because a dot precedes it, and the version came
    /// out "v1.two point three".
    private static func gluedToAWord(_ ns: NSString, _ start: Int) -> Bool {
        var i = start
        while i > 0 {
            guard let scalar = scalarAt(ns, i - 1) else { return false }
            // `-` and `+` are in the walk because an exponent puts one
            // between the letter and the digits: in `1e-3` the scan starts at
            // the `3`, walks back over `-` to `e`, and stops calling it a
            // number. A bare `-5` is unaffected, the walk reaches a space and
            // finds no letter.
            // A *grouping* space is crossed too, and only under the same strictness
            // as the non-backtracking ports. `x200 000` binds as a single match in Go
            // and Rust, whose engines do not backtrack, so their lookbehind refuses the
            // whole run; a backtracking engine that matched the standalone `000` reads
            // "x200 zero zero zero", half a token spoken, which is the class the
            // right-hand guard exists to stop.
            //
            // Judged by the group the walk steps *out of*, plus a digit behind the
            // space. The looser shapes each break on a real input: "a digit on each
            // side" crosses `R2 5`, which is not
            // a grouped number; "exactly three digits behind" alone breaks `a1 000 000`,
            // whose first group is legitimately one digit; and dropping the digit-behind
            // test lets the walk cross space after space, so `Sold 200 000` reaches
            // "Sold" and refuses a number nothing was glued to.
            let groupingSpace = scalar == " " && isDigit(ns, i - 2) && continuesAGroup(ns, i)
            if !groupingSpace, !(SpeechText.isLetter(scalar) || SpeechText.isDecimalDigit(scalar)
                || scalar == "_" || scalar == "." || scalar == ","
                || scalar == "-" || scalar == "+") {
                return false
            }
            i -= width(scalar)
            if SpeechText.isLetter(scalar) { return true }
        }
        return false
    }

    /// Whether the run at `i` is exactly one thousands group: `groupDigits`
    /// digits and no fourth, the shape every group after the first has in
    /// `digitRun`.
    ///
    /// Asked of the group the backward walk steps *out of*, because that is the
    /// half whose width the pattern fixes, the first group may be one to three
    /// digits and says nothing about whether the space behind it groups.
    ///
    /// The fourth-digit clause is what keeps `e3 1000` readable: four digits
    /// behind the space are not a group, the space never grouped, and without
    /// the clause the walk crosses it, reaches the `e` and refuses a thousand
    /// that nothing is glued to.
    private static func continuesAGroup(_ ns: NSString, _ i: Int) -> Bool {
        (0..<groupDigits).allSatisfy { isDigit(ns, i + $0) } && !isDigit(ns, i + groupDigits)
    }

    /// Whether the run at `i` is at least a thousands group wide, the forward
    /// walk's half of the same question, and deliberately the looser half.
    ///
    /// A fourth digit makes the run ragged, and a ragged run has to *reach* this
    /// walk rather than end it. `1 0023R` binds as `1 002` in the engines that
    /// do not backtrack, so their forward walk runs on into the `R` and the
    /// whole token is left written; here `(?! ?[0-9])` backtracks to a bare `1`,
    /// and refusing to cross at the fourth digit read "en 0023R", one digit run
    /// spoken, the next welded to a letter, which is the shape
    /// `docs/design/preprocess.md` refuses outright.
    ///
    /// The two ends cannot share one test. Relaxing the backward end the same
    /// way crosses the space in `e3 1000`, four digits behind it and a letter
    /// behind those, and refuses a thousand nothing is glued to.
    private static func reachesAGroup(_ ns: NSString, _ i: Int) -> Bool {
        (0..<groupDigits).allSatisfy { isDigit(ns, i + $0) }
    }

    /// A decimal digit at `at`, with an out-of-range index answering no rather
    /// than trapping, both walks read past either end of the string.
    ///
    /// `\p{Nd}`, the class the other four ports test, and **not**
    /// `Character.isNumber`, which is Numeric_Type and answers true for `\u{4e00}`
    /// (`Lo`, numeric value 1) and for every `No` and `Nl`: a character with a
    /// numeric value is not a digit, and counting one as a digit carries a run
    /// past the number it is reading.
    private static func isDigit(_ ns: NSString, _ at: Int) -> Bool {
        guard let sc = scalarAt(ns, at) else { return false }
        return SpeechText.isDecimalDigit(sc)
    }

    /// An E.164 telephone number, read digit by digit and taken before the
    /// digit run, which cannot decline it: `+48 123 456 789` is a valid
    /// one-to-three-then-threes grouping, so read as a cardinal it is
    /// forty-eight billion. The plus is the evidence, E.164 requires one and a
    /// grouped thousand never carries one.
    private static let phoneRun = try! NSRegularExpression(pattern: "\\+[0-9][0-9 ]*[0-9]")

    /// Digits in a thousands group: every group after the first is exactly
    /// this many.
    private static let groupDigits = 3

    /// ISO 8601's 24:00. Admitted as an hour, and only with a zero minute.
    private static let endOfDayHour: Int64 = 24

    /// Below this a plus-signed run is a delta, not a telephone number.
    private static let minE164Digits = 8

    /// U+2212 MINUS SIGN and U+2010 HYPHEN, folded to ASCII where a digit
    /// follows. Everything downstream reads the sign as `-`, so unfolded, a
    /// typographic minus is not a sign at all: it reaches the punctuation
    /// pass, becomes a space, and `−5` reads as *five*. Not U+2013, which
    /// writes a range.
    private static let unicodeMinus = try! NSRegularExpression(
        pattern: "[\u{2212}\u{2010}](?=[0-9])")

    /// The mark `language` writes between a whole number and its fraction.
    ///
    /// Exposed because the speech funnel needs it outside the number pass: a
    /// currency amount is the one place a dot between digits is known not to be
    /// a clock time, and the funnel must say so while the symbol is in hand.
    public static func decimalSeparator(_ language: String) -> String {
        grammars[language]?.decimalSeparator ?? "."
    }

    /// Foreign digit systems and their separators, as this language spells
    /// them.
    ///
    /// Beside NFC because it is the same kind of pass: one spelling for every
    /// pass that follows, and early enough that the symbol table still sees
    /// the folded percent sign.
    ///
    /// Language-dependent for the separators, and that is not a detail. U+066B
    /// is a *decimal* separator, so folding it to a dot everywhere turned
    /// "٣٫١٤" into "3.14", which in the eleven languages that write decimals
    /// with a comma is the written form of a clock time, read out as *drei Uhr
    /// vierzehn*.
    public static func foldForeignDigits(_ text: String, language: String) -> String {
        let decimal = grammars[language]?.decimalSeparator ?? "."
        let grouping = decimal == "." ? "," : "."
        var out = ""
        out.reserveCapacity(text.count)
        for scalar in text.unicodeScalars {
            // Both Arabic-Indic blocks are ten consecutive code points, so the
            // arithmetic in the first two cases lands in 48...57 and the ASCII
            // digit always exists.
            switch scalar.value {
            case 0x0660...0x0669:
                out.append(Character(UnicodeScalar(scalar.value - 0x0660 + 48)!))
            case 0x06F0...0x06F9:
                out.append(Character(UnicodeScalar(scalar.value - 0x06F0 + 48)!))
            case 0x066B: out += decimal
            case 0x066C: out += grouping
            case 0x066A: out.append("%")
            default: out.unicodeScalars.append(scalar)
            }
        }
        return out
    }

    /// E.164 numbers, digit by digit. See `phoneRun`.
    private static func expandPhoneNumbers(_ text: String, language: String) -> String {
        let ns = text as NSString
        var out = ""
        var cursor = 0
        for m in phoneRun.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            // The same two guards the digit run answers to, for the same
            // reason: a run inside a word is part of an identifier, and
            // `+12345678abc` is not a telephone number in any country.
            guard !gluedToAWord(ns, m.range.location),
                  !gluedForward(ns, m.range.location + m.range.length)
            else { continue }
            let whole = ns.substring(with: m.range)
            let digits = whole.filter { $0.isNumber }
            guard digits.count >= minE164Digits else { continue }
            let said = digits.compactMap { cardinal(Int64(String($0)) ?? 0, language: language) }
            guard said.count == digits.count else { continue }
            out += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
            out += said.joined(separator: " ")
            cursor = m.range.location + m.range.length
        }
        out += ns.substring(from: cursor)
        return out
    }

    /// Every run of digits in `text`, said as words: the seam between the
    /// verbalizer and the funnel.
    ///
    /// Never fails and never leaves digits behind. A number past every scale is
    /// read digit by digit, because it is almost always an identifier, and only
    /// the language's own decimal mark is a decimal mark; the other one is
    /// grouping, dropped the way a reader drops it.
    ///
    /// Mirrors `loudkit.frontend.numbers.expand`.
    public static func expandNumbers(_ text: String, language: String) -> String {
        guard let g = grammars[language] else { return text }
        // Both before anything looks for a digit run: the sign has to be ASCII
        // by the time the pattern matches one, and a phone number has to be
        // gone before the grouping rule meets a shape it cannot decline.
        let folded = expandPhoneNumbers(
            unicodeMinus.stringByReplacingMatches(
                in: text, range: NSRange(location: 0, length: (text as NSString).length),
                withTemplate: "-"),
            language: language)
        let ns = folded as NSString
        let text = folded
        var out = ""
        var cursor = 0
        for m in digitRun.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            out += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
            let whole = ns.substring(with: m.range)
            // See `gluedToAWord`: the lookbehind sees one character and an
            // identifier can put a dot between its letter and its digits.
            let matchEnd = m.range.location + m.range.length
            if gluedToAWord(ns, m.range.location)
                || gluedForward(ns, matchEnd)
                || truncatedByAFraction(ns, matchEnd) {
                out += whole
                cursor = m.range.location + m.range.length
                continue
            }
            // Normalised once, here, so everything downstream sees one shape: a
            // sign kept apart from the digits, and thousands spaces gone. The
            // alternative was teaching `isQuantity` and `sayNumber` about two
            // more spellings of a number. Mirrors Python's `say` in
            // `expand_numbers`.
            let sign = m.range(at: 1).location != NSNotFound
            let digits = ns.substring(with: m.range(at: 2)).replacingOccurrences(of: " ", with: "")
            let fraction = m.range(at: 3).location == NSNotFound
                ? "" : ns.substring(with: m.range(at: 3))
            let literal = digits + fraction
            if isQuantity(literal, g) {
                let said = sayNumber(literal, g, language)
                out += sign && !g.minusWord.isEmpty ? "\(g.minusWord) \(said)" : said
            } else {
                out += whole
            }
            cursor = m.range.location + m.range.length
        }
        out += ns.substring(from: cursor)
        return out
    }

    /// Whether a digit run is a number rather than a version, an address or a
    /// date.
    ///
    /// `1.2.3`, `192.168.0.1` and `12.03.2026` all match the digit-run pattern
    /// and none is a quantity. Reading one as a quantity says "nineteen million
    /// two hundred sixteen thousand eight hundred one" for an IP address, and in
    /// the Python reference is a hard crash.
    ///
    /// A run is a quantity when it has at most one separator, or when its
    /// separators genuinely group: every segment after the first exactly three
    /// digits, the first one to three. Anything else is left as written.
    private static func isQuantity(_ literal: String, _ g: Grammar) -> Bool {
        let grouping: Character = g.decimalSeparator == "." ? "," : "."
        let decimal = Character(g.decimalSeparator)
        let whole: Substring
        let fraction: Substring
        if let cut = literal.firstIndex(of: decimal) {
            whole = literal[literal.startIndex..<cut]
            fraction = literal[literal.index(after: cut)...]
        } else {
            whole = literal[...]
            fraction = ""
        }
        // A second mark in what should be the fraction: the split happens once,
        // so this is where "1.2.3" left "2.3" and the reference crashed on it.
        if fraction.contains(grouping) || fraction.contains(decimal) { return false }
        let segments = whole.split(separator: grouping, omittingEmptySubsequences: false)
        if segments.count == 1 { return true }
        let grouped = (1...3).contains(segments[0].count)
            && segments.dropFirst().allSatisfy { $0.count == 3 }
        if grouped { return true }
        // Two segments and no fraction is the "2.5 GB" shape: the mark that is
        // not this language's decimal separator, used as one anyway.
        return segments.count == 2 && !literal.contains(decimal)
    }

    private static func sayNumber(_ literal: String, _ g: Grammar, _ language: String) -> String {
        // The non-decimal mark is only grouping when it groups: every
        // following segment exactly three digits. Polish "1.000" is a
        // thousand; Polish "2.5" is a de-facto decimal, and 2.5 read as 25 is
        // a changed meaning.
        let grouping = g.decimalSeparator == "." ? "," : "."
        var pieces = literal.components(separatedBy: g.decimalSeparator)
        let segments = pieces[0].components(separatedBy: grouping)
        if segments.count > 1 {
            if segments.dropFirst().allSatisfy({ $0.count == 3 }) {
                pieces[0] = segments.joined()
            } else if pieces.count == 1 && segments.count == 2 {
                pieces = [segments[0], segments[1]]
            } else {
                pieces[0] = segments.joined()
            }
        }
        if pieces.count > 1 {
            pieces[1] = pieces[1].replacingOccurrences(of: grouping, with: "")
        }
        var parts = [sayInteger(pieces[0], language)]
        if pieces.count > 1 && !pieces[1].isEmpty {
            parts.append(g.decimalWord)
            // Digit by digit, "point four nine", never "point forty-nine":
            // leading zeros carry meaning there that a cardinal would eat.
            parts.append(contentsOf: digitByDigit(pieces[1], language))
        }
        return parts.joined(separator: " ")
    }

    private static func sayInteger(_ digits: String, _ language: String) -> String {
        // Leading zeros mean a code, not a quantity: 0042 is zero zero four two.
        if digits.count > 1 && digits.hasPrefix("0") {
            return digitByDigit(digits, language).joined(separator: " ")
        }
        if let n = Int64(digits), let said = cardinal(n, language: language) {
            return said
        }
        return digitByDigit(digits, language).joined(separator: " ")
    }

    private static func digitByDigit(_ digits: String, _ language: String) -> [String] {
        digits.compactMap { ch in
            ch.wholeNumberValue.flatMap { cardinal(Int64($0), language: language) }
        }
    }
}
