/// Python reference: `loudkit/frontend/numbers.py`.
import CryptoKit
import Foundation

/// Numbers, said out loud — the Swift half of `loudkit.frontend.numbers`.
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

    /// First 16 hex characters of the SHA-256 of the grammar file this bundle
    /// carries. Hashed as raw bytes, like every other implementation, so the
    /// five agree only when they ship the same file.
    /// Grammar **and** respelling lexicon, in that order.
    ///
    /// The lexicon is a funnel input exactly as the grammar is and it changes
    /// the spoken tokens, so both files hash into the fingerprint. Leaving the
    /// lexicon out covers 55 KB of rules but not 6.5 MB of vocabulary, and a
    /// build whose lexicon has drifted says different words under the same
    /// sixteen hex digits.
    ///
    /// Both files are resolved the way the passes that *read* them resolve
    /// them: `ChatterboxAssets` first, `Bundle.module` second. Hashing
    /// `Bundle.module` unconditionally while `LexicalRespelling` prefers
    /// the asset channel makes an application shipping its own `pl_en_respell`
    /// speak from one lexicon and report the digest of another — a
    /// fingerprint describing a file that is not in use, which is the one
    /// thing this digest exists to make impossible.
    /// A bundled JSON resource, resolved the way every pass that reads one
    /// resolves it: an application's own copy through `ChatterboxAssets` wins,
    /// the packaged copy is the fallback.
    static func resourceBytes(_ name: String) -> Data? {
        let url = ChatterboxAssets.url(forResource: name, withExtension: "json")
            ?? Bundle.module.url(forResource: name, withExtension: "json")
        guard let url else { return nil }
        return try? Data(contentsOf: url)
    }

    public static let grammarDigest: String = {
        guard let grammar = resourceBytes("numbers"),
              let respell = resourceBytes("pl_en_respell")
        else { return "" }
        return SHA256.hash(data: grammar + respell).map { String(format: "%02x", $0) }
            .joined().prefix(16).description
    }()

    /// The language ids ``cardinal(_:language:gender:)`` can verbalize, sorted
    /// — the roster in `numbers.json`, and the allowlist `TextFrontend`
    /// enforces.
    ///
    /// Mirrors `loudkit.frontend.numbers.supported_languages`. One authority for both
    /// questions: a port that keeps a second copy of the roster is a port that
    /// disagrees with Python the next time a grammar is added.
    public static var supportedLanguages: [String] { grammars.keys.sorted() }

    private static func loadGrammars() -> [String: Grammar] {
        guard let url = Bundle.module.url(forResource: "numbers", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let doc = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let languages = doc["languages"] as? [String: [String: Any]]
        else {
            // The file ships in the bundle; its absence is a build defect, and
            // an empty table makes every call fail loudly rather than wrongly.
            return [:]
        }
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
    /// language or a value past the grammar's largest scale returns nil —
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

    /// ASCII digits only, explicitly — see the Python module for why.
    // Python's `_TIME_RUN` guards both ends with lookaround:
    // `(?<![\d.,:]) … (?![.,:]?\d)`. ICU rejects that exact combination —
    // each piece parses alone, the whole does not — so the right-hand guard
    // lives in `hasNeighbouringDigit` below, called on every match.
    //
    // The left-hand `\b` is NOT equivalent to Python's lookbehind and is a
    // known, measured divergence: `a14:30` reads as *afourteen thirty* in
    // Python and *afourteen:thirty* here, because `\b` fires between a letter
    // and a digit where the lookbehind does not. Recorded in the conformance
    // fixture's `divergent` block rather than left for someone to hear.
    // No `\b`: Python guards this with `(?<![\d.,:]) … (?![.,:]?\d)`, which
    // rejects a *digit or separator* on either side and says nothing about
    // letters. `\b` fires between a letter and a digit as well, so `a14:30`
    // matched in Python and not here — the same string read two ways by two
    // implementations that report one fingerprint.
    //
    // ICU rejects Python's lookaround combination outright (each half parses,
    // the whole does not), so both guards live in `attachedToDigits`, which was
    // already doing exactly this job for the right-hand side.
    private static let timeRun = try! NSRegularExpression(
        pattern: "([01]?[0-9]|2[0-4])[:.]([0-5][0-9])")

    /// Whether the match at `range` has a digit or a separator touching either
    /// end — what tells `14:30` from the `12.03` inside a date. `12.03` matches
    /// inside `12.03.2026`, the ordinary written date of German, Polish,
    /// Danish, Finnish and Norwegian, which must not be read as twelve o'clock
    /// three with the year trailing behind it.
    ///
    /// Checked here rather than with lookarounds in the pattern: ICU rejects the
    /// combination this needs, and Go and Rust already do it by index because
    /// their engines have no lookaround at all. Three of the five agreeing on
    /// one shape beats two spellings of the same rule.
    ///
    /// A trailing sentence period is fine — what follows it is not a digit.
    private static func attachedToDigits(_ ns: NSString, _ range: NSRange) -> Bool {
        let separators: Set<Character> = [".", ",", ":"]
        let start = range.location
        let end = range.location + range.length
        if start > 0 {
            let before = Character(ns.substring(with: NSRange(location: start - 1, length: 1)))
            if before.isNumber || separators.contains(before) { return true }
        }
        if end < ns.length {
            let after = Character(ns.substring(with: NSRange(location: end, length: 1)))
            if after.isNumber { return true }
            if separators.contains(after), end + 1 < ns.length {
                let next = Character(ns.substring(with: NSRange(location: end + 1, length: 1)))
                if next.isNumber { return true }
            }
        }
        return false
    }

    /// Clock times as words — see the Python module for the shape.
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
            // as the clock — "$0.49" as *zero forty-nine*, "3.14" as *three
            // fourteen* — and the shared fixture pinned one of them, so all
            // five implementations agreed on it.
            let separator = ns.substring(
                with: NSRange(location: m.range(at: 1).location + m.range(at: 1).length, length: 1))
            if separator == ".", g.decimalSeparator == "." { continue }
            out += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
            let hour = Int64(ns.substring(with: m.range(at: 1))) ?? 0
            let minute = Int64(ns.substring(with: m.range(at: 2))) ?? 0
            // 24 is admitted only with a zero minute: ISO 8601 writes
            // end-of-day as 24:00, and without it the two halves were read as
            // unrelated numbers with the colon left standing between them.
            // 24:30 is not a time in any convention and stays as written.
            if hour == endOfDayHour, minute != 0 { continue }
            var words: [String] = []
            if let said = cardinal(hour, language: language) { words.append(said) }
            if !g.timeInfix.isEmpty { words.append(g.timeInfix) }
            if minute != 0, let said = cardinal(minute, language: language) {
                words.append(said)
            }
            out += words.joined(separator: " ")
            var end = m.range.location + m.range.length
            if !g.timeInfix.isEmpty {
                end = consumeWrittenInfix(ns, end, g.timeInfix)
            }
            cursor = end
        }
        out += ns.substring(from: cursor)
        return out
    }

    /// Extends `end` past a written infix word — German writes "um 14.30 Uhr",
    /// and the spoken reading already puts the infix where it belongs, between
    /// hour and minutes (*vierzehn Uhr dreißig*). Leaving the written word
    /// standing said it twice. Consumed only when it is a whole word
    /// immediately after the time; *Uhrzeit* keeps its head.
    ///
    /// ASCII checks throughout — space and tab are single UTF-16 units and the
    /// trailing guard tests ASCII alphanumerics only — so this matches the
    /// other four implementations exactly.
    private static func consumeWrittenInfix(_ ns: NSString, _ end: Int, _ infix: String) -> Int {
        var i = end
        while i < ns.length {
            let ch = ns.substring(with: NSRange(location: i, length: 1))
            if ch == " " || ch == "\t" { i += 1 } else { break }
        }
        if i == end { return end }
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

    /// The authority-listed abbreviations, written out — see the Python module.
    public static func expandAbbreviations(_ text: String, language: String) -> String {
        guard let g = grammars[language], !g.abbreviations.isEmpty else { return text }
        var out = text
        for (written, spoken) in g.abbreviations {
            let pattern = "(^|[^\\w.])" + NSRegularExpression.escapedPattern(for: written)
                + "($|[^\\w.])"
            out = out.replacingOccurrences(
                of: pattern, with: "$1" + spoken + "$2", options: .regularExpression)
        }
        return out
    }

    /// Python's `_DIGIT_RUN`, character for character.
    ///
    /// Each part of the pattern is audible when missing:
    ///
    /// * `(?<![\p{L}\p{N}_])` — a run glued to a word is part of that word.
    ///   Without it `iOS18` comes out *iOSeighteen* here and stays `iOS18` in
    ///   Python.
    /// * `(-(?=[0-9]))?` — a minus in front of digits belongs to the number.
    ///   Without it `-5` reads as *five*, dropping the sign entirely.
    /// * `[0-9]{1,3}(?: [0-9]{3})+` — space-grouped thousands are one number.
    ///   Without it `1 000` reads as *one zero zero zero*.
    /// * `(?! ?[0-9])` — a grouped run must reach a boundary. Without it the
    ///   engine takes the longest prefix that fits and abandons the rest, so
    ///   `1 202 555 0199` matches `1 202 555 019` and reads as a ten-digit
    ///   cardinal with a bare `9` trailing behind it.
    ///
    /// The guards spell the word class out rather than writing `\w`, for the
    /// same reason JS does and the opposite problem. ICU's `\w` is
    /// `[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}]` plus the two joiners — it counts a *combining
    /// mark* as a word character, and no other port does: Python's `\w` is
    /// `str.isalnum()` plus underscore, JS spells the class out, and Go and Rust
    /// apply the guard by hand. So `a̬123` — a base letter, a combining caron
    /// below, three digits — read as *a hundre og tjuetre* in four ports and
    /// stayed `a 123` here, because the mark behind the digits looked like the
    /// tail of a word. The walks below were already right: they ask
    /// `isLetter`/`isNumber`, which a mark is neither.
    private static let digitRun = try! NSRegularExpression(
        pattern: "(?<![\\p{L}\\p{N}_])(-(?=[0-9]))?"
            + "([0-9]{1,3}(?: [0-9]{3})+(?! ?[0-9])|[0-9]+)"
            + "((?:[.,][0-9]+)*)(?![\\p{L}\\p{N}_])")

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
            let c = Character(UnicodeScalar(ns.character(at: i)) ?? " ")
            if c.isLetter { return true }
            if c.isNumber || c == "_" || c == "." || c == "," || c == "-" || c == "+" {
                i += 1
                continue
            }
            // A thousands group after the space — `gluedToAWord`'s check, one
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
        let sep = Character(UnicodeScalar(ns.character(at: end)) ?? " ")
        let after = Character(UnicodeScalar(ns.character(at: end + 1)) ?? " ")
        return (sep == "." || sep == ",") && after.isNumber
    }

    /// Whether the digit run at `start` sits inside a token containing a
    /// letter — Python's backward walk over word characters and dots, which is
    /// the question its one-character lookbehind could not ask. In `v1.2.3` the
    /// scan starts at the `2`, because a dot precedes it, and the version came
    /// out "v1.two point three".
    private static func gluedToAWord(_ ns: NSString, _ start: Int) -> Bool {
        var i = start
        while i > 0 {
            let scalar = Character(UnicodeScalar(ns.character(at: i - 1)) ?? " ")
            // `-` and `+` are in the walk because an exponent puts one
            // between the letter and the digits: in `1e-3` the scan starts at
            // the `3`, walks back over `-` to `e`, and stops calling it a
            // number. A bare `-5` is unaffected — the walk reaches a space and
            // finds no letter.
            // A *grouping* space is crossed too, and only under the same strictness
            // as the non-backtracking ports. `x200 000` binds as a single match in Go
            // and Rust, whose engines do not backtrack, so their lookbehind refuses the
            // whole run; a backtracking engine that matched the standalone `000` reads
            // "x200 zero zero zero" — half a token spoken, which is the class the
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
            if !groupingSpace, !(scalar.isLetter || scalar.isNumber
                || scalar == "_" || scalar == "." || scalar == ","
                || scalar == "-" || scalar == "+") {
                return false
            }
            i -= 1
            if scalar.isLetter { return true }
        }
        return false
    }

    /// Whether the run at `i` is exactly one thousands group: `groupDigits`
    /// digits and no fourth, the shape every group after the first has in
    /// `digitRun`.
    ///
    /// Asked of the group the backward walk steps *out of*, because that is the
    /// half whose width the pattern fixes — the first group may be one to three
    /// digits and says nothing about whether the space behind it groups.
    ///
    /// The fourth-digit clause is what keeps `e3 1000` readable: four digits
    /// behind the space are not a group, the space never grouped, and without
    /// the clause the walk crosses it, reaches the `e` and refuses a thousand
    /// that nothing is glued to.
    private static func continuesAGroup(_ ns: NSString, _ i: Int) -> Bool {
        (0..<groupDigits).allSatisfy { isDigit(ns, i + $0) } && !isDigit(ns, i + groupDigits)
    }

    /// Whether the run at `i` is at least a thousands group wide — the forward
    /// walk's half of the same question, and deliberately the looser half.
    ///
    /// A fourth digit makes the run ragged, and a ragged run has to *reach* this
    /// walk rather than end it. `1 0023R` binds as `1 002` in the engines that
    /// do not backtrack, so their forward walk runs on into the `R` and the
    /// whole token is left written; here `(?! ?[0-9])` backtracks to a bare `1`,
    /// and refusing to cross at the fourth digit read "en 0023R" — one digit run
    /// spoken, the next welded to a letter, which is the shape
    /// `docs/reference/preprocess.md` refuses outright.
    ///
    /// The two ends cannot share one test. Relaxing the backward end the same
    /// way crosses the space in `e3 1000` — four digits behind it and a letter
    /// behind those — and refuses a thousand nothing is glued to.
    private static func reachesAGroup(_ ns: NSString, _ i: Int) -> Bool {
        (0..<groupDigits).allSatisfy { isDigit(ns, i + $0) }
    }

    /// An ASCII-or-not digit at `at`, with an out-of-range index answering no
    /// rather than trapping — both walks read past either end of the string.
    private static func isDigit(_ ns: NSString, _ at: Int) -> Bool {
        guard at >= 0, at < ns.length else { return false }
        return Character(UnicodeScalar(ns.character(at: at)) ?? " ").isNumber
    }

    /// An E.164 telephone number, read digit by digit and taken before the
    /// digit run, which cannot decline it: `+48 123 456 789` is a valid
    /// one-to-three-then-threes grouping, so read as a cardinal it is
    /// forty-eight billion. The plus is the evidence — E.164 requires one and a
    /// grouped thousand never carries one.
    private static let phoneRun = try! NSRegularExpression(pattern: "\\+[0-9][0-9 ]*[0-9]")

    /// ISO 8601's 24:00. Admitted as an hour, and only with a zero minute.
    /// Digits in a thousands group: every group after the first is exactly
    /// this many.
    private static let groupDigits = 3

    private static let endOfDayHour: Int64 = 24

    /// Below this a plus-signed run is a delta, not a telephone number.
    private static let minE164Digits = 8

    /// U+2212 MINUS SIGN and U+2010 HYPHEN, folded to ASCII where a digit
    /// follows. Everything downstream reads the sign as `-`, so a typographic
    /// minus was not a sign at all: it reached the punctuation pass, became a
    /// space, and `−5` was read as *five*. Not U+2013, which writes a range.
    private static let unicodeMinus = try! NSRegularExpression(
        pattern: "[\u{2212}\u{2010}](?=[0-9])")

    /// Every run of digits in `text`, said as words — the seam between the
    /// verbalizer and the funnel. Never fails and never leaves digits behind:
    /// a number past every scale is read digit by digit (it is almost always
    /// an identifier), and only the language's own decimal mark is a decimal
    /// mark — the other one is grouping, dropped the way a reader drops it.
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
    /// "٣٫١٤" into "3.14" — which in the eleven languages that write decimals
    /// with a comma is the written form of a clock time, read out as *drei Uhr
    /// vierzehn*.
    public static func foldForeignDigits(_ text: String, language: String) -> String {
        let decimal = grammars[language]?.decimalSeparator ?? "."
        let grouping = decimal == "." ? "," : "."
        var out = ""
        out.reserveCapacity(text.count)
        for scalar in text.unicodeScalars {
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
    /// two hundred sixteen thousand eight hundred one" for an IP address — and in
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
            // Digit by digit — "point four nine", never "point forty-nine":
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
