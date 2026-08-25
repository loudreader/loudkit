/// Python reference: `loudkit/frontend/dates.py`.
import Foundation

/// Dates and ordinals, said the way each language says them.
///
/// A port of `loudkit.frontend.dates`: `12.03.2026` is the
/// ordinary written date of five of these twelve languages, and without this
/// funnel it reads as a clock time with a stray year, or as one eight-digit
/// number. `1st`
/// arrives as *onest*, because the number pass expands the digits and leaves the
/// suffix stuck to them.
///
/// Every rule is data from the shared `numbers.json` — month names, day forms,
/// the infixes Spanish and Portuguese speak between the parts, the German
/// oblique triggers, the ordinal tables. What is code here is the *shape*: which
/// written forms are dates at all, and how each language reads a year.
///
/// Two refusals are as deliberate as anything it does. A yearless `12.3.` is
/// never matched — its closing period is indistinguishable from a sentence's, so
/// `Die Zahl ist 3.5.` would otherwise come out as *dritte Mai*. And `3/12/2026` is left alone
/// in English, where it is March twelfth to half the world and the third of
/// December to the other half: a listener recovers from hearing digits, not from
/// a confident wrong month.
public enum Dates {

    /// Above this a four-digit run is an identifier, not a year.
    private static let maxYear = 2999
    /// A three-digit year exists; a three-digit *anything* is far more often a
    /// quantity, and nothing in the string separates them.
    private static let minYear = 1000
    /// February is 29 on purpose: a plausibility bound, not a calendar.
    /// Refusing 29 February in a common year would reject a date a human wrote
    /// deliberately, and accepting it costs nothing.
    private static let daysInMonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    struct Rules {
        var dayForm = "cardinal"
        var dayWords: [Int: String] = [:]
        var dayWordsOblique: [Int: String] = [:]
        var obliqueTriggers: [String] = []
        var dayOneWord = ""
        var months: [String] = []
        var dayMonthInfix = ""
        var monthYearInfix = ""
        var dayFirstPrefix = ""
        var dayFirstInfix = ""
        var yearRule = ""
        var yearUnits: [Int: String] = [:]
        var yearTeens: [Int: String] = [:]
        var yearTens: [Int: String] = [:]
        var yearTwoThousand = ""
        var dottedIsAmbiguous = false
        var noDottedDates = false
        var ordinalSuffixes: [String] = []
        var ordinalUnits: [Int: String] = [:]
        var ordinalTeens: [Int: String] = [:]
        var ordinalTens: [Int: String] = [:]
        var ordinalJoiner = "-"
    }

    static let rules: [String: Rules] = {
        guard let url = Bundle.module.url(forResource: "numbers", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let doc = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let languages = doc["languages"] as? [String: [String: Any]]
        else { return [:] }

        func intKeys(_ raw: Any?) -> [Int: String] {
            var out: [Int: String] = [:]
            for (k, v) in (raw as? [String: String]) ?? [:] {
                if let n = Int(k), !v.isEmpty { out[n] = v }
            }
            return out
        }

        var result: [String: Rules] = [:]
        for (lang, entry) in languages {
            guard let dates = entry["dates"] as? [String: Any] else { continue }
            var r = Rules()
            r.dayForm = dates["day_form"] as? String ?? "cardinal"
            r.dayWords = intKeys(dates["day_words"])
            r.dayWordsOblique = intKeys(dates["day_words_oblique"])
            r.obliqueTriggers = (dates["oblique_triggers"] as? [String]) ?? []
            r.dayOneWord = dates["day_one_word"] as? String ?? ""
            r.months = (dates["months"] as? [String]) ?? []
            r.dayMonthInfix = dates["day_month_infix"] as? String ?? ""
            r.monthYearInfix = dates["month_year_infix"] as? String ?? ""
            r.dayFirstPrefix = dates["day_first_prefix"] as? String ?? ""
            r.dayFirstInfix = dates["day_first_infix"] as? String ?? ""
            r.yearRule = dates["year_rule"] as? String ?? ""
            r.yearUnits = intKeys(dates["year_units"])
            r.yearTeens = intKeys(dates["year_teens"])
            r.yearTens = intKeys(dates["year_tens"])
            r.yearTwoThousand = dates["year_two_thousand"] as? String ?? ""
            r.dottedIsAmbiguous = dates["dotted_is_ambiguous"] as? Bool ?? false
            r.noDottedDates = dates["no_dotted_dates"] as? Bool ?? false
            if let ord = entry["ordinals"] as? [String: Any] {
                r.ordinalSuffixes = (ord["suffixes"] as? [String]) ?? []
                r.ordinalUnits = intKeys(ord["units"])
                r.ordinalTeens = intKeys(ord["teens"])
                r.ordinalTens = intKeys(ord["tens"])
                r.ordinalJoiner = ord["tens_joiner"] as? String ?? "-"
            }
            result[lang] = r
        }
        return result
    }()

    /// The languages with a date grammar, sorted.
    public static var supportedLanguages: [String] { rules.keys.sorted() }

    // MARK: pieces

    static func monthName(_ month: Int, language: String) -> String? {
        guard let r = rules[language], (1...12).contains(month), r.months.count == 12
        else { return nil }
        return r.months[month - 1]
    }

    /// The day-of-month word, in whatever form this language's dates take.
    ///
    /// `oblique` is German only — the `-en` ending that `am`/`den`/`vom` select.
    /// Ignored elsewhere, because no other language here inflects the day by its
    /// frame.
    static func ordinalDay(_ day: Int, language: String, oblique: Bool = false) -> String? {
        guard let r = rules[language], (1...31).contains(day) else { return nil }
        if oblique, let w = r.dayWordsOblique[day] { return w }
        if let w = r.dayWords[day] { return w }
        // Cardinal languages: the day is just a number, except where the first
        // of the month is lexicalised.
        if day == 1, !r.dayOneWord.isEmpty { return r.dayOneWord }
        return Numbers.cardinal(Int64(day), language: language)
    }

    /// A year, read the way this language reads years.
    ///
    /// English and Norwegian split it; German, Dutch and Swedish group it in
    /// hundreds; the rest say one plain cardinal. Spanish is the explicit case —
    /// the RAE writes that a year is read as its cardinal and *not* in
    /// two-figure blocks as in English, so 2021 is *dos mil veintiuno*.
    static func sayYear(_ year: Int, language: String) -> String {
        let cardinal = { (n: Int) in Numbers.cardinal(Int64(n), language: language) ?? "" }
        guard let r = rules[language] else { return cardinal(year) }
        switch r.yearRule {
        case "en_split": return yearEnglish(year)
        case "de_hundreds": return yearHundreds(year, language: "de", joiner: "hundert", range: 1100...1999)
        case "nl_hundreds": return yearHundreds(year, language: "nl", joiner: "honderd", range: 1100...1999)
        case "sv_hundreds": return yearHundreds(year, language: "sv", joiner: "hundra", range: 1100...2099)
        case "no_split": return yearNorwegian(year)
        case "da_long": return yearDanish(year)
        case "pl_ordinal_genitive": return yearPolish(year, r)
        default: return cardinal(year)
        }
    }

    private static func card(_ n: Int, _ lang: String) -> String {
        Numbers.cardinal(Int64(n), language: lang) ?? ""
    }

    private static func yearEnglish(_ year: Int) -> String {
        if year == 1000 || year == 2000 || (2001...2009).contains(year) {
            return card(year, "en")
        }
        if (1001...1999).contains(year) || year >= 2100 {
            let century = year / 100, rest = year % 100
            if rest == 0 { return "\(card(century, "en")) hundred" }
            // "nineteen oh five" — never "nineteen five", which nobody says.
            if rest < 10 { return "\(card(century, "en")) oh \(card(rest, "en"))" }
            return "\(card(century, "en")) \(card(rest, "en"))"
        }
        if (2010...2099).contains(year) { return "twenty \(card(year % 100, "en"))" }
        return card(year, "en")
    }

    /// German, Dutch and Swedish all write `<century><joiner><rest>` solid; only
    /// the joiner and the range differ. German's upper bound is 1999 because the
    /// GfdS explicitly rejects `zwanzighundert…` — German did not follow the
    /// English "twenty-sixteen" shift. Swedish runs to 2099 because
    /// Isof/Språkrådet has recommended the `tjugohundra…` series for decades.
    private static func yearHundreds(
        _ year: Int, language: String, joiner: String, range: ClosedRange<Int>
    ) -> String {
        guard range.contains(year) else { return card(year, language) }
        let century = year / 100, rest = year % 100
        let head = "\(card(century, language))\(joiner)"
        return rest == 0 ? head : "\(head)\(card(rest, language))"
    }

    /// Norwegian splits 1100–1999 and drops `hundre`: 1972 is `nittensyttito`.
    /// Språkrådet's main recommendation from 2000 on is the `totusenog…` form.
    private static func yearNorwegian(_ year: Int) -> String {
        guard (1100...1999).contains(year) else { return card(year, "no") }
        let century = year / 100, rest = year % 100
        if rest == 0 { return "\(card(century, "no"))hundre" }
        return "\(card(century, "no"))\(card(rest, "no"))"
    }

    /// Dansk Sprognævn: the long form works for every year, and the short
    /// "telephone-number" form is explicitly poor for a century's first decade.
    private static func yearDanish(_ year: Int) -> String {
        guard (1100...1999).contains(year) else { return card(year, "da") }
        let century = year / 100, rest = year % 100
        let head = "\(card(century, "da")) hundrede"
        return rest == 0 ? head : "\(head) og \(card(rest, "da"))"
    }

    /// Only the tens and units of a Polish year decline.
    ///
    /// PWN's worked example is *tysiąc dziewięćset dziewięćdziesiątego
    /// drugiego*: the thousands and hundreds keep their cardinal form and the
    /// ordinal genitive lands on the last two digits. Where those are zero the
    /// declension moves left, which is why 2000 has its own word.
    private static func yearPolish(_ year: Int, _ r: Rules) -> String {
        if year == 2000, !r.yearTwoThousand.isEmpty { return r.yearTwoThousand }
        let head = year / 100, rest = year % 100
        let lead = head != 0 ? card(head * 100, "pl") : ""
        if rest == 0 { return lead }
        let tail: String
        if let teen = r.yearTeens[rest] {
            tail = teen
        } else {
            let words = [r.yearTens[(rest / 10) * 10] ?? "", r.yearUnits[rest % 10] ?? ""]
            tail = words.filter { !$0.isEmpty }.joined(separator: " ")
        }
        return "\(lead) \(tail)".trimmingCharacters(in: .whitespaces)
    }

    private static func valid(day: Int, month: Int, year: Int?) -> Bool {
        guard (1...12).contains(month) else { return false }
        guard (1...daysInMonth[month - 1]).contains(day) else { return false }
        guard let y = year else { return true }
        return (minYear...maxYear).contains(y)
    }

    private static func spoken(
        day: Int, month: Int, year: Int?, language: String, oblique: Bool
    ) -> String? {
        guard let r = rules[language],
              let dayWord = ordinalDay(day, language: language, oblique: oblique),
              let monthWord = monthName(month, language: language)
        else { return nil }
        var parts = [dayWord]
        if !r.dayMonthInfix.isEmpty { parts.append(r.dayMonthInfix) }
        parts.append(monthWord)
        if let y = year {
            if !r.monthYearInfix.isEmpty { parts.append(r.monthYearInfix) }
            parts.append(sayYear(y, language: language))
        }
        return parts.joined(separator: " ")
    }

    // MARK: written forms

    private static let iso = try! NSRegularExpression(
        pattern: #"(?<![\d.,:/-])([12][0-9]{3})-([01][0-9])-([0-3][0-9])(?![\d-])"#)
    /// With the year, which is what makes it a date rather than a guess. The
    /// yearless `12.3.` is deliberately not matched — see the type's note.
    private static let dotted = try! NSRegularExpression(
        pattern: #"(?<![\d.,:/-])([0-3]?[0-9])\.([01]?[0-9])\.([12][0-9]{3})\b"#)
    /// Day-first in every language here; English is handled in the callback,
    /// where the field order is genuinely ambiguous.
    private static let slashed = try! NSRegularExpression(
        pattern: #"(?<![\d.,:/-])([0-3]?[0-9])/([01]?[0-9])/([12][0-9]{3})(?![\d/])"#)

    /// Every written date in `text`, said the way `language` says it.
    ///
    /// Never throws and never invents: a run failing the bounds check, or whose
    /// field order cannot be resolved, comes back exactly as it was written.
    public static func expandDates(_ text: String, language: String) -> String {
        guard let r = rules[language] else { return text }
        var out = text

        out = replace(out, iso) { groups, at, whole in
            guard let y = Int(groups[1]), let m = Int(groups[2]), let d = Int(groups[3]),
                  valid(day: d, month: m, year: y)
            else { return nil }
            return spoken(day: d, month: m, year: y, language: language,
                          oblique: isOblique(at: at, in: whole, r))
        }
        out = replace(out, dotted) { groups, at, whole in
            // Swedish marks an ordinal with a colon (`1:a`), never a trailing
            // period, so `12.` there is a list number or a sentence end. English
            // writes dotted dates almost never, and when it does the field order
            // is as unresolvable as in the slashed form.
            guard !r.noDottedDates, !r.dottedIsAmbiguous else { return nil }
            guard let d = Int(groups[1]), let m = Int(groups[2]), let y = Int(groups[3]),
                  valid(day: d, month: m, year: y)
            else { return nil }
            return spoken(day: d, month: m, year: y, language: language,
                          oblique: isOblique(at: at, in: whole, r))
        }
        out = replace(out, slashed) { groups, at, whole in
            guard let d = Int(groups[1]), let m = Int(groups[2]), let y = Int(groups[3])
            else { return nil }
            // `3/12/2026` is March twelfth to half the English-speaking world
            // and the third of December to the other half, and nothing in the
            // string says which.
            if language == "en", d <= 12 { return nil }
            guard valid(day: d, month: m, year: y) else { return nil }
            return spoken(day: d, month: m, year: y, language: language,
                          oblique: isOblique(at: at, in: whole, r))
        }
        return textual(out, language: language, r)
    }

    /// `12 marca 2026`, `12. März 2026`, `March 12, 2026` — a written month name
    /// beside a bare day. The name is the disambiguator, so this runs for every
    /// language including English.
    private static func textual(_ text: String, language: String, _ r: Rules) -> String {
        guard r.months.count == 12 else { return text }
        let names = r.months.map { NSRegularExpression.escapedPattern(for: $0) }
            .joined(separator: "|")
        // Spanish and Portuguese speak a preposition between every part, so the
        // written form carries it too: "12 de marzo de 2026". Optional in the
        // pattern rather than a second pattern, because a language either has
        // the infix everywhere or nowhere.
        let infix = r.dayMonthInfix.isEmpty
            ? "" : #"(?:\s+\#(NSRegularExpression.escapedPattern(for: r.dayMonthInfix)))?"#
        let yinfix = r.monthYearInfix.isEmpty
            ? "" : #"(?:\s+\#(NSRegularExpression.escapedPattern(for: r.monthYearInfix)))?"#

        var out = text
        let dayFirstPattern =
            #"(?<![\w])([0-3]?[0-9])\.?\#(infix)\s+(\#(names))(?:\#(yinfix)\s+([12][0-9]{3}))?(?!\w)"#
        if let dayFirst = try? NSRegularExpression(
            pattern: dayFirstPattern, options: [.caseInsensitive]) {
            out = replace(out, dayFirst) { groups, at, whole in
                guard let d = Int(groups[1]), let m = monthIndex(groups[2], r) else { return nil }
                let y = groups.count > 3 ? Int(groups[3]) : nil
                guard valid(day: d, month: m, year: y) else { return nil }
                if !r.dayFirstPrefix.isEmpty || !r.dayFirstInfix.isEmpty {
                    // English written day-first reads "the twelfth of March":
                    // both dialects say it that way, so no locale flag is
                    // needed to choose.
                    guard let head = ordinalDay(d, language: language),
                          let monthWord = monthName(m, language: language) else { return nil }
                    var rest = [monthWord]
                    if let y { rest.append(sayYear(y, language: language)) }
                    let prefix = r.dayFirstPrefix.isEmpty ? "" : "\(r.dayFirstPrefix) "
                    let join = r.dayFirstInfix.isEmpty ? " " : " \(r.dayFirstInfix) "
                    return "\(prefix)\(head)\(join)\(rest.joined(separator: " "))"
                }
                return spoken(day: d, month: m, year: y, language: language,
                              oblique: isOblique(at: at, in: whole, r))
            }
        }

        // Month-first is an English shape. Reading it in a language that never
        // writes it would be inventing a construction nobody used.
        guard !r.dayFirstInfix.isEmpty else { return out }
        let monthFirstPattern =
            #"(?<![\w])(\#(names))\s+([0-3]?[0-9])(?:(?:st|nd|rd|th)\b)?,?(?:\s+([12][0-9]{3}))?(?!\w)"#
        guard let monthFirst = try? NSRegularExpression(
            pattern: monthFirstPattern, options: [.caseInsensitive]) else { return out }
        return replace(out, monthFirst) { groups, _, _ in
            guard let m = monthIndex(groups[1], r), let d = Int(groups[2]) else { return nil }
            let y = groups.count > 3 ? Int(groups[3]) : nil
            guard valid(day: d, month: m, year: y),
                  let monthWord = monthName(m, language: language),
                  let dayWord = ordinalDay(d, language: language) else { return nil }
            var parts = [monthWord, dayWord]
            if let y { parts.append(sayYear(y, language: language)) }
            return parts.joined(separator: " ")
        }
    }

    private static func monthIndex(_ name: String, _ r: Rules) -> Int? {
        let lowered = name.lowercased()
        for (i, candidate) in r.months.enumerated() where candidate.lowercased() == lowered {
            return i + 1
        }
        return nil
    }

    /// German only: `am`/`den`/`vom` before the day select the `-en` ending.
    private static func isOblique(at: Int, in whole: String, _ r: Rules) -> Bool {
        guard !r.obliqueTriggers.isEmpty else { return false }
        let ns = whole as NSString
        guard at <= ns.length else { return false }
        let before = ns.substring(to: at).replacingOccurrences(
            of: #"\s+$"#, with: "", options: .regularExpression)
        guard let last = before.split(separator: " ").last else { return false }
        let tail = last.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: ",;:"))
        return r.obliqueTriggers.contains { $0.lowercased() == tail }
    }

    // MARK: ordinals

    /// `value` as a written-out ordinal, or `nil` if this language has no table.
    ///
    /// Composed rather than enumerated past ninety-nine: the hundreds and above
    /// stay cardinal and only the last two digits become an ordinal, so *101st*
    /// is "one hundred and first". The irregulars a suffix rule gets wrong —
    /// fifth, eighth, ninth, twelfth, twentieth — are inside the two-digit
    /// tables and written out there.
    public static func ordinal(_ value: Int, language: String) -> String? {
        guard let r = rules[language], !r.ordinalUnits.isEmpty, value >= 0 else { return nil }
        let head = value / 100, rest = value % 100
        guard let tail = twoDigitOrdinal(rest, r) else { return nil }
        if head == 0 { return tail }
        let lead = card(head * 100, language)
        return rest != 0 ? "\(lead) \(tail)" : lead
    }

    private static func twoDigitOrdinal(_ value: Int, _ r: Rules) -> String? {
        if let teen = r.ordinalTeens[value] { return teen }
        let tens = value / 10, units = value % 10
        if units == 0 { return r.ordinalTens[tens * 10] }
        if tens == 0 { return r.ordinalUnits[units] }
        guard let unitWord = r.ordinalUnits[units] else { return nil }
        // The tens word is English's, because English is the only language of
        // the twelve that writes an ordinal as digits plus a suffix.
        return "\(card(tens * 10, "en"))\(r.ordinalJoiner)\(unitWord)"
    }

    /// `1st` and `22nd` as words.
    ///
    /// English is the only one of the twelve writing an ordinal as digits plus a
    /// letter suffix, so for every other language this is a no-op. It runs
    /// before the number pass, which would otherwise expand the digits and leave
    /// the suffix stuck to them: *onest*, *fiveth place*, *twenty-twond*.
    ///
    /// A value the tables cannot say is left exactly as written, suffix
    /// included, rather than half-said.
    public static func expandOrdinals(_ text: String, language: String) -> String {
        guard let r = rules[language], !r.ordinalSuffixes.isEmpty else { return text }
        let suffixes = r.ordinalSuffixes.joined(separator: "|")
        guard let re = try? NSRegularExpression(
            pattern: #"\b([0-9]+)(\#(suffixes))\b"#, options: [.caseInsensitive])
        else { return text }
        return replace(text, re) { groups, _, _ in
            guard let n = Int(groups[1]) else { return nil }
            return ordinal(n, language: language)
        }
    }

    // MARK: substitution

    /// Replace every match, right to left so earlier offsets stay valid.
    ///
    /// The callback receives the capture groups (index 0 is the whole match),
    /// the match's offset, and the string being scanned — the last two because
    /// the German oblique test reads the word *before* the date. Returning `nil`
    /// leaves that match exactly as written, which is this module's answer
    /// whenever the evidence runs out.
    private static func replace(
        _ text: String,
        _ re: NSRegularExpression,
        _ body: ([String], Int, String) -> String?
    ) -> String {
        let ns = text as NSString
        let matches = re.matches(in: text, range: NSRange(location: 0, length: ns.length))
        guard !matches.isEmpty else { return text }
        var out = text
        for m in matches.reversed() {
            var groups: [String] = []
            for i in 0..<m.numberOfRanges {
                let r = m.range(at: i)
                groups.append(r.location == NSNotFound ? "" : ns.substring(with: r))
            }
            // Trailing optional groups that did not participate come back empty;
            // drop them so `groups.count` says how many actually matched.
            while groups.count > 1, groups.last == "" { groups.removeLast() }
            guard let said = body(groups, m.range.location, text) else { continue }
            out = (out as NSString).replacingCharacters(in: m.range, with: said)
        }
        return out
    }
}
