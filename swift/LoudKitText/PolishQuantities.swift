import Foundation

/// Agreement for written prices, measurements and common selection prompts.
/// Leave the digits for the shared cardinal pass, which also handles signs.
enum PolishQuantities {
    private static let amount = try! NSRegularExpression(
        pattern: #"([0-9]+(?:[ \u00A0\u202F][0-9]{3})*)(?:,([0-9]+))?(?:[ \t\u00A0\u202F]*(zł|kg|m²))?"#)
    private static let choice = try! NSRegularExpression(
        pattern: #"([Ww]ybierz|[Zz]aznacz|[Ww]skaż) +([1-9]) +z +(10|[2-9]) +(ofert|opcji|pozycji|propozycji|wiadomości|produktów|plików|biletów|elementów)"#)
    private static let feminine: Set<String> = ["ofert", "opcji", "pozycji", "propozycji", "wiadomości"]
    private static let messages = try! NSRegularExpression(pattern: #"([Mm]asz) +2( +(?:nowe +)?wiadomości)"#)
    private static let genitives = [
        "dwóch", "trzech", "czterech", "pięciu", "sześciu", "siedmiu", "ośmiu", "dziewięciu", "dziesięciu",
    ]
    private static let złoty = ["złoty", "złote", "złotych"]
    private static let grosz = ["grosz", "grosze", "groszy"]
    private static let kilograms = ["kilogram", "kilogramy", "kilogramów"]
    private static let squareMetres = ["metr kwadratowy", "metry kwadratowe", "metrów kwadratowych"]

    static func expand(_ text: String) -> String {
        let choices = replace(choice, in: text) { match, ns in
            let chosen = ns.substring(with: match.range(at: 2))
            let total = ns.substring(with: match.range(at: 3))
            guard let count = Int(total) else { return nil }
            let verb = ns.substring(with: match.range(at: 1))
            let noun = ns.substring(with: match.range(at: 4))
            let numeral = feminine.contains(noun) ? (["1": "jedną", "2": "dwie"][chosen] ?? chosen) : chosen
            return "\(verb) \(numeral) z \(genitives[count - 2]) \(noun)"
        }
        let notifications = replace(messages, in: choices) { match, ns in
            ns.substring(with: match.range(at: 1)) + " dwie" + ns.substring(with: match.range(at: 2))
        }
        return replace(amount, in: notifications) { match, ns in
            let unitRange = match.range(at: 3)
            guard unitRange.location != NSNotFound else { return nil }
            // A suffix of a version or decimal is not a separate amount.
            let start = match.range.location
            if let prior = Numbers.scalarAt(ns, start - 1), prior == "." || prior == ",",
               let before = Numbers.scalarAt(ns, start - 2), ("0"..."9").contains(before) {
                return nil
            }
            let whole = ns.substring(with: match.range(at: 1))
            let parts = whole.split(whereSeparator: { $0.isWhitespace })
            guard parts.count == 1 || (parts.first?.count ?? 0) <= 3 else { return nil }
            let digits = parts.joined()
            guard digits.count <= 9, digits == "0" || digits.first != "0",
                  let value = Int64(digits) else { return nil }
            let fractionRange = match.range(at: 2)
            let fraction = fractionRange.location == NSNotFound ? "" : ns.substring(with: fractionRange)
            let unit = ns.substring(with: unitRange)
            if unit == "zł" {
                guard fraction.count <= 2 else { return nil }
                let cents = (Int64(fraction) ?? 0) * (fraction.count == 1 ? 10 : 1)
                let major = "\(digits) \(Numbers.scaleWord(value, złoty))"
                guard cents > 0 else { return major }
                let minor = "\(cents) \(Numbers.scaleWord(cents, grosz))"
                return value == 0 ? minor : "\(major) i \(minor)"
            }
            guard fraction.count <= 6 else { return nil }
            if fraction.contains(where: { $0 != "0" }) {
                let form = unit == "kg" ? "kilograma" : "metra kwadratowego"
                return "\(digits),\(fraction) \(form)"
            }
            return "\(digits) \(Numbers.scaleWord(value, unit == "kg" ? kilograms : squareMetres))"
        }
    }

    private static func word(_ scalar: Unicode.Scalar?) -> Bool {
        guard let scalar else { return false }
        return scalar == "_" || SpeechText.isLetter(scalar) || scalar.properties.numericType != nil
    }

    private static func replace(
        _ pattern: NSRegularExpression, in text: String,
        with transform: (NSTextCheckingResult, NSString) -> String?
    ) -> String {
        let ns = text as NSString
        var out = ""
        var start = 0
        for match in pattern.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let end = NSMaxRange(match.range)
            guard !word(Numbers.scalarAt(ns, match.range.location - 1)),
                  !word(Numbers.scalarAt(ns, end)), let spoken = transform(match, ns) else { continue }
            out += ns.substring(with: NSRange(location: start, length: match.range.location - start))
            out += spoken
            start = end
        }
        if start == 0 { return text }
        out += ns.substring(from: start)
        return out
    }
}
