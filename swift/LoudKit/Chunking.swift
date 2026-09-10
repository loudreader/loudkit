/// Python reference: `loudkit/frontend/chunking.py`.
import Foundation

/// Splitting text that is longer than one window.
///
/// A window carries 255 speech tokens, about ten seconds of speech, and about
/// 127 characters of prepared text. Anything longer has to be split, generated
/// in pieces and joined, and **where the splits fall is audible**: a break at a
/// full stop is inaudible, a break mid-clause is not. That makes it an
/// algorithm-layer decision rather than a caller's convenience, which is why it
/// lives in `AlgorithmConfig` and has to be identical in every port.
///
/// A caller who splits differently gets different chunk boundaries,
/// therefore different derived seeds, therefore **different audio from every
/// other port**, while `AlgorithmConfig.fingerprint()` goes on declaring the
/// chunking recipe they are not applying.
///
/// Not every period-space is a full stop, and the period is the one separator
/// that is ambiguous: `!` and `?` are unmistakable sentence ends and `;` and
/// `,` are not sentence ends at all. Breaking `"But Mr. Smith went home"`
/// after the title hands the renderer a seven-character chunk: its own
/// utterance, its own derived seed, and a token ceiling proportional to seven
/// characters. See `ChunkConfig.midSentencePeriod`.
public enum Chunking {
    /// Exactly what `str.strip()` removes, which is what every trim in this
    /// file means.
    ///
    /// `CharacterSet.whitespacesAndNewlines` was standing here and it is the
    /// wrong set twice over. It **contains** U+200B ZERO WIDTH SPACE, which
    /// the reference does not strip, so a chunk beginning with one lost it and
    /// every later boundary in that passage shifted with the length. And it is
    /// **missing** U+001C to U+001F, which the reference does strip, so a
    /// chunk edged by a separator character kept it where the reference did
    /// not. Both directions move the cut, and a moved cut is a different
    /// derived seed and therefore different audio.
    ///
    /// The 29 code points are `str.isspace()`, enumerated rather than named by
    /// a Foundation constant, for the reason the funnel's own table is written
    /// out beside it: a port whose set depends on which ICU it links against
    /// agrees by coincidence. It is deliberately *not* the funnel's
    /// `WHITE_SPACE`, which is Unicode White_Space and stops four short of
    /// this: the funnel means "what a reader hears as a gap" and this means
    /// "what the reference's strip removes".
    static let strippable: CharacterSet = {
        var set = CharacterSet()
        for scalar: Unicode.Scalar in [
            "\u{0009}", "\u{000A}", "\u{000B}", "\u{000C}", "\u{000D}",
            "\u{001C}", "\u{001D}", "\u{001E}", "\u{001F}", "\u{0020}",
            "\u{0085}", "\u{00A0}", "\u{1680}",
            "\u{2000}", "\u{2001}", "\u{2002}", "\u{2003}", "\u{2004}",
            "\u{2005}", "\u{2006}", "\u{2007}", "\u{2008}", "\u{2009}",
            "\u{200A}", "\u{2028}", "\u{2029}", "\u{202F}", "\u{205F}",
            "\u{3000}",
        ] {
            set.insert(scalar)
        }
        return set
    }()

    /// Characters of prepared text per speech token.
    ///
    /// Measured on the reference voice across English, Polish (after the
    /// respelling funnel) and German: 0.53–0.64. The constant is the **low end
    /// with margin** (0.5 < the 0.53 measured minimum) because it is used to
    /// *stay under* a limit, never to predict a length: picking the middle of
    /// the range would let the worst case overflow the window more often still.
    ///
    /// It is a budget, not a guarantee. Measured over 9920 rendered chunks in
    /// ten languages, 54 overran the window anyway: a mean cannot bound a
    /// variance, and most of those are chunks that should have fitted and did
    /// not because the model emitted no stop token in time. An overflow is not
    /// an error either: the generator stops at the cap mid-word and the
    /// remainder is never spoken, which is what `cap_resplit` and
    /// `split_in_half` exist for.
    ///
    /// It must equal `loudkit.frontend.chunking.CHARS_PER_TOKEN`: a different value is a
    /// different set of joins and therefore a different reading.
    public static let charsPerToken = 0.5

    /// A conservative upper estimate of the speech tokens `text` will produce.
    ///
    /// Counts Unicode scalars, not UTF-16 code units: `String.count` in Swift
    /// counts grapheme clusters, which is a third answer again. The other four
    /// ports count code points, and the fixture is the arbiter.
    public static func estimateTokens(_ text: String) -> Int {
        Int(Double(text.unicodeScalars.count) / charsPerToken) + 1
    }

    /// Split `text` into pieces that each fit one window, in order, together
    /// covering the input. Never empty for non-empty input.
    ///
    /// Indexed by Unicode scalar rather than by `String.Index` arithmetic on
    /// UTF-16 offsets: a cut in the wrong units lands inside a character, and
    /// that is the shape of bug every port here has had at least once.
    public static func splitText(_ text: String, config: ChunkConfig) -> [String] {
        let trimmed = text.trimmingCharacters(in: strippable)
        if trimmed.isEmpty { return [] }
        if !config.enabled || estimateTokens(trimmed) <= config.maxTokens { return [trimmed] }

        let budget = Int(Double(config.maxTokens) * charsPerToken)
        var chunks: [String] = []
        var rest = Array(trimmed.unicodeScalars)

        while !rest.isEmpty {
            if rest.count <= budget {
                chunks.append(scalars(rest).trimmingCharacters(in: strippable))
                break
            }
            let headScalars = Array(rest.prefix(budget + 1))
            let head = scalars(headScalars)
            // One scalar past the window, and used only by `holds`: the latest
            // candidate can end the window exactly, and the test reads the
            // character after it. The search itself stays inside the budget.
            let look = Array(rest.prefix(budget + 2))
            var cut = -1
            // Strongest separator first, and within a separator the LATEST
            // break, so chunks run as long as they may rather than as short as
            // they can.
            for separator in config.splitOn {
                let sepScalars = Array(separator.unicodeScalars)
                var before = lastIndex(of: sepScalars, in: headScalars, before: headScalars.count)
                // A period inside a sentence is not a boundary, so the search
                // keeps walking back through this separator's own occurrences
                // before it gives up and tries a weaker one. Searching before
                // the held one skips an occurrence overlapping it, which no
                // separator here can have.
                while before > 0 && holds(look, before, sepScalars, config) {
                    before = lastIndex(of: sepScalars, in: headScalars, before: before)
                }
                if before > 0 {
                    cut = before + sepScalars.count
                    break
                }
            }
            if cut <= 0 {
                // No punctuation in a whole window's worth of text. Break at the
                // last word boundary; it will be heard, and that is the point.
                //
                // Scanning scalars, not `range(of:)`: that search is
                // grapheme-cluster and canonically equivalent, so a space
                // followed by a combining mark is one cluster and not equal to
                // " ", and Swift saw no boundary where the other four did. The
                // rest of this function was already hand-rolled for that reason.
                // `wordBoundaries`, not U+0020, because NBSP survives the funnel.
                let scalars = Array(head.unicodeScalars)
                var before = -1
                for (i, s) in scalars.enumerated() where wordBoundaries.contains(s) {
                    before = i
                }
                if before > 0 { cut = before }
            }
            if cut <= 0 {
                cut = budget  // one unbroken token longer than a window: mid-word
            }
            // Never zero: a cut of 0 leaves `rest` unchanged and the loop spins
            // forever.
            cut = max(cut, 1)

            chunks.append(
                scalars(Array(rest.prefix(cut))).trimmingCharacters(in: strippable))
            rest = Array(rest.dropFirst(cut))
            while let first = rest.first, strippable.contains(first) {
                rest.removeFirst()
            }
        }
        return chunks.filter { !$0.isEmpty }
    }

    /// The latest start offset of `needle` in `haystack` strictly below `limit`,
    /// or -1. Scalar offsets throughout, so the units never change hands.
    private static func lastIndex(
        of needle: [Unicode.Scalar], in haystack: [Unicode.Scalar], before limit: Int
    ) -> Int {
        if needle.isEmpty { return -1 }
        var start = min(limit, haystack.count) - needle.count
        while start >= 0 {
            if matches(needle, in: haystack, at: start) { return start }
            start -= 1
        }
        return -1
    }

    /// Whether `haystack` carries `needle` starting at `start`.
    ///
    /// Scalar by scalar rather than through `String`: Swift's string comparison
    /// is canonical, so a decomposed "\u{15B}" would match a composed one here
    /// and in no other port. Exact code points are the contract.
    private static func matches(
        _ needle: [Unicode.Scalar], in haystack: [Unicode.Scalar], at start: Int
    ) -> Bool {
        if start < 0 || start + needle.count > haystack.count { return false }
        for (i, scalar) in needle.enumerated() where haystack[start + i] != scalar {
            return false
        }
        return true
    }

    /// Whether `scalar` is an ASCII letter or digit.
    ///
    /// A scalar test, not `CharacterSet.alphanumerics`: the five
    /// implementations have to answer this identically, and every language's
    /// idea of "letter" is its own. ASCII is the part they cannot disagree on.
    private static func isASCIIWord(_ scalar: Unicode.Scalar) -> Bool {
        let v = scalar.value
        return (v >= 48 && v <= 57) || (v >= 65 && v <= 90) || (v >= 97 && v <= 122)
    }

    /// Whether the candidate at scalar offset `at` is a period inside a
    /// sentence rather than the end of one.
    ///
    /// `look` is the search window plus one scalar, because the test below
    /// reads the character *after* the separator and the latest candidate can
    /// end the window exactly.
    ///
    /// Gated on the period: `"! "` and `"? "` end sentences and `"; "` and
    /// `", "` do not end them at all, so neither is ever in doubt.
    private static func holds(
        _ look: [Unicode.Scalar], _ at: Int, _ separator: [Unicode.Scalar],
        _ config: ChunkConfig
    ) -> Bool {
        guard config.midSentencePeriod == .hold, separator.first == "." else { return false }
        // A sentence does not resume in lower case. This is what catches the
        // ellipsis the funnel folds to a single period, which no abbreviation
        // list reaches. ASCII only, and measured rather than assumed: over 2253
        // periods in ten languages, four are followed by a word starting with a
        // non-ASCII lowercase letter, and reading the whole Unicode Lowercase
        // property instead moves one passage in 1200.
        let after = at + separator.count
        if after < look.count {
            let v = look[after].value
            if v >= 97 && v <= 122 { return true }
        }
        // Or the token in front of the period is a listed abbreviation. Entries
        // carry no period of their own, the period belongs to the separator.
        for abbreviation in config.abbreviations {
            let entry = Array(abbreviation.unicodeScalars)
            let before = at - entry.count
            if before < 0 || !matches(entry, in: look, at: before) { continue }
            // The tail of a longer word, not a word of its own.
            if before > 0 && isASCIIWord(look[before - 1]) { continue }
            return true
        }
        return false
    }

    private static func scalars(_ values: [Unicode.Scalar]) -> String {
        var view = String.UnicodeScalarView()
        view.append(contentsOf: values)
        return String(view)
    }

    /// Characters ``splitInHalf(_:)`` may cut on, written out rather than
    /// tested for.
    ///
    /// A predicate would be shorter and the five ports do not agree on one:
    /// measured, Python's `str.isspace()` treats U+001C-U+001F as whitespace
    /// where the other four do not, and JS alone KEEPS U+0085 where the other
    /// four strip it. Swift alone strips U+200B, JS alone strips U+FEFF; the
    /// funnel removes both before the splitter sees them, but U+0085 and
    /// U+001C-U+001F survive it. A disagreement there is a different split
    /// point, which is different audio for the same text and seed. A hand-
    /// written set cannot drift.
    ///
    /// The funnel does not remove these. NBSP in particular is ordinary in real
    /// prose ("10 000", French punctuation, typeset copy), and a capped chunk
    /// whose only boundaries are NBSP comes back unsplittable and ships its
    /// truncation unless NBSP is in this set.
    public static let wordBoundaries: Set<Unicode.Scalar> = [
        "\u{0020}", "\u{0009}", "\u{000a}", "\u{000d}", "\u{00a0}", "\u{2007}", "\u{202f}",
    ]

    /// Halve a chunk the window could not hold, at a word boundary.
    ///
    /// ``splitText(_:config:)``'s estimate is conservative but not a guarantee:
    /// it budgets characters against a constant, and a speaker slower than that
    /// constant fills the window before the text runs out. The generator then
    /// stops at the cap mid-word, and the words that did not fit are *lost*
    /// rather than deferred, because chunk texts are fixed before any of them
    /// is rendered. Measured across ten languages, 54 of 9920 chunks reached a
    /// cap and 30 were still speaking when it closed, over five voices; gating
    /// on the window leaves 51 and 27, on two.
    ///
    /// The boundary is the nearest WORD break, and punctuation is not sought.
    /// The reason is mechanical rather than comparative: a comma is an
    /// instruction to pause, this model has no pause-duration prior, and fed
    /// one it overshoots. No measurement compares seeking punctuation against
    /// this law; see `docs/design/text-funnel.md`. Not sought is not avoided:
    /// the nearest word break can follow a comma.
    ///
    /// `nil` for a single unbroken run. Splitting it would have to cut a word,
    /// which is worse than the truncation it would be repairing.
    public static func splitInHalf(_ text: String) -> (String, String)? {
        // Unicode scalars, not `Character`. `Array(text)` yields grapheme
        // clusters, and CRLF is ONE cluster in Swift and TWO scalars in the
        // other four ports -- so a passage with Windows line endings, which the
        // funnel passes through verbatim, would be halved at a different point
        // here than everywhere else. Same text, same seed, different audio.
        let chars = Array(text.unicodeScalars)
        let middle = Double(chars.count) / 2
        var best: (distance: Double, at: Int)?
        for (i, c) in chars.enumerated() {
            // Interior only: a boundary at either end yields an empty half.
            guard wordBoundaries.contains(c), i > 0, i + 1 < chars.count else { continue }
            let distance = abs(Double(i) - middle)
            // `.infinity` for "no boundary yet": the first one always wins.
            if distance < (best?.distance ?? .infinity) {
                best = (distance, i + 1)
            }
        }
        guard let at = best?.at else { return nil }
        let first = String(String.UnicodeScalarView(chars[..<at]))
            .trimmingCharacters(in: strippable)
        let second = String(String.UnicodeScalarView(chars[at...]))
            .trimmingCharacters(in: strippable)
        // Trimming can empty a half the scan thought was interior, on input
        // whose boundary run is all whitespace. `splitText` trims before this
        // is ever called, so the engine cannot reach it, but this is public and
        // a caller handed an empty half would render silence and call it speech.
        if first.isEmpty || second.isEmpty { return nil }
        return (first, second)
    }
}
