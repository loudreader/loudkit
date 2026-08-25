/// Python reference: `loudkit/frontend/chunking.py`.
import Foundation

/// Splitting text that is longer than one window.
///
/// A window carries 255 speech tokens — about ten seconds of speech, and about
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
public enum Chunking {
    /// Characters of prepared text per speech token.
    ///
    /// Measured on the reference voice across English, Polish (after the
    /// respelling funnel) and German: 0.53–0.64. The constant is the **low end
    /// with margin** (0.5 < the 0.53 measured minimum) because it is used to
    /// *stay under* a limit, never to predict a length — picking the middle of
    /// the range would let the worst case overflow the window, and an overflow
    /// is a hard failure.
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
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return [] }
        if !config.enabled || estimateTokens(trimmed) <= config.maxTokens { return [trimmed] }

        let budget = Int(Double(config.maxTokens) * charsPerToken)
        var chunks: [String] = []
        var rest = Array(trimmed.unicodeScalars)

        while !rest.isEmpty {
            if rest.count <= budget {
                chunks.append(scalars(rest).trimmingCharacters(in: .whitespacesAndNewlines))
                break
            }
            let head = scalars(Array(rest.prefix(budget + 1)))
            var cut = -1
            // Strongest separator first, and within a separator the LATEST
            // break, so chunks run as long as they may rather than as short as
            // they can.
            for separator in config.splitOn {
                if let at = head.range(of: separator, options: .backwards) {
                    let before = head.unicodeScalars.distance(
                        from: head.unicodeScalars.startIndex, to: at.lowerBound)
                    if before > 0 {
                        cut = before + separator.unicodeScalars.count
                        break
                    }
                }
            }
            if cut <= 0, let at = head.range(of: " ", options: .backwards) {
                // No punctuation in a whole window's worth of text. Break at the
                // last word boundary; it will be heard, and that is the point.
                let before = head.unicodeScalars.distance(
                    from: head.unicodeScalars.startIndex, to: at.lowerBound)
                if before > 0 { cut = before }
            }
            if cut <= 0 {
                cut = budget  // one unbroken token longer than a window: mid-word
            }
            // Never zero: a cut of 0 leaves `rest` unchanged and the loop spins
            // forever.
            cut = max(cut, 1)

            chunks.append(
                scalars(Array(rest.prefix(cut))).trimmingCharacters(in: .whitespacesAndNewlines))
            rest = Array(rest.dropFirst(cut))
            while let first = rest.first, CharacterSet.whitespacesAndNewlines.contains(first) {
                rest.removeFirst()
            }
        }
        return chunks.filter { !$0.isEmpty }
    }

    private static func scalars(_ values: [Unicode.Scalar]) -> String {
        var view = String.UnicodeScalarView()
        view.append(contentsOf: values)
        return String(view)
    }
}
