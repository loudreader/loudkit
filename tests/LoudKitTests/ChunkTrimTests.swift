import Foundation
import XCTest

@testable import LoudKit

/// The splitter trims exactly what `str.strip()` trims.
///
/// `CharacterSet.whitespacesAndNewlines` was the wrong set twice over: it
/// contains U+200B ZERO WIDTH SPACE, which the reference does not strip, and it
/// is missing U+001C to U+001F, which the reference does. Because a stripped
/// character changes the chunk's length, every later boundary in the passage
/// moves with it, and a moved boundary is a different derived seed and
/// different audio. The engine funnels before it chunks and the funnel removes
/// both, so no render reaches this; `Chunking.splitText` and
/// `Chunking.splitInHalf` are public and documented, and a caller who chunks
/// their own text got different boundaries on Apple than everywhere else.
///
/// Expectations are `loudkit.frontend.chunking`'s answers, measured.
final class ChunkTrimTests: XCTestCase {
    /// Long enough that the shipping recipe has to split it more than once.
    private let long = """
        measurement lazy punctuation quick brown fox jumps over the sleeping \
        dog again and again with more words to overflow the window twice over \
        so that the splitter has to choose. And a second sentence follows it \
        here, with a comma, and then more words to push past the budget again.
        """

    /// The reference's answers for the head of the first chunk and the first
    /// half, with nothing at the edge to trim.
    private let firstChunk = """
        measurement lazy punctuation quick brown fox jumps over the sleeping \
        dog again and again with more words to overflow the window
        """
    private let firstHalf = """
        measurement lazy punctuation quick brown fox jumps over the sleeping \
        dog again and again with more words to overflow the window twice over
        """

    /// U+200B is not whitespace to the reference, so it stays at the edge --
    /// and because it stays, the window fills one word earlier.
    func testZeroWidthSpaceSurvivesTheTrim() {
        let config = ChunkConfig()
        let shorter = String(firstChunk.dropLast(" window".count))
        XCTAssertEqual(
            Chunking.splitText("\u{200B} " + long, config: config).first,
            "\u{200B} " + shorter)
        XCTAssertEqual(
            Chunking.splitText("\u{200B}" + long, config: config).first,
            "\u{200B}" + shorter)
        XCTAssertEqual(
            Chunking.splitInHalf("\u{200B} " + long)?.0, "\u{200B} " + firstHalf)
        XCTAssertEqual(
            Chunking.splitInHalf("\u{200B}" + long)?.0, "\u{200B}" + firstHalf)
    }

    /// U+001C to U+001F are whitespace to the reference, so they go.
    func testSeparatorCharactersAreTrimmed() {
        let config = ChunkConfig()
        for separator in ["\u{001C}", "\u{001D}", "\u{001E}", "\u{001F}"] {
            let code = separator.unicodeScalars.first!.value
            XCTAssertEqual(
                Chunking.splitText(separator + " " + long, config: config).first,
                firstChunk, "leading U+\(String(code, radix: 16)) was kept")
            XCTAssertEqual(
                Chunking.splitText(separator + long, config: config).first,
                firstChunk, "glued U+\(String(code, radix: 16)) was kept")
            XCTAssertEqual(
                Chunking.splitInHalf(separator + " " + long)?.0,
                firstHalf, "splitInHalf kept U+\(String(code, radix: 16))")
        }
    }

    /// The set itself, so a later edit cannot quietly widen or narrow it.
    func testTheTrimSetIsExactlyTheReferenceStripSet() {
        let want: Set<UInt32> = [
            0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x001C, 0x001D, 0x001E,
            0x001F, 0x0020, 0x0085, 0x00A0, 0x1680, 0x2000, 0x2001, 0x2002,
            0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A,
            0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
        ]
        var got: Set<UInt32> = []
        for cp in UInt32(0)...0xFFFF {
            guard let scalar = Unicode.Scalar(cp) else { continue }
            if Chunking.strippable.contains(scalar) { got.insert(cp) }
        }
        XCTAssertEqual(got, want)
    }
}
