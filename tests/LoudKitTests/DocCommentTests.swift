import XCTest

/// Every public declaration in the two library targets carries a doc comment,
/// and no doc line sits at column zero inside an indented block.
///
/// Both failures are invisible to the compiler and to swiftlint. A block that
/// drifts one declaration down still compiles: DocC attaches it to the
/// neighbour and the symbol it was written for goes undocumented, which is how
/// seven of these ended up describing the wrong thing. Reading the sources
/// rather than the built module because a doc comment leaves no trace in the
/// binary.
final class DocCommentTests: XCTestCase {
    private static var libraryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // LoudKitTests
            .deletingLastPathComponent()  // tests
            .deletingLastPathComponent()  // repository root
            .appendingPathComponent("swift")
    }

    private func sources() throws -> [(url: URL, lines: [String])] {
        var out: [(URL, [String])] = []
        for target in ["LoudKit", "LoudKitText"] {
            let dir = Self.libraryRoot.appendingPathComponent(target)
            let names = try FileManager.default.contentsOfDirectory(atPath: dir.path)
            for name in names.sorted() where name.hasSuffix(".swift") {
                let url = dir.appendingPathComponent(name)
                out.append((url, try String(contentsOf: url, encoding: .utf8)
                    .components(separatedBy: "\n")))
            }
        }
        XCTAssertGreaterThan(out.count, 20, "the library sources did not resolve")
        return out
    }

    func testEveryPublicDeclarationIsDocumented() throws {
        var undocumented: [String] = []
        for (url, lines) in try sources() {
            for (i, line) in lines.enumerated() {
                let text = line.trimmingCharacters(in: .whitespaces)
                guard text.hasPrefix("public ") else { continue }
                // A memberwise `public init() {}` on a struct of documented
                // fields says nothing a reader does not already have.
                if text == "public init() {}" { continue }
                var j = i - 1
                while j >= 0, lines[j].trimmingCharacters(in: .whitespaces).hasPrefix("@") {
                    j -= 1
                }
                if j >= 0, lines[j].trimmingCharacters(in: .whitespaces).hasPrefix("///") {
                    continue
                }
                undocumented.append("\(url.lastPathComponent):\(i + 1): \(text)")
            }
        }
        XCTAssertEqual(undocumented, [], "undocumented public declarations")
    }

    /// No em dash in the prose the package ships, comments and messages alike.
    ///
    /// A house rule the compiler cannot enforce, and the Python runtime was
    /// swept for it while the four ports were not. The one permitted occurrence
    /// is the footnote-marker character class, where the em dash is a character
    /// being matched rather than punctuation being written.
    func testNoEmDashInTheProseThePackageShips() throws {
        var offenders: [String] = []
        for (url, lines) in try sources() {
            for (i, line) in lines.enumerated() where line.contains("\u{2014}") {
                // The footnote-marker class: hyphen, en dash, em dash.
                if line.contains("\\-\u{2013}\u{2014}") { continue }
                offenders.append("\(url.lastPathComponent):\(i + 1): \(line.trimmingCharacters(in: .whitespaces))")
            }
        }
        XCTAssertEqual(offenders, [], "em dashes in shipped prose")
    }

    /// A doc line at column zero inside an indented block is a block that was
    /// pasted over another one: the merged paragraph then documents whichever
    /// declaration follows.
    func testNoDocLineSitsAtColumnZeroInsideAnIndentedBlock() throws {
        var stray: [String] = []
        for (url, lines) in try sources() {
            for (i, line) in lines.enumerated() where line.hasPrefix("///") {
                let previous = i > 0 ? lines[i - 1] : ""
                let indented = previous.hasPrefix("    ")
                    && previous.trimmingCharacters(in: .whitespaces).hasPrefix("///")
                if indented { stray.append("\(url.lastPathComponent):\(i + 1): \(line)") }
            }
        }
        XCTAssertEqual(stray, [], "doc lines that lost their indentation")
    }
}
