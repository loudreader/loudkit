import Foundation
import XCTest

/// The ```swift blocks on the front page and the landing page compile against
/// this package.
///
/// Every block that imports `LoudKit` becomes `main.swift` of an executable
/// target in a throwaway package that depends on this repository by path, and
/// `swift build` is the judge. Weight-free: nothing is run. The first build in
/// a scratch directory takes seconds, and later ones are incremental.
final class SnippetTests: XCTestCase {
    static let pages = ["README.md", "site/src/handwritten/index.mdx"]

    func testTheUserPageSwiftSnippetsBuild() throws {
        let root = Fixture.repoRoot
        var targets: [(name: String, page: String, body: String)] = []
        for page in Self.pages {
            let text = try String(contentsOf: root.appendingPathComponent(page), encoding: .utf8)
            let blocks = Self.swiftBlocks(text).enumerated().filter { $0.element.contains("import LoudKit") }
            XCTAssertFalse(
                blocks.isEmpty, "\(page) shows no Swift block that imports LoudKit; the gate is looking at nothing")
            for (i, body) in blocks {
                let stem = page.replacingOccurrences(of: "[/.-]", with: "_", options: .regularExpression)
                targets.append(("Snippet_\(stem)_\(i + 1)", page, body))
            }
        }

        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("loudkit-snippet-\(ProcessInfo.processInfo.processIdentifier)")
        try? FileManager.default.removeItem(at: scratch)
        defer { try? FileManager.default.removeItem(at: scratch) }
        for target in targets {
            let dir = scratch.appendingPathComponent("Sources/\(target.name)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try target.body.write(
                to: dir.appendingPathComponent("main.swift"), atomically: true, encoding: .utf8)
        }
        // A path dependency is named after its directory, whatever the manifest says.
        let package = root.lastPathComponent.lowercased()
        let manifest = """
            // swift-tools-version: 5.9
            import PackageDescription
            let package = Package(
                name: "Snippet",
                platforms: [.macOS(.v14)],
                dependencies: [.package(path: "\(root.path)")],
                targets: [
            \(targets.map {
                "        .executableTarget(name: \"\($0.name)\", "
                    + "dependencies: [.product(name: \"LoudKit\", package: \"\(package)\")]),"
            }.joined(separator: "\n"))
                ]
            )
            """
        try manifest.write(
            to: scratch.appendingPathComponent("Package.swift"), atomically: true, encoding: .utf8)

        // Warm across runs: the dependency's build sits beside this package's own.
        let build = root.appendingPathComponent(".build/snippets").path
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "swift", "build", "--package-path", scratch.path, "--scratch-path", build
        ]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        process.waitUntilExit()
        let blocks = targets.map { "\($0.page):\n\($0.body)" }.joined(separator: "\n")
        XCTAssertEqual(
            process.terminationStatus, 0,
            "a Swift block on a user page does not build:\n\(output)\n--- the blocks ---\n\(blocks)")
    }

    /// Every ```swift fence on a page, with a tab component's indentation removed.
    static func swiftBlocks(_ text: String) -> [String] {
        var blocks: [String] = []
        var lines: [String]?
        for line in text.components(separatedBy: "\n") {
            let stripped = line.drop(while: { $0 == " " || $0 == "\t" })
            if var open = lines {
                if stripped.hasPrefix("```") {
                    blocks.append(dedent(open))
                    lines = nil
                } else {
                    open.append(line)
                    lines = open
                }
            } else if Self.fenceTag(String(stripped)) == "swift" {
                lines = []
            }
        }
        return blocks
    }

    /// The language a fence opens with: `swift` for "```swift", not for "```swiftui".
    static func fenceTag(_ line: String) -> String? {
        guard line.hasPrefix("```") else { return nil }
        return line.dropFirst(3).split(whereSeparator: { $0 == " " || $0 == "\t" }).first.map(String.init)
    }

    static func dedent(_ lines: [String]) -> String {
        let indent = lines
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .map { $0.prefix(while: { $0 == " " || $0 == "\t" }).count }
            .min() ?? 0
        return lines.map { String($0.dropFirst(min(indent, $0.count))) }.joined(separator: "\n") + "\n"
    }
}
