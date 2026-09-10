import CryptoKit
import Foundation
import XCTest

@testable import LoudKit

/// The fetch, against a Hub that is not there.
///
/// A `URLProtocol` answers the revision, the tree API and the `resolve/`
/// routes out of a dictionary, so the plan, the pagination, the receipt and
/// the verification are decidable in a second on a machine with no weights.
final class DownloadTests: XCTestCase {
    private var root = URL(fileURLWithPath: NSTemporaryDirectory())

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("loudkit-download-\(UUID().uuidString)")
        StubHub.reset()
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
        StubHub.reset()
    }

    /// The receipt file as written, unjudged: for forging one field of it.
    private func receiptOnDisk(_ root: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: root.appendingPathComponent(LoudKit.receiptName))
        else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    // MARK: the shared fixtures

    /// The glob is `fnmatch`, where `*` crosses `/`, which is how voices arrive.
    func testGlobIsFnmatch() throws {
        let cases = try Fixture.shared("glob.json")["cases"] as! [[String: Any]]
        XCTAssertGreaterThanOrEqual(cases.count, 10, "the fixture holds probes")
        for c in cases {
            let (pattern, name) = (c["pattern"] as! String, c["name"] as! String)
            XCTAssertEqual(LoudKit.matches(name, pattern), c["match"] as! Bool, "\(pattern) ~ \(name)")
        }
    }

    /// Compare selected files; Swift uses native generation and needs no T3 packages.
    func testThePlanIsTheSharedFixture() throws {
        let fixture = try Fixture.shared("release_plan.json")
        let listing = (fixture["listing"] as! [String]).map { LoudKit.RemoteFile(path: $0, size: 1) }
        var seen = 0
        for c in fixture["cases"] as! [[String: Any]] where c["backend"] as! String == "coreml" {
            seen += 1
            let cloning = c["cloning"] as! Bool
            let patterns = LoudKit.releasePatterns(cloning: cloning)
            XCTAssertEqual(patterns.ignore, c["ignore"] as! [String], "cloning=\(cloning) ignore")
            XCTAssertEqual(LoudKit.plan(listing, cloning: cloning).map(\.path).sorted(),
                           c["wanted"] as! [String], "cloning=\(cloning) plan")
        }
        XCTAssertEqual(seen, 2, "the fixture holds two coreml cases")
    }

    func testARepoIdIsWhatTheSharedFixtureSays() throws {
        let cases = try Fixture.shared("repo_id.json")["cases"] as! [[String: Any]]
        XCTAssertGreaterThanOrEqual(cases.count, 10, "the fixture holds probes")
        for c in cases {
            XCTAssertEqual(LoudKit.isRepoID(c["ref"] as! String), c["is_repo_id"] as! Bool,
                           c["ref"] as! String)
        }
        // A path that exists is a path, however it is spelled.
        XCTAssertFalse(LoudKit.isRepoID(NSTemporaryDirectory()))
    }

    // MARK: the fetch

    func testFetchesThePlanAndNothingElseBookkeepingFirst() async throws {
        let directory = try await StubHub.download(to: root)
        XCTAssertEqual(directory, root)
        let fm = FileManager.default
        XCTAssertTrue(fm.fileExists(atPath: directory.appendingPathComponent(
            "loudr-1.safetensors").path))
        XCTAssertTrue(fm.fileExists(atPath: directory.appendingPathComponent(
            "coreml/vocoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin").path))
        XCTAssertTrue(fm.fileExists(atPath: directory.appendingPathComponent(
            "voices/joe.safetensors").path))
        XCTAssertFalse(fm.fileExists(atPath: directory.appendingPathComponent(
            "onnx/t3_step.onnx").path))
        XCTAssertFalse(fm.fileExists(atPath: directory.appendingPathComponent(
            "ve.safetensors").path))
        // A repo that is not a release is refused before its weights move,
        // so the two files that say what it is come first.
        XCTAssertEqual(Array(StubHub.served.prefix(2)), ["SHA256SUMS", "release.json"])
        // What lands is a bundle, by the discovery rules, not just a pile.
        let bundle = try ModelBundle(directory: directory)
        XCTAssertEqual(bundle.voiceNames, ["ada", "joe"])
    }

    /// A repository big enough to page: a truncated listing is a fetch that
    /// silently leaves files behind.
    func testFollowsThePaginationCursor() async throws {
        StubHub.pageSize = 4
        let directory = try await StubHub.download(to: root)
        XCTAssertNoThrow(try ModelBundle(directory: directory))
        XCTAssertGreaterThan(StubHub.listings, 1, "the listing fitted in one page; not paged")
    }

    /// The receipt names the commit the revision still resolves to, so the
    /// second fetch is a hit. It reports no progress either: a hit neither
    /// fetches nor hashes, and has nothing to draw.
    func testASecondFetchMovesNoBytes() async throws {
        _ = try await StubHub.download(to: root)
        XCTAssertGreaterThan(StubHub.served.count, 0)
        StubHub.served = []
        let seen = Recorder()
        let directory = try await StubHub.download(to: root) { seen.record($0) }
        XCTAssertEqual(directory, root)
        XCTAssertEqual(StubHub.served, [], "a finished fetch re-downloaded")
        XCTAssertTrue(seen.all.isEmpty, "a hit reported progress, so it did some work")
    }

    /// A hit trusts the receipt: a file edited in place under a receipt that
    /// still matches is not looked at, which is what "hashes nothing" means.
    func testAHitHashesNothing() async throws {
        let directory = try await StubHub.download(to: root)
        let voice = directory.appendingPathComponent("voices/joe.safetensors")
        try Data(repeating: 0xEE, count: 256).write(to: voice)
        StubHub.served = []
        _ = try await StubHub.download(to: root)
        XCTAssertEqual(StubHub.served, [])
        XCTAssertEqual(try Data(contentsOf: voice), Data(repeating: 0xEE, count: 256))
    }

    /// The revision moved: the manifest comes again and every file on disk is
    /// hashed against it. A truncated voice is caught here, on the miss, and
    /// only it is fetched; the receipt then names the new commit. One more
    /// move, with a voice corrupted at its full length, fetches that one alone.
    func testAMovedRevisionRefetchesOnlyWhatDoesNotHash() async throws {
        let directory = try await StubHub.download(to: root)
        let joe = directory.appendingPathComponent("voices/joe.safetensors")
        try Data("tr".utf8).write(to: joe)
        StubHub.commit = String(repeating: "b", count: 40)
        StubHub.served = []
        _ = try await StubHub.download(to: root)
        XCTAssertEqual(StubHub.served, ["SHA256SUMS", "voices/joe.safetensors"])
        XCTAssertEqual(try Data(contentsOf: joe), StubHub.files["voices/joe.safetensors"])
        XCTAssertEqual(LoudKit.readReceipt(root, repo: StubHub.repo, cloning: false)?.commit,
                       StubHub.commit)

        let ada = directory.appendingPathComponent("voices/ada.safetensors")
        try Data(repeating: 0xEE, count: 256).write(to: ada)
        StubHub.commit = String(repeating: "c", count: 40)
        StubHub.served = []
        _ = try await StubHub.download(to: root)
        XCTAssertEqual(StubHub.served, ["SHA256SUMS", "voices/ada.safetensors"])
        XCTAssertEqual(try Data(contentsOf: ada), StubHub.files["voices/ada.safetensors"])
    }

    /// A receipt vouches for every file the plan selects from `SHA256SUMS`,
    /// not only the files the port cannot run without: a voice is one of
    /// many and its absence is a miss. An edit in place rides on a matching
    /// receipt, because a hit hashes nothing.
    func testAListedFileDeletedUnderAReceiptIsFetchedAgain() async throws {
        let directory = try await StubHub.download(to: root)
        let ada = directory.appendingPathComponent("voices/ada.safetensors")
        try FileManager.default.removeItem(at: ada)
        StubHub.served = []
        _ = try await StubHub.download(to: root)
        XCTAssertEqual(StubHub.served, ["SHA256SUMS", "voices/ada.safetensors"])
        XCTAssertEqual(try Data(contentsOf: ada), StubHub.files["voices/ada.safetensors"])
        try Data(repeating: 0xEE, count: 256).write(to: ada)
        StubHub.served = []
        _ = try await StubHub.download(to: root)
        XCTAssertEqual(StubHub.served, [])
        XCTAssertEqual(try Data(contentsOf: ada), Data(repeating: 0xEE, count: 256))
    }

    /// Offline, a pin is answered by the bytes it names or by nothing.
    ///
    /// A caller asking for one revision and getting another is the failure a
    /// pin exists to prevent, so the receipt's own commit and ref are what the
    /// request is held to. `main` is this port's unpinned request, because
    /// `revision` defaults to it and cannot be told from one a caller wrote.
    func testTheHubOutOfReachRefusesACacheThatIsNotThePin() async throws {
        _ = try await StubHub.download(to: root)
        StubHub.offline = true

        // Forty hex names a commit, which the receipt records exactly.
        do {
            _ = try await LoudKit.download(StubHub.fetch(revision: String(repeating: "b", count: 40)),
                                           to: root, progress: nil)
            XCTFail("a cache at another commit answered a pinned request")
        } catch let error as LoudKitError {
            XCTAssertTrue("\(error)".contains("not a preference"), "\(error)")
        }
        let atCommit = try await LoudKit.download(StubHub.fetch(revision: StubHub.commit),
                                                  to: root, progress: nil)
        XCTAssertEqual(atCommit, root)

        // A ref cannot be resolved with no hub, and the receipt records which
        // one this copy answers for.
        do {
            _ = try await LoudKit.download(StubHub.fetch(revision: "v0.2.0"), to: root,
                                           progress: nil)
            XCTFail("a cache fetched for another ref answered a pinned request")
        } catch let error as LoudKitError {
            XCTAssertTrue("\(error)".contains("cannot be resolved without it"), "\(error)")
        }

        // The default is not a pin: it takes whatever the cache holds.
        let unpinned = try await StubHub.download(to: root)
        XCTAssertEqual(unpinned, root)
    }

    /// The Hub out of reach is an outage, not a miss: a directory holding a
    /// receipt and the whole set is used, with one line on stderr. Without a
    /// receipt nothing vouches for the files, and the error names the Hub.
    func testTheHubOutOfReachUsesTheReceipt() async throws {
        _ = try await StubHub.download(to: root)
        StubHub.offline = true
        StubHub.served = []
        let seen = Recorder()
        let (directory, stderr) = try await capturingStderr {
            try await StubHub.download(to: root) { seen.record($0) }
        }
        XCTAssertEqual(directory, root)
        XCTAssertEqual(StubHub.served, [])
        XCTAssertTrue(seen.all.isEmpty)
        // The one line, the way the other four ports assert it: it names the
        // outage, the directory and the commit the receipt vouches for.
        XCTAssertEqual(stderr.split(separator: "\n").count, 1, stderr)
        XCTAssertTrue(stderr.contains("cannot be reached"), stderr)
        XCTAssertTrue(stderr.contains(root.path), stderr)
        XCTAssertTrue(stderr.contains(StubHub.commit), stderr)
        // A directory verified as one repo does not answer for another.
        try LoudKit.writeReceipt(root, repo: "loudreader/other", revision: "main",
                                 commit: StubHub.commit)
        do {
            _ = try await StubHub.download(to: root)
            XCTFail("a receipt for another repo answered for this one with the Hub out of reach")
        } catch {
            XCTAssertTrue(String(describing: error).contains("cannot reach"),
                          String(describing: error))
        }
        try FileManager.default.removeItem(at: root.appendingPathComponent(LoudKit.receiptName))
        do {
            _ = try await StubHub.download(to: root)
            XCTFail("a directory with no receipt was used with the Hub out of reach")
        } catch {
            XCTAssertTrue(String(describing: error).contains("cannot reach"),
                          String(describing: error))
        }
        // Nor is a receipt no download wrote: one field forged is no receipt.
        try LoudKit.writeReceipt(root, repo: StubHub.repo, revision: "main", commit: StubHub.commit)
        let good = try XCTUnwrap(receiptOnDisk(root))
        var noCommit = good
        noCommit.removeValue(forKey: "commit")
        let forgeries: [(String, [String: Any])] = [
            ("no commit", noCommit),
            ("an empty commit", good.merging(["commit": ""]) { $1 }),
            ("the wrong digest", good.merging(["sha256sums": String(repeating: "0", count: 64)]) { $1 })
        ]
        for (what, forged) in forgeries {
            try JSONSerialization.data(withJSONObject: forged)
                .write(to: root.appendingPathComponent(LoudKit.receiptName))
            do {
                _ = try await StubHub.download(to: root)
                XCTFail("offline, a receipt with \(what) answered")
            } catch {
                XCTAssertTrue(String(describing: error).contains("cannot reach"), what)
            }
        }
    }

    /// The receipt is the shared fixture's: the fields in order, and the hit
    /// rule case for case.
    func testTheReceiptIsTheSharedFixture() async throws {
        let fixture = try Fixture.shared("release_receipt.json")
        XCTAssertEqual(LoudKit.receiptName, fixture["name"] as? String)
        let directory = try await StubHub.download(to: root)
        let text = try String(contentsOf: directory.appendingPathComponent(LoudKit.receiptName),
                              encoding: .utf8)
        let keys = text.split(separator: "\n").compactMap { line -> String? in
            guard line.hasPrefix(" \"") else { return nil }
            return String(line.dropFirst(2).prefix { $0 != "\"" })
        }
        XCTAssertEqual(keys, fixture["fields"] as? [String])
        XCTAssertTrue(text.contains(" \"repo\": \"\(StubHub.repo)\","), "slashes escaped: \(text)")
        let read = { LoudKit.readReceipt(directory, repo: StubHub.repo, cloning: false) }
        let receipt = try XCTUnwrap(read())
        XCTAssertEqual(receipt.repo, StubHub.repo)
        XCTAssertEqual(receipt.revision, "main")
        XCTAssertEqual(receipt.commit, StubHub.commit)
        XCTAssertEqual(receipt.sha256sums,
                       try LoudKit.fileSHA256(directory.appendingPathComponent("SHA256SUMS")))
        // The example's shape, `YYYY-MM-DDTHH:MM:SSZ`.
        XCTAssertEqual(receipt.fetchedAt.count,
                       ((fixture["example"] as? [String: Any])?["fetched_at"] as? String)?.count)
        // Over the example's manifest, the example's digest.
        try Data((fixture["sums"] as! String).utf8)
            .write(to: directory.appendingPathComponent("SHA256SUMS"))
        try LoudKit.writeReceipt(directory, repo: StubHub.repo, revision: "main", commit: StubHub.commit)
        XCTAssertEqual(read()?.sha256sums,
                       (fixture["example"] as? [String: Any])?["sha256sums"] as? String)
        // No manifest in the release: null, not a missing key.
        try FileManager.default.removeItem(at: directory.appendingPathComponent("SHA256SUMS"))
        try LoudKit.writeReceipt(directory, repo: StubHub.repo, revision: "main", commit: StubHub.commit)
        XCTAssertTrue(try XCTUnwrap(receiptOnDisk(directory))["sha256sums"] is NSNull)
        XCTAssertNil(try XCTUnwrap(read()).sha256sums)

        let cases = fixture["cases"] as! [[String: Any]]
        XCTAssertGreaterThanOrEqual(cases.count, 20, "the fixture holds probes")
        for (index, c) in cases.enumerated() {
            let name = c["name"] as! String
            let dir = root.appendingPathComponent("case-\(index)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            // Whatever JSON the case holds, an object or not; only null is no file.
            if let receipt = c["receipt"], !(receipt is NSNull) {
                try JSONSerialization.data(withJSONObject: receipt, options: .fragmentsAllowed)
                    .write(to: dir.appendingPathComponent(LoudKit.receiptName))
            }
            if let sums = c["sums"] as? String {
                try Data(sums.utf8).write(to: dir.appendingPathComponent("SHA256SUMS"))
            }
            for file in c["files"] as! [String] {
                try Data(file.utf8).write(to: dir.appendingPathComponent(file))
            }
            let repo = c["repo"] as! String
            XCTAssertEqual(LoudKit.readReceipt(dir, repo: repo, cloning: false) != nil,
                           c["offline"] as? String == "use", name)
            XCTAssertEqual(
                LoudKit.receiptHit(dir, repo: repo, commit: c["commit"] as! String, cloning: false),
                c["online"] as? String == "hit", name)
        }
    }

    func testProgressEndsAtTheWholePlan() async throws {
        let seen = Recorder()
        _ = try await StubHub.download(to: root) { seen.record($0) }
        let final = try XCTUnwrap(seen.all.last)
        XCTAssertEqual(final.fileIndex, final.fileCount)
        XCTAssertEqual(final.bytesWritten, final.bytesTotal)
    }

    /// A release short of a package is refused before a byte moves.
    func testAnIncompleteReleaseIsRefusedBeforeAnythingMoves() async throws {
        StubHub.files = StubHub.files.filter { !$0.key.contains("flow_estimator") }
        StubHub.files["SHA256SUMS"] = StubHub.sums(of: StubHub.files)
        do {
            _ = try await StubHub.download(to: root)
            XCTFail("a release with no flow_estimator was accepted")
        } catch {
            XCTAssertTrue(String(describing: error).contains("flow_estimator"),
                          String(describing: error))
        }
        XCTAssertEqual(StubHub.served, [])
    }

    /// Missing renderer packages are refused before weights move.
    func testAReleaseWithoutRendererPackagesIsRefused() async throws {
        StubHub.repo = "loudreader/loudr-1-turbo"
        StubHub.files = [
            "loudr-1-turbo.safetensors": Data(repeating: 1, count: 512),
            "manifest.json": Data("{}".utf8),
            "tokenizer.json": Data("{}".utf8),
            "release.json": Data("{\"profile\": \"turbo-0.1\", \"verified\": true}".utf8),
            "voices/joe.safetensors": Data(repeating: 4, count: 256),
        ]
        StubHub.files["SHA256SUMS"] = StubHub.sums(of: StubHub.files)
        do {
            _ = try await StubHub.download(to: root)
            XCTFail("a turbo release was fetched")
        } catch {
            let message = String(describing: error)
            XCTAssertTrue(message.contains("ships no coreml/flow_encoder.mlpackage"), message)
            XCTAssertTrue(message.contains("loudreader/loudr-1"), message)
        }
        XCTAssertEqual(StubHub.served, [])
    }

    /// A bundle loaded by repo id fetches the three enrollment packages into
    /// its own directory, with the manifest that vouches for them, and the
    /// second enrollment fetches nothing. A directory of your own is told how
    /// to fetch them and moves no bytes.
    func testEnrollOnARepoBundleFetchesTheCloningSetOnce() async throws {
        let directory = try await StubHub.download(to: root)
        var bundle = try ModelBundle(directory: directory)
        XCTAssertNil(bundle.origin, "a directory is not a repo")
        bundle.origin = StubHub.fetch()
        StubHub.served = []
        try await bundle.fetchCloning()
        let packages = Set(StubHub.served.filter { $0 != LoudKit.sumsName }.map {
            $0.split(separator: "/").prefix(2).joined(separator: "/")
        })
        XCTAssertEqual(packages, Set(LoudKit.coremlEnrollment))
        XCTAssertEqual(StubHub.served.first, LoudKit.sumsName)
        XCTAssertEqual(StubHub.served.count, 1 + 3 * 3, "\(StubHub.served)")
        XCTAssertNoThrow(try LoudKit.verifyInventory(at: directory, cloning: true))
        XCTAssertNotNil(LoudKit.readReceipt(directory, repo: StubHub.repo, cloning: true),
                        "the receipt was not widened to the cloning set")
        StubHub.served = []
        try await bundle.fetchCloning()
        XCTAssertEqual(StubHub.served, [], "the second enrollment fetched something")

        let plain = root.appendingPathComponent("plain")
        let other = try ModelBundle(directory: try await StubHub.download(to: plain))
        StubHub.served = []
        do {
            try await other.fetchCloning()
            XCTFail("a directory bundle fetched for itself")
        } catch {
            XCTAssertTrue(String(describing: error).contains("cloning: true"), String(describing: error))
        }
        XCTAssertEqual(StubHub.served, [])
    }

    func testBothModelNamesUseTheSameDownloadAndReceipt() async throws {
        for model in ["loudr-1", "loudr-1-turbo"] {
            StubHub.reset()
            StubHub.repo = "loudreader/" + model
            if model != "loudr-1" {
                StubHub.files[model + ".safetensors"] = StubHub.files.removeValue(forKey: "loudr-1.safetensors")
                StubHub.files["SHA256SUMS"] = StubHub.sums(of: StubHub.files)
            }
            let directory = root.appendingPathComponent(model)
            _ = try await StubHub.download(to: directory)
            XCTAssertEqual(try ModelBundle(directory: directory).checkpoint.lastPathComponent,
                           model + ".safetensors")
            let count = StubHub.served.count
            _ = try await StubHub.download(to: directory)
            XCTAssertEqual(StubHub.served.count, count)
            XCTAssertEqual(LoudKit.modelRepo(model), "loudreader/" + model)
        }
    }

    /// The cache layout is the fixture's: one path for a given root and repo,
    /// and the one variable that moves it.
    func testTheCachePathIsTheSharedFixture() throws {
        let fixture = try Fixture.shared("cache_path.json")
        XCTAssertEqual(fixture["env"] as? String, "LOUDKIT_CACHE")
        let cases = fixture["cases"] as! [[String: String]]
        XCTAssertGreaterThanOrEqual(cases.count, 4, "the fixture holds probes")
        for c in cases {
            XCTAssertEqual(LoudKit.cachePath(root: URL(fileURLWithPath: c["root"]!), repo: c["repo"]!).path,
                           c["path"]!, "\(c)")
        }
        for o in fixture["override"] as! [[String: String]] {
            XCTAssertEqual(LoudKit.cacheDirectory(repo: o["repo"]!, environment: ["LOUDKIT_CACHE": o["cache"]!]).path,
                           o["path"]!, "\(o)")
        }
        let plain = LoudKit.cacheDirectory(repo: "loudreader/loudr-1", environment: [:])
        XCTAssertEqual(plain.lastPathComponent, "loudreader--loudr-1")
        XCTAssertEqual(plain.deletingLastPathComponent().lastPathComponent, "loudkit")
    }

    /// A receipt for the synthesis set answers a plain load when the Hub is
    /// away, and an enrollment gets the sentence naming the three packages,
    /// not the transport error.
    func testOfflineWithoutTheCloningSetTheSentenceNamesThePackages() async throws {
        let directory = try await StubHub.download(to: root)
        StubHub.offline = true
        let again = try await StubHub.download(to: root)
        XCTAssertEqual(again, directory)
        do {
            _ = try await StubHub.download(to: root, cloning: true)
            XCTFail("offline without the packages, the fetch answered")
        } catch {
            let message = String(describing: error)
            XCTAssertTrue(message.contains(
                "holds no enrollment packages (coreml/s3_tokenizer.mlpackage, coreml/camp.mlpackage, "
                + "coreml/voice_encoder.mlpackage). Connect once to fetch them."), message)
        }
    }

    /// An empty file, garbage, and a file too large to be a receipt are no
    /// receipt, and the last is not read.
    func testAReceiptThatIsNotARecordIsNothing() throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let path = root.appendingPathComponent(LoudKit.receiptName)
        let record = "{\"repo\": \"someone/loudr-1\", \"revision\": \"main\", \"commit\": "
            + "\"\(StubHub.commit)\", \"sha256sums\": null, \"fetched_at\": \"2026-09-02T12:00:00Z\""
        try Data((record + "}\n").utf8).write(to: path)
        XCTAssertNotNil(LoudKit.readReceipt(root, repo: "someone/loudr-1", cloning: false),
                        "the record itself was refused")
        let bodies: [(String, Data)] = [
            ("an empty file", Data()),
            ("garbage", Data(repeating: 0xff, count: 1 << 16)),
            ("a record padded past the limit",
             Data((record + ", \"pad\": \"" + String(repeating: "x", count: 1 << 20) + "\"}").utf8))
        ]
        for (name, body) in bodies {
            try body.write(to: path)
            XCTAssertNil(LoudKit.readReceipt(root, repo: "someone/loudr-1", cloning: false), name)
        }
    }

    func testAMissingRepoSaysSo() async throws {
        StubHub.files = [:]
        do {
            _ = try await StubHub.download(to: root)
            XCTFail("an empty repo was accepted")
        } catch {
            XCTAssertTrue(String(describing: error).contains("none of the files"),
                          String(describing: error))
        }
    }

    /// The listing decides what filenames this process writes, so a name that
    /// climbs out of the download directory is refused before anything lands.
    func testARepoCannotNameAFileOutsideTheDirectory() async throws {
        StubHub.files["../escaped.safetensors"] = Data("no".utf8)
        do {
            _ = try await StubHub.download(to: root)
            XCTFail("a path with .. in it was fetched")
        } catch {
            XCTAssertTrue(String(describing: error).contains("not a path inside"),
                          String(describing: error))
        }
        XCTAssertEqual(StubHub.served, [])
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: root.deletingLastPathComponent()
                .appendingPathComponent("escaped.safetensors").path))
    }

    func testTheTraversalRule() {
        XCTAssertTrue(LoudKit.isSafePath("voices/joe.safetensors"))
        XCTAssertTrue(LoudKit.isSafePath("coreml/vocoder.mlpackage/Data/x"))
        XCTAssertFalse(LoudKit.isSafePath("../x"))
        XCTAssertFalse(LoudKit.isSafePath("a/../../x"))
        XCTAssertFalse(LoudKit.isSafePath("/etc/passwd"))
        XCTAssertFalse(LoudKit.isSafePath("a//b"))
        XCTAssertFalse(LoudKit.isSafePath(""))
    }

    // MARK: the checksums

    /// A miss hashes what is on disk: the truncated voice no longer matches
    /// the manifest, so it is fetched again, and the bytes the Hub now serves
    /// match it no better, so they are refused and removed. The stale receipt
    /// went before the fetch, so nothing vouches for what is left.
    func testATamperedFileIsRefusedAndRemoved() async throws {
        _ = try await StubHub.download(to: root)
        StubHub.rewrite("voices/joe.safetensors", resign: false)
        try Data("tr".utf8).write(to: root.appendingPathComponent("voices/joe.safetensors"))
        StubHub.commit = String(repeating: "b", count: 40)
        do {
            _ = try await StubHub.download(to: root)
            XCTFail("a file whose bytes do not match its manifest was accepted")
        } catch {
            let message = String(describing: error)
            XCTAssertTrue(message.contains("failed the release checksum"), message)
            XCTAssertTrue(message.contains("voices/joe.safetensors"), message)
        }
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("voices/joe.safetensors").path),
            "the bad file was left behind")
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: root.appendingPathComponent(LoudKit.receiptName).path),
            "a stale receipt outlived the files it vouched for")
    }

    /// Weights the manifest says nothing about are weights nothing vouches for.
    func testAnUnlistedFileIsRefused() async throws {
        StubHub.files["voices/zed.safetensors"] = Data(repeating: 9, count: 256)
        do {
            _ = try await StubHub.download(to: root)
            XCTFail("a voice outside SHA256SUMS was accepted")
        } catch {
            let message = String(describing: error)
            XCTAssertTrue(message.contains("does not list"), message)
            XCTAssertTrue(message.contains("voices/zed.safetensors"), message)
        }
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("voices/zed.safetensors").path))
    }

    /// Every loudreader release ships a manifest: one arriving without it is
    /// refused before a byte moves.
    func testAnOfficialReleaseWithoutAManifestIsRefusedBeforeAnythingMoves() async throws {
        StubHub.files["SHA256SUMS"] = nil
        do {
            _ = try await StubHub.download(to: root)
            XCTFail("an official release with no SHA256SUMS was accepted")
        } catch {
            XCTAssertTrue(String(describing: error).contains("no SHA256SUMS"),
                          String(describing: error))
        }
        XCTAssertEqual(StubHub.served, [])
    }

    /// Checksums say the bytes arrived intact and nothing about what they are:
    /// a development bundle carries a perfectly valid manifest. It is refused
    /// after the two bookkeeping files and before any weights.
    func testADevelopmentBundleIsRefusedBeforeItsWeightsMove() async throws {
        StubHub.files["release.json"] = Data("{\"profile\": \"lenient\", \"verified\": true}".utf8)
        StubHub.files["SHA256SUMS"] = StubHub.sums(of: StubHub.files)
        do {
            _ = try await StubHub.download(to: root)
            XCTFail("a lenient bundle in an official repo was accepted")
        } catch {
            XCTAssertTrue(String(describing: error).contains("development bundle"),
                          String(describing: error))
        }
        XCTAssertEqual(StubHub.served, ["SHA256SUMS", "release.json"])
    }

    func testAnUngatedBuildIsRefused() async throws {
        StubHub.files["release.json"] = Data("{\"profile\": \"full-0.1\", \"verified\": false}".utf8)
        StubHub.files["SHA256SUMS"] = StubHub.sums(of: StubHub.files)
        do {
            _ = try await StubHub.download(to: root)
            XCTFail("an ungated bundle was accepted")
        } catch {
            XCTAssertTrue(String(describing: error).contains("verified: true"),
                          String(describing: error))
        }
    }

    /// One gate for both models; what keeps turbo out of this package is the
    /// packages check, which this listing passes.
    func testTheTurboProfileIsAReleaseToo() async throws {
        StubHub.files["release.json"] = Data("{\"profile\": \"turbo-0.1\", \"verified\": true}".utf8)
        StubHub.files["SHA256SUMS"] = StubHub.sums(of: StubHub.files)
        let directory = try await StubHub.download(to: root)
        XCTAssertNoThrow(try ModelBundle(directory: directory))
    }

    /// A re-signed manifest passes: what the check refuses is bytes that do not
    /// match the manifest beside them, not bytes that changed.
    func testARewrittenAndResignedReleasePasses() async throws {
        StubHub.rewrite("voices/joe.safetensors", resign: true)
        let directory = try await StubHub.download(to: root)
        XCTAssertNoThrow(try ModelBundle(directory: directory))
    }

    /// A mangled manifest fails loudly rather than verifying nothing.
    func testTheManifestFormatIsTheOneTheBuilderWrites() throws {
        let sums = root.appendingPathComponent("SHA256SUMS")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let good = String(repeating: "a", count: 64)

        func parse(_ text: String) throws -> [String: String] {
            try Data(text.utf8).write(to: sums)
            return try LoudKit.parseSums(sums)
        }
        XCTAssertEqual(try parse("\(good)  voices/joe.safetensors\n"),
                       ["voices/joe.safetensors": good])
        // One space, not two.
        XCTAssertThrowsError(try parse("\(good) voices/joe.safetensors\n"))
        // Short and upper-case digests are not this format.
        XCTAssertThrowsError(try parse("abc  voices/joe.safetensors\n"))
        XCTAssertThrowsError(try parse("\(String(repeating: "A", count: 64))  joe\n"))
        // Names that address bytes the release does not contain.
        XCTAssertThrowsError(try parse("\(good)  ../joe.safetensors\n"))
        XCTAssertThrowsError(try parse("\(good)  /etc/passwd\n"))
        XCTAssertThrowsError(try parse("\(good)  c:/joe\n"))
        XCTAssertThrowsError(try parse("\(good)  voices//joe\n"))
        XCTAssertThrowsError(try parse("\(good)  a\\b\n"))
        // A manifest that disagrees with itself about one file.
        XCTAssertThrowsError(try parse("\(good)  joe\n\(good)  joe\n"))
        // Blank lines are furniture; a manifest of nothing but them is not.
        XCTAssertEqual(try parse("\n\(good)  joe\n\n").count, 1)
        XCTAssertThrowsError(try parse("\n\n"))
    }

    func testFileSHA256IsThePublishedDigest() throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let file = root.appendingPathComponent("abc")
        try Data("abc".utf8).write(to: file)
        // The published digest of "abc", not one computed here with the same
        // primitive.
        XCTAssertEqual(try LoudKit.fileSHA256(file),
                       "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }
}

/// What `body` wrote to stderr, beside its result. The Hub writes its offline
/// line through `FileHandle.standardError`, so file descriptor 2 is pointed at
/// a pipe for the call and restored after it; nothing else may write to stderr
/// meanwhile, which is why the capture is as narrow as one call.
func capturingStderr<T>(_ body: () async throws -> T) async throws -> (T, String) {
    let pipe = Pipe()
    let saved = dup(STDERR_FILENO)
    dup2(pipe.fileHandleForWriting.fileDescriptor, STDERR_FILENO)
    let restore = {
        dup2(saved, STDERR_FILENO)
        close(saved)
        try? pipe.fileHandleForWriting.close()
    }
    let result: T
    do {
        result = try await body()
    } catch {
        restore()
        throw error
    }
    restore()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    return (result, String(decoding: data, as: UTF8.self))
}

/// Somewhere for a `@Sendable` progress closure to put what it saw. The
/// callbacks arrive on the session's queue, so a captured `var` is a data race
/// the compiler is right about.
final class Recorder: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [LoudKit.DownloadProgress] = []

    func record(_ progress: LoudKit.DownloadProgress) {
        lock.lock()
        defer { lock.unlock() }
        events.append(progress)
    }

    var all: [LoudKit.DownloadProgress] {
        lock.lock()
        defer { lock.unlock() }
        return events
    }
}

// MARK: - the Hub that is not there

/// A repository served out of a dictionary, over `URLProtocol`.
enum StubHub {
    static let endpoint = "https://hub.invalid"
    nonisolated(unsafe) static var repo = "loudreader/loudr-1"
    /// What `main` resolves to; a test moves it to make the next fetch a miss.
    nonisolated(unsafe) static var commit = "3f2c1e0d9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e"
    /// Every route fails with a transport error, the way a Hub nobody can
    /// reach does.
    nonisolated(unsafe) static var offline = false

    nonisolated(unsafe) static var files: [String: Data] = defaultFiles
    /// Which files were served, in order, so idempotence and "nothing moved
    /// before the refusal" are measurable.
    nonisolated(unsafe) static var served: [String] = []
    /// How many tree pages were served.
    nonisolated(unsafe) static var listings = 0
    nonisolated(unsafe) static var pageSize = 1_000

    static func reset() {
        repo = "loudreader/loudr-1"
        commit = "3f2c1e0d9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e"
        offline = false
        files = defaultFiles
        served = []
        listings = 0
        pageSize = 1_000
    }

    /// The repository as the tree API describes it.
    static var repository: [LoudKit.RemoteFile] {
        files.keys.sorted().map {
            LoudKit.RemoteFile(path: $0, size: Int64(files[$0]?.count ?? 0))
        }
    }

    /// A fetch that goes through the stub rather than the network.
    static func fetch(cloning: Bool = false, revision: String = "main") -> LoudKit.Fetch {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubProtocol.self]
        return LoudKit.Fetch(repo: repo, revision: revision, cloning: cloning,
                             endpoint: endpoint, configuration: configuration)
    }

    static func download(
        to directory: URL,
        cloning: Bool = false,
        progress: (@Sendable (LoudKit.DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        try await LoudKit.download(fetch(cloning: cloning), to: directory, progress: progress)
    }

    /// A release with enough of every kind of file to exercise the plan, with
    /// a `SHA256SUMS` computed over the rest the way `tools/build_release.py`
    /// writes it.
    static let defaultFiles: [String: Data] = {
        var files: [String: Data] = [
            "loudr-1.safetensors": Data(repeating: 1, count: 4_096),
            "loudr-1-enrollment.safetensors": Data(repeating: 2, count: 2_048),
            "ve.safetensors": Data(repeating: 3, count: 512),
            "manifest.json": Data("{\"format\": \"loudkit-checkpoint\"}".utf8),
            "tokenizer.json": Data("{}".utf8),
            "release.json": Data("{\"profile\": \"full-0.1\", \"verified\": true}".utf8),
            "voices/joe.safetensors": Data(repeating: 4, count: 256),
            "voices/ada.safetensors": Data(repeating: 5, count: 256),
            "coreml/export.json": Data("{\"checkpoint\": \"loudr-1\"}".utf8),
            "onnx/t3_step.onnx": Data(repeating: 6, count: 1_024),
            "README.md": Data("# not part of the set\n".utf8),
        ]
        for package in LoudKit.coremlSynthesis + LoudKit.coremlEnrollment {
            files["\(package)/Manifest.json"] = Data("{}".utf8)
            files["\(package)/Data/com.apple.CoreML/model.mlmodel"] = Data(repeating: 7, count: 128)
            files["\(package)/Data/com.apple.CoreML/weights/weight.bin"] =
                Data(repeating: 8, count: 512)
        }
        files["SHA256SUMS"] = sums(of: files)
        return files
    }()

    /// The manifest a builder would write for these bytes: every file but the
    /// manifest itself, sorted, `<digest>  <name>`.
    static func sums(of files: [String: Data]) -> Data {
        var text = ""
        for name in files.keys.sorted() where name != "SHA256SUMS" {
            let digest = SHA256.hash(data: files[name] ?? Data())
                .map { String(format: "%02x", $0) }.joined()
            text += "\(digest)  \(name)\n"
        }
        return Data(text.utf8)
    }

    /// Change one file's bytes without changing its length, and re-sign the
    /// manifest or not.
    static func rewrite(_ name: String, resign: Bool) {
        let old = files[name] ?? Data()
        files[name] = Data(repeating: 0xEE, count: old.count)
        if resign { files["SHA256SUMS"] = sums(of: files) }
    }
}

/// Answers the revision, the tree API and `resolve/` out of `StubHub`.
final class StubProtocol: URLProtocol {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        guard let url = request.url else { return }
        if StubHub.offline {
            client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
            return
        }
        let (status, body, headers) = Self.answer(url)
        let response = HTTPURLResponse(url: url, statusCode: status,
                                       httpVersion: "HTTP/1.1", headerFields: headers)
        if let response {
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        }
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    private static func answer(_ url: URL) -> (Int, Data, [String: String]) {
        let path = url.path
        if path == "/api/models/\(StubHub.repo)/revision/main" {
            let body = (try? JSONSerialization.data(withJSONObject: ["sha": StubHub.commit])) ?? Data()
            return (200, body, ["Content-Type": "application/json"])
        }
        let treePrefix = "/api/models/\(StubHub.repo)/tree/main"
        if path == treePrefix || path == treePrefix + "/" {
            return listing(url)
        }
        let resolvePrefix = "/\(StubHub.repo)/resolve/main/"
        guard path.hasPrefix(resolvePrefix) else { return (404, Data(), [:]) }
        let name = String(path.dropFirst(resolvePrefix.count))
        guard let body = StubHub.files[name] else { return (404, Data(), [:]) }
        StubHub.served.append(name)
        return (200, body, ["Content-Length": String(body.count)])
    }

    /// One page of the tree, with a `Link` header when there is more.
    private static func listing(_ url: URL) -> (Int, Data, [String: String]) {
        StubHub.listings += 1
        let all = StubHub.repository
        let cursor = URLComponents(url: url, resolvingAgainstBaseURL: false)?
            .queryItems?.first { $0.name == "cursor" }?.value
        let start = Int(cursor ?? "0") ?? 0
        let end = min(start + StubHub.pageSize, all.count)
        let page = all[start..<end].map {
            ["type": "file", "path": $0.path, "size": $0.size] as [String: Any]
        }
        // A directory entry rides along: the reader must skip it rather than
        // plan a fetch for a name that serves no bytes.
        var entries: [[String: Any]] = page
        if start == 0 { entries.insert(["type": "directory", "path": "voices"], at: 0) }
        let body = (try? JSONSerialization.data(withJSONObject: entries)) ?? Data()
        var headers = ["Content-Type": "application/json"]
        if end < all.count {
            headers["Link"] =
                "<\(StubHub.endpoint)/api/models/\(StubHub.repo)/tree/main"
                + "?recursive=1&cursor=\(end)>; rel=\"next\""
        }
        return (200, body, headers)
    }
}
