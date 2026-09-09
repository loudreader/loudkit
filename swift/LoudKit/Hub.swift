import Foundation

/// Fetching a release, with nothing else installed.
///
/// The plan is `python/loudkit/release.py`'s `release_patterns("coreml")` set,
/// spelled the same way, so both languages fetch the same files from the same
/// repo. The transport is Foundation: the Hub's tree API says what a repo
/// holds, and `resolve/` serves the bytes.
public enum LoudKit {

    // MARK: the plan

    /// The synthesis artefact's name in a release.
    public static let checkpointName = "loudr-1.safetensors"
    /// The turbo release's synthesis artefact, under its own canonical name.
    public static let turboCheckpointName = "loudr-1-turbo.safetensors"
    /// Every canonical synthesis name, in the order a listing names them.
    static let checkpointNames = [checkpointName, turboCheckpointName]
    /// The enrollment artefact, and the utterance voice encoder. Both are
    /// torch's: this port enrols through the CoreML graphs and never opens
    /// them, so they stay out of the fetch even when cloning was asked for.
    static let enrollmentName = "loudr-1-enrollment.safetensors"
    static let voiceEncoderName = "ve.safetensors"

    /// Everything a release carries whatever the backend, from
    /// `release._RELEASE_CORE`.
    ///
    /// `*.safetensors` is the checkpoint and, nested, every voice: these are
    /// `fnmatch` patterns, where `*` crosses `/`, which is the rule the Python
    /// plan was written against.
    static let releaseCore = [
        "*.safetensors",
        "manifest.json",
        "tokenizer.json",
        "release.json",
        "voices/*",
        "SHA256SUMS"
    ]

    /// The three renderer packages this port runs.
    static let coremlSynthesis = [
        "coreml/flow_encoder.mlpackage",
        "coreml/flow_estimator.mlpackage",
        "coreml/vocoder.mlpackage"
    ]

    /// The three enrollment packages `Enrollment.Enroller` runs.
    static let coremlEnrollment = [
        "coreml/s3_tokenizer.mlpackage",
        "coreml/camp.mlpackage",
        "coreml/voice_encoder.mlpackage"
    ]

    /// The record naming the checkpoint the packages were exported from.
    /// Fetched with them: a record left on the Hub is a check nobody runs.
    static let coremlRecord = "coreml/export.json"

    /// `(allow, ignore)` for a CoreML fetch, mirroring
    /// `release.release_patterns("coreml")`.
    ///
    /// A `.mlpackage` is a directory tree, so each package is fetched as
    /// `<package>/*` and lands as the several files it is.
    static func releasePatterns(cloning: Bool) -> (allow: [String], ignore: [String]) {
        var allow = releaseCore
        allow += coremlSynthesis.map { "\($0)/*" }
        allow.append(coremlRecord)
        if cloning { allow += coremlEnrollment.map { "\($0)/*" } }
        return (allow, [voiceEncoderName, enrollmentName])
    }

    /// One file in a repo, as the tree API describes it.
    struct RemoteFile: Equatable {
        let path: String
        /// The file's size, not its LFS pointer's. `-1` when the API did not say.
        let size: Int64
    }

    /// The files a fetch will move, in the order it will move them.
    ///
    /// Ignore wins over allow, the same precedence `huggingface_hub` applies,
    /// because the two enrollment names are carved back out of a `*.safetensors`
    /// that would otherwise match them.
    static func plan(_ files: [RemoteFile], cloning: Bool) -> [RemoteFile] {
        let (allow, ignore) = releasePatterns(cloning: cloning)
        return files.filter { file in
            !ignore.contains { matches(file.path, $0) }
                && allow.contains { matches(file.path, $0) }
        }
    }

    /// `fnmatch` for the two wildcards the plan uses, with `*` crossing `/`.
    ///
    /// That is Python's rule rather than a shell's, and the plan depends on it:
    /// `*.safetensors` is written once and means the checkpoint at the root and
    /// all twenty voices under `voices/`.
    static func matches(_ path: String, _ pattern: String) -> Bool {
        let p = Array(path), q = Array(pattern)
        var i = 0, j = 0, star = -1, mark = 0
        while i < p.count {
            if j < q.count, q[j] == "*" {
                star = j
                mark = i
                j += 1
            } else if j < q.count, q[j] == "?" || q[j] == p[i] {
                i += 1
                j += 1
            } else if star >= 0 {
                j = star + 1
                mark += 1
                i = mark
            } else {
                return false
            }
        }
        while j < q.count, q[j] == "*" { j += 1 }
        return j == q.count
    }

    // MARK: the fetch

    /// How far along a `download` is. Reported per file and for the plan.
    public struct DownloadProgress: Sendable {
        /// Repo-relative path of the file being fetched.
        public let file: String
        /// 1-based position of `file` in the plan, and the plan's length.
        public let fileIndex: Int
        /// Files in the plan.
        public let fileCount: Int
        /// Bytes written for `file`, and what it will be when finished.
        public let fileBytesWritten: Int64
        /// Size of `file`, from the listing.
        public let fileBytesTotal: Int64
        /// Bytes written across the whole plan, and the plan's size. A file
        /// already on disk counts as written, so a resumed fetch starts part way
        /// along rather than at zero.
        public let bytesWritten: Int64
        /// Size of the whole plan.
        public let bytesTotal: Int64

        /// `bytesWritten / bytesTotal`, or zero when the plan's size is not
        /// known yet.
        public var fraction: Double {
            bytesTotal > 0 ? Double(bytesWritten) / Double(bytesTotal) : 0
        }
    }

    /// Where an unnamed fetch lands: `cachePath` under `~/Library/Caches`, or
    /// `$LOUDKIT_CACHE/<org>--<name>` when that variable is set. The same
    /// layout in Go, Rust and JS, pinned by
    /// `tests/data/conformance/cache_path.json`, so a cache one port wrote is
    /// a cache the other three read.
    public static func cacheDirectory(repo: String) -> URL {
        cacheDirectory(repo: repo, environment: ProcessInfo.processInfo.environment)
    }

    static func cacheDirectory(repo: String, environment: [String: String]) -> URL {
        if let override = environment["LOUDKIT_CACHE"], !override.isEmpty {
            return URL(fileURLWithPath: override).appendingPathComponent(cacheSlug(repo))
        }
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return cachePath(root: caches, repo: repo)
    }

    /// `<root>/loudkit/<org>--<name>`.
    public static func cachePath(root: URL, repo: String) -> URL {
        root.appendingPathComponent("loudkit").appendingPathComponent(cacheSlug(repo))
    }

    static func cacheSlug(_ repo: String) -> String {
        modelRepo(repo).replacingOccurrences(of: "/", with: "--")
    }

    static func modelRepo(_ name: String) -> String {
        ["loudr-1", "loudr-1-turbo"].contains(name) ? "loudreader/" + name : name
    }

    /// Fetch a release's CoreML set into `directory`, and answer with it.
    ///
    /// The directory carries a receipt naming the commit it holds; a call
    /// whose revision still resolves to that commit, over a directory whose
    /// receipt `readReceipt` accepts and which still holds the set,
    /// fetches and hashes no weight. Any other call re-fetches `SHA256SUMS`,
    /// keeps every file that hashes to what it lists and fetches the rest,
    /// verifying each as it arrives; the two bookkeeping files come first,
    /// so a repo that is not a release is refused before its weights move. A
    /// Hub that cannot be reached is covered by the receipt, with one line
    /// on stderr saying so.
    ///
    /// - Parameters:
    ///   - repo: a Hub repo id, `org/name`.
    ///   - directory: where the release lands. Created if it is not there.
    ///   - revision: branch, tag or commit. Pin it for a build that must not move.
    ///   - cloning: also fetch the three enrollment packages `Enrollment` needs.
    ///   - progress: called on the session's queue, many times per file. Nil
    ///     prints one line per file to stderr.
    @discardableResult
    public static func download(
        repo: String,
        to directory: URL,
        revision: String = "main",
        cloning: Bool = false,
        progress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        try await download(Fetch(repo: repo, revision: revision, cloning: cloning),
                           to: directory, progress: progress ?? stderrProgress)
    }

    /// One line per file on stderr, the default for a fetch nobody is drawing.
    static let stderrProgress: @Sendable (DownloadProgress) -> Void = { p in
        guard p.fileBytesWritten == p.fileBytesTotal else { return }
        FileHandle.standardError.write(Data(
            "loudkit: [\(p.fileIndex)/\(p.fileCount)] \(p.file)\n".utf8))
    }

    /// What to fetch, and from where.
    ///
    /// The last two are the seams the tests drive the fetch through: a
    /// `URLProtocol` stub and a hostname that does not resolve, so the plan, the
    /// pagination and the idempotence are decidable without a network.
    struct Fetch {
        var repo: String
        var revision: String = "main"
        var cloning: Bool = false
        var endpoint: String = defaultEndpoint
        var configuration: URLSessionConfiguration = .ephemeral
    }

    /// The default Hub. `HF_ENDPOINT` moves it, the same variable every other
    /// client reads, for a mirror or an enterprise Hub.
    ///
    /// A `String` rather than a `URL` because every use joins escaped path
    /// components onto it, and one conversion at the end says where a bad value
    /// came from instead of forcing an optional open at the top.
    static var defaultEndpoint: String {
        let raw = ProcessInfo.processInfo.environment["HF_ENDPOINT"] ?? ""
        return raw.isEmpty ? "https://huggingface.co" : raw
    }

    /// The whole fetch.
    static func download(
        _ fetch: Fetch,
        to directory: URL,
        progress: (@Sendable (DownloadProgress) -> Void)?
    ) async throws -> URL {
        var fetch = fetch
        fetch.repo = modelRepo(fetch.repo)
        let repo = fetch.repo
        guard isRepoID(repo) else {
            throw LoudKitError.asset(
                "\(repo): not a Hugging Face repo id (those look like 'org/name')")
        }
        let fetcher = Fetcher(configuration: fetch.configuration)
        defer { fetcher.invalidate() }

        let commit: String
        let complete = { (try? verifyInventory(at: directory, cloning: fetch.cloning)) != nil }
        do {
            commit = try await resolveCommit(repo: repo, revision: fetch.revision,
                                             endpoint: fetch.endpoint, fetcher: fetcher)
        } catch let error as URLError {
            if let receipt = readReceipt(directory, repo: repo, cloning: fetch.cloning), complete() {
                FileHandle.standardError.write(Data((
                    "loudkit: the hub cannot be reached; using \(directory.path), which holds "
                    + "\(receipt.repo) at \(receipt.commit) (fetched \(receipt.fetchedAt))\n").utf8))
                return directory
            }
            // A receipt for the synthesis set does not cover an enrollment:
            // the sentence names the packages, not the transport.
            if fetch.cloning, readReceipt(directory, repo: repo, cloning: false) != nil,
               (try? verifyInventory(at: directory, cloning: false)) != nil {
                throw LoudKitError.asset(
                    "\(repo): the hub cannot be reached, and \(directory.path) holds no enrollment "
                    + "packages (\(coremlEnrollment.joined(separator: ", "))). Connect once to fetch them.")
            }
            throw LoudKitError.asset("cannot reach \(fetch.endpoint): \(error.localizedDescription)")
        }
        if receiptHit(directory, repo: repo, commit: commit, cloning: fetch.cloning), complete() {
            return directory
        }
        // A stale receipt must not outlive the files it vouched for.
        try? FileManager.default.removeItem(at: directory.appendingPathComponent(receiptName))

        let files = try await listFiles(repo: repo, revision: fetch.revision,
                                        endpoint: fetch.endpoint, fetcher: fetcher)
        let wanted = bookkeepingFirst(plan(files, cloning: fetch.cloning))
        guard !wanted.isEmpty else {
            throw LoudKitError.asset(
                "\(repo) at \(fetch.revision): holds none of the files a loudkit release "
                + "carries. Check the repo id and the revision.")
        }
        try refuseBeforeFetching(repo: repo, revision: fetch.revision, wanted: wanted.map(\.path))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let planBytes = wanted.reduce(Int64(0)) { $0 + max($1.size, 0) }
        var done: Int64 = 0
        var sums: [String: String]?
        for (index, file) in wanted.enumerated() {
            let destination = directory.appendingPathComponent(file.path)
            let already = done
            func report(_ written: Int64, _ total: Int64) {
                progress?(DownloadProgress(
                    file: file.path, fileIndex: index + 1, fileCount: wanted.count,
                    fileBytesWritten: written, fileBytesTotal: total,
                    bytesWritten: already + max(written, 0), bytesTotal: planBytes))
            }
            var fetched = false
            if matchesSums(destination, name: file.path, sums: sums) {
                done += max(file.size, 0)
                report(file.size, file.size)
            } else {
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.createDirectory(
                    at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                let url = try resolveURL(endpoint: fetch.endpoint, repo: repo,
                                         revision: fetch.revision, path: file.path)
                try await reachable(fetch.endpoint) {
                    try await fetcher.download(request(url), to: destination) { written, total in
                        report(written, total > 0 ? total : file.size)
                    }
                }
                done += max(file.size, 0)
                report(file.size, file.size)
                fetched = true
            }
            if file.path == sumsName {
                sums = try parseSums(destination)
                continue
            }
            if file.path == releaseRecord, isOfficial(repo) {
                try requireReleasable(directory, repo: repo, sums: sums ?? [:])
            }
            // A file kept was hashed against the manifest a moment ago.
            if fetched {
                try verifyFetched(directory, repo: repo, name: file.path, sums: sums)
            }
        }
        // After the hashes: a set can be intact and still short, because the
        // plan fetches whatever subset the repo holds and says nothing about the
        // rest.
        try verifyInventory(at: directory, cloning: fetch.cloning)
        try writeReceipt(directory, repo: repo, revision: fetch.revision, commit: commit)
        return directory
    }

    /// What the listing alone can say about a repo, before a byte moves: an
    /// official repo must ship the two bookkeeping files, and a repo without
    /// the CoreML packages cannot run here whatever else it carries.
    static func refuseBeforeFetching(repo: String, revision: String, wanted: [String]) throws {
        if isOfficial(repo) {
            for name in [sumsName, releaseRecord] where !wanted.contains(name) {
                throw LoudKitError.asset(
                    "\(repo) at \(revision): no \(name). Every \(officialOrg) release ships one, "
                    + "so this cannot be checked against anything and will not be fetched. Pin a "
                    + "revision you trust.")
            }
        }
        for package in coremlSynthesis where !wanted.contains(where: { $0.hasPrefix(package + "/") }) {
            throw LoudKitError.asset(
                "\(repo) at \(revision) ships no \(package), which this package runs on. Use "
                + "loudreader/loudr-1, or pin a revision that carries the packages.")
        }
    }

    /// The two bookkeeping files ahead of everything else, the rest in order.
    static func bookkeepingFirst(_ files: [RemoteFile]) -> [RemoteFile] {
        func rank(_ path: String) -> Int {
            path == sumsName ? 0 : path == releaseRecord ? 1 : 2
        }
        return files.enumerated().sorted {
            (rank($0.element.path), $0.offset) < (rank($1.element.path), $1.offset)
        }.map(\.element)
    }

    /// Whether a repo-relative path may be joined onto a local directory.
    ///
    /// The path comes from the server, and the fetch turns it into a file to
    /// write. `../../.ssh/authorized_keys` is a valid string and a valid Git
    /// path is not the same thing as a valid destination, so the join is
    /// refused rather than made safe afterwards.
    static func isSafePath(_ path: String) -> Bool {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\"),
              !path.contains("\0") else { return false }
        let segments = path.split(separator: "/", omittingEmptySubsequences: false)
        return segments.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }

    /// The rule from `hub.py`, pinned to the shared fixture: anything that
    /// exists on disk is a path, and so is anything path-shaped; `org/name`
    /// is a repo id.
    static func isRepoID(_ ref: String) -> Bool {
        if ref.isEmpty || FileManager.default.fileExists(atPath: ref) { return false }
        if ref.hasPrefix(".") || ref.hasPrefix("/") || ref.hasPrefix("~")
            || ref.hasSuffix(".safetensors") { return false }
        let parts = ref.split(separator: "/", omittingEmptySubsequences: false)
        let allowed = { (c: Character) in c.isLetter || c.isNumber || c == "_" || c == "-" || c == "." }
        return parts.count == 2 && parts.allSatisfy { !$0.isEmpty && $0.allSatisfy(allowed) }
    }

    /// Whether the file already at `url` hashes to what `sums` lists for
    /// `name`. The manifest itself never does: it cannot vouch for itself.
    static func matchesSums(_ url: URL, name: String, sums: [String: String]?) -> Bool {
        guard name != sumsName, let expected = sums?[name],
              (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true
        else { return false }
        return (try? fileSHA256(url)) == expected
    }

    static func resolveURL(endpoint: String, repo: String, revision: String,
                           path: String) throws -> URL {
        try url(join(endpoint, repo, "resolve", revision, path))
    }

    /// Join path components onto a base, escaping each one.
    ///
    /// A component may itself hold separators (`coreml/vocoder.mlpackage/…`),
    /// which stay separators: the escape is per segment, so a file name with a
    /// space in it survives and the tree does not collapse.
    static func join(_ base: String, _ components: String...) -> String {
        var out = base.hasSuffix("/") ? String(base.dropLast()) : base
        for component in components {
            for segment in component.split(separator: "/") {
                let escaped = segment.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)
                out += "/" + (escaped ?? String(segment))
            }
        }
        return out
    }

    static func url(_ string: String) throws -> URL {
        guard let url = URL(string: string) else {
            throw LoudKitError.asset("\(string): not a URL")
        }
        return url
    }

    /// A request carrying the token, when there is one.
    ///
    /// Read from the environment rather than taken as a parameter: a gated repo
    /// is the only reason this port needs one, and a token that travels through
    /// an API is a token that ends up in a log.
    static func request(_ url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        let environment = ProcessInfo.processInfo.environment
        if let token = environment["HF_TOKEN"] ?? environment["HUGGING_FACE_HUB_TOKEN"],
           !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    /// The commit `revision` names on the Hub today.
    ///
    /// A `URLError` passes through untouched: the caller decides whether the
    /// receipt covers it. An HTTP status is an error, not an outage.
    static func resolveCommit(repo: String, revision: String, endpoint: String,
                              fetcher: Fetcher) async throws -> String {
        let url = try url(join(endpoint, "api", "models", repo, "revision", revision))
        let (data, response) = try await fetcher.data(request(url))
        try check(response, what: "\(repo) at \(revision)")
        guard let info = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let sha = info["sha"] as? String, !sha.isEmpty else {
            throw LoudKitError.asset("\(repo) at \(revision): the Hub did not say which commit that is")
        }
        return sha
    }

    /// Everything the repo holds, following the tree API's pagination.
    ///
    /// The cursor is followed rather than ignored because a truncated listing is
    /// a fetch that silently leaves files behind, and the failure would appear
    /// much later as a bundle that does not load.
    static func listFiles(repo: String, revision: String, endpoint: String,
                          fetcher: Fetcher) async throws -> [RemoteFile] {
        var next: URL? = try url(
            join(endpoint, "api", "models", repo, "tree", revision) + "?recursive=1")
        var out: [RemoteFile] = []
        var pages = 0
        while let url = next {
            pages += 1
            // A release is tens of files and the tree API pages by the
            // thousand, so a listing past this is a `Link: rel="next"` cycle
            // rather than a large repository, and following it never returns.
            guard pages <= maxListingPages else {
                throw LoudKitError.asset("\(repo): the file listing does not end")
            }
            let (data, response) = try await reachable(endpoint) {
                try await fetcher.data(request(url))
            }
            try check(response, what: "the file listing for \(repo)")
            guard let entries = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                throw LoudKitError.asset("\(repo): the file listing is not a JSON array")
            }
            for entry in entries where entry["type"] as? String == "file" {
                guard let path = entry["path"] as? String else { continue }
                guard isSafePath(path) else {
                    throw LoudKitError.asset(
                        "\(repo): the listing names a file at \(path), which is not a path "
                        + "inside the download directory. Nothing was written.")
                }
                let lfs = (entry["lfs"] as? [String: Any])?["size"] as? NSNumber
                let size = lfs ?? entry["size"] as? NSNumber
                out.append(RemoteFile(path: path, size: size?.int64Value ?? -1))
            }
            next = nextLink(response)
        }
        return out
    }

    /// `Link: <…>; rel="next"`, which is how the tree API pages.
    static func nextLink(_ response: URLResponse?) -> URL? {
        guard let http = response as? HTTPURLResponse,
              let link = http.value(forHTTPHeaderField: "Link") else { return nil }
        for part in link.components(separatedBy: ",") {
            let fields = part.components(separatedBy: ";")
            guard fields.count >= 2,
                  fields.dropFirst().contains(where: { $0.contains("rel=\"next\"") }) else { continue }
            let target = fields[0].trimmingCharacters(in: .whitespaces)
                .trimmingCharacters(in: CharacterSet(charactersIn: "<>"))
            if let url = URL(string: target) { return url }
        }
        return nil
    }

    /// Turn a transport failure into one sentence.
    ///
    /// A `URLError` prints as three screens of `NSUnderlyingError` dictionaries,
    /// and this one arrives at a person who is offline or behind a proxy on the
    /// first line of their first program.
    static func reachable<T>(_ endpoint: String, _ work: () async throws -> T) async throws -> T {
        do {
            return try await work()
        } catch let error as URLError {
            throw LoudKitError.asset("cannot reach \(endpoint): \(error.localizedDescription)")
        }
    }

    /// Turn an HTTP status into the sentence a caller can act on.
    static func check(_ response: URLResponse?, what: String) throws {
        guard let http = response as? HTTPURLResponse else { return }
        switch http.statusCode {
        case 200..<300:
            return
        case 401, 403:
            throw LoudKitError.asset(
                "\(what): HTTP \(http.statusCode). The repo is private or gated. "
                + "set HF_TOKEN to a token that can read it.")
        case 404:
            throw LoudKitError.asset(
                "\(what): HTTP 404. No such repo, revision or file.")
        default:
            throw LoudKitError.asset("\(what): HTTP \(http.statusCode).")
        }
    }

    // MARK: the receipt

    /// What a verified download directory carries: `repo` and `revision` as
    /// asked, `commit` as the Hub resolved it, `sha256sums` (null when the
    /// release ships none) and `fetched_at` in UTC. The fields, the predicate
    /// and the hit rule are the shared fixture `release_receipt.json`'s.
    static let receiptName = ".loudkit-release.json"
    /// The most a receipt file is read: five short fields. A file past it is
    /// not a receipt, and is not read.
    static let receiptLimit = 1 << 20
    /// The most pages of a file listing that are followed.
    static let maxListingPages = 100

    /// A receipt `readReceipt` accepted.
    struct Receipt: Equatable {
        let repo: String
        /// As asked; `main` when the caller named none.
        let revision: String
        let commit: String
        /// The digest of `SHA256SUMS`, or nil for a release without one.
        let sha256sums: String?
        /// UTC, to the second.
        let fetchedAt: String
    }

    /// The receipt under `root` when it vouches for `repo`, else nil: every
    /// field present with its type (null is the wrong type for all but
    /// `sha256sums`; keys it does not know are ignored), `repo` the one
    /// asked, `commit` forty lowercase hex, `sha256sums` the digest of the
    /// `SHA256SUMS` on disk (nil for none), and every file it lists that the
    /// plan selects present. No weight is hashed: the receipt is checked, not
    /// the release. This is the only place the file is read.
    static func readReceipt(_ root: URL, repo: String, cloning: Bool) -> Receipt? {
        let path = root.appendingPathComponent(receiptName)
        guard let size = (try? path.resourceValues(forKeys: [.fileSizeKey]))?.fileSize, size <= receiptLimit,
              let data = try? Data(contentsOf: path),
              let record = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let named = record["repo"] as? String,
              let revision = record["revision"] as? String,
              let commit = record["commit"] as? String,
              let fetchedAt = record["fetched_at"] as? String,
              let digest = record["sha256sums"],
              digest is String || digest is NSNull
        else { return nil }
        let receipt = Receipt(repo: named, revision: revision, commit: commit,
                              sha256sums: digest as? String, fetchedAt: fetchedAt)
        guard receipt.repo == repo, commit.count == 40,
              commit.allSatisfy({ $0.isASCII && ($0.isNumber || ("a"..."f").contains($0)) })
        else { return nil }
        guard isFile(root, sumsName) else { return receipt.sha256sums == nil ? receipt : nil }
        let sums = root.appendingPathComponent(sumsName)
        guard let onDisk = try? fileSHA256(sums), receipt.sha256sums == onDisk,
              let listed = try? parseSums(sums)
        else { return nil }
        let files = listed.keys.map { RemoteFile(path: $0, size: -1) }
        return plan(files, cloning: cloning).allSatisfy { isFile(root, $0.path) } ? receipt : nil
    }

    /// Record that `root` holds `repo` at `commit`, verified.
    ///
    /// Written field by field because `JSONSerialization` orders keys as it
    /// likes, and the fixture fixes the order; slashes unescaped, so the
    /// bytes are the other four writers' bytes.
    static func writeReceipt(_ root: URL, repo: String, revision: String, commit: String) throws {
        var digest: Any = NSNull()
        if isFile(root, sumsName) { digest = try fileSHA256(root.appendingPathComponent(sumsName)) }
        let fields: [(String, Any)] = [
            ("repo", repo),
            ("revision", revision),
            ("commit", commit),
            ("sha256sums", digest),
            ("fetched_at", ISO8601DateFormatter().string(from: Date()))
        ]
        var lines: [String] = []
        for (key, value) in fields {
            let json = try JSONSerialization.data(
                withJSONObject: value, options: [.fragmentsAllowed, .withoutEscapingSlashes])
            guard let text = String(bytes: json, encoding: .utf8) else {
                throw LoudKitError.asset("\(receiptName): \(key) cannot be written as JSON")
            }
            lines.append(" \"\(key)\": \(text)")
        }
        try Data(("{\n" + lines.joined(separator: ",\n") + "\n}\n").utf8)
            .write(to: root.appendingPathComponent(receiptName))
    }

    /// Whether `root` already holds `repo` at `commit`: a receipt
    /// `readReceipt` accepts, naming that commit. An empty commit never hits:
    /// nothing was resolved.
    static func receiptHit(_ root: URL, repo: String, commit: String, cloning: Bool) -> Bool {
        !commit.isEmpty && readReceipt(root, repo: repo, cloning: cloning)?.commit == commit
    }

    // MARK: the inventory

    /// The inventory the plan promised must be on disk, mirroring
    /// `release.verify_release_inventory`.
    ///
    /// A pattern fetch is a request, not a receipt: the Hub serves whatever
    /// subset of the patterns the repo actually holds and says nothing about the
    /// rest, so a release missing its packages comes back looking like one that
    /// has them. The shortfall is named here, at the fetch, rather than found
    /// later inside CoreML.
    public static func verifyInventory(at root: URL, cloning: Bool = false) throws {
        var missing: [String] = []
        let checkpoint = try? ModelBundle.findCheckpoint(in: root)
        if checkpoint == nil { missing.append(checkpointName) }
        for name in ["manifest.json", "tokenizer.json"] where !isFile(root, name) {
            missing.append(name)
        }
        for package in coremlSynthesis + (cloning ? coremlEnrollment : [])
        where !isPackage(root.appendingPathComponent(package)) {
            missing.append(package)
        }
        let voices = (try? FileManager.default.contentsOfDirectory(
            atPath: root.appendingPathComponent(ModelBundle.voiceDirectory).path)) ?? []
        if !voices.contains(where: { $0.hasSuffix(".safetensors") }) {
            missing.append("\(ModelBundle.voiceDirectory)/*.safetensors")
        }
        guard missing.isEmpty else {
            throw LoudKitError.asset(
                "\(root.path): this fetch does not add up to a usable CoreML set, missing: "
                + missing.sorted().joined(separator: ", ") + ". The release does not carry "
                + "these files, or the fetch was interrupted; retry, or pin a revision that "
                + "ships them.")
        }
    }

    static func isFile(_ root: URL, _ name: String) -> Bool {
        var isDirectory: ObjCBool = false
        let there = FileManager.default.fileExists(
            atPath: root.appendingPathComponent(name).path, isDirectory: &isDirectory)
        return there && !isDirectory.boolValue
    }

    /// Whether `node` has the structure of a CoreML package: a `Manifest.json`
    /// beside a non-empty `Data/`. "A directory with something in it" passes for
    /// a package whose weights never arrived, which is the one shortfall a
    /// pattern fetch actually produces.
    ///
    /// A precompiled `.mlmodelc` beside it answers instead, because that is the
    /// other form `Engine.load` accepts.
    static func isPackage(_ node: URL) -> Bool {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        if node.pathExtension == "mlpackage" {
            let compiled = node.deletingPathExtension().appendingPathExtension("mlmodelc")
            if fm.fileExists(atPath: compiled.path, isDirectory: &isDirectory), isDirectory.boolValue {
                return true
            }
        }
        guard fm.fileExists(atPath: node.path, isDirectory: &isDirectory), isDirectory.boolValue,
              fm.fileExists(atPath: node.appendingPathComponent("Manifest.json").path) else {
            return false
        }
        let data = node.appendingPathComponent("Data")
        guard fm.fileExists(atPath: data.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return false
        }
        return !((try? fm.contentsOfDirectory(atPath: data.path))?.isEmpty ?? true)
    }
}

/// One URLSession, driven from `async`, that streams to disk and says how far
/// it has got.
///
/// A download delegate rather than `URLSession.download(from:)`: the async form
/// hands back a finished file and no way to watch a 747 MB one arrive, and a
/// first mile with a silent three-minute pause in it reads as a hang.
final class Fetcher {
    private let watcher = Watcher()
    private let session: URLSession

    init(configuration: URLSessionConfiguration) {
        let queue = OperationQueue()
        // Serial: the watcher holds one download's state at a time, and the
        // delegate callbacks are the only writers to it.
        queue.maxConcurrentOperationCount = 1
        session = URLSession(configuration: configuration, delegate: watcher, delegateQueue: queue)
    }

    func invalidate() { session.finishTasksAndInvalidate() }

    func data(_ request: URLRequest) async throws -> (Data, URLResponse) {
        try await session.data(for: request)
    }

    /// Fetch one file to `destination`, replacing whatever is there.
    func download(_ request: URLRequest, to destination: URL,
                  onProgress: @escaping (Int64, Int64) -> Void) async throws {
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
            watcher.expect(destination, onProgress: onProgress, continuation: c)
            session.downloadTask(with: request).resume()
        }
    }
}

/// The delegate half, its own class so the session can be a `let`.
private final class Watcher: NSObject, URLSessionDownloadDelegate {
    private var continuation: CheckedContinuation<Void, Error>?
    private var destination: URL?
    private var failure: Error?
    private var onProgress: ((Int64, Int64) -> Void)?

    func expect(_ destination: URL, onProgress: @escaping (Int64, Int64) -> Void,
                continuation: CheckedContinuation<Void, Error>) {
        self.destination = destination
        self.onProgress = onProgress
        self.continuation = continuation
        failure = nil
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        onProgress?(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // The temp file is gone when this returns, so the move happens here and
        // any failure is carried to `didCompleteWithError`, which always runs.
        do {
            try LoudKit.check(downloadTask.response,
                              what: downloadTask.originalRequest?.url?.path ?? "the file")
            guard let destination else { return }
            let fm = FileManager.default
            if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
            try fm.moveItem(at: location, to: destination)
        } catch {
            failure = error
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        let outcome = failure ?? error
        let pending = continuation
        continuation = nil
        onProgress = nil
        destination = nil
        if let outcome {
            pending?.resume(throwing: outcome)
        } else {
            pending?.resume()
        }
    }
}
