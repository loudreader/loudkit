import Foundation

/// A downloaded release, as one directory.
///
/// `Engine.load(checkpoint:coremlAssets:)` asks the caller for the pieces, and
/// so did every example: a checkpoint path, a voice path, a `coreml/` path, and
/// a `tokenizer.json` nobody names until it is missing. That is four chances to
/// get a path wrong before hearing anything. A release has a fixed layout, so
/// this reads it, and the caller passes the directory the fetch wrote.
///
/// The rules are `python/loudkit/release.py`'s `_only_checkpoint_in`, so both
/// languages accept the same directories: the canonical checkpoint name first,
/// then the file that declares the synthesis role, then the sole candidate.
public struct ModelBundle {
    /// Where a release keeps its voice profiles, relative to `directory`.
    public static let voiceDirectory = "voices"
    /// Where a release keeps its exported CoreML packages.
    public static let coremlDirectory = "coreml"

    /// The release directory itself.
    public let directory: URL
    /// The synthesis checkpoint.
    public let checkpoint: URL
    /// The text tokenizer, beside the checkpoint.
    public let tokenizer: URL
    /// Where the `.mlpackage` (or precompiled `.mlmodelc`) stages live.
    public let coreml: URL
    /// The fetch this release came from, so `fetchCloning` can fetch the rest
    /// of it into `directory`. Nil for a directory of your own.
    var origin: LoudKit.Fetch?

    /// Read a release directory, or say what it is short of.
    ///
    /// Everything missing is named at once. A loader that stops at the first
    /// absence makes a reader fix a fetch one file per run.
    ///
    /// `manifest.json` is not asked for, though the fetch's own inventory does
    /// ask for it: it is a human-readable mirror of the manifest the checkpoint
    /// already embeds, and the engine reads the embedded one. Requiring the
    /// mirror here would refuse a hand-assembled directory that loads.
    public init(directory: URL) throws {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw LoudKitError.asset(
                "\(directory.path): not a directory. Pass the directory "
                + "`LoudKit.download` wrote, not a file inside it.")
        }
        var missing: [String] = []
        let tokenizer = directory.appendingPathComponent("tokenizer.json")
        if !LoudKit.isFile(directory, "tokenizer.json") { missing.append("tokenizer.json") }
        let coreml = directory.appendingPathComponent(Self.coremlDirectory)
        for package in LoudKit.coremlSynthesis
        where !LoudKit.isPackage(directory.appendingPathComponent(package)) {
            missing.append(package)
        }
        let found: URL
        do {
            found = try Self.findCheckpoint(in: directory)
        } catch {
            // Rethrown rather than folded into the list: "missing:
            // loudr-1.safetensors" is the wrong sentence for a directory
            // holding two of them, and that is the one case where the reader
            // has to be told something other than "fetch it".
            guard !missing.isEmpty else { throw error }
            throw LoudKitError.asset(
                "\(error). Also missing: " + missing.sorted().joined(separator: ", ") + ".")
        }
        guard missing.isEmpty else {
            throw LoudKitError.asset(
                "\(directory.path): not a loudkit release, missing: "
                + missing.sorted().joined(separator: ", ")
                + ". Fetch one with LoudKit.download(repo:to:).")
        }
        self.directory = directory
        self.checkpoint = found
        self.tokenizer = tokenizer
        self.coreml = coreml
    }

    /// The voices this release ships, sorted. Empty when it ships none.
    public var voiceNames: [String] {
        let names = (try? FileManager.default.contentsOfDirectory(
            atPath: directory.appendingPathComponent(Self.voiceDirectory).path)) ?? []
        return names.filter { $0.hasSuffix(".safetensors") && !$0.hasPrefix(".") }
            .map { String($0.dropLast(".safetensors".count)) }
            .sorted()
    }

    /// One shipped voice, by name. The error lists what the release does have,
    /// because a wrong voice name is a typo and a typo wants the list.
    public func voice(named name: String) throws -> VoiceProfile {
        let url = directory.appendingPathComponent(Self.voiceDirectory)
            .appendingPathComponent("\(name).safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoudKitError.voiceNotFound(
                name: name, directory: directory.path, shipped: voiceNames)
        }
        return try VoiceProfile.load(url: url)
    }

    /// The enroller over this release's enrollment packages, for cloning a
    /// voice from a recording. Needs the packages on disk: `fetchCloning`
    /// brings them for a release loaded by repo id, and a directory of your
    /// own needs a fetch made with `cloning: true`.
    public func enroller() throws -> Enrollment.Enroller {
        let absent = absentEnrollmentPackages
        guard absent.isEmpty else {
            throw LoudKitError.asset(
                "\(directory.path): enrollment needs \(absent.sorted().joined(separator: ", ")). "
                + "Fetch them with LoudKit.download(repo:to:cloning: true).")
        }
        return try Enrollment.Enroller(coremlDir: coreml)
    }

    /// Make sure the release holds the three enrollment packages. A release
    /// loaded by repo id fetches them into its own directory, through the
    /// same receipt-aware `LoudKit.download`, so the cache grows to the wider
    /// set once and stays a hit afterwards. A directory of your own is told
    /// how to fetch them.
    public func fetchCloning() async throws {
        let absent = absentEnrollmentPackages
        if absent.isEmpty { return }
        guard var fetch = origin else {
            throw LoudKitError.asset(
                "\(directory.path): enrollment needs \(absent.sorted().joined(separator: ", ")). "
                + "Fetch them with LoudKit.download(repo:to:cloning: true).")
        }
        fetch.cloning = true
        try await LoudKit.download(fetch, to: directory, progress: LoudKit.stderrProgress)
    }

    private var absentEnrollmentPackages: [String] {
        LoudKit.coremlEnrollment.filter { !LoudKit.isPackage(directory.appendingPathComponent($0)) }
    }

    /// The synthesis checkpoint in `directory`, mirroring
    /// `release._only_checkpoint_in`.
    ///
    /// Three rules in order: exactly one canonical name; then the file whose
    /// manifest declares the synthesis role; then the sole candidate, with
    /// anything declaring the *enrollment* role set aside. A release is two
    /// checkpoints now: the synthesis half beside the enrollment half is
    /// ordinary, and two *models* under their own names is not an ambiguity a
    /// resolver may settle, because they are different releases with different
    /// decode loops.
    static func findCheckpoint(in directory: URL) throws -> URL {
        let canonical = LoudKit.checkpointNames.filter {
            FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        if canonical.count == 1, let only = canonical.first {
            let named = directory.appendingPathComponent(only)
            try refuse(named, unless: "synthesis")
            return named
        }
        if canonical.count > 1 {
            throw LoudKitError.asset(
                "\(directory.path): \(canonical.count) models here "
                + "(\(canonical.joined(separator: ", "))): name the one you mean with "
                + "Engine.load(checkpoint:). They are different releases with different "
                + "decode loops, so there is no right one to pick.")
        }
        let found = rootCheckpoints(in: directory)
        let declared = found.filter { role($0) == "synthesis" }
        let candidates = declared.isEmpty ? found.filter { role($0) != "enrollment" } : declared
        if candidates.count == 1, let only = candidates.first { return only }
        if found.isEmpty {
            throw LoudKitError.asset(
                "\(directory.path): no *.safetensors here. A loudkit release is a "
                + "synthesis checkpoint and a voices/ directory, with the enrollment "
                + "checkpoint beside them when it ships. Fetch one with "
                + "LoudKit.download(repo:to:).")
        }
        if candidates.isEmpty {
            throw LoudKitError.asset(
                "\(directory.path): the only checkpoint here is a release's enrollment "
                + "artefact. It carries the two modules a clone needs and nothing "
                + "synthesis reads. Fetch the release's \(LoudKit.checkpointName) beside it.")
        }
        throw LoudKitError.asset(
            "\(directory.path): \(candidates.count) checkpoints "
            + "(\(candidates.map(\.lastPathComponent).joined(separator: ", "))): "
            + "name the one you mean with Engine.load(checkpoint:).")
    }

    /// Root-level files that could be a checkpoint, sorted.
    ///
    /// Voices are safetensors too and live in `voices/`, so the search does not
    /// recurse. `ve.safetensors` is at the root and is not one: excluding it by
    /// name keeps this a question about the weights rather than about how many
    /// safetensors a release happens to carry.
    static func rootCheckpoints(in directory: URL) -> [URL] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        return names.filter {
            $0.hasSuffix(".safetensors") && !$0.hasPrefix(".") && $0 != LoudKit.voiceEncoderName
                && LoudKit.isFile(directory, $0)
        }
        .sorted()
        .map(directory.appendingPathComponent)
    }

    /// `manifest["artifact_role"]`, or nil for a file that makes no claim.
    ///
    /// Nil covers both a pre-split checkpoint, whose manifest predates the
    /// field and which does carry every tensor, and a file this cannot read at
    /// all. The field is only ever read to refuse, so a missing claim never
    /// promotes a file to a role it did not ask for.
    static func role(_ url: URL) -> String? {
        guard let store = try? Safetensors(url: url),
              let json = store.metadata["manifest"],
              let parsed = try? JSONSerialization.jsonObject(with: Data(json.utf8))
                as? [String: Any] else { return nil }
        return parsed["artifact_role"] as? String
    }

    static func refuse(_ url: URL, unless expected: String) throws {
        if let role = role(url), role != expected {
            throw LoudKitError.asset(
                "\(url.lastPathComponent): this is a release's \(role) artefact, and the "
                + "\(expected) artefact is what was asked for.")
        }
    }
}

extension Engine {
    /// Load from a release directory: one URL instead of four.
    ///
    /// `Engine.load(checkpoint:coremlAssets:execution:)` is still there for a
    /// layout this does not describe: an app bundle, a hand-assembled set, a
    /// checkpoint beside precompiled `.mlmodelc` stages.
    public static func load(bundle directory: URL,
                            execution: ExecutionConfig = ExecutionConfig()) throws -> Engine {
        try load(bundle: ModelBundle(directory: directory), execution: execution)
    }

    /// The same, for a bundle already read.
    public static func load(bundle: ModelBundle,
                            execution: ExecutionConfig = ExecutionConfig()) throws -> Engine {
        let engine = try load(
            checkpoint: bundle.checkpoint, coremlAssets: bundle.coreml, execution: execution)
        engine.bundle = bundle
        return engine
    }

    /// Open a release: a directory holding one, or a repo id such as
    /// `"loudreader/loudr-1"`, which is fetched into
    /// `LoudKit.cacheDirectory(repo:)` and kept there. Every load goes
    /// through `LoudKit.download`, whose receipt makes the second one free.
    /// Progress goes to stderr unless `progress` is given.
    ///
    /// Anything that exists on disk is a path; `org/name` is a repo id;
    /// anything else is a path that does not exist, and the error says so.
    public static func load(_ ref: String,
                            revision: String = "main",
                            execution: ExecutionConfig = ExecutionConfig(),
                            progress: (@Sendable (LoudKit.DownloadProgress) -> Void)? = nil
    ) async throws -> Engine {
        if FileManager.default.fileExists(atPath: ref) {
            return try load(bundle: URL(fileURLWithPath: ref), execution: execution)
        }
        let ref = LoudKit.modelRepo(ref)
        guard LoudKit.isRepoID(ref) else {
            throw LoudKitError.asset(
                "\(ref): no such file or directory, and not a Hugging Face repo id "
                + "(those look like 'org/name')")
        }
        let directory = try await LoudKit.download(
            repo: ref, to: LoudKit.cacheDirectory(repo: ref), revision: revision, progress: progress)
        var bundle = try ModelBundle(directory: directory)
        bundle.origin = LoudKit.Fetch(repo: ref, revision: revision)
        return try load(bundle: bundle, execution: execution)
    }
}
