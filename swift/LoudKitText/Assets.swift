import Foundation

/// Where the engine's assets live.
///
/// A package cannot look these up through `Bundle.main`: the same sources
/// serve a command-line tool, an application and a test target, and only one of
/// those has a main bundle worth asking. So the lookup is a single seam, set
/// once before the engine loads.
///
/// Same shape as `Bundle.url(forResource:withExtension:)` on purpose: a call
/// site differs from the Foundation one by a single identifier, which is what
/// makes the two cheap to keep in step by eye.
public enum ChatterboxAssets {

    /// Directories searched in order. Set this before `Engine.load`.
    ///
    /// `nonisolated(unsafe)`: written once at startup, read from every
    /// synthesis actor afterwards. Making it an actor would put an `await` in
    /// front of every table read on the load path for a value that never
    /// changes after the first line of `main`.
    public nonisolated(unsafe) static var searchPaths: [URL] = defaultSearchPaths

    /// The bundle's own resources, if there are any, then the executable's
    /// directory, which for a command-line binary is where a sidecar asset
    /// folder naturally sits.
    static var defaultSearchPaths: [URL] {
        var paths: [URL] = []
        if let resources = Bundle.main.resourceURL { paths.append(resources) }
        let executable = Bundle.main.bundleURL.deletingLastPathComponent()
        if !paths.contains(executable) { paths.append(executable) }
        return paths
    }

    /// The first `name.ext` found on `searchPaths`, or `nil`.
    ///
    /// Every pass that reads a data file resolves it through here, so an
    /// application shipping its own copy is read by the passes and hashed by
    /// the digest alike.
    public static func url(forResource name: String, withExtension ext: String) -> URL? {
        for base in searchPaths {
            let candidate = base.appendingPathComponent("\(name).\(ext)")
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        return nil
    }
}
