import Foundation

/// Where the engine's assets live.
///
/// In the app these files are bundle resources and every lookup went through
/// `Bundle.main`. A package cannot assume that: the same sources now serve a
/// command-line tool, a menu-bar app, and a test target, and only one of those
/// has a main bundle worth asking. So the lookup is a single seam, set once
/// before the engine loads.
///
/// Same shape as `Bundle.url(forResource:withExtension:)` on purpose — the
/// call sites in the copied engine sources differ from the app's by one
/// identifier, which is what makes them cheap to keep in step by eye.
public enum ChatterboxAssets {

    /// Directories searched in order. Set this before `SpeechEngine.load`.
    ///
    /// `nonisolated(unsafe)`: written once at startup, read from every
    /// synthesis actor afterwards. Making it an actor would put an `await` in
    /// front of every table read on the load path for a value that never
    /// changes after the first line of `main`.
    public nonisolated(unsafe) static var searchPaths: [URL] = defaultSearchPaths

    /// The bundle's own resources, if there are any, then the executable's
    /// directory — which for a command-line binary is where a sidecar asset
    /// folder naturally sits.
    static var defaultSearchPaths: [URL] {
        var paths: [URL] = []
        if let resources = Bundle.main.resourceURL { paths.append(resources) }
        let executable = Bundle.main.bundleURL.deletingLastPathComponent()
        if !paths.contains(executable) { paths.append(executable) }
        return paths
    }

    public static func url(forResource name: String, withExtension ext: String) -> URL? {
        for base in searchPaths {
            let candidate = base.appendingPathComponent("\(name).\(ext)")
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        return nil
    }
}
