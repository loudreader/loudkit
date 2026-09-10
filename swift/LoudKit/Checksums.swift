import CryptoKit
import Foundation

/// The release's own `SHA256SUMS`, checked against the bytes that arrived.
///
/// Every file a fetch brings is hashed once, when it arrives, and a file
/// already on disk is hashed against the manifest before it is kept; a
/// directory whose valid receipt names today's commit neither fetches nor
/// hashes a weight.
/// The rules are `python/loudkit/checksums.py`'s, so the package refuses what
/// the Python client refuses.
extension LoudKit {
    static let sumsName = "SHA256SUMS"
    static let releaseRecord = "release.json"

    /// The org whose releases this package vouches for: the only case where a
    /// missing manifest is a release that lost a file rather than a stranger's
    /// bare upload.
    static let officialOrg = "loudreader"

    /// The profiles a builder stamps on a releasable bundle: `full-0.1` for
    /// loudr-1 and `turbo-0.1` for loudr-1-turbo. Neither is written until
    /// the builder's load-and-speak gate passes.
    static let strictProfiles = ["full-0.1", "turbo-0.1"]

    // MARK: the check

    /// Hash one file this fetch brought against the release's `SHA256SUMS`.
    ///
    /// No manifest is fine only for a stranger's repo. A fetched file the
    /// manifest does not list is refused when it is weights, refused under an
    /// official repo whatever it is, and reported otherwise. A digest that
    /// does not match is refused, and the file removed so the next run
    /// fetches it again.
    static func verifyFetched(_ root: URL, repo: String, name: String,
                              sums: [String: String]?) throws {
        guard let sums else { return }
        let target = root.appendingPathComponent(name)
        guard let expected = sums[name] else {
            if name.hasSuffix(".safetensors") {
                try? FileManager.default.removeItem(at: target)
                throw LoudKitError.asset(
                    "\(root.path): \(sumsName) does not list \(name), which is weights loudkit "
                    + "would open with nothing vouching for them. Pin a revision you trust.")
            }
            if isOfficial(repo) {
                try? FileManager.default.removeItem(at: target)
                throw LoudKitError.asset(
                    "\(root.path): \(sumsName) does not list \(name). A \(officialOrg) release "
                    + "checksums every file it ships, so this did not come from the release.")
            }
            FileHandle.standardError.write(Data(
                ("loudkit: \(name) is not covered by \(sumsName) and therefore not verified\n").utf8))
            return
        }
        if try fileSHA256(target) != expected {
            // Reported as attempted, not as done: the removal is best effort
            // and a caller told the file is gone when it is still there runs
            // the download again into the same failure.
            let removed = (try? FileManager.default.removeItem(at: target)) != nil
            throw LoudKitError.asset(
                "\(root.path): \(name) failed the release checksum. "
                + (removed ? "The file has been removed; run the download again, "
                           : "The file could not be removed; delete it, then run the download again, ")
                + "or pin a revision you trust.")
        }
    }

    /// An official repo's `release.json` must name a release profile, record
    /// `verified: true`, and be listed by the manifest that vouches for it.
    static func requireReleasable(_ root: URL, repo: String, sums: [String: String]) throws {
        let record = root.appendingPathComponent(releaseRecord)
        guard isFile(root, releaseRecord) else {
            throw LoudKitError.asset(
                "\(repo): no release.json. Every \(officialOrg) release records its "
                + "profile and its verified flag there, so this download cannot prove it "
                + "is a release. Pin a revision you trust.")
        }
        guard let expected = sums[releaseRecord] else {
            throw LoudKitError.asset(
                "\(repo): \(sumsName) does not list release.json, so the record that would "
                + "vouch for this release is vouched for by nothing. Pin a revision you trust.")
        }
        guard try fileSHA256(record) == expected else {
            throw LoudKitError.asset(
                "\(repo): release.json failed the release checksum. Delete the directory "
                + "and download again.")
        }
        guard let parsed = try? JSONSerialization.jsonObject(with: Data(contentsOf: record)),
              let claim = parsed as? [String: Any] else {
            throw LoudKitError.asset(
                "\(repo): release.json is unreadable. Delete the directory and download again.")
        }
        let profile = claim["profile"] as? String
        guard let profile, strictProfiles.contains(profile) else {
            throw LoudKitError.asset(
                "\(repo): release.json says profile \(profile.map { "\"\($0)\"" } ?? "none"), "
                + "and a \(officialOrg) release is one of \(strictProfiles.joined(separator: ", ")). "
                + "This is a development bundle, not the release.")
        }
        guard claim["verified"] as? Bool == true else {
            throw LoudKitError.asset(
                "\(repo): release.json does not record verified: true, so the bundle never "
                + "passed the builder's load-and-speak gate.")
        }
    }

    static func isOfficial(_ repo: String) -> Bool {
        let parts = repo.split(separator: "/", omittingEmptySubsequences: false)
        return parts.count == 2 && parts[0].lowercased() == officialOrg
    }

    // MARK: the manifest

    /// Parse `SHA256SUMS`, refusing every line it cannot understand: each
    /// non-empty line is a 64-hex digest, two spaces, and a normalised
    /// relative POSIX path. A name that escapes the directory, or one listed
    /// twice, is refused with the line it came from.
    static func parseSums(_ sums: URL) throws -> [String: String] {
        guard let text = String(bytes: try Data(contentsOf: sums), encoding: .utf8) else {
            throw LoudKitError.asset("\(sums.path): not UTF-8, so it is not a manifest")
        }
        var out: [String: String] = [:]
        for (index, raw) in text.split(separator: "\n", omittingEmptySubsequences: false)
            .enumerated() {
            let line = raw.hasSuffix("\r") ? String(raw.dropLast()) : String(raw)
            let number = index + 1
            if line.trimmingCharacters(in: .whitespaces).isEmpty { continue }
            guard let (digest, name) = sumsLine(line) else {
                throw LoudKitError.asset(
                    "\(sums.path): malformed \(sumsName) line \(number): \"\(line)\"; this does "
                    + "not look like a loudkit release manifest, refusing to verify against it")
            }
            if let why = rejectedName(name) {
                throw LoudKitError.asset(
                    "\(sums.path): line \(number): \"\(name)\" \(why); refusing to verify "
                    + "against a manifest that names files outside the release it describes")
            }
            guard out[name] == nil else {
                throw LoudKitError.asset(
                    "\(sums.path): line \(number): duplicate entry for \"\(name)\"; the "
                    + "manifest disagrees with itself about one file. Rebuild the release.")
            }
            out[name] = digest
        }
        guard !out.isEmpty else {
            throw LoudKitError.asset("\(sums.path): no checksum entries")
        }
        return out
    }

    /// `^([0-9a-f]{64})  (\S.*)$`, spelled out.
    static func sumsLine(_ line: String) -> (digest: String, name: String)? {
        let hex = Set("0123456789abcdef")
        let chars = Array(line)
        guard chars.count > 66, chars[64] == " ", chars[65] == " ",
              chars[66] != " ", chars[66] != "\t",
              chars[..<64].allSatisfy(hex.contains) else { return nil }
        return (String(chars[..<64]), String(chars[66...]))
    }

    /// Why `name` cannot address bytes inside the directory, or nil.
    static func rejectedName(_ name: String) -> String? {
        if name.contains("\\") { return "is not a POSIX path (it contains a backslash)" }
        if name.hasPrefix("/") || isDriveRooted(name) {
            return "is absolute, and a manifest name is relative to the release root"
        }
        let parts = name.split(separator: "/", omittingEmptySubsequences: false)
        if parts.contains("..") { return "escapes the release root with '..'" }
        if name.isEmpty || parts.contains(where: { $0.isEmpty || $0 == "." }) {
            return "is not normalised (an empty or '.' path component)"
        }
        return nil
    }

    /// `c:/x`, the other spelling of "rooted at a filesystem". Narrow on
    /// purpose: a colon is legal in a POSIX filename.
    static func isDriveRooted(_ name: String) -> Bool {
        let chars = Array(name)
        return chars.count >= 3 && chars[0].isLetter && chars[0].isASCII
            && chars[1] == ":" && chars[2] == "/"
    }

    /// SHA-256 of a file, in 1 MB blocks: the checkpoint is 747 MB.
    static func fileSHA256(_ url: URL) throws -> String {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            throw LoudKitError.asset("\(url.path): cannot be read to check it (\(error))")
        }
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            guard let block = try handle.read(upToCount: 1 << 20), !block.isEmpty else { break }
            hasher.update(data: block)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
