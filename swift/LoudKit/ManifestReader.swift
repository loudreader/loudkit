import Foundation

/// Reading a checkpoint manifest's numbers, names and flags, refusing a value
/// that has no reading here.
///
/// Answering the default for a value with no reading hands the engine a setting
/// the manifest never declared. `window.static_length: "255"` ignored is a
/// ragged window under a manifest that asked for a padded one, and
/// `sampling_defaults.temperature: "0.9"` ignored is 0.8 under a manifest that
/// asked for 0.9. The fallback is then written into the canonical form as if
/// the manifest had declared it, so the fingerprint agrees with the misreading
/// instead of reporting it, and that is the divergence class this library
/// exists to prevent.
///
/// `manifest.algorithm_from` converts each of these with `int()`, `float()` or
/// `str()`, which refuse a null, a list and a non-numeric string. Go reads them
/// through `config.numbers`, Rust through `checkpoint::Numbers`; this port
/// throws where those two accumulate, because `fromManifest` already throws and
/// the answer is the same either way: the first value the reader cannot read is
/// refused, by name, before any of it reaches a config or a fingerprint.
enum ManifestReader {
    /// The one numeric reading of a decoded JSON value.
    ///
    /// `JSONSerialization` returns an `NSNumber` for a JSON number *and* for a
    /// JSON boolean, so `as? NSNumber` on its own accepts `true` and reads it
    /// as one. `CFBooleanGetTypeID` is the only reliable way to tell those
    /// apart here: `as? Bool` succeeds for any number that happens to equal
    /// zero or one, so it cannot be used to reject one, and `min_p: 0` and
    /// `guidance_rate: 0` are values a manifest is entitled to declare.
    ///
    /// The reference is looser on one input and this port does not follow it
    /// there. `int("255")` is 255 in Python, so a manifest carrying a quoted
    /// length loads in the reference and is refused here, as it is in Go and
    /// Rust. Refusing is the safe side: the alternative is not accepting it, it
    /// is accepting it *differently* in five languages, each with its own
    /// reading of `"255 "`, `"0x10"` and `"1e3"`.
    static func asNumber(_ value: Any) -> Double? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
        return number.doubleValue
    }

    /// The one boolean reading, by the same rule read the other way round.
    static func asBoolean(_ value: Any) -> Bool? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) == CFBooleanGetTypeID() else { return nil }
        return number.boolValue
    }

    /// A JSON number as an `Int`, truncated toward zero the way `int()`
    /// truncates it.
    ///
    /// A magnitude no `Int` can hold is refused rather than converted: the
    /// conversion traps, and a trap is a crash report where the manifest key
    /// that caused it does not appear.
    private static func held(_ value: Double, _ path: String) throws -> Int {
        guard value.isFinite, value.magnitude < 9.007199254740992e15 else {
            throw LoudKitError.manifest("\(path) is not a whole number this port can hold: \(value)")
        }
        return Int(value)
    }

    /// A JSON number as the count a manifest key that counts things carries:
    /// whole, and within what this port holds.
    ///
    /// A count of 2.7 is refused rather than truncated. A token id, a window
    /// length or a token budget written as a fraction was computed wrong, and
    /// reading it as 2 turns that mistake into a chunker that breathes in a
    /// different place, under a `recipe_version` saying the five
    /// implementations agree. `manifest._int` refuses it in this sentence.
    ///
    /// The fraction is named before the magnitude because the reference names
    /// only the fraction: a negative fraction is refused for what it is.
    ///
    /// Not the float fields. A temperature, a token rate, a guidance rate and
    /// a fade length take 2.7 as a value, and a check applied to them would
    /// refuse a manifest that is correct.
    private static func whole(_ value: Double, _ path: String) throws -> Int {
        guard value == value.rounded(.towardZero) else {
            throw LoudKitError.manifest("\(path) must be a whole number, got \(value)")
        }
        return try held(value, path)
    }

    /// A declared format version, truncated the way `int()` truncates it.
    ///
    /// Not `count`. The reference reads this key with a bare `int()` in
    /// `checkpoint._read_manifest`, one module away from the reader that
    /// refuses a fractional count, so a manifest saying `format_version: 2.7`
    /// opens in all five. What decides the file here is the number, which the
    /// caller checks against the versions this build implements.
    static func formatVersion(
        _ manifest: [String: Any], _ key: String, _ fallback: Int
    ) throws -> Int {
        guard manifest[key] != nil else { return fallback }
        return try held(try number(manifest, "manifest", key, Double(fallback)),
                        "manifest['\(key)']")
    }

    /// A block of the manifest, or an empty one when the key is absent.
    ///
    /// A key of the wrong type is refused rather than skipped: the whole block
    /// used to vanish on a wrong type, taking its range checks with it, so a
    /// manifest that declared a sampling law ran under the shipping one.
    /// `manifest._block` refuses the same key the same way.
    static func block(_ manifest: [String: Any], _ key: String) throws -> [String: Any] {
        guard let raw = manifest[key] else { return [:] }
        guard let map = raw as? [String: Any] else {
            throw LoudKitError.manifest(
                "manifest['\(key)'] must be a mapping, got \(type(of: raw))")
        }
        return map
    }

    /// A numeric key of `block`, or the default for an absent one.
    ///
    /// Absent is not zero: only a missing key takes the default, so a manifest
    /// declaring `min_p: 0` (no truncation, a legal and meaningful setting)
    /// keeps its zero, and so does a deliberately disabled EOS floor.
    static func number(
        _ block: [String: Any], _ path: String, _ key: String, _ fallback: Double
    ) throws -> Double {
        guard let raw = block[key] else { return fallback }
        guard let value = asNumber(raw) else {
            throw LoudKitError.manifest("\(path)['\(key)'] should be a number, got \(raw)")
        }
        return value
    }

    /// A numeric key of `block` held here as a count or an id.
    static func count(
        _ block: [String: Any], _ path: String, _ key: String, _ fallback: Int
    ) throws -> Int {
        guard block[key] != nil else { return fallback }
        return try whole(try number(block, path, key, Double(fallback)), "\(path)['\(key)']")
    }

    /// An optional length: a number, an explicit null for the unset reading, or
    /// a refusal for anything else.
    ///
    /// `count` has no null reading because the values it reads have none: a
    /// token rate is a number or it is a mistake. A window length is a number
    /// or the ragged reading, and `_window_from` spells that with one `opt`
    /// helper that reads an absent key and an explicit null the same way, while
    /// reading `max_speech_tokens` with a bare `int()` that refuses a null like
    /// any other non-number.
    static func optCount(_ block: [String: Any], _ path: String, _ key: String) throws -> Int? {
        guard let raw = block[key], !(raw is NSNull) else { return nil }
        guard let value = asNumber(raw) else {
            throw LoudKitError.manifest(
                "\(path)['\(key)'] should be a number or null, got \(raw)")
        }
        return try whole(value, "\(path)['\(key)']")
    }

    /// A string key of `block`, or nothing for an absent one.
    ///
    /// Every caller names a closed set and checks the answer against it; this
    /// settles only the shape, so the "unknown mode" refusals keep naming their
    /// own options. A present key ignored for having the wrong type is a law
    /// the manifest declared and this port did not run. The reference arrives
    /// at the same refusal from the other side: `str()` renders the value and
    /// the membership check fails on what it rendered.
    static func text(_ block: [String: Any], _ path: String, _ key: String) throws -> String? {
        guard let raw = block[key] else { return nil }
        guard let value = raw as? String else {
            throw LoudKitError.manifest("\(path)['\(key)'] must be a string, got \(type(of: raw))")
        }
        return value
    }

    /// A boolean key of `block`, or nothing for an absent one.
    ///
    /// Not ignored when it is not a boolean: a manifest that turns the splitter
    /// off and a runtime that leaves it on read the same `recipe_version` and
    /// deliver different audio, and `enabled` is the one key whose misreading
    /// moves every join at once. The reference coerces with `bool()`, which
    /// reads `null` and `[]` as off; a refusal is the safe side of a coercion
    /// no two languages spell alike.
    static func flag(_ block: [String: Any], _ path: String, _ key: String) throws -> Bool? {
        guard let raw = block[key] else { return nil }
        guard let value = asBoolean(raw) else {
            throw LoudKitError.manifest("\(path)['\(key)'] must be JSON true or false, got \(raw)")
        }
        return value
    }

    /// A list of strings under `block`, or nothing for an absent key.
    ///
    /// A bare string is refused rather than iterated: `split_on: ". "` is two
    /// separators of one character each once Swift walks it, which is not what
    /// any manifest means. An element of the wrong type is refused for the same
    /// reason one level down; the reference renders each with `str()`, a
    /// coercion this port does not copy, for the reason `asNumber` gives.
    static func strings(
        _ block: [String: Any], _ path: String, _ key: String
    ) throws -> [String]? {
        guard let raw = block[key] else { return nil }
        guard let list = raw as? [Any] else {
            throw LoudKitError.manifest(
                "\(path)['\(key)'] must be a list of strings, got \(type(of: raw))")
        }
        return try list.enumerated().map { index, item in
            guard let value = item as? String else {
                throw LoudKitError.manifest(
                    "\(path)['\(key)'][\(index)] must be a string, got \(type(of: item))")
            }
            return value
        }
    }

    /// A top-level list of token ids, or an empty census for an absent key.
    ///
    /// The container is refused when it is not a list, and a string by name:
    /// `silence_token_ids: "123"` is a `Sequence` in the reference too, and
    /// must not load as three tokens. An element with no numeric reading is
    /// refused for the reason the whole list is: the reference builds the
    /// census with `int()` per element, and one element dropped leaves the
    /// stall detector reading a census the manifest never declared while the
    /// canonical form records the short one.
    static func ids(_ manifest: [String: Any], _ key: String) throws -> [Int] {
        guard let raw = manifest[key] else { return [] }
        guard let list = raw as? [Any] else {
            throw LoudKitError.manifest("manifest['\(key)'] must be a list, got \(type(of: raw))")
        }
        return try list.enumerated().map { index, item in
            guard let value = asNumber(item) else {
                throw LoudKitError.manifest(
                    "manifest['\(key)'][\(index)] should be a number, got \(item)")
            }
            return try whole(value, "manifest['\(key)'][\(index)]")
        }
    }

    /// The explicit Euler time grid, or nothing for the cosine schedule.
    ///
    /// Absent and null are the cosine schedule. A point with no numeric reading
    /// is refused rather than dropped: a shorter grid integrates on a schedule
    /// the manifest never declared, and the canonical form records the short
    /// one. `algorithm_from` refuses this key by name.
    static func floats(_ manifest: [String: Any], _ key: String) throws -> [Double]? {
        guard let raw = manifest[key], !(raw is NSNull) else { return nil }
        guard let list = raw as? [Any] else {
            throw LoudKitError.manifest(
                "manifest['\(key)'] must be a list of floats or null, got \(type(of: raw))")
        }
        return try list.enumerated().map { index, item in
            guard let value = asNumber(item) else {
                throw LoudKitError.manifest(
                    "manifest['\(key)'][\(index)] should be a number, got \(item)")
            }
            return value
        }
    }

    // MARK: duplicate keys

    /// Parse a JSON document the way `json.loads` parses it: two members with
    /// the same name leave the **last** one standing.
    ///
    /// Foundation keeps the **first**, and there is no option to change it --
    /// `JSONSerialization` and `JSONDecoder` both do. Python, Go's
    /// `encoding/json`, `serde_json` and `JSON.parse` all keep the last. So a
    /// manifest carrying `"sample_rate":24000,"sample_rate":48000` ran four
    /// engines at 48 kHz and this one at 24, from identical bytes, and because
    /// each port hashed the value it had read, the fingerprints agreed about a
    /// disagreement. The same held for a nested block.
    ///
    /// The earlier member is removed from the raw bytes before Foundation sees
    /// them. A document with no duplicate -- every pack this project ships --
    /// is handed over untouched, so the common path is byte-for-byte what it
    /// always was. The scan's only job is finding duplicates: parsing and
    /// validating stay Foundation's, and a document the scan cannot follow is
    /// passed through for Foundation to refuse.
    ///
    /// `what` is what the caller was reading, and it is not optional because a
    /// caller that cannot name it has nothing to put in the message. A parse
    /// failure used to surface as `NSCocoaErrorDomain Code=3840 "Unable to
    /// parse empty data."`, which says neither which file nor which of its
    /// parts. Go says `%s: bad manifest JSON: %w` and Rust
    /// `{path}: bad manifest JSON: {e}`; this is the same sentence.
    ///
    /// Fragments are allowed for the same reason `json.loads` allows them: a
    /// header that is a bare string or number has to reach the caller's type
    /// check, which refuses it by name, rather than dying as a syntax error.
    static func json(_ data: Data, _ what: String) throws -> Any {
        var document = data
        if let cut = try? duplicateSpans(in: data), !cut.isEmpty {
            var kept = Data(capacity: data.count)
            var at = data.startIndex
            for span in cut {
                kept.append(contentsOf: data[at..<span.lowerBound])
                at = span.upperBound
            }
            kept.append(contentsOf: data[at...])
            document = kept
        }
        do {
            return try JSONSerialization.jsonObject(
                with: document, options: [.fragmentsAllowed])
        } catch {
            throw LoudKitError.asset(
                "\(what): \((error as NSError).localizedDescription)")
        }
    }

    /// What a decoded JSON value is, in the words a manifest is written in,
    /// as a phrase that reads after "is": "an array", "a string", "null".
    ///
    /// `type(of:)` answers `__NSArrayI`, `__NSCFNumber` and `NSNull` here,
    /// which are Foundation's business and not the reader's: a message naming
    /// them tells whoever wrote the file nothing about what to change.
    static func jsonTypePhrase(_ value: Any) -> String {
        switch value {
        case is [String: Any]: return "an object"
        case is [Any]: return "an array"
        case is String: return "a string"
        case is NSNull: return "null"
        case let number as NSNumber:
            return CFGetTypeID(number) == CFBooleanGetTypeID() ? "a boolean" : "a number"
        default: return "a \(type(of: value))"
        }
    }

    /// A decoded JSON value spelled the way the reference's refusals quote it
    /// back, so the same bad file reads the same in both.
    ///
    /// Foundation's own rendering is the reason this exists: `"\(value)"`
    /// answers `<null>` for a null, `1` for `true`, `abc` for a string that
    /// was quoted in the file, and a multi-line property list for an array or
    /// a dictionary. Object keys are sorted, matching the other three ports.
    static func pyRepr(_ value: Any) -> String {
        switch value {
        case is NSNull: return "None"
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                return number.boolValue ? "True" : "False"
            }
            // A float keeps its point and an integer has none, which is how
            // `repr` tells 5.0 from 5.
            return CFNumberIsFloatType(number)
                ? "\(number.doubleValue)" : "\(number.int64Value)"
        case let text as String:
            // repr's own rule: single quotes unless the value holds one and no
            // double quote.
            if text.contains("'") && !text.contains("\"") { return "\"\(text)\"" }
            let escaped = text
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "'", with: "\\'")
            return "'\(escaped)'"
        case let list as [Any]:
            return "[\(list.map(pyRepr).joined(separator: ", "))]"
        case let block as [String: Any]:
            let body = block.keys.sorted().map { "\(pyRepr($0)): \(pyRepr(block[$0]!))" }
            return "{\(body.joined(separator: ", "))}"
        default: return "\(value)"
        }
    }

    /// Errors that mean "this scanner cannot follow the document", never
    /// "the document is bad". Only `json` sees them, and it answers by
    /// standing aside.
    private struct NotFollowable: Error {}

    /// Byte ranges of the members a last-wins reading discards, in order and
    /// non-overlapping. Each range covers the member and the comma after it.
    private static func duplicateSpans(in data: Data) throws -> [Range<Data.Index>] {
        var scanner = JSONScan(bytes: [UInt8](data), base: data.startIndex)
        try scanner.value(depth: 0)
        scanner.skipSpace()
        guard scanner.at == scanner.bytes.count else { throw NotFollowable() }
        return scanner.drop.sorted { $0.lowerBound < $1.lowerBound }
    }

    /// Just enough JSON to know where each object member starts and ends.
    ///
    /// Not a parser: it reads no values and builds no tree. RFC 8259 shapes
    /// only, so that a `{`, a `,` or a `"` inside a string is never mistaken
    /// for structure.
    private struct JSONScan {
        let bytes: [UInt8]
        let base: Data.Index
        var at = 0
        var drop: [Range<Data.Index>] = []

        /// Deep enough for any manifest, shallow enough that a hostile file
        /// cannot run this recursion out of stack.
        static let maxDepth = 64

        mutating func skipSpace() {
            while at < bytes.count, bytes[at] == 0x20 || bytes[at] == 0x09
                    || bytes[at] == 0x0A || bytes[at] == 0x0D {
                at += 1
            }
        }

        mutating func expect(_ byte: UInt8) throws {
            guard at < bytes.count, bytes[at] == byte else { throw NotFollowable() }
            at += 1
        }

        /// The bytes of a string token, quotes included, or nil when there is
        /// no string here.
        mutating func string() throws -> [UInt8] {
            try expect(0x22)  // "
            let from = at - 1
            while at < bytes.count {
                let byte = bytes[at]
                if byte == 0x5C {  // backslash: the next byte cannot end the string
                    at += 2
                    continue
                }
                if byte == 0x22 {
                    at += 1
                    return Array(bytes[from..<at])
                }
                at += 1
            }
            throw NotFollowable()
        }

        mutating func value(depth: Int) throws {
            guard depth < Self.maxDepth else { throw NotFollowable() }
            skipSpace()
            guard at < bytes.count else { throw NotFollowable() }
            switch bytes[at] {
            case 0x7B: try object(depth: depth)  // {
            case 0x5B: try array(depth: depth)  // [
            case 0x22: _ = try string()  // "
            default: try scalar()
            }
        }

        mutating func array(depth: Int) throws {
            try expect(0x5B)
            skipSpace()
            if at < bytes.count, bytes[at] == 0x5D { at += 1; return }
            while true {
                try value(depth: depth + 1)
                skipSpace()
                guard at < bytes.count else { throw NotFollowable() }
                if bytes[at] == 0x2C { at += 1; continue }  // ,
                if bytes[at] == 0x5D { at += 1; return }  // ]
                throw NotFollowable()
            }
        }

        mutating func object(depth: Int) throws {
            try expect(0x7B)
            skipSpace()
            if at < bytes.count, bytes[at] == 0x7D { at += 1; return }
            // Name -> the span of the member seen so far under that name,
            // including the comma that follows it. The span is only ever
            // discarded once a *later* member takes the name, and a member
            // that is discarded is by definition not the object's last, so
            // the comma after it always exists.
            var seen: [String: Range<Int>] = [:]
            while true {
                let from = at
                let raw = try string()
                guard let name = Self.decode(raw) else { throw NotFollowable() }
                skipSpace()
                try expect(0x3A)  // :
                try value(depth: depth + 1)
                skipSpace()
                guard at < bytes.count else { throw NotFollowable() }
                let comma = bytes[at] == 0x2C
                if !comma, bytes[at] != 0x7D { throw NotFollowable() }
                if let previous = seen[name] {
                    drop.append((base + previous.lowerBound)..<(base + previous.upperBound))
                }
                if comma {
                    at += 1
                    seen[name] = from..<at
                    skipSpace()
                    continue
                }
                at += 1  // }
                return
            }
        }

        /// A number, `true`, `false` or `null`: run to the next structural
        /// byte. The document is Foundation's to validate, so the shape of
        /// what is skipped does not have to be checked, only its end found.
        mutating func scalar() throws {
            let from = at
            while at < bytes.count {
                let byte = bytes[at]
                if byte == 0x2C || byte == 0x7D || byte == 0x5D || byte == 0x20
                    || byte == 0x09 || byte == 0x0A || byte == 0x0D { break }
                if byte == 0x22 || byte == 0x7B || byte == 0x5B { throw NotFollowable() }
                at += 1
            }
            if at == from { throw NotFollowable() }
        }

        /// A member name, unescaped. `"a"` and `"a"` are one name, so the
        /// comparison is on the decoded text and not on the token.
        ///
        /// A token with no backslash in it *is* its own bytes, and taking that
        /// path matters: the other one costs a `JSONSerialization` call per
        /// member, and `tokenizer.json` has a member per vocabulary entry.
        ///
        /// Failable rather than lossy on both paths: bytes that are not UTF-8
        /// would otherwise become replacement characters, and two names that
        /// differ only in the bytes replaced would compare equal. `nil` here
        /// stops the scan, and Foundation refuses the document itself.
        static func decode(_ token: [UInt8]) -> String? {
            let inner = token[(token.startIndex + 1)..<(token.endIndex - 1)]
            if !inner.contains(0x5C) { return String(bytes: inner, encoding: .utf8) }
            guard let any = try? JSONSerialization.jsonObject(
                with: Data(token), options: [.fragmentsAllowed]) else { return nil }
            return any as? String
        }
    }
}
