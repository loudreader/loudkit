/// Python reference: `loudkit/frontend/polish.py`.
import Foundation
import LoudKitText

/// Text to text-tokens, exactly as the shipped engine and `loudkit.frontend.text`
/// do it. Lifted from the production tokenizer port (ChatterboxTokenizer.swift,
/// bit-parity tested against the Python reference) and kept deliberately thin:
/// lowercase, NFKD, a language tag, spaces to `[SPACE]`, then plain BPE over
/// Unicode *scalars* — NFKD combining marks stay separate symbols, exactly
/// like Python code points.
public final class TextFrontend {
    /// Refused languages whose refusal has a *specific* reason worth stating:
    /// their upstream pipeline wants Cangjie codes, kana conversion,
    /// diacritisation, jamo decomposition or stress marks, none of which this
    /// frontend carries. A subset of "not on the roster", kept so the message
    /// can say why rather than just no.
    static let needsModelPreprocessing: Set<String> = ["zh", "ja", "he", "ko", "ru"]

    /// The allowlist: the twelve ids in `numbers.json`.
    ///
    /// This was a blacklist of the five above, and the difference matters
    /// because the tokenizer's vocabulary carries tags for 31 languages. A
    /// blacklist let the other 26 through and the tag was emitted, so
    /// `encode(text, language: "bg")` NFKD-mangled Cyrillic into ids the model
    /// reads as sounds it was never trained to make — no error, plausible
    /// audio, wrong language.
    static var supportedLanguages: [String] { Numbers.supportedLanguages }

    private let vocab: [String: Int]
    private let mergeRank: [String: Int]
    private let addedTokens: [(content: String, id: Int)]
    private let addedRegex: NSRegularExpression?
    private let preTokRegex: NSRegularExpression
    private let unkId: Int

    public init(tokenizerURL: URL) throws {
        let data = try Data(contentsOf: tokenizerURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let vocabRaw = model["vocab"] as? [String: Any],
              let mergesRaw = model["merges"] as? [Any],
              let addedRaw = root["added_tokens"] as? [[String: Any]] else {
            throw LoudKitError.asset("bad tokenizer file: \(tokenizerURL.lastPathComponent)")
        }
        var v = [String: Int](minimumCapacity: vocabRaw.count)
        for (k, value) in vocabRaw { v[k] = (value as? NSNumber)?.intValue ?? -1 }
        vocab = v
        var ranks = [String: Int](minimumCapacity: mergesRaw.count)
        for (i, m) in mergesRaw.enumerated() {
            if let s = m as? String {
                let parts = s.split(separator: " ", maxSplits: 1).map(String.init)
                if parts.count == 2 { ranks[parts[0] + "\u{0}" + parts[1]] = i }
            } else if let pair = m as? [String], pair.count == 2 {
                ranks[pair[0] + "\u{0}" + pair[1]] = i
            }
        }
        mergeRank = ranks
        var added: [(String, Int)] = addedRaw.compactMap { entry in
            guard let content = entry["content"] as? String,
                  let id = (entry["id"] as? NSNumber)?.intValue else { return nil }
            return (content, id)
        }
        added.sort { $0.0.count > $1.0.count }  // longest-first greedy literal match
        addedTokens = added
        let alternation = added
            .map { NSRegularExpression.escapedPattern(for: $0.0) }
            .joined(separator: "|")
        // A tokenizer with no added tokens makes this the empty pattern, which
        // ICU accepts and which then matches the empty string at every position
        // — every character its own token. Unreachable with the shipped
        // tokenizer, and one manifest away from being reachable, so it is a
        // `nil` rather than a pattern that cannot fail loudly.
        addedRegex = added.isEmpty
            ? nil
            : try NSRegularExpression(pattern: alternation)
        preTokRegex = try NSRegularExpression(
            pattern: "\\w+|[^\\w\\s]+", options: [.useUnicodeWordBoundaries])
        guard let unk = v["[UNK]"], v["[START]"] != nil, v["[STOP]"] != nil,
              v["[SPACE]"] != nil else {
            throw LoudKitError.asset("tokenizer vocabulary is missing special tokens")
        }
        unkId = unk
    }

    /// Normalise and tokenise. Same text and language give the same ids —
    /// and the same ids as `GraphemeTextFrontend.encode` on the Python side
    /// (the conformance fixture pins several trap sentences).
    public func encode(_ text: String, language: String = "en") throws -> [Int] {
        let lang = language.lowercased()
        let roster = Self.supportedLanguages
        if !roster.contains(lang) {
            let why = Self.needsModelPreprocessing.contains(lang)
                ? "needs model-based text preprocessing "
                    + "(Cangjie/kana/diacritics/jamo/stress) that this frontend does not carry"
                : "is not one of the languages this build's text layer is written for"
            throw LoudKitError.asset(
                "language \(lang) \(why). Supported: \(roster.joined(separator: ", "))")
        }
        var t = text.lowercased()
        t = t.decomposedStringWithCompatibilityMapping  // NFKD
        // Square brackets never reach the tokenizer from user text: the
        // vocabulary holds 117 bracket control tokens ([sigh], [gasp], the
        // language tags) and matches them greedily, so "he [sigh]ed" would
        // make the model sigh. The language tag added next is the one bracket
        // that belongs.
        t = t.replacingOccurrences(of: "[", with: " ")
        t = t.replacingOccurrences(of: "]", with: " ")
        t = "[\(lang)]" + t
        t = t.replacingOccurrences(of: " ", with: "[SPACE]")

        var ids: [Int] = []
        let ns = t as NSString
        var cursor = 0
        let matches = addedRegex?.matches(
            in: t, range: NSRange(location: 0, length: ns.length)) ?? []
        for m in matches {
            if m.range.location > cursor {
                encodeSegment(
                    ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor)),
                    into: &ids)
            }
            let content = ns.substring(with: m.range)
            if let id = addedTokens.first(where: { $0.content == content })?.id {
                ids.append(id)
            }
            cursor = m.range.location + m.range.length
        }
        if cursor < ns.length {
            encodeSegment(ns.substring(from: cursor), into: &ids)
        }
        return ids
    }

    private func encodeSegment(_ segment: String, into ids: inout [Int]) {
        let ns = segment as NSString
        for m in preTokRegex.matches(in: segment, range: NSRange(location: 0, length: ns.length)) {
            bpe(ns.substring(with: m.range), into: &ids)
        }
    }

    private func bpe(_ word: String, into ids: inout [Int]) {
        var symbols: [String] = word.unicodeScalars.map { String($0) }
        guard !symbols.isEmpty else { return }
        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0..<(symbols.count - 1) {
                if let r = mergeRank[symbols[i] + "\u{0}" + symbols[i + 1]], r < bestRank {
                    bestRank = r
                    bestIdx = i
                }
            }
            if bestIdx < 0 { break }
            symbols[bestIdx] += symbols[bestIdx + 1]
            symbols.remove(at: bestIdx + 1)
        }
        for s in symbols {
            ids.append(vocab[s] ?? unkId)  // fuse_unk=false: each unknown scalar -> its own UNK
        }
    }
}
