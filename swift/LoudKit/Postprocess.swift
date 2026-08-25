/// Python reference: `loudkit/frontend/postprocess.py`.
import Foundation

/// Deciding where a generated chunk actually ended.
///
/// Mirrors `loudkit.postprocess`. This is a **detector**, not a filter: it reads
/// the speech tokens a chunk produced, answers one question — where did the
/// sentence really stop? — and returns a verdict. It never touches a sample of
/// audio.
///
/// The artifact it removes is generated, not spectral. The decoder is
/// free-running, and silence tokens are exempt from both the repetition penalty
/// and the `min_p` cutoff (penalising silence measurably removes pauses), so
/// once the sentence is over those tokens keep probability mass indefinitely.
/// The decoder free-runs silence, and any step where a non-silence token
/// survives the cutoff becomes a hallucinated word — heard as "it finished, then
/// a long gap, then one random word".
///
/// Every constant came from a device trace or a regression, and every rule is
/// pinned by `tests/data/conformance/postprocess.json`, which all five ports
/// run. Provenance is in `docs/reference/postprocess.md`.
public enum Postprocess {

    /// What the engine does with a verdict.
    ///
    /// `trim` applies the cut, which changes the audio and therefore travels in
    /// the fingerprint like every other audible decision. `report` runs the
    /// detectors and attaches the verdict without acting on it. `off` skips
    /// them entirely.
    public enum Mode: String, Sendable, Equatable, CaseIterable {
        case off
        case report
        case trim
    }

    /// Which rule fired. `clean` means none did.
    public enum Reason: String, Sendable, Equatable {
        case clean
        case dropout
        case repetition
        case silenceTail = "silence_tail"
        case terminalEcho = "terminal_echo"
        case desperation
        case endedTail = "ended_tail"
    }

    /// What the detectors concluded about one chunk.
    public struct Inspection: Sendable, Equatable {
        /// How many leading tokens survive — equal to the input length when
        /// nothing fired, so a caller can always slice by it without branching.
        public var keep: Int
        public var reason: Reason
        /// The row is impossibly long for its text and no anchor agreed where
        /// to cut. Not an error and not a cut: a report. Shipping such a row
        /// silently is how the artifact reached listeners in the first place.
        public var suspect: Bool

        /// Whether anything was removed.
        public var cut: Bool { reason != .clean }
    }

    /// Everything the detectors need to know about one generated chunk.
    public struct Request: Sendable {
        /// The denominator of every ratio rule.
        public var textTokenCount: Int
        /// The EOS floor this row was generated under.
        public var minTokens: Int
        /// Step at which the stop token was most probable, or negative if it
        /// was never observed.
        public var eosPeakAt: Int
        public var eosPeakProb: Double
        /// Whether generation stopped at the stop token rather than a cap.
        public var ended: Bool
        /// Whether this chunk ends the passage. A continuation chunk has no
        /// sentence end, so its stop peak means nothing.
        public var isTerminal: Bool
        /// Whether generation was stopped by the length ceiling.
        public var hitCeiling: Bool

        public init(
            textTokenCount: Int, minTokens: Int, eosPeakAt: Int, eosPeakProb: Double,
            ended: Bool, isTerminal: Bool, hitCeiling: Bool
        ) {
            self.textTokenCount = textTokenCount
            self.minTokens = minTokens
            self.eosPeakAt = eosPeakAt
            self.eosPeakProb = eosPeakProb
            self.ended = ended
            self.isTerminal = isTerminal
            self.hitCeiling = hitCeiling
        }
    }
}

/// The detector constants. Algorithm layer: a port that uses a different
/// number produces different audio, so these are hashed into the fingerprint
/// rather than left as static constants.
public struct PostprocessConfig: Sendable, Equatable {
    public var mode: Postprocess.Mode = .trim

    /// Hard stop for generation, as a multiple of the text-token count.
    ///
    /// Device trace of the showcase render: `t3.overrun gen=92 ceiling=92
    /// bestEOS=74@0.003 floor=31` — ~26 text tokens stopped only because it hit
    /// the ceiling, mid-sentence, already at 3.5 speech tokens per text token.
    /// NOT the chunker's 2.6: there, guessing high only wastes window; here,
    /// guessing low cuts a sentence off.
    public var ceilingSpeechPerTextToken: Double = 4.0
    /// Carries the very short texts, where a ratio alone is unsafe (1.6 s).
    public var ceilingSlackTokens: Int = 40

    /// Share of a tail that must be silence before it counts as one.
    public var trailingFillerThreshold: Double = 0.7
    /// An unbroken silence run marking a structural boundary (~0.5 s at 25 Hz).
    ///
    /// A hallucinated word sits *behind* such a seam; under the share test
    /// alone its burst lowers the silence ratio below threshold, so the
    /// ugliest tails are exactly the ones the rescue refuses to cut.
    public var trailingSilenceRunTokens: Int = 12
    /// Top of the stop-peak acceptance band in ``Postprocess/desperationCut``,
    /// as a multiple of the text-token count.
    ///
    /// Measured reads run 1.75–2.35 speech tokens per text token, so the band
    /// reaches past every legitimate ending while staying well under the 4.5x
    /// garbage threshold.
    public var desperationBandRatio: Double = 2.6
    /// Slack above the proportional band, in speech tokens (~0.5 s). Carries
    /// the short texts, where the ratio alone would close the band on endings
    /// a legitimate read had already reached.
    public var desperationBandFloor: Int = 12
    /// How confident the best stop must be before the share/run test is
    /// consulted at all. EOS-defence bench, variant B.
    public var fillerMinEosProbability: Double = 0.05
    /// How much speech may follow a seam and still be a hallucinated word
    /// rather than a continuing clause (~0.4 s).
    ///
    /// Deliberately separate from `endedTailWordMax` despite holding the same
    /// number: they govern different rows, so
    /// loosening the trim on terminal chunks must not silently loosen this.
    public var fillerMaxSpeechAfterRun: Int = 10

    /// Past this ratio the row certainly contains garbage, whatever its stop
    /// confidence said.
    ///
    /// "It was as he expected." — 14 text tokens — came back as 96 speech
    /// tokens of sentence-then-dense-babble, with the stop peak at the right
    /// *place* (45) but confidence 0.000, so every probability-gated rescue
    /// refused. Real speech runs 1.75–2.35 speech tokens per text token.
    public var desperationSpeechPerTextToken: Double = 4.5
    /// Tiny texts are exempt: fixed overheads give a clean "No!" a ratio of 6+
    /// by itself.
    public var desperationMinTextTokens: Int = 10

    /// Silence before a blip that counts as stranding it (~0.24 s).
    public var endedTailSilenceRun: Int = 6
    /// <= 80 ms of "speech" is a click, not a word.
    public var endedTailBlipMax: Int = 2
    /// A stray word behind a full seam on a *terminal* chunk is cut with it.
    /// Continuation chunks keep their tails — their pauses are the sentence's
    /// rhythm and their "end" is not an end.
    public var endedTailWordMax: Int = 10
    /// Pause left in place after trimming (~0.2 s).
    public var endedTailKeep: Int = 5

    /// The ordinary terminal echo: a confident stop, late, with at most ~1.2 s
    /// after it. The position rule keeps a real clause pause from reading as an
    /// ending.
    public var echoStrongEosProbability: Double = 0.1
    public var echoStrongMaxTail: Int = 30
    public var echoStrongMinPositionPct: Int = 68

    /// The narrow second path, for one regression ("...but a brigand. Pass.
    /// Four.": `gen=124/124, bestEOS=109@0.004`). Confidence this weak is
    /// accepted only with every corroborator at once.
    public var echoWeakEosProbability: Double = 0.003
    public var echoWeakMaxTail: Int = 16
    public var echoWeakMinPositionPct: Int = 85

    /// How many re-rolls a condemned window may get before shipping as is.
    /// Only dropout and suspect retry; each attempt draws a derived seed, so
    /// the ladder is a pure function of the caller's seed.
    public var retryMaxAttempts: Int = 2

    /// How far a chunk's pace may drift from the passage's median before it
    /// is flagged (multiplicative, both directions).
    public var pacingTolerance: Double = 1.6

    /// Longest cycle, in tokens (~0.5 s), that counts as a stuck decoder. Above
    /// it a repeated block is a phrase, and a repeated phrase is rhetoric.
    public var repetitionMaxPeriod: Int = 12

    /// How many consecutive identical cycles a loop needs. Two is a repeated
    /// phrase; three is necessary, not sufficient.
    public var repetitionMinCycles: Int = 3

    /// How many tokens the repeating region must cover (~1.0 s). The constant
    /// that does the work: measured across 27 renders in nine languages, a
    /// healthy row repeats for at most 10 tokens. Cycle count alone fired on
    /// 22 of those 27.
    public var repetitionMinSpan: Int = 24

    /// Early truncation: the row is too short to be the text it was asked for.
    /// Reported, never cut — there is nothing to cut, and it is the most damaging
    /// failure in the set because a listener cannot hear that content is absent.
    /// The 25-token floor is the published criterion for a catastrophic
    /// neural-codec TTS failure; the proportional test exempts a genuinely short
    /// line, since the shortest healthy reads measured run 35 tokens.
    public var dropoutMinTokens: Int = 25

    public init() {}
}

extension Postprocess {

    /// Speech tokens at which the decoder is stopped whatever it thinks.
    ///
    /// Applied *during* generation: the tokens past it cost real time on a
    /// device and are certain to be discarded. It only ever stops a row that
    /// was going to run away — a model that stops on its own never reaches it.
    public static func ceiling(
        forTextTokens count: Int, config: PostprocessConfig, window: Int
    ) -> Int {
        // Clamped to `window`, not `window + 15`. `Windowing` refuses
        // anything past `maxSpeechTokens`, so those fifteen tokens could never
        // be rendered: a row allowed to reach 270 was stopped at 270 and then
        // rejected at 255 — real time on a device for tokens certain to be
        // discarded. Changed in all five together with the `funnel-2` bump,
        // because a ceiling change moves audio and has to be visible in the
        // fingerprint.
        let proportional = Int(Double(count) * config.ceilingSpeechPerTextToken)
        return min(window, proportional + config.ceilingSlackTokens)
    }

    private static func silenceFlags(_ tokens: ArraySlice<Int>, _ silence: Set<Int>) -> [Bool] {
        tokens.map { silence.contains($0) }
    }

    /// Whether what follows `index` is a trailing tail rather than more
    /// sentence.
    ///
    /// The overrun rescue cuts back to where the model came closest to
    /// stopping, and that peak is a hint, not a verdict. Trusting it alone
    /// truncated whole sentences: a voice reading a language its tag does not
    /// match may never commit to stopping, so its best moment of hesitation
    /// lands a third of the way in.
    ///
    /// So the peak is corroborated by *what it proposes to discard* — either
    /// the tail is mostly silence by share, or it holds a long unbroken run with
    /// only a stray word behind it. Without that second half, a rhetorical pause
    /// mid-tail (25 silent tokens, then 80 of speech) matched the run rule and
    /// the rescue cut the rest of the sentence off.
/// Indices of chunks whose pace drifts past the tolerance from the median.
    ///
    /// Long-form drift: per-chunk pace (speech tokens / text tokens) against
    /// the passage's own median, report-only. The median rather than the mean,
    /// so one broken chunk cannot drag the baseline toward itself and hide.
    public static func pacingOutliers(
        _ ratios: [Double], config: PostprocessConfig
    ) -> [Int] {
        guard ratios.count >= 3 else {
            // One chunk has no neighbours; two cannot say which drifted.
            return []
        }
        let ordered = ratios.sorted()
        let mid = ordered.count / 2
        let median =
            ordered.count % 2 == 0 ? (ordered[mid - 1] + ordered[mid]) / 2 : ordered[mid]
        guard median > 0 else { return [] }
        return ratios.enumerated().compactMap { i, ratio in
            (ratio > median * config.pacingTolerance
                || ratio < median / config.pacingTolerance) ? i : nil
        }
    }

    /// Whether the row is too short to be the text it was asked for.
    ///
    /// Two conditions, both required. The absolute floor catches a row that
    /// stopped almost immediately whatever the text was; the proportional one
    /// keeps a genuinely short line exempt, because a read producing less than
    /// one speech token per text token has not said the text under any
    /// pronunciation.
    public static func isDropout(
        _ tokenCount: Int, _ textTokenCount: Int, config: PostprocessConfig
    ) -> Bool {
        if tokenCount >= config.dropoutMinTokens { return false }
        return textTokenCount > 0 && tokenCount < textTokenCount
    }

    /// Where a stuck decoder started looping, or `nil`.
    ///
    /// The failure the tail rules cannot see, because it happens *inside* the
    /// row. The mechanism is the one behind the trailing hallucinated word — the
    /// model's own output becomes its context — but it strikes mid-sequence, so
    /// no rule that reads the end can find it.
    ///
    /// Deliberately hard to trigger, because it is the only rule here that cuts
    /// mid-sequence: a short cycle, repeated many times, matched exactly. A
    /// decoder that has genuinely locked up emits the same tokens rather than
    /// similar ones, and a fuzzy match on a signal this destructive would
    /// truncate real speech.
    ///
    /// A cycle that is entirely silence is never a loop — silence repeating is
    /// what silence is, and the tail rules already judge pauses against where
    /// they sit.
    ///
    /// Returns one full cycle past the loop's start: the first instance is
    /// plausibly the word the sentence wanted.
    public static func repetitionCut(
        _ tokens: [Int], silence: Set<Int>, config: PostprocessConfig
    ) -> Int? {
        let n = tokens.count
        guard n >= config.repetitionMinSpan else { return nil }
        let quiet = tokens.map { silence.contains($0) }

        // Earliest loop wins: a row that locks up twice locked up first at the
        // first one, and everything after it is already inside the failure.
        var best: Int?
        let longestPeriod = min(config.repetitionMaxPeriod, n / config.repetitionMinCycles)
        for period in 1...max(longestPeriod, 1) where period <= longestPeriod {
            var start = 0
            while start + period * config.repetitionMinCycles <= n {
                var cycles = 1
                var at = start + period
                while at + period <= n,
                      Array(tokens[at..<(at + period)]) == Array(tokens[start..<(start + period)]) {
                    cycles += 1
                    at += period
                }
                let allQuiet = quiet[start..<(start + period)].allSatisfy { $0 }
                if cycles >= config.repetitionMinCycles,
                   cycles * period >= config.repetitionMinSpan, !allQuiet {
                    let candidate = start + period
                    if best == nil || candidate < best! { best = candidate }
                    break
                }
                start += 1
            }
        }
        return best
    }

    public static func isTrailingFiller(
        _ tokens: [Int], from index: Int, silence: Set<Int>, config: PostprocessConfig
    ) -> Bool {
        guard index >= 0, index < tokens.count else { return false }
        let flags = silenceFlags(tokens[index...], silence)

        var silent = 0
        var run = 0
        var longestRun = 0
        for isSilent in flags {
            if isSilent {
                silent += 1
                run += 1
                longestRun = max(longestRun, run)
            } else {
                run = 0
            }
        }
        if Double(silent) / Double(flags.count) >= config.trailingFillerThreshold { return true }
        guard longestRun >= config.trailingSilenceRunTokens else { return false }

        // Collect qualifying runs, then require every gap of speech between
        // them — and after the last — to be a stray word or less.
        // [seam][real sentence][seam][word] fails: the tokens between the two
        // seams are the sentence itself, not filler trailing the first
        // boundary.
        var runs: [(Int, Int)] = []
        var scanRun = 0
        var scanStart = 0
        for (i, isSilent) in flags.enumerated() {
            if isSilent {
                if scanRun == 0 { scanStart = i }
                scanRun += 1
                if scanRun == config.trailingSilenceRunTokens {
                    runs.append((scanStart, i + 1))
                }
            } else {
                scanRun = 0
            }
        }
        if runs.isEmpty { return false }
        if runs[0].0 > config.fillerMaxSpeechAfterRun { return false }
        let last = runs[runs.count - 1]
        if flags.count - last.1 > config.fillerMaxSpeechAfterRun { return false }
        for i in 1..<runs.count where runs[i].0 - runs[i - 1].1 > config.fillerMaxSpeechAfterRun {
            return false
        }
        return true
    }

    /// The rescue for rows whose *length* is the evidence.
    ///
    /// Past the ratio the row is certainly broken, so the question is where to
    /// cut, not whether: at the first long silence run that starts past the
    /// floor (a run straddling the floor belongs to the sentence, which is why
    /// the run's *start* is tested), else at the stop peak if it sits in a band
    /// a real read could have ended in. The band protects the
    /// mislabeled-language case (92 generated / 26 text = 3.5x), whose kind of
    /// row must never be cut at a peak landing a third of the way in.
    ///
    /// `peakAllowed` is false for a continuation chunk: it has no sentence end,
    /// so its stop peak means nothing.
    public static func desperationCut(
        _ tokens: [Int], textTokenCount: Int, minTokens: Int, eosPeakAt: Int,
        silence: Set<Int>, config: PostprocessConfig, peakAllowed: Bool = true
    ) -> Int? {
        guard textTokenCount >= config.desperationMinTextTokens else { return nil }
        guard Double(tokens.count)
            >= Double(textTokenCount) * config.desperationSpeechPerTextToken
        else { return nil }

        let earliest = max(minTokens, 10)
        let flags = silenceFlags(tokens[...], silence)

        var runStart = -1
        var run = 0
        for (i, isSilent) in flags.enumerated() {
            if isSilent {
                if run == 0 { runStart = i }
                run += 1
                if run >= config.trailingSilenceRunTokens && runStart >= earliest {
                    return runStart
                }
            } else {
                run = 0
            }
        }

        // No seam — the babble is dense; fall back to the model's own best
        // stop, if it lands where a real read could have ended.
        guard peakAllowed else { return nil }
        let bandTop =
            Int(config.desperationBandRatio * Double(textTokenCount))
            + config.desperationBandFloor
        if eosPeakAt >= earliest, eosPeakAt <= bandTop, eosPeakAt < tokens.count {
            return eosPeakAt
        }
        return nil
    }

    /// Dead air past the sentence on a row that stopped when it meant to.
    ///
    /// Walked backward as `[sentence][r1 silence][burst][r2 silence]`. Three
    /// shapes come off: a bare silence run half a second long; a silence run
    /// with a 1–2 token blip right before the stop (the device specimen ended
    /// `.......#`); and, on a *terminal* chunk only, a stray word behind a full
    /// seam.
    public static func endedTailTrim(
        _ tokens: [Int], silence: Set<Int>, config: PostprocessConfig, isTerminal: Bool = false
    ) -> Int? {
        let flags = silenceFlags(tokens[...], silence)
        var j = tokens.count - 1

        var r2 = 0
        while j >= 0, flags[j] {
            r2 += 1
            j -= 1
        }
        guard j >= 0 else { return nil }
        if r2 >= config.trailingSilenceRunTokens {
            let n = j + 1 + min(r2, config.endedTailKeep)
            return n < tokens.count ? n : nil
        }

        var burst = 0
        while j >= 0, !flags[j] {
            burst += 1
            j -= 1
        }
        var r1 = 0
        while j >= 0, flags[j] {
            r1 += 1
            j -= 1
        }
        guard j >= 0 else { return nil }  // the "burst" was the sentence

        let strandedClick = burst <= config.endedTailBlipMax && r1 >= config.endedTailSilenceRun
        let strandedWord = isTerminal && burst <= config.endedTailWordMax
            && r1 >= config.trailingSilenceRunTokens
        guard strandedClick || strandedWord else { return nil }
        let n = j + 1 + min(r1, config.endedTailKeep)
        return n < tokens.count ? n : nil
    }

    /// A terminal chunk that ended correctly and then free-ran an extra word.
    ///
    /// There is no silence seam here, so ``isTrailingFiller(_:from:silence:config:)``
    /// has nothing to anchor on. Instead the earlier stop candidate must be
    /// strong, late and followed by a short tail. The second acceptance path is
    /// narrower and exists for one regression where the model never sampled a
    /// stop token but its best — very weak — stop was 15 tokens before the hard
    /// ceiling.
    public static func terminalEchoCut(
        tokenCount: Int, eosPeakAt: Int, eosPeakProb: Double, minTokens: Int,
        isTerminal: Bool, hitCeiling: Bool, config: PostprocessConfig
    ) -> Int? {
        guard isTerminal, eosPeakAt > max(minTokens, 10), eosPeakAt < tokenCount else {
            return nil
        }
        let tail = tokenCount - eosPeakAt
        let strongPeak = eosPeakProb >= config.echoStrongEosProbability
            && tail <= config.echoStrongMaxTail
            && eosPeakAt * 100 >= tokenCount * config.echoStrongMinPositionPct
        let weakLatePeakAtCeiling = hitCeiling
            && eosPeakProb >= config.echoWeakEosProbability
            && tail <= config.echoWeakMaxTail
            && eosPeakAt * 100 >= tokenCount * config.echoWeakMinPositionPct
        return (strongPeak || weakLatePeakAtCeiling) ? eosPeakAt : nil
    }

    /// Run every detector in precedence order and return one verdict.
    ///
    /// The shipped reader grew five entry points, one per field bug, and left
    /// the ordering to each call site. Here they are one resolver with the
    /// precedence written down, because an order that lives in a caller is an
    /// order the next caller gets wrong.
    ///
    /// Peak-anchored rescues first, then the length-anchored one — it is the
    /// bluntest, and it applies to *ended* rows too, because a model that
    /// babbles past its sentence and only then samples a stop token has
    /// forfeited the trust that stopping implies. The ended-tail trim runs only
    /// when nothing above fired.
    public static func inspect(
        _ tokens: [Int], request: Request, silence: Set<Int>, config: PostprocessConfig
    ) -> Inspection {
        if config.mode == .off || tokens.isEmpty {
            return Inspection(keep: tokens.count, reason: .clean, suspect: false)
        }

        var cut: Int?
        var reason: Reason = .clean

        // Terminal chunks only, like its three siblings. `isTerminal` means a
        // continuation chunk's stop peak is meaningless and its pauses are
        // rhythm rather than dead air — and this rule reads exactly those two
        // signals, so it was trimming mid-passage chunks on evidence the
        // contract says is not evidence. Changed in all five implementations
        // together; postprocess is a bit-parity surface.
        let fillerCut = request.isTerminal
            && !request.ended
            && request.eosPeakProb > config.fillerMinEosProbability
            && request.eosPeakAt > max(request.minTokens, 10)
            && request.eosPeakAt < tokens.count
            && isTrailingFiller(
                tokens, from: request.eosPeakAt, silence: silence, config: config)

        // Early truncation first: nothing below can help a row that is already
        // too short, and the verdict is "incomplete", not "wrongly ended".
        if isDropout(tokens.count, request.textTokenCount, config: config) {
            return Inspection(keep: tokens.count, reason: .dropout, suspect: true)
        }

        // Then repetition, because it is the only rule that knows *exactly*
        // where the failure began. Every other anchor here is inferred from a
        // signal that might mean something else; a repeated cycle is not.
        if let looped = repetitionCut(tokens, silence: silence, config: config) {
            cut = looped
            reason = .repetition
        } else if fillerCut {
            cut = request.eosPeakAt
            reason = .silenceTail
        } else if let echo = terminalEchoCut(
            tokenCount: tokens.count, eosPeakAt: request.eosPeakAt,
            eosPeakProb: request.eosPeakProb, minTokens: request.minTokens,
            isTerminal: request.isTerminal, hitCeiling: request.hitCeiling, config: config)
        {
            cut = echo
            reason = .terminalEcho
        } else if let desperate = desperationCut(
            tokens, textTokenCount: request.textTokenCount, minTokens: request.minTokens,
            eosPeakAt: request.eosPeakAt, silence: silence, config: config,
            peakAllowed: request.isTerminal)
        {
            cut = desperate
            reason = .desperation
        }

        if cut == nil, request.ended,
           let trimmed = endedTailTrim(
            tokens, silence: silence, config: config, isTerminal: request.isTerminal)
        {
            cut = trimmed
            reason = .endedTail
        }

        // A condemned row that dodged every token anchor. Reported, never cut:
        // no rule could say where, and cutting at a guess is how the rescue
        // truncated whole sentences before the corroboration rules were added.
        let suspect = cut == nil
            && request.textTokenCount >= config.desperationMinTextTokens
            && Double(tokens.count)
                >= Double(request.textTokenCount) * config.desperationSpeechPerTextToken
        return Inspection(keep: cut ?? tokens.count, reason: reason, suspect: suspect)
    }
}
