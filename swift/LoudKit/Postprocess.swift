/// Python reference: `loudkit/postprocess.py`.
import Foundation

/// Deciding where a generated chunk actually ended.
///
/// Mirrors `loudkit.postprocess`. This is a **detector**, not a filter: it reads
/// the speech tokens a chunk produced, answers one question (where did the
/// sentence really stop?) and returns a verdict. It never touches a sample of
/// audio.
///
/// The artifact it removes is generated, not spectral. The decoder is
/// free-running, and silence tokens are exempt from the `min_p` cutoff (a
/// pause token is the only way to pause, and a filter that removes it removes
/// prosody), so once the sentence is over those tokens keep probability mass
/// indefinitely. The decoder free-runs silence, and any step where a
/// non-silence token survives the cutoff becomes a hallucinated word, heard
/// as "it finished, then a long gap, then one random word". The repetition
/// penalty applies to silence as well, because exempting it there too makes a
/// silence run absorbing mid-row: see
/// ``Postprocess/isStalled(_:hitCeiling:silence:config:)`` for the failure that
/// leaves behind and the sampler for the measurement.
///
/// Every constant came from a device trace or a regression, and every rule is
/// pinned by `tests/data/conformance/postprocess.json`, which all five ports
/// run. Provenance is in `docs/design/postprocess.md`, and the measurements
/// behind each detector, the specimens included, are in
/// `docs/design/postprocess-detectors.md`.
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

    /// What a qualifying loop the decoder *resumed from* receives in
    /// ``inspect(_:request:silence:config:)``. `condemn` is the shipping
    /// default; `cut` names the pre-amendment law. See
    /// ``PostprocessConfig/repetitionResume``.
    public enum RepetitionResume: String, Sendable, Equatable, CaseIterable {
        case condemn
        case cut
    }

    /// Which silence family the all-silence-cycle exemption in
    /// ``repetitionCut(_:silence:config:)`` reads. `acoustic` is the shipping
    /// default; `sampling` names the pre-amendment law. See
    /// ``PostprocessConfig/repetitionSilence``.
    public enum RepetitionSilence: String, Sendable, Equatable, CaseIterable {
        case acoustic
        case sampling
    }

    /// Which rule fired. `clean` means none did.
    public enum Reason: String, Sendable, Equatable {
        case clean
        case dropout
        case stall
        case repetition
        case silenceTail = "silence_tail"
        case terminalEcho = "terminal_echo"
        case desperation
        case endedTail = "ended_tail"
    }

    /// What the detectors concluded about one chunk.
    public struct Inspection: Sendable, Equatable {
        /// How many leading tokens survive, equal to the input length when
        /// nothing fired, so a caller can always slice by it without branching.
        public var keep: Int
        /// Which rule fired, or `clean`.
        public var reason: Reason
        /// The row is certainly wrong in a way no cut can fix. Set with
        /// `dropout` (content missing), with `stall` (the row is dead air
        /// where speech should be), with `repetition` on a loop the decoder
        /// resumed from (the cut would delete what it came back to say, so
        /// the row is handed back whole), with a starved `desperation` cut (a
        /// cap-hit trim that keeps less than any full read of its text, here
        /// `keep` still holds the cut, as the fallback if every retry is also
        /// condemned), and on a row impossibly long for its text that dodged
        /// every token anchor. Not an error and not a cut: a report, and the
        /// engine's signal to retry. Shipping such a row silently is how the
        /// artifact reached listeners in the first place.
        public var suspect: Bool

        /// Whether a rule fired, which is not the same as whether tokens came off.
        ///
        /// `dropout` and `stall` condemn a row without cutting it, `keep` still holds
        /// the whole input, and both make this true. A caller reading it as "tokens
        /// were removed" and slicing on that gets the right answer anyway, because
        /// `keep` is the length; a caller counting cuts with it over-counts by every
        /// condemned row. "Whether anything was removed" is what this said, and it was
        /// wrong for the two verdicts that exist precisely because nothing can be
        /// removed.
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
        /// Probability of the stop token at that step.
        public var eosPeakProb: Double
        /// Whether generation stopped at the stop token rather than a cap.
        public var ended: Bool
        /// Whether this chunk ends the passage. A continuation chunk has no
        /// sentence end, so its stop peak means nothing.
        public var isTerminal: Bool
        /// Whether generation was stopped by the length ceiling.
        public var hitCeiling: Bool

        /// Every field is required: the detectors read all seven, and a
        /// default here would let a caller condemn a row on values it never set.
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
    /// The one knob a caller changes: whether the detectors cut, only report,
    /// or do not run.
    public var mode: Postprocess.Mode = .trim

    /// Hard stop for generation, as a multiple of the text-token count.
    ///
    /// Device trace of the showcase render: `t3.overrun gen=92 ceiling=92
    /// bestEOS=74@0.003 floor=31`, ~26 text tokens stopped only because it hit
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
    /// "It was as he expected.", 14 text tokens, came back as 96 speech
    /// tokens of sentence-then-dense-babble, with the stop peak at the right
    /// *place* (45) but confidence 0.000, so every probability-gated rescue
    /// refused. Real speech runs 1.75–2.35 speech tokens per text token.
    public var desperationSpeechPerTextToken: Double = 4.5
    /// Tiny texts are exempt: fixed overheads give a clean "No!" a ratio of 6+
    /// by itself.
    public var desperationMinTextTokens: Int = 10
    /// A cap-hit row whose desperation cut keeps fewer speech tokens than this
    /// many per text token is condemned into the retry ladder instead of
    /// shipping the trim.
    ///
    /// The defect it closes: a cap-hit row whose seam cut kept a short span
    /// of audio that renders near-silent through ids outside both manifest
    /// censuses, so no set-membership rule can see it. The trim shipped a
    /// mute chunk and the caller was never told to retry.
    ///
    /// Calibrated on every cap-hit desperation rescue in the interior-stall
    /// and acceptance batteries (nine rows, four voices, four languages):
    /// every keep at or below 1.57 per text token was mute or missing much of
    /// its text, and every keep at or above 1.85 carried real speech. 1.7
    /// sits inside that gap, and deliberately under 1.75, the floor of the
    /// measured healthy band of speech tokens per text token, so a complete
    /// read is never condemned. Cap-hit rows only, a row that ended on its
    /// own corroborated its trim with a stop token. Zero disables the
    /// trigger. The specimen and the calibration rows are in
    /// `docs/design/postprocess-detectors.md`.
    public var desperationMinKeepPerTextToken: Double = 1.7

    /// Silence before a blip that counts as stranding it (~0.24 s).
    public var endedTailSilenceRun: Int = 6
    /// <= 80 ms of "speech" is a click, not a word.
    public var endedTailBlipMax: Int = 2
    /// A stray word behind a full seam on a *terminal* chunk is cut with it.
    /// Continuation chunks keep their tails, their pauses are the sentence's
    /// rhythm and their "end" is not an end.
    public var endedTailWordMax: Int = 10
    /// Pause left in place after trimming (~0.2 s).
    public var endedTailKeep: Int = 5

    /// The ordinary terminal echo: a confident stop, late, with at most ~1.2 s
    /// after it. The position rule keeps a real clause pause from reading as an
    /// ending.
    public var echoStrongEosProbability: Double = 0.1
    /// Longest tail this path will cut, in tokens.
    public var echoStrongMaxTail: Int = 30
    /// Earliest position, as a percentage of the row, that may read as an
    /// ending.
    public var echoStrongMinPositionPct: Int = 68

    /// The narrow second path, for one regression ("...but a brigand. Pass.
    /// Four.": `gen=124/124, bestEOS=109@0.004`). Confidence this weak is
    /// accepted only with every corroborator at once.
    public var echoWeakEosProbability: Double = 0.003
    /// Longest tail this path will cut, in tokens.
    public var echoWeakMaxTail: Int = 16
    /// Earliest position, as a percentage of the row, that may read as an
    /// ending.
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
    /// that does the work, and the one this layer was calibrated on: across
    /// the 27-render set (nine languages, three length classes, one voice held
    /// constant) a healthy row repeated for at most 10 tokens and the single
    /// runaway for 44, so 24 sits between them with 2.4x margin over the
    /// healthy maximum. Cycle count alone fired on 22 of those 27; with the
    /// span it fires on none. The set is described in
    /// `docs/design/postprocess.md`, which is also where the constants
    /// measured on other populations are kept apart from this one.
    public var repetitionMinSpan: Int = 24

    /// What a qualifying loop the decoder *resumed from* receives.
    ///
    /// The guard that holds without a census. A genuine lock-up is a tail
    /// pathology: the model's own output is its context, the state is
    /// absorbing, and the repeating region runs to the end of the row. Every
    /// fire in the conformance fixture resumes by zero tokens, and a ceiling
    /// can truncate at most one incomplete copy, `period - 1` tokens, at
    /// most 11. A qualifying repetition followed by a full period or more of
    /// other content is therefore a different event: a decoder that resumed
    /// was never locked, and on a checkpoint without render censuses the
    /// thing it resumed from is a pause parked on a silent-rendering id the
    /// sampler list cannot name.
    ///
    /// `condemn` (the default): the row is reported whole (`keep` is the full
    /// row, verdict repetition, suspect) and routed into the retry
    /// ladder like `stall`. The cut is refused because the cut *is* the
    /// defect: on the measured specimen it deleted a pause together with the
    /// correctly-read speech behind it and reported the row fluent.
    /// `repetitionSilence` closes that on a manifest that carries the
    /// censuses; this field closes it on every checkpoint, including ids no
    /// census lists.
    ///
    /// Not a discard-fraction guard, though one was calibrated first: a
    /// fraction reads geometry, so a pause with less speech behind it slips
    /// under any cap and ships the deletion as clean. The largest resume a
    /// truncated genuine loop can produce is `period - 1`, so the law is
    /// `resume >= period`: integer-exact, no constant to tune, and a cut
    /// that survives it only ever removes a tail, like every other rule in
    /// the layer.
    ///
    /// `cut` names the pre-amendment law, for a checkpoint measured under
    /// it. The bare rule (``Postprocess/repetitionCut(_:silence:config:)``)
    /// reports the loop either way; this field decides what the resolver
    /// does with one that resumed.
    ///
    /// The specimen is in `docs/design/postprocess-detectors.md`, measured on
    /// the published census-less pack. ``repetitionSilence`` is measured on
    /// the same chunk rendered on a pack that does name its silent ids, so
    /// the two runs are not quite the same row.
    public var repetitionResume: Postprocess.RepetitionResume = .condemn

    /// Which silence family the all-silence-cycle exemption reads.
    ///
    /// `acoustic`: the union of the configured sampler silence ids and both
    /// render censuses (`silenceRenderIds`, `quietRenderIds`). A pause parked
    /// on *any* silent-rendering id is never mistaken for a decoder loop.
    /// Keyed to the sampler list alone the exemption cannot see a pause
    /// parked on ids that render true silence but sit outside that list: the
    /// run fires as a loop, and the cut keeps one cycle and deletes the pause
    /// together with the correctly-read speech behind it, verdict repetition,
    /// not suspect, no retry, audibly fluent. Most of the measured
    /// checkpoint's truly-silent ids sit outside the sampler list, so this
    /// was the rule's default behaviour on most real pauses, and the shape is
    /// inaudible content loss shipping as clean.
    ///
    /// `sampling`: the configured sampler list alone, the pre-amendment
    /// law, nameable so a checkpoint measured under it can declare what it
    /// measured. A checkpoint without censuses gets this behaviour under
    /// either value, since the union degenerates to the sampler list.
    ///
    /// This family feeds the loop exemption only. The tail rules
    /// (`silence_tail`, `ended_tail`, the filler and desperation seams) stay
    /// keyed to the sampler list they were calibrated against; see
    /// `docs/design/postprocess.md` for the two-lists decision. The specimen,
    /// the id census it turned on and the measured prevalence are in
    /// `docs/design/postprocess-detectors.md`.
    public var repetitionSilence: Postprocess.RepetitionSilence = .acoustic

    /// A non-tail dead-air run this long condemns the row (~1.0 s at 25 Hz).
    ///
    /// The run is measured two-class, and the two classes are essential: only
    /// true-silence ids (`silenceRenderIds`) count toward this threshold, but
    /// the run *continues* across quiet-family ids (`quietRenderIds`),
    /// breath and decay tokens that render inaudible in context. Single-set
    /// counting was measured broken: one breath token in the middle of real
    /// dead air split a 47-token run into two short ones and the rule missed
    /// it.
    ///
    /// Calibrated across all ten shipping languages (120 passages per arm):
    /// healthy interior runs top out at 13–19 tokens and healthy leading runs
    /// at 11, so 25 is outside anything ordinary prose produced anywhere
    /// while sitting under every measured stall. 20 also clears the healthy
    /// maxima; 25 is the shipped margin.
    public var stallRunTokens: Int = 25

    /// Token ids that render as true digital silence.
    ///
    /// A property of the checkpoint, measured by rendering (per-id median
    /// energy below -80 dBFS across two independent censuses), and therefore
    /// supplied by the manifest, top level, beside `silence_token_ids`,
    /// precisely so the next backend cannot re-guess it. Empty means the
    /// checkpoint predates the census; the stall rule then runs its run
    /// trigger only, keyed to the configured `silence_token_ids`, degraded
    /// (only 8 of that list's 31 ids actually render silent, so the whole-row
    /// and majority triggers cannot be trusted with it) but safe.
    ///
    /// NOT a sampling exemption list. Widening the sampler's `min_p`
    /// exemption to exactly these ids was measured harmful (pause-time share
    /// doubles) and the repetition penalty applies to every token
    /// regardless. This list exists so the detectors read dead air where dead
    /// air actually is.
    public var silenceRenderIds: [Int] = []

    /// The contextually-quiet family: breath and decay ids.
    ///
    /// Measured by per-instance RMS attribution (at least 90% of instances
    /// quiet, 5+ sightings), minus the true-silence census. Dead-air runs
    /// continue across these ids but they never count toward the run gate, a
    /// breath inside dead air is still dead air, and a breath between words
    /// is not. Manifest-supplied like `silenceRenderIds`; empty when the
    /// checkpoint predates the census.
    public var quietRenderIds: [Int] = []

    /// Early truncation: the row is too short to be the text it was asked for.
    /// Reported, never cut: there is nothing to cut, and it is the most damaging
    /// failure in the set because a listener cannot hear that content is absent.
    /// The 25-token floor is the published criterion for a catastrophic
    /// neural-codec TTS failure; the proportional test exempts a genuinely short
    /// line, since the shortest healthy reads measured run 35 tokens.
    public var dropoutMinTokens: Int = 25

    public init() {}

    /// How many retry streams the seed ladder has room for: the gap between the
    /// engine's retry stream at 8 and its chunk streams at 16. Mirrors
    /// `RETRY_LADDER_HEADROOM`.
    static let retryLadderHeadroom = 8

    /// Every float on this struct, so a NaN cannot walk through the validator.
    private var finiteFields: [(String, Double)] {
        [("ceiling_speech_per_text_token", ceilingSpeechPerTextToken),
         ("trailing_filler_threshold", trailingFillerThreshold),
         ("filler_min_eos_probability", fillerMinEosProbability),
         ("desperation_band_ratio", desperationBandRatio),
         ("desperation_speech_per_text_token", desperationSpeechPerTextToken),
         ("desperation_min_keep_per_text_token", desperationMinKeepPerTextToken),
         ("echo_strong_eos_probability", echoStrongEosProbability),
         ("echo_weak_eos_probability", echoWeakEosProbability),
         ("pacing_tolerance", pacingTolerance)]
    }

    /// Refuse a preset the detectors cannot run, with `_validate_ranges`'s
    /// sentences.
    ///
    /// Three of these were traps rather than refusals: a NaN
    /// `ceiling_speech_per_text_token` reaches `Int(...)` in `ceiling` and
    /// kills the process, `repetition_min_cycles: 0` divides by zero in
    /// `repetitionCut`, and `stall_run_tokens: 0` makes every row with one
    /// silence token a stall.
    func validate() throws {  // swiftlint:disable:this cyclomatic_complexity
        // One branch per constant. Grouping them to satisfy a branch count
        // would read as if the groupings meant something, and they do not.
        for (name, value) in finiteFields where !value.isFinite {
            throw LoudKitError.manifest("\(name) must be a finite number: \(value)")
        }
        guard retryMaxAttempts >= 0, retryMaxAttempts < Self.retryLadderHeadroom else {
            throw LoudKitError.manifest(
                "retry_max_attempts must be in [0, \(Self.retryLadderHeadroom)): "
                + "\(retryMaxAttempts). Above that the ladder's derived seeds run "
                + "into the streams the chunk seeds use.")
        }
        guard repetitionMinCycles >= 2 else {
            // One cycle is not a repetition and two is the definition of one; a
            // threshold below two would cut every row that says a word twice.
            throw LoudKitError.manifest(
                "repetition_min_cycles must be at least 2: \(repetitionMinCycles)")
        }
        guard repetitionMaxPeriod >= 1 else {
            throw LoudKitError.manifest(
                "repetition_max_period must be positive: \(repetitionMaxPeriod)")
        }
        guard stallRunTokens >= 1 else {
            // At zero every row with a single silence token before speech is a
            // stall, and "condemned" stops meaning anything.
            throw LoudKitError.manifest("stall_run_tokens must be positive: \(stallRunTokens)")
        }
        guard repetitionMinSpan >= repetitionMinCycles else {
            // A span shorter than the cycle count is unreachable: the shortest
            // qualifying loop is min_cycles copies of a one-token cycle.
            throw LoudKitError.manifest(
                "repetition_min_span (\(repetitionMinSpan)) must be at least "
                + "repetition_min_cycles (\(repetitionMinCycles))")
        }
        guard ceilingSpeechPerTextToken > 0 else {
            throw LoudKitError.manifest(
                "ceiling_speech_per_text_token must be positive: \(ceilingSpeechPerTextToken)")
        }
        guard desperationSpeechPerTextToken > ceilingSpeechPerTextToken else {
            // Below the ceiling, the rule that means "certainly broken" fires on
            // rows the ceiling stopped correctly.
            throw LoudKitError.manifest(
                "desperation_speech_per_text_token (\(desperationSpeechPerTextToken)) must "
                + "exceed ceiling_speech_per_text_token (\(ceilingSpeechPerTextToken)): "
                + "below it, the rule that means 'certainly broken' fires on rows "
                + "the ceiling stopped correctly")
        }
        guard desperationMinKeepPerTextToken >= 0 else {
            throw LoudKitError.manifest(
                "desperation_min_keep_per_text_token must be >= 0: "
                + "\(desperationMinKeepPerTextToken)")
        }
        guard desperationMinKeepPerTextToken <= desperationBandRatio else {
            // The band top is where a real read could still have ended.
            throw LoudKitError.manifest(
                "desperation_min_keep_per_text_token (\(desperationMinKeepPerTextToken)) must "
                + "not exceed desperation_band_ratio (\(desperationBandRatio))")
        }
        guard trailingFillerThreshold > 0, trailingFillerThreshold <= 1 else {
            throw LoudKitError.manifest(
                "trailing_filler_threshold must be in (0, 1]: \(trailingFillerThreshold)")
        }
        guard fillerMinEosProbability >= 0, fillerMinEosProbability < 1 else {
            throw LoudKitError.manifest(
                "filler_min_eos_probability out of range: \(fillerMinEosProbability)")
        }
        let counts = [
            ("ceiling_slack_tokens", ceilingSlackTokens),
            ("trailing_silence_run_tokens", trailingSilenceRunTokens),
            // In the reference's order, which names these two among the
            // counts: a negative band floor closes the band the field exists
            // to open, and a negative dropout minimum leaves no row short
            // enough to be one.
            ("desperation_band_floor", desperationBandFloor),
            ("desperation_min_text_tokens", desperationMinTextTokens),
            ("ended_tail_silence_run", endedTailSilenceRun),
            ("ended_tail_blip_max", endedTailBlipMax),
            ("ended_tail_word_max", endedTailWordMax),
            ("filler_max_speech_after_run", fillerMaxSpeechAfterRun),
            ("ended_tail_keep", endedTailKeep),
            ("echo_strong_max_tail", echoStrongMaxTail),
            ("echo_weak_max_tail", echoWeakMaxTail),
            ("dropout_min_tokens", dropoutMinTokens)
        ]
        for (name, value) in counts where value < 0 {
            throw LoudKitError.manifest("\(name) must be >= 0: \(value)")
        }
        let percentages = [
            ("echo_strong_min_position_pct", echoStrongMinPositionPct),
            ("echo_weak_min_position_pct", echoWeakMinPositionPct)
        ]
        for (name, pct) in percentages where !(0...100).contains(pct) {
            throw LoudKitError.manifest("\(name) is a percentage: \(pct)")
        }
    }
}

extension Postprocess {

    /// Speech tokens at which the decoder is stopped whatever it thinks.
    ///
    /// Applied *during* generation: the tokens past it cost real time on a
    /// device and are certain to be discarded. It only ever stops a row that
    /// was going to run away: a model that stops on its own never reaches it.
    public static func ceiling(
        forTextTokens count: Int, config: PostprocessConfig, window: Int
    ) -> Int {
        // Clamped to `window`, not `window + 15`. `Windowing` refuses
        // anything past `maxSpeechTokens`, so those fifteen tokens could never
        // be rendered: a row allowed to reach 270 was stopped at 270 and then
        // rejected at 255, real time on a device for tokens certain to be
        // discarded. Changed in all five together with the `funnel-2` bump,
        // because a ceiling change moves audio and has to be visible in the
        // fingerprint.
        let proportional = Int(Double(count) * config.ceilingSpeechPerTextToken)
        return min(window, proportional + config.ceilingSlackTokens)
    }

    private static func silenceFlags(_ tokens: ArraySlice<Int>, _ silence: Set<Int>) -> [Bool] {
        tokens.map { silence.contains($0) }
    }

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
    /// row. The mechanism is the one behind the trailing hallucinated word, the
    /// model's own output becomes its context, but it strikes mid-sequence, so
    /// no rule that reads the end can find it.
    ///
    /// Deliberately hard to trigger, because it is the only rule here that
    /// anchors mid-sequence: a short cycle, repeated many times, matched
    /// exactly. A decoder that has genuinely locked up emits the same tokens
    /// rather than similar ones, and a fuzzy match on a signal this
    /// destructive would truncate real speech. And under
    /// ``RepetitionResume/condemn`` an *applied* cut only ever removes a
    /// tail: a loop the decoder resumed from is condemned by the resolver
    /// instead (``inspect(_:request:silence:config:)`` reads the resumption
    /// off `loopCandidate` and judges it), so the mid-sequence anchor never
    /// deletes what followed it.
    ///
    /// A cycle that is entirely silence is never a loop, silence repeating is
    /// what silence is, and the tail rules already judge pauses against where
    /// they sit. Under ``RepetitionSilence/acoustic`` the exemption reads
    /// *acoustic* silence, the passed ids unioned with both render censuses,
    /// the same way ``isStalled(_:hitCeiling:silence:config:)`` reads its
    /// censuses off the config. Keyed to the sampler list alone it was blind
    /// to six of the eight truly-silent ids, and a long pause parked on one
    /// of them fired as a period-1 loop whose cut deleted the pause and every
    /// correctly-read token behind it; the render that shows it is the one
    /// ``PostprocessConfig/repetitionSilence`` is written against. A cycle
    /// mixing silence with speech still counts, a word-then-pause stutter is
    /// one of the shapes this failure takes.
    ///
    /// Returns one full cycle past the loop's start: the first instance is
    /// plausibly the word the sentence wanted.
    public static func repetitionCut(
        _ tokens: [Int], silence: Set<Int>, config: PostprocessConfig
    ) -> Int? {
        loopCandidate(tokens, silence: silence, config: config)?.cut
    }

    /// The earliest qualifying loop: `(cut index, decoder resumed)`.
    ///
    /// One search serves both questions. The cut index is
    /// ``repetitionCut(_:silence:config:)``'s contract, unchanged. `resumed`
    /// is whether the winning loop's repeating region ends `period` or more
    /// tokens before the row does: a locked decoder emits its cycle to the
    /// end of the row, and a ceiling can truncate at most one incomplete
    /// copy (`period - 1` tokens), so a full period of anything else after
    /// the region means the decoder came back, which a locked decoder, by
    /// definition, does not. No extra scan pays for it: a matching full copy
    /// would have been counted as another cycle, so `n - at >= period`
    /// already implies a deviation.
    private static func loopCandidate(
        _ tokens: [Int], silence: Set<Int>, config: PostprocessConfig
    ) -> (cut: Int, resumed: Bool)? {
        let n = tokens.count
        guard n >= config.repetitionMinSpan else { return nil }
        // The exemption's family, not the run rules': the tail rules keep
        // reading the sampler list they were calibrated against. Resolved
        // here rather than by the caller for the same reason `isStalled`
        // reads its censuses off the config, a family that lives in a
        // caller is a family the next caller feeds wrong. Without censuses
        // the union is the sampler list, unchanged.
        let family = config.repetitionSilence == .acoustic
            ? silence.union(config.silenceRenderIds).union(config.quietRenderIds)
            : silence
        let quiet = tokens.map { family.contains($0) }

        // Earliest loop wins: a row that locks up twice locked up first at the
        // first one, and everything after it is already inside the failure.
        var best: (cut: Int, resumed: Bool)?
        let longestPeriod = min(config.repetitionMaxPeriod, n / config.repetitionMinCycles)
        guard longestPeriod >= 1 else { return nil }
        for period in 1...longestPeriod {
            var start = 0
            while start + period * config.repetitionMinCycles <= n {
                var cycles = 1
                var at = start + period
                while at + period <= n,
                      tokens[at..<(at + period)] == tokens[start..<(start + period)] {
                    cycles += 1
                    at += period
                }
                let allQuiet = quiet[start..<(start + period)].allSatisfy { $0 }
                if cycles >= config.repetitionMinCycles,
                   cycles * period >= config.repetitionMinSpan, !allQuiet {
                    let candidate = start + period
                    // `Int.max` for "nothing found yet": the first candidate
                    // always wins, and no cut can be larger than the row.
                    if candidate < (best?.cut ?? Int.max) {
                        best = (candidate, n - at >= period)
                    }
                    break
                }
                start += 1
            }
        }
        return best
    }

    /// Whether the decoder spent this row trapped in silence.
    ///
    /// The failure the tail rules structurally cannot see. The decoder enters
    /// a silence run at a pause point of its own argmax and, with `min_p`
    /// stripping every non-silence candidate while the exemption re-admits
    /// the listed silence ids, the run has no exit. That trap is the
    /// sampler's, and so are the numbers that size it: they are attached to
    /// the line that closes it, the repetition penalty in
    /// ``LRSamplerV1/sample(logits:step:seen:)``, and written up in
    /// `docs/design/postprocess.md`. What belongs here is why the detector
    /// exists after the penalty: it catches what still gets through, and any
    /// checkpoint or configuration where the trap re-opens. The rows it
    /// catches all shipped as `clean` before it, because no other rule looks
    /// inside the sequence: the rest read the tail, or count the whole row
    /// against its text.
    ///
    /// Three triggers, all integer-exact, any one condemns:
    ///
    /// - **no speech at all**: every generated token is in the
    ///   silence-or-quiet family. Measured: mute rows are 255/255 silence
    ///   tokens, a seed lottery (they recur at 3/18 re-renders), and a retry
    ///   rescues 9/10.
    /// - **a non-tail dead-air run** of at least `stallRunTokens`
    ///   true-silence tokens. Two-class: quiet-family ids extend a run
    ///   without counting toward it (see `stallRunTokens` for why single-set
    ///   counting is broken). Tail runs are excluded, the tail rules own the
    ///   tail, and a trailing pause is judged against the place it sits in.
    /// - **a ceiling overrun that is mostly silence**: `hitCeiling` and the
    ///   family holds a strict majority of the row. A tail run on a cap-hit
    ///   row is not a natural tail: the ceiling truncated the read, so the
    ///   dead air is a stall the cap happened to interrupt (the reference
    ///   specimen is a row that ran to the cap almost entirely in
    ///   silence; see `docs/design/postprocess-detectors.md`). This also
    ///   closes a structural hole: the ceiling clips rows to 4.0x text tokens
    ///   + 40, so past 80 text tokens a cap-hit stall can never reach the
    ///   4.5x desperation threshold, the rule that means "certainly broken"
    ///   was unreachable by the most broken rows this layer sees.
    ///
    /// Without a census (`silenceRenderIds` empty, a checkpoint packed
    /// before it) only the run trigger fires, keyed to the configured
    /// `silence_token_ids`. That is the measured-safe subset: 13 of the
    /// configured list's 31 ids render audible speech, so whole-row
    /// membership in that list does not prove a mute row, and a whole-row
    /// trigger keyed to it could condemn real speech. Degraded-but-safe beats
    /// a fallback that lies.
    ///
    /// Returns `true` for a condemned row. There is nothing to cut: the
    /// failure is a hole, not a tail, and the fix is the retry ladder: the
    /// same route `dropout` takes, for the same reason.
    public static func isStalled(
        _ tokens: [Int], hitCeiling: Bool, silence: Set<Int>, config: PostprocessConfig
    ) -> Bool {
        guard !tokens.isEmpty else { return false }
        let census = !config.silenceRenderIds.isEmpty
        let gate = census ? Set(config.silenceRenderIds) : silence
        let family = gate.union(config.quietRenderIds)

        let inFamily = tokens.map { family.contains($0) }
        let familyCount = inFamily.lazy.filter { $0 }.count
        if census, familyCount == tokens.count { return true }
        if census, hitCeiling, 2 * familyCount > tokens.count { return true }

        var gateCount = 0
        for (token, flag) in zip(tokens, inFamily) {
            if flag {
                if gate.contains(token) { gateCount += 1 }
            } else {
                // The run ended before the row did, so it is not the tail.
                if gateCount >= config.stallRunTokens { return true }
                gateCount = 0
            }
        }
        return false
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
    /// So the peak is corroborated by what it proposes to discard: either the
    /// tail is mostly silence by share, or it holds a long unbroken run with
    /// only a stray word behind it. Without that second half, a rhetorical
    /// pause mid-tail (25 silent tokens, then 80 of speech) matched the run
    /// rule and the rescue cut the rest of the sentence off.
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
        // them, and after the last, to be a stray word or less.
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

        // No seam, because the babble is dense; fall back to the model's own best
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
    /// stop token but its best, very weak, stop was 15 tokens before the hard
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
    /// The order, which is the contract:
    ///
    /// 1. `dropout`: the row is too short for the text. Reported whole, never
    ///    cut: nothing below can help a row that is missing content.
    /// 2. `repetition`: an exact repeated cycle. First of the cuts, because it is
    ///    the only rule that knows exactly where the failure began; every other
    ///    anchor here is inferred. A cycle the decoder came back from is condemned
    ///    whole rather than cut, since the cut would delete what it came back to say.
    /// 3. `stall`: a mid-row hole. Condemned whole, before any tail rescue: a tail
    ///    cut cannot remove a hole in the middle, and a rescue firing here would
    ///    trim the tail and ship the hole under its own reason.
    /// 4. `silence_tail`: the peak-anchored filler trim.
    /// 5. `terminal_echo`: then `desperation`, the length-anchored one is the
    ///    bluntest, and it applies to *ended* rows too, because a model that babbles
    ///    past its sentence and only then samples a stop token has forfeited the
    ///    trust that stopping implies.
    /// 6. `ended_tail_trim`: only when nothing above fired.
    ///
    /// `repetition` and `stall` come before the peak-anchored rescues and
    /// neither is peak-anchored, so this list is in the resolver's order and
    /// not in the order the rules were measured.
    public static func inspect(
        _ tokens: [Int], request: Request, silence: Set<Int>, config: PostprocessConfig
    ) -> Inspection {
        if config.mode == .off || tokens.isEmpty {
            return Inspection(keep: tokens.count, reason: .clean, suspect: false)
        }

        var cut: Int?
        var reason: Reason = .clean
        var starved = false

        // Terminal chunks only, like its three siblings. `isTerminal` means a
        // continuation chunk's stop peak is meaningless and its pauses are
        // rhythm rather than dead air, and this rule reads exactly those two
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
        let looped = loopCandidate(tokens, silence: silence, config: config)
        if let looped, looped.resumed, config.repetitionResume == .condemn {
            // The decoder came back after the repeating region, so it was
            // never locked, and the cut would delete whatever it came back
            // to say: the defect `repetitionResume` names, on any checkpoint
            // whose silence family cannot name the pause the region was.
            // Condemned like `stall`, whole: unlike a starved desperation
            // cut there is no trim worth keeping as a fallback, because the
            // trim is the defect.
            return Inspection(keep: tokens.count, reason: .repetition, suspect: true)
        }
        if let looped {
            cut = looped.cut
            reason = .repetition
        } else if isStalled(
            tokens, hitCeiling: request.hitCeiling, silence: silence, config: config)
        {
            // Condemned, never cut, before any tail rescue can run: a mid-row
            // hole is not removable by a tail cut, and a rescue that fired
            // here would trim the tail and ship the hole under its own
            // reason. Routed like `dropout`, reported whole, suspect, into
            // the retry ladder.
            return Inspection(keep: tokens.count, reason: .stall, suspect: true)
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
            // The starved rescue. On a cap-hit row the trim has no stop token
            // corroborating it, and a cut keeping fewer than
            // `desperationMinKeepPerTextToken` speech tokens per text token
            // kept less than any full read of the text. The kept audio can be
            // near-silence through ids no census lists, almost all of it, a
            // mute chunk shipped as fixed, so the keep's *length* is the only
            // evidence there is. Condemned like `stall`, but the cut stands:
            // if the retry ladder exhausts, the trim ships, today's audio,
            // flagged `suspect`, rather than the untrimmed babble.
            starved = request.hitCeiling
                && Double(desperate)
                    < Double(request.textTokenCount) * config.desperationMinKeepPerTextToken
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
        let suspect = starved
            || (cut == nil
                && request.textTokenCount >= config.desperationMinTextTokens
                && Double(tokens.count)
                    >= Double(request.textTokenCount) * config.desperationSpeechPerTextToken)
        return Inspection(keep: cut ?? tokens.count, reason: reason, suspect: suspect)
    }
}
