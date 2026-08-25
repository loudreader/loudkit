import Foundation

/// Playing faster without talking higher — WSOLA, from first principles.
///
/// Mirrors `loudkit.models.timestretch`. "Speed" in a reading app means what it
/// means on a video player: 1.5x is the same voice, sooner. Resampling gives
/// you a chipmunk; what is wanted is *time* stretched while *pitch* is left
/// alone.
///
/// **Why WSOLA and not a phase vocoder.** The phase vocoder is the other
/// standard answer and is better on sustained, harmonic material — held notes,
/// chords. Speech is the opposite kind of signal: it is mostly transients
/// (plosives, the attack of every syllable) sitting on a pitch that moves
/// continuously. A phase vocoder resynthesises from magnitudes and unwrapped
/// phases, and its characteristic failure on that material is transient
/// smearing — a /t/ arriving as a soft thud, "phasiness" on voiced segments —
/// which is precisely the part intelligibility rests on. WSOLA never leaves the
/// time domain: it copies real waveform segments and only chooses *where* to
/// copy them from, so a plosive is either included whole or not at all. It
/// cannot smear what it never transforms.
///
/// **The algorithm.** Cut the input into overlapping ~25 ms frames. Write them
/// back out at a hop that is fixed by the output rate (50 % overlap), and read
/// them in at a hop scaled by `speed`. The read position is not used as
/// computed: it is moved by up to ±10 ms to whichever offset best matches what
/// the previously written frame *would* naturally have been followed by. That
/// search is the "waveform similarity" in the name, and it keeps
/// successive frames in phase with each other, so the overlap-add
/// reinforces rather than cancels. A plain OLA without the search is the same
/// code with the search window set to zero, and it sounds like it: periodic
/// warble at the frame rate.
///
/// Everything here is deterministic — no RNG, no adaptivity, no libraries. The
/// constants are derived from the sample rate rather than written as sample
/// counts, so the same code is correct at 16 kHz or 48 kHz, and the five
/// implementations derive them the same way.
///
/// **What it costs.** At 1.25x this is hard to tell from a native reading. At
/// 2x, or at 0.5x, it is audibly processed: the alignment search cannot always
/// find a match, and the artefact is a faint roughness or a doubled consonant.
/// That is the practical range, and the bounds below are set where the result
/// stops being worth offering rather than where the arithmetic stops working.
///
/// Speed is **not** an algorithm value and is not in
/// `AlgorithmConfig.fingerprint()`: it is an execution input like the seed and
/// the text, and two engines that disagree about it are still computing the
/// same thing.
public enum TimeStretch {
    public static let minSpeed = 0.5
    public static let maxSpeed = 2.0

    /// Analysis/synthesis frame. Long enough to hold two periods of the lowest
    /// voiced pitch this is used on (~80 Hz), short enough that a frame is
    /// inside one phone.
    private static let frameMs = 25.0

    /// How far the read position may move to find a better join — a bit under
    /// one pitch period at the low end of the voiced range, which is what the
    /// search is looking for.
    private static let searchMs = 10.0

    /// Frames overlap by half. A periodic Hann window at hop = frame/2 sums to
    /// exactly one, so the overlap-add needs no normalisation of its own — the
    /// denominator in ``timeStretch(_:sampleRate:speed:)`` only ever corrects
    /// the ends and the places the alignment search moved a frame off the grid.
    private static let hannCOLAHop = 2

    /// Accept `speed` or throw, with a message that says the range.
    ///
    /// Kept here rather than in the engine so that every entry point — three
    /// engine methods, and whatever a host app puts in front of them — refuses
    /// the same values with the same words, and a new entry point cannot forget
    /// to.
    ///
    /// `LoudKitError.shape` rather than a new case: this enum's vocabulary
    /// already spends `.shape` on "what you passed is not something this can
    /// run" (`Windowing.requireFits`, "nothing to speak"), and a sixth case for
    /// one bounded scalar would buy a caller nothing they cannot get from the
    /// message.
    ///
    /// **Refused, not clamped.** A caller who asked for 3x and silently got 2x
    /// has a bug that only a stopwatch finds.
    public static func validateSpeed(_ speed: Double) throws {
        // Non-finite first: `nan` compares false against both bounds, so a
        // naive range test lets it through and the length arithmetic below then
        // produces an empty waveform rather than an error.
        guard speed.isFinite else {
            throw LoudKitError.shape("speed must be a finite number, not \(speed)")
        }
        guard speed >= minSpeed, speed <= maxSpeed else {
            throw LoudKitError.shape(
                "speed \(speed) is outside [\(minSpeed), \(maxSpeed)]. Beyond that "
                    + "range the time-stretch is audibly processed rather than merely "
                    + "faster or slower, so it is refused rather than clamped.")
        }
    }

    /// How long `n` samples become at `speed`.
    ///
    /// Written as `floor(n / speed + 0.5)` rather than with `rounded()` on
    /// purpose: Python rounds halves to even, Go, Rust, Swift and JavaScript do
    /// not, and a one-sample disagreement between ports on an exact half is the
    /// kind of thing that is found six months later in a conformance run.
    ///
    /// Only meaningful for a speed that has passed
    /// ``validateSpeed(_:)``. Zero for a division that has no finite answer,
    /// because `Int(Double.infinity)` is a trap in Swift where it is an
    /// exception in Python, and a library that crashes the host app on a
    /// caller's bad argument is the worse of the two.
    public static func stretchedLength(_ n: Int, speed: Double) -> Int {
        let scaled = (Double(n) / speed + 0.5).rounded(.down)
        guard scaled.isFinite, scaled >= 0, scaled < Double(Int.max) else { return 0 }
        return Int(scaled)
    }

    /// `audio` played at `speed`, same pitch.
    ///
    /// - Parameters:
    ///   - audio: mono samples.
    ///   - sampleRate: theirs. The frame, hop and search window are derived
    ///     from it, so this is not decorative.
    ///   - speed: greater than one shortens, less than one lengthens. `1.0`
    ///     returns the input array itself — the engine's default must be a
    ///     bypass, and "bit-identical" is easier to trust when there is no
    ///     arithmetic to be identical about.
    /// - Returns: exactly `stretchedLength(audio.count, speed:)` samples.
    public static func timeStretch(
        _ audio: [Float], sampleRate: Int, speed: Double
    ) throws -> [Float] {
        try validateSpeed(speed)
        if speed == 1.0 { return audio }

        let n = audio.count
        let outLen = stretchedLength(n, speed: speed)
        let frame = Int((Double(sampleRate) * frameMs / 1000.0 + 0.5).rounded(.down))
        let hop = frame / hannCOLAHop
        if n <= frame || outLen <= 0 || hop <= 0 {
            // Nothing to overlap-add: a fragment shorter than one frame has no
            // second frame to align against. Cut or zero-padded to the right
            // length instead, which is wrong in the way silence is wrong rather
            // than in the way a pitch shift is. At 24 kHz a frame is 600
            // samples — a fortieth of a second, below anything the engine
            // renders.
            //
            // A zero hop joins that branch rather than looping forever —
            // `writeAt += hop` would never advance. It takes a sample rate
            // under 60 Hz to reach, so it is not a behaviour difference in any
            // case a caller can hit; it turns a hang, which no stack trace
            // explains, into the short-fragment path. Python, Go and Rust all
            // guard it, and this port did not: the hop was computed *below* the
            // guard, so there was nothing to test.
            var out = [Float](repeating: 0, count: max(outLen, 0))
            for i in 0..<min(max(outLen, 0), n) { out[i] = audio[i] }
            return out
        }

        let search = Int((Double(sampleRate) * searchMs / 1000.0 + 0.5).rounded(.down))
        // Periodic Hann, i.e. 2*pi*i/frame and not /(frame-1). The periodic
        // form is the one that sums to exactly one at 50 % overlap; the
        // symmetric form is off by a hair at every frame boundary, which reads
        // as a low-level buzz at the frame rate — 40 Hz here, right in the
        // range a listener notices.
        var window = [Double](repeating: 0, count: frame)
        for i in 0..<frame {
            window[i] = 0.5 - 0.5 * cos(2.0 * Double.pi * Double(i) / Double(frame))
        }

        // Float64 throughout, as in every other port: the correlation below
        // sums `frame` products, and in Float32 the last bits of that sum are
        // noise that can pick a different offset.
        let x = audio.map(Double.init)
        // Room for the last frame to be written whole; trimmed at the end.
        var acc = [Double](repeating: 0, count: outLen + frame)
        var weight = [Double](repeating: 0, count: outLen + frame)

        var lastFrameAt = 0
        var writeAt = 0
        var k = 0
        while writeAt < outLen {
            let ideal = Int((Double(k) * Double(hop) * speed + 0.5).rounded(.down))
            var readAt = 0
            if k > 0 {
                // What the previous frame would naturally have been followed
                // by. The search asks which nearby segment continues *this*,
                // not which one the arithmetic pointed at.
                let from = min(lastFrameAt + hop, n)
                let to = min(from + frame, n)
                readAt = bestMatch(
                    x, target: from..<to, ideal: ideal, search: search, frame: frame)
            }
            readAt = min(max(readAt, 0), n - frame)

            for i in 0..<frame {
                acc[writeAt + i] += window[i] * x[readAt + i]
                weight[writeAt + i] += window[i]
            }

            lastFrameAt = n >= frame + hop ? min(readAt, n - frame - hop) : readAt
            writeAt += hop
            k += 1
        }

        // The Hann pair sums to one in the interior, so this division is the
        // identity almost everywhere; it earns its place at the two ends, where
        // only one frame contributes and the raw sum would fade in and out.
        var out = [Float](repeating: 0, count: outLen)
        for i in 0..<outLen where weight[i] > 1e-12 {
            out[i] = Float(acc[i] / weight[i])
        }
        return out
    }

    /// The offset within ±`search` of `ideal` whose frame best continues the
    /// segment at `target`.
    ///
    /// Scored by cross-correlation normalised by the *candidate's* energy only
    /// — the target's is the same for every candidate and cancels out of the
    /// ranking. Without that normalisation the search prefers whichever
    /// candidate is loudest rather than whichever fits, which at a syllable
    /// onset is exactly the wrong one.
    ///
    /// Ties go to the lower offset (the comparison is a strict `>`), so the
    /// choice does not depend on iteration order and the five ports agree.
    ///
    /// `target` is a range into `x` rather than a copied slice, and the scoring
    /// loop runs over a raw buffer pointer: this is the hot loop of the whole
    /// module — `2 * search + 1` candidates of `frame` samples for every output
    /// frame — and it is the one place here where the shape of the Swift costs
    /// enough to be worth writing around.
    private static func bestMatch(
        _ x: [Double], target: Range<Int>, ideal: Int, search: Int, frame: Int
    ) -> Int {
        let n = x.count
        let lo = max(0, ideal - search)
        let hi = min(n - frame, ideal + search)
        if hi < lo || target.count < frame {
            return min(max(ideal, 0), n - frame)
        }

        var bestAt = lo
        var bestScore = -Double.infinity
        x.withUnsafeBufferPointer { buf in
            // Unreachable for an empty input — `hi < lo` above already returned
            // — and a `guard` rather than a `!` so that stays true if it ever is.
            guard let base = buf.baseAddress else { return }
            let tgt = base + target.lowerBound
            for at in lo...hi {
                let cand = base + at
                var energy = 0.0
                var cross = 0.0
                for i in 0..<frame {
                    let c = cand[i]
                    energy += c * c
                    cross += c * tgt[i]
                }
                // A silent candidate scores zero rather than dividing by
                // nothing.
                let score = energy <= 0.0 ? 0.0 : cross / energy.squareRoot()
                if score > bestScore {
                    bestScore = score
                    bestAt = at
                }
            }
        }
        return bestAt
    }
}
