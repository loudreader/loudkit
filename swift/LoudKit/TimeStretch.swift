import Foundation

/// Playing faster without talking higher: WSOLA, from first principles.
///
/// Mirrors `loudkit.models.timestretch`. "Speed" in a reading app means what it
/// means on a video player: 1.5x is the same voice, sooner. Resampling gives
/// you a chipmunk; what is wanted is *time* stretched while *pitch* is left
/// alone.
///
/// **Why WSOLA and not a phase vocoder.** The phase vocoder is the other
/// standard answer and is better on sustained, harmonic material, held notes,
/// chords. Speech is the opposite kind of signal: it is mostly transients
/// (plosives, the attack of every syllable) sitting on a pitch that moves
/// continuously. A phase vocoder resynthesises from magnitudes and unwrapped
/// phases, and its characteristic failure on that material is transient
/// smearing, a /t/ arriving as a soft thud, "phasiness" on voiced segments,
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
/// Everything here is deterministic, no RNG, no adaptivity, no libraries. The
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
    /// Slowest reading offered. Below it the alignment artefacts above are
    /// audible on speech.
    public static let minSpeed = 0.5
    /// Fastest reading offered. Above it the words stop being words.
    public static let maxSpeed = 2.0

    /// Analysis/synthesis frame. Long enough to hold two periods of the lowest
    /// voiced pitch this is used on (~80 Hz), short enough that a frame is
    /// inside one phone.
    private static let frameMs = 25.0

    /// How far the read position may move to find a better join, a bit under
    /// one pitch period at the low end of the voiced range, which is what the
    /// search is looking for.
    private static let searchMs = 10.0

    /// Frames overlap by half. A periodic Hann window at hop = frame/2 sums to
    /// exactly one, so the overlap-add needs no normalisation of its own, the
    /// denominator in ``timeStretch(_:sampleRate:speed:)`` only ever corrects
    /// the ends and the places the alignment search moved a frame off the grid.
    private static let hannCOLAHop = 2

    /// Accept `speed` or throw, with a message that says the range.
    ///
    /// Kept here rather than in the engine so that every entry point, three
    /// engine methods, and whatever a host app puts in front of them, refuses
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
    ///     returns the input array itself, the engine's default must be a
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
            // samples, a fortieth of a second, below anything the engine
            // renders.
            //
            // A zero hop joins that branch rather than looping forever on a
            // `writeAt += hop` that never advances. It takes a sample rate
            // under 60 Hz to reach, so no caller can hit it; the guard turns a
            // hang, which no stack trace explains, into the short-fragment
            // path. Python, Go and Rust guard it in the same place.
            var out = [Float](repeating: 0, count: max(outLen, 0))
            for i in 0..<min(max(outLen, 0), n) { out[i] = audio[i] }
            return out
        }

        let search = Int((Double(sampleRate) * searchMs / 1000.0 + 0.5).rounded(.down))
        // Periodic Hann, i.e. 2*pi*i/frame and not /(frame-1). The periodic
        // form is the one that sums to exactly one at 50 % overlap; the
        // symmetric form is off by a hair at every frame boundary, which reads
        // as a low-level buzz at the frame rate, 40 Hz here, right in the
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
    /// Scored by cross-correlation normalised by the *candidate's* energy only,
    /// the target's is the same for every candidate and cancels out of the
    /// ranking. Without that normalisation the search prefers whichever
    /// candidate is loudest rather than whichever fits, which at a syllable
    /// onset is exactly the wrong one.
    ///
    /// Ties go to the lower offset (the comparison is a strict `>`), so the
    /// choice does not depend on iteration order and the five ports agree.
    ///
    /// `target` is a range into `x` rather than a copied slice, and the scoring
    /// loop runs over a raw buffer pointer: this is the hot loop of the whole
    /// module, `2 * search + 1` candidates of `frame` samples for every output
    /// frame, and it is the one place here where the shape of the Swift costs
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
            // Unreachable for an empty input, `hi < lo` above already returned,
            // and a `guard` rather than a `!` so that stays true if it ever is.
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

    /// The raised-cosine ramp `fadeEdges` puts on both ends of a rendered window,
    /// in seconds. Below any speech feature, above the step it removes.
    public static let edgeFadeSeconds = 0.02

    // The Python float32 ramps at 24 kHz, pinned by the shared edge_fade fixture.
    // The historical 5 ms ramp, 120 samples.
    private static let edgeFade24k5ms: [Float] = ([
        0, 959885312, 976660992, 986541824, 993429632, 999196544, 1003297920, 1007385600,
        1010173536, 1013327328, 1015933152, 1017872448, 1019991008, 1022287328, 1024084992, 1025408456,
        1026818224, 1028313304, 1029892656, 1031555192, 1032549260, 1033461936, 1034414372, 1035405908,
        1036435856, 1037503496, 1038608080, 1039748848, 1040556192, 1041161546, 1041783760, 1042422400,
        1043077022, 1043747166, 1044432370, 1045132152, 1045846030, 1046573500, 1047314058, 1048067190,
        1048704182, 1049092528, 1049486360, 1049885403, 1050289379, 1050698008, 1051111005, 1051528080,
        1051948944, 1052373305, 1052800865, 1053231326, 1053664389, 1054099753, 1054537114, 1054976164,
        1055416602, 1055858118, 1056300406, 1056743155, 1057075334, 1057296710, 1057517852, 1057738611,
        1057958830, 1058178356, 1058397035, 1058614717, 1058831249, 1059046480, 1059260259, 1059472440,
        1059682872, 1059891410, 1060097908, 1060302222, 1060504211, 1060703732, 1060900649, 1061094821,
        1061286114, 1061474398, 1061659537, 1061841404, 1062019874, 1062194820, 1062366121, 1062533657,
        1062697312, 1062856972, 1063012526, 1063163864, 1063310882, 1063453478, 1063591550, 1063725006,
        1063853749, 1063977692, 1064096746, 1064210830, 1064319865, 1064423773, 1064522482, 1064615926,
        1064704036, 1064786752, 1064864017, 1064935777, 1065001982, 1065062585, 1065117544, 1065166822,
        1065210384, 1065248199, 1065280241, 1065306488, 1065326920, 1065341526, 1065350293, 1065353216
    ] as [UInt32]).map { Float(bitPattern: $0) }

    // The shipped 20 ms ramp, 480 samples.
    private static let edgeFade24k20ms: [Float] = ([
        0, 926187520, 942956544, 952823808, 959735808, 965537792, 969602048, 973741056,
        976511488, 979650560, 982312704, 984251136, 986373888, 988680704, 990513664, 991851008,
        993280384, 994801664, 996414592, 998119424, 999080064, 1000024128, 1001013824, 1002049280,
        1003130432, 1004257152, 1005429504, 1006640128, 1007271776, 1007926112, 1008603168, 1009302816,
        1010025120, 1010770016, 1011537440, 1012327392, 1013139840, 1013974752, 1014832064, 1015366672,
        1015817680, 1016279856, 1016753168, 1017237584, 1017733088, 1018239664, 1018757280, 1019285936,
        1019825584, 1020376224, 1020937808, 1021510320, 1022093744, 1022688064, 1023293216, 1023659696,
        1023973096, 1024291880, 1024616040, 1024945568, 1025280432, 1025620632, 1025966152, 1026316976,
        1026673088, 1027034464, 1027401104, 1027772984, 1028150088, 1028532400, 1028919912, 1029312592,
        1029710432, 1030113408, 1030521520, 1030934728, 1031353032, 1031776400, 1032001804, 1032218536,
        1032437772, 1032659508, 1032883732, 1033110436, 1033339612, 1033571244, 1033805324, 1034041848,
        1034280800, 1034522172, 1034765952, 1035012128, 1035260692, 1035511636, 1035764944, 1036020608,
        1036278616, 1036538960, 1036801624, 1037066596, 1037333868, 1037603432, 1037875268, 1038149372,
        1038425724, 1038704320, 1038985148, 1039268184, 1039553432, 1039840868, 1040130488, 1040304832,
        1040451804, 1040599844, 1040748950, 1040899114, 1041050330, 1041202592, 1041355894, 1041510228,
        1041665586, 1041821962, 1041979352, 1042137748, 1042297142, 1042457528, 1042618898, 1042781246,
        1042944566, 1043108848, 1043274088, 1043440278, 1043607412, 1043775480, 1043944476, 1044114392,
        1044285222, 1044456958, 1044629594, 1044803122, 1044977534, 1045152822, 1045328978, 1045505994,
        1045683866, 1045862584, 1046042140, 1046222528, 1046403738, 1046585762, 1046768594, 1046952226,
        1047136648, 1047321858, 1047507836, 1047694586, 1047882096, 1048070358, 1048259362, 1048449102,
        1048607785, 1048703378, 1048799326, 1048895626, 1048992272, 1049089262, 1049186590, 1049284254,
        1049382247, 1049480564, 1049579206, 1049678166, 1049777438, 1049877020, 1049976906, 1050077094,
        1050177578, 1050278353, 1050379416, 1050480762, 1050582388, 1050684288, 1050786458, 1050888891,
        1050991588, 1051094542, 1051197749, 1051301204, 1051404902, 1051508839, 1051613011, 1051717413,
        1051822041, 1051926890, 1052031956, 1052137234, 1052242720, 1052348408, 1052454294, 1052560376,
        1052666646, 1052773102, 1052879738, 1052986550, 1053093533, 1053200682, 1053307994, 1053415462,
        1053523083, 1053630853, 1053738765, 1053846817, 1053955002, 1054063316, 1054171755, 1054280315,
        1054388990, 1054497777, 1054606669, 1054715663, 1054824754, 1054933936, 1055043206, 1055152558,
        1055261989, 1055371493, 1055481065, 1055590699, 1055700394, 1055810144, 1055919943, 1056029787,
        1056139672, 1056249591, 1056359542, 1056469519, 1056579517, 1056689531, 1056799557, 1056909591,
        1056992117, 1057047134, 1057102146, 1057157153, 1057212152, 1057267141, 1057322116, 1057377076,
        1057432018, 1057486940, 1057541840, 1057596715, 1057651562, 1057706380, 1057761166, 1057815918,
        1057870634, 1057925309, 1057979944, 1058034535, 1058089080, 1058143577, 1058198023, 1058252416,
        1058306754, 1058361034, 1058415254, 1058469412, 1058523504, 1058577530, 1058631486, 1058685370,
        1058739181, 1058792915, 1058846571, 1058900145, 1058953637, 1059007043, 1059060360, 1059113588,
        1059166724, 1059219764, 1059272708, 1059325552, 1059378296, 1059430934, 1059483466, 1059535891,
        1059588205, 1059640406, 1059692492, 1059744461, 1059796310, 1059848037, 1059899640, 1059951118,
        1060002466, 1060053684, 1060104769, 1060155719, 1060206530, 1060257204, 1060307735, 1060358123,
        1060408364, 1060458459, 1060508402, 1060558194, 1060607828, 1060657308, 1060706628, 1060755788,
        1060804785, 1060853616, 1060902280, 1060950776, 1060999099, 1061047249, 1061095223, 1061143020,
        1061190636, 1061238072, 1061285322, 1061332388, 1061379266, 1061425953, 1061472448, 1061518750,
        1061564856, 1061610764, 1061656472, 1061701978, 1061747281, 1061792378, 1061837267, 1061881946,
        1061926414, 1061970667, 1062014706, 1062058528, 1062102131, 1062145513, 1062188672, 1062231606,
        1062274314, 1062316793, 1062359042, 1062401059, 1062442842, 1062484390, 1062525700, 1062566771,
        1062607601, 1062648188, 1062688530, 1062728627, 1062768476, 1062808074, 1062847422, 1062886516,
        1062925356, 1062963939, 1063002264, 1063040330, 1063078134, 1063115675, 1063152950, 1063189960,
        1063226703, 1063263176, 1063299378, 1063335308, 1063370964, 1063406344, 1063441448, 1063476272,
        1063510816, 1063545079, 1063579059, 1063612754, 1063646164, 1063679285, 1063712118, 1063744661,
        1063776912, 1063808870, 1063840534, 1063871902, 1063902972, 1063933744, 1063964217, 1063994388,
        1064024257, 1064053822, 1064083083, 1064112037, 1064140683, 1064169020, 1064197049, 1064224766,
        1064252170, 1064279262, 1064306038, 1064332500, 1064358644, 1064384469, 1064409976, 1064435162,
        1064460027, 1064484570, 1064508789, 1064532684, 1064556252, 1064579495, 1064602410, 1064624996,
        1064647253, 1064669180, 1064690774, 1064712037, 1064732966, 1064753562, 1064773822, 1064793746,
        1064813334, 1064832582, 1064851494, 1064870066, 1064888298, 1064906190, 1064923739, 1064940946,
        1064957810, 1064974330, 1064990506, 1065006337, 1065021822, 1065036960, 1065051750, 1065066194,
        1065080288, 1065094033, 1065107428, 1065120474, 1065133168, 1065145512, 1065157502, 1065169142,
        1065180428, 1065191360, 1065201938, 1065212162, 1065222032, 1065231546, 1065240704, 1065249508,
        1065257954, 1065266043, 1065273776, 1065281150, 1065288168, 1065294827, 1065301128, 1065307070,
        1065312654, 1065317878, 1065322743, 1065327248, 1065331394, 1065335180, 1065338606, 1065341672,
        1065344377, 1065346722, 1065348706, 1065350330, 1065351592, 1065352494, 1065353036, 1065353216
    ] as [UInt32]).map { Float(bitPattern: $0) }

    /// The pinned ramp of `n` samples, or nil when no table covers that length.
    private static func edgeFadeTable(_ n: Int) -> [Float]? {
        if n == edgeFade24k5ms.count { return edgeFade24k5ms }
        if n == edgeFade24k20ms.count { return edgeFade24k20ms }
        return nil
    }

    /// Taper both waveform edges; see docs/design/postprocess.md.
    public static func fadeEdges(_ audio: [Float], sampleRate: Int,
                                 seconds: Double = edgeFadeSeconds) -> [Float] {
        let n = Int(seconds * Double(sampleRate))
        guard n > 0, audio.count >= 2 * n else { return audio }
        var out = audio
        let last = out.count - 1
        // A length no table covers is computed in double and narrowed. That ramp
        // tracks the float32 reference to within two units in the last place,
        // under 1.2e-07 at full scale, about -138 dBFS: the identity contract's
        // equivalent class, not its bit-exact one.
        let table = edgeFadeTable(n)
        for i in 0..<n {
            let w = table?[i] ?? Float(0.5 - 0.5 * cos(Double.pi * Double(i) / Double(n - 1)))
            out[i] *= w
            out[last - i] *= w
        }
        return out
    }
}
