import Foundation
import XCTest

@testable import LoudKit

/// WSOLA: the properties, not the samples.
///
/// Nothing here pins a waveform. A time-stretch has no golden output that
/// survives a change of compiler, and the five ports sum the same doubles in
/// their own order — so what is asserted is what a listener would notice if it
/// broke: the length, the pitch, the loudness, and the fact that `speed = 1.0`
/// is not a stretch at all but a bypass.
///
/// The same properties are asserted in Python, Go, Rust and TypeScript. **A
/// shared byte-level fixture was considered and rejected**: the alignment
/// search picks between candidates whose cross-correlations can differ in the
/// last bit across languages, so one offset chosen differently would move every
/// sample after it, and the fixture would fail for a reason that is not a
/// defect. Property tests fail only when the behaviour is actually wrong.
///
/// No weights are involved — this is arithmetic over a synthetic signal, so it
/// runs in every checkout.
final class TimeStretchTests: XCTestCase {
    private static let sampleRate = 24_000
    private static let speeds: [Double] = [0.5, 0.75, 0.9, 1.25, 1.5, 2.0]

    /// A voiced-ish test signal: a low fundamental, a harmonic, and a sweep.
    ///
    /// Deterministic by construction — no RNG anywhere in this file, because
    /// the stretcher has none either and a flaky DSP test is worse than no DSP
    /// test.
    private func signal(seconds: Double = 1.0, f0: Double = 220.0) -> [Float] {
        let n = Int(Double(Self.sampleRate) * seconds)
        return (0..<n).map { i in
            let t = Double(i) / Double(Self.sampleRate)
            let wave =
                0.5 * sin(2 * .pi * f0 * t)
                + 0.25 * sin(2 * .pi * 2 * f0 * t)
                + 0.15 * sin(2 * .pi * 600 * t * (1 + t))
            return Float(wave)
        }
    }

    /// Fundamental by autocorrelation. Enough to catch a chipmunk.
    private func pitchHz(_ x: [Float]) -> Double {
        let from = Self.sampleRate / 4
        let count = 4096
        XCTAssertGreaterThanOrEqual(x.count, from + count, "signal too short to measure")
        var window = (0..<count).map { Double(x[from + $0]) }
        let mean = window.reduce(0, +) / Double(count)
        for i in 0..<count { window[i] -= mean }

        let lo = Self.sampleRate / 500
        let hi = Self.sampleRate / 80
        var bestLag = lo
        var best = -Double.infinity
        for lag in lo..<hi {
            var sum = 0.0
            for i in 0..<(count - lag) { sum += window[i] * window[i + lag] }
            if sum > best {
                best = sum
                bestLag = lag
            }
        }
        return Double(Self.sampleRate) / Double(bestLag)
    }

    private func rms(_ x: [Float]) -> Double {
        guard !x.isEmpty else { return 0 }
        let total = x.reduce(0.0) { $0 + Double($1) * Double($1) }
        return (total / Double(x.count)).squareRoot()
    }

    // MARK: unity is a bypass

    func testUnitySpeedReturnsTheInputBitForBit() throws {
        // The engine's default must not depend on a DSP path being lossless —
        // it must not enter the DSP path at all.
        let x = signal(seconds: 0.3)
        let got = try TimeStretch.timeStretch(x, sampleRate: Self.sampleRate, speed: 1.0)
        XCTAssertEqual(got.count, x.count)
        XCTAssertTrue(got.elementsEqual(x), "speed 1.0 changed a sample")
        XCTAssertTrue(
            got.withUnsafeBytes { a in x.withUnsafeBytes { b in a.elementsEqual(b) } },
            "speed 1.0 is not byte-for-byte the input")
    }

    // MARK: length

    func testTheOutputIsExactlyAsLongAsAsked() throws {
        let x = signal(seconds: 1.0)
        for speed in Self.speeds {
            let got = try TimeStretch.timeStretch(x, sampleRate: Self.sampleRate, speed: speed)
            XCTAssertEqual(
                got.count, TimeStretch.stretchedLength(x.count, speed: speed),
                "wrong length at \(speed)x")
        }
    }

    func testTheLengthFormulaIsHalfUpNotHalfEven() {
        // Python rounds halves to even; Go, Rust, Swift and JavaScript do not.
        // A one-sample disagreement on an exact half is found six months later,
        // in a conformance run, by somebody else — so all five write the
        // formula out as floor(n / speed + 0.5).
        XCTAssertEqual(TimeStretch.stretchedLength(5, speed: 2.0), 3)  // 2.5 -> 3, not 2
        XCTAssertEqual(TimeStretch.stretchedLength(3, speed: 2.0), 2)  // 1.5 -> 2
    }

    func testAFragmentShorterThanAFrameIsStillTheRightLength() throws {
        // No overlap to align, so it is cut or padded. At 24 kHz this is under
        // 25 ms — below anything the engine renders, and the alternative is a
        // crash on the degenerate case.
        let tiny = [Float](repeating: 1, count: 64)
        XCTAssertEqual(
            try TimeStretch.timeStretch(tiny, sampleRate: Self.sampleRate, speed: 2.0).count, 32)
        let slow = try TimeStretch.timeStretch(tiny, sampleRate: Self.sampleRate, speed: 0.5)
        XCTAssertEqual(slow.count, 128)
        XCTAssertEqual(slow[64], 0, "the pad is silence, not a repeat")
    }

    func testAnEmptyWaveformStaysEmpty() throws {
        XCTAssertTrue(
            try TimeStretch.timeStretch([], sampleRate: Self.sampleRate, speed: 1.5).isEmpty)
    }

    // MARK: it is a stretch and not a resample

    func testPitchIsPreserved() throws {
        // The entire point. A resampler would move the fundamental by exactly
        // `speed`; this must not move it at all.
        let x = signal(seconds: 2.0)
        let before = pitchHz(x)
        for speed in Self.speeds {
            let got = try TimeStretch.timeStretch(x, sampleRate: Self.sampleRate, speed: speed)
            XCTAssertEqual(
                pitchHz(got), before, accuracy: before * 0.03, "pitch moved at \(speed)x")
        }
    }

    func testLoudnessSurvives() throws {
        // Overlap-add with a window that does not sum to one is the classic way
        // to get a 6 dB drop or a comb filter. The periodic Hann at 50 %
        // overlap sums to one, and the denominator corrects the ends.
        let x = signal(seconds: 1.0)
        let before = rms(x)
        for speed in Self.speeds {
            let got = try TimeStretch.timeStretch(x, sampleRate: Self.sampleRate, speed: speed)
            XCTAssertEqual(rms(got), before, accuracy: before * 0.15, "RMS moved at \(speed)x")
        }
    }

    func testNothingClipsOrGoesNonFinite() throws {
        let x = signal(seconds: 1.0)
        for speed in Self.speeds {
            let got = try TimeStretch.timeStretch(x, sampleRate: Self.sampleRate, speed: speed)
            XCTAssertTrue(got.allSatisfy { $0.isFinite }, "non-finite sample at \(speed)x")
            XCTAssertLessThanOrEqual(got.map { abs($0) }.max() ?? 0, 1.2, "clipped at \(speed)x")
        }
    }

    func testSilenceStaysSilent() throws {
        // The correlation search divides by candidate energy; a silent frame is
        // where that division has to not happen.
        let silence = [Float](repeating: 0, count: Self.sampleRate)
        let got = try TimeStretch.timeStretch(silence, sampleRate: Self.sampleRate, speed: 1.5)
        XCTAssertTrue(got.allSatisfy { $0 == 0 })
    }

    func testTwoCallsAgreeBitForBit() throws {
        // No RNG, no adaptivity, no wall-clock. Same in, same out, forever.
        let x = signal(seconds: 1.0)
        let first = try TimeStretch.timeStretch(x, sampleRate: Self.sampleRate, speed: 1.4)
        let second = try TimeStretch.timeStretch(x, sampleRate: Self.sampleRate, speed: 1.4)
        XCTAssertTrue(first.elementsEqual(second))
    }

    func testTheConstantsAreDerivedFromTheSampleRate() throws {
        // The frame is 25 ms and the search 10 ms at whatever rate is passed,
        // never a hardcoded sample count — otherwise a 16 kHz caller gets a
        // 37 ms frame and a different reading from every other port. Visible
        // from outside as the frame threshold moving: 500 samples is under one
        // frame at 24 kHz (600) and over it at 16 kHz (400), so the same input
        // is zero-padded by the degenerate branch in one case and actually
        // overlap-added in the other.
        let short = [Float](repeating: 1, count: 500)
        let at24 = try TimeStretch.timeStretch(short, sampleRate: 24_000, speed: 0.5)
        let at16 = try TimeStretch.timeStretch(short, sampleRate: 16_000, speed: 0.5)
        XCTAssertEqual(at24.count, 1_000)
        XCTAssertEqual(at16.count, 1_000)
        XCTAssertEqual(at24[900], 0, "24 kHz: 500 samples is under a frame, so the tail is pad")
        // A constant signal overlap-added through a window and divided by that
        // same window is exactly the constant again, at every sample the frames
        // reach.
        XCTAssertEqual(at16[900], 1, accuracy: 1e-6, "16 kHz: 500 samples is over a frame")
    }

    // MARK: validation

    func testOutOfRangeIsRefusedNotClamped() {
        // A caller who asked for 3x and silently got 2x has a bug only a
        // stopwatch finds.
        for speed in [0.49, 2.01, 0.0, -1.0, 10.0] {
            XCTAssertThrowsError(try TimeStretch.validateSpeed(speed), "\(speed) was accepted") {
                XCTAssertTrue(
                    "\($0)".contains("outside [0.5, 2.0]"),
                    "the message has to name the range: \($0)")
            }
        }
    }

    func testNonFiniteIsRefusedBeforeTheRangeCheck() {
        // `nan` compares false against both bounds, so a naive range test lets
        // it through and produces an empty waveform.
        for speed in [Double.nan, .infinity, -.infinity] {
            XCTAssertThrowsError(try TimeStretch.validateSpeed(speed)) {
                XCTAssertTrue("\($0)".contains("finite"), "wrong reason: \($0)")
            }
        }
    }

    func testTheBoundsThemselvesAreAllowed() {
        for speed in [TimeStretch.minSpeed, 1.0, TimeStretch.maxSpeed] {
            XCTAssertNoThrow(try TimeStretch.validateSpeed(speed))
        }
    }

    func testStretchingRefusesTheSameValuesValidationDoes() {
        // The stretch validates first, so an out-of-range speed cannot reach
        // the DSP and come back as a mysteriously short array.
        let x = signal(seconds: 0.1)
        XCTAssertThrowsError(
            try TimeStretch.timeStretch(x, sampleRate: Self.sampleRate, speed: 4.0))
    }

    /// The guard that no implementation tested, which is how two of the five —
    /// this one included — shipped without it.
    ///
    /// Below ~60 Hz the derived frame is one sample, so the hop (`frame /
    /// hannCOLAHop`) is zero and `writeAt += hop` never advances. This port
    /// computed the hop *after* the degenerate-shape guard, so a 64-sample
    /// buffer at 40 Hz spun past five million iterations while Python returned
    /// 43 samples in microseconds. Nothing was red, because nothing asked.
    ///
    /// Bounded by the clock as well as by the length: a regression here is a
    /// hang, and a hang wedges the suite rather than failing it.
    func testASampleRateTooLowToHaveAHopDoesNotHang() {
        let started = Date()
        let got = try? TimeStretch.timeStretch([Float](repeating: 0, count: 64),
                                               sampleRate: 40, speed: 1.5)
        XCTAssertLessThan(
            Date().timeIntervalSince(started), 5.0,
            "the overlap-add loop did not terminate")
        XCTAssertEqual(got?.count, TimeStretch.stretchedLength(64, speed: 1.5))
        XCTAssertEqual(got?.count, 43)
    }
}
