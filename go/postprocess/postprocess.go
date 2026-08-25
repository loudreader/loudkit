// Package postprocess mirrors loudkit.postprocess — deciding where a generated
// chunk actually ended.
//
// This is a detector, not a filter. It reads the speech tokens a chunk produced
// and answers one question — where did the sentence really stop? — then returns
// a verdict. It never touches a sample of audio.
//
// The artifact it removes is generated, not spectral. The decoder is
// free-running, and silence tokens are exempt from both the repetition penalty
// and the min_p cutoff (penalising silence measurably removes pauses), so once
// the sentence is over those tokens keep probability mass indefinitely. The
// decoder free-runs silence, and any step where a non-silence token survives
// the cutoff becomes a hallucinated word — heard as "it finished, then a long
// gap, then one random word".
//
// Every constant here came from a device trace or a regression, and every rule
// is pinned by tests/data/conformance/postprocess.json, which all five ports
// run. Provenance is in docs/reference/postprocess.md.//
// Python reference: `loudkit/postprocess.py`.
package postprocess

import "sort"

// Mode is what the engine does with a verdict.
//
// "trim" applies the cut, which changes the audio and therefore travels in the
// fingerprint like every other audible decision. "report" runs the detectors
// and attaches the verdict without acting on it. "off" skips them entirely.
const (
	ModeOff    = "off"
	ModeReport = "report"
	ModeTrim   = "trim"
)

// Reasons a cut was made. ReasonClean means no rule fired.
const (
	ReasonClean        = "clean"
	ReasonDropout      = "dropout"
	ReasonRepetition   = "repetition"
	ReasonSilenceTail  = "silence_tail"
	ReasonTerminalEcho = "terminal_echo"
	ReasonDesperation  = "desperation"
	ReasonEndedTail    = "ended_tail"
)

// Config holds the detector constants. Algorithm layer: a port that
// uses a different number produces different audio, so these are hashed into
// the fingerprint rather than left as package constants.
type Config struct {
	Mode string

	// CeilingSpeechPerTextToken is the hard stop for generation, as a multiple
	// of the text-token count. Device trace of the showcase render:
	// "t3.overrun gen=92 ceiling=92 bestEOS=74@0.003 floor=31" — ~26 text
	// tokens stopped only because it hit the ceiling, mid-sentence, already at
	// 3.5 speech tokens per text token. NOT the chunker's 2.6: there, guessing
	// high only wastes window; here, guessing low cuts a sentence off.
	CeilingSpeechPerTextToken float64
	// CeilingSlackTokens carries the very short texts, where a ratio alone is
	// unsafe (1.6 s of audio).
	CeilingSlackTokens int

	// TrailingFillerThreshold is the share of a tail that must be silence
	// before it counts as one.
	TrailingFillerThreshold float64
	// TrailingSilenceRunTokens is an unbroken silence run that marks a
	// structural boundary (~0.5 s at 25 Hz). A hallucinated word sits behind
	// such a seam; under the share test alone its burst lowers the silence
	// ratio below threshold, so the ugliest tails are exactly the ones the
	// rescue refuses to cut.
	TrailingSilenceRunTokens int
	// DesperationBandRatio is the top of the stop-peak acceptance band in
	// DesperationCut, as a multiple of the text-token count. Measured reads
	// run 1.75-2.35 speech tokens per text token, so the band reaches past
	// every legitimate ending while staying well under the 4.5x garbage
	// threshold.
	DesperationBandRatio float64
	// DesperationBandFloor is the slack above the proportional band, in
	// speech tokens (~0.5 s). Carries the short texts, where the ratio alone
	// would close the band on endings a legitimate read had already reached.
	DesperationBandFloor int
	// FillerMinEosProbability is how confident the best stop must be before the
	// share/run test is consulted at all. EOS-defence bench, variant B.
	FillerMinEosProbability float64
	// FillerMaxSpeechAfterRun is how much speech may follow a seam and still be
	// a hallucinated word rather than a continuing clause (~0.4 s). Deliberately
	// separate from EndedTailWordMax despite the same number: they govern
	// different rows.
	FillerMaxSpeechAfterRun int

	// DesperationSpeechPerTextToken: past this the row certainly contains
	// garbage. "It was as he expected." — 14 text tokens — came back as 96
	// speech tokens of sentence-then-dense-babble with the stop peak at the
	// right place (45) but confidence 0.000, so every probability-gated rescue
	// refused. Real speech runs 1.75-2.35 per text token.
	DesperationSpeechPerTextToken float64
	// DesperationMinTextTokens exempts tiny texts, where fixed overheads give a
	// clean "No!" a ratio of 6+ by itself.
	DesperationMinTextTokens int

	// EndedTailSilenceRun is the silence before a blip that counts as stranding
	// it (~0.24 s).
	EndedTailSilenceRun int
	// EndedTailBlipMax: <= 80 ms of "speech" is a click, not a word.
	EndedTailBlipMax int
	// EndedTailWordMax: a stray word behind a full seam on a terminal chunk is
	// cut with it. Continuation chunks keep their tails — their pauses are the
	// sentence's rhythm and their "end" is not an end.
	EndedTailWordMax int
	// EndedTailKeep is the pause left in place after trimming (~0.2 s).
	EndedTailKeep int

	// The ordinary terminal echo: a confident stop, late, with at most ~1.2 s
	// after it. The position rule keeps a real clause pause from reading as an
	// ending.
	EchoStrongEosProbability float64
	EchoStrongMaxTail        int
	EchoStrongMinPositionPct int

	// The narrow second path, for one regression ("...but a brigand. Pass.
	// Four.": gen=124/124, bestEOS=109@0.004). Confidence this weak is accepted
	// only with every corroborator at once.
	EchoWeakEosProbability float64
	EchoWeakMaxTail        int
	EchoWeakMinPositionPct int

	// RetryMaxAttempts is how many re-rolls a condemned window may get before
	// shipping as is. Only dropout and suspect retry; each attempt draws a
	// derived seed, so the ladder is a pure function of the caller's seed.
	RetryMaxAttempts int

	// PacingTolerance is how far a chunk's pace may drift from the passage's
	// median before it is flagged (multiplicative, both directions).
	PacingTolerance float64

	// RepetitionMaxPeriod is the longest cycle, in tokens (~0.5 s), that counts
	// as a stuck decoder. Above it a repeated block is a phrase, and a repeated
	// phrase is rhetoric rather than a lock-up.
	RepetitionMaxPeriod int

	// RepetitionMinCycles is how many consecutive identical cycles a loop needs.
	// Two is a repeated phrase; three is necessary, not sufficient.
	RepetitionMinCycles int

	// RepetitionMinSpan is how many tokens the repeating region must cover
	// (~1.0 s). The constant that does the work: measured across 27 renders in
	// nine languages, a healthy row repeats for at most 10 tokens. Keying on
	// cycle count alone fired on 22 of those 27.
	RepetitionMinSpan int

	// Early truncation: the row is too short to be the text it was asked for.
	// Reported, never cut — there is nothing to cut, and it is the most damaging
	// failure in the set because a listener cannot hear that content is absent.
	// The 25-token floor is the published criterion for a catastrophic
	// neural-codec TTS failure; the proportional test exempts a genuinely short
	// line, since the shortest healthy reads measured run 35 tokens.
	DropoutMinTokens int
}

// Production is the shipping detector configuration.
func Production() Config {
	return Config{
		Mode:                          ModeTrim,
		CeilingSpeechPerTextToken:     4.0,
		CeilingSlackTokens:            40,
		TrailingFillerThreshold:       0.7,
		TrailingSilenceRunTokens:      12,
		DesperationBandRatio:          2.6,
		DesperationBandFloor:          12,
		FillerMinEosProbability:       0.05,
		FillerMaxSpeechAfterRun:       10,
		DesperationSpeechPerTextToken: 4.5,
		DesperationMinTextTokens:      10,
		EndedTailSilenceRun:           6,
		EndedTailBlipMax:              2,
		EndedTailWordMax:              10,
		EndedTailKeep:                 5,
		EchoStrongEosProbability:      0.1,
		EchoStrongMaxTail:             30,
		EchoStrongMinPositionPct:      68,
		EchoWeakEosProbability:        0.003,
		EchoWeakMaxTail:               16,
		EchoWeakMinPositionPct:        85,
		RetryMaxAttempts:              2,
		PacingTolerance:               1.6,
		RepetitionMaxPeriod:           12,
		RepetitionMinCycles:           3,
		RepetitionMinSpan:             24,
		DropoutMinTokens:              25,
	}
}

// Inspection is what the detectors concluded about one chunk.
type Inspection struct {
	// Keep is how many leading tokens survive — equal to the input length when
	// nothing fired, so a caller can always slice by it without branching.
	Keep   int
	Reason string
	// Suspect means the row is impossibly long for its text and no anchor
	// agreed where to cut. Not an error and not a cut: a report. Shipping such
	// a row silently is how the artifact reached listeners in the first place.
	Suspect bool
}

// Cut reports whether anything was removed.
func (i Inspection) Cut() bool { return i.Reason != ReasonClean }

// Request is everything the detectors need about one generated chunk.
type Request struct {
	// TextTokenCount is the denominator of every ratio rule.
	TextTokenCount int
	// MinTokens is the EOS floor this row was generated under.
	MinTokens int
	// EosPeakAt is the step at which the stop token was most probable, or
	// negative if it was never observed; EosPeakProb is that probability.
	EosPeakAt   int
	EosPeakProb float64
	// Ended is whether generation stopped at the stop token rather than a cap.
	Ended bool
	// IsTerminal is whether this chunk ends the passage. A continuation chunk
	// has no sentence end, so its stop peak means nothing.
	IsTerminal bool
	// HitCeiling is whether generation was stopped by the length ceiling.
	HitCeiling bool
}

// CeilingFor is the speech-token count at which the decoder is stopped whatever
// it thinks. Applied during generation: tokens past it cost real time on a
// device and are certain to be discarded.
func CeilingFor(textTokenCount int, cfg Config, window int) int {
	proportional := int(float64(textTokenCount)*cfg.CeilingSpeechPerTextToken) +
		cfg.CeilingSlackTokens
	if clamp := window; proportional > clamp {
		return clamp
	}
	return proportional
}

func silenceFlags(tokens []int, silence []int) []bool {
	set := make(map[int]struct{}, len(silence))
	for _, id := range silence {
		set[id] = struct{}{}
	}
	flags := make([]bool, len(tokens))
	for i, t := range tokens {
		_, flags[i] = set[t]
	}
	return flags
}

// IsTrailingFiller reports whether what follows index is a trailing tail rather
// than more sentence.
//
// The overrun rescue cuts back to where the model came closest to stopping, and
// that peak is a hint, not a verdict. Trusting it alone truncated whole
// sentences: a voice reading a language its tag does not match may never commit
// to stopping, so its best moment of hesitation lands a third of the way in. So
// the peak is corroborated by what it proposes to discard — either the tail is
// mostly silence by share, or it holds a long unbroken run with only a stray
// word behind it. Without that second half, a rhetorical pause mid-tail (25
// silent tokens, then 80 of speech) matched the run rule and the rescue cut the
// rest of the sentence off.
// IsDropout reports whether the row is too short to be the text it was asked
// for.
//
// Two conditions, both required. The absolute floor catches a row that stopped
// almost immediately whatever the text was; the proportional one keeps a
// genuinely short line exempt, because a read producing less than one speech
// token per text token has not said the text under any pronunciation.
func IsDropout(tokenCount, textTokenCount int, cfg Config) bool {
	if tokenCount >= cfg.DropoutMinTokens {
		return false
	}
	return textTokenCount > 0 && tokenCount < textTokenCount
}

// PacingOutliers reports indices of chunks whose pace drifts past the
// tolerance from the median. Long-form drift: per-chunk pace (speech tokens / text tokens) against the passage's own median, report-only. The median rather than the mean, so one broken chunk cannot drag the baseline toward itself and hide.
func PacingOutliers(ratios []float64, cfg Config) []int {
	if len(ratios) < 3 {
		// One chunk has no neighbours; two cannot say which of them drifted.
		return nil
	}
	ordered := append([]float64(nil), ratios...)
	sort.Float64s(ordered)
	mid := len(ordered) / 2
	median := ordered[mid]
	if len(ordered)%2 == 0 {
		median = (ordered[mid-1] + ordered[mid]) / 2
	}
	if median <= 0 {
		return nil
	}
	var out []int
	for i, ratio := range ratios {
		if ratio > median*cfg.PacingTolerance || ratio < median/cfg.PacingTolerance {
			out = append(out, i)
		}
	}
	return out
}

// RepetitionCut reports where a stuck decoder started looping, or -1.
//
// The failure the tail rules cannot see, because it happens *inside* the row.
// The mechanism is the one behind the trailing hallucinated word — the model's
// own output becomes its context — but it strikes mid-sequence, so no rule that
// reads the end can find it.
//
// Deliberately hard to trigger, because it is the only rule here that cuts
// mid-sequence: a short cycle, repeated many times, matched exactly. A decoder
// that has genuinely locked up emits the same tokens rather than similar ones,
// and a fuzzy match on a signal this destructive would truncate real speech.
//
// A cycle that is entirely silence is never a loop — silence repeating is what
// silence is, and the tail rules already judge pauses against where they sit.
//
// Returns one full cycle past the loop's start: the first instance is plausibly
// the word the sentence wanted.
func RepetitionCut(tokens []int, silence []int, cfg Config) int {
	n := len(tokens)
	if n < cfg.RepetitionMinSpan {
		return -1
	}
	quiet := silenceFlags(tokens, silence)

	// Earliest loop wins: a row that locks up twice locked up first at the
	// first one, and everything after it is already inside the failure.
	best := -1
	longestPeriod := cfg.RepetitionMaxPeriod
	if limit := n / cfg.RepetitionMinCycles; limit < longestPeriod {
		longestPeriod = limit
	}
	for period := 1; period <= longestPeriod; period++ {
		for start := 0; start+period*cfg.RepetitionMinCycles <= n; start++ {
			cycles := 1
			for at := start + period; at+period <= n; at += period {
				same := true
				for i := 0; i < period; i++ {
					if tokens[at+i] != tokens[start+i] {
						same = false
						break
					}
				}
				if !same {
					break
				}
				cycles++
			}
			allQuiet := true
			for i := 0; i < period; i++ {
				if !quiet[start+i] {
					allQuiet = false
					break
				}
			}
			if cycles >= cfg.RepetitionMinCycles && cycles*period >= cfg.RepetitionMinSpan && !allQuiet {
				if best < 0 || start+period < best {
					best = start + period
				}
				break
			}
		}
	}
	return best
}

func IsTrailingFiller(tokens []int, index int, silence []int, cfg Config) bool {
	if index < 0 || index >= len(tokens) {
		return false
	}
	flags := silenceFlags(tokens[index:], silence)

	silent, run, longestRun := 0, 0, 0
	for _, isSilent := range flags {
		if isSilent {
			silent++
			run++
			if run > longestRun {
				longestRun = run
			}
		} else {
			run = 0
		}
	}
	if float64(silent)/float64(len(flags)) >= cfg.TrailingFillerThreshold {
		return true
	}
	if longestRun < cfg.TrailingSilenceRunTokens {
		return false
	}

	// Collect qualifying runs, then require every gap of speech between them —
	// and after the last — to be a stray word or less. [seam][real
	// sentence][seam][word] fails: the tokens between the two seams are the
	// sentence itself, not filler trailing the first boundary.
	type runSpan struct{ start, end int }
	var runs []runSpan
	scanRun, scanStart := 0, 0
	for i, isSilent := range flags {
		if isSilent {
			if scanRun == 0 {
				scanStart = i
			}
			scanRun++
			if scanRun == cfg.TrailingSilenceRunTokens {
				runs = append(runs, runSpan{scanStart, i + 1})
			}
		} else {
			scanRun = 0
		}
	}
	if len(runs) == 0 {
		return false
	}
	if runs[0].start > cfg.FillerMaxSpeechAfterRun {
		return false
	}
	last := runs[len(runs)-1]
	if len(flags)-last.end > cfg.FillerMaxSpeechAfterRun {
		return false
	}
	for i := 1; i < len(runs); i++ {
		if runs[i].start-runs[i-1].end > cfg.FillerMaxSpeechAfterRun {
			return false
		}
	}
	return true
}

// DesperationCut is the rescue for rows whose length is the evidence. It
// returns the token count to keep, or -1 for no cut.
//
// Past the ratio the row is certainly broken, so the question is where to cut,
// not whether: at the first long silence run that starts past the floor (a run
// straddling the floor belongs to the sentence, which is why the run's start is
// tested), else at the stop peak if it sits in a band a real read could have
// ended in. The band protects the mislabeled-language case (92 generated / 26
// text = 3.5x), whose kind of row must never be cut at a peak landing a third
// of the way in.
//
// peakAllowed is false for a continuation chunk: it has no sentence end, so its
// stop peak means nothing.
func DesperationCut(tokens []int, textTokenCount, minTokens, eosPeakAt int,
	silence []int, cfg Config, peakAllowed bool) int {
	if textTokenCount < cfg.DesperationMinTextTokens {
		return -1
	}
	if float64(len(tokens)) < float64(textTokenCount)*cfg.DesperationSpeechPerTextToken {
		return -1
	}
	earliest := minTokens
	if earliest < 10 {
		earliest = 10
	}

	flags := silenceFlags(tokens, silence)
	runStart, run := -1, 0
	for i, isSilent := range flags {
		if isSilent {
			if run == 0 {
				runStart = i
			}
			run++
			if run >= cfg.TrailingSilenceRunTokens && runStart >= earliest {
				return runStart
			}
		} else {
			run = 0
		}
	}

	if !peakAllowed {
		return -1
	}
	bandTop := int(cfg.DesperationBandRatio*float64(textTokenCount)) +
		cfg.DesperationBandFloor
	if eosPeakAt >= earliest && eosPeakAt <= bandTop && eosPeakAt < len(tokens) {
		return eosPeakAt
	}
	return -1
}

// EndedTailTrim removes dead air past the sentence on a row that stopped when
// it meant to. Returns the token count to keep, or -1 for no trim.
//
// Walked backward as [sentence][r1 silence][burst][r2 silence]. Three shapes
// come off: a bare silence run half a second long; a silence run with a 1-2
// token blip right before the stop (the device specimen ended ".......#"); and,
// on a terminal chunk only, a stray word behind a full seam.
func EndedTailTrim(tokens []int, silence []int, cfg Config, isTerminal bool) int {
	flags := silenceFlags(tokens, silence)
	j := len(tokens) - 1

	r2 := 0
	for j >= 0 && flags[j] {
		r2++
		j--
	}
	if j < 0 {
		return -1
	}
	if r2 >= cfg.TrailingSilenceRunTokens {
		keep := r2
		if keep > cfg.EndedTailKeep {
			keep = cfg.EndedTailKeep
		}
		if n := j + 1 + keep; n < len(tokens) {
			return n
		}
		return -1
	}

	burst := 0
	for j >= 0 && !flags[j] {
		burst++
		j--
	}
	r1 := 0
	for j >= 0 && flags[j] {
		r1++
		j--
	}
	if j < 0 {
		return -1 // the "burst" was the sentence
	}

	strandedClick := burst <= cfg.EndedTailBlipMax && r1 >= cfg.EndedTailSilenceRun
	strandedWord := isTerminal && burst <= cfg.EndedTailWordMax &&
		r1 >= cfg.TrailingSilenceRunTokens
	if !strandedClick && !strandedWord {
		return -1
	}
	keep := r1
	if keep > cfg.EndedTailKeep {
		keep = cfg.EndedTailKeep
	}
	if n := j + 1 + keep; n < len(tokens) {
		return n
	}
	return -1
}

// TerminalEchoCut handles a terminal chunk that ended correctly and then
// free-ran an extra word. Returns the token count to keep, or -1.
//
// There is no silence seam here, so IsTrailingFiller has nothing to anchor on.
// Instead the earlier stop candidate must be strong, late and followed by a
// short tail. The second acceptance path is narrower and exists for one
// regression where the model never sampled a stop token but its best — very
// weak — stop was 15 tokens before the hard ceiling.
func TerminalEchoCut(tokenCount, eosPeakAt int, eosPeakProb float64,
	minTokens int, isTerminal, hitCeiling bool, cfg Config) int {
	if !isTerminal {
		return -1
	}
	floor := minTokens
	if floor < 10 {
		floor = 10
	}
	if eosPeakAt <= floor || eosPeakAt >= tokenCount {
		return -1
	}

	tail := tokenCount - eosPeakAt
	strongPeak := eosPeakProb >= cfg.EchoStrongEosProbability &&
		tail <= cfg.EchoStrongMaxTail &&
		eosPeakAt*100 >= tokenCount*cfg.EchoStrongMinPositionPct
	weakLatePeakAtCeiling := hitCeiling &&
		eosPeakProb >= cfg.EchoWeakEosProbability &&
		tail <= cfg.EchoWeakMaxTail &&
		eosPeakAt*100 >= tokenCount*cfg.EchoWeakMinPositionPct
	if strongPeak || weakLatePeakAtCeiling {
		return eosPeakAt
	}
	return -1
}

// Inspect runs every detector in precedence order and returns one verdict.
//
// The shipped reader grew five entry points, one per field bug, and left the
// ordering to each call site. Here they are one resolver with the precedence
// written down, because an order that lives in a caller is an order the next
// caller gets wrong.
//
// Peak-anchored rescues first, then the length-anchored one — it is the
// bluntest, and it applies to ended rows too, because a model that babbles past
// its sentence and only then samples a stop token has forfeited the trust that
// stopping implies. The ended-tail trim runs only when nothing above fired.
func Inspect(tokens []int, req Request, silence []int, cfg Config) Inspection {
	if cfg.Mode == ModeOff || len(tokens) == 0 {
		return Inspection{Keep: len(tokens), Reason: ReasonClean}
	}

	floor := req.MinTokens
	if floor < 10 {
		floor = 10
	}
	cut, reason := -1, ReasonClean

	// Terminal chunks only, like its three siblings. IsTerminal means a
	// continuation chunk's stop peak is meaningless and its pauses are rhythm
	// rather than dead air — and this rule reads exactly those two signals, so
	// it was trimming mid-passage chunks on evidence the contract says is not
	// evidence. Changed in all five implementations together; postprocess is a
	// bit-parity surface.
	fillerCut := req.IsTerminal &&
		!req.Ended &&
		req.EosPeakProb > cfg.FillerMinEosProbability &&
		req.EosPeakAt > floor &&
		req.EosPeakAt < len(tokens) &&
		IsTrailingFiller(tokens, req.EosPeakAt, silence, cfg)
	// Early truncation first: nothing below can help a row that is already too
	// short, and the verdict is "incomplete" rather than "wrongly ended".
	if IsDropout(len(tokens), req.TextTokenCount, cfg) {
		return Inspection{Keep: len(tokens), Reason: ReasonDropout, Suspect: true}
	}

	// Then repetition, because it is the only rule that knows *exactly* where
	// the failure began. Every other anchor here is inferred from a signal that
	// might mean something else; an exactly repeated cycle is not.
	if looped := RepetitionCut(tokens, silence, cfg); looped >= 0 {
		cut, reason = looped, ReasonRepetition
	} else if fillerCut {
		cut, reason = req.EosPeakAt, ReasonSilenceTail
	} else if echo := TerminalEchoCut(len(tokens), req.EosPeakAt, req.EosPeakProb,
		req.MinTokens, req.IsTerminal, req.HitCeiling, cfg); echo >= 0 {
		cut, reason = echo, ReasonTerminalEcho
	} else if desperate := DesperationCut(tokens, req.TextTokenCount, req.MinTokens,
		req.EosPeakAt, silence, cfg, req.IsTerminal); desperate >= 0 {
		cut, reason = desperate, ReasonDesperation
	}

	if cut < 0 && req.Ended {
		if trimmed := EndedTailTrim(tokens, silence, cfg, req.IsTerminal); trimmed >= 0 {
			cut, reason = trimmed, ReasonEndedTail
		}
	}

	keep := len(tokens)
	if cut >= 0 {
		keep = cut
	}
	// A condemned row that dodged every token anchor. Reported, never cut: no
	// rule could say where, and cutting at a guess is how the rescue truncated
	// whole sentences before the corroboration rules were added.
	suspect := cut < 0 &&
		req.TextTokenCount >= cfg.DesperationMinTextTokens &&
		float64(len(tokens)) >= float64(req.TextTokenCount)*cfg.DesperationSpeechPerTextToken
	return Inspection{Keep: keep, Reason: reason, Suspect: suspect}
}
