// Package sampler implements LR-SAMPLER-v1, a bit-parity port of
// loudkit.sampler. min_p is evaluated in logit space and selection is
// Gumbel-argmax, so the choice is identical on every backend.
package sampler

import (
	"math"

	"github.com/loudreader/loudkit/go/rng"
)

// Config is the sampling law (mirror of loudkit.config.SamplingConfig).
type Config struct {
	Temperature        float64
	RepetitionPenalty  float64
	MinP               float64
	MaxNewTokens       int
	SilenceTokenIds    []int
	MinTokensFloor     int
	MinTokensTextRatio float64
}

const samplingStream uint32 = 0

// Sampler chooses the next token from raw logits. It caches a block of
// precomputed Gumbel noise, because generating ten Philox rounds per token
// costs more than running the entire model.
type Sampler struct {
	config  Config
	seed    uint64
	block   int
	noise   []float64
	base    int
	silence map[int]bool

	// Observation of how close each step came to stopping. Never feeds back
	// into the draw; read by the postprocess detectors after generation.
	// stopToken < 0 disables it, and with it its cost — one exponential and
	// one sum over the vocabulary per step.
	stopToken int
	eosFloor  int
	peakAt    int
	peakProb  float64
}

func New(config Config, seed uint64) *Sampler {
	return NewWithBlock(config, seed, 256)
}

func NewWithBlock(config Config, seed uint64, block int) *Sampler {
	sil := make(map[int]bool)
	for _, t := range config.SilenceTokenIds {
		sil[t] = true
	}
	return &Sampler{
		config: config, seed: seed, block: block, silence: sil,
		stopToken: -1, peakAt: -1,
	}
}

// ObserveEOS enables the stop-token observation the postprocess layer reads.
//
// Done here, in the sampler, rather than by changing the generator: every
// backend already calls the sampler on every step — it owns the RNG stream, so
// a backend that skipped it would produce different tokens — which means the
// observation reaches every generation path without a new seam.
//
// eosFloor is the floor this generation runs under. The peak is only recorded
// past it, matching the shipped engine: below the floor the generator masks the
// stop token, so its probability there describes the mask rather than the model.
func (s *Sampler) ObserveEOS(stopToken, eosFloor int) {
	s.stopToken = stopToken
	s.eosFloor = eosFloor
	s.peakAt = -1
	s.peakProb = 0
}

// EOSPeak reports where the model came closest to stopping, as (step,
// probability). (-1, 0) when the stop token was never plausible, or when
// ObserveEOS was not called.
//
// If the model never stops, that peak is where the sentence really ended —
// which is what makes the number worth carrying.
func (s *Sampler) EOSPeak() (int, float64) { return s.peakAt, s.peakProb }

// observeEOS records how close this step came to stopping. Never changes the
// draw.
//
// The quantity is the shipped engine's, reproduced exactly: the stop token's
// softmax weight over the sum of the weights that survived min_p. The numerator
// is taken BEFORE the cutoff is applied, so a step where the stop token was
// itself filtered out still reports how near it came — the number answers "how
// close was this to being the end", not "what was the chance of stopping", and
// the first question is the one the detectors need, because the rows they exist
// to rescue are precisely the ones where stopping never won.
//
// The floor is > and not >=: at exactly the floor step the generator has only
// just unmasked the stop token, and the shipped engine records from the step
// after.
func (s *Sampler) observeEOS(scaled []float64, maxS float64, threshold float64, step int) {
	if step <= s.eosFloor || s.stopToken < 0 || s.stopToken >= len(scaled) {
		return
	}
	total := 0.0
	for i := range scaled {
		keep := s.config.MinP == 0 || scaled[i] >= threshold || s.silence[i]
		if keep {
			total += math.Exp(scaled[i] - maxS)
		}
	}
	if total <= 0 {
		return
	}
	prob := math.Exp(scaled[s.stopToken]-maxS) / total
	if prob > s.peakProb {
		s.peakProb = prob
		s.peakAt = step
	}
}

func (s *Sampler) noiseFor(step, width int) []float64 {
	if s.noise == nil || step < s.base || step >= s.base+s.block || len(s.noise) != s.block*width {
		s.base = (step / s.block) * s.block
		s.noise = rng.GumbelNoise(s.seed, samplingStream, s.base, s.block, width)
	}
	start := (step - s.base) * width
	return s.noise[start : start+width]
}

// Call chooses the next token from raw, unnormalised logits.
func (s *Sampler) Call(logits []float32, step int, seen []bool) int {
	cfg := s.config
	n := len(logits)
	z := make([]float64, n)
	for i := range logits {
		z[i] = float64(logits[i])
	}

	if cfg.RepetitionPenalty != 1.0 {
		for i := range z {
			if seen[i] && !s.silence[i] {
				if z[i] > 0 {
					z[i] /= cfg.RepetitionPenalty
				} else {
					z[i] *= cfg.RepetitionPenalty
				}
			}
		}
	}

	scaled := make([]float64, n)
	maxS := math.Inf(-1)
	for i := range z {
		scaled[i] = z[i] / cfg.Temperature
		if scaled[i] > maxS {
			maxS = scaled[i]
		}
	}

	// min_p in logit space: keep i iff s[i] >= max(s) + ln(min_p).
	threshold := math.Inf(-1)
	if cfg.MinP > 0 {
		threshold = maxS + math.Log(cfg.MinP)
	}

	if s.stopToken >= 0 {
		s.observeEOS(scaled, maxS, threshold, step)
	}

	g := s.noiseFor(step, n)
	best := math.Inf(-1)
	bestIdx := -1
	for i := range scaled {
		keep := cfg.MinP == 0 || scaled[i] >= threshold || s.silence[i]
		if !keep {
			continue
		}
		v := scaled[i] + g[i]
		if v > best {
			best = v
			bestIdx = i
		}
	}
	if bestIdx == -1 {
		return 0 // all kept values -inf; argmax falls back to index 0
	}
	return bestIdx
}
