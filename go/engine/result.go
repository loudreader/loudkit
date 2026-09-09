package engine

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"time"

	"github.com/loudreader/loudkit/go/timing"
)

// ErrCancelled is what Synthesize and SynthesizeWindow return when
// Options.ShouldCancel returned true: nothing was produced. Test for it with
// errors.Is. Stream does not return it; the chunks already handed to the
// callback are the partial, and the caller flipped the flag.
var ErrCancelled = errors.New("cancelled: ShouldCancel returned true")

// Options is what varies about one synthesis besides the text and the voice.
// The zero value is seed 0, the voice's own language, normal speed, no
// previous tokens and no cancellation.
type Options struct {
	Seed uint64
	// Language of the text. Empty means the voice's own; name one only to
	// read text in a language the voice was not enrolled in.
	Language string
	// Speed is playback speed in [0.5, 2.0], pitch preserved. Zero means 1.0,
	// which is an exact bypass.
	Speed float64
	// PreviousTokens are the Tokens of an earlier Result, so this utterance
	// continues its pitch contour instead of restarting like a fresh
	// sentence. Any length; only the tail is used.
	PreviousTokens []int
	// ShouldCancel is polled on every decode step. The chunk being generated
	// is discarded, not rendered; Synthesize then returns ErrCancelled and
	// Stream returns nil after the chunks it already delivered.
	ShouldCancel func() bool
}

func (o Options) speed() float64 {
	if o.Speed == 0 {
		return 1.0
	}
	return o.Speed
}

// Result is one rendered passage.
type Result struct {
	Audio      []float32
	SampleRate int
	Tokens     []int
	// Mel is the [80, frames] spectrogram the audio was vocoded from,
	// row-major, for comparing two backends.
	Mel []float32
	// Chunks tile the audio in order: chunk k's End is chunk k+1's Start.
	Chunks []timing.ChunkTiming
	// HitTokenCap is set when generation stopped at the token cap rather than
	// at a stop token, so the audio is real but the reading is probably cut
	// off.
	HitTokenCap bool
}

// Duration of the audio.
func (r *Result) Duration() time.Duration {
	if r.SampleRate <= 0 {
		return 0
	}
	return time.Duration(float64(len(r.Audio)) / float64(r.SampleRate) * float64(time.Second))
}

// WriteWav writes the result as a mono 16-bit PCM WAV at its own sample rate.
//
// Quantisation is floor(x * 32768) clipped to the int16 range, the rule
// python/loudkit/synthesis.py writes with, so five ports turn the same samples
// into the same bytes.
func (r *Result) WriteWav(w io.Writer) error {
	if r.SampleRate <= 0 {
		return fmt.Errorf("sample rate must be positive, got %d", r.SampleRate)
	}
	data := len(r.Audio) * 2
	buf := bufio.NewWriter(w)
	put := func(v any) {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	buf.WriteString("RIFF")
	put(uint32(36 + data))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	put(uint32(16))
	put(uint16(1)) // PCM
	put(uint16(1)) // mono
	put(uint32(r.SampleRate))
	put(uint32(r.SampleRate * 2)) // byte rate
	put(uint16(2))                // block align
	put(uint16(16))               // bits per sample
	buf.WriteString("data")
	put(uint32(data))
	frame := make([]byte, 2)
	for _, x := range r.Audio {
		binary.LittleEndian.PutUint16(frame, uint16(quantise(x)))
		if _, err := buf.Write(frame); err != nil {
			return err
		}
	}
	return buf.Flush()
}

// SaveWav writes the result to path.
func (r *Result) SaveWav(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := r.WriteWav(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// quantise is one float sample as an int16 frame. A NaN sample is silence.
//
// The NaN branch is a rule, not a description. Both comparisons below
// are false for NaN, so without the branch NaN reaches the conversion, and the
// Go specification makes a conversion whose value the result type cannot
// represent implementation-dependent. wav_quantise.json carries a "nan" probe,
// which would then be pinning an accident; the branch is what makes it a
// guarantee, and it returns the 0 the fixture already holds.
func quantise(x float32) int16 {
	scaled := math.Floor(float64(x) * 32768.0)
	if math.IsNaN(scaled) {
		return 0
	}
	switch {
	case scaled > 32767:
		return 32767
	case scaled < -32768:
		return -32768
	}
	return int16(scaled)
}
