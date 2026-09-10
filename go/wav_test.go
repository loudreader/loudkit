package loudkit

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/loudreader/loudkit/go/engine"
)

func TestSaveWavAndReadBack(t *testing.T) {
	path := filepath.Join(t.TempDir(), "hello.wav")
	r := &engine.Result{Audio: []float32{0, 0.5, -0.5, 1, -1}, SampleRate: 24000}
	if err := r.SaveWav(path); err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	audio, rate, err := readWav(bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	if rate != 24000 {
		t.Errorf("rate = %d", rate)
	}
	want := []float32{0, 0.5, -0.5, 32767.0 / 32768, -1}
	if len(audio) != len(want) {
		t.Fatalf("read %d samples, wrote %d", len(audio), len(want))
	}
	for i := range want {
		if math.Abs(float64(audio[i]-want[i])) > 1e-6 {
			t.Errorf("sample %d = %v, want %v", i, audio[i], want[i])
		}
	}
}

// buildWav writes a header plus one raw data chunk, for the reader's sake.
func buildWav(format uint16, channels, rate, bits int, data []byte) []byte {
	var buf bytes.Buffer
	put := func(v any) { _ = binary.Write(&buf, binary.LittleEndian, v) }
	buf.WriteString("RIFF")
	put(uint32(36 + len(data)))
	buf.WriteString("WAVE")
	// A LIST chunk between fmt and data, which real recorders write and a
	// reader has to step over.
	buf.WriteString("LIST")
	put(uint32(4))
	buf.WriteString("INFO")
	buf.WriteString("fmt ")
	put(uint32(16))
	put(format)
	put(uint16(channels))
	put(uint32(rate))
	put(uint32(rate * channels * bits / 8))
	put(uint16(channels * bits / 8))
	put(uint16(bits))
	buf.WriteString("data")
	put(uint32(len(data)))
	buf.Write(data)
	return buf.Bytes()
}

func TestReadWavDownmixesAndSkipsChunks(t *testing.T) {
	// Two channels at 16 bit: +1.0 left against -1.0 right averages to silence.
	data := []byte{0x00, 0x40, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x00}
	audio, rate, err := readWav(bytes.NewReader(buildWav(1, 2, 16000, 16, data)))
	if err != nil {
		t.Fatal(err)
	}
	if rate != 16000 {
		t.Errorf("rate = %d", rate)
	}
	if len(audio) != 2 || audio[0] != 0 || audio[1] != 0 {
		t.Errorf("downmix = %v, want two silent frames", audio)
	}
}

func TestReadWavFloat32(t *testing.T) {
	var data bytes.Buffer
	for _, x := range []float32{0.25, -0.75} {
		_ = binary.Write(&data, binary.LittleEndian, math.Float32bits(x))
	}
	audio, _, err := readWav(bytes.NewReader(buildWav(3, 1, 24000, 32, data.Bytes())))
	if err != nil {
		t.Fatal(err)
	}
	if len(audio) != 2 || audio[0] != 0.25 || audio[1] != -0.75 {
		t.Errorf("float32 samples = %v", audio)
	}
}

// A chunk size is a 32-bit field taken straight from the file, and Enroll is
// the door a user's own recording comes through. Allocating the declared size
// before reading a byte of it turns a truncated or hostile header into an
// out-of-memory kill instead of an error, so the refusal has to come
// out of the bytes that arrive, not out of the number in the header.
func TestReadWavRefusesAChunkLargerThanTheFile(t *testing.T) {
	for _, tc := range []struct {
		name  string
		build func() []byte
	}{
		{
			name: "a data chunk claiming 4 GiB",
			build: func() []byte {
				w := buildWav(1, 1, 24000, 16, []byte{1, 2, 3, 4})
				return retag(w, "data", 0xFFFFFFFF)
			},
		},
		{
			name: "a fmt chunk claiming 4 GiB",
			build: func() []byte {
				w := buildWav(1, 1, 24000, 16, []byte{1, 2, 3, 4})
				return retag(w, "fmt ", 0xFFFFFFFF)
			},
		},
		{
			name: "a data chunk a few bytes past the end",
			build: func() []byte {
				w := buildWav(1, 1, 24000, 16, []byte{1, 2, 3, 4})
				return retag(w, "data", 64)
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := tc.build()
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			if _, _, err := readWav(bytes.NewReader(body)); err == nil {
				t.Fatal("a chunk larger than the file must be refused")
			}
			runtime.ReadMemStats(&after)
			// Generous by three orders of magnitude against the 4 GiB the
			// declaration asks for, and tight enough that an allocation sized
			// from the header fails here.
			if got := after.TotalAlloc - before.TotalAlloc; got > 1<<20 {
				t.Fatalf("refusing the chunk allocated %d bytes; the allocation must follow the file, not the header", got)
			}
		})
	}
}

// retag rewrites the declared size of the named chunk in a built WAV, leaving
// the bytes after it alone.
func retag(wav []byte, id string, size uint32) []byte {
	out := append([]byte(nil), wav...)
	at := bytes.Index(out[12:], []byte(id))
	if at < 0 {
		panic("no " + id + " chunk to retag")
	}
	binary.LittleEndian.PutUint32(out[12+at+4:], size)
	return out
}

func TestReadWavRefusesWhatItIsNot(t *testing.T) {
	if _, _, err := readWav(bytes.NewReader([]byte("not a wav at all"))); err == nil {
		t.Error("a non-RIFF stream is not a WAV")
	}
	// A-law: a real encoding this reader does not decode, named rather than
	// silently read as something else.
	if _, _, err := readWav(bytes.NewReader(buildWav(6, 1, 8000, 8, []byte{1, 2}))); err == nil {
		t.Error("an encoding this cannot decode must be refused")
	}
}
