package loudkit

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// readWav reads a RIFF/WAVE stream as mono float32, and says at what rate.
//
// Enrollment takes a recording from wherever the user has one, so this accepts
// the encodings a recorder writes: 8, 16, 24 and 32-bit integer PCM, and
// 32-bit float. Channels are averaged down to the mono the enroller works in.
func readWav(r io.Reader) ([]float32, int, error) {
	head := make([]byte, 12)
	if _, err := io.ReadFull(r, head); err != nil {
		return nil, 0, fmt.Errorf("not a WAV file: %w", err)
	}
	if string(head[0:4]) != "RIFF" || string(head[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a WAV file (no RIFF/WAVE header)")
	}
	var (
		format     uint16
		channels   int
		sampleRate int
		bits       int
		haveFormat bool
	)
	for {
		var chunk [8]byte
		if _, err := io.ReadFull(r, chunk[:]); err != nil {
			return nil, 0, fmt.Errorf("WAV ends before its data chunk: %w", err)
		}
		id := string(chunk[0:4])
		size := int64(binary.LittleEndian.Uint32(chunk[4:8]))
		switch id {
		case "fmt ":
			body, err := chunkBody(r, "fmt", size)
			if err != nil {
				return nil, 0, err
			}
			if len(body) < 16 {
				return nil, 0, fmt.Errorf("WAV fmt chunk is %d bytes, need 16", len(body))
			}
			format = binary.LittleEndian.Uint16(body[0:2])
			channels = int(binary.LittleEndian.Uint16(body[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(body[4:8]))
			bits = int(binary.LittleEndian.Uint16(body[14:16]))
			// WAVE_FORMAT_EXTENSIBLE carries the real format in the first two
			// bytes of its GUID.
			if format == 0xFFFE && len(body) >= 26 {
				format = binary.LittleEndian.Uint16(body[24:26])
			}
			haveFormat = true
		case "data":
			if !haveFormat {
				return nil, 0, fmt.Errorf("WAV data chunk comes before its fmt chunk")
			}
			body, err := chunkBody(r, "data", size)
			if err != nil {
				return nil, 0, err
			}
			audio, err := decodeFrames(body, format, channels, bits)
			if err != nil {
				return nil, 0, err
			}
			return audio, sampleRate, nil
		default:
			// Chunks are word-aligned: an odd length is followed by a pad byte
			// that belongs to nothing.
			if _, err := io.CopyN(io.Discard, r, size+size%2); err != nil {
				return nil, 0, fmt.Errorf("WAV ends before its data chunk: %w", err)
			}
			continue
		}
		if size%2 == 1 {
			if _, err := io.CopyN(io.Discard, r, 1); err != nil {
				return nil, 0, err
			}
		}
	}
}

// chunkBody reads one chunk's declared bytes.
//
// The allocation follows the bytes that arrive rather than the declaration.
// The size is a 32-bit field taken straight from the file, and Enroll is the
// door a user's own recording comes through, so sizing the buffer from the
// header lets a truncated or hostile file claiming 0xFFFFFFFF take 4 GiB
// before a single byte is read: an out-of-memory kill rather than an error
// the caller sees.
//
// The short read is still reported, and the message keeps io.ReadFull's words
// so a caller matching on them still matches.
func chunkBody(r io.Reader, id string, size int64) ([]byte, error) {
	body, err := io.ReadAll(io.LimitReader(r, size))
	if err != nil {
		return nil, fmt.Errorf("WAV %s chunk: %w", id, err)
	}
	if int64(len(body)) != size {
		return nil, fmt.Errorf("WAV %s chunk declares %d bytes and the file holds %d: unexpected EOF",
			id, size, len(body))
	}
	return body, nil
}

// decodeFrames turns one data chunk into mono float32 in [-1, 1).
func decodeFrames(body []byte, format uint16, channels, bits int) ([]float32, error) {
	if channels < 1 {
		return nil, fmt.Errorf("WAV declares %d channels", channels)
	}
	width := bits / 8
	if width < 1 || bits%8 != 0 {
		return nil, fmt.Errorf("WAV declares %d bits per sample", bits)
	}
	var sample func([]byte) float32
	switch {
	case format == 1 && bits == 8:
		// 8-bit PCM is unsigned by the RIFF spec, unlike every wider width.
		sample = func(b []byte) float32 { return (float32(b[0]) - 128) / 128 }
	case format == 1 && bits == 16:
		sample = func(b []byte) float32 {
			return float32(int16(binary.LittleEndian.Uint16(b))) / 32768
		}
	case format == 1 && bits == 24:
		sample = func(b []byte) float32 {
			v := int32(b[0]) | int32(b[1])<<8 | int32(int8(b[2]))<<16
			return float32(v) / 8388608
		}
	case format == 1 && bits == 32:
		sample = func(b []byte) float32 {
			return float32(int32(binary.LittleEndian.Uint32(b))) / 2147483648
		}
	case format == 3 && bits == 32:
		sample = func(b []byte) float32 {
			return math.Float32frombits(binary.LittleEndian.Uint32(b))
		}
	default:
		return nil, fmt.Errorf("WAV encoding %d at %d bits is not one this reads "+
			"(8, 16, 24 or 32-bit integer PCM, or 32-bit float)", format, bits)
	}
	// At least one byte wide: a width or a channel count under one is refused above.
	stride := width * channels
	out := make([]float32, len(body)/stride)
	for i := range out {
		frame := body[i*stride:]
		sum := float32(0)
		for c := 0; c < channels; c++ {
			sum += sample(frame[c*width:])
		}
		out[i] = sum / float32(channels)
	}
	return out, nil
}
