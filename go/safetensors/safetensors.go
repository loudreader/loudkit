// Package safetensors reads the loudkit checkpoint and voice files — enough
// of the format to pull the embedding tables and the manifest. 8-byte
// little-endian header length, a JSON header naming each tensor with its
// dtype, shape and byte offsets, then the raw tensors.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Tensor is one named tensor: its shape and a byte slice of its data.
type Tensor struct {
	Dtype string
	Shape []int64
	Data  []byte
}

// File is a parsed safetensors container.
type File struct {
	Tensors  map[string]Tensor
	Metadata map[string]string
}

// Open reads and parses a safetensors file.
func Open(path string) (*File, error) {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(buf) < 8 {
		return nil, fmt.Errorf("%s: too small to be safetensors", path)
	}
	headerLen := binary.LittleEndian.Uint64(buf[:8])
	if headerLen > uint64(len(buf)-8) {
		return nil, fmt.Errorf("%s: bad header length %d", path, headerLen)
	}
	var header map[string]interface{}
	if err := json.Unmarshal(buf[8:8+headerLen], &header); err != nil {
		return nil, fmt.Errorf("%s: bad header JSON: %w", path, err)
	}
	f := &File{
		Tensors:  map[string]Tensor{},
		Metadata: map[string]string{},
	}
	base := 8 + headerLen
	for name, raw := range header {
		if name == "__metadata__" {
			meta, ok := raw.(map[string]interface{})
			if !ok {
				continue
			}
			for k, v := range meta {
				if s, ok := v.(string); ok {
					f.Metadata[k] = s
				}
			}
			continue
		}
		spec, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		dtype, _ := spec["dtype"].(string)
		offsets, _ := spec["data_offsets"].([]interface{})
		if len(offsets) != 2 {
			return nil, fmt.Errorf("%s: tensor %q missing data_offsets", path, name)
		}
		begin := uint64(toFloat(offsets[0]))
		end := uint64(toFloat(offsets[1]))
		shape := []int64{}
		if shapeRaw, ok := spec["shape"].([]interface{}); ok {
			for _, d := range shapeRaw {
				shape = append(shape, int64(toFloat(d)))
			}
		}
		// Validate against base+end, not end: the slice below is indexed from
		// the end of the header, so a tensor whose offsets fit inside the
		// file but not inside the *data section* would slice out of range and
		// panic. base+end can also wrap on a hostile header, hence the
		// overflow check.
		if begin > end || base+end < base || base+end > uint64(len(buf)) {
			return nil, fmt.Errorf("%s: tensor %q out of range", path, name)
		}
		// The shape must account for exactly the bytes claimed.
		//
		// The range check above stops a slice panic, but callers read Shape to
		// size their work — a header declaring [256] over four bytes of payload
		// is not a bad tensor, it is a reader that computes with a length the
		// data does not have. The typed accessors below also divided by the
		// element width and silently dropped a partial tail; with this check a
		// partial tail cannot exist.
		width, ok := byteWidth(dtype)
		if !ok {
			return nil, fmt.Errorf("%s: tensor %q has unknown dtype %q", path, name, dtype)
		}
		elements := int64(1)
		for _, dim := range shape {
			if dim < 0 {
				return nil, fmt.Errorf("%s: tensor %q has a negative dimension in %v",
					path, name, shape)
			}
			if dim != 0 && elements > math.MaxInt64/dim {
				return nil, fmt.Errorf("%s: tensor %q shape %v overflows", path, name, shape)
			}
			elements *= dim
		}
		if declared := elements * int64(width); declared != int64(end-begin) {
			return nil, fmt.Errorf(
				"%s: tensor %q declares shape %v of %s (%d bytes) but occupies %d bytes — "+
					"the header does not describe the payload",
				path, name, shape, dtype, declared, end-begin)
		}
		f.Tensors[name] = Tensor{
			Dtype: dtype,
			Shape: shape,
			Data:  buf[base+begin : base+end],
		}
	}
	return f, nil
}

// byteWidth is bytes per element, and false for a dtype this reader does not
// know. Listed rather than inferred: an unknown dtype must be refused at load,
// not discovered later by whichever accessor is asked for it first.
func byteWidth(dtype string) (int, bool) {
	switch dtype {
	case "F64", "I64", "U64":
		return 8, true
	case "F32", "I32", "U32":
		return 4, true
	case "F16", "BF16", "I16", "U16":
		return 2, true
	case "I8", "U8", "BOOL":
		return 1, true
	}
	return 0, false
}

func toFloat(x interface{}) float64 {
	switch v := x.(type) {
	case float64:
		return v
	case int:
		return float64(v)
	}
	return 0
}

// F32 returns a tensor's data as []float32 (F32, or F16 upcast exactly).
func (f *File) F32(name string) ([]float32, error) {
	t, ok := f.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("no tensor %q", name)
	}
	switch t.Dtype {
	case "F32":
		if len(t.Data)%4 != 0 {
			return nil, fmt.Errorf("%s: F32 data not 4-aligned", name)
		}
		out := make([]float32, len(t.Data)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.Data[i*4:]))
		}
		return out, nil
	case "F16":
		out := make([]float32, len(t.Data)/2)
		for i := range out {
			h := binary.LittleEndian.Uint16(t.Data[i*2:])
			out[i] = HalfToFloat32(h)
		}
		return out, nil
	}
	return nil, fmt.Errorf("%s: expected F32/F16, got %s", name, t.Dtype)
}

// I64 returns a tensor's data as []int64 (I64 only).
func (f *File) I64(name string) ([]int64, error) {
	t, ok := f.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("no tensor %q", name)
	}
	if t.Dtype != "I64" {
		return nil, fmt.Errorf("%s: expected I64, got %s", name, t.Dtype)
	}
	out := make([]int64, len(t.Data)/8)
	for i := range out {
		out[i] = int64(binary.LittleEndian.Uint64(t.Data[i*8:]))
	}
	return out, nil
}

// HalfToFloat32 upcasts an IEEE half-precision value exactly.
func HalfToFloat32(h uint16) float32 {
	sign := float32(1)
	if h&0x8000 != 0 {
		sign = -1
	}
	exp := (h >> 10) & 0x1f
	frac := h & 0x3ff
	switch {
	case exp == 0:
		return sign * float32(frac) * float32(math.Pow(2, -24))
	case exp == 31:
		if frac == 0 {
			return sign * float32(math.Inf(1))
		}
		return float32(math.NaN())
	}
	return sign * (1 + float32(frac)/1024) * float32(math.Pow(2, float64(exp)-15))
}
