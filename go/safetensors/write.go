package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// Entry is one tensor to write: its name, dtype, shape and raw bytes.
type Entry struct {
	Name  string
	Dtype string
	Shape []int64
	Data  []byte
}

// dtypeRank orders tensors the way the safetensors library lays them out:
// widest dtype first, then by name. Only the order the reader inverts; the
// format itself accepts any.
var dtypeRank = map[string]int{
	"BOOL": 0, "U8": 1, "I8": 2, "F8_E5M2": 3, "F8_E4M3": 4, "I16": 5, "U16": 6,
	"F16": 7, "BF16": 8, "I32": 9, "U32": 10, "F32": 11, "F64": 12, "I64": 13, "U64": 14,
}

// Write lays out tensors and metadata as a safetensors file: an 8-byte
// little-endian header length, the JSON header padded to a multiple of eight
// with spaces, then the payloads in header order.
func Write(path string, entries []Entry, metadata map[string]string) error {
	sorted := append([]Entry(nil), entries...)
	sort.SliceStable(sorted, func(i, j int) bool {
		ri, rj := dtypeRank[sorted[i].Dtype], dtypeRank[sorted[j].Dtype]
		if ri != rj {
			return ri > rj
		}
		return sorted[i].Name < sorted[j].Name
	})
	var header bytes.Buffer
	header.WriteByte('{')
	if len(metadata) > 0 {
		meta, err := json.Marshal(metadata)
		if err != nil {
			return err
		}
		header.WriteString(`"__metadata__":`)
		header.Write(meta)
	}
	offset := 0
	for _, e := range sorted {
		width, ok := byteWidth(e.Dtype)
		if !ok {
			return fmt.Errorf("%s: unknown dtype %q", e.Name, e.Dtype)
		}
		elements := int64(1)
		for _, d := range e.Shape {
			elements *= d
		}
		if elements*int64(width) != int64(len(e.Data)) {
			return fmt.Errorf("%s: shape %v of %s is %d bytes, data is %d",
				e.Name, e.Shape, e.Dtype, elements*int64(width), len(e.Data))
		}
		if header.Len() > 1 {
			header.WriteByte(',')
		}
		name, _ := json.Marshal(e.Name)
		shape, _ := json.Marshal(e.Shape)
		fmt.Fprintf(&header, `%s:{"dtype":%q,"shape":%s,"data_offsets":[%d,%d]}`,
			name, e.Dtype, shape, offset, offset+len(e.Data))
		offset += len(e.Data)
	}
	header.WriteByte('}')
	for header.Len()%8 != 0 {
		header.WriteByte(' ')
	}
	var out bytes.Buffer
	var length [8]byte
	binary.LittleEndian.PutUint64(length[:], uint64(header.Len()))
	out.Write(length[:])
	out.Write(header.Bytes())
	for _, e := range sorted {
		out.Write(e.Data)
	}
	return os.WriteFile(path, out.Bytes(), 0o600)
}

// F32Bytes is float32 values as little-endian bytes.
func F32Bytes(values []float32) []byte {
	out := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(v))
	}
	return out
}

// I64Bytes is int64 values as little-endian bytes.
func I64Bytes(values []int64) []byte {
	out := make([]byte, 8*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint64(out[8*i:], uint64(v))
	}
	return out
}
