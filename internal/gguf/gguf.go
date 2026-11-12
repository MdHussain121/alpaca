// Package gguf parses just enough of a GGUF file to describe a model and size
// its tensors. Tensor data is never read, and large metadata arrays (tokenizer
// vocabularies, merges) are skipped without being loaded into memory.
package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// LayerBytes holds the weight bytes of one repeating layer.
type LayerBytes struct {
	Total  uint64
	Expert uint64
}

// Metadata is what the launcher needs to know about a model file.
type Metadata struct {
	Path string
	Name string
	Arch string

	FileType uint32
	Quant    string

	ContextLength    uint64
	BlockCount       uint64
	EmbeddingLen     uint64
	HeadCount        uint64
	HeadCountKV      []uint64
	KeyLength        uint64
	ValueLength      uint64
	ExpertCount      uint64
	ExpertUsed       uint64
	FullAttnInterval uint64
	SlidingWindow    uint64
	SlidingWindowPat []bool
	RecurrentLayers  []bool

	Layers      []LayerBytes
	GlobalBytes uint64
	EmbedBytes  uint64
	TotalBytes  uint64
	TotalParams uint64

	FileSize           int64
	UnknownTensorTypes map[uint32]bool
}

// IsMoE reports whether the file has mixture-of-experts tensors.
func (m *Metadata) IsMoE() bool { return m.ExpertCount > 0 }

// HeadDim is the per-head dimension, falling back to embedding/heads.
func (m *Metadata) HeadDim() uint64 {
	if m.KeyLength > 0 {
		return m.KeyLength
	}
	if m.HeadCount > 0 {
		return m.EmbeddingLen / m.HeadCount
	}
	return 0
}

// ValueDim is the per-head value dimension used for the V cache.
func (m *Metadata) ValueDim() uint64 {
	if m.ValueLength > 0 {
		return m.ValueLength
	}
	return m.HeadDim()
}

// KVHeadsAt returns the KV head count for a layer, supporting per-layer arrays.
func (m *Metadata) KVHeadsAt(layer int) uint64 {
	if len(m.HeadCountKV) == 0 {
		return m.HeadCount
	}
	if layer < len(m.HeadCountKV) {
		return m.HeadCountKV[layer]
	}
	return m.HeadCountKV[len(m.HeadCountKV)-1]
}

// PerLayerKV reports whether the model declares a different KV head count or
// attention pattern per layer (hybrid, recurrent, or sliding-window attention).
func (m *Metadata) PerLayerKV() bool {
	if len(m.RecurrentLayers) > 0 || m.FullAttnInterval > 1 || m.SlidingWindow > 0 {
		return true
	}
	if len(m.HeadCountKV) < 2 {
		return false
	}
	first := m.HeadCountKV[0]
	for _, v := range m.HeadCountKV[1:] {
		if v != first {
			return true
		}
	}
	return false
}

// IsKVLayer reports whether layer allocates a standard or SWA KV cache.
// Recurrent/SSM/shortconv layers (where KV head count is 0 or recurrent_layers
// is true or outside full_attention_interval) allocate no KV cache.
func (m *Metadata) IsKVLayer(layer int) bool {
	if m.KVHeadsAt(layer) == 0 {
		return false
	}
	if len(m.RecurrentLayers) > 0 {
		idx := layer % len(m.RecurrentLayers)
		if m.RecurrentLayers[idx] {
			return false
		}
		return true
	}
	if m.FullAttnInterval > 1 {
		return uint64(layer+1)%m.FullAttnInterval == 0
	}
	return true
}

// IsSWALayer reports whether layer uses a bounded sliding-window KV cache.
func (m *Metadata) IsSWALayer(layer int) bool {
	if m.SlidingWindow == 0 || !m.IsKVLayer(layer) {
		return false
	}
	if len(m.SlidingWindowPat) > 0 {
		return m.SlidingWindowPat[layer%len(m.SlidingWindowPat)]
	}
	return true
}

// ParamsString renders the parameter count compactly (4.0B, 8.2B, 761M).
func (m *Metadata) ParamsString() string {
	p := m.TotalParams
	switch {
	case p >= 1_000_000_000:
		return fmt.Sprintf("%.1fB", float64(p)/1e9)
	case p >= 1_000_000:
		return fmt.Sprintf("%.0fM", float64(p)/1e6)
	case p == 0:
		return "?"
	default:
		return fmt.Sprintf("%d", p)
	}
}

// DisplayName is the metadata name, falling back to the file name.
func (m *Metadata) DisplayName() string {
	if strings.TrimSpace(m.Name) != "" {
		return m.Name
	}
	return strings.TrimSuffix(filepath.Base(m.Path), ".gguf")
}

var magic = [4]byte{'G', 'G', 'U', 'F'}

// maxLayers bounds the repeating-layer table. No real model comes close, and
// the bound keeps a crafted file from making the parser allocate gigabytes.
const maxLayers = 1024

// typeInfo maps a ggml tensor type to its (block elements, stored bytes).
var typeInfo = map[uint32][2]int64{
	0:  {1, 4},     // F32
	1:  {1, 2},     // F16
	2:  {32, 18},   // Q4_0
	3:  {32, 20},   // Q4_1
	6:  {32, 22},   // Q5_0
	7:  {32, 24},   // Q5_1
	8:  {32, 34},   // Q8_0
	9:  {32, 36},   // Q8_1
	10: {256, 84},  // Q2_K
	11: {256, 110}, // Q3_K
	12: {256, 144}, // Q4_K
	13: {256, 176}, // Q5_K
	14: {256, 210}, // Q6_K
	15: {256, 292}, // Q8_K
	16: {256, 66},  // IQ2_XXS
	17: {256, 74},  // IQ2_XS
	18: {256, 98},  // IQ3_XXS
	19: {256, 50},  // IQ1_S
	20: {32, 18},   // IQ4_NL
	21: {256, 110}, // IQ3_S
	22: {256, 82},  // IQ2_S
	23: {256, 136}, // IQ4_XS
	24: {1, 1},     // I8
	25: {1, 2},     // I16
	26: {1, 4},     // I32
	27: {1, 8},     // I64
	28: {1, 8},     // F64
	29: {256, 56},  // IQ1_M
	30: {1, 2},     // BF16
	31: {32, 18},   // Q4_0_4_4
	32: {32, 18},   // Q4_0_4_8
	33: {32, 18},   // Q4_0_8_8
	34: {256, 54},  // TQ1_0
	35: {256, 66},  // TQ2_0
	36: {32, 18},   // IQ4_NL_4_4
	37: {32, 18},   // IQ4_NL_4_8
	38: {32, 18},   // IQ4_NL_8_8
	39: {32, 17},   // MXFP4
	40: {64, 36},   // NVFP4 (QK_NVFP4=64, 4 E4M3 scales + 32 bytes packed nibbles)
	41: {128, 18},  // Q1_0 (QK1_0=128, f16 delta + 16 bytes packed bits)
	42: {64, 18},   // Q2_0 (QK2_0=64, f16 delta + 16 bytes packed 2-bit quants)
	43: {1, 1},     // F8_E4M3
}

var fileTypeNames = map[uint32]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
	10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
	14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
	19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS",
	24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
	29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
	33: "Q4_0_4_4", 34: "Q4_0_4_8", 35: "Q4_0_8_8", 36: "TQ1_0", 37: "TQ2_0",
	38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0", 42: "F8_E4M3",
}

var typeNames = map[uint32]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
	9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
	15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
	20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
	26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 31: "Q4_0_4_4",
	32: "Q4_0_4_8", 33: "Q4_0_8_8", 34: "TQ1_0", 35: "TQ2_0", 36: "IQ4_NL_4_4",
	37: "IQ4_NL_4_8", 38: "IQ4_NL_8_8", 39: "MXFP4",
	40: "NVFP4", 41: "Q1_0", 42: "Q2_0", 43: "F8_E4M3",
}

// reader tracks an absolute file position so large arrays can be skipped with a
// seek instead of being read.
type reader struct {
	f   *os.File
	br  *bufio.Reader
	pos int64
}

func (r *reader) readN(n int) ([]byte, error) {
	b := make([]byte, n)
	if _, err := io.ReadFull(r.br, b); err != nil {
		return nil, err
	}
	r.pos += int64(n)
	return b, nil
}

func (r *reader) skip(n int64) error {
	if n < 0 {
		return errors.New("negative GGUF skip length")
	}
	if n == 0 {
		return nil
	}
	if n <= int64(r.br.Buffered()) {
		if _, err := r.br.Discard(int(n)); err != nil {
			return err
		}
		r.pos += n
		return nil
	}
	r.br.Reset(r.f)
	if _, err := r.f.Seek(r.pos+n, io.SeekStart); err != nil {
		return err
	}
	r.pos += n
	return nil
}

func (r *reader) u32() (uint32, error) {
	b, err := r.readN(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}

func (r *reader) u64() (uint64, error) {
	b, err := r.readN(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b), nil
}

func (r *reader) str() (string, error) {
	n, err := r.u64()
	if err != nil {
		return "", err
	}
	if n > 1<<24 {
		return "", fmt.Errorf("string length %d out of range", n)
	}
	b, err := r.readN(int(n))
	if err != nil {
		return "", err
	}
	return string(b), nil
}

var errShort = errors.New("unexpected end of GGUF file")

// ShardPaths returns path followed by any existing sibling shards for a split
// GGUF (matching -NNNNN-of-MMMMM.gguf). Single-file GGUFs return []string{path}.
func ShardPaths(path string) []string {
	out := []string{path}
	mm := reShard.FindStringSubmatch(filepath.Base(path))
	if mm == nil {
		return out
	}
	var first, total int
	fmt.Sscanf(mm[1], "%d", &first)
	fmt.Sscanf(mm[2], "%d", &total)
	if first < 1 || total < 2 || total > 256 {
		return out
	}
	dir := filepath.Dir(path)
	base := filepath.Base(path)
	for i := 1; i <= total; i++ {
		if i == first {
			continue
		}
		name := reShard.ReplaceAllString(base, fmt.Sprintf("-%05d-of-%05d.gguf", i, total))
		sp := filepath.Join(dir, name)
		if _, err := os.Stat(sp); err == nil {
			out = append(out, sp)
		}
	}
	return out
}

// Parse reads the header, metadata and tensor table of a GGUF file. Split
// models are resolved by reading every shard's tensor table and summing it.
func Parse(path string) (*Metadata, error) {
	m, err := parseOne(path)
	if err != nil {
		return nil, err
	}
	for _, sp := range ShardPaths(path)[1:] {
		o, err := parseOne(sp)
		if err != nil {
			continue
		}
		m.mergeShard(o)
	}
	return m, nil
}

// mergeShard folds another shard's tensor table into m.
func (m *Metadata) mergeShard(o *Metadata) {
	m.TotalBytes += o.TotalBytes
	m.TotalParams += o.TotalParams
	m.GlobalBytes += o.GlobalBytes
	m.EmbedBytes += o.EmbedBytes
	m.FileSize += o.FileSize
	for i, lb := range o.Layers {
		for len(m.Layers) <= i {
			m.Layers = append(m.Layers, LayerBytes{})
		}
		m.Layers[i].Total += lb.Total
		m.Layers[i].Expert += lb.Expert
	}
	for t := range o.UnknownTensorTypes {
		if m.UnknownTensorTypes == nil {
			m.UnknownTensorTypes = map[uint32]bool{}
		}
		m.UnknownTensorTypes[t] = true
	}
}

func parseOne(path string) (*Metadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, err
	}

	r := &reader{f: f, br: bufio.NewReaderSize(f, 1<<20)}
	m := &Metadata{Path: path, FileSize: st.Size(), UnknownTensorTypes: map[uint32]bool{}}

	var hdr [4]byte
	if _, err := io.ReadFull(r.br, hdr[:]); err != nil {
		return nil, errShort
	}
	r.pos += 4
	if hdr != magic {
		return nil, fmt.Errorf("not a GGUF file (bad magic %q)", string(hdr[:]))
	}

	version, err := r.u32()
	if err != nil {
		return nil, errShort
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("unsupported GGUF version %d", version)
	}

	tensorCount, err := r.u64()
	if err != nil {
		return nil, errShort
	}
	kvCount, err := r.u64()
	if err != nil {
		return nil, errShort
	}
	if kvCount > 1<<20 || tensorCount > 1<<24 {
		return nil, errors.New("implausible header counts; file is probably corrupt")
	}

	capHint := int(kvCount)
	if capHint > 256 {
		capHint = 256
	}
	kv := make(map[string]any, capHint)
	for i := uint64(0); i < kvCount; i++ {
		key, err := r.str()
		if err != nil {
			return nil, fmt.Errorf("metadata key %d: %w", i, err)
		}
		vtype, err := r.u32()
		if err != nil {
			return nil, errShort
		}
		keep := keepKey(key)
		val, err := r.value(vtype, keep)
		if err != nil {
			return nil, fmt.Errorf("metadata %q: %w", key, err)
		}
		if keep && val != nil {
			kv[key] = val
		}
	}

	m.applyKV(kv)

	// Tensor info table.
	typeCounts := map[uint32]uint64{}
	for i := uint64(0); i < tensorCount; i++ {
		name, err := r.str()
		if err != nil {
			return nil, fmt.Errorf("tensor %d name: %w", i, err)
		}
		nDims, err := r.u32()
		if err != nil {
			return nil, errShort
		}
		if nDims > 8 {
			return nil, fmt.Errorf("tensor %q has %d dims", name, nDims)
		}
		elems := uint64(1)
		for d := uint32(0); d < nDims; d++ {
			dim, err := r.u64()
			if err != nil {
				return nil, errShort
			}
			if dim != 0 && elems > math.MaxUint64/dim {
				return nil, fmt.Errorf("tensor %q dimensions overflow", name)
			}
			elems *= dim
		}
		ttype, err := r.u32()
		if err != nil {
			return nil, errShort
		}
		if _, err := r.u64(); err != nil { // offset, unused
			return nil, errShort
		}

		ti, ok := typeInfo[ttype]
		if !ok {
			m.UnknownTensorTypes[ttype] = true
			continue
		}
		blocks := (elems + uint64(ti[0]) - 1) / uint64(ti[0])
		bytes := blocks * uint64(ti[1])

		m.TotalBytes += bytes
		m.TotalParams += elems
		typeCounts[ttype] += bytes

		layer := layerIndex(name)
		if layer < 0 {
			m.GlobalBytes += bytes
			if strings.HasPrefix(name, "token_embd.") || strings.HasPrefix(name, "token_types.") {
				m.EmbedBytes += bytes
			}
			continue
		}
		for len(m.Layers) <= layer {
			m.Layers = append(m.Layers, LayerBytes{})
		}
		m.Layers[layer].Total += bytes
		if isExpertTensor(name) {
			m.Layers[layer].Expert += bytes
		}
	}

	if m.Quant == "" {
		var best uint32
		var bestBytes uint64
		for t, b := range typeCounts {
			if b > bestBytes {
				best, bestBytes = t, b
			}
		}
		m.Quant = typeNames[best]
	}
	if m.Quant == "" {
		m.Quant = "?"
	}
	if len(m.Layers) == 0 && m.BlockCount > 0 {
		m.Layers = make([]LayerBytes, m.BlockCount)
	}
	return m, nil
}

// keepKey avoids materialising enormous tokenizer arrays.
func keepKey(key string) bool {
	switch key {
	case "general.architecture", "general.name", "general.file_type":
		return true
	}
	if strings.HasPrefix(key, "tokenizer.") {
		return false
	}
	if strings.HasPrefix(key, "general.tags") || strings.HasPrefix(key, "general.description") {
		return false
	}
	return true
}

func (r *reader) value(vtype uint32, keep bool) (any, error) {
	switch vtype {
	case 0: // uint8
		b, err := r.readN(1)
		if err != nil {
			return nil, err
		}
		return uint64(b[0]), nil
	case 1: // int8
		b, err := r.readN(1)
		if err != nil {
			return nil, err
		}
		return uint64(int64(int8(b[0]))), nil
	case 2: // uint16
		b, err := r.readN(2)
		if err != nil {
			return nil, err
		}
		return uint64(binary.LittleEndian.Uint16(b)), nil
	case 3: // int16
		b, err := r.readN(2)
		if err != nil {
			return nil, err
		}
		return uint64(int64(int16(binary.LittleEndian.Uint16(b)))), nil
	case 4: // uint32
		v, err := r.u32()
		return uint64(v), err
	case 5: // int32
		v, err := r.u32()
		return uint64(int64(int32(v))), err
	case 6: // float32
		v, err := r.u32()
		return float64(math.Float32frombits(v)), err
	case 7: // bool
		b, err := r.readN(1)
		if err != nil {
			return nil, err
		}
		return b[0] != 0, nil
	case 8: // string
		s, err := r.str()
		if err != nil {
			return nil, err
		}
		if !keep {
			return nil, nil
		}
		return s, nil
	case 9: // array
		elemType, err := r.u32()
		if err != nil {
			return nil, err
		}
		count, err := r.u64()
		if err != nil {
			return nil, err
		}
		if !keep || count > 65536 {
			return nil, r.skipArray(elemType, count)
		}
		out := make([]any, 0, count)
		for i := uint64(0); i < count; i++ {
			v, err := r.value(elemType, true)
			if err != nil {
				return nil, err
			}
			out = append(out, v)
		}
		return out, nil
	case 10: // uint64
		return r.u64()
	case 11: // int64
		v, err := r.u64()
		return v, err
	case 12: // float64
		v, err := r.u64()
		return math.Float64frombits(v), err
	default:
		return nil, fmt.Errorf("unknown metadata value type %d", vtype)
	}
}

var scalarSize = map[uint32]int64{
	0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
}

func (r *reader) skipArray(elemType uint32, count uint64) error {
	if sz, ok := scalarSize[elemType]; ok {
		if count > uint64(math.MaxInt64/sz) {
			return errors.New("GGUF array size overflow")
		}
		return r.skip(int64(count) * sz)
	}
	if count > 1<<24 {
		return errors.New("GGUF array count out of range")
	}
	if elemType == 8 {
		for i := uint64(0); i < count; i++ {
			n, err := r.u64()
			if err != nil {
				return err
			}
			if n > uint64(math.MaxInt64) {
				return errors.New("GGUF string length overflow")
			}
			if err := r.skip(int64(n)); err != nil {
				return err
			}
		}
		return nil
	}
	// Nested arrays are rare; walk them.
	if elemType == 9 {
		for i := uint64(0); i < count; i++ {
			et, err := r.u32()
			if err != nil {
				return err
			}
			c, err := r.u64()
			if err != nil {
				return err
			}
			if err := r.skipArray(et, c); err != nil {
				return err
			}
		}
		return nil
	}
	return fmt.Errorf("cannot skip array of type %d", elemType)
}

var reBlk = regexp.MustCompile(`^blk\.(\d+)\.`)

func layerIndex(name string) int {
	mm := reBlk.FindStringSubmatch(name)
	if mm == nil {
		return -1
	}
	n, err := strconv.Atoi(mm[1])
	if err != nil || n < 0 || n >= maxLayers {
		return -1
	}
	return n
}

func isExpertTensor(name string) bool {
	return strings.Contains(name, "_exps") || strings.Contains(name, "expert")
}

func (m *Metadata) applyKV(kv map[string]any) {
	m.Arch = asString(kv["general.architecture"])
	m.Name = asString(kv["general.name"])
	if ft, ok := kv["general.file_type"]; ok {
		m.FileType = uint32(asUint(ft))
		m.Quant = fileTypeNames[m.FileType]
	}

	prefix := m.Arch
	if prefix != "" {
		m.ContextLength = asUint(kv[prefix+".context_length"])
		m.BlockCount = asUint(kv[prefix+".block_count"])
		m.EmbeddingLen = asUint(kv[prefix+".embedding_length"])
		m.HeadCount = asUint(kv[prefix+".attention.head_count"])
		m.KeyLength = asUint(kv[prefix+".attention.key_length"])
		m.ValueLength = asUint(kv[prefix+".attention.value_length"])
		m.ExpertCount = asUint(kv[prefix+".expert_count"])
		m.ExpertUsed = asUint(kv[prefix+".expert_used_count"])
		m.HeadCountKV = asUintSlice(kv[prefix+".attention.head_count_kv"])
		m.FullAttnInterval = asUint(kv[prefix+".full_attention_interval"])
		m.SlidingWindow = asUint(kv[prefix+".attention.sliding_window"])
		m.SlidingWindowPat = asBoolSlice(kv[prefix+".attention.sliding_window_pattern"])
		m.RecurrentLayers = asBoolSlice(kv[prefix+".attention.recurrent_layers"])
	}
	if m.ContextLength == 0 {
		m.ContextLength = asUint(kv["general.context_length"])
	}
	if m.BlockCount == 0 {
		m.BlockCount = asUint(kv["general.block_count"])
	}
	// A corrupt header must not size an allocation or drive a layer loop.
	if m.BlockCount > maxLayers {
		m.BlockCount = 0
	}
	if m.ExpertCount == 0 {
		m.ExpertCount = asUint(kv[prefix+".moe.expert_count"])
	}
}

func asString(v any) string {
	s, _ := v.(string)
	return s
}

func asUint(v any) uint64 {
	switch t := v.(type) {
	case uint64:
		return t
	case int64:
		if t < 0 {
			return 0
		}
		return uint64(t)
	case float64:
		if t < 0 {
			return 0
		}
		return uint64(t)
	case bool:
		if t {
			return 1
		}
		return 0
	}
	return 0
}

func asUintSlice(v any) []uint64 {
	if v == nil {
		return nil
	}
	if arr, ok := v.([]any); ok {
		out := make([]uint64, 0, len(arr))
		for _, e := range arr {
			out = append(out, asUint(e))
		}
		return out
	}
	if n := asUint(v); n > 0 {
		return []uint64{n}
	}
	return nil
}

func asBoolSlice(v any) []bool {
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]bool, 0, len(arr))
	for _, e := range arr {
		switch t := e.(type) {
		case bool:
			out = append(out, t)
		default:
			out = append(out, asUint(t) != 0)
		}
	}
	return out
}

// Candidate is a model file found by Discover.
type Candidate struct {
	Path    string
	RelPath string
	Size    int64
}

var reShard = regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)

// skipDirs keeps the default scan of a drive root (or the repo) from walking
// runtime folders and system metadata.
func skipDirs(name string) bool {
	switch strings.ToLower(name) {
	case ".git", ".obsidian", "bin", "build", "node_modules",
		"$recycle.bin", "system volume information":
		return true
	}
	return false
}

// DefaultMaxDiscoverDepth limits recursive folder traversal so scanning a
// drive root does not crawl arbitrary deep trees.
const DefaultMaxDiscoverDepth = 3

// Discover walks dir recursively up to DefaultMaxDiscoverDepth subdirectories
// for .gguf files, collapsing split models into a single candidate.
func Discover(dir string) ([]Candidate, error) {
	return DiscoverWithDepth(dir, DefaultMaxDiscoverDepth)
}

// DiscoverWithDepth walks dir up to maxDepth subdirectories below dir
// (maxDepth <= 0 disables depth limiting).
func DiscoverWithDepth(dir string, maxDepth int) ([]Candidate, error) {
	var out []Candidate
	cleanRoot := filepath.Clean(dir)
	err := filepath.WalkDir(cleanRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // ignore unreadable subtrees
		}
		if d.IsDir() {
			if path != cleanRoot {
				if skipDirs(d.Name()) {
					return filepath.SkipDir
				}
				if maxDepth > 0 {
					if rel, relErr := filepath.Rel(cleanRoot, path); relErr == nil {
						depth := strings.Count(filepath.ToSlash(rel), "/") + 1
						if depth > maxDepth {
							return filepath.SkipDir
						}
					}
				}
			}
			return nil
		}
		if !strings.EqualFold(filepath.Ext(path), ".gguf") {
			return nil
		}
		if mm := reShard.FindStringSubmatch(d.Name()); mm != nil && mm[1] != "00001" {
			return nil
		}
		rel, relErr := filepath.Rel(cleanRoot, path)
		if relErr != nil {
			rel = d.Name()
		}
		var size int64
		if info, statErr := d.Info(); statErr == nil {
			size = info.Size()
		}
		out = append(out, Candidate{Path: path, RelPath: rel, Size: size})
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}
