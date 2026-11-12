package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// ggufBuilder writes a tiny but structurally valid GGUF file.
type ggufBuilder struct {
	buf        bytes.Buffer
	tensorCnt  uint64
	kvBody     bytes.Buffer
	kvCnt      uint64
	tensorBody bytes.Buffer
}

func newBuilder() *ggufBuilder {
	b := &ggufBuilder{}
	b.buf.WriteString("GGUF")
	binary.Write(&b.buf, binary.LittleEndian, uint32(3))
	return b
}

func (b *ggufBuilder) kVKey(s string) {
	binary.Write(&b.kvBody, binary.LittleEndian, uint64(len(s)))
	b.kvBody.WriteString(s)
	b.kvCnt++
}

func (b *ggufBuilder) u32(v uint32) { binary.Write(&b.kvBody, binary.LittleEndian, v) }
func (b *ggufBuilder) u64(v uint64) { binary.Write(&b.kvBody, binary.LittleEndian, v) }

func (b *ggufBuilder) addUint32(key string, v uint32) {
	b.kVKey(key)
	b.u32(4)
	b.u32(v)
}

func (b *ggufBuilder) addString(key, v string) {
	b.kVKey(key)
	b.u32(8)
	binary.Write(&b.kvBody, binary.LittleEndian, uint64(len(v)))
	b.kvBody.WriteString(v)
}

func (b *ggufBuilder) addUint32Array(key string, vals []uint32) {
	b.kVKey(key)
	b.u32(9)
	b.u32(4)
	binary.Write(&b.kvBody, binary.LittleEndian, uint64(len(vals)))
	for _, v := range vals {
		b.u32(v)
	}
}

func (b *ggufBuilder) addStringArray(key string, vals []string) {
	b.kVKey(key)
	b.u32(9)
	b.u32(8)
	binary.Write(&b.kvBody, binary.LittleEndian, uint64(len(vals)))
	for _, v := range vals {
		binary.Write(&b.kvBody, binary.LittleEndian, uint64(len(v)))
		b.kvBody.WriteString(v)
	}
}

func (b *ggufBuilder) addUint64(key string, v uint64) {
	b.kVKey(key)
	b.u32(10)
	b.u64(v)
}

func (b *ggufBuilder) addFloat32(key string, v float32) {
	b.kVKey(key)
	b.u32(6)
	binary.Write(&b.kvBody, binary.LittleEndian, math.Float32bits(v))
}

func (b *ggufBuilder) addBoolArray(key string, vals []bool) {
	b.kVKey(key)
	b.u32(9)
	b.u32(7)
	binary.Write(&b.kvBody, binary.LittleEndian, uint64(len(vals)))
	for _, v := range vals {
		var x byte
		if v {
			x = 1
		}
		b.kvBody.WriteByte(x)
	}
}

// addTensor appends a tensor-info entry. dims are element counts per axis.
func (b *ggufBuilder) addTensor(name string, dims []uint64, ggmlType uint32) {
	binary.Write(&b.tensorBody, binary.LittleEndian, uint64(len(name)))
	b.tensorBody.WriteString(name)
	binary.Write(&b.tensorBody, binary.LittleEndian, uint32(len(dims)))
	for _, d := range dims {
		binary.Write(&b.tensorBody, binary.LittleEndian, d)
	}
	binary.Write(&b.tensorBody, binary.LittleEndian, ggmlType)
	binary.Write(&b.tensorBody, binary.LittleEndian, uint64(0))
	b.tensorCnt++
}

func (b *ggufBuilder) write(t *testing.T) string {
	t.Helper()
	return b.writeAt(t, filepath.Join(t.TempDir(), "synthetic.gguf"))
}

func (b *ggufBuilder) writeAt(t *testing.T, path string) string {
	t.Helper()
	binary.Write(&b.buf, binary.LittleEndian, b.tensorCnt)
	binary.Write(&b.buf, binary.LittleEndian, b.kvCnt)
	b.buf.Write(b.kvBody.Bytes())
	b.buf.Write(b.tensorBody.Bytes())

	if err := os.WriteFile(path, b.buf.Bytes(), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	return path
}

func TestParseMetadataAndTensors(t *testing.T) {
	b := newBuilder()
	b.addString("general.architecture", "qwen3moe")
	b.addString("general.name", "Synthetic Qwen")
	b.addUint32("general.file_type", 15) // Q4_K_M
	b.addUint32("qwen3moe.block_count", 4)
	b.addUint32("qwen3moe.context_length", 32768)
	b.addUint32("qwen3moe.embedding_length", 4096)
	b.addUint32("qwen3moe.attention.head_count", 32)
	b.addUint32Array("qwen3moe.attention.head_count_kv", []uint32{8, 8, 4, 4})
	b.addUint32("qwen3moe.attention.key_length", 128)
	b.addUint32("qwen3moe.expert_count", 128)
	b.addUint32("qwen3moe.expert_used_count", 8)

	// A large tokenizer array must be skipped, not loaded.
	vocab := make([]string, 5000)
	for i := range vocab {
		vocab[i] = "tok"
	}
	b.addStringArray("tokenizer.ggml.tokens", vocab)
	b.addFloat32("tokenizer.ggml.scores", 0)

	// Global tensors.
	b.addTensor("token_embd.weight", []uint64{4096, 32000}, 12)
	b.addTensor("output_norm.weight", []uint64{4096}, 0)
	// Four layers: attention (Q4_K) plus experts (Q4_K).
	for l := 0; l < 4; l++ {
		b.addTensor(blk(l, "attn_q.weight"), []uint64{4096, 4096}, 12)
		b.addTensor(blk(l, "ffn_gate_exps.weight"), []uint64{4096, 1408, 128}, 12)
		b.addTensor(blk(l, "ffn_down_exps.weight"), []uint64{1408, 4096, 128}, 12)
	}

	path := b.write(t)
	meta, err := Parse(path)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	if meta.Arch != "qwen3moe" {
		t.Errorf("arch = %q", meta.Arch)
	}
	if meta.Name != "Synthetic Qwen" {
		t.Errorf("name = %q", meta.Name)
	}
	if meta.Quant != "Q4_K_M" {
		t.Errorf("quant = %q", meta.Quant)
	}
	if meta.BlockCount != 4 {
		t.Errorf("layers = %d", meta.BlockCount)
	}
	if !meta.IsMoE() {
		t.Error("expected MoE")
	}
	if meta.ExpertCount != 128 {
		t.Errorf("experts = %d", meta.ExpertCount)
	}
	if len(meta.Layers) != 4 {
		t.Fatalf("layer bytes len = %d", len(meta.Layers))
	}
	if !meta.PerLayerKV() {
		t.Error("expected per-layer KV heads")
	}
	if got := meta.KVHeadsAt(0); got != 8 {
		t.Errorf("kv heads layer 0 = %d", got)
	}
	if got := meta.KVHeadsAt(3); got != 4 {
		t.Errorf("kv heads layer 3 = %d", got)
	}
	if meta.HeadDim() != 128 {
		t.Errorf("head dim = %d", meta.HeadDim())
	}

	// 4096x4096 Q4_K = 4096*4096/256*144 bytes.
	wantAttn := uint64(4096*4096) / 256 * 144
	gotAttn := meta.Layers[0].Total - meta.Layers[0].Expert
	if gotAttn != wantAttn {
		t.Errorf("attn bytes = %d, want %d", gotAttn, wantAttn)
	}
	if meta.Layers[0].Expert == 0 {
		t.Error("expert bytes not attributed")
	}
	if meta.GlobalBytes == 0 {
		t.Error("global bytes missing")
	}
	if meta.TotalBytes <= meta.GlobalBytes {
		t.Error("total bytes should exceed global bytes")
	}
	if meta.TotalParams == 0 {
		t.Error("param count missing")
	}
}

func blk(l int, name string) string {
	return "blk." + itoa(l) + "." + name
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// A corrupt header used to size an allocation directly: this file is 106 bytes
// and claimed 10^18 layers, which panicked with "makeslice: len out of range".
func TestParseAbsurdBlockCountDoesNotPanic(t *testing.T) {
	b := newBuilder()
	b.addString("general.architecture", "llama")
	b.addUint64("llama.block_count", 1_000_000_000_000_000_000)
	b.addTensor("token_embd.weight", []uint64{8}, 0)

	meta, err := Parse(b.write(t))
	if err != nil {
		t.Fatalf("must not fail on a bad count: %v", err)
	}
	if meta.BlockCount != 0 {
		t.Fatalf("block count = %d, want the implausible value dropped", meta.BlockCount)
	}
}

// A tensor name can claim any layer index; it must not grow the layer table.
func TestParseAbsurdLayerIndexIsIgnored(t *testing.T) {
	b := newBuilder()
	b.addString("general.architecture", "llama")
	b.addUint32("llama.block_count", 2)
	b.addTensor("blk.999999999.attn_q.weight", []uint64{8}, 0)

	meta, err := Parse(b.write(t))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(meta.Layers) > 2 {
		t.Fatalf("layer table = %d, want it bounded by block_count", len(meta.Layers))
	}
}

func TestParseMergesSplitShards(t *testing.T) {
	dir := t.TempDir()

	first := newBuilder()
	first.addString("general.architecture", "llama")
	first.addUint32("llama.block_count", 2)
	first.addTensor("blk.0.attn_q.weight", []uint64{4096, 4096}, 12)
	firstPath := filepath.Join(dir, "m-00001-of-00002.gguf")
	first.writeAt(t, firstPath)

	second := newBuilder()
	second.addString("general.architecture", "llama")
	second.addUint32("llama.block_count", 2)
	second.addTensor("blk.1.attn_q.weight", []uint64{4096, 4096}, 12)
	second.writeAt(t, filepath.Join(dir, "m-00002-of-00002.gguf"))

	meta, err := Parse(firstPath)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	one := uint64(4096*4096) / 256 * 144
	if meta.TotalBytes != 2*one {
		t.Fatalf("total bytes = %d, want both shards (%d)", meta.TotalBytes, 2*one)
	}
	if len(meta.Layers) != 2 {
		t.Fatalf("layers = %d, want 2", len(meta.Layers))
	}
}

func TestParseRejectsTooManyDimensions(t *testing.T) {
	b := newBuilder()
	b.addString("general.architecture", "llama")
	name := "x"
	binary.Write(&b.tensorBody, binary.LittleEndian, uint64(len(name)))
	b.tensorBody.WriteString(name)
	binary.Write(&b.tensorBody, binary.LittleEndian, uint32(9))
	for i := 0; i < 9; i++ {
		binary.Write(&b.tensorBody, binary.LittleEndian, uint64(1))
	}
	binary.Write(&b.tensorBody, binary.LittleEndian, uint32(0))
	binary.Write(&b.tensorBody, binary.LittleEndian, uint64(0))
	b.tensorCnt++

	if _, err := Parse(b.write(t)); err == nil {
		t.Fatal("a 9-dimensional tensor was accepted")
	}
}

func TestParseRejectsTensorDimensionOverflow(t *testing.T) {
	b := newBuilder()
	b.addString("general.architecture", "llama")
	b.addTensor("y", []uint64{math.MaxUint64, 3}, 0)

	if _, err := Parse(b.write(t)); err == nil {
		t.Fatal("overflowing dimensions were accepted")
	}
}

func TestParseRejectsTruncatedTensorTable(t *testing.T) {
	b := newBuilder()
	b.addString("general.architecture", "llama")
	b.addTensor("blk.0.attn_q.weight", []uint64{256}, 0)
	path := b.write(t)

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	trunc := filepath.Join(t.TempDir(), "trunc.gguf")
	if err := os.WriteFile(trunc, data[:len(data)-6], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Parse(trunc); err == nil {
		t.Fatal("truncated tensor table accepted")
	}
}

func TestParseRejectsTruncatedMetadata(t *testing.T) {
	b := newBuilder()
	b.addString("general.architecture", "llama")
	path := b.write(t)

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	trunc := filepath.Join(t.TempDir(), "trunc-meta.gguf")
	if err := os.WriteFile(trunc, data[:24], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Parse(trunc); err == nil {
		t.Fatal("truncated metadata accepted")
	}
}

func TestParseOnDirectoryFails(t *testing.T) {
	if _, err := Parse(t.TempDir()); err == nil {
		t.Fatal("a directory was accepted as a model")
	}
}

func TestParseSkipsNestedLargeArray(t *testing.T) {
	b := newBuilder()
	b.addString("general.architecture", "llama")
	b.kVKey("x.nested")
	b.u32(9) // array
	b.u32(9) // of arrays
	b.u64(1)
	b.u32(4) // inner element type
	b.u64(70000)
	for i := 0; i < 70000; i++ {
		b.u32(uint32(i))
	}

	if _, err := Parse(b.write(t)); err != nil {
		t.Fatalf("parse: %v", err)
	}
}

func TestParseRejectsNonGGUF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.gguf")
	if err := os.WriteFile(path, []byte("not a model at all"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Parse(path); err == nil {
		t.Fatal("expected an error")
	}
}

func TestParseTruncatedFile(t *testing.T) {
	b := newBuilder()
	b.addString("general.architecture", "llama")
	b.addUint32("llama.block_count", 4)
	path := b.write(t)

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	trunc := filepath.Join(t.TempDir(), "trunc.gguf")
	if err := os.WriteFile(trunc, data[:len(data)-2], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Parse(trunc); err == nil {
		t.Fatal("expected an error for a truncated file")
	}
}

// Every metadata value type must be decoded without touching tensor data.
func TestParseEveryMetadataValueType(t *testing.T) {
	b := newBuilder()
	b.addString("general.architecture", "llama")
	b.addUint32("llama.block_count", 2)

	b.kVKey("t.u8")
	b.u32(0)
	b.kvBody.WriteByte(255)
	b.kVKey("t.i8")
	b.u32(1)
	b.kvBody.WriteByte(0xff)
	b.kVKey("t.u16")
	b.u32(2)
	binary.Write(&b.kvBody, binary.LittleEndian, uint16(65535))
	b.kVKey("t.i16")
	b.u32(3)
	binary.Write(&b.kvBody, binary.LittleEndian, int16(-2))
	b.kVKey("t.u32")
	b.u32(4)
	b.u32(7)
	b.kVKey("t.i32")
	b.u32(5)
	b.u32(uint32(0xffffffff))
	b.kVKey("t.f32")
	b.u32(6)
	b.u32(math.Float32bits(1.5))
	b.kVKey("t.bool")
	b.u32(7)
	b.kvBody.WriteByte(1)
	b.kVKey("t.str")
	b.u32(8)
	b.u64(3)
	b.kvBody.WriteString("abc")
	b.kVKey("t.arr")
	b.u32(9)
	b.u32(4)
	b.u64(3)
	for i := 0; i < 3; i++ {
		b.u32(uint32(i))
	}
	b.kVKey("t.u64")
	b.u32(10)
	b.u64(9)
	b.kVKey("t.i64")
	b.u32(11)
	b.u64(0xffffffffffffffff)
	b.kVKey("t.f64")
	b.u32(12)
	b.u64(math.Float64bits(2.5))

	if _, err := Parse(b.write(t)); err != nil {
		t.Fatalf("parse: %v", err)
	}
}

func TestParseRejectsUnknownMetadataType(t *testing.T) {
	b := newBuilder()
	b.addString("general.architecture", "llama")
	b.kVKey("t.bad")
	b.u32(99)

	if _, err := Parse(b.write(t)); err == nil {
		t.Fatal("unknown value type accepted")
	}
}

// A large array under a kept key must be skipped, not loaded.
func TestParseSkipsHugeArrayOnKeptKey(t *testing.T) {
	b := newBuilder()
	b.addString("general.architecture", "llama")
	b.kVKey("x.big")
	b.u32(9)
	b.u32(4)
	b.u64(70000)
	for i := 0; i < 70000; i++ {
		b.u32(uint32(i))
	}

	if _, err := Parse(b.write(t)); err != nil {
		t.Fatalf("parse: %v", err)
	}
}

func TestParseUnknownTensorTypeIsRecorded(t *testing.T) {
	b := newBuilder()
	b.addString("general.architecture", "llama")
	b.addUint32("llama.block_count", 1)
	b.addTensor("blk.0.attn_q.weight", []uint64{256}, 250) // not in the table

	meta, err := Parse(b.write(t))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !meta.UnknownTensorTypes[250] {
		t.Fatal("unknown tensor type not recorded")
	}
}

func TestParseRejectsUnsupportedVersions(t *testing.T) {
	for _, v := range []uint32{1, 4} {
		buf := &bytes.Buffer{}
		buf.WriteString("GGUF")
		binary.Write(buf, binary.LittleEndian, v)
		binary.Write(buf, binary.LittleEndian, uint64(0))
		binary.Write(buf, binary.LittleEndian, uint64(0))

		path := filepath.Join(t.TempDir(), "v.gguf")
		if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := Parse(path); err == nil {
			t.Fatalf("version %d accepted", v)
		}
	}
}

func TestHeadDimFallsBackToEmbeddingOverHeads(t *testing.T) {
	m := &Metadata{EmbeddingLen: 4096, HeadCount: 32}
	if got := m.HeadDim(); got != 128 {
		t.Fatalf("head dim = %d", got)
	}
	if got := m.ValueDim(); got != 128 {
		t.Fatalf("value dim = %d", got)
	}
	empty := &Metadata{}
	if empty.HeadDim() != 0 || empty.ValueDim() != 0 {
		t.Fatal("empty metadata must not divide by zero")
	}
}

func TestParamsStringAndDisplayName(t *testing.T) {
	cases := []struct {
		p    uint64
		want string
	}{
		{8_200_000_000, "8.2B"},
		{761_000_000, "761M"},
		{0, "?"},
		{900, "900"},
	}
	for _, tc := range cases {
		if got := (&Metadata{TotalParams: tc.p}).ParamsString(); got != tc.want {
			t.Errorf("params %d = %q, want %q", tc.p, got, tc.want)
		}
	}
	if got := (&Metadata{Path: `models\demo.gguf`}).DisplayName(); got != "demo" {
		t.Errorf("display name = %q", got)
	}
}

func TestKVHeadsAtFallsBackToLastEntry(t *testing.T) {
	m := &Metadata{HeadCount: 32, HeadCountKV: []uint64{8, 4}}
	if got := m.KVHeadsAt(5); got != 4 {
		t.Fatalf("KVHeadsAt(5) = %d", got)
	}
	if got := (&Metadata{HeadCount: 32}).KVHeadsAt(0); got != 32 {
		t.Fatalf("fallback = %d", got)
	}
}

// The default scan root is the app folder, which also holds the runtime: the
// walk must not crawl bin/, build/ or system metadata.
func TestDiscoverSkipsRuntimeAndSystemFolders(t *testing.T) {
	dir := t.TempDir()
	for _, sub := range []string{"bin", "build", "System Volume Information", "models"} {
		if err := os.MkdirAll(filepath.Join(dir, sub), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for _, p := range []string{
		filepath.Join(dir, "bin", "hidden.gguf"),
		filepath.Join(dir, "build", "hidden.gguf"),
		filepath.Join(dir, "models", "wanted.gguf"),
	} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	cands, err := Discover(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(cands) != 1 || filepath.Base(cands[0].Path) != "wanted.gguf" {
		t.Fatalf("candidates = %+v, want only models/wanted.gguf", cands)
	}
}

func TestDiscoverToleratesMissingDirectory(t *testing.T) {
	if _, err := Discover(filepath.Join(t.TempDir(), "absent")); err != nil {
		t.Fatalf("missing models dir must not fail: %v", err)
	}
}

func TestDiscoverSkipsLaterShards(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{
		"a-00001-of-00003.gguf",
		"a-00002-of-00003.gguf",
		"a-00003-of-00003.gguf",
		"plain.gguf",
		"notes.txt",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cands, err := Discover(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(cands) != 2 {
		t.Fatalf("got %d candidates, want 2: %+v", len(cands), cands)
	}
}

func TestParseScalarKVHeadsSWAAndRecurrent(t *testing.T) {
	// Scalar head_count_kv + SWA pattern (Spark / Gemma3 style).
	b := newBuilder()
	b.addString("general.architecture", "spark2_5")
	b.addUint32("spark2_5.block_count", 4)
	b.addUint32("spark2_5.attention.head_count", 16)
	b.addUint32("spark2_5.attention.head_count_kv", 4)
	b.addUint32("spark2_5.attention.sliding_window", 512)
	b.addBoolArray("spark2_5.attention.sliding_window_pattern", []bool{true, true, true, false})
	b.addTensor("token_embd.weight", []uint64{256, 1024}, 0) // F32: 1 MiB
	b.addTensor("output.weight", []uint64{256, 1024}, 0)     // F32: 1 MiB

	m, err := Parse(b.write(t))
	if err != nil {
		t.Fatalf("Parse SWA: %v", err)
	}
	if got := m.KVHeadsAt(0); got != 4 {
		t.Fatalf("KVHeadsAt(0) = %d, want 4", got)
	}
	if !m.PerLayerKV() {
		t.Fatal("expected PerLayerKV = true for SWA model")
	}
	if !m.IsSWALayer(0) || !m.IsSWALayer(2) || m.IsSWALayer(3) {
		t.Fatalf("unexpected SWA pattern: 0=%v 2=%v 3=%v", m.IsSWALayer(0), m.IsSWALayer(2), m.IsSWALayer(3))
	}
	if m.EmbedBytes != 256*1024*4 {
		t.Fatalf("EmbedBytes = %d, want %d", m.EmbedBytes, 256*1024*4)
	}
	if m.GlobalBytes != 2*256*1024*4 {
		t.Fatalf("GlobalBytes = %d, want %d", m.GlobalBytes, 2*256*1024*4)
	}

	// Recurrent layers (Qwen3.5 style: every 4th layer full attention).
	q := newBuilder()
	q.addString("general.architecture", "qwen35")
	q.addUint32("qwen35.block_count", 4)
	q.addUint32("qwen35.attention.head_count", 16)
	q.addUint32("qwen35.attention.head_count_kv", 2)
	q.addUint32("qwen35.full_attention_interval", 4)
	q.addBoolArray("qwen35.attention.recurrent_layers", []bool{true, true, true, false})

	qm, err := Parse(q.write(t))
	if err != nil {
		t.Fatalf("Parse recurrent: %v", err)
	}
	for l := 0; l < 3; l++ {
		if qm.IsKVLayer(l) {
			t.Fatalf("layer %d should be recurrent (no KV cache)", l)
		}
	}
	if !qm.IsKVLayer(3) {
		t.Fatal("layer 3 should be full attention KV layer")
	}
}

func TestSkipArrayOverflowRejected(t *testing.T) {
	b := newBuilder()
	b.kVKey("tokenizer.ggml.tokens") // keepKey returns false -> takes skipArray path
	b.u32(9)                         // array
	b.u32(10)                        // uint64 elements (8 bytes each)
	b.u64((1 << 62))                 // count * 8 overflows int64!
	if _, err := Parse(b.write(t)); err == nil {
		t.Fatal("expected error for overflowing array skip count")
	}
}
