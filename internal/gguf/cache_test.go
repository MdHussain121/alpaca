package gguf

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestCacheHitAndInvalidation(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "tiny-NVFP4.gguf")
	b := newBuilder()
	b.addString("general.architecture", "llama")
	b.addString("general.name", "Tiny NVFP4")
	b.addUint32("general.file_type", 39)
	b.addUint32("llama.block_count", 2)
	b.addUint32("llama.embedding_length", 64)
	b.addUint32("llama.attention.head_count", 4)
	b.addUint32("llama.attention.head_count_kv", 2)
	b.addTensor("token_embd.weight", []uint64{64, 16}, 43)    // F8_E4M3 -> 1024 B
	b.addTensor("blk.0.attn_q.weight", []uint64{64, 64}, 40)  // NVFP4 -> 64*64/64*36 = 2304 B
	b.addTensor("blk.1.attn_k.weight", []uint64{128, 16}, 41) // Q1_0 -> 2048/128*18 = 288 B
	b.addTensor("blk.1.attn_v.weight", []uint64{64, 16}, 42)  // Q2_0 -> 1024/64*18 = 288 B
	b.writeAt(t, modelPath)

	cacheFile := filepath.Join(dir, ".gguf-cache.json")
	c := LoadCache(cacheFile)

	m1, hit1, err := ParseCached(c, modelPath)
	if err != nil || hit1 || m1 == nil {
		t.Fatalf("first ParseCached = (%v, hit=%v, err=%v)", m1, hit1, err)
	}
	if len(m1.UnknownTensorTypes) != 0 {
		t.Fatalf("unexpected UnknownTensorTypes for 40-43: %v", m1.UnknownTensorTypes)
	}
	if m1.Quant != "NVFP4" {
		t.Fatalf("Quant = %q, want NVFP4", m1.Quant)
	}
	if want := uint64(1024 + 2304 + 288 + 288); m1.TotalBytes != want {
		t.Fatalf("TotalBytes = %d, want %d", m1.TotalBytes, want)
	}
	if err := c.SaveIfDirty(); err != nil {
		t.Fatalf("SaveIfDirty: %v", err)
	}

	c2 := LoadCache(cacheFile)
	m2, hit2, err := ParseCached(c2, modelPath)
	if err != nil || !hit2 || m2 == nil || m2.Name != m1.Name || m2.TotalBytes != m1.TotalBytes {
		t.Fatalf("second ParseCached = (%v, hit=%v, err=%v)", m2, hit2, err)
	}

	future := time.Now().Add(2 * time.Second)
	if err := os.Chtimes(modelPath, future, future); err != nil {
		t.Fatal(err)
	}
	m3, hit3, err := ParseCached(c2, modelPath)
	if err != nil || hit3 || m3 == nil {
		t.Fatalf("post-mtime ParseCached = (%v, hit=%v, err=%v)", m3, hit3, err)
	}
}

func TestDiscoverDepthLimitAndTypes40to43(t *testing.T) {
	for id, name := range map[uint32]string{40: "NVFP4", 41: "Q1_0", 42: "Q2_0", 43: "F8_E4M3"} {
		if _, ok := typeInfo[id]; !ok {
			t.Fatalf("missing typeInfo[%d] (%s)", id, name)
		}
		if typeNames[id] != name {
			t.Fatalf("typeNames[%d] = %q, want %q", id, typeNames[id], name)
		}
	}

	root := t.TempDir()
	shallow := filepath.Join(root, "a", "b", "c", "ok.gguf")
	deep := filepath.Join(root, "a", "b", "c", "d", "too-deep.gguf")
	for _, p := range []string{shallow, deep} {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got, err := Discover(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || filepath.Base(got[0].Path) != "ok.gguf" {
		t.Fatalf("Discover depth=3 got %+v, want only ok.gguf", got)
	}
}

func TestShardPathsEnumeratesSplitShards(t *testing.T) {
	dir := t.TempDir()
	s1 := filepath.Join(dir, "big-00001-of-00003.gguf")
	s2 := filepath.Join(dir, "big-00002-of-00003.gguf")
	s3 := filepath.Join(dir, "big-00003-of-00003.gguf")
	for _, p := range []string{s1, s2, s3} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got := ShardPaths(s1)
	if len(got) != 3 || got[0] != s1 || got[1] != s2 || got[2] != s3 {
		t.Fatalf("ShardPaths = %v", got)
	}
	single := filepath.Join(dir, "single.gguf")
	if gotSingle := ShardPaths(single); len(gotSingle) != 1 || gotSingle[0] != single {
		t.Fatalf("single ShardPaths = %v", gotSingle)
	}
}

func TestCacheBenchEnvIfSet(t *testing.T) {
	dir := os.Getenv("GGUF_SCAN_BENCH_DIR")
	if dir == "" {
		return
	}
	cacheFile := filepath.Join(t.TempDir(), ".gguf-cache.json")
	t0 := time.Now()
	cands, err := DiscoverWithDepth(dir, DefaultMaxDiscoverDepth)
	if err != nil {
		t.Fatal(err)
	}
	c := LoadCache(cacheFile)
	for _, cd := range cands {
		if _, _, perr := ParseCached(c, cd.Path); perr != nil {
			t.Fatal(perr)
		}
	}
	_ = c.SaveIfDirty()
	cold := time.Since(t0)

	t1 := time.Now()
	cands2, err := DiscoverWithDepth(dir, DefaultMaxDiscoverDepth)
	if err != nil {
		t.Fatal(err)
	}
	c2 := LoadCache(cacheFile)
	hits := 0
	for _, cd := range cands2 {
		_, hit, perr := ParseCached(c2, cd.Path)
		if perr != nil {
			t.Fatal(perr)
		}
		if hit {
			hits++
		}
	}
	warm := time.Since(t1)
	t.Logf("dir=%s models=%d cold=%v warm=%v hits=%d/%d", dir, len(cands), cold, warm, hits, len(cands2))
}
