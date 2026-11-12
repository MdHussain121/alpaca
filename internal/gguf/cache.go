package gguf

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
)

const cacheSchemaVersion = 2

type shardStamp struct {
	Path        string `json:"path"`
	Size        int64  `json:"size"`
	ModUnixNano int64  `json:"mod_unix_nano"`
}

type cacheEntry struct {
	Size        int64        `json:"size"`
	ModUnixNano int64        `json:"mod_unix_nano"`
	Shards      []shardStamp `json:"shards,omitempty"`
	Meta        *Metadata    `json:"meta"`
}

type cacheDisk struct {
	Version int                    `json:"version"`
	Entries map[string]*cacheEntry `json:"entries"`
}

// Cache persists parsed GGUF headers keyed by file path, size, and mtime so
// slow USB portable drive scans do not re-open multi-GB models on every launch.
type Cache struct {
	mu      sync.RWMutex
	path    string
	dirty   bool
	entries map[string]*cacheEntry
}

// NewCache creates an in-memory GGUF metadata cache backed by path (may be empty).
func NewCache(path string) *Cache {
	return &Cache{
		path:    path,
		entries: make(map[string]*cacheEntry),
	}
}

// LoadCache reads a persistent GGUF metadata cache from path. Missing or corrupt
// cache files safely yield an empty cache.
func LoadCache(path string) *Cache {
	c := NewCache(path)
	if path == "" {
		return c
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return c
	}
	var disk cacheDisk
	if err := json.Unmarshal(b, &disk); err != nil || disk.Version != cacheSchemaVersion {
		return c
	}
	if disk.Entries != nil {
		c.entries = disk.Entries
	}
	return c
}

// ParseCached returns cached Metadata when file size and modification timestamp
// (plus any shards) match, otherwise parses the file and stores it in c.
func ParseCached(c *Cache, path string) (meta *Metadata, hit bool, err error) {
	if c == nil {
		m, perr := Parse(path)
		return m, false, perr
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		absPath = filepath.Clean(path)
	}
	st, err := os.Stat(absPath)
	if err != nil {
		return nil, false, err
	}
	size := st.Size()
	mtime := st.ModTime().UnixNano()

	c.mu.RLock()
	ent := c.entries[absPath]
	c.mu.RUnlock()

	if ent != nil && ent.Meta != nil && ent.Size == size && ent.ModUnixNano == mtime && shardsMatch(ent.Shards) {
		return cloneMetadata(ent.Meta), true, nil
	}

	m, err := Parse(absPath)
	if err != nil {
		return nil, false, err
	}

	newEnt := &cacheEntry{
		Size:        size,
		ModUnixNano: mtime,
		Shards:      collectShardStamps(absPath),
		Meta:        cloneMetadata(m),
	}
	c.mu.Lock()
	c.entries[absPath] = newEnt
	c.dirty = true
	c.mu.Unlock()

	return m, false, nil
}

// SaveIfDirty atomically writes the cache to disk if any entries changed.
func (c *Cache) SaveIfDirty() error {
	if c == nil || c.path == "" {
		return nil
	}
	c.mu.Lock()
	if !c.dirty {
		c.mu.Unlock()
		return nil
	}
	snap := cacheDisk{
		Version: cacheSchemaVersion,
		Entries: make(map[string]*cacheEntry, len(c.entries)),
	}
	for k, v := range c.entries {
		snap.Entries[k] = v
	}
	c.dirty = false
	c.mu.Unlock()

	b, err := json.Marshal(snap)
	if err != nil {
		return err
	}
	dir := filepath.Dir(c.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp := c.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, c.path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func collectShardStamps(path string) []shardStamp {
	siblings := ShardPaths(path)[1:]
	if len(siblings) == 0 {
		return nil
	}
	out := make([]shardStamp, 0, len(siblings))
	for _, sp := range siblings {
		st, err := os.Stat(sp)
		if err != nil {
			continue
		}
		out = append(out, shardStamp{
			Path:        sp,
			Size:        st.Size(),
			ModUnixNano: st.ModTime().UnixNano(),
		})
	}
	return out
}

func shardsMatch(shards []shardStamp) bool {
	for _, s := range shards {
		st, err := os.Stat(s.Path)
		if err != nil || st.Size() != s.Size || st.ModTime().UnixNano() != s.ModUnixNano {
			return false
		}
	}
	return true
}

func cloneMetadata(m *Metadata) *Metadata {
	if m == nil {
		return nil
	}
	cp := *m
	if m.HeadCountKV != nil {
		cp.HeadCountKV = append([]uint64(nil), m.HeadCountKV...)
	}
	if m.SlidingWindowPat != nil {
		cp.SlidingWindowPat = append([]bool(nil), m.SlidingWindowPat...)
	}
	if m.RecurrentLayers != nil {
		cp.RecurrentLayers = append([]bool(nil), m.RecurrentLayers...)
	}
	if m.Layers != nil {
		cp.Layers = append([]LayerBytes(nil), m.Layers...)
	}
	if m.UnknownTensorTypes != nil {
		cp.UnknownTensorTypes = make(map[uint32]bool, len(m.UnknownTensorTypes))
		for k, v := range m.UnknownTensorTypes {
			cp.UnknownTensorTypes[k] = v
		}
	}
	return &cp
}
