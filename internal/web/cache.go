package web

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	SearchCacheTTL = 10 * time.Minute
	PageCacheTTL   = 45 * time.Minute
	defaultCacheMB = 32
)

type Cache struct {
	Dir      string
	TTL      time.Duration
	MaxBytes int64

	mu sync.Mutex
}

func NewCache(dir string, ttl time.Duration) *Cache {
	if strings.TrimSpace(dir) == "" {
		dir = DefaultCacheDir()
	}
	if ttl <= 0 {
		ttl = SearchCacheTTL
	}
	return &Cache{Dir: dir, TTL: ttl, MaxBytes: defaultCacheMB << 20}
}

func CacheKey(parts ...string) string {
	h := sha256.New()
	for _, p := range parts {
		h.Write([]byte(strings.TrimSpace(p)))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func (c *Cache) GetJSON(key string, dst any) bool {
	raw, ok := c.GetBytes(key)
	if !ok {
		return false
	}
	return json.Unmarshal(raw, dst) == nil
}

func (c *Cache) PutJSON(key string, v any) {
	raw, err := json.Marshal(v)
	if err != nil {
		return
	}
	c.PutBytes(key, raw)
}

func (c *Cache) GetBytes(key string) ([]byte, bool) {
	if c == nil || c.Dir == "" {
		return nil, false
	}
	path := filepath.Join(c.Dir, key+".json")
	st, err := os.Stat(path)
	if err != nil || (c.TTL > 0 && time.Since(st.ModTime()) > c.TTL) {
		if err == nil {
			_ = os.Remove(path)
		}
		return nil, false
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}
	return raw, true
}

func (c *Cache) PutBytes(key string, raw []byte) {
	if c == nil || c.Dir == "" || len(raw) == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := os.MkdirAll(c.Dir, 0o755); err != nil {
		return
	}
	path := filepath.Join(c.Dir, key+".json")
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		_ = os.Remove(tmp)
		return
	}
	_ = os.Rename(tmp, path)
	c.pruneLocked()
}

func (c *Cache) pruneLocked() {
	if c.MaxBytes <= 0 {
		return
	}
	entries, err := os.ReadDir(c.Dir)
	if err != nil {
		return
	}
	type fileInfo struct {
		path string
		size int64
		mod  time.Time
	}
	var files []fileInfo
	var total int64
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		files = append(files, fileInfo{filepath.Join(c.Dir, e.Name()), info.Size(), info.ModTime()})
		total += info.Size()
	}
	if total <= c.MaxBytes {
		return
	}
	sort.Slice(files, func(i, j int) bool { return files[i].mod.Before(files[j].mod) })
	for _, f := range files {
		if total <= c.MaxBytes {
			break
		}
		if os.Remove(f.path) == nil {
			total -= f.size
		}
	}
}

// DefaultCacheDir prefers the portable launcher folder, then the user cache
// directory. A read-only drive falls back without surfacing an error.
func DefaultCacheDir() string {
	var candidates []string
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(exe), "cache", "web"))
	}
	if ud, err := os.UserCacheDir(); err == nil {
		candidates = append(candidates, filepath.Join(ud, "alpaca", "web"))
	}
	candidates = append(candidates, filepath.Join(os.TempDir(), "alpaca-web-cache"))
	for _, dir := range candidates {
		if os.MkdirAll(dir, 0o755) != nil {
			continue
		}
		probe := filepath.Join(dir, ".write-test")
		if os.WriteFile(probe, []byte("ok"), 0o644) == nil {
			_ = os.Remove(probe)
			return dir
		}
	}
	return ""
}
