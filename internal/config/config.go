// Package config persists global settings and named presets.
package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/BurntSushi/toml"

	"github.com/MdHussain121/alpaca/internal/flags"
)

// WebConfig controls keyless web search and fetch limits.
type WebConfig struct {
	SearXNGURL    string `toml:"searxng_url"`
	MaxFetchBytes int    `toml:"max_fetch_bytes"`
}

// SamplingProfile defines a named generation sampling preset applied via /profile.
type SamplingProfile struct {
	Description string  `toml:"description"`
	Temp        float64 `toml:"temp"`
	TopP        float64 `toml:"top_p"`
	TopK        int     `toml:"top_k"`
	MinP        float64 `toml:"min_p"`
}

// Config is the contents of config.toml.
type Config struct {
	Theme                  string `toml:"theme"`
	Backend                string `toml:"backend"`
	Port                   int    `toml:"port"`
	ModelsDir              string `toml:"models_dir"`
	ScanDepth              int    `toml:"scan_depth"`
	ASCII                  bool   `toml:"ascii"`
	CompactionThresholdPct int    `toml:"compaction_threshold_pct"`
	LongGenBell            bool   `toml:"long_gen_bell"`
	ActiveProfile          string `toml:"active_profile"`
	DisableSystemPrompt    bool   `toml:"disable_system_prompt"`
	// ExtraModels are explicit .gguf paths outside ModelsDir, e.g. on another
	// drive. They are scanned alongside the folder.
	ExtraModels []string                   `toml:"extra_models"`
	Web         WebConfig                  `toml:"web"`
	Profiles    map[string]SamplingProfile `toml:"profiles"`
	LastUsed    map[string]flags.Settings  `toml:"last_used"`

	root string
}

// DefaultProfiles returns the built-in sampling profiles.
func DefaultProfiles() map[string]SamplingProfile {
	return map[string]SamplingProfile{
		"coding": {
			Description: "Deterministic code editing and tool calls (temp 0.20, top-p 0.90)",
			Temp:        0.20,
			TopP:        0.90,
			TopK:        40,
			MinP:        0.05,
		},
		"precise": {
			Description: "Factual, low-variance reasoning and strict JSON (temp 0.05, top-p 0.85)",
			Temp:        0.05,
			TopP:        0.85,
			TopK:        20,
			MinP:        0.08,
		},
		"creative": {
			Description: "Brainstorming and expressive writing (temp 0.85, top-p 0.95)",
			Temp:        0.85,
			TopP:        0.95,
			TopK:        64,
			MinP:        0.02,
		},
	}
}

// ResolvedProfiles merges built-in profiles with any user-defined overrides from config.toml.
func (c *Config) ResolvedProfiles() map[string]SamplingProfile {
	out := DefaultProfiles()
	if c != nil {
		for k, v := range c.Profiles {
			name := strings.ToLower(strings.TrimSpace(k))
			if name != "" {
				out[name] = v
			}
		}
	}
	return out
}

// ProfileNames returns profile names in stable UI order (built-ins first, custom sorted).
func (c *Config) ProfileNames() []string {
	m := c.ResolvedProfiles()
	ordered := make([]string, 0, len(m))
	for _, b := range []string{"coding", "precise", "creative"} {
		if _, ok := m[b]; ok {
			ordered = append(ordered, b)
		}
	}
	var custom []string
	for k := range m {
		if k != "coding" && k != "precise" && k != "creative" {
			custom = append(custom, k)
		}
	}
	sort.Strings(custom)
	return append(ordered, custom...)
}

// EffectiveCompactionPct returns the auto-compaction threshold percentage (default 80).
func (c *Config) EffectiveCompactionPct() int {
	if c == nil || c.CompactionThresholdPct <= 0 || c.CompactionThresholdPct > 98 {
		return 80
	}
	return c.CompactionThresholdPct
}

// Preset is a saved configuration for a model, stored in presets/<id>.toml.
type Preset struct {
	Name  string `toml:"name"`
	Model string `toml:"model"`
	flags.Settings

	ID string `toml:"-"`
}

// Default returns the config used when no config.toml exists.
func Default() *Config {
	return &Config{
		Theme:                  "mocha",
		Backend:                "auto",
		Port:                   8080,
		CompactionThresholdPct: 80,
		ActiveProfile:          "coding",
		// "." means the folder alpaca.exe lives in, so models dropped next to the
		// exe are found with no setup. Set models_dir to a folder to narrow it.
		ModelsDir: ".",
		ScanDepth: 3,
		LastUsed:  map[string]flags.Settings{},
	}
}

const configName = "config.toml"

// Load reads config.toml from root (falling back to legacy llm.toml or pllama.toml),
// creating a default config.toml if none is present.
func Load(root string) *Config {
	cfg := Default()
	cfg.root = root
	path := filepath.Join(root, configName)
	if _, err := os.Stat(path); err != nil {
		if legacyPath := findLegacyConfigFile(root); legacyPath != "" {
			if _, derr := toml.DecodeFile(legacyPath, cfg); derr == nil {
				cfg.root = root
				normalizeLoadedConfig(cfg)
				_ = cfg.Save()
				return cfg
			}
		}
		_ = cfg.Save()
		return cfg
	}
	if _, err := toml.DecodeFile(path, cfg); err != nil {
		// A corrupt config must not stop the launcher.
		cfg = Default()
		cfg.root = root
		return cfg
	}
	cfg.root = root
	normalizeLoadedConfig(cfg)
	return cfg
}

func normalizeLoadedConfig(cfg *Config) {
	if cfg.LastUsed == nil {
		cfg.LastUsed = map[string]flags.Settings{}
	}
	if cfg.ModelsDir == "" {
		cfg.ModelsDir = "."
	}
	if cfg.ScanDepth <= 0 {
		cfg.ScanDepth = 3
	}
	if cfg.Port == 0 {
		cfg.Port = 8080
	}
	if cfg.Theme == "" {
		cfg.Theme = "mocha"
	}
	if cfg.Backend == "" {
		cfg.Backend = "auto"
	}
}

// Root is the application folder the config lives in.
func (c *Config) Root() string { return c.root }

// AddExtraModel remembers an explicit model path, ignoring duplicates.
func (c *Config) AddExtraModel(path string) {
	for _, p := range c.ExtraModels {
		if strings.EqualFold(p, path) {
			return
		}
	}
	c.ExtraModels = append(c.ExtraModels, path)
}

// Clone returns a copy that can be written from another goroutine without
// racing the UI, which keeps reading the live config.
func (c *Config) Clone() *Config {
	cp := *c
	cp.LastUsed = make(map[string]flags.Settings, len(c.LastUsed))
	for k, v := range c.LastUsed {
		cp.LastUsed[k] = v
	}
	if c.Profiles != nil {
		cp.Profiles = make(map[string]SamplingProfile, len(c.Profiles))
		for k, v := range c.Profiles {
			cp.Profiles[k] = v
		}
	}
	cp.ExtraModels = append([]string(nil), c.ExtraModels...)
	return &cp
}

// ModelsPath resolves the models directory. "." is the app folder itself, so
// the default scan is the folder alpaca.exe was started from.
func (c *Config) ModelsPath() string {
	if filepath.IsAbs(c.ModelsDir) {
		return c.ModelsDir
	}
	return filepath.Join(c.root, c.ModelsDir)
}

// Save writes config.toml atomically.
func (c *Config) Save() error {
	path := filepath.Join(c.root, configName)
	return writeTOML(path, c)
}

// Remember stores the last-used settings for a model file name.
func (c *Config) Remember(model string, s flags.Settings) {
	if c.LastUsed == nil {
		c.LastUsed = map[string]flags.Settings{}
	}
	c.LastUsed[model] = s
}

// Recall returns the stored settings for a model, if any.
func (c *Config) Recall(model string) (flags.Settings, bool) {
	s, ok := c.LastUsed[model]
	return s, ok
}

func writeTOML(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := toml.NewEncoder(f).Encode(v); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// PresetsDir is the folder holding preset files.
func (c *Config) PresetsDir() string { return filepath.Join(c.root, "presets") }

// LogsDir is the folder holding per-launch logs.
func (c *Config) LogsDir() string { return filepath.Join(c.root, "logs") }

// GGUFCachePath returns the persistent metadata cache path next to config.toml.
func (c *Config) GGUFCachePath() string { return filepath.Join(c.root, ".gguf-cache.json") }

// LocalModelsDir returns the local disk directory used by opt-in "Copy model
// to local disk" (prefers %LOCALAPPDATA%\alpaca\models).
func (c *Config) LocalModelsDir() string {
	if base := strings.TrimSpace(os.Getenv("LOCALAPPDATA")); base != "" {
		return filepath.Join(base, "alpaca", "models")
	}
	if cacheDir, err := os.UserCacheDir(); err == nil && cacheDir != "" {
		return filepath.Join(cacheDir, "alpaca", "models")
	}
	return filepath.Join(c.root, "models")
}

// ListPresets loads every preset in presets/, sorted by name.
func (c *Config) ListPresets() ([]Preset, error) {
	dir := c.PresetsDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var out []Preset
	for _, e := range entries {
		if e.IsDir() || !strings.EqualFold(filepath.Ext(e.Name()), ".toml") {
			continue
		}
		var p Preset
		if _, err := toml.DecodeFile(filepath.Join(dir, e.Name()), &p); err != nil {
			continue
		}
		p.ID = strings.TrimSuffix(e.Name(), filepath.Ext(e.Name()))
		if p.Name == "" {
			p.Name = p.ID
		}
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool { return strings.ToLower(out[i].Name) < strings.ToLower(out[j].Name) })
	return out, nil
}

// SavePreset writes a preset. The ID defaults to a slug of the name.
func (c *Config) SavePreset(p Preset) (Preset, error) {
	id := p.ID
	if id == "" {
		id = slug(p.Name)
	}
	if id == "" {
		return p, errors.New("preset needs a name")
	}
	p.ID = id
	if err := writeTOML(filepath.Join(c.PresetsDir(), id+".toml"), p); err != nil {
		return p, err
	}
	return p, nil
}

// DeletePreset removes a preset by ID.
func (c *Config) DeletePreset(id string) error {
	if id == "" {
		return errors.New("missing preset id")
	}
	err := os.Remove(filepath.Join(c.PresetsDir(), id+".toml"))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}

// RenamePreset changes a preset's display name and ID.
func (c *Config) RenamePreset(p Preset, newName string) (Preset, error) {
	if strings.TrimSpace(newName) == "" {
		return p, errors.New("new name is empty")
	}
	oldID := p.ID
	p.Name = newName
	p.ID = slug(newName)
	if p.ID == "" {
		p.ID = oldID
	}
	if p.ID != oldID {
		if err := c.DeletePreset(oldID); err != nil {
			return p, err
		}
	}
	return c.SavePreset(p)
}

func slug(s string) string {
	var b strings.Builder
	lastDash := false
	for _, r := range strings.ToLower(strings.TrimSpace(s)) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
			lastDash = false
		case r == ' ' || r == '-' || r == '_':
			if !lastDash && b.Len() > 0 {
				b.WriteByte('-')
				lastDash = true
			}
		}
	}
	return strings.Trim(b.String(), "-")
}

// BuiltinPresets returns the shipped per-model-class presets for a model.
func BuiltinPresets(modelName string, trainedCtx int, isMoE bool) []Preset {
	base := flags.Default(0, 8080)
	mk := func(name string, mutate func(*flags.Settings)) Preset {
		s := base
		s.AutoKV = false
		mutate(&s)
		s.Normalize()
		return Preset{Name: name, Model: modelName, Settings: s}
	}

	fast := mk("Fast (8K, q8_0)", func(s *flags.Settings) {
		s.Context = 8192
		s.KCache = "q8_0"
		s.VCache = "f16"
	})
	long := mk("Long context (32K)", func(s *flags.Settings) {
		s.Context = 32768
		s.KCache = "q8_0"
		s.VCache = "q8_0"
	})
	maxfit := mk("Max fit", func(s *flags.Settings) {
		s.Context = 8192
		s.KCache = "q8_0"
		s.VCache = "q4_0"
		s.UBatch = 256
		if isMoE {
			s.NCpuMoe = 24
		}
	})

	if trainedCtx > 0 {
		for _, p := range []*Preset{&fast, &long, &maxfit} {
			if p.Context > trainedCtx {
				p.Context = trainedCtx
			}
		}
	}
	return []Preset{fast, long, maxfit}
}

// String describes a preset for the UI.
func (p Preset) String() string {
	return fmt.Sprintf("%s · %dK ctx · %s/%s", p.Name, p.Context/1024, p.KCache, p.VCache)
}
