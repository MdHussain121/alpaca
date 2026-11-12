package mcp

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

// ServerConfig configures one stdio MCP server in mcp.toml.
type ServerConfig struct {
	Command       string            `toml:"command"`
	Args          []string          `toml:"args"`
	Env           map[string]string `toml:"env"`
	Enabled       *bool             `toml:"enabled"`
	TimeoutSec    int               `toml:"timeout_s"`
	Class         string            `toml:"class"` // "net" (default) or "exec"
	DisabledTools []string          `toml:"disabled_tools"`
}

// Config holds all configured stdio MCP servers loaded from mcp.toml.
type Config struct {
	Servers map[string]ServerConfig `toml:"servers"`
}

// IsEnabled returns true unless enabled = false is set in mcp.toml.
func (s ServerConfig) IsEnabled() bool {
	if s.Enabled == nil {
		return true
	}
	return *s.Enabled
}

// Timeout returns the per-call timeout, defaulting to 30s.
func (s ServerConfig) Timeout() time.Duration {
	if s.TimeoutSec <= 0 {
		return 30 * time.Second
	}
	return time.Duration(s.TimeoutSec) * time.Second
}

// NormalizedClass returns "exec" if configured as "exec", otherwise "net".
func (s ServerConfig) NormalizedClass() string {
	if strings.EqualFold(strings.TrimSpace(s.Class), "exec") {
		return "exec"
	}
	return "net"
}

// LoadConfig reads mcp.toml. Missing files return an empty Config without error.
func LoadConfig(path string) (Config, error) {
	cfg := Config{Servers: make(map[string]ServerConfig)}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return cfg, fmt.Errorf("read %s: %w", path, err)
	}
	if _, err := toml.Decode(string(data), &cfg); err != nil {
		return Config{Servers: make(map[string]ServerConfig)}, fmt.Errorf("parse %s: %w", path, err)
	}
	if cfg.Servers == nil {
		cfg.Servers = make(map[string]ServerConfig)
	}
	return cfg, nil
}

// OrderedNames returns server names sorted alphabetically.
func (c Config) OrderedNames() []string {
	out := make([]string, 0, len(c.Servers))
	for k := range c.Servers {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// SanitizeToolSegment makes a server or tool name safe to compose into `server.tool`.
func SanitizeToolSegment(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "unnamed"
	}
	var b strings.Builder
	for _, r := range s {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}

// ResolveCommand resolves a relative command path against rootDir if present there,
// otherwise leaving command as-is so exec.LookPath handles PATH lookup.
func ResolveCommand(rootDir, cmd string) string {
	cmd = strings.TrimSpace(cmd)
	if cmd == "" || filepath.IsAbs(cmd) {
		return cmd
	}
	if rootDir != "" {
		cand := filepath.Join(rootDir, cmd)
		if info, err := os.Stat(cand); err == nil && !info.IsDir() {
			return cand
		}
	}
	return cmd
}

// RedactSecrets scrubs configured secret env values (length >= 4) from text
// so server environment secrets never leak to logs or model outputs.
func RedactSecrets(env map[string]string, text string) string {
	if len(env) == 0 || text == "" {
		return text
	}
	vals := make([]string, 0, len(env))
	for _, v := range env {
		if len(v) >= 4 {
			vals = append(vals, v)
		}
	}
	sort.Slice(vals, func(i, j int) bool { return len(vals[i]) > len(vals[j]) })
	out := text
	for _, v := range vals {
		out = strings.ReplaceAll(out, v, "[REDACTED]")
	}
	return out
}
