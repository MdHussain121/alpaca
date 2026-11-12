package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/MdHussain121/alpaca/internal/agent"
	"github.com/MdHussain121/alpaca/internal/runner"
	"github.com/MdHussain121/alpaca/internal/session"
)

const (
	ManyToolsThreshold      = 12
	HighSchemaTokensWarning = 1500
	defaultMaxRestarts      = 3
	defaultRestartWindow    = 60 * time.Second
)

// ToolStatus summarizes a single discovered MCP tool and its toggle state.
type ToolStatus struct {
	Server       string
	RawName      string
	FullName     string // server.tool
	Description  string
	Enabled      bool
	SchemaTokens int
	Schema       ToolSchema
}

// ServerStatus summarizes one configured MCP server (env secret values excluded).
type ServerStatus struct {
	Name             string
	Command          string
	Enabled          bool
	Running          bool
	State            string
	LastError        string
	RestartCount     int
	Class            string
	EnvKeyCount      int
	ToolCount        int
	EnabledToolCount int
	SchemaTokens     int
	WarnManyTools    bool
	Tools            []ToolStatus
}

// Summary aggregates MCP server and tool state for /mcp, /tools, and warnings.
type Summary struct {
	ConfigPath        string
	ConfigExists      bool
	Servers           []ServerStatus
	TotalServers      int
	RunningServers    int
	TotalTools        int
	TotalEnabledTools int
	TotalSchemaTokens int
	Warning           string
}

type managedServer struct {
	name          string
	cfg           ServerConfig
	enabled       bool
	disabledTools map[string]bool // keyed by RawName
	client        *Client
	gen           uint64
	state         string
	lastError     string
	restarts      []time.Time
	registered    []string
}

// Manager owns stdio MCP server lifecycles and syncs discovered tools with agent.Registry.
type Manager struct {
	mu           sync.Mutex
	rootDir      string
	workDir      string
	configPath   string
	registry     *agent.Registry
	servers      map[string]*managedServer
	order        []string
	shuttingDown bool
	genSeq       uint64
	maxRestarts  int
	onChange     func()
}

// NewManager creates an MCP Manager bound to rootDir/mcp.toml and reg.
func NewManager(rootDir, workDir, configPath string, reg *agent.Registry) *Manager {
	return &Manager{
		rootDir:     rootDir,
		workDir:     workDir,
		configPath:  configPath,
		registry:    reg,
		servers:     make(map[string]*managedServer),
		maxRestarts: defaultMaxRestarts,
	}
}

// SetWorkDir updates the working directory used for newly started MCP child servers.
func (m *Manager) SetWorkDir(workDir string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.workDir = workDir
}

// SetOnChange registers a callback invoked asynchronously when background restarts finish.
func (m *Manager) SetOnChange(fn func()) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onChange = fn
}

// Reload reads mcp.toml, stops previous servers, starts enabled servers, and syncs Registry.
func (m *Manager) Reload(ctx context.Context) (Summary, error) {
	cfg, err := LoadConfig(m.configPath)

	m.mu.Lock()
	m.shuttingDown = false
	m.genSeq++
	gen := m.genSeq
	oldClients := make([]*Client, 0, len(m.servers))
	for _, s := range m.servers {
		if s.client != nil {
			oldClients = append(oldClients, s.client)
		}
		m.unregisterServerLocked(s)
	}
	m.servers = make(map[string]*managedServer)
	m.order = nil
	workDir := m.workDir
	rootDir := m.rootDir
	m.mu.Unlock()

	for _, c := range oldClients {
		_ = c.Close()
	}

	if err != nil {
		return m.Summary(), err
	}

	names := cfg.OrderedNames()
	type startTarget struct {
		name string
		srv  *managedServer
	}
	var targets []startTarget

	m.mu.Lock()
	for _, name := range names {
		sc := cfg.Servers[name]
		disabled := make(map[string]bool, len(sc.DisabledTools))
		for _, dt := range sc.DisabledTools {
			disabled[strings.TrimSpace(dt)] = true
		}
		srv := &managedServer{
			name:          name,
			cfg:           sc,
			enabled:       sc.IsEnabled(),
			disabledTools: disabled,
			gen:           gen,
			state:         "disabled",
		}
		if srv.enabled {
			srv.state = "starting"
			targets = append(targets, startTarget{name: name, srv: srv})
		}
		m.servers[name] = srv
		m.order = append(m.order, name)
	}
	m.mu.Unlock()

	for _, tgt := range targets {
		_ = m.startServer(ctx, tgt.srv, rootDir, workDir, gen)
	}
	return m.Summary(), nil
}

func (m *Manager) startServer(ctx context.Context, srv *managedServer, rootDir, workDir string, gen uint64) error {
	name := srv.name
	sc := srv.cfg
	opts := ClientOptions{
		Name:    name,
		RootDir: rootDir,
		WorkDir: workDir,
		Config:  sc,
		OnExit: func(exitErr error) {
			m.handleUnexpectedExit(name, gen, exitErr)
		},
	}
	c, err := StartClient(ctx, opts)

	m.mu.Lock()
	defer m.mu.Unlock()
	cur, ok := m.servers[name]
	if !ok || cur != srv || cur.gen != gen || m.shuttingDown || !cur.enabled {
		if c != nil {
			go c.Close()
		}
		return err
	}
	if err != nil {
		cur.client = nil
		cur.state = "error"
		cur.lastError = RedactSecrets(sc.Env, err.Error())
		m.unregisterServerLocked(cur)
		return err
	}
	cur.client = c
	cur.state = "running"
	cur.lastError = ""
	m.syncServerToolsLocked(cur)
	return nil
}

func (m *Manager) handleUnexpectedExit(name string, gen uint64, exitErr error) {
	m.mu.Lock()
	srv, ok := m.servers[name]
	if !ok || srv.gen != gen || m.shuttingDown || !srv.enabled {
		m.mu.Unlock()
		return
	}
	srv.client = nil
	m.unregisterServerLocked(srv)

	now := time.Now()
	cutoff := now.Add(-defaultRestartWindow)
	var recent []time.Time
	for _, ts := range srv.restarts {
		if ts.After(cutoff) {
			recent = append(recent, ts)
		}
	}
	if len(recent) >= m.maxRestarts {
		srv.restarts = recent
		srv.state = "crashed (limit reached)"
		msg := "exited unexpectedly"
		if exitErr != nil {
			msg = RedactSecrets(srv.cfg.Env, exitErr.Error())
		}
		srv.lastError = fmt.Sprintf("restart limit (%d) reached: %s", m.maxRestarts, msg)
		cb := m.onChange
		m.mu.Unlock()
		if cb != nil {
			cb()
		}
		return
	}
	attempt := len(recent)
	recent = append(recent, now)
	srv.restarts = recent
	srv.state = fmt.Sprintf("restarting (%d/%d)", attempt+1, m.maxRestarts)
	if exitErr != nil {
		srv.lastError = RedactSecrets(srv.cfg.Env, exitErr.Error())
	}
	rootDir := m.rootDir
	workDir := m.workDir
	m.genSeq++
	nextGen := m.genSeq
	srv.gen = nextGen
	cb := m.onChange
	m.mu.Unlock()

	if cb != nil {
		cb()
	}

	backoff := time.Duration(200*(1<<attempt)) * time.Millisecond
	if backoff > 2*time.Second {
		backoff = 2 * time.Second
	}
	time.Sleep(backoff)

	m.mu.Lock()
	if m.shuttingDown {
		m.mu.Unlock()
		return
	}
	srv2, ok := m.servers[name]
	if !ok || srv2 != srv || srv2.gen != nextGen || !srv2.enabled {
		m.mu.Unlock()
		return
	}
	m.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_ = m.startServer(ctx, srv, rootDir, workDir, nextGen)

	m.mu.Lock()
	cb2 := m.onChange
	m.mu.Unlock()
	if cb2 != nil {
		cb2()
	}
}

func (m *Manager) unregisterServerLocked(srv *managedServer) {
	if m.registry == nil || len(srv.registered) == 0 {
		srv.registered = nil
		return
	}
	for _, fullName := range srv.registered {
		m.registry.Unregister(fullName)
	}
	srv.registered = nil
}

func (m *Manager) syncServerToolsLocked(srv *managedServer) {
	m.unregisterServerLocked(srv)
	if m.registry == nil || srv.client == nil || !srv.enabled {
		return
	}
	cls := agent.ClassNet
	if srv.cfg.NormalizedClass() == "exec" {
		cls = agent.ClassExec
	}
	srvSeg := SanitizeToolSegment(srv.name)
	usedNames := make(map[string]bool)

	for _, ts := range srv.client.Tools() {
		if srv.disabledTools[ts.Name] {
			continue
		}
		toolSeg := SanitizeToolSegment(ts.Name)
		fullName := srvSeg + "." + toolSeg
		if usedNames[fullName] {
			fullName = fmt.Sprintf("%s.%s_%d", srvSeg, toolSeg, len(usedNames)+1)
		}
		usedNames[fullName] = true

		desc := strings.TrimSpace(ts.Description)
		if desc == "" {
			desc = fmt.Sprintf("MCP tool %s from server %s", ts.Name, srv.name)
		}
		desc = RedactSecrets(srv.cfg.Env, desc)

		rawName := ts.Name
		srvName := srv.name
		params := ts.InputSchema
		if params == nil {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}

		spec := agent.ToolSpec{
			Name:        fullName,
			Group:       "mcp",
			Class:       cls,
			Description: desc,
			Parameters:  params,
			InspectPerm: func(tctx agent.ToolContext, args map[string]any) (agent.PermissionInput, string, string, int, int, error) {
				summary := fmt.Sprintf("mcp %s(%s)", fullName, compactArgsPreview(args))
				return agent.PermissionInput{
					Tool:          fullName,
					CanonicalArgs: args,
					Class:         cls,
					InWorkspace:   false,
					Tainted:       tctx.Tainted,
					Summary:       summary,
				}, summary, "", 0, 0, nil
			},
			Execute: func(ctx context.Context, tctx agent.ToolContext, args map[string]any) agent.ToolExecResult {
				out, isErr, callErr := m.callTool(ctx, srvName, rawName, args)
				if callErr != nil {
					if ctx.Err() != nil {
						return agent.ToolExecResult{
							Output:      "MCP tool call interrupted.",
							Summary:     "interrupted",
							IsError:     true,
							Interrupted: true,
						}
					}
					return agent.ToolExecResult{
						Output:  callErr.Error(),
						Summary: "error",
						IsError: true,
					}
				}
				return agent.ToolExecResult{
					Output:  out,
					Summary: fmt.Sprintf("ok (%d B)", len(out)),
					IsError: isErr,
				}
			},
		}
		m.registry.Register(spec)
		srv.registered = append(srv.registered, fullName)
	}
}

func compactArgsPreview(args map[string]any) string {
	if len(args) == 0 {
		return ""
	}
	b, err := json.Marshal(args)
	if err != nil {
		return "..."
	}
	s := string(b)
	if len(s) > 72 {
		return s[:69] + "..."
	}
	return s
}

func (m *Manager) callTool(ctx context.Context, serverName, rawToolName string, args map[string]any) (string, bool, error) {
	m.mu.Lock()
	srv, ok := m.servers[serverName]
	var c *Client
	if ok && srv.enabled {
		c = srv.client
	}
	m.mu.Unlock()

	if c == nil {
		return "", true, fmt.Errorf("mcp server %q is not running", serverName)
	}
	return c.CallTool(ctx, rawToolName, args)
}

// ToggleServer enables or disables an entire configured MCP server cleanly.
func (m *Manager) ToggleServer(ctx context.Context, name string) (bool, error) {
	m.mu.Lock()
	srv, ok := m.servers[name]
	if !ok {
		m.mu.Unlock()
		return false, fmt.Errorf("unknown MCP server %q", name)
	}
	next := !srv.enabled
	srv.enabled = next
	m.genSeq++
	gen := m.genSeq
	srv.gen = gen
	srv.restarts = nil
	oldClient := srv.client
	if !next {
		srv.client = nil
		srv.state = "disabled"
		srv.lastError = ""
		m.unregisterServerLocked(srv)
		m.mu.Unlock()
		if oldClient != nil {
			_ = oldClient.Close()
		}
		return false, nil
	}
	srv.state = "starting"
	srv.lastError = ""
	rootDir := m.rootDir
	workDir := m.workDir
	m.mu.Unlock()

	if oldClient != nil {
		_ = oldClient.Close()
	}
	err := m.startServer(ctx, srv, rootDir, workDir, gen)
	return true, err
}

// ToggleTool enables or disables an individual tool belonging to serverName.
func (m *Manager) ToggleTool(serverName, rawToolName string) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	srv, ok := m.servers[serverName]
	if !ok {
		return false, fmt.Errorf("unknown MCP server %q", serverName)
	}
	if srv.disabledTools == nil {
		srv.disabledTools = make(map[string]bool)
	}
	nextEnabled := srv.disabledTools[rawToolName]
	if nextEnabled {
		delete(srv.disabledTools, rawToolName)
	} else {
		srv.disabledTools[rawToolName] = true
	}
	m.syncServerToolsLocked(srv)
	return nextEnabled, nil
}

// Summary builds a thread-safe snapshot of all configured MCP servers, schema token costs,
// and warnings when tool counts become large for small local models.
func (m *Manager) Summary() Summary {
	m.mu.Lock()
	defer m.mu.Unlock()

	sum := Summary{
		ConfigPath: m.configPath,
		Servers:    make([]ServerStatus, 0, len(m.order)),
	}
	if _, err := LoadConfig(m.configPath); err == nil {
		sum.ConfigExists = len(m.order) > 0
	}

	for _, name := range m.order {
		srv := m.servers[name]
		if srv == nil {
			continue
		}
		st := ServerStatus{
			Name:         srv.name,
			Command:      srv.cfg.Command,
			Enabled:      srv.enabled,
			Running:      srv.client != nil && srv.state == "running",
			State:        srv.state,
			LastError:    srv.lastError,
			RestartCount: len(srv.restarts),
			Class:        srv.cfg.NormalizedClass(),
			EnvKeyCount:  len(srv.cfg.Env),
		}
		if srv.client != nil {
			tools := srv.client.Tools()
			srvSeg := SanitizeToolSegment(srv.name)
			st.ToolCount = len(tools)
			st.Tools = make([]ToolStatus, 0, len(tools))
			for _, ts := range tools {
				en := !srv.disabledTools[ts.Name] && srv.enabled
				tok := estimateMCPToolTokens(srvSeg+"."+SanitizeToolSegment(ts.Name), ts)
				if en {
					st.EnabledToolCount++
					st.SchemaTokens += tok
				}
				st.Tools = append(st.Tools, ToolStatus{
					Server:       srv.name,
					RawName:      ts.Name,
					FullName:     srvSeg + "." + SanitizeToolSegment(ts.Name),
					Description:  RedactSecrets(srv.cfg.Env, ts.Description),
					Enabled:      en,
					SchemaTokens: tok,
					Schema:       ts,
				})
			}
			sort.Slice(st.Tools, func(i, j int) bool {
				return st.Tools[i].RawName < st.Tools[j].RawName
			})
		}
		if st.ToolCount >= ManyToolsThreshold || st.SchemaTokens >= HighSchemaTokensWarning {
			st.WarnManyTools = true
		}
		sum.TotalServers++
		if st.Running {
			sum.RunningServers++
		}
		sum.TotalTools += st.ToolCount
		sum.TotalEnabledTools += st.EnabledToolCount
		sum.TotalSchemaTokens += st.SchemaTokens
		sum.Servers = append(sum.Servers, st)
	}

	if sum.TotalEnabledTools >= ManyToolsThreshold || sum.TotalSchemaTokens >= HighSchemaTokensWarning {
		sum.Warning = fmt.Sprintf(
			"Warning: %d MCP tools active (~%d schema tok) — small models degrade with long tool lists.",
			sum.TotalEnabledTools, sum.TotalSchemaTokens,
		)
	}
	return sum
}

func estimateMCPToolTokens(fullName string, ts ToolSchema) int {
	def := runner.ToolDef{
		Type: "function",
		Function: runner.ToolFunctionDef{
			Name:        fullName,
			Description: ts.Description,
			Parameters:  ts.InputSchema,
		},
	}
	b, _ := json.Marshal(def)
	return session.EstimateTokensText(string(b))
}

// Shutdown terminates all running MCP child servers and their Job Objects cleanly.
func (m *Manager) Shutdown() {
	m.mu.Lock()
	m.shuttingDown = true
	m.genSeq++
	clients := make([]*Client, 0, len(m.servers))
	for _, srv := range m.servers {
		if srv.client != nil {
			clients = append(clients, srv.client)
			srv.client = nil
		}
		m.unregisterServerLocked(srv)
	}
	m.mu.Unlock()

	var wg sync.WaitGroup
	for _, c := range clients {
		wg.Add(1)
		go func(cl *Client) {
			defer wg.Done()
			_ = cl.Close()
		}(c)
	}
	wg.Wait()
}
