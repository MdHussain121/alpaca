package ui

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/MdHussain121/alpaca/internal/agent"
	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/estimate"
	"github.com/MdHussain121/alpaca/internal/flags"
	"github.com/MdHussain121/alpaca/internal/gguf"
	"github.com/MdHussain121/alpaca/internal/hw"
	"github.com/MdHussain121/alpaca/internal/mcp"
	"github.com/MdHussain121/alpaca/internal/runner"
	"github.com/MdHussain121/alpaca/internal/session"
	"github.com/MdHussain121/alpaca/internal/skills"
	"github.com/MdHussain121/alpaca/internal/web"
)

type screen int

const (
	screenLauncher screen = iota
	screenRun
)

type modelEntry struct {
	cand gguf.Candidate
	meta *gguf.Metadata
	err  error
}

// Model is the whole TUI state.
type Model struct {
	root string
	cfg  *config.Config
	sty  styleSet

	width, height int
	ready         bool

	models     []modelEntry
	visible    []int
	cursor     int
	filter     string
	filtering  bool
	scanning   bool
	scanErr    error
	hw         hw.Info
	loadedMeta int

	meta       *gguf.Metadata
	settings   flags.Settings
	resolved   flags.Settings
	est        estimate.Result
	focus      int // 0 = models, 1 = settings
	fieldIdx   int
	cmdPreview string

	presets    []config.Preset
	presetIdx  int
	showPreset bool

	editing bool
	input   textinput.Model

	screen   screen
	proc     *runner.Process
	status   string
	statusDe string
	metrics  runner.Metrics
	logs     []string
	logView  viewport.Model
	spinner  spinner.Model
	healthCh chan error

	webNoticeShown bool
	quitConfirm    bool
	backConfirm    bool
	runQuitConfirm bool
	lanConfirm     bool
	showHelp       bool
	quit           bool

	launching     bool
	healthCancel  context.CancelFunc
	quitCtx       context.Context
	quitCancel    context.CancelFunc
	editingPreset int
	editingField  int
	editingPath   bool
	editingTitle  string

	activePort        int
	hasRecalled       bool
	userEditedThreads bool
	userEditedAutoFit bool
	gpuProbed         bool

	ggufCache         *gguf.Cache
	pendingSelectPath string

	modelDriveKind  string
	modelEstLoad    time.Duration
	launchDriveKind string
	launchETA       time.Duration
	launchAt        time.Time
	launchRetried   bool

	copying      bool
	copyTitle    string
	copyProgress copyProgressMsg
	copyCancel   context.CancelFunc
	copyCh       <-chan copyProgressMsg

	runTab        runTab
	chatMsgs      []chatTurn
	chatView      viewport.Model
	chatInput     textinput.Model
	chatStreaming bool
	chatCancel    context.CancelFunc
	chatGen       uint64
	chatCtxUsed   int
	chatBaseURL   string
	modelCaps     runner.ModelProps
	codeLineNums  bool
	popup         suggestionPopup
	palette       commandPalette

	agentWorkDir       string
	sessionID          string
	agentTurnCounter   int
	sessionTainted     bool
	toolRegistry       *agent.Registry
	permEngine         *agent.PermissionEngine
	checkpoints        *agent.CheckpointStore
	toolSchemaTokens   int
	toolSchemaLive     bool
	execNoticeShown    bool
	permModal          *permModalState
	askModal           *askModalState
	toolsModal         *toolsModalState
	mcpModal           *mcpModalState
	mcpManager         *mcp.Manager
	mcpLoaded          bool
	mcpConfigOv        string
	permsModal         *permsModalState
	skillsModal        *skillsModalState
	skillApproveModal  *skillApproveModalState
	contextModal       *contextModalState
	attachModal        *attachModalState
	sessionsModal      *sessionsModalState
	skillsCatalog      skills.Catalog
	skillListTokens    int
	skillsDirOverride  string
	sessionDoc         *session.SessionDoc
	sessionsDirOv      string
	activeProfile      string
	breakdownCache     *session.BreakdownCache
	genStartTime       time.Time
	attachmentTokens   int
	cachedPrefixTokens int
	webClientOpts      web.ClientOptions

	telemetry        runner.Telemetry
	procRAMBytes     uint64
	procPrivBytes    uint64
	sysRAMAvail      uint64
	sysRAMTotal      uint64
	telemetryPending bool
	telemetryTick    int

	toast          string
	toastUntil     time.Time
	toastSeq       uint64
	startupNotices []string
	version        string
	commit         string
	llamaTag       string
	err            error
}

// --- messages ---

type scanMsg struct {
	entries []modelEntry
	cache   *gguf.Cache
	err     error
}

type metaMsg struct {
	idx  int
	meta *gguf.Metadata
	err  error
}

type hwMsg struct{ info hw.Info }

type gpuMsg struct {
	name              string
	total             uint64
	free              uint64
	ok                bool
	devices           []hw.GPUDevice
	unsupportedReason string
}

type presetMsg struct {
	presets []config.Preset
	err     error
	saved   string
	renamed string
}

type lineMsg string
type logBatchMsg []string
type linesClosedMsg struct{}

type toastMsg string
type toastExpireMsg struct{ seq uint64 }

type healthMsg struct {
	err  error
	proc *runner.Process
}

// pathMsg carries a typed model path or models folder.
type pathMsg struct {
	modelsDir string
	model     string
	err       error
}

// launchMsg carries the result of starting llama-server off the UI thread.
type launchMsg struct {
	proc *runner.Process
	port int
	err  error
}

// stoppedMsg reports a finished (possibly asynchronous) stop.
type stoppedMsg struct{ action string }

type exitMsg struct{ err error }

type tickMsg time.Time

type telemetryMsg struct {
	telemetry runner.Telemetry
	procWS    uint64
	procPriv  uint64
	ramTotal  uint64
	ramAvail  uint64
}

type propsMsg struct {
	props runner.ModelProps
	err   error
}

// New builds the launcher rooted at dir.
func New(root string, cfg *config.Config) *Model {
	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = newStyles(ThemeByName(cfg.Theme), cfg.ASCII).accent()

	ti := textinput.New()
	ti.Placeholder = "--cache-reuse 256"
	ti.CharLimit = 256

	ci := textinput.New()
	ci.Prompt = ""
	ci.Placeholder = "Type a message… (enter send · /effort <lvl> · /help)"
	ci.CharLimit = 8192

	vp := viewport.New(80, 12)
	cv := viewport.New(80, 12)
	quitCtx, quitCancel := context.WithCancel(context.Background())

	fastHW := hw.DetectFast()

	m := &Model{
		root:          root,
		cfg:           cfg,
		hw:            fastHW,
		sysRAMAvail:   fastHW.RAMAvail,
		sysRAMTotal:   fastHW.RAMTotal,
		sty:           newStyles(ThemeByName(cfg.Theme), cfg.ASCII),
		spinner:       sp,
		input:         ti,
		chatInput:     ci,
		palette:       newCommandPalette(),
		logView:       vp,
		chatView:      cv,
		settings:      flags.Default(0, cfg.Port),
		status:        "IDLE",
		scanning:      true,
		quitCtx:       quitCtx,
		quitCancel:    quitCancel,
		editingPreset: -1,
		editingField:  -1,
		codeLineNums:  true,
	}
	m.ensureAgentReady()
	m.refreshChatView(false)
	return m
}

// SetVersionInfo records build metadata displayed in the UI and --version output.
func (m *Model) SetVersionInfo(version, commit, llamaTag string) {
	m.version = strings.TrimSpace(version)
	m.commit = strings.TrimSpace(commit)
	m.llamaTag = strings.TrimSpace(llamaTag)
}

// SetStartupNotices queues non-fatal startup migration or environment notices for display.
func (m *Model) SetStartupNotices(notices []string) {
	m.startupNotices = append([]string(nil), notices...)
	if len(notices) > 0 {
		m.toast = strings.Join(notices, " · ")
		m.toastUntil = time.Now().Add(8 * time.Second)
		m.toastSeq++
	}
}

// CompactBanner renders the 3-line geometric alpaca mark and wordmark for
// --version, the start screen, and the about/help overlay.
func CompactBanner(ascii bool) string {
	return BrandBanner(ascii)
}

// BrandBanner returns the 3-line alpaca header with an ASCII fallback for legacy consoles.
func BrandBanner(ascii bool) string {
	if ascii {
		return strings.Join([]string{
			`  /\/\     [ alpaca ]`,
			`  | o \__  portable local-LLM launcher`,
			`  |===\_/  CUDA | CPU | Tools | MCP | Skills`,
		}, "\n")
	}
	return strings.Join([]string{
		"  ╭─╮╭─╮    ▄▀█ █   █▀█ ▄▀█ █▀▀ ▄▀█",
		"  │ •  ╰─╮  █▀█ █▄▄ █▀▀ █▀█ █▄▄ █▀█",
		"  │ ═══╭─╯  portable launcher · llama.cpp",
	}, "\n")
}

// Init starts the async scan, GPU probe and initial spinner (stops once idle).
func (m *Model) Init() tea.Cmd {
	cmds := []tea.Cmd{m.scanCmd(), probeGPUCmd(m.root, m.hw)}
	if m.needsSpinner() {
		cmds = append(cmds, m.spinner.Tick)
	}
	if m.toast != "" {
		cmds = append(cmds, toastExpireCmd(m.toastSeq))
	}
	return tea.Batch(cmds...)
}

func (m *Model) needsSpinner() bool {
	return (m.screen == screenLauncher && m.scanning) ||
		(m.screen == screenRun && (m.status == "LOADING" || m.chatStreaming))
}

func runPollCmd(interval time.Duration) tea.Cmd {
	if interval < 2*time.Second {
		interval = 2 * time.Second
	}
	return tea.Tick(interval, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func toastExpireCmd(seq uint64) tea.Cmd {
	return tea.Tick(4*time.Second, func(time.Time) tea.Msg { return toastExpireMsg{seq: seq} })
}

func gpuCmd() tea.Cmd {
	return probeGPUCmd("", hw.Info{})
}

func probeGPUCmd(root string, fastHW hw.Info) tea.Cmd {
	return func() tea.Msg {
		devs, reason := hw.DetectGPUDevices(nil)
		if len(devs) > 0 && reason == "" {
			var sumT, sumF uint64
			for _, d := range devs {
				sumT += d.VRAMTotal
				sumF += d.VRAMFree
			}
			name := devs[0].Name
			if len(devs) > 1 {
				name = fmt.Sprintf("%dx %s", len(devs), name)
			}
			// Cross-check llama-server --list-devices when packaged binary exists.
			if root != "" {
				cudaExe := filepath.Join(root, "bin", "cuda", "llama-server.exe")
				if _, err := os.Stat(cudaExe); err == nil {
					if probed, pErr := hw.ProbeBackendDevices(context.Background(), cudaExe); pErr == nil && len(probed) > 0 {
						devs = probed
						sumT, sumF = 0, 0
						for _, d := range devs {
							sumT += d.VRAMTotal
							sumF += d.VRAMFree
						}
					}
				}
			}
			return gpuMsg{name: name, total: sumT, free: sumF, ok: true, devices: devs}
		}
		if root != "" && fastHW.HasVulkan {
			vulkanExe := filepath.Join(root, "bin", "vulkan", "llama-server.exe")
			if _, err := os.Stat(vulkanExe); err == nil {
				if probed, pErr := hw.ProbeBackendDevices(context.Background(), vulkanExe); pErr == nil && len(probed) > 0 {
					var sumT, sumF uint64
					for _, d := range probed {
						sumT += d.VRAMTotal
						sumF += d.VRAMFree
					}
					name := probed[0].Name
					if len(probed) > 1 {
						name = fmt.Sprintf("%dx %s", len(probed), name)
					}
					return gpuMsg{name: name, total: sumT, free: sumF, ok: false, devices: probed, unsupportedReason: reason}
				}
			}
		}
		return gpuMsg{unsupportedReason: reason}
	}
}

// scanCmd lists model files, then parses each header asynchronously. Explicit
// paths outside the models folder are listed too, so models on another drive
// (D:\...) work without moving them.
func (m *Model) scanCmd() tea.Cmd {
	dir := m.cfg.ModelsPath()
	depth := m.cfg.ScanDepth
	cachePath := m.cfg.GGUFCachePath()
	extras := append([]string(nil), m.cfg.ExtraModels...)
	return func() tea.Msg {
		cache := gguf.LoadCache(cachePath)
		cands, err := gguf.DiscoverWithDepth(dir, depth)
		entries := make([]modelEntry, 0, len(cands)+len(extras))
		seen := make(map[string]bool, len(cands))
		for _, c := range cands {
			seen[strings.ToLower(c.Path)] = true
			entries = append(entries, modelEntry{cand: c})
		}
		for _, p := range extras {
			if seen[strings.ToLower(p)] {
				continue
			}
			st, statErr := os.Stat(p)
			if statErr != nil || st.IsDir() {
				continue
			}
			seen[strings.ToLower(p)] = true
			entries = append(entries, modelEntry{cand: gguf.Candidate{Path: p, RelPath: p, Size: st.Size()}})
		}
		if err != nil && len(entries) == 0 {
			return scanMsg{cache: cache, err: err}
		}
		sort.Slice(entries, func(i, j int) bool {
			return strings.ToLower(entries[i].cand.RelPath) < strings.ToLower(entries[j].cand.RelPath)
		})
		return scanMsg{entries: entries, cache: cache}
	}
}

// rescan drops the current list and lists the models folder again.
func (m *Model) rescan() tea.Cmd {
	m.scanning = true
	m.scanErr = nil
	m.models = nil
	m.meta = nil
	m.applyFilter()
	m.recompute()
	return m.scanCmd()
}

func parseMetaCmd(idx int, path string, cache *gguf.Cache) tea.Cmd {
	return func() tea.Msg {
		meta, _, err := gguf.ParseCached(cache, path)
		return metaMsg{idx: idx, meta: meta, err: err}
	}
}

func saveCacheCmd(cache *gguf.Cache) tea.Cmd {
	return func() tea.Msg {
		if cache != nil {
			_ = cache.SaveIfDirty()
		}
		return nil
	}
}

func presetCmd(cfg *config.Config) tea.Cmd {
	return func() tea.Msg {
		ps, err := cfg.ListPresets()
		return presetMsg{presets: ps, err: err}
	}
}

// --- helpers ---

func (m *Model) layers() int {
	if m.meta == nil {
		return 0
	}
	if m.meta.BlockCount > 0 {
		return int(m.meta.BlockCount)
	}
	return len(m.meta.Layers)
}

func (m *Model) nonzeroIndex() int {
	if len(m.visible) == 0 {
		return -1
	}
	if m.cursor < 0 {
		return m.visible[0]
	}
	if m.cursor >= len(m.visible) {
		return m.visible[len(m.visible)-1]
	}
	return m.visible[m.cursor]
}

func (m *Model) current() *modelEntry {
	i := m.nonzeroIndex()
	if i < 0 || i >= len(m.models) {
		return nil
	}
	return &m.models[i]
}

func (m *Model) toastf(format string, args ...any) {
	m.toast = fmt.Sprintf(format, args...)
	m.toastUntil = time.Now().Add(4 * time.Second)
	m.toastSeq++
}

func (m *Model) applyFilter() {
	m.visible = m.visible[:0]
	q := strings.ToLower(strings.TrimSpace(m.filter))
	for i, e := range m.models {
		name := strings.ToLower(e.cand.RelPath)
		if q == "" || strings.Contains(name, q) {
			m.visible = append(m.visible, i)
		}
	}
	if m.cursor >= len(m.visible) {
		m.cursor = len(m.visible) - 1
	}
	if m.cursor < 0 {
		m.cursor = 0
	}
}

// selectCurrent loads the highlighted model's metadata and settings.
func (m *Model) selectCurrent() {
	e := m.current()
	if e == nil {
		m.meta = nil
		m.modelDriveKind = ""
		m.modelEstLoad = 0
		m.recompute()
		return
	}
	m.meta = e.meta
	if e.meta == nil {
		m.modelDriveKind, m.modelEstLoad = estimateModelLoad(e.cand.Path, uint64(max(0, e.cand.Size)))
		m.recompute()
		return
	}
	key := filepath.Base(e.meta.Path)
	if s, ok := m.cfg.Recall(key); ok {
		m.settings = s
		m.hasRecalled = true
		m.userEditedAutoFit = true
		m.userEditedThreads = true
	} else {
		m.settings = flags.Default(0, m.cfg.Port)
		m.hasRecalled = false
		m.userEditedAutoFit = false
		m.userEditedThreads = false
	}
	totalBytes := e.meta.TotalBytes
	if totalBytes == 0 && e.cand.Size > 0 {
		totalBytes = uint64(e.cand.Size)
	}
	m.modelDriveKind, m.modelEstLoad = estimateModelLoad(e.meta.Path, totalBytes)
	// A vision model usually ships its projector next to the weights.
	if m.settings.MMProj == "" {
		m.settings.MMProj = findProjector(e.meta.Path)
	}
	m.clampToModel()
	m.recompute()
}

func (m *Model) applySmartAutoFit() {
	if m.meta == nil || m.hasRecalled || m.userEditedAutoFit || m.hw.VRAMTotal == 0 {
		return
	}
	s := m.settings
	if s.AutoKV {
		s.KCache, s.VCache, _, _ = estimate.SmartKV(m.meta, s, m.hw)
		s.Normalize()
	}
	res := estimate.Compute(m.meta, s, m.hw)
	m.settings.AutoFit = estimate.NeedsAutoFit(res)
}

func (m *Model) clampToModel() {
	if m.meta == nil {
		return
	}
	if n := m.layers(); n > 0 {
		if m.settings.GPULayers > n {
			m.settings.GPULayers = flags.AllCount
		}
		if m.settings.NCpuMoe > n {
			m.settings.NCpuMoe = n
		}
	}
	// A context larger than the trained one is allowed on purpose (RoPE
	// scaling); only absurd values are clamped.
	if m.settings.Context > 4*1024*1024 {
		m.settings.Context = 4 * 1024 * 1024
	}
	m.settings.Normalize()
}

// recompute refreshes the estimate and the command preview.
func (m *Model) recompute() {
	m.settings.Normalize()
	if m.meta == nil {
		m.est = estimate.Result{}
		m.resolved = m.settings
		m.cmdPreview = ""
		return
	}
	m.applySmartAutoFit()
	resolved, res := estimate.ResolveSettings(m.settings, m.meta, m.hw)
	m.resolved = resolved
	m.est = res
	m.cmdPreview = flags.Quote(resolved.Args(m.meta.Path))
}

func (m *Model) backendDir() string {
	name := m.cfg.Backend
	switch name {
	case "auto":
		if m.hw.HasCUDA && dirExists(filepath.Join(m.root, "bin", "cuda")) {
			return "cuda"
		}
		if m.hw.HasVulkan && dirExists(filepath.Join(m.root, "bin", "vulkan")) {
			return "vulkan"
		}
		if !m.hw.HasAVX2 && dirExists(filepath.Join(m.root, "bin", "cpu-compat")) {
			return "cpu-compat"
		}
		return "cpu"
	case "cpu":
		if !m.hw.HasAVX2 && dirExists(filepath.Join(m.root, "bin", "cpu-compat")) {
			return "cpu-compat"
		}
		return "cpu"
	case "cuda", "vulkan", "cpu-compat":
		return name
	}
	return "cpu"
}

func dirExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

// --- update ---

// Update handles messages and keys.
func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	prevToastSeq := m.toastSeq
	nextM, cmd := m.updateInner(msg)
	if cmd == nil && m.toastSeq != prevToastSeq && m.toast != "" {
		return nextM, toastExpireCmd(m.toastSeq)
	}
	return nextM, cmd
}

func (m *Model) updateInner(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.ready = true
		m.resizeRunViews()
		return m, nil

	case tea.KeyMsg:
		if msg.Type == tea.KeyCtrlC || msg.String() == "ctrl+c" {
			return m.forceQuit()
		}
		if m.permModal != nil {
			return m.handlePermModalKey(msg)
		}
		if m.skillApproveModal != nil {
			return m.handleSkillApproveModalKey(msg)
		}
		if m.askModal != nil {
			return m.handleAskModalKey(msg)
		}
		if m.toolsModal != nil {
			return m.handleToolsModalKey(msg)
		}
		if m.mcpModal != nil {
			return m.handleMCPModalKey(msg)
		}
		if m.permsModal != nil {
			return m.handlePermsModalKey(msg)
		}
		if m.skillsModal != nil {
			return m.handleSkillsModalKey(msg)
		}
		if m.contextModal != nil {
			return m.handleContextModalKey(msg)
		}
		if m.attachModal != nil {
			return m.handleAttachModalKey(msg)
		}
		if m.sessionsModal != nil {
			return m.handleSessionsModalKey(msg)
		}
		if m.palette.open {
			return m.updatePaletteKey(msg)
		}
		if msg.String() == "ctrl+p" && !m.editing && !m.showHelp && !m.quitConfirm && !m.runQuitConfirm && !m.backConfirm && !m.lanConfirm {
			m.openCommandPalette()
			return m, nil
		}
		if m.editing {
			return m.updateEditing(msg)
		}
		if m.screen == screenRun {
			return m.updateRun(msg)
		}
		return m.updateLauncher(msg)

	case tea.MouseMsg:
		return m.updateMouse(msg)

	case scanMsg:
		m.scanning = false
		m.scanErr = msg.err
		m.models = msg.entries
		m.ggufCache = msg.cache
		m.applyFilter()
		if m.pendingSelectPath != "" {
			for pos, idx := range m.visible {
				if idx >= 0 && idx < len(m.models) && strings.EqualFold(m.models[idx].cand.Path, m.pendingSelectPath) {
					m.cursor = pos
					break
				}
			}
			m.pendingSelectPath = ""
		}
		m.loadedMeta = 0
		cmds := []tea.Cmd{}
		for i := range m.models {
			cmds = append(cmds, parseMetaCmd(i, m.models[i].cand.Path, m.ggufCache))
		}
		if len(m.models) > 0 {
			cmds = append(cmds, presetCmd(m.cfg))
		}
		return m, tea.Batch(cmds...)

	case metaMsg:
		if msg.idx >= 0 && msg.idx < len(m.models) {
			m.models[msg.idx].meta = msg.meta
			m.models[msg.idx].err = msg.err
			m.loadedMeta++
		}
		if idx := m.nonzeroIndex(); idx == msg.idx {
			m.selectCurrent()
		}
		if m.loadedMeta >= len(m.models) && m.ggufCache != nil {
			return m, saveCacheCmd(m.ggufCache)
		}
		return m, nil

	case gpuMsg:
		m.gpuProbed = true
		m.hw.GPUName = msg.name
		m.hw.VRAMTotal = msg.total
		m.hw.VRAMFree = msg.free
		m.hw.HasCUDA = msg.ok
		m.hw.Devices = msg.devices
		m.hw.CUDAUnsupportedReason = msg.unsupportedReason
		m.recompute()
		return m, nil

	case hwMsg:
		m.hw = msg.info
		m.sysRAMAvail = msg.info.RAMAvail
		m.sysRAMTotal = msg.info.RAMTotal
		m.recompute()
		return m, nil

	case copyProgressMsg:
		if !m.copying {
			return m, nil
		}
		m.copyProgress = msg
		if !msg.done {
			return m, waitCopyProgressCmd(m.copyCh)
		}
		m.copying = false
		m.copyCancel = nil
		m.copyCh = nil
		if msg.err != nil {
			if !errors.Is(msg.err, context.Canceled) {
				m.toastf("copy failed: %v", msg.err)
			}
			return m, nil
		}
		m.pendingSelectPath = msg.destPath
		m.cfg.AddExtraModel(msg.destPath)
		if msg.alreadyExists {
			m.toastf("local copy already exists: %s", filepath.Base(msg.destPath))
		} else {
			m.toastf("copied to local disk (%.0f MB/s) — ready", msg.speedMBps)
		}
		return m, tea.Batch(saveConfigCmd(m.cfg.Clone()), m.rescan())

	case presetMsg:
		if msg.err != nil {
			m.toastf("%v", msg.err)
			return m, nil
		}
		m.presets = msg.presets
		if msg.saved != "" {
			m.toastf("saved preset %q", msg.saved)
		}
		if msg.renamed != "" {
			m.toastf("renamed to %q", msg.renamed)
		}
		return m, nil

	case toastExpireMsg:
		if msg.seq == m.toastSeq {
			m.toast = ""
		}
		return m, nil

	case tickMsg:
		if time.Now().After(m.toastUntil) {
			m.toast = ""
		}
		if m.screen != screenRun || m.proc == nil || m.proc.Exited() {
			return m, nil
		}
		m.telemetryTick++
		interval := 2 * time.Second
		if m.status == "READY" && !m.chatStreaming {
			interval = 4 * time.Second
		}
		cmds := []tea.Cmd{runPollCmd(interval)}
		if !m.telemetryPending && m.proc.PID() > 0 {
			m.telemetryPending = true
			pollHTTP := m.status == "READY" && m.wantsHTTPMetrics()
			cmds = append(cmds, telemetryPollCmd(m.effectivePort(), uint32(m.proc.PID()), pollHTTP))
		}
		return m, tea.Batch(cmds...)

	case telemetryMsg:
		m.telemetryPending = false
		if msg.ramTotal > 0 {
			m.sysRAMTotal = msg.ramTotal
			m.sysRAMAvail = msg.ramAvail
		}
		if msg.procWS > 0 {
			m.procRAMBytes = msg.procWS
			m.procPrivBytes = msg.procPriv
		}
		if m.status == "READY" {
			// Preserve positive rates if server just finished initializing metrics,
			// otherwise reflect live Prometheus /metrics and /slots readings.
			if msg.telemetry.NCtx > 0 || msg.telemetry.GenTokPerSec > 0 || msg.telemetry.PromptTokSec > 0 || msg.telemetry.KVCacheRatio > 0 || msg.telemetry.RequestsProcessing > 0 {
				m.telemetry = msg.telemetry
			}
			if m.telemetry.GenTokPerSec > 0 {
				m.metrics.GenTokPerSec = m.telemetry.GenTokPerSec
			}
			if m.telemetry.PromptTokSec > 0 {
				m.metrics.PromptTokSec = m.telemetry.PromptTokSec
			}
		}
		return m, nil

	case chatStreamEvt:
		if msg.gen != m.chatGen {
			return m, nil
		}
		if msg.delta.Done {
			return m, readChatEvtCmd(msg.gen, msg.ch, msg.errCh)
		}
		m.handleChatDelta(msg.delta)
		return m, readChatEvtCmd(msg.gen, msg.ch, msg.errCh)

	case chatDoneMsg:
		if msg.gen != m.chatGen {
			return m, nil
		}
		m.finishChatStream(msg.err)
		return m, nil

	case agentLoopMsg:
		return m.handleAgentLoopMsg(msg)

	case agentToolTokensMsg:
		m.toolSchemaTokens = msg.tokens
		m.toolSchemaLive = msg.live
		return m, nil

	case logBatchMsg:
		m.appendLogBatch([]string(msg))
		return m, readLogBatchCmd(m.proc)

	case lineMsg:
		m.appendLog(string(msg))
		return m, readLogBatchCmd(m.proc)

	case linesClosedMsg:
		return m, nil

	case healthMsg:
		// Ignore a poll that belongs to a server we already stopped, or the
		// next server could be marked READY by the previous one's result.
		if m.proc == nil || msg.proc != m.proc {
			return m, nil
		}
		if msg.err == nil {
			if m.status == "LOADING" {
				m.status = "READY"
				m.statusDe = ""
				m.launchRetried = false
				m.refreshChatView(false)
				cmds := []tea.Cmd{fetchPropsCmd(m.effectivePort(), m.chatBaseURL)}
				if m.proc != nil && m.proc.PID() > 0 {
					m.telemetryPending = true
					cmds = append(cmds, telemetryPollCmd(m.effectivePort(), uint32(m.proc.PID()), m.wantsHTTPMetrics()))
				}
				return m, tea.Batch(cmds...)
			}
		} else if m.status == "LOADING" {
			m.status = "ERROR"
			m.statusDe = msg.err.Error()
			m.refreshChatView(false)
		}
		return m, nil

	case propsMsg:
		if msg.err == nil && msg.props.Loaded {
			m.modelCaps = msg.props
		}
		return m, nil

	case launchMsg:
		m.launching = false
		if m.quit || (m.quitCtx != nil && m.quitCtx.Err() != nil) {
			if msg.proc != nil {
				_ = msg.proc.StopTimeout(0)
				msg.proc.Release()
			}
			return m, nil
		}
		if msg.err != nil {
			m.status = "ERROR"
			m.statusDe = msg.err.Error()
			m.logs = nil
			m.refreshChatView(false)
			return m, nil
		}
		m.proc = msg.proc
		m.activePort = msg.port
		if msg.port != m.settings.Port {
			m.toastf("port busy, using %d", msg.port)
		}
		m.status = "LOADING"
		m.statusDe = ""
		if m.meta != nil {
			m.cfg.Remember(filepath.Base(m.meta.Path), m.settings)
		}
		ctx, cancel := context.WithCancel(context.Background())
		m.healthCancel = cancel
		cmds := []tea.Cmd{
			readLogBatchCmd(m.proc),
			waitHealthCmd(ctx, m.proc, healthTimeoutForETA(m.launchETA)),
			waitExitCmd(m.proc),
			m.spinner.Tick,
			runPollCmd(2 * time.Second),
			saveConfigCmd(m.cfg.Clone()),
		}
		if m.proc.PID() > 0 {
			m.telemetryPending = true
			cmds = append(cmds, telemetryPollCmd(m.effectivePort(), uint32(m.proc.PID()), false))
		}
		return m, tea.Batch(cmds...)

	case stoppedMsg:
		m.stopHealth()
		m.stopChatStream()
		m.proc = nil
		m.activePort = 0
		m.telemetryPending = false
		switch msg.action {
		case "restart":
			m.logs = nil
			m.metrics = runner.Metrics{}
			m.telemetry = runner.Telemetry{}
			m.logView.SetContent("")
			m.launchRetried = false
			return m.launch()
		case "back":
			m.screen = screenLauncher
			m.status = "STOPPED"
			m.statusDe = ""
			m.chatInput.Blur()
		default:
			m.status = "STOPPED"
			m.statusDe = "stopped by user"
			m.refreshChatView(false)
		}
		return m, nil

	case toastMsg:
		m.toastf("%s", string(msg))
		return m, nil

	case pathMsg:
		if msg.err != nil {
			m.toastf("%v", msg.err)
			return m, nil
		}
		if msg.modelsDir != "" {
			m.cfg.ModelsDir = msg.modelsDir
			m.toastf("models folder: %s", msg.modelsDir)
		}
		if msg.model != "" {
			m.cfg.AddExtraModel(msg.model)
			m.toastf("added %s", filepath.Base(msg.model))
		}
		return m, tea.Batch(saveConfigCmd(m.cfg.Clone()), m.rescan())

	case exitMsg:
		if m.proc == nil {
			return m, nil
		}
		m.proc.Release()
		m.stopHealth()
		m.stopChatStream()
		if m.status == "STOPPING" || m.status == "STOPPED" {
			return m, nil
		}
		// One automatic retry: some model chat templates raise a Jinja
		// exception on --reasoning-effort values they do not support
		// ("Unexpected reasoning effort max. Supported types are high and
		// low."), which kills the server before it becomes ready. Relaunch
		// once with the effort flag dropped to the template default instead
		// of leaving the user with a dead server.
		if m.status == "LOADING" && !m.launchRetried &&
			m.settings.ReasoningEffort != "" && m.settings.ReasoningEffort != "default" &&
			effortRejectedByTemplate(m.logs) {
			m.launchRetried = true
			rejected := m.settings.ReasoningEffort
			m.settings.ReasoningEffort = "default"
			m.proc = nil
			m.activePort = 0
			m.telemetryPending = false
			m.toastf("model template rejected --reasoning-effort %s — retrying with the template default", rejected)
			return m.launch()
		}
		// A server that dies must never leave the screen claiming READY.
		m.status = "ERROR"
		m.statusDe = "llama-server exited"
		if msg.err != nil {
			m.statusDe = msg.err.Error()
		}
		if reason := startupReason(m.logs); reason != "" {
			m.statusDe = "llama-server exited: " + reason
		}
		if effortRejectedByTemplate(m.logs) {
			m.statusDe += " — set Reasoning effort to a supported level (ctrl+e) or check the log"
		}
		m.refreshChatView(false)
		return m, nil
	case spinner.TickMsg:
		if !m.needsSpinner() {
			return m, nil
		}
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		if m.screen == screenRun && m.chatStreaming {
			m.refreshChatView(false)
		}
		return m, cmd
	}
	return m, nil
}

func (m *Model) resizeLogView() {
	w := m.width - 4
	if w < 10 {
		w = 10
	}
	h := m.height - 6
	if h < 3 {
		h = 3
	}
	m.logView.Width = w
	m.logView.Height = h
}

// Shutdown stops any running server. Called after the program exits.
func (m *Model) Shutdown() {
	if m.quitCancel != nil {
		m.quitCancel()
	}
	if m.copyCancel != nil {
		m.copyCancel()
	}
	m.stopChatStream()
	if len(m.chatMsgs) > 0 {
		m.saveCurrentSession()
	}
	m.stopHealth()
	if m.mcpManager != nil {
		m.mcpManager.Shutdown()
		m.mcpManager = nil
	}
	if m.proc != nil {
		grace := 250 * time.Millisecond
		if m.quit {
			grace = 0
		}
		_ = m.proc.StopTimeout(grace)
		m.proc = nil
	}
}

func (m *Model) forceQuit() (tea.Model, tea.Cmd) {
	m.editing = false
	m.filtering = false
	m.showPreset = false
	m.showHelp = false
	m.lanConfirm = false
	m.quitConfirm = false
	m.runQuitConfirm = false
	m.backConfirm = false
	m.copying = false
	if m.copyCancel != nil {
		m.copyCancel()
		m.copyCancel = nil
	}
	m.copyCh = nil
	m.quit = true
	if m.quitCancel != nil {
		m.quitCancel()
	}
	m.stopChatStream()
	if len(m.chatMsgs) > 0 {
		m.saveCurrentSession()
	}
	m.stopHealth()
	if m.mcpManager != nil {
		m.mcpManager.Shutdown()
		m.mcpManager = nil
	}
	return m, tea.Quit
}

func (m *Model) updateMouse(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	var delta int
	switch msg.Button {
	case tea.MouseButtonWheelUp:
		delta = -1
	case tea.MouseButtonWheelDown:
		delta = 1
	default:
		return m, nil
	}
	if m.screen == screenRun {
		if m.runTab == runTabChat {
			if delta < 0 {
				m.chatView.ScrollUp(3)
			} else {
				m.chatView.ScrollDown(3)
			}
		} else {
			if delta < 0 {
				m.logView.ScrollUp(3)
			} else {
				m.logView.ScrollDown(3)
			}
		}
		return m, nil
	}
	if m.editing || m.showHelp || m.lanConfirm || m.quitConfirm {
		return m, nil
	}
	if m.showPreset {
		if delta < 0 && m.presetIdx > 0 {
			m.presetIdx--
		} else if delta > 0 && m.presetIdx < len(m.presets)-1 {
			m.presetIdx++
		}
		return m, nil
	}
	if m.focus == 0 {
		next := clampInt(m.cursor+delta, 0, max(0, len(m.visible)-1))
		if next != m.cursor {
			m.cursor = next
			m.selectCurrent()
		}
	} else {
		m.fieldIdx = clampInt(m.fieldIdx+delta, 0, max(0, len(m.visibleFields())-1))
	}
	return m, nil
}

// stopHealth cancels the health poll of the current server.
func (m *Model) stopHealth() {
	if m.healthCancel != nil {
		m.healthCancel()
		m.healthCancel = nil
	}
}

const (
	logRingMax     = 5000
	logRingTrimTo  = 4000
	logBatchWindow = 45 * time.Millisecond
	logBatchMax    = 512
)

func (m *Model) wantsHTTPMetrics() bool {
	return strings.Contains(" "+m.settings.ExtraArgs+" ", " --metrics ")
}

func (m *Model) appendLog(line string) {
	m.appendLogBatch([]string{line})
}

func (m *Model) appendLogBatch(lines []string) {
	if len(lines) == 0 {
		return
	}
	wasAtBottom := len(m.logs) == 0 || m.logView.AtBottom()
	for _, line := range lines {
		m.logs = append(m.logs, line)
		runner.ParseLine(line, &m.metrics)
		if runner.IsErrorLine(line) && m.status == "LOADING" {
			m.statusDe = strings.TrimSpace(line)
		}
	}
	if len(m.logs) > logRingMax {
		keep := logRingTrimTo
		start := len(m.logs) - keep
		copy(m.logs[:keep], m.logs[start:])
		clear(m.logs[keep:])
		m.logs = m.logs[:keep]
	}
	m.logView.SetContent(strings.Join(m.logs, "\n"))
	if wasAtBottom {
		m.logView.GotoBottom()
	}
}

func (m *Model) updateEditing(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c":
		return m.forceQuit()
	case "esc":
		m.editing = false
		m.editingPreset = -1
		m.editingField = -1
		m.editingPath = false
		m.editingTitle = ""
		m.input.Blur()
		return m, nil
	case "enter":
		m.editing = false
		m.editingTitle = ""
		m.input.Blur()
		text := m.input.Value()
		if m.editingPath {
			m.editingPath = false
			if strings.TrimSpace(text) == "" {
				return m, nil
			}
			return m, openPathCmd(text)
		}
		if idx := m.editingPreset; idx >= 0 {
			m.editingPreset = -1
			if idx < len(m.presets) {
				return m, renamePresetCmd(m.cfg, m.presets[idx], text)
			}
			return m, nil
		}
		if m.editingField >= 0 {
			beforeThreads := m.settings.Threads
			m.applyTypedValue(text)
			if m.settings.Threads != beforeThreads {
				m.userEditedThreads = true
			}
			return m, nil
		}
		m.settings.ExtraArgs = text
		m.recompute()
		return m, nil
	}
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

func (m *Model) updateLauncher(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if m.copying {
		switch msg.String() {
		case "ctrl+c":
			return m.forceQuit()
		case "esc":
			if m.copyCancel != nil {
				m.copyCancel()
				m.copyCancel = nil
			}
			m.copying = false
			m.copyCh = nil
			m.toastf("copy cancelled")
		}
		return m, nil
	}
	if m.showHelp {
		m.showHelp = false
		return m, nil
	}
	if m.lanConfirm {
		switch msg.String() {
		case "y":
			m.lanConfirm = false
		case "n", "esc":
			m.lanConfirm = false
			m.settings.LAN = false
			m.recompute()
		}
		return m, nil
	}
	if m.quitConfirm {
		switch msg.String() {
		case "y":
			m.quitConfirm = false
			m.quit = true
			return m, tea.Quit
		case "n", "esc":
			m.quitConfirm = false
		}
		return m, nil
	}
	if m.filtering {
		switch msg.String() {
		case "esc":
			m.filtering = false
			m.filter = ""
			m.applyFilter()
		case "enter":
			m.filtering = false
		case "backspace":
			r := []rune(m.filter)
			if len(r) > 0 {
				m.filter = string(r[:len(r)-1])
			}
			m.applyFilter()
		default:
			if len(msg.Runes) > 0 {
				m.filter += string(msg.Runes)
				m.applyFilter()
			}
		}
		return m, nil
	}

	if m.showPreset {
		return m.updatePresetOverlay(msg)
	}

	switch msg.String() {
	case "ctrl+c":
		return m.forceQuit()
	case "q":
		if m.status == "READY" || m.status == "LOADING" {
			m.quitConfirm = true
			return m, nil
		}
		m.quit = true
		return m, tea.Quit

	case "tab", "shift+tab":
		m.focus = (m.focus + 1) % 2
		return m, nil

	case "up", "k":
		if m.focus == 0 {
			if m.cursor > 0 {
				m.cursor--
				m.selectCurrent()
			}
		} else if m.fieldIdx > 0 {
			m.fieldIdx--
		}
		return m, nil
	case "down", "j":
		if m.focus == 0 {
			if m.cursor < len(m.visible)-1 {
				m.cursor++
				m.selectCurrent()
			}
		} else if m.fieldIdx < len(m.visibleFields())-1 {
			m.fieldIdx++
		}
		return m, nil
	case "pgup", "ctrl+u":
		if m.focus == 0 {
			m.cursor = clampInt(m.cursor-5, 0, max(0, len(m.visible)-1))
			m.selectCurrent()
		} else {
			m.fieldIdx = clampInt(m.fieldIdx-5, 0, max(0, len(m.visibleFields())-1))
		}
		return m, nil
	case "pgdown", "ctrl+d":
		if m.focus == 0 {
			m.cursor = clampInt(m.cursor+5, 0, max(0, len(m.visible)-1))
			m.selectCurrent()
		} else {
			m.fieldIdx = clampInt(m.fieldIdx+5, 0, max(0, len(m.visibleFields())-1))
		}
		return m, nil
	case "home":
		if m.focus == 0 && len(m.visible) > 0 {
			m.cursor = 0
			m.selectCurrent()
		} else {
			m.fieldIdx = 0
		}
		return m, nil
	case "end":
		if m.focus == 0 && len(m.visible) > 0 {
			m.cursor = len(m.visible) - 1
			m.selectCurrent()
		} else if fs := m.visibleFields(); len(fs) > 0 {
			m.fieldIdx = len(fs) - 1
		}
		return m, nil
	case "left", "h":
		if m.focus == 1 {
			prevTheme := m.cfg.Theme
			prevThreads := m.settings.Threads
			m.adjustField(-1)
			if m.settings.Threads != prevThreads {
				m.userEditedThreads = true
			}
			if m.cfg.Theme != prevTheme {
				return m, saveConfigCmd(m.cfg.Clone())
			}
		}
		return m, nil
	case "right", "l":
		if m.focus == 1 {
			prevTheme := m.cfg.Theme
			prevThreads := m.settings.Threads
			m.adjustField(1)
			if m.settings.Threads != prevThreads {
				m.userEditedThreads = true
			}
			if m.cfg.Theme != prevTheme {
				return m, saveConfigCmd(m.cfg.Clone())
			}
		}
		return m, nil

	case "enter":
		if m.focus == 1 {
			if m.beginEdit() {
				return m, nil
			}
			prevTheme := m.cfg.Theme
			f := m.visibleFields()
			if m.fieldIdx < len(f) && f[m.fieldIdx].activate != nil {
				f[m.fieldIdx].activate()
			}
			if m.cfg.Theme != prevTheme {
				return m, saveConfigCmd(m.cfg.Clone())
			}
			return m, nil
		}
		m.launchRetried = false
		return m.launch()

	case "/":
		m.filtering = true
		m.filter = ""
		m.applyFilter()
		return m, nil
	case "o":
		m.editing = true
		m.editingField = -1
		m.editingPreset = -1
		m.editingPath = true
		m.editingTitle = "Open model (.gguf) or models folder"
		m.input.SetValue("")
		m.input.Placeholder = `.\models  or  .\models\model.gguf`
		m.input.Focus()
		return m, nil
	case "c":
		if m.cmdPreview != "" {
			return m, copyCmd(m.cmdPreview)
		}
		return m, nil
	case "C":
		return m.startCopyLocal()
	case "p":
		m.showPreset = true
		m.presetIdx = 0
		return m, presetCmd(m.cfg)
	case "s":
		return m, m.saveCurrentPreset()
	case "?":
		m.showHelp = true
		return m, nil
	case "t":
		m.cycleTheme(1)
		return m, saveConfigCmd(m.cfg.Clone())
	}
	return m, nil
}

func (m *Model) cycleTheme(dir int) {
	m.cfg.Theme = NextTheme(m.cfg.Theme, dir)
	m.sty = newStyles(ThemeByName(m.cfg.Theme), m.cfg.ASCII)
	m.spinner.Style = m.sty.accent()
	m.toastf("theme: %s", m.sty.t.Name)
}

func (m *Model) updatePresetOverlay(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc", "p":
		m.showPreset = false
	case "up", "k":
		if m.presetIdx > 0 {
			m.presetIdx--
		}
	case "down", "j":
		if m.presetIdx < len(m.presets)-1 {
			m.presetIdx++
		}
	case "enter":
		if m.presetIdx >= 0 && m.presetIdx < len(m.presets) {
			p := m.presets[m.presetIdx]
			m.settings = p.Settings
			m.showPreset = false
			m.clampToModel()
			m.recompute()
			m.toastf("loaded %q", p.Name)
		}
	case "d":
		if m.presetIdx >= 0 && m.presetIdx < len(m.presets) {
			m.showPreset = false
			return m, deletePresetCmd(m.cfg, m.presets[m.presetIdx])
		}
	case "r":
		if m.presetIdx >= 0 && m.presetIdx < len(m.presets) {
			p := m.presets[m.presetIdx]
			m.showPreset = false
			m.editingPreset = m.presetIdx
			m.editingPath = false
			m.editingField = -1
			m.editingTitle = "Rename preset"
			m.editing = true
			m.input.Placeholder = "preset name"
			m.input.SetValue(p.Name)
			m.input.Focus()
		}
	}
	return m, nil
}

func (m *Model) updateRun(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if m.runQuitConfirm {
		switch msg.String() {
		case "y":
			m.runQuitConfirm = false
			m.quit = true
			return m, tea.Quit
		case "n", "esc":
			m.runQuitConfirm = false
		}
		return m, nil
	}
	if m.backConfirm {
		switch msg.String() {
		case "y":
			m.backConfirm = false
			m.stopChatStream()
			if m.proc != nil && !m.proc.Exited() {
				m.status = "STOPPING"
				return m, stopCmd(m.proc, "back")
			}
			m.screen = screenLauncher
			m.status = "STOPPED"
			m.chatInput.Blur()
		case "n", "esc":
			m.backConfirm = false
		}
		return m, nil
	}

	if m.runTab == runTabChat && m.popup.open {
		if handled, mdl, c := m.handlePopupKey(msg); handled {
			return mdl, c
		}
	}

	if msg.String() == "tab" || msg.String() == "shift+tab" {
		m.popup.reset()
		if m.runTab == runTabChat {
			m.runTab = runTabLog
			m.chatInput.Blur()
		} else {
			m.runTab = runTabChat
			m.chatInput.Focus()
			m.resizeRunViews()
		}
		return m, nil
	}

	if m.runTab == runTabChat {
		switch msg.String() {
		case "ctrl+c":
			return m.forceQuit()
		case "esc":
			if m.chatStreaming {
				m.stopChatStream()
				m.toastf("generation aborted")
				return m, nil
			}
			if m.proc != nil && !m.proc.Exited() {
				m.backConfirm = true
				return m, nil
			}
			m.screen = screenLauncher
			m.chatInput.Blur()
			return m, nil
		case "ctrl+e":
			m.cycleReasoningEffortLive()
			return m, nil
		case "ctrl+t":
			m.toggleThinkingDisplay()
			m.refreshChatView(false)
			return m, nil
		case "ctrl+o":
			m.toggleToolCardsDisplay()
			return m, nil
		case "ctrl+y", "alt+c":
			if cmd := m.copyLatestCodeOrReply(); cmd != nil {
				return m, cmd
			}
			return m, nil
		case "c":
			if !m.chatInput.Focused() {
				if cmd := m.copyLatestCodeOrReply(); cmd != nil {
					return m, cmd
				}
				return m, nil
			}
		case "ctrl+l":
			m.resetChat()
			m.toastf("chat cleared")
			return m, nil
		case "up":
			m.chatView.ScrollUp(2)
			return m, nil
		case "down":
			m.chatView.ScrollDown(2)
			return m, nil
		case "pgup":
			m.chatView.ViewUp()
			return m, nil
		case "pgdown":
			m.chatView.ViewDown()
			return m, nil
		case "ctrl+u":
			m.chatView.HalfViewUp()
			return m, nil
		case "ctrl+d":
			m.chatView.HalfViewDown()
			return m, nil
		case "home":
			m.chatView.GotoTop()
			return m, nil
		case "end":
			m.chatView.GotoBottom()
			return m, nil
		case "enter":
			return m.submitChatInput()
		}
		var cmd tea.Cmd
		m.chatInput, cmd = m.chatInput.Update(msg)
		m.syncChatPopup()
		return m, cmd
	}

	switch msg.String() {
	case "ctrl+c":
		return m.forceQuit()
	case "q":
		// Quitting while a server loads or runs must be confirmed.
		if m.proc != nil && !m.proc.Exited() {
			m.runQuitConfirm = true
			return m, nil
		}
		m.quit = true
		return m, tea.Quit
	case "x":
		m.stopChatStream()
		if m.proc != nil && !m.proc.Exited() {
			m.status = "STOPPING"
			m.statusDe = "stopping…"
			return m, stopCmd(m.proc, "stop")
		}
		m.status = "STOPPED"
		m.statusDe = "stopped by user"
		return m, nil
	case "r":
		if m.meta == nil || m.launching {
			return m, nil
		}
		m.stopChatStream()
		if m.proc != nil && !m.proc.Exited() {
			m.status = "STOPPING"
			m.statusDe = "restarting…"
			return m, stopCmd(m.proc, "restart")
		}
		m.proc = nil
		m.logs = nil
		m.metrics = runner.Metrics{}
		m.telemetry = runner.Telemetry{}
		m.logView.SetContent("")
		m.launchRetried = false
		return m.launch()
	case "ctrl+e":
		m.cycleReasoningEffortLive()
		return m, nil
	case "esc":
		if m.proc != nil && !m.proc.Exited() {
			m.backConfirm = true
			return m, nil
		}
		m.screen = screenLauncher
		return m, nil
	case "up", "k", "pgup":
		m.logView, _ = m.logView.Update(msg)
		return m, nil
	case "ctrl+u":
		m.logView.HalfViewUp()
		return m, nil
	case "down", "j", "pgdown":
		m.logView, _ = m.logView.Update(msg)
		return m, nil
	case "ctrl+d":
		m.logView.HalfViewDown()
		return m, nil
	case "home", "g":
		m.logView.GotoTop()
		return m, nil
	case "end", "G":
		m.logView.GotoBottom()
		return m, nil
	}
	return m, nil
}

func readLogBatchCmd(p *runner.Process) tea.Cmd {
	if p == nil {
		return nil
	}
	return func() tea.Msg {
		ch := p.Lines()
		first, ok := <-ch
		if !ok {
			return linesClosedMsg{}
		}
		batch := make([]string, 1, 32)
		batch[0] = first
		timer := time.NewTimer(logBatchWindow)
		defer timer.Stop()
		for len(batch) < logBatchMax {
			select {
			case line, ok := <-ch:
				if !ok {
					return logBatchMsg(batch)
				}
				batch = append(batch, line)
			case <-timer.C:
				return logBatchMsg(batch)
			}
		}
		return logBatchMsg(batch)
	}
}

// startupReason picks the most informative log line for a server that died
// before becoming ready. llama.cpp prints a generic fatal line last ("exiting
// due to model loading error"), so the actionable error sits above it in the
// log; prefer chat-template and Jinja failures, then fall back to the tail.
func startupReason(logs []string) string {
	for i := len(logs) - 1; i >= 0; i-- {
		l := strings.ToLower(logs[i])
		if strings.Contains(l, "reasoning effort") || strings.Contains(l, "reasoning_effort") ||
			strings.Contains(l, "chat template parsing error") || strings.Contains(l, "jinja exception") {
			return strings.TrimSpace(logs[i])
		}
	}
	for i := len(logs) - 1; i >= 0; i-- {
		if s := strings.TrimSpace(logs[i]); s != "" {
			return s
		}
	}
	return ""
}

// effortRejectedByTemplate reports whether the log shows a model chat template
// that raised a Jinja exception over the --reasoning-effort value. Such
// templates (e.g. reasoning models accepting only high/low) kill the server
// before it becomes ready; the fix is to relaunch without the flag.
func effortRejectedByTemplate(logs []string) bool {
	for i := len(logs) - 1; i >= 0; i-- {
		l := strings.ToLower(logs[i])
		if strings.Contains(l, "reasoning effort") || strings.Contains(l, "reasoning_effort") {
			return true
		}
	}
	return false
}

func waitHealthCmd(ctx context.Context, p *runner.Process, timeout time.Duration) tea.Cmd {
	return func() tea.Msg {
		if timeout <= 0 {
			timeout = 10 * time.Minute
		}
		ctx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		return healthMsg{err: p.Health(ctx), proc: p}
	}
}

func waitExitCmd(p *runner.Process) tea.Cmd {
	return func() tea.Msg {
		<-p.Done()
		return exitMsg{err: p.ExitError()}
	}
}

func telemetryPollCmd(port int, pid uint32, ready bool) tea.Cmd {
	return func() tea.Msg {
		msg := telemetryMsg{}
		if pid > 0 {
			if ws, priv, err := hw.ProcessRAM(pid); err == nil {
				msg.procWS = ws
				msg.procPriv = priv
			}
		}
		if total, avail, err := hw.MemoryStatus(); err == nil {
			msg.ramTotal = total
			msg.ramAvail = avail
		}
		if ready && port > 0 {
			ctx, cancel := context.WithTimeout(context.Background(), 900*time.Millisecond)
			defer cancel()
			if tel, err := runner.FetchTelemetry(ctx, port); err == nil {
				msg.telemetry = tel
			}
		}
		return msg
	}
}

// launch switches to the run screen and starts the server off the UI thread.
func (m *Model) launch() (tea.Model, tea.Cmd) {
	if m.meta == nil {
		m.toastf("no model selected")
		return m, nil
	}
	if m.launching || (m.proc != nil && !m.proc.Exited()) {
		return m, nil
	}
	m.screen = screenRun
	m.status = "LOADING"
	m.statusDe = ""
	m.logs = nil
	m.metrics = runner.Metrics{}
	m.telemetry = runner.Telemetry{}
	m.modelCaps = runner.ModelProps{}
	m.procRAMBytes = 0
	m.procPrivBytes = 0
	m.activePort = m.settings.Port
	m.logView.SetContent("")
	m.resetChat()
	if m.settings.LaunchMode == "chat" {
		m.runTab = runTabChat
		m.chatInput.Focus()
	} else {
		m.runTab = runTabLog
		m.chatInput.Blur()
	}
	m.refreshChatView(false)
	m.resizeRunViews()
	totalBytes := m.meta.TotalBytes
	if e := m.current(); totalBytes == 0 && e != nil && e.cand.Size > 0 {
		totalBytes = uint64(e.cand.Size)
	}
	m.launchDriveKind, m.launchETA = estimateModelLoad(m.meta.Path, totalBytes)
	m.launchAt = time.Now()
	m.launching = true
	return m, m.launchCmd()
}

func (m *Model) startCopyLocal() (tea.Model, tea.Cmd) {
	if m.meta == nil {
		m.toastf("no model selected")
		return m, nil
	}
	if m.copying {
		return m, nil
	}
	destDir := m.cfg.LocalModelsDir()
	parentCtx := m.quitCtx
	if parentCtx == nil {
		parentCtx = context.Background()
	}
	ctx, cancel := context.WithCancel(parentCtx)
	ch := make(chan copyProgressMsg, 8)
	m.copying = true
	m.copyTitle = filepath.Base(m.meta.Path)
	m.copyProgress = copyProgressMsg{}
	m.copyCancel = cancel
	m.copyCh = ch

	src := m.meta.Path
	mmproj := m.settings.MMProj
	go copyModelToLocal(ctx, src, mmproj, destDir, hw.DiskFreeBytes, ch)
	return m, waitCopyProgressCmd(ch)
}

// launchCmd keeps the port probe and the process creation off the UI thread.
func (m *Model) launchCmd() tea.Cmd {
	meta := m.meta
	settings := m.settings
	hwInfo := m.hw
	backend := m.backendDir()
	dir := filepath.Join(m.root, "bin", backend)
	exe := filepath.Join(dir, "llama-server.exe")
	logPath := filepath.Join(m.cfg.LogsDir(),
		fmt.Sprintf("alpaca-%s.log", time.Now().Format("20060102-150405")))
	quitCtx := m.quitCtx

	return func() tea.Msg {
		if quitCtx != nil && quitCtx.Err() != nil {
			return launchMsg{err: context.Canceled}
		}
		if _, err := os.Stat(exe); err != nil {
			return launchMsg{err: fmt.Errorf(
				"llama-server.exe not found in bin%c%s — run scripts/package.ps1",
				filepath.Separator, backend)}
		}
		port := settings.Port
		if !runner.PortFree(port) {
			port = runner.NextFreePort(port)
			settings.Port = port
		}
		resolved, _ := estimate.ResolveSettings(settings, meta, hwInfo)
		proc, err := runner.Start(exe, dir, resolved.Args(meta.Path), logPath)
		if err != nil {
			if hint := runner.MissingDLL(err, dir); hint != "" {
				return launchMsg{err: fmt.Errorf("%w — %s", err, hint)}
			}
			return launchMsg{err: err}
		}
		if quitCtx != nil && quitCtx.Err() != nil {
			_ = proc.StopTimeout(0)
			proc.Release()
			return launchMsg{err: context.Canceled}
		}
		return launchMsg{proc: proc, port: port}
	}
}

func saveConfigCmd(c *config.Config) tea.Cmd {
	return func() tea.Msg {
		if err := c.Save(); err != nil {
			return toastMsg("could not save config.toml (read-only drive?): " + err.Error())
		}
		return nil
	}
}

func savePresetCmd(cfg *config.Config, p config.Preset) tea.Cmd {
	return func() tea.Msg {
		saved, err := cfg.SavePreset(p)
		if err != nil {
			return presetMsg{err: err}
		}
		ps, lerr := cfg.ListPresets()
		return presetMsg{presets: ps, err: lerr, saved: saved.Name}
	}
}

func renamePresetCmd(cfg *config.Config, p config.Preset, name string) tea.Cmd {
	return func() tea.Msg {
		if _, err := cfg.RenamePreset(p, name); err != nil {
			return presetMsg{err: err}
		}
		ps, err := cfg.ListPresets()
		return presetMsg{presets: ps, err: err, renamed: name}
	}
}

func deletePresetCmd(cfg *config.Config, p config.Preset) tea.Cmd {
	return func() tea.Msg {
		if err := cfg.DeletePreset(p.ID); err != nil {
			return presetMsg{err: err}
		}
		ps, err := cfg.ListPresets()
		return presetMsg{presets: ps, err: err}
	}
}

// openPathCmd accepts either a model file or a new models folder.
func openPathCmd(value string) tea.Cmd {
	return func() tea.Msg {
		value = strings.TrimSpace(strings.Trim(value, `"`))
		st, err := os.Stat(value)
		if err != nil {
			return pathMsg{err: fmt.Errorf("not found: %s", value)}
		}
		if st.IsDir() {
			abs, aerr := filepath.Abs(value)
			if aerr == nil {
				value = abs
			}
			return pathMsg{modelsDir: value}
		}
		if !strings.EqualFold(filepath.Ext(value), ".gguf") {
			return pathMsg{err: fmt.Errorf("%s is not a .gguf file or a folder", value)}
		}
		abs, aerr := filepath.Abs(value)
		if aerr == nil {
			value = abs
		}
		return pathMsg{model: value}
	}
}

func copyCmd(text string) tea.Cmd {
	return func() tea.Msg {
		if err := copyToClipboard(text); err != nil {
			return toastMsg("copy failed: " + err.Error())
		}
		return toastMsg("command copied to clipboard")
	}
}

func stopCmd(p *runner.Process, action string) tea.Cmd {
	return func() tea.Msg {
		p.Stop()
		return stoppedMsg{action: action}
	}
}

var clipboardWriter = copyClipboardClipExe

// copyToClipboard pipes text into the configured clipboard writer (clip.exe by default).
func copyToClipboard(s string) error {
	return clipboardWriter(s)
}

func copyClipboardClipExe(s string) error {
	cmd := exec.Command("clip.exe")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	if _, err := stdin.Write([]byte(s)); err != nil {
		return err
	}
	if err := stdin.Close(); err != nil {
		return err
	}
	return cmd.Wait()
}

func (m *Model) saveCurrentPreset() tea.Cmd {
	if m.meta == nil {
		m.toastf("no model selected")
		return nil
	}
	name := fmt.Sprintf("%s %dK", strings.TrimSuffix(filepath.Base(m.meta.Path), ".gguf"), m.settings.Context/1024)
	return savePresetCmd(m.cfg, config.Preset{
		Name:     name,
		Model:    filepath.Base(m.meta.Path),
		Settings: m.settings,
	})
}
