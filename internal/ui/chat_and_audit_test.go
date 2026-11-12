package ui

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/MdHussain121/alpaca/internal/flags"
	"github.com/MdHussain121/alpaca/internal/gguf"
	"github.com/MdHussain121/alpaca/internal/hw"
	"github.com/MdHussain121/alpaca/internal/runner"
)

func TestSlidersRenderAcrossSettingsAndGPULayersDecrementsFromAll(t *testing.T) {
	m := testUI(t)
	m.meta = testMeta(8_000_000_000, 120) // test >99 layer model too
	m.settings.GPULayers = flags.AllCount
	m.recompute()

	m.focus = 1
	m.width, m.height = 120, 42

	sliderLabels := []string{
		"Context",
		"GPU layers",
		"Threads",
		"Batch",
		"Ubatch",
		"Parallel slots",
		"Temp",
		"Top-p",
		"Top-k",
		"Min-p",
		"Reasoning effort",
		"Reasoning budget",
		"Max tokens",
		"Theme",
	}

	for _, label := range sliderLabels {
		idx := fieldIndexByLabel(m, label)
		if idx < 0 {
			t.Fatalf("missing field %q", label)
		}
		m.fieldIdx = idx
		v := m.View()
		if !strings.Contains(v, "●") {
			t.Fatalf("field %q did not render a slider thumb in view:\n%s", label, v)
		}
	}

	// Decrementing GPU layers from `all` (999) on a 120-layer model must step to 119,
	// not snap back up to AllCount.
	m.fieldIdx = fieldIndexByLabel(m, "GPU layers")
	m.adjustField(-1)
	if m.settings.GPULayers != 119 {
		t.Fatalf("GPU layers after Left from all on 120L model = %d, want 119", m.settings.GPULayers)
	}
	m.adjustField(1)
	if m.settings.GPULayers != flags.AllCount {
		t.Fatalf("GPU layers after Right from 119 = %d, want AllCount (%d)", m.settings.GPULayers, flags.AllCount)
	}

	// ASCII mode renders ASCII slider characters without Unicode glyph breakage.
	m.cfg.ASCII = true
	m.sty = newStyles(m.sty.t, true)
	vASCII := m.View()
	if !strings.Contains(vASCII, "O") {
		t.Fatalf("ASCII slider thumb missing:\n%s", vASCII)
	}
}

func TestModalsIsolateKeysAndRenderEditingInputAndPreserveRecalledThreads(t *testing.T) {
	// 1. Editing input modal renders typed text & rune-safe backspace on `/`.
	m := testUI(t)
	m.width, m.height = 100, 30
	press(m, "/")
	if !m.filtering {
		t.Fatal("`/` did not enter filter mode")
	}
	m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("模型")})
	if m.filter != "模型" {
		t.Fatalf("filter = %q, want \"模型\"", m.filter)
	}
	m.Update(tea.KeyMsg{Type: tea.KeyBackspace})
	if m.filter != "模" {
		t.Fatalf("after rune backspace filter = %q, want \"模\"", m.filter)
	}
	press(m, "esc")

	// 2. overlayInput must render input modal text on screen when editing a setting.
	m.focus = 1
	m.fieldIdx = fieldIndexByLabel(m, "System prompt")
	press(m, "enter")
	if !m.editing {
		t.Fatal("Enter on System prompt did not open inline/modal editor")
	}
	m.input.SetValue("Be concise.")
	if v := m.View(); !strings.Contains(v, "Be concise.") || !strings.Contains(v, "System prompt") {
		t.Fatalf("overlayInput did not render editor modal:\n%s", v)
	}
	press(m, "enter")
	if m.settings.SystemPrompt != "Be concise." {
		t.Fatalf("SystemPrompt = %q, want %q", m.settings.SystemPrompt, "Be concise.")
	}

	// 3. quitConfirm / lanConfirm / runQuitConfirm / backConfirm swallow stray navigation keys.
	m.quitConfirm = true
	beforeIdx := m.fieldIdx
	press(m, "down")
	press(m, "up")
	press(m, "t")
	if m.fieldIdx != beforeIdx || !m.quitConfirm {
		t.Fatalf("quitConfirm leaked keys or closed prematurely: fieldIdx=%d quitConfirm=%v", m.fieldIdx, m.quitConfirm)
	}
	press(m, "n")
	if m.quitConfirm {
		t.Fatal("`n` did not dismiss quitConfirm")
	}

	// 4. hwMsg must not overwrite recalled/user-tuned Threads after preset/last-used recall.
	m.userEditedThreads = true
	m.settings.Threads = 4
	m.Update(hwMsg{info: hw.Info{PhysCores: 12, LogicCores: 24, RAMTotal: 32 << 30, RAMAvail: 20 << 30}})
	if m.settings.Threads != 4 {
		t.Fatalf("hwMsg overwrote recalled Threads: got %d, want 4", m.settings.Threads)
	}

	// 5. selectCurrent calls recompute even when entry meta is still nil.
	m.models = append(m.models, modelEntry{cand: gguf.Candidate{RelPath: "Unloaded.gguf", Path: "unloaded.gguf"}, meta: nil})
	m.visible = []int{0, 1}
	m.cursor = 1
	m.selectCurrent()
	if m.meta != nil {
		t.Fatal("expected nil meta for unloaded entry")
	}
}

func TestLaunchModeChatEntersChatTabAndStreamsTelemetry(t *testing.T) {
	var captured struct {
		Effort string `json:"reasoning_effort"`
		Temp   float64
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions":
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("decode chat request: %v", err)
			}
			if e, _ := body["reasoning_effort"].(string); e != "" {
				captured.Effort = e
			}
			w.Header().Set("Content-Type", "text/event-stream")
			fl, _ := w.(http.Flusher)
			chunks := []string{
				`{"choices":[{"delta":{"reasoning_content":"Let me check. "}}]}`,
				`{"choices":[{"delta":{"content":"42"}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":120,"completion_tokens":28},"timings":{"prompt_n":120,"prompt_per_second":610.5,"predicted_n":28,"predicted_per_second":48.2}}`,
			}
			for _, c := range chunks {
				fmt.Fprintf(w, "data: %s\n\n", c)
				if fl != nil {
					fl.Flush()
				}
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
			if fl != nil {
				fl.Flush()
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	m := testUI(t)
	m.width, m.height = 120, 36
	m.settings.LaunchMode = "chat"
	m.settings.Context = 8192
	m.settings.Reasoning = "on"
	m.settings.ReasoningEffort = "medium"
	m.meta = testMeta(4_000_000_000, 36)
	m.recompute()

	_, _ = m.launch()
	if m.screen != screenRun || m.runTab != runTabChat {
		t.Fatalf("launch in chat mode: screen=%v runTab=%v", m.screen, m.runTab)
	}
	// Double-launch guard must ignore second launch while m.launching is true.
	if _, secondCmd := m.launch(); secondCmd != nil {
		t.Fatal("double-launch guard failed to suppress second launchCmd")
	}

	// Simulate server readiness: right after load, before inference, tok/s must NOT show 0.0 t/s.
	m.launching = false
	m.proc = &runner.Process{}
	m.status = "READY"
	m.chatBaseURL = srv.URL

	initialView := m.View()
	if strings.Contains(initialView, "0.0 t/s") {
		t.Fatalf("fresh READY view should show dash instead of 0.0 t/s:\n%s", initialView)
	}
	if !strings.Contains(initialView, "— t/s") {
		t.Fatalf("fresh READY view missing `— t/s` placeholder:\n%s", initialView)
	}

	// Cycle effort live via Ctrl+E -> high.
	m.Update(tea.KeyMsg{Type: tea.KeyCtrlE})
	if m.settings.ReasoningEffort != "high" {
		t.Fatalf("Ctrl+E ReasoningEffort = %q, want high", m.settings.ReasoningEffort)
	}

	// Send a message and pump SSE Bubble Tea commands to completion.
	m.chatInput.SetValue("What is 6*7?")
	cmd := press(m, "enter")
	for cmd != nil {
		msg := cmd()
		_, cmd = m.Update(msg)
	}

	if captured.Effort != "high" {
		t.Fatalf("sent reasoning_effort = %q, want high", captured.Effort)
	}
	if got := m.effectiveCtxUsed(); got != 148 {
		t.Fatalf("effectiveCtxUsed = %d, want 148 (120+28)", got)
	}
	if got := m.effectiveGenTokSec(); got != 48.2 {
		t.Fatalf("effectiveGenTokSec = %.1f, want 48.2", got)
	}

	// Telemetry poll updates host/process RAM and fallback rates.
	m.Update(telemetryMsg{
		ramTotal: 16 << 30,
		ramAvail: 8 << 30,
		procWS:   2 << 30,
		telemetry: runner.Telemetry{
			NCtx:         8192,
			GenTokPerSec: 48.2,
		},
	})

	chatView := m.View()
	for _, want := range []string{"Thought for", "42", "48.2 t/s", "148/8192", "effort high", "srv 2.0 GB"} {
		if !strings.Contains(chatView, want) {
			t.Fatalf("chat view missing %q:\n%s", want, chatView)
		}
	}
	if strings.Contains(chatView, "Let me check.") {
		t.Fatalf("completed thinking must auto-collapse:\n%s", chatView)
	}

	// Expand thinking trace via Ctrl+T, then collapse again.
	m.Update(tea.KeyMsg{Type: tea.KeyCtrlT})
	if v := m.View(); !strings.Contains(v, "Let me check.") {
		t.Fatalf("Ctrl+T did not expand collapsed reasoning trace:\n%s", v)
	}
	m.Update(tea.KeyMsg{Type: tea.KeyCtrlT})
	if !m.settings.HideThinking {
		t.Fatal("second Ctrl+T did not set HideThinking")
	}
	if v := m.View(); strings.Contains(v, "Let me check.") {
		t.Fatalf("second Ctrl+T did not collapse reasoning trace:\n%s", v)
	}

	// `/effort off` updates reasoning mode cleanly.
	m.chatInput.SetValue("/effort off")
	_ = press(m, "enter")
	if m.settings.Reasoning != "off" {
		t.Fatalf("/effort off left Reasoning = %q", m.settings.Reasoning)
	}

	// Tab switches between TUI Chat and llama-server log cleanly.
	press(m, "tab")
	if m.runTab != runTabLog {
		t.Fatalf("Tab did not switch to runTabLog: %v", m.runTab)
	}
	if v := m.View(); !strings.Contains(v, "llama-server log") {
		t.Fatalf("log tab view missing header:\n%s", v)
	}
	press(m, "tab")
	if m.runTab != runTabChat {
		t.Fatalf("Tab did not switch back to runTabChat: %v", m.runTab)
	}

	// `/clear` resets chat turns and context usage.
	m.chatInput.SetValue("/clear")
	_ = press(m, "enter")
	if m.effectiveCtxUsed() != 0 || len(m.chatMsgs) != 0 {
		t.Fatalf("/clear did not reset chat state: ctx=%d msgs=%d", m.effectiveCtxUsed(), len(m.chatMsgs))
	}
}

func TestTitleBarTrueColorContrastAndAllThemesValid(t *testing.T) {
	themes := AllThemes()
	if len(themes) < 10 {
		t.Fatalf("expected at least 10 themes, got %d", len(themes))
	}
	for _, th := range themes {
		s := newStyles(th, false)
		s.true = true
		bar := s.titleBar(90, " llm · launcher ", "CUDA · 8 GB")
		if w := lipgloss.Width(bar); w != 90 {
			t.Fatalf("theme %q titleBar width = %d, want 90", th.Name, w)
		}
		// Must set dark foreground matching Bg hex so title text stays legible on bright accent gradients.
		fg := hexRGB(string(th.Bg))
		wantFG := fmt.Sprintf("38;2;%d;%d;%d", fg[0], fg[1], fg[2])
		if !strings.Contains(bar, wantFG) {
			t.Fatalf("theme %q TrueColor titleBar missing Bg foreground %q", th.Name, wantFG)
		}
	}
}

func TestChatAndLogScrollingWithKeysAndMouseWheel(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 100, 26
	m.screen = screenRun
	m.runTab = runTabChat
	m.status = "READY"
	m.resizeRunViews()

	// Add enough turns to overflow the chat viewport cleanly.
	var long strings.Builder
	for i := 0; i < 45; i++ {
		fmt.Fprintf(&long, "Line %02d of assistant output to test vertical scrolling.\n", i)
	}
	m.chatMsgs = []chatTurn{
		{Role: "user", Content: "Scroll check"},
		{Role: "assistant", Content: long.String()},
	}
	m.refreshChatView(true)
	if !m.chatView.AtBottom() {
		t.Fatal("expected initial chat view at bottom")
	}
	bottomOffset := m.chatView.YOffset
	if bottomOffset <= 0 {
		t.Fatalf("expected scrollable chat viewport, YOffset=%d", bottomOffset)
	}

	// Arrow Up and Mouse Wheel Up must scroll UP in TUI Chat even with chatInput focused.
	m.chatInput.Focus()
	press(m, "up")
	if m.chatView.YOffset >= bottomOffset {
		t.Fatalf("Up key did not scroll chat up: before=%d after=%d", bottomOffset, m.chatView.YOffset)
	}
	afterUp := m.chatView.YOffset
	m.Update(tea.MouseMsg{Button: tea.MouseButtonWheelUp, Action: tea.MouseActionPress})
	if m.chatView.YOffset >= afterUp {
		t.Fatalf("Mouse wheel up did not scroll chat up: before=%d after=%d", afterUp, m.chatView.YOffset)
	}
	if v := m.View(); !strings.Contains(v, "↑") {
		t.Fatalf("chat header missing scrolled-up indicator:\n%s", v)
	}

	// Streaming deltas while scrolled up must NOT yank user back to the bottom.
	savedOffset := m.chatView.YOffset
	m.handleChatDelta(runner.ChatDelta{Content: "more tokens"})
	if m.chatView.YOffset != savedOffset {
		t.Fatalf("handleChatDelta yanked scroll position from %d to %d", savedOffset, m.chatView.YOffset)
	}

	// Down key and Mouse Wheel Down scroll back down cleanly.
	m.Update(tea.KeyMsg{Type: tea.KeyEnd})
	if !m.chatView.AtBottom() {
		t.Fatal("End key did not return chat to bottom")
	}

	// Switch to Logs tab and verify scroll up + appendLog preservation.
	press(m, "tab")
	for i := 0; i < 40; i++ {
		m.appendLog(fmt.Sprintf("log line %02d", i))
	}
	m.Update(tea.MouseMsg{Button: tea.MouseButtonWheelUp, Action: tea.MouseActionPress})
	if m.logView.AtBottom() {
		t.Fatal("Mouse wheel up did not scroll logs up")
	}
	logOffset := m.logView.YOffset
	m.appendLog("new telemetry line")
	if m.logView.YOffset != logOffset {
		t.Fatalf("appendLog yanked scrolled log view from %d to %d", logOffset, m.logView.YOffset)
	}
}

func TestCtrlCExitsImmediatelyFromEveryScreenAndOverlay(t *testing.T) {
	states := []struct {
		name  string
		setup func(m *Model)
	}{
		{"launcher", func(m *Model) { m.screen = screenLauncher }},
		{"editing", func(m *Model) { m.editing = true; m.input.Focus() }},
		{"quitConfirm", func(m *Model) { m.quitConfirm = true }},
		{"runChatStreaming", func(m *Model) {
			m.screen = screenRun
			m.runTab = runTabChat
			m.chatStreaming = true
			m.proc = &runner.Process{}
		}},
		{"runLogConfirm", func(m *Model) {
			m.screen = screenRun
			m.runTab = runTabLog
			m.runQuitConfirm = true
			m.proc = &runner.Process{}
		}},
	}
	for _, st := range states {
		t.Run(st.name, func(t *testing.T) {
			m := testUI(t)
			st.setup(m)
			_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
			if !m.quit {
				t.Fatalf("%s: ctrl+c did not set m.quit=true", st.name)
			}
			if cmd == nil {
				t.Fatalf("%s: ctrl+c did not return tea.Quit command", st.name)
			}
			if m.quitCtx.Err() == nil {
				t.Fatalf("%s: ctrl+c did not cancel quitCtx", st.name)
			}
		})
	}
}

func TestRunViewShowsModelAndKVOffloadSplit(t *testing.T) {
	m := testUI(t)
	m.width, m.height = 160, 40
	m.screen = screenRun
	m.status = "READY"
	m.metrics = runner.Metrics{
		KVActualMiB:   300,
		GPUKVMiB:      200,
		CPUKVMiB:      100,
		LayersOnGPU:   20,
		TotalLayers:   36,
		CPUModelMiB:   2400,
		CPUComputeMiB: 64,
	}
	m.Update(telemetryMsg{
		ramTotal: 16 << 30,
		ramAvail: 8 << 30,
		procWS:   5 << 30,
	})
	v := m.View()
	for _, want := range []string{"20/36L (CPU 16L)", "200M VRAM+100M RAM", "buf 2564M · srv 5.0 GB"} {
		if !strings.Contains(v, want) {
			t.Fatalf("offload status missing %q:\n%s", want, v)
		}
	}
}
