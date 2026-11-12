package ui

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/spinner"

	"github.com/MdHussain121/alpaca/internal/hw"
	"github.com/MdHussain121/alpaca/internal/runner"
)

func TestPerformanceTargets(t *testing.T) {
	t0 := time.Now()
	m := testUI(t)
	m.width, m.height = 160, 40
	firstFrame := m.View()
	firstPaint := time.Since(t0)
	if len(firstFrame) == 0 {
		t.Fatal("empty initial view")
	}
	if firstPaint > 300*time.Millisecond {
		t.Fatalf("first paint = %v, want < 300ms", firstPaint)
	}

	// Render latency at 160x40 across Launcher, Help, Presets, and Run Chat must be < 5ms.
	const iters = 50
	start := time.Now()
	for i := 0; i < iters; i++ {
		_ = m.View()
	}
	launcherAvg := time.Since(start) / iters
	if !raceDetectorActive && launcherAvg > 5*time.Millisecond {
		t.Fatalf("launcher View() avg = %v, want < 5ms", launcherAvg)
	}

	m.screen = screenRun
	m.status = "READY"
	m.runTab = runTabChat
	m.chatMsgs = []chatTurn{
		{Role: "user", Content: strings.Repeat("Explain quicksort cleanly. ", 8)},
		{Role: "assistant", Reasoning: strings.Repeat("thinking step ", 20), Content: strings.Repeat("Quicksort partitions arrays in O(n log n) average time. ", 12), GenTokSec: 72.5, Tokens: 120},
	}
	m.resizeRunViews()
	start = time.Now()
	for i := 0; i < iters; i++ {
		_ = m.View()
	}
	chatAvg := time.Since(start) / iters
	if !raceDetectorActive && chatAvg > 5*time.Millisecond {
		t.Fatalf("run chat View() avg = %v, want < 5ms", chatAvg)
	}

	// Idle screens must NOT re-arm spinner or poll ticks.
	m.screen = screenLauncher
	m.scanning = false
	if _, cmd := m.Update(spinner.TickMsg{ID: m.spinner.ID()}); cmd != nil {
		t.Fatal("idle launcher re-armed spinner tick")
	}
	if _, cmd := m.Update(tickMsg(time.Now())); cmd != nil {
		t.Fatal("idle launcher re-armed run poll tick")
	}
	m.screen = screenRun
	m.status = "STOPPED"
	m.proc = nil
	if _, cmd := m.Update(spinner.TickMsg{ID: m.spinner.ID()}); cmd != nil {
		t.Fatal("stopped run screen re-armed spinner tick")
	}

	// 100k-line log flood via batched updates must stay bounded and < 60 MB RSS.
	m.screen = screenRun
	m.status = "READY"
	m.runTab = runTabLog
	batch := make([]string, 500)
	for i := range batch {
		batch[i] = fmt.Sprintf("2026-11-12 INFO server slot %03d generated token sample line for memory flood benchmark", i)
	}
	for b := 0; b < 200; b++ { // 200 * 500 = 100,000 lines
		m.appendLogBatch(batch)
	}
	if len(m.logs) > logRingMax {
		t.Fatalf("log ring exceeded max: %d > %d", len(m.logs), logRingMax)
	}
	runtime.GC()
	debug.FreeOSMemory()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	ws, _, _ := hw.ProcessRAM(uint32(os.Getpid()))
	t.Logf("firstPaint=%v launcherAvg=%v chatAvg=%v heap=%.1fMiB procWS=%.1fMiB",
		firstPaint, launcherAvg, chatAvg, float64(ms.HeapAlloc)/(1<<20), float64(ws)/(1<<20))
	if ms.HeapAlloc > 60<<20 {
		t.Fatalf("heap after 100k log flood = %.1f MiB, want < 60 MiB", float64(ms.HeapAlloc)/(1<<20))
	}
	if !raceDetectorActive && ws > 0 && ws > 60<<20 {
		t.Fatalf("process RSS after 100k log flood = %.1f MiB, want < 60 MiB", float64(ws)/(1<<20))
	}
}

func TestChatCoalescingAndWrapCache(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	raw := make(chan runner.ChatDelta, 512)
	out := make(chan runner.ChatDelta, 64)
	go coalesceChatDeltas(ctx, raw, out)

	// Push 300 rapid tokens immediately (~<2ms wall time).
	for i := 0; i < 300; i++ {
		raw <- runner.ChatDelta{Content: "tok ", GenTokPerSec: 80.0}
	}
	raw <- runner.ChatDelta{Done: true}
	close(raw)

	var got []runner.ChatDelta
	var combined strings.Builder
	for d := range out {
		got = append(got, d)
		combined.WriteString(d.Content)
	}
	// First token flushes immediately + remaining burst coalesces -> far fewer than 300 frames.
	if len(got) > 16 {
		t.Fatalf("coalescer emitted %d frames for 300 rapid tokens, expected <= 16", len(got))
	}
	wantText := strings.Repeat("tok ", 300)
	if combined.String() != wantText {
		t.Fatalf("coalesced text length = %d, want %d", combined.Len(), len(wantText))
	}

	// Verify historical turns reuse cached rendered lines without re-wrapping.
	m := testUI(t)
	m.width, m.height = 120, 36
	m.chatMsgs = []chatTurn{
		{Role: "user", Content: "First prompt"},
		{Role: "assistant", Content: "First answer"},
		{Role: "assistant", Content: ""},
	}
	m.chatStreaming = true
	m.resizeRunViews()
	firstCachedPtr := &m.chatMsgs[1].cache.rendered[0]

	m.handleChatDelta(runner.ChatDelta{Content: "Streaming delta"})
	if &m.chatMsgs[1].cache.rendered[0] != firstCachedPtr {
		t.Fatal("historical turn cache was unnecessarily invalidated during streaming delta")
	}
}

func BenchmarkViewLauncher160x40(b *testing.B) {
	m := testUI(&testing.T{})
	m.width, m.height = 160, 40
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.View()
	}
}

func BenchmarkViewRunChat160x40(b *testing.B) {
	m := testUI(&testing.T{})
	m.width, m.height = 160, 40
	m.screen = screenRun
	m.status = "READY"
	m.runTab = runTabChat
	for i := 0; i < 10; i++ {
		m.chatMsgs = append(m.chatMsgs,
			chatTurn{Role: "user", Content: "How do I optimize Go allocations in a terminal UI?"},
			chatTurn{Role: "assistant", Reasoning: "Think about pre-built styles and caching wrapped lines.", Content: "Pre-build lipgloss styles once, cache wrapped chat turns, and coalesce token redraws.", GenTokSec: 68.4, Tokens: 45},
		)
	}
	m.resizeRunViews()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.View()
	}
}

func BenchmarkLogBatch100k(b *testing.B) {
	batch := make([]string, 500)
	for i := range batch {
		batch[i] = fmt.Sprintf("eval time = 120.50 ms / 10 runs ( 82.99 tokens per second)")
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		m := testUI(&testing.T{})
		m.screen = screenRun
		m.status = "READY"
		for c := 0; c < 200; c++ {
			m.appendLogBatch(batch)
		}
	}
}
