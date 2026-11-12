package ui

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/charmbracelet/lipgloss"

	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/gguf"
	"github.com/MdHussain121/alpaca/internal/runner"
)

// View renders the current screen. The frame is always exactly width x height.
func (m *Model) View() string {
	if !m.ready {
		return "starting alpaca…"
	}
	if m.showHelp {
		return m.viewHelp()
	}
	if m.width < 80 || m.height < 24 {
		return m.viewTooSmall()
	}
	if m.screen == screenRun {
		return m.viewRun()
	}
	return m.viewLauncher()
}

func (m *Model) backendBadge() string {
	if m.hw.HasCUDA {
		gpu := m.hw.GPUName
		if gpu == "" {
			gpu = "CUDA"
		}
		gpu = truncate(gpu, 24)
		cudaCount := 0
		for _, d := range m.hw.Devices {
			if d.Backend == "cuda" {
				cudaCount++
			}
		}
		if cudaCount > 1 {
			return fmt.Sprintf(" ● CUDA (%dx) · %s %s GB ", cudaCount, gpu, trimGB(m.hw.VRAMTotal))
		}
		return fmt.Sprintf(" ● CUDA · %s %s GB ", gpu, trimGB(m.hw.VRAMTotal))
	}
	name := m.backendDir()
	if name == "vulkan" && m.hw.GPUName != "" {
		return fmt.Sprintf(" ● VULKAN · %s ", truncate(m.hw.GPUName, 20))
	}
	if name == "cpu" {
		return " ● CPU "
	}
	return " ● " + strings.ToUpper(name) + " "
}

func trimGB(b uint64) string {
	v := float64(b) / (1 << 30)
	if v == float64(int64(v)) {
		return fmt.Sprintf("%.0f", v)
	}
	return fmt.Sprintf("%.1f", v)
}

func (m *Model) viewLauncher() string {
	w, h := m.width, m.height

	title := m.sty.titleBar(w, " alpaca  ·  portable model launcher ", m.backendBadge())

	const cmdH, memH, helpH = 4, 5, 1
	var bannerBlock string
	bannerH := 0
	if h >= 30 {
		bannerLines := strings.Split(BrandBanner(m.sty.ascii), "\n")
		styled := make([]string, len(bannerLines))
		for i, bl := range bannerLines {
			if i == 0 {
				styled[i] = m.sty.accent().Render(bl)
			} else if i == 1 {
				styled[i] = m.sty.accent2().Render(bl)
			} else {
				styled[i] = m.sty.muted().Render(bl)
			}
		}
		bannerBlock = strings.Join(styled, "\n")
		bannerH = len(bannerLines)
	}
	bodyH := h - 1 - bannerH - cmdH - memH - helpH
	if bodyH < 8 {
		bodyH = 8
	}

	var body string
	if w >= 100 {
		leftW := w * 38 / 100
		if leftW < 34 {
			leftW = 34
		}
		rightW := w - leftW
		body = lipgloss.JoinHorizontal(lipgloss.Top,
			m.viewModels(leftW, bodyH),
			m.viewSettings(rightW, bodyH),
		)
	} else {
		top := bodyH * 45 / 100
		if top < 4 {
			top = 4
		}
		body = lipgloss.JoinVertical(lipgloss.Left,
			m.viewModels(w, top),
			m.viewSettings(w, bodyH-top),
		)
	}

	cmd := m.viewCommand(w, cmdH)
	mem := m.viewMemory(w, memH)
	help := m.viewHelpBar(w)

	var out string
	if bannerBlock != "" {
		out = lipgloss.JoinVertical(lipgloss.Left, title, bannerBlock, body, mem, cmd, help)
	} else {
		out = lipgloss.JoinVertical(lipgloss.Left, title, body, mem, cmd, help)
	}

	if m.showPreset {
		out = m.overlayPresets(out)
	}
	if m.editing {
		out = m.overlayInput(out)
	}
	if m.copying {
		out = m.overlayCopyProgress(out)
	}
	if m.lanConfirm {
		out = m.overlayConfirm(out, "Expose llama-server on the LAN?", "no API key is set — y expose · n keep 127.0.0.1")
	}
	if m.quitConfirm {
		out = m.overlayConfirm(out, "Stop llama-server and quit?", "y quit · n stay")
	}
	if m.palette.open {
		out = m.overlayPalette(out)
	}
	if m.toolsModal != nil {
		out = m.overlayToolsModal(out)
	}
	if m.mcpModal != nil {
		out = m.overlayMCPModal(out)
	}
	if m.permsModal != nil {
		out = m.overlayPermsModal(out)
	}
	if m.skillsModal != nil {
		out = m.overlaySkillsModal(out)
	}
	if m.contextModal != nil {
		out = m.overlayContextModal(out)
	}
	if m.attachModal != nil {
		out = m.overlayAttachModal(out)
	}
	if m.sessionsModal != nil {
		out = m.overlaySessionsModal(out)
	}
	if m.permModal != nil {
		out = m.overlayPermModal(out)
	}
	if m.skillApproveModal != nil {
		out = m.overlaySkillApproveModal(out)
	}
	if m.askModal != nil {
		out = m.overlayAskModal(out)
	}
	if m.toast != "" {
		out = m.overlayToast(out)
	}
	return frame(out, w, h)
}

// frame guarantees exactly h lines so the alt screen never scrolls.
func frame(s string, w, h int) string {
	lines := strings.Split(s, "\n")
	if len(lines) > h {
		lines = lines[:h]
	}
	for len(lines) < h {
		lines = append(lines, "")
	}
	return strings.Join(lines, "\n")
}

func (m *Model) viewModels(w, h int) string {
	innerW := w - 4
	innerH := h - 2
	if innerW < 4 || innerH < 2 {
		return ""
	}
	bodyH := innerH - 1 // panel reserves row 0 for title

	body := make([]string, 0, bodyH)

	switch {
	case m.scanning:
		body = append(body, m.sty.muted().Render(truncate(" "+m.spinner.View()+" scanning models…", innerW)))
	case m.scanErr != nil:
		body = append(body, m.sty.danger().Render(truncate(" scan failed: "+m.scanErr.Error(), innerW)))
	case m.filtering:
		body = append(body, m.sty.accent().Render(truncate("/"+m.filter+"_", innerW)))
	case m.filter != "":
		body = append(body, m.sty.muted().Render(truncate("filter: "+m.filter, innerW)))
	default:
		body = append(body, m.sty.muted().Render(truncate(fmt.Sprintf("%d model(s)", len(m.models)), innerW)))
	}

	if len(m.visible) == 0 && !m.scanning {
		body = append(body, "")
		body = append(body, m.sty.muted().Render(truncate(" no .gguf files in models/", innerW)))
		return m.sty.panel(w, h, "Models", m.focus == 0, body)
	}

	rowsAvail := (bodyH - 1) / 2
	if rowsAvail < 1 {
		rowsAvail = 1
	}
	start := 0
	if m.cursor >= rowsAvail {
		start = m.cursor - rowsAvail + 1
	}
	end := start + rowsAvail
	if end > len(m.visible) {
		end = len(m.visible)
	}

	for pos := start; pos < end; pos++ {
		idx := m.visible[pos]
		if idx < 0 || idx >= len(m.models) {
			continue
		}
		e := m.models[idx]
		selected := pos == m.cursor

		// The file name is what the user recognises and what presets are keyed
		// on. Some GGUFs carry a hash or a different model's name in
		// general.name, so that field is not used as the label.
		name := strings.TrimSuffix(filepath.Base(e.cand.RelPath), ".gguf")

		cursorMark := "  "
		if selected && m.focus == 0 {
			cursorMark = "▸ "
		} else if selected {
			cursorMark = "• "
		}

		line1 := truncate(cursorMark+name, innerW)
		if selected {
			line1 = m.sty.accent().Render(line1)
		} else {
			line1 = m.sty.text().Render(line1)
		}
		body = append(body, line1)

		detail := ""
		switch {
		case e.err != nil:
			detail = "unreadable: " + e.err.Error()
		case e.meta == nil:
			detail = "reading header…"
		default:
			detail = fmt.Sprintf("%s · %s · %dL", e.meta.Quant, e.meta.ParamsString(), modellayers(e.meta))
			if e.meta.Arch != "" {
				detail += " · " + e.meta.Arch
			}
			if e.meta.IsMoE() {
				detail += " · MoE"
			}
		}
		badgeStr := ""
		if e.err == nil && e.meta != nil {
			if rating, ok := config.EvalRatingForFile(e.cand.RelPath); ok {
				label := fmt.Sprintf("tools:%s (%.0f%%)", rating.Badge, rating.SuccessRatePct)
				switch rating.Badge {
				case config.EvalBadgeRecommended:
					badgeStr = " · " + m.sty.success().Render(label)
				case config.EvalBadgeWorksLimits:
					badgeStr = " · " + m.sty.warning().Render(label)
				default:
					badgeStr = " · " + m.sty.danger().Render(label)
				}
			}
		}
		var d string
		if e.err != nil {
			d = m.sty.danger().Render(truncate("  "+detail, innerW))
		} else if badgeStr != "" {
			d = truncateANSI("  "+m.sty.muted().Render(detail)+badgeStr, innerW)
		} else {
			d = m.sty.muted().Render(truncate("  "+detail, innerW))
		}
		body = append(body, d)

		if len(body) >= bodyH {
			break
		}
	}

	return m.sty.panel(w, h, "Models", m.focus == 0, body)
}

func modellayers(meta *gguf.Metadata) int {
	if meta.BlockCount > 0 {
		return int(meta.BlockCount)
	}
	return len(meta.Layers)
}

func (m *Model) viewSettings(w, h int) string {
	innerW := w - 4
	innerH := h - 2
	if innerW < 8 || innerH < 2 {
		return ""
	}
	bodyH := innerH - 1 // panel reserves row 0 for title
	if bodyH < 1 {
		bodyH = 1
	}

	if m.meta == nil {
		body := []string{
			"",
			m.sty.muted().Render(truncate("  select a model to configure it", innerW)),
		}
		if m.cfg != nil && m.cfg.HasWebSearchKeyInFile() {
			body = append(body, "", m.sty.warning().Render(truncate("  ⚠ API key in config.toml travels with the drive", innerW)))
		}
		return m.sty.panel(w, h, "Settings", m.focus == 1, body)
	}

	fs := m.visibleFields()
	if m.fieldIdx >= len(fs) && len(fs) > 0 {
		m.fieldIdx = len(fs) - 1
	}

	hintRows := 0
	if bodyH >= 4 {
		hintRows = 1
	}
	rowsAvail := bodyH - hintRows
	if rowsAvail < 1 {
		rowsAvail = 1
	}

	start := 0
	if m.fieldIdx >= rowsAvail {
		start = m.fieldIdx - rowsAvail + 1
	}
	end := start + rowsAvail
	if end > len(fs) {
		end = len(fs)
	}

	labelW := 18
	if innerW < 42 {
		labelW = 14
	}
	sliderW := 0
	switch {
	case innerW >= 58:
		sliderW = 10
	case innerW >= 46:
		sliderW = 8
	}

	body := make([]string, 0, bodyH)
	for i := start; i < end; i++ {
		f := fs[i]
		selected := i == m.fieldIdx && m.focus == 1
		marker := "  "
		if selected {
			marker = "▸ "
		}

		val := f.value()
		note := ""
		if f.note != nil {
			note = f.note()
		}

		var sliderPart string
		if sliderW > 0 && f.slider != nil {
			if ratio, ok := f.slider(); ok {
				sliderPart = m.sty.sliderRatio(sliderW, ratio) + " "
			}
		}

		prefixTxt := marker + pad(f.label, labelW) + " "
		prefixW := displayWidth(prefixTxt)
		sliderUsed := 0
		if sliderPart != "" {
			sliderUsed = sliderW + 1
		}
		valAvail := innerW - prefixW - sliderUsed
		if valAvail < 4 {
			valAvail = 4
		}
		valTrunc := truncate(val, valAvail)

		var line string
		if selected {
			line = m.sty.accent().Render(prefixTxt) + sliderPart + m.sty.accent().Bold(true).Render(valTrunc)
		} else {
			line = m.sty.text().Render(prefixTxt) + sliderPart + m.sty.text().Render(valTrunc)
		}

		if note != "" {
			used := prefixW + sliderUsed + displayWidth(valTrunc)
			rem := innerW - used - 2
			if rem >= 6 {
				line += "  " + m.sty.muted().Render(truncate(note, rem))
			}
		}
		body = append(body, truncateANSI(line, innerW))
	}

	if hintRows > 0 {
		for len(body) < bodyH-1 {
			body = append(body, "")
		}
		if m.cfg != nil && m.cfg.HasWebSearchKeyInFile() {
			warn := "  ⚠ API key in config.toml travels with the drive"
			body = append(body, m.sty.warning().Render(truncate(warn, innerW)))
		} else {
			scrollHint := fmt.Sprintf("%d/%d", m.fieldIdx+1, len(fs))
			hint := fmt.Sprintf("  tab panels · ←→ slide · enter edit · t theme · %s", scrollHint)
			body = append(body, m.sty.muted().Render(truncate(hint, innerW)))
		}
	}

	return m.sty.panel(w, h, "Settings", m.focus == 1, body)
}

func (m *Model) slowDriveOversizedRAM() bool {
	if m.meta == nil {
		return false
	}
	if m.modelDriveKind != "removable" && m.modelDriveKind != "network" {
		return false
	}
	modelBytes := uint64(0)
	if m.meta.FileSize > 0 {
		modelBytes = uint64(m.meta.FileSize)
	} else if m.meta.TotalBytes > 0 {
		modelBytes = m.meta.TotalBytes
	}
	availRAM := m.est.RAMAvail
	if availRAM == 0 {
		availRAM = m.hw.RAMAvail
	}
	if availRAM == 0 {
		availRAM = m.hw.RAMTotal
	}
	return availRAM > 0 && modelBytes > availRAM
}

func (m *Model) viewMemory(w, h int) string {
	innerW := w - 4
	if innerW < 20 {
		return ""
	}

	vramTxt := fmt.Sprintf("%s/%s GB", fmtGB(m.est.VRAMUsed()), fmtGB(m.est.VRAMAvail))
	ramTxt := fmt.Sprintf("%s/%s GB", fmtGB(m.est.RAMUsed()), fmtGB(m.est.RAMAvail))

	var body []string
	// "VRAM " + gauge + " " + txt + "   RAM " + gauge + " " + txt
	const fixed = 14
	gaugeW := (innerW - fixed - len(vramTxt) - len(ramTxt)) / 2
	if gaugeW >= 5 {
		line := "VRAM " + m.sty.gauge(gaugeW, m.est.VRAMRatio()) + " " + vramTxt +
			"   RAM " + m.sty.gauge(gaugeW, m.est.RAMRatio()) + " " + ramTxt
		body = append(body, truncateANSI(line, innerW))
	} else {
		body = append(body, m.sty.text().Render(
			truncate("VRAM "+vramTxt+"   RAM "+ramTxt, innerW)))
	}

	// The second line is the warning when there is one, otherwise the breakdown.
	switch {
	case m.meta != nil && !m.est.Fits() && m.slowDriveOversizedRAM():
		body = append(body, m.sty.danger().Render(
			truncate("  ✗ over budget & > avail RAM on slow drive — press C to copy local", innerW)))
	case m.meta != nil && !m.est.Fits() && m.est.CommitAvail > 0 && m.est.CommitRatio() > 1.0 && m.est.VRAMRatio() <= 1.0 && m.est.RAMRatio() <= 1.0:
		body = append(body, m.sty.danger().Render(
			truncate("  ✗ over Windows commit limit — raise pagefile or lower ctx", innerW)))
	case m.meta != nil && !m.est.Fits():
		msg := "  ✗ over budget — offload layers/MoE/KV to fit VRAM & RAM"
		if strings.Contains(m.est.KVReason, "-nkvo") && !m.settings.NoKVOffload && m.est.VRAMRatio() > 1.0 {
			msg = "  ✗ over budget (>16K blocks q4_0 KV) — try -nkvo or lower context"
		}
		body = append(body, m.sty.danger().Render(
			truncate(msg, innerW)))
	case m.slowDriveOversizedRAM():
		body = append(body, m.sty.warning().Render(
			truncate("  ! model > avail RAM on slow drive — press C to copy local", innerW)))
	case m.meta != nil && m.est.LoadModeWarning != "":
		body = append(body, m.sty.warning().Render(
			truncate("  "+m.est.LoadModeWarning, innerW)))
	case m.meta != nil && m.est.CommitAvail > 0 && m.est.CommitRatio() > 0.90:
		body = append(body, m.sty.warning().Render(
			truncate("  ! near Windows commit limit — close apps or raise pagefile", innerW)))
	case m.meta != nil && (m.est.VRAMRatio() > 0.85 || m.est.RAMRatio() > 0.85):
		body = append(body, m.sty.warning().Render(
			truncate("  ! tight fit — leave headroom for KV cache & OS RAM", innerW)))
	default:
		line := "  " + m.est.Breakdown()
		if m.est.UnifiedMemory {
			line += " · shared iGPU/RAM"
		}
		if m.meta != nil && m.est.KVQualityNote != "" {
			line += " · " + m.est.KVQualityNote
		} else {
			if m.est.Approximate {
				line += " · approx"
			}
			if len(m.est.Notes) > 0 {
				line += " · " + m.est.Notes[0]
			} else if !m.hw.HasCUDA && m.hw.CUDAUnsupportedReason != "" {
				line += " · " + m.hw.CUDAUnsupportedReason
			}
		}
		body = append(body, m.sty.muted().Render(truncate(line, innerW)))
	}

	return m.sty.panel(w, h, "Memory", false, body)
}

func (m *Model) viewCommand(w, h int) string {
	innerW := w - 4
	if innerW < 10 {
		return ""
	}
	line := m.cmdPreview
	if line == "" {
		line = "llama-server (no model selected)"
	}
	// Keep the program name and the trailing flags; elide the model path.
	if len([]rune(line)) > innerW {
		const prefix = "llama-server "
		r := []rune(line)
		keep := innerW - len(prefix) - 1
		if keep < 4 {
			keep = 4
		}
		if keep > len(r) {
			keep = len(r)
		}
		line = prefix + "…" + string(r[len(r)-keep:])
	}
	body := []string{m.sty.muted().Render(truncate(line, innerW))}
	cmdTitle := "Command"
	if m.meta != nil && m.modelDriveKind == "removable" && m.modelEstLoad >= 5*time.Second {
		cmdTitle = fmt.Sprintf("Command · USB est %s load (C copy local)", fmtETA(m.modelEstLoad))
	}
	return m.sty.panel(w, h, cmdTitle, false, body)
}

func (m *Model) viewHelpBar(w int) string {
	var hints [][2]string
	switch {
	case m.showPreset:
		hints = [][2]string{{"↑↓", "pick"}, {"enter", "load"}, {"r", "rename"}, {"d", "delete"}, {"esc", "close"}}
	case m.focus == 0:
		hints = [][2]string{
			{"↑↓", "model"}, {"tab", "settings"}, {"enter", "launch"},
			{"/", "filter"}, {"o", "path"}, {"p", "presets"}, {"s", "save"},
			{"c", "copy"}, {"C", "local"}, {"t", "theme"}, {"?", "help"}, {"q", "quit"},
		}
	default:
		hints = [][2]string{
			{"↑↓", "field"}, {"←→", "slide"}, {"enter", "edit/toggle"},
			{"tab", "models"}, {"o", "path"}, {"p", "presets"}, {"s", "save"},
			{"t", "theme"}, {"?", "help"}, {"q", "quit"},
		}
	}
	return m.sty.helpBar(w, hints)
}

func (m *Model) viewRun() string {
	w, h := m.width, m.height

	right := " " + m.status + " "
	title := m.sty.titleBar(w, " alpaca  ·  "+m.modelTitle(), right)

	metrics := m.viewMetrics(w)
	panelH := h - 3
	if panelH < 6 {
		panelH = 6
	}

	var mainPanel, help string
	if m.runTab == runTabChat {
		mainPanel = m.viewChatPanel(w, panelH)
		help = m.sty.helpBar(w, [][2]string{
			{"enter", "send"}, {"tab", "logs"}, {"↑↓/pgup", "scroll"}, {"ctrl+e", "effort"},
			{"ctrl+t", "thinking"}, {"ctrl+l", "clear"}, {"esc", "stop/back"}, {"ctrl+c", "quit"},
		})
	} else {
		m.logView.Width = w - 4
		m.logView.Height = panelH - 3
		if m.logView.Height < 1 {
			m.logView.Height = 1
		}
		logTitle := "llama-server log  [tab: chat]"
		if m.logView.TotalLineCount() > m.logView.VisibleLineCount() && !m.logView.AtBottom() {
			logTitle = fmt.Sprintf("llama-server log · ↑ %.0f%%  [tab: chat]", m.logView.ScrollPercent()*100)
		}
		logBody := strings.Split(m.logView.View(), "\n")
		mainPanel = m.sty.panel(w, panelH, logTitle, true, logBody)
		help = m.sty.helpBar(w, [][2]string{
			{"tab", "chat"}, {"↑↓", "scroll"}, {"ctrl+e", "effort"},
			{"x", "stop"}, {"r", "restart"}, {"esc", "back"}, {"ctrl+c", "quit"},
		})
	}

	out := lipgloss.JoinVertical(lipgloss.Left, title, metrics, mainPanel, help)

	if m.runQuitConfirm {
		out = m.overlayConfirm(out, "Stop llama-server and quit?", "y quit · n stay")
	}
	if m.backConfirm {
		out = m.overlayConfirm(out, "Stop llama-server and go back?", "y yes · n no")
	}
	if m.palette.open {
		out = m.overlayPalette(out)
	}
	if m.toolsModal != nil {
		out = m.overlayToolsModal(out)
	}
	if m.mcpModal != nil {
		out = m.overlayMCPModal(out)
	}
	if m.permsModal != nil {
		out = m.overlayPermsModal(out)
	}
	if m.skillsModal != nil {
		out = m.overlaySkillsModal(out)
	}
	if m.contextModal != nil {
		out = m.overlayContextModal(out)
	}
	if m.attachModal != nil {
		out = m.overlayAttachModal(out)
	}
	if m.sessionsModal != nil {
		out = m.overlaySessionsModal(out)
	}
	if m.permModal != nil {
		out = m.overlayPermModal(out)
	}
	if m.skillApproveModal != nil {
		out = m.overlaySkillApproveModal(out)
	}
	if m.askModal != nil {
		out = m.overlayAskModal(out)
	}
	if m.toast != "" {
		out = m.overlayToast(out)
	}
	return frame(out, w, h)
}

func (m *Model) viewChatPanel(w, h int) string {
	innerW := w - 4
	if innerW < 10 {
		innerW = 10
	}
	bodyH := h - 3
	if bodyH < 4 {
		bodyH = 4
	}

	chatH := bodyH - 4
	if chatH < 1 {
		chatH = 1
	}
	m.chatView.Width = innerW
	m.chatView.Height = chatH

	ctxTotal := m.effectiveCtxTotal()
	ctxUsed := m.effectiveCtxUsed()
	ctxRatio := 0.0
	if ctxTotal > 0 {
		ctxRatio = float64(ctxUsed) / float64(ctxTotal)
	}
	ctxPct := int(ctxRatio * 100)
	if ctxPct > 100 {
		ctxPct = 100
	}

	tokStr := "— t/s"
	if g := m.effectiveGenTokSec(); g > 0 {
		tokStr = fmt.Sprintf("%.1f t/s", g)
	}

	thinkLabel := "on"
	if m.settings.HideThinking {
		thinkLabel = "off"
	}

	barW := 10
	if innerW < 64 {
		barW = 6
	}
	statusLine := m.sty.muted().Render("ctx ") +
		m.sty.gauge(barW, ctxRatio) + " " +
		m.sty.text().Render(fmt.Sprintf("%s/%s (%d%%)", formatCompactTok(ctxUsed), formatCompactTok(ctxTotal), ctxPct)) +
		"  " + m.sty.muted().Render("gen ") + m.sty.success().Render(tokStr) +
		"  " + m.sty.muted().Render("effort ") + m.sty.accent().Render(m.reasoningBadgeLabel()) +
		"  " + m.sty.muted().Render("think ") + m.sty.text().Render(thinkLabel) +
		"  " + m.sty.muted().Render("prof:") + m.sty.accent().Render(m.effectiveProfileLabel()) +
		"  " + m.sty.muted().Render("perm:") + m.sty.accent2().Render(m.permissionBadgeLabel()) +
		"  " + m.sty.muted().Render("ws:") + m.sty.text().Render(truncate(m.workspaceLabel(), 16))
	if m.sessionDoc != nil && len(m.sessionDoc.Meta.Branches) > 1 {
		statusLine += "  " + m.sty.muted().Render("br:") + m.sty.accent2().Render(m.sessionDoc.Meta.ActiveBranch)
	}
	if m.sessionTainted {
		statusLine += "  " + m.sty.badge(" TAINTED ", m.sty.t.Warning)
	}

	divChar := "─"
	if m.sty.ascii {
		divChar = "-"
	}
	div := m.sty.muted().Render(strings.Repeat(divChar, innerW))

	body := make([]string, 0, bodyH)
	body = append(body, truncateANSI(statusLine, innerW))
	body = append(body, div)

	viewLines := strings.Split(m.chatView.View(), "\n")
	for i := 0; i < chatH; i++ {
		if i < len(viewLines) {
			body = append(body, viewLines[i])
		} else {
			body = append(body, "")
		}
	}
	if m.popup.open {
		if popLines := renderPopupBox(m.sty, m.popup, innerW, chatH); len(popLines) > 0 {
			start := 2 + chatH - len(popLines)
			if start < 2 {
				start = 2
			}
			for pi, pl := range popLines {
				if start+pi < 2+chatH {
					body[start+pi] = pl
				}
			}
		}
	}
	body = append(body, div)

	promptPrefix := m.sty.sAccentBold.Render("› ")
	if m.chatStreaming {
		promptPrefix = m.sty.sWarning.Bold(true).Render("⋯ ")
	}
	m.chatInput.Width = innerW - 3
	body = append(body, truncateANSI(promptPrefix+m.chatInput.View(), innerW))

	chatTitle := "TUI Chat  [tab: logs]"
	if m.chatView.TotalLineCount() > m.chatView.VisibleLineCount() && !m.chatView.AtBottom() {
		chatTitle = fmt.Sprintf("TUI Chat · ↑ %.0f%%  [tab: logs]", m.chatView.ScrollPercent()*100)
	}
	return m.sty.panel(w, h, chatTitle, true, body)
}

func formatCompactTok(n int) string {
	if n >= 10000 {
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	}
	return fmt.Sprintf("%d", n)
}

func (m *Model) modelTitle() string {
	if m.meta == nil {
		return "no model"
	}
	port := m.effectivePort()
	return truncate(m.meta.DisplayName(), 44) + fmt.Sprintf(" · http://127.0.0.1:%d/v1", port)
}

func (m *Model) viewMetrics(w int) string {
	bits := []string{}

	status := m.status
	colour := m.sty.t.Muted
	switch status {
	case "READY":
		colour = m.sty.t.Success
	case "LOADING":
		colour = m.sty.t.Warning
	case "ERROR":
		colour = m.sty.t.Error
	}
	bits = append(bits, m.sty.badge(" "+status+" ", colour))

	if m.statusDe != "" {
		bits = append(bits, m.sty.danger().Render(truncate(m.statusDe, 48)))
	}
	if m.status == "LOADING" {
		pct := ""
		if len(m.logs) > 0 {
			for i := len(m.logs) - 1; i >= 0 && i >= len(m.logs)-6; i-- {
				if p, ok := runner.ProgressFromLine(m.logs[i]); ok {
					pct = fmt.Sprintf(" %.0f%%", p*100)
					break
				}
			}
		}
		loadInfo := pct
		if m.launchETA >= 5*time.Second {
			elapsed := time.Since(m.launchAt)
			rem := m.launchETA - elapsed
			src := ""
			if m.launchDriveKind == "removable" {
				src = " · USB"
			}
			if rem > time.Second {
				loadInfo += fmt.Sprintf(" (%s left%s)", fmtETA(rem), src)
			} else {
				loadInfo += fmt.Sprintf(" (%s elapsed%s)", fmtDur(elapsed), src)
			}
		}
		bits = append(bits, m.spinner.View()+m.sty.warning().Render(loadInfo))
	}

	if m.status == "READY" {
		gen := m.effectiveGenTokSec()
		if gen > 0 {
			bits = append(bits, m.sty.muted().Render("gen"), m.sty.success().Render(fmt.Sprintf("%.1f t/s", gen)))
		} else {
			bits = append(bits, m.sty.muted().Render("gen"), m.sty.muted().Render("— t/s"))
		}
		if p := m.effectivePromptTokSec(); p > 0 {
			bits = append(bits, m.sty.muted().Render("prompt"), m.sty.text().Render(fmt.Sprintf("%.0f t/s", p)))
		}
		if m.metrics.KVActualMiB > 0 {
			kvStr := fmt.Sprintf("%.0f MiB", m.metrics.KVActualMiB)
			switch {
			case m.metrics.GPUKVMiB > 0 && m.metrics.CPUKVMiB > 0:
				kvStr = fmt.Sprintf("%.0fM VRAM+%.0fM RAM", m.metrics.GPUKVMiB, m.metrics.CPUKVMiB)
			case m.metrics.CPUKVMiB > 0:
				kvStr += " RAM"
			case m.metrics.GPUKVMiB > 0:
				kvStr += " VRAM"
			}
			if w >= 140 {
				if estMiB := float64(m.est.KVVRAM+m.est.KVRAM) / (1 << 20); estMiB > 0 {
					kvStr += fmt.Sprintf(" (est %.0f)", estMiB)
				}
			}
			bits = append(bits, m.sty.muted().Render("KV"), m.sty.text().Render(kvStr))
		}
		if m.metrics.TotalLayers > 0 {
			offLabel := fmt.Sprintf("%d/%dL", m.metrics.LayersOnGPU, m.metrics.TotalLayers)
			if m.metrics.LayersOnGPU < m.metrics.TotalLayers {
				offLabel += fmt.Sprintf(" (CPU %dL)", m.metrics.TotalLayers-m.metrics.LayersOnGPU)
			}
			bits = append(bits, m.sty.muted().Render("GPU"), m.sty.text().Render(offLabel))
		}
	}

	// Show live process RAM usage alongside llama-server host buffer allocations.
	if ramStr := m.runtimeRAMLabel(); ramStr != "" {
		bits = append(bits, m.sty.muted().Render("RAM"), m.sty.text().Render(ramStr))
	}

	if m.proc != nil && !m.proc.Exited() {
		bits = append(bits, m.sty.muted().Render("up"), m.sty.text().Render(fmtDur(m.proc.Uptime())))
	}

	line := strings.Join(bits, "  ")
	return truncateANSI(line, w)
}

func (m *Model) runtimeRAMLabel() string {
	hostMiB := m.metrics.CPUModelMiB + m.metrics.CPUKVMiB + m.metrics.CPUComputeMiB
	switch {
	case m.procRAMBytes > 0:
		base := "srv " + fmtBytesShort(m.procRAMBytes)
		if hostMiB > 0 {
			base = fmt.Sprintf("buf %.0fM · %s", hostMiB, base)
		}
		if m.sysRAMTotal > 0 && m.sysRAMAvail > 0 {
			usedSys := m.sysRAMTotal - m.sysRAMAvail
			return fmt.Sprintf("%s · sys %s/%s GB", base, fmtGB(usedSys), fmtGB(m.sysRAMTotal))
		}
		return base
	case hostMiB > 0:
		return fmt.Sprintf("buf %.0f MiB", hostMiB)
	case m.meta != nil && m.est.RAMUsed() > 0:
		return "est " + fmtGB(m.est.RAMUsed()) + " GB"
	}
	return ""
}

func fmtBytesShort(b uint64) string {
	if b >= 1<<30 {
		return fmt.Sprintf("%.1f GB", float64(b)/(1<<30))
	}
	return fmt.Sprintf("%.0f MB", float64(b)/(1<<20))
}

func fmtDur(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm%02ds", int(d.Minutes()), int(d.Seconds())%60)
	}
	return fmt.Sprintf("%dh%02dm", int(d.Hours()), int(d.Minutes())%60)
}

func fmtGB(b uint64) string {
	return fmt.Sprintf("%.1f", float64(b)/(1<<30))
}

func (m *Model) viewTooSmall() string {
	msg := fmt.Sprintf("Terminal too small\n\nneed at least 80x24, have %dx%d", m.width, m.height)
	return lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Center,
		m.sty.warning().Render(msg))
}

func (m *Model) viewHelp() string {
	w, h := m.width, m.height
	if w <= 0 || h <= 0 {
		return "help"
	}
	verLine := "alpaca"
	if m.version != "" {
		verLine = fmt.Sprintf("alpaca %s (commit %s, llama.cpp %s)", m.version, m.commit, m.llamaTag)
	}
	lines := []string{
		m.sty.accent().Render(BrandBanner(m.sty.ascii)),
		m.sty.muted().Render("  " + verLine),
		"",
		m.sty.accent().Render("keys"),
		"",
		"  ↑↓ / kj        move selection or field",
		"  ←→ / hl        adjust slider / option (offload, KV, reasoning, etc.)",
		"  tab            switch Models / Settings (or Chat / Logs on run screen)",
		"  enter          launch model, or edit/toggle the focused setting",
		"  /              filter models",
		"  o              open a model file or change the models folder",
		"  p / s          presets list (r rename · d delete) / save preset",
		"  c / C          copy command line / copy model to local disk",
		"  t              cycle color theme",
		"  ? / q          close help / quit",
		"",
		m.sty.muted().Render("  chat: enter send · ctrl+e effort · ctrl+t thinking · ctrl+l clear · /help"),
		m.sty.muted().Render("  logs: x stop · r restart · esc back"),
		"",
		m.sty.muted().Render("  press ? or esc to close"),
	}
	box := m.sty.sHelpBox.Render(strings.Join(lines, "\n"))

	out := lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, box)
	return frame(out, w, h)
}

func (m *Model) overlayInput(base string) string {
	w := m.width - 8
	if w > 64 {
		w = 64
	}
	if w < 30 {
		w = 30
	}
	title := m.editingTitle
	if title == "" {
		title = "Edit value"
	}
	m.input.Width = w - 6
	content := m.sty.accent().Render(title) + "\n\n" +
		m.input.View() + "\n\n" +
		m.sty.muted().Render("enter apply · esc cancel")
	box := m.sty.sInputBox.Width(w).Render(content)
	return overlay(base, box)
}

func (m *Model) overlayPresets(base string) string {
	w := m.width
	if w > 70 {
		w = 70
	}
	if w < 30 {
		w = 30
	}
	var lines []string
	if len(m.presets) == 0 {
		lines = append(lines, m.sty.muted().Render("  no presets yet — press s to save one"))
	}
	for i, p := range m.presets {
		row := "  " + pad(p.String(), w-8)
		if i == m.presetIdx {
			lines = append(lines, m.sty.accent().Render(truncate("▸ "+p.String(), w-4)))
		} else {
			lines = append(lines, m.sty.text().Render(truncate(row, w-4)))
		}
	}
	box := m.sty.sPresetBox.Width(w).Render(strings.Join(lines, "\n"))
	return overlay(base, box)
}

func (m *Model) overlayConfirm(base, title, hint string) string {
	box := m.sty.sConfirmBox.
		Render(m.sty.warning().Render(title) + "\n\n" + m.sty.muted().Render("  "+hint))
	return overlay(base, box)
}

func (m *Model) overlayToast(base string) string {
	box := m.sty.sToastBox.
		Render(m.sty.accent().Render(m.toast))
	return overlay(base, box)
}

func (m *Model) overlayCopyProgress(base string) string {
	w := m.width - 8
	if w > 64 {
		w = 64
	}
	if w < 34 {
		w = 34
	}
	barW := w - 12
	if barW < 10 {
		barW = 10
	}
	var ratio float64
	if m.copyProgress.totalBytes > 0 {
		ratio = float64(m.copyProgress.doneBytes) / float64(m.copyProgress.totalBytes)
	}
	progressStr := fmt.Sprintf("%s / %s GB (%.0f%%)",
		fmtGB(uint64(max64(0, m.copyProgress.doneBytes))),
		fmtGB(uint64(max64(0, m.copyProgress.totalBytes))),
		ratio*100)
	if m.copyProgress.speedMBps > 0 {
		progressStr += fmt.Sprintf(" · %.0f MB/s", m.copyProgress.speedMBps)
	}
	if m.copyProgress.eta > 0 {
		progressStr += " · " + fmtETA(m.copyProgress.eta) + " left"
	}
	content := m.sty.accent().Render("Copying to local disk") + "\n" +
		m.sty.muted().Render(truncate(m.copyTitle, w-6)) + "\n\n" +
		m.sty.gauge(barW, ratio) + fmt.Sprintf(" %3.0f%%\n", ratio*100) +
		m.sty.text().Render(truncate(progressStr, w-6)) + "\n\n" +
		m.sty.muted().Render("dest: "+truncate(m.cfg.LocalModelsDir(), w-12)) + "\n" +
		m.sty.muted().Render("esc cancel")
	box := m.sty.sInputBox.Width(w).Render(content)
	return overlay(base, box)
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// overlay centers box over base without changing the frame size.
func overlay(base, box string) string {
	baseLines := strings.Split(base, "\n")
	boxLines := strings.Split(box, "\n")
	boxH := len(boxLines)
	baseW := 0
	for _, l := range baseLines {
		if w := lipgloss.Width(l); w > baseW {
			baseW = w
		}
	}
	boxW := 0
	for _, l := range boxLines {
		if w := lipgloss.Width(l); w > boxW {
			boxW = w
		}
	}
	top := (len(baseLines) - boxH) / 2
	if top < 0 {
		top = 0
	}
	left := (baseW - boxW) / 2
	if left < 0 {
		left = 0
	}
	for i, bl := range boxLines {
		row := top + i
		if row < 0 || row >= len(baseLines) {
			continue
		}
		bw := lipgloss.Width(bl)
		prefix := ansiLeft(baseLines[row], left)
		suffix := ansiFrom(baseLines[row], left+bw)
		baseLines[row] = prefix + "\x1b[0m" + bl + "\x1b[0m" + suffix
	}
	return strings.Join(baseLines, "\n")
}

// ansiLeft returns the leading cells up to width without cutting ANSI sequences,
// padding with spaces if the line is shorter than width.
func ansiLeft(s string, width int) string {
	if width <= 0 {
		return ""
	}
	var b strings.Builder
	w := 0
	for i := 0; i < len(s); {
		if s[i] == 0x1b {
			j := i + 1
			for j < len(s) && s[j] != 'm' {
				j++
			}
			if j < len(s) {
				j++
			}
			b.WriteString(s[i:j])
			i = j
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		rw := runeWidth(r)
		if w+rw > width {
			break
		}
		b.WriteRune(r)
		w += rw
		i += size
	}
	if w < width {
		b.WriteString(strings.Repeat(" ", width-w))
	}
	return b.String()
}

// ansiFrom skips skip visible cells and returns the trailing slice of s,
// replaying the most recent SGR styling active at the cut point.
func ansiFrom(s string, skip int) string {
	if skip <= 0 {
		return s
	}
	var lastSGR strings.Builder
	var out strings.Builder
	w := 0
	copying := false
	for i := 0; i < len(s); {
		if s[i] == 0x1b {
			j := i + 1
			for j < len(s) && s[j] != 'm' {
				j++
			}
			if j < len(s) {
				j++
			}
			seq := s[i:j]
			if copying {
				out.WriteString(seq)
			} else {
				if seq == "\x1b[0m" || seq == "\x1b[m" {
					lastSGR.Reset()
				} else {
					lastSGR.WriteString(seq)
				}
			}
			i = j
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		rw := runeWidth(r)
		if !copying {
			if w+rw > skip {
				copying = true
				out.WriteString(lastSGR.String())
				if w < skip {
					out.WriteString(strings.Repeat(" ", w+rw-skip))
					w += rw
					i += size
					continue
				}
			} else {
				w += rw
				i += size
				continue
			}
		}
		out.WriteRune(r)
		i += size
	}
	return out.String()
}
