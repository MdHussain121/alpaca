package ui

import (
	"math"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/termenv"
)

// Theme is a named colour palette.
type Theme struct {
	Name    string
	Bg      lipgloss.Color
	Surface lipgloss.Color
	Text    lipgloss.Color
	Muted   lipgloss.Color
	Accent  lipgloss.Color
	Accent2 lipgloss.Color
	Success lipgloss.Color
	Warning lipgloss.Color
	Error   lipgloss.Color
}

// Mocha is the default Catppuccin Mocha palette from the spec.
var Mocha = Theme{
	Name:    "mocha",
	Bg:      "#1e1e2e",
	Surface: "#313244",
	Text:    "#cdd6f4",
	Muted:   "#a6adc8",
	Accent:  "#cba6f7",
	Accent2: "#89b4fa",
	Success: "#a6e3a1",
	Warning: "#f9e2af",
	Error:   "#f38ba8",
}

// TokyoNight is the alternate palette.
var TokyoNight = Theme{
	Name:    "tokyo-night",
	Bg:      "#1a1b26",
	Surface: "#24283b",
	Text:    "#c0caf5",
	Muted:   "#7f8aa8",
	Accent:  "#bb9af7",
	Accent2: "#7aa2f7",
	Success: "#9ece6a",
	Warning: "#e0af68",
	Error:   "#f7768e",
}

// Dracula is a vibrant purple/pink dark palette.
var Dracula = Theme{
	Name:    "dracula",
	Bg:      "#282a36",
	Surface: "#44475a",
	Text:    "#f8f8f2",
	Muted:   "#9aa2be",
	Accent:  "#bd93f9",
	Accent2: "#ff79c6",
	Success: "#50fa7b",
	Warning: "#f1fa8c",
	Error:   "#ff5555",
}

// Gruvbox is a warm retro amber/olive dark palette.
var Gruvbox = Theme{
	Name:    "gruvbox",
	Bg:      "#1d2021",
	Surface: "#3c3836",
	Text:    "#ebdbb2",
	Muted:   "#a89984",
	Accent:  "#fe8019",
	Accent2: "#fabd2f",
	Success: "#b8bb26",
	Warning: "#d79921",
	Error:   "#fb4934",
}

// Nord is an arctic slate/frost blue palette.
var Nord = Theme{
	Name:    "nord",
	Bg:      "#2e3440",
	Surface: "#3b4252",
	Text:    "#eceff4",
	Muted:   "#93a1b8",
	Accent:  "#88c0d0",
	Accent2: "#81a1c1",
	Success: "#a3be8c",
	Warning: "#ebcb8b",
	Error:   "#bf616a",
}

// RosePine is a muted plum and rose gold palette.
var RosePine = Theme{
	Name:    "rose-pine",
	Bg:      "#191724",
	Surface: "#26233a",
	Text:    "#e0def4",
	Muted:   "#908caa",
	Accent:  "#ebbcba",
	Accent2: "#9ccfd8",
	Success: "#9ccfd8",
	Warning: "#f6c177",
	Error:   "#eb6f92",
}

// Kanagawa is a deep ink-blue and warm sand palette.
var Kanagawa = Theme{
	Name:    "kanagawa",
	Bg:      "#1f1f28",
	Surface: "#2a2a37",
	Text:    "#dcd7ba",
	Muted:   "#9ca69a",
	Accent:  "#957fb8",
	Accent2: "#7e9cd8",
	Success: "#98bb6c",
	Warning: "#e6c384",
	Error:   "#e46876",
}

// Monokai is a classic high-contrast olive/violet/cyan palette.
var Monokai = Theme{
	Name:    "monokai",
	Bg:      "#1e1f1c",
	Surface: "#2e2f2a",
	Text:    "#f8f8f2",
	Muted:   "#a59f85",
	Accent:  "#ae81ff",
	Accent2: "#66d9ef",
	Success: "#a6e22e",
	Warning: "#e6db74",
	Error:   "#f92672",
}

// SolarizedDark is Ethan Schoonover's dark teal/blue palette.
var SolarizedDark = Theme{
	Name:    "solarized-dark",
	Bg:      "#002b36",
	Surface: "#073642",
	Text:    "#eee8d5",
	Muted:   "#93a1a1",
	Accent:  "#2aa198",
	Accent2: "#268bd2",
	Success: "#859900",
	Warning: "#b58900",
	Error:   "#dc322f",
}

// Cyberpunk is a high-contrast neon teal/cyan dark palette.
var Cyberpunk = Theme{
	Name:    "cyberpunk",
	Bg:      "#0d1117",
	Surface: "#161b22",
	Text:    "#e6edf3",
	Muted:   "#8b949e",
	Accent:  "#00f5d4",
	Accent2: "#00bbf9",
	Success: "#39d353",
	Warning: "#fee440",
	Error:   "#ff4d6d",
}

// AllThemes returns every built-in color theme in cycle order.
func AllThemes() []Theme {
	return []Theme{
		Mocha, TokyoNight, Dracula, Gruvbox, Nord,
		RosePine, Kanagawa, Monokai, SolarizedDark, Cyberpunk,
	}
}

// ThemeNames returns the built-in theme names in cycle order.
func ThemeNames() []string {
	all := AllThemes()
	out := make([]string, len(all))
	for i, t := range all {
		out[i] = t.Name
	}
	return out
}

// ThemeByName resolves a config theme name.
func ThemeByName(name string) Theme {
	n := strings.ToLower(strings.TrimSpace(name))
	if n == "tokyo" {
		return TokyoNight
	}
	if n == "solarized" {
		return SolarizedDark
	}
	for _, t := range AllThemes() {
		if t.Name == n {
			return t
		}
	}
	return Mocha
}

// NextTheme returns the next theme name after cur in cycle order.
func NextTheme(cur string, dir int) string {
	themes := AllThemes()
	resolved := ThemeByName(cur).Name
	idx := 0
	for i, t := range themes {
		if t.Name == resolved {
			idx = i
			break
		}
	}
	if dir == 0 {
		dir = 1
	}
	idx = (idx + dir) % len(themes)
	if idx < 0 {
		idx += len(themes)
	}
	return themes[idx].Name
}

type styleSet struct {
	t     Theme
	ascii bool
	true  bool

	sText          lipgloss.Style
	sMuted         lipgloss.Style
	sAccent        lipgloss.Style
	sAccentBold    lipgloss.Style
	sAccent2Bold   lipgloss.Style
	sSuccess       lipgloss.Style
	sWarning       lipgloss.Style
	sDanger        lipgloss.Style
	sPanelNormal   lipgloss.Style
	sPanelFocused  lipgloss.Style
	sGaugeSuccess  lipgloss.Style
	sGaugeWarn     lipgloss.Style
	sGaugeErr      lipgloss.Style
	sTitleFallback lipgloss.Style

	sHelpBox    lipgloss.Style
	sInputBox   lipgloss.Style
	sPresetBox  lipgloss.Style
	sConfirmBox lipgloss.Style
	sToastBox   lipgloss.Style

	accentRGB  [3]int
	accent2RGB [3]int
	bgRGB      [3]int
}

func newStyles(t Theme, ascii bool) styleSet {
	border := lipgloss.RoundedBorder()
	if ascii {
		border = lipgloss.NormalBorder()
	}
	return styleSet{
		t:              t,
		ascii:          ascii,
		true:           termenv.ColorProfile() == termenv.TrueColor,
		sText:          lipgloss.NewStyle().Foreground(t.Text),
		sMuted:         lipgloss.NewStyle().Foreground(t.Muted),
		sAccent:        lipgloss.NewStyle().Foreground(t.Accent),
		sAccentBold:    lipgloss.NewStyle().Foreground(t.Accent).Bold(true),
		sAccent2Bold:   lipgloss.NewStyle().Foreground(t.Accent2).Bold(true),
		sSuccess:       lipgloss.NewStyle().Foreground(t.Success),
		sWarning:       lipgloss.NewStyle().Foreground(t.Warning),
		sDanger:        lipgloss.NewStyle().Foreground(t.Error),
		sPanelNormal:   lipgloss.NewStyle().Border(border).BorderForeground(t.Muted).Padding(0, 1),
		sPanelFocused:  lipgloss.NewStyle().Border(border).BorderForeground(t.Accent).Padding(0, 1),
		sGaugeSuccess:  lipgloss.NewStyle().Foreground(t.Success),
		sGaugeWarn:     lipgloss.NewStyle().Foreground(t.Warning),
		sGaugeErr:      lipgloss.NewStyle().Foreground(t.Error),
		sTitleFallback: lipgloss.NewStyle().Background(t.Accent).Foreground(t.Bg).Bold(true),
		sHelpBox:       lipgloss.NewStyle().Border(border).BorderForeground(t.Accent).Padding(1, 3),
		sInputBox:      lipgloss.NewStyle().Border(border).BorderForeground(t.Accent).Padding(1, 2),
		sPresetBox:     lipgloss.NewStyle().Border(border).BorderForeground(t.Accent2).Padding(1, 2),
		sConfirmBox:    lipgloss.NewStyle().Border(border).BorderForeground(t.Warning).Padding(1, 3),
		sToastBox:      lipgloss.NewStyle().Border(border).BorderForeground(t.Accent).Padding(0, 2),
		accentRGB:      hexRGB(string(t.Accent)),
		accent2RGB:     hexRGB(string(t.Accent2)),
		bgRGB:          hexRGB(string(t.Bg)),
	}
}

func (s styleSet) text() lipgloss.Style    { return s.sText }
func (s styleSet) muted() lipgloss.Style   { return s.sMuted }
func (s styleSet) accent() lipgloss.Style  { return s.sAccent }
func (s styleSet) accent2() lipgloss.Style { return s.sAccent2Bold.Bold(false) }

// stripANSI removes ANSI CSI escape sequences cleanly for plain-text diff & width checks.
func stripANSI(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		if s[i] == 0x1b && i+1 < len(s) && s[i+1] == '[' {
			i += 2
			for i < len(s) && (s[i] < 0x40 || s[i] > 0x7e) {
				i++
			}
			continue
		}
		b.WriteByte(s[i])
	}
	return b.String()
}
func (s styleSet) success() lipgloss.Style { return s.sSuccess }
func (s styleSet) warning() lipgloss.Style { return s.sWarning }
func (s styleSet) danger() lipgloss.Style  { return s.sDanger }

func (s styleSet) key(k, desc string) string {
	return s.accent().Render(k) + " " + s.muted().Render(desc)
}

// runeWidth is the number of terminal cells a rune occupies. Wide East Asian
// characters and emoji take two, which plain rune counts get wrong.
func runeWidth(r rune) int {
	switch {
	case r >= 0x1100 && r <= 0x115f,
		r == 0x2329 || r == 0x232a,
		r >= 0x2e80 && r <= 0xa4cf && r != 0x303f,
		r >= 0xac00 && r <= 0xd7a3,
		r >= 0xf900 && r <= 0xfaff,
		r >= 0xfe10 && r <= 0xfe19,
		r >= 0xfe30 && r <= 0xfe6f,
		r >= 0xff00 && r <= 0xff60,
		r >= 0xffe0 && r <= 0xffe6,
		r >= 0x1f300 && r <= 0x1f64f,
		r >= 0x1f900 && r <= 0x1f9ff,
		r >= 0x20000 && r <= 0x3fffd:
		return 2
	}
	return 1
}

func displayWidth(s string) int {
	n := 0
	for _, r := range s {
		n += runeWidth(r)
	}
	return n
}

// truncate shortens s to at most w cells, appending an ellipsis when cut.
func truncate(s string, w int) string {
	if w <= 0 {
		return ""
	}
	if displayWidth(s) <= w {
		return s
	}
	if w == 1 {
		for _, r := range s {
			return string(r)
		}
		return ""
	}
	var b strings.Builder
	used := 0
	for _, r := range s {
		rw := runeWidth(r)
		if used+rw > w-1 {
			break
		}
		b.WriteRune(r)
		used += rw
	}
	b.WriteString("…")
	return b.String()
}

// truncateANSI shortens a string to w cells without cutting escape sequences.
func truncateANSI(s string, w int) string {
	if w <= 0 {
		return ""
	}
	if lipgloss.Width(s) <= w {
		return s
	}
	var b strings.Builder
	width := 0
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
		if width+1 > w-1 {
			break
		}
		b.WriteRune(r)
		width++
		i += size
	}
	b.WriteString("…")
	b.WriteString("\x1b[0m")
	return b.String()
}

// Pad right-pads s to exactly w cells.
func pad(s string, w int) string {
	n := displayWidth(s)
	if n >= w {
		return truncate(s, w)
	}
	return s + strings.Repeat(" ", w-n)
}

// Panel draws a bordered card of the exact total size w x h.
func (s styleSet) panel(w, h int, title string, focused bool, body []string) string {
	if w < 6 || h < 3 {
		return ""
	}
	innerW := w - 4 // border 2 + padding 2
	innerH := h - 2 // top and bottom border

	lines := make([]string, 0, innerH)
	head := s.sMuted.Render(truncate(title, innerW))
	if focused {
		head = s.sAccent.Render(truncate(title, innerW))
	}
	lines = append(lines, head)
	for _, l := range body {
		if len(lines) >= innerH {
			break
		}
		lines = append(lines, l)
	}
	for len(lines) < innerH {
		lines = append(lines, "")
	}

	box := s.sPanelNormal
	if focused {
		box = s.sPanelFocused
	}
	return box.Width(w - 2).Height(innerH).Render(strings.Join(lines, "\n"))
}

// Badge renders a filled status chip.
func (s styleSet) badge(label string, colour lipgloss.Color) string {
	style := lipgloss.NewStyle().
		Background(colour).
		Foreground(s.t.Bg).
		Bold(true).
		Padding(0, 1)
	if !s.true {
		style = lipgloss.NewStyle().Foreground(colour).Bold(true).Padding(0, 1)
	}
	return style.Render(label)
}

var blocksFull = []rune("█▉▊▋▌▍▎▏")
var blocksEmpty = '░'

// gauge draws a horizontal meter of exactly width cells.
func (s styleSet) gauge(width int, ratio float64) string {
	if width <= 0 {
		return ""
	}
	if ratio < 0 {
		ratio = 0
	}
	if ratio > 1.5 {
		ratio = 1.5
	}
	filled := ratio * float64(width)
	full := int(math.Floor(filled))
	frac := filled - float64(full)
	if full > width {
		full = width
		frac = 0
	}

	gaugeStyle := s.sGaugeSuccess
	switch {
	case ratio > 1.0:
		gaugeStyle = s.sGaugeErr
	case ratio > 0.85:
		gaugeStyle = s.sGaugeWarn
	}

	if s.ascii {
		filledBar := strings.Repeat("#", full)
		emptyBar := strings.Repeat("-", width-full)
		return gaugeStyle.Render(filledBar) + s.muted().Render(emptyBar)
	}

	var bFilled strings.Builder
	bFilled.Grow(width * 3)
	for i := 0; i < full; i++ {
		bFilled.WriteRune(blocksFull[0])
	}
	if full < width && frac > 0 {
		idx := int(frac * float64(len(blocksFull)))
		if idx >= len(blocksFull) {
			idx = len(blocksFull) - 1
		}
		bFilled.WriteRune(blocksFull[idx])
		full++
	}

	var bEmpty strings.Builder
	bEmpty.Grow((width - full) * 3)
	for i := full; i < width; i++ {
		bEmpty.WriteRune(blocksEmpty)
	}
	return gaugeStyle.Render(bFilled.String()) + s.muted().Render(bEmpty.String())
}

// slider renders a selectable discrete position indicator of exactly width cells.
func (s styleSet) slider(width, pos, total int) string {
	if total <= 1 {
		return s.sliderRatio(width, 0)
	}
	return s.sliderRatio(width, float64(pos)/float64(total-1))
}

// sliderRatio renders a continuous slider bar of exact width cells for ratio in [0..1].
func (s styleSet) sliderRatio(width int, ratio float64) string {
	if width <= 0 {
		return ""
	}
	if ratio < 0 {
		ratio = 0
	}
	if ratio > 1 {
		ratio = 1
	}
	idx := int(math.Round(ratio * float64(width-1)))
	if idx < 0 {
		idx = 0
	}
	if idx >= width {
		idx = width - 1
	}
	if s.ascii {
		left := strings.Repeat("=", idx) + "O"
		right := strings.Repeat("-", width-1-idx)
		return s.accent().Render(left) + s.muted().Render(right)
	}
	left := strings.Repeat("━", idx) + "●"
	right := strings.Repeat("─", width-1-idx)
	return s.accent().Render(left) + s.muted().Render(right)
}

// titleBar renders a full-width gradient header with a right-aligned badge.
func (s styleSet) titleBar(width int, left, right string) string {
	if width <= 0 {
		return ""
	}
	l := truncate(left, width)
	r := truncate(right, width)
	if displayWidth(l)+displayWidth(r)+2 > width {
		r = ""
	}

	cells := make([]rune, width)
	for i := range cells {
		cells[i] = ' '
	}
	col := 0
	for _, ch := range l {
		cw := runeWidth(ch)
		if cw <= 0 {
			continue
		}
		if col+cw > width {
			break
		}
		cells[col] = ch
		if cw == 2 && col+1 < width {
			cells[col+1] = 0
		}
		col += cw
	}
	if r != "" {
		rw := displayWidth(r)
		start := width - rw
		if start < 0 {
			start = 0
		}
		c := start
		for _, ch := range r {
			cw := runeWidth(ch)
			if cw <= 0 {
				continue
			}
			if c+cw > width {
				break
			}
			cells[c] = ch
			if cw == 2 && c+1 < width {
				cells[c+1] = 0
			}
			c += cw
		}
	}

	if !s.true {
		var pb strings.Builder
		for _, ch := range cells {
			if ch != 0 {
				pb.WriteRune(ch)
			}
		}
		return s.sTitleFallback.Render(pb.String())
	}

	from := s.accentRGB
	to := s.accent2RGB
	fg := s.bgRGB
	var buf []byte
	buf = make([]byte, 0, width*36)
	pos := 0
	for _, ch := range cells {
		if ch == 0 {
			continue
		}
		t := 0.0
		if width > 1 {
			t = float64(pos) / float64(width-1)
		}
		c := lerp(from, to, t)
		buf = append(buf, "\x1b[48;2;"...)
		buf = strconv.AppendInt(buf, int64(c[0]), 10)
		buf = append(buf, ';')
		buf = strconv.AppendInt(buf, int64(c[1]), 10)
		buf = append(buf, ';')
		buf = strconv.AppendInt(buf, int64(c[2]), 10)
		buf = append(buf, "m\x1b[38;2;"...)
		buf = strconv.AppendInt(buf, int64(fg[0]), 10)
		buf = append(buf, ';')
		buf = strconv.AppendInt(buf, int64(fg[1]), 10)
		buf = append(buf, ';')
		buf = strconv.AppendInt(buf, int64(fg[2]), 10)
		buf = append(buf, 'm')
		buf = utf8.AppendRune(buf, ch)
		pos += runeWidth(ch)
	}
	buf = append(buf, "\x1b[0m"...)
	return string(buf)
}

func hexRGB(h string) [3]int {
	h = strings.TrimPrefix(h, "#")
	if len(h) != 6 {
		return [3]int{255, 255, 255}
	}
	v, err := strconv.ParseUint(h, 16, 32)
	if err != nil {
		return [3]int{255, 255, 255}
	}
	return [3]int{int(v >> 16 & 0xff), int(v >> 8 & 0xff), int(v & 0xff)}
}

func lerp(a, b [3]int, t float64) [3]int {
	return [3]int{
		int(float64(a[0]) + (float64(b[0])-float64(a[0]))*t),
		int(float64(a[1]) + (float64(b[1])-float64(a[1]))*t),
		int(float64(a[2]) + (float64(b[2])-float64(a[2]))*t),
	}
}

// helpBar renders key hints, wrapping to the available width.
func (s styleSet) helpBar(width int, hints [][2]string) string {
	var parts []string
	for _, h := range hints {
		parts = append(parts, s.key(h[0], h[1]))
	}
	line := strings.Join(parts, s.muted().Render(" · "))
	if lipgloss.Width(line) <= width {
		return line
	}
	// Drop hints from the end until it fits.
	for len(parts) > 1 {
		parts = parts[:len(parts)-1]
		line = strings.Join(parts, s.muted().Render(" · "))
		if lipgloss.Width(line) <= width {
			return line
		}
	}
	return truncate(hints[0][0]+" "+hints[0][1], width)
}
