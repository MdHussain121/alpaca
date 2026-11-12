package ui

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/alecthomas/chroma/v2"
	"github.com/alecthomas/chroma/v2/formatters"
	"github.com/charmbracelet/lipgloss"
)

// codeBlock is a raw fenced code snippet extracted from markdown.
type codeBlock struct {
	Lang   string
	Code   string
	Closed bool
}

type mdSegment struct {
	isCode bool
	text   string
	block  codeBlock
}

type cachedCodeLines struct {
	lang         string
	themeName    string
	ascii        bool
	closed       bool
	rawLines     []string
	styledLines  []string // per-raw-line highlighted ANSI string (no trailing \n)
	highlightOps int      // incremental highlight invocations (benchmark/test visibility)
}

// markdownRenderCache keeps per-block Chroma highlighting incremental during streaming
// so completed lines above the cursor line are never re-highlighted per delta.
type markdownRenderCache struct {
	blocks []*cachedCodeLines
	ops    int
}

func (c *markdownRenderCache) reset() {
	c.blocks = nil
	c.ops = 0
}

// extractCodeBlocks returns all fenced code blocks in markdown content.
func extractCodeBlocks(src string) []codeBlock {
	segs := parseMarkdownSegments(src)
	var out []codeBlock
	for _, s := range segs {
		if s.isCode {
			out = append(out, s.block)
		}
	}
	return out
}

func parseMarkdownSegments(src string) []mdSegment {
	lines := strings.Split(strings.ReplaceAll(src, "\r\n", "\n"), "\n")
	var segs []mdSegment
	var prose strings.Builder

	inFence := false
	fenceChar := byte(0)
	fenceLen := 0
	lang := ""
	var codeBuf []string

	flushProse := func() {
		if prose.Len() > 0 {
			segs = append(segs, mdSegment{text: strings.TrimSuffix(prose.String(), "\n")})
			prose.Reset()
		}
	}

	for _, line := range lines {
		trimmed := strings.TrimLeft(line, " \t")
		if !inFence {
			if fc, fl, info, ok := matchFenceOpen(trimmed); ok {
				flushProse()
				inFence = true
				fenceChar = fc
				fenceLen = fl
				lang = info
				codeBuf = codeBuf[:0]
				continue
			}
			prose.WriteString(line)
			prose.WriteByte('\n')
			continue
		}

		if matchFenceClose(trimmed, fenceChar, fenceLen) {
			segs = append(segs, mdSegment{
				isCode: true,
				block: codeBlock{
					Lang:   lang,
					Code:   strings.Join(codeBuf, "\n"),
					Closed: true,
				},
			})
			inFence = false
			lang = ""
			codeBuf = nil
			continue
		}
		codeBuf = append(codeBuf, line)
	}

	if inFence {
		segs = append(segs, mdSegment{
			isCode: true,
			block: codeBlock{
				Lang:   lang,
				Code:   strings.Join(codeBuf, "\n"),
				Closed: false,
			},
		})
	} else {
		flushProse()
	}
	return segs
}

func matchFenceOpen(trimmed string) (char byte, count int, lang string, ok bool) {
	if len(trimmed) < 3 {
		return 0, 0, "", false
	}
	fc := trimmed[0]
	if fc != '`' && fc != '~' {
		return 0, 0, "", false
	}
	i := 0
	for i < len(trimmed) && trimmed[i] == fc {
		i++
	}
	if i < 3 {
		return 0, 0, "", false
	}
	rest := strings.TrimSpace(trimmed[i:])
	// Backtick info strings cannot contain backticks in CommonMark.
	if fc == '`' && strings.ContainsRune(rest, '`') {
		return 0, 0, "", false
	}
	if fields := strings.Fields(rest); len(fields) > 0 {
		lang = fields[0]
	}
	return fc, i, lang, true
}

func matchFenceClose(trimmed string, fc byte, openLen int) bool {
	if len(trimmed) < openLen || trimmed[0] != fc {
		return false
	}
	i := 0
	for i < len(trimmed) && trimmed[i] == fc {
		i++
	}
	if i < openLen {
		return false
	}
	return strings.TrimSpace(trimmed[i:]) == ""
}

// renderMarkdownLines renders prose and fenced code blocks within innerW columns.
func renderMarkdownLines(sty styleSet, innerW int, src string, showLineNums bool, streamingCursor bool, cache *markdownRenderCache) []string {
	if innerW < 16 {
		innerW = 16
	}
	if cache == nil {
		cache = &markdownRenderCache{}
	}
	segs := parseMarkdownSegments(src)
	if len(segs) == 0 && streamingCursor {
		return []string{"  " + sty.sText.Render("▌")}
	}

	var out []string
	codeIdx := 0
	for si, seg := range segs {
		isLast := si == len(segs)-1
		if !seg.isCode {
			text := seg.text
			if isLast && streamingCursor {
				text += "▌"
			}
			out = append(out, renderProseLines(sty, innerW-2, text)...)
			continue
		}

		lines := renderCodeBlock(sty, innerW-2, seg.block, codeIdx, showLineNums, isLast && streamingCursor, cache)
		out = append(out, lines...)
		codeIdx++
	}
	if streamingCursor && len(segs) > 0 && segs[len(segs)-1].isCode && segs[len(segs)-1].block.Closed {
		out = append(out, "  "+sty.sText.Render("▌"))
	}
	return out
}

func renderProseLines(sty styleSet, width int, text string) []string {
	if width < 12 {
		width = 12
	}
	rawLines := strings.Split(text, "\n")
	var out []string
	for _, line := range rawLines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			out = append(out, "")
			continue
		}
		if strings.HasPrefix(trimmed, "### ") || strings.HasPrefix(trimmed, "## ") || strings.HasPrefix(trimmed, "# ") {
			h := strings.TrimLeft(trimmed, "# ")
			for _, wl := range wrapText(h, width) {
				out = append(out, "  "+sty.sAccentBold.Render(wl))
			}
			continue
		}
		if strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* ") {
			item := strings.TrimSpace(trimmed[2:])
			bullet := "• "
			if sty.ascii {
				bullet = "- "
			}
			wrapped := wrapText(item, max(8, width-2))
			for i, wl := range wrapped {
				prefix := "  "
				if i == 0 {
					prefix += sty.sAccent.Render(bullet)
				} else {
					prefix += "  "
				}
				out = append(out, prefix+renderInlineMarkdown(sty, wl))
			}
			continue
		}
		for _, wl := range wrapText(line, width) {
			out = append(out, "  "+renderInlineMarkdown(sty, wl))
		}
	}
	return out
}

// renderInlineMarkdown highlights `inline code` and **bold** spans cleanly.
func renderInlineMarkdown(sty styleSet, line string) string {
	var b strings.Builder
	for i := 0; i < len(line); {
		if line[i] == '`' {
			end := strings.IndexByte(line[i+1:], '`')
			if end > 0 {
				code := line[i+1 : i+1+end]
				b.WriteString(sty.sAccent.Render(code))
				i += end + 2
				continue
			}
		}
		if strings.HasPrefix(line[i:], "**") {
			end := strings.Index(line[i+2:], "**")
			if end > 0 {
				bold := line[i+2 : i+2+end]
				b.WriteString(sty.sText.Bold(true).Render(bold))
				i += end + 4
				continue
			}
		}
		nextTick := strings.IndexByte(line[i:], '`')
		nextBold := strings.Index(line[i:], "**")
		cut := len(line) - i
		if nextTick >= 0 && nextTick < cut {
			cut = nextTick
		}
		if nextBold >= 0 && nextBold < cut {
			cut = nextBold
		}
		if cut == 0 {
			b.WriteByte(line[i])
			i++
			continue
		}
		b.WriteString(sty.sText.Render(line[i : i+cut]))
		i += cut
	}
	return b.String()
}

func renderCodeBlock(sty styleSet, boxW int, blk codeBlock, idx int, showLineNums bool, cursorAtEnd bool, cache *markdownRenderCache) []string {
	if boxW < 18 {
		boxW = 18
	}
	tl, tr, bl, br, hl, vl := "╭", "╮", "╰", "╯", "─", "│"
	if sty.ascii {
		tl, tr, bl, br, hl, vl = "+", "+", "+", "+", "-", "|"
	}

	langLabel := blk.Lang
	if langLabel == "" {
		langLabel = "code"
	}
	copyHint := "[c] copy"
	innerBoxW := boxW - 2 // inside left and right border

	leftHdr := hl + " " + langLabel + " "
	rightHdr := " " + copyHint + " " + hl
	fill := innerBoxW - displayWidth(leftHdr) - displayWidth(rightHdr)
	var topLine string
	if fill >= 1 {
		topLine = "  " + sty.sMuted.Render(tl+leftHdr) + sty.sMuted.Render(strings.Repeat(hl, fill)) + sty.sMuted.Render(" ") + sty.sAccent.Render(copyHint) + sty.sMuted.Render(" "+hl+tr)
	} else {
		topLine = "  " + sty.sMuted.Render(tl+truncate(leftHdr, max(1, innerBoxW))+tr)
	}

	rawLines := getHighlightedCodeLines(sty, blk, idx, cache)
	if len(rawLines) == 0 {
		rawLines = []string{""}
	}

	numW := len(strconv.Itoa(len(rawLines)))
	gutter := ""
	blankGutter := ""
	gutterW := 0
	if showLineNums {
		gutterW = numW + 3 // "<num> │ "
	}
	contentW := innerBoxW - 2 - gutterW // 1 space pad left & right inside box
	if contentW < 8 {
		contentW = 8
	}

	out := make([]string, 0, len(rawLines)+2)
	out = append(out, topLine)

	leftRail := sty.sMuted.Render(vl + " ")
	rightRail := sty.sMuted.Render(" " + vl)

	for li, ansiLine := range rawLines {
		if cursorAtEnd && !blk.Closed && li == len(rawLines)-1 {
			ansiLine += sty.sText.Render("▌")
		}
		wrapped := wrapANSI(ansiLine, contentW)
		if len(wrapped) == 0 {
			wrapped = []string{""}
		}
		for wi, wl := range wrapped {
			if showLineNums {
				if wi == 0 {
					gutter = sty.sMuted.Render(fmt.Sprintf("%*d %s ", numW, li+1, vl))
				} else {
					blankGutter = sty.sMuted.Render(strings.Repeat(" ", numW) + " " + vl + " ")
					gutter = blankGutter
				}
			} else {
				gutter = ""
			}
			padded := widthClampANSI(wl, contentW)
			out = append(out, "  "+leftRail+gutter+padded+rightRail)
		}
	}

	bottom := "  " + sty.sMuted.Render(bl+strings.Repeat(hl, innerBoxW)+br)
	out = append(out, bottom)
	return out
}

func getHighlightedCodeLines(sty styleSet, blk codeBlock, idx int, cache *markdownRenderCache) []string {
	rawLines := strings.Split(blk.Code, "\n")
	for i := range rawLines {
		rawLines[i] = expandTabs(rawLines[i], 4)
	}

	for len(cache.blocks) <= idx {
		cache.blocks = append(cache.blocks, &cachedCodeLines{})
	}
	bc := cache.blocks[idx]

	themeOrASCII := sty.t.Name
	if sty.ascii {
		themeOrASCII = "ascii"
	}

	sameHeader := bc.lang == blk.Lang && bc.themeName == themeOrASCII && bc.ascii == sty.ascii
	if sameHeader && bc.closed && blk.Closed && len(bc.rawLines) == len(rawLines) && stringSlicesEqual(bc.rawLines, rawLines) {
		return bc.styledLines
	}

	// Incremental update: preserve highlighted ANSI strings for untouched completed lines
	// (every line before len(rawLines)-1 when still streaming).
	preserved := 0
	if sameHeader && !sty.ascii {
		limit := len(bc.rawLines)
		if !blk.Closed && limit > 0 {
			// The last line from the previous frame may have grown, so only reuse lines strictly before it.
			limit--
		}
		if limit > len(rawLines) {
			limit = len(rawLines)
		}
		for preserved < limit && bc.rawLines[preserved] == rawLines[preserved] {
			preserved++
		}
	}

	if !sameHeader {
		preserved = 0
	}

	// If closed, re-highlight cleanly once if any earlier lines existed while open, OR keep
	// incremental path during streaming so completed lines aren't re-highlighted.
	styled := make([]string, len(rawLines))
	if preserved > 0 {
		copy(styled[:preserved], bc.styledLines[:preserved])
	}

	if sty.ascii {
		for i := preserved; i < len(rawLines); i++ {
			styled[i] = rawLines[i]
		}
	} else {
		lex := chromaLexerFor(blk.Lang)
		cStyle := chromaStyleForTheme(sty.t)
		fmtName := "terminal256"
		if sty.true {
			fmtName = "terminal16m"
		}
		formatter := formatters.Get(fmtName)
		if formatter == nil {
			formatter = formatters.Fallback
		}
		for i := preserved; i < len(rawLines); i++ {
			styled[i] = highlightSingleLine(lex, formatter, cStyle, rawLines[i])
			bc.highlightOps++
			cache.ops++
		}
	}

	bc.lang = blk.Lang
	bc.themeName = themeOrASCII
	bc.ascii = sty.ascii
	bc.closed = blk.Closed
	bc.rawLines = append(bc.rawLines[:0], rawLines...)
	bc.styledLines = styled
	return styled
}

func highlightSingleLine(lex chroma.Lexer, formatter chroma.Formatter, style *chroma.Style, line string) string {
	if line == "" {
		return ""
	}
	it, err := lex.Tokenise(nil, line)
	if err != nil {
		return line
	}
	var buf bytes.Buffer
	if err := formatter.Format(&buf, style, it); err != nil {
		return line
	}
	return strings.TrimRight(buf.String(), "\r\n")
}

func expandTabs(s string, tabWidth int) string {
	if !strings.ContainsRune(s, '\t') {
		return s
	}
	var b strings.Builder
	col := 0
	for _, r := range s {
		if r == '\t' {
			spaces := tabWidth - (col % tabWidth)
			b.WriteString(strings.Repeat(" ", spaces))
			col += spaces
			continue
		}
		b.WriteRune(r)
		col += runeWidth(r)
	}
	return b.String()
}

// wrapANSI wraps an ANSI-colored string to width display cells without breaking escape codes
// and replays active SGR sequences onto wrapped continuation lines.
func wrapANSI(s string, width int) []string {
	if width <= 1 {
		return []string{s}
	}
	if lipgloss.Width(s) <= width {
		return []string{s}
	}

	var lines []string
	var cur strings.Builder
	activeSGR := ""
	curW := 0

	flushLine := func() {
		line := cur.String()
		if activeSGR != "" {
			line += "\x1b[0m"
		}
		lines = append(lines, line)
		cur.Reset()
		if activeSGR != "" {
			cur.WriteString(activeSGR)
		}
		curW = 0
	}

	for i := 0; i < len(s); {
		if s[i] == 0x1b && i+1 < len(s) && s[i+1] == '[' {
			j := i + 2
			for j < len(s) && s[j] != 'm' {
				j++
			}
			if j < len(s) {
				j++
			}
			seq := s[i:j]
			cur.WriteString(seq)
			if seq == "\x1b[0m" || seq == "\x1b[m" {
				activeSGR = ""
			} else {
				activeSGR += seq
			}
			i = j
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		rw := runeWidth(r)
		if curW+rw > width && curW > 0 {
			flushLine()
		}
		cur.WriteRune(r)
		curW += rw
		i += size
	}
	if cur.Len() > 0 {
		line := cur.String()
		if activeSGR != "" && !strings.HasSuffix(line, "\x1b[0m") {
			line += "\x1b[0m"
		}
		lines = append(lines, line)
	}
	if len(lines) == 0 {
		return []string{""}
	}
	return lines
}

func stringSlicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
