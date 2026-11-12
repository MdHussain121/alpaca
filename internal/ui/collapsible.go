package ui

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// collapsibleBlock describes a fold-out UI section shared across thinking traces,
// tool-call cards, and skill panels.
type collapsibleBlock struct {
	Summary      string
	Body         string
	RightHint    string
	Expanded     bool
	Streaming    bool
	PreviewLines int
	AccentHeader bool
}

// renderCollapsible formats b to fit within width terminal columns using sty.
func renderCollapsible(sty styleSet, width int, spin string, b collapsibleBlock) []string {
	if width < 12 {
		width = 12
	}
	arrow := "▸"
	rail := "│ "
	if sty.ascii {
		arrow = ">"
		rail = "| "
	}

	leftIcon := arrow + " "
	if b.Expanded {
		if sty.ascii {
			arrow = "v"
		} else {
			arrow = "▾"
		}
		leftIcon = arrow + " "
	}
	if b.Streaming && spin != "" {
		leftIcon += spin + " "
	}

	hintPlain := ""
	if b.RightHint != "" {
		hintPlain = "  " + b.RightHint
	}
	maxHdr := width - 2
	hdrText := b.Summary
	avail := maxHdr - displayWidth(leftIcon)
	if hintPlain != "" && avail > displayWidth(hintPlain)+8 {
		hdrLeft := truncate(hdrText, avail-displayWidth(hintPlain))
		gap := avail - displayWidth(hdrLeft) - displayWidth(hintPlain)
		if gap < 1 {
			gap = 1
		}
		var rendered string
		if b.AccentHeader {
			rendered = "  " + sty.sAccent.Render(leftIcon+hdrLeft) + strings.Repeat(" ", gap) + sty.sMuted.Render(b.RightHint)
		} else {
			rendered = "  " + sty.sMuted.Render(leftIcon+hdrLeft) + strings.Repeat(" ", gap) + sty.sMuted.Render(b.RightHint)
		}
		return finishCollapsibleBody(sty, width, rail, rendered, b)
	}

	renderedHdr := "  " + sty.sMuted.Render(leftIcon+truncate(hdrText, max(4, avail)))
	if b.AccentHeader {
		renderedHdr = "  " + sty.sAccent.Render(leftIcon+truncate(hdrText, max(4, avail)))
	}
	return finishCollapsibleBody(sty, width, rail, renderedHdr, b)
}

func finishCollapsibleBody(sty styleSet, width int, rail, headerLine string, b collapsibleBlock) []string {
	out := []string{truncateANSI(headerLine, width)}
	if b.Body == "" {
		return out
	}

	bodyW := width - 4 - displayWidth(rail)
	if bodyW < 8 {
		bodyW = 8
	}

	switch {
	case b.Expanded:
		for _, l := range wrapText(b.Body, bodyW) {
			out = append(out, "    "+sty.sMuted.Render(rail+l))
		}
	case b.Streaming && b.PreviewLines > 0:
		wrapped := wrapText(b.Body, bodyW)
		if len(wrapped) > b.PreviewLines {
			wrapped = wrapped[len(wrapped)-b.PreviewLines:]
		}
		for _, l := range wrapped {
			out = append(out, "    "+sty.sMuted.Render(rail+l))
		}
	}
	return out
}

// widthClampANSI pads or truncates an ANSI string to an exact target cell width.
func widthClampANSI(s string, w int) string {
	cur := lipgloss.Width(s)
	if cur > w {
		return truncateANSI(s, w)
	}
	if cur < w {
		return s + strings.Repeat(" ", w-cur)
	}
	return s
}
