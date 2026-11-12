package runner

import "strings"

// ThinkStreamParser incrementally splits inline <think>...</think> blocks from
// streamed content chunks so reasoning is separated cleanly even across chunk boundaries.
type ThinkStreamParser struct {
	inThink    bool
	sawClose   bool
	trimLeadNL bool
	buf        string
}

const (
	openThinkTag  = "<think>"
	closeThinkTag = "</think>"
)

// Feed processes an incoming content fragment and returns separated
// (reasoning, content) text ready to append to the active turn.
func (p *ThinkStreamParser) Feed(chunk string) (reasoning, content string) {
	if chunk == "" {
		return "", ""
	}
	p.buf += chunk

	var rOut, cOut strings.Builder
	for len(p.buf) > 0 {
		if p.inThink {
			idx := strings.Index(p.buf, closeThinkTag)
			if idx >= 0 {
				rOut.WriteString(p.buf[:idx])
				p.buf = p.buf[idx+len(closeThinkTag):]
				p.inThink = false
				p.sawClose = true
				p.trimLeadNL = true
				continue
			}
			// Keep up to len(closeThinkTag)-1 bytes in case </think> spans chunks.
			keep := partialSuffixPrefix(p.buf, closeThinkTag)
			emit := len(p.buf) - keep
			if emit > 0 {
				rOut.WriteString(p.buf[:emit])
				p.buf = p.buf[emit:]
			}
			break
		}

		if p.trimLeadNL {
			p.buf = strings.TrimLeft(p.buf, "\r\n")
			if p.buf == "" {
				break
			}
			p.trimLeadNL = false
		}

		idx := strings.Index(p.buf, openThinkTag)
		if idx >= 0 {
			cOut.WriteString(p.buf[:idx])
			p.buf = p.buf[idx+len(openThinkTag):]
			p.inThink = true
			continue
		}
		keep := partialSuffixPrefix(p.buf, openThinkTag)
		emit := len(p.buf) - keep
		if emit > 0 {
			cOut.WriteString(p.buf[:emit])
			p.buf = p.buf[emit:]
		}
		break
	}
	return rOut.String(), cOut.String()
}

// Flush emits any buffered bytes remaining when the stream ends.
func (p *ThinkStreamParser) Flush() (reasoning, content string) {
	if p.buf == "" {
		return "", ""
	}
	rem := p.buf
	p.buf = ""
	if p.inThink {
		return rem, ""
	}
	if p.trimLeadNL {
		rem = strings.TrimLeft(rem, "\r\n")
		p.trimLeadNL = false
	}
	return "", rem
}

// Closed reports whether an inline <think> block completed via </think>.
func (p *ThinkStreamParser) Closed() bool {
	return p.sawClose
}

// partialSuffixPrefix returns the longest suffix length of s (at most len(tag)-1)
// that matches a prefix of tag, so partial tags split across chunks are buffered.
func partialSuffixPrefix(s, tag string) int {
	maxCheck := len(tag) - 1
	if len(s) < maxCheck {
		maxCheck = len(s)
	}
	for n := maxCheck; n > 0; n-- {
		if strings.HasSuffix(s, tag[:n]) {
			return n
		}
	}
	return 0
}
