package agent

import (
	"fmt"
	"strings"
)

type diffOpKind int

const (
	diffEq diffOpKind = iota
	diffDel
	diffAdd
)

type diffOp struct {
	kind diffOpKind
	text string
}

// UnifiedDiff produces a compact unified diff between before and after for relPath,
// returning the formatted patch along with added and removed line counts.
func UnifiedDiff(relPath, before, after string) (patch string, added, removed int) {
	aLines := splitLinesPreserve(before)
	bLines := splitLinesPreserve(after)
	ops := computeLCSDiff(aLines, bLines)

	for _, op := range ops {
		switch op.kind {
		case diffAdd:
			added++
		case diffDel:
			removed++
		}
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("--- a/%s\n+++ b/%s\n", relPath, relPath))
	if before == "" {
		b.WriteString(fmt.Sprintf("@@ -0,0 +1,%d @@\n", len(bLines)))
		for _, ln := range bLines {
			b.WriteByte('+')
			b.WriteString(ln)
			b.WriteByte('\n')
		}
		return strings.TrimSuffix(b.String(), "\n"), len(bLines), 0
	}
	if added == 0 && removed == 0 {
		b.WriteString("(no changes)")
		return b.String(), 0, 0
	}

	const ctxLines = 3
	// Group changed regions into hunks with ctxLines surrounding lines.
	type hunkRange struct{ start, end int }
	var hunks []hunkRange
	for i, op := range ops {
		if op.kind == diffEq {
			continue
		}
		s := max(0, i-ctxLines)
		e := min(len(ops)-1, i+ctxLines)
		if len(hunks) > 0 && s <= hunks[len(hunks)-1].end+1 {
			hunks[len(hunks)-1].end = e
		} else {
			hunks = append(hunks, hunkRange{start: s, end: e})
		}
	}

	// Track line numbers up to each hunk start cleanly.
	oldPos, newPos := 1, 1
	opIdx := 0
	for _, h := range hunks {
		for opIdx < h.start {
			if ops[opIdx].kind != diffAdd {
				oldPos++
			}
			if ops[opIdx].kind != diffDel {
				newPos++
			}
			opIdx++
		}
		hOldStart, hNewStart := oldPos, newPos
		oldCnt, newCnt := 0, 0
		var body strings.Builder
		for opIdx <= h.end {
			op := ops[opIdx]
			switch op.kind {
			case diffEq:
				body.WriteByte(' ')
				body.WriteString(op.text)
				body.WriteByte('\n')
				oldCnt++
				newCnt++
				oldPos++
				newPos++
			case diffDel:
				body.WriteByte('-')
				body.WriteString(op.text)
				body.WriteByte('\n')
				oldCnt++
				oldPos++
			case diffAdd:
				body.WriteByte('+')
				body.WriteString(op.text)
				body.WriteByte('\n')
				newCnt++
				newPos++
			}
			opIdx++
		}
		b.WriteString(fmt.Sprintf("@@ -%d,%d +%d,%d @@\n", hOldStart, oldCnt, hNewStart, newCnt))
		b.WriteString(body.String())
	}

	return strings.TrimSuffix(b.String(), "\n"), added, removed
}

func splitLinesPreserve(s string) []string {
	if s == "" {
		return nil
	}
	norm := strings.ReplaceAll(s, "\r\n", "\n")
	norm = strings.TrimSuffix(norm, "\n")
	return strings.Split(norm, "\n")
}

func computeLCSDiff(a, b []string) []diffOp {
	m, n := len(a), len(b)
	// Fallback cleanly on very large files (>2000 lines) to avoid O(mn) memory spike.
	if m*n > 2_000_000 {
		out := make([]diffOp, 0, m+n)
		for _, ln := range a {
			out = append(out, diffOp{kind: diffDel, text: ln})
		}
		for _, ln := range b {
			out = append(out, diffOp{kind: diffAdd, text: ln})
		}
		return out
	}

	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := m - 1; i >= 0; i-- {
		for j := n - 1; j >= 0; j-- {
			if a[i] == b[j] {
				dp[i][j] = dp[i+1][j+1] + 1
			} else if dp[i+1][j] >= dp[i][j+1] {
				dp[i][j] = dp[i+1][j]
			} else {
				dp[i][j] = dp[i][j+1]
			}
		}
	}

	var ops []diffOp
	i, j := 0, 0
	for i < m && j < n {
		if a[i] == b[j] {
			ops = append(ops, diffOp{kind: diffEq, text: a[i]})
			i++
			j++
		} else if dp[i+1][j] >= dp[i][j+1] {
			ops = append(ops, diffOp{kind: diffDel, text: a[i]})
			i++
		} else {
			ops = append(ops, diffOp{kind: diffAdd, text: b[j]})
			j++
		}
	}
	for ; i < m; i++ {
		ops = append(ops, diffOp{kind: diffDel, text: a[i]})
	}
	for ; j < n; j++ {
		ops = append(ops, diffOp{kind: diffAdd, text: b[j]})
	}
	return ops
}
