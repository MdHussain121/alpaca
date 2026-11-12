//go:build !windows

package ui

import (
	"encoding/base64"
	"fmt"
	"io"
	"os"
)

var (
	osc52Writer io.Writer = os.Stdout
	bellWriter  io.Writer = os.Stdout
)

func init() {
	clipboardWriter = copyClipboardOSC52
}

func copyClipboardOSC52(s string) error {
	return writeOSC52(osc52Writer, s)
}

func writeOSC52(w io.Writer, s string) error {
	if w == nil {
		return fmt.Errorf("no OSC 52 output writer")
	}
	enc := base64.StdEncoding.EncodeToString([]byte(s))
	_, err := fmt.Fprintf(w, "\x1b]52;c;%s\a", enc)
	return err
}
