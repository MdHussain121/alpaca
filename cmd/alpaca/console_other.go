//go:build !windows

package main

import (
	"os"
	"strings"
)

func initConsole() (bool, string) {
	return os.Getenv("ALPACA_ASCII") == "1", ""
}

func prefersASCIIBanner() bool {
	if os.Getenv("ALPACA_ASCII") == "1" {
		return true
	}
	return strings.EqualFold(strings.TrimSpace(os.Getenv("LANG")), "C")
}
