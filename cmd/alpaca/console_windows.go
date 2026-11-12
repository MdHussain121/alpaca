//go:build windows

package main

import (
	"os"
	"strings"

	"golang.org/x/sys/windows"
)

var (
	kernel32DLL         = windows.NewLazySystemDLL("kernel32.dll")
	procGetConsoleOutCP = kernel32DLL.NewProc("GetConsoleOutputCP")
)

func initConsole() (asciiFallback bool, vtNotice string) {
	hOut := windows.Stdout
	var mode uint32
	if err := windows.GetConsoleMode(hOut, &mode); err == nil {
		// Enable Virtual Terminal Processing on Windows 10 1809+ legacy consoles.
		if mode&windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING == 0 {
			if err := windows.SetConsoleMode(hOut, mode|windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING); err != nil {
				vtNotice = "legacy console without VT mode; running plain output"
				asciiFallback = true
			}
		}
	}
	cp, _, _ := procGetConsoleOutCP.Call()
	wt := os.Getenv("WT_SESSION") != ""
	if os.Getenv("ALPACA_ASCII") == "1" {
		asciiFallback = true
	} else if !wt && cp != 0 && cp != 65001 {
		asciiFallback = true
	}
	return asciiFallback, vtNotice
}

func prefersASCIIBanner() bool {
	if os.Getenv("ALPACA_ASCII") == "1" {
		return true
	}
	if strings.EqualFold(strings.TrimSpace(os.Getenv("LANG")), "C") {
		return true
	}
	if os.Getenv("WT_SESSION") != "" {
		return false
	}
	cp, _, _ := procGetConsoleOutCP.Call()
	return cp != 65001
}
