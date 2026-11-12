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
	procSetConsoleOutCP = kernel32DLL.NewProc("SetConsoleOutputCP")
	procSetConsoleCP    = kernel32DLL.NewProc("SetConsoleCP")
)

func initConsole() (asciiFallback bool, vtNotice string) {
	// Enable UTF-8 code page (65001) for Unicode box drawing, sliders, and gauge blocks.
	procSetConsoleOutCP.Call(65001)
	procSetConsoleCP.Call(65001)

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
	if os.Getenv("ALPACA_ASCII") == "1" {
		asciiFallback = true
	} else if cp != 0 && cp != 65001 {
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
	procSetConsoleOutCP.Call(65001)
	if os.Getenv("WT_SESSION") != "" {
		return false
	}
	cp, _, _ := procGetConsoleOutCP.Call()
	return cp != 65001
}
