//go:build windows

package ui

import (
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"runtime"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	cfUnicodeText = 13
	gmemMoveable  = 0x0002
)

var (
	user32DLL         = windows.NewLazySystemDLL("user32.dll")
	kernel32DLL       = windows.NewLazySystemDLL("kernel32.dll")
	procOpenClipboard = user32DLL.NewProc("OpenClipboard")
	procCloseClip     = user32DLL.NewProc("CloseClipboard")
	procEmptyClip     = user32DLL.NewProc("EmptyClipboard")
	procSetClipData   = user32DLL.NewProc("SetClipboardData")
	procGlobalAlloc   = kernel32DLL.NewProc("GlobalAlloc")
	procGlobalLock    = kernel32DLL.NewProc("GlobalLock")
	procGlobalUnlock  = kernel32DLL.NewProc("GlobalUnlock")
	procGlobalFree    = kernel32DLL.NewProc("GlobalFree")
	procRtlMoveMemory = kernel32DLL.NewProc("RtlMoveMemory")

	osc52Writer io.Writer = os.Stdout
	bellWriter  io.Writer = os.Stdout
)

func init() {
	clipboardWriter = copyClipboardWinOrOSC52
}

// copyClipboardWinOrOSC52 writes text to the Windows clipboard using pure-Go Win32
// syscalls (no cgo, no external subprocess) and falls back to OSC 52 on failure.
func copyClipboardWinOrOSC52(s string) error {
	if err := copyWin32Clipboard(s); err == nil {
		return nil
	}
	return writeOSC52(osc52Writer, s)
}

func copyWin32Clipboard(s string) error {
	u16, err := windows.UTF16FromString(s)
	if err != nil {
		return err
	}
	r, _, e := procOpenClipboard.Call(0)
	if r == 0 {
		if e != nil && e != windows.ERROR_SUCCESS {
			return e
		}
		return fmt.Errorf("OpenClipboard failed")
	}
	defer procCloseClip.Call()

	procEmptyClip.Call()

	byteLen := uintptr(len(u16) * 2)
	hMem, _, e := procGlobalAlloc.Call(gmemMoveable, byteLen)
	if hMem == 0 {
		return fmt.Errorf("GlobalAlloc failed: %v", e)
	}
	ptr, _, e := procGlobalLock.Call(hMem)
	if ptr == 0 {
		procGlobalFree.Call(hMem)
		return fmt.Errorf("GlobalLock failed: %v", e)
	}
	procRtlMoveMemory.Call(ptr, uintptr(unsafe.Pointer(&u16[0])), byteLen)
	runtime.KeepAlive(u16)
	procGlobalUnlock.Call(hMem)

	hSet, _, e := procSetClipData.Call(cfUnicodeText, hMem)
	if hSet == 0 {
		procGlobalFree.Call(hMem)
		return fmt.Errorf("SetClipboardData failed: %v", e)
	}
	return nil
}

func writeOSC52(w io.Writer, s string) error {
	if w == nil {
		return fmt.Errorf("no OSC 52 output writer")
	}
	enc := base64.StdEncoding.EncodeToString([]byte(s))
	_, err := fmt.Fprintf(w, "\x1b]52;c;%s\a", enc)
	return err
}
