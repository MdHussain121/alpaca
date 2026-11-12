//go:build windows

package runner

import "unsafe"

var unsafeSizeofJobInfo = uint32(unsafe.Sizeof(jobObjectExtendedLimitInformation{}))

func unsafePointerJobInfo(v *jobObjectExtendedLimitInformation) unsafe.Pointer {
	return unsafe.Pointer(v)
}
