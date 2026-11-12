//go:build windows

package hw

import (
	"encoding/binary"
	"unsafe"
)

const (
	unsafeSizeofMemoryStatusEx     = uint32(unsafe.Sizeof(memoryStatusEx{}))
	unsafeSizeofProcessMemCounters = uint32(unsafe.Sizeof(processMemoryCountersEx{}))
)

func unsafePointer(v *memoryStatusEx) unsafe.Pointer {
	return unsafe.Pointer(v)
}

func unsafeProcMemPtr(v *processMemoryCountersEx) unsafe.Pointer {
	return unsafe.Pointer(v)
}

func unsafeByteSlicePtr(b []byte) unsafe.Pointer {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Pointer(&b[0])
}

func unsafeUint32Ptr(v *uint32) unsafe.Pointer {
	return unsafe.Pointer(v)
}

func readLE32(b []byte) uint32 { return binary.LittleEndian.Uint32(b) }
func readLE16(b []byte) uint16 { return binary.LittleEndian.Uint16(b) }
func readLE64(b []byte) uint64 { return binary.LittleEndian.Uint64(b) }
