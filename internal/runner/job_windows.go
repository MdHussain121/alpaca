//go:build windows

package runner

import (
	"fmt"
	"io"
	"net"

	"golang.org/x/sys/windows"
)

const (
	processSetQuota            = 0x0100
	processTerminate           = 0x0001
	jobObjectLimitKillOnClose  = 0x2000
	jobObjectExtendedLimitInfo = 9
)

type jobObjectExtendedLimitInformation struct {
	BasicLimitInformation jobObjectBasicLimitInformation
	IoInfo                ioTimes
	ProcessMemoryLimit    uintptr
	JobMemoryLimit        uintptr
	PeakProcessMemoryUsed uintptr
	PeakJobMemoryUsed     uintptr
}

type jobObjectBasicLimitInformation struct {
	PerProcessUserTimeLimit int64
	PerJobUserTimeLimit     int64
	LimitFlags              uint32
	MinimumWorkingSetSize   uintptr
	MaximumWorkingSetSize   uintptr
	ActiveProcessLimit      uint32
	Affinity                uintptr
	PriorityClass           uint32
	SchedulingClass         uint32
}

type ioTimes struct {
	ReadOperationCount  uint64
	WriteOperationCount uint64
	OtherOperationCount uint64
	ReadTransferCount   uint64
	WriteTransferCount  uint64
	OtherTransferCount  uint64
}

// attachJob puts pid into a Job Object that kills the process when the job
// handle closes, so alpaca.exe dying never leaves an orphan llama-server.
func attachJob(pid uint32) (windows.Handle, error) {
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return 0, err
	}

	info := jobObjectExtendedLimitInformation{}
	info.BasicLimitInformation.LimitFlags = jobObjectLimitKillOnClose
	if _, err := windows.SetInformationJobObject(
		job,
		jobObjectExtendedLimitInfo,
		uintptr(unsafePointerJobInfo(&info)),
		uint32(unsafeSizeofJobInfo),
	); err != nil {
		windows.CloseHandle(job)
		return 0, err
	}

	h, err := windows.OpenProcess(processSetQuota|processTerminate|0x1000, false, pid)
	if err != nil {
		windows.CloseHandle(job)
		return 0, err
	}
	defer windows.CloseHandle(h)

	if err := windows.AssignProcessToJobObject(job, h); err != nil {
		windows.CloseHandle(job)
		return 0, err
	}
	return job, nil
}

// JobHandle owns a Windows Job Object configured with KILL_ON_CLOSE so child
// processes and their descendants terminate deterministically together.
type JobHandle struct {
	h windows.Handle
}

// AttachJob assigns pid to a new Job Object.
func AttachJob(pid uint32) (*JobHandle, error) {
	h, err := attachJob(pid)
	if err != nil {
		return nil, err
	}
	return &JobHandle{h: h}, nil
}

// Terminate kills every process currently inside the Job Object and closes it.
func (j *JobHandle) Terminate() error {
	if j == nil || j.h == 0 {
		return nil
	}
	h := j.h
	j.h = 0
	_ = windows.TerminateJobObject(h, 1)
	return windows.CloseHandle(h)
}

// Close closes the underlying Job Object handle (killing descendants via KILL_ON_CLOSE).
func (j *JobHandle) Close() error {
	if j == nil || j.h == 0 {
		return nil
	}
	h := j.h
	j.h = 0
	return windows.CloseHandle(h)
}

func listenTCP(addr string) (io.Closer, error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("%w", err)
	}
	return ln, nil
}
