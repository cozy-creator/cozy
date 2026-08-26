//go:build windows

package orchestrator

import (
	"strconv"

	"golang.org/x/sys/windows"
)

// birthOf on Windows: the process CREATION TIME from GetProcessTimes, as 100ns ticks
// since 1601. Same contract as the Linux and Darwin spellings — a reused pid has a
// different birth.
func birthOf(pid int) string {
	handle, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return ""
	}
	defer windows.CloseHandle(handle)
	var creation, exit, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(handle, &creation, &exit, &kernel, &user); err != nil {
		return ""
	}
	ticks := uint64(creation.HighDateTime)<<32 | uint64(creation.LowDateTime)
	return strconv.FormatUint(ticks, 10)
}
