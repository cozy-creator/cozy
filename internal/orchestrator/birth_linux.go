//go:build linux

package orchestrator

import (
	"os"
	"strconv"
	"strings"
)

// birthOf is the OS process-birth identity: /proc/<pid>/stat field 22, the kernel's own
// start time in clock ticks. A reused pid has a different birth, which is why a pid
// alone is never enough to adopt a worker as warm. `birth_windows.go` and
// `birth_darwin.go` answer the same question with the identity their kernel keeps.
func birthOf(pid int) string {
	data, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return ""
	}
	// The comm field may contain spaces; everything after the last ')' is positional.
	tail := string(data)
	if i := strings.LastIndex(tail, ")"); i >= 0 {
		tail = tail[i+1:]
	}
	fields := strings.Fields(tail)
	if len(fields) < 20 {
		return ""
	}
	return fields[19] // (22) starttime, offset by the two fields consumed above
}
