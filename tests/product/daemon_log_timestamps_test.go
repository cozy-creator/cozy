package producttest

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// Every daemon log line starts with its UTC write time, so the log lines up with run events.
func TestDaemonLogLinesCarryTheirTime(t *testing.T) {
	root := t.TempDir()
	startDaemonProcess(t, root)
	if code, out := runCozy(t, root, "down", "--json"); code != 0 {
		t.Fatalf("stopping the daemon [%d]: %s", code, out)
	}
	raw, err := os.ReadFile(filepath.Join(root, "daemon.log"))
	must(t, err)
	stamped := regexp.MustCompile(`^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z `)
	lines := strings.Split(strings.TrimRight(string(raw), "\n"), "\n")
	if len(lines) < 2 {
		t.Fatalf("the daemon logged too little to check: %q", raw)
	}
	for _, line := range lines {
		if !stamped.MatchString(line) {
			t.Fatalf("an unstamped daemon log line: %q", line)
		}
	}
}
