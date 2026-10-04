package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// A machine granted no WebRTC port serves no cozy/1: its receipt names no webrtc member, and it
// listens on nothing but its worker and media ports.
func TestWebRTCListensOnlyWhenGranted(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host=<cozy-machine>")
	}
	root, err := os.MkdirTemp(os.TempDir(), "czn")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if !t.Failed() {
			_ = removeAllForce(root)
		}
	})
	if code, out := runCozy(t, root, "machine", "start"); code != 0 {
		t.Fatalf("machine start [exit %d]\n%s", code, out)
	}
	dir := filepath.Join(root, "machine")
	var agent struct {
		PID        int `json:"pid"`
		WorkerPort int `json:"worker_port"`
		MediaPort  int `json:"media_port"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "agent.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &agent))
	var envelope struct {
		Payload []byte `json:"payload"`
	}
	var receipt struct {
		WebRTC *struct {
			Port int `json:"port"`
		} `json:"webrtc"`
	}
	raw, err = os.ReadFile(filepath.Join(dir, "root/run/cozy/bootstrap/readiness-envelope.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &envelope))
	must(t, json.Unmarshal(envelope.Payload, &receipt))
	ports := processListenerPorts(t, agent.PID)
	if receipt.WebRTC != nil || !slices.Equal(ports, slices.Sorted(slices.Values([]int{agent.WorkerPort, agent.MediaPort}))) {
		t.Fatalf("a machine granted no WebRTC port names %+v in its receipt and listens on %v", receipt.WebRTC, ports)
	}
}

func processListenerPorts(t *testing.T, pid int) []int {
	t.Helper()
	sockets := map[string]bool{}
	fds, err := os.ReadDir(fmt.Sprintf("/proc/%d/fd", pid))
	must(t, err)
	for _, fd := range fds {
		if link, err := os.Readlink(fmt.Sprintf("/proc/%d/fd/%s", pid, fd.Name())); err == nil && strings.HasPrefix(link, "socket:[") {
			sockets[strings.TrimSuffix(strings.TrimPrefix(link, "socket:["), "]")] = true
		}
	}
	ports := []int{}
	for _, table := range []string{"tcp", "tcp6"} {
		raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/net/%s", pid, table))
		must(t, err)
		for _, line := range strings.Split(string(raw), "\n") {
			fields := strings.Fields(line)
			if len(fields) < 10 || fields[3] != "0A" || !sockets[fields[9]] {
				continue
			}
			_, hex, _ := strings.Cut(fields[1], ":")
			port, err := strconv.ParseUint(hex, 16, 16)
			must(t, err)
			ports = append(ports, int(port))
		}
	}
	slices.Sort(ports)
	return slices.Compact(ports)
}
