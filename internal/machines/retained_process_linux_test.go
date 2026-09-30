package machines

import (
	"bufio"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestRetainedAgentProcessHelper(t *testing.T) {
	if os.Getenv("COZY_TEST_AGENT_IDENTITY_HELPER") != "1" {
		return
	}
	_, _ = io.WriteString(os.Stdout, "ready\n")
	_, _ = io.Copy(io.Discard, os.Stdin)
	os.Exit(0)
}

func TestRecordedRetainedAgentSurvivesLauncherReplacement(t *testing.T) {
	h := NewHost(t.TempDir(), "", nil)
	source, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	digest, err := fileDigest(source)
	if err != nil {
		t.Fatal(err)
	}
	application := filepath.Join(h.Root(), "usr/local/lib/cozy-machine", digest, "cozy-machine")
	if err := os.MkdirAll(filepath.Dir(application), 0755); err != nil {
		t.Fatal(err)
	}
	if err := copyFile(source, application, 0755); err != nil {
		t.Fatal(err)
	}
	command := exec.Command(application, "-test.run=^TestRetainedAgentProcessHelper$")
	command.Env = append(os.Environ(), "COZY_TEST_AGENT_IDENTITY_HELPER=1")
	input, err := command.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	output, err := command.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = input.Close()
		_ = command.Process.Kill()
		_ = command.Wait()
	})
	if ready, err := bufio.NewReader(output).ReadString('\n'); err != nil || ready != "ready\n" {
		t.Fatalf("helper not ready: %q %v", ready, err)
	}
	record := hostRecord{PID: command.Process.Pid, StartTicks: processStartTicks(command.Process.Pid)}
	if record.StartTicks == 0 {
		t.Fatal("helper process has no start identity")
	}
	writeRecord := func(value hostRecord) {
		t.Helper()
		raw, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err := writePrivate(h.path("agent.json"), raw); err != nil {
			t.Fatal(err)
		}
	}
	if h.alive(record.PID) {
		t.Fatal("an unrecorded retained process was adopted")
	}
	writeRecord(record)
	if !h.alive(record.PID) {
		t.Fatal("the recorded retained agent was reported dead")
	}
	if err := os.MkdirAll(filepath.Dir(h.binary()), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(source, h.binary()); err != nil {
		t.Fatal(err)
	}
	if !h.alive(record.PID) {
		t.Fatal("selecting a different launcher lost the live process")
	}
	if status, problem := h.Status(); problem != nil || !status.Running || status.PID != record.PID {
		t.Fatalf("status lost the retained process: %+v %v", status, problem)
	}
	for _, altered := range []hostRecord{
		{PID: record.PID, StartTicks: record.StartTicks + 1},
		{PID: record.PID, StartTicks: 0},
		{PID: record.PID + 1, StartTicks: record.StartTicks},
	} {
		writeRecord(altered)
		if h.alive(record.PID) {
			t.Fatal("a stale or incomplete record authorized a retained process")
		}
	}
	writeRecord(record)
	for _, target := range []string{
		filepath.Join(t.TempDir(), "usr/local/lib/cozy-machine", digest, "cozy-machine"),
		filepath.Join(h.Root(), "usr/local/lib/cozy-machine", strings.Repeat("z", 64), "cozy-machine"),
		filepath.Join(h.Root(), "usr/local/lib/cozy-machine", digest, "another-program"),
	} {
		if h.recordedApplication(record.PID, target) {
			t.Fatalf("unrelated path was accepted: %s", target)
		}
	}
}
