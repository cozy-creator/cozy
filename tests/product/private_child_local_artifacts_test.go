package producttest

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

// The same A/B author files used by the real rental proof run here through the
// ordinary local CLI. No Docker, PodHost replacement, native ACK or cache is faked.
func TestPrivateChildLocalArtifactsShareWorkspaceMemoization(t *testing.T) {
	control := filepath.Join(t.TempDir(), "control")
	uv := func(args ...string) {
		t.Helper()
		if out, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
			t.Fatalf("uv: %v\n%s", err, out)
		}
	}
	uv("venv", control, "--python", "3.12")
	runtime := "cozy-runtime==0.10.0"
	if *privateChildRuntimeWheel != "" {
		runtime = *privateChildRuntimeWheel
	}
	install := []string{"pip", "install", "--python", filepath.Join(control, "bin", "python"), runtime}
	if *privateChildTensorFSWheel != "" {
		install = append(install, *privateChildTensorFSWheel)
	}
	uv(install...)
	root, err := os.MkdirTemp("", "cozy-native-memo-")
	must(t, err)
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		args := []string{"down"}
		if !t.Failed() {
			args = append(args, "--all")
		}
		_, _ = runCozyPath(t, root, path, args...)
		if t.Failed() {
			t.Logf("local native memo evidence retained at %s", root)
		} else {
			_ = os.RemoveAll(root)
		}
	})
	defer tracePrivateChildWait(t, root)()
	project := copyPrivateTensorProject(t, root, "")
	script := filepath.Join(project, "recipe.py")
	run := func(args ...string) string {
		t.Helper()
		code, out := runCozyPath(t, root, path, append(args, "--json")...)
		if code != 0 {
			t.Fatalf("cozy %s failed [%d]: %s", strings.Join(args, " "), code, out)
		}
		return out
	}
	restartDaemon := func() {
		t.Helper()
		lock, err := os.ReadFile(filepath.Join(root, "daemon.lock"))
		must(t, err)
		pid := 0
		for _, line := range strings.Split(string(lock), "\n") {
			if value, ok := strings.CutPrefix(line, "pid="); ok {
				pid, err = strconv.Atoi(value)
				must(t, err)
			}
		}
		if pid <= 0 {
			t.Fatal("the test daemon has no process identity")
		}
		process, err := os.FindProcess(pid)
		must(t, err)
		must(t, process.Kill())
		run("up") // normal startup reaps prior worker processes and reopens the journal
	}
	var store *records.Store
	var originalA, originalB, latestB records.Request
	checkTensor := func(producer records.Request, value int) {
		t.Helper()
		outputs, problem := store.AllModelTransferWeights(producer.ID, producer.Ordinal)
		fatal(t, problem)
		if len(outputs) != 1 {
			t.Fatalf("native result has no exact producer receipt: %+v", outputs)
		}
		command := exec.Command(filepath.Join(control, "bin", "python"), filepath.Join("testdata", "private_child_read.py"),
			outputs[0].ManifestID, strconv.Itoa(value), filepath.Join(root, "tensorfs"))
		out, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("native tensor after restart and GC: %v %s", err, out)
		}
		t.Logf("retained native tensor: %s", out)
	}
	for cycle := 0; cycle < 4; cycle++ {
		if cycle == 1 {
			body, err := os.ReadFile(script)
			must(t, err)
			must(t, os.WriteFile(script, []byte(strings.Replace(string(body), "factor=0", "factor=2", 1)), 0o600))
		}
		if cycle == 2 {
			file := filepath.Join(project, "candidate", "tensor_candidate.py")
			body, err := os.ReadFile(file)
			must(t, err)
			must(t, os.WriteFile(file, []byte(strings.Replace(string(body), "value * factor for value", "value * factor + 1 for value", 1)), 0o600))
		}
		code, out := runCozyPath(t, root, path, "run", script, "--await", "--json")
		if cycle == 0 {
			if code == 0 || !strings.Contains(out, "candidate quality gate failed") {
				t.Fatalf("B's author failure was not the result [%d]: %s", code, out)
			}
			opened, issue := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, issue)
			store = opened
			defer store.Close()
		} else if code != 0 {
			t.Fatalf("fresh local cycle %d failed [%d]: %s", cycle, code, out)
		}
		parent, problem := store.RequestByReference(strconv.Itoa(1 + 3*cycle))
		fatal(t, problem)
		if parent == nil || parent.RetryOf != "" || parent.Worker != "" {
			t.Fatalf("local fresh run gained retry or rental identity: %+v", parent)
		}
		children, problem := store.Children(parent.ID)
		fatal(t, problem)
		if len(children) != 2 {
			t.Fatalf("cycle %d children: %+v", cycle, children)
		}
		if cycle == 0 {
			originalA, originalB = children[0], children[1]
			if originalA.State != "succeeded" || originalB.State != "blocked" {
				t.Fatalf("A/B failure custody was not preserved: %+v", children)
			}
			continue
		}
		if children[0].Ordinal != 0 || children[0].ReusedFrom != originalA.ID || children[0].ChildTargetDigest != originalA.ChildTargetDigest {
			t.Fatalf("cycle %d recomputed or changed A: %+v", cycle, children[0])
		}
		if cycle < 3 {
			if children[1].Ordinal != 1 || children[1].ReusedFrom != "" {
				t.Fatalf("changed B was not executed exactly once: %+v", children[1])
			}
			if cycle == 1 && (children[1].ChildTargetDigest != originalB.ChildTargetDigest || children[1].ChildIntentDigest == originalB.ChildIntentDigest) {
				t.Fatal("parameter edit changed B's implementation or reused its failed input")
			}
			if cycle == 2 && (children[1].ChildTargetDigest == latestB.ChildTargetDigest || children[1].ChildIntentDigest != latestB.ChildIntentDigest) {
				t.Fatal("same-version library edit did not change only implementation identity")
			}
			latestB = children[1]
		} else if children[1].Ordinal != 0 || children[1].ReusedFrom != latestB.ID {
			t.Fatalf("unchanged B did not use the shared workspace result: %+v", children[1])
		}
		run("run", "cancel", strconv.Itoa(1+3*(cycle-1)))
		restartDaemon()
		checkTensor(originalA, 7)
		value := 15
		if cycle == 1 {
			value = 14
		}
		checkTensor(latestB, value)
	}
	var removed uint32
	var reclaimed uint64
	for deadline := time.Now().Add(10 * time.Second); ; {
		var result struct {
			Removed uint32 `json:"removed_entries"`
			Bytes   uint64 `json:"reclaimed_bytes"`
			Busy    bool   `json:"store_busy"`
		}
		must(t, json.Unmarshal([]byte(run("cache", "prune")), &result))
		removed += result.Removed
		reclaimed += result.Bytes
		if !result.Busy {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("native GC remained busy after all package workers stopped")
		}
		time.Sleep(50 * time.Millisecond)
	}
	if removed == 0 || reclaimed == 0 {
		t.Fatalf("obsolete B cache was not pruned: entries=%d bytes=%d", removed, reclaimed)
	}
	checkTensor(originalA, 7)
	checkTensor(latestB, 15)
	for _, name := range []string{"source", "candidate"} {
		if _, err := os.Stat(filepath.Join(project, name, "uv.lock")); !os.IsNotExist(err) {
			t.Fatal("private intake modified the author's lock files")
		}
	}
	t.Logf("local native memo cycles, daemon restarts and prune passed: entries=%d bytes=%d", removed, reclaimed)
}
