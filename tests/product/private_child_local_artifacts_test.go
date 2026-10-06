package producttest

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/records"
)

// The same A/B author files used by the real rental proof run here through the
// ordinary local CLI on this computer's machine, whose Store keeps the results. No Docker,
// PodHost replacement, native ACK or cache is faked.
func TestUnpublishedChildLocalArtifactsShareWorkspaceMemoization(t *testing.T) {
	integration(t)
	control := filepath.Join(t.TempDir(), "control")
	uv := func(args ...string) {
		t.Helper()
		if out, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
			t.Fatalf("uv: %v\n%s", err, out)
		}
	}
	uv("venv", control, "--python", "3.12")
	// This environment runs Creator's provisioning commands. The authored
	// packages keep their independent SDK floor in copyPrivateTensorProject.
	runtime := "cozy-runtime>=" + hostruntime.ToolFloor
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
	unpressuredMachine(t, root)
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if t.Failed() {
			t.Logf("local native memo evidence retained at %s", root)
		} else {
			must(t, removeAllForce(root))
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
		run("up") // normal startup reattaches the Runtime-owned journal
	}
	var store *records.Store
	var originalA, originalB, latestB machineChildProof
	reader := storeReader(t)
	checkTensor := func(producer machineChildProof, value int) {
		t.Helper()
		artifact, problem := records.DecodeModelArtifact(producer.Result)
		fatal(t, problem)
		if artifact == nil {
			t.Fatal("native result has no exact artifact")
		}
		command := exec.Command(reader, filepath.Join("testdata", "private_child_read.py"),
			artifact.Manifest.Digest, strconv.Itoa(value), machineStore(root))
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
		if cycle == 3 {
			// The caller returns B's checkpoint; changing only the caller must still
			// reuse both operation results.
			body, err := os.ReadFile(script)
			must(t, err)
			body = []byte(strings.Replace(string(body), "async def main():", "from cozy_runtime.author import ModelArtifact\n\nasync def main() -> ModelArtifact:", 1))
			body = []byte(strings.Replace(string(body), "    await candidate(", "    result = await candidate(", 1) + "    return result\n")
			must(t, os.WriteFile(script, body, 0o600))
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
		parent, problem := store.RequestByReference(strconv.Itoa(1 + cycle))
		fatal(t, problem)
		if parent == nil || parent.RetryOf != "" || parent.Worker != "" {
			t.Fatalf("local fresh run gained retry or rental identity: %+v", parent)
		}
		children := machineChildren(t, root, store, strconv.Itoa(1+cycle))
		if len(children) != 2 {
			t.Fatalf("cycle %d children: %+v", cycle, children)
		}
		if cycle == 0 {
			originalA, originalB = children[0], children[1]
			if originalA.State != "succeeded" || originalB.State != "failed" {
				t.Fatalf("A/B failure custody was not preserved: %+v", children)
			}
			continue
		}
		if children[0].Executions != 0 || children[0].Computation != originalA.Computation || string(children[0].Result) != string(originalA.Result) {
			t.Fatalf("cycle %d recomputed or changed A: original=%+v actual=%+v", cycle, originalA, children[0])
		}
		if cycle < 3 {
			if children[1].Executions != 1 {
				t.Fatalf("changed B was not executed exactly once: %+v", children[1])
			}
			if cycle == 1 && children[1].Intent == originalB.Intent {
				t.Fatal("parameter edit reused B's failed input")
			}
			if cycle == 2 && (children[1].Computation == latestB.Computation || children[1].Intent != latestB.Intent) {
				t.Fatal("same-version library edit did not change only implementation identity")
			}
			latestB = children[1]
		} else if children[1].Executions != 0 || children[1].Computation != latestB.Computation || string(children[1].Result) != string(latestB.Result) {
			t.Fatalf("unchanged B did not use the shared workspace result: %+v", children[1])
		}
		link, issue := store.MachineExecution(parent.ID)
		fatal(t, issue)
		owed, issue := store.MachineExecutionOwesWork(parent.ID)
		fatal(t, issue)
		// The machine keeps its runs' checkpoints itself: a collected run owes this computer
		// nothing, even one returning its checkpoint. checkTensor proves each one survives.
		if link == nil || !link.Collected || owed {
			t.Fatalf("a collected run still owes this computer work: link=%+v owed=%v", link, owed)
		}
		if cycle == 3 {
			artifact, problem := records.DecodeModelArtifact(latestB.Result)
			fatal(t, problem)
			var shown struct {
				Result struct {
					Value struct {
						Manifest struct {
							Digest string `json:"digest"`
						} `json:"manifest"`
					} `json:"value"`
				} `json:"result"`
			}
			must(t, json.Unmarshal([]byte(run("run", "show", strconv.Itoa(1+cycle))), &shown))
			if artifact == nil || shown.Result.Value.Manifest.Digest != artifact.Manifest.Digest {
				t.Fatalf("the caller's result is not B's checkpoint: %+v", shown)
			}
		}
		if cycle == 1 {
			// Only the failed predecessor still needs explicit abandonment.
			run("run", "cancel", "1")
		}
		restartDaemon()
		checkTensor(originalA, 7)
		value := 15
		if cycle == 1 {
			value = 14
		}
		checkTensor(latestB, value)
	}
	var down struct {
		NotClosed int `json:"not_closed_cleanly"`
	}
	must(t, json.Unmarshal([]byte(run("down", "--all")), &down))
	if down.NotClosed != 0 {
		t.Fatalf("shutdown abandoned ordinary native cleanup: %+v", down)
	}
	for _, name := range []string{"source", "candidate"} {
		if _, err := os.Stat(filepath.Join(project, name, "uv.lock")); !os.IsNotExist(err) {
			t.Fatal("private intake modified the author's lock files")
		}
	}
}
