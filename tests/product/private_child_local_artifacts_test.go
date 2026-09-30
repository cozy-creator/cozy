package producttest

import (
	"database/sql"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

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
	checkTensor := func(producer machineChildProof, value int) {
		t.Helper()
		artifact, problem := records.DecodeModelArtifact(producer.Result)
		fatal(t, problem)
		if artifact == nil {
			t.Fatal("native result has no exact artifact")
		}
		command := exec.Command(filepath.Join(control, "bin", "python"), filepath.Join("testdata", "private_child_read.py"),
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
			// A completed no-output caller no longer owns its intermediates. Return
			// the final checkpoint explicitly before testing pruning and restart;
			// changing only this caller must still reuse both operation results.
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
		if link == nil || !link.Collected || owed != (cycle == 3) {
			t.Fatalf("successful caller collection lost explicit result custody: link=%+v owed=%v", link, owed)
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
	var removed uint32
	var reclaimed uint64
	journal, err := sql.Open("sqlite", "file:"+machineJournal(root)+"?mode=ro")
	must(t, err)
	defer journal.Close()
	journal.SetMaxOpenConns(1)
	for deadline := time.Now().Add(10 * time.Second); ; {
		var result struct {
			Removed uint32 `json:"removed_entries"`
			Bytes   uint64 `json:"reclaimed_bytes"`
			Busy    bool   `json:"store_busy"`
		}
		must(t, json.Unmarshal([]byte(run("cache", "prune")), &result))
		removed += result.Removed
		reclaimed += result.Bytes
		// Collection can finish while a child's temporary custody is still settling.
		// A non-busy native GC may therefore skip A. Observe its exact cache mapping
		// leaving before the fresh caller proves recomputation below.
		var cached bool
		must(t, journal.QueryRow("SELECT EXISTS(SELECT 1 FROM operation_cache WHERE lower(hex(key))=?)",
			originalA.Computation).Scan(&cached))
		if !result.Busy && !cached {
			break
		}
		if time.Now().After(deadline) {
			logNativeMemoCustody(t, journal)
			t.Fatalf("native pruning did not remove unused A: cached=%t store_busy=%t entries=%d bytes=%d",
				cached, result.Busy, removed, reclaimed)
		}
		time.Sleep(50 * time.Millisecond)
	}
	if removed == 0 || reclaimed == 0 {
		t.Fatalf("obsolete B cache was not pruned: entries=%d bytes=%d", removed, reclaimed)
	}
	checkTensor(latestB, 15)
	var down struct {
		NotClosed int `json:"not_closed_cleanly"`
	}
	must(t, json.Unmarshal([]byte(run("down", "--all")), &down))
	if down.NotClosed != 0 {
		t.Fatalf("shutdown abandoned ordinary native cleanup: %+v", down)
	}
	// The caller retained B, so pruning could remove only unused A and obsolete B.
	// A fresh caller recomputes A; its identical manifest still hits retained B.
	run("run", script, "--await")
	children := machineChildren(t, root, store, "5")
	if len(children) != 2 || children[0].Executions != 1 || children[1].Executions != 0 ||
		children[1].Computation != latestB.Computation || string(children[1].Result) != string(latestB.Result) {
		t.Fatalf("unused input did not recompute or retained result was lost: %+v", children)
	}
	checkTensor(children[0], 7)
	for _, name := range []string{"source", "candidate"} {
		if _, err := os.Stat(filepath.Join(project, name, "uv.lock")); !os.IsNotExist(err) {
			t.Fatal("private intake modified the author's lock files")
		}
	}
	t.Logf("local native memo cycles, daemon restarts and prune passed: entries=%d bytes=%d", removed, reclaimed)
}

// Bounded, read-only ownership facts for a failed pruning barrier; no result payloads
// or delegated credentials enter the test log.
func logNativeMemoCustody(t *testing.T, journal *sql.DB) {
	t.Helper()
	for _, query := range []struct{ name, sql string }{
		{"memo", `SELECT json_group_array(json_object('id',id,'state',state,
 'source',json_extract(CAST(body AS TEXT),'$.source.request_id'),
 'holds',json_extract(CAST(body AS TEXT),'$.holds'))) FROM (SELECT * FROM operation_cache LIMIT 32)`},
		{"readers", `SELECT json_group_array(json_object('consumer',consumer,'state',state,'cache',cache_id))
 FROM (SELECT * FROM operation_lookups LIMIT 32)`},
		{"holds", `SELECT json_group_array(json_object('id',id,'state',state,'transaction',transaction_id))
 FROM (SELECT * FROM holds WHERE state<>'released' LIMIT 64)`},
		{"recipients", `SELECT json_group_array(json_object('recipient',recipient,'path',path))
 FROM (SELECT * FROM execution_model_holds LIMIT 64)`},
		{"executions", `SELECT json_group_array(json_object('request',request,'state',state,'desired',desired,
 'collected',collected,'retention_waived',retention_waived)) FROM (SELECT * FROM executions LIMIT 32)`},
	} {
		var value string
		if err := journal.QueryRow(query.sql).Scan(&value); err != nil {
			t.Logf("native memo %s: %v", query.name, err)
		} else {
			t.Logf("native memo %s: %s", query.name, value)
		}
	}
}
