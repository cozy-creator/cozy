package producttest

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/flock"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/userunit"
)

func TestLegacyLiveRecordRefusesNewNamespace(t *testing.T) {
	h := machines.NewHost(t.TempDir(), "", nil)
	old := filepath.Join(h.Root(), "usr/local/bin/pod-supervisor")
	must(t, os.MkdirAll(filepath.Dir(old), 0755))
	sleep, err := exec.LookPath("sleep")
	must(t, err)
	data, err := os.ReadFile(sleep)
	must(t, err)
	must(t, os.WriteFile(old, data, 0755))
	process := exec.Command(old, "3600")
	must(t, process.Start())
	t.Cleanup(func() { _ = process.Process.Kill(); _ = process.Wait() })
	record, _ := json.Marshal(map[string]any{"pid": process.Process.Pid})
	must(t, os.WriteFile(filepath.Join(filepath.Dir(h.Root()), "host.json"), record, 0600))
	receipt := filepath.Join(h.Root(), "run/cozy/bootstrap/readiness-envelope.json")
	must(t, os.MkdirAll(filepath.Dir(receipt), 0700))
	must(t, os.WriteFile(receipt, []byte("old receipt"), 0600))
	_, problem := h.Ensure(context.Background(), "", nil, true)
	if problem == nil || problem.ErrName() != "machine.legacy_process_running" {
		t.Fatalf("new lifecycle admitted over old process: %v", problem)
	}
	if body, err := os.ReadFile(receipt); err != nil || string(body) != "old receipt" {
		t.Fatal("refusal changed shared readiness")
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(h.Root()), "agent.json")); !os.IsNotExist(err) {
		t.Fatal("refusal wrote current process identity")
	}
}

func TestMissingInstallationMetadataNeverBootstrapsOverLiveAgent(t *testing.T) {
	h := machines.NewHost(t.TempDir(), "", nil)
	binary := filepath.Join(h.Root(), "usr/local/bin/cozy-machine")
	must(t, os.MkdirAll(filepath.Dir(binary), 0755))
	sleep, err := exec.LookPath("sleep")
	must(t, err)
	data, err := os.ReadFile(sleep)
	must(t, err)
	must(t, os.WriteFile(binary, data, 0755))
	process := exec.Command(binary, "3600")
	must(t, process.Start())
	t.Cleanup(func() { _ = process.Process.Kill(); _ = process.Wait() })
	record, _ := json.Marshal(map[string]any{"pid": process.Process.Pid})
	dir := filepath.Dir(h.Root())
	must(t, os.WriteFile(filepath.Join(dir, "agent.json"), record, 0600))
	_, problem := h.Install(t.Context(), machines.Source{}, "must-not-run-uv")
	if problem == nil || problem.ErrName() != "machine.installation_unreadable" {
		t.Fatalf("bootstrap admitted over live agent: %v", problem)
	}
	if err := process.Process.Signal(syscall.Signal(0)); err != nil {
		t.Fatalf("live agent was terminated: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "installed.json")); !os.IsNotExist(err) {
		t.Fatal("bootstrap fabricated installation metadata")
	}
}

func TestSharedExecutableDoesNotAuthorizeProcessTermination(t *testing.T) {
	for _, name := range []string{"cozy", "cozy-machine"} {
		t.Run(name, func(t *testing.T) {
			h := machines.NewHost(t.TempDir(), "", nil)
			binary := filepath.Join(h.Root(), "usr/local/bin", name)
			must(t, os.MkdirAll(filepath.Dir(binary), 0755))
			sleep, err := exec.LookPath("sleep")
			must(t, err)
			must(t, os.Symlink(sleep, binary))
			process := exec.Command(sleep, "3600")
			must(t, process.Start())
			t.Cleanup(func() { _ = process.Process.Kill(); _ = process.Wait() })
			record, _ := json.Marshal(map[string]any{"pid": process.Process.Pid})
			must(t, os.WriteFile(filepath.Join(filepath.Dir(h.Root()), "agent.json"), record, 0600))
			fatal(t, h.Stop(t.Context()))
			if err := process.Process.Signal(syscall.Signal(0)); err != nil {
				t.Fatalf("unrelated shared executable was terminated: %v", err)
			}
		})
	}
}

func TestBootstrapRefusesKernelOwnedRootWithoutClientRecords(t *testing.T) {
	for _, relative := range []string{"var/lib/cozy/machine/agent.lock", "run/cozy/worker/worker.lock"} {
		t.Run(relative, func(t *testing.T) {
			dir := t.TempDir()
			h := machines.NewHost(dir, "", nil)
			path := filepath.Join(h.Root(), relative)
			must(t, os.MkdirAll(filepath.Dir(path), 0700))
			file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0600)
			must(t, err)
			defer file.Close()
			must(t, flock.Exclusive(file))
			defer flock.Release(file)
			_, err = file.WriteString("owned journal lease")
			must(t, err)
			_, problem := h.Install(t.Context(), machines.Source{}, "must-not-run-uv")
			if problem == nil || problem.ErrName() != "machine.busy" {
				t.Fatalf("bootstrap ignored kernel ownership: %v", problem)
			}
			for _, name := range []string{"installed.json", "agent.json", "identity", "root/authorized_keys"} {
				if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
					t.Fatalf("refused bootstrap wrote %s", name)
				}
			}
			body, err := os.ReadFile(path)
			must(t, err)
			if string(body) != "owned journal lease" {
				t.Fatal("refused bootstrap changed lease contents")
			}
			// With installed software but a lost client launch record, Ensure
			// must honor the same kernel owner before rotating its receipt key.
			must(t, os.WriteFile(filepath.Join(dir, "installed.json"), []byte(`{"host":{"module":"github.com/cozy-creator/cozy-runtime/machine-agent"},"host_pinned":true}`), 0600))
			receiptKey := filepath.Join(dir, "receipt-key")
			must(t, os.WriteFile(receiptKey, []byte("retained private key"), 0600))
			_, problem = h.Ensure(t.Context(), "", nil, true)
			if problem == nil || problem.ErrName() != "machine.busy" {
				t.Fatalf("Ensure ignored kernel ownership: %v", problem)
			}
			afterKey, err := os.ReadFile(receiptKey)
			must(t, err)
			if string(afterKey) != "retained private key" {
				t.Fatal("Ensure rotated the current owner's receipt key")
			}
			for _, name := range []string{"agent.json", "identity", "root/authorized_keys"} {
				if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
					t.Fatalf("refused Ensure wrote %s", name)
				}
			}
			// The failed acquisition must release any earlier lease in its pair.
			if relative == "run/cozy/worker/worker.lock" {
				other, err := os.OpenFile(filepath.Join(h.Root(), "var/lib/cozy/machine/agent.lock"), os.O_RDWR, 0600)
				must(t, err)
				defer other.Close()
				must(t, flock.Exclusive(other))
				must(t, flock.Release(other))
			}
		})
	}
}

func TestEnsureRefusesConcurrentHostMutationWithoutWaiting(t *testing.T) {
	dir := t.TempDir()
	file, err := os.OpenFile(filepath.Join(dir, "host.lock"), os.O_CREATE|os.O_RDWR, 0600)
	must(t, err)
	defer file.Close()
	must(t, flock.Exclusive(file))
	defer flock.Release(file)
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	_, problem := machines.NewHost(dir, "", nil).Ensure(ctx, "", nil, true)
	if ctx.Err() != nil || problem == nil || problem.ErrName() != "machine.busy" {
		t.Fatalf("Ensure waited for another controller's lifecycle lock: %v (context %v)", problem, ctx.Err())
	}
	if _, err := os.Stat(filepath.Join(dir, "agent.json")); !os.IsNotExist(err) {
		t.Fatal("busy Ensure wrote launch authority")
	}
}

func TestStaleProcessGenerationDoesNotAuthorizeTermination(t *testing.T) {
	dir := t.TempDir()
	h := machines.NewHost(dir, "", nil)
	binary := filepath.Join(h.Root(), "usr/local/bin/cozy-machine")
	must(t, os.MkdirAll(filepath.Dir(binary), 0755))
	sleep, err := exec.LookPath("sleep")
	must(t, err)
	data, err := os.ReadFile(sleep)
	must(t, err)
	must(t, os.WriteFile(binary, data, 0755))
	process := exec.Command(binary, "3600")
	must(t, process.Start())
	t.Cleanup(func() { _ = process.Process.Kill(); _ = process.Wait() })
	record, _ := json.Marshal(map[string]any{"pid": process.Process.Pid, "start_ticks": uint64(1)})
	must(t, os.WriteFile(filepath.Join(dir, "agent.json"), record, 0600))
	fatal(t, h.Stop(t.Context()))
	if err := process.Process.Signal(syscall.Signal(0)); err != nil {
		t.Fatalf("stale process generation authorized termination: %v", err)
	}
}

func TestLiveRuntimeMissingMetadataPreservesPausedWork(t *testing.T) {
	_, h, launch := scopedMachine(t, true)
	journal := func() string {
		t.Helper()
		script := `import sqlite3,sys,json
c=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro',uri=True)
r=c.execute("SELECT state,desired,generation,ordinal,collected,retention_waived FROM executions WHERE owner='cozy-local-client' AND request='scoped-paused'").fetchone()
assert r and r[0]=='paused',r
print(json.dumps(r))`
		out, err := exec.Command(machinePython(t), "-I", "-c", script, filepath.Join(h.Root(), "var/lib/tensorfs/.cozy-workspace/journal.sqlite3")).CombinedOutput()
		if err != nil {
			t.Fatalf("read retained execution: %v %s", err, out)
		}
		return string(out)
	}
	beforeJournal := journal()
	path := filepath.Join(filepath.Dir(h.Root()), "installed.json")
	body, err := os.ReadFile(path)
	must(t, err)
	must(t, os.Remove(path))
	defer func() { must(t, os.WriteFile(path, body, 0600)) }()
	authority := func() string {
		t.Helper()
		files := map[string]*string{}
		for _, relative := range []string{"var/lib/cozy/machine/hub-access.json", "run/cozy/bootstrap/machine-hubs.json"} {
			data, err := os.ReadFile(filepath.Join(h.Root(), relative))
			if os.IsNotExist(err) {
				files[relative] = nil
				continue
			}
			must(t, err)
			value := string(data)
			files[relative] = &value
		}
		data, err := json.Marshal(files)
		must(t, err)
		return string(data)
	}
	before := authority()
	_, problem := h.Install(t.Context(), machines.Source{}, "must-not-run-uv")
	if problem == nil || problem.ErrName() != "machine.installation_unreadable" {
		t.Fatalf("missing metadata admitted live bootstrap: %v", problem)
	}
	if authority() != before {
		t.Fatal("refused bootstrap changed machine authority")
	}
	awaitScopedMachine(t, h, launch)
	if journal() != beforeJournal {
		t.Fatal("refused bootstrap changed retained execution")
	}
}

// A live agent whose launch record is missing is read back from its systemd user unit and
// adopted. Without a user manager there is no unit to read it from: the launch is refused and
// the agent keeps running untouched.
func TestMissingLaunchRecordAdoptsKernelOwnedRuntime(t *testing.T) {
	_, h, launch := scopedMachine(t, true)
	dir := filepath.Dir(h.Root())
	recordPath := filepath.Join(dir, "agent.json")
	record, err := os.ReadFile(recordPath)
	must(t, err)
	paths := []string{filepath.Join(dir, "receipt-key")}
	pin, problem := h.Pin()
	fatal(t, problem)
	transport := &http.Transport{TLSClientConfig: pin.TLSConfig()}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport}
	readReceipt := func() []byte {
		t.Helper()
		request, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://"+launch.MediaAddr+"/v1/bootstrap/receipt", nil)
		must(t, err)
		response, err := client.Do(request)
		must(t, err)
		defer response.Body.Close()
		if response.StatusCode != http.StatusOK {
			t.Fatalf("receipt read: HTTP %d", response.StatusCode)
		}
		body, err := io.ReadAll(response.Body)
		must(t, err)
		return body
	}
	receiptBefore := readReceipt()
	before := map[string][]byte{}
	for _, path := range paths {
		body, err := os.ReadFile(path)
		must(t, err)
		before[path] = body
	}
	// Restore only this isolated fixture's original ownership records for cleanup.
	defer func() {
		must(t, os.WriteFile(recordPath, record, 0600))
		for path, body := range before {
			must(t, os.WriteFile(path, body, 0600))
		}
	}()
	must(t, os.Remove(recordPath))
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	adopts := userunit.Available()
	adopted, problem := machines.NewHost(dir, "", nil).Ensure(ctx, "", nil, true)
	if adopts && (problem != nil || adopted.PID != launch.PID || adopted.BootID != launch.BootID) {
		t.Errorf("launch did not adopt the running kernel owner %d: %+v, %v", launch.PID, adopted, problem)
	}
	if !adopts && (problem == nil || (problem.ErrName() != "machine.process_untracked" && problem.ErrName() != "machine.busy")) {
		t.Errorf("launch ignored an existing kernel owner: %v", problem)
	}
	for path, body := range before {
		after, err := os.ReadFile(path)
		if err != nil || string(after) != string(body) {
			t.Errorf("launch changed the running owner's %s", filepath.Base(path))
		}
	}
	if _, err := os.Stat(recordPath); adopts && err != nil {
		t.Errorf("the adopted owner has no launch record: %v", err)
	} else if !adopts && !os.IsNotExist(err) {
		t.Error("launch replaced missing ownership with another process")
	}
	if string(readReceipt()) != string(receiptBefore) {
		t.Error("launch changed the running owner's public identity receipt")
	}
	awaitScopedMachine(t, h, launch)
}
