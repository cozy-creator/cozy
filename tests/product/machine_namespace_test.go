package producttest

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/flock"
	"github.com/cozy-creator/cozy/internal/machines"
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
	_, problem := h.Ensure(context.Background(), nil)
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
	// A live machine holds its root's lock for its life.
	lock := filepath.Join(h.Root(), "var/lib/cozy/machine/agent.lock")
	must(t, os.MkdirAll(filepath.Dir(lock), 0700))
	held, err := os.OpenFile(lock, os.O_CREATE|os.O_RDWR, 0600)
	must(t, err)
	defer held.Close()
	must(t, flock.Exclusive(held))
	defer flock.Release(held)
	_, problem := h.Install(t.Context(), machines.Source{}, "must-not-run-uv")
	if problem == nil || problem.ErrName() != "machine.busy" {
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
			for _, name := range []string{"installed.json", "agent.json", "identity", "owner.pem"} {
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
			_, problem = h.Ensure(t.Context(), nil)
			if problem == nil || problem.ErrName() != "machine.busy" {
				t.Fatalf("Ensure ignored kernel ownership: %v", problem)
			}
			afterKey, err := os.ReadFile(receiptKey)
			must(t, err)
			if string(afterKey) != "retained private key" {
				t.Fatal("Ensure rotated the current owner's receipt key")
			}
			for _, name := range []string{"agent.json", "identity", "owner.pem"} {
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
	_, problem := machines.NewHost(dir, "", nil).Ensure(ctx, nil)
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
