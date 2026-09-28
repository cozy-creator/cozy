package producttest

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/flock"
	"github.com/cozy-creator/cozy/internal/home"
)

// The CLI is real; the owner fixture lets us control each publication phase and
// generations independently. Its advertised PID is a disposable child, so a
// regression that signals lock metadata cannot terminate the test process.
func compatibilityOwner(t *testing.T) (home.Layout, *os.File, int, <-chan error) {
	t.Helper()
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	process := exec.Command("sleep", "60")
	must(t, process.Start())
	done := make(chan error, 1)
	go func() { done <- process.Wait() }()
	t.Cleanup(func() { _ = process.Process.Kill() })
	lock, err := os.OpenFile(layout.Daemon, os.O_RDWR|os.O_CREATE, 0600)
	must(t, err)
	must(t, flock.Exclusive(lock))
	t.Cleanup(func() { _ = flock.Release(lock); _ = lock.Close() })
	return layout, lock, process.Process.Pid, done
}

func publishCompatibilityOwner(t *testing.T, layout home.Layout, lock *os.File, pid int, address, schema string) {
	t.Helper()
	body := fmt.Sprintf("addr=%s\nsocket=\npid=%d\nsince=2026-09-12T00:00:00Z\n%s", address, pid, schema)
	must(t, lock.Truncate(0))
	_, err := lock.WriteAt([]byte(body), 0)
	must(t, err)
	if address != "" {
		_, problem := api.Mint(layout)
		fatal(t, problem)
	}
}

func compatibilityCLI(t *testing.T, root string, args ...string) (string, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 13*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, cozyBin, args...)
	command.Env = childEnv(t, root)
	out, err := command.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("CLI readiness never returned: %s", out)
	}
	return string(out), err
}

func TestDaemonAPICompatibilityPreservesLiveOwner(t *testing.T) {
	for _, schema := range []string{"", "schema=1\n", "schema=999\n"} {
		t.Run(strings.TrimSpace(schema), func(t *testing.T) {
			layout, lock, pid, done := compatibilityOwner(t)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/requests" {
					_, _ = w.Write([]byte(`{"requests":[]}`))
				}
			}))
			defer server.Close()
			publishCompatibilityOwner(t, layout, lock, pid, strings.TrimPrefix(server.URL, "http://"), schema)
			before, err := os.ReadFile(layout.Daemon)
			must(t, err)
			if output, err := compatibilityCLI(t, layout.Root, "run", "list", "--json"); err != nil {
				t.Fatalf("compatible API refused: %v %s", err, output)
			}
			after, err := os.ReadFile(layout.Daemon)
			must(t, err)
			if string(before) != string(after) {
				t.Fatal("API access changed the owner's record")
			}
			select {
			case <-done:
				t.Fatal("API access stopped the owner")
			default:
			}
		})
	}
}

func TestDaemonCLIWaitsForPublication(t *testing.T) {
	for _, partial := range []string{"", "addr=", "addr=127.0.0.1:"} {
		t.Run(partial, func(t *testing.T) {
			layout, lock, pid, done := compatibilityOwner(t)
			_, err := lock.WriteAt([]byte(partial), 0)
			must(t, err)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/requests" {
					_, _ = w.Write([]byte(`{"requests":[]}`))
				}
			}))
			defer server.Close()
			command := exec.Command(cozyBin, "run", "list", "--json")
			command.Env = childEnv(t, layout.Root)
			out := make(chan error, 1)
			must(t, command.Start())
			go func() { out <- command.Wait() }()
			t.Cleanup(func() { _ = command.Process.Kill() })
			select {
			case err := <-out:
				t.Fatalf("CLI treated publication in progress as an operator: %v", err)
			case <-time.After(200 * time.Millisecond):
			}
			publishCompatibilityOwner(t, layout, lock, pid, strings.TrimPrefix(server.URL, "http://"), "")
			select {
			case err := <-out:
				must(t, err)
			case <-time.After(3 * time.Second):
				t.Fatal("CLI did not follow the published ephemeral address")
			}
			select {
			case <-done:
				t.Fatal("CLI terminated startup")
			default:
			}
		})
	}
}

func TestDaemonHeldRecordNeverFallsBackToConfiguredPort(t *testing.T) {
	layout, lock, pid, _ := compatibilityOwner(t)
	_, err := lock.WriteAt([]byte("pid="+strconv.Itoa(pid)+"\n"), 0)
	must(t, err)
	state := daemon.Probe(config.Config{Home: layout.Root, Port: 8080})
	if !state.Up || state.Addr != "" || state.OperatorOwned {
		t.Fatalf("incomplete owner record became a configured address or operator: %+v", state)
	}
}

func TestDaemonReadinessWaitFollowsOwnerBeyondTenSeconds(t *testing.T) {
	fullRun(t, "outwaits the deleted ten-second readiness deadline")
	layout, lock, pid, ownerDone := compatibilityOwner(t)
	publishCompatibilityOwner(t, layout, lock, pid, "127.0.0.1:0", "")
	command := exec.Command(cozyBin, "run", "list", "--json")
	command.Env = childEnv(t, layout.Root)
	must(t, command.Start())
	done := make(chan error, 1)
	go func() { done <- command.Wait() }()
	t.Cleanup(func() { _ = command.Process.Kill() })
	// Exercise the previous production deadline: the live owner is still
	// initializing and must not become a failure merely because ten seconds pass.
	select {
	case err := <-done:
		t.Fatalf("live startup returned before readiness: %v", err)
	case <-time.After(11 * time.Second):
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/requests" {
			_, _ = w.Write([]byte(`{"requests":[]}`))
		}
	}))
	defer server.Close()
	publishCompatibilityOwner(t, layout, lock, pid, strings.TrimPrefix(server.URL, "http://"), "")
	select {
	case err := <-done:
		must(t, err)
	case <-time.After(3 * time.Second):
		t.Fatal("CLI did not observe delayed readiness")
	}
	select {
	case <-ownerDone:
		t.Fatal("readiness wait stopped the owner")
	default:
	}
}

func TestDaemonReadinessWaitCanBeInterruptedWithoutStoppingOwner(t *testing.T) {
	layout, lock, pid, ownerDone := compatibilityOwner(t)
	publishCompatibilityOwner(t, layout, lock, pid, "127.0.0.1:0", "")
	command := exec.Command(cozyBin, "up", "--json")
	command.Env = childEnv(t, layout.Root)
	must(t, command.Start())
	done := make(chan error, 1)
	go func() { done <- command.Wait() }()
	t.Cleanup(func() { _ = command.Process.Kill() })
	select {
	case err := <-done:
		t.Fatalf("unready owner did not keep caller waiting: %v", err)
	case <-time.After(200 * time.Millisecond):
	}
	must(t, command.Process.Signal(os.Interrupt))
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("interruption reported success before readiness")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("caller ignored interruption")
	}
	select {
	case <-ownerDone:
		t.Fatal("interrupting caller stopped the owner")
	default:
	}
	if state := daemon.Probe(config.Config{Home: layout.Root}); !state.Up || state.PID != pid {
		t.Fatalf("interruption changed owner: %+v", state)
	}
}

func TestDaemonOperatorLockRefusesBeforeAPIReadiness(t *testing.T) {
	layout, lock, pid, done := compatibilityOwner(t)
	publishCompatibilityOwner(t, layout, lock, pid, "", "")
	output, err := compatibilityCLI(t, layout.Root, "run", "list", "--json")
	if err == nil || !strings.Contains(output, "daemon.operator_owned") {
		t.Fatalf("operator handoff was not preserved: %v %s", err, output)
	}
	select {
	case <-done:
		t.Fatal("CLI stopped the operator")
	default:
	}
	if _, err := os.Stat(filepath.Join(layout.Root, "creator.sqlite")); !os.IsNotExist(err) {
		t.Fatal("operator conflict opened a records database")
	}
}

func TestConcurrentCLIStartsConvergeOnOneEphemeralDaemon(t *testing.T) {
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("port: 0\n"), 0600))
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	commands := []*exec.Cmd{
		exec.CommandContext(ctx, cozyBin, "run", "list", "--json"),
		exec.CommandContext(ctx, cozyBin, "run", "list", "--json"),
	}
	outputs := make([]bytes.Buffer, len(commands))
	for i, command := range commands {
		command.Stdout = &outputs[i]
		command.Stderr = &outputs[i]
		command.Env = childEnv(t, root)
		must(t, command.Start())
	}
	for i, command := range commands {
		if err := command.Wait(); err != nil {
			t.Errorf("concurrent child %d: %v %s", i, err, outputs[i].String())
		}
	}
	if t.Failed() {
		t.FailNow()
	}
	state := daemon.Probe(config.Config{Home: root})
	if !state.Up || state.PID <= 0 || state.Addr == "" || strings.HasSuffix(state.Addr, ":0") {
		t.Fatalf("concurrent startup did not publish one bound daemon: %+v", state)
	}
	if output, err := compatibilityCLI(t, root, "run", "list", "--json"); err != nil {
		t.Fatalf("winner did not remain ready: %v %s", err, output)
	}
	if next := daemon.Probe(config.Config{Home: root}); next.PID != state.PID {
		t.Fatal("a subsequent client replaced the startup winner")
	}
}
