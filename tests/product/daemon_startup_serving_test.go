package producttest

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/flock"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/userunit"
)

// silentHub accepts connections and never answers them: a Tensorhub that is up but stuck.
func silentHub(t *testing.T) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	must(t, err)
	held := make(chan net.Conn, 256)
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			select {
			case held <- conn:
			default:
				conn.Close()
			}
		}
	}()
	t.Cleanup(func() {
		listener.Close()
		for len(held) > 0 {
			(<-held).Close()
		}
	})
	return "http://" + listener.Addr().String()
}

// holdRecords is another live writer on the records: it holds SQLite's write lock until
// the returned function is called.
func holdRecords(t *testing.T, root string) func() {
	t.Helper()
	db, err := sql.Open("sqlite", filepath.Join(root, "creator.sqlite"))
	must(t, err)
	conn, err := db.Conn(context.Background())
	must(t, err)
	_, err = conn.ExecContext(context.Background(), "BEGIN IMMEDIATE")
	must(t, err)
	released := false
	release := func() {
		if !released {
			released = true
			_, _ = conn.ExecContext(context.Background(), "ROLLBACK")
			conn.Close()
			db.Close()
		}
	}
	t.Cleanup(release)
	return release
}

func daemonLogHas(root, text string) bool {
	raw, _ := os.ReadFile(filepath.Join(root, "daemon.log"))
	return strings.Contains(string(raw), text)
}

// The owner's restart, 2026-10-10: the new daemon bound 127.0.0.1:8818 and then sat for
// minutes in its startup work (blocked jobs, a pending output export, failed rentals, a Hub
// that answered nothing) while every client's connection queued in the kernel unanswered.
// Startup work waits here on another writer holding the records, the way it waits on any
// slow step; the API must answer regardless, and the work must finish once it can.
func TestADaemonServesWhileItsStartupWorkWaits(t *testing.T) {
	root := t.TempDir()
	t.Cleanup(func() { _, _ = runCozy(t, root, "down") })
	hub := silentHub(t)
	must(t, os.WriteFile(filepath.Join(root, config.FileName),
		[]byte("tensorhub_url: "+hub+"\ntensorhub_token: startup-test\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	for i := range 3 {
		id, rental := fmt.Sprintf("job-blocked-%d", i), fmt.Sprintf("pr-failed-%d", i)
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, Package: "proof/blocked", Entrypoint: "main",
			Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true, Hub: hub})
		fatal(t, problem)
		fatal(t, store.LinkMachineExecution(id, rental))
		fatal(t, store.RecordRental(records.Rental{ID: rental, MachineName: rental, SKU: "cpu", AcceleratorModel: "CPU",
			AcceleratorCount: 1, HourlyRateUSDMicros: 1, State: "failed", Hub: hub,
			Failure: records.RentalFailure{Code: "supervisor_never_started"}}))
	}
	_, _, problem = store.Submit(records.Request{ID: "req-export", IdemKey: "req-export", Package: "proof/export",
		Entrypoint: "main", Payload: []byte(`{}`), BodyDigest: childDigest("2"),
		OutputExport: &records.OutputExportIntent{Directory: filepath.Join(root, "out"),
			Outputs: []records.OutputExportEntry{{OutputID: "image", MediaType: "image/webp"}}}})
	fatal(t, problem)
	store.Close()
	db, err := sql.Open("sqlite", filepath.Join(root, "creator.sqlite"))
	must(t, err)
	_, err = db.Exec(`UPDATE requests SET state='blocked' WHERE id LIKE 'job-blocked-%'`)
	must(t, err)
	_, err = db.Exec(`UPDATE requests SET state='succeeded' WHERE id='req-export'`)
	must(t, err)
	must(t, db.Close())

	release := holdRecords(t, root)
	began := time.Now()
	code, out := cozyWithin(t, root, 90*time.Second, "run", "list", "--no-watch", "--json")
	if code != 0 || !strings.Contains(out, "proof/blocked/main") {
		t.Fatalf("cozy run list while startup work waits [exit %d]: %s", code, out)
	}
	t.Logf("cozy run list answered in %s while startup work waited", time.Since(began).Round(time.Millisecond))
	if daemonLogHas(root, "startup recovery:") {
		t.Fatalf("recovery finished while another writer held the records:\n%s", tail(filepath.Join(root, "daemon.log")))
	}

	release()
	eventually(t, root, "startup work finishing once the records are free", func() bool {
		return daemonLogHas(root, "startup tmp sweep:")
	})
	raw, _ := os.ReadFile(filepath.Join(root, "daemon.log"))
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.Contains(line, " startup ") {
			t.Log(line)
		}
	}
	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	export, problem := store.OutputExportOf("req-export")
	fatal(t, problem)
	if export == nil || export.State == "pending" {
		t.Fatalf("the pending output export was not settled by startup recovery: %+v\n%s", export, tail(filepath.Join(root, "daemon.log")))
	}
	// Plain down with startup work done: the daemon drains and goes at once.
	if code, out := cozyWithin(t, root, time.Minute, "down", "--json"); code != 0 {
		t.Fatalf("cozy down [exit %d]: %s", code, out)
	}
}

// Opening the records can wait too — a schema step on another writer's lock. The daemon
// answers from the moment it holds the root: a typed "starting" refusal, never a
// connection the kernel accepted and nobody serves.
func TestADaemonAnswersWhileItOpensItsRecords(t *testing.T) {
	root := t.TempDir()
	t.Cleanup(func() { _, _ = runCozy(t, root, "down") })
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	store.Close()
	db, err := sql.Open("sqlite", filepath.Join(root, "creator.sqlite"))
	must(t, err)
	_, err = db.Exec(`DROP INDEX requests_state`) // the daemon's open recreates it: a write
	must(t, err)
	must(t, db.Close())
	release := holdRecords(t, root)

	up := make(chan string, 1)
	command := exec.Command(cozyBin, "up", "--json")
	command.Env = childEnv(t, root)
	go func() {
		out, _ := command.CombinedOutput()
		up <- fmt.Sprintf("[exit %d] %s", command.ProcessState.ExitCode(), out)
	}()
	var addr string
	eventually(t, root, "the daemon publishing its address", func() bool {
		state := daemon.Probe(config.Config{Home: root})
		addr = state.Addr
		return state.Up && addr != ""
	})
	response, err := (&http.Client{Timeout: 30 * time.Second}).Get("http://" + addr + "/v1/requests")
	if err != nil {
		t.Fatalf("the daemon holds %s but does not answer it: %v", addr, err)
	}
	body, _ := io.ReadAll(response.Body)
	response.Body.Close()
	var refusal struct {
		Error struct{ Code string } `json:"error"`
	}
	_ = json.Unmarshal(body, &refusal)
	if response.StatusCode != http.StatusServiceUnavailable || refusal.Error.Code != "daemon_starting" {
		t.Fatalf("while opening its records the daemon answered %d %s", response.StatusCode, body)
	}

	release()
	select {
	case result := <-up:
		if !strings.Contains(result, "[exit 0]") {
			t.Fatalf("cozy up once the records were free: %s", result)
		}
	case <-time.After(2 * time.Minute):
		t.Fatalf("cozy up never finished once the records were free\n%s", tail(filepath.Join(root, "daemon.log")))
	}
	if code, out := cozyWithin(t, root, 90*time.Second, "run", "list", "--no-watch", "--json"); code != 0 {
		t.Fatalf("cozy run list after startup [exit %d]: %s", code, out)
	}
}

// `cozy down` waits for the daemon it stopped, not for the root to be empty: a client
// retrying through the restart may start the next daemon first, and that one is not stopping.
func TestDownFinishesWhenTheNextDaemonAlreadyHoldsTheRoot(t *testing.T) {
	root := t.TempDir()
	if code, out := runCozy(t, root, "up", "--json"); code != 0 {
		t.Fatalf("cozy up [exit %d]: %s", code, out)
	}
	// The next owner, waiting on the root's claim the moment the first lets go.
	claim, err := os.OpenFile(filepath.Join(root, "daemon.lock"), os.O_RDWR, 0o600)
	must(t, err)
	defer claim.Close()
	successor := make(chan struct{})
	go func() {
		defer close(successor)
		if flock.Block(claim) == nil {
			_, _ = claim.WriteAt([]byte(fmt.Sprintf("addr=127.0.0.1:1\nsocket=\npid=%d\nsince=%s\n",
				os.Getpid(), time.Now().UTC().Format(time.RFC3339))), 0)
		}
	}()
	began := time.Now()
	code, out := cozyWithin(t, root, 2*time.Minute, "down", "--json")
	if code != 0 || !strings.Contains(out, `"stopped"`) || time.Since(began) > 30*time.Second {
		t.Fatalf("cozy down with the next daemon already up took %s [exit %d]: %s", time.Since(began), code, out)
	}
	<-successor
	_ = flock.Release(claim)
}

// A daemon's last words — here the goroutine dump SIGQUIT prints — outlive the next start:
// daemon-startup.log appends after a start marker instead of being truncated by it.
func TestADaemonsDumpOutlivesTheNextStart(t *testing.T) {
	if !userunit.Available() {
		t.Skip("requires a systemd user manager")
	}
	root := t.TempDir()
	unit := userunit.Name("cozy-daemon", root, false)
	t.Cleanup(func() { _ = userunit.Stop(unit) })
	if code, out := runCozy(t, root, "up", "--json"); code != 0 {
		t.Fatalf("cozy up [exit %d]: %s", code, out)
	}
	pid := userunit.MainPID(unit)
	if pid == 0 {
		t.Fatalf("the daemon is not running as unit %s", unit)
	}
	must(t, syscall.Kill(pid, syscall.SIGQUIT))
	eventually(t, root, "the dumped daemon ending", func() bool { return !userunit.Running(unit) })
	if code, out := runCozy(t, root, "up", "--json"); code != 0 {
		t.Fatalf("cozy up after the dump [exit %d]: %s", code, out)
	}
	raw, err := os.ReadFile(filepath.Join(root, "daemon-startup.log"))
	must(t, err)
	log := string(raw)
	first, dump := strings.Index(log, "--- cozy-daemon start "), strings.Index(log, "SIGQUIT")
	second := strings.LastIndex(log, "--- cozy-daemon start ")
	if first < 0 || dump < first || second < dump {
		t.Fatalf("daemon-startup.log lost the previous daemon's dump:\n%s", log)
	}
	_, _ = runCozy(t, root, "down")
}
