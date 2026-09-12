package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// Lifecycle commands operate on an existing request. A miss must reach the
// daemon's typed job lookup, without treating pause/resume as package names or
// creating a replacement transaction.
func TestUnpublishedTransactionCommandsResolveExistingRequest(t *testing.T) {
	root := t.TempDir()
	daemon := startDaemonProcess(t, root)
	for _, action := range []string{"pause", "resume"} {
		t.Run(action, func(t *testing.T) {
			response := daemon.call(t, http.MethodPost,
				"/v1/local/jobs/req-private-transaction-absent/"+action,
				map[string]any{"actor": "product proof"})
			if response.Status != http.StatusNotFound || response.code() != "not_found" {
				t.Errorf("%s must resolve an existing transaction: %s", action, response.brief())
			}
			code, stdout, stderr := runCozyStreams(t, root, "run", action,
				"req-private-transaction-absent", "--json")
			var document struct {
				Error struct {
					Code string `json:"code"`
				} `json:"error"`
			}
			if err := json.Unmarshal([]byte(stdout), &document); err != nil ||
				code == 0 || document.Error.Code != "not_found" {
				t.Errorf("run %s must report the owner's missing transaction [exit %d]: stdout=%s stderr=%s",
					action, code, stdout, stderr)
			}
		})
	}
	if requests := listInvocations(t, root); len(requests) != 0 {
		t.Fatalf("lifecycle commands created requests: %+v", requests)
	}
}

// A paused request remains an unfinished obligation after a hard daemon exit.
// This arm starts before dispatch deliberately: pausing a queued transaction
// must not require starting its Python program just to stop it again.
func TestUnpublishedTransactionQueuedPauseSurvivesDaemonCrash(t *testing.T) {
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName),
		[]byte("daemon:\n  idle_shutdown_s: 0\n"), 0o600))
	daemon := startDaemonProcess(t, root)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	before := recordPrivateTransaction(t, store, "queued", "")
	reference := strconv.FormatInt(before.Number, 10)
	unauthorized := daemon.call(t, http.MethodPost, "/v1/local/jobs/"+before.ID+"/pause",
		map[string]any{"actor": "untrusted caller"}, "Authorization", "")
	if unauthorized.Status != http.StatusUnauthorized {
		t.Fatalf("unauthenticated pause crossed the daemon boundary: %s", unauthorized.brief())
	}
	assertPrivateTransactionIdentity(t, store, before, before.State)
	for range 2 {
		code, out := runCozy(t, root, "run", "pause", reference, "--json")
		if code != 0 || !strings.Contains(out, `"status":"paused"`) {
			t.Fatalf("queued pause [exit %d]: %s", code, out)
		}
	}
	assertPrivateTransactionIdentity(t, store, before, "paused")
	if attempts, problem := store.Attempts(before.ID); problem != nil || len(attempts) != 0 {
		t.Fatalf("queued pause invented computation: %+v %v", attempts, problem)
	}
	if code, out := runCozy(t, root, "run", "watch", reference, "--json"); code != 0 || !strings.Contains(out, `"status":"paused"`) {
		t.Fatalf("watch did not acknowledge retained pause [exit %d]: %s", code, out)
	}

	daemon = crashAndRestartTransactionDaemon(t, daemon)
	state := daemon.call(t, http.MethodGet, "/v1/local/jobs/"+before.ID, nil)
	if state.Status != http.StatusOK || !bytes.Contains(state.Body, []byte(`"status":"paused"`)) {
		t.Fatalf("restarted owner lost the paused state: %s", state.brief())
	}
	assertPrivateTransactionIdentity(t, store, before, "paused")
	owed, problem := store.Owed()
	fatal(t, problem)
	for _, request := range owed {
		if request.ID == before.ID {
			t.Fatal("restarted owner queues paused work without explicit resume")
		}
	}

	// Cancel means permanent abandonment, distinct from pause. Resuming the same
	// number or global ID must refuse and leave the original identity in history.
	if code, out := runCozy(t, root, "run", "cancel", reference, "--json"); code != 0 {
		t.Fatalf("cancel paused transaction [exit %d]: %s", code, out)
	}
	waitUntil(t, "paused cancellation settles", func() bool {
		row, problem := store.RequestRow(before.ID)
		fatal(t, problem)
		return row.State == "canceled"
	})
	for _, ref := range []string{reference, before.ID} {
		response := daemon.call(t, http.MethodPost, "/v1/local/jobs/"+ref+"/resume",
			map[string]any{"actor": "product proof"})
		if response.Status != http.StatusConflict {
			t.Fatalf("permanently canceled transaction resumed: %s", response.brief())
		}
	}
	assertPrivateTransactionIdentity(t, store, before, "canceled")
	if requests := listInvocations(t, root); len(requests) != 1 || requests[0].ID != before.ID {
		t.Fatalf("pause/restart/cancel changed request identity: %+v", requests)
	}
}

// The rental holding transaction state is owned by every retained request, not
// only by its original buyer. A separate idle rental proves the real sweep has
// run; elapsed time alone is not evidence that the retained rental was examined.
func TestUnpublishedTransactionsShareRentalRetention(t *testing.T) {
	root := t.TempDir()
	hub := newFakeRentalHub(t, 0)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hub.server.URL+"\ntensorhub_token: rental-idle-test\n"+
			"rentals:\n  idle_release_s: 1\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	daemon := startDaemonProcess(t, root)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const retained = "pr-transaction-retained"
	first := recordPrivateTransaction(t, store, "first", retained)
	second := recordPrivateTransaction(t, store, "second", retained)
	response := daemon.call(t, http.MethodPost, "/v1/local/jobs/"+first.ID+"/pause",
		map[string]any{"actor": "product proof"})
	if response.Status != http.StatusOK || !bytes.Contains(response.Body, []byte(`"status":"paused"`)) {
		t.Fatalf("pause retained request: %s", response.brief())
	}
	blocked, problem := store.BlockRetainedWork(second.ID, "author_exception", "step B failed")
	fatal(t, problem)
	if !blocked {
		t.Fatal("failed transaction did not retain its state")
	}
	plant := func(id, machine, buyer string) {
		t.Helper()
		hub.add(id, machine)
		fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
			ID: id, MachineName: machine, SKU: "cpu", AcceleratorModel: "CPU",
			HourlyRateUSDMicros: 100_000, State: "ready", Hub: hub.server.URL,
			Address: "127.0.0.1:1", CertPath: filepath.Join(root, id+".pem"),
			ManagedRequestID: buyer,
		}))
	}
	plant(retained, "otter", first.ID)

	assertHeldAfterSweep := func(witness, machine string) {
		t.Helper()
		plant(witness, machine, "")
		awaitRentalGone(t, store, witness, 15*time.Second, filepath.Join(root, "daemon.log"))
		row, problem := store.RentalRow(retained)
		fatal(t, problem)
		if row == nil || row.State != "ready" || hub.releases(retained) != 0 {
			t.Fatalf("rental with retained transaction released: row=%+v deletes=%d", row, hub.releases(retained))
		}
	}
	assertHeldAfterSweep("pr-transaction-witness-one", "heron")
	daemon = crashAndRestartTransactionDaemon(t, daemon)
	assertHeldAfterSweep("pr-transaction-witness-two", "curlew")
	assertPrivateTransactionIdentity(t, store, first, "paused")
	assertPrivateTransactionIdentity(t, store, second, "blocked")

	// Cancel the original buyer. The second request is now the only reason to
	// retain the machine; consulting ManagedRequestID alone would delete it.
	response = daemon.call(t, http.MethodPost, "/v1/local/jobs/"+first.ID+"/cancel", nil)
	if response.Status != http.StatusOK && response.Status != http.StatusAccepted {
		t.Fatalf("cancel original buyer: %s", response.brief())
	}
	assertHeldAfterSweep("pr-transaction-witness-three", "kestrel")
	assertPrivateTransactionIdentity(t, store, second, "blocked")
	response = daemon.call(t, http.MethodPost, "/v1/local/jobs/"+second.ID+"/cancel", nil)
	if response.Status != http.StatusOK && response.Status != http.StatusAccepted {
		t.Fatalf("cancel final owner: %s", response.brief())
	}
	awaitRentalGone(t, store, retained, 15*time.Second, filepath.Join(root, "daemon.log"))
	if hub.releases(retained) != 1 {
		t.Fatalf("final owner abandonment produced %d rental releases", hub.releases(retained))
	}
}

// Disabling automatic idle cleanup must not suppress an explicit transaction
// abandonment. This failed against the real idle_release_s=0 development home.
func TestPrivateCancellationReleasesManagedRentalWithIdleCleanupDisabled(t *testing.T) {
	root := t.TempDir()
	hub := newFakeRentalHub(t, 0)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hub.server.URL+"\ntensorhub_token: retained-cancel-proof\n"+
			"rentals:\n  idle_release_s: 0\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	daemon := startDaemonProcess(t, root)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const managed, manual = "pr-explicit-cancel-managed", "pr-explicit-cancel-manual"
	request := recordPrivateTransaction(t, store, "explicit-cancel", managed)
	for id, buyer := range map[string]string{managed: request.ID, manual: ""} {
		machine := "otter"
		if id == manual {
			machine = "heron"
		}
		hub.add(id, machine)
		fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1, ID: id, MachineName: machine,
			State: "ready", SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000,
			Hub: hub.server.URL, Address: "127.0.0.1:1", CertPath: filepath.Join(root, id+".pem"),
			ManagedRequestID: buyer}))
	}
	response := daemon.call(t, http.MethodPost, "/v1/local/jobs/"+request.ID+"/cancel", nil)
	if response.Status != http.StatusOK && response.Status != http.StatusAccepted {
		t.Fatalf("explicit abandonment failed: %s", response.brief())
	}
	awaitRentalGone(t, store, managed, 10*time.Second, filepath.Join(root, "daemon.log"))
	if hub.releases(managed) != 1 {
		t.Fatal("explicit final-owner abandonment did not release exactly one managed rental")
	}
	row, problem := store.RentalRow(manual)
	fatal(t, problem)
	if row == nil || row.State != "ready" || hub.releases(manual) != 0 {
		t.Fatal("explicit transaction abandonment released the independent manual reservation")
	}
}

func recordPrivateTransaction(t *testing.T, store *records.Store, label, rentalID string) records.Request {
	t.Helper()
	request := records.Request{
		ID: "req-private-" + label, IdemKey: "idem-private-" + label,
		BodyDigest: "sha256:" + strings.Repeat("a", 64),
		Package:    "local/private-proof", Entrypoint: "prepare", Kind: "job", Org: "local",
		Release: "1.0.0", PlanID: "sha256:" + strings.Repeat("b", 64),
		LocalPackageDigest: "sha256:" + strings.Repeat("c", 64),
		Payload:            []byte(`{"source_revision":"frozen","seed":17,"quality_bar":0.95}`),
		Worker:             rentalID, Rental: rentalID != "", RentalRequired: rentalID != "",
		RetainWork: true,
	}
	_, fresh, problem := store.Submit(request)
	fatal(t, problem)
	if !fresh {
		t.Fatal("private transaction fixture was not newly recorded")
	}
	row, problem := store.RequestByReference(request.ID)
	fatal(t, problem)
	return *row
}

func assertPrivateTransactionIdentity(t *testing.T, store *records.Store, before records.Request, state string) {
	t.Helper()
	after, problem := store.RequestByReference(before.ID)
	fatal(t, problem)
	if after == nil || after.State != state || after.ID != before.ID || after.Number != before.Number ||
		after.IdemKey != before.IdemKey || after.BodyDigest != before.BodyDigest || after.CreatedAt != before.CreatedAt ||
		after.Package != before.Package || after.Entrypoint != before.Entrypoint || after.Release != before.Release ||
		after.PlanID != before.PlanID || after.LocalPackageDigest != before.LocalPackageDigest ||
		after.EnvironmentDigest != before.EnvironmentDigest || !bytes.Equal(after.Payload, before.Payload) ||
		!after.RetainWork || after.Worker != before.Worker || after.Ordinal != before.Ordinal {
		t.Fatalf("transaction identity or state changed: before=%+v after=%+v want-state=%s", before, after, state)
	}
}

func crashAndRestartTransactionDaemon(t *testing.T, first *daemonProcess) *daemonProcess {
	t.Helper()
	must(t, first.cmd.Process.Kill())
	select {
	case <-first.exited:
	case <-time.After(10 * time.Second):
		t.Fatal("owned daemon process did not exit after SIGKILL")
	}
	second := startDaemonProcess(t, first.root)
	// A hard crash leaves the old lock. Observe the new process replace it;
	// deleting that lock here would hide the recovery behavior under test.
	waitUntil(t, "restarted daemon publishes its new credential", func() bool {
		data, err := os.ReadFile(filepath.Join(first.root, "daemon.lock"))
		if err != nil {
			return false
		}
		var address, token string
		for _, line := range strings.Split(string(data), "\n") {
			if value, ok := strings.CutPrefix(line, "addr="); ok {
				address = strings.TrimSpace(value)
			}
			if value, ok := strings.CutPrefix(line, "token="); ok {
				token = strings.TrimSpace(value)
			}
		}
		if token == "" || token == first.token || address == "" {
			return false
		}
		second.addr, second.token = address, token
		return true
	})
	return second
}
