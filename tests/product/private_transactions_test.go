package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

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

func recordPrivateTransaction(t *testing.T, store *records.Store, label, rentalID string) records.Request {
	t.Helper()
	request := records.Request{
		ID: "req-private-" + label, IdemKey: "idem-private-" + label,
		BodyDigest: "sha256:" + strings.Repeat("a", 64),
		Package:    "local/private-proof", Entrypoint: "prepare", Kind: "job", Org: "local",
		Release: "1.0.0", PlanID: "sha256:" + strings.Repeat("b", 64),
		LocalInstallationID: "sha256:" + strings.Repeat("c", 64),
		Payload:             []byte(`{"source_revision":"frozen","seed":17,"quality_bar":0.95}`),
		Worker:              rentalID, Rental: rentalID != "", RentalRequired: rentalID != "",
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
		after.PlanID != before.PlanID || after.LocalInstallationID != before.LocalInstallationID ||
		after.InstallationID != before.InstallationID || !bytes.Equal(after.Payload, before.Payload) ||
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
