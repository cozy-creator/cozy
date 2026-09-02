package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// TestRunListMachineColumn is cl-090 as behaviour: `cozy run list` says WHERE each run
// executes without changing the one monotonic numbering. A rental-claimed run carries the
// rental's owner-scoped machine word (cl-088), an unclaimed one stays blank, and the fact
// is read from current placement state — the same `requests.worker` row the fleet's
// AssignManagedRental writes — so a queued run gains its machine the moment a rental
// claims it. The local arm lives in TestProductPath, beside the runs that produce it.
func TestRunListMachineColumn(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "machine-column")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	// An unreachable hub on purpose: naming a venue is a local read, never a paid call.
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: http://127.0.0.1:1\n"+
			"tensorhub_token: machine-column-test\n"+
			"daemon:\n  idle_shutdown_s: 0\n"), 0o600))

	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.RecordRental(records.Rental{
		ID: "pr-machine-column", MachineName: "otter", SKU: "cpu", AcceleratorModel: "CPU",
		HourlyRateUSDMicros: 100_000, State: "ready", Hub: "http://127.0.0.1:1",
		Address: "127.0.0.1:1", CertPath: filepath.Join(root, "pr-machine-column.pem"),
	}))
	startDaemonProcess(t, root)
	// Planted after startup, so the startup requeue does not race the daemon's own
	// acquisition attempt against the deliberately unreachable hub: both runs WAIT,
	// which is the state whose venue this test reads.
	submit := func(id, worker string) {
		t.Helper()
		if _, _, problem := store.Submit(records.Request{
			ID: id, IdemKey: "idem-" + id, BodyDigest: "sha256:" + strings.Repeat("ab", 32),
			Package: "fake/machine", Entrypoint: "generate", Payload: []byte("{}"),
			Rental: true, Worker: worker,
		}); problem != nil {
			t.Fatal(problem.Message)
		}
	}
	submit("req-machine-unclaimed", "")
	submit("req-machine-claimed", "pr-machine-column")

	type listedRun struct {
		Number  string `json:"number"`
		ID      string `json:"id"`
		Machine string `json:"machine"`
		Status  string `json:"status"`
	}
	listRuns := func() map[string]listedRun {
		t.Helper()
		code, out := runCozy(t, root, "run", "list", "--json", "--full")
		var document struct {
			Invocations []listedRun `json:"invocations"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &document) != nil {
			t.Fatalf("run list failed [exit %d]\n%s", code, out)
		}
		rows := make(map[string]listedRun, len(document.Invocations))
		for _, row := range document.Invocations {
			rows[row.ID] = row
		}
		return rows
	}
	rows := listRuns()
	if row := rows["req-machine-unclaimed"]; row.Status != "queued" || row.Machine != "" {
		t.Fatalf("an unclaimed --rental run must wait with a BLANK machine: %+v", row)
	}
	if row := rows["req-machine-claimed"]; row.Status != "queued" || row.Machine != "otter" {
		t.Fatalf("a rental-claimed run must carry the rental's machine word: %+v", row)
	}
	if rows["req-machine-unclaimed"].Number == "" || rows["req-machine-claimed"].Number == "" {
		t.Fatalf("the venue must never replace the monotonic number: %+v", rows)
	}
	// The default JSON document carries the field too — machine is a first-class column,
	// not a --full extra.
	if code, out := runCozy(t, root, "run", "list", "--json"); code != 0 ||
		!strings.Contains(out, `"machine":"otter"`) || !strings.Contains(out, `"machine":""`) {
		t.Fatalf("default run list JSON lost the machine field [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "list"); code != 0 ||
		!strings.Contains(out, "MACHINE") || !strings.Contains(out, "otter") {
		t.Fatalf("human run list does not show the MACHINE column [exit %d]\n%s", code, out)
	}
	// A pipe is always one snapshot: automation never inherits an endless refresh loop.
	// Explicit watch likewise refuses without a terminal, and JSON is always snapshot-shaped.
	if code, out := runCozy(t, root, "run", "list", "--watch"); code == 0 ||
		!strings.Contains(out, "--watch requires interactive terminal output") {
		t.Fatalf("piped watch did not refuse clearly [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "list", "--watch", "--json"); code == 0 ||
		!strings.Contains(out, "--watch requires interactive terminal output") {
		t.Fatalf("JSON watch did not refuse clearly [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "list", "--no-watch"); code != 0 ||
		!strings.Contains(out, "MACHINE") {
		t.Fatalf("explicit snapshot failed [exit %d]\n%s", code, out)
	}

	// The blank→name transition: the exact durable claim the fleet records, observed by
	// the very next list. The list reads placement state as it is NOW, not as submitted.
	assigned, problem := store.AssignManagedRental("req-machine-unclaimed", "pr-machine-column")
	fatal(t, problem)
	if !assigned {
		t.Fatal("the queued run refused its rental assignment")
	}
	if row := listRuns()["req-machine-unclaimed"]; row.Machine != "otter" {
		t.Fatalf("the queued run did not gain its machine when the rental claimed it: %+v", row)
	}
}
