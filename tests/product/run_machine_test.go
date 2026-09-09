package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// TestRunListMachineColumn is cl-090 and cl-107 as behaviour: `cozy run list` says WHERE
// each run executes without changing the one monotonic numbering. A rental-claimed run
// carries the rental's owner-scoped machine word (cl-088), an unclaimed one stays blank,
// and the word is RECORDED on the request when a rental is bound to it — never a
// read-time join — so it survives the rental row's release, a run bound at acquisition
// start names its machine even when acquisition fails before any claim, and the raw
// `pr-…` id appears nowhere in default human output (it stays a --full/--json fact,
// `rental_id`). The local arm lives in TestProductPath, beside the runs that produce it.
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

	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
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
	for _, id := range []string{"newer-other-a", "newer-other-b", "newer-other-c"} {
		if _, _, problem := store.Submit(records.Request{
			ID: id, IdemKey: "idem-" + id, BodyDigest: "sha256:" + strings.Repeat("cd", 32),
			Package: "other/package", Entrypoint: "generate", Payload: []byte("{}"),
			Rental: true,
		}); problem != nil {
			t.Fatal(problem.Message)
		}
	}

	type listedRun struct {
		Number          int64    `json:"number"`
		ID              string   `json:"id"`
		Machine         string   `json:"machine"`
		RentalID        string   `json:"rental_id"`
		Status          string   `json:"status"`
		QueuedMS        int64    `json:"queued_ms"`
		ExecutionMS     int64    `json:"execution_ms"`
		Attempts        int      `json:"attempts"`
		ProgressStage   string   `json:"progress_stage"`
		StageFraction   *float64 `json:"stage_fraction"`
		OverallFraction *float64 `json:"overall_fraction"`
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
	if rows["req-machine-unclaimed"].Number < 1 || rows["req-machine-claimed"].Number < 1 {
		t.Fatalf("the venue must never replace the monotonic number: %+v", rows)
	}
	for _, id := range []string{"req-machine-unclaimed", "req-machine-claimed"} {
		row := rows[id]
		if row.QueuedMS < 0 || row.ExecutionMS != 0 || row.Attempts != 0 ||
			row.ProgressStage != "" || row.StageFraction != nil || row.OverallFraction != nil {
			t.Fatalf("queued run %s carries presentation text or live progress: %+v", id, row)
		}
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
	// Filtering happens in the store before LIMIT. Newer runs for another package may not
	// hide an older matching run from a deliberately small result window.
	if code, out := runCozy(t, root, "run", "list", "--package", "fake/machine", "--limit", "2", "--json", "--full"); code != 0 || strings.Contains(out, "other/package") ||
		!strings.Contains(out, "req-machine-unclaimed") || !strings.Contains(out, "req-machine-claimed") {
		t.Fatalf("package filtering did not precede the list limit [exit %d]\n%s", code, out)
	}

	// The blank→name transition: the exact durable claim the fleet records, observed by
	// the very next list. The list reads placement state as it is NOW, not as submitted.
	assigned, problem := store.PinRental("req-machine-unclaimed", "pr-machine-column", nil)
	fatal(t, problem)
	if !assigned {
		t.Fatal("the queued run refused its rental assignment")
	}
	if row := listRuns()["req-machine-unclaimed"]; row.Machine != "otter" {
		t.Fatalf("the queued run did not gain its machine when the rental claimed it: %+v", row)
	}

	// cl-107: the word is HISTORY the run keeps, not a live join. Deleting the rental
	// row — exactly what releasing a rental does — must leave every claimed run still
	// saying `otter`, with the raw id kept as the --full/--json `rental_id` fact.
	forgotten, problem := store.ForgetRental("pr-machine-column")
	fatal(t, problem)
	if !forgotten {
		t.Fatal("the planted rental row was already gone")
	}
	rows = listRuns()
	for _, id := range []string{"req-machine-claimed", "req-machine-unclaimed"} {
		if row := rows[id]; row.Machine != "otter" || row.RentalID != "pr-machine-column" {
			t.Fatalf("a released rental's word must persist on the run %s "+
				"(machine word + rental_id): %+v", id, row)
		}
	}

	// cl-107 addendum: a run BOUND at acquisition start names its machine even when
	// acquisition fails before any claim. BeginRentalOperation mints the word and
	// stamps it on the managed request in one transaction; no rental row exists yet and
	// none ever will here (the hub is unreachable) — the shape a failed acquisition
	// leaves behind, which must not render as `-`.
	submit("req-machine-acquiring", "")
	op, replay, problem := store.BeginRentalOperation(records.RentalOperation{
		Key: "op-machine-acquiring", Hub: "http://127.0.0.1:1",
		Reason:              "managed-rental-req-machine-acquiring",
		HourlyRateUSDMicros: 100_000, ManagedRequestID: "req-machine-acquiring",
	}, 10_000_000, 0, func(machineName string) ([]byte, string, *exit.Error) {
		return []byte(`{"name":"` + machineName + `"}`), "sha256:" + strings.Repeat("cd", 32), nil
	}, nil)

	fatal(t, problem)
	if replay {
		t.Fatal("a fresh rental operation replayed")
	}
	var minted struct {
		Name string `json:"name"`
	}
	must(t, json.Unmarshal(op.RequestBody, &minted))
	if minted.Name == "" {
		t.Fatalf("the rental operation minted no machine word: %s", op.RequestBody)
	}
	if row := listRuns()["req-machine-acquiring"]; row.Machine != minted.Name || row.RentalID != "" {
		t.Fatalf("a run bound at acquisition start must already name its machine "+
			"(want %q, no rental id yet): %+v", minted.Name, row)
	}

	// The id string is a machine fact: absent from default human output entirely.
	if code, out := runCozy(t, root, "run", "list"); code != 0 ||
		!strings.Contains(out, "otter") || strings.Contains(out, "pr-") {
		t.Fatalf("default human run list must show words, never a pr- id [exit %d]\n%s", code, out)
	}
}
