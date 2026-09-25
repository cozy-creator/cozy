package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// The machine column names the selected venue even while the request is queued.
// Status and attempt facts still distinguish a local queue from remote execution.
// The recorded machine survives rental cleanup.
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
		RequestedRental string   `json:"requested_rental"`
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
		t.Fatalf("a selected queued run must show its planned machine: %+v", row)
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

	// Assignment makes the planned venue visible; it does not start an attempt.
	assigned, problem := store.PinRental("req-machine-unclaimed", "pr-machine-column", nil)
	fatal(t, problem)
	if !assigned {
		t.Fatal("the queued run refused its rental assignment")
	}
	if row := listRuns()["req-machine-unclaimed"]; row.Machine != "otter" || row.Attempts != 0 || row.Status != "queued" {
		t.Fatalf("selected venue lost its queued status: %+v", row)
	}
	// Replanning must not show the previous rental's retained historical name.
	released, problem := store.ReleaseUnattemptedRentalAssignment("req-machine-unclaimed", "pr-machine-column")
	fatal(t, problem)
	if !released {
		t.Fatal("unattempted automatic assignment was not released")
	}
	if row := listRuns()["req-machine-unclaimed"]; row.Machine != "" || row.RentalID != "" {
		t.Fatalf("released assignment still appears selected: %+v", row)
	}
	assigned, problem = store.PinRental("req-machine-unclaimed", "pr-machine-column", nil)
	fatal(t, problem)
	if !assigned {
		t.Fatal("queued run could not select its rental again")
	}
	// An explicit rental preference is still local waiting until an attempt starts.
	_, _, problem = store.Submit(records.Request{
		ID: "req-machine-explicit", IdemKey: "idem-machine-explicit", BodyDigest: "sha256:" + strings.Repeat("ef", 32),
		Package: "fake/explicit", Entrypoint: "generate", Payload: []byte("{}"),
		Rental: true, RequestedRental: "pr-machine-column",
	})
	fatal(t, problem)
	if row := listRuns()["req-machine-explicit"]; row.Machine != "otter" || row.RequestedRental != "pr-machine-column" || row.Attempts != 0 || row.Status != "queued" {
		t.Fatalf("explicit affinity lost its planned machine or queued state: %+v", row)
	}
	if code, out := runCozy(t, root, "run", "list", "--package=fake/explicit"); code != 0 ||
		!strings.Contains(out, "waiting for rental otter") {
		t.Fatalf("explicit affinity is not shown as waiting: [%d] %s", code, out)
	}
	fatal(t, store.SpawnWorker(records.WorkerProcess{
		InstanceID: "ins-machine", Package: "fake/machine", WorkerID: "remote", Devices: []string{"cpu"},
	}))
	for _, id := range []string{"req-machine-claimed", "req-machine-unclaimed"} {
		session := "session-" + id
		attempt, problem := store.Dispatch(records.Attempt{RequestID: id, SessionID: session,
			InstanceID: "ins-machine", InvocationDigest: "sha256:" + strings.Repeat("de", 32), InvocationCanonical: []byte("{}")})
		fatal(t, problem)
		fatal(t, store.OfferDispatch(id, attempt, session))
		fatal(t, store.Accepted(id, attempt, session))
		if row := listRuns()[id]; row.Machine != "otter" || row.Attempts != 1 {
			t.Fatalf("actual attempt did not gain its venue: %+v", row)
		}
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

	// An acquisition intent without a current phase is not yet a selected machine.
	// The same recorded name can also belong to an unpinned earlier rental.
	submit("req-machine-acquiring", "")
	op, replay, problem := store.BeginRentalOperation(records.RentalOperation{
		Key: "op-machine-acquiring", Hub: "http://127.0.0.1:1",
		Reason:              "managed-rental-req-machine-acquiring",
		HourlyRateUSDMicros: 100_000, ManagedRequestID: "req-machine-acquiring",
	}, func(machineName string) ([]byte, string, *exit.Error) {
		return []byte(`{"name":"` + machineName + `"}`), "sha256:" + strings.Repeat("cd", 32), nil
	})

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
	if row := listRuns()["req-machine-acquiring"]; row.Machine != "" || row.RentalID != "" {
		t.Fatalf("acquisition with no current phase must leave its venue blank "+
			"(want %q, no rental id yet): %+v", minted.Name, row)
	}

	// The id string is a machine fact: absent from default human output entirely.
	if code, out := runCozy(t, root, "run", "list"); code != 0 ||
		!strings.Contains(out, "otter") || strings.Contains(out, "pr-") {
		t.Fatalf("default human run list must show words, never a pr- id [exit %d]\n%s", code, out)
	}
}

// Older daemons already carry the pinned or preparing machine in separate facts.
// A new client must show those facts without requiring a daemon restart.
func TestQueuedRunMachineProjection(t *testing.T) {
	for _, tc := range []struct {
		name string
		life api.Lifecycle
		want string
	}{
		{"pinned", api.Lifecycle{Status: "queued", RequestedMachine: "shuutarou"}, "shuutarou"},
		{"downloading", api.Lifecycle{Status: "queued", Phase: "downloading", PhaseMachine: "shuutarou"}, "shuutarou"},
		{"new pin supersedes previous phase", api.Lifecycle{Status: "queued", RequestedMachine: "new-pod", PhaseMachine: "old-pod"}, "new-pod"},
		{"current selection", api.Lifecycle{Status: "queued", Machine: "selected", PhaseMachine: "old-pod"}, "selected"},
		{"attempt", api.Lifecycle{Status: "in_progress", Machine: "executing", RequestedMachine: "planned", PhaseMachine: "preparing"}, "executing"},
		{"completed", api.Lifecycle{Status: "completed", Machine: "historical", PhaseMachine: "preparing"}, "historical"},
		{"unassigned", api.Lifecycle{Status: "queued"}, ""},
		{"stale phase after failure", api.Lifecycle{Status: "failed", PhaseMachine: "old-pod"}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.life.DisplayMachine(); got != tc.want {
				t.Fatalf("machine = %q, want %q", got, tc.want)
			}
		})
	}
}
