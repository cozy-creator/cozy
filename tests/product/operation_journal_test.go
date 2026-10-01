package producttest

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
)

// heldModel serves a real CozyTensors model to a machine's own TensorFS. While holding,
// every request for the asset after the first servedBeforeHold stalls without a body until
// its client leaves or the hold is released.
type heldModel struct {
	fixture        cancelFixture
	mu             sync.Mutex
	holding        bool
	heldDigest     string
	open           int
	gets           map[string]int
	distinct       []string
	ranges         []string
	held, left     chan struct{}
	heldOnce, gone sync.Once
	release        chan struct{}
}

func newHeldModel(t *testing.T) *heldModel {
	tfs, err := exec.LookPath("tfs")
	if err != nil {
		t.Skip("requires the native tfs CLI to encode the model")
	}
	return &heldModel{fixture: newCancelFixture(t, tfs), holding: true, gets: map[string]int{},
		held: make(chan struct{}), left: make(chan struct{}), release: make(chan struct{})}
}

func (m *heldModel) serve(h *machineHub) {
	f := m.fixture
	bodies := map[string][]byte{strings.TrimPrefix(f.manifestID, "sha256:"): f.manifest}
	objects := []any{}
	for id, body := range f.objects {
		bodies[strings.TrimPrefix(id, "sha256:")] = body
		objects = append(objects, map[string]any{"length": len(body), "sha256": strings.TrimPrefix(id, "sha256:")})
	}
	public := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models/resolve" {
			_ = json.NewEncoder(w).Encode(map[string]any{"model": "proof/model", "manifest_id": f.manifestID,
				"manifest_length": len(f.manifest), "bytes": f.totalBytes, "components": []string{"transformer"}})
			return
		}
		public.ServeHTTP(w, r)
	})
	doors := h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/tensorfs/closure":
			var asked struct{ Lane string }
			_ = json.NewDecoder(r.Body).Decode(&asked)
			_ = json.NewEncoder(w).Encode(map[string]any{"complete": true, "lane": asked.Lane, "model": "proof/model",
				"manifest":            map[string]any{"length": len(f.manifest), "sha256": strings.TrimPrefix(f.manifestID, "sha256:")},
				"objects":             objects,
				"presign_max_digests": 64, "release": "", "scope": "runtime"})
		case r.URL.Path == "/v1/tensorfs/presign":
			var asked struct{ Digests []string }
			_ = json.NewDecoder(r.Body).Decode(&asked)
			urls := map[string]string{}
			for _, digest := range asked.Digests {
				urls[digest] = h.worker.URL + "/o/" + digest
			}
			now := time.Now().Unix()
			_ = json.NewEncoder(w).Encode(map[string]any{"expires_at_unix": now + 3600, "server_time_unix": now, "urls": urls})
		case strings.HasPrefix(r.URL.Path, "/o/") && bodies[strings.TrimPrefix(r.URL.Path, "/o/")] != nil:
			m.object(w, r, strings.TrimPrefix(r.URL.Path, "/o/"), bodies[strings.TrimPrefix(r.URL.Path, "/o/")])
		default:
			doors.ServeHTTP(w, r)
		}
	})
}

func (m *heldModel) object(w http.ResponseWriter, r *http.Request, digest string, body []byte) {
	asset := "sha256:"+digest != m.fixture.headerID && "sha256:"+digest != m.fixture.manifestID
	m.mu.Lock()
	m.gets[digest]++
	if r.Header.Get("Range") != "" {
		m.ranges = append(m.ranges, r.Header.Get("Range"))
	}
	if asset && !slices.Contains(m.distinct, digest) {
		m.distinct = append(m.distinct, digest)
		if len(m.distinct) == servedBeforeHold+1 {
			m.heldDigest = digest
		}
	}
	hold := m.holding && digest == m.heldDigest
	if hold {
		m.open++
	}
	m.mu.Unlock()
	if !hold {
		_, _ = w.Write(body)
		return
	}
	// A trickle, not a stall: the transfer keeps measurable progress, so nothing but the
	// client leaving or the release ends it.
	w.Header().Set("Content-Length", fmt.Sprint(len(body)))
	w.WriteHeader(http.StatusOK)
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	for sent, done := 0, false; !done; {
		select {
		case <-r.Context().Done():
			done = true
		case <-m.release:
			_, _ = w.Write(body[sent:])
			done = true
		case <-tick.C:
			if sent+128 < len(body) {
				_, _ = w.Write(body[sent : sent+64])
				w.(http.Flusher).Flush()
				sent += 64
			}
			m.heldOnce.Do(func() { close(m.held) })
		}
	}
	m.mu.Lock()
	m.open--
	if m.open == 0 && r.Context().Err() != nil {
		m.gone.Do(func() { close(m.left) })
	}
	m.mu.Unlock()
}

type journaled struct {
	Number    int64  `json:"number"`
	Kind      string `json:"kind"`
	Target    string `json:"target"`
	Machine   string `json:"machine"`
	Status    string `json:"status"`
	ErrorCode string `json:"error_code"`
	Error     string `json:"error"`
	Changed   *bool  `json:"changed"`
	Models    []struct {
		Model string `json:"model"`
		Moved int64  `json:"moved_bytes"`
		Total int64  `json:"total_bytes"`
	} `json:"models"`
}

func journalOf(t *testing.T, root string, args ...string) (int, journaled, string) {
	t.Helper()
	code, out := runCozy(t, root, append(args, "--json", "--full")...)
	var row journaled
	if code == 0 {
		must(t, json.Unmarshal([]byte(out), &row))
	}
	return code, row, out
}

// A model download onto a rented machine is a numbered operation on the run verbs: it
// shows its bytes while it lands, it names itself to a Runtime update it blocks, its
// cancel detaches it from the machine's transfer, and the same download resumes.
// Everything below the CLI is real: daemon, agent, Runtime and TensorFS.
func TestMachineDownloadIsAJournaledOperation(t *testing.T) {
	model := newHeldModel(t)
	f := nativeMaintenance(t, model.serve)
	ref := "proof/model#" + model.fixture.manifestID
	code, accepted, out := journalOf(t, f.root, "model", "download", ref, "--rental=tessa")
	if code != 0 || accepted.Number == 0 || accepted.Kind != "download" || accepted.Machine != "tessa" {
		t.Fatalf("the download was not journaled [exit %d]: %s", code, out)
	}
	number := fmt.Sprint(accepted.Number)
	if code, again, out := journalOf(t, f.root, "model", "download", ref, "--rental=tessa"); code != 0 || again.Number != accepted.Number {
		t.Fatalf("a repeated pending download was journaled again [exit %d]: %s", code, out)
	}
	select {
	case <-model.held:
	case <-time.After(5 * time.Minute):
		t.Fatalf("the machine never reached the held object\n%s", tail(f.root+"/daemon.log"))
	}
	// The machine's plan counts every object it lands, the manifest included.
	planned := model.fixture.totalBytes + int64(len(model.fixture.manifest))
	var shown journaled
	last := ""
	eventually(t, f.root, "the download's bytes in run show", func() bool {
		time.Sleep(time.Second)
		_, shown, last = journalOf(t, f.root, "run", "show", number)
		last = fmt.Sprintf("want %d total bytes in progress; saw %s", planned, last)
		return shown.Status == "in_progress" && len(shown.Models) == 1 && shown.Models[0].Moved > 0 && shown.Models[0].Total == planned
	}, &last)
	code, out = runCozy(t, f.root, "run", "list", "--json", "--no-watch")
	if code != 0 || !strings.Contains(out, `"kind":"download"`) || !strings.Contains(out, fmt.Sprintf(`"number":%s`, number)) {
		t.Fatalf("run list does not show the download [exit %d]: %s", code, out)
	}
	code, out = runCozy(t, f.root, "rental", "update", "tessa", "--json")
	if code == 0 || !strings.Contains(out, "blocked by #"+number+" (download proof/model") || !strings.Contains(out, "cozy run cancel "+number) {
		t.Fatalf("the update did not name the download blocking it [exit %d]: %s", code, out)
	}
	code, canceled, out := journalOf(t, f.root, "run", "cancel", number)
	if code != 0 || canceled.Status != "canceled" || canceled.Changed == nil || !*canceled.Changed {
		t.Fatalf("cancel did not end the download [exit %d]: %s", code, out)
	}
	cancels := false
	eventually(t, f.root, "the machine's answer to the cancel", func() bool {
		time.Sleep(time.Second)
		_, shown, last = journalOf(t, f.root, "run", "show", number)
		cancels = shown.ErrorCode == "operation.canceled"
		return cancels || shown.ErrorCode == "machine.cancel_unavailable"
	}, &last)
	t.Logf("the machine answered the cancel: %s: %s", shown.ErrorCode, shown.Error)
	if cancels {
		select {
		case <-model.left:
		case <-time.After(time.Minute):
			t.Fatal("the machine kept transferring the canceled download's object")
		}
	} else {
		close(model.release) // a machine that predates cancellation finishes the transfer
	}
	model.mu.Lock()
	model.holding = false
	before := map[string]int{}
	for digest, n := range model.gets {
		before[digest] = n
	}
	model.mu.Unlock()
	code, done, out := journalOf(t, f.root, "model", "download", ref, "--rental=tessa", "--await")
	if code != 0 || done.Status != "completed" || done.Number <= accepted.Number || len(done.Models) > 0 && done.Models[0].Moved != planned {
		t.Fatalf("the same download did not complete as a new number [exit %d]: %s", code, out)
	}
	model.mu.Lock()
	defer model.mu.Unlock()
	for _, digest := range model.distinct[:servedBeforeHold] {
		if model.gets[digest] != before[digest] {
			t.Fatalf("an object that landed before the cancel moved again: %s (ranges %v)", digest, model.ranges)
		}
	}
}

// Run numbers are stored. Runs an older Creator recorded without one are numbered once,
// in their recorded order, so each keeps the number it was shown with; a download then
// takes the next number in the same sequence, and both list newest first.
func TestJournalKeepsRecordedRunNumbers(t *testing.T) {
	layout, store := rentalInstallStore(t)
	for i := range 3 {
		_, _, problem := store.Submit(records.Request{ID: fmt.Sprintf("req-journal-%d", i), IdemKey: fmt.Sprintf("journal-%d", i),
			BodyDigest: childDigest(fmt.Sprint(i)), Package: "proof/journal", Entrypoint: "run", Payload: []byte(`{}`)})
		fatal(t, problem)
		time.Sleep(time.Millisecond)
	}
	store.Close()
	db, err := sql.Open("sqlite", layout.DB)
	must(t, err)
	_, err = db.Exec(`DROP TABLE journal`)
	must(t, err)
	must(t, db.Close())
	reopened, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer reopened.Close()
	for i := range 3 {
		row, problem := reopened.RequestByReference(fmt.Sprint(i + 1))
		fatal(t, problem)
		if row == nil || row.ID != fmt.Sprintf("req-journal-%d", i) || row.Journal != "run" {
			t.Fatalf("run %d lost its number: %+v", i+1, row)
		}
	}
	download, fresh, problem := reopened.BeginInstall(machines.Local, "", "key", records.InstallSelection{Models: []records.ModelRef{{
		Model: "proof/model", Release: "1", Manifest: childDigest("d")}}})
	fatal(t, problem)
	again, _, problem := reopened.BeginInstall(machines.Local, "", "key", download.Install)
	fatal(t, problem)
	if !fresh || download.Number != 4 || download.Kind != "download" || again.ID != download.ID {
		t.Fatalf("the download did not take the next number once: %+v %+v", download, again)
	}
	if _, _, problem := reopened.BeginInstall(machines.Local, "", "key", records.InstallSelection{Package: "proof/journal", Release: "1"}); problem == nil {
		t.Fatal("one idempotency key named two operations")
	}
	runs, problem := reopened.RequestsBefore("", "", 10, 0)
	fatal(t, problem)
	operations, problem := reopened.OperationsBefore("", "", "", "", 10, 0)
	fatal(t, problem)
	if len(runs) != 3 || runs[0].Number != 3 || len(operations) != 1 || operations[0].Number != 4 {
		t.Fatalf("history pages are not one newest-first sequence: %+v %+v", runs, operations)
	}
	changed, problem := reopened.CancelOperation(download.ID, "cozy run cancel")
	fatal(t, problem)
	canceled, problem := reopened.Operation("4")
	fatal(t, problem)
	if !changed || canceled.Status() != "canceled" || canceled.CanceledBy != "cozy run cancel" {
		t.Fatalf("a queued download was not withdrawn: %+v", canceled)
	}
}
