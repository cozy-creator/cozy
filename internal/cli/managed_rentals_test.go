package cli

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
)

const testWheelhouse = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func TestManagedRentalIdlePolicySurvivesRestartAndKeepsJobsEphemeral(t *testing.T) {
	var mu sync.Mutex
	released := map[string]int{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.PathValue("id")
		if id == "" {
			id = r.URL.Path[len("/v1/rentals/"):]
		}
		mu.Lock()
		deletes := released[id]
		if r.Method == http.MethodDelete {
			released[id]++
		}
		mu.Unlock()
		switch r.Method {
		case http.MethodDelete:
			w.WriteHeader(http.StatusNoContent)
		case http.MethodGet:
			state := "ready"
			if deletes > 0 {
				state = "released"
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"rental_id": id, "state": state,
				"requested_accelerator_model": "CPU", "hourly_rate_usd_micros": 70_000,
				"wheelhouse_manifest_digest": testWheelhouse,
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	root := t.TempDir()
	layout, problem := home.Open(root)
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(filepath.Join(root, "records.db"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	var output bytes.Buffer
	ctx := &Context{Inv: &Invocation{}, Out: &output, Err: &output, Cfg: config.Config{
		Home: root, HubURL: server.URL, HubToken: secret.New("test-token"),
		RentalsMaxHourlySpendUSDMicros: 1_000_000,
	}}

	recordSettledRental(t, store, server.URL, "pr-serving", "serving", true)
	restarted := &managedRentals{
		ctx: ctx, layout: layout, store: store, idleGrace: 25 * time.Millisecond,
	}
	restarted.releaseOrphaned()
	mu.Lock()
	gotDeletes := released["pr-serving"]
	mu.Unlock()
	if gotDeletes != 0 {
		t.Fatalf("serving rental released before its grace: deletes=%d", gotDeletes)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		gotDeletes = released["pr-serving"]
		mu.Unlock()
		if gotDeletes == 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	row, problem := store.RentalRow("pr-serving")
	if problem != nil || gotDeletes != 1 || row != nil {
		t.Fatalf("serving expiry: deletes=%d row=%+v problem=%v output=%s",
			gotDeletes, row, problem, output.String())
	}

	recordSettledRental(t, store, server.URL, "pr-job", "job", false)
	jobFleet := &managedRentals{ctx: ctx, layout: layout, store: store, idleGrace: time.Hour}
	if _, problem := jobFleet.release("pr-job"); problem != nil {
		t.Fatal(problem)
	}
	mu.Lock()
	gotDeletes = released["pr-job"]
	mu.Unlock()
	row, problem = store.RentalRow("pr-job")
	if problem != nil || gotDeletes != 1 || row != nil {
		t.Fatalf("job release: deletes=%d row=%+v problem=%v", gotDeletes, row, problem)
	}
}

func recordSettledRental(t *testing.T, store *records.Store, hub, id, kind string, terminal bool) {
	t.Helper()
	requestID := "req-" + id
	if problem := store.RecordRental(records.Rental{
		ID: id, MachineName: id[3:], SKU: "cpu", AcceleratorModel: "CPU",
		HourlyRateUSDMicros: 70_000, ManagedRequestID: requestID,
		State: "ready", Hub: hub, WheelhouseManifestDigest: testWheelhouse,
	}); problem != nil {
		t.Fatal(problem)
	}
	if problem := store.AttachWorker(records.WorkerProcess{
		InstanceID: id, Package: "proof/package", WorkerID: "worker",
	}); problem != nil {
		t.Fatal(problem)
	}
	request := records.Request{
		ID: requestID, IdemKey: requestID, BodyDigest: requestID, Kind: kind,
		Package: "proof/package", Entrypoint: "marco", Payload: []byte(`{}`),
		Worker: id, Rental: true,
	}
	if _, fresh, problem := store.Submit(request); problem != nil || !fresh {
		t.Fatalf("submit %s: fresh=%v problem=%v", requestID, fresh, problem)
	}
	if !terminal {
		if problem := store.SettleRequest(requestID, "succeeded"); problem != nil {
			t.Fatal(problem)
		}
		return
	}
	attempt, problem := store.Dispatch(records.Attempt{
		RequestID: requestID, InstanceID: id, SessionID: "boot",
		InvocationDigest: requestID, InvocationCanonical: []byte(`{}`),
	})
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := store.OfferDispatch(requestID, attempt, "boot"); problem != nil {
		t.Fatal(problem)
	}
	applied, problem := store.AcceptTerminal(records.Terminal{
		RequestID: requestID, Attempt: attempt, SessionID: "boot",
		InvocationDigest: requestID, TerminalID: "outcome", TerminalDigest: requestID,
		Status: "SUCCEEDED", RequestState: "succeeded",
	})
	if problem != nil || !applied {
		t.Fatalf("terminal %s: applied=%v problem=%v", requestID, applied, problem)
	}
}
