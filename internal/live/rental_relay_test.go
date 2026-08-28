package live

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator/internal/records"
	"github.com/cozy-creator/cozy-creator/internal/rental"
	"github.com/cozy-creator/cozy-creator/internal/secret"
)

func TestRentalRelayRefusalIsDurableAndClearsOnSuccess(t *testing.T) {
	accept := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { //cozy:allow the LIVE arm hosts a typed Tensorhub refusal/acceptance boundary
		if r.URL.Path != "/v1/private-rentals/rental-relay-one/worker-session-observations" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if !accept {
			w.WriteHeader(http.StatusConflict)
			fmt.Fprint(w, `{"error":{"code":"rental.relay_rejected","message":"worker evidence was rejected","remedy":"replace the incompatible worker"}}`)
			return
		}
		fmt.Fprint(w, `{"state":"ready","desired_revision":1,"accepted_revision":1,"converged_revision":1}`)
	}))
	defer server.Close()

	o := hostOwner(t, "rental-relay-refusal")
	defer o.close()
	row := records.Rental{
		ID: "rental-relay-one", EndpointRef: "cozy/fake/v1/run",
		AcceleratorModel: "NVIDIA H200", State: hub.RentalConverging, Hub: server.URL,
	}
	fatal(t, o.store.RecordRental(row))
	o.cfg.HubURL = server.URL
	client := hub.New(o.cfg, "cozy-live/rental-relay")
	relay := rental.RelayWorkerSession(o.store, client)
	connection := &orchestrator.WorkerConnection{
		RentalID: row.ID, Token: secret.New("rental-owner-token"),
	}
	evidence := orchestrator.RentalSessionEvidence{
		ClaimAck: []byte{1}, Snapshot: []byte{2}, ObservedState: []byte{3}, DesiredRevision: 1,
	}

	problem := relay(context.Background(), connection, evidence)
	if problem == nil || problem.Code != exit.Conflict || problem.ErrName() != "rental.relay_rejected" {
		t.Fatalf("relay refusal = %v", problem)
	}
	stored, readProblem := o.store.RentalRelayRefusal(row.ID)
	fatal(t, readProblem)
	if stored == nil || stored.Message != "worker evidence was rejected" ||
		stored.Remedy != "replace the incompatible worker" {
		t.Fatalf("stored relay refusal = %#v", stored)
	}
	if _, knownProblem := rental.Known(o.store)(row.ID); knownProblem == nil ||
		knownProblem.ErrName() != problem.ErrName() || knownProblem.Message != problem.Message {
		t.Fatalf("Known did not surface the stored verdict: %v", knownProblem)
	}

	accept = true
	fatal(t, relay(context.Background(), connection, evidence))
	stored, readProblem = o.store.RentalRelayRefusal(row.ID)
	fatal(t, readProblem)
	if stored != nil {
		t.Fatalf("accepted relay left a stale refusal: %#v", stored)
	}
	updated, readProblem := o.store.RentalRow(row.ID)
	fatal(t, readProblem)
	if updated == nil || updated.State != hub.RentalReady {
		t.Fatalf("accepted relay state = %#v", updated)
	}
}
