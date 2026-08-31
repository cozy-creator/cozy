package producttest

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestRemoteRequestBindsOneWorkerResolvedPlan(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.db"))
	fatal(t, problem)
	t.Cleanup(store.Close)

	request, fresh, problem := store.Submit(records.Request{
		ID: "req-marco", IdemKey: "idem-marco", BodyDigest: "sha256:marco",
		Package: "cozy/marco-polo", Entrypoint: "marco", Payload: []byte(`{"value":"marco"}`),
		Worker: "rnt-marco",
	})
	fatal(t, problem)
	if !fresh || request.PlanID != "" {
		t.Fatalf("fresh remote request already selected a plan: %+v", request)
	}

	const plan = "sha256:8fbe95f587b06b979d612a17cb9a63d686d8b11db5c27728c88c33fe600cb0c8"
	fatal(t, store.BindRequestPlan(request.ID, plan))
	fatal(t, store.BindRequestPlan(request.ID, plan)) // exact replay is a no-op
	if problem := store.BindRequestPlan(request.ID,
		"sha256:36ba4a5884bafad99ae4dc7eb92641844e69bf4dac2d44caf712f25828b0fb93"); problem == nil || problem.ErrName() != "request_plan_changed" {
		t.Fatalf("changed worker plan was not refused: %v", problem)
	}
	stored, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if stored == nil || stored.PlanID != plan {
		t.Fatalf("worker-resolved plan did not survive readback: %+v", stored)
	}
}
