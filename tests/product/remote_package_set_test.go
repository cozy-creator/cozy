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
	const release = "sha256:bc346950b7223be1aa3ec66c239c25538d374906df532c27a039e149bb2e632a"
	const environment = "sha256:6973d360c952f850ae3238e24a32dd36044973b981d20400c72af30040046fa4"
	const config = "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
	fatal(t, store.BindRemoteInvocation(request.ID, plan, release, environment, config))
	fatal(t, store.BindRemoteInvocation(request.ID, plan, release, environment, config)) // exact replay
	if problem := store.BindRemoteInvocation(request.ID, plan, release,
		"sha256:36ba4a5884bafad99ae4dc7eb92641844e69bf4dac2d44caf712f25828b0fb93", config); problem == nil || problem.ErrName() != "request_invocation_identity_changed" {
		t.Fatalf("changed worker invocation identity was not refused: %v", problem)
	}
	stored, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if stored == nil || stored.PlanID != plan || stored.PackageRevisionDigest != release ||
		stored.EnvironmentDigest != environment || stored.ConfigDigest != config {
		t.Fatalf("worker-resolved plan did not survive readback: %+v", stored)
	}
}
