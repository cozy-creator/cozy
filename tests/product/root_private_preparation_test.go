package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// A root private serving request has no binding plan until Host prepares its
// captured code. This exercises the actual controller and wire protocol; the
// independent Host peer supplies prepared bytes, not Python execution evidence.
func TestRootPrivateModelFreeServingObtainsPlanFromPreparation(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	revision := stageLocalRevision(t, root)
	o := hostOwner(t, "root-private-no-plan", rentalWiring(connection, private), func(opt *orchestrator.Options) {
		opt.Packages = localLauncher{revision: revision}
	})
	install := records.PackageInstall{ID: "root-private", Package: revision.Package, Major: 1, Version: revision.Release,
		SourceKind: "local", SourceRef: filepath.Join(o.root, "checkout"),
		Dir: filepath.Join(o.root, "installs", "root-private"), Python: "/usr/bin/python3", Platform: "linux-x86"}
	_, problem := o.store.Activate(install)
	fatal(t, problem)
	requestID, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "root-private", Package: revision.Package, Entrypoint: "tile", Release: revision.Release,
		LocalInstallationID: revision.ID, Payload: []byte(`{"size":48}`), Outputs: []string{"image"},
		Worker: podRental, InstallID: install.ID, Rental: true, RentalRequired: true,
	})
	fatal(t, problem)
	waitUntil(t, "root private serving offer after code preparation", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.offers) > 0
	})
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if len(pod.offers) != 1 || len(pod.localPrepares) != 1 || len(pod.prepares) != 0 {
		t.Fatalf("root private path: offers=%d private=%d published=%d", len(pod.offers), len(pod.localPrepares), len(pod.prepares))
	}
	row, problem := o.store.RequestRow(requestID)
	fatal(t, problem)
	if row.ParentRequestID != "" || len(row.Models) != 0 || row.PlanID != podPlanID(revision.Package) || row.Ordinal != 1 {
		t.Fatalf("root request did not obtain its model-free plan from preparation: %+v", row)
	}
	if row.LocalInstallationID != revision.ID || row.LocalPackageUploadedBootID != podBootID {
		t.Fatal("private preparation lost its exact revision or verified upload boot")
	}
	spec, err := canonical.Read(pod.offers[0].InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
	must(t, err)
	if spec.Sub("serving").Str("entrypoint_binding_digest") != row.PlanID {
		t.Fatal("offer did not use the binding prepared for the root request")
	}
}
