package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestPrivateJobPreparationDoesNotActivateServing(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, jobReady: true, localJobOnly: true}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	revision := stageLocalRevision(t, root)
	o := hostOwner(t, "private-job-preparation", rentalWiring(connection, private), func(opt *orchestrator.Options) {
		opt.Packages = localLauncher{revision: revision}
	})
	fatal(t, o.store.RecordRental(records.Rental{AcceleratorCount: 1, ID: podRental, MachineName: "otter", State: "ready",
		SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000,
		Address: connection.Addr, CertPath: connection.CACert,
		ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}))
	install := records.PackageInstall{ID: "inst-private-job", Package: revision.Package, Major: 1, Version: revision.Release,
		SourceKind: "local", SourceRef: filepath.Join(o.root, "checkout"), SourceDigest: revision.SourceDigest,
		Dir: filepath.Join(o.root, "installs", "job"), Python: "/usr/bin/python3", Platform: "linux-x86"}
	_, problem := o.store.Activate(install)
	fatal(t, problem)
	requestID, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "private-job", Package: revision.Package, Entrypoint: "prepare", PlanID: "sha256:" + strings.Repeat("34", 32),
		Release: revision.Release, LocalPackageDigest: revision.Digest, Payload: []byte(`{"value":7}`),
		Worker: podRental, InstallID: install.ID, Rental: true, RentalRequired: true, Kind: "job", RetainWork: true,
	})
	fatal(t, problem)
	waitUntil(t, "private job offered after exact preparation", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.offers) > 0
	})
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if len(pod.localPrepares) != 1 || len(pod.jobDirectives) != 1 {
		t.Fatalf("private job preparation or directive repeated: prepares=%d jobs=%d", len(pod.localPrepares), len(pod.jobDirectives))
	}
	for _, desired := range pod.desired {
		if desired.GetPlacementSet() != nil {
			t.Fatal("job-only preparation activated a serving placement")
		}
	}
	if pod.offers[0].RequestId != requestID {
		t.Fatal("preparation dispatched a different request")
	}
}
