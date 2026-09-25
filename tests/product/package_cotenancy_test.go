package producttest

// CO-TENANCY ON ONE RENTAL (run 146's defect, h3a-010). Routing may place several
// packages on one rented machine; the Runtime prepares exactly one package per
// PreparePackageSet. The daemon therefore issues ONE prepare per co-resident package —
// each under a download set naming exactly its own package and its own models — and sends
// the united placements as the one full-replace set. These proofs run the real
// orchestrator against the fakePod's second implementation of the pod side, which
// refuses a multi-package download set with the Runtime's own verdict.

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"

	"os"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// downloadSetPackages reads the package names a prepare's download set selects.
func downloadSetPackages(t *testing.T, call *pb.PreparePackageSetCall) []string {
	t.Helper()
	doc, err := canonical.Read(call.PackageSet.DownloadDelegation, &pb.DownloadDelegation{})
	must(t, err)
	names := []string(nil)
	for _, row := range doc.List("packages") {
		names = append(names, row.Str("package"))
	}
	return names
}

// assertOnePackagePerPrepare is THE invariant this lane exists for: every download set the
// pod was ever asked to prepare names exactly one package.
func assertOnePackagePerPrepare(t *testing.T, prepares []*pb.PreparePackageSetCall) map[string][][]byte {
	t.Helper()
	sets := map[string][][]byte{}
	for i, call := range prepares {
		names := downloadSetPackages(t, call)
		if len(names) != 1 {
			t.Fatalf("prepare %d presents a download set naming %d packages (%v); want exactly one per prepare",
				i, len(names), names)
		}
		sets[names[0]] = append(sets[names[0]], call.PackageSet.DownloadDelegation)
	}
	return sets
}

// placementsOf parses one desired placement_set into placement_id -> package name.
func placementsOf(t *testing.T, d *pb.DesiredWorkerState) map[string]string {
	t.Helper()
	set := d.GetPlacementSet()
	if set == nil {
		t.Fatalf("the desired state carries no placement_set: %T", d.Mode)
	}
	doc, err := canonical.Read(set.PlacementSetCanonicalBytes, &pb.PlacementSet{})
	must(t, err)
	out := map[string]string{}
	for _, placement := range doc.List("placements") {
		out[placement.Str("placement_id")] = placement.Sub("package").Str("package")
	}
	return out
}

// TestCoTenantPackagesPreparePerPackage: two published weightless packages routed onto
// ONE rental — run 146's co-tenancy shape — both prepare and both dispatch. Every
// download set the pod sees names exactly one package; the package already serving is
// not prepared again and keeps its exact download-set bytes and placement_id when the
// second joins (the Runtime seeds
// placement_id from those bytes, so re-authoring different ones would restage a
// serving placement); and the final full-replace set carries both placements.
func TestCoTenantPackagesPreparePerPackage(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, slots: 2}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	o := hostOwner(t, "podhost-cotenant", rentalWiring(connection, private))

	submit := func(pkg, idem string) string {
		submission := orchestrator.Submission{
			IdemKey: idem, Package: pkg, Entrypoint: "tile", PlanID: podPlanID(pkg),
			Release: "1.0.0",
			Payload: []byte(`{"size":16}`), Outputs: []string{"image"},
			Worker: podRental, Rental: true, RentalRequired: true,
		}
		if pkg == "acme/beta" {
			submission.Models = []orchestrator.ModelRef{{
				Package: pkg, Slot: "tile.models.model", Model: "paul/anima",
				Release: "1.0.0", Lane: "bf16", Manifest: "sha256:" + strings.Repeat("43", 32),
				ManifestLength: 2048,
			}}
		}
		requestID, _, e := o.c.Submit(submission)
		fatal(t, e)
		return requestID
	}

	// The first tenant prepares, converges, and is OFFERED before the second exists.
	submit("acme/alpha", "cotenant-alpha")
	waitUntil(t, "the first tenant's offer", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.offers) >= 1
	})
	pod.mu.Lock()
	firstSets := len(pod.desired)
	alphaSet := placementsOf(t, pod.desired[firstSets-1])
	pod.mu.Unlock()
	if len(alphaSet) != 1 {
		t.Fatalf("the first tenant's set carries %d placements; want its one", len(alphaSet))
	}
	if reason, _ := o.c.RentalStanding(podRental, true); reason != orchestrator.ExcludedModeConflict {
		t.Fatalf("an offered serving invocation allowed a job mode replacement: %s", reason)
	}

	// The second tenant joins the SAME rental (run 146: routing co-tenants a second
	// package onto the machine already serving the first).
	submit("acme/beta", "cotenant-beta")
	waitUntil(t, "the second tenant's offer", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.offers) >= 2
	})

	pod.mu.Lock()
	defer pod.mu.Unlock()
	sets := assertOnePackagePerPrepare(t, pod.prepares)
	if len(sets["acme/alpha"]) != 1 || len(sets["acme/beta"]) != 1 {
		t.Fatalf("prepares per package alpha=%d beta=%d; want each prepared once — alpha's "+
			"unchanged download set was already prepared on this boot when beta joined",
			len(sets["acme/alpha"]), len(sets["acme/beta"]))
	}
	for pkg, issued := range sets {
		for _, set := range issued[1:] {
			if !bytes.Equal(set, issued[0]) {
				t.Fatalf("%s was prepared under two different download sets; a set is a pure "+
					"function of its content, or the serving placement_id changes for nothing", pkg)
			}
		}
	}
	beta, err := canonical.Read(sets["acme/beta"][0], &pb.DownloadDelegation{})
	must(t, err)
	models := beta.List("models")
	if len(models) != 1 || models[0].Str("lane") != "bf16" {
		t.Fatalf("beta's download set lost the selected model lane: %v", models)
	}
	united := placementsOf(t, pod.desired[len(pod.desired)-1])
	if len(united) != 2 {
		t.Fatalf("the united set carries %d placements (%v); want both tenants", len(united), united)
	}
	hosts := map[string]string{}
	for id, pkg := range united {
		hosts[pkg] = id
	}
	for id, pkg := range alphaSet {
		if hosts[pkg] != id {
			t.Fatalf("placement %s of %s was restaged as %s when the second tenant joined", id, pkg, hosts[pkg])
		}
	}
	specs := map[string]bool{}
	for _, offer := range pod.offers {
		spec, err := canonical.Read(offer.InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
		must(t, err)
		specs[spec.Sub("serving").Str("entrypoint_binding_digest")] = true
	}
	if !specs[podPlanID("acme/alpha")] || !specs[podPlanID("acme/beta")] {
		t.Fatalf("the offers do not cover both tenants' plans: %v", specs)
	}
	log, err := os.ReadFile(o.root + "/orchestrator.log")
	must(t, err)
	if strings.Contains(string(log), "package_prepare_selection_invalid") {
		t.Fatalf("the pod refused a multi-package download set; the daemon still merges prepares")
	}
}

// TestRun146ShapePreparesClean is the exact producer arc that died on run 146: a
// weightless producer package resident first, then a sibling's model-bearing package
// co-tenanted onto the same rental. Both prepare cleanly; each download set names exactly
// its own package, and only the model-bearing package's set carries models.
func TestRun146ShapePreparesClean(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	o := hostOwner(t, "podhost-run146", rentalWiring(connection, private))

	instance, _, _, e := o.c.EnsureRental(podRental)
	fatal(t, e)
	fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{
		Package: "paul/minimax-h3-tools", Release: "2.2.1"}}, nil))
	waitUntil(t, "the producer's placement_set", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.desired) >= 1
	})
	fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{
		Package: "paul/anima", Release: "1.2.0"}},
		[]*pb.DownloadModelRef{{Package: "paul/anima", Slot: "unet", Model: "paul/anima-weights",
			Release: "1.0.0", Lane: "bf16", Manifest: "sha256:" + strings.Repeat("43", 32)}}))
	waitUntil(t, "the united two-package placement_set", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.desired) >= 2 && len(placementsOf(t, pod.desired[len(pod.desired)-1])) == 2
	})

	pod.mu.Lock()
	defer pod.mu.Unlock()
	sets := assertOnePackagePerPrepare(t, pod.prepares)
	if len(sets["paul/minimax-h3-tools"]) == 0 || len(sets["paul/anima"]) == 0 {
		t.Fatalf("prepares per package: %d producer, %d anima; want both prepared",
			len(sets["paul/minimax-h3-tools"]), len(sets["paul/anima"]))
	}
	models := func(downloadSet []byte) []string {
		doc, err := canonical.Read(downloadSet, &pb.DownloadDelegation{})
		must(t, err)
		names := []string(nil)
		for _, row := range doc.List("models") {
			names = append(names, row.Str("model"))
		}
		return names
	}
	for _, set := range sets["paul/minimax-h3-tools"] {
		if got := models(set); len(got) != 0 {
			t.Fatalf("the weightless producer's download set carries models %v; a model rides only "+
				"with its own package", got)
		}
	}
	for _, set := range sets["paul/anima"] {
		if got := models(set); len(got) != 1 || got[0] != "paul/anima-weights" {
			t.Fatalf("anima's download set carries models %v; want exactly its own", got)
		}
	}
	log, err := os.ReadFile(o.root + "/orchestrator.log")
	must(t, err)
	if strings.Contains(string(log), "package_prepare_selection_invalid") {
		t.Fatalf("run 146's refusal is back: a download set named more than one package")
	}
}

// TestPodRefusesAMergedDownloadSet is the red arm for the fakePod's fence: a download set
// naming two packages — what the daemon used to sign — is refused by the pod side with
// the Runtime's exact verdict. This is the refusal run 146 died on; the fence must be
// live for the green proofs above to mean anything.
func TestPodRefusesAMergedDownloadSet(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public}
	root := t.TempDir()
	connection, pemPath := startFakePod(t, root, pod)

	merged, problem := rentalWiringDownloadSet(connection, private, []*pb.DownloadPackageRef{
		{Package: "acme/alpha", Release: "1.0.0"},
		{Package: "acme/beta", Release: "1.0.0"},
	}, nil)
	if problem != nil {
		t.Fatal(problem.Message)
	}
	leaf, err := os.ReadFile(pemPath)
	must(t, err)
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(leaf) {
		t.Fatal("the pod leaf certificate is unreadable")
	}
	client, err := grpc.NewClient(connection.Addr, grpc.WithTransportCredentials(
		credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12, RootCAs: pool, ServerName: "cozy-worker"})))
	must(t, err)
	defer client.Close()
	proof, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: 7, WorkerId: podWorkerID,
		WorkerBootId: podBootID, WorkerTlsCertificateDigest: pod.leafDigest})
	must(t, err)
	facts := testPrepareFacts("acme/alpha", "1.0.0")
	stream, err := pb.NewPodHostClient(client).PreparePackageSet(t.Context(), &pb.PreparePackageSetCall{
		Claim:              &pb.Claim{RecordOwnerEpoch: 7, WorkerBootId: podBootID, Proof: ed25519.Sign(private, proof)},
		PackageSet:         &pb.DesiredPackageSet{DownloadDelegation: merged},
		Application:        facts.Application,
		ModelSlotPaths:     facts.ModelSlotPaths,
		ImageInventory:     facts.ImageInventory,
		LockedRequirements: facts.LockedRequirements,
	})
	must(t, err)
	deadline := time.Now().Add(10 * time.Second)
	for {
		if time.Now().After(deadline) {
			t.Fatal("no terminal prepare event for the merged download set")
		}
		event, err := stream.Recv()
		must(t, err)
		if event.Stage == pb.PrepareStage_PREPARE_STAGE_PREPARED {
			t.Fatal("the pod prepared a download set naming two packages; the Runtime's rule is not enforced")
		}
		if event.Stage == pb.PrepareStage_PREPARE_STAGE_REFUSED {
			if !strings.Contains(event.SafeDetail, "package_prepare_selection_invalid") {
				t.Fatalf("the refusal does not carry the Runtime's verdict: %s: %s", event.SafeCode, event.SafeDetail)
			}
			return
		}
	}
}

// rentalWiringDownloadSet authors a download set exactly as rentalWiring's
// RentalPackageSet does, callable outside an Options mutation.
func rentalWiringDownloadSet(connection *orchestrator.WorkerConnection, signer ed25519.PrivateKey,
	packages []*pb.DownloadPackageRef, models []*pb.DownloadModelRef) ([]byte, *exit.Error) {
	var options orchestrator.Options
	rentalWiring(connection, signer)(&options)
	return options.RentalPackageSet(packages, models)
}
