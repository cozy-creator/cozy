package producttest

// CO-TENANCY ON ONE RENTAL (run 146's defect, h3a-010). Routing may place several
// packages on one rented machine; the Runtime prepares exactly one package per
// PreparePackageSet. The daemon therefore issues ONE prepare per co-resident package —
// each under a delegation naming exactly its own package and its own models — and sends
// the united placements as the one full-replace set. These proofs run the real
// orchestrator against the fakePod's second implementation of the pod side, which
// refuses a multi-package delegation with the Runtime's own verdict.

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

// delegationPackages reads the package names a prepare's delegation selects.
func delegationPackages(t *testing.T, call *pb.PreparePackageSetCall) []string {
	t.Helper()
	doc, err := canonical.Read(call.PackageSet.DownloadDelegation, &pb.DownloadDelegation{})
	must(t, err)
	names := []string(nil)
	for _, row := range doc.List("packages") {
		names = append(names, row.Str("package"))
	}
	return names
}

// assertOnePackagePerPrepare is THE invariant this lane exists for: every delegation the
// pod was ever asked to prepare names exactly one package.
func assertOnePackagePerPrepare(t *testing.T, prepares []*pb.PreparePackageSetCall) map[string][][]byte {
	t.Helper()
	delegations := map[string][][]byte{}
	for i, call := range prepares {
		names := delegationPackages(t, call)
		if len(names) != 1 {
			t.Fatalf("prepare %d presents a delegation naming %d packages (%v); want exactly one per prepare",
				i, len(names), names)
		}
		delegations[names[0]] = append(delegations[names[0]], call.PackageSet.DownloadDelegation)
	}
	return delegations
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
// delegation the pod sees names exactly one package; the package already serving keeps
// its exact delegation bytes and placement_id when the second joins (the Runtime seeds
// placement_id from the delegation bytes, so a re-signed credential would restage a
// serving placement); and the final full-replace set carries both placements.
func TestCoTenantPackagesPreparePerPackage(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, slots: 2}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	o := hostOwner(t, "podhost-cotenant", rentalWiring(connection, private))

	submit := func(pkg, idem, digest string) string {
		requestID, _, e := o.c.Submit(orchestrator.Submission{
			IdemKey: idem, Package: pkg, Entrypoint: "tile", PlanID: podPlanID(pkg),
			Release: "1.0.0", ReleaseDigest: "sha256:" + strings.Repeat(digest, 32),
			Payload: []byte(`{"size":16}`), Outputs: []string{"image"},
			Worker: podRental, Rental: true, RentalRequired: true,
		})
		fatal(t, e)
		return requestID
	}

	// The first tenant prepares, converges, and is OFFERED before the second exists.
	submit("acme/alpha", "cotenant-alpha", "31")
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

	// The second tenant joins the SAME rental (run 146: routing co-tenants a second
	// package onto the machine already serving the first).
	submit("acme/beta", "cotenant-beta", "32")
	waitUntil(t, "the second tenant's offer", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.offers) >= 2
	})

	pod.mu.Lock()
	defer pod.mu.Unlock()
	delegations := assertOnePackagePerPrepare(t, pod.prepares)
	if len(delegations["acme/alpha"]) < 2 || len(delegations["acme/beta"]) < 1 {
		t.Fatalf("prepares per package alpha=%d beta=%d; want alpha prepared again when beta joined",
			len(delegations["acme/alpha"]), len(delegations["acme/beta"]))
	}
	for pkg, issued := range delegations {
		for _, delegation := range issued[1:] {
			if !bytes.Equal(delegation, issued[0]) {
				t.Fatalf("%s was prepared under two different delegations; the signed bytes must be "+
					"reused while they live, or the serving placement_id changes with the credential", pkg)
			}
		}
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
		t.Fatalf("the pod refused a multi-package delegation; the daemon still merges prepares")
	}
}

// TestRun146ShapePreparesClean is the exact producer arc that died on run 146: a
// weightless producer package resident first, then a sibling's model-bearing package
// co-tenanted onto the same rental. Both prepare cleanly; each delegation names exactly
// its own package, and only the model-bearing package's delegation carries models.
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
		Package: "paul/minimax-h3-tools", Release: "2.2.1",
		ReleaseDigest: "sha256:" + strings.Repeat("41", 32)}}, nil))
	waitUntil(t, "the producer's placement_set", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.desired) >= 1
	})
	fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{
		Package: "paul/anima", Release: "1.2.0",
		ReleaseDigest: "sha256:" + strings.Repeat("42", 32)}},
		[]*pb.DownloadModelRef{{Package: "paul/anima", Slot: "unet", Model: "paul/anima-weights",
			Release: "1.0.0", Manifest: "sha256:" + strings.Repeat("43", 32)}}))
	waitUntil(t, "the united two-package placement_set", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.desired) >= 2 && len(placementsOf(t, pod.desired[len(pod.desired)-1])) == 2
	})

	pod.mu.Lock()
	defer pod.mu.Unlock()
	delegations := assertOnePackagePerPrepare(t, pod.prepares)
	if len(delegations["paul/minimax-h3-tools"]) == 0 || len(delegations["paul/anima"]) == 0 {
		t.Fatalf("prepares per package: %d producer, %d anima; want both prepared",
			len(delegations["paul/minimax-h3-tools"]), len(delegations["paul/anima"]))
	}
	models := func(delegation []byte) []string {
		doc, err := canonical.Read(delegation, &pb.DownloadDelegation{})
		must(t, err)
		names := []string(nil)
		for _, row := range doc.List("models") {
			names = append(names, row.Str("model"))
		}
		return names
	}
	for _, delegation := range delegations["paul/minimax-h3-tools"] {
		if got := models(delegation); len(got) != 0 {
			t.Fatalf("the weightless producer's delegation carries models %v; a model rides only "+
				"with its own package", got)
		}
	}
	for _, delegation := range delegations["paul/anima"] {
		if got := models(delegation); len(got) != 1 || got[0] != "paul/anima-weights" {
			t.Fatalf("anima's delegation carries models %v; want exactly its own", got)
		}
	}
	log, err := os.ReadFile(o.root + "/orchestrator.log")
	must(t, err)
	if strings.Contains(string(log), "package_prepare_selection_invalid") {
		t.Fatalf("run 146's refusal is back: a delegation named more than one package")
	}
}

// TestPodRefusesMergedDelegation is the red arm for the fakePod's fence: a delegation
// naming two packages — what the daemon used to sign — is refused by the pod side with
// the Runtime's exact verdict. This is the refusal run 146 died on; the fence must be
// live for the green proofs above to mean anything.
func TestPodRefusesMergedDelegation(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public}
	root := t.TempDir()
	connection, pemPath := startFakePod(t, root, pod)

	merged, signature, problem := rentalWiringSigner(connection, private, []*pb.DownloadPackageRef{
		{Package: "acme/alpha", Release: "1.0.0", ReleaseDigest: "sha256:" + strings.Repeat("31", 32)},
		{Package: "acme/beta", Release: "1.0.0", ReleaseDigest: "sha256:" + strings.Repeat("31", 32)},
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
	stream, err := pb.NewPodHostClient(client).PreparePackageSet(t.Context(), &pb.PreparePackageSetCall{
		Claim:      &pb.Claim{RecordOwnerEpoch: 7, WorkerBootId: podBootID, Proof: ed25519.Sign(private, proof)},
		PackageSet: &pb.DesiredPackageSet{DownloadDelegation: merged, DownloadDelegationSignature: signature},
	})
	must(t, err)
	deadline := time.Now().Add(10 * time.Second)
	for {
		if time.Now().After(deadline) {
			t.Fatal("no terminal prepare event for the merged delegation")
		}
		event, err := stream.Recv()
		must(t, err)
		if event.Stage == pb.PrepareStage_PREPARE_STAGE_PREPARED {
			t.Fatal("the pod prepared a delegation naming two packages; the Runtime's rule is not enforced")
		}
		if event.Stage == pb.PrepareStage_PREPARE_STAGE_REFUSED {
			if !strings.Contains(event.SafeDetail, "package_prepare_selection_invalid") {
				t.Fatalf("the refusal does not carry the Runtime's verdict: %s: %s", event.SafeCode, event.SafeDetail)
			}
			return
		}
	}
}

// rentalWiringSigner signs a delegation exactly as rentalWiring's RentalPackageSet does,
// callable outside an Options mutation.
func rentalWiringSigner(connection *orchestrator.WorkerConnection, signer ed25519.PrivateKey,
	packages []*pb.DownloadPackageRef, models []*pb.DownloadModelRef) ([]byte, []byte, *exit.Error) {
	var options orchestrator.Options
	rentalWiring(connection, signer)(&options)
	return options.RentalPackageSet(connection, packages, models)
}
