package producttest

// CO-TENANCY ON ONE RENTAL (run 146's defect, h3a-010). Routing may place several
// packages on one rented machine; the Runtime prepares exactly one package per
// PreparePackageSet. The daemon therefore issues ONE prepare per co-resident package —
// each under a download set naming exactly its own package and its own models — and sends
// the united placements as the one full-replace set. These proofs run the real
// orchestrator against the fakePod's second implementation of the pod side, which
// refuses a multi-package download set with the Runtime's own verdict.

import (
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
	stream, err := pb.NewPodHostClient(client).PreparePackageSet(t.Context(), &pb.PreparePackageSetCall{
		Claim:      &pb.Claim{RecordOwnerEpoch: 7, WorkerBootId: podBootID, Proof: ed25519.Sign(private, proof)},
		PackageSet: &pb.DesiredPackageSet{DownloadDelegation: merged},
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

// rentalWiringDownloadSet authors the unsigned download-set document a rental prepares.
func rentalWiringDownloadSet(_ *orchestrator.WorkerConnection, _ ed25519.PrivateKey,
	packages []*pb.DownloadPackageRef, models []*pb.DownloadModelRef) ([]byte, *exit.Error) {
	body, err := canonical.Bytes(&pb.DownloadDelegation{Models: models, Packages: packages})
	if err != nil {
		return nil, exit.Internalf("%v", err)
	}
	return body, nil
}
