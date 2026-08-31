package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"math/big"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestRentalCreatorIdentityAndClaim(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	identity, problem := rental.PendingCreatorIdentity(layout, "paid-operation-1")
	fatal(t, problem)
	again, problem := rental.PendingCreatorIdentity(layout, "paid-operation-1")
	fatal(t, problem)
	if identity.PublicKey() != again.PublicKey() {
		t.Fatal("one paid-operation replay minted two Creator keys")
	}
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	if err != nil || len(public) != ed25519.PublicKeySize {
		t.Fatalf("Creator public key is malformed: %v", err)
	}
	info, err := os.Stat(layout.PendingRentalCreatorIdentity("paid-operation-1"))
	must(t, err)
	if runtime.GOOS != "windows" && info.Mode().Perm() != 0o600 {
		t.Fatalf("Creator identity mode = %#o, want 0600", info.Mode().Perm())
	}
	finalIdentity := layout.RentalCreatorIdentity("rnt-01K5PROTO013")
	must(t, os.Rename(layout.PendingRentalCreatorIdentity("paid-operation-1"), finalIdentity))
	workerCert := testWorkerCertificate(t)
	workerCertPath := layout.RentalCert("rnt-01K5PROTO013")
	must(t, os.WriteFile(workerCertPath, workerCert, 0o644))
	connection := &orchestrator.WorkerConnection{
		RentalID: "rnt-01K5PROTO013", CACert: workerCertPath,
		WorkerID: "wrk-4070", WorkerBootID: "boot-9f21",
	}
	claimSigner := rental.ClaimProof(layout)
	claimSignature, problem := claimSigner(connection, 41)
	fatal(t, problem)
	certificateBlock, _ := pem.Decode(workerCert)
	certificateDigest := canonical.Digest(certificateBlock.Bytes)
	claimBytes, err := canonical.Bytes(&pb.ClaimProof{
		RecordOwnerEpoch: 41, WorkerBootId: "boot-9f21", WorkerId: "wrk-4070",
		WorkerTlsCertificateDigest: certificateDigest,
	})
	must(t, err)
	if len(claimSignature) != ed25519.SignatureSize || !ed25519.Verify(public, claimBytes, claimSignature) {
		t.Fatal("ClaimProof was not signed by the persisted rental key")
	}
	fixedDigest, _ := canonical.Raw("sha256:4c5b5699f2d99ebf9195c1e518d13c94539bbd4ba670a46199f6d79d58d15721")
	fixedClaim, err := canonical.Bytes(&pb.ClaimProof{
		RecordOwnerEpoch: 41, WorkerBootId: "boot-9f21", WorkerId: "wrk-4070",
		WorkerTlsCertificateDigest: fixedDigest,
	})
	must(t, err)
	wantClaim := []byte(`{"format":"cozy.worker.v1.ClaimProof/1","record_owner_epoch":41,"worker_boot_id":"boot-9f21","worker_id":"wrk-4070","worker_tls_certificate_digest":"sha256:4c5b5699f2d99ebf9195c1e518d13c94539bbd4ba670a46199f6d79d58d15721"}`)
	if !bytes.Equal(fixedClaim, wantClaim) {
		t.Fatalf("ClaimProof differs from worker-protocol vector:\n got %s\nwant %s", fixedClaim, wantClaim)
	}
	packages := []*pb.DownloadPackageRef{{Package: "cozy/marco-polo", Release: "1.0.0",
		ReleaseDigest: "sha256:bc346950b7223be1aa3ec66c239c25538d374906df532c27a039e149bb2e632a"}}
	expires := time.Now().Add(30 * time.Minute)
	delegation, signature, problem := rental.SignDownloadDelegation(layout, connection, packages,
		nil, expires)
	fatal(t, problem)
	if len(signature) != ed25519.SignatureSize || !ed25519.Verify(public, delegation, signature) {
		t.Fatal("download delegation was not signed by the rental Creator key")
	}
	doc, err := canonical.Read(delegation, &pb.DownloadDelegation{})
	must(t, err)
	if doc.Str("rental_id") != connection.RentalID || doc.Str("worker_id") != connection.WorkerID ||
		doc.Str("worker_boot_id") != connection.WorkerBootID || len(doc.List("packages")) != 1 ||
		len(doc.List("models")) != 0 {
		t.Fatalf("download delegation lost logical or worker identity: %s", delegation)
	}
	replayed, replaySignature, problem := rental.SignDownloadDelegation(layout, connection,
		packages, nil, expires)
	fatal(t, problem)
	if !bytes.Equal(replayed, delegation) || !bytes.Equal(replaySignature, signature) {
		t.Fatal("exact logical package-set replay changed delegation bytes or signature")
	}
	empty, emptySignature, problem := rental.SignDownloadDelegation(layout, connection,
		nil, nil, time.Now().Add(30*time.Minute))
	fatal(t, problem)
	if !bytes.Contains(empty, []byte(`"models":[]`)) || !bytes.Contains(empty, []byte(`"packages":[]`)) ||
		!ed25519.Verify(public, empty, emptySignature) {
		t.Fatalf("empty package_set authority is not explicit and signed: %s", empty)
	}

	request, problem := hub.RentalRequestBytes("cpu", strings.Repeat("1", 64),
		identity.PublicKey())
	fatal(t, problem)
	var body map[string]any
	must(t, json.Unmarshal(request, &body))
	if body["creator_public_key"] != identity.PublicKey() ||
		body["media_token_sha256"] != strings.Repeat("1", 64) || body["package_ref"] != nil ||
		body["model_selections"] != nil || body["renter_token_sha256"] != nil {
		t.Fatalf("rental create authority is not the hardcut shape: %s", request)
	}
}

func testWorkerCertificate(t *testing.T) []byte {
	t.Helper()
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "cozy-worker"},
		NotBefore: time.Now().Add(-time.Minute), NotAfter: time.Now().Add(time.Hour),
		KeyUsage: x509.KeyUsageDigitalSignature,
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, public, private)
	must(t, err)
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
}
