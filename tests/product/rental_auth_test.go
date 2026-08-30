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

func TestRentalCreatorIdentityAndDelegation(t *testing.T) {
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
	claimSigner := rental.ClaimProof(layout)
	claimSignature, problem := claimSigner(&orchestrator.WorkerConnection{
		RentalID: "rnt-01K5PROTO013", CACert: workerCertPath,
		WorkerID: "wrk-4070", WorkerBootID: "boot-9f21",
	}, 41)
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

	request, problem := hub.RentalRequestBytes("cozy/marco-polo/v1/marco", "cpu",
		strings.Repeat("1", 64), identity.PublicKey())
	fatal(t, problem)
	var body map[string]any
	must(t, json.Unmarshal(request, &body))
	if body["creator_public_key"] != identity.PublicKey() ||
		body["media_token_sha256"] != strings.Repeat("1", 64) || body["renter_token_sha256"] != nil {
		t.Fatalf("rental create authority is not the hardcut shape: %s", request)
	}

	certificateDigest = fixedDigest
	document := &pb.ArtifactDelegation{
		RentalId: "rnt-01K5PROTO013", WorkerId: "wrk-4070", WorkerBootId: "boot-9f21",
		WorkerTlsCertificateDigest: certificateDigest, Revision: 4,
		PackageReleaseIds: []string{"cozy/marco-polo@v1", "cozy/upscale@v3"},
		ModelManifestIds: []string{
			"sha256:6d71b9c29a444029755492e433597672b5f757c6fc5a6a83bdd26fc9a6153718",
			"sha256:d984214cfcb6d61878a1251037851276f0c892ae90f5359a5bee522635975e19",
		},
		DelegationId: "dlg-2026-08-29-a", ExpiresAtUnix: 1787004000,
	}
	canonicalBytes, err := canonical.Bytes(document)
	must(t, err)
	want := []byte(`{"delegation_id":"dlg-2026-08-29-a","expires_at_unix":1787004000,"format":"cozy.worker.v1.ArtifactDelegation/2","model_manifest_ids":["sha256:6d71b9c29a444029755492e433597672b5f757c6fc5a6a83bdd26fc9a6153718","sha256:d984214cfcb6d61878a1251037851276f0c892ae90f5359a5bee522635975e19"],"package_release_ids":["cozy/marco-polo@v1","cozy/upscale@v3"],"rental_id":"rnt-01K5PROTO013","revision":4,"worker_boot_id":"boot-9f21","worker_id":"wrk-4070","worker_tls_certificate_digest":"sha256:4c5b5699f2d99ebf9195c1e518d13c94539bbd4ba670a46199f6d79d58d15721"}`)
	if !bytes.Equal(canonicalBytes, want) {
		t.Fatalf("ArtifactDelegation differs from worker-protocol vector:\n got %s\nwant %s", canonicalBytes, want)
	}
	signature := identity.Sign(canonicalBytes)
	if len(signature) != ed25519.SignatureSize || !ed25519.Verify(public, canonicalBytes, signature) {
		t.Fatal("the persisted rental key did not sign the exact delegation bytes")
	}
	noModel, err := canonical.Bytes(&pb.ArtifactDelegation{
		RentalId: "rnt-01K5PROTO013", WorkerId: "wrk-4070", WorkerBootId: "boot-9f21",
		WorkerTlsCertificateDigest: certificateDigest, Revision: 5,
		PackageReleaseIds: []string{"cozy/marco-polo@v1"}, ModelManifestIds: []string{},
		DelegationId: "dlg-2026-08-29-no-model", ExpiresAtUnix: 1787004000,
	})
	must(t, err)
	wantNoModel := []byte(`{"delegation_id":"dlg-2026-08-29-no-model","expires_at_unix":1787004000,"format":"cozy.worker.v1.ArtifactDelegation/2","model_manifest_ids":[],"package_release_ids":["cozy/marco-polo@v1"],"rental_id":"rnt-01K5PROTO013","revision":5,"worker_boot_id":"boot-9f21","worker_id":"wrk-4070","worker_tls_certificate_digest":"sha256:4c5b5699f2d99ebf9195c1e518d13c94539bbd4ba670a46199f6d79d58d15721"}`)
	if !bytes.Equal(noModel, wantNoModel) {
		t.Fatalf("no-model ArtifactDelegation omitted the explicit empty set:\n got %s\nwant %s", noModel, wantNoModel)
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
