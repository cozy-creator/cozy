package producttest

import (
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
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

	request, problem := hub.RentalRequestBytes("cozy/marco-polo/v1/marco", "cpu",
		strings.Repeat("1", 64), identity.PublicKey())
	fatal(t, problem)
	var body map[string]any
	must(t, json.Unmarshal(request, &body))
	if body["creator_public_key"] != identity.PublicKey() ||
		body["media_token_sha256"] != strings.Repeat("1", 64) || body["renter_token_sha256"] != nil {
		t.Fatalf("rental create authority is not the hardcut shape: %s", request)
	}

	certificateDigest, _ := canonical.Raw("sha256:4c5b5699f2d99ebf9195c1e518d13c94539bbd4ba670a46199f6d79d58d15721")
	document := &pb.ArtifactDelegation{
		RentalId: "rnt-01K5PROTO013", WorkerId: "wrk-4070", WorkerBootId: "boot-9f21",
		WorkerTlsCertificateDigest: certificateDigest, Revision: 4,
		PackageReleaseIds: []string{"cozy/marco-polo@v1", "cozy/upscale@v3"},
		ModelCheckpointIds: []string{
			"sha256:3be1c8def2c244b9c2a727a6ef424fd222850dd39795d262a8520ecff74e2831",
			"sha256:92ba6b2513e9e1531f2a902536113b8e53c4c750861c150730baefa9ed811ba5",
		},
		DelegationId: "dlg-2026-08-29-a", ExpiresAtUnix: 1787004000,
	}
	canonicalBytes, err := canonical.Bytes(document)
	must(t, err)
	want := []byte(`{"delegation_id":"dlg-2026-08-29-a","expires_at_unix":1787004000,"format":"cozy.worker.v1.ArtifactDelegation/1","model_checkpoint_ids":["sha256:3be1c8def2c244b9c2a727a6ef424fd222850dd39795d262a8520ecff74e2831","sha256:92ba6b2513e9e1531f2a902536113b8e53c4c750861c150730baefa9ed811ba5"],"package_release_ids":["cozy/marco-polo@v1","cozy/upscale@v3"],"rental_id":"rnt-01K5PROTO013","revision":4,"worker_boot_id":"boot-9f21","worker_id":"wrk-4070","worker_tls_certificate_digest":"sha256:4c5b5699f2d99ebf9195c1e518d13c94539bbd4ba670a46199f6d79d58d15721"}`)
	if !bytes.Equal(canonicalBytes, want) {
		t.Fatalf("ArtifactDelegation differs from worker-protocol vector:\n got %s\nwant %s", canonicalBytes, want)
	}
	signature := identity.Sign(canonicalBytes)
	if len(signature) != ed25519.SignatureSize || !ed25519.Verify(public, canonicalBytes, signature) {
		t.Fatal("the persisted rental key did not sign the exact delegation bytes")
	}
}
