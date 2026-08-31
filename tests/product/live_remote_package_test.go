package producttest

import (
	"crypto/ed25519"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"flag"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

var livePackageE2E = flag.String("live-package-e2e", "", "real worker package E2E config JSON")

// TestLiveRemoteMarcoPolo is the opt-in cross-repository proof. Its supplied worker must
// be the real Tensorhub pod-supervisor hosting the real cozy-runtime; no fake WorkerControl
// frame is admitted here. The harness passes one JSON file with -live-package-e2e.
func TestLiveRemoteMarcoPolo(t *testing.T) {
	path := *livePackageE2E
	if path == "" {
		t.Skip("-live-package-e2e is not set")
	}
	var live struct {
		RentalID        string `json:"rental_id"`
		WorkerID        string `json:"worker_id"`
		WorkerBootID    string `json:"worker_boot_id"`
		WorkerAddr      string `json:"worker_addr"`
		WorkerCertPEM   string `json:"worker_cert_pem"`
		MediaAddr       string `json:"media_addr"`
		MediaToken      string `json:"media_token"`
		MediaCertPEM    string `json:"media_cert_pem"`
		CreatorKeySeed  string `json:"creator_private_key_seed_b64url"`
		Package         string `json:"package"`
		ReleaseDetail   string `json:"release_detail_json"`
		ExpectedMessage string `json:"expected_message"`
	}
	data, err := os.ReadFile(path)
	must(t, err)
	must(t, json.Unmarshal(data, &live))
	base := filepath.Dir(path)
	resolve := func(value string) string {
		if value != "" && !filepath.IsAbs(value) {
			return filepath.Join(base, value)
		}
		return value
	}
	live.WorkerCertPEM = resolve(live.WorkerCertPEM)
	live.MediaCertPEM = resolve(live.MediaCertPEM)
	live.ReleaseDetail = resolve(live.ReleaseDetail)
	if live.RentalID == "" || live.WorkerID == "" || live.WorkerBootID == "" ||
		live.WorkerAddr == "" || live.MediaAddr == "" || live.MediaToken == "" || live.Package == "" ||
		live.CreatorKeySeed == "" || live.WorkerCertPEM == "" || live.MediaCertPEM == "" ||
		live.ReleaseDetail == "" {
		t.Fatal("live package E2E config is incomplete")
	}

	var detail hub.PackageReleaseDetail
	detailBytes, err := os.ReadFile(live.ReleaseDetail)
	must(t, err)
	must(t, json.Unmarshal(detailBytes, &detail))
	if int64(len(detail.PackageDescriptor)) != detail.Release.PackageDescriptorLength {
		t.Fatal("release-detail fixture descriptor length does not match its release fact")
	}
	descriptor, problem := launch.DecodeDescriptor(detail.PackageDescriptor)
	fatal(t, problem)
	if descriptor.Digest != detail.Release.PackageDescriptorDigest {
		t.Fatal("release-detail fixture descriptor digest does not match its release fact")
	}
	entrypoint, problem := descriptor.Function("marco")
	fatal(t, problem)
	if len(descriptor.Entrypoints) != 1 || len(entrypoint.Models) != 0 {
		t.Fatal("live proof package is not one weightless Marco entrypoint")
	}

	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	must(t, os.MkdirAll(layout.Rentals, 0o700))
	writeCreatorKey(t, live.CreatorKeySeed, layout.RentalCreatorIdentity(live.RentalID))
	copyFile(t, live.WorkerCertPEM, layout.RentalCert(live.RentalID), 0o644)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	_, problem = store.Activate(records.PackageInstall{
		ID: "live-marco", Package: live.Package, Major: 1, Version: detail.Release.Release,
		SourceKind: "tensorhub", SourceRef: live.Package + "@" + detail.Release.Release,
		SourceDigest: detail.Release.ReleaseDigest, Verified: true,
		Dir: filepath.Join(root, "release-metadata"), LinkMode: "none",
		PackageDescriptor: detail.Release.PackageDescriptorDigest,
	})
	fatal(t, problem)
	connection := &orchestrator.WorkerConnection{
		RentalID: live.RentalID, Addr: live.WorkerAddr, CACert: layout.RentalCert(live.RentalID),
		WorkerID: live.WorkerID, WorkerBootID: live.WorkerBootID,
		Media: &media.Spec{Addr: live.MediaAddr, Token: secret.New(live.MediaToken),
			CACert: resolve(live.MediaCertPEM)},
	}
	launcher := livePackageLauncher{logical: orchestrator.LogicalPackage{
		Package: live.Package, Release: detail.Release.Release,
		ReleaseDigest: detail.Release.ReleaseDigest, InstallID: "live-marco",
		Function: "marco", Outputs: launch.AssetPaths(entrypoint.Result),
	}}
	owner, problem := orchestrator.Open(orchestrator.Options{
		Cfg: config.Config{Home: root}, Layout: layout, Store: store, Packages: launcher,
		Rentals: func(id string) (*orchestrator.RemoteTarget, *exit.Error) {
			if id != live.RentalID {
				return nil, exit.New(exit.NotFound, "unknown live rental %s", id)
			}
			return &orchestrator.RemoteTarget{Connection: connection}, nil
		},
		ObserveRental: func(observation orchestrator.RentalObservation) *exit.Error {
			if observation.RentalID != live.RentalID || observation.WorkerID != live.WorkerID ||
				observation.WorkerBootID != live.WorkerBootID {
				return exit.New(exit.Conflict, "live worker readback changed identity")
			}
			return nil
		},
		RentalClaimProof: rental.ClaimProof(layout), RentalPackageSet: rental.PackageSetSigner(layout),
		ConfigDigest: "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
		MaxOutputMiB: 8,
	})
	fatal(t, problem)
	go func() { _ = owner.Serve() }()
	defer owner.Close(20 * time.Second)

	requestID, _, problem := owner.Submit(orchestrator.Submission{
		IdemKey: "live-marco-polo", Package: live.Package, Entrypoint: "marco",
		Payload: []byte(`{"message":"marco"}`), Worker: live.RentalID, InstallID: "live-marco",
		Outputs: launch.AssetPaths(entrypoint.Result),
	})
	fatal(t, problem)
	result, problem := owner.AwaitSettled(requestID, 3*time.Minute)
	fatal(t, problem)
	if result.Status != "SUCCEEDED" {
		t.Fatalf("live Marco invocation settled %s: %s", result.Status, result.Cause)
	}
	doc, err := canonical.Read(result.Body, &pb.AttemptOutcomeBody{})
	must(t, err)
	inline, err := base64.StdEncoding.DecodeString(doc.Sub("result").Str("inline_result"))
	must(t, err)
	var polo struct {
		Message string `json:"message"`
	}
	must(t, json.Unmarshal(inline, &polo))
	want := live.ExpectedMessage
	if want == "" {
		want = "polo2222"
	}
	if polo.Message != want {
		t.Fatalf("live Marco response = %q, want %q", polo.Message, want)
	}
	row, problem := store.RequestRow(requestID)
	fatal(t, problem)
	if row == nil || row.PlanID == "" || row.PackageRevisionDigest != detail.Release.ReleaseDigest ||
		row.EnvironmentDigest == "" || row.ConfigDigest == "" {
		t.Fatalf("live request did not pin worker-resolved invocation identity: %+v", row)
	}
}

type livePackageLauncher struct{ logical orchestrator.LogicalPackage }

func (l livePackageLauncher) ResolveLogicalInstall(installID, function string) (orchestrator.LogicalPackage, *exit.Error) {
	if installID != l.logical.InstallID || function != l.logical.Function {
		return orchestrator.LogicalPackage{}, exit.New(exit.NotFound, "unknown live package selection")
	}
	return l.logical, nil
}

func (livePackageLauncher) ResolvePlacement(string) (orchestrator.DesiredPlacement, *exit.Error) {
	return orchestrator.DesiredPlacement{}, exit.Unavailablef("live proof resolves no local placement")
}
func (livePackageLauncher) Resolve(string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return orchestrator.WorkerLaunchSpec{}, exit.Unavailablef("live proof launches no local worker")
}
func (livePackageLauncher) ResolveInstall(string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return orchestrator.WorkerLaunchSpec{}, exit.Unavailablef("live proof launches no local install")
}
func (livePackageLauncher) ResolveJob(string, string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return orchestrator.WorkerLaunchSpec{}, exit.Unavailablef("live proof runs no jobs")
}
func (livePackageLauncher) ResolveJobInstall(string, string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return orchestrator.WorkerLaunchSpec{}, exit.Unavailablef("live proof runs no jobs")
}

func copyFile(t *testing.T, source, destination string, mode os.FileMode) {
	t.Helper()
	data, err := os.ReadFile(source)
	must(t, err)
	must(t, os.WriteFile(destination, data, mode))
}

func writeCreatorKey(t *testing.T, encoded, destination string) {
	t.Helper()
	seed, err := base64.RawURLEncoding.DecodeString(encoded)
	must(t, err)
	if len(seed) != ed25519.SeedSize {
		t.Fatalf("live Creator key seed is %d B, want %d", len(seed), ed25519.SeedSize)
	}
	der, err := x509.MarshalPKCS8PrivateKey(ed25519.NewKeyFromSeed(seed))
	must(t, err)
	must(t, os.WriteFile(destination,
		pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der}), 0o600))
}
