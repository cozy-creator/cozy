package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"flag"
	"os"
	"path/filepath"
	"strings"
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
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/privatepackage"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

var livePackageE2E = flag.String("live-package-e2e", "", "real worker package E2E config JSON")
var livePrivatePackageE2E = flag.String("live-private-package-e2e", "",
	"real worker private-package E2E config JSON")
var liveProductionDescriptor = flag.String("live-production-descriptor", "",
	"exact published ordinary H3 producer descriptor JSON")
var liveQuantizeDescriptor = flag.String("live-quantize-descriptor", "",
	"exact published quantization descriptor JSON")

func TestLiveModelProducerDescriptor(t *testing.T) {
	if *liveProductionDescriptor == "" {
		t.Skip("-live-production-descriptor is not set")
	}
	raw, err := os.ReadFile(*liveProductionDescriptor)
	must(t, err)
	descriptor, problem := launch.DecodeDescriptor(raw)
	fatal(t, problem)
	producer, problem := descriptor.Function("four-lane")
	fatal(t, problem)
	if producer.Kind != "job" || len(producer.ArtifactOutputs) != 4 || len(producer.Models) != 2 {
		t.Fatalf("published producer shape = kind %s, %d outputs, %d sources",
			producer.Kind, len(producer.ArtifactOutputs), len(producer.Models))
	}
	outputs := map[string]bool{}
	for _, output := range producer.ArtifactOutputs {
		outputs[output.OutputID] = output.RequiredContract != nil
	}
	for _, name := range []string{"bf16-full", "bf16-adaln-pruned", "fp8-adaln-pruned",
		"mxfp8-adaln-pruned"} {
		if !outputs[name] {
			t.Fatalf("published production omits output %s", name)
		}
	}
	for _, model := range producer.Models {
		if !strings.HasPrefix(model.SourceProfile, "hf/minimax-h3/") {
			t.Fatalf("producer input %s has unexpected source profile %s", model.Param,
				model.SourceProfile)
		}
	}
}

func TestLiveQuantizeDescriptor(t *testing.T) {
	if *liveQuantizeDescriptor == "" {
		t.Skip("-live-quantize-descriptor is not set")
	}
	raw, err := os.ReadFile(*liveQuantizeDescriptor)
	must(t, err)
	descriptor, problem := launch.DecodeDescriptor(raw)
	fatal(t, problem)
	if len(descriptor.Jobs) != 2 {
		t.Fatalf("quantize descriptor has %d jobs", len(descriptor.Jobs))
	}
	for _, name := range []string{"fp8", "mxfp8"} {
		job, problem := descriptor.Function(name)
		fatal(t, problem)
		if job.Kind != "job" || len(job.Models) != 1 || job.Models[0].Param != "source" ||
			len(job.ArtifactOutputs) != 1 || job.ArtifactOutputs[0].OutputID != "model" {
			t.Fatalf("quantize job %s = %+v", name, job)
		}
	}
}

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
	connection := &orchestrator.WorkerConnection{
		RentalID: live.RentalID, Addr: live.WorkerAddr, CACert: layout.RentalCert(live.RentalID),
		WorkerID: live.WorkerID, WorkerBootID: live.WorkerBootID,
		Media: &media.Spec{Addr: live.MediaAddr, Token: secret.New(live.MediaToken),
			CACert: resolve(live.MediaCertPEM)},
	}
	bindingBytes, err := canonical.Write(map[string]canonical.Value{
		"name": "marco", "slots": []canonical.Value{},
	})
	must(t, err)
	planID, err := canonical.Spell(canonical.Digest(bindingBytes))
	must(t, err)
	owner, problem := orchestrator.Open(orchestrator.Options{
		Cfg: config.Config{Home: root}, Layout: layout, Store: store, Log: os.Stderr,
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
		Payload: []byte(`{"message":"marco"}`), Worker: live.RentalID,
		Release: detail.Release.Release, ReleaseDigest: detail.Release.ReleaseDigest,
		PlanID: planID, Outputs: launch.AssetPaths(entrypoint.Result),
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

type privateLiveLauncher struct {
	installID string
	revision  privatepackage.Revision
}

func (l privateLiveLauncher) unavailable() (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return orchestrator.WorkerLaunchSpec{}, exit.Unavailablef("live private launcher is remote-only")
}
func (l privateLiveLauncher) ResolvePlacement(string) (orchestrator.DesiredPlacement, *exit.Error) {
	return orchestrator.DesiredPlacement{}, exit.Unavailablef("live private launcher is remote-only")
}
func (l privateLiveLauncher) Resolve(string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return l.unavailable()
}
func (l privateLiveLauncher) ResolveInstall(string, []orchestrator.ModelRef) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return l.unavailable()
}
func (l privateLiveLauncher) ResolveJob(string, string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return l.unavailable()
}
func (l privateLiveLauncher) ResolveJobInstall(string, string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return l.unavailable()
}
func (l privateLiveLauncher) PrivateRevision(installID, digest string) (privatepackage.Revision, *exit.Error) {
	if installID != l.installID || digest != l.revision.Digest {
		return privatepackage.Revision{}, exit.New(exit.Conflict,
			"live private revision changed")
	}
	return l.revision, nil
}

// TestLivePrivateRemoteMarcoPolo crosses Creator's sender, a real TLS pod-supervisor,
// Runtime's private preparation seam, and the real Marco job attempt. No Hub package route
// or fake WorkerControl implementation participates.
func TestLivePrivateRemoteMarcoPolo(t *testing.T) {
	path := *livePrivatePackageE2E
	if path == "" {
		t.Skip("-live-private-package-e2e is not set")
	}
	var live struct {
		RentalID       string `json:"rental_id"`
		WorkerID       string `json:"worker_id"`
		WorkerBootID   string `json:"worker_boot_id"`
		WorkerAddr     string `json:"worker_addr"`
		WorkerCertPEM  string `json:"worker_cert_pem"`
		MediaAddr      string `json:"media_addr"`
		MediaToken     string `json:"media_token"`
		MediaCertPEM   string `json:"media_cert_pem"`
		CreatorKeySeed string `json:"creator_private_key_seed_b64url"`
		PrivateProject string `json:"private_project"`
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
	live.PrivateProject = resolve(live.PrivateProject)
	if live.RentalID == "" || live.WorkerID == "" || live.WorkerBootID == "" ||
		live.WorkerAddr == "" || live.MediaAddr == "" || live.MediaToken == "" ||
		live.CreatorKeySeed == "" || live.WorkerCertPEM == "" || live.MediaCertPEM == "" ||
		live.PrivateProject == "" {
		t.Fatal("live private-package E2E config is incomplete")
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

	pack, problem := packagepublish.PrepareLocalFrom(live.PrivateProject)
	fatal(t, problem)
	sourceDigest, _, _, problem := pack.SourceIdentity()
	fatal(t, problem)
	fatal(t, pack.Build(context.Background()))
	descriptorBytes, err := os.ReadFile(pack.Descriptor)
	must(t, err)
	descriptor, problem := launch.DecodeDescriptor(descriptorBytes)
	fatal(t, problem)
	job, problem := descriptor.Function("marco_job")
	fatal(t, problem)
	if job.Kind != "job" || len(job.Models) != 0 {
		t.Fatal("live private proof is not one weightless Marco job")
	}
	installID := "install-live-private-marco"
	install := records.PackageInstall{ID: installID, Package: "local/" + pack.Name,
		Major: 1, Version: pack.Release, SourceKind: "local", SourceRef: pack.Tree,
		SourceDigest: sourceDigest, Dir: layout.GenerationDir(installID),
		PackageDescriptor: descriptor.Digest}
	pack.Close()
	_, problem = store.Activate(install)
	fatal(t, problem)
	revision, problem := privatepackage.Stage(context.Background(), layout, install)
	fatal(t, problem)
	launcher := privateLiveLauncher{installID: installID, revision: revision}
	connection := &orchestrator.WorkerConnection{RentalID: live.RentalID,
		Addr: live.WorkerAddr, CACert: layout.RentalCert(live.RentalID),
		WorkerID: live.WorkerID, WorkerBootID: live.WorkerBootID,
		Media: &media.Spec{Addr: live.MediaAddr, Token: secret.New(live.MediaToken),
			CACert: live.MediaCertPEM}}
	owner, problem := orchestrator.Open(orchestrator.Options{Cfg: config.Config{Home: root},
		Layout: layout, Store: store, Log: os.Stderr, Packages: launcher,
		Rentals: func(id string) (*orchestrator.RemoteTarget, *exit.Error) {
			if id != live.RentalID {
				return nil, exit.New(exit.NotFound, "unknown live rental %s", id)
			}
			return &orchestrator.RemoteTarget{Connection: connection}, nil
		}, ObserveRental: func(observation orchestrator.RentalObservation) *exit.Error {
			if observation.RentalID != live.RentalID || observation.WorkerID != live.WorkerID ||
				observation.WorkerBootID != live.WorkerBootID {
				return exit.New(exit.Conflict, "live worker readback changed identity")
			}
			return nil
		}, RentalClaimProof: rental.ClaimProof(layout), MaxOutputMiB: 8})
	fatal(t, problem)
	go func() { _ = owner.Serve() }()
	defer owner.Close(20 * time.Second)
	requestID, _, problem := owner.Submit(orchestrator.Submission{IdemKey: "live-private-marco",
		Kind: "job", Package: install.Package, Entrypoint: "marco_job", PlanID: job.DescriptorID,
		Payload: []byte(`{"message":"marco"}`), Worker: live.RentalID, Rental: true,
		InstallID: installID, Release: revision.Release, ReleaseDigest: revision.SourceDigest,
		PrivatePackageDigest: revision.Digest, Outputs: launch.AssetPaths(job.Result)})
	fatal(t, problem)
	result, problem := owner.AwaitSettled(requestID, 3*time.Minute)
	fatal(t, problem)
	if result.Status != "SUCCEEDED" {
		t.Fatalf("live private Marco job settled %s: %s", result.Status, result.Cause)
	}
	doc, err := canonical.Read(result.Body, &pb.AttemptOutcomeBody{})
	must(t, err)
	inline, err := base64.StdEncoding.DecodeString(doc.Sub("result").Str("inline_result"))
	must(t, err)
	var polo struct {
		Message string `json:"message"`
	}
	must(t, json.Unmarshal(inline, &polo))
	if polo.Message != "polo" {
		t.Fatalf("live private Marco response = %q", polo.Message)
	}
	row, problem := store.RequestRow(requestID)
	fatal(t, problem)
	if row == nil || row.PrivatePackageDigest != revision.Digest ||
		row.PrivatePackageUploadedBootID != live.WorkerBootID ||
		row.PackageRevisionDigest != revision.SourceDigest || row.PlanID != job.DescriptorID {
		t.Fatalf("live private request lost exact revision/worker identity: %+v", row)
	}
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
