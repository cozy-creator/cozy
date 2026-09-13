package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
)

// Runtime prepares a real unpublished wheel; Creator consumes those exact bytes
// and emits its real remote InvocationSpec; Runtime judges that emitted document.
func TestCapturedPrivateEnvironmentReachesRuntimeAdmission(t *testing.T) {
	runtime := "cozy-runtime==" + runtimeFixtureVersion(t, *privateChildRuntimeWheel)
	if *privateChildRuntimeWheel != "" {
		runtime = *privateChildRuntimeWheel
	}
	root := t.TempDir()
	// Runtime seals installed generations read-only; only this test's owned tree
	// is made removable after its helper and protocol servers have stopped.
	t.Cleanup(func() { must(t, removeAllForce(root)) })
	python := filepath.Join(root, "control", "bin", "python")
	run := func(name string, args ...string) {
		t.Helper()
		out, err := exec.Command(name, args...).CombinedOutput()
		if err != nil {
			t.Fatalf("%s: %v\n%s", name, err, out)
		}
		t.Logf("%s", out)
	}
	run("uv", "venv", "--python", "3.12", filepath.Dir(filepath.Dir(python)))
	run("uv", "pip", "install", "--python", python, runtime)
	helper := filepath.Join("testdata", "local_serving_preparation", "private_environment.py")
	run(python, helper, "prepare", root)
	raw, err := os.ReadFile(filepath.Join(root, "prepared.json"))
	must(t, err)
	set, err := canonical.Read(raw, &pb.PlacementSet{})
	must(t, err)
	placement := set.List("placements")[0]
	environment := placement.Str("environment_digest")
	if environment == "" || placement.Sub("development").Sub("project_wheel").Sub("ref").Str("digest") == "" {
		t.Fatal("Runtime did not prepare a captured wheel with an exact Environment")
	}
	var receipt struct {
		Package                string              `json:"package"`
		Release                string              `json:"release"`
		SourceDigest           string              `json:"source_digest"`
		Digest                 string              `json:"digest"`
		PackageInterfaceDigest string              `json:"package_interface_digest"`
		PackageInterfaceLength int64               `json:"package_interface_length"`
		Files                  []localpackage.File `json:"files"`
	}
	metadata, err := os.ReadFile(filepath.Join(root, "revision.json"))
	must(t, err)
	must(t, json.Unmarshal(metadata, &receipt))
	revision := localpackage.Revision{
		Package: receipt.Package, Release: receipt.Release, SourceDigest: receipt.SourceDigest,
		Digest: receipt.Digest, PackageInterfaceDigest: receipt.PackageInterfaceDigest,
		PackageInterfaceLength: receipt.PackageInterfaceLength, Files: receipt.Files,
	}
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true}
	pod.localPrepare = func(call *pb.PrepareLocalPackageCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
		if err := pod.verifyClaim(call.Claim, false); err != nil {
			return err
		}
		if spellOfBytes(call.LocalPackageSet.Package.LocalRevisionDigest) != revision.Digest {
			t.Error("Creator changed the Runtime-measured revision")
		}
		pod.mu.Lock()
		pod.preparedSet, pod.preparedDig = raw, canonical.Digest(raw)
		pod.localPrepares = append(pod.localPrepares, call)
		pod.mu.Unlock()
		return stream.Send(&pb.PrepareEvent{
			Stage:        pb.PrepareStage_PREPARE_STAGE_PREPARED,
			PlacementSet: &pb.DesiredPlacementSet{PlacementSetDigest: canonical.Digest(raw), PlacementSetCanonicalBytes: raw},
		})
	}
	connection, _ := startFakePod(t, root, pod)
	o := hostOwner(t, "native-private-environment", rentalWiring(connection, private), func(opt *orchestrator.Options) {
		opt.Packages = localLauncher{revision: revision}
	})
	install := records.PackageInstall{ID: "inst-native-environment", Package: revision.Package, Major: 1,
		Version: revision.Release, SourceKind: "local", SourceRef: root, SourceDigest: revision.SourceDigest,
		Dir: filepath.Join(o.root, "installs", "fixture"), Python: python, Platform: "linux-x86"}
	_, problem := o.store.Activate(install)
	fatal(t, problem)
	requestID, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "native-private-environment", Package: revision.Package, Entrypoint: "tile",
		PlanID:  placement.List("entrypoints")[0].Str("entrypoint_binding_digest"),
		Release: revision.Release, LocalPackageDigest: revision.Digest,
		Payload: []byte(`{"size":48}`), Worker: podRental, InstallID: install.ID,
		Rental: true, RentalRequired: true,
	})
	fatal(t, problem)
	waitUntil(t, "actual remote InvocationSpec from Runtime-prepared wheel", func() bool {
		request, problem := o.store.RequestRow(requestID)
		fatal(t, problem)
		if request != nil && (request.State == "failed" || request.State == "blocked") {
			log, _ := os.ReadFile(filepath.Join(o.root, "orchestrator.log"))
			t.Fatalf("request settled before native admission: %s\n%s", request.State, log)
		}
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.offers) == 1
	})
	pod.mu.Lock()
	offer := proto.Clone(pod.offers[0]).(*pb.AttemptOffer)
	desired := append([]byte(nil), pod.desired[0].GetPlacementSet().PlacementSetCanonicalBytes...)
	pod.mu.Unlock()
	if offer.RequestId != requestID || !bytes.Equal(desired, raw) {
		t.Fatal("Creator changed the exact prepared placement or request")
	}
	invocation, err := canonical.Read(offer.InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
	must(t, err)
	if invocation.Str("environment_digest") != environment {
		t.Fatalf("InvocationSpec environment %q differs from prepared %q", invocation.Str("environment_digest"), environment)
	}
	must(t, os.WriteFile(filepath.Join(root, "invocation.json"), offer.InvocationSpecCanonicalBytes, 0600))
	offerBytes, err := proto.Marshal(offer)
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, "offer.bin"), offerBytes, 0600))
	run(python, helper, "accept", root)
}

func spellOfBytes(raw []byte) string {
	spelled, _ := canonical.Spell(raw)
	return spelled
}
