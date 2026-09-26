package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
)

// Runtime prepares a real unpublished wheel; Creator consumes those exact bytes
// and emits its real remote InvocationSpec; Runtime judges that emitted document.
func TestCapturedPrivateInstallationReachesRuntimeAdmission(t *testing.T) {
	integration(t)
	// This fixture exercises the installed-resource protocol introduced after the
	// generic controller floor. An explicit candidate wheel qualifies newer cohorts.
	version := "0.18.28"
	runtime := "cozy-runtime==" + version
	if *privateChildRuntimeWheel != "" {
		version, runtime = runtimeFixtureVersion(t, *privateChildRuntimeWheel), *privateChildRuntimeWheel
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
	// Freeze the fixture's SDK as package-owned wheels and exact public refs.
	// Its child must not inherit the control interpreter's site-packages.
	sdkProject := filepath.Join(root, "sdk-project")
	must(t, os.Mkdir(sdkProject, 0700))
	sdkMetadata := fmt.Sprintf("[project]\nname=\"private-sdk-fixture\"\nversion=\"1.0\"\nrequires-python=\">=3.12,<3.13\"\ndependencies=[%s]\n", strconv.Quote("cozy-runtime=="+version))
	if *privateChildRuntimeWheel != "" {
		sdkMetadata += "[tool.uv.sources]\ncozy-runtime={path=" + strconv.Quote(runtime) + "}\n"
	}
	must(t, os.WriteFile(filepath.Join(sdkProject, "pyproject.toml"), []byte(sdkMetadata), 0600))
	run("uv", "lock", "--project", sdkProject)
	lock := filepath.Join(root, "pylock.sdk.toml")
	run("uv", "export", "--quiet", "--locked", "--no-dev", "--no-default-groups", "--no-emit-project", "--no-emit-local", "--format", "pylock.toml", "--output-file", lock, "--project", sdkProject)
	lockBytes, err := os.ReadFile(lock)
	must(t, err)
	sdkRows, sdkProblem := packagepublish.RegistryRowsFromLock(lockBytes, nil, "")
	fatal(t, sdkProblem)
	dependencyRequirements := packagepublish.RegistryRequirements(sdkRows)
	must(t, os.WriteFile(filepath.Join(root, "dependency-requirements.txt"), dependencyRequirements, 0600))
	must(t, os.WriteFile(filepath.Join(root, "sdk-wheel.txt"), []byte(*privateChildRuntimeWheel), 0600))
	helper := filepath.Join("testdata", "local_serving_preparation", "private_environment.py")
	run(python, helper, "prepare", root)
	raw, err := os.ReadFile(filepath.Join(root, "prepared.json"))
	must(t, err)
	set, err := canonical.Read(raw, &pb.PlacementSet{})
	must(t, err)
	placement := set.List("placements")[0]
	environment := placement.Str("installation_id")
	if environment == "" || placement.Sub("development").Str("installation_id") != environment {
		t.Fatal("Runtime did not prepare a captured wheel with an exact Environment")
	}
	var revision localpackage.Installation
	metadata, err := os.ReadFile(filepath.Join(root, "installation.json"))
	must(t, err)
	must(t, json.Unmarshal(metadata, &revision))
	revision.DependencyRequirements = dependencyRequirements
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true}
	pod.localPrepare = func(call *pb.PrepareLocalPackageCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
		if err := pod.verifyClaim(call.Claim, false); err != nil {
			return err
		}
		if call.LocalPackageSet.Package.InstallationId != revision.ID {
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
		Version: revision.Release, SourceKind: "local", SourceRef: root,
		Dir: filepath.Join(o.root, "installs", "fixture"), Python: python, Platform: "linux-x86"}
	_, problem := o.store.Activate(install)
	fatal(t, problem)
	requestID, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "native-private-environment", Package: revision.Package, Entrypoint: "tile",
		PlanID:  placement.List("entrypoints")[0].Str("entrypoint_binding_digest"),
		Release: revision.Release, LocalInstallationID: revision.ID,
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
	if invocation.Str("installation_id") != environment {
		t.Fatalf("InvocationSpec environment %q differs from prepared %q", invocation.Str("installation_id"), environment)
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
