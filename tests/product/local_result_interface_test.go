package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestLocalResultInterfaceSurvivesInstallRemoval(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	raw := []byte(`{"application":"proof:app","format":"cozy.package.interface/1","entrypoints":[],"jobs":[{"name":"main","models":[],"publishes":false,"weights_outputs":[],"request":{"fields":[]},"result":{"fields":[{"name":"clip","type":{"asset":"video"},"asset_bound":{"max_bytes":200000,"media_types":["video/mp4"]}}]}}]}`)
	surface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	entry, problem := surface.Function("main")
	fatal(t, problem)
	installed := &pb.InstalledPackage{InstallationId: "local-result-install", Package: "local/example", Release: "1.0.0", PackageInterface: raw}
	capture, captureDigest, err := canonical.Identity(&pb.MachineExecutionCapture{RootInstallationId: installed.InstallationId, InstalledPackages: []*pb.InstalledPackage{installed}})
	must(t, err)
	request, _, problem := store.Submit(records.Request{ID: "job-local-result", IdemKey: "local-result", Package: installed.Package, Release: installed.Release, Entrypoint: "main", Kind: "job", PlanID: entry.DescriptorID, Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true, LocalInstallationID: installed.InstallationId})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "pr-owned-machine"))
	spec := []byte(`{"invocation":"immutable"}`)
	fatal(t, store.RecordMachineSubmission(request.ID, &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "persistent-workspace", SubmissionId: request.IdemKey, CaptureCanonicalBytes: capture, CaptureDigest: captureDigest, Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}))
	// The accepted graph retains the schema itself, so no install or source files
	// are needed and mutable replacement pins cannot alter the output contract.
	resolver := cli.NewResolver(store, config.Config{Home: root, HubURL: "http://127.0.0.1:1"})
	bound, problem := resolver.CapturedByteOutputBound(request, "clip", "video/mp4")
	fatal(t, problem)
	if bound != 200000 {
		t.Fatalf("captured bound changed: %d", bound)
	}
	if _, problem := resolver.CapturedByteOutputBound(request, "unknown", "video/mp4"); problem == nil {
		t.Fatal("undeclared output was accepted")
	}
	if _, problem := resolver.CapturedByteOutputBound(request, "clip", "image/png"); problem == nil {
		t.Fatal("wrong output media type was accepted")
	}

}
