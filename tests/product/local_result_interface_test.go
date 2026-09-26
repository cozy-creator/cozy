package producttest

import (
	"os"
	"path/filepath"
	"strings"
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
	ifaceDigest, err := canonical.Raw(assessmentDigest(surface.Raw))
	must(t, err)
	revision := &pb.LocalPackageRevision{Package: "local/example", Release: "1.0.0", SourceDigest: canonical.Digest([]byte("source")), PackageInterface: &pb.Ref{Digest: ifaceDigest, Length: uint64(len(raw))}}
	_, revisionDigest, err := canonical.Identity(revision)
	must(t, err)
	spelled, err := canonical.Spell(revisionDigest)
	must(t, err)
	capture, captureDigest, err := canonical.Identity(&pb.MachineExecutionCapture{RootInstallationId: revisionDigest, Revisions: []*pb.LocalPackageRevision{revision}})
	must(t, err)
	request, _, problem := store.Submit(records.Request{ID: "job-local-result", IdemKey: "local-result", Package: revision.Package, Release: revision.Release, Entrypoint: "main", Kind: "job", PlanID: entry.DescriptorID, Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true, LocalInstallationID: spelled})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "pr-owned-machine"))
	spec := []byte(`{"invocation":"immutable"}`)
	fatal(t, store.RecordMachineSubmission(request.ID, &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "persistent-workspace", SubmissionId: request.IdemKey, CaptureCanonicalBytes: capture, CaptureDigest: captureDigest, Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}))
	directory := filepath.Join(layout.LocalPackages, strings.TrimPrefix(spelled, "sha256:"))
	must(t, os.MkdirAll(directory, 0700))
	path := filepath.Join(directory, launch.PackageInterfaceFile)
	must(t, os.WriteFile(path, raw, 0600))
	// An old run can recover after its original install AND staged revision
	// disappear, but only from another byte-identical schema copy.
	replacement := filepath.Join(root, "replacement")
	must(t, os.MkdirAll(filepath.Join(replacement, "documents"), 0700))
	replacementPath := launch.PackageInterfacePath(replacement)
	must(t, os.WriteFile(replacementPath, []byte(`{}`), 0600))
	_, problem = store.Activate(records.PackageInstall{ID: "replacement", Package: request.Package, Major: 1, Version: "1.0.1", Dir: replacement})
	fatal(t, problem)
	must(t, os.Remove(path))
	resolver := cli.NewResolver(store, config.Config{Home: root, HubURL: "http://127.0.0.1:1"}, nil)
	if _, problem := resolver.CapturedByteOutputBound(request, "clip", "video/mp4"); problem == nil {
		t.Fatal("mismatched replacement metadata was accepted")
	}
	must(t, os.WriteFile(replacementPath, raw, 0600))
	bound, problem := resolver.CapturedByteOutputBound(request, "clip", "video/mp4")
	fatal(t, problem)
	if bound != 200000 {
		t.Fatalf("captured bound changed: %d", bound)
	}
	// The request now owns its exact schema independently of installation/source cleanup.
	must(t, os.Remove(replacementPath))
	bound, problem = resolver.CapturedByteOutputBound(request, "clip", "video/mp4")
	fatal(t, problem)
	if bound != 200000 {
		t.Fatal("source cleanup lost the captured result contract")
	}
	request.LocalInstallationID = childDigest("9")
	if _, problem := resolver.CapturedByteOutputBound(request, "clip", "video/mp4"); problem == nil {
		t.Fatal("another local root reused this result contract")
	}
}
