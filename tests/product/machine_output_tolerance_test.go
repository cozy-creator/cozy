package producttest

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type machineByteHolder interface {
	RetainByteTree(context.Context, *pb.NativeByteRetentionCall) (*pb.NativeByteRetentionResult, error)
	ReleaseByteTree(context.Context, *pb.NativeByteRetentionCall) (*pb.NativeByteRetentionResult, error)
	ReadByteTreeObject(*pb.NativeByteReadCall, grpc.ServerStreamingServer[pb.NativeByteReadChunk]) error
}

func (p *fakePod) RetainByteTree(ctx context.Context, call *pb.NativeByteRetentionCall) (*pb.NativeByteRetentionResult, error) {
	if holder, ok := p.machine.(machineByteHolder); ok {
		return holder.RetainByteTree(ctx, call)
	}
	return p.UnimplementedPodHostServer.RetainByteTree(ctx, call)
}

func (p *fakePod) ReleaseByteTree(ctx context.Context, call *pb.NativeByteRetentionCall) (*pb.NativeByteRetentionResult, error) {
	if holder, ok := p.machine.(machineByteHolder); ok {
		return holder.ReleaseByteTree(ctx, call)
	}
	return p.UnimplementedPodHostServer.ReleaseByteTree(ctx, call)
}

func (p *fakePod) ReadByteTreeObject(call *pb.NativeByteReadCall, stream grpc.ServerStreamingServer[pb.NativeByteReadChunk]) error {
	if holder, ok := p.machine.(machineByteHolder); ok {
		return holder.ReadByteTreeObject(call, stream)
	}
	return p.UnimplementedPodHostServer.ReadByteTreeObject(call, stream)
}

// fileMachine holds its result bytes by digest until the collector releases them.
type fileMachine struct {
	finishedMachine
	objects map[string][]byte
}

func (m *fileMachine) RetainByteTree(_ context.Context, call *pb.NativeByteRetentionCall) (*pb.NativeByteRetentionResult, error) {
	return &pb.NativeByteRetentionResult{RetentionId: call.Request.RetentionId, Source: call.Request.Source}, nil
}

func (m *fileMachine) ReleaseByteTree(_ context.Context, call *pb.NativeByteRetentionCall) (*pb.NativeByteRetentionResult, error) {
	return &pb.NativeByteRetentionResult{RetentionId: call.Request.RetentionId, Source: call.Request.Source, Released: true}, nil
}

func (m *fileMachine) ReadByteTreeObject(call *pb.NativeByteReadCall, stream grpc.ServerStreamingServer[pb.NativeByteReadChunk]) error {
	data, ok := m.objects[string(call.Object.Digest)]
	if !ok {
		return status.Error(codes.NotFound, "no such object")
	}
	return stream.Send(&pb.NativeByteReadChunk{Offset: 0, Data: data})
}

// The machine is an independently upgraded peer. An output it returns without a declaration
// is ignored with a warning; a declared output it does not return fails alone, and the
// outputs it did return are still collected.
func TestMachineOutputsAreCollectedPastExtrasAndOmissions(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	identity, problem := rental.PendingCreatorIdentity(layout, "output-tolerance")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)

	asset := `{"type":{"asset":"image"},"wire":"required","asset_bound":{"max_bytes":1000,"media_types":["image/png"]}}`
	surface := `{"application":"proof:app","format":"cozy.package.interface/1","entrypoints":[],"jobs":[{"name":"main","models":[],"publishes":false,"weights_outputs":[],"request":{"fields":[]},"result":{"fields":[` +
		`{"name":"image",` + asset[1:] + `,{"name":"mask",` + asset[1:] + `]}}]}`
	installed := &pb.InstalledPackage{InstallationId: "tolerance-install", Package: "local/example", Release: "1.0.0", PackageInterface: []byte(surface)}
	capture, captureDigest, err := canonical.Identity(&pb.MachineExecutionCapture{RootInstallationId: installed.InstallationId, InstalledPackages: []*pb.InstalledPackage{installed}})
	must(t, err)
	request, _, problem := store.Submit(records.Request{ID: "job-output-tolerance", IdemKey: "output-tolerance", Package: installed.Package,
		Release: installed.Release, Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"),
		MachineExecutionObserver: true, LocalInstallationID: installed.InstallationId})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, podRental))
	spec := []byte(`{"invocation":"immutable"}`)
	submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "workspace", SubmissionId: request.IdemKey,
		CaptureCanonicalBytes: capture, CaptureDigest: captureDigest,
		Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
	fatal(t, store.RecordMachineSubmission(request.ID, submission))
	receipt := &pb.MachineExecutionReceipt{RequestId: request.ID, SubmissionId: request.IdemKey, CaptureDigest: submission.CaptureDigest,
		InvocationSpecDigest: submission.Offer.InvocationSpecDigest, AcceptedAtMs: uint64(time.Now().UnixMilli()),
		WorkerId: podWorkerID, WorkerBootId: podBootID, ExecutionWorkspaceId: "workspace"}
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))

	image := []byte("\x89PNG returned by the machine")
	extra := []byte("an output this package never declared")
	entry := func(name string, data []byte, marker byte) (*pb.OutputEntry, map[string]any) {
		sum := sha256.Sum256(data)
		spelled, err := canonical.Spell(sum[:])
		must(t, err)
		return &pb.OutputEntry{OutputId: name, Digest: sum[:], Length: uint64(len(data)), MimeType: "image/png",
				NativeTree: &pb.NativeByteTreeRef{ProducerRootId: childDigest("5"), ReceiptDigest: bytes.Repeat([]byte{marker}, 32),
					Manifest: &pb.Ref{Digest: bytes.Repeat([]byte{marker + 1}, 32), Length: 64}, ContentBytes: uint64(len(data))}},
			map[string]any{"asset_ref": spelled, "digest": spelled, "kind": "image", "media_type": "image/png", "size_bytes": len(data)}
	}
	imageEntry, imageRow := entry("image", image, 1)
	extraEntry, _ := entry("extra", extra, 3)
	_, maskRow := entry("mask", []byte("never sent"), 5)
	inline, err := json.Marshal(map[string]any{"image": imageRow, "mask": maskRow})
	must(t, err)
	specDigest, err := canonical.Spell(receipt.InvocationSpecDigest)
	must(t, err)
	body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: request.ID, AttemptOrdinal: 1,
		InvocationSpecDigest: specDigest, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED,
		Result:         &pb.ResultEnvelope{InlineResult: inline},
		OutputManifest: &pb.OutputManifest{Outputs: []*pb.OutputEntry{imageEntry, extraEntry}}})
	must(t, err)
	machine := &fileMachine{
		finishedMachine: finishedMachine{
			state: &pb.MachineExecutionState{RequestId: request.ID, WorkerId: podWorkerID, WorkerBootId: podBootID,
				ExecutionWorkspaceId: "workspace", Generation: 1, AttemptOrdinal: 1, State: "succeeded"},
			outcome: &pb.AttemptOutcome{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: receipt.InvocationSpecDigest,
				OutcomeId: "tolerance-outcome", OutcomeDigest: digest, OutcomeCanonicalBytes: body},
		},
		objects: map[string][]byte{string(imageEntry.Digest): image, string(extraEntry.Digest): extra},
	}

	pod := &fakePod{controlKey: public, machine: machine}
	connection, certPath := startFakePod(t, root, pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	hub := newFakeRentalHub(t, 0)
	hub.publishListing()
	hub.add(podRental, "collector")
	hub.set(podRental, "requested_accelerator_model", "fake-4090")
	row := records.Rental{ID: podRental, MachineName: "collector", State: "ready", SKU: "cpu", AcceleratorModel: "fake-4090",
		AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: hub.server.URL, Address: connection.Addr,
		MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}
	fatal(t, rental.Attach(layout, store, row, string(cert), connection.Media.Token, identity))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	startDaemonProcess(t, root)

	code, out := runCozy(t, root, "run", "watch", request.ID)
	if code != 0 {
		t.Fatalf("the run failed on its machine's outputs [exit %d]: %s\n%s", code, out, tail(filepath.Join(root, "daemon.log")))
	}
	for _, want := range []string{`warning: the machine returned output "extra", which the package does not declare; ignored`,
		`warning: output "mask" failed: the machine did not return it`} {
		if !strings.Contains(out, want) {
			t.Fatalf("watch did not show %q:\n%s", want, out)
		}
	}
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if link == nil || !link.Collected {
		t.Fatal("the result was not collected")
	}
	outputs, problem := store.VisibleOutputs(request.ID)
	fatal(t, problem)
	if len(outputs) != 1 || outputs[0].OutputID != "image" {
		t.Fatalf("collected outputs = %+v, want only the declared image", outputs)
	}
	received, err := os.ReadFile(outputs[0].Path)
	must(t, err)
	if !bytes.Equal(received, image) {
		t.Fatal("the collected image differs from the machine's bytes")
	}
}
