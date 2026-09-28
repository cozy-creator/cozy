package producttest

// THE HOST LANE (proto-025), observed end to end against a second implementation of the pod
// side: real TLS, real protocol bytes, a real PodHost service that verifies the owner's
// ClaimProof exactly as pod-supervisor does. Nothing in the orchestrator knows this pod exists.

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/mediawire"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

const (
	podWorkerID = "wrk-pod-1"
	podBootID   = "boot-pod-1"
	podRental   = "pr-test-0001"
)

// fakePod hosts WorkerControl and PodHost on one pinned TLS listener, the way pod-supervisor
// does. It verifies the owner's Ed25519 ClaimProof on both services against the control key
// the rental's auth document would carry, records what crossed, and answers minimally.
type fakePod struct {
	// releases are the published releases this pod's machine reads at its own Hub.
	releases         map[string]*pb.DescribedRelease
	keepalive        func(*pb.KeepRentalAliveRequest) (*pb.KeepRentalAliveResult, error)
	identity         string
	noSeats          bool
	mediaReservation func(int64, int) error
	mediaRequest     func(http.ResponseWriter, *http.Request) bool
	// sourceRuntime delegates checkpoint metadata/bytes to an actual installed Runtime.
	sourceRuntime pb.RuntimePreparationClient
	weightsReady  func(*pb.WeightsIntentReadyRequest) (*pb.WeightsHostAck, error)
	protocolInfo  func(context.Context, *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error)
	watchProgress func(*pb.ProgressOpen, pb.WorkerControl_WatchProgressServer) error
	derivedRetain func(context.Context, *pb.DerivedRetentionCall) (*pb.DerivedRetentionResult, error)
	// artifactTransfer answers `cozy run upload`'s retained-artifact commands.
	artifactTransfer func(context.Context, *pb.NativeArtifactTransferCall) (*pb.NativeArtifactTransferStatus, error)
	derivedRelease   func(context.Context, *pb.DerivedRetentionCall) (*pb.DerivedRetentionResult, error)
	resultRelease    func(context.Context, *pb.DerivedResultReleaseCall) (*pb.DerivedResultReleaseResult, error)
	controlDefaults
	pb.UnimplementedPodHostServer
	controlKey ed25519.PublicKey
	// rebooted is the boot a restarted pod came back on; nil keeps its first boot.
	rebooted   atomic.Pointer[string]
	wireMinor  uint32 // zero uses the current protocol; tests can negotiate an older peer
	leafDigest []byte
	// mutateHostDigest is the red arm: the host document's digest stops hashing its bytes.
	mutateHostDigest bool

	// latch is the materialization fault this pod keeps against every placement_set it
	// accepts (the Runtime's own shape: accepted, never converged, the placement ABSENT
	// and OFFLINE, the fault replayed on every report). Nil reports nothing.
	latch *pb.Fault
	// serve makes every accepted placement_set converge at once: STAGED, DISPATCHABLE,
	// every entrypoint advertised, one free seat. Offers are recorded, never answered.
	serve bool
	// jobReady advertises the independent job seat after accepting a JobDirective.
	jobReady bool
	// snapshotHeld replays actual retained attempt identities during reconnect tests.
	snapshotHeld []*pb.HeldAttempt
	hostHeld     func() []*pb.HeldAttempt
	// localJobOnly supplies a prepared interface without any serving entrypoints.
	localJobOnly   bool
	privatePrepare func(*pb.PreparePrivatePlacementCall, grpc.ServerStreamingServer[pb.PrepareEvent]) error
	localPrepare   func(*pb.PrepareLocalPackageCall, grpc.ServerStreamingServer[pb.PrepareEvent]) error
	localUpload    func(grpc.BidiStreamingServer[pb.LocalPackageUploadFrame, pb.LocalPackageFileStatus]) error
	// onJobReady can delay and sequence the independent peer's readiness facts.
	onJobReady func(*pb.WorkerFrame, func(*pb.WorkerFrame) error) error
	// answerOffer supplies a protocol outcome when a test exercises settlement.
	answerOffer func(*pb.AttemptOffer) (*pb.AttemptOutcome, error)
	// machine answers Runtime-owned machine execution observation and collection.
	machine machineExecutionPeer
	// onFrame lets a product test delegate selected frames to a real Runtime peer.
	onFrame func(*pb.RecordOwnerFrame, func(*pb.WorkerFrame) error) (bool, error)
	// preparedPlacement supplies a complete second-implementation placement for
	// tests of modeled callable routing. The normal package preparation path still runs.
	preparedPlacement func([]byte, string, string) *pb.Placement
	// slots is the advertised seat count while serving; zero means one.
	slots uint32
	// reobserve re-sends the serving report for the last desired state, as the Runtime's
	// report cadence would after its seats change.
	reobserve func() error
	// deviceCount is the width this pod's ClaimAck reports — how many accelerators the
	// worker actually found. Zero reports one card, which is what every pod was until
	// wide products became buyable. A test sets it to say what the PROVIDER delivered,
	// which the owner holds against the width the rental was paid for (cl-179).
	deviceCount uint32
	// downloadSamples is how many DOWNLOADING events the prepare stream reports before
	// it finishes. One is what a pod host that only reports the stage boundary sends;
	// more is what one that reports while the bytes are moving sends. It exists so the
	// phase lane is proven on a stream that ADVANCES, not only on one that is wired.
	downloadSamples int
	prepareEvent    func(*pb.PrepareEvent)
	// retainPrepared answers an identical call with the PREPARED event it already sent,
	// as the pod host does (workerhost.Host.prepare): keyed by the exact call bytes.
	retainPrepared bool
	retained       map[string]*pb.PrepareEvent

	mu                 sync.Mutex
	acks               []*pb.SnapshotAck
	desired            []*pb.DesiredWorkerState
	prepares           []*pb.PreparePackageSetCall
	protocolReads      int // each dial of the pod probes it once
	localPrepares      []*pb.PrepareLocalPackageCall
	uploads            []*pb.LocalPackageFileRef
	uploadEnds         []codes.Code
	uploadCalls        int
	lanes              []string // the order lanes were used: fetch, prepare_private, placement_set
	prepareCodes       []codes.Code
	prepareUnavailable int
	// refusePrepare maps a package to the Runtime refusal its own preparation answers.
	refusePrepare map[string]*pb.PrepareEvent
	// stalePlacementSets answers that many placement-set desires with Runtime's
	// placement_set_reprepare_required fault instead of applying them.
	stalePlacementSets int
	hostDigest         []byte
	preparedSet        []byte
	preparedDig        []byte
	reports            map[string]int // fault text -> reports that carried it
	offers             []*pb.AttemptOffer
	finalizations      []*pb.WeightsFinalizeRequest
	// stagedJobBuild is what THIS pod wrote into the job plan records it staged during
	// preparation — `build_id`, its own placement's environment identity, exactly as
	// `package_prepare.py::_stage_job_plans` writes it. A JobDirective naming anything
	// else is what `session.py::apply_job_directive` refuses as `job_plan_mismatch`.
	stagedJobBuild string
	jobDirectives  []*pb.JobDirective
}

func (p *fakePod) ProtocolInfo(ctx context.Context, request *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error) {
	p.mu.Lock()
	p.protocolReads++
	p.mu.Unlock()
	if p.protocolInfo != nil {
		return p.protocolInfo(ctx, request)
	}
	version := p.wireMinor
	if version == 0 {
		version = pb.WireMinor
	}
	return &pb.ProtocolInfoResult{WireMinor: version, MinimumWireMinor: min(version, pb.MinCompatibleWireMinor)}, nil
}

// served is the serve arm's ObservedWorkerState: the exact set accepted and converged,
// the one placement staged and dispatchable under every binding it names.
func (p *fakePod) workerID() string {
	if p.identity == "" {
		return podWorkerID
	}
	return "wrk-" + p.identity
}
func (p *fakePod) bootID() string {
	if boot := p.rebooted.Load(); boot != nil {
		return *boot
	}
	if p.identity == "" {
		return podBootID
	}
	return "boot-" + p.identity
}
func (p *fakePod) rentalID() string {
	if p.identity == "" {
		return podRental
	}
	return "pr-" + p.identity
}

func (p *fakePod) served(d *pb.DesiredWorkerState, epoch uint64) *pb.WorkerFrame {
	set := d.GetPlacementSet()
	doc, err := canonical.Read(set.PlacementSetCanonicalBytes, &pb.PlacementSet{})
	if err != nil {
		return nil
	}
	var placements []*pb.PlacementStatus
	for _, placement := range doc.List("placements") {
		row := &pb.PlacementStatus{
			PlacementId:     placement.Str("placement_id"),
			Materialization: pb.MaterializationState_MATERIALIZATION_STATE_STAGED,
			Serving:         pb.ServingState_SERVING_STATE_DISPATCHABLE, ExecutorEpoch: 1,
			PlacementSetDigest: set.PlacementSetDigest,
			InstallationId:     placement.Str("installation_id"),
		}
		for _, entrypoint := range placement.List("entrypoints") {
			if digest, err := canonical.Raw(entrypoint.Str("entrypoint_binding_digest")); err == nil {
				row.DispatchableBindingDigests = append(row.DispatchableBindingDigests, digest)
			}
		}
		placements = append(placements, row)
	}
	p.mu.Lock()
	seats := uint32(1)
	if p.slots > 1 {
		seats = p.slots
	}
	if p.noSeats {
		seats = 0
	}
	p.mu.Unlock()
	return &pb.WorkerFrame{Msg: &pb.WorkerFrame_ObservedState{ObservedState: &pb.ObservedWorkerState{
		RecordOwnerEpoch: d.RecordOwnerEpoch, ControlStreamEpoch: epoch, WorkerBootId: p.bootID(),
		AcceptedDesiredStateRevision: d.Revision, ConvergedRevision: d.Revision,
		AcceptedPlacementSetDigest: set.PlacementSetDigest,
		WorkerPhase:                pb.WorkerPhase_WORKER_PHASE_ONLINE, AppliedWireMinor: pb.WireMinor,
		AdmissionState: pb.AdmissionState_ADMISSION_STATE_OPEN, AdmissionEpoch: 7, AvailableAttemptSlots: seats,
		Placements: placements,
	}}}
}

// report is the latch arm's ObservedWorkerState: the desired revision accepted and not
// converged, the placement absent, and the fault, exactly as a Runtime that latched a
// materialization refusal reports every ReportCadence.
func (p *fakePod) report(d *pb.DesiredWorkerState, epoch uint64) *pb.WorkerFrame {
	p.mu.Lock()
	defer p.mu.Unlock()
	fault := p.latch
	if fault == nil {
		return nil
	}
	if p.reports == nil {
		p.reports = map[string]int{}
	}
	p.reports[fault.Reason+"/"+fault.Detail]++
	set := d.GetPlacementSet()
	doc, err := canonical.Read(set.PlacementSetCanonicalBytes, &pb.PlacementSet{})
	if err != nil {
		return nil
	}
	var placements []*pb.PlacementStatus
	for _, placement := range doc.List("placements") {
		placements = append(placements, &pb.PlacementStatus{
			PlacementId:        placement.Str("placement_id"),
			Materialization:    pb.MaterializationState_MATERIALIZATION_STATE_ABSENT,
			Serving:            pb.ServingState_SERVING_STATE_OFFLINE,
			PlacementSetDigest: set.PlacementSetDigest,
		})
	}
	return &pb.WorkerFrame{Msg: &pb.WorkerFrame_ObservedState{ObservedState: &pb.ObservedWorkerState{
		RecordOwnerEpoch: d.RecordOwnerEpoch, ControlStreamEpoch: epoch, WorkerBootId: p.bootID(),
		AcceptedDesiredStateRevision: d.Revision, AcceptedPlacementSetDigest: set.PlacementSetDigest,
		WorkerPhase: pb.WorkerPhase_WORKER_PHASE_ONLINE, AppliedWireMinor: pb.WireMinor,
		AdmissionState: pb.AdmissionState_ADMISSION_STATE_CLOSED, AdmissionEpoch: 4,
		Placements: placements,
		Faults:     []*pb.Fault{{Kind: fault.Kind, Subject: placements[0].PlacementId, Reason: fault.Reason, Detail: fault.Detail}},
	}}}
}

func (p *fakePod) verifyClaim(claim *pb.Claim, stream bool) error {
	if claim == nil {
		return status.Error(codes.Unauthenticated, "no Claim")
	}
	if !stream && claim.ControlStreamEpoch != 0 {
		return status.Error(codes.FailedPrecondition, "a host call is not stream-scoped")
	}
	proof, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: claim.RecordOwnerEpoch,
		WorkerId: p.workerID(), WorkerBootId: p.bootID(), WorkerTlsCertificateDigest: p.leafDigest})
	if err != nil {
		return err
	}
	if len(claim.Proof) != ed25519.SignatureSize || !ed25519.Verify(p.controlKey, proof, claim.Proof) {
		return status.Error(codes.Unauthenticated, "the ClaimProof does not verify")
	}
	return nil
}

func (p *fakePod) Control(stream grpc.BidiStreamingServer[pb.RecordOwnerFrame, pb.WorkerFrame]) error {
	var sendMu sync.Mutex
	send := func(frame *pb.WorkerFrame) error {
		sendMu.Lock()
		defer sendMu.Unlock()
		return stream.Send(frame)
	}
	for {
		frame, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		if p.onFrame != nil {
			handled, err := p.onFrame(frame, send)
			if err != nil {
				return err
			}
			if handled {
				continue
			}
		}
		switch m := frame.Msg.(type) {
		case *pb.RecordOwnerFrame_Claim:
			if err := p.verifyClaim(m.Claim, true); err != nil {
				return err
			}
			minor := p.wireMinor
			if minor == 0 {
				minor = pb.WireMinor
			}
			if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: &pb.ClaimAck{
				RecordOwnerEpoch: m.Claim.RecordOwnerEpoch, ControlStreamEpoch: 1, WorkerBootId: p.bootID(),
				Accepted: true, WireMinor: minor, WorkerId: p.workerID(), WorkerInstanceId: "inst-pod-1",
				Resources: &pb.WorkerResources{Backend: "cuda", DeviceName: "fake-4090",
					DeviceCount: max(p.deviceCount, 1), DeviceMemoryTotalBytes: 24 << 30},
			}}}); err != nil {
				return err
			}
			body, digest, err := canonical.Identity(&pb.WorkerSnapshotBody{
				WorkerPhase: pb.WorkerPhase_WORKER_PHASE_ONLINE, AdmissionEpoch: 1,
				AdmissionState: pb.AdmissionState_ADMISSION_STATE_CLOSED, AvailableAttemptSlots: 2,
				HeldAttempts: p.snapshotHeld})
			if err != nil {
				return err
			}
			var retained []*pb.HeldAttempt
			if p.hostHeld != nil {
				retained = p.hostHeld()
			}
			hostBody, hostDigest, err := canonical.Identity(&pb.HostSnapshotBody{HeldOutcomes: retained,
				WeightsTransactions: []*pb.WeightsTransactionStatus{{
					WeightsTransactionId: "sha256:" + strings.Repeat("ab", 32), RequestId: "job-prior",
					AttemptOrdinal: 1, InvocationSpecDigest: "sha256:" + strings.Repeat("cd", 32),
					OutputSlot: "model", WriterEpoch: 1,
					State:                     pb.WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_INTENT,
					TensorfsDeclarationDigest: bytes.Repeat([]byte{0xef}, 32)}}})
			if err != nil {
				return err
			}
			if p.mutateHostDigest {
				hostDigest = bytes.Repeat([]byte{0x11}, 32)
			}
			p.mu.Lock()
			p.hostDigest = hostDigest
			p.mu.Unlock()
			if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_Snapshot{Snapshot: &pb.WorkerSnapshot{
				RecordOwnerEpoch: m.Claim.RecordOwnerEpoch, ControlStreamEpoch: 1, WorkerBootId: p.bootID(),
				SnapshotId: "snp-pod-1", SnapshotDigest: digest, SnapshotCanonicalBytes: body,
				HostSnapshotDigest: hostDigest, HostSnapshotCanonicalBytes: hostBody,
			}}}); err != nil {
				return err
			}
		case *pb.RecordOwnerFrame_SnapshotAck:
			p.mu.Lock()
			p.acks = append(p.acks, m.SnapshotAck)
			p.mu.Unlock()
		case *pb.RecordOwnerFrame_DesiredState:
			p.mu.Lock()
			p.desired = append(p.desired, m.DesiredState)
			p.lanes = append(p.lanes, "placement_set")
			if job := m.DesiredState.GetJob(); job != nil {
				p.jobDirectives = append(p.jobDirectives, job)
			}
			latched, serve := p.latch != nil, p.serve
			stale := p.stalePlacementSets > 0 && m.DesiredState.GetPlacementSet() != nil
			if stale {
				p.stalePlacementSets--
			}
			p.mu.Unlock()
			if stale {
				d := m.DesiredState
				if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ObservedState{ObservedState: &pb.ObservedWorkerState{
					RecordOwnerEpoch: d.RecordOwnerEpoch, ControlStreamEpoch: 1, WorkerBootId: p.bootID(),
					WorkerPhase: pb.WorkerPhase_WORKER_PHASE_ONLINE, AppliedWireMinor: pb.WireMinor,
					Faults: []*pb.Fault{{Kind: pb.FaultKind_FAULT_KIND_ARTIFACT_FETCH_FAILED, Subject: "revision " + strconv.FormatUint(d.Revision, 10),
						Reason: "placement_set_reprepare_required", Detail: "this placement set was prepared by an older Runtime"}},
				}}}); err != nil {
					return err
				}
				continue
			}
			if p.jobReady && m.DesiredState.GetJob() != nil {
				d := m.DesiredState
				ready := &pb.WorkerFrame{Msg: &pb.WorkerFrame_ObservedState{ObservedState: &pb.ObservedWorkerState{
					RecordOwnerEpoch: d.RecordOwnerEpoch, ControlStreamEpoch: 1, WorkerBootId: p.bootID(),
					AcceptedDesiredStateRevision: d.Revision, ConvergedRevision: d.Revision,
					WorkerPhase: pb.WorkerPhase_WORKER_PHASE_ONLINE, AppliedWireMinor: pb.WireMinor,
					AdmissionState: pb.AdmissionState_ADMISSION_STATE_OPEN, AdmissionEpoch: 7,
					AvailableAttemptSlots: 1, JobCapacity: &pb.JobCapacity{JobsAvailable: 1},
				}}}
				var err error
				if p.onJobReady != nil {
					err = p.onJobReady(ready, send)
				} else {
					err = send(ready)
				}
				if err != nil {
					return err
				}
			}
			if serve && m.DesiredState.GetPlacementSet() != nil {
				desired := m.DesiredState
				p.mu.Lock()
				p.reobserve = func() error {
					if frame := p.served(desired, 1); frame != nil {
						return send(frame)
					}
					return nil
				}
				p.mu.Unlock()
				if frame := p.served(m.DesiredState, 1); frame != nil {
					if err := send(frame); err != nil {
						return err
					}
				}
			}
			if latched && m.DesiredState.GetPlacementSet() != nil {
				go func(d *pb.DesiredWorkerState) {
					for {
						select {
						case <-stream.Context().Done():
							return
						case <-time.After(25 * time.Millisecond):
						}
						if frame := p.report(d, 1); frame != nil {
							if err := send(frame); err != nil {
								return
							}
						}
					}
				}(m.DesiredState)
			}
		case *pb.RecordOwnerFrame_AttemptOffer:
			p.mu.Lock()
			p.offers = append(p.offers, m.AttemptOffer)
			p.mu.Unlock()
			if p.answerOffer != nil {
				outcome, err := p.answerOffer(m.AttemptOffer)
				if err != nil {
					return err
				}
				if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptOutcome{AttemptOutcome: outcome}}); err != nil {
					return err
				}
			}
		case *pb.RecordOwnerFrame_WeightsFinalizeRequest:
			p.mu.Lock()
			p.finalizations = append(p.finalizations, m.WeightsFinalizeRequest)
			p.mu.Unlock()
		}
	}
}

// PrepareLocalPackage is the pod host's private lane: the set names wheels the pod
// already holds verified, and the prepared placement is a development one carrying the
// exact project wheel and the local revision digest, as Runtime authors it.
func (p *fakePod) LocalPackageUpload(stream grpc.BidiStreamingServer[pb.LocalPackageUploadFrame, pb.LocalPackageFileStatus]) error {
	if p.localUpload != nil {
		return p.localUpload(stream)
	}
	frame, err := stream.Recv()
	if err != nil {
		return err
	}
	header := frame.GetHeader()
	if header == nil || header.File == nil {
		return status.Error(codes.InvalidArgument, "header required")
	}
	if err := p.verifyClaim(header.Claim, false); err != nil {
		return err
	}
	p.mu.Lock()
	uploadIndex := p.uploadCalls
	p.uploadCalls++
	var uploadEnd codes.Code
	if uploadIndex < len(p.uploadEnds) {
		uploadEnd = p.uploadEnds[uploadIndex]
	}
	p.mu.Unlock()
	if uploadEnd != codes.OK {
		return status.Error(uploadEnd, "invalid local package upload header: filename refused")
	}
	file := header.File
	if file.Length == 0 || file.Length > 512<<20 {
		return status.Error(codes.InvalidArgument, "file size")
	}
	state := &pb.LocalPackageFileStatus{OperationId: header.OperationId,
		Digest: file.Digest, Filename: file.Filename, Length: file.Length,
		State: pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING}
	if err := stream.Send(state); err != nil {
		return err
	}
	hash := sha256.New()
	for state.ReceivedBytes < file.Length {
		frame, err := stream.Recv()
		if err != nil {
			return err
		}
		chunk := frame.GetChunk()
		if chunk == nil || chunk.Offset != state.ReceivedBytes || len(chunk.Data) == 0 ||
			len(chunk.Data) > 1<<20 || uint64(len(chunk.Data)) > file.Length-state.ReceivedBytes {
			return status.Error(codes.InvalidArgument, "chunk offset or size")
		}
		_, _ = hash.Write(chunk.Data)
		state.ReceivedBytes += uint64(len(chunk.Data))
		if state.ReceivedBytes == file.Length {
			if !bytes.Equal(hash.Sum(nil), file.Digest) {
				return status.Error(codes.DataLoss, "wheel digest")
			}
			state.State = pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED
		}
		if err := stream.Send(state); err != nil {
			return err
		}
	}
	p.mu.Lock()
	if len(p.uploads) == 0 {
		p.lanes = append(p.lanes, "upload")
	}
	p.uploads = append(p.uploads, file)
	p.mu.Unlock()
	return nil
}

func (p *fakePod) PrepareLocalPackage(call *pb.PrepareLocalPackageCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
	if p.localPrepare != nil {
		return p.localPrepare(call, stream)
	}
	if err := p.verifyClaim(call.Claim, false); err != nil {
		return err
	}
	selected := call.LocalPackageSet
	if selected == nil || selected.Package == nil || len(selected.Files) == 0 {
		return status.Error(codes.InvalidArgument, "no local package set")
	}
	var project *pb.LocalPackageFileRef
	var total uint64
	for _, file := range selected.Files {
		total += file.Length
		// Wire 30: no kind row travels; the project wheel is the row whose filename
		// names the package's own distribution and release.
		if strings.HasPrefix(file.Filename, "weightless-"+selected.Package.Release+"-") {
			project = file
		}
	}
	if project == nil {
		return status.Error(codes.FailedPrecondition, "the local package set names no project wheel")
	}
	p.mu.Lock()
	p.localPrepares = append(p.localPrepares, call)
	p.lanes = append(p.lanes, "prepare_private")
	p.mu.Unlock()
	entrypoints := []*pb.Entrypoint{podEntrypoint("tile")}
	if p.localJobOnly {
		entrypoints = nil
	}
	placement := &pb.Placement{
		PlacementId: "package-" + selected.OperationId,
		PackageMode: &pb.Placement_Development{Development: &pb.DevelopmentPackage{
			Package: selected.Package.Package, Release: selected.Package.Release,
			InstallationId: selected.Package.InstallationId,
		}},
		InstallationId:   selected.Package.InstallationId,
		PackageInterface: fixturePackageInterface,
		BindingsDigest:   bytes.Repeat([]byte{0x25}, 32),
		Entrypoints:      entrypoints,
	}
	sealPodBindings(placement)
	setBytes, setDigest, err := canonical.Identity(&pb.PlacementSet{Placements: []*pb.Placement{placement}})
	if err != nil {
		return err
	}
	p.mu.Lock()
	p.preparedSet, p.preparedDig = setBytes, setDigest
	p.mu.Unlock()
	for _, event := range []*pb.PrepareEvent{
		{Stage: pb.PrepareStage_PREPARE_STAGE_RESOLVED, TotalBytes: total, TransferredBytes: total},
		{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARING, TotalBytes: total, TransferredBytes: total},
		{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED, TotalBytes: total, TransferredBytes: total,
			PlacementSet: &pb.DesiredPlacementSet{PlacementSetDigest: setDigest, PlacementSetCanonicalBytes: setBytes}},
	} {
		if err := stream.Send(event); err != nil {
			return err
		}
	}
	return nil
}

// PreparePackageSet is the published package lane the way the pod actually runs it:
// pod-supervisor downloads what the desired set names and the Runtime prepares EXACTLY
// ONE package per call — a set naming more than one is refused with the Runtime's own
// verdict (run 146's failure text), and the prepared single placement is derived from
// the set the way the Runtime derives it, placement_id seeded from its exact bytes.
func (p *fakePod) PreparePackageSet(call *pb.PreparePackageSetCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
	if err := p.verifyClaim(call.Claim, false); err != nil {
		p.mu.Lock()
		p.prepareCodes = append(p.prepareCodes, status.Code(err))
		p.mu.Unlock()
		return err
	}
	if call.PackageSet == nil || len(call.PackageSet.DownloadDelegation) == 0 {
		return status.Error(codes.InvalidArgument, "no desired download set")
	}
	// RED ARM for the delegation deletion: the pod is handed no credential.
	if len(call.PackageSet.DownloadDelegationSignature) != 0 {
		return status.Error(codes.InvalidArgument, "the download set still carries a signature")
	}
	downloadSet, err := canonical.Read(call.PackageSet.DownloadDelegation, &pb.DownloadDelegation{})
	if err != nil {
		return status.Errorf(codes.InvalidArgument, "download set: %v", err)
	}
	p.mu.Lock()
	p.prepares = append(p.prepares, call)
	unavailable := p.prepareUnavailable > 0
	if unavailable {
		p.prepareUnavailable--
	}
	p.mu.Unlock()
	if unavailable {
		return status.Error(codes.Unavailable, "Tensorhub is restarting")
	}
	total := uint64(18_874_368)
	selected := downloadSet.List("packages")
	if len(selected) != 1 {
		for _, event := range []*pb.PrepareEvent{
			{Stage: pb.PrepareStage_PREPARE_STAGE_RESOLVED, TotalBytes: total},
			{Stage: pb.PrepareStage_PREPARE_STAGE_REFUSED, TotalBytes: total,
				SafeCode: "runtime_preparation_failed", SafeDetail: "package_prepare_selection_invalid: " +
					"exactly one package and a model array are required"},
		} {
			if err := stream.Send(event); err != nil {
				return err
			}
		}
		return nil
	}
	name, release := selected[0].Str("package"), selected[0].Str("release")
	// The Runtime binds only the prepared package's own slots (package_prepare
	// _selected_models): a row naming another package is refused.
	for _, model := range downloadSet.List("models") {
		if model.Str("package") != name {
			return stream.Send(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_REFUSED, TotalBytes: total,
				SafeCode: "runtime_preparation_failed", SafeDetail: "package_prepare_model_selection_mismatch: package"})
		}
	}
	key, err := proto.MarshalOptions{Deterministic: true}.Marshal(call)
	if err != nil {
		return err
	}
	p.mu.Lock()
	refusal := p.refusePrepare[name]
	answer := p.retained[string(key)]
	p.mu.Unlock()
	if answer != nil {
		return stream.Send(answer)
	}
	if refusal != nil {
		for _, event := range []*pb.PrepareEvent{
			{Stage: pb.PrepareStage_PREPARE_STAGE_RESOLVED, TotalBytes: total},
			{Stage: pb.PrepareStage_PREPARE_STAGE_REFUSED, TotalBytes: total, SafeCode: refusal.SafeCode, SafeDetail: refusal.SafeDetail},
		} {
			if err := stream.Send(event); err != nil {
				return err
			}
		}
		return nil
	}
	distribution := name[strings.IndexByte(name, '/')+1:]
	placement := podPlacement(call.PackageSet.DownloadDelegation, name, release, distribution)
	if p.preparedPlacement != nil {
		placement = p.preparedPlacement(call.PackageSet.DownloadDelegation, name, release)
	}
	sealPodBindings(placement)
	setBytes, setDigest, err := canonical.Identity(&pb.PlacementSet{Placements: []*pb.Placement{
		placement,
	}})
	if err != nil {
		return err
	}
	prepared, err := canonical.Read(setBytes, &pb.PlacementSet{})
	if err != nil {
		return err
	}
	p.mu.Lock()
	p.preparedSet, p.preparedDig = setBytes, setDigest
	// The Runtime stages this package's job plan records inside THIS call, before the
	// owner has united anything, so the only build identity it can write is its own
	// placement's.
	p.stagedJobBuild = prepared.List("placements")[0].Str("installation_id")
	p.mu.Unlock()
	samples := max(p.downloadSamples, 1)
	events := []*pb.PrepareEvent{{Stage: pb.PrepareStage_PREPARE_STAGE_RESOLVED, TotalBytes: total}}
	for i := 1; i <= samples; i++ {
		events = append(events, &pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_DOWNLOADING,
			TotalBytes: total, TransferredBytes: total / 2 * uint64(i) / uint64(samples)})
	}
	events = append(events,
		&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARING, TotalBytes: total, TransferredBytes: total},
		&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED, TotalBytes: total, TransferredBytes: total,
			PlacementSet: &pb.DesiredPlacementSet{PlacementSetDigest: setDigest, PlacementSetCanonicalBytes: setBytes}})
	if p.machine != nil {
		// A Runtime that owns executions names the installation it prepared beside the set.
		events[len(events)-1].InstalledPackage = &pb.InstalledPackage{InstallationId: placement.InstallationId,
			Package: name, Release: release, PackageInterface: placement.PackageInterface}
	}
	if p.retainPrepared {
		p.mu.Lock()
		if p.retained == nil {
			p.retained = map[string]*pb.PrepareEvent{}
		}
		p.retained[string(key)] = events[len(events)-1]
		p.mu.Unlock()
	}
	for i, event := range events {
		if p.prepareEvent != nil {
			p.prepareEvent(event)
		}
		if err := stream.Send(event); err != nil {
			return err
		}
		if i < len(events)-1 && p.downloadSamples > 0 {
			// A measurable interval between samples. The owner computes a rate from the
			// gap between two reports and declines to compute one when there is no gap,
			// so a pod that reports instantly proves nothing about a rate.
			time.Sleep(2 * time.Millisecond)
		}
	}
	return nil
}

// podPlacement is the one placement a prepared package download set yields, on the
// Runtime's own derivations: `package-` + the first 24 hex of sha256 over the exact
// download-set bytes, and one deterministic entrypoint per package so a test can name the
// plan it will dispatch (podPlanID).
func podPlacement(downloadSet []byte, name, release, distribution string) *pb.Placement {
	seed := sha256.Sum256(downloadSet)
	_ = distribution
	return &pb.Placement{
		PlacementId: "package-" + hex.EncodeToString(seed[:])[:24],
		PackageMode: &pb.Placement_Package{Package: &pb.PackageSelection{
			Package: name, Release: release}},
		InstallationId:   "fixture-" + hex.EncodeToString(seed[:])[:24],
		PackageInterface: fixturePackageInterface,
		BindingsDigest:   sha256Of([]byte("bindings:" + name)),
		Entrypoints:      []*pb.Entrypoint{podEntrypoint("tile")},
	}
}

func podEntrypoint(name string) *pb.Entrypoint {
	raw, _ := canonical.Write(map[string]canonical.Value{"name": name, "slots": []canonical.Value{}})
	return &pb.Entrypoint{Name: name, EntrypointBindingDigest: canonical.Digest(raw)}
}

func sealPodBindings(placement *pb.Placement) {
	raw, _, _ := canonical.Identity(&pb.PlacementSet{Placements: []*pb.Placement{placement}})
	set, _ := canonical.Read(raw, &pb.PlacementSet{})
	row := set.List("placements")[0]
	for index, entry := range row.List("entrypoints") {
		slots := canonical.Value([]canonical.Value{})
		if value, exists := entry["slots"]; exists {
			slots = value
		}
		raw, _ := canonical.Write(map[string]canonical.Value{"name": entry.Str("name"), "slots": slots})
		placement.Entrypoints[index].EntrypointBindingDigest = canonical.Digest(raw)
	}
	raw, _, _ = canonical.Identity(&pb.PlacementSet{Placements: []*pb.Placement{placement}})
	set, _ = canonical.Read(raw, &pb.PlacementSet{})
	row = set.List("placements")[0]
	fields := map[string]canonical.Value{"entrypoints": []canonical.Value{}, "models": []canonical.Value{}}
	for field := range fields {
		if value, exists := row[field]; exists {
			fields[field] = value
		}
	}
	raw, _ = canonical.Write(fields)
	placement.BindingsDigest = canonical.Digest(raw)
}

// startFakePod mints the pod leaf, binds the pinned listener, and serves the media health
// route the owner dials before it will attach.
func startFakePod(t *testing.T, root string, pod *fakePod) (*orchestrator.WorkerConnection, string) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	must(t, err)
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: workertls.ServerName},
		NotBefore: time.Now().Add(-time.Minute), NotAfter: time.Now().Add(time.Hour),
		KeyUsage: x509.KeyUsageDigitalSignature, ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames: []string{workertls.ServerName}, IPAddresses: []net.IP{net.IPv4(127, 0, 0, 1)},
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	must(t, err)
	pemPath := filepath.Join(root, "pod-leaf.pem")
	must(t, os.WriteFile(pemPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600))
	pin, err := workertls.LoadPin(pemPath)
	must(t, err)
	pod.leafDigest = pin.Digest()
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12,
		Certificates: []tls.Certificate{{Certificate: [][]byte{der}, PrivateKey: key}}})))
	pb.RegisterWorkerControlServer(server, pod)
	pb.RegisterPodHostServer(server, pod)
	listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow the test's POD binds; the product dials
	must(t, err)
	go server.Serve(listener)
	t.Cleanup(server.Stop)
	rev := mediawire.ContractRev
	mediaRoot := filepath.Join(root, "pod-media")
	var mediaWrites sync.Mutex
	reserved := map[string]bool{}
	mediaPlane := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if pod.mediaRequest != nil && pod.mediaRequest(w, r) {
			return
		}
		mediaWrites.Lock()
		defer mediaWrites.Unlock()
		switch {
		case r.URL.Path == "/v1/health":
			_ = json.NewEncoder(w).Encode(mediawire.Health{Service: mediawire.Service, ContractRev: &rev})
		case r.Method == http.MethodPut && strings.HasPrefix(r.URL.Path, "/v1/attempts/") && strings.Contains(r.URL.Path, "/inputs/"):
			parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/v1/attempts/"), "/inputs/")
			if len(parts) != 2 || !reserved[parts[0]] {
				t.Errorf("input upload arrived without its attempt reservation: %s", r.URL.Path)
				w.WriteHeader(http.StatusConflict)
				return
			}
			body, _ := io.ReadAll(r.Body)
			path := filepath.Join(mediaRoot, "inputs", strings.TrimPrefix(r.URL.Path, "/v1/attempts/"))
			_ = os.MkdirAll(filepath.Dir(path), 0o755)
			_ = os.WriteFile(path, body, 0o644)
			w.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(w).Encode(map[string]any{"path": path, "length": len(body), "digest": "sha256:" + hex.EncodeToString(sha256Of(body))})
		case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/v1/outputs/"):
			count, err := strconv.Atoi(r.URL.Query().Get("output_count"))
			if err != nil || count < 0 {
				t.Errorf("output reservation omitted its exact file count: %s", r.URL)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			if pod.mediaReservation != nil {
				bytes, err := strconv.ParseInt(r.URL.Query().Get("max_bytes"), 10, 64)
				if err != nil {
					t.Error(err)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				if err := pod.mediaReservation(bytes, count); err != nil {
					w.WriteHeader(http.StatusInsufficientStorage)
					return
				}
			}
			slot := strings.TrimPrefix(r.URL.Path, "/v1/outputs/")
			reserved[slot] = true
			dir := filepath.Join(mediaRoot, "outputs", slot)
			_ = os.MkdirAll(dir, 0o755)
			w.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(w).Encode(map[string]any{"dir": dir})
		case r.Method == http.MethodDelete:
			delete(reserved, strings.TrimPrefix(r.URL.Path, "/v1/attempts/"))
			_ = json.NewEncoder(w).Encode(map[string]any{})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	mediaPlane.TLS = &tls.Config{MinVersion: tls.VersionTLS12,
		Certificates: []tls.Certificate{{Certificate: [][]byte{der}, PrivateKey: key}}}
	mediaPlane.StartTLS()
	t.Cleanup(mediaPlane.Close)
	return &orchestrator.WorkerConnection{
		RentalID: pod.rentalID(), Addr: listener.Addr().String(), CACert: pemPath,
		WorkerID: pod.workerID(), WorkerBootID: pod.bootID(),
		// ONE PROVISIONED IDENTITY, TWO LISTENERS: the byte plane presents the SAME pinned
		// leaf as the control leg, which is the shape `rental.Resolver` resolves in
		// production — it hands the media client the rental's pinned certificate, so a
		// plain-http harness plane would be a leg this suite exercises and the daemon never
		// does.
		Media: &media.Spec{Addr: strings.TrimPrefix(mediaPlane.URL, "https://"),
			Token: secret.New("media-token"), CACert: pemPath},
	}, pemPath
}

// rentalWiring is the production entrypoint's rental hooks with a test key: the same
// ClaimProof/1 signature, and the same unsigned download-set document.
func rentalWiring(connection *orchestrator.WorkerConnection, signer ed25519.PrivateKey) func(*orchestrator.Options) {
	return func(o *orchestrator.Options) {
		o.Rentals = func(id string) (*orchestrator.RemoteTarget, *exit.Error) {
			if id != connection.RentalID {
				return nil, exit.New(exit.NotFound, "no rental %s", id)
			}
			return &orchestrator.RemoteTarget{Connection: connection}, nil
		}
		o.RentalClaimProof = func(c *orchestrator.WorkerConnection, epoch uint64) ([]byte, *exit.Error) {
			pin, err := workertls.LoadPin(c.CACert)
			if err != nil {
				return nil, exit.Internalf("%v", err)
			}
			body, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: epoch, WorkerId: c.WorkerID,
				WorkerBootId: c.WorkerBootID, WorkerTlsCertificateDigest: pin.Digest()})
			if err != nil {
				return nil, exit.Internalf("%v", err)
			}
			return ed25519.Sign(signer, body), nil
		}
	}
}

func waitUntil(t *testing.T, what string, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if ok() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func (p *fakePod) CheckpointPage(ctx context.Context, call *pb.CheckpointPageCall) (*pb.CheckpointPageResult, error) {
	if err := p.verifyClaim(call.GetClaim(), false); err != nil {
		return nil, err
	}
	if p.sourceRuntime == nil {
		return nil, status.Error(codes.Unimplemented, "no source Runtime")
	}
	return p.sourceRuntime.CheckpointPage(ctx, call.GetRequest())
}

func (p *fakePod) WeightsIntentReady(_ context.Context, call *pb.WeightsIntentReadyCall) (*pb.WeightsHostAck, error) {
	if err := p.verifyClaim(call.GetClaim(), false); err != nil {
		return nil, err
	}
	if p.weightsReady == nil {
		return nil, status.Error(codes.Unimplemented, "no weights Ready peer")
	}
	return p.weightsReady(call.GetRequest())
}
func (p *fakePod) CheckpointTransfer(ctx context.Context, call *pb.CheckpointTransferCall) (*pb.CheckpointTransferStatus, error) {
	if err := p.verifyClaim(call.GetClaim(), false); err != nil {
		return nil, err
	}
	if p.sourceRuntime == nil {
		return nil, status.Error(codes.Unimplemented, "no source Runtime")
	}
	return p.sourceRuntime.CheckpointTransfer(ctx, call.GetRequest())
}

func (p *fakePod) WatchProgress(open *pb.ProgressOpen, stream pb.WorkerControl_WatchProgressServer) error {
	if p.watchProgress != nil {
		return p.watchProgress(open, stream)
	}
	<-stream.Context().Done()
	return nil
}

func (p *fakePod) KeepRentalAlive(ctx context.Context, request *pb.KeepRentalAliveRequest) (*pb.KeepRentalAliveResult, error) {
	if err := p.verifyClaim(request.Claim, false); err != nil {
		return nil, err
	}
	if p.keepalive == nil {
		return nil, status.Error(codes.Unimplemented, "keepalive unavailable")
	}
	return p.keepalive(request)
}
