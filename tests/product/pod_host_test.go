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
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/mediawire"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
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
	identity         string
	noSeats          bool
	mediaReservation func(int64, int) error
	// sourceRuntime delegates checkpoint metadata/bytes to an actual installed Runtime.
	sourceRuntime   pb.RuntimePreparationClient
	sourceRelease   func(context.Context, *pb.ModelSourceReleaseCall) (*pb.ReleaseModelSourceResult, error)
	sourceControl   func(*pb.ModelSourceControlCall) (*pb.ModelSourceControlResult, error)
	weightsReady    func(*pb.WeightsIntentReadyRequest) (*pb.WeightsHostAck, error)
	protocolInfo    func(context.Context, *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error)
	watchProgress   func(*pb.ProgressOpen, pb.WorkerControl_WatchProgressServer) error
	recordOperation func(*pb.RecordOperationResultCall) (*pb.RecordOperationResultResult, error)
	derivedRelease  func(context.Context, *pb.DerivedRetentionCall) (*pb.DerivedRetentionResult, error)
	resultRelease   func(context.Context, *pb.DerivedResultReleaseCall) (*pb.DerivedResultReleaseResult, error)
	controlDefaults
	pb.UnimplementedPodHostServer
	controlKey ed25519.PublicKey
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
	// onJobReady can delay and sequence the independent peer's readiness facts.
	onJobReady func(*pb.WorkerFrame, func(*pb.WorkerFrame) error) error
	// answerOffer supplies a protocol outcome when a test exercises settlement.
	answerOffer func(*pb.AttemptOffer) (*pb.AttemptOutcome, error)
	// onFrame lets a product test delegate selected frames to a real Runtime peer.
	onFrame func(*pb.RecordOwnerFrame, func(*pb.WorkerFrame) error) (bool, error)
	// preparedPlacement supplies a complete second-implementation placement for
	// tests of modeled callable routing. The normal package preparation path still runs.
	preparedPlacement func([]byte, string, string) *pb.Placement
	numericalDigest   []byte
	// slots is the advertised seat count while serving; zero means one.
	slots uint32
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

	mu                 sync.Mutex
	acks               []*pb.SnapshotAck
	desired            []*pb.DesiredWorkerState
	prepares           []*pb.PreparePackageSetCall
	localPrepares      []*pb.PrepareLocalPackageCall
	uploads            []*pb.LocalPackageFileRef
	uploadEnds         []codes.Code
	uploadCalls        int
	lanes              []string // the order lanes were used: fetch, prepare_private, placement_set
	prepareCodes       []codes.Code
	prepareUnavailable int
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
			EnvironmentDigest:  placement.Str("environment_digest"),
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

func (p *fakePod) setLatch(fault *pb.Fault) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.latch = fault
}

func (p *fakePod) reported(reason string) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.reports[reason]
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
			p.mu.Unlock()
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
		case *pb.RecordOwnerFrame_LocalPackageFetchRequest:
			return status.Error(codes.PermissionDenied, "captured wheels must use direct upload")
		}
	}
}

// PrepareLocalPackage is the pod host's private lane: the set names wheels the pod
// already holds verified, and the prepared placement is a development one carrying the
// exact project wheel and the local revision digest, as Runtime authors it.
func (p *fakePod) LocalPackageUpload(stream grpc.BidiStreamingServer[pb.LocalPackageUploadFrame, pb.LocalPackageFileStatus]) error {
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
	state := &pb.LocalPackageFileStatus{OperationId: header.OperationId, SourceDigest: header.SourceDigest,
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
			SourceDigest: selected.Package.SourceDigest, LocalRevisionDigest: selected.Package.LocalRevisionDigest,
			ProjectWheel: &pb.WheelFact{Ref: &pb.Ref{Digest: project.Digest, Length: project.Length},
				Distribution: "weightless", Version: selected.Package.Release, Filename: project.Filename,
				ImportRoots: []string{"weightless"}, Tags: []string{"py3-none-any"}}}},
		EnvironmentDigest: bytes.Repeat([]byte{0x23}, 32),
		PackageInterface:  &pb.Ref{Digest: bytes.Repeat([]byte{0x24}, 32), Length: 2048},
		BindingsDigest:    bytes.Repeat([]byte{0x25}, 32),
		Entrypoints:       entrypoints,
		Environment:       &pb.Environment{},
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
	// MINOR 31 (xs-019): the release facts ride the call — the pod host's exact
	// refusal (internal/workerhost/prepare.go), the one run 178 died on.
	if call.Application == "" || len(call.LockedRequirements) == 0 ||
		len(call.LockedRequirements) > pb.MaxLockedRequirementsBytes ||
		len(call.ModelSlotPaths) > pb.MaxModelSlotPaths {
		return status.Error(codes.InvalidArgument,
			"PreparePackageSet requires the release facts: application, bounded model_slot_paths, and the locked requirements export")
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
	if call.ImageInventory == nil {
		// The Runtime's own verdict: preparation refuses a request without the
		// placed image's inventory (package_prepare._request_facts).
		for _, event := range []*pb.PrepareEvent{
			{Stage: pb.PrepareStage_PREPARE_STAGE_RESOLVED, TotalBytes: total},
			{Stage: pb.PrepareStage_PREPARE_STAGE_REFUSED, TotalBytes: total,
				SafeCode: "package_prepare_image_inventory_missing", SafeDetail: "image_inventory"},
		} {
			if err := stream.Send(event); err != nil {
				return err
			}
		}
		return nil
	}
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
	p.stagedJobBuild = prepared.List("placements")[0].Str("environment_digest")
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
		EnvironmentDigest: sha256Of([]byte("environment:" + name)),
		PackageInterface:  &pb.Ref{Digest: sha256Of([]byte("interface:" + name)), Length: 2048},
		BindingsDigest:    sha256Of([]byte("bindings:" + name)),
		Entrypoints:       []*pb.Entrypoint{podEntrypoint("tile")},
		Environment: &pb.Environment{LockedRequirements: &pb.Ref{
			Digest: sha256Of([]byte("locked:" + name)), Length: 1024}},
	}
}

// podPlanID is the plan a request binds to reach podPlacement's entrypoint for `name`.
func podPlanID(name string) string {
	spelled, _ := canonical.Spell(podEntrypoint("tile").EntrypointBindingDigest)
	return spelled
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
		mediaWrites.Lock()
		defer mediaWrites.Unlock()
		switch {
		case r.URL.Path == "/v1/health":
			_ = json.NewEncoder(w).Encode(mediawire.Health{Service: mediawire.Service, ContractRev: &rev, AttemptScopedInputs: true})
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
		o.ObserveRental = func(orchestrator.RentalObservation) *exit.Error { return nil }
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
		o.RentalPackageSet = func(packages []*pb.DownloadPackageRef,
			models []*pb.DownloadModelRef) ([]byte, *exit.Error) {
			body, err := canonical.Bytes(&pb.DownloadDelegation{Models: models, Packages: packages})
			if err != nil {
				return nil, exit.Internalf("%v", err)
			}
			return body, nil
		}
		o.RentalPrepareFacts = func(_ context.Context, _ *orchestrator.WorkerConnection,
			ref *pb.DownloadPackageRef) (orchestrator.PrepareFacts, *exit.Error) {
			return testPrepareFacts(ref.Package, ref.Release), nil
		}
	}
}

// testPrepareFacts is the hub-known release half of a PreparePackageSetCall in this
// suite: what the rental-scoped prepare-facts route would answer for one release.
func testPrepareFacts(pkg, release string) orchestrator.PrepareFacts {
	return orchestrator.PrepareFacts{
		Application:    "comfyui",
		ModelSlotPaths: []string{"unet"},
		ImageInventory: &pb.ImageInventory{Profile: "python3.12-cpu-linux-x86", Python: "3.12.8",
			Distributions: []*pb.ImageDistribution{{Distribution: "numpy", Version: "2.1.0"}}},
		LockedRequirements: []byte("--index-url https://pypi.org/simple\n" +
			"--extra-index-url https://hub.invalid/v1/index/acme/simple/\n\n" +
			pkg[strings.IndexByte(pkg, '/')+1:] + "==" + release +
			" --hash=sha256:" + strings.Repeat("11", 32) + "\n"),
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

// TestPodHostThreeStepSequence: the owner prepares through PodHost, then sends the exact
// prepared placement_set bytes itself on WorkerControl. No package_set ever crosses the
// control stream, and the snapshot ack echoes the host document's digest.
func TestPodHostThreeStepSequence(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	o := hostOwner(t, "podhost", rentalWiring(connection, private))

	instance, _, _, e := o.c.EnsureRental(podRental)
	fatal(t, e)
	waitUntil(t, "the peer receiving the snapshot acknowledgement", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.acks) > 0
	})
	pod.mu.Lock()
	acks, hostDigest := append([]*pb.SnapshotAck(nil), pod.acks...), pod.hostDigest
	pod.mu.Unlock()
	if len(acks) != 1 || !bytes.Equal(acks[0].HostSnapshotDigest, hostDigest) {
		t.Fatalf("the snapshot ack did not echo the host document digest: %d ack(s) %x vs %x",
			len(acks), acks[0].GetHostSnapshotDigest(), hostDigest)
	}

	fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{
		Package: "cozy/h3-package", Release: "1.0.7"}}, nil))
	waitUntil(t, "the prepared placement_set on WorkerControl", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.desired) >= 1
	})
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if len(pod.prepares) != 1 {
		t.Fatalf("PodHost.PreparePackageSet was called %d times, want 1", len(pod.prepares))
	}
	if got := pod.prepares[0].Claim; got.ControlStreamEpoch != 0 || got.WorkerBootId != podBootID {
		t.Fatalf("the host call's Claim is stream-scoped or misaddressed: %+v", got)
	}
	if first := pod.prepares[0]; first.Application == "" || len(first.ModelSlotPaths) == 0 ||
		first.ImageInventory.GetPython() == "" || len(first.LockedRequirements) == 0 {
		t.Fatalf("the host call does not carry the release facts (fields 3-6): app=%q slots=%v inventory=%v",
			first.Application, first.ModelSlotPaths, first.ImageInventory)
	}
	for _, d := range pod.desired {
		if d.GetPackageSet() != nil || d.GetLocalPackageSet() != nil || d.GetPrivatePlacementSet() != nil {
			t.Fatalf("a host mode crossed WorkerControl: %T", d.Mode)
		}
	}
	sent := pod.desired[0].GetPlacementSet()
	if sent == nil || !bytes.Equal(sent.PlacementSetCanonicalBytes, pod.preparedSet) ||
		!bytes.Equal(sent.PlacementSetDigest, pod.preparedDig) {
		t.Fatalf("the desired state does not carry the exact bytes the host prepared")
	}
	facts := o.c.Worker(instance)
	if facts == nil || facts.DesiredRevision != pod.desired[0].Revision {
		t.Fatalf("the owner's desired revision %v is not the one it sent (%d)", facts, pod.desired[0].Revision)
	}
	log, err := os.ReadFile(filepath.Join(o.root, "orchestrator.log"))
	must(t, err)
	for _, want := range []string{"PodHost prepare package_set", "PREPARED", "prepared by the host as package_set"} {
		if !strings.Contains(string(log), want) {
			t.Errorf("the owner log does not show %q", want)
		}
	}
}

// TestPodHostRefusesUnverifiedHostDocument: a host document whose digest does not hash its
// bytes is refused at the barrier exactly like a worker document would be. Dispatch never
// opens, and the pod sees no ack.
func TestPodHostRefusesUnverifiedHostDocument(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, mutateHostDigest: true}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	o := hostOwner(t, "podhost-red", rentalWiring(connection, private))

	done := make(chan *exit.Error, 1)
	go func() { _, _, _, e := o.c.EnsureRental(podRental); done <- e }()
	waitUntil(t, "the owner's refusal of the host document", func() bool {
		log, _ := os.ReadFile(filepath.Join(o.root, "orchestrator.log"))
		return strings.Contains(string(log), "host_snapshot_digest") &&
			strings.Contains(string(log), "NOT acknowledged")
	})
	pod.mu.Lock()
	acks := len(pod.acks)
	pod.mu.Unlock()
	if acks != 0 {
		t.Fatalf("the owner acknowledged a host document whose digest does not hash its bytes")
	}
	select {
	case e := <-done:
		t.Fatalf("EnsureRental returned (%s) although the barrier never opened", briefly(e))
	default:
	}
}

// localLauncher is the one Launcher method a rental-bound editable request uses: the
// sealed revision the request's install names. Everything local is out of scope on a pod.
type localLauncher struct {
	orchestrator.Launcher
	revision localpackage.Revision
}

// Wire-peer controls use declared synthetic revisions; actual CLI tests exercise
// the real resolver's sealed metadata gate.
func (l localLauncher) ValidateExecutionCapture(records.Request) *exit.Error { return nil }

func (l localLauncher) LocalRevision(installID, digest string) (localpackage.Revision, *exit.Error) {
	if digest != l.revision.Digest {
		return localpackage.Revision{}, exit.New(exit.NotFound, "no local revision %s", digest)
	}
	return l.revision, nil
}

// stageLocalRevision writes one project wheel and one dependency wheel under the root
// and seals them the way `localpackage.Stage` does: files sorted by digest, one project.
func stageLocalRevision(t *testing.T, root string) localpackage.Revision {
	t.Helper()
	write := func(name string, body []byte) localpackage.File {
		path := filepath.Join(root, name)
		must(t, os.WriteFile(path, body, 0o600))
		digest, _ := canonical.Spell(sha256Of(body))
		kind := "dependency"
		if strings.HasPrefix(name, "weightless-") {
			kind = "project"
		}
		return localpackage.File{Digest: digest, Filename: name, Kind: kind, Path: path, Length: int64(len(body))}
	}
	files := []localpackage.File{
		write("weightless-1.0.0-py3-none-any.whl", bytes.Repeat([]byte("project wheel bytes\n"), 180)),
		write("helper-0.3.0-py3-none-any.whl", bytes.Repeat([]byte("dependency wheel bytes\n"), 90)),
	}
	if files[1].Digest < files[0].Digest {
		files[0], files[1] = files[1], files[0]
	}
	return localpackage.Revision{Package: "local/weightless", Release: "1.0.0",
		SourceDigest: "sha256:" + strings.Repeat("31", 32), Digest: "sha256:" + strings.Repeat("32", 32),
		PackageInterfaceDigest: "sha256:" + strings.Repeat("33", 32), PackageInterfaceLength: 512, Files: files}
}

// submitPrivateRental records the editable install the request names and queues the
// rental-bound request, exactly as `cozy run local/... --rental-only` does.
func submitPrivateRental(t *testing.T, o *owner, revision localpackage.Revision, idem string) string {
	t.Helper()
	install := records.PackageInstall{ID: "inst-" + idem, Package: revision.Package, Major: 1, Version: revision.Release,
		SourceKind: "local", SourceRef: filepath.Join(o.root, "checkout"), SourceDigest: revision.SourceDigest,
		Dir: filepath.Join(o.root, "installs", idem), Python: "/usr/bin/python3", Platform: "linux-x86"}
	_, e := o.store.Activate(install)
	fatal(t, e)
	planID := podPlanID(revision.Package)
	requestID, _, e := o.c.Submit(orchestrator.Submission{
		IdemKey: idem, Package: revision.Package, Entrypoint: "tile", PlanID: planID,
		Release:            revision.Release,
		LocalPackageDigest: revision.Digest, Payload: []byte(`{"size":48}`), Outputs: []string{"image"},
		Worker: podRental, InstallID: install.ID, Rental: true, RentalRequired: true,
	})
	fatal(t, e)
	return requestID
}

// TestPodHostLocalRevisionGrantsProjectWheel: a rental-bound editable request grants
// every wheel of its sealed revision to the pod — the project wheel among them — BEFORE
// the private set is prepared through PodHost, and the placement_set the owner then
// sends is the exact bytes the host prepared, carrying that wheel.
func TestPodHostLocalRevisionGrantsProjectWheel(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	revision := stageLocalRevision(t, root)
	pod.serve = true
	o := hostOwner(t, "podhost-private", rentalWiring(connection, private), func(opt *orchestrator.Options) {
		opt.Packages = localLauncher{revision: revision}
	})
	requestID := submitPrivateRental(t, o, revision, "private-grants")

	waitUntil(t, "the prepared placement_set on WorkerControl", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.desired) >= 1
	})
	func() {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		if got := strings.Join(pod.lanes, ","); got != "upload,prepare_private,placement_set" {
			t.Fatalf("the pod saw the lanes in the order %q; want upload, prepare_private, placement_set", got)
		}
		project := 0
		for _, grant := range pod.uploads {
			spelled, _ := canonical.Spell(grant.Digest)
			if strings.HasPrefix(grant.Filename, "weightless-1.0.0-") {
				project++
				if spelled != revision.Files[0].Digest && spelled != revision.Files[1].Digest ||
					grant.Length == 0 {
					t.Fatalf("the project wheel grant names %s %s, not the sealed revision's wheel", spelled, grant.Filename)
				}
			}
		}
		if len(pod.uploads) != len(revision.Files) || project != 1 {
			t.Fatalf("%d grant(s) with %d project wheel(s); want %d grants naming exactly one project wheel",
				len(pod.uploads), project, len(revision.Files))
		}
		if len(pod.localPrepares) != 1 || len(pod.localPrepares[0].LocalPackageSet.Files) != len(revision.Files) ||
			pod.localPrepares[0].LocalPackageSet.OperationId != requestID {
			t.Fatalf("PodHost.PrepareLocalPackage saw %d call(s) for %v; want one naming request %s and every wheel",
				len(pod.localPrepares), pod.localPrepares, requestID)
		}
		sent := pod.desired[0].GetPlacementSet()
		if sent == nil || !bytes.Equal(sent.PlacementSetCanonicalBytes, pod.preparedSet) {
			t.Fatalf("the desired state does not carry the exact bytes the host prepared")
		}
		doc, err := canonical.Read(sent.PlacementSetCanonicalBytes, &pb.PlacementSet{})
		must(t, err)
		wheel := doc.List("placements")[0].Sub("development").Sub("project_wheel").Sub("ref").Str("digest")
		if wheel != revision.Files[0].Digest && wheel != revision.Files[1].Digest {
			t.Fatalf("the placement's project wheel %s is not one the owner granted", wheel)
		}
		row, e := o.store.RequestRow(requestID)
		fatal(t, e)
		if row.LocalPackageUploadedBootID != podBootID {
			t.Fatalf("the request row records upload boot %q; want the pod's %s", row.LocalPackageUploadedBootID, podBootID)
		}
	}()
	// THE POD SERVES: the request is DISPATCHED to the rented worker exactly once — the
	// install that pins a local relaunch never hides the rental — and the spec names the
	// captured local revision's exact prepared Environment.
	waitUntil(t, "the attempt offer on the pod", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.offers) >= 1
	})
	time.Sleep(300 * time.Millisecond)
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if len(pod.offers) != 1 || len(pod.desired) != 1 {
		t.Fatalf("%d offer(s) over %d desired state(s); want one offer over the one prepared set",
			len(pod.offers), len(pod.desired))
	}
	spec, err := canonical.Read(pod.offers[0].InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
	must(t, err)
	prepared, err := canonical.Read(pod.desired[0].GetPlacementSet().PlacementSetCanonicalBytes, &pb.PlacementSet{})
	must(t, err)
	environment := prepared.List("placements")[0].Str("environment_digest")
	if environment == "" || spec.Str("environment_digest") != environment {
		t.Fatalf("captured serving must name its actual prepared environment: got %q, want %q", spec.Str("environment_digest"), environment)

	}
}

// TestPodPlacementRefusedRepeats: a pod that latches one materialization fault against the
// desired revision and replays it unchanged on every report is answered by a typed request
// failure after the fleet's still-factor of identical reports — never a timer, never an
// unbounded queue. A fault whose text changes starts the count over; the rental stays
// attached for the idle release; the owner sends no second desired state.
func TestPodPlacementRefusedRepeats(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	revision := stageLocalRevision(t, root)
	o := hostOwner(t, "podhost-latched", rentalWiring(connection, private), func(opt *orchestrator.Options) {
		opt.Packages = localLauncher{revision: revision}
	})
	missing := "sha256:" + strings.Repeat("2964e74c", 8) + " is reached by PlacementSet/1 and absent locally and from the plan"
	first := &pb.Fault{Kind: pb.FaultKind_FAULT_KIND_ARTIFACT_FETCH_FAILED, Reason: "download_plan_missing", Detail: missing}
	pod.setLatch(first)
	requestID := submitPrivateRental(t, o, revision, "latched")

	state := func() string {
		row, e := o.store.RequestRow(requestID)
		fatal(t, e)
		return row.State
	}
	// Six identical reports, then the text changes: the count starts over, and the request
	// is still queued after fourteen reports that never agreed eight times running.
	waitUntil(t, "six identical fault reports", func() bool { return pod.reported(first.Reason+"/"+first.Detail) >= 6 })
	changed := &pb.Fault{Kind: first.Kind, Reason: first.Reason, Detail: missing + " (grant refreshed)"}
	pod.setLatch(changed)
	waitUntil(t, "six changed fault reports", func() bool { return pod.reported(changed.Reason+"/"+changed.Detail) >= 6 })
	if got := state(); got != "submitted" && got != "queued" {
		t.Fatalf("the request is %s after twelve reports of two different faults; want it still waiting", got)
	}
	// The same fault, eight reports running: the worker's final word on this revision.
	pod.setLatch(first)
	waitUntil(t, "the typed request failure", func() bool { return state() == "failed" })
	rows, e := o.store.EventsAfter(requestID, 0, 100)
	fatal(t, e)
	var failed map[string]any
	for _, row := range rows {
		if row.Type == "request.failed" {
			failed = row.Payload
		}
	}
	if failed == nil || failed["error_type"] != "worker.placement_refused" ||
		!strings.HasPrefix(failed["error"].(string), "download_plan_missing: "+missing) {
		t.Fatalf("request.failed payload = %v; want worker.placement_refused carrying the fault text", failed)
	}
	if n := pod.reported(first.Reason + "/" + first.Detail); n < 6+orchestrator.StillFactor {
		t.Fatalf("the owner failed the request after %d identical reports; want at least %d", n, 6+orchestrator.StillFactor)
	}
	pod.mu.Lock()
	sent := len(pod.desired)
	pod.mu.Unlock()
	if sent != 1 {
		t.Fatalf("the owner sent %d desired states; want the one — a latched fault is never answered by a re-send", sent)
	}
	row, e := o.store.RequestRow(requestID)
	fatal(t, e)
	if row.Requeues != 0 {
		t.Fatalf("the request was requeued %d time(s); want 0", row.Requeues)
	}
	instance, _, _, e := o.c.EnsureRental(podRental)
	fatal(t, e)
	facts := o.c.Worker(instance)
	if facts == nil || facts.Exited {
		t.Fatalf("the rented worker is gone (%v); the rental is the idle release's to end", facts)
	}
	log, err := os.ReadFile(filepath.Join(o.root, "orchestrator.log"))
	must(t, err)
	if !strings.Contains(string(log), "repeated unchanged on "+strconv.Itoa(orchestrator.StillFactor)+" consecutive reports") {
		t.Errorf("the owner log does not name the repetition verdict")
	}
}

// submitPublishedRentalJob queues one published JOB bound to the rented pod, the way
// `cozy model upload --producer ... --rental-only` does: a job-shaped callable, no
// install, no model release of its own.
func submitPublishedRentalJob(t *testing.T, o *owner, pkg, release, planID, idem string) string {
	t.Helper()
	requestID, _, e := o.c.Submit(orchestrator.Submission{
		IdemKey: idem, Package: pkg, Entrypoint: "four-lane", PlanID: planID, Release: release,
		Kind: "job", Org: "paul", Payload: []byte(`{"steps":4}`), Outputs: []string{"model"},
		Worker: podRental, Rental: true, RentalRequired: true,
	})
	fatal(t, e)
	return requestID
}

// TestPodHostJobDirectiveNamesTheStagedBuild: the JOB lane's directive names the BUILD the
// rented worker staged its job plan records under — its own prepared placement's identity —
// and not this owner's PlacementSet digest, which the worker never saw while it was
// staging. Naming the set digest is what `session.py::apply_job_directive` refuses as
// `job_plan_mismatch`, and it refuses BEFORE any offer, so no producer run can reach a
// first attempt. The producer path (a job-shaped callable on a rented pod, zero model
// releases) is the combination that had no test.
func TestPodHostJobDirectiveNamesTheStagedBuild(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	o := hostOwner(t, "podhost-job-build", rentalWiring(connection, private))

	planID := "sha256:" + strings.Repeat("35", 32)
	submitPublishedRentalJob(t, o, "cozy/h3-package", "1.0.7", planID, "job-build-1")

	waitUntil(t, "the JobDirective on WorkerControl", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.jobDirectives) >= 1
	})
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if pod.stagedJobBuild == "" {
		t.Fatalf("the pod staged its job plans under no build identity")
	}
	directive := pod.jobDirectives[0]
	if directive.BuildId != pod.stagedJobBuild {
		t.Fatalf("the JobDirective names build %q; the pod staged its job plan under %q",
			directive.BuildId, pod.stagedJobBuild)
	}
	if directive.JobDescriptorId != planID {
		t.Fatalf("the JobDirective names descriptor %q, not the requested %q",
			directive.JobDescriptorId, planID)
	}
	if !directive.ReclaimOnTerminal {
		t.Fatalf("the JobDirective does not reclaim on terminal")
	}
	setDigest, _ := canonical.Spell(pod.preparedDig)
	if directive.BuildId == setDigest {
		t.Fatalf("the JobDirective names the PlacementSet digest, which the worker never staged under")
	}
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
