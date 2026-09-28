package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// A published root reaches its machine as one message: the release, the callable, the
// payload, the caller's Model choices and its inputs. The machine installs what it lacks from
// its own Hub, resolves every open slot for its own devices, prepares and mints the offer;
// nothing here reads the Hub, a ladder or a package.

// releaseRoot answers whether a request is submitted as one root: every published root, and
// an unpublished installation's root with a provider-source Model, which only a root carries.
func releaseRoot(request records.Request) bool {
	if request.LocalInstallationID != "" {
		return sourced(request.Models)
	}
	return !strings.HasPrefix(request.Package, "local/")
}

// capturedRevision is the unpublished installation a root names. A root carries that one
// installation; one whose package calls other unpublished packages still goes by capture.
func (m *machineRuns) capturedRevision(request records.Request) (localpackage.Installation, *exit.Error) {
	capture, problem := m.resolver.CaptureMachineExecution(request)
	if problem != nil {
		return localpackage.Installation{}, problem
	}
	if len(capture.Installations) != 1 || capture.Installations[0].ID != request.LocalInstallationID {
		return localpackage.Installation{}, exit.Named(exit.Structural, "machine_execution.captured_callees_unsupported",
			"a provider-source Model on a package that calls other unpublished packages is not supported yet")
	}
	return capture.Installations[0], nil
}

func (m *machineRuns) releaseRootSubmission(ctx context.Context, request records.Request, connection *machineConnection) (*pb.MachineExecutionSubmit, *exit.Error) {
	root := &pb.ReleaseRoot{Package: request.Package, Release: request.Release, Entrypoint: request.Entrypoint,
		DeadlineUnixMs: uint64(max(request.DeadlineUnixMS, 0)), AttentionKernel: request.AttentionKernel}
	if request.LocalInstallationID != "" {
		if _, problem := m.capturedRevision(request); problem != nil {
			return nil, problem
		}
		root.Release, root.InstallationId = "", request.LocalInstallationID
	}
	if request.IsJob() {
		root.Job, root.PublicationGrant = true, home.ScratchRepo(request.Org, request.ID)
		if request.ModelTransfer != nil {
			root.WeightsDestination = request.ModelTransfer.Destination
		}
	}
	for _, model := range request.Models {
		parameter := model.Slot[strings.LastIndex(model.Slot, ".")+1:]
		if model.Source != "" {
			root.Models = append(root.Models, &pb.ModelChoice{Parameter: parameter, Source: model.Source,
				Profiles: model.Profiles})
			continue
		}
		choice := &pb.ModelChoice{Parameter: parameter,
			Repository: model.Model, Release: model.Release, Lane: model.Lane}
		if model.Manifest != "" {
			digest, err := canonical.Raw(model.Manifest)
			if err != nil {
				return nil, exit.Named(exit.Validation, "model.manifest_invalid", "%s names no exact manifest", model.Slot)
			}
			choice.Manifest = &pb.Ref{Digest: digest, Length: uint64(max(model.ManifestLength, 0))}
		}
		root.Models = append(root.Models, choice)
	}
	sort.Slice(root.Models, func(i, j int) bool { return root.Models[i].Parameter < root.Models[j].Parameter })
	if request.Capture != "" {
		root.Capture = &pb.ActivationCapture{}
		if err := json.Unmarshal([]byte(request.Capture), root.Capture); err != nil {
			return nil, exit.New(exit.Validation, "recorded capture options are invalid")
		}
	}
	began := time.Now()
	access, problem := m.stageMachineInputs(ctx, request, connection)
	if problem != nil {
		return nil, problem
	}
	if root.InputAccess = access; len(access) > 0 {
		m.submissionStage(request.ID, "inputs", fmt.Sprintf("%d input(s)", len(access)), began)
	}
	for _, asset := range request.Assets {
		root.Inputs = append(root.Inputs, &pb.InputBinding{InputId: asset.FieldPath, Digest: asset.Digest,
			Length: uint64(asset.Length), KindMime: asset.MediaType, Order: asset.Order})
	}
	return &pb.MachineExecutionSubmit{SubmissionId: records.MachineSubmissionID(request.IdemKey),
		Offer: &pb.AttemptOffer{RequestId: request.ID}, PayloadCanonicalBytes: request.Payload,
		ReleaseRoot: root}, nil
}

// sendReleaseRoot replays the one frozen submission until the machine answers a receipt; while
// it prepares, the machine answers with its progress.
func (m *machineRuns) sendReleaseRoot(ctx context.Context, request records.Request, connection *machineConnection, frozen *pb.MachineExecutionSubmit) *exit.Error {
	progress := ""
	// The Models' download, kept for `cozy run show` once it ends: its bytes, time and rate.
	var began time.Time
	var downloaded *pb.PrepareEvent
	recordDownload := func() {
		if payload := orchestrator.PrepareStagePayload(request.Package, pb.PrepareStage_PREPARE_STAGE_DOWNLOADING, began, downloaded); payload != nil {
			_ = m.store.AppendEvent(request.ID, "request.preparing", 0, payload)
		}
		downloaded = nil
	}
	defer recordDownload()
	for {
		submission := proto.Clone(frozen).(*pb.MachineExecutionSubmit)
		submission.SourceCredentials, submission.Claim = m.resolver.SourceCredentials(), connection.Claim
		submission.Offer.WorkerBootId, submission.Offer.RecordOwnerEpoch = connection.Claim.WorkerBootId, connection.Claim.RecordOwnerEpoch
		var trailer metadata.MD
		receipt, err := connection.Host.SubmitMachineExecution(ctx, submission, grpc.Trailer(&trailer))
		if err == nil {
			if receipt == nil || receipt.WorkerId != connection.Claim.WorkerId || receipt.WorkerBootId == "" {
				return exit.New(exit.Conflict, "execution was accepted by an unexpected worker")
			}
			return m.store.AcceptMachineExecution(request.ID, receipt)
		}
		codeOf := trailer.Get("cozy-error-code")
		switch {
		case slices.Contains(codeOf, "release_root_installation_absent"):
			// The machine lost the installation it was given (a restart): not now. The
			// next pass prepares it again and asks with the same submission.
			return exit.Named(exit.Unavailable, "machine_execution.installation_absent", "%s", status.Convert(err).Message())
		case slices.Contains(codeOf, "release_root_preparing"):
			if message := status.Convert(err).Message(); message != progress {
				progress = message
				_ = m.store.AppendEvent(request.ID, "request.preparing", 0, map[string]any{"stage": "machine", "detail": message})
			}
			if event := m.observeRootDownload(request.ID, trailer); event != nil {
				if downloaded == nil {
					began = time.Now()
				}
				downloaded = event
			} else {
				recordDownload()
			}
		case status.Code(err) == codes.DeadlineExceeded && ctx.Err() == nil:
			// The Host bounds one admission; the machine keeps preparing: ask again.
		default:
			if slices.Contains(codeOf, "execution_workspace_changed") {
				connection.KeepWorkspace(nil)
			}
			return m.submissionRefused(request.ID, trailer, err)
		}
	}
}

// observeRootDownload shows a preparing root's Models download as the run's live download
// phase: the machine answers the bytes landed of the total, and the phase lane measures rate.
func (m *machineRuns) observeRootDownload(request string, trailer metadata.MD) *pb.PrepareEvent {
	counts := trailer.Get("cozy-progress-bytes")
	var moved, total uint64
	if len(counts) != 1 {
		return nil
	}
	if _, err := fmt.Sscanf(counts[0], "%d %d", &moved, &total); err != nil || total == 0 {
		return nil
	}
	event := &pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_DOWNLOADING, TransferredBytes: moved, TotalBytes: total}
	if m.fleet != nil && m.fleet.owner != nil {
		m.fleet.owner.ObservePrepareEvent(request, "", "", event)
	}
	return event
}

// submissionRefused names a failed release-root submission; a definitive refusal is recorded.
func (m *machineRuns) submissionRefused(requestID string, trailer metadata.MD, err error) *exit.Error {
	for _, code := range trailer.Get("cozy-error-code") {
		switch code {
		case "execution_workspace_changed", "execution_workspace_required":
			return exit.Named(exit.Conflict, "machine_execution.workspace_changed", "execution workspace no longer matches the frozen submission; prior acceptance remains unresolved")
		case "execution_submission_refused":
			return m.unaccepted(requestID, err)
		}
	}
	return machineTransport(err)
}
