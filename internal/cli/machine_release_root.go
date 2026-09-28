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

// releaseRoot answers whether a request is submitted by its release: every published root
// except those on the temporary prepared path.
func releaseRoot(request records.Request) bool {
	return request.LocalInstallationID == "" && !strings.HasPrefix(request.Package, "local/") &&
		!temporaryPreparedPath(request)
}

// temporaryPreparedPath names the published requests release roots do not carry yet:
// adapters (--lora). They keep the prepared submission until tracker proto-062 R2 moves
// them; delete this with it. (Intake captures input trees as staged inputs and keeps
// model-transfer acquisitions off machine executions.)
func temporaryPreparedPath(request records.Request) bool {
	for _, model := range request.Models {
		if len(model.Adapters) > 0 {
			return true
		}
	}
	return false
}

func (m *machineRuns) releaseRootSubmission(ctx context.Context, request records.Request, connection *machineConnection) (*pb.MachineExecutionSubmit, *exit.Error) {
	root := &pb.ReleaseRoot{Package: request.Package, Release: request.Release, Entrypoint: request.Entrypoint,
		DeadlineUnixMs: uint64(max(request.DeadlineUnixMS, 0)), AttentionKernel: request.AttentionKernel}
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
		case slices.Contains(codeOf, "release_root_preparing"):
			if message := status.Convert(err).Message(); message != progress {
				progress = message
				_ = m.store.AppendEvent(request.ID, "request.preparing", 0, map[string]any{"stage": "machine", "detail": message})
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
