package orchestrator

import (
	"bytes"
	"context"
	"io"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// Source roots belong to the request, not to the Python process's lifetime.
// Wait for the original Host to fence and drain its source operation before
// relinquishing retention. An explicitly ended rental has already lost its disk.
func (c *Orchestrator) releaseRetainedSource(request records.Request) *exit.Error {
	if request.Worker == "" || !request.ModelTransfer.HasAcquisition() {
		return nil
	}
	row, problem := c.opt.Store.RentalRow(request.Worker)
	if problem != nil {
		return problem
	}
	if row != nil && row.State == "released" {
		return nil
	}
	s, problem := c.rentalControl(request.Worker)
	if problem != nil {
		_, _, _, _ = c.EnsureRental(request.Worker)
		return problem
	}
	if s.host == nil || s.claim == nil {
		return exit.Unavailablef("source release is awaiting the claimed original pod")
	}
	selection, err := canonical.Raw(request.ModelTransfer.SourceSelection)
	if err != nil {
		return exit.Internalf("retained source selection is malformed")
	}
	answer, err := s.host.ModelSourceRelease(s.ctx, &pb.ModelSourceReleaseCall{
		Claim: s.claim, OperationId: request.ID, SourceSelectionDigest: selection,
	})
	if err != nil {
		return exit.Unavailablef("source release is awaiting the original pod")
	}
	if answer == nil || !answer.Released || answer.OperationId != request.ID {
		return exit.Named(exit.Structural, "request.source_release_changed", "source release did not acknowledge this request")
	}
	return nil
}

// adoptRetriedSource asks the same claimed pod to independently retain source
// computation for an edited request. The native owner validates the old chain
// against the new source plan and produces new operation-bound checkpoints.
func (c *Orchestrator) adoptRetriedSource(ctx context.Context, request records.Request, s *session) *exit.Error {
	if request.RetryOf == "" || !request.RetainWork {
		return nil
	}
	priorID, progress, problem := c.opt.Store.RetriedSourceCheckpoints(request.ID)
	if problem != nil || priorID == "" || len(progress) == 0 {
		return problem
	}
	if s == nil || s.host == nil || s.claim == nil {
		return exit.Unavailablef("retained source adoption requires the claimed original pod")
	}
	prior, problem := c.opt.Store.RequestRow(priorID)
	if problem != nil || prior == nil {
		return exit.Unavailablef("retained source predecessor metadata is unavailable")
	}
	if problem := declareAdoptionSource(ctx, s, *prior); problem != nil {
		return problem
	}
	selection, err := canonical.Raw(request.ModelTransfer.SourceSelection)
	if err != nil {
		return exit.Internalf("retained source selection is malformed")
	}
	call := &pb.ModelSourceAdoptCall{Claim: s.claim, FromOperationId: priorID,
		Request: &pb.ModelSourcePrepareRequest{RecordOwnerEpoch: recordOwnerEpoch,
			WorkerBootId: s.bootID, OperationId: request.ID, SourceSelectionDigest: selection,
			SourceUri: request.ModelTransfer.Source, DeclaredLicense: request.ModelTransfer.SourceLicense}}
	for _, progress := range progress {
		if progress.WorkerBootID != s.bootID {
			return exit.Named(exit.Conflict, "request.state_lost", "retained source checkpoint belongs to another worker boot; local bytes cannot be claimed as resumed")
		}
		checkpoint := progress.Observed
		head, headErr := canonical.Raw(checkpoint.HeadID)
		plan, planErr := canonical.Raw(checkpoint.PlanDigest)
		if headErr != nil || planErr != nil {
			return exit.Internalf("retained source checkpoint is malformed")
		}
		call.Request.Checkpoints = append(call.Request.Checkpoints, &pb.ModelSourceCheckpoint{Slot: checkpoint.Slot,
			Head: &pb.Ref{Digest: head, Length: uint64(checkpoint.HeadLength)}, PlanDigest: plan,
			Index: uint64(checkpoint.Index), Bytes: uint64(checkpoint.Bytes)})
	}
	for slot, profile := range request.ModelTransfer.SourceProfiles {
		call.Request.Profiles = append(call.Request.Profiles, &pb.ModelSourceProfile{Slot: slot, Profile: profile})
	}
	sort.Slice(call.Request.Profiles, func(i, j int) bool { return call.Request.Profiles[i].Slot < call.Request.Profiles[j].Slot })
	if proto.Size(call) > pb.MaxInlineControlBytes {
		return exit.Named(exit.Structural, "request.source_adoption_too_large", "retained source adoption exceeds the bounded control message")
	}
	answer, err := s.host.ModelSourceAdopt(ctx, call)
	if err != nil {
		return sourceAdoptionError(err)
	}
	if answer == nil || answer.OperationId != request.ID || answer.WorkerBootId != s.bootID ||
		answer.RecordOwnerEpoch != recordOwnerEpoch || answer.ControlStreamEpoch != 0 ||
		!bytes.Equal(answer.SourceSelectionDigest, selection) || proto.Size(answer) > pb.MaxInlineControlBytes {
		return exit.Named(exit.Structural, "request.source_adoption_changed", "source adoption changed its claimed request, source, or worker")
	}
	if answer.Outcome == pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REFUSED {
		return exit.Named(exit.Conflict, answer.SafeCode, "source adoption refused: %s", answer.SafeDetail)
	}
	if len(answer.Checkpoints) != len(call.Request.Checkpoints) {
		return exit.Named(exit.Structural, "request.source_adoption_incomplete", "source adoption did not independently retain every prior checkpoint")
	}
	wanted := make(map[string]*pb.ModelSourceCheckpoint, len(call.Request.Checkpoints))
	for _, checkpoint := range call.Request.Checkpoints {
		wanted[checkpoint.Slot] = checkpoint
	}
	for _, checkpoint := range answer.Checkpoints {
		if checkpoint == nil {
			return exit.Named(exit.Structural, "request.source_adoption_changed", "source adoption returned an absent checkpoint")
		}
		prior := wanted[checkpoint.Slot]
		if prior == nil || checkpoint.Head == nil || bytes.Equal(checkpoint.Head.Digest, prior.Head.Digest) ||
			!bytes.Equal(checkpoint.PlanDigest, prior.PlanDigest) {
			return exit.Named(exit.Structural, "request.source_adoption_changed", "source adoption must retain the same source progress under a new operation-bound head")
		}
		delete(wanted, checkpoint.Slot)
	}
	c.onModelSourcePrepared(s, answer)
	observed, problem := c.opt.Store.ModelSourceProgress(request.ID)
	if problem != nil {
		return problem
	}
	if len(observed) != len(progress) {
		return exit.Named(exit.Structural, "request.source_adoption_unrecorded", "source adoption was not accepted into the request's own progress")
	}
	previous := make([]records.ModelCheckpoint, len(progress))
	adopted := make([]records.ModelCheckpoint, len(observed))
	for i := range progress {
		previous[i] = progress[i].Observed
	}
	for i := range observed {
		adopted[i] = observed[i].Observed
	}
	return c.opt.Store.RecordRetriedSourceAdoption(request.ID, priorID, s.bootID, previous, adopted)
}

// The Host's source admission roster is session memory. After its restart, the
// owner must redeclare the old operation's exact metadata before asking to adopt
// its retained native checkpoint. Empty URLs cannot fetch any provider bodies.
func declareAdoptionSource(ctx context.Context, s *session, prior records.Request) *exit.Error {
	if prior.ModelTransfer == nil {
		return exit.New(exit.Validation, "retained source predecessor has no acquisition")
	}
	selection, err := canonical.Raw(prior.ModelTransfer.SourceSelection)
	if err != nil {
		return exit.Internalf("retained source predecessor selection is malformed")
	}
	provider := pb.ModelSourceProvider_MODEL_SOURCE_PROVIDER_UNSPECIFIED
	if strings.HasPrefix(prior.ModelTransfer.Source, "hf://") {
		provider = pb.ModelSourceProvider_MODEL_SOURCE_PROVIDER_HUGGING_FACE
	}
	if strings.HasPrefix(prior.ModelTransfer.Source, "civitai://") {
		provider = pb.ModelSourceProvider_MODEL_SOURCE_PROVIDER_CIVITAI
	}
	for _, file := range prior.ModelTransfer.SourceFiles {
		request := &pb.ModelSourceFileRequest{RecordOwnerEpoch: recordOwnerEpoch, WorkerBootId: s.bootID, OperationId: prior.ID, SourceSelectionDigest: selection,
			Member: file.Member, ObjectId: "sha256:" + file.SHA256, Length: uint64(file.Length), Header: file.Header, Provider: provider, CapabilityRevision: 1}
		stream, err := s.host.ModelSourceFile(ctx, &pb.ModelSourceFileCall{Claim: s.claim, Request: request})
		if err != nil {
			return sourceAdoptionError(err)
		}
		for {
			answer, err := stream.Recv()
			if err == io.EOF {
				break
			}
			if err != nil {
				return sourceAdoptionError(err)
			}
			if answer == nil || answer.OperationId != prior.ID || answer.Member != file.Member || answer.ObjectId != request.ObjectId || answer.Length != request.Length || !bytes.Equal(answer.SourceSelectionDigest, selection) {
				return exit.Named(exit.Structural, "request.source_adoption_changed", "retained source declaration changed its exact metadata")
			}
		}
	}
	return nil
}

func sourceAdoptionError(err error) *exit.Error {
	if result := classifyPrepareEnd(err); result.refusal != "" {
		return exit.Named(exit.Structural, "request.source_adoption_refused", "retained source adoption refused (%s): %s", status.Code(err), refusalDetail(err))
	}
	return exit.Unavailablef("retained source adoption awaits the original pod (%s)", status.Code(err))
}
