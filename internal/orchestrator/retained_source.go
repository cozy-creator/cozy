package orchestrator

import (
	"bytes"
	"context"
	"sort"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

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
		return exit.Unavailablef("retained source adoption is awaiting the original pod")
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
