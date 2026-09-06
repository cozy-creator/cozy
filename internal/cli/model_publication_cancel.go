package cli

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

// AbandonModelTransferPublications reconciles only this request's unfinished
// output sessions. Completed repository checkpoints are retained.
func (o *modelTransferOwner) AbandonModelTransferPublications(ctx context.Context, requestID string) *exit.Error {
	request, problem := o.store.RequestRow(requestID)
	if problem != nil {
		return problem
	}
	transfer, problem := o.store.ModelTransferOf(requestID)
	if problem != nil {
		return problem
	}
	if request == nil || transfer == nil || (transfer.State != "canceling" && transfer.State != "canceled") {
		return exit.New(exit.Conflict, "publication cleanup requires explicit cancellation")
	}
	if transfer.Kind != "model-upload" {
		return nil
	}
	outputs, problem := o.store.AllModelTransferWeights(requestID, request.Ordinal)
	if problem != nil || len(outputs) == 0 {
		return problem
	}
	ref, problem := hub.ParseRef(transfer.Destination)
	if problem != nil {
		return problem
	}
	client, problem := ownedPublication(o.cliContext(transfer.ModelTransferIntent, request.Worker != ""), ref)
	if problem != nil {
		return problem
	}
	for _, output := range outputs {
		if output.FinalID != "" {
			continue
		}
		operation := transferOutputOperation(requestID, output.OutputSlot)
		problem := client.AbandonPublication(ctx, ref, operation)
		if problem == nil || problem.ErrName() == "publication.not_found" {
			continue
		}
		if problem.ErrName() != "publication.not_abandonable" {
			return problem
		}
		// A finalization response may have been lost. Confirm the existing immutable
		// checkpoint rather than deleting it or creating a replacement publication.
		objects := make([]hub.Object, 0, len(output.Objects))
		for _, object := range output.Objects {
			objects = append(objects, hub.Object{ID: object.ObjectID, Length: object.Length})
		}
		opened, problem := client.OpenPublication(ctx, ref, operation, objects, "reconcile canceled publication")
		if problem != nil {
			return problem
		}
		if opened.Created || opened.Publication.State != "checkpointed" {
			return exit.New(exit.Conflict, "canceled publication is neither abandoned nor checkpointed")
		}
	}
	return nil
}
