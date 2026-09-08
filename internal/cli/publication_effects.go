package cli

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/publication"
	"github.com/cozy-creator/cozy/internal/records"
)

func (o *modelTransferOwner) publicationEffectClient(ctx context.Context, call records.NativeCall) (*hub.Client, *exit.Error) {
	ref, problem := publication.EffectDestination(call.Operation, call.Request)
	if problem != nil {
		return nil, problem
	}
	configured := &Context{Cfg: o.cfg, AccountAuth: o.auth}
	return ownedPublication(configured, ref)
}
func (o *modelTransferOwner) PreparePublicationEffect(ctx context.Context, call records.NativeCall) ([]byte, *exit.Error) {
	client, problem := o.publicationEffectClient(ctx, call)
	if problem != nil {
		return nil, problem
	}
	if call.Operation != "publish_release" {
		return nil, exit.Named(exit.Unavailable, "publication.operation_mismatch", "release service received a non-release effect")
	}
	intent, problem := publication.PreparePublish(ctx, client, call.Request)
	if problem != nil {
		return nil, problem
	}
	return publication.Canonical(intent)
}
func (o *modelTransferOwner) ApplyPublicationEffect(ctx context.Context, call records.NativeCall) ([]byte, *exit.Error) {
	client, problem := o.publicationEffectClient(ctx, call)
	if problem != nil {
		return nil, problem
	}
	if call.Operation != "publish_release" {
		return nil, exit.Named(exit.Unavailable, "publication.operation_mismatch", "release service received a non-release effect")
	}
	var intent publication.ReleaseIntent
	if problem := publication.DecodeEffect(call.Frozen, &intent); problem != nil {
		return nil, problem
	}
	receipt, problem := publication.ApplyRelease(ctx, client, intent, call.State == "executing", func() *exit.Error {
		parent, problem := o.store.RequestRow(call.ParentRequestID)
		if problem != nil {
			return problem
		}
		if parent != nil && parent.State != "dispatching" && parent.State != "canceling" && parent.State != "canceled" && parent.State != "releasing" {
			return exit.Named(exit.Unavailable, "publication.parent_paused", "publication waits for its parent to resume")
		}
		if parent == nil || parent.State != "dispatching" {
			return exit.Named(exit.Canceled, "publication.parent_stopped", "stopped parent cannot issue another publication mutation")
		}
		return o.store.StartNativeCall(call.ID)
	})
	if problem != nil {
		return nil, problem
	}
	return publication.Canonical(receipt)
}

func (o *modelTransferOwner) UploadPublicationEffect(ctx context.Context, call records.NativeCall, intent publication.UploadIntent, upload publication.ObjectUploader) ([]byte, *exit.Error) {
	client, problem := o.publicationEffectClient(ctx, call)
	if problem != nil {
		return nil, problem
	}
	beforeWrite := func() *exit.Error {
		parent, problem := o.store.RequestRow(call.ParentRequestID)
		if problem != nil {
			return problem
		}
		if parent != nil && parent.State != "dispatching" && parent.State != "canceling" && parent.State != "canceled" && parent.State != "releasing" {
			return exit.Named(exit.Unavailable, "publication.parent_paused", "publication waits for its parent to resume")
		}
		if parent == nil || parent.State != "dispatching" {
			return exit.Named(exit.Canceled, "publication.parent_stopped", "stopped parent cannot authorize upload or checkpoint finalization")
		}
		return o.store.StartNativeCall(call.ID)
	}
	receipt, problem := publication.UploadCheckpoint(ctx, client, call.ID, intent, beforeWrite, upload)
	if problem != nil {
		return nil, problem
	}
	return publication.Canonical(receipt)
}
