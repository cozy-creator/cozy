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
	return ownedPublication(o.cliContext(o.hubOf(call.ParentRequestID), records.ModelTransferIntent{}, false), ref)
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
	receipt, problem := publication.ApplyRelease(ctx, client, intent, call.State == "executing", func() *exit.Error { return o.store.StartNativeEffectWrite(call.ID) })
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
	beforeWrite := func() *exit.Error { return o.store.StartNativeEffectWrite(call.ID) }

	receipt, problem := publication.UploadCheckpoint(ctx, client, call.ID, intent, beforeWrite, upload)
	if problem != nil {
		return nil, problem
	}
	return publication.Canonical(receipt)
}
