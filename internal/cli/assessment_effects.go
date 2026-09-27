package cli

import (
	"context"

	"github.com/cozy-creator/cozy/internal/assessment"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/publication"
	"github.com/cozy-creator/cozy/internal/records"
)

func (o *modelTransferOwner) PrepareAssessmentPublication(ctx context.Context, call records.NativeCall, producer string, report, workloads []byte) (publication.AssessmentIntent, *exit.Error) {
	var intent publication.AssessmentIntent
	if problem := publication.DecodeEffect(call.Request, &intent.Request); problem != nil {
		return intent, problem
	}
	client, problem := o.publicationEffectClient(ctx, call)
	if problem != nil {
		return intent, problem
	}
	ref, problem := publication.EffectDestination(call.Operation, call.Request)
	if problem != nil {
		return intent, problem
	}
	intent.Account, problem = publication.Authorize(ctx, client, ref, o.assessmentOperator(o.hubOf(call.ParentRequestID)))
	if problem != nil {
		return intent, problem
	}
	checkpoint, problem := client.CheckpointManifest(ctx, ref, intent.Request.Checkpoint.Checkpoint)
	if problem != nil {
		return intent, problem
	}
	digest, _ := canonical.Spell(canonical.Digest(checkpoint))
	if digest != intent.Request.Checkpoint.Manifest.Digest || int64(len(checkpoint)) != intent.Request.Checkpoint.Manifest.Length {
		return intent, exit.New(exit.Conflict, "assessment checkpoint publication changed")
	}
	inspected, problem := assessment.Inspect(ctx, report, o.cfg.Tool())
	if problem != nil {
		return intent, problem
	}
	info, problem := assessment.ReadRenderInspection(report, inspected)
	if problem != nil {
		return intent, problem
	}
	if info.Subject.Candidate != intent.Request.Checkpoint.Checkpoint {
		return intent, exit.New(exit.Conflict, "assessment candidate differs from published checkpoint")
	}
	bound, problem := assessment.VerifyRenderBindings(o.store, producer, info, workloads)
	if problem != nil {
		return intent, problem
	}
	intent.RenderHolds, problem = assessment.VerifyObservations(o.store, producer, info, workloads, bound)
	if problem != nil {
		return intent, problem
	}
	reportDigest, _ := canonical.Spell(canonical.Digest(report))
	workloadDigest, _ := canonical.Spell(canonical.Digest(workloads))
	if reportDigest != intent.Request.Report || workloadDigest != intent.Request.Workloads {
		return intent, exit.New(exit.Conflict, "assessment files changed their granted content identities")
	}
	intent.Report = records.ArtifactObjectRef{Digest: reportDigest, Length: int64(len(report))}
	intent.Workloads = records.ArtifactObjectRef{Digest: workloadDigest, Length: int64(len(workloads))}
	intent.Verdict = info.Verdict
	intent.RenderOwner = producer
	return intent, nil
}
func (o *modelTransferOwner) ApplyAssessmentPublication(ctx context.Context, call records.NativeCall, intent publication.AssessmentIntent, report []byte) ([]byte, *exit.Error) {
	var request publication.AssessmentRequest
	if problem := publication.DecodeEffect(call.Request, &request); problem != nil {
		return nil, problem
	}
	if !publication.SameAssessmentRequest(request, intent.Request) {
		return nil, exit.New(exit.Conflict, "assessment intent changed after admission")
	}
	client, problem := o.publicationEffectClient(ctx, call)
	if problem != nil {
		return nil, problem
	}
	ref, problem := publication.EffectDestination(call.Operation, call.Request)
	if problem != nil {
		return nil, problem
	}
	account, problem := publication.Authorize(ctx, client, ref, o.assessmentOperator(o.hubOf(call.ParentRequestID)))
	if problem != nil {
		return nil, problem
	}
	receipt, problem := publication.ApplyAssessment(ctx, client, intent, account, report, func() *exit.Error { return o.store.StartNativeEffectWrite(call.ID) })
	if problem != nil {
		return nil, problem
	}
	return publication.Canonical(receipt)
}

func (o *modelTransferOwner) assessmentOperator(origin string) bool {
	command := o.cliContext(origin, records.ModelTransferIntent{}, false)
	return command.Cfg.HubToken.Present() && !command.AccountAuth.CredentialPresent()
}
