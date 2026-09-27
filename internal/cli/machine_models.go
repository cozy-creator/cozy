package cli

import (
	"bytes"
	"context"
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// machineModelPlan is what one outcome's model outputs leave to collect. Each model output
// stands on its own custody record; one that fails is a warning, never the others' loss.
type machineModelPlan struct {
	models   []launch.RetainedModelResult
	warnings []string
	native   bool
}

func (m *machineRuns) planMachineModels(request records.Request, connection *machineConnection, body *pb.AttemptOutcomeBody) (*machineModelPlan, *exit.Error) {
	if body.Result == nil || !request.ChildArtifacts && len(body.Result.RetainedModels) == 0 {
		return nil, nil
	}
	surface, problem := m.resolver.capturedResultInterface(request)
	if problem != nil {
		return nil, problem
	}
	entrypoint, problem := surface.Function(request.Entrypoint)
	if problem != nil {
		return nil, problem
	}
	if body.Result.ResultBlob != nil {
		return nil, nil // ordinary file custody needs its separate byte transport
	}
	if len(launch.ModelArtifactPaths(entrypoint.Result)) > 0 && connection.WireMinor < 53 {
		return nil, exit.Named(exit.Unavailable, "machine_execution.model_collection_upgrade_required", "native model collection requires actual Runtime protocol 53; its result remains retained")
	}
	schema, problem := machineResultSchema(surface, request.Entrypoint)
	if problem != nil {
		return nil, problem
	}
	models, warnings, native, problem := launch.ValidateMachineModelResults(schema, body.Result)
	if problem != nil {
		return nil, problem
	}
	return &machineModelPlan{models: models, warnings: warnings, native: native}, nil
}

// covers is whether a collected model output already holds this weights transaction.
func (p *machineModelPlan) covers(receipt *pb.WeightsReceipt) bool {
	if p == nil || !p.native {
		return false
	}
	for _, model := range p.models {
		if receipt.WeightsTransactionId == model.Retention.WeightsTransactionId && receipt.TensorfsReceiptDigest == model.Artifact.TensorFSReceiptDigest {
			return true
		}
	}
	return false
}

// retainMachineWeights gives each weights output no collected model result carries a
// recipient custody of its own on the machine: a derived retention this host holds. The
// bytes then stay retained on the rental, counted as its disk, until released or the
// rental ends, and collection completes: nothing waits on a transfer nobody drives.
func (m *machineRuns) retainMachineWeights(ctx context.Context, request records.Request, connection *machineConnection,
	outcome *pb.AttemptOutcome, body *pb.AttemptOutcomeBody, plan *machineModelPlan) *exit.Error {
	for _, ref := range body.WeightsReceipts {
		var receipt pb.WeightsReceipt
		if ref == nil || !bytes.Equal(canonical.Digest(ref.WeightsReceiptCanonicalBytes), ref.WeightsReceiptDigest) ||
			canonical.Unmarshal(ref.WeightsReceiptCanonicalBytes, &receipt) != nil || plan.covers(&receipt) {
			continue
		}
		digest, err := canonical.Raw(receipt.TensorfsReceiptDigest)
		if err != nil {
			continue
		}
		if connection.WireMinor < 53 {
			return exit.Named(exit.Unavailable, "machine_execution.result_custody_required",
				"weights output %q can be held on the machine only by Runtime protocol 53 or newer", receipt.OutputSlot)
		}
		artifact := records.ModelArtifact{ProducerRequestID: request.ID, OutputSlot: receipt.OutputSlot, TensorFSReceiptDigest: receipt.TensorfsReceiptDigest}
		hold := records.MachineModelRetention{OutcomeID: outcome.OutcomeId, ResultPointer: "weights/" + receipt.OutputSlot,
			TransactionID: receipt.WeightsTransactionId, ReceiptDigest: digest,
			RetentionID: records.ArtifactRetentionID(request.ID, "machine-weights", receipt.OutputSlot, artifact)}
		// The retention id is fixed by the output, so an ask repeated after a lost answer
		// holds the same bytes once; the root hold is released only after this is recorded.
		received, err := connection.retainModel(ctx, modelRetentionRequest(hold))
		if err != nil {
			return machineTransport(err)
		}
		if received == nil || received.WeightsTransactionId != hold.TransactionID || !bytes.Equal(received.TensorfsReceiptDigest, digest) ||
			received.RetentionId != hold.RetentionID || received.Released || received.Manifest == nil {
			return exit.New(exit.Conflict, "machine returned a different weights custody receipt")
		}
		artifact.Manifest.Digest, _ = canonical.Spell(received.Manifest.Digest)
		artifact.Manifest.Length = int64(received.Manifest.Length)
		if hold.Artifact, err = json.Marshal(artifact); err != nil {
			return exit.Internalf("cannot record weights custody: %s", err)
		}
		if problem := m.store.FreezeMachineModelRetention(request.ID, hold); problem != nil {
			return problem
		}
		if problem := m.store.AdvanceMachineModelRetention(request.ID, hold.RetentionID, "held"); problem != nil {
			return problem
		}
	}
	return nil
}

func (m *machineRuns) collectMachineModels(ctx context.Context, request records.Request, connection *machineConnection, outcome *pb.AttemptOutcome, plan *machineModelPlan) (bool, *exit.Error) {
	if plan == nil || !plan.native {
		return plan != nil && plan.native, nil
	}
	for _, model := range plan.models {
		hold := records.MachineModelRetention{
			OutcomeID: outcome.OutcomeId, ResultPointer: model.Pointer, Artifact: model.Canonical,
			TransactionID: model.Retention.WeightsTransactionId, ReceiptDigest: model.Retention.TensorfsReceiptDigest,
			SourceRetentionID: model.Retention.RetentionId,
			RetentionID:       records.ArtifactRetentionID(request.ID, "machine-result", model.Pointer, model.Artifact),
		}
		if hold.RetentionID == hold.SourceRetentionID {
			return false, exit.New(exit.Conflict, "model recipient custody must be independent of the Runtime root hold")
		}
		if problem := m.store.FreezeMachineModelRetention(request.ID, hold); problem != nil {
			return false, problem
		}
		received, err := connection.retainModel(ctx, modelRetentionRequest(hold))
		if err != nil {
			return false, machineTransport(err)
		}
		if problem := verifyModelRetention(hold, received, false); problem != nil {
			return false, problem
		}
		if problem := m.store.AdvanceMachineModelRetention(request.ID, hold.RetentionID, "held"); problem != nil {
			return false, problem
		}
	}
	return true, nil
}

func modelRetentionRequest(hold records.MachineModelRetention) *pb.DerivedRetentionRequest {
	return &pb.DerivedRetentionRequest{WeightsTransactionId: hold.TransactionID, TensorfsReceiptDigest: hold.ReceiptDigest, RetentionId: hold.RetentionID}
}

func verifyModelRetention(hold records.MachineModelRetention, received *pb.DerivedRetentionResult, released bool) *exit.Error {
	if received == nil || received.WeightsTransactionId != hold.TransactionID || !bytes.Equal(received.TensorfsReceiptDigest, hold.ReceiptDigest) || received.RetentionId != hold.RetentionID || received.Released != released {
		return exit.New(exit.Conflict, "machine returned a different model custody receipt")
	}
	if !released || received.Manifest != nil {
		artifact, problem := records.DecodeModelArtifact(hold.Artifact)
		if problem != nil || artifact == nil {
			return exit.New(exit.Conflict, "recorded model artifact is invalid")
		}
		digest, _ := canonical.Raw(artifact.Manifest.Digest)
		if received.Manifest == nil || !bytes.Equal(digest, received.Manifest.Digest) || received.Manifest.Length != uint64(artifact.Manifest.Length) {
			return exit.New(exit.Conflict, "machine model custody names a different manifest")
		}
	}
	return nil
}

func (m *machineRuns) releaseMachineModels(ctx context.Context, request string, connection *machineConnection) *exit.Error {
	holds, problem := m.store.MachineModelRetentions(request)
	if problem != nil {
		return problem
	}
	for _, hold := range holds {
		if hold.State == "released" {
			continue
		}
		if problem := m.store.AdvanceMachineModelRetention(request, hold.RetentionID, "releasing"); problem != nil {
			return problem
		}
		result, err := connection.releaseModel(ctx, modelRetentionRequest(hold))
		if err != nil {
			return machineTransport(err)
		}
		if problem := verifyModelRetention(hold, result, true); problem != nil {
			return problem
		}
		if problem := m.store.AdvanceMachineModelRetention(request, hold.RetentionID, "released"); problem != nil {
			return problem
		}
	}
	return nil
}
