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

func (m *machineRuns) collectMachineModels(ctx context.Context, request records.Request, connection *machineConnection, outcome *pb.AttemptOutcome, body *pb.AttemptOutcomeBody) (bool, *exit.Error) {
	if body.Result == nil || !request.ChildArtifacts && len(body.Result.RetainedModels) == 0 {
		return false, nil
	}
	_, surface, problem := m.resolver.installPackageInterface(request.InstallID)
	if problem != nil {
		return false, problem
	}
	entrypoint, problem := surface.Function(request.Entrypoint)
	if problem != nil {
		return false, problem
	}
	if len(launch.RetainedAssetPaths(entrypoint)) > 0 || len(body.GetOutputManifest().GetOutputs()) > 0 || body.Result.ResultBlob != nil {
		return false, nil // ordinary file custody needs its separate byte transport
	}
	if len(launch.ModelArtifactPaths(entrypoint.Result)) > 0 && connection.wireMinor < 53 {
		return false, exit.Named(exit.Unavailable, "machine_execution.model_collection_upgrade_required", "native model collection requires actual Runtime protocol 53; its result remains retained")
	}
	var document struct {
		Jobs []struct {
			Name   string          `json:"name"`
			Result json.RawMessage `json:"result"`
		} `json:"jobs"`
	}
	if json.Unmarshal(surface.Raw, &document) != nil {
		return false, exit.New(exit.Conflict, "captured result schema is unreadable")
	}
	var schema json.RawMessage
	for _, job := range document.Jobs {
		if job.Name == request.Entrypoint {
			schema = job.Result
		}
	}
	models, native, problem := launch.ValidateMachineModelResults(schema, body.Result)
	if problem != nil || !native {
		return native, problem
	}
	for _, ref := range body.WeightsReceipts {
		var output pb.WeightsReceipt
		if ref == nil || !bytes.Equal(canonical.Digest(ref.WeightsReceiptCanonicalBytes), ref.WeightsReceiptDigest) || canonical.Unmarshal(ref.WeightsReceiptCanonicalBytes, &output) != nil {
			return false, exit.New(exit.Conflict, "machine model output receipt changed its exact bytes")
		}
		covered := false
		for _, model := range models {
			covered = covered || output.WeightsTransactionId == model.Retention.WeightsTransactionId && output.TensorfsReceiptDigest == model.Artifact.TensorFSReceiptDigest
		}
		if !covered {
			return false, exit.Named(exit.Unavailable, "machine_execution.result_custody_required", "a declared model output has no returned artifact custody descriptor; outputs remain retained")
		}
	}
	for _, model := range models {
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
