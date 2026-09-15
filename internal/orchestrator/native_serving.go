package orchestrator

import (
	"context"
	"sort"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// nativeServingModels binds child input custody before preparation can read it.
// These references authorize no registry request and carry no mutable model alias.
func (c *Orchestrator) nativeServingModels(request records.Request) ([]*pb.NativeModelBinding, *exit.Error) {
	if request.IsJob() || request.ParentRequestID == "" {
		return nil, exit.New(exit.Validation, "native serving requires an admitted private child")
	}
	if problem := c.retainChildInputs(request); problem != nil {
		return nil, problem
	}
	native, problem := c.opt.Store.NativeArtifactRetentions(request.ID)
	if problem != nil {
		return nil, problem
	}
	weights, problem := c.opt.Store.WeightsRetentions(request.ID)
	if problem != nil {
		return nil, problem
	}
	bindings := make([]*pb.NativeModelBinding, 0, len(request.Models))
	for _, model := range request.Models {
		if model.Downloadable() {
			continue // Hub inputs use the same exact-root downloader as top-level calls.
		}
		inputSlot := "result/" + model.Slot // childArtifacts uses one rooted JSON path for its custody rows.
		var source *pb.DerivedRetentionRequest
		for _, held := range native {
			if held.Kind == "input" && held.Slot == inputSlot && held.State == "held" && held.ArtifactKind == "derived" && held.ManifestID == model.Manifest && held.ManifestLength == model.ManifestLength {
				source = nativeTransferSource(held)
			}
		}
		for _, held := range weights {
			if held.Kind != "input" || held.Slot != inputSlot || held.State != "held" {
				continue
			}
			output, problem := c.opt.Store.ModelTransferWeights(held.ProducerRequestID, held.ProducerAttempt, held.ProducerOutputSlot)
			if problem != nil {
				return nil, problem
			}
			if output == nil || output.ManifestID != model.Manifest || output.ManifestLength != model.ManifestLength {
				continue
			}
			receipt, err := canonical.Read(output.Receipt, &pb.WeightsReceipt{})
			if err != nil {
				return nil, exit.Internalf("retained model receipt is not canonical")
			}
			digest, err := canonical.Raw(receipt.Str("tensorfs_receipt_digest"))
			if err != nil {
				return nil, exit.Internalf("retained model receipt identity is invalid")
			}
			source = &pb.DerivedRetentionRequest{WeightsTransactionId: output.TransactionID, TensorfsReceiptDigest: digest, RetentionId: held.RetentionID}
		}
		if source == nil || model.BindingPath == "" {
			return nil, exit.Named(exit.Conflict, "child.model_custody_absent", "serving preparation has no exact held model input")
		}
		digest, err := canonical.Raw(model.Manifest)
		if err != nil {
			return nil, exit.Internalf("retained model manifest identity is invalid")
		}
		bindings = append(bindings, &pb.NativeModelBinding{Slot: model.BindingPath, Model: model.Model, Manifest: &pb.Ref{Digest: digest, Length: uint64(model.ManifestLength)}, Retention: source, Adapters: downloadAdapters(model.Adapters)})
	}
	sort.Slice(bindings, func(i, j int) bool { return bindings[i].Slot < bindings[j].Slot })
	return bindings, nil
}

func mixedModelInputs(models []ModelRef) bool {
	var downloaded, native bool
	for _, model := range models {
		if model.Downloadable() {
			downloaded = true
		} else {
			native = true
		}
	}
	return downloaded && native
}

// The optional preparation capability is independent of protocol minor. Older
// peers omit it and remain usable for every single-source model selection.
func requireMixedModelInputs(s *session, mixed bool) *exit.Error {
	if !mixed {
		return nil
	}
	if s == nil {
		return exit.Unavailablef("mixed model preparation awaits its worker connection")
	}
	ctx, cancel := context.WithTimeout(s.ctx, 10*time.Second)
	defer cancel()
	var info *pb.ProtocolInfoResult
	var err error
	if s.host != nil {
		info, err = s.host.ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
	} else if s.preparation != nil {
		info, err = s.preparation.ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
	}
	if err != nil {
		return exit.Named(exit.Unavailable, "mixed_model_inputs_probe_failed", "worker model-input capability is unavailable")
	}
	if info == nil || !info.SupportsMixedModelInputs {
		return exit.Named(exit.Conflict, "mixed_model_inputs_unsupported", "this worker does not support retained and downloaded model inputs together").WithRemedy("update the worker Runtime and supervisor before running this model selection")
	}
	return nil
}
