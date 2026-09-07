package orchestrator

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func childArtifacts(raw []byte, paths [][]string) (map[string]records.ModelArtifact, *exit.Error) {
	var value any
	if json.Unmarshal(raw, &value) != nil {
		return nil, exit.New(exit.Validation, "child artifact result is not JSON")
	}
	out := map[string]records.ModelArtifact{}
	var walk func(any, []string, string) *exit.Error
	walk = func(value any, segments []string, path string) *exit.Error {
		if value == nil {
			return nil
		}
		if len(segments) == 0 {
			body, _ := json.Marshal(value)
			artifact, problem := records.DecodeModelArtifact(body)
			if problem != nil {
				return problem
			}
			if len(out) >= 32 {
				return exit.New(exit.Validation, "child artifact references exceed 32")
			}
			out[path] = *artifact
			return nil
		}
		if len(segments) > 12 {
			return exit.New(exit.Validation, "child artifact nesting exceeds its bound")
		}
		if segments[0] == "*" {
			values, ok := value.([]any)
			if !ok {
				return exit.New(exit.Validation, "child artifact list differs from its result schema")
			}
			for index, child := range values {
				if problem := walk(child, segments[1:], fmt.Sprintf("%s/%d", path, index)); problem != nil {
					return problem
				}
			}
			return nil
		}
		object, ok := value.(map[string]any)
		if !ok {
			return exit.New(exit.Validation, "child artifact structure differs from its result schema")
		}
		child, exists := object[segments[0]]
		if !exists {
			return nil
		}
		escaped := strings.ReplaceAll(strings.ReplaceAll(segments[0], "~", "~0"), "/", "~1")
		return walk(child, segments[1:], path+"/"+escaped)
	}
	for _, path := range paths {
		if problem := walk(value, path, "result"); problem != nil {
			return nil, problem
		}
	}
	return out, nil
}

func (c *Orchestrator) childResultArtifacts(request records.Request, raw []byte) (map[string]records.ModelArtifact, *exit.Error) {
	if !request.ChildArtifacts && (request.WeightsOutputs == "" || request.WeightsOutputs == "[]") {
		return nil, nil
	}
	resolver, ok := c.opt.Packages.(interface {
		PrivateArtifactPaths(records.Request) ([][]string, *exit.Error)
	})
	if !ok {
		return nil, exit.Internalf("child result schema owner is unavailable")
	}
	paths, problem := resolver.PrivateArtifactPaths(request)
	if problem != nil {
		return nil, problem
	}
	artifacts, problem := childArtifacts(raw, paths)
	if problem != nil {
		return nil, problem
	}
	original := request.ID
	if request.ReusedFrom != "" {
		original = request.ReusedFrom
	}
	producer, problem := c.opt.Store.RequestRow(original)
	if problem != nil || producer == nil {
		return nil, exit.Unavailablef("child result producer is unavailable")
	}
	outputs, problem := c.opt.Store.AllModelTransferWeights(original, producer.Ordinal)
	if problem != nil {
		return nil, problem
	}
	declared, problem := decodeWeightsOutputs(request.WeightsOutputs)
	if problem != nil {
		return nil, problem
	}
	if len(outputs) != len(declared) {
		return nil, exit.Unavailablef("child native output receipts are not yet fully observed")
	}
	for _, output := range outputs {
		receipt, err := canonical.Read(output.Receipt, &pb.WeightsReceipt{})
		if err != nil {
			return nil, exit.Internalf("child native output receipt is malformed")
		}
		artifacts["weights/"+output.OutputSlot] = records.ModelArtifact{ProducerRequestID: original, OutputSlot: output.OutputSlot,
			Manifest: records.ArtifactObjectRef{Digest: output.ManifestID, Length: output.ManifestLength}, TensorFSReceiptDigest: receipt.Str("tensorfs_receipt_digest")}
	}
	return artifacts, nil
}

func (c *Orchestrator) retainChildArtifacts(ctx context.Context, consumer records.Request, kind string, artifacts map[string]records.ModelArtifact) *exit.Error {
	slots := make([]string, 0, len(artifacts))
	for slot := range artifacts {
		slots = append(slots, slot)
	}
	sort.Strings(slots)
	for _, slot := range slots {
		artifact := artifacts[slot]
		weights, problem := c.opt.Store.ArtifactOutput(artifact)
		if problem != nil {
			return problem
		}
		owned, problem := c.opt.Store.ArtifactHasCustody(weights.RequestID, weights.Attempt, weights.OutputSlot, consumer.ReuseScope)
		if problem != nil {
			return problem
		}
		if !owned && consumer.ReusedFrom != weights.RequestID {
			return exit.Named(exit.Conflict, "child.artifact_released", "model artifact has no retained native owner")
		}
		retentionID := records.ArtifactRetentionID(consumer.ID, kind, slot, artifact)
		retention, problem := c.opt.Store.RecordWeightsRetention(records.WeightsRetention{RequestID: consumer.ID, Kind: kind, Slot: slot,
			ProducerRequestID: weights.RequestID, ProducerAttempt: weights.Attempt, ProducerOutputSlot: weights.OutputSlot, RetentionID: retentionID})
		if problem != nil {
			return problem
		}
		if retention.State == "held" {
			continue
		}
		if problem := c.changeDerivedRetention(ctx, retention, false); problem != nil {
			return problem
		}
	}
	return nil
}

func (c *Orchestrator) changeDerivedRetention(ctx context.Context, retention records.WeightsRetention, release bool) *exit.Error {
	weights, problem := c.opt.Store.ModelTransferWeights(retention.ProducerRequestID, retention.ProducerAttempt, retention.ProducerOutputSlot)
	if problem != nil || weights == nil {
		return exit.Named(exit.Conflict, "child.artifact_provenance_lost", "native artifact provenance is unavailable")
	}
	receipt, err := canonical.Read(weights.Receipt, &pb.WeightsReceipt{})
	if err != nil {
		return exit.Internalf("native artifact receipt is not canonical")
	}
	digest, err := canonical.Raw(receipt.Str("tensorfs_receipt_digest"))
	if err != nil {
		return exit.Internalf("native artifact receipt digest is malformed")
	}
	producer, problem := c.opt.Store.RequestRow(weights.RequestID)
	if problem != nil || producer == nil {
		return exit.Unavailablef("native artifact producer is unavailable")
	}
	var session *session
	if producer.Worker != "" {
		session, problem = c.rentalControl(producer.Worker)
	} else {
		// All local package workers share the same authoritative TensorFS home.
		// Any current claimed preparation service can acquire the native root.
		c.mu.Lock()
		for _, candidate := range c.sessions {
			if candidate.host == nil && candidate.preparation != nil {
				session = candidate
				break
			}
		}
		c.mu.Unlock()
	}
	if problem != nil {
		return problem
	}
	if session == nil {
		return exit.Unavailablef("native artifact retention awaits a live preparation service")
	}
	request := &pb.DerivedRetentionRequest{WeightsTransactionId: weights.TransactionID, TensorfsReceiptDigest: digest, RetentionId: retention.RetentionID}
	var result *pb.DerivedRetentionResult
	if session.host != nil {
		call := &pb.DerivedRetentionCall{Claim: session.claim, Request: request}
		if release {
			result, err = session.host.ReleaseDerivedRetention(ctx, call)
		} else {
			result, err = session.host.RetainDerivedResult(ctx, call)
		}
	} else if release {
		result, err = session.preparation.ReleaseDerivedRetention(ctx, request)
	} else {
		result, err = session.preparation.RetainDerivedResult(ctx, request)
	}
	if err != nil {
		switch status.Code(err) {
		case codes.InvalidArgument, codes.PermissionDenied, codes.FailedPrecondition, codes.NotFound:
			return exit.Named(exit.Conflict, "child.artifact_retention_refused", "native artifact retention refused its captured owner or receipt")
		}
		return exit.Unavailablef("native artifact retention awaits its current owner")
	}
	if result == nil || result.RetentionId != request.RetentionId || result.WeightsTransactionId != request.WeightsTransactionId || !bytes.Equal(result.TensorfsReceiptDigest, digest) || result.Released != release {
		return exit.Named(exit.Structural, "child.artifact_retention_changed", "native artifact retention changed its exact original receipt or owner intent")
	}
	if result.Manifest != nil {
		manifest, _ := canonical.Spell(result.Manifest.Digest)
		if manifest != weights.ManifestID || result.Manifest.Length != uint64(weights.ManifestLength) {
			return exit.Named(exit.Structural, "child.artifact_manifest_changed", "native artifact retention changed its original manifest")
		}
	} else if !release {
		return exit.Named(exit.Structural, "child.artifact_manifest_missing", "native artifact retention returned no manifest")
	}
	if release {
		return c.opt.Store.CompleteWeightsRetentionRelease(retention.RetentionID)
	}
	return c.opt.Store.ConfirmWeightsRetention(retention.RetentionID, session.instanceID, session.bootID)
}

func (c *Orchestrator) releaseOriginalDerivedResults(id string) *exit.Error {
	request, problem := c.opt.Store.RequestRow(id)
	if problem != nil || request == nil {
		return problem
	}
	if request.ParentRequestID == "" && !request.ChildArtifacts {
		return nil
	}
	outputs, problem := c.opt.Store.AllModelTransferWeights(id, request.Ordinal)
	if problem != nil {
		return problem
	}
	if len(outputs) == 0 {
		return nil
	}
	var session *session
	if request.Worker != "" {
		session, problem = c.rentalControl(request.Worker)
	} else {
		c.mu.Lock()
		for _, candidate := range c.sessions {
			if candidate.host == nil && candidate.preparation != nil {
				session = candidate
				break
			}
		}
		c.mu.Unlock()
	}
	if problem != nil {
		return problem
	}
	if session == nil {
		return exit.Unavailablef("original artifact release awaits its native owner")
	}
	for _, output := range outputs {
		receipt, err := canonical.Read(output.Receipt, &pb.WeightsReceipt{})
		if err != nil {
			return exit.Internalf("original artifact receipt is not canonical")
		}
		digest, err := canonical.Raw(receipt.Str("tensorfs_receipt_digest"))
		if err != nil {
			return exit.Internalf("original artifact native receipt digest is malformed")
		}
		call := &pb.DerivedResultReleaseRequest{WeightsTransactionId: output.TransactionID, TensorfsReceiptDigest: digest}
		var result *pb.DerivedResultReleaseResult
		if session.host != nil {
			result, err = session.host.ReleaseDerivedResult(context.Background(), &pb.DerivedResultReleaseCall{Claim: session.claim, Request: call})
		} else {
			result, err = session.preparation.ReleaseDerivedResult(context.Background(), call)
		}
		if err != nil {
			return exit.Unavailablef("original artifact release awaits native disposal")
		}
		if result == nil || !result.Released || result.WeightsTransactionId != output.TransactionID || !bytes.Equal(result.TensorfsReceiptDigest, digest) {
			return exit.Named(exit.Structural, "child.original_release_changed", "native disposal changed the original result authority")
		}
	}
	return nil
}

func (c *Orchestrator) retainChildInputs(request records.Request) *exit.Error {
	paths := make([][]string, 0, len(request.Models))
	for _, model := range request.Models {
		paths = append(paths, []string{model.Slot})
	}
	artifacts, problem := childArtifacts(request.Payload, paths)
	if problem != nil {
		return problem
	}
	return c.retainChildArtifacts(context.Background(), request, "input", artifacts)
}

func (c *Orchestrator) releaseChildRetentions(id string, inputsOnly bool) *exit.Error {
	if problem := c.opt.Store.BeginWeightsRetentionRelease(id, inputsOnly); problem != nil {
		return problem
	}
	retentions, problem := c.opt.Store.WeightsRetentions(id)
	if problem != nil {
		return problem
	}
	for _, retention := range retentions {
		if retention.State == "releasing" {
			if problem := c.changeDerivedRetention(context.Background(), retention, true); problem != nil {
				return problem
			}
		}
	}
	return nil
}
