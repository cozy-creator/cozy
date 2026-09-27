package launch

import (
	"bytes"
	"encoding/json"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type RetainedModelResult struct {
	Pointer   string
	Canonical []byte
	Artifact  records.ModelArtifact
	Retention *pb.DerivedRetentionRequest
}

// ValidateMachineModelResults checks only declared model positions, using the
// exact captured result schema. Metadata is verified as a complete set before
// any independent native hold is acquired.
func ValidateMachineModelResults(schema json.RawMessage, envelope *pb.ResultEnvelope) ([]RetainedModelResult, bool, *exit.Error) {
	refuse := func(message string) ([]RetainedModelResult, bool, *exit.Error) {
		return nil, false, exit.Named(exit.Conflict, "machine_execution.model_result_invalid", "%s", message)
	}
	var declared Struct
	if json.Unmarshal(schema, &declared) != nil {
		return refuse("captured result schema is unreadable")
	}
	paths := ModelArtifactPaths(declared)
	if len(paths) == 0 && len(envelope.GetRetainedModels()) == 0 {
		return nil, false, nil
	}
	if envelope == nil || envelope.ResultBlob != nil || len(envelope.InlineResult) == 0 || len(envelope.RetainedModels) > 32 {
		return refuse("model result needs its bounded inline result and complete custody descriptors")
	}
	drift, problem := ValidateMachineResult(schema, envelope)
	if problem != nil {
		return refuse(problem.Message)
	}
	for _, path := range paths {
		if !drift.Usable(strings.Join(path, ".")) {
			return refuse("a declared model output failed: " + strings.Join(drift.Warnings(), "; "))
		}
	}
	inline := envelope.InlineResult
	expected := map[string][]byte{}
	var visit func(json.RawMessage, []string, string)
	visit = func(value json.RawMessage, path []string, pointer string) {
		if len(path) == 0 {
			model, problem := records.DecodeModelArtifact(value)
			if problem == nil && model != nil {
				expected[pointer], _ = canonical.NormalizeJCS(value)
			}
			return
		}
		if path[0] == "*" {
			var values []json.RawMessage
			if json.Unmarshal(value, &values) == nil {
				for i, child := range values {
					visit(child, path[1:], pointer+"/"+strconv.Itoa(i))
				}
			}
			return
		}
		var object map[string]json.RawMessage
		if json.Unmarshal(value, &object) == nil {
			segment := strings.ReplaceAll(strings.ReplaceAll(path[0], "~", "~0"), "/", "~1")
			visit(object[path[0]], path[1:], pointer+"/"+segment)
		}
	}
	for _, path := range paths {
		visit(inline, path, "")
	}
	if len(expected) != len(envelope.RetainedModels) {
		return refuse("model result custody descriptors omit or add a declared artifact")
	}
	var results []RetainedModelResult
	for i, retained := range envelope.RetainedModels {
		if retained == nil || len(retained.ResultPointer) > 1024 || len(retained.ModelArtifactCanonicalBytes) > 4096 ||
			i > 0 && envelope.RetainedModels[i-1].ResultPointer >= retained.ResultPointer ||
			!bytes.Equal(expected[retained.ResultPointer], retained.ModelArtifactCanonicalBytes) {
			return refuse("model result pointer or exact artifact bytes differ from its schema position")
		}
		artifact, problem := records.DecodeModelArtifact(retained.ModelArtifactCanonicalBytes)
		if problem != nil || artifact == nil || retained.Retention == nil {
			return refuse("model result has no closed artifact and native custody descriptor")
		}
		receipt, err := canonical.Raw(artifact.TensorFSReceiptDigest)
		if err != nil || !bytes.Equal(receipt, retained.Retention.TensorfsReceiptDigest) {
			return refuse("model result names a different native receipt")
		}
		if _, err := canonical.Raw(retained.Retention.WeightsTransactionId); err != nil {
			return refuse("model result has no actual native transaction identity")
		}
		if _, err := canonical.Raw(retained.Retention.RetentionId); err != nil {
			return refuse("model result has no actual root-held retention identity")
		}
		results = append(results, RetainedModelResult{Pointer: retained.ResultPointer, Canonical: retained.ModelArtifactCanonicalBytes, Artifact: *artifact, Retention: retained.Retention})
	}
	return results, len(paths) > 0, nil
}
