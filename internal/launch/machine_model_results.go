package launch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

type RetainedModelResult struct {
	Pointer   string
	Canonical []byte
	Artifact  records.ModelArtifact
	Retention *pb.DerivedRetentionRequest
}

// ValidateMachineModelResults checks only declared model positions, using the exact captured
// result schema. Each model output stands on its own custody record: one that is missing,
// duplicated with different bytes, or fails its integrity check fails alone with a warning,
// and every other output is returned for collection. Only a result whose metadata cannot be
// read at all is refused.
func ValidateMachineModelResults(schema json.RawMessage, envelope *pb.ResultEnvelope) ([]RetainedModelResult, []string, bool, *exit.Error) {
	refuse := func(message string) ([]RetainedModelResult, []string, bool, *exit.Error) {
		return nil, nil, false, exit.Named(exit.Conflict, "machine_execution.model_result_invalid", "%s", message)
	}
	var declared Struct
	if json.Unmarshal(schema, &declared) != nil {
		return refuse("captured result schema is unreadable")
	}
	paths := ModelArtifactPaths(declared)
	if len(paths) == 0 && len(envelope.GetRetainedModels()) == 0 {
		return nil, nil, false, nil
	}
	if envelope == nil || envelope.ResultBlob != nil || len(envelope.InlineResult) == 0 || len(envelope.RetainedModels) > 32 {
		return refuse("model result needs its bounded inline result and bounded custody descriptors")
	}
	drift, problem := ValidateMachineResult(schema, envelope)
	if problem != nil {
		return refuse(problem.Message)
	}
	warnings := drift.Warnings()
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
		if !drift.Usable(strings.Join(path, ".")) {
			continue // its result position already failed, and says why
		}
		visit(envelope.InlineResult, path, "")
	}
	records_ := map[string]*pb.RetainedModelResult{}
	failed := map[string]string{}
	for _, retained := range envelope.RetainedModels {
		if retained == nil || len(retained.ResultPointer) > 1024 || len(retained.ModelArtifactCanonicalBytes) > 4096 {
			warnings = append(warnings, "the machine returned a malformed model custody record; ignored")
			continue
		}
		pointer := retained.ResultPointer
		if _, declared := expected[pointer]; !declared {
			warnings = append(warnings, fmt.Sprintf("the machine returned model custody for %q, which the result does not declare; ignored", pointer))
			continue
		}
		if prior, seen := records_[pointer]; seen {
			if !proto.Equal(prior, retained) {
				failed[pointer] = "the machine returned two different custody records for it"
			}
			continue
		}
		records_[pointer] = retained
	}
	var results []RetainedModelResult
	for _, pointer := range slices.Sorted(maps.Keys(expected)) {
		retained := records_[pointer]
		reason := failed[pointer]
		if reason == "" && retained == nil {
			reason = "the machine returned no custody record for it"
		}
		var artifact *records.ModelArtifact
		if reason == "" {
			artifact, reason = verifyModelCustody(expected[pointer], retained)
		}
		if reason != "" {
			warnings = append(warnings, fmt.Sprintf("model output %q failed: %s", modelOutputName(pointer), reason))
			continue
		}
		results = append(results, RetainedModelResult{Pointer: pointer, Canonical: retained.ModelArtifactCanonicalBytes, Artifact: *artifact, Retention: retained.Retention})
	}
	return results, warnings, len(paths) > 0, nil
}

// verifyModelCustody is one model output's own integrity check: its custody record names
// the exact artifact at its schema position, that artifact's native receipt, and actual
// transaction and root-held retention identities.
func verifyModelCustody(expected []byte, retained *pb.RetainedModelResult) (*records.ModelArtifact, string) {
	if !bytes.Equal(expected, retained.ModelArtifactCanonicalBytes) {
		return nil, "its custody record names different artifact bytes than its result position"
	}
	artifact, problem := records.DecodeModelArtifact(retained.ModelArtifactCanonicalBytes)
	if problem != nil || artifact == nil || retained.Retention == nil {
		return nil, "its custody record has no closed artifact and native custody descriptor"
	}
	receipt, err := canonical.Raw(artifact.TensorFSReceiptDigest)
	if err != nil || !bytes.Equal(receipt, retained.Retention.TensorfsReceiptDigest) {
		return nil, "its custody record names a different native receipt"
	}
	if _, err := canonical.Raw(retained.Retention.WeightsTransactionId); err != nil {
		return nil, "its custody record has no actual native transaction identity"
	}
	if _, err := canonical.Raw(retained.Retention.RetentionId); err != nil {
		return nil, "its custody record has no actual root-held retention identity"
	}
	return artifact, ""
}

// modelOutputName spells an RFC 6901 result pointer as the dotted output path.
func modelOutputName(pointer string) string {
	if pointer == "" {
		return "result"
	}
	parts := strings.Split(strings.TrimPrefix(pointer, "/"), "/")
	for i, part := range parts {
		parts[i] = strings.ReplaceAll(strings.ReplaceAll(part, "~1", "/"), "~0", "~")
	}
	return strings.Join(parts, ".")
}
