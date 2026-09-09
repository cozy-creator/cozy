package orchestrator

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"math"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

func byteRetentionRequest(h records.NativeArtifactRetention) *pb.NativeByteRetentionRequest {
	receipt, _ := canonical.Raw(h.ReceiptDigest)
	manifest, _ := canonical.Raw(h.ManifestID)
	return &pb.NativeByteRetentionRequest{RetentionId: h.RetentionID, Source: &pb.NativeByteTreeRef{ProducerRootId: h.TransactionID, ReceiptDigest: receipt, Manifest: &pb.Ref{Digest: manifest, Length: uint64(h.ManifestLength)}, ContentBytes: uint64(h.ContentBytes)}}
}
func (c *Orchestrator) changeByteRetention(ctx context.Context, h records.NativeArtifactRetention, release bool) *exit.Error {
	session, problem := c.workspaceControlContext(ctx, h.OwnerWorker)
	if problem != nil {
		return problem
	}
	if session == nil {
		return exit.Unavailablef("byte artifact awaits its original private workspace")
	}
	request := byteRetentionRequest(h)
	call := &pb.NativeByteRetentionCall{Claim: session.claim, Request: request}
	var result *pb.NativeByteRetentionResult
	var err error
	if session.host != nil {
		if release {
			result, err = session.host.ReleaseByteTree(ctx, call)
		} else {
			result, err = session.host.RetainByteTree(ctx, call)
		}
	} else if release {
		result, err = session.preparation.WorkspaceReleaseByteTree(ctx, call)
	} else {
		result, err = session.preparation.WorkspaceRetainByteTree(ctx, call)
	}
	if err != nil {
		return exit.Unavailablef("native byte custody awaits its retained workspace")
	}
	if result == nil || !proto.Equal(result.Source, request.Source) || result.RetentionId != h.RetentionID || result.Released != release {
		return exit.Named(exit.Structural, "child.byte_retention_changed", "byte custody acknowledgement changed its exact subject")
	}
	if release {
		return c.opt.Store.CompleteNativeArtifactRelease(h.RetentionID)
	}
	if problem := c.opt.Store.ConfirmNativeArtifact(h.RetentionID, session.instanceID, session.bootID); problem != nil {
		if problem.ErrName() == "child.artifact_release_pending" {
			_ = c.changeByteRetention(ctx, h, true)
		}
		return problem
	}
	return nil
}
func (c *Orchestrator) childByteGrants(s *session, call *pb.ChildCallRequest, request records.Request) ([]*pb.ChildByteResultGrant, *pb.ExecutionObservation, *exit.Error) {
	if _, problem := c.childParent(s, call.ParentRequestId, call.ParentAttemptOrdinal, call.ParentInvocationSpecDigest); problem != nil {
		return nil, nil, problem
	}
	sourceID := request.ID
	if request.ReusedFrom != "" {
		sourceID = request.ReusedFrom
	}
	source, problem := c.opt.Store.RequestRow(sourceID)
	if problem != nil || source == nil {
		return nil, nil, exit.Unavailablef("byte result source is absent")
	}
	if source.Worker != request.Worker {
		return nil, nil, exit.Unavailablef("byte result moved from its private workspace")
	}
	outputs, problem := c.opt.Store.ByteOutputs(source.ID, source.Ordinal)
	if problem != nil {
		return nil, nil, problem
	}
	grants := make([]*pb.ChildByteResultGrant, 0, len(outputs))
	for _, output := range outputs {
		h, problem := c.opt.Store.ReserveByteResult(request.ID, output)
		if problem != nil {
			return nil, nil, problem
		}
		if h.State != "held" {
			if problem := c.changeByteRetention(s.ctx, h, false); problem != nil {
				return nil, nil, problem
			}
		}
		grants = append(grants, &pb.ChildByteResultGrant{OutputId: output.OutputID, Source: output.NativeRef(), RetentionId: h.RetentionID})
	}
	attempt, problem := c.opt.Store.AttemptRow(source.ID, source.Ordinal)
	if problem != nil || attempt == nil {
		return nil, nil, exit.Unavailablef("byte result outcome is absent")
	}
	doc, err := canonical.Read(attempt.TerminalBody, &pb.AttemptOutcomeBody{})
	if err != nil {
		return nil, nil, exit.Internalf("byte result outcome is invalid")
	}
	observation, problem := readExecutionObservation(doc.Sub("observation"))
	return grants, observation, problem
}

func readExecutionObservation(doc canonical.Doc) (*pb.ExecutionObservation, *exit.Error) {
	if len(doc) == 0 {
		return nil, nil
	}
	raw, err := json.Marshal(doc)
	if err != nil {
		return nil, exit.New(exit.Validation, "execution observation is invalid")
	}
	// Canonical documents spell digests, while protobuf JSON uses base64 for bytes.
	var value map[string]any
	_ = json.Unmarshal(raw, &value)
	for field, keys := range map[string][]string{"environment": {"worker_image_digest", "execution_contract_digest"}, "capture": {"content_digest"}} {
		row, _ := value[field].(map[string]any)
		for _, key := range keys {
			if text, ok := row[key].(string); ok {
				digest, e := canonical.Raw(text)
				if e != nil {
					return nil, exit.New(exit.Validation, "execution observation digest is invalid")
				}
				row[key] = base64.StdEncoding.EncodeToString(digest)
			}
		}
	}
	raw, _ = json.Marshal(value)
	var observation pb.ExecutionObservation
	if err := protojson.Unmarshal(raw, &observation); err != nil {
		return nil, exit.New(exit.Validation, "execution observation has an unsupported shape")
	}
	return &observation, nil
}

func (c *Orchestrator) privateByteOutputs(req records.Request, attempt records.Attempt, doc canonical.Doc) ([]records.ByteOutput, *exit.Error) {
	var out []records.ByteOutput
	rows := doc.Sub("output_manifest").List("outputs")
	if len(rows) > pb.MaxChildArtifactGrants {
		return nil, exit.New(exit.Validation, "child byte output count exceeds 32")
	}
	bounds, ok := c.opt.Packages.(interface {
		PrivateByteOutputBound(records.Request, string, string) (int64, *exit.Error)
	})
	if !ok && len(rows) > 0 {
		return nil, exit.Unavailablef("byte result schema verifier is absent")
	}

	var result any
	inline, err := base64.StdEncoding.Strict().DecodeString(doc.Sub("result").Str("inline_result"))
	if err != nil || len(inline) > childCallMaxBytes {
		return nil, exit.New(exit.Validation, "byte result metadata exceeds its ordinary bound")
	}
	if len(inline) > 0 {
		decoder := json.NewDecoder(strings.NewReader(string(inline)))
		decoder.UseNumber()
		if decoder.Decode(&result) != nil {
			return nil, exit.New(exit.Validation, "byte result metadata is invalid")
		}
	}
	observation, problem := readExecutionObservation(doc.Sub("observation"))
	if problem != nil {
		return nil, problem
	}
	if observation != nil && observation.Environment != nil && observation.Environment.WorkerBootId != attempt.SessionID {
		return nil, exit.New(exit.Validation, "execution observation names another worker boot")
	}
	if req.Capture != "" && outcomeStatus(doc.Int("status")) == "SUCCEEDED" && (observation == nil || observation.Capture == nil || observation.Capture.OutputId != "runtime.capture" || len(observation.Capture.ContentDigest) != 32) {
		return nil, exit.New(exit.Validation, "capture request lacks its exact outcome observation")
	}
	if req.Capture == "" && observation != nil && observation.Capture != nil {
		return nil, exit.New(exit.Validation, "outcome contains unrequested capture")
	}
	declared := map[string]bool{}
	for _, id := range splitList(req.Outputs) {
		declared[id] = true
	}
	for _, value := range rows {
		e := value
		native := e.Sub("native_tree")
		manifest := native.Sub("manifest")
		b := records.ByteOutput{RequestID: req.ID, Attempt: attempt.Attempt, OutputID: e.Str("output_id"), Digest: e.Str("digest"), Length: e.Int("length"), MimeType: e.Str("mime_type"), ProducerRootID: native.Str("producer_root_id"), ReceiptDigest: native.Str("receipt_digest"), ManifestID: manifest.Str("digest"), ManifestLength: manifest.Int("length"), ContentBytes: native.Int("content_bytes")}
		if len(native) == 0 || b.Length < 0 || b.ContentBytes < 0 || b.ManifestLength <= 0 || len(b.OutputID) > 512 || strings.ContainsAny(b.OutputID, "/\\") || !declared[b.OutputID] {
			return nil, exit.New(exit.Validation, "byte result names an ungranted field or invalid native custody")
		}
		if problem := records.ValidateByteRef(b.NativeRef()); problem != nil {
			return nil, problem
		}
		var maximum int64
		var problem *exit.Error
		if b.OutputID == "runtime.capture" && req.Capture != "" {
			maximum = min(int64(c.maxOutputBytes()), 128<<20)
		} else {
			maximum, problem = bounds.PrivateByteOutputBound(req, b.OutputID, b.MimeType)
		}
		if problem != nil {
			return nil, problem
		}
		if b.Length > maximum || b.ContentBytes > maximum || uint64(b.Length) > math.MaxInt64 {
			return nil, exit.New(exit.Validation, "byte output exceeds its declared capacity")
		}
		if b.OutputID != "runtime.capture" {
			if problem := verifyByteResultRow(result, b); problem != nil {
				return nil, problem
			}
		} else if b.MimeType != "application/vnd.cozy.tree-manifest" || b.Digest != b.ManifestID || b.Length != b.ManifestLength {
			return nil, exit.New(exit.Validation, "capture is not its native Tree manifest")
		}
		delete(declared, b.OutputID)
		out = append(out, b)
	}
	if outcomeStatus(doc.Int("status")) == "SUCCEEDED" && len(declared) > 0 {
		return nil, exit.New(exit.Validation, "successful child omitted a declared byte output")
	}
	return out, nil
}

func captureOptions(value *pb.ActivationCapture) (string, *exit.Error) {
	if value == nil {
		return "", nil
	}
	if len(value.Components) == 0 || len(value.Components) > 32 || len(value.Steps) > 4096 {
		return "", exit.New(exit.Validation, "capture exceeds component or timestep bounds")
	}
	seen := map[string]bool{}
	for _, name := range value.Components {
		if name == "" || len(name) > 128 || seen[name] {
			return "", exit.New(exit.Validation, "capture component selection is invalid")
		}
		seen[name] = true
	}
	for index, step := range value.Steps {
		if (index == 0 && step != 0) || (index > 0 && step <= value.Steps[index-1]) {
			return "", exit.New(exit.Validation, "capture steps must start at zero and increase")
		}
	}
	steps := append([]uint32{}, value.Steps...)
	raw, _ := json.Marshal(map[string]any{"components": value.Components, "steps": steps})
	raw, err := canonical.NormalizeJCS(raw)
	if err != nil {
		return "", exit.New(exit.Validation, "capture identity is invalid")
	}
	return string(raw), nil
}

func verifyByteResultRow(result any, b records.ByteOutput) *exit.Error {
	value := result
	for _, part := range strings.Split(b.OutputID, ".") {
		switch current := value.(type) {
		case map[string]any:
			value = current[part]
		case []any:
			index, err := strconv.Atoi(part)
			if err != nil || index < 0 || index >= len(current) {
				return exit.New(exit.Validation, "byte result path is absent")
			}
			value = current[index]
		default:
			return exit.New(exit.Validation, "byte result path is not declared metadata")
		}
	}
	row, ok := value.(map[string]any)
	if !ok {
		return exit.New(exit.Validation, "byte result has no final asset row")
	}
	size, err := strconv.ParseInt(stringNumber(row["size_bytes"]), 10, 64)
	if err != nil || row["asset_ref"] != b.Digest || row["digest"] != b.Digest || size != b.ContentBytes {
		return exit.New(exit.Validation, "byte result row changed its final native identity")
	}
	if row["kind"] == "tree" {
		if b.MimeType != "application/vnd.cozy.tree-manifest" || b.Digest != b.ManifestID || b.Length != b.ManifestLength {
			return exit.New(exit.Validation, "Tree result confuses manifest length with content size")
		}
	} else {
		if row["media_type"] != b.MimeType || b.ContentBytes != b.Length {
			return exit.New(exit.Validation, "asset result differs from encoded file facts")
		}
	}
	return nil
}
func stringNumber(value any) string {
	if number, ok := value.(json.Number); ok {
		return string(number)
	}
	return ""
}
