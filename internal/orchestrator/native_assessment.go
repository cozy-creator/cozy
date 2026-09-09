package orchestrator

import (
	"bytes"
	"context"
	"crypto/sha256"
	"io"

	"github.com/cozy-creator/cozy/internal/assessment"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/publication"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type assessmentEffectOwner interface {
	PrepareAssessmentPublication(context.Context, records.NativeCall, string, []byte, []byte) (publication.AssessmentIntent, *exit.Error)
	ApplyAssessmentPublication(context.Context, records.NativeCall, publication.AssessmentIntent, []byte) ([]byte, *exit.Error)
}

func (c *Orchestrator) runNativeAssessment(ctx context.Context, call records.NativeCall) ([]byte, *exit.Error) {
	owner, ok := c.opt.ModelTransfers.(assessmentEffectOwner)
	if !ok {
		return nil, exit.Unavailablef("assessment publication owner is unavailable")
	}
	var request publication.AssessmentRequest
	if problem := publication.DecodeEffect(call.Request, &request); problem != nil {
		return nil, problem
	}
	var intent publication.AssessmentIntent
	if len(call.Frozen) > 0 {
		if problem := publication.DecodeEffect(call.Frozen, &intent); problem != nil {
			return nil, problem
		}
		// A committed public association can be reconciled even if the original pod
		// has disappeared. A read-only miss never authorizes another write by itself.
		result, problem := owner.ApplyAssessmentPublication(ctx, call, intent, nil)
		if problem == nil || problem.Code != exit.NotFound {
			return result, problem
		}
	}
	var holds []records.NativeArtifactRetention
	producer := ""
	var problem *exit.Error
	if len(call.Frozen) == 0 {
		holds, producer, problem = c.opt.Store.ReserveAssessmentBytes(call.ID, call.ParentRequestID, request.Report, request.Workloads)
	} else {
		holds, problem = c.opt.Store.NativeArtifactRetentions(call.ID)
		producer = intent.RenderOwner
	}
	if problem != nil {
		return nil, problem
	}
	files := map[string][]byte{}
	for _, hold := range holds {
		if hold.Kind != "effect" || hold.ArtifactKind != "tree" || (hold.Slot != "report" && hold.Slot != "workloads") || (hold.State != "held" && hold.State != "pending") || hold.ProducerID != producer {
			return nil, exit.Named(exit.Conflict, "assessment.custody_changed", "assessment lacks its exact retained input files")
		}
		if problem := c.changeByteRetention(ctx, hold, false); problem != nil {
			return nil, problem
		}
		bytes, problem := c.readByteArtifact(ctx, hold, assessment.MaxBytes)
		if problem != nil {
			return nil, problem
		}
		files[hold.Slot] = bytes
	}
	if len(files) != 2 {
		return nil, exit.New(exit.Conflict, "assessment input custody is incomplete")
	}
	if len(call.Frozen) == 0 {
		intent, problem = owner.PrepareAssessmentPublication(ctx, call, producer, files["report"], files["workloads"])
		if problem != nil {
			return nil, problem
		}
		for _, hold := range intent.RenderHolds {
			if problem := c.changeByteRetention(ctx, hold, false); problem != nil {
				return nil, problem
			}
		}
		frozen, problem := publication.Canonical(intent)
		if problem != nil {
			return nil, problem
		}
		if problem := c.opt.Store.FreezeNativeCall(call.ID, frozen); problem != nil {
			return nil, problem
		}
		call.Frozen = frozen
		call.State = "frozen"
	}
	// A new PUT still requires live original observation custody. Executing-phase
	// retry uses its already verified frozen intent and current write authorization.
	if call.State != "executing" {
		for _, hold := range intent.RenderHolds {
			if problem := c.changeByteRetention(ctx, hold, false); problem != nil {
				return nil, problem
			}
		}
	}
	return owner.ApplyAssessmentPublication(ctx, call, intent, files["report"])
}
func (c *Orchestrator) readByteArtifact(ctx context.Context, hold records.NativeArtifactRetention, limit int) ([]byte, *exit.Error) {
	output, problem := c.opt.Store.ByteOutputForRetention(hold)
	if problem != nil {
		return nil, problem
	}
	if output.Length <= 0 || output.Length > int64(limit) || output.ContentBytes != output.Length || output.MimeType == "application/vnd.cozy.tree-manifest" {
		return nil, exit.New(exit.Validation, "assessment requires a bounded native file payload")
	}
	session, problem := c.workspaceControl(hold.OwnerWorker)
	if problem != nil {
		return nil, problem
	}
	if session == nil {
		return nil, exit.Unavailablef("assessment artifact awaits its private workspace")
	}
	digest, err := canonical.Raw(output.Digest)
	if err != nil {
		return nil, exit.New(exit.Validation, "assessment artifact digest is invalid")
	}
	call := &pb.NativeByteReadCall{Claim: session.claim, Source: byteRetentionRequest(hold), Object: &pb.Ref{Digest: digest, Length: uint64(output.Length)}}
	var stream interface {
		Recv() (*pb.NativeByteReadChunk, error)
	}
	if session.host != nil {
		stream, err = session.host.ReadByteTreeObject(ctx, call)
	} else {
		stream, err = session.preparation.WorkspaceReadByteTreeObject(ctx, call)
	}
	if err != nil {
		return nil, assessmentReadError(err)
	}
	body := make([]byte, 0, int(output.Length))
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, assessmentReadError(err)
		}
		if chunk == nil || chunk.Offset != uint64(len(body)) || len(chunk.Data) == 0 || len(chunk.Data) > pb.MaxNativeByteReadChunkBytes || len(chunk.Data) > int(output.Length)-len(body) {
			return nil, exit.New(exit.Conflict, "assessment byte stream changed its object bounds")
		}
		body = append(body, chunk.Data...)
	}
	sum := sha256.Sum256(body)
	if len(body) != int(output.Length) || !bytes.Equal(sum[:], digest) {
		return nil, exit.New(exit.Conflict, "assessment byte stream differs from its immutable file identity")
	}
	return body, nil
}

func assessmentReadError(err error) *exit.Error {
	switch status.Code(err) {
	case codes.NotFound, codes.PermissionDenied, codes.InvalidArgument, codes.FailedPrecondition, codes.DataLoss:
		return exit.Named(exit.Conflict, "assessment.artifact_unavailable", "the native store refused the exact assessment file custody")
	}
	return exit.Unavailablef("assessment artifact read was interrupted")
}
