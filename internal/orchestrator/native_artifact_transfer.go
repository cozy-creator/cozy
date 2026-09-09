package orchestrator

import (
	"bytes"
	"context"
	"fmt"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/publication"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
	"sort"
)

type pendingNativeArtifact struct {
	session *session
	request *pb.NativeArtifactTransfer
	result  chan *pb.NativeArtifactTransferStatus
}

func (c *Orchestrator) artifactRoundtrip(ctx context.Context, worker string, command *pb.NativeArtifactTransfer) (*pb.NativeArtifactTransferStatus, *exit.Error) {
	session, problem := c.workspaceControl(worker)
	if problem != nil {
		return nil, problem
	}
	if session == nil {
		return nil, exit.Unavailablef("native artifact transfer awaits its claimed workspace")
	}
	c.mu.Lock()
	currentWorker := c.workers[session.instanceID]
	capable := currentWorker != nil && currentWorker.wireMinor >= 42
	c.mu.Unlock()
	if !capable {
		return nil, exit.Named(exit.Unavailable, "publication.worker_upgrade_required", "native artifact publication requires worker protocol42")
	}
	command.RecordOwnerEpoch, command.ControlStreamEpoch, command.WorkerBootId = recordOwnerEpoch, session.epoch, session.bootID
	command.CommandId = c.artifactSequence.Add(1)
	key := fmt.Sprintf("%s/%d", command.EffectId, command.CommandId)
	pending := &pendingNativeArtifact{session: session, request: proto.Clone(command).(*pb.NativeArtifactTransfer), result: make(chan *pb.NativeArtifactTransferStatus, 1)}
	c.artifactPending.Store(key, pending)
	defer c.artifactPending.Delete(key)
	if !session.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_NativeArtifactTransfer{NativeArtifactTransfer: command}}) {
		return nil, exit.Unavailablef("native artifact transfer stream is unavailable")
	}
	select {
	case result := <-pending.result:
		if problem := publication.ArtifactTransferRefusal(result.SafeCode, result.SafeDetail); problem != nil {
			return nil, problem
		}
		return result, nil
	case <-ctx.Done():
		return nil, exit.New(exit.Canceled, "native artifact transfer caller stopped")
	case <-session.ctx.Done():
		return nil, exit.Unavailablef("native artifact transfer stream ended before acknowledgement")
	case <-c.done:
		return nil, exit.Unavailablef("native artifact transfer owner is stopping")
	}
}
func (c *Orchestrator) onNativeArtifactTransfer(s *session, result *pb.NativeArtifactTransferStatus) {
	if result == nil || c.fenced(s, result.RecordOwnerEpoch, result.ControlStreamEpoch, result.WorkerBootId) {
		return
	}
	found, ok := c.artifactPending.Load(fmt.Sprintf("%s/%d", result.EffectId, result.CommandId))
	if !ok {
		return
	}
	pending := found.(*pendingNativeArtifact)
	request := pending.request
	if pending.session != s || !proto.Equal(result.Source, request.Source) || !proto.Equal(result.Manifest, request.Manifest) || result.GrantRevision != request.GrantRevision || len(result.Objects) > 128 || proto.Size(result) > pb.MaxInlineControlBytes {
		return
	}
	if request.Grant != nil && result.ObjectId != request.Grant.ObjectId {
		return
	}
	select {
	case pending.result <- proto.Clone(result).(*pb.NativeArtifactTransferStatus):
	default:
	}
}
func (c *Orchestrator) nativeEffectSource(ctx context.Context, call records.NativeCall, artifact records.ModelArtifact) (records.NativeArtifactRetention, *exit.Error) {
	held, problem := c.opt.Store.ReserveNativeArtifact(call.ID, call.ParentRequestID, "effect", "source", artifact)
	if problem != nil {
		return held, problem
	}
	if held.State != "held" {
		if problem := c.changeNativeArtifactRetention(ctx, held, false); problem != nil {
			return held, problem
		}
	}
	return held, nil
}

func nativeTransferSource(h records.NativeArtifactRetention) *pb.DerivedRetentionRequest {
	digest, _ := canonical.Raw(h.ReceiptDigest)
	return &pb.DerivedRetentionRequest{WeightsTransactionId: h.TransactionID, TensorfsReceiptDigest: digest, RetentionId: h.RetentionID}
}
func nativeTransferManifest(h records.NativeArtifactRetention) *pb.Ref {
	digest, _ := canonical.Raw(h.ManifestID)
	return &pb.Ref{Digest: digest, Length: uint64(h.ManifestLength)}
}

func (c *Orchestrator) runNativeUpload(ctx context.Context, call records.NativeCall) ([]byte, *exit.Error) {
	var request publication.UploadRequest
	if problem := publication.DecodeEffect(call.Request, &request); problem != nil {
		return nil, problem
	}
	var h records.NativeArtifactRetention
	var problem *exit.Error
	if len(call.Frozen) > 0 {
		holds, problem := c.opt.Store.NativeArtifactRetentions(call.ID)
		if problem != nil {
			return nil, problem
		}
		for _, held := range holds {
			if held.Kind == "effect" && held.Slot == "source" && held.State == "held" {
				h = held
			}
		}
		if h.RetentionID == "" {
			return nil, exit.Named(exit.Conflict, "publication.source_hold_absent", "recorded effect has no retained native source")
		}
	} else {
		h, problem = c.nativeEffectSource(ctx, call, request.Artifact)
		if problem != nil {
			return nil, problem
		}
	}
	var intent publication.UploadIntent
	if len(call.Frozen) == 0 {
		intent.Request = request
		var closure []byte
		for offset := uint32(0); ; {
			page, problem := c.artifactRoundtrip(ctx, h.OwnerWorker, &pb.NativeArtifactTransfer{EffectId: call.ID, Source: nativeTransferSource(h), Manifest: nativeTransferManifest(h), Offset: offset, Limit: 128})
			if problem != nil {
				return nil, problem
			}
			if len(page.ClosureDigest) != 32 || (closure != nil && !bytes.Equal(closure, page.ClosureDigest)) {
				return nil, exit.New(exit.Conflict, "native artifact inventory changed between pages")
			}
			closure = append([]byte(nil), page.ClosureDigest...)
			for _, object := range page.Objects {
				if object.Length == 0 || object.Length > (1<<53)-1 {
					return nil, exit.New(exit.Validation, "native artifact object length is invalid")
				}
				if _, err := canonical.Raw(object.ObjectId); err != nil {
					return nil, exit.New(exit.Validation, "native artifact object digest is invalid")
				}
				if len(intent.Objects) > 0 && intent.Objects[len(intent.Objects)-1].ID >= object.ObjectId {
					return nil, exit.New(exit.Conflict, "native artifact inventory repeats or regresses")
				}
				intent.Objects = append(intent.Objects, hub.Object{ID: object.ObjectId, Length: int64(object.Length)})
			}
			if len(intent.Objects) > 100000 {
				return nil, exit.New(exit.Validation, "native artifact inventory exceeds its object bound")
			}
			if !page.HasMore {
				break
			}
			if page.NextOffset <= offset {
				return nil, exit.New(exit.Conflict, "native artifact page did not advance")
			}
			offset = page.NextOffset
		}
		rows := make([][]any, 0, len(intent.Objects))
		for _, object := range intent.Objects {
			rows = append(rows, []any{object.ID, object.Length})
		}
		raw, problem := publication.Canonical(rows)
		if problem != nil {
			return nil, problem
		}
		if !bytes.Equal(canonical.Digest(raw), closure) {
			return nil, exit.New(exit.Conflict, "native artifact inventory differs from its closure identity")
		}
		intent.ClosureDigest, _ = canonical.Spell(closure)
		frozen, problem := publication.Canonical(intent)
		if problem != nil {
			return nil, problem
		}
		if problem := c.opt.Store.FreezeNativeCall(call.ID, frozen); problem != nil {
			return nil, problem
		}
		call.Frozen = frozen
		call.State = "frozen"
	} else if problem := publication.DecodeEffect(call.Frozen, &intent); problem != nil {
		return nil, problem
	}
	owner, ok := c.opt.ModelTransfers.(interface {
		UploadPublicationEffect(context.Context, records.NativeCall, publication.UploadIntent, publication.ObjectUploader) ([]byte, *exit.Error)
	})
	if !ok {
		return nil, exit.Unavailablef("checkpoint effect has no publication owner")
	}
	return owner.UploadPublicationEffect(ctx, call, intent, func(ctx context.Context, grant hub.Grant, serverTime int64) *exit.Error {
		headers := make([]*pb.WeightsUploadHeader, 0, len(grant.Headers))
		for name, value := range grant.Headers {
			headers = append(headers, &pb.WeightsUploadHeader{Name: name, Value: value})
		}
		sort.Slice(headers, func(i, j int) bool { return headers[i].Name < headers[j].Name })
		uploaded, problem := c.artifactRoundtrip(ctx, h.OwnerWorker, &pb.NativeArtifactTransfer{EffectId: call.ID, Source: nativeTransferSource(h), Manifest: nativeTransferManifest(h), GrantRevision: 1, ServerTimeUnix: serverTime, Grant: &pb.WeightsUploadGrant{ObjectId: grant.ObjectID, Length: uint64(grant.Length), Url: grant.URL, ExpiresAtUnix: uint64(grant.ExpiresAtUnix), RequiredHeaders: headers}})
		if problem != nil {
			return problem
		}
		if uploaded.Outcome != pb.WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_UPLOADED && uploaded.Outcome != pb.WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT {
			return exit.Unavailablef("native artifact object upload did not finish")
		}
		if uploaded.Outcome == pb.WeightsUploadOutcome_WEIGHTS_UPLOAD_OUTCOME_UPLOADED && (uploaded.ChecksumSha256 != grant.ObjectID || uploaded.TransferredBytes != uint64(grant.Length)) {
			return exit.New(exit.Conflict, "native artifact upload changed exact bytes")
		}
		return nil
	})
}
