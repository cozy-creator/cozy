package orchestrator

import (
	"bytes"
	"context"
	"sort"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/publication"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// ArtifactRoundtrip sends one NativeArtifactTransfer to the machine holding the artifact and
// answers its status, a refusal already typed.
type ArtifactRoundtrip func(context.Context, *pb.NativeArtifactTransfer) (*pb.NativeArtifactTransferStatus, *exit.Error)

// HeldClosure pages the exact object closure of a held artifact, as its machine's native
// store names it, and returns it sorted with its closure digest. operation names the upload
// the pages are read for.
func HeldClosure(ctx context.Context, roundtrip ArtifactRoundtrip, operation string, source *pb.DerivedRetentionRequest,
	manifest *pb.Ref) ([]hub.Object, string, *exit.Error) {
	var objects []hub.Object
	var closure []byte
	for offset := uint32(0); ; {
		page, problem := roundtrip(ctx, &pb.NativeArtifactTransfer{EffectId: operation, Source: source, Manifest: manifest, Offset: offset, Limit: 128})
		if problem != nil {
			return nil, "", problem
		}
		if len(page.ClosureDigest) != 32 || (closure != nil && !bytes.Equal(closure, page.ClosureDigest)) {
			return nil, "", exit.New(exit.Conflict, "native artifact inventory changed between pages")
		}
		closure = append([]byte(nil), page.ClosureDigest...)
		for _, object := range page.Objects {
			if object.Length == 0 || object.Length > (1<<53)-1 {
				return nil, "", exit.New(exit.Validation, "native artifact object length is invalid")
			}
			if _, err := canonical.Raw(object.ObjectId); err != nil {
				return nil, "", exit.New(exit.Validation, "native artifact object digest is invalid")
			}
			if len(objects) > 0 && objects[len(objects)-1].ID >= object.ObjectId {
				return nil, "", exit.New(exit.Conflict, "native artifact inventory repeats or regresses")
			}
			objects = append(objects, hub.Object{ID: object.ObjectId, Length: int64(object.Length)})
		}
		if len(objects) > 100000 {
			return nil, "", exit.New(exit.Validation, "native artifact inventory exceeds its object bound")
		}
		if !page.HasMore {
			break
		}
		if page.NextOffset <= offset {
			return nil, "", exit.New(exit.Conflict, "native artifact page did not advance")
		}
		offset = page.NextOffset
	}
	rows := make([][]any, 0, len(objects))
	for _, object := range objects {
		rows = append(rows, []any{object.ID, object.Length})
	}
	raw, problem := publication.Canonical(rows)
	if problem != nil {
		return nil, "", problem
	}
	if !bytes.Equal(canonical.Digest(raw), closure) {
		return nil, "", exit.New(exit.Conflict, "native artifact inventory differs from its closure identity")
	}
	digest, _ := canonical.Spell(closure)
	return objects, digest, nil
}

// PushHeld sends one granted closure object of a held artifact from the machine holding it
// straight to the Hub.
func PushHeld(ctx context.Context, roundtrip ArtifactRoundtrip, operation string, source *pb.DerivedRetentionRequest,
	manifest *pb.Ref, grant hub.Grant, serverTime int64) *exit.Error {
	headers := make([]*pb.WeightsUploadHeader, 0, len(grant.Headers))
	for name, value := range grant.Headers {
		headers = append(headers, &pb.WeightsUploadHeader{Name: name, Value: value})
	}
	sort.Slice(headers, func(i, j int) bool { return headers[i].Name < headers[j].Name })
	uploaded, problem := roundtrip(ctx, &pb.NativeArtifactTransfer{EffectId: operation, Source: source, Manifest: manifest,
		GrantRevision: 1, ServerTimeUnix: serverTime, Grant: &pb.WeightsUploadGrant{ObjectId: grant.ObjectID, Length: uint64(grant.Length),
			Url: grant.URL, ExpiresAtUnix: uint64(grant.ExpiresAtUnix), RequiredHeaders: headers}})
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
}
