package orchestrator

import (
	"bytes"
	"context"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func (c *Orchestrator) changeNativeArtifactRetention(ctx context.Context, h records.NativeArtifactRetention, release bool) *exit.Error {
	if h.ArtifactKind == "tree" {
		return c.changeByteRetention(ctx, h, release)
	}
	owner, problem := c.opt.Store.RequestRow(h.OwnerRequestID)
	if problem != nil || owner == nil {
		return exit.Unavailablef("native artifact owner request is unavailable")
	}
	session, problem := c.workspaceControl(h.OwnerWorker)
	if problem != nil {
		return problem
	}
	if session == nil {
		return exit.Unavailablef("native artifact awaits its claimed workspace")
	}
	digest, err := canonical.Raw(h.ReceiptDigest)
	if err != nil {
		return exit.Internalf("native artifact receipt digest is malformed")
	}
	request := &pb.DerivedRetentionRequest{WeightsTransactionId: h.TransactionID, TensorfsReceiptDigest: digest, RetentionId: h.RetentionID}
	call := &pb.DerivedRetentionCall{Claim: session.claim, Request: request}
	var result *pb.DerivedRetentionResult
	if session.host != nil {
		if release {
			result, err = session.host.ReleaseDerivedRetention(ctx, call)
		} else {
			result, err = session.host.RetainDerivedResult(ctx, call)
		}
	} else if release {
		result, err = session.preparation.WorkspaceReleaseDerivedRetention(ctx, call)
	} else {
		result, err = session.preparation.WorkspaceRetainDerivedResult(ctx, call)
	}
	if err != nil {
		return exit.Unavailablef("native artifact custody request awaits its original workspace")
	}
	if result == nil || result.RetentionId != h.RetentionID || result.WeightsTransactionId != h.TransactionID || !bytes.Equal(result.TensorfsReceiptDigest, digest) || result.Released != release {
		return exit.Named(exit.Structural, "child.artifact_retention_changed", "native retention reply changed its exact identity")
	}
	if !release {
		if result.Manifest == nil {
			return exit.Named(exit.Structural, "child.artifact_manifest_missing", "native retention acknowledged no artifact")
		}
		manifest, _ := canonical.Spell(result.Manifest.Digest)
		if manifest != h.ManifestID || result.Manifest.Length != uint64(h.ManifestLength) {
			return exit.Named(exit.Structural, "child.artifact_manifest_changed", "native retention changed its manifest")
		}
	}
	if release {
		return c.opt.Store.CompleteNativeArtifactRelease(h.RetentionID)
	}
	if problem := c.opt.Store.ConfirmNativeArtifact(h.RetentionID, session.instanceID, session.bootID); problem != nil {
		// Cancellation may have committed before the acquire response. Release the
		// native hold against its durable tombstone instead of resurrecting it.
		if problem.ErrName() == "child.artifact_release_pending" {
			_ = c.changeNativeArtifactRetention(ctx, h, true)
		}
		return problem
	}
	return nil
}
