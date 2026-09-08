package publication

import (
	"context"
	"github.com/cozy-creator/cozy/internal/canonical"
	"sort"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/transfer"
)

type UploadIntent struct {
	Request       UploadRequest `json:"request"`
	Objects       []hub.Object  `json:"objects"`
	ClosureDigest string        `json:"closure_digest"`
}
type ObjectUploader func(context.Context, hub.Grant, int64) *exit.Error

// UploadCheckpoint uses Tensorhub's existing publication/object journal. The
// retained private Runtime streams each granted object directly to its destination.
func UploadCheckpoint(ctx context.Context, client *hub.Client, operation string, intent UploadIntent, beforeWrite func() *exit.Error, upload ObjectUploader) (CheckpointRef, *exit.Error) {
	ref, problem := hub.ParseRef(intent.Request.Destination)
	if problem != nil {
		return CheckpointRef{}, problem
	}
	artifact := intent.Request.Artifact
	observed, lookup := client.CheckpointManifest(ctx, ref, artifact.Manifest.Digest)
	if lookup == nil {
		digest, _ := canonical.Spell(canonical.Digest(observed))
		if digest != artifact.Manifest.Digest || int64(len(observed)) != artifact.Manifest.Length {
			return CheckpointRef{}, exit.New(exit.Conflict, "checkpoint readback changed its exact manifest")
		}
		return CheckpointRef{Destination: ref.String(), Checkpoint: artifact.Manifest.Digest, Manifest: artifact.Manifest, Publication: operation, Observation: "observed_convergence"}, nil
	}
	if lookup.Code != exit.NotFound {
		return CheckpointRef{}, lookup
	}
	if beforeWrite == nil || upload == nil {
		return CheckpointRef{}, exit.Internalf("checkpoint publication has no owner or native mover")
	}
	if problem := beforeWrite(); problem != nil {
		return CheckpointRef{}, problem
	}
	opened, problem := client.OpenPublication(ctx, ref, operation, intent.Objects, "private script checkpoint upload")
	if problem != nil {
		return CheckpointRef{}, problem
	}
	totals, problem := transfer.ValidateOpenedPublication(opened, operation, intent.Objects)
	if problem != nil {
		return CheckpointRef{}, problem
	}
	lengths := map[string]int64{}
	for _, object := range intent.Objects {
		lengths[object.ID] = object.Length
	}
	pending := []string{}
	for _, row := range opened.Publication.Objects {
		if row.State != "accepted" && row.State != "verifying" {
			pending = append(pending, row.ObjectID)
		}
	}
	sort.Strings(pending)
	// Mint immediately before the object is spent. Earlier objects already
	// accepted by the Hub never enter this loop, including after lost replies.
	for _, id := range pending {
		if problem := beforeWrite(); problem != nil {
			return CheckpointRef{}, problem
		}
		granted, problem := client.GrantKnownTransfers(ctx, ref, operation, []string{id}, "private script checkpoint upload")
		if problem != nil {
			return CheckpointRef{}, problem
		}
		if len(granted.Held) == 1 {
			if granted.Held[0].Length != lengths[id] {
				return CheckpointRef{}, exit.New(exit.Conflict, "held object changed its exact length")
			}
			continue
		}
		if len(granted.Grants) != 1 || granted.Grants[0].Length != lengths[id] {
			return CheckpointRef{}, exit.New(exit.Conflict, "upload grant changed its exact inventory")
		}
		if problem := upload(ctx, granted.Grants[0], granted.ServerTimeUnix); problem != nil {
			return CheckpointRef{}, problem
		}
	}
	if problem := beforeWrite(); problem != nil {
		return CheckpointRef{}, problem
	}
	checkpoint, problem := client.FinalizePublication(ctx, ref, operation, hub.FinalizePublicationRequest{ManifestID: artifact.Manifest.Digest, ManifestLength: artifact.Manifest.Length}, "private script checkpoint upload")
	if problem != nil {
		return CheckpointRef{}, problem
	}
	if problem := transfer.ValidateFinalizedCheckpoint(checkpoint, operation, artifact.Manifest.Digest, artifact.Manifest.Length, totals); problem != nil {
		return CheckpointRef{}, problem
	}
	return CheckpointRef{Destination: ref.String(), Checkpoint: checkpoint.CheckpointID, Manifest: artifact.Manifest, Publication: checkpoint.PublishID, Observation: "acknowledged"}, nil
}
