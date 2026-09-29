package publication

import (
	"bytes"
	"context"
	"encoding/json"
	"io"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

type CheckpointRef struct {
	Destination string                    `json:"destination"`
	Checkpoint  string                    `json:"checkpoint"`
	Manifest    records.ArtifactObjectRef `json:"manifest"`
	Publication string                    `json:"publication"`
	Observation string                    `json:"observation"`
}

type PublishRequest struct {
	Destination      string                   `json:"destination"`
	Release          string                   `json:"release"`
	Lanes            map[string]CheckpointRef `json:"lanes"`
	ExpectedRevision *int64                   `json:"expected_revision"`
}

type UploadRequest struct {
	Artifact    records.ModelArtifact `json:"artifact"`
	Destination string                `json:"destination"`
}

// DecodeEffect reads one effect request by the members this cozy knows; a newer Runtime's
// others are ignored.
func DecodeEffect(raw []byte, value any) *exit.Error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	if decoder.Decode(value) != nil || decoder.Decode(&struct{}{}) != io.EOF {
		return exit.New(exit.Validation, "publication effect request is not one JSON object of its interface")
	}
	return nil
}
func Canonical(value any) ([]byte, *exit.Error) {
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, exit.Internalf("cannot encode publication effect: %s", err)
	}
	raw, err = canonical.NormalizeJCS(raw)
	if err != nil {
		return nil, exit.Internalf("cannot normalize publication effect: %s", err)
	}
	return raw, nil
}

func PreparePublish(ctx context.Context, client Releases, raw []byte) (ReleaseIntent, *exit.Error) {
	var request PublishRequest
	if problem := DecodeEffect(raw, &request); problem != nil {
		return ReleaseIntent{}, problem
	}
	lanes := make(map[string]string, len(request.Lanes))
	for name, checkpoint := range request.Lanes {
		if checkpoint.Destination != request.Destination || checkpoint.Checkpoint != checkpoint.Manifest.Digest || checkpoint.Manifest.Length <= 0 || checkpoint.Publication == "" {
			return ReleaseIntent{}, exit.New(exit.Validation, "release checkpoint reference differs from its destination or exact manifest")
		}
		lanes[name] = checkpoint.Checkpoint
	}
	return PrepareRelease(ctx, client, ReleaseRequest{Destination: request.Destination, Release: request.Release, Lanes: lanes, ExpectedRevision: request.ExpectedRevision})
}

func EffectDestination(operation string, raw []byte) (hub.Ref, *exit.Error) {
	var destination string
	switch operation {
	case "publish_release":
		var request PublishRequest
		if problem := DecodeEffect(raw, &request); problem != nil {
			return hub.Ref{}, problem
		}
		destination = request.Destination

	case "attach_assessment":
		var request AssessmentRequest
		if problem := DecodeEffect(raw, &request); problem != nil {
			return hub.Ref{}, problem
		}
		if problem := ValidateAssessmentRequest(request); problem != nil {
			return hub.Ref{}, problem
		}
		destination = request.Checkpoint.Destination
	case "upload_checkpoint":
		var request UploadRequest
		if problem := DecodeEffect(raw, &request); problem != nil {
			return hub.Ref{}, problem
		}
		destination = request.Destination
	default:
		return hub.Ref{}, exit.New(exit.Validation, "unknown fixed publication effect")
	}
	ref, problem := hub.ParseRef(destination)
	if problem != nil {
		return ref, problem
	}
	if ref.Org == "local" {
		return ref, exit.New(exit.Validation, "local aliases cannot be published")
	}
	return ref, nil
}
