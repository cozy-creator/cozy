package publication

import (
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
