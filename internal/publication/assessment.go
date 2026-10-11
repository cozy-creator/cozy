package publication

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

type AssessmentRequest struct {
	Checkpoint CheckpointRef `json:"checkpoint"`
	Report     string        `json:"report"`
	Workloads  string        `json:"workloads"`
}
type AssessmentIntent struct {
	Request     AssessmentRequest                 `json:"request"`
	Account     string                            `json:"account"`
	Report      records.ArtifactObjectRef         `json:"report"`
	Workloads   records.ArtifactObjectRef         `json:"workloads"`
	Verdict     string                            `json:"verdict"`
	RenderOwner string                            `json:"render_owner"`
	RenderHolds []records.NativeArtifactRetention `json:"render_holds"`
}
type AssessmentRef struct {
	Destination string                    `json:"destination"`
	Checkpoint  string                    `json:"checkpoint"`
	Report      records.ArtifactObjectRef `json:"report"`
	Verdict     string                    `json:"verdict"`
	Observation string                    `json:"observation"`
}

type Assessments interface {
	AssessmentReport(context.Context, hub.Ref, string, string) ([]byte, *exit.Error)
	AttachAssessment(context.Context, hub.Ref, string, string, string, []byte) *exit.Error
}
