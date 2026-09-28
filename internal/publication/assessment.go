package publication

import (
	"bytes"
	"context"

	"github.com/cozy-creator/cozy/internal/canonical"
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

func ValidateAssessmentRequest(request AssessmentRequest) *exit.Error {
	checkpoint := request.Checkpoint
	if checkpoint.Checkpoint != checkpoint.Manifest.Digest || checkpoint.Manifest.Length <= 0 || checkpoint.Publication == "" {
		return exit.New(exit.Validation, "assessment checkpoint is not an exact publication receipt")
	}
	for _, digest := range []string{checkpoint.Checkpoint, request.Report, request.Workloads} {
		if _, err := canonical.Raw(digest); err != nil {
			return exit.New(exit.Validation, "assessment inputs require immutable content identities")
		}
	}
	ref, problem := hub.ParseRef(checkpoint.Destination)
	if problem != nil {
		return problem
	}
	if ref.Org == "local" {
		return exit.New(exit.Validation, "assessment destination must be a public model repository")
	}
	return nil
}

type Assessments interface {
	AssessmentReport(context.Context, hub.Ref, string, string) ([]byte, *exit.Error)
	AttachAssessment(context.Context, hub.Ref, string, string, string, []byte) *exit.Error
}

// ApplyAssessment first reconciles the immutable destination, including after a
// lost write response. A nil report performs readback only. BeforeWrite remains the
// existing durable cancellation/authority fence and is called only for a new PUT.
func ApplyAssessment(ctx context.Context, client Assessments, intent AssessmentIntent, account string, report []byte, beforeWrite func() *exit.Error) (AssessmentRef, *exit.Error) {
	if problem := ValidateAssessmentRequest(intent.Request); problem != nil {
		return AssessmentRef{}, problem
	}
	if intent.Account == "" || account != intent.Account || intent.Report.Digest != intent.Request.Report || intent.Workloads.Digest != intent.Request.Workloads || intent.Report.Length <= 0 || intent.Report.Length > 8<<20 || intent.Workloads.Length <= 0 || intent.Workloads.Length > 8<<20 || intent.RenderOwner == "" {
		return AssessmentRef{}, exit.Named(exit.Conflict, "assessment.intent_changed", "assessment identity or account changed after verification")
	}
	switch intent.Verdict {
	case "pass", "fail", "indeterminate":
	default:
		return AssessmentRef{}, exit.New(exit.Validation, "assessment verdict is unsupported")
	}
	ref, problem := hub.ParseRef(intent.Request.Checkpoint.Destination)
	if problem != nil {
		return AssessmentRef{}, problem
	}
	answer := AssessmentRef{Destination: ref.String(), Checkpoint: intent.Request.Checkpoint.Checkpoint, Report: intent.Report, Verdict: intent.Verdict}
	existing, problem := client.AssessmentReport(ctx, ref, answer.Checkpoint, intent.Report.Digest)
	if problem == nil {
		digest, _ := canonical.Spell(canonical.Digest(existing))
		if digest != intent.Report.Digest || int64(len(existing)) != intent.Report.Length {
			return answer, exit.Named(exit.Conflict, "assessment.readback_changed", "assessment readback differs from the frozen report")
		}
		answer.Observation = "observed_convergence"
		return answer, nil
	}
	if problem.Code != exit.NotFound || report == nil {
		return answer, problem
	}
	digest, _ := canonical.Spell(canonical.Digest(report))
	if digest != intent.Report.Digest || int64(len(report)) != intent.Report.Length {
		return answer, exit.New(exit.Conflict, "assessment bytes differ from verified artifact")
	}
	if beforeWrite == nil {
		return answer, exit.Internalf("assessment has no durable pre-write owner")
	}
	if problem := beforeWrite(); problem != nil {
		return answer, problem
	}
	if problem := client.AttachAssessment(ctx, ref, answer.Checkpoint, intent.Report.Digest, intent.Verdict, report); problem != nil {
		return answer, problem
	}
	// The client has compared both its typed acknowledgement and exact GET bytes.
	answer.Observation = "acknowledged"
	return answer, nil
}

func SameAssessmentRequest(left, right AssessmentRequest) bool {
	a, _ := Canonical(left)
	b, _ := Canonical(right)
	return bytes.Equal(a, b)
}
