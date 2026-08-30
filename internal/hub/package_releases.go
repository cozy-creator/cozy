package hub

// Typed package release/profile routes.

import (
	"context"
	"net/http"
	"net/url"

	"github.com/cozy-creator/cozy/internal/exit"
)

type PackageUpload struct {
	Kind            string            `json:"kind,omitempty"`
	Path            string            `json:"path,omitempty"`
	URL             string            `json:"url"`
	RequiredHeaders map[string]string `json:"required_headers"`
	AlreadyUploaded bool              `json:"already_uploaded"`
}

type PackageProfileState struct {
	Profile             string `json:"profile"`
	State               string `json:"state"`
	CandidateID         string `json:"candidate_id,omitempty"`
	BaseRealizationKind string `json:"base_realization_kind,omitempty"`
	RefusalCode         string `json:"refusal_code,omitempty"`
	RefusalDetail       string `json:"refusal_detail,omitempty"`
}

type PackageReleaseBegin struct {
	State              string        `json:"state"`
	ProjectWheelUpload PackageUpload `json:"project_wheel_upload"`
}

type PackageUploads struct {
	Uploads []PackageUpload `json:"uploads"`
}

type PackageReleaseFinalize struct {
	CompatibleProfiles []string              `json:"compatible_profiles"`
	PackageExecutions  []PackageExecution    `json:"package_executions"`
	Profiles           []PackageProfileState `json:"profiles"`
	QualificationError string                `json:"qualification_error,omitempty"`
	QualificationState string                `json:"qualification_state"`
	Requirements       []string              `json:"requirements"`
	RequiresPython     string                `json:"requires_python"`
}

type PackageExecution struct {
	Digest   string `json:"digest"`
	Function string `json:"function"`
	Profile  string `json:"profile"`
	State    string `json:"state"`
}

func packageReleasePath(ref Ref, release string) string {
	return resourcePath("packages", ref) + "/releases/" + url.PathEscape(release)
}

func (c *Client) BeginPackageRelease(ctx context.Context, ref Ref, release, reason string) (PackageReleaseBegin, *exit.Error) {
	var out PackageReleaseBegin
	e := c.do(ctx, call{method: http.MethodPost,
		path: packageReleasePath(ref, release), auth: true, reason: reason,
		body: map[string]any{}, strict: true}, &out)
	return out, e
}

func (c *Client) PackageReleaseUploads(ctx context.Context, ref Ref, release string,
	paths, dependencyWheels []string, reason string,
) (PackageUploads, *exit.Error) {
	var out PackageUploads
	e := c.do(ctx, call{method: http.MethodPost,
		path: packageReleasePath(ref, release) + "/uploads", auth: true, reason: reason,
		body: map[string]any{"paths": paths, "dependency_wheels": dependencyWheels}, strict: true}, &out)
	return out, e
}

func (c *Client) FinalizePackageRelease(ctx context.Context, ref Ref, release, reason string) (PackageReleaseFinalize, *exit.Error) {
	var out PackageReleaseFinalize
	e := c.do(ctx, call{method: http.MethodPut,
		path: packageReleasePath(ref, release), auth: true, reason: reason,
		body: map[string]any{}, patient: true, strict: true}, &out)
	return out, e
}
