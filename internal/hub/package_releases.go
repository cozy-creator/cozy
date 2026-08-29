package hub

// Typed package release/profile routes.

import (
	"context"
	"net/http"
	"net/url"

	"github.com/cozy-creator/cozy/internal/exit"
)

type ObjectRef struct {
	Digest string `json:"digest"`
	Length int64  `json:"length"`
}

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
	Created            bool                  `json:"created"`
	ExecutionCount     int                   `json:"execution_count"`
	Profiles           []PackageProfileState `json:"profiles"`
	QualificationError string                `json:"qualification_error,omitempty"`
	QualificationState string                `json:"qualification_state"`
	Requirements       []string              `json:"requirements"`
	RequiresPython     string                `json:"requires_python"`
}

func packageReleasePath(ref Ref, release string) string {
	return resourcePath("packages", ref) + "/releases/" + url.PathEscape(release)
}

func packageProfilePath(ref Ref, release, profile string) string {
	return packageReleasePath(ref, release) + "/profiles/" + url.PathEscape(profile)
}

func (c *Client) BeginPackageRelease(ctx context.Context, ref Ref, release, reason string) (PackageReleaseBegin, *exit.Error) {
	var out PackageReleaseBegin
	e := c.do(ctx, call{method: http.MethodPost,
		path: packageReleasePath(ref, release), auth: true, reason: reason,
		body: map[string]any{}}, &out)
	return out, e
}

func (c *Client) PackageReleaseUploads(ctx context.Context, ref Ref, release string,
	paths, dependencyWheels []string, reason string,
) (PackageUploads, *exit.Error) {
	var out PackageUploads
	e := c.do(ctx, call{method: http.MethodPost,
		path: packageReleasePath(ref, release) + "/uploads", auth: true, reason: reason,
		body: map[string]any{"paths": paths, "dependency_wheels": dependencyWheels}}, &out)
	return out, e
}

func (c *Client) FinalizePackageRelease(ctx context.Context, ref Ref, release, reason string) (PackageReleaseFinalize, *exit.Error) {
	var out PackageReleaseFinalize
	e := c.do(ctx, call{method: http.MethodPut,
		path: packageReleasePath(ref, release), auth: true, reason: reason,
		body: map[string]any{}, patient: true}, &out)
	return out, e
}

type ExactDocument struct {
	Digest         string `json:"digest"`
	Length         int64  `json:"length"`
	CanonicalBytes []byte `json:"canonical_bytes_base64"`
}

type DownloadGrant struct {
	Role      string    `json:"role"`
	Ref       ObjectRef `json:"ref"`
	URL       string    `json:"url"`
	ExpiresAt string    `json:"expires_at"`
}

type BaseRealization struct {
	Kind   string `json:"kind"`
	Digest string `json:"digest"`
}

type LocalQualificationMaterials struct {
	CandidateID            string          `json:"candidate_id"`
	Profile                string          `json:"profile"`
	LeaseID                string          `json:"lease_id"`
	LeaseExpiresAt         string          `json:"lease_expires_at"`
	BaseRealization        BaseRealization `json:"base_realization"`
	PackageEnvironmentSpec ExactDocument   `json:"package_environment_spec"`
	PackageBundle          ExactDocument   `json:"package_bundle"`
	PackageDescriptor      ExactDocument   `json:"package_descriptor"`
	WheelhouseManifest     ExactDocument   `json:"wheelhouse_manifest"`
	ResolvedWheelSet       ExactDocument   `json:"resolved_wheel_set"`
	ResolutionLock         ExactDocument   `json:"resolution_lock"`
	Downloads              []DownloadGrant `json:"downloads"`
}

func (c *Client) PackageLocalQualificationMaterials(ctx context.Context, ref Ref, release, profile string,
	reason string,
) (LocalQualificationMaterials, *exit.Error) {
	var out LocalQualificationMaterials
	e := c.do(ctx, call{method: http.MethodPost,
		path:  packageProfilePath(ref, release, profile) + "/install",
		auth: true, reason: reason, byBytes: true, patient: true,
		body: map[string]any{}}, &out)
	return out, e
}
