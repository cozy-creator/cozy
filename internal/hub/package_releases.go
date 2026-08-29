package hub

// Typed package release/profile routes.

import (
	"context"
	"net/http"
	"net/url"

	"github.com/cozy-creator/cozy-creator/internal/exit"
)

type ObjectRef struct {
	Digest string `json:"digest"`
	Length int64  `json:"length"`
}

type PackageUpload struct {
	Path            string            `json:"path,omitempty"`
	URL             string            `json:"url"`
	RequiredHeaders map[string]string `json:"required_headers"`
	ExpiresAt       string            `json:"expires_at"`
	AlreadyUploaded bool              `json:"already_uploaded"`
}

type PackageProfileState struct {
	Profile                string    `json:"profile"`
	State                  string    `json:"state"`
	CandidateID            string    `json:"candidate_id,omitempty"`
	BaseRealizationKind    string    `json:"base_realization_kind,omitempty"`
	BaseRealizationDigest  string    `json:"base_realization_digest,omitempty"`
	PackageEnvironmentSpec ObjectRef `json:"package_environment_spec,omitempty"`
	ResolvedWheelSet       ObjectRef `json:"resolved_wheel_set,omitempty"`
	ResolutionLock         ObjectRef `json:"resolution_lock,omitempty"`
	RefusalCode            string    `json:"refusal_code,omitempty"`
	RefusalDetail          string    `json:"refusal_detail,omitempty"`
}

type PackageReleaseBegin struct {
	Release            string        `json:"release"`
	State              string        `json:"state"`
	Created            bool          `json:"created"`
	ProjectWheelUpload PackageUpload `json:"project_wheel_upload"`
}

type PackageSourceUploads struct {
	Uploads []PackageUpload `json:"uploads"`
}

type PackageExecution struct {
	Profile  string `json:"profile"`
	Function string `json:"function"`
	Digest   string `json:"digest"`
	State    string `json:"state"`
}

type PackageReleaseFinalize struct {
	Created           bool                  `json:"created"`
	Release           string                `json:"release"`
	Profiles          []PackageProfileState `json:"profiles"`
	PackageExecutions []PackageExecution    `json:"package_executions"`
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
		path: packageReleasePath(ref, release) + "/begin", admin: true, reason: reason,
		body: map[string]any{}}, &out)
	return out, e
}

func (c *Client) PackageReleaseSourceUploads(ctx context.Context, ref Ref, release string,
	paths []string, reason string,
) (PackageSourceUploads, *exit.Error) {
	var out PackageSourceUploads
	e := c.do(ctx, call{method: http.MethodPost,
		path: packageReleasePath(ref, release) + "/uploads", admin: true, reason: reason,
		body: map[string]any{"paths": paths}}, &out)
	return out, e
}

func (c *Client) FinalizePackageRelease(ctx context.Context, ref Ref, release, reason string) (PackageReleaseFinalize, *exit.Error) {
	var out PackageReleaseFinalize
	e := c.do(ctx, call{method: http.MethodPost,
		path: packageReleasePath(ref, release) + "/finalize", admin: true, reason: reason,
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

type NativeWheelProof struct {
	ExpectedResultDigest string `json:"expected_result_digest"`
	Fixture              string `json:"fixture"`
}

type LocalQualificationMaterials struct {
	CandidateID            string            `json:"candidate_id"`
	Profile                string            `json:"profile"`
	LeaseID                string            `json:"lease_id"`
	LeaseExpiresAt         string            `json:"lease_expires_at"`
	BaseRealization        BaseRealization   `json:"base_realization"`
	PackageEnvironmentSpec ExactDocument     `json:"package_environment_spec"`
	PackageBundle          ExactDocument     `json:"package_bundle"`
	PackageDescriptor      ExactDocument     `json:"package_descriptor"`
	WheelhouseManifest     ExactDocument     `json:"wheelhouse_manifest"`
	ResolvedWheelSet       ExactDocument     `json:"resolved_wheel_set"`
	ResolutionLock         ExactDocument     `json:"resolution_lock"`
	Downloads              []DownloadGrant   `json:"downloads"`
	NativeWheelProof       *NativeWheelProof `json:"native_wheel_proof,omitempty"`
}

func (c *Client) PackageLocalQualificationMaterials(ctx context.Context, ref Ref, release, profile string,
	grantTTLSeconds int64, reason string,
) (LocalQualificationMaterials, *exit.Error) {
	var out LocalQualificationMaterials
	e := c.do(ctx, call{method: http.MethodPost,
		path:  packageProfilePath(ref, release, profile) + "/local-qualification-materials",
		admin: true, reason: reason, byBytes: true, patient: true,
		body: map[string]int64{"grant_ttl_seconds": grantTTLSeconds}}, &out)
	return out, e
}
