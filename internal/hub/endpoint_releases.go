package hub

// Typed endpoint release/profile routes. Declaration bytes are caller-canonical and
// replayed exactly; this client never re-marshals them into a second identity.

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

type EndpointUpload struct {
	Role            string            `json:"role"`
	Ref             ObjectRef         `json:"ref"`
	URL             string            `json:"url"`
	RequiredHeaders map[string]string `json:"required_headers"`
	ExpiresAt       string            `json:"expires_at"`
	AlreadyHeld     bool              `json:"already_held"`
}

type EndpointProfileState struct {
	Profile                 string    `json:"profile"`
	State                   string    `json:"state"`
	CandidateID             string    `json:"candidate_id,omitempty"`
	BaseRealizationKind     string    `json:"base_realization_kind,omitempty"`
	BaseRealizationDigest   string    `json:"base_realization_digest,omitempty"`
	EndpointEnvironmentSpec ObjectRef `json:"endpoint_environment_spec,omitempty"`
	ResolvedWheelSet        ObjectRef `json:"resolved_wheel_set,omitempty"`
	ResolutionLock          ObjectRef `json:"resolution_lock,omitempty"`
	RefusalCode             string    `json:"refusal_code,omitempty"`
	RefusalDetail           string    `json:"refusal_detail,omitempty"`
}

type EndpointReleaseBegin struct {
	DeclarationDigest string                 `json:"declaration_digest"`
	State             string                 `json:"state"`
	Uploads           []EndpointUpload       `json:"uploads"`
	Profiles          []EndpointProfileState `json:"profiles"`
}

type EndpointExecution struct {
	Profile  string `json:"profile"`
	Function string `json:"function"`
	Digest   string `json:"digest"`
	State    string `json:"state"`
}

type EndpointReleaseFinalize struct {
	Created            bool                   `json:"created"`
	Release            string                 `json:"release"`
	DeclarationDigest  string                 `json:"declaration_digest"`
	Profiles           []EndpointProfileState `json:"profiles"`
	EndpointExecutions []EndpointExecution    `json:"endpoint_executions"`
}

func endpointReleasePath(ref Ref, release string) string {
	return resourcePath("endpoints", ref) + "/releases/" + url.PathEscape(release)
}

func endpointProfilePath(ref Ref, release, profile string) string {
	return endpointReleasePath(ref, release) + "/profiles/" + url.PathEscape(profile)
}

func (c *Client) BeginEndpointRelease(ctx context.Context, ref Ref, release string,
	declaration []byte, reason string,
) (EndpointReleaseBegin, *exit.Error) {
	var out EndpointReleaseBegin
	e := c.do(ctx, call{method: http.MethodPost,
		path: endpointReleasePath(ref, release) + "/begin", admin: true, reason: reason,
		bodyBytes: declaration, byBytes: true, patient: true}, &out)
	return out, e
}

func (c *Client) FinalizeEndpointRelease(ctx context.Context, ref Ref, release string,
	declaration []byte, reason string,
) (EndpointReleaseFinalize, *exit.Error) {
	var out EndpointReleaseFinalize
	e := c.do(ctx, call{method: http.MethodPost,
		path: endpointReleasePath(ref, release) + "/finalize", admin: true, reason: reason,
		bodyBytes: declaration, byBytes: true, patient: true}, &out)
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
	CandidateID             string            `json:"candidate_id"`
	Profile                 string            `json:"profile"`
	LeaseID                 string            `json:"lease_id"`
	LeaseExpiresAt          string            `json:"lease_expires_at"`
	BaseRealization         BaseRealization   `json:"base_realization"`
	EndpointEnvironmentSpec ExactDocument     `json:"endpoint_environment_spec"`
	EndpointBundle          ExactDocument     `json:"endpoint_bundle"`
	Descriptor              ExactDocument     `json:"descriptor"`
	EvaluatedConfig         ExactDocument     `json:"evaluated_config"`
	WheelhouseManifest      ExactDocument     `json:"wheelhouse_manifest"`
	ResolvedWheelSet        ExactDocument     `json:"resolved_wheel_set"`
	ResolutionLock          ExactDocument     `json:"resolution_lock"`
	Downloads               []DownloadGrant   `json:"downloads"`
	NativeWheelProof        *NativeWheelProof `json:"native_wheel_proof,omitempty"`
}

func (c *Client) EndpointLocalQualificationMaterials(ctx context.Context, ref Ref, release, profile string,
	grantTTLSeconds int64, reason string,
) (LocalQualificationMaterials, *exit.Error) {
	var out LocalQualificationMaterials
	e := c.do(ctx, call{method: http.MethodPost,
		path:  endpointProfilePath(ref, release, profile) + "/local-qualification-materials",
		admin: true, reason: reason, byBytes: true, patient: true,
		body: map[string]int64{"grant_ttl_seconds": grantTTLSeconds}}, &out)
	return out, e
}
