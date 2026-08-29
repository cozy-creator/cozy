package hub

// Typed endpoint release/profile routes. Declaration bytes are caller-canonical and
// replayed exactly; this client never re-marshals them into a second identity.

import (
	"context"
	"encoding/json"
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
	BaseWorkerImageDigest   string    `json:"base_worker_image_digest,omitempty"`
	BaseRealizationKind     string    `json:"base_realization_kind,omitempty"`
	BaseRealizationDigest   string    `json:"base_realization_digest,omitempty"`
	EndpointEnvironmentSpec ObjectRef `json:"endpoint_environment_spec,omitempty"`
	ResolvedWheelSet        ObjectRef `json:"resolved_wheel_set,omitempty"`
	ResolutionLock          ObjectRef `json:"resolution_lock,omitempty"`
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

type EndpointQualificationRequest struct {
	AcceleratorModel               string `json:"accelerator_model"`
	ProviderExposureLimitUSDMicros int64  `json:"provider_exposure_limit_usd_micros"`
	DurationCapSeconds             int64  `json:"duration_cap_s"`
}

type EndpointQualification struct {
	Profile                   string          `json:"profile"`
	State                     string          `json:"state"`
	AcceleratorModel          string          `json:"accelerator_model"`
	ProviderResourceID        string          `json:"provider_resource_id"`
	ProviderExposureUSDMicros int64           `json:"provider_exposure_usd_micros"`
	ObservedCostUSDMicros     int64           `json:"observed_cost_usd_micros"`
	Reclaimed                 bool            `json:"reclaimed"`
	ModelQualificationSpec    json.RawMessage `json:"model_qualification_spec"`
	ExecutionObservation      json.RawMessage `json:"execution_observation"`
	ModelAdmissionDecision    json.RawMessage `json:"model_admission_decision"`
	FailureCode               string          `json:"failure_code"`
	FailureDetail             string          `json:"failure_detail"`
}

func endpointProfilePath(ref Ref, release, profile string) string {
	return endpointReleasePath(ref, release) + "/profiles/" + url.PathEscape(profile)
}

func (c *Client) QualifyEndpointProfile(ctx context.Context, ref Ref, release, profile string,
	request EndpointQualificationRequest, reason string,
) (EndpointQualification, *exit.Error) {
	var out EndpointQualification
	e := c.do(ctx, call{method: http.MethodPost,
		path:  endpointProfilePath(ref, release, profile) + "/qualify",
		admin: true, reason: reason, body: request, byBytes: true, patient: true}, &out)
	return out, e
}

func (c *Client) EndpointProfileQualification(ctx context.Context, ref Ref, release,
	profile string,
) (EndpointQualification, *exit.Error) {
	var out EndpointQualification
	e := c.do(ctx, call{method: http.MethodGet,
		path:  endpointProfilePath(ref, release, profile) + "/qualification",
		admin: true, byBytes: true}, &out)
	return out, e
}

type ServingTarget struct {
	Major    string `json:"major"`
	Function string `json:"function"`
}

type ServingResult struct {
	EndpointRef              string   `json:"endpoint_ref"`
	EndpointExecutionDigests []string `json:"endpoint_execution_digests"`
}

type EndpointPromotion struct {
	Release string          `json:"release"`
	Serving []ServingResult `json:"serving"`
}

func (c *Client) PromoteEndpointRelease(ctx context.Context, ref Ref, release string,
	serving []ServingTarget, reason string,
) (EndpointPromotion, *exit.Error) {
	var out EndpointPromotion
	e := c.do(ctx, call{method: http.MethodPost,
		path: endpointReleasePath(ref, release) + "/promote", admin: true, reason: reason,
		body: map[string]any{"serving": serving}, byBytes: true}, &out)
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

type LocalExecutionGrant struct {
	CandidateID             string          `json:"candidate_id"`
	Profile                 string          `json:"profile"`
	BaseWorkerImageDigest   string          `json:"base_worker_image_digest"`
	BaseRealization         BaseRealization `json:"base_realization"`
	EndpointEnvironmentSpec ExactDocument   `json:"endpoint_environment_spec"`
	WheelhouseManifest      ExactDocument   `json:"wheelhouse_manifest"`
	ResolvedWheelSet        ExactDocument   `json:"resolved_wheel_set"`
	ResolutionLock          ExactDocument   `json:"resolution_lock"`
	Downloads               []DownloadGrant `json:"downloads"`
}

func (c *Client) EndpointLocalExecution(ctx context.Context, ref Ref, release, profile string,
	grantTTLSeconds int64, reason string,
) (LocalExecutionGrant, *exit.Error) {
	var out LocalExecutionGrant
	e := c.do(ctx, call{method: http.MethodPost,
		path:  endpointProfilePath(ref, release, profile) + "/local-execution",
		admin: true, reason: reason, byBytes: true, patient: true,
		body: map[string]int64{"grant_ttl_seconds": grantTTLSeconds}}, &out)
	return out, e
}
