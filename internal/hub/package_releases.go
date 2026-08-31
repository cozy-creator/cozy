package hub

// Typed package release/profile routes.

import (
	"context"
	"encoding/json"
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

type PackageReleaseDraft struct {
	State   string          `json:"state"`
	Uploads []PackageUpload `json:"uploads"`
}

type PackageReleaseCommit struct {
	State              string `json:"state"`
	ReleaseDigest      string `json:"release_digest"`
	QualificationState string `json:"qualification_state"`
	QualificationError string `json:"qualification_error,omitempty"`
}

type PackageReleaseYank struct {
	Changed  bool   `json:"changed"`
	Release  string `json:"release"`
	State    string `json:"state"`
	YankedAt string `json:"yanked_at"`
}

type PackageReleaseDetail struct {
	Release struct {
		Release                 string `json:"release"`
		ReleaseDigest           string `json:"release_digest"`
		PackageDescriptorDigest string `json:"package_descriptor_digest"`
		PackageDescriptorLength int64  `json:"package_descriptor_length"`
		CreatedAt               string `json:"created_at"`
		CommittedAt             string `json:"committed_at,omitempty"`
		Yanked                  bool   `json:"yanked,omitempty"`
		YankedAt                string `json:"yanked_at,omitempty"`
	} `json:"release"`
	Document          json.RawMessage `json:"document"`
	PackageDescriptor json.RawMessage `json:"package_descriptor"`
}

type PackageInstallDownload struct {
	Digest string `json:"digest"`
	Kind   string `json:"kind"`
	Length int64  `json:"length"`
	Path   string `json:"path"`
	URL    string `json:"url"`
}

type PackageDownloadPlan struct {
	Release            string                   `json:"release"`
	Profile            string                   `json:"profile"`
	PlacementSet       ExactDocument            `json:"placement_set"`
	PackageDescriptor  ExactDocument            `json:"package_descriptor"`
	Qualification      ExactDocument            `json:"qualification"`
	EnvironmentReceipt ExactDocument            `json:"environment_receipt"`
	WheelhouseManifest ExactDocument            `json:"wheelhouse_manifest"`
	Downloads          []PackageInstallDownload `json:"downloads"`
}

// PackageInstallTarget carries only measured local compatibility facts. Tensorhub
// remains the authority that ranks qualified profiles and returns one exact selection.
type PackageInstallTarget struct {
	Accelerator       string `json:"accelerator"`
	OS                string `json:"os"`
	Arch              string `json:"arch"`
	DriverCUDA        string `json:"driver_cuda"`
	ComputeCapability string `json:"compute_capability"`
}

func packageReleasePath(ref Ref, release string) string {
	return resourcePath("packages", ref) + "/releases/" + url.PathEscape(release)
}

func packagePublishPath(ref Ref, release string) string {
	return resourcePath("packages", ref) + "/publish/" + url.PathEscape(release)
}

func (c *Client) DeclarePackageRelease(ctx context.Context, ref Ref, release string,
	paths, dependencyWheels []string, reason string) (PackageReleaseDraft, *exit.Error) {
	var out PackageReleaseDraft
	e := c.do(ctx, call{method: http.MethodPost,
		path: packagePublishPath(ref, release), auth: true, reason: reason,
		body: map[string]any{"paths": paths, "dependency_wheels": dependencyWheels}, strict: true}, &out)
	return out, e
}

func (c *Client) CommitPackageRelease(ctx context.Context, ref Ref, release, reason string) (PackageReleaseCommit, *exit.Error) {
	var out PackageReleaseCommit
	e := c.do(ctx, call{method: http.MethodPost,
		path: packagePublishPath(ref, release) + "/finalize", auth: true, reason: reason,
		body: map[string]any{}, patient: true, strict: true}, &out)
	return out, e
}

func (c *Client) YankPackageRelease(ctx context.Context, ref Ref, release, reason string) (PackageReleaseYank, *exit.Error) {
	var out PackageReleaseYank
	e := c.do(ctx, call{method: http.MethodDelete,
		path: packageReleasePath(ref, release), auth: true, reason: reason,
		body: map[string]any{}, strict: true}, &out)
	return out, e
}

func (c *Client) PackageRelease(ctx context.Context, ref Ref,
	release string) (PackageReleaseDetail, *exit.Error) {
	var out PackageReleaseDetail
	e := c.do(ctx, call{method: http.MethodGet, path: packageReleasePath(ref, release),
		strict: true, responseBytes: 16 << 20}, &out)
	return out, e
}

func (c *Client) PackageDownloads(ctx context.Context, ref Ref, release string,
	target PackageInstallTarget,
) (PackageDownloadPlan, *exit.Error) {
	var out PackageDownloadPlan
	if target.OS == "" || target.Arch == "" ||
		(target.Accelerator != "cpu" && target.Accelerator != "nvidia") ||
		target.Accelerator == "cpu" && (target.DriverCUDA != "" || target.ComputeCapability != "") ||
		target.Accelerator == "nvidia" && (target.DriverCUDA == "" || target.ComputeCapability == "") {
		return out, exit.Internalf("package install target is incomplete or contradictory")
	}
	body := struct {
		Capability      PackageInstallTarget `json:"capability"`
		ModelSelections []ModelSelection     `json:"model_selections"`
	}{Capability: target, ModelSelections: []ModelSelection{}}
	query := url.Values{}
	if release != "" {
		query.Set("release", release)
	}
	path := resourcePath("packages", ref) + "/download"
	if len(query) != 0 {
		path += "?" + query.Encode()
	}
	e := c.do(ctx, call{method: http.MethodPost,
		path: path, body: body, strict: true,
		responseBytes: 16 << 20}, &out)
	return out, e
}
