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

type PackageReleaseDraft struct {
	State   string          `json:"state"`
	Uploads []PackageUpload `json:"uploads"`
}

type PackageReleaseCommit struct {
	State          string        `json:"state"`
	PackageRelease ExactDocument `json:"package_release"`
}

type PackageInstallDownload struct {
	Digest string `json:"digest"`
	Kind   string `json:"kind"`
	Length int64  `json:"length"`
	Path   string `json:"path"`
	URL    string `json:"url"`
}

type PackageInstallPlan struct {
	Profile           string                   `json:"profile"`
	PlacementSet      ExactDocument            `json:"placement_set"`
	PackageRelease    ExactDocument            `json:"package_release"`
	PackageDescriptor ExactDocument            `json:"package_descriptor"`
	Qualification     ExactDocument            `json:"qualification"`
	Downloads         []PackageInstallDownload `json:"downloads"`
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

func (c *Client) DeclarePackageRelease(ctx context.Context, ref Ref, release string,
	paths, dependencyWheels []string, reason string) (PackageReleaseDraft, *exit.Error) {
	var out PackageReleaseDraft
	e := c.do(ctx, call{method: http.MethodPost,
		path: packageReleasePath(ref, release), auth: true, reason: reason,
		body: map[string]any{"paths": paths, "dependency_wheels": dependencyWheels}, strict: true}, &out)
	return out, e
}

func (c *Client) CommitPackageRelease(ctx context.Context, ref Ref, release, reason string) (PackageReleaseCommit, *exit.Error) {
	var out PackageReleaseCommit
	e := c.do(ctx, call{method: http.MethodPut,
		path: packageReleasePath(ref, release), auth: true, reason: reason,
		body: map[string]any{}, patient: true, strict: true}, &out)
	return out, e
}

func (c *Client) PackageInstallPlan(ctx context.Context, ref Ref, release string,
	target PackageInstallTarget,
) (PackageInstallPlan, *exit.Error) {
	var out PackageInstallPlan
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
	e := c.do(ctx, call{method: http.MethodPost,
		path: packageReleasePath(ref, release) + "/selection", body: body, strict: true,
		responseBytes: 16 << 20}, &out)
	return out, e
}
