package hub

// Typed package publication and download routes.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/url"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

// MaxPackageInstallDownloads is one project wheel, the complete bounded rental
// dependency-wheel closure, and the source-carried Runtime/TensorFS wheels used
// only to materialize Creator's independent local environment.
const MaxPackageInstallDownloads = 131

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
	State         string `json:"state"`
	ReleaseDigest string `json:"release_digest"`
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

// Requirements returns the immutable execution dependencies from the exact
// PackageRelease/1 bytes. Creator needs only this one release fact to choose a
// CPU or accelerator product; it does not reinterpret the package descriptor's
// model inputs as hardware requirements.
func (d PackageReleaseDetail) Requirements() ([]string, *exit.Error) {
	requirements, _, problem := d.Constraints()
	return requirements, problem
}

// Constraints returns the release's execution dependencies AND its interpreter floor from
// the same exact bytes. RequiresPython is the second half of what the SKU's base profile
// can be checked against before renting; both are Tensorhub's own derivation from the
// installed project wheel, so neither is re-derived here.
func (d PackageReleaseDetail) Constraints() ([]string, string, *exit.Error) {
	// Document is embedded inside another JSON response. The outer encoder may spell
	// `<`, `>` and `&` as Unicode escapes, so RawMessage preserves transport tokens,
	// not necessarily the stored PackageRelease bytes. Normalize the parsed content
	// before checking its stored-byte identity; a semantic change still moves the hash.
	canonicalDocument, err := canonical.NormalizeJCS(d.Document)
	if err != nil {
		return nil, "", exit.Named(exit.Structural, "hub.package_release_invalid",
			"Tensorhub returned an invalid PackageRelease/1 document")
	}
	sum := sha256.Sum256(canonicalDocument)
	want := "sha256:" + hex.EncodeToString(sum[:])
	if d.Release.ReleaseDigest != want {
		return nil, "", exit.Named(exit.Conflict, "hub.package_release_digest_mismatch",
			"Tensorhub package release bytes do not match release digest %s", d.Release.ReleaseDigest)
	}
	var document struct {
		Format         string   `json:"format"`
		Requirements   []string `json:"requirements"`
		RequiresPython string   `json:"requires_python"`
	}
	if err := json.Unmarshal(canonicalDocument, &document); err != nil ||
		document.Format != "cozy.package.release/1" || document.Requirements == nil {
		return nil, "", exit.Named(exit.Structural, "hub.package_release_invalid",
			"Tensorhub returned an invalid PackageRelease/1 document")
	}
	for i, requirement := range document.Requirements {
		if requirement == "" || i > 0 && requirement <= document.Requirements[i-1] {
			return nil, "", exit.Named(exit.Structural, "hub.package_release_invalid",
				"Tensorhub returned unsorted or empty package requirements")
		}
	}
	return append([]string(nil), document.Requirements...), document.RequiresPython, nil
}

type PackageInstallDownload struct {
	Digest       string   `json:"digest"`
	Distribution string   `json:"distribution"`
	ImportRoots  []string `json:"import_roots"`
	Kind         string   `json:"kind"`
	Length       int64    `json:"length"`
	Path         string   `json:"path"`
	Tags         []string `json:"tags"`
	URL          string   `json:"url"`
	Version      string   `json:"version"`
}

type PackageDownloadPlan struct {
	Downloads         []PackageInstallDownload `json:"downloads"`
	PackageConfig     ExactDocument            `json:"package_config"`
	PackageDescriptor ExactDocument            `json:"package_descriptor"`
	Pyproject         ExactDocument            `json:"pyproject"`
	Release           string                   `json:"release"`
	ReleaseDigest     string                   `json:"release_digest"`
	UVLock            ExactDocument            `json:"uv_lock"`
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

func (c *Client) PackageDownloads(ctx context.Context, ref Ref, release string) (PackageDownloadPlan, *exit.Error) {
	var out PackageDownloadPlan
	query := url.Values{}
	if release != "" {
		query.Set("release", release)
	}
	path := resourcePath("packages", ref) + "/download"
	if len(query) != 0 {
		path += "?" + query.Encode()
	}
	e := c.do(ctx, call{method: http.MethodPost,
		path: path, body: struct{}{}, strict: true,
		responseBytes: 16 << 20}, &out)
	if e == nil && (len(out.Downloads) == 0 || len(out.Downloads) > MaxPackageInstallDownloads) {
		e = exit.Named(exit.Structural, "hub.package_download_plan_invalid",
			"Tensorhub returned %d package wheel downloads; expected 1 through %d",
			len(out.Downloads), MaxPackageInstallDownloads)
	}
	return out, e
}
