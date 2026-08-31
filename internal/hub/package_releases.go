package hub

// Typed package publication and download routes.

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
	PackageDescriptor ExactDocument            `json:"package_descriptor"`
	Release           string                   `json:"release"`
	ReleaseDigest     string                   `json:"release_digest"`
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
	return out, e
}
