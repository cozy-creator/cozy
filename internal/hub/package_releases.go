package hub

// Typed package publication and download routes.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// PackageDeclaredFile is one ordinary file in a publication session. Tensorhub journals
// the declaration and echoes these refs only to authorize missing uploads.
type PackageDeclaredFile struct {
	Digest string `json:"digest"`
	Length int64  `json:"length"`
	Path   string `json:"path"`
}

// PackageRegistryRow names one locked registry dependency for the hub to fetch
// itself; the client never proxies registry bytes (cl-078).
type PackageRegistryRow struct {
	Name    string `json:"name"`
	SHA256  string `json:"sha256"`
	Size    int64  `json:"size"`
	URL     string `json:"url"`
	Version string `json:"version"`
}

type PackagePresignedUpload struct {
	URL             string            `json:"url"`
	RequiredHeaders map[string]string `json:"required_headers"`
}

type PackageFileGrant struct {
	PackageDeclaredFile
	Present bool                    `json:"present"`
	Upload  *PackagePresignedUpload `json:"upload,omitempty"`
}

type PackageReleaseDraft struct {
	PublicationID string             `json:"publication_id"`
	Files         []PackageFileGrant `json:"files"`
}

type PackageReleaseCommit struct {
	PublicationID string                 `json:"publication_id"`
	State         string                 `json:"state"`
	StatusURL     string                 `json:"status_url,omitempty"`
	Error         *PackageReleaseFailure `json:"error,omitempty"`
	// Warnings is Tensorhub's advice about a committed release; it never blocks one.
	Warnings []string `json:"warnings,omitempty"`
}

// PackageReleaseFailure is a durable terminal refusal reported by an
// asynchronous finalizer. The status route deliberately answers HTTP 200 so a
// caller can resume after losing the original finalize response and still see
// the Hub's typed refusal.
type PackageReleaseFailure struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Remedy  string `json:"remedy,omitempty"`
}

func (c PackageReleaseCommit) failure() *exit.Error {
	if c.Error == nil || c.Error.Code == "" {
		return exit.Named(exit.Failed, "package_release.failed",
			"Tensorhub failed package finalization")
	}
	e := exit.Named(exit.Failed, c.Error.Code, "%s", c.Error.Message)
	if c.Error.Remedy != "" {
		e.WithRemedy("%s", c.Error.Remedy)
	}
	return e
}

type PackageReleaseYank struct {
	Changed  bool   `json:"changed"`
	Release  string `json:"release"`
	State    string `json:"state"`
	YankedAt string `json:"yanked_at"`
}

type PackageReleaseDetail struct {
	Release struct {
		Release                string `json:"release"`
		PackageInterfaceDigest string `json:"package_interface_digest"`
		PackageInterfaceLength int64  `json:"package_interface_length"`
		CreatedAt              string `json:"created_at"`
		CommittedAt            string `json:"committed_at,omitempty"`
		Yanked                 bool   `json:"yanked,omitempty"`
		YankedAt               string `json:"yanked_at,omitempty"`
	} `json:"release"`
	PackageInterface      json.RawMessage `json:"package_interface"`
	ExecutionRequirements []string        `json:"requirements"`
	RequiresPython        string          `json:"requires_python"`
	PythonVersion         string          `json:"python_version"`
}

// PackageSourceArchive is the immutable source distribution retained for
// public inspection. The URL is a short-lived object capability.
type PackageSourceArchive struct {
	Digest string `json:"digest"`
	Length int64  `json:"length"`
	URL    string `json:"url"`
}

// Requirements returns the immutable execution dependencies from the exact
// package environment facts. Creator needs only this one release fact to choose
// a CPU or accelerator product; it does not reinterpret model inputs as hardware requirements.
func (d PackageReleaseDetail) Requirements() ([]string, *exit.Error) {
	requirements, _, problem := d.Constraints()
	return requirements, problem
}

// Constraints returns the release's execution dependencies and interpreter floor,
// which Tensorhub derives from the committed wheel environment; absent, it has none.
func (d PackageReleaseDetail) Constraints() ([]string, string, *exit.Error) {
	requirements := slices.DeleteFunc(slices.Clone(d.ExecutionRequirements), func(r string) bool { return r == "" })
	slices.Sort(requirements)
	return slices.Compact(requirements), d.RequiresPython, nil
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
	PythonVersion    string                   `json:"python_version"`
	Downloads        []PackageInstallDownload `json:"downloads"`
	PackageConfig    ExactDocument            `json:"package_config"`
	PackageInterface ExactDocument            `json:"package_interface"`
	Pyproject        ExactDocument            `json:"pyproject"`
	Release          string                   `json:"release"`
	UVLock           ExactDocument            `json:"uv_lock"`
}

func packageReleasePath(ref Ref, release string) string {
	return resourcePath("packages", ref) + "/releases/" + url.PathEscape(release)
}

func packagePublishPath(ref Ref, release string) string {
	return resourcePath("packages", ref) + "/publish/" + url.PathEscape(release)
}

func (c *Client) DeclarePackageRelease(ctx context.Context, ref Ref, release string,
	files []PackageDeclaredFile, reason string) (PackageReleaseDraft, *exit.Error) {
	var out PackageReleaseDraft
	e := c.do(ctx, call{method: http.MethodPost,
		path: packagePublishPath(ref, release), auth: true, reason: reason,
		body:          map[string]any{"files": files},
		responseBytes: 16 << 20}, &out)
	return out, e
}

func (c *Client) CommitPackageRelease(ctx context.Context, ref Ref, release string,
	publicationID string, registry []PackageRegistryRow, reason string, pythonVersion ...string,
) (PackageReleaseCommit, *exit.Error) {
	if registry == nil {
		registry = []PackageRegistryRow{}
	}
	body := map[string]any{"publication_id": publicationID, "registry": registry}
	if len(pythonVersion) > 0 && pythonVersion[0] != "" {
		body["python_version"] = pythonVersion[0]
	}
	var out PackageReleaseCommit
	e := c.do(ctx, call{method: http.MethodPost,
		path: packagePublishPath(ref, release) + "/finalize", auth: true, reason: reason,
		body: body}, &out)
	return out, e
}

// PackageReleaseStatus reads the short, authenticated status projection for a
// queued finalization. It intentionally uses the canonical route instead of
// trusting a server-provided absolute URL as a new origin.
func (c *Client) PackageReleaseStatus(ctx context.Context, ref Ref, release string) (PackageReleaseCommit, *exit.Error) {
	var out PackageReleaseCommit
	e := c.do(ctx, call{method: http.MethodGet,
		path: packagePublishPath(ref, release) + "/status", auth: true, responseBytes: 1 << 20}, &out)
	return out, e
}

const packageFinalizePollInterval = 2 * time.Second

// WaitPackageRelease follows a 202 finalization until the durable commit or
// typed failure is visible. The caller context controls the whole wait.
func (c *Client) WaitPackageRelease(ctx context.Context, ref Ref, release string,
	initial PackageReleaseCommit, progress func(PackageReleaseCommit),
) (PackageReleaseCommit, *exit.Error) {
	state := initial
	for {
		if progress != nil {
			progress(state)
		}
		if state.State == "committed" {
			return state, nil
		}
		if state.State == "failed" {
			return state, state.failure()
		}
		// Queued, verifying, retrying, or a state a newer Hub added: still pending.
		timer := time.NewTimer(packageFinalizePollInterval)
		select {
		case <-ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return state, exit.New(exit.Canceled, "package finalization was canceled")
		case <-timer.C:
		}
		next, problem := c.PackageReleaseStatus(ctx, ref, release)
		if transientPoll(ctx, problem) {
			continue
		}
		if problem != nil {
			return state, problem
		}
		state = next
	}
}

func (c *Client) YankPackageRelease(ctx context.Context, ref Ref, release, reason string) (PackageReleaseYank, *exit.Error) {
	var out PackageReleaseYank
	e := c.do(ctx, call{method: http.MethodDelete,
		path: packageReleasePath(ref, release), auth: true, reason: reason,
		body: map[string]any{}}, &out)
	return out, e
}

func (c *Client) PackageRelease(ctx context.Context, ref Ref,
	release string) (PackageReleaseDetail, *exit.Error) {
	var out PackageReleaseDetail
	kept := c.releaseCachePath("release", ref, release, "")
	if loadRelease(kept, &out) && out.Release.Release == release {
		return out, nil
	}
	e := c.do(ctx, call{method: http.MethodGet, path: packageReleasePath(ref, release), responseBytes: 16 << 20}, &out)
	if e == nil && out.Release.Release == release {
		storeRelease(kept, out)
	}
	return out, e
}

func (c *Client) PackageSourceArchive(ctx context.Context, ref Ref, release string) (PackageSourceArchive, *exit.Error) {
	var out PackageSourceArchive
	e := c.do(ctx, call{method: http.MethodGet,
		path:          packageReleasePath(ref, release) + "/source",
		responseBytes: 1 << 20}, &out)
	return out, e
}

func (c *Client) PackageDownloads(ctx context.Context, ref Ref, release string, pythonVersion ...string) (PackageDownloadPlan, *exit.Error) {
	var out PackageDownloadPlan
	kept := c.releaseCachePath("downloads", ref, release, strings.Join(pythonVersion, ","))
	if loadRelease(kept, &out) && out.Release == release && len(out.Downloads) > 0 {
		return out, nil
	}
	query := url.Values{}
	if release != "" {
		query.Set("release", release)
	}
	if len(pythonVersion) > 0 && pythonVersion[0] != "" {
		query.Set("python_version", pythonVersion[0])
	}
	path := resourcePath("packages", ref) + "/download"
	if len(query) != 0 {
		path += "?" + query.Encode()
	}
	e := c.do(ctx, call{method: http.MethodPost,
		path: path, body: struct{}{},
		responseBytes: 16 << 20}, &out)
	if e == nil && len(out.Downloads) == 0 {
		e = exit.Named(exit.Structural, "hub.package_download_plan_invalid",
			"Tensorhub returned no package wheel downloads")
	}
	if e == nil && out.Release == release {
		// The wheel URLs are short-lived capabilities no caller reads; the facts are kept.
		for i := range out.Downloads {
			out.Downloads[i].URL = ""
		}
		storeRelease(kept, out)
	}
	return out, e
}
