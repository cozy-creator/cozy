package hub

// Typed package publication and download routes.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"
	"unicode"

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
// which Tensorhub derives from the committed wheel environment.
func (d PackageReleaseDetail) Constraints() ([]string, string, *exit.Error) {
	if d.ExecutionRequirements == nil {
		return nil, "", exit.Named(exit.Structural, "hub.package_release_invalid",
			"Tensorhub returned no package requirements")
	}
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
	// patient: the grant answers one store HEAD per declared subject — work
	// bounded by the declaration itself, not by a wall clock. A 174-file
	// declaration against remote object custody exceeded the 10s header clock.
	e := c.do(ctx, call{method: http.MethodPost,
		path: packagePublishPath(ref, release), auth: true, reason: reason,
		body: map[string]any{"files": files}, patient: true,
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
		body:    body,
		patient: true}, &out)
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
// typed failure is visible. Each status call has a short header/body deadline;
// the caller context, rather than a fixed wall clock, controls the whole wait.
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
		if state.State != "queued" && state.State != "verifying" && state.State != "retrying" {
			return state, exit.Named(exit.Structural, "hub.package_release_invalid",
				"Tensorhub returned unknown package finalization state %q", state.State)
		}
		timer := time.NewTimer(packageFinalizePollInterval)
		select {
		case <-ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return state, exit.New(exit.Canceled, "package finalization was canceled")
		case <-timer.C:
		}
		callCtx, cancel := context.WithTimeout(ctx, Timeout)
		next, problem := c.PackageReleaseStatus(callCtx, ref, release)
		cancel()
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

// ---------------------------------------------------------------- bindings (th-116)

// PackageBindingRow is one mutable owner override: a model release and its GPU-to-lane
// fit map. Explicit run selection takes precedence; absent rows use immutable defaults
// from the selected package interface. Only owner writes change these rows.
type PackageBindingRow struct {
	Slot      string        `json:"slot"`
	Model     string        `json:"model"`
	Release   string        `json:"release"`
	Ladder    []BindingRung `json:"ladder"`
	Revision  int64         `json:"revision"`
	UpdatedAt string        `json:"updated_at"`
}

// BindingRung pairs an accelerator pattern with the lane that fits it. GPU is matched as
// a case-insensitive in-order token subsequence of a machine's accelerator_model ("H100"
// fits "NVIDIA H100 80GB HBM3" and "NVIDIA H100 NVL"); the literal "*" is the catch-all
// and is allowed only as the last rung.
type BindingRung struct {
	GPU  string `json:"gpu"`
	GPUs int    `json:"gpus,omitempty"`
	Lane string `json:"lane"`
}

func (r BindingRung) String() string {
	if r.GPUs > 0 {
		return fmt.Sprintf("%dx%s=%s", r.GPUs, r.GPU, r.Lane)
	}
	return r.GPU + "=" + r.Lane
}

// Ref renders the row as its org/model@release spelling.
func (b PackageBindingRow) Ref() string { return b.Model + "@" + b.Release }

// LadderText is the ladder as one readable line: `H100=fp8, *=bf16`.
func LadderText(ladder []BindingRung) string {
	parts := make([]string, 0, len(ladder))
	for _, rung := range ladder {
		parts = append(parts, rung.String())
	}
	return strings.Join(parts, ", ")
}

// ValidateLadder is the shape every ladder must have before it is written or trusted:
// at least one rung, every rung a gpu pattern and a lane, "*" nowhere but last.
func ValidateLadder(ladder []BindingRung) *exit.Error {
	if len(ladder) == 0 {
		return exit.Usagef("a binding needs at least one --gpu <GPU>=<lane> rung")
	}
	for i, rung := range ladder {
		if rung.GPUs < 0 {
			return exit.Usagef("rung %d GPU count must be positive when supplied", i+1)
		}
		if len(rung.GPU) == 0 || len(rung.GPU) > 64 || strings.TrimSpace(rung.GPU) != rung.GPU ||
			(rung.GPU != "*" && strings.IndexFunc(rung.GPU, func(r rune) bool { return unicode.IsLetter(r) || unicode.IsNumber(r) }) < 0) ||
			strings.TrimSpace(rung.Lane) == "" || strings.ContainsAny(rung.Lane, " \t") {
			return exit.Usagef("rung %d is not <GPU>=<lane>: %q", i+1, rung.String())
		}
		if rung.GPU == "*" && i != len(ladder)-1 {
			return exit.Usagef("the catch-all rung '*' must be the last rung, not rung %d of %d", i+1, len(ladder))
		}
	}
	return nil
}

// PackageBindingRows is the anonymous read of a package's owner overrides, as written.
func (c *Client) PackageBindingRows(ctx context.Context, ref Ref) ([]PackageBindingRow, *exit.Error) {
	var out struct {
		Bindings []PackageBindingRow `json:"bindings"`
	}
	if e := c.do(ctx, call{method: http.MethodGet, path: resourcePath("packages", ref) + "/bindings"}, &out); e != nil {
		return nil, e
	}
	return out.Bindings, nil
}

// PackageBindings is the owner overrides for `slots` (every slot when none is named), each
// usable. An unusable row for one of them is refused, not skipped: skipping it would silently
// drop the owner's choice. The refusal names the row and how to repair it. Rows for other
// slots are not read.
func (c *Client) PackageBindings(ctx context.Context, ref Ref, slots ...string) ([]PackageBindingRow, *exit.Error) {
	rows, e := c.PackageBindingRows(ctx, ref)
	if e != nil {
		return nil, e
	}
	seen := make(map[string]bool, len(rows))
	var used []PackageBindingRow
	for _, row := range rows {
		if len(slots) > 0 && !slices.Contains(slots, row.Slot) {
			continue
		}
		reason := ""
		switch {
		case row.Slot == "":
			return nil, exit.Named(exit.Structural, "hub.package_bindings_invalid",
				"an owner binding of %s names no slot", ref.String()).
				WithRemedy("only Tensorhub can repair a slotless row; meanwhile choose the model per run: model.<param>=org/model@release")
		case seen[row.Slot]:
			reason = "is duplicated"
		case row.Model == "" || row.Release == "":
			reason = "names no model release"
		case row.Revision < 1:
			reason = "has no revision"
		default:
			if problem := ValidateLadder(row.Ladder); problem != nil {
				reason = "has an invalid ladder: " + problem.Message
			}
		}
		if reason != "" {
			return nil, exit.Named(exit.Structural, "hub.package_bindings_invalid",
				"the owner binding of %s for slot %s %s", ref.String(), row.Slot, reason).
				WithRemedy("rebind it: cozy package bind %s %s org/model@release --gpu <GPU>=<lane>; or remove it: cozy package unbind %s %s",
					ref.String(), row.Slot, ref.String(), row.Slot)
		}
		seen[row.Slot] = true
		used = append(used, row)
	}
	return used, nil
}

type PackageBindingWrite struct {
	Binding PackageBindingRow `json:"binding"`
	Changed bool              `json:"changed"`
}

// BindPackageSlot sets one package slot's owner override to a model release and ladder
// under CAS on expectedRevision (0 creates an unbound row).
func (c *Client) BindPackageSlot(ctx context.Context, ref Ref, slot, model, release string,
	ladder []BindingRung, expectedRevision int64, reason string,
) (PackageBindingWrite, *exit.Error) {
	var out PackageBindingWrite
	e := c.do(ctx, call{method: http.MethodPut,
		path: resourcePath("packages", ref) + "/bindings/" + url.PathEscape(slot),
		auth: true, reason: reason,
		body: map[string]any{"model": model, "release": release, "ladder": ladder,
			"expected_revision": expectedRevision}}, &out)
	return out, e
}

// PackageBindingReset removes an owner override so authored defaults can apply.
type PackageBindingReset struct {
	Slot    string `json:"slot"`
	Changed bool   `json:"changed"`
}

func (c *Client) UnbindPackageSlot(ctx context.Context, ref Ref, slot string, expectedRevision int64, reason string) (PackageBindingReset, *exit.Error) {
	var out PackageBindingReset
	problem := c.do(ctx, call{method: http.MethodDelete,
		path: resourcePath("packages", ref) + "/bindings/" + url.PathEscape(slot),
		auth: true, reason: reason,
		body: map[string]any{"expected_revision": expectedRevision}}, &out)
	return out, problem
}
