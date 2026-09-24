package hub

// Typed package publication and download routes.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

// MaxPackageInstallDownloads is one project wheel, the complete bounded rental
// dependency-wheel closure, and the source-carried Runtime/TensorFS wheels used
// only to materialize Creator's independent local environment.
const MaxPackageInstallDownloads = 131

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
		Defective              bool   `json:"defective,omitempty"`
		DefectiveAt            string `json:"defective_at,omitempty"`
		DefectiveCode          string `json:"defective_code,omitempty"`
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

// Constraints returns the release's execution dependencies and interpreter floor.
// Tensorhub derives both from the exact committed wheel environment; Creator verifies
// the carried PackageInterface ref but does not derive execution requirements from it.
func (d PackageReleaseDetail) Constraints() ([]string, string, *exit.Error) {
	// PackageInterface is embedded inside another JSON response. The outer encoder may spell
	// `<`, `>` and `&` as Unicode escapes, so RawMessage preserves transport tokens,
	// not necessarily the stored PackageRelease bytes. Normalize the parsed content
	// before checking its stored-byte identity; a semantic change still moves the hash.
	canonicalInterface, err := canonical.NormalizeJCS(d.PackageInterface)
	if err != nil {
		return nil, "", exit.Named(exit.Structural, "hub.package_release_invalid",
			"Tensorhub returned an invalid package interface")
	}
	sum := sha256.Sum256(canonicalInterface)
	want := "sha256:" + hex.EncodeToString(sum[:])
	if d.Release.PackageInterfaceDigest != want ||
		d.Release.PackageInterfaceLength != int64(len(canonicalInterface)) {
		return nil, "", exit.Named(exit.Conflict, "hub.package_interface_identity_mismatch",
			"Tensorhub package interface bytes do not match their exact ref")
	}
	if d.ExecutionRequirements == nil {
		return nil, "", exit.Named(exit.Structural, "hub.package_release_invalid",
			"Tensorhub returned no package requirements")
	}
	for i, requirement := range d.ExecutionRequirements {
		if requirement == "" || i > 0 && requirement <= d.ExecutionRequirements[i-1] {
			return nil, "", exit.Named(exit.Structural, "hub.package_release_invalid",
				"Tensorhub returned unsorted or empty package requirements")
		}
	}
	return append([]string(nil), d.ExecutionRequirements...), d.RequiresPython, nil
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
		body: map[string]any{"files": files}, patient: true, strict: true,
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
		patient: true, strict: true}, &out)
	return out, e
}

// PackageReleaseStatus reads the short, authenticated status projection for a
// queued finalization. It intentionally uses the canonical route instead of
// trusting a server-provided absolute URL as a new origin.
func (c *Client) PackageReleaseStatus(ctx context.Context, ref Ref, release string) (PackageReleaseCommit, *exit.Error) {
	var out PackageReleaseCommit
	e := c.do(ctx, call{method: http.MethodGet,
		path: packagePublishPath(ref, release) + "/status", auth: true,
		strict: true, responseBytes: 1 << 20}, &out)
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
		var problem *exit.Error
		state, problem = c.PackageReleaseStatus(callCtx, ref, release)
		cancel()
		if problem != nil {
			return state, problem
		}
	}
}

type PackageDefectReport struct {
	Code     string `json:"code"`
	Detail   string `json:"detail"`
	RentalID string `json:"rental_id"`
}

type PackageDefectResult struct {
	Changed    bool   `json:"changed"`
	Code       string `json:"code"`
	Release    string `json:"release"`
	ReportedAt string `json:"reported_at"`
	State      string `json:"state"`
}

// ReportPackageDefect relays a package-interface-falsifying pod refusal (th-106). The
// hub authorizes the report by rental OWNERSHIP: this account owns the named rental
// against a committed release. The signed-delegation chain that used to prove the
// rental had downloaded these exact bytes is deleted (owner ruling 2026-09-03), so the
// report's provenance is no longer proven.
func (c *Client) ReportPackageDefect(ctx context.Context, ref Ref, release string,
	report PackageDefectReport, reason string,
) (PackageDefectResult, *exit.Error) {
	var out PackageDefectResult
	e := c.do(ctx, call{method: http.MethodPost,
		path: packageReleasePath(ref, release) + "/defects", auth: true, reason: reason,
		body: report, strict: true}, &out)
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

func (c *Client) PackageSourceArchive(ctx context.Context, ref Ref, release string) (PackageSourceArchive, *exit.Error) {
	var out PackageSourceArchive
	e := c.do(ctx, call{method: http.MethodGet,
		path: packageReleasePath(ref, release) + "/source", strict: true,
		responseBytes: 1 << 20}, &out)
	return out, e
}

func (c *Client) PackageDownloads(ctx context.Context, ref Ref, release string, pythonVersion ...string) (PackageDownloadPlan, *exit.Error) {
	var out PackageDownloadPlan
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
		path: path, body: struct{}{}, strict: true,
		responseBytes: 16 << 20}, &out)
	if e == nil && (len(out.Downloads) == 0 || len(out.Downloads) > MaxPackageInstallDownloads) {
		e = exit.Named(exit.Structural, "hub.package_download_plan_invalid",
			"Tensorhub returned %d package wheel downloads; expected 1 through %d",
			len(out.Downloads), MaxPackageInstallDownloads)
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
	Lane string `json:"lane"`
}

func (r BindingRung) String() string { return r.GPU + "=" + r.Lane }

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

// PackageBindings is the anonymous read of a package's current owner overrides.
func (c *Client) PackageBindings(ctx context.Context, ref Ref) ([]PackageBindingRow, *exit.Error) {
	var out struct {
		Bindings []PackageBindingRow `json:"bindings"`
	}
	e := c.do(ctx, call{method: http.MethodGet,
		path: resourcePath("packages", ref) + "/bindings", strict: true}, &out)
	if e != nil {
		return nil, e
	}
	seen := make(map[string]bool, len(out.Bindings))
	for _, row := range out.Bindings {
		if row.Slot == "" || row.Model == "" || row.Release == "" || row.Revision < 1 ||
			seen[row.Slot] || ValidateLadder(row.Ladder) != nil {
			return nil, exit.Named(exit.Structural, "hub.package_bindings_invalid",
				"Tensorhub returned an incomplete or duplicate package binding row for slot %q", row.Slot)
		}
		seen[row.Slot] = true
	}
	return out.Bindings, nil
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
		auth: true, reason: reason, strict: true,
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
		auth: true, reason: reason, strict: true,
		body: map[string]any{"expected_revision": expectedRevision}}, &out)
	return out, problem
}
