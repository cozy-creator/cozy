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
	PublicationID   string   `json:"publication_id"`
	State           string   `json:"state"`
	BindingWarnings []string `json:"binding_warnings,omitempty"`
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
	publicationID string, registry []PackageRegistryRow, reason string,
) (PackageReleaseCommit, *exit.Error) {
	if registry == nil {
		registry = []PackageRegistryRow{}
	}
	var out PackageReleaseCommit
	e := c.do(ctx, call{method: http.MethodPost,
		path: packagePublishPath(ref, release) + "/finalize", auth: true, reason: reason,
		body:    map[string]any{"publication_id": publicationID, "registry": registry},
		patient: true, strict: true}, &out)
	return out, e
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

// ---------------------------------------------------------------- bindings (th-116)

// PackageBindingRow is one mutable hub default: which model a package slot
// loads when no `model.<param>=` run key speaks. Seeded from the shipped
// package.toml at release commit and owner-mutable afterwards; the hub row is
// the ONE source and the in-release toml is never consulted post-seed.
type PackageBindingRow struct {
	Slot      string `json:"slot"`
	Model     string `json:"model"`
	Release   string `json:"release,omitempty"`
	Lane      string `json:"lane,omitempty"`
	Revision  int64  `json:"revision"`
	UpdatedAt string `json:"updated_at"`
}

// Ref renders the row as the ladder's org/model[@release] spelling.
func (b PackageBindingRow) Ref() string {
	if b.Release == "" {
		return b.Model
	}
	return b.Model + "@" + b.Release
}

// PackageBindings is the anonymous read of a package's current default bindings.
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
		if row.Slot == "" || row.Model == "" || row.Revision < 1 || seen[row.Slot] {
			return nil, exit.Named(exit.Structural, "hub.package_bindings_invalid",
				"Tensorhub returned an incomplete or duplicate package binding row")
		}
		seen[row.Slot] = true
	}
	return out.Bindings, nil
}

type PackageBindingWrite struct {
	Binding PackageBindingRow `json:"binding"`
	Changed bool              `json:"changed"`
}

// BindPackageSlot moves one package slot's default to an arbitrary
// model/release/lane under CAS on expectedRevision (0 creates an unseeded row).
func (c *Client) BindPackageSlot(ctx context.Context, ref Ref, slot, model, release, lane string,
	expectedRevision int64, reason string,
) (PackageBindingWrite, *exit.Error) {
	var out PackageBindingWrite
	e := c.do(ctx, call{method: http.MethodPut,
		path: resourcePath("packages", ref) + "/bindings/" + url.PathEscape(slot),
		auth: true, reason: reason, strict: true,
		body: map[string]any{"model": model, "release": release, "lane": lane,
			"expected_revision": expectedRevision}}, &out)
	return out, e
}
