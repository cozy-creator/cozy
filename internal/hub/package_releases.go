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

// PackageDeclaredFile is one publication subject: the caller's digest claim.
// The hub answers each from the store's own HEAD (th-094 shape) and re-hashes
// the stored bytes itself at finalize.
type PackageDeclaredFile struct {
	Digest string `json:"digest"`
	Kind   string `json:"kind"`
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
	State string             `json:"state"`
	Files []PackageFileGrant `json:"files"`
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
		Defective               bool   `json:"defective,omitempty"`
		DefectiveAt             string `json:"defective_at,omitempty"`
		DefectiveCode           string `json:"defective_code,omitempty"`
	} `json:"release"`
	Document          json.RawMessage `json:"document"`
	PackageDescriptor json.RawMessage `json:"package_descriptor"`
}

// Requirements returns the immutable execution dependencies from the exact
// PackageManifest/1 bytes. Creator needs only this one release fact to choose a
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
			"Tensorhub returned an invalid PackageManifest/1 document")
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
		document.Format != "cozy.package.manifest/1" || document.Requirements == nil {
		return nil, "", exit.Named(exit.Structural, "hub.package_release_invalid",
			"Tensorhub returned an invalid PackageManifest/1 document")
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
	files []PackageDeclaredFile, registry []PackageRegistryRow, reason string,
) (PackageReleaseCommit, *exit.Error) {
	if registry == nil {
		registry = []PackageRegistryRow{}
	}
	var out PackageReleaseCommit
	e := c.do(ctx, call{method: http.MethodPost,
		path: packagePublishPath(ref, release) + "/finalize", auth: true, reason: reason,
		body: map[string]any{"files": files, "registry": registry}, patient: true, strict: true}, &out)
	return out, e
}

type PackageDefectReport struct {
	Code                string `json:"code"`
	Detail              string `json:"detail"`
	RentalID            string `json:"rental_id"`
	DelegationBase64URL string `json:"delegation_base64url"`
	SignatureBase64URL  string `json:"signature_base64url"`
}

type PackageDefectResult struct {
	Changed    bool   `json:"changed"`
	Code       string `json:"code"`
	Release    string `json:"release"`
	ReportedAt string `json:"reported_at"`
	State      string `json:"state"`
}

// ReportPackageDefect relays a descriptor-falsifying pod refusal (th-106). The
// hub authorizes the report by its chain: this account owns the named rental
// and the presented creator-signed delegation named this exact release.
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
