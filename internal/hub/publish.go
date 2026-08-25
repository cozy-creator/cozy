package hub

// th-002's publish protocol and th-002/th-003's checkpoint reads, as methods on the
// ONE client (cl-012). `do` still owns the request build, the credential, the reason
// header and the whole error mapping — these add shapes, never a second client.
//
// The protocol, in the hub's own words (tensorhub README, "The publish protocol"):
// declare the whole canonical object set before a byte moves; upload only what the
// hub says it lacks, straight to the FINAL content keys, under every condition the
// grant signed; the hub streams every object back and hashes it ITSELF; completion
// runs the hermetic verifier and installs exactly one root. A client receipt never
// substitutes for the hub's own proof (law 18), which is why nothing here reports
// what a byte hashed to.

import (
	"context"
	"encoding/base64"
	"net/http"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// BeginRequest is the declaration. `closure` and `code_topology` are the exact bytes
// tfs produced; `manifest` is the exact snapshot manifest object, which is NOT in the
// closure (tensorfs's PublishClosure covers what a publish MOVES and names the
// snapshot separately) and therefore rides the declaration.
//
// THE MODEL FAMILY IS NOT DECLARED (th-003, tensorhub 1353e53). th-002 took it as a
// declaration because no classifier existed; th-003 derives it at completion by
// comparing the installed checkpoint's topology_digest against the structure fixtures
// tfs banks — family is detectable from artifacts, never self-declared. The hub
// decodes with DisallowUnknownFields, so a client that still sends the field cannot
// publish at all; cl-006's real-hub side-check found exactly that and cl-010 removed
// it, along with `cozy push --family`.
type BeginRequest struct {
	Session      string   `json:"session"`
	Closure      string   `json:"closure"`
	CodeTopology string   `json:"code_topology"`
	SnapshotID   string   `json:"snapshot_id"`
	Manifest     string   `json:"manifest"`
	HeaderID     string   `json:"header_id"`
	Objects      []Object `json:"objects"`
	RawCarrier   bool     `json:"raw_carrier"`
}

// Object is one declared object identity and its length.
type Object struct {
	ID     string `json:"object_id"`
	Length int64  `json:"length"`
}

// B64 wraps exact document bytes for the declaration.
func B64(raw []byte) string { return base64.StdEncoding.EncodeToString(raw) }

// Session is the hub's view of one publish.
type Session struct {
	ID              string `json:"publish_id"`
	Session         string `json:"session"`
	State           string `json:"state"`
	ClosureID       string `json:"closure_id"`
	SnapshotID      string `json:"snapshot_id"`
	HeaderID        string `json:"header_id"`
	DeclaredObjects int    `json:"declared_objects"`
	DeclaredBytes   int64  `json:"declared_bytes"`
	ExpiresAt       string `json:"expires_at"`
}

// Totals is what the declaration adds up to, as the HUB counted it. The names are
// the hub's own; nothing here re-derives a count it was told.
type Totals struct {
	DeclaredObjects int   `json:"declared_objects"`
	DeclaredBytes   int64 `json:"declared_bytes"`
	MissingObjects  int   `json:"missing_objects"`
	// MissingBytes is what this publish will actually MOVE. Everything else the
	// declaration named, the hub already holds — and a progress line that does not
	// print both is claiming credit for bytes it never sent.
	MissingBytes int64 `json:"missing_bytes"`
	HeldObjects  int   `json:"held_objects"`
}

// HeldBytes is what the hub already holds, which it does not count for us: it is the
// declaration minus what has to move. Subtraction, not a second census.
func (t Totals) HeldBytes() int64 { return t.DeclaredBytes - t.MissingBytes }

// Missing is one object the hub does not hold, with the state it is in. A resumed
// publish reads these states rather than any local record.
type Missing struct {
	ID     string `json:"object_id"`
	Length int64  `json:"length"`
	State  string `json:"state"`
}

// BeginResponse is the missing-object answer. `held` is a list of IDENTITIES: the
// hub says which objects it already has, not how big they were — it knows that from
// the declaration it just read.
type BeginResponse struct {
	Publish Session   `json:"publish"`
	Missing []Missing `json:"missing"`
	Held    []string  `json:"held"`
	Totals  Totals    `json:"totals"`
	Created bool      `json:"created"`
}

// Begin declares the whole set. It is idempotent on the closure digest: re-declaring
// the same set in the same repo returns the SAME session, RE-PLANNED — which is what
// makes an interrupted publish resumable with no client-side state at all.
func (c *Client) Begin(ctx context.Context, ref Ref, req BeginRequest, reason string) (BeginResponse, *exit.Error) {
	var out BeginResponse
	e := c.do(ctx, call{
		method: http.MethodPost, path: publishes(ref), admin: true, reason: reason,
		body: req, timeout: Transfer,
	}, &out)
	return out, e
}

// Grant is one authorization to write one object at its final content key. Either a
// single signed PUT, or a ranged plan whose parts never enter identity.
type Grant struct {
	ObjectID string            `json:"object_id"`
	Length   int64             `json:"length"`
	Key      string            `json:"key"`
	Method   string            `json:"method"`
	URL      string            `json:"url"`
	Expires  string            `json:"expires_at"`
	Headers  map[string]string `json:"required_headers"`
	UploadID string            `json:"upload_id"`
	Parts    []Part            `json:"parts"`
}

// Part is one ranged upload leg. Its number and range are transport, never identity.
type Part struct {
	Number int    `json:"number"`
	URL    string `json:"url"`
	Offset int64  `json:"offset"`
	Length int64  `json:"length"`
}

// Multipart says whether this grant is the ranged plan. R2 implements no sha256
// checksum on multipart (decisions #310), so the ranged path keeps no-clobber and
// the hub's own streaming hash discharges the digest.
func (g Grant) Multipart() bool { return g.Method == "MULTIPART" }

// Grants asks for authorization over the named objects; an empty list means every
// missing object. An expired grant comes back as `grant.expired_replan` — a REPLAN,
// never a retry-as-failure.
func (c *Client) Grants(ctx context.Context, ref Ref, publishID string, ids []string, reason string) ([]Grant, *exit.Error) {
	var out struct {
		Grants []Grant `json:"grants"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost, path: publishes(ref) + "/" + publishID + "/grants",
		admin: true, reason: reason, timeout: Transfer,
		body: map[string]any{"object_ids": ids},
	}, &out)
	return out.Grants, e
}

// FinishMultipart assembles a ranged upload. The HUB owns this call because it
// carries the no-clobber precondition; a client that assembled its own could replace
// verified bytes.
func (c *Client) FinishMultipart(ctx context.Context, ref Ref, publishID, objectID string, etags []string, reason string) (bool, *exit.Error) {
	var out struct {
		Conflict bool `json:"conflict"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost, path: publishes(ref) + "/" + publishID + "/multipart/complete",
		admin: true, reason: reason, timeout: Transfer,
		body: map[string]any{"object_id": objectID, "etags": etags},
	}, &out)
	return out.Conflict, e
}

// VerifyReport is what the publisher observed at the storage edge, and it is
// deliberately thin: whether the write hit an existing key. It carries no checksum,
// because the hub does not accept one.
type VerifyReport struct {
	ObjectID string `json:"object_id"`
	Conflict bool   `json:"conflict"`
}

// Verdict is the hub's own reading of an object it streamed back and hashed.
type Verdict struct {
	ObjectID       string `json:"object_id"`
	State          string `json:"state"`
	Bytes          int64  `json:"bytes"`
	ChecksumSource string `json:"checksum_source"`
	Precondition   string `json:"precondition"`
	MS             int64  `json:"ms"`
	Detail         string `json:"detail"`
}

// VerifyObjects asks the hub to prove every uploaded object for itself: it HEADs the
// key, streams the object back through its own sha256, and mints the receipt from
// what it observed. PUT success and an ETag are not proof.
func (c *Client) VerifyObjects(ctx context.Context, ref Ref, publishID string, reports []VerifyReport, reason string) ([]Verdict, *exit.Error) {
	var out struct {
		Objects []Verdict `json:"objects"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost, path: publishes(ref) + "/" + publishID + "/objects/verify",
		admin: true, reason: reason, timeout: Transfer,
		body: map[string]any{"objects": reports},
	}, &out)
	return out.Objects, e
}

// Root is the installed checkpoint: exactly one snapshot, committed idempotently.
type Root struct {
	SnapshotID     string `json:"snapshot_id"`
	HeaderID       string `json:"header_id"`
	TopologyDigest string `json:"topology_digest"`
	CatalogRootID  string `json:"catalog_root_id"`
	CatalogRootKey string `json:"catalog_root_key"`
	Objects        int    `json:"objects"`
	Bytes          int64  `json:"bytes"`
	VerifierBuild  string `json:"verifier_build"`
	Grade          string `json:"grade"`
}

// CompleteResponse is the committed result. A duplicate completion returns the
// ORIGINAL result out of the ledger with duplicate=true — the same root, the same
// catalog root id, byte for byte.
type CompleteResponse struct {
	PublishID string `json:"publish_id"`
	Root      Root   `json:"root"`
	// Verifier is the hermetic verdict, consumed as the hub rendered it. The field
	// spellings are the verifier's own (capitalised), and they are copied, not
	// re-modelled: an alternate semantic model of somebody else's verdict is how a
	// consumer starts disagreeing with the thing that decided.
	Verifier struct {
		Report       string `json:"Report"`
		Lane         string `json:"Lane"`
		Satisfaction string `json:"Satisfaction"`
		Grade        string `json:"Grade"`
	} `json:"verifier"`
	MS                    map[string]int64 `json:"ms"`
	ReingestedHeldObjects int              `json:"reingested_held_objects"`
	Duplicate             bool             `json:"duplicate"`
}

// Complete constructs the exact subject, runs the hermetic verifier, installs one
// root and commits the catalog. It re-verifies every object it already held (law 18:
// the hub re-proves the bytes regardless of what the publisher declared).
func (c *Client) Complete(ctx context.Context, ref Ref, publishID, reason string) (CompleteResponse, *exit.Error) {
	var out CompleteResponse
	e := c.do(ctx, call{
		method: http.MethodPost, path: publishes(ref) + "/" + publishID + "/complete",
		admin: true, reason: reason, timeout: Transfer,
	}, &out)
	return out, e
}

// Abort drops a session: it aborts the session's ranged uploads (an unaborted one is
// billed storage no object listing can show) and drops its scratch.
func (c *Client) Abort(ctx context.Context, ref Ref, publishID, reason string) *exit.Error {
	return c.do(ctx, call{
		method: http.MethodDelete, path: publishes(ref) + "/" + publishID,
		admin: true, reason: reason,
	}, nil)
}

// ---------------------------------------------------------------- checkpoint reads

// Checkpoint is one installed checkpoint row. Public: reads carry no credential.
type Checkpoint struct {
	Org            string `json:"org"`
	Name           string `json:"name"`
	SnapshotID     string `json:"snapshot_id"`
	HeaderID       string `json:"header_id"`
	TopologyDigest string `json:"topology_digest"`
	Objects        int    `json:"objects"`
	Bytes          int64  `json:"bytes"`
	Grade          string `json:"grade"`
	VerifierBuild  string `json:"verifier_build"`
	InstalledAt    string `json:"installed_at"`
}

// Checkpoints lists what a repo holds. With an empty ref it lists the whole catalog.
func (c *Client) Checkpoints(ctx context.Context, ref Ref) ([]Checkpoint, *exit.Error) {
	path := "/v1/checkpoints"
	if ref.Org != "" {
		path = "/v1/repos/" + ref.Org + "/" + ref.Name + "/checkpoints"
	}
	var out struct {
		Checkpoints []Checkpoint `json:"checkpoints"`
	}
	e := c.do(ctx, call{method: http.MethodGet, path: path}, &out)
	return out.Checkpoints, e
}

// Manifest reads the exact SnapshotManifest bytes back, verbatim. They are handed
// straight to the byte plane, which admits them only if they hash to the snapshot id
// asked for — so a hub that lied about a manifest cannot install one.
func (c *Client) Manifest(ctx context.Context, ref Ref, snapshot string) ([]byte, *exit.Error) {
	var raw []byte
	e := c.do(ctx, call{
		method: http.MethodGet,
		path:   "/v1/repos/" + ref.Org + "/" + ref.Name + "/checkpoints/" + snapshot + "/manifest",
		raw:    &raw, timeout: Transfer,
	}, nil)
	return raw, e
}

// Read is one authorization to fetch one object. It is the mirror of Grant, and the
// same rule holds: the URL is presigned at the final content key and nothing about
// it is a claim regarding the bytes — the fetcher proves those itself, on the way in.
type Read struct {
	ObjectID string `json:"object_id"`
	Length   int64  `json:"length"`
	URL      string `json:"url"`
	Expires  string `json:"expires_at"`
}

// Reads asks for download authorization over the named objects of one installed
// checkpoint.
//
// AWAITING A HUB ROUTE. th-002 landed the whole write side and no read side: the hub
// signs PUTs at final keys and streams objects back for its OWN verification, but
// exposes no route that hands a client a presigned GET (`objstore.PresignGet` exists
// and has no caller behind a route). Until one lands, a Launch-1 hub is
// publish-only, and this refuses by NAME — a fetch that silently invented a bucket
// URL, or that reached object storage with a credential of its own, would be a
// second custody authority, which is exactly what law 2 forbids.
func (c *Client) Reads(ctx context.Context, ref Ref, snapshot string, ids []string) ([]Read, *exit.Error) {
	var out struct {
		Reads []Read `json:"reads"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost, timeout: Transfer,
		path: "/v1/repos/" + ref.Org + "/" + ref.Name + "/checkpoints/" + snapshot + "/reads",
		body: map[string]any{"object_ids": ids},
	}, &out)
	if e != nil && (e.Name == "hub.untyped_refusal" || e.Name == "route.not_found") {
		return nil, exit.Named(exit.Unavailable, "hub.no_read_plane",
			"the hub at %s serves no object-read route: POST %s answered %q", c.base,
			"…/checkpoints/{snapshot}/reads", e.Name).
			WithRemedy("this hub can take custody of bytes and cannot hand them back yet; the read grant is the missing half of th-002's transfer protocol").
			WithNext("cozy push <org/repo> <sha256:…>", "cozy pull --dry-run "+ref.String())
	}
	return out.Reads, e
}

func publishes(ref Ref) string {
	return "/v1/repos/" + ref.Org + "/" + ref.Name + "/publishes"
}

// PublishState is the resume read: the session and every declared object's state.
// Public — a publisher can see where it got to without a credential.
func (c *Client) PublishState(ctx context.Context, ref Ref, publishID string) (Session, []Verdict, *exit.Error) {
	var out struct {
		Publish Session   `json:"publish"`
		Objects []Verdict `json:"objects"`
	}
	e := c.do(ctx, call{method: http.MethodGet, path: publishes(ref) + "/" + publishID}, &out)
	return out.Publish, out.Objects, e
}
