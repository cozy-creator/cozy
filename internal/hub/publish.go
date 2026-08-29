package hub

// Tensorhub's incremental model-publication protocol and checkpoint reads, as methods
// on the ONE client. `do` still owns request construction, credentials, reasons, and
// error mapping. A publication opens under a stable operation id, claims known object
// transfers, settles each transfer through grant -> received -> verify, then seals the
// exact TensorFS documents. The retired declare-whole route has no compatibility path.

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/url"

	"github.com/cozy-creator/cozy-creator/internal/exit"
)

// Object is one known ObjectRef transfer identity and its length.
type Object struct {
	ID     string `json:"object_id"`
	Length int64  `json:"length"`
}

// B64 wraps exact document bytes for the declaration.
func B64(raw []byte) string { return base64.StdEncoding.EncodeToString(raw) }

// Session is the hub's durable view of one publication.
type Session struct {
	ID              string `json:"publish_id"`
	Session         string `json:"session"`
	StorageDomain   string `json:"storage_domain"`
	CustodyScope    string `json:"custody_scope"`
	State           string `json:"state"`
	ClosureID       string `json:"closure_id"`
	SnapshotID      string `json:"snapshot_id"`
	HeaderID        string `json:"header_id"`
	DeclaredObjects int    `json:"declared_objects"`
	DeclaredBytes   int64  `json:"declared_bytes"`
	ExpiresAt       string `json:"expires_at"`
}

// Totals is Creator's accounting over the exact transfer rows Tensorhub returned.
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

type OpenPublicationResponse struct {
	Publication Session `json:"publication"`
	Created     bool    `json:"created"`
}

func (c *Client) OpenPublication(ctx context.Context, ref Ref, operationID, reason string) (OpenPublicationResponse, *exit.Error) {
	var out OpenPublicationResponse
	e := c.do(ctx, call{
		method: http.MethodPost, path: publications(ref), admin: true, reason: reason,
		body: map[string]string{"operation_id": operationID},
	}, &out)
	return out, e
}

// Transfer is one durable known-object transfer row. Its state is the restart journal.
type Transfer struct {
	TransferID    string `json:"transfer_id"`
	IntakeMode    string `json:"intake_mode"`
	ObjectID      string `json:"object_id"`
	Length        int64  `json:"length"`
	State         string `json:"state"`
	ReceivedBytes int64  `json:"received_bytes"`
	VerifiedBytes int64  `json:"verified_bytes"`
}

func (c *Client) ClaimKnownTransfers(ctx context.Context, ref Ref, publicationID string,
	objects []Object, reason string,
) ([]Transfer, *exit.Error) {
	var out struct {
		Transfers []Transfer `json:"transfers"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost,
		path:   publications(ref) + "/" + publicationID + "/known-transfers",
		admin:  true, reason: reason, byBytes: true,
		body: map[string]any{"objects": objects},
	}, &out)
	return out.Transfers, e
}

// Grant is one authorization to write one known object at its final content key.
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

type GrantResponse struct {
	Grants []Grant    `json:"grants"`
	Held   []Transfer `json:"held"`
}

// GrantKnownTransfer asks for one transfer immediately before its bytes move. One at
// a time keeps a large multipart response bounded.
func (c *Client) GrantKnownTransfer(ctx context.Context, ref Ref, publicationID,
	transferID, reason string,
) (GrantResponse, *exit.Error) {
	var out GrantResponse
	e := c.do(ctx, call{
		method: http.MethodPost,
		path:   publications(ref) + "/" + publicationID + "/known-transfers/grants",
		admin:  true, reason: reason, byBytes: true,
		body: map[string]any{"transfer_ids": []string{transferID}},
	}, &out)
	if e != nil {
		return GrantResponse{}, e
	}
	if len(out.Grants)+len(out.Held) != 1 {
		return GrantResponse{}, exit.Internalf(
			"grant for transfer %s answered %d grants and %d held rows",
			transferID, len(out.Grants), len(out.Held),
		)
	}
	return out, nil
}

type ReceivedTransfer struct {
	TransferID    string `json:"transfer_id"`
	ReceivedBytes int64  `json:"received_bytes"`
	Precondition  string `json:"precondition"`
}

func (c *Client) MarkTransfersReceived(ctx context.Context, ref Ref, publicationID string,
	transfers []ReceivedTransfer, reason string,
) ([]Transfer, *exit.Error) {
	var out struct {
		Transfers []Transfer `json:"transfers"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost,
		path:   publications(ref) + "/" + publicationID + "/known-transfers/received",
		admin:  true, reason: reason, byBytes: true,
		body: map[string]any{"transfers": transfers},
	}, &out)
	return out.Transfers, e
}

// Verdict is Tensorhub's own full-byte reading of one transfer.
type Verdict struct {
	ObjectID       string `json:"object_id"`
	State          string `json:"state"`
	Bytes          int64  `json:"bytes"`
	ChecksumSource string `json:"checksum_source"`
	Precondition   string `json:"precondition"`
	MS             int64  `json:"ms"`
	Detail         string `json:"detail"`
}

func (c *Client) VerifyKnownTransfers(ctx context.Context, ref Ref, publicationID string,
	transferIDs []string, reason string,
) ([]Verdict, *exit.Error) {
	var out struct {
		Transfers []Verdict `json:"transfers"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost,
		path:   publications(ref) + "/" + publicationID + "/known-transfers/verify",
		admin:  true, reason: reason, byBytes: true,
		body: map[string]any{"transfer_ids": transferIDs},
	}, &out)
	return out.Transfers, e
}

// FinishMultipart assembles a ranged upload. Tensorhub owns the final no-clobber
// conditional.
func (c *Client) FinishMultipart(ctx context.Context, ref Ref, publicationID, objectID string, etags []string, reason string) (bool, *exit.Error) {
	var out struct {
		Conflict bool `json:"conflict"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost,
		path:   publications(ref) + "/" + publicationID + "/multipart/complete",
		admin:  true, reason: reason, byBytes: true,
		body: map[string]any{"object_id": objectID, "etags": etags},
	}, &out)
	return out.Conflict, e
}

type SealPublicationRequest struct {
	Closure      string           `json:"closure"`
	CodeTopology string           `json:"code_topology"`
	SnapshotID   string           `json:"snapshot_id"`
	Manifest     string           `json:"manifest"`
	HeaderID     string           `json:"header_id"`
	Stamps       []map[string]any `json:"stamps"`
	Receipt      any              `json:"receipt"`
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

// CompleteResponse is the committed seal result. Exact replay returns the original
// result with duplicate=true.
type CompleteResponse struct {
	PublishID string `json:"publish_id"`
	Root      Root   `json:"root"`
	Verifier  struct {
		Report       string `json:"Report"`
		Lane         string `json:"Lane"`
		Satisfaction string `json:"Satisfaction"`
		Grade        string `json:"Grade"`
	} `json:"verifier"`
	MS                    map[string]int64 `json:"ms"`
	ReingestedHeldObjects int              `json:"reingested_held_objects"`
	Duplicate             bool             `json:"duplicate"`
}

func (c *Client) SealPublication(ctx context.Context, ref Ref, publicationID string,
	request SealPublicationRequest, reason string,
) (CompleteResponse, *exit.Error) {
	var out CompleteResponse
	e := c.do(ctx, call{
		method: http.MethodPost,
		path:   publications(ref) + "/" + publicationID + "/seal",
		admin:  true, reason: reason, byBytes: true,
		body: request,
	}, &out)
	return out, e
}

func publications(ref Ref) string {
	return "/v1/models/" + ref.Org + "/" + ref.Name + "/publications"
}

// The old whole-closure Begin/Grant/VerifyObjects/Complete API is absent.
// Publication progress is durable only as transfer rows in Tensorhub.

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

// ModelResolution is Tensorhub's exact answer to a human model ref. Download never
// lists checkpoints and guesses: digest or release selection happens at this route.
type ModelResolution struct {
	Ref        string   `json:"ref"`
	Model      string   `json:"model"`
	Release    string   `json:"release"`
	Alias      string   `json:"alias"`
	Lane       string   `json:"lane"`
	LaneID     string   `json:"lane_id"`
	Checkpoint string   `json:"checkpoint_id"`
	HeaderID   string   `json:"header_digest"`
	Structure  string   `json:"structure"`
	Encodings  []string `json:"encodings"`
	Objects    int      `json:"objects"`
	Bytes      int64    `json:"bytes"`
	Pinned     string   `json:"pinned"`
	Note       string   `json:"note"`
}

func (c *Client) ResolveModel(ctx context.Context, spec, lane string) (ModelResolution, *exit.Error) {
	query := url.Values{"ref": []string{spec}}
	if lane != "" {
		query.Set("lane", lane)
	}
	var out ModelResolution
	e := c.do(ctx, call{method: http.MethodGet,
		path: "/v1/models/resolve?" + query.Encode()}, &out)
	return out, e
}

// Manifest reads the exact SnapshotManifest bytes back, verbatim. They are handed
// straight to the byte plane, which admits them only if they hash to the snapshot id
// asked for — so a hub that lied about a manifest cannot install one.
func (c *Client) Manifest(ctx context.Context, ref Ref, snapshot string) ([]byte, *exit.Error) {
	var raw []byte
	e := c.do(ctx, call{
		method: http.MethodGet,
		path:   "/v1/models/" + ref.Org + "/" + ref.Name + "/checkpoints/" + snapshot + "/manifest",
		raw:    &raw, byBytes: true,
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

// Reads asks Tensorhub's live typed model route for download authorization over the
// named objects of one installed checkpoint. Creator never constructs a bucket URL or
// reaches storage with credentials of its own.
func (c *Client) Reads(ctx context.Context, ref Ref, snapshot string, ids []string) ([]Read, *exit.Error) {
	var out struct {
		Reads []Read `json:"reads"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost, byBytes: true,
		path: "/v1/models/" + ref.Org + "/" + ref.Name + "/checkpoints/" + snapshot + "/reads",
		body: map[string]any{"object_ids": ids},
	}, &out)
	if e != nil && (e.Name == "hub.untyped_refusal" || e.Name == "route.not_found") {
		return nil, exit.Named(exit.Unavailable, "hub.no_read_plane",
			"the hub at %s serves no object-read route: POST %s answered %q", c.base,
			"…/checkpoints/{snapshot}/reads", e.Name).
			WithRemedy("this hub can take custody of bytes and cannot hand them back yet; the read grant is the missing half of th-002's transfer protocol").
			WithNext("cozy model publish <org/model> <sha256:…>", "cozy model download --dry-run "+ref.String())
	}
	return out.Reads, e
}
