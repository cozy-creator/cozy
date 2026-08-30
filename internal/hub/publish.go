package hub

// Tensorhub's incremental model-publication protocol and manifest reads, as methods
// on the ONE client. `do` still owns request construction, credentials, reasons, and
// error mapping. A publication opens under a stable operation id, claims known object
// transfers, settles each transfer through grant -> received -> verify, then seals the
// exact TensorFS documents. The retired declare-whole route has no compatibility path.

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/url"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Object is one known ObjectRef transfer identity and its length.
type Object struct {
	ID     string `json:"object_id"`
	Length int64  `json:"length"`
}

// B64 wraps exact document bytes for the declaration.
func B64(raw []byte) string { return base64.StdEncoding.EncodeToString(raw) }

// Session is the compact durable view returned by the idempotent publication PUT.
type Session struct {
	Operation string     `json:"operation"`
	Release   string     `json:"release"`
	Lane      string     `json:"lane"`
	State     string     `json:"state"`
	Objects   []Transfer `json:"objects"`
}

// Totals is Cozy's accounting over the exact transfer rows Tensorhub returned.
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

func (c *Client) OpenPublication(ctx context.Context, ref Ref, operationID, release, lane string,
	objects []Object, reason string,
) (OpenPublicationResponse, *exit.Error) {
	var out OpenPublicationResponse
	e := c.do(ctx, call{
		method: http.MethodPut,
		path:   publications(ref) + "/" + url.PathEscape(operationID), auth: true, reason: reason,
		body:    map[string]any{"release": release, "lane": lane, "objects": objects},
		byBytes: true, patient: true,
	}, &out)
	return out, e
}

// Transfer is one durable known-object transfer row. Its state is the restart journal.
type Transfer struct {
	ObjectID string `json:"object_id"`
	Length   int64  `json:"length"`
	State    string `json:"state"`
}

// Grant is one authorization to write one known object at its final content key.
type Grant struct {
	ObjectID string            `json:"object_id"`
	Length   int64             `json:"length"`
	Key      string            `json:"key"`
	URL      string            `json:"url"`
	Expires  string            `json:"expires_at"`
	Headers  map[string]string `json:"required_headers"`
}

type GrantResponse struct {
	Grants []Grant    `json:"grants"`
	Held   []Transfer `json:"held"`
}

// GrantKnownTransfers asks for one bounded transfer batch immediately before its
// bytes move. Tensorhub returns exactly one grant or held row per requested object.
func (c *Client) GrantKnownTransfers(ctx context.Context, ref Ref, operation string,
	objectIDs []string, reason string,
) (GrantResponse, *exit.Error) {
	var out GrantResponse
	e := c.do(ctx, call{
		method: http.MethodPost,
		path:   publications(ref) + "/" + url.PathEscape(operation) + "/grants",
		auth:   true, reason: reason, byBytes: true, patient: true,
		body: map[string]any{"object_ids": objectIDs},
	}, &out)
	if e != nil {
		return GrantResponse{}, e
	}
	if len(out.Grants)+len(out.Held) != len(objectIDs) {
		return GrantResponse{}, exit.Internalf(
			"grant for %d objects answered %d grants and %d held rows",
			len(objectIDs), len(out.Grants), len(out.Held),
		)
	}
	want := make(map[string]bool, len(objectIDs))
	for _, objectID := range objectIDs {
		if objectID == "" || want[objectID] {
			return GrantResponse{}, exit.Internalf("grant request contains an empty or duplicate object id")
		}
		want[objectID] = true
	}
	seen := make(map[string]bool, len(objectIDs))
	for _, grant := range out.Grants {
		if !want[grant.ObjectID] || seen[grant.ObjectID] {
			return GrantResponse{}, exit.Internalf("grant response contains an absent or duplicate object %s", grant.ObjectID)
		}
		seen[grant.ObjectID] = true
	}
	for _, held := range out.Held {
		if !want[held.ObjectID] || seen[held.ObjectID] {
			return GrantResponse{}, exit.Internalf("grant response contains an absent or duplicate held object %s", held.ObjectID)
		}
		seen[held.ObjectID] = true
	}
	return out, nil
}

type SettleObject struct {
	ObjectID       string `json:"object_id"`
	AlreadyPresent bool   `json:"already_present,omitempty"`
}

type SettledObject struct {
	ObjectID       string `json:"object_id"`
	State          string `json:"state"`
	ChecksumSource string `json:"checksum_source"`
	Conflict       bool   `json:"conflict"`
}

func (c *Client) SettleObjects(ctx context.Context, ref Ref, operation string,
	objects []SettleObject, reason string,
) ([]SettledObject, *exit.Error) {
	var out struct {
		Objects []SettledObject `json:"objects"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost,
		path:   publications(ref) + "/" + url.PathEscape(operation) + "/settle",
		auth:   true, reason: reason, byBytes: true, patient: true,
		body: map[string]any{"objects": objects},
	}, &out)
	return out.Objects, e
}

type SealPublicationRequest struct {
	Manifest              string `json:"manifest"`
	ReleaseEvidenceBase64 string `json:"release_evidence_base64"`
}

type ManifestRef struct {
	SHA256 string `json:"sha256"`
	Length int64  `json:"length"`
}

// CompleteResponse is the committed (release,lane)->manifest result. Exact replay
// returns the original result with duplicate=true.
type CompleteResponse struct {
	PublishID             string      `json:"publish_id"`
	Release               string      `json:"release"`
	Lane                  string      `json:"lane"`
	Manifest              ManifestRef `json:"manifest"`
	TopologyDigest        string      `json:"topology_digest"`
	Objects               int         `json:"objects"`
	Bytes                 int64       `json:"bytes"`
	ReleaseEvidenceBase64 string      `json:"release_evidence_base64"`
	Duplicate             bool        `json:"duplicate"`
}

func (c *Client) SealPublication(ctx context.Context, ref Ref, operation string,
	request SealPublicationRequest, reason string,
) (CompleteResponse, *exit.Error) {
	var out CompleteResponse
	e := c.do(ctx, call{
		method: http.MethodPost,
		path:   publications(ref) + "/" + url.PathEscape(operation) + "/seal",
		auth:   true, reason: reason, byBytes: true,
		body: request, patient: true,
	}, &out)
	return out, e
}

func publications(ref Ref) string {
	return "/v1/models/" + ref.Org + "/" + ref.Name + "/publications"
}

// The old whole-closure Begin/Grant/VerifyObjects/Complete API is absent.
// Publication progress is durable only as transfer rows in Tensorhub.

// ---------------------------------------------------------------- manifest reads

// ModelManifest is one resolved immutable model tree.
type ModelManifest struct {
	Org             string `json:"org"`
	Name            string `json:"name"`
	Release         string `json:"release"`
	Lane            string `json:"lane"`
	ManifestID      string `json:"manifest_id"`
	HeaderID        string `json:"header_digest"`
	Objects         int    `json:"objects"`
	Bytes           int64  `json:"bytes"`
	ReleaseEvidence []byte `json:"-"`
}

// ModelResolution is Tensorhub's exact answer to a human model ref. Download never
// lists manifests and guesses: digest or release selection happens at this route.
type ModelResolution struct {
	Model                 string `json:"model"`
	Release               string `json:"release"`
	Lane                  string `json:"lane"`
	ManifestID            string `json:"manifest_id"`
	HeaderID              string `json:"header_digest"`
	Objects               int    `json:"objects"`
	Bytes                 int64  `json:"bytes"`
	ReleaseEvidenceBase64 string `json:"release_evidence_base64"`
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

// Manifest reads the exact manifest bytes back, verbatim. They are handed straight
// to the byte plane, which admits them only if they hash to the manifest id
// asked for — so a hub that lied about a manifest cannot install one.
func (c *Client) Manifest(ctx context.Context, ref Ref, manifestID string) ([]byte, *exit.Error) {
	var raw []byte
	e := c.do(ctx, call{
		method: http.MethodGet,
		path:   "/v1/models/" + ref.Org + "/" + ref.Name + "/manifests/" + manifestID,
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
// named objects of one released manifest. Cozy never constructs a bucket URL or
// reaches storage with credentials of its own.
func (c *Client) Reads(ctx context.Context, ref Ref, manifestID string, ids []string) ([]Read, *exit.Error) {
	var out struct {
		Reads []Read `json:"reads"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost, byBytes: true,
		path: "/v1/models/" + ref.Org + "/" + ref.Name + "/manifests/" + manifestID + "/reads",
		body: map[string]any{"object_ids": ids},
	}, &out)
	if e != nil && (e.Name == "hub.untyped_refusal" || e.Name == "route.not_found") {
		return nil, exit.Named(exit.Unavailable, "hub.no_read_plane",
			"the hub at %s serves no object-read route: POST %s answered %q", c.base,
			"…/manifests/{manifest}/reads", e.Name).
			WithRemedy("this hub can take custody of bytes and cannot hand them back yet; the read grant is the missing half of th-002's transfer protocol").
			WithNext("cozy model publish <org/model> <sha256:…>", "cozy model download --dry-run "+ref.String())
	}
	return out.Reads, e
}
