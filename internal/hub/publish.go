package hub

// Tensorhub's incremental model-upload protocol and manifest reads, as methods
// on the ONE client. `do` still owns request construction, credentials, reasons, and
// error mapping. A publication opens under a stable operation id, claims known object
// transfers, uploads through bounded grants, finalizes one owner checkpoint, then
// retains that checkpoint. Mutable release pointers are a separate call.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/url"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
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

func (c *Client) OpenPublication(ctx context.Context, ref Ref, operationID string,
	objects []Object, reason string,
) (OpenPublicationResponse, *exit.Error) {
	var out OpenPublicationResponse
	e := c.do(ctx, call{
		method: http.MethodPut,
		path:   publications(ref) + "/" + url.PathEscape(operationID), auth: true, reason: reason,
		body:    map[string]any{"objects": objects},
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
	URL      string            `json:"url"`
	Headers  map[string]string `json:"required_headers"`
	// ExpiresAtUnix is when the STORE stops honouring this signature, on the hub's clock.
	// It is only meaningful beside GrantResponse.ServerTimeUnix: the difference is the life
	// the hub signed for, and a client that has spent half of it asks for a new grant.
	ExpiresAtUnix int64 `json:"expires_at_unix"`
}

// HeldTransfer is Tensorhub's current durable transfer-row projection. Grants
// return it only when a concurrent publisher accepted an object after this
// client opened the publication.
type HeldTransfer struct {
	Transfer
	TransferID     string  `json:"transfer_id"`
	GrantExpiresAt *string `json:"grant_expires_at,omitempty"`
	RefusalCode    *string `json:"refusal_code,omitempty"`
	RefusalDetail  *string `json:"refusal_detail,omitempty"`
}

type GrantResponse struct {
	Grants []Grant        `json:"grants"`
	Held   []HeldTransfer `json:"held"`
	// ServerTimeUnix is the signer's own now. Staleness is measured against it and never
	// against the local clock, so skew between this daemon and the hub cannot misjudge
	// how much of a grant is left.
	ServerTimeUnix int64 `json:"server_time_unix"`
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

type FinalizePublicationRequest struct {
	ManifestID     string `json:"manifest_id"`
	ManifestLength int64  `json:"manifest_length"`
}

type ManifestRef struct {
	SHA256 string `json:"sha256"`
	Length int64  `json:"length"`
}

// CheckpointPublication is the durable checkpoint returned by finalize.
type CheckpointPublication struct {
	PublishID    string      `json:"publish_id"`
	CheckpointID string      `json:"checkpoint_id"`
	Manifest     ManifestRef `json:"manifest"`
	Objects      int         `json:"objects"`
	Bytes        int64       `json:"bytes"`
	State        string      `json:"state"`
	Duplicate    bool        `json:"duplicate"`
}

// RemoveCheckpoint removes one retained, unreleased checkpoint through the
// model's existing custody API. Referencing releases must be withdrawn first.
func (c *Client) RemoveCheckpoint(ctx context.Context, ref Ref, checkpointID, reason string) *exit.Error {
	var out struct {
		CheckpointID     string `json:"checkpoint_id"`
		RepositorySHA256 string `json:"repository_sha256"`
		Removed          bool   `json:"removed"`
	}
	if problem := c.do(ctx, call{method: http.MethodDelete, auth: true, reason: reason,
		path: "/v1/models/" + ref.Org + "/" + ref.Name + "/checkpoints/" + url.PathEscape(checkpointID)}, &out); problem != nil {
		return problem
	}
	if _, err := canonical.Raw(out.RepositorySHA256); err != nil || out.CheckpointID != checkpointID {
		return exit.Internalf("Tensorhub did not confirm the requested checkpoint removal")
	}
	return nil
}

// modelFinalizationView is the durable asynchronous response returned by
// Tensorhub after model finalization was moved off the request context. Result
// is intentionally raw because it is the completed CheckpointPublication
// document, while error is the hub's typed terminal refusal.
type modelFinalizationView struct {
	Operation string                    `json:"operation"`
	State     string                    `json:"state"`
	StatusURL string                    `json:"status_url"`
	Result    json.RawMessage           `json:"result,omitempty"`
	Error     *modelFinalizationFailure `json:"error,omitempty"`
}

type modelFinalizationFailure struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Remedy  string `json:"remedy,omitempty"`
}

// modelFinalizePollInterval is a variable so focused protocol tests can avoid
// waiting on the production two-second retry interval. The caller context, not
// a fixed wall clock, bounds the complete operation.
var modelFinalizePollInterval = 2 * time.Second

func decodeModelFinalization(raw []byte) (CheckpointPublication, modelFinalizationView, *exit.Error) {
	var view modelFinalizationView
	if err := json.Unmarshal(raw, &view); err != nil {
		return CheckpointPublication{}, modelFinalizationView{}, exit.Named(exit.Internal,
			"hub.unreadable_answer", "Tensorhub returned an invalid model finalization response: %s", err)
	}
	if view.Operation == "" || view.State == "" {
		return CheckpointPublication{}, view, exit.Named(exit.Structural,
			"hub.model_finalization_invalid", "Tensorhub returned incomplete model finalization state")
	}
	switch view.State {
	case "queued", "running":
		return CheckpointPublication{}, view, nil
	case "completed":
		if len(view.Result) == 0 {
			return CheckpointPublication{}, view, exit.Named(exit.Structural,
				"hub.model_finalization_invalid", "Tensorhub completed model finalization without a checkpoint result")
		}
		var checkpoint CheckpointPublication
		if err := json.Unmarshal(view.Result, &checkpoint); err != nil {
			return CheckpointPublication{}, view, exit.Named(exit.Structural,
				"hub.model_finalization_invalid", "Tensorhub returned an invalid model checkpoint result: %s", err)
		}
		return checkpoint, view, nil
	case "failed":
		if view.Error == nil || view.Error.Code == "" {
			return CheckpointPublication{}, view, exit.Named(exit.Failed,
				"upload.finalization_failed", "Tensorhub failed model finalization")
		}
		failure := exit.Named(exit.Failed, view.Error.Code, "%s", view.Error.Message)
		if view.Error.Remedy != "" {
			failure.WithRemedy("%s", view.Error.Remedy)
		}
		return CheckpointPublication{}, view, failure
	default:
		return CheckpointPublication{}, view, exit.Named(exit.Structural,
			"hub.model_finalization_invalid", "Tensorhub returned unknown model finalization state %q", view.State)
	}
}

func (c *Client) readModelFinalization(ctx context.Context, ref Ref, operation string) ([]byte, *exit.Error) {
	var raw []byte
	e := c.do(ctx, call{
		method: http.MethodGet,
		path:   publications(ref) + "/" + url.PathEscape(operation) + "/finalization",
		auth:   true, responseBytes: 1 << 20, raw: &raw,
	}, nil)
	return raw, e
}

// ModelFinalizationAnswer is Hub's exact answer to the owner's read of one publication's
// finalization: a 2xx document, or a 404 naming an absent publication or finalization.
// Any other answer settles nothing, and the caller keeps waiting.
func (c *Client) ModelFinalizationAnswer(ctx context.Context, ref Ref, operation string) (int, []byte, *exit.Error) {
	var raw []byte
	var status int
	e := c.do(ctx, call{
		method: http.MethodGet,
		path:   publications(ref) + "/" + url.PathEscape(operation) + "/finalization",
		auth:   true, responseBytes: 1 << 20, raw: &raw, status: &status,
	}, nil)
	if e == nil || status == http.StatusNotFound &&
		(e.ErrName() == "publication.not_found" || e.ErrName() == "publication.finalization_absent") {
		return status, raw, nil
	}
	return 0, nil, e
}

func (c *Client) FinalizePublication(ctx context.Context, ref Ref, operation string,
	request FinalizePublicationRequest, reason string,
) (CheckpointPublication, *exit.Error) {
	var raw []byte
	e := c.do(ctx, call{
		method: http.MethodPost,
		path:   publications(ref) + "/" + url.PathEscape(operation) + "/finalize",
		auth:   true, reason: reason, byBytes: true,
		body: request, patient: true, raw: &raw,
	}, nil)
	if e != nil {
		return CheckpointPublication{}, e
	}
	checkpoint, state, problem := decodeModelFinalization(raw)
	if problem != nil {
		return CheckpointPublication{}, problem
	}
	if state.Operation != "" && state.Operation != operation {
		return CheckpointPublication{}, exit.Named(exit.Conflict,
			"hub.model_finalization_mismatch", "Tensorhub returned model finalization for operation %q, expected %q", state.Operation, operation)
	}
	if state.State == "completed" {
		return checkpoint, nil
	}
	for {
		timer := time.NewTimer(modelFinalizePollInterval)
		select {
		case <-ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return CheckpointPublication{}, exit.New(exit.Canceled, "model finalization was canceled")
		case <-timer.C:
		}
		callCtx, cancel := context.WithTimeout(ctx, Timeout)
		raw, problem = c.readModelFinalization(callCtx, ref, operation)
		cancel()
		if transientPoll(ctx, problem) {
			continue
		}
		if problem != nil {
			return CheckpointPublication{}, problem
		}
		checkpoint, state, problem = decodeModelFinalization(raw)
		if problem != nil {
			return CheckpointPublication{}, problem
		}
		if state.Operation != "" && state.Operation != operation {
			return CheckpointPublication{}, exit.Named(exit.Conflict,
				"hub.model_finalization_mismatch", "Tensorhub returned model finalization for operation %q, expected %q", state.Operation, operation)
		}
		if state.State == "completed" {
			return checkpoint, nil
		}
	}
}

type ModelReleaseLane struct {
	Lane         string `json:"lane"`
	CheckpointID string `json:"checkpoint_id"`
	Objects      int    `json:"objects"`
	Bytes        int64  `json:"bytes"`
}

// ModelRelease is one revision of a mutable human release label. Checkpoints
// remain immutable; only these lane pointers move.
type ModelRelease struct {
	Release          string             `json:"release"`
	Revision         int64              `json:"revision"`
	Yanked           bool               `json:"yanked"`
	Lanes            []ModelReleaseLane `json:"lanes"`
	RepositorySHA256 string             `json:"repository_sha256"`
	Changed          bool               `json:"changed"`
}

func modelReleasePath(ref Ref, release string) string {
	return "/v1/models/" + ref.Org + "/" + ref.Name + "/releases/" + url.PathEscape(release)
}

func (c *Client) ModelRelease(ctx context.Context, ref Ref, release string) (ModelRelease, *exit.Error) {
	card, problem := c.ModelCard(ctx, ref)
	if problem != nil {
		return ModelRelease{}, problem
	}
	for _, summary := range card.Releases {
		if summary.Release != release {
			continue
		}
		lanes := make([]ModelReleaseLane, 0, len(summary.Lanes))
		for _, lane := range summary.Lanes {
			lanes = append(lanes, ModelReleaseLane{Lane: lane.Lane,
				CheckpointID: lane.ManifestID})
		}
		return ModelRelease{Release: summary.Release, Revision: summary.Revision,
			Lanes: lanes}, nil
	}
	return ModelRelease{}, exit.New(exit.NotFound, "model release %s@%s is absent",
		ref.String(), release)
}

func (c *Client) UpdateModelRelease(ctx context.Context, ref Ref, release string,
	expectedRevision int64, setLanes map[string]string, removeLanes []string, reason string,
) (ModelRelease, *exit.Error) {
	if setLanes == nil {
		setLanes = map[string]string{}
	}
	if removeLanes == nil {
		removeLanes = []string{}
	}
	var out ModelRelease
	e := c.do(ctx, call{method: http.MethodPost, path: modelReleasePath(ref, release),
		auth: true, reason: reason, patient: true,
		body: map[string]any{"expected_revision": expectedRevision,
			"set_lanes": setLanes, "remove_lanes": removeLanes}}, &out)
	return out, e
}

// RetargetModelLane moves ONE existing release lane pointer to another
// checkpoint the model already retains. The hub creates nothing on this route:
// an unknown release, lane, or checkpoint and a yanked release refuse typed.
func (c *Client) RetargetModelLane(ctx context.Context, ref Ref, release, lane,
	checkpointID, reason string,
) (ModelRelease, *exit.Error) {
	var out ModelRelease
	e := c.do(ctx, call{method: http.MethodPut,
		path: modelReleasePath(ref, release) + "/lanes/" + url.PathEscape(lane),
		auth: true, reason: reason, patient: true,
		body: map[string]any{"checkpoint_id": checkpointID}}, &out)
	return out, e
}

// ModelDeletion is Tensorhub's account of one deleted model. Reclaimable counts
// the distinct objects nothing else references; the Hub's next GC deletes them.
type ModelDeletion struct {
	Model              string `json:"model"`
	Checkpoints        int    `json:"checkpoints"`
	ReclaimableObjects int    `json:"reclaimable_objects"`
	ReclaimableBytes   int64  `json:"reclaimable_bytes"`
}

// DeleteModel deletes one model whose releases are all yanked.
func (c *Client) DeleteModel(ctx context.Context, ref Ref, reason string) (ModelDeletion, *exit.Error) {
	var out ModelDeletion
	if e := c.do(ctx, call{method: http.MethodDelete, path: "/v1/models/" + ref.Org + "/" + ref.Name,
		auth: true, reason: reason, patient: true}, &out); e != nil {
		return out, e
	}
	if out.Model != ref.String() {
		return out, exit.Internalf("Tensorhub did not confirm deleting %s", ref)
	}
	return out, nil
}

func (c *Client) YankModelRelease(ctx context.Context, ref Ref, release, reason string) (ModelRelease, *exit.Error) {
	var out ModelRelease
	e := c.do(ctx, call{method: http.MethodDelete, path: modelReleasePath(ref, release),
		auth: true, reason: reason, patient: true}, &out)
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
	Org        string   `json:"org"`
	Name       string   `json:"name"`
	Release    string   `json:"release"`
	Lane       string   `json:"lane"`
	ManifestID string   `json:"manifest_id"`
	HeaderID   string   `json:"header_digest"`
	Components []string `json:"components"`
	Objects    int      `json:"objects"`
	Bytes      int64    `json:"bytes"`
}

// ModelResolution is Tensorhub's exact answer to a human model ref. Download never
// lists manifests and guesses: digest or release selection happens at this route.
type ModelResolution struct {
	Model          string           `json:"model"`
	Release        string           `json:"release"`
	Lane           string           `json:"lane"`
	ManifestID     string           `json:"manifest_id"`
	ManifestLength int64            `json:"manifest_length"`
	ComponentBytes map[string]int64 `json:"component_bytes,omitempty"`
	HeaderID       string           `json:"header_digest"`
	Components     []string         `json:"components"`
	Objects        int              `json:"objects"`
	Bytes          int64            `json:"bytes"`
}

func (c *Client) ResolveModel(ctx context.Context, spec, lane string) (ModelResolution, *exit.Error) {
	query := url.Values{"ref": []string{spec}}
	if lane != "" {
		query.Set("lane", lane)
	}
	var out ModelResolution
	// An unpublished checkpoint resolves only for its owner, so a signed-in reader presents
	// its bearer; a signed-out one reads published releases.
	e := c.do(ctx, call{method: http.MethodGet, optionalAuth: true,
		path: "/v1/models/resolve?" + query.Encode()}, &out)
	return out, e
}

// Manifest reads the exact manifest bytes back, verbatim. They are handed straight
// to the byte plane, which admits them only if they hash to the manifest id
// asked for — so a hub that lied about a manifest cannot install one.
func (c *Client) ReleaseManifest(ctx context.Context, ref Ref, release, lane string) ([]byte, *exit.Error) {
	var raw []byte
	e := c.do(ctx, call{
		method: http.MethodGet,
		path: "/v1/models/" + ref.Org + "/" + ref.Name + "/releases/" +
			url.PathEscape(release) + "/lanes/" + url.PathEscape(lane) + "/manifest",
		raw: &raw, byBytes: true,
	}, nil)
	return raw, e
}

func (c *Client) CheckpointManifest(ctx context.Context, ref Ref, checkpointID string) ([]byte, *exit.Error) {
	var raw []byte
	e := c.do(ctx, call{
		method: http.MethodGet, optionalAuth: true,
		path: "/v1/models/" + ref.Org + "/" + ref.Name + "/checkpoints/" + checkpointID,
		raw:  &raw, byBytes: true,
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
func (c *Client) ReleaseReads(ctx context.Context, ref Ref, release, lane string, ids []string) ([]Read, *exit.Error) {
	var out struct {
		Reads []Read `json:"reads"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost, byBytes: true,
		path: "/v1/models/" + ref.Org + "/" + ref.Name + "/releases/" + url.PathEscape(release) +
			"/lanes/" + url.PathEscape(lane) + "/reads",
		body: map[string]any{"object_ids": ids},
	}, &out)
	return out.Reads, e
}

func (c *Client) CheckpointReads(ctx context.Context, ref Ref, checkpointID string, ids []string) ([]Read, *exit.Error) {
	var out struct {
		Reads []Read `json:"reads"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost, byBytes: true, optionalAuth: true,
		path: "/v1/models/" + ref.Org + "/" + ref.Name + "/checkpoints/" + checkpointID + "/reads",
		body: map[string]any{"object_ids": ids},
	}, &out)
	return out.Reads, e
}

// transientPoll: a status read that failed in transit while the caller still waits. The
// finalization is durable on the Hub, so the next poll asks again instead of abandoning it.
func transientPoll(ctx context.Context, problem *exit.Error) bool {
	return problem != nil && ctx.Err() == nil &&
		(problem.Code == exit.Unavailable || problem.Code == exit.Deadline || problem.Code == exit.Capacity)
}
