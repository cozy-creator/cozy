// Package media is the OWNER's side of a rented pod's media server (cl-014, ruled #506b):
// the only byte channel there is between this host and a pod, in either direction.
//
// Before this existed there was none. `attachRemote` dialled a pod's control leg, named
// plan ids the pod had never been given, and minted DeliveryGrant access out of paths on
// the OWNER's disk — three separate ways of pretending a boundary was not there. The wire
// always had the shape for it (`InputAccess.Url` to read from, `OutputAccess.Url` to write
// to, `DeliveryGrant.Credential` to present); nothing implemented either end.
//
// WHY THE POD HOSTS IT and not this host: an owner-side byte plane would have to bind
// off-loopback, which is exactly what `internal/api`'s one-bind-site fence refuses, and it
// would have to mint a bearer for the pod to present, which the no-minted-token fence
// refuses. Putting the plane on the pod resolves both without weakening either — the pod
// already listens off-loopback for its control leg, already holds a provisioned
// credential, and is the machine the bytes have to be on anyway (#506b).
//
// This client is a plain HTTPS client pinned to the pod's certificate. It holds the
// rental's owner token because it is the one place that credential becomes an
// `Authorization` header for the media plane — the same rule `internal/hub` keeps for the
// hub's and `internal/orchestrator/owner.go` keeps for `Claim.proof`.
package media

import (
	"bytes"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

// Spec is the pod's media plane as a rental pins it: where it answers, the certificate to
// trust, and the bearer to present. It is the media half of the dial triple — same shape,
// same rules, a different port.
type Spec struct {
	Addr   string       `json:"addr"`
	Token  secret.Value `json:"token"`
	CACert string       `json:"ca_cert"`
}

// Client talks to ONE pod's media server.
type Client struct {
	spec   Spec
	http   *http.Client
	scheme string
	budget time.Duration
	// maxObject is the largest body this host will pull from the pod. A pod is a machine
	// somebody else is running; without a bound here, one that streams forever streams
	// into this process's memory. It is the OWNER's own per-output ceiling, passed in —
	// the same number the DeliveryGrant bounds the write by, so both ends of one object
	// are held to one figure.
	maxObject int64
}

// Budget is how long ONE media call may go unanswered before this host decides the pod's
// byte plane is not there. It is DERIVED, not chosen: the caller passes the same silence
// budget the pod's CONTROL leg is already judged by — the count of report periods a worker
// may miss before it is called stalled (`orchestrator.SilentReports * orchestrator.ReportCadence`). A pod
// that has not answered its byte plane in the time it owes eight reports is not slow.
//
// It is not optional, and the reason is an observation: a client with no bound at all hung
// on `connect()` against a REAL rented pod whose provider had mapped the control port and
// not the media one. Filtered is indistinguishable from slow at the socket, so the wait has
// to end on something, and the honest something is the budget the same pod is already held
// to on its other listener.

// Dial builds the client. The certificate is PINNED — trusting exactly that PEM and no CA
// is what makes a pin a pin, the same choice the control leg makes (#445). A spec with no
// certificate is a loopback harness pod and speaks plain http; a spec with one speaks TLS
// and will not fall back.
func Dial(spec Spec, budget time.Duration, maxObject int64) (*Client, *exit.Error) {
	if spec.Addr == "" {
		return nil, exit.Unavailablef("this rental pins no media address")
	}
	if !spec.Token.Present() {
		return nil, exit.New(exit.Credential,
			"this rental pins a media address and no credential to present to it")
	}
	c := &Client{spec: spec, scheme: "http"}
	transport := &http.Transport{}
	if spec.CACert != "" {
		pem, err := os.ReadFile(spec.CACert)
		if err != nil {
			return nil, exit.New(exit.NotFound,
				"the pinned media certificate %s cannot be read: %s", spec.CACert, err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, exit.New(exit.Validation,
				"%s holds no usable certificate to pin", spec.CACert)
		}
		transport.TLSClientConfig = &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12}
		c.scheme = "https"
	}
	if budget <= 0 {
		return nil, exit.Internalf(
			"the media plane was dialled with no silence budget; see media.Client's Budget note")
	}
	if maxObject <= 0 {
		return nil, exit.Internalf(
			"the media plane was dialled with no object bound; see media.Client's maxObject")
	}
	c.http = &http.Client{Transport: transport, Timeout: budget}
	c.budget, c.maxObject = budget, maxObject
	return c, nil
}

// Addr is where this client talks, for a log line that must not say the credential.
func (c *Client) Addr() string { return c.spec.Addr }

func (c *Client) url(path string) string { return c.scheme + "://" + c.spec.Addr + path }

// answer is the media server's own document. Every route answers one shape or a typed
// refusal; this side never guesses which by status code alone.
type answer struct {
	Path   string `json:"path"`
	Dir    string `json:"dir"`
	Digest string `json:"digest"`
	PlanID string `json:"plan_id"`
	Length int64  `json:"length"`
	Error  struct {
		Code    string `json:"code"`
		Message string `json:"message"`
		Remedy  string `json:"remedy"`
	} `json:"error"`
}

// call is the ONE request builder, and therefore the one place the owner token becomes an
// Authorization header for this plane.
func (c *Client) call(method, path string, body []byte) (answer, []byte, *exit.Error) {
	request, err := http.NewRequest(method, c.url(path), bytes.NewReader(body))
	if err != nil {
		return answer{}, nil, exit.Internalf("cannot build the media request: %s", err)
	}
	request.Header.Set("Authorization", "Bearer "+c.spec.Token.Reveal()) //cozy:allow-reveal the one place a rental's owner token becomes the media plane's Authorization header
	if body != nil {
		request.Header.Set("Content-Type", "application/octet-stream")
		request.ContentLength = int64(len(body))
	}
	response, err := c.http.Do(request)
	if err != nil {
		return answer{}, nil, exit.Named(exit.Unavailable, "media_unreachable",
			"the pod's media server at %s did not answer inside the %s this pod's control "+
				"leg is already judged by: %s", c.spec.Addr, c.budget, err).
			WithRemedy("a rented pod's byte plane is a CO-RESIDENT process (cl-014) on its " +
				"own port; a pod that runs a worker and no media server can be dialled and " +
				"cannot be fed, and this host will not fall back to granting paths on its " +
				"own disk that the pod cannot reach").
			WithNext("cozy rent ls")
	}
	defer response.Body.Close()
	if response.ContentLength > c.maxObject {
		return answer{}, nil, exit.Named(exit.Validation, "media_over_bound",
			"the pod declares %d B for %s and this host admits %d B per object",
			response.ContentLength, path, c.maxObject).
			WithRemedy("the DECLARED length refuses before the bytes move; the bytes meet " +
				"the same bound below")
	}
	// The bytes are held to the same figure, because a declaration is a claim: one extra
	// byte past the bound is read so the overrun is DETECTED rather than silently accepted.
	data, err := io.ReadAll(io.LimitReader(response.Body, c.maxObject+1))
	if err != nil {
		return answer{}, nil, exit.Unavailablef(
			"the pod's media server answered %d and the body ended early: %s",
			response.StatusCode, err)
	}
	if int64(len(data)) > c.maxObject {
		return answer{}, nil, exit.Named(exit.Validation, "media_over_bound",
			"the pod's answer for %s passed %d B and was cut there", path, c.maxObject).
			WithRemedy("a stream may not exceed what this host admits, and finding out " +
				"afterwards is not a bound")
	}
	if response.StatusCode >= 400 {
		var doc answer
		_ = json.Unmarshal(data, &doc)
		if doc.Error.Code == "" {
			return answer{}, nil, exit.Unavailablef(
				"the pod's media server answered %d with no typed refusal: %s",
				response.StatusCode, brief(string(data)))
		}
		e := exit.Named(codeFor(response.StatusCode), doc.Error.Code, "%s", doc.Error.Message)
		if doc.Error.Remedy != "" {
			e.WithRemedy("%s", doc.Error.Remedy)
		}
		return answer{}, nil, e
	}
	// A GET of an output is bytes, not a document. Both come back from here so there is
	// one authorized request builder and not two.
	if method == http.MethodGet && strings.Contains(path, "/outputs/") {
		return answer{Digest: response.Header.Get("X-Cozy-Digest")}, data, nil
	}
	var doc answer
	if err := json.Unmarshal(data, &doc); err != nil {
		return answer{}, nil, exit.Internalf(
			"the pod's media server answered a document this host cannot read: %s", err)
	}
	return doc, data, nil
}

// codeFor maps the media server's HTTP status onto this product's exit matrix. The pod's
// refusal keeps its own NAME; only the class is decided here.
func codeFor(status int) exit.Code {
	switch status {
	case http.StatusUnauthorized, http.StatusForbidden:
		return exit.Credential
	case http.StatusNotFound:
		return exit.NotFound
	case http.StatusBadRequest, http.StatusRequestEntityTooLarge:
		return exit.Validation
	case http.StatusConflict:
		return exit.Conflict
	default:
		return exit.Unavailable
	}
}

// Health is the pod-side liveness question, asked with the credential so that "the pod
// answers" and "the pod admits this host" are one answer instead of two.
func (c *Client) Health() *exit.Error {
	_, _, e := c.call(http.MethodGet, "/v1/health", nil)
	return e
}

// PutInput uploads one attempt input and answers the POD-LOCAL PATH it landed at. That
// path is what the owner mints into `InputAccess.Url`: the owner never guesses where the
// pod's disk is, and the pod never learns where the owner's is.
func (c *Client) PutInput(blob string, data []byte) (string, *exit.Error) {
	doc, _, e := c.call(http.MethodPut, "/v1/inputs/"+blob, data)
	if e != nil {
		return "", e
	}
	if doc.Path == "" {
		return "", exit.Internalf("the pod accepted %d B and named no path for them", len(data))
	}
	// The pod's own digest, checked against ours. It is cheap and it closes the one gap a
	// path answer leaves: an upload that landed truncated would otherwise be discovered by
	// the worker as a digest refusal on the input, one whole dispatch later.
	if want := digestOf(data); doc.Digest != "" && doc.Digest != want {
		return "", exit.New(exit.Failed,
			"the pod says the %d B it landed hash to %s and they hash to %s here",
			len(data), shortDigest(doc.Digest), shortDigest(want))
	}
	return doc.Path, nil
}

// PutPlan relays one exact canonical EntrypointBindingPlan. The pod hashes the
// whole byte string and checks the claimed subject id before keeping it; neither
// side renders a local binding record.
func (c *Client) PutPlan(planID string, record []byte) (string, *exit.Error) {
	doc, _, e := c.call(http.MethodPut,
		"/v1/plans/"+strings.TrimPrefix(planID, "sha256:"), record)
	if e != nil {
		return "", e.WithRemedy("%s", or(e.Remedy,
			"a binding plan is named by the digest of its identity; the pod recomputes it "+
				"rather than trusting the name it arrived under"))
	}
	return doc.Path, nil
}

// ReserveOutputs charges the exact sum of the attempt's OutputBinding max_bytes values,
// then creates its output directory and answers the pod-local destination. Charging first
// is what lets direct worker filesystem writes and owner HTTP uploads share one quota.
func (c *Client) ReserveOutputs(slot string, maxBytes int64) (string, *exit.Error) {
	doc, _, e := c.call(http.MethodPost,
		"/v1/outputs/"+slot+"?max_bytes="+strconv.FormatInt(maxBytes, 10), nil)
	if e != nil {
		return "", e
	}
	if doc.Dir == "" {
		return "", exit.Internalf("the pod reserved an output slot and named no directory")
	}
	return doc.Dir, nil
}

// DropAttempt removes the pod-side inputs and outputs owned by one attempt. It is
// idempotent, serving both failed-grant rollback and post-ack terminal cleanup.
func (c *Client) DropAttempt(slot string) *exit.Error {
	_, _, e := c.call(http.MethodDelete, "/v1/attempts/"+slot, nil)
	return e
}

// GetOutput fetches one committed output back to this host. It is the MIRROR's transport:
// the bytes it returns are verified against the terminal's manifest by the orchestrator
// before anything becomes visible, and this function verifies nothing itself beyond the
// pod's own declared digest — deciding that an output may be shown is the orchestrator's
// job and never a transport's.
func (c *Client) GetOutput(slot, name string) ([]byte, *exit.Error) {
	doc, data, e := c.call(http.MethodGet, "/v1/outputs/"+slot+"/"+name, nil)
	if e != nil {
		return nil, e
	}
	if want := digestOf(data); doc.Digest != "" && doc.Digest != want {
		return nil, exit.New(exit.Failed,
			"the pod declares this output as %s and the %d B that arrived hash to %s",
			shortDigest(doc.Digest), len(data), shortDigest(want))
	}
	return data, nil
}

func digestOf(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func shortDigest(d string) string {
	bare := strings.TrimPrefix(d, "sha256:")
	if len(bare) > 12 {
		return "sha256:" + bare[:12]
	}
	return d
}

func brief(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 200 {
		return s[:200] + "…"
	}
	return s
}

func or(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// Slot is the OPAQUE per-attempt name the media plane is addressed by. It is DERIVED from
// the attempt's own identity rather than minted and remembered: a orchestrator that
// restarted between dispatch and terminal must still be able to name the slot its own
// grant pointed at, and a remembered id would be exactly the state a restart loses.
func Slot(requestID string, attempt uint64) string {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s#%d", requestID, attempt)))
	return "a" + hex.EncodeToString(sum[:10])
}
