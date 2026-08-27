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
	"context"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
	"github.com/cozy-creator/cozy-creator-v2/internal/workertls"
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

// Budget is the connection/response allowance before declared transfer bytes are counted.
// It is DERIVED, not chosen: the caller passes the same silence
// budget the pod's CONTROL leg is already judged by — the count of report periods a worker
// may miss before it is called stalled (`orchestrator.SilentReports * orchestrator.ReportCadence`). A pod
// A call's total deadline adds its exact byte length at the explicit 1 MiB/s floor below;
// large valid media therefore does not inherit a hidden high-bandwidth requirement.
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
		transport.TLSClientConfig = &tls.Config{
			RootCAs: pool, MinVersion: tls.VersionTLS12, ServerName: workertls.ServerName,
		}
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
	c.http = &http.Client{Transport: transport}
	c.budget, c.maxObject = budget, maxObject
	return c, nil
}

// Addr is where this client talks, for a log line that must not say the credential.
func (c *Client) Addr() string { return c.spec.Addr }

func (c *Client) url(path string) string { return c.scheme + "://" + c.spec.Addr + path }

const minimumTransferBytesPerSecond = int64(1 << 20)

// transferDeadline keeps every call bounded without making total wall time a hidden
// bandwidth requirement. The control-plane silence budget pays for connection/response
// latency; declared bytes then receive time at a conservative 1 MiB/s minimum rate.
func (c *Client) transferDeadline(length int64) time.Duration {
	if length < minimumTransferBytesPerSecond {
		length = minimumTransferBytesPerSecond
	}
	seconds := length / minimumTransferBytesPerSecond
	remainder := length % minimumTransferBytesPerSecond
	const maximum = time.Duration(1<<63 - 1)
	if c.budget >= maximum || seconds > int64((maximum-c.budget)/time.Second) {
		return maximum
	}
	transfer := time.Duration(seconds) * time.Second
	fraction := time.Duration(remainder) * time.Second / time.Duration(minimumTransferBytesPerSecond)
	if transfer > maximum-c.budget-fraction {
		return maximum
	}
	transfer += fraction
	return c.budget + transfer
}

func (c *Client) bounded(request *http.Request, length int64) (*http.Request, context.CancelFunc) {
	ctx, cancel := context.WithTimeout(request.Context(), c.transferDeadline(length))
	return request.WithContext(ctx), cancel
}

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
	request, cancel := c.bounded(request, int64(len(body)))
	defer cancel()
	response, err := c.http.Do(request)
	if err != nil {
		return answer{}, nil, exit.Named(exit.Unavailable, "media_unreachable",
			"the pod's media server at %s did not complete inside the %s derived from its "+
				"declared bytes and the control-leg silence budget: %s",
			c.spec.Addr, c.transferDeadline(int64(len(body))), err).
			WithRemedy("a rented pod's byte plane is a CO-RESIDENT process (cl-014) on its " +
				"own port; a pod that runs a worker and no media server can be dialled and " +
				"cannot be fed, and this host will not fall back to granting paths on its " +
				"own disk that the pod cannot reach").
			WithNext("cozy rent ls")
	}
	defer response.Body.Close()
	const maxAnswerBytes = int64(1 << 20)
	if response.ContentLength > maxAnswerBytes {
		return answer{}, nil, exit.Named(exit.Validation, "media_over_bound",
			"the pod declares %d B for the %s answer and this host admits %d B",
			response.ContentLength, path, maxAnswerBytes).
			WithRemedy("the DECLARED length refuses before the bytes move; the bytes meet " +
				"the same bound below")
	}
	// The bytes are held to the same figure, because a declaration is a claim: one extra
	// byte past the bound is read so the overrun is DETECTED rather than silently accepted.
	data, err := io.ReadAll(io.LimitReader(response.Body, maxAnswerBytes+1))
	if err != nil {
		return answer{}, nil, exit.Unavailablef(
			"the pod's media server answered %d and the body ended early: %s",
			response.StatusCode, err)
	}
	if int64(len(data)) > maxAnswerBytes {
		return answer{}, nil, exit.Named(exit.Validation, "media_over_bound",
			"the pod's answer for %s passed %d B and was cut there", path, maxAnswerBytes).
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

// PutInputFile streams one exact verified file to the pod. The digest and length are the
// request record's facts; neither side needs a whole-file byte slice.
func (c *Client) PutInputFile(blob, path, wantDigest string, wantLength int64) (string, *exit.Error) {
	file, err := os.Open(path)
	if err != nil {
		return "", exit.New(exit.NotFound, "input asset %s: %s", filepath.Base(path), err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() != wantLength {
		return "", exit.Named(exit.Conflict, "input_asset_changed",
			"input asset %s no longer has its recorded %d-byte length", filepath.Base(path), wantLength)
	}
	hash := sha256.New()
	request, err := http.NewRequest(http.MethodPut, c.url("/v1/inputs/"+blob),
		io.TeeReader(file, hash))
	if err != nil {
		return "", exit.Internalf("cannot build the media upload: %s", err)
	}
	request.Header.Set("Authorization", "Bearer "+c.spec.Token.Reveal()) //cozy:allow-reveal
	request.Header.Set("Content-Type", "application/octet-stream")
	request.ContentLength = wantLength
	request, cancel := c.bounded(request, wantLength)
	defer cancel()
	response, err := c.http.Do(request)
	if err != nil {
		return "", exit.Named(exit.Unavailable, "media_unreachable",
			"the pod's media server at %s did not accept %s: %s", c.spec.Addr, filepath.Base(path), err)
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, 1<<20+1))
	if err != nil || len(data) > 1<<20 {
		return "", exit.Unavailablef("the pod's media upload answer is unreadable or oversized")
	}
	if response.StatusCode >= 400 {
		return "", mediaRefusal(response.StatusCode, data)
	}
	var doc answer
	if err := json.Unmarshal(data, &doc); err != nil || doc.Path == "" {
		return "", exit.Internalf("the pod accepted an input and returned no readable path")
	}
	gotDigest := "sha256:" + hex.EncodeToString(hash.Sum(nil))
	if gotDigest != wantDigest || doc.Length != wantLength ||
		(doc.Digest != "" && doc.Digest != wantDigest) {
		return "", exit.Named(exit.Conflict, "input_asset_changed",
			"input asset %s did not retain its recorded digest/length during upload", filepath.Base(path))
	}
	return doc.Path, nil
}

func mediaRefusal(status int, data []byte) *exit.Error {
	var doc answer
	_ = json.Unmarshal(data, &doc)
	if doc.Error.Code == "" {
		return exit.Unavailablef("the pod's media server answered %d with no typed refusal", status)
	}
	problem := exit.Named(codeFor(status), doc.Error.Code, "%s", doc.Error.Message)
	if doc.Error.Remedy != "" {
		problem.WithRemedy("%s", doc.Error.Remedy)
	}
	return problem
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

// GetOutputTo mirrors one exact output directly into an atomic local file while hashing
// and counting the received stream.
func (c *Client) GetOutputTo(slot, name, destination, wantDigest string,
	wantLength int64) (int64, *exit.Error) {
	request, err := http.NewRequest(http.MethodGet, c.url("/v1/outputs/"+slot+"/"+name), nil)
	if err != nil {
		return 0, exit.Internalf("cannot build the media output request: %s", err)
	}
	request.Header.Set("Authorization", "Bearer "+c.spec.Token.Reveal()) //cozy:allow-reveal
	request, cancel := c.bounded(request, wantLength)
	defer cancel()
	response, err := c.http.Do(request)
	if err != nil {
		return 0, exit.Named(exit.Unavailable, "media_unreachable",
			"the pod's media server at %s could not stream output %s: %s", c.spec.Addr, name, err)
	}
	defer response.Body.Close()
	if response.StatusCode >= 400 {
		data, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20+1))
		return 0, mediaRefusal(response.StatusCode, data)
	}
	if response.ContentLength >= 0 && response.ContentLength != wantLength {
		return 0, exit.Named(exit.Failed, "media_length_mismatch",
			"the pod declares output %s as %d B; its terminal declares %d B",
			name, response.ContentLength, wantLength)
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		return 0, exit.Internalf("cannot create output mirror directory: %s", err)
	}
	staging, err := os.CreateTemp(filepath.Dir(destination), "."+filepath.Base(destination)+".mirroring-*")
	if err != nil {
		return 0, exit.Internalf("cannot stage mirrored output %s: %s", name, err)
	}
	stagingPath := staging.Name()
	keep := false
	defer func() {
		_ = staging.Close()
		if !keep {
			_ = os.Remove(stagingPath)
		}
	}()
	hash := sha256.New()
	written, err := io.Copy(io.MultiWriter(staging, hash), io.LimitReader(response.Body, wantLength+1))
	if err != nil {
		return written, exit.Unavailablef("output %s ended while it was mirrored: %s", name, err)
	}
	if written != wantLength {
		return written, exit.Named(exit.Failed, "media_length_mismatch",
			"output %s delivered %d B; its terminal declares %d B", name, written, wantLength)
	}
	gotDigest := "sha256:" + hex.EncodeToString(hash.Sum(nil))
	headerDigest := response.Header.Get("X-Cozy-Digest")
	if gotDigest != wantDigest || (headerDigest != "" && headerDigest != wantDigest) {
		return written, exit.Named(exit.Failed, "media_digest_mismatch",
			"output %s delivered %s; its terminal declares %s", name, gotDigest, wantDigest)
	}
	if err := staging.Sync(); err != nil {
		return written, exit.Internalf("cannot sync mirrored output %s: %s", name, err)
	}
	if err := staging.Close(); err != nil {
		return written, exit.Internalf("cannot close mirrored output %s: %s", name, err)
	}
	if err := os.Rename(stagingPath, destination); err != nil {
		return written, exit.Internalf("cannot commit mirrored output %s: %s", name, err)
	}
	keep = true
	return written, nil
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
