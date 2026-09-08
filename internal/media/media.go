// Package media is the OWNER's side of a rented pod's media server (cl-014, ruled #506b):
// the only byte channel there is between this host and a pod, in either direction.
//
// This plane carries per-invocation inputs and outputs only. Package source, plans,
// wheels, and model-object sets are resolved directly by the worker from signed intent;
// there is no package-distribution route on this client or the pod server.
//
// WHY THE POD HOSTS IT and not this host: an owner-side byte plane would have to bind
// off-loopback, which is exactly what `internal/api`'s one-bind-site fence refuses, and it
// would have to mint a bearer for the pod to present, which the no-minted-token fence
// refuses. Putting the plane on the pod resolves both without weakening either — the pod
// already listens off-loopback for its control leg, already holds a provisioned
// credential, and is the machine the bytes have to be on anyway (#506b).
//
// This client is a plain HTTPS client pinned to the pod's certificate. It holds the
// rental's media bearer because it is the one place that credential becomes an
// `Authorization` header for the media plane — the same rule `internal/hub` keeps for the
// hub's; this bearer has no WorkerControl or Tensorhub authority.
package media

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/mediawire"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/workertls"
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

// Budget is the I/O stall allowance: how long one call may go with no byte moving in either
// direction. A transfer that keeps moving bytes, however slowly, is never failed by it:
// total wall time is not a bandwidth claim, and this bound cannot settle or retire a worker.
//
// It is not optional, and the reason is an observation: a client with no bound at all hung
// on `connect()` against a REAL rented pod whose provider had mapped the control port and
// not the media one. Filtered is indistinguishable from slow at the socket, so the wait has
// to end on something, and the honest something is the budget the same pod is already held
// to on its other listener.

// Dial builds the client. The certificate is PINNED — the leaf the pod presents must be
// byte-for-byte the pinned PEM, the same rule the control leg keeps (#445). A spec with no
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
		pin, err := workertls.LoadPin(spec.CACert)
		if err != nil {
			return nil, exit.New(exit.Validation, "pinned media certificate: %s", err)
		}
		transport.TLSClientConfig = pin.TLSConfig()
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

// stall bounds one call by PROGRESS, not by a clock over the whole transfer: its context
// ends only after `budget` passes with no byte moved in either direction. Connect and
// headers are under the same budget, because nothing moves during them.
type stall struct {
	ctx    context.Context
	cancel context.CancelFunc
	timer  *time.Timer
	budget time.Duration
	done   atomic.Bool
}

var errStalled = fmt.Errorf("stalled")

func (c *Client) stall(request *http.Request) (*http.Request, *stall) {
	ctx, cancel := context.WithCancelCause(request.Context())
	g := &stall{ctx: ctx, budget: c.budget}
	g.timer = time.AfterFunc(c.budget, func() { cancel(errStalled) })
	g.cancel = func() { g.done.Store(true); g.timer.Stop(); cancel(nil) }
	if request.Body != nil && request.Body != http.NoBody {
		request.Body = g.reader(request.Body)
	}
	return request.WithContext(ctx), g
}

func (g *stall) touch() {
	if !g.done.Load() {
		g.timer.Reset(g.budget)
	}
}

// reader reports every chunk that moves through it as progress.
func (g *stall) reader(r io.ReadCloser) io.ReadCloser {
	return &progress{ReadCloser: r, touch: g.touch}
}

// why names the stall as such when that is what ended the call.
func (g *stall) why(err error) error {
	if context.Cause(g.ctx) == errStalled {
		return fmt.Errorf("no byte moved for %s (the control-leg silence budget)", g.budget)
	}
	return err
}

type progress struct {
	io.ReadCloser
	touch func()
}

func (p *progress) Read(b []byte) (int, error) {
	n, err := p.ReadCloser.Read(b)
	if n > 0 {
		p.touch()
	}
	return n, err
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

// call is the ONE request builder, and therefore the one place the media bearer becomes an
// Authorization header for this plane.
func (c *Client) call(method, path string, body []byte) (answer, []byte, *exit.Error) {
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	request, err := http.NewRequest(method, c.url(path), reader)
	if err != nil {
		return answer{}, nil, exit.Internalf("cannot build the media request: %s", err)
	}
	request.Header.Set("Authorization", "Bearer "+c.spec.Token.Reveal()) //cozy:allow-reveal the rental media bearer becomes an Authorization header only here
	if body != nil {
		request.Header.Set("Content-Type", "application/octet-stream")
		request.ContentLength = int64(len(body))
	}
	request, guard := c.stall(request)
	defer guard.cancel()
	response, err := c.http.Do(request)
	if err != nil {
		return answer{}, nil, exit.Named(exit.Unavailable, "media_unreachable",
			"the pod's media server at %s did not answer %s %s: %s",
			c.spec.Addr, method, path, guard.why(err)).
			WithRemedy("a rented pod's byte plane is a CO-RESIDENT process (cl-014) on its " +
				"own port; a pod that runs a worker and no media server can be dialled and " +
				"cannot be fed, and this host will not fall back to granting paths on its " +
				"own disk that the pod cannot reach").
			WithNext("cozy rental")
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
	data, err := io.ReadAll(io.LimitReader(guard.reader(response.Body), maxAnswerBytes+1))
	if err != nil {
		return answer{}, nil, exit.Unavailablef(
			"the pod's media server answered %d and the body ended early: %s",
			response.StatusCode, guard.why(err))
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
// answers" and "the pod admits this host" are one answer instead of two — and it is the
// media plane's ONE version handshake, which is why it runs before any byte moves.
//
// The two ends of this plane are released separately: Tensorhub compiles `pod-supervisor`
// into its base worker image while this client floats with Cozy master, so a route or a
// field can move on one side alone. The revision closes that: a plane at another revision,
// or one too old to declare a revision at all, is refused here rather than fed bytes whose
// answer shape this host would misread. This media contract has its own explicit revision;
// it is independent of the protobuf worker protocol.
func (c *Client) Health() *exit.Error {
	_, data, e := c.call(http.MethodGet, "/v1/health", nil)
	if e != nil {
		return e
	}
	var said mediawire.Health
	if err := json.Unmarshal(data, &said); err != nil {
		return c.skew("answered a health document this host cannot read: %s", err)
	}
	if said.Service != mediawire.Service {
		return c.skew("calls itself %q and this host dials %q",
			brief(said.Service), mediawire.Service)
	}
	if said.ContractRev == nil {
		return c.skew("declares NO media contract revision and this host speaks rev %d",
			mediawire.ContractRev)
	}
	if *said.ContractRev != mediawire.ContractRev {
		return c.skew("speaks media contract rev %d and this host speaks rev %d",
			*said.ContractRev, mediawire.ContractRev)
	}
	if !said.AttemptScopedInputs {
		return exit.Named(exit.Conflict, "media_input_scope_unsupported",
			"the pod's media plane at %s does not support attempt-scoped inputs", c.spec.Addr).
			WithRemedy("update the pod supervisor before attaching this rental; inputs require an existing attempt reservation")
	}
	return nil
}

// skew is the one refusal shape for a byte plane this host cannot trust the answers of.
func (c *Client) skew(format string, args ...any) *exit.Error {
	return exit.Named(exit.Conflict, "media_contract_mismatch",
		"the pod's media plane at %s "+format, append([]any{c.spec.Addr}, args...)...).
		WithRemedy("the pod's media server is built into its image from a pinned "+
			"Tensorhub recipe commit while this host floats with Cozy master, so the "+
			"two ends can differ. Rebuild "+
			"the pod image from a commit that speaks rev %d, or run an owner that speaks "+
			"what the pod does. Nothing is uploaded to a plane whose answers this host "+
			"cannot read: a misparsed field is worse than a refused rental.",
			mediawire.ContractRev).
		WithNext("cozy rental")
}

// PutInput uploads one attempt input and answers the POD-LOCAL PATH it landed at. That
// path is what the owner mints into `InputAccess.Url`: the owner never guesses where the
// pod's disk is, and the pod never learns where the owner's is.
func (c *Client) PutInput(slot, inputID string, data []byte) (string, *exit.Error) {
	return c.putInput(slot, inputID, bytes.NewReader(data), digestOf(data), int64(len(data)))
}

// PutInputFile streams one exact verified file to the pod. It borrows the original
// path; neither this client nor attempt cleanup moves or removes that file.
func (c *Client) PutInputFile(slot, inputID, path, wantDigest string, wantLength int64) (string, *exit.Error) {
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
	return c.putInput(slot, inputID, file, wantDigest, wantLength)
}

// putInput is shared by payload bytes and borrowed files. The receiver records
// this exact input against the reserved attempt before returning its cache path.
func (c *Client) putInput(slot, inputID string, body io.Reader, wantDigest string, wantLength int64) (string, *exit.Error) {
	if slot == "" || inputID == "" {
		return "", exit.New(exit.Validation, "media input requires an attempt slot and input ID")
	}
	hash := sha256.New()
	request, err := http.NewRequest(http.MethodPut,
		c.url("/v1/attempts/"+url.PathEscape(slot)+"/inputs/"+url.PathEscape(inputID)),
		io.TeeReader(body, hash))
	if err != nil {
		return "", exit.Internalf("cannot build the media upload: %s", err)
	}
	request.Header.Set("Authorization", "Bearer "+c.spec.Token.Reveal()) //cozy:allow-reveal
	request.Header.Set("Content-Type", "application/octet-stream")
	request.ContentLength = wantLength
	if wantLength == 0 {
		request.Body = http.NoBody
	}
	request, guard := c.stall(request)
	defer guard.cancel()
	response, err := c.http.Do(request)
	if err != nil {
		return "", exit.Named(exit.Unavailable, "media_unreachable",
			"the pod's media server at %s did not accept input %s: %s",
			c.spec.Addr, inputID, guard.why(err))
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(guard.reader(response.Body), 1<<20+1))
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
	if gotDigest != wantDigest || doc.Length != wantLength || doc.Digest != wantDigest {
		return "", exit.Named(exit.Conflict, "input_asset_changed",
			"input %s did not retain its recorded digest/length during upload", inputID)
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

// ReserveOutputs reserves the total output byte bound and exact output count
// before the worker can write. The receiver uses the count to protect the inodes
// still owed by this attempt, including when the declared count is zero.
func (c *Client) ReserveOutputs(slot string, maxBytes int64, outputCount int) (string, *exit.Error) {
	if maxBytes < 0 || outputCount < 0 {
		return "", exit.New(exit.Validation, "output reservation requires non-negative bytes and file count")
	}
	doc, _, e := c.call(http.MethodPost,
		"/v1/outputs/"+url.PathEscape(slot)+"?max_bytes="+strconv.FormatInt(maxBytes, 10)+
			"&output_count="+strconv.Itoa(outputCount), nil)
	if e != nil {
		return "", e
	}
	if doc.Dir == "" {
		return "", exit.Internalf("the pod reserved an output slot and named no directory")
	}
	return doc.Dir, nil
}

// DropAttempt releases one attempt through the receiver's existing lifecycle.
// Shared cached input bytes remain available to other attempts. The call is
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
	request, guard := c.stall(request)
	defer guard.cancel()
	response, err := c.http.Do(request)
	if err != nil {
		return 0, exit.Named(exit.Unavailable, "media_unreachable",
			"the pod's media server at %s could not stream output %s: %s",
			c.spec.Addr, name, guard.why(err))
	}
	defer response.Body.Close()
	body := guard.reader(response.Body)
	if response.StatusCode >= 400 {
		data, _ := io.ReadAll(io.LimitReader(body, 1<<20+1))
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
	written, err := io.Copy(io.MultiWriter(staging, hash), io.LimitReader(body, wantLength+1))
	if err != nil {
		return written, exit.Unavailablef("output %s ended while it was mirrored: %s",
			name, guard.why(err))
	}
	if written != wantLength {
		return written, exit.Named(exit.Failed, "media_length_mismatch",
			"output %s delivered %d B; its terminal declares %d B", name, written, wantLength)
	}
	gotDigest := "sha256:" + hex.EncodeToString(hash.Sum(nil))
	if gotDigest != wantDigest {
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

// MaxTriageBundle is the protocol's bound on one triage bundle (`TriageBundleRef.length`
// <= 1 MiB, worker-protocol/01). The pod's plane refuses to serve past it and this client
// refuses to read past it, so the terminal's declared length is checked against a figure
// both ends already hold.
const MaxTriageBundle = 1 << 20

// GetTriage reads one attempt's triage bundle by the OPAQUE subject its terminal named,
// and proves the bytes are the ones the terminal claimed before handing them back. Like
// GetOutputTo this is a transport check of the pod's own declaration; whether the bundle
// is KEPT is the orchestrator's decision against the terminal it accepted.
func (c *Client) GetTriage(subject, wantDigest string, wantLength int64) ([]byte, *exit.Error) {
	if wantLength <= 0 || wantLength > MaxTriageBundle {
		return nil, exit.Named(exit.Validation, "triage_length_invalid",
			"the terminal declares triage bundle %s as %d B; the protocol bounds one at %d B",
			subject, wantLength, MaxTriageBundle)
	}
	request, err := http.NewRequest(http.MethodGet, c.url("/v1/triage/"+subject), nil)
	if err != nil {
		return nil, exit.Internalf("cannot build the media triage request: %s", err)
	}
	request.Header.Set("Authorization", "Bearer "+c.spec.Token.Reveal()) //cozy:allow-reveal
	request, guard := c.stall(request)
	defer guard.cancel()
	response, err := c.http.Do(request)
	if err != nil {
		return nil, exit.Named(exit.Unavailable, "media_unreachable",
			"the pod's media server at %s could not serve triage bundle %s: %s",
			c.spec.Addr, subject, guard.why(err))
	}
	defer response.Body.Close()
	body := guard.reader(response.Body)
	if response.StatusCode >= 400 {
		data, _ := io.ReadAll(io.LimitReader(body, 1<<20+1))
		return nil, mediaRefusal(response.StatusCode, data)
	}
	if response.ContentLength >= 0 && response.ContentLength != wantLength {
		return nil, exit.Named(exit.Failed, "media_length_mismatch",
			"the pod declares triage bundle %s as %d B; its terminal declares %d B",
			subject, response.ContentLength, wantLength)
	}
	data, err := io.ReadAll(io.LimitReader(body, wantLength+1))
	if err != nil {
		return nil, exit.Unavailablef("triage bundle %s ended while it was read: %s",
			subject, guard.why(err))
	}
	if int64(len(data)) != wantLength {
		return nil, exit.Named(exit.Failed, "media_length_mismatch",
			"triage bundle %s delivered %d B; its terminal declares %d B",
			subject, len(data), wantLength)
	}
	if got := digestOf(data); got != wantDigest {
		return nil, exit.Named(exit.Failed, "media_digest_mismatch",
			"triage bundle %s delivered %s; its terminal declares %s", subject, got, wantDigest)
	}
	return data, nil
}

func digestOf(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func brief(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 200 {
		return s[:200] + "…"
	}
	return s
}

// Slot is the OPAQUE per-attempt name the media plane is addressed by. It is DERIVED from
// the attempt's own identity rather than minted and remembered: a orchestrator that
// restarted between dispatch and terminal must still be able to name the slot its own
// grant pointed at, and a remembered id would be exactly the state a restart loses.
func Slot(requestID string, attempt uint64) string {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s#%d", requestID, attempt)))
	return "a" + hex.EncodeToString(sum[:10])
}
