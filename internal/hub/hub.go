// Package hub is the ONE client onto tensorhub's HTTP API (cl-011). Every catalog
// read and every first-party write in this binary goes through it, and every refusal
// the hub sends reaches the user VERBATIM — its code, its message, its remedy — under
// a code from the shared exit matrix.
//
// Launch 1 has no identity plane (decisions #229): reads are public and carry no
// credential; writes carry the ONE static admin token the config authority holds.
// There is no login act, no session, no refresh and no org — th-031 arms all of that
// in Wave 2, and nothing here anticipates it.
//
// This package is also the seam cl-012 (publish/transfer) extends: `do` owns the
// request build, the credential, the reason header and the whole error mapping, so
// th-002's upload-session verbs (begin -> grants -> verify -> complete) land as
// methods beside these, never as a second client.
package hub

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync/atomic"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/secret"
)

// Timeout bounds one hub call end to end. A hub that accepts a connection and then
// says nothing is exit 10 (deadline), not exit 9 — the two failures have different
// remedies and the matrix keeps them apart.
const Timeout = 10 * time.Second

// maxBody caps a response read. The catalog answers are small documents; a hub that
// streams something enormous at us is a fault, not a listing.
const maxBody = 4 << 20

// maxDocument caps a canonical document read back verbatim (a snapshot manifest).
// It is the hub's own declaration-document ceiling: a manifest larger than the hub
// would accept cannot be one the hub installed.
const maxDocument = 64 << 20

var resourceSlug = regexp.MustCompile(`^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$`)

// A call whose work is proportional to the bytes it moves is bounded by THOSE BYTES,
// at the storage edge where they are (`transfer.mover`). There is no `Transfer`
// constant any more: a 30-minute total is a ceiling on how big a checkpoint may be,
// and it wrapped the per-object bound so tightly that the inner one could never fire.

// Client is one configured hub package. It holds no state between calls.
type Client struct {
	base   string
	token  secret.Value
	source string // where the token came from, for the remedy text
	http   *http.Client
	// slow answers calls whose work or response is bounded by bytes rather than total
	// wall time. Connection setup and headers are still bounded, and response bodies are
	// guarded below by observed byte progress.
	slow    *http.Client
	patient *http.Client // paid/proof operations: caller envelope bounds work, not a header clock
	agent   string
}

// New builds the client from the frozen config value. It reads no environment.
func New(cfg config.Config, agent string) *Client {
	return &Client{
		base:    strings.TrimRight(cfg.HubURL, "/"),
		token:   cfg.HubToken,
		source:  cfg.HubTokenSource,
		http:    &http.Client{Timeout: Timeout},
		slow:    &http.Client{Transport: slowTransport()},
		patient: &http.Client{Transport: patientTransport()},
		agent:   agent,
	}
}

func slowTransport() *http.Transport {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.DialContext = (&net.Dialer{Timeout: Timeout, KeepAlive: 30 * time.Second}).DialContext
	t.ResponseHeaderTimeout = Timeout
	return t
}

func patientTransport() *http.Transport {
	t := slowTransport()
	t.ResponseHeaderTimeout = 0
	return t
}

var errResponseStalled = errors.New("hub response body stalled")

type responseProgress struct {
	ctx    context.Context
	cancel context.CancelCauseFunc
	timer  *time.Timer
	done   atomic.Bool
}

func guardResponse(req *http.Request) (*http.Request, *responseProgress) {
	ctx, cancel := context.WithCancelCause(req.Context())
	return req.WithContext(ctx), &responseProgress{ctx: ctx, cancel: cancel}
}

func (g *responseProgress) start() {
	g.timer = time.AfterFunc(Timeout, func() { g.cancel(errResponseStalled) })
}

func (g *responseProgress) stop() {
	g.done.Store(true)
	if g.timer != nil {
		g.timer.Stop()
	}
	g.cancel(nil)
}

func (g *responseProgress) touch() {
	if !g.done.Load() && g.timer != nil {
		g.timer.Reset(Timeout)
	}
}

func (g *responseProgress) reader(r io.Reader) io.Reader {
	return &progressReader{reader: r, touch: g.touch}
}

func (g *responseProgress) stalled() bool {
	return errors.Is(context.Cause(g.ctx), errResponseStalled)
}

type progressReader struct {
	reader io.Reader
	touch  func()
}

func (r *progressReader) Read(p []byte) (int, error) {
	n, err := r.reader.Read(p)
	if n > 0 {
		r.touch()
	}
	return n, err
}

func (c *Client) Base() string        { return c.base }
func (c *Client) Token() secret.Value { return c.token }

// Resource is the shared shape of a package or model returned by its typed public
// route. The route supplies the type; the document therefore carries no `kind`
// discriminator and Cozy never guesses one from its contents.
type Resource struct {
	Org       string `json:"org"`
	Name      string `json:"name"`
	CreatedAt string `json:"created_at"`
}

func (r Resource) Ref() string { return r.Org + "/" + r.Name }

type call struct {
	method    string
	path      string
	body      any
	bodyBytes []byte // exact caller-persisted JSON; never re-marshaled on replay
	admin     bool   // carries the admin token
	reason    string // X-Tensorhub-Reason; the hub refuses a mutation without one
	// idempotency is the caller-owned operation identity for a paid mutation. It is
	// distinct from Tensorhub's provider operation id and survives a lost HTTP answer.
	idempotency string
	// byBytes drops the total wall clock for work bounded by bytes. Connection setup and
	// headers remain bounded, and an answer body must keep making byte progress.
	byBytes bool
	// patient removes the response-header clock for a server-side operation already
	// bounded by its own explicit resource/cost/duration envelope. Once headers arrive,
	// the ordinary response-body progress guard applies.
	patient bool
	// responseBytes widens the ordinary small-JSON cap for one explicitly bounded
	// response shape and moves the call onto the slow client: a large snapshot body
	// is bounded by these bytes, not by Timeout.
	responseBytes int64
	// raw takes the answer's exact bytes instead of decoding it. The snapshot
	// manifest route answers a canonical document verbatim, and this client must
	// carry it the same way — nothing here re-encodes one.
	raw *[]byte
}

// WithToken returns a copy of the client carrying a credential supplied for this
// invocation (`--token-stdin`) instead of the configured one. The value never
// reaches argv, a record, or a log line — the secret fence keeps Reveal() here.
func (c *Client) WithToken(v secret.Value, source string) *Client {
	d := *c
	d.token, d.source = v, source
	return &d
}

type ResourceSearch struct {
	Total  int    `json:"total"`
	Limit  int    `json:"limit"`
	Capped bool   `json:"capped"`
	Query  string `json:"q"`
}

// Packages searches package resources server-side. Public — no credential.
func (c *Client) Packages(ctx context.Context, query string) ([]Resource, ResourceSearch, *exit.Error) {
	var out struct {
		Packages []Resource     `json:"packages"`
		Search   ResourceSearch `json:"search"`
	}
	if e := c.do(ctx, call{method: http.MethodGet, path: resourceSearchPath("packages", query)}, &out); e != nil {
		return nil, ResourceSearch{}, e
	}
	return out.Packages, out.Search, nil
}

// Models searches model resources server-side. Public.
func (c *Client) Models(ctx context.Context, query string) ([]Resource, ResourceSearch, *exit.Error) {
	var out struct {
		Models []Resource     `json:"models"`
		Search ResourceSearch `json:"search"`
	}
	if e := c.do(ctx, call{method: http.MethodGet, path: resourceSearchPath("models", query)}, &out); e != nil {
		return nil, ResourceSearch{}, e
	}
	return out.Models, out.Search, nil
}

func resourceSearchPath(collection, query string) string {
	if query == "" {
		return "/v1/" + collection
	}
	return "/v1/" + collection + "?" + url.Values{"q": []string{query}}.Encode()
}

// Package resolves one package through its typed public route.
func (c *Client) Package(ctx context.Context, ref Ref) (Resource, *exit.Error) {
	var out struct {
		Package Resource `json:"package"`
	}
	e := c.do(ctx, call{method: http.MethodGet, path: resourcePath("packages", ref)}, &out)
	return out.Package, e
}

// Model resolves one model through its typed public route.
func (c *Client) Model(ctx context.Context, ref Ref) (Resource, *exit.Error) {
	var out struct {
		Model Resource `json:"model"`
	}
	e := c.do(ctx, call{method: http.MethodGet, path: resourcePath("models", ref)}, &out)
	return out.Model, e
}

// CreatePackage is the first-party package creation door.
func (c *Client) CreatePackage(ctx context.Context, org, name, reason string) (Resource, *exit.Error) {
	var out struct {
		Package Resource `json:"package"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost, path: "/v1/packages", admin: true, reason: reason,
		body: map[string]string{"org": org, "name": name},
	}, &out)
	return out.Package, e
}

// CreateModel is the first-party model creation door.
func (c *Client) CreateModel(ctx context.Context, org, name, reason string) (Resource, *exit.Error) {
	var out struct {
		Model Resource `json:"model"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost, path: "/v1/models", admin: true, reason: reason,
		body: map[string]string{"org": org, "name": name},
	}, &out)
	return out.Model, e
}

func resourcePath(collection string, ref Ref) string {
	return "/v1/" + collection + "/" + ref.Org + "/" + ref.Name
}

func (c *Client) do(ctx context.Context, cl call, out any) *exit.Error {
	// A missing credential is answered BEFORE the dial: a round trip cannot tell the
	// operator anything the local configuration does not already say.
	if cl.admin && !c.token.Present() {
		return exit.Named(exit.Credential, "hub.token_missing",
			"%s %s is a first-party route and no admin token is configured", cl.method, cl.path).
			WithRemedy("set TENSORHUB_TOKEN to the hub's admin.token; catalog reads need no credential").
			WithNext("cozy package search", "cozy model search")
	}

	var body io.Reader
	if cl.body != nil && cl.bodyBytes != nil {
		return exit.Internalf("hub call %s %s supplied both structured and exact request bytes", cl.method, cl.path)
	}
	if cl.bodyBytes != nil {
		body = bytes.NewReader(cl.bodyBytes)
	} else if cl.body != nil {
		b, err := json.Marshal(cl.body)
		if err != nil {
			return exit.Internalf("encoding the request body failed: %s", err)
		}
		body = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(ctx, cl.method, c.base+cl.path, body)
	if err != nil {
		return exit.Usagef("%q is not a usable hub URL: %s", c.base, err).
			WithRemedy("set TENSORHUB_URL to a base URL, e.g. https://hub.example.com")
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", c.agent)
	if cl.body != nil || cl.bodyBytes != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if cl.reason != "" {
		req.Header.Set("X-Tensorhub-Reason", cl.reason)
	}
	if cl.idempotency != "" {
		req.Header.Set("Idempotency-Key", cl.idempotency)
	}
	if cl.admin {
		// The ONE raw read of the credential in this binary. It goes into a header on
		// a request and nowhere else: not a log line, not a record, not a rendering.
		req.Header.Set("Authorization", "Bearer "+c.token.Reveal())
	}

	client := c.http
	var progress *responseProgress
	if cl.byBytes || cl.responseBytes > 0 {
		client = c.slow
		req, progress = guardResponse(req)
	}
	if cl.patient {
		client = c.patient
		if progress == nil {
			req, progress = guardResponse(req)
		}
	}
	resp, err := client.Do(req)
	if err != nil {
		if progress != nil {
			progress.stop()
		}
		return c.transport(err)
	}
	defer resp.Body.Close()
	if progress != nil {
		progress.start()
		defer progress.stop()
	}

	cap := int64(maxBody)
	if cl.raw != nil {
		cap = maxDocument
	}
	if cl.responseBytes > cap {
		cap = cl.responseBytes
	}
	responseBody := io.Reader(resp.Body)
	if progress != nil {
		responseBody = progress.reader(responseBody)
	}
	raw, err := io.ReadAll(io.LimitReader(responseBody, cap))
	if err != nil {
		if progress != nil && progress.stalled() {
			return exit.Named(exit.Deadline, "hub.response_stalled",
				"the hub at %s stopped sending its response body for %s", c.base, Timeout).
				WithRemedy("retry; if it persists the hub is up but its response stream is stalled")
		}
		return c.transport(err)
	}
	if resp.StatusCode >= 400 {
		return c.refusal(resp.StatusCode, raw)
	}
	if cl.raw != nil {
		*cl.raw = raw
		return nil
	}
	if out != nil {
		if err := json.Unmarshal(raw, out); err != nil {
			return exit.Named(exit.Internal, "hub.unreadable_answer",
				"%s %s answered %d with a body this client cannot read: %s",
				cl.method, cl.path, resp.StatusCode, err).
				WithRemedy("check that TENSORHUB_URL names a tensorhub, not a proxy or a login page").
				WithNext("cozy package search")
		}
	}
	return nil
}

// transport maps a failure that never became an HTTP answer. Unreachable is 9;
// a deadline is 10 — a hub that is up but stuck is a different problem.
func (c *Client) transport(err error) *exit.Error {
	var netErr net.Error
	if errors.Is(err, context.DeadlineExceeded) || strings.Contains(err.Error(), "Client.Timeout") ||
		(errors.As(err, &netErr) && netErr.Timeout()) {
		return exit.New(exit.Deadline, "the hub at %s did not answer within %s", c.base, Timeout).
			WithRemedy("retry; if it persists the hub is up but not serving")
	}
	return exit.Unavailablef("Tensorhub at %s is unavailable. Try again later.", c.base).
		WithRemedy("Set TENSORHUB_URL to use a different Tensorhub.")
}

// refusal renders the hub's own typed envelope under a shared-matrix code. The hub's
// code, message and remedy are passed through unchanged (cozy-creator.md: "hub
// refusals reach the user verbatim"); only the exit code is ours.
func (c *Client) refusal(status int, raw []byte) *exit.Error {
	var env struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
			Remedy  string `json:"remedy"`
		} `json:"error"`
	}
	code := statusCode(status)
	if err := json.Unmarshal(raw, &env); err != nil || env.Error.Code == "" {
		// Not every refusal is typed: an unknown route answers from net/http's own
		// mux as plain text. Say what came back rather than inventing a code.
		snippet := strings.TrimSpace(string(raw))
		if len(snippet) > 200 {
			snippet = snippet[:200] + "…"
		}
		if snippet == "" {
			snippet = "(empty body)"
		}
		return exit.Named(code, "hub.untyped_refusal",
			"the hub answered %d with no typed error envelope: %s", status, snippet).
			WithRemedy("this route may not exist on this hub build; `cozy package search` names it and its version").
			WithNext("cozy package search")
	}

	e := exit.Named(code, env.Error.Code, "%s", env.Error.Message)
	if env.Error.Remedy != "" {
		e.WithRemedy("%s", env.Error.Remedy)
	}
	if code == exit.Credential {
		// The one remedy the hub cannot write for us: it does not know where OUR
		// token came from. Launch 1 has no login, so the next step is never `cozy login`.
		e.WithNext("cozy package search")
		if c.source == "unset" {
			e.WithRemedy("set TENSORHUB_TOKEN to the hub's admin.token (%s)", e.Remedy)
		}
	}
	return e
}

// statusCode maps an HTTP status onto the ONE shared exit matrix.
func statusCode(status int) exit.Code {
	switch {
	case status == http.StatusUnauthorized, status == http.StatusForbidden:
		return exit.Credential
	case status == http.StatusNotFound:
		return exit.NotFound
	case status == http.StatusConflict:
		return exit.Conflict
	case status == http.StatusRequestTimeout, status == http.StatusGatewayTimeout:
		return exit.Deadline
	case status == http.StatusTooManyRequests:
		return exit.Unavailable
	case status >= 500:
		return exit.Unavailable
	case status >= 400:
		return exit.Validation
	}
	return exit.Internal
}

// Ref is one typed `org/name` package or model resource. Commands that accept a release
// pin split it before calling this resource parser.
type Ref struct {
	Org  string
	Name string
}

func (r Ref) String() string { return r.Org + "/" + r.Name }

func ParseRef(s string) (Ref, *exit.Error) {
	if strings.Contains(s, "@") {
		return Ref{}, exit.Usagef("%q is not a resource name", s).
			WithRemedy("name exactly org/name here; model download parses @release and @sha256 pins separately").
			WithNext("cozy package search", "cozy model search")
	}
	org, name, ok := strings.Cut(s, "/")
	if !ok || !resourceSlug.MatchString(org) || !resourceSlug.MatchString(name) {
		return Ref{}, exit.Usagef("%q is not a model or package ref: expected exactly one org/name separator", s).
			WithRemedy("org and name use lowercase letters, digits, dots, underscores, or hyphens").
			WithNext("cozy package search", "cozy model search")
	}
	return Ref{Org: org, Name: name}, nil
}

// Context bounds one hub call. Handlers never build their own.
func Context() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), Timeout)
}

// LongContext carries a whole transfer — many calls, some of them proportional to the
// bytes moved. A catalog read that takes minutes is broken; a publish that does is
// working, and one deadline cannot mean both — so this one carries NO deadline. Each
// object at the storage edge is bounded by its own byte counter, and a transfer is a
// finite list of those; the only thing left for this context to carry is the caller's
// cancel.
func LongContext() (context.Context, context.CancelFunc) {
	return context.WithCancel(context.Background())
}
