// Package hub is the ONE client onto tensorhub's HTTP API (cl-011). Every catalog
// read and every first-party write in this binary goes through it, and every refusal
// the hub sends reaches the user VERBATIM — its code, its message, its remedy — under
// a code from the shared exit matrix.
//
// Catalog reads are public. Authenticated calls use either the operator's configured
// token or Creator's machine-key TokenSource; the resulting short bearer exists only
// in memory and this package remains its one HTTP carrier.
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
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/secret"
)

// maxBody caps a response read. The catalog answers are small documents; a hub that
// streams something enormous at us is a fault, not a listing.
const maxBody = 4 << 20

// maxDocument caps a canonical document read back verbatim (a snapshot manifest).
// It is the hub's own declaration-document ceiling: a manifest larger than the hub
// would accept cannot be one the hub installed.
const maxDocument = 64 << 20

var resourceSlug = regexp.MustCompile(`^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$`)

// CanonicalName is the one spelling of an account, org or resource name. Names are
// never case-sensitive: `Paul/SDXL` is `paul/sdxl`.
func CanonicalName(name string) string { return strings.ToLower(strings.TrimSpace(name)) }

// A call whose work is proportional to the bytes it moves is bounded by THOSE BYTES,
// at the storage edge where they are (`transfer.mover`). There is no `Transfer`
// constant any more: a 30-minute total is a ceiling on how big a checkpoint may be,
// and it wrapped the per-object bound so tightly that the inner one could never fire.

// Client is one configured hub package. It holds no state between calls.
type Client struct {
	base   string
	named  string // the hub as a person knows it: "local (http://127.0.0.1:8819)"
	label  string // its --tensorhub spelling
	token  secret.Value
	source string // where the token came from, for the remedy text
	http   *http.Client
	agent  string
	tokens TokenSource
	// waiting hears "waiting on Tensorhub" while a call has no answer; nil hears nothing.
	waiting io.Writer
	// releases keeps committed release documents (release_cache.go); empty keeps none.
	releases string
}

// TokenSource silently turns a persisted machine key into a short AuthKit bearer.
// The interface lives here so this HTTP package does not own account-key storage.
type TokenSource interface {
	AccessToken(context.Context) (secret.Value, *exit.Error)
}

// New builds the client from the frozen config value. It reads no environment.
func New(cfg config.Config, agent string) *Client {
	return &Client{
		base:   strings.TrimRight(cfg.HubURL, "/"),
		named:  cfg.HubText(cfg.HubURL),
		label:  cfg.HubLabel(cfg.HubURL),
		token:  cfg.HubToken,
		source: cfg.HubTokenSource,
		http:   HTTP(cfg.HubLiveness),
		agent:  agent,
		releases: func() string {
			if cfg.Home == "" {
				return ""
			}
			return filepath.Join(cfg.Home, "releases")
		}(),
	}
}

// WithWaiting returns a copy that says on w, every waitReport, how long a call has been
// waiting for its answer.
func (c *Client) WithWaiting(w io.Writer) *Client {
	d := *c
	d.waiting = w
	return &d
}

// waitReport is how often a call still waiting for its answer says so. It only reports.
const waitReport = 5 * time.Second

// awaiting reports the wait until the returned func is called, which returns once no
// report is being written.
func (c *Client) awaiting() func() {
	if c.waiting == nil {
		return func() {}
	}
	began, ticker, done, gone := time.Now(), time.NewTicker(waitReport), make(chan struct{}), make(chan struct{})
	go func() {
		defer close(gone)
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				fmt.Fprintf(c.waiting, "waiting on Tensorhub at %s (%s)\n", c.named, time.Since(began).Round(time.Second))
			}
		}
	}()
	return func() {
		ticker.Stop()
		close(done)
		<-gone
	}
}

// transports holds one pooled transport per liveness window: the process's config is
// frozen, so in practice one.
var transports sync.Map

// HTTP is the one client every Hub request uses. No clock bounds a Hub that is alive,
// however slowly it answers. A connection is dead once it shows no sign of life for the
// liveness window: half of it without a frame sends a probe (an HTTP/2 PING, and a TCP
// keepalive under any protocol), and the other half without its ACK closes the connection.
// A dial gets the same half for its SYN-ACK and a TLS handshake the whole window; a
// refusal, an unreachable route or a TLS failure ends a call at once.
func HTTP(liveness time.Duration) *http.Client {
	if t, ok := transports.Load(liveness); ok {
		return &http.Client{Transport: t.(*http.Transport)}
	}
	probe := liveness / 2
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.DialContext = (&net.Dialer{Timeout: probe, KeepAliveConfig: net.KeepAliveConfig{
		Enable: true, Idle: probe, Interval: probe / 5, Count: 5}}).DialContext
	t.TLSHandshakeTimeout = liveness
	t.HTTP2 = &http.HTTP2Config{SendPingTimeout: probe, PingTimeout: probe, WriteByteTimeout: liveness}
	pooled, _ := transports.LoadOrStore(liveness, t)
	return &http.Client{Transport: pooled.(*http.Transport)}
}

func (c *Client) Base() string { return c.base }

// Resource is the shared shape of a package or model returned by its typed public
// route. The route supplies the type; the document therefore carries no `kind`
// discriminator and Cozy never guesses one from its contents.
type Resource struct {
	Org           string `json:"org"`
	Name          string `json:"name"`
	Family        string `json:"family,omitempty"`
	CreatedAt     string `json:"created_at"`
	LatestRelease string `json:"latest_release,omitempty"`
}

type ReleaseSummary struct {
	CutAt    string `json:"cut_at"`
	Release  string `json:"release"`
	Yanked   bool   `json:"yanked,omitempty"`
	YankedAt string `json:"yanked_at,omitempty"`
}

type PackageCard struct {
	Package  Resource         `json:"package"`
	Releases []ReleaseSummary `json:"releases"`
}

type ModelLaneSummary struct {
	Lane       string   `json:"lane"`
	ManifestID string   `json:"manifest_id"`
	Components []string `json:"components"`
	// ComponentBytes is each component's stored bytes (th-181); absent when the hub has
	// not derived them for this checkpoint.
	ComponentBytes map[string]int64 `json:"component_bytes,omitempty"`
	Bytes          int64            `json:"bytes"`
}

type ModelReleaseSummary struct {
	ReleaseSummary
	Revision int64              `json:"revision"`
	Lanes    []ModelLaneSummary `json:"lanes"`
	// DefaultLane is the lane a ref naming this release and no lane reads (th-245).
	DefaultLane string `json:"default_lane"`
}

type ModelCard struct {
	Model    Resource              `json:"model"`
	Releases []ModelReleaseSummary `json:"releases"`
}

func (r Resource) Ref() string { return r.Org + "/" + r.Name }

type call struct {
	method       string
	path         string
	body         any
	bodyBytes    []byte // exact caller-persisted JSON; never re-marshaled on replay
	auth         bool   // carries the configured operator token or a short user token
	optionalAuth bool   // checkpoint reads use available credentials without requiring login
	reason       string // X-Tensorhub-Reason; the hub refuses a mutation without one
	// idempotency is the caller-owned operation identity for a paid mutation. It is
	// distinct from Tensorhub's provider operation id and survives a lost HTTP answer.
	idempotency string
	// responseBytes widens the ordinary small-JSON cap for one explicitly bounded
	// response shape.
	responseBytes int64
	// raw takes the answer's exact bytes instead of decoding it. The snapshot
	// manifest route answers a canonical document verbatim, and this client must
	// carry it the same way — nothing here re-encodes one.
	raw *[]byte
	// status receives the answer's HTTP status; with raw, a refusal's exact bytes too.
	status *int
	// trustRoot receives the root of an already verified TLS chain for delegation to an agent.
	trustRoot *[]byte
}

// WithToken returns a copy of the client carrying a credential supplied for this
// invocation (`--token-stdin`) instead of the configured one. The value never
// reaches argv, a record, or a log line — the secret fence keeps Reveal() here.
func (c *Client) WithToken(v secret.Value, source string) *Client {
	d := *c
	d.token, d.source, d.tokens = v, source, nil
	return &d
}

// WithTokenSource returns a copy that prefers a short user bearer. A configured
// operator token is only the fallback when this machine has never enrolled.
func (c *Client) WithTokenSource(source TokenSource) *Client {
	d := *c
	d.tokens = source
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
func (c *Client) Models(ctx context.Context, query, family string) ([]Resource, ResourceSearch, *exit.Error) {
	var out struct {
		Models []Resource     `json:"models"`
		Search ResourceSearch `json:"search"`
	}
	path := resourceSearchPath("models", query)
	separator := "?"
	if strings.Contains(path, "?") {
		separator = "&"
	}
	if family != "" {
		path += separator + url.Values{"family": []string{family}}.Encode()
	}
	if e := c.do(ctx, call{method: http.MethodGet, path: path}, &out); e != nil {
		return nil, ResourceSearch{}, e
	}
	return out.Models, out.Search, nil
}

func (c *Client) SetModelFamily(ctx context.Context, ref Ref, family string) (Resource, *exit.Error) {
	var out struct {
		Model Resource `json:"model"`
	}
	e := c.do(ctx, call{
		method: http.MethodPatch, path: resourcePath("models", ref), auth: true,
		reason: "update model discovery family", body: map[string]string{"family": family},
	}, &out)
	return out.Model, e
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

func (c *Client) PackageCard(ctx context.Context, ref Ref) (PackageCard, *exit.Error) {
	var out PackageCard
	e := c.do(ctx, call{method: http.MethodGet, path: resourcePath("packages", ref)}, &out)
	return out, e
}

func (c *Client) ModelCard(ctx context.Context, ref Ref) (ModelCard, *exit.Error) {
	var out ModelCard
	// A signed-in owner sees its own private repository; a signed-out reader sees public ones.
	e := c.do(ctx, call{method: http.MethodGet, path: resourcePath("models", ref), optionalAuth: true}, &out)
	return out, e
}

func resourcePath(collection string, ref Ref) string {
	return "/v1/" + collection + "/" + ref.Org + "/" + ref.Name
}

func (c *Client) do(ctx context.Context, cl call, out any) *exit.Error {
	status, e := c.doOnce(ctx, cl, out)
	if status != http.StatusUnauthorized || !cl.auth || c.tokens == nil {
		return e
	}
	// A bearer the hub no longer recognizes -- it restarted or rotated its signing
	// state without expiring our cached copy -- is re-minted from the durable machine
	// key exactly once. A 401 arrives before the hub does any work, so the replay is
	// safe; a machine key the hub truly revoked fails the re-mint and surfaces there.
	source, renewable := c.tokens.(interface{ Invalidate() })
	if !renewable {
		return e
	}
	source.Invalidate()
	_, e = c.doOnce(ctx, cl, out)
	return e
}

func (c *Client) doOnce(ctx context.Context, cl call, out any) (int, *exit.Error) {
	token := c.token
	if (cl.auth || cl.optionalAuth) && c.tokens != nil {
		userToken, problem := c.tokens.AccessToken(ctx)
		switch {
		case problem == nil:
			token = userToken
		case cl.optionalAuth && problem.ErrName() == "auth.machine_key_missing":
			// A signed-out checkpoint reader needs no credential. An explicitly
			// configured operator token, if present, still takes its existing path.
		case !token.Present() || problem.ErrName() != "auth.machine_key_missing":
			return 0, problem
		}
	}
	if cl.auth && !token.Present() {
		return 0, exit.Named(exit.Credential, "hub.token_missing",
			"%s %s requires a Tensorhub login", cl.method, cl.path).
			WithNext("cozy auth login <email>")
	}

	var body io.Reader
	if cl.body != nil && cl.bodyBytes != nil {
		return 0, exit.Internalf("hub call %s %s supplied both structured and exact request bytes", cl.method, cl.path)
	}
	if cl.bodyBytes != nil {
		body = bytes.NewReader(cl.bodyBytes)
	} else if cl.body != nil {
		b, err := json.Marshal(cl.body)
		if err != nil {
			return 0, exit.Internalf("encoding the request body failed: %s", err)
		}
		body = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(ctx, cl.method, c.base+cl.path, body)
	if err != nil {
		return 0, exit.Usagef("%q is not a usable hub URL: %s", c.base, err).
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
	if (cl.auth || cl.optionalAuth) && token.Present() {
		// The ONE raw read of the credential in this binary. It goes into a header on
		// a request and nowhere else: not a log line, not a record, not a rendering.
		req.Header.Set("Authorization", "Bearer "+token.Reveal())
	}

	defer c.awaiting()()
	resp, err := c.http.Do(req)
	if err != nil {
		return 0, TransportFailure(c.base, err)
	}
	defer resp.Body.Close()
	if cl.trustRoot != nil && resp.TLS != nil && len(resp.TLS.VerifiedChains) > 0 {
		chain := resp.TLS.VerifiedChains[0]
		if len(chain) > 0 {
			*cl.trustRoot = append([]byte(nil), chain[len(chain)-1].Raw...)
		}
	}

	cap := int64(maxBody)
	if cl.raw != nil {
		cap = maxDocument
	}
	if cl.responseBytes > cap {
		cap = cl.responseBytes
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, cap))
	if err != nil {
		return resp.StatusCode, TransportFailure(c.base, err)
	}
	if cl.status != nil {
		*cl.status = resp.StatusCode
		if cl.raw != nil {
			*cl.raw = raw
		}
	}
	if resp.StatusCode >= 400 {
		return resp.StatusCode, c.refusal(resp.StatusCode, raw)
	}
	if cl.raw != nil {
		*cl.raw = raw
		return resp.StatusCode, nil
	}
	if out != nil && len(bytes.TrimSpace(raw)) > 0 {
		if err := json.Unmarshal(raw, out); err != nil {
			return resp.StatusCode, exit.Named(exit.Internal, "hub.unreadable_answer",
				"%s %s answered %d with a body this client cannot read: %s",
				cl.method, cl.path, resp.StatusCode, err).
				WithRemedy("check that TENSORHUB_URL names a tensorhub, not a proxy or a login page").
				WithNext("cozy package search")
		}
	}
	return resp.StatusCode, nil
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
	// Absence is a fact about one hub; with several configured, a bare "no X" misleads.
	if code == exit.NotFound && c.named != "" {
		e.Message += " on hub " + c.named
		e.Details = map[string]any{"tensorhub": c.base}
	}
	if code == exit.Credential {
		// The one next step the hub cannot write for us: it does not know where this
		// Creator stores its machine credential, or which hub this command addressed.
		e.WithNext("cozy auth login <email> --tensorhub=" + c.label)
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
	case status == http.StatusPaymentRequired:
		return exit.Capacity
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
	org, name, ok := strings.Cut(CanonicalName(s), "/")
	if !ok || !resourceSlug.MatchString(org) || !resourceSlug.MatchString(name) {
		return Ref{}, exit.Usagef("%q is not a model or package ref: expected exactly one org/name separator", s).
			WithRemedy("org and name use letters, digits, dots, underscores, or hyphens").
			WithNext("cozy package search", "cozy model search")
	}
	return Ref{Org: org, Name: name}, nil
}

// Context is one Hub call's context: its caller may cancel it, and no clock does. A Hub
// that is alive may take as long as it needs; a dead connection ends the call (HTTP).
func Context() (context.Context, context.CancelFunc) {
	return context.WithCancel(context.Background())
}
