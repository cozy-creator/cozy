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
	"net/http"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/config"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
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

// Transfer bounds one call whose work is proportional to the bytes it moves.
const Transfer = 30 * time.Minute

// Client is one configured hub endpoint. It holds no state between calls.
type Client struct {
	base   string
	token  secret.Value
	source string // where the token came from, for the remedy text
	http   *http.Client
	agent  string
}

// New builds the client from the frozen config value. It reads no environment.
func New(cfg config.Config, agent string) *Client {
	return &Client{
		base:   strings.TrimRight(cfg.HubURL, "/"),
		token:  cfg.HubToken,
		source: cfg.HubTokenSource,
		http:   &http.Client{Timeout: Timeout},
		agent:  agent,
	}
}

func (c *Client) Base() string        { return c.base }
func (c *Client) Token() secret.Value { return c.token }

// Repo is one catalog row as the public listing renders it. The grammar is the hub's
// (th-003 freezes it); this consumes it verbatim and defines none of its own.
type Repo struct {
	Org       string `json:"org"`
	Name      string `json:"name"`
	Kind      string `json:"kind"`
	CreatedAt string `json:"created_at"`
}

func (r Repo) Ref() string { return r.Org + "/" + r.Name }

// Health is GET /healthz.
type Health struct {
	Status string `json:"status"`
	Env    string `json:"env"`
}

// ConfigKey is one row of the hub's effective configuration. A secret key renders as
// its digest on the hub side; this client never sees the value and cannot print one.
type ConfigKey struct {
	Key    string `json:"key"`
	Value  string `json:"value"`
	Source string `json:"source"`
	Secret bool   `json:"secret"`
	Doc    string `json:"doc"`
}

type call struct {
	method string
	path   string
	body   any
	admin  bool   // carries the admin token
	reason string // X-Tensorhub-Reason; the hub refuses a mutation without one
	// timeout overrides Timeout for a call whose work is bounded by BYTES rather
	// than by the hub's own latency (completion re-streams and re-hashes every
	// declared object). A catalog read that is slow is broken; a completion that
	// is slow is working.
	timeout time.Duration
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

// Health reads the hub's liveness. Public.
func (c *Client) Health(ctx context.Context) (Health, *exit.Error) {
	var out Health
	e := c.do(ctx, call{method: http.MethodGet, path: "/healthz"}, &out)
	return out, e
}

// Repos lists the catalog. Public — no credential, and a tokenless caller sees
// exactly what a credentialed one sees.
func (c *Client) Repos(ctx context.Context) ([]Repo, *exit.Error) {
	var out struct {
		Repos []Repo `json:"repos"`
	}
	if e := c.do(ctx, call{method: http.MethodGet, path: "/v1/repos"}, &out); e != nil {
		return nil, e
	}
	return out.Repos, nil
}

// CreateRepo is the first-party publish door: create a model or endpoint repo under
// the admin token, with a reason the hub records durably before it acts.
func (c *Client) CreateRepo(ctx context.Context, org, name, kind, reason string) (Repo, *exit.Error) {
	var out struct {
		Repo Repo `json:"repo"`
	}
	e := c.do(ctx, call{
		method: http.MethodPost, path: "/v1/repos", admin: true, reason: reason,
		body: map[string]string{"org": org, "name": name, "kind": kind},
	}, &out)
	return out.Repo, e
}

// EffectiveConfig reads the hub's running configuration with per-key provenance.
// Admin. Secret values arrive already digested by the hub.
func (c *Client) EffectiveConfig(ctx context.Context) (string, []ConfigKey, *exit.Error) {
	var out struct {
		Env  string      `json:"env"`
		Keys []ConfigKey `json:"keys"`
	}
	e := c.do(ctx, call{method: http.MethodGet, path: "/v1/admin/effective-config", admin: true}, &out)
	return out.Env, out.Keys, e
}

func (c *Client) do(ctx context.Context, cl call, out any) *exit.Error {
	// A missing credential is answered BEFORE the dial: a round trip cannot tell the
	// operator anything the local configuration does not already say.
	if cl.admin && !c.token.Present() {
		return exit.Named(exit.Credential, "hub.token_missing",
			"%s %s is a first-party route and no admin token is configured", cl.method, cl.path).
			WithRemedy("set TENSORHUB_TOKEN to the hub's admin.token; catalog reads need no credential").
			WithNext("cozy hub status", "cozy search")
	}

	var body io.Reader
	if cl.body != nil {
		b, err := json.Marshal(cl.body)
		if err != nil {
			return exit.Internalf("encoding the request body failed: %s", err)
		}
		body = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(ctx, cl.method, c.base+cl.path, body)
	if err != nil {
		return exit.Usagef("%q is not a usable hub URL: %s", c.base, err).
			WithRemedy("set TENSORHUB_URL to a base URL, e.g. https://hub.example.com").
			WithNext("cozy hub status")
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", c.agent)
	if cl.body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if cl.reason != "" {
		req.Header.Set("X-Tensorhub-Reason", cl.reason)
	}
	if cl.admin {
		// The ONE raw read of the credential in this binary. It goes into a header on
		// a request and nowhere else: not a log line, not a record, not a rendering.
		req.Header.Set("Authorization", "Bearer "+c.token.Reveal())
	}

	client := c.http
	if cl.timeout > 0 {
		wider := *c.http
		wider.Timeout = cl.timeout
		client = &wider
	}
	resp, err := client.Do(req)
	if err != nil {
		return c.transport(err)
	}
	defer resp.Body.Close()

	cap := int64(maxBody)
	if cl.raw != nil {
		cap = maxDocument
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, cap))
	if err != nil {
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
				WithNext("cozy hub status")
		}
	}
	return nil
}

// transport maps a failure that never became an HTTP answer. Unreachable is 9;
// a deadline is 10 — a hub that is up but stuck is a different problem.
func (c *Client) transport(err error) *exit.Error {
	if errors.Is(err, context.DeadlineExceeded) || strings.Contains(err.Error(), "Client.Timeout") {
		return exit.New(exit.Deadline, "the hub at %s did not answer within %s", c.base, Timeout).
			WithRemedy("retry; if it persists the hub is up but not serving").
			WithNext("cozy hub status")
	}
	return exit.Unavailablef("the hub at %s is unreachable: %s", c.base, unwrapURL(err)).
		WithRemedy("check TENSORHUB_URL and that the hub is running").
		WithNext("cozy hub status")
}

// unwrapURL strips net/http's URL wrapper so the message names the cause, not the
// verb and the URL a second time.
func unwrapURL(err error) string {
	var ue interface{ Unwrap() error }
	if errors.As(err, &ue) {
		if inner := ue.Unwrap(); inner != nil {
			return inner.Error()
		}
	}
	return err.Error()
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
			WithRemedy("this route may not exist on this hub build; `cozy hub status` names it and its version").
			WithNext("cozy hub status")
	}

	e := exit.Named(code, env.Error.Code, "%s", env.Error.Message)
	if env.Error.Remedy != "" {
		e.WithRemedy("%s", env.Error.Remedy)
	}
	if code == exit.Credential {
		// The one remedy the hub cannot write for us: it does not know where OUR
		// token came from. Launch 1 has no login, so the next step is never `cozy login`.
		e.WithNext("cozy hub status")
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

// Ref is `org/name` — one model or endpoint repo. Release addressing
// (`@release` / `@sha256:…`) is th-003's grammar and is refused by name until it
// exists, rather than being parsed into something this build cannot resolve.
type Ref struct {
	Org  string
	Name string
}

func (r Ref) String() string { return r.Org + "/" + r.Name }

func ParseRef(s string) (Ref, *exit.Error) {
	if strings.Contains(s, "@") {
		return Ref{}, exit.Usagef("%q pins a release, which this build cannot resolve", s).
			WithRemedy("release addressing (@release, @sha256:…) lands with th-003; name the repo alone").
			WithNext("cozy search")
	}
	org, name, ok := strings.Cut(s, "/")
	if !ok || org == "" || name == "" || strings.Contains(name, "/") {
		return Ref{}, exit.Usagef("%q is not a repo ref: expected exactly one org/name separator", s).
			WithRemedy("the grammar is org/name, e.g. cozy/sdxl").
			WithNext("cozy search")
	}
	return Ref{Org: org, Name: name}, nil
}

// Kinds is the repo-kind vocabulary, the hub's own (its CHECK constraint).
var Kinds = []string{"model", "endpoint"}

func CheckKind(kind string) *exit.Error {
	for _, k := range Kinds {
		if kind == k {
			return nil
		}
	}
	return exit.Usagef("%q is not a repo kind", kind).
		WithRemedy("kind is one of: %s", strings.Join(Kinds, ", "))
}

// Context bounds one hub call. Handlers never build their own.
func Context() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), Timeout)
}

// LongContext bounds a whole transfer — many calls, some of them proportional to the
// bytes moved. A catalog read that takes minutes is broken; a publish that does is
// working, and one deadline cannot mean both.
func LongContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), Transfer)
}
