// Package client is the CLI's client of the LOCAL CLIENT API (cl-010). It exists so
// invoke, unload, and down commands are the API's FIRST
// CLIENT rather than a parallel implementation the HTTP surface later wraps: every one of
// them speaks the routes in `docs/client-contract.md` over a real socket, exactly as
// cl-007's UI and cozy.art do.
//
// Three things this package deliberately does NOT do:
//
//   - It does not define the contract's shapes. `api.Submission`, `api.Handle`,
//     `api.Lifecycle`, `api.MediaRef` ARE the contract, and a client struct that
//     re-declared them would be a second spelling that drifts.
//   - It does not invent a refusal vocabulary. Every non-2xx answer is the typed
//     envelope, and `api.CodeOf` (the server's own status mapping, inverted beside it)
//     turns it back into a shared-matrix exit code with the server's name kept.
//   - It does not hold a credential of its own. `api.ClientCredential` reads the 0600
//     handoff file, and `api.Authorize` is the one place it becomes a header.
package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/secret"
)

// Client is one CLI process's connection to the running Cozy daemon.
type Client struct {
	base  string
	token secret.Value
	http  *http.Client
}

// Open reads the running daemon's address and its 0600 credential. It never probes:
// the caller already passed the shared exit-9 gate, and a second probe here would be a
// second spelling of "is it up".
func Open(cfg config.Config, st daemon.State) (*Client, *exit.Error) {
	l, e := home.Open(cfg.Home)
	if e != nil {
		return nil, e
	}
	token, e := api.ClientCredential(l)
	if e != nil {
		return nil, e
	}
	return &Client{
		base:  "http://" + st.Addr,
		token: token,
		// No client-wide clock decides whether a local request stalled.
		// Explicit caller deadlines cancel their request; worker liveness and measured
		// no-progress facts decide operational failure.
		http: &http.Client{},
	}, nil
}

// Addr is the daemon address this client talks to, for rendering.
func (c *Client) Addr() string { return strings.TrimPrefix(c.base, "http://") }

func (c *Client) request(method, path string, body any, headers ...string) (*http.Request, *exit.Error) {
	var reader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, exit.Internalf("cannot render the request body: %s", err)
		}
		reader = bytes.NewReader(data)
	}
	req, err := http.NewRequest(method, c.base+path, reader)
	if err != nil {
		return nil, exit.Internalf("cannot build a request for %s: %s", path, err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	for i := 0; i+1 < len(headers); i += 2 {
		req.Header.Set(headers[i], headers[i+1])
	}
	// No Origin header: the CLI is not a browser, and the API admits an absent Origin
	// precisely because a non-browser client is the only caller that legitimately omits
	// one. Sending a fabricated one would be the client pretending to be a page.
	api.Authorize(req, c.token)
	return req, nil
}

// call issues one request and decodes `out`, or turns the typed envelope into a typed
// CLI error. Every refusal a caller sees came from the server, spelled the server's way.
func (c *Client) call(method, path string, body, out any, headers ...string) *exit.Error {
	return c.callContext(context.Background(), method, path, body, out, headers...)
}

func (c *Client) callContext(ctx context.Context, method, path string, body, out any,
	headers ...string,
) *exit.Error {
	req, e := c.request(method, path, body, headers...)
	if e != nil {
		return e
	}
	req = req.WithContext(ctx)
	res, err := c.http.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return exit.New(exit.Canceled, "the Cozy daemon request was canceled: %s", ctx.Err())
		}
		return c.unreachable(err)
	}
	defer res.Body.Close()
	data, err := io.ReadAll(io.LimitReader(res.Body, 64<<20))
	if err != nil {
		return exit.Internalf("the Cozy daemon answer could not be read: %s", err)
	}
	if res.StatusCode >= 300 {
		return Refusal(res.StatusCode, data)
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(data, out); err != nil {
		return exit.Internalf("the Cozy daemon answered %s with a body this client cannot read: %s",
			path, err)
	}
	return nil
}

// unreachable is what a dead or dying daemon looks like from a client's seat: not a
// refusal document, a transport failure. It carries the SAME remedy every server-backed
// verb's exit-9 gate carries, because it is the same condition arriving later.
func (c *Client) unreachable(err error) *exit.Error {
	return exit.Unavailablef("the Cozy daemon stopped answering on %s: %s", c.Addr(), err).
		WithRemedy("it may have stopped mid-request; retry or run `cozy up`").
		WithNext("cozy up", "cozy run list")
}

// Refusal turns one typed error envelope into a typed CLI error. The NAME is the
// server's own (`override_unresolved`, `bundle_corrupt`, `not_found`); the CODE is the
// shared matrix code the server's status mapped out of. A body that is not an envelope
// is reported as what it was, never as a guess.
func Refusal(status int, data []byte) *exit.Error {
	var doc struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
			Remedy  string `json:"remedy"`
		} `json:"error"`
	}
	code := api.CodeOf(status)
	if json.Unmarshal(data, &doc) != nil || doc.Error.Code == "" {
		body := strings.TrimSpace(string(data))
		if len(body) > 200 {
			body = body[:200] + "…"
		}
		return exit.Named(code, "untyped_answer",
			"the Cozy daemon answered %d with no typed envelope: %s", status, body)
	}
	e := exit.Named(code, doc.Error.Code, "%s", doc.Error.Message)
	if doc.Error.Remedy != "" {
		e.WithRemedy("%s", doc.Error.Remedy)
	}
	return e
}

// ---------------------------------------------------------------- the contract CORE

// Submit posts one request under an idempotency key. `replay` is the server's own
// answer, never inferred from equal ids.
func (c *Client) Submit(sub api.Submission, key string) (api.Handle, *exit.Error) {
	var h api.Handle
	e := c.call("POST", "/v1/requests", sub, &h, "Idempotency-Key", key)
	return h, e
}

// Request reads one request's lifecycle document.
// Triage fetches one attempt's kept triage bundle — raw bytes, already verified by the
// daemon against the terminal's own reference before it was stored (cl-116).
func (c *Client) Triage(attemptKey string) ([]byte, *exit.Error) {
	req, e := c.request(http.MethodGet, "/v1/local/attempts/"+url.PathEscape(attemptKey)+"/triage", nil)
	if e != nil {
		return nil, e
	}
	res, err := c.http.Do(req)
	if err != nil {
		return nil, c.unreachable(err)
	}
	defer res.Body.Close()
	data, err := io.ReadAll(io.LimitReader(res.Body, 2<<20))
	if err != nil {
		return nil, exit.Internalf("the triage bundle could not be read: %s", err)
	}
	if res.StatusCode >= 300 {
		return nil, Refusal(res.StatusCode, data)
	}
	return data, nil
}

func (c *Client) Request(id string) (api.Lifecycle, *exit.Error) {
	var life api.Lifecycle
	e := c.call("GET", "/v1/requests/"+id, nil, &life)
	return life, e
}

// Requests lists ordinary invocations and jobs through the one request lifecycle
// authority. Kind distinguishes their execution expectation without creating a
// second public inventory.
func (c *Client) Requests(ctx context.Context, status, packageName string, limit int) ([]api.Lifecycle, *exit.Error) {
	var out struct {
		Requests []api.Lifecycle `json:"requests"`
	}
	path := fmt.Sprintf("/v1/requests?limit=%d", limit)
	if status != "" {
		path += "&status=" + url.QueryEscape(status)
	}
	if packageName != "" {
		path += "&package=" + url.QueryEscape(packageName)
	}
	problem := c.callContext(ctx, http.MethodGet, path, nil, &out)
	return out.Requests, problem
}

// Cancel REQUESTS cancellation. The attempt's own journaled terminal settles it, so this
// returns as soon as the request is recorded and the caller keeps watching the stream.
// Cancel is an ATTRIBUTED act (cl-108): the actor names who is canceling — a person's
// explicit `cozy run cancel`, a caller-authored deadline — and rides the durable record.
func (c *Client) Cancel(id, actor string) *exit.Error {
	return c.call("POST", "/v1/requests/"+id+"/cancel", map[string]string{"actor": actor}, nil)
}

// ------------------------------------------------------------ the LOCAL extension

// StartResult is the rental-claim answer.
type StartResult struct {
	InstanceID string `json:"instance_id"`
	Package    string `json:"package"`
	Change     string `json:"change"`
	Note       string `json:"note"`
}

// EnsureRental directly claims one attached empty worker and waits for ClaimAck.
func (c *Client) EnsureRental(rentalID string) (StartResult, *exit.Error) {
	var res StartResult
	e := c.call("POST", "/v1/local/rentals/"+url.PathEscape(rentalID)+"/claim", map[string]any{}, &res)
	return res, e
}

// DetachRental waits until the daemon no longer holds this rental's worker-control slot.
// The result is false when the slot was already absent.
func (c *Client) DetachRental(rentalID string) (bool, *exit.Error) {
	var out struct {
		Changed bool `json:"changed"`
	}
	e := c.call(http.MethodDelete,
		"/v1/local/rentals/"+url.PathEscape(rentalID)+"/claim", nil, &out)
	return out.Changed, e
}

// Unload asks the daemon to stop only definitely-idle local serving workers. It
// never touches remote rentals, run-once jobs, active work, or installed disk bytes.
func (c *Client) Unload() (api.UnloadResult, *exit.Error) {
	var out api.UnloadResult
	e := c.call(http.MethodPost, "/v1/local/daemon/unload", map[string]any{}, &out)
	return out, e
}

// Down performs the daemon-side lifecycle fence. Under all=false, active work or
// rentals refuse without mutation. Under all=true, the daemon requests cancellation
// and returns the exact paid obligations the caller must terminate and confirm through
// Tensorhub before retrying. ShuttingDown=true means cooperative down was accepted.
func (c *Client) Down(all bool) (api.DownResult, *exit.Error) {
	var out api.DownResult
	e := c.call(http.MethodPost, "/v1/local/daemon/down", map[string]bool{"all": all}, &out)
	return out, e
}

// ------------------------------------------------------------------ the JOB family

// SubmitJob posts one bounded job under an idempotency key. It never answers "busy":
// no capacity is a queued STATE, and the handle says which.
func (c *Client) SubmitJob(sub api.JobSubmission, key string) (api.JobHandle, *exit.Error) {
	var h api.JobHandle
	e := c.call("POST", "/v1/local/jobs", sub, &h, "Idempotency-Key", key)
	return h, e
}

// Job reads one job's state document.
func (c *Client) Job(id string) (api.JobState, *exit.Error) {
	var state api.JobState
	e := c.call("GET", "/v1/local/jobs/"+id, nil, &state)
	return state, e
}

// CancelJob REQUESTS cancellation. A running job's own journaled terminal settles it; a
// queued one leaves the queue and settles here. The actor names who is canceling.
func (c *Client) CancelJob(id, actor string) *exit.Error {
	return c.call("POST", "/v1/local/jobs/"+id+"/cancel", map[string]string{"actor": actor}, nil)
}
