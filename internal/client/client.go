// Package client is the CLI's client of the LOCAL CLIENT API (cl-010). It exists so
// `cozy run`, `cozy start`, `cozy stop`, `cozy logs` and `cozy doctor` are the API's FIRST
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
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/api"
	"github.com/cozy-creator/cozy-creator-v2/internal/config"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
	"github.com/cozy-creator/cozy-creator-v2/internal/service"
)

// Client is one CLI process's connection to the running LocalService.
type Client struct {
	base  string
	token secret.Value
	http  *http.Client
}

// Open reads the running service's address and its 0600 credential. It never probes:
// the caller already passed the shared exit-9 gate, and a second probe here would be a
// second spelling of "is it up".
func Open(cfg config.Config, st service.State) (*Client, *exit.Error) {
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
		// No client-wide clock decides whether a local workflow or request stalled.
		// Explicit caller deadlines cancel their request; worker liveness and measured
		// no-progress facts decide operational failure.
		http: &http.Client{},
	}, nil
}

// Addr is the service address this client talks to, for rendering.
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
	req, e := c.request(method, path, body, headers...)
	if e != nil {
		return e
	}
	res, err := c.http.Do(req)
	if err != nil {
		return c.unreachable(err)
	}
	defer res.Body.Close()
	data, err := io.ReadAll(io.LimitReader(res.Body, 64<<20))
	if err != nil {
		return exit.Internalf("the LocalService answer could not be read: %s", err)
	}
	if res.StatusCode >= 300 {
		return Refusal(res.StatusCode, data)
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(data, out); err != nil {
		return exit.Internalf("the LocalService answered %s with a body this client cannot read: %s",
			path, err)
	}
	return nil
}

// unreachable is what a dead or dying service looks like from a client's seat: not a
// refusal document, a transport failure. It carries the SAME remedy every server-backed
// verb's exit-9 gate carries, because it is the same condition arriving later.
func (c *Client) unreachable(err error) *exit.Error {
	return exit.Unavailablef("the cozy LocalService stopped answering on %s: %s", c.Addr(), err).
		WithRemedy("it may have exited mid-request; its log is in the local root").
		WithNext("cozy status", "cozy up")
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
			"the LocalService answered %d with no typed envelope: %s", status, body)
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
func (c *Client) Request(id string) (api.Lifecycle, *exit.Error) {
	var life api.Lifecycle
	e := c.call("GET", "/v1/requests/"+id, nil, &life)
	return life, e
}

// Cancel REQUESTS cancellation. The attempt's own journaled terminal settles it, so this
// returns as soon as the request is recorded and the caller keeps watching the stream.
func (c *Client) Cancel(id string) *exit.Error {
	return c.call("POST", "/v1/requests/"+id+"/cancel", nil, nil)
}

// Media hands one output's bytes to `write` and returns what it wrote plus the digest the
// server declared. The id is OPAQUE: this client composes no path and cannot ask for one.
func (c *Client) Media(mediaID string, write func(io.Reader) (int64, *exit.Error)) (int64, string, *exit.Error) {
	req, e := c.request("GET", "/v1/media/"+mediaID, nil)
	if e != nil {
		return 0, "", e
	}
	res, err := c.http.Do(req)
	if err != nil {
		return 0, "", c.unreachable(err)
	}
	defer res.Body.Close()
	if res.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(res.Body, 1<<20))
		return 0, "", Refusal(res.StatusCode, data)
	}
	n, e := write(res.Body)
	return n, res.Header.Get("X-Cozy-Digest"), e
}

// ------------------------------------------------------------ the LOCAL extension

// Endpoints lists what this host can serve.
func (c *Client) Endpoints() ([]api.EndpointRow, *exit.Error) {
	var out struct {
		Endpoints []api.EndpointRow `json:"endpoints"`
	}
	e := c.call("GET", "/v1/local/endpoints", nil, &out)
	return out.Endpoints, e
}

// Worker is one live worker, as the orchestrator reports it. The fields are
// `orchestrator.WorkerFacts` on the wire; a client reads the ones it renders.
type Worker struct {
	InstanceID string   `json:"instance_id"`
	Endpoint   string   `json:"endpoint"`
	ReleaseID  string   `json:"release_id"`
	SessionID  string   `json:"session_id"`
	PID        int      `json:"pid"`
	Exited     bool     `json:"exited"`
	Devices    []string `json:"devices"`
	// THE TWO AXES, and the machine phase they were pulled out of (#473/#482).
	// `intake_state` is retired with both of its uses: one enum could not say "staged on
	// disk but offline", which is the exact state an outgoing spec holds under
	// fallback-retention.
	Phase           string   `json:"worker_phase"`
	Materialization string   `json:"materialization"`
	Serving         string   `json:"serving"`
	Plans           []string `json:"dispatchable_plan_ids"`
	// QuietMS measures missed protocol reports. ErrorForMS is diagnostic only; typed
	// faults/refusals settle immediately and no elapsed duration decides readiness.
	QuietMS    int64  `json:"quiet_ms"`
	ErrorForMS int64  `json:"error_for_ms"`
	Fault      string `json:"fault"`
	// Refusal is the HOST's own verdict about this worker, distinct from the worker's own
	// fault: a fault may clear, a refusal is settled.
	Refusal string `json:"refusal"`
}

// Dispatchable answers whether this worker can take an attempt — the protocol's own fact,
// not a guess from liveness: a process that is up but has loaded nothing is not warm, and
// a placement that is STAGED but OFFLINE is not capacity however much of it is on disk.
func (w Worker) Dispatchable() bool {
	return !w.Exited && w.Serving == "DISPATCHABLE" && len(w.Plans) > 0
}

func (c *Client) Workers() ([]Worker, *exit.Error) {
	var out struct {
		Workers []Worker `json:"workers"`
	}
	e := c.call("GET", "/v1/local/workers", nil, &out)
	return out.Workers, e
}

// StartResult is the prewarm answer. `Change` says WHICH of the three things happened —
// `none`, `worker_started`, or `placement_added` (#484). The old `Resident bool` could
// only answer "was it already there", which is true both of an idempotent no-op and of a
// live worker that just gained a placement, and those are different answers.
type StartResult struct {
	InstanceID string `json:"instance_id"`
	Endpoint   string `json:"endpoint"`
	Change     string `json:"change"`
	Note       string `json:"note"`
}

// StartWorker makes an endpoint resident. `warm` false asks the worker to skip its boot
// warm pass — the route's `warm` field is omitted entirely when it is true, so the wire
// carries a choice only when one was made.
func (c *Client) EnsureWorker(endpoint string, warm bool) (StartResult, *exit.Error) {
	var res StartResult
	body := map[string]any{"endpoint": endpoint}
	if !warm {
		body["warm"] = false
	}
	e := c.call("POST", "/v1/local/workers", body, &res)
	return res, e
}

// StopResult is the drain answer. `Stopped` false means there was nothing to stop, which
// is a successful no-op and not a refusal.
type StopResult struct {
	InstanceID string `json:"instance_id"`
	Stopped    bool   `json:"stopped"`
	Note       string `json:"note"`
}

func (c *Client) ShutdownWorker(instance string) (StopResult, *exit.Error) {
	var res StopResult
	e := c.call("DELETE", "/v1/local/workers/"+instance, nil, &res)
	return res, e
}

// ShutdownService is `cozy down`'s cooperative ask (#449): the authenticated route that
// takes the same path a SIGTERM takes. The 202 means the ask was DELIVERED; the exit is
// proved by the service lock, which the caller watches.
func (c *Client) ShutdownService() *exit.Error {
	var out map[string]any
	return c.call("POST", "/v1/local/service/shutdown", map[string]any{}, &out)
}

// Doctor is the host/service document, verbatim. cozy-creator renders it; it derives
// nothing the server already answered.
func (c *Client) Doctor() (map[string]any, *exit.Error) {
	var out map[string]any
	e := c.call("GET", "/v1/local/doctor", nil, &out)
	return out, e
}

// Triage is the retained bundle by OPAQUE attempt key, with the server's own `explain`
// projection beside the whole document.
type Triage struct {
	AttemptKey string         `json:"attempt_key"`
	SubjectID  string         `json:"subject_id"`
	Length     int64          `json:"length"`
	Digest     string         `json:"digest"`
	Verified   bool           `json:"verified"`
	Explain    []string       `json:"explain"`
	Bundle     map[string]any `json:"bundle"`
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

// Jobs lists jobs newest-first with the server's own per-state counts.
func (c *Client) Jobs(status, endpoint string, limit int) ([]api.JobState, map[string]int, *exit.Error) {
	var out struct {
		Jobs   []api.JobState `json:"jobs"`
		States map[string]int `json:"states"`
	}
	path := fmt.Sprintf("/v1/local/jobs?limit=%d", limit)
	if status != "" {
		path += "&status=" + status
	}
	if endpoint != "" {
		path += "&endpoint=" + endpoint
	}
	e := c.call("GET", path, nil, &out)
	return out.Jobs, out.States, e
}

// CancelJob REQUESTS cancellation. A running job's own journaled terminal settles it; a
// queued one leaves the queue and settles here.
func (c *Client) CancelJob(id string) *exit.Error {
	return c.call("POST", "/v1/local/jobs/"+id+"/cancel", nil, nil)
}

func (c *Client) Triage(attemptKey string) (Triage, *exit.Error) {
	var t Triage
	e := c.call("GET", "/v1/local/attempts/"+attemptKey+"/triage", nil, &t)
	return t, e
}
