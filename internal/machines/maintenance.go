package machines

import (
	"context"
	"crypto/ed25519"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/exit"
)

type RuntimeState struct {
	Phase        string   `json:"phase"`
	Capabilities []string `json:"capabilities"`
	Runtime      string   `json:"runtime"`
	TensorFS     string   `json:"tensorfs"`
	Agent        struct {
		Version   string `json:"version"`
		SHA256    string `json:"sha256"`
		Selection string `json:"selection"`
	} `json:"agent"`
	Bootstrap struct {
		ABI            string `json:"abi"`
		Version        string `json:"version"`
		UpdateBoundary string `json:"update_boundary"`
	} `json:"bootstrap"`
	Update *RuntimeUpdateState `json:"update"`
}

// RuntimePair is one immutable software pair. From remains active while To is a
// candidate in prepared/waiting_activation states.
type RuntimePair struct {
	Runtime  string `json:"runtime"`
	TensorFS string `json:"tensorfs"`
}

// RuntimeUpdateState is additive across bootstrap versions. Terminal succeeded is the
// only state that activates To; prepared and waiting_activation are durable
// candidate states that callers may report without claiming activation.
type RuntimeUpdateState struct {
	Operation        string      `json:"operation"`
	State            string      `json:"state"`
	Error            string      `json:"error"`
	From             RuntimePair `json:"from"`
	To               RuntimePair `json:"to"`
	Pinned           bool        `json:"pinned,omitempty"`
	PreviouslyPinned bool        `json:"previously_pinned,omitempty"`
}

func (u *RuntimeUpdateState) PendingActivation() bool {
	return u != nil && (u.State == "prepared" || u.State == "waiting_activation")
}

type Maintenance struct {
	Base    string
	Client  *http.Client
	Machine string
	Public  ed25519.PublicKey
	Sign    func([]byte) []byte
	Log     io.Writer // optional: says when an upload is sent again
}

// ReadSoftware observes a running agent without launching it or changing its receipt.
func (h *Host) ReadSoftware(ctx context.Context) (*RuntimeState, *exit.Error) {
	record, problem := h.record()
	if problem != nil || record == nil {
		return nil, problem
	}
	if !h.alive(record.PID) {
		return nil, nil
	}
	client, problem := h.maintenanceFor(&Launch{WorkerID: record.WorkerID, Addr: "127.0.0.1:" + strconv.Itoa(record.WorkerPort)})
	if problem != nil {
		return nil, problem
	}
	defer client.Client.CloseIdleConnections()
	return client.State(ctx)
}

func (c *Maintenance) Do(ctx context.Context, method, path string, body io.Reader, into any) (int, *exit.Error) {
	token, err := capability.MintSigned(c.Public, c.Sign, capability.Grant{Machine: c.Machine, Action: capability.Maintenance,
		Expires: time.Now().Add(10 * time.Minute).Unix()})
	if err != nil {
		return 0, exit.Internalf("cannot mint the maintenance capability: %s", err)
	}
	request, err := http.NewRequestWithContext(ctx, method, c.Base+path, body)
	if err != nil {
		return 0, exit.Internalf("cannot address the machine: %s", err)
	}
	request.Header.Set("Authorization", "Cozy-Cap "+token)
	response, err := c.Client.Do(request)
	if err != nil {
		var verification *tls.CertificateVerificationError
		if errors.As(err, &verification) {
			return 0, exit.Named(exit.Credential, "machine.certificate_untrusted", "the machine did not prove its pinned identity: %s", err)
		}
		return 0, exit.Unavailablef("the machine did not answer: %s", err)
	}
	defer response.Body.Close()
	raw, readError := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if readError != nil && response.StatusCode/100 == 2 {
		return response.StatusCode, exit.Unavailablef("the machine's response was interrupted: %s", readError)
	}
	if response.StatusCode/100 != 2 {
		var refusal struct{ Code, Message string }
		if json.Unmarshal(raw, &refusal) == nil && refusal.Code == "runtime_starting" {
			return response.StatusCode, exit.Named(exit.Unavailable, "machine.runtime_starting", "%s", refusal.Message)
		}
		text := strings.TrimSpace(string(raw))
		if reason, lease := strings.CutPrefix(text, "rental_authority_unavailable: "); lease && response.StatusCode == http.StatusServiceUnavailable && !strings.Contains(reason, "denied") {
			// The agent admits controls from its Hub's lease; without one it started nothing.
			return response.StatusCode, exit.Named(exit.Unavailable, AuthorityPending, "the machine is waiting for its Hub's rental authority and started nothing: %s", reason)
		}
		return response.StatusCode, exit.New(exit.Failed, "the machine refused %s %s (HTTP %d): %s", method, path, response.StatusCode, text)
	}
	if into != nil && json.Unmarshal(raw, into) != nil {
		return response.StatusCode, exit.New(exit.Structural, "the machine answered %s %s with unreadable JSON", method, path)
	}
	return response.StatusCode, nil
}

// State reports the authenticated current Runtime and update operation.
// A peer lacking the current maintenance contract is refused.
func (c *Maintenance) State(ctx context.Context) (*RuntimeState, *exit.Error) {
	var state RuntimeState
	_, problem := c.Do(ctx, http.MethodGet, "/v1/machine/runtime", nil, &state)
	if problem != nil {
		return nil, problem
	}
	if !slices.Contains(state.Capabilities, RuntimeUpdateCapability) {
		return nil, exit.Named(exit.Structural, "machine.agent_update_required", "the machine does not advertise %s", RuntimeUpdateCapability)
	}
	digest, err := hex.DecodeString(state.Agent.SHA256)
	if state.Bootstrap.ABI != BootstrapCapability || state.Agent.Version == "" || err != nil || len(digest) != 32 ||
		(state.Agent.Selection != "bundled" && state.Agent.Selection != "explicit") {
		return nil, exit.Named(exit.Structural, "machine.agent_update_required", "the machine must report its running agent and support transactional agent replacement; install a current worker image or service bootstrap")
	}
	return &state, nil
}

// newestPublished is a distribution's newest release on the package index.
func NewestPublished(ctx context.Context, name string) (string, *exit.Error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://pypi.org/pypi/"+name+"/json", nil)
	if err != nil {
		return "", exit.Internalf("cannot address the package index: %s", err)
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return "", exit.Unavailablef("the package index did not answer: %s", err)
	}
	defer response.Body.Close()
	var project struct {
		Info struct {
			Version string `json:"version"`
		} `json:"info"`
	}
	if response.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(response.Body, 32<<20)).Decode(&project) != nil || project.Info.Version == "" {
		return "", exit.New(exit.Unavailable, "the package index has no release of %s", name)
	}
	return project.Info.Version, nil
}

// Stage sends one wheel to the machine and returns the digest the machine kept. A starting
// Runtime or a missing authority lease is waited out. A transport fault before the machine
// acknowledged the wheel staged nothing: it is sent once more when the machine answers again.
func (c *Maintenance) Stage(ctx context.Context, path, name string) (string, *exit.Error) {
	for resent := false; ; {
		file, err := os.Open(path)
		if err != nil {
			return "", exit.New(exit.NotFound, "cannot open update wheel %s: %s", name, err)
		}
		var staged struct {
			SHA256 string `json:"sha256"`
		}
		_, problem := c.Do(ctx, http.MethodPut, "/v1/machine/runtime/wheels/"+name, file, &staged)
		file.Close()
		if problem == nil {
			return staged.SHA256, nil
		}
		waits := problem.ErrName() == "machine.runtime_starting" || problem.ErrName() == AuthorityPending
		if problem.Code != exit.Unavailable || !waits && resent {
			return "", problem
		}
		if _, admission := c.AwaitUpdateAdmission(ctx); admission != nil {
			return "", problem
		}
		if !waits {
			resent = true
			if c.Log != nil {
				fmt.Fprintf(c.Log, "machine %s: sending %s again: %s\n", c.Machine, name, problem.Message)
			}
		}
	}
}

// AuthorityPending names the agent's refusal of a control it never started: it holds no current
// rental authority lease from its Hub (after boot, or while its Hub is out of reach).
const AuthorityPending = "machine.authority_pending"

// Wait on the agent's observed lifecycle, never infer readiness from elapsed time.
func (c *Maintenance) AwaitUpdateAdmission(ctx context.Context) (*RuntimeState, *exit.Error) {
	for {
		state, problem := c.State(ctx)
		if problem != nil && problem.ErrName() != "machine.runtime_starting" && problem.ErrName() != AuthorityPending {
			return nil, problem
		}
		if problem == nil {
			switch state.Phase {
			case "ready", "failed":
				return state, nil
			case "starting", "booting":
			default:
				return nil, exit.Named(exit.Structural, "machine.agent_update_required", "the machine does not report its Runtime lifecycle phase")
			}
		}
		select {
		case <-ctx.Done():
			return nil, exit.Named(exit.Canceled, "machine.readiness_observation_lost", "Runtime readiness observation ended")
		case <-time.After(time.Second):
		}
	}
}

// AwaitUpdate observes one accepted operation across application replacement.
// Readiness and transport gaps are observations, never authority to submit again.
func (c *Maintenance) AwaitUpdate(ctx context.Context, operation string) (*RuntimeState, *exit.Error) {
	return c.awaitUpdate(ctx, operation, false)
}

// AwaitUpdateOrPending observes until the operation activates or durably prepares
// a candidate whose activation waits for current work to drain.
func (c *Maintenance) AwaitUpdateOrPending(ctx context.Context, operation string) (*RuntimeState, *exit.Error) {
	return c.awaitUpdate(ctx, operation, true)
}

func (c *Maintenance) awaitUpdate(ctx context.Context, operation string, pending bool) (*RuntimeState, *exit.Error) {
	detached := func() (*RuntimeState, *exit.Error) {
		return nil, exit.Named(exit.Canceled, "machine.update_observation_lost", "update %s continues on the machine; observation ended: %s", operation, ctx.Err())
	}
	for {
		if ctx.Err() != nil {
			return detached()
		}
		state, problem := c.State(ctx)
		if ctx.Err() != nil {
			return detached()
		}
		if problem != nil {
			if problem.Code != exit.Unavailable {
				return nil, exit.Named(problem.Code, "machine.update_observation_lost", "update %s continues on the machine; observation ended: %s", operation, problem)
			}
		} else {
			if state.Update == nil || state.Update.Operation != operation {
				return nil, exit.Named(exit.Unavailable, "machine.update_observation_lost", "update %s is no longer the machine's current operation; its outcome must be inspected before another install", operation)
			}
			if updateTerminal(state.Update.State) || pending && state.Update.PendingActivation() {
				return state, nil
			}
		}
		select {
		case <-ctx.Done():
			return detached()
		case <-time.After(time.Second):
		}
	}
}
