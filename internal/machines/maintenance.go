package machines

import (
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
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
	Update *struct {
		Operation string `json:"operation"`
		State     string `json:"state"`
		Error     string `json:"error"`
		From      struct {
			Runtime  string `json:"runtime"`
			TensorFS string `json:"tensorfs"`
		} `json:"from"`
	} `json:"update"`
}

type Maintenance struct {
	Base    string
	Client  *http.Client
	Machine string
	Public  ed25519.PublicKey
	Sign    func([]byte) []byte
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
		return 0, exit.Unavailablef("the machine did not answer: %s", err)
	}
	defer response.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if response.StatusCode/100 != 2 {
		var refusal struct{ Code, Message string }
		if json.Unmarshal(raw, &refusal) == nil && refusal.Code == "runtime_starting" {
			return response.StatusCode, exit.Named(exit.Unavailable, "machine.runtime_starting", "%s", refusal.Message)
		}
		return response.StatusCode, exit.New(exit.Failed, "the machine refused %s %s (HTTP %d): %s", method, path, response.StatusCode, strings.TrimSpace(string(raw)))
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

// Wait on the agent's observed lifecycle, never infer readiness from elapsed time.
func (c *Maintenance) AwaitUpdateAdmission(ctx context.Context) (*RuntimeState, *exit.Error) {
	for {
		state, problem := c.State(ctx)
		if problem != nil && problem.ErrName() != "machine.runtime_starting" {
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
