package machines

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/google/uuid"
)

// updateLocked is the client of the same durable installer used by rental machines.
// It never kills the agent, invokes uv, or treats observer loss as a failed update.
func (h *Host) updateLocked(ctx context.Context, source Source) (*Installed, *exit.Error) {
	selectedHost := source.Host

	launch, problem := h.ensureLocked(ctx, "", nil)
	if problem != nil {
		return nil, problem
	}
	client, problem := h.maintenanceFor(launch)
	if problem != nil {
		return nil, problem
	}
	defer client.Client.CloseIdleConnections()
	state, problem := client.AwaitUpdateAdmission(ctx)
	if problem != nil {
		return nil, problem
	}
	if state.Update != nil && !updateTerminal(state.Update.State) {
		if state.Update.PendingActivation() {
			return nil, pendingUpdateConflict(state.Update)
		}
		return nil, exit.Named(exit.Conflict, "machine.update_in_progress", "the machine is already completing update %s", state.Update.Operation)
	}
	agent := "bundled"
	if selectedHost != "" {
		digest, err := fileDigest(selectedHost)
		if err != nil {
			return nil, exit.New(exit.NotFound, "cannot read the selected agent: %s", err)
		}
		if digest != state.Agent.SHA256 {
			return nil, exit.Named(exit.Structural, "machine.agent_bundle_required", "an existing machine can retain its running agent or select the agent in a Runtime wheel; bundle a different agent in the selected Runtime wheel and omit --host")
		}
		agent = "explicit"
	}
	operation := uuid.NewString()
	type choice struct {
		File    string `json:"file,omitempty"`
		SHA256  string `json:"sha256,omitempty"`
		Version string `json:"version,omitempty"`
	}
	body := struct {
		Operation string `json:"operation"`
		Pin       bool   `json:"pin"`
		Agent     string `json:"agent"`
		Runtime   choice `json:"runtime"`
		TensorFS  choice `json:"tensorfs"`
	}{Operation: operation, Agent: agent, Pin: source.Pinned || source.RuntimeWheel != "" || source.TensorFSWheel != ""}
	for _, item := range []struct {
		name, path string
		out        *choice
	}{{hostruntime.Distribution, source.RuntimeWheel, &body.Runtime}, {"tensorfs", source.TensorFSWheel, &body.TensorFS}} {
		if item.path == "" {
			version, problem := NewestPublished(ctx, item.name)
			if problem != nil {
				return nil, problem
			}
			item.out.Version = version
			continue
		}
		digest, err := fileDigest(item.path)
		if err != nil {
			return nil, exit.New(exit.NotFound, "cannot read update wheel: %s", err)
		}
		file, err := os.Open(item.path)
		if err != nil {
			return nil, exit.New(exit.NotFound, "cannot open update wheel: %s", err)
		}
		var staged struct {
			SHA256 string `json:"sha256"`
		}
		_, problem := client.Do(ctx, http.MethodPut, "/v1/machine/runtime/wheels/"+filepath.Base(item.path), file, &staged)
		file.Close()
		if problem != nil {
			return nil, problem
		}
		if staged.SHA256 != digest {
			return nil, exit.New(exit.Conflict, "the staged update wheel differs from the selected bytes")
		}
		item.out.File, item.out.SHA256 = filepath.Base(item.path), digest
	}
	raw, _ := json.Marshal(body)
	for {
		_, problem := client.Do(ctx, http.MethodPost, "/v1/machine/runtime/update", bytes.NewReader(raw), nil)
		if problem == nil {
			break
		}
		if problem.ErrName() != "machine.runtime_starting" {
			return nil, exit.Named(exit.Unavailable, "machine.update_observation_lost", "update %s may continue on the machine; observation ended: %s", operation, problem)
		}
		if _, problem = client.AwaitUpdateAdmission(ctx); problem != nil {
			return nil, problem
		}
	}
	state, problem = client.AwaitUpdateOrPending(ctx, operation)
	if problem != nil {
		return nil, problem
	}
	if state.Update.PendingActivation() {
		return pendingInstalled(state), nil
	}
	if update := state.Update; update.State != "succeeded" {
		return nil, exit.Named(exit.Failed, "machine.update_failed", "update %s: %s; machine now runs Runtime %s / TensorFS %s", operation, update.Error, state.Runtime, state.TensorFS)
	}
	if state.Agent.Selection != agent || state.Agent.SHA256 == "" {
		return nil, exit.Named(exit.Structural, "machine.agent_selection_unconfirmed", "update %s completed without confirming the selected running agent", operation)
	}
	installed := &Installed{InstalledAt: time.Now().UTC(), HostPinned: state.Agent.Selection == "explicit",
		Host:    installedArtifact{Name: "cozy-machine " + state.Agent.Version, SHA256: state.Agent.SHA256, Module: AgentModule},
		Runtime: installedArtifact{Name: hostruntime.Distribution + " " + state.Runtime}, TensorFS: installedArtifact{Name: "tensorfs " + state.TensorFS}}
	for _, pair := range []struct {
		file   string
		target *installedArtifact
	}{{source.RuntimeWheel, &installed.Runtime}, {source.TensorFSWheel, &installed.TensorFS}} {
		if pair.file != "" {
			pair.target.Name = filepath.Base(pair.file)
			pair.target.SHA256, _ = fileDigest(pair.file)
		}
	}
	if selectedHost != "" {
		installed.Host.Name = filepath.Base(selectedHost)
	}
	if err := h.recordInstalled(*installed); err != nil {
		return nil, exit.Internalf("updated Runtime but cannot retain installation metadata: %s", err)
	}
	return installed, nil
}

func pendingUpdateConflict(update *RuntimeUpdateState) *exit.Error {
	return exit.Named(exit.Conflict, "machine.update_in_progress", "Runtime update %s is awaiting activation; observe the existing candidate before starting another install", update.Operation)
}

// pendingInstalled reports the active pair and candidate operation without
// persisting the candidate as installed. Further installs wait until Runtime
// activates the candidate or rolls it back.
func pendingInstalled(state *RuntimeState) *Installed {
	return &Installed{
		Host:       installedArtifact{Name: "cozy-machine " + state.Agent.Version, SHA256: state.Agent.SHA256, Module: AgentModule},
		Runtime:    installedArtifact{Name: hostruntime.Distribution + " " + state.Runtime},
		TensorFS:   installedArtifact{Name: "tensorfs " + state.TensorFS},
		HostPinned: state.Agent.Selection == "explicit",
		Pending:    state.Update,
	}
}

func (h *Host) maintenanceFor(launch *Launch) (*Maintenance, *exit.Error) {
	// Observing an existing machine must never create new authority.
	if _, err := os.ReadFile(h.path("owner.pem")); err != nil {
		return nil, exit.New(exit.Credential, "the retained machine owner key is unavailable")
	}
	owner, problem := h.Owner()
	if problem != nil {
		return nil, problem
	}
	public, err := base64.RawURLEncoding.DecodeString(owner.PublicKey())
	if err != nil || len(public) != ed25519.PublicKeySize {
		return nil, exit.New(exit.Credential, "the machine owner key is unreadable")
	}
	pin, problem := h.Pin()
	if problem != nil {
		return nil, problem
	}
	transport := &http.Transport{TLSClientConfig: pin.TLSConfig()}
	return &Maintenance{Base: "https://" + launch.Addr, Machine: launch.WorkerID, Public: public,
		Sign: owner.Sign, Client: &http.Client{Transport: transport}}, nil
}

func updateTerminal(state string) bool {
	return state == "succeeded" || state == "rolled_back" || state == "failed"
}

func (h *Host) recordInstalled(installed Installed) error {
	raw, _ := json.MarshalIndent(installed, "", "  ")
	if err := writePrivate(h.path("installed.json"), raw); err != nil {
		return err
	}
	// Atomically replace the obsolete entry point; never follow its old symlink
	// into a shared CLI executable.
	target := filepath.Join(h.Root(), "usr/local/bin/pod-supervisor")
	file, err := os.CreateTemp(filepath.Dir(target), ".legacy-refusal-*")
	if err != nil {
		return err
	}
	defer os.Remove(file.Name())
	_, err = file.WriteString("#!/bin/sh\necho 'legacy machine control retired; use the current cozy CLI' >&2\nexit 6\n")
	if err == nil {
		err = file.Chmod(0755)
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(file.Name(), target)
}
