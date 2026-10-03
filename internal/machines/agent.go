package machines

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

const AgentModule = "github.com/cozy-creator/cozy-runtime/machine-agent"
const HubAccessCapability = "hub-access/1"
const RuntimeUpdateCapability = "runtime-update/1"
const BootstrapCapability = "machine-bootstrap/1"

func (h *Host) bundledAgent() string { return filepath.Join(h.python(), "bin/cozy-machine") }

// An agent is accepted by its own identity (`version --json`): its name, a usable wire range
// and the capabilities this client drives. The implementation (the Go agent of a Runtime
// wheel, the Rust machine, a later one) and release numbers are descriptive.
func compatibleAgent(ctx context.Context, path string) bool {
	out, err := exec.CommandContext(ctx, path, "version", "--json").Output()
	var version struct {
		Name             string   `json:"name"`
		WireMinor        uint32   `json:"wire_minor"`
		MinimumWireMinor uint32   `json:"minimum_wire_minor"`
		Capabilities     []string `json:"capabilities"`
	}
	return err == nil && json.Unmarshal(out, &version) == nil && version.Name == "cozy-machine" &&
		version.WireMinor >= pb.MinCompatibleWireMinor && version.MinimumWireMinor <= pb.WireMinor &&
		slices.Contains(version.Capabilities, HubAccessCapability) && slices.Contains(version.Capabilities, RuntimeUpdateCapability) &&
		slices.Contains(version.Capabilities, BootstrapCapability)
}

func (h *Host) defaultAgent(ctx context.Context) (string, *exit.Error) {
	path := h.bundledAgent()
	if !compatibleAgent(ctx, path) {
		return "", exit.Named(exit.Structural, "machine.agent_update_required", "the Runtime wheel must contain a cozy-machine with %s, %s and %s", HubAccessCapability, RuntimeUpdateCapability, BootstrapCapability).
			WithRemedy("install the current published Runtime pair, or supply a current development agent with --host")
	}
	return path, nil
}

// Link the wheel-owned executable so later Runtime updates also update the agent.
func (h *Host) linkAgent() (installedArtifact, error) {
	source := h.bundledAgent()
	digest, err := fileDigest(source)
	if err != nil {
		return installedArtifact{}, err
	}
	artifact := installedArtifact{Name: "cozy-machine", SHA256: digest, Module: HostModule(source)}
	target := filepath.Join(h.Root(), "usr/local/bin/cozy-machine")
	if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
		return artifact, err
	}
	file, err := os.CreateTemp(filepath.Dir(target), ".agent-link-*")
	if err != nil {
		return artifact, err
	}
	staged := file.Name()
	file.Close()
	defer os.Remove(staged)
	if err := os.Remove(staged); err != nil {
		return artifact, err
	}
	if err := os.Symlink(source, staged); err != nil {
		return artifact, err
	}
	return artifact, os.Rename(staged, target)
}
