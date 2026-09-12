package packagepublish

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"io"
	"os/exec"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/wheel"
)

//go:embed active_requirements.py
var activeRequirementsScript string

type RequirementSelection struct {
	Requirements []string            `json:"requirements"`
	Extras       map[string][]string `json:"extras"`
}

func ActiveWheelRequirements(ctx context.Context, project string, extras []string, paths []string) (RequirementSelection, *exit.Error) {
	metadata := map[string]string{}
	for _, path := range paths {
		raw, problem := wheel.Metadata(path)
		if problem != nil {
			return RequirementSelection{}, problem
		}
		name, _, problem := wheel.MetadataIdentity(raw)
		if problem != nil {
			return RequirementSelection{}, problem
		}
		name = normalizedProjectName(name)
		if !ImageOwnedDistribution(name) {
			if _, duplicate := metadata[name]; duplicate {
				return RequirementSelection{}, exit.New(exit.Conflict, "captured wheel metadata repeats a distribution")
			}
			metadata[name] = string(raw)
		}
	}
	return ActiveRequirements(ctx, project, extras, metadata)
}

// ActiveRequirements evaluates standard PEP 508 markers in a trusted tool
// environment. Captured metadata arrives on stdin; no captured module, .pth,
// interpreter or resolver is executed. Image wheels own their dependency subtree.
func ActiveRequirements(ctx context.Context, project string, extras []string, metadata map[string]string) (RequirementSelection, *exit.Error) {
	var result RequirementSelection
	if extras == nil {
		extras = []string{}
	}
	raw, err := json.Marshal(map[string]any{"project": project, "extras": extras,
		"metadata": metadata, "image": json.RawMessage(baseDistributionsJSON)})
	if err != nil || len(raw) == 0 || int64(len(raw)) > maxLockBytes {
		return result, exit.New(exit.Validation, "captured requirements exceed the metadata bound")
	}
	command := exec.CommandContext(ctx, "uv", "run", "--isolated", "--no-project", "--no-config",
		"--python", "3.12", "--with", "packaging==26.2", "python", "-I", "-c", activeRequirementsScript)
	command.Env, command.Stdin = config.Frozen().Tool(), bytes.NewReader(raw)
	out, err := command.StdoutPipe()
	if err != nil || command.Start() != nil {
		return result, exit.New(exit.Structural, "cannot start the requirement metadata reader")
	}
	answer, readErr := io.ReadAll(io.LimitReader(out, maxLockBytes+1))
	if int64(len(answer)) > maxLockBytes {
		_ = command.Process.Kill()
	}
	err = command.Wait()
	if err != nil || readErr != nil || int64(len(answer)) > maxLockBytes || json.Unmarshal(answer, &result) != nil {
		return result, exit.New(exit.Validation, "captured dependency requirements could not be evaluated")
	}
	return result, nil
}

func (s RequirementSelection) ImageRequirements() []string {
	var out []string
	for _, raw := range s.Requirements {
		if req, problem := parseRequirement(raw); problem == nil && ImageOwnedDistribution(req.name) {
			out = append(out, raw)
		}
	}
	return out
}
