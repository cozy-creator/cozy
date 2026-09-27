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
	Markers        []bool              `json:"markers"`
	RequiresPython string              `json:"requires_python"`
	Requirements   []string            `json:"requirements"`
	Extras         map[string][]string `json:"extras"`
}

func ActiveWheelRequirements(ctx context.Context, project string, extras []string, paths []string, python string) (RequirementSelection, *exit.Error) {
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
		if _, duplicate := metadata[name]; duplicate {
			return RequirementSelection{}, exit.New(exit.Conflict, "captured wheel metadata repeats a distribution")
		}
		metadata[name] = string(raw)
	}
	return ActiveRequirements(ctx, project, extras, metadata, python)
}

// ActiveRequirements binds selected extras using the standard PEP 508 parser,
// preserving target-specific markers. Captured metadata arrives on stdin; no
// captured module, .pth, interpreter or resolver runs. Every selected dependency contributes its own subtree.
func ActiveRequirements(ctx context.Context, project string, extras []string, metadata map[string]string, python string) (RequirementSelection, *exit.Error) {
	if extras == nil {
		extras = []string{}
	}
	return readActiveRequirements(ctx, map[string]any{"project": project, "extras": extras,
		"metadata": metadata, "python": python})
}

func readActiveRequirements(ctx context.Context, input map[string]any) (RequirementSelection, *exit.Error) {
	var result RequirementSelection
	raw, err := json.Marshal(input)
	if err != nil || len(raw) == 0 || int64(len(raw)) > maxLockBytes {
		return result, exit.New(exit.Validation, "captured requirements exceed the metadata bound")
	}
	// Metadata is evaluated by a trusted tool interpreter; the target version
	// stays in the input document and does not select or execute package code.
	command := exec.CommandContext(ctx, "uv", "run", "--isolated", "--no-project", "--no-config",
		"--python", ">=3.12", "--no-python-downloads", "--with", "packaging==26.2", "python", "-I", "-c", activeRequirementsScript)
	command.Env, command.Stdin = config.Frozen().Tool(), bytes.NewReader(raw) //cozy:stdin-value bounded metadata, never an interactive prompt
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
