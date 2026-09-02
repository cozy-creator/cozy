package packagepublish

// cl-078: the author derives. When the descriptor declares source profiles, the
// publish runs cozy-model-contract-proof in the project's own locked venv over
// the hub-resolved (snapshot, config, hardware variants) per declared pair, and
// uploads the produced envelope — result.json plus the digest-named contract and
// code-topology documents — beside the descriptor. The hub validates those
// documents statically and never executes package code (th-106).

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/wheel"
)

// DeriveInput is one resolved source profile: the exact snapshot manifest, the
// canonical construction config bytes, and the hub's active hardware variants.
type DeriveInput struct {
	Snapshot     string
	ConfigDigest string
	ConfigBytes  []byte
	Variants     []string
}

// ProfileResolver answers one declared "org/model/release/lane[/config]"
// profile from the hub. config is the optional fifth segment (th-114): the
// named construction config of a multi-config checkpoint, empty when the
// profile declares none.
type ProfileResolver func(ctx context.Context, model, release, lane, config string) (DeriveInput, *exit.Error)

var evidenceDocName = regexp.MustCompile(`^[0-9a-f]{64}\.json$`)

type declaredBinding struct {
	path    string
	profile string
}

// declaredBindings reads the descriptor's profiled model slots. The descriptor
// was just produced by describe, so this is a projection, not a validation.
func declaredBindings(descriptorPath string) ([]declaredBinding, *exit.Error) {
	raw, err := os.ReadFile(descriptorPath)
	if err != nil {
		return nil, exit.Named(exit.Structural, "package_descriptor_invalid",
			"the built descriptor is unreadable: %v", err)
	}
	var document struct {
		Entrypoints []struct {
			Models []struct {
				Path          string `json:"path"`
				SourceProfile string `json:"source_profile"`
			} `json:"models"`
		} `json:"entrypoints"`
		Jobs []struct {
			Models []struct {
				Path          string `json:"path"`
				SourceProfile string `json:"source_profile"`
			} `json:"models"`
		} `json:"jobs"`
	}
	if err := json.Unmarshal(raw, &document); err != nil {
		return nil, exit.Named(exit.Structural, "package_descriptor_invalid",
			"the built descriptor is not JSON: %v", err)
	}
	out := []declaredBinding{}
	for _, callable := range document.Entrypoints {
		for _, slot := range callable.Models {
			if slot.SourceProfile != "" {
				out = append(out, declaredBinding{path: slot.Path, profile: slot.SourceProfile})
			}
		}
	}
	for _, callable := range document.Jobs {
		for _, slot := range callable.Models {
			if slot.SourceProfile != "" {
				out = append(out, declaredBinding{path: slot.Path, profile: slot.SourceProfile})
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].path < out[j].path })
	return out, nil
}

// DeriveEvidence produces the publish-time envelope for every declared pair.
// A package with no declared profiles produces none, and that is the design:
// static grading exists exactly for the pairs the author declared and tested.
func (p *Package) DeriveEvidence(ctx context.Context, org string, resolve ProfileResolver) *exit.Error {
	if p.Root == "" || p.Descriptor == "" || p.Wheel == "" {
		return exit.Internalf("DeriveEvidence before Build")
	}
	bindings, problem := declaredBindings(p.Descriptor)
	if problem != nil {
		return problem
	}
	if len(bindings) == 0 {
		return nil
	}
	inputs := map[string]DeriveInput{}
	stage := filepath.Join(p.Root, "evidence-inputs")
	if err := os.MkdirAll(stage, 0o700); err != nil {
		return exit.Internalf("stage evidence inputs: %v", err)
	}
	requestBindings := make([]map[string]any, 0, len(bindings))
	for i, binding := range bindings {
		// Model releases are mutable pointers with unordered labels (#689), so a
		// model has no default release and the declared profile must name one:
		// <org>/<model>/<release>/<lane>. A fifth segment (th-114) names the
		// construction config of a multi-config checkpoint.
		segments := strings.Split(binding.profile, "/")
		if len(segments) < 4 || len(segments) > 5 ||
			(len(segments) == 5 && segments[4] == "") {
			return exit.Named(exit.Validation, "source_profile_unresolvable",
				"%s declares source profile %q; evidence resolution needs org/model/release/lane with an optional /config",
				binding.path, binding.profile).
				WithRemedy("declare the profile as <org>/<model>/<release>/<lane>[/<config>]")
		}
		config := ""
		if len(segments) == 5 {
			config = segments[4]
		}
		input, held := inputs[binding.profile]
		if !held {
			resolved, problem := resolve(ctx, segments[0]+"/"+segments[1], segments[2], segments[3], config)
			if problem != nil {
				return problem
			}
			if len(resolved.Variants) == 0 {
				return exit.Named(exit.Unavailable, "derive_inputs_unavailable",
					"the hub advertises no active hardware variants to derive against")
			}
			inputs[binding.profile] = resolved
			input = resolved
		}
		configPath := filepath.Join(stage, fmt.Sprintf("config-%02d.json", i))
		if err := os.WriteFile(configPath, input.ConfigBytes, 0o600); err != nil {
			return exit.Internalf("write evidence config: %v", err)
		}
		variants := append([]string(nil), input.Variants...)
		sort.Strings(variants)
		requestBindings = append(requestBindings, map[string]any{
			"path":     binding.path,
			"snapshot": input.Snapshot,
			"config": map[string]any{"digest": input.ConfigDigest,
				"length": len(input.ConfigBytes), "path": configPath},
			"hardware_variants": variants,
		})
	}
	descriptorRaw, err := os.ReadFile(p.Descriptor)
	if err != nil {
		return exit.Internalf("read descriptor: %v", err)
	}
	wheelIdentity, problem := wheel.InspectIdentity(p.Wheel)
	if problem != nil {
		return problem
	}
	wheelDigest, problem := dependencyDigest(p.Wheel)
	if problem != nil {
		return problem
	}
	descriptorSum := sha256.Sum256(descriptorRaw)
	request := map[string]any{
		"bindings": requestBindings,
		"package": map[string]any{
			"package": org + "/" + p.Name,
			"release": p.Release,
			"project_wheel": map[string]any{
				"distribution": wheelIdentity.Distribution,
				"ref":          map[string]any{"digest": wheelDigest, "length": wheelIdentity.Length},
				"version":      wheelIdentity.Version,
			},
		},
		"package_descriptor": map[string]any{
			"digest": "sha256:" + hex.EncodeToString(descriptorSum[:]),
			"length": len(descriptorRaw),
			"path":   p.Descriptor,
		},
		"project": p.Tree,
	}
	requestRaw, err := json.Marshal(request)
	if err != nil {
		return exit.Internalf("encode evidence request: %v", err)
	}
	requestPath := filepath.Join(stage, "request.json")
	if err := os.WriteFile(requestPath, requestRaw, 0o600); err != nil {
		return exit.Internalf("write evidence request: %v", err)
	}
	outDir := filepath.Join(p.Root, "evidence-out")
	cmd := exec.CommandContext(ctx, "uv", "run", "--locked", "--no-progress",
		"cozy-model-contract-proof", requestPath, outDir)
	cmd.Dir = p.Tree
	cmd.Env = config.Frozen().Tool()
	output, err := cmd.CombinedOutput()
	if err != nil {
		detail := strings.Join(strings.Fields(string(output)), " ")
		if len(detail) > 4096 {
			detail = detail[:4096]
		}
		return exit.Named(exit.Validation, "derive_evidence_refused",
			"cozy-model-contract-proof refused: %s", detail).
			WithRemedy("repair the declared model pairs, then publish again")
	}
	// The proof publishes a read-only tree (0o500 directories); reopen it so
	// Package.Close's RemoveAll can reap the disposable root.
	_ = filepath.WalkDir(outDir, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr == nil && entry.IsDir() {
			_ = os.Chmod(path, 0o700)
		}
		return nil
	})
	evidence := map[string]string{}
	index := filepath.Join(outDir, "result.json")
	if _, err := os.Stat(index); err != nil {
		return exit.Named(exit.Structural, "derive_evidence_invalid",
			"the evidence run produced no result.json")
	}
	evidence["evidence/result.json"] = index
	for _, directory := range []string{"contracts", "code-topologies"} {
		entries, err := os.ReadDir(filepath.Join(outDir, directory))
		if err != nil {
			return exit.Named(exit.Structural, "derive_evidence_invalid",
				"the evidence run produced no %s documents", directory)
		}
		for _, entry := range entries {
			if entry.IsDir() || !evidenceDocName.MatchString(entry.Name()) {
				return exit.Named(exit.Structural, "derive_evidence_invalid",
					"the evidence run produced an unexpected %s entry %q", directory, entry.Name())
			}
			evidence["evidence/"+directory+"/"+entry.Name()] =
				filepath.Join(outDir, directory, entry.Name())
		}
	}
	p.Evidence = evidence
	return nil
}
