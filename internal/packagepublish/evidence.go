package packagepublish

// cr-077: the author derives SLOT FACTS. A slot's entire declaration is its class
// annotation; per slot class the publish derives exactly two facts — the shape-only
// code topology digest and the acceptable encodings set — by running the class's
// factory on the meta device in the project's own locked venv. The construction SEED
// (which checkpoint snapshot/config the factory runs against) comes from the slot's
// default `package.toml [bindings]` entry; its identity never enters the published
// facts. The hub validates the produced envelope statically and never executes
// package code (th-106); compatibility is graded on demand from
// grade(code topology × checkpoint topology) + encoding/device qualification.

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
	"github.com/pelletier/go-toml/v2"
)

// DeriveInput is one resolved construction seed: the exact snapshot manifest, the
// canonical construction config bytes (the runtime-parity document — one config, or
// the assembled name-keyed mapping of every named config), and the hub's active
// hardware variants.
type DeriveInput struct {
	Snapshot     string
	ConfigDigest string
	ConfigBytes  []byte
	Variants     []string
}

// SeedResolver answers one fully named default binding — model, release, lane —
// from the hub's derive-inputs read.
type SeedResolver func(ctx context.Context, model, release, lane string) (DeriveInput, *exit.Error)

var evidenceDocName = regexp.MustCompile(`^[0-9a-f]{64}\.json$`)

type declaredSlot struct {
	path  string
	class string
}

// declaredSlots reads every model slot of the descriptor. The descriptor was just
// produced by describe, so this is a projection, not a validation.
func declaredSlots(descriptorPath string) ([]declaredSlot, *exit.Error) {
	raw, err := os.ReadFile(descriptorPath)
	if err != nil {
		return nil, exit.Named(exit.Structural, "package_descriptor_invalid",
			"the built descriptor is unreadable: %v", err)
	}
	var document struct {
		Entrypoints []struct {
			Models []struct {
				Path  string `json:"path"`
				Class string `json:"class"`
			} `json:"models"`
		} `json:"entrypoints"`
		Jobs []struct {
			Models []struct {
				Path  string `json:"path"`
				Class string `json:"class"`
			} `json:"models"`
		} `json:"jobs"`
	}
	if err := json.Unmarshal(raw, &document); err != nil {
		return nil, exit.Named(exit.Structural, "package_descriptor_invalid",
			"the built descriptor is not JSON: %v", err)
	}
	out := []declaredSlot{}
	for _, callable := range document.Entrypoints {
		for _, slot := range callable.Models {
			out = append(out, declaredSlot{path: slot.Path, class: slot.Class})
		}
	}
	for _, callable := range document.Jobs {
		for _, slot := range callable.Models {
			out = append(out, declaredSlot{path: slot.Path, class: slot.Class})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].path < out[j].path })
	return out, nil
}

// seedRef is one fully named default binding: the checkpoint the class's factory
// derives against. Anything less than model+release+lane is not a derivation seed.
type seedRef struct {
	model   string
	release string
	lane    string
}

func (s seedRef) key() string { return s.model + "/" + s.release + "/" + s.lane }

// slotSeeds resolves each slot's default binding from package.toml [bindings],
// with the runtime's exact precedence: the path spelling wins over the class
// spelling. A slot with no complete binding derives no facts and is reported as
// skipped — publishable, but ungradeable until the author names a default. Two
// slots of one class resolving different seeds refuse: a class carries exactly
// one topology fact.
func slotSeeds(tree string, slots []declaredSlot) (map[string]seedRef, []string, *exit.Error) {
	raw, err := os.ReadFile(filepath.Join(tree, "package.toml"))
	if err != nil {
		return nil, nil, exit.Named(exit.Validation, "package_config_unreadable",
			"package.toml is unreadable: %v", err)
	}
	var document struct {
		Bindings map[string]struct {
			Model   string `toml:"model"`
			Release string `toml:"release"`
			Lane    string `toml:"lane"`
		} `toml:"bindings"`
	}
	if err := toml.Unmarshal(raw, &document); err != nil {
		return nil, nil, exit.Named(exit.Validation, "package_config_invalid",
			"package.toml is not valid TOML: %v", err)
	}
	known := map[string]bool{}
	for _, slot := range slots {
		known[slot.path] = true
		known[slot.class] = true
	}
	for key := range document.Bindings {
		if !known[key] {
			return nil, nil, exit.Named(exit.Validation, "slot_binding_unknown",
				"package.toml binds %q, which this release declares no slot for", key).
				WithRemedy("bind a declared slot path or model class, or remove the entry")
		}
	}
	seeds := map[string]seedRef{}
	skipped := []string{}
	for _, slot := range slots {
		entry, held := document.Bindings[slot.path]
		if !held {
			entry, held = document.Bindings[slot.class]
		}
		if !held || entry.Model == "" || entry.Release == "" || entry.Lane == "" {
			if _, derived := seeds[slot.class]; !derived {
				skipped = append(skipped, slot.class)
			}
			continue
		}
		seed := seedRef{model: entry.Model, release: entry.Release, lane: entry.Lane}
		if previous, held := seeds[slot.class]; held && previous != seed {
			return nil, nil, exit.Named(exit.Validation, "slot_binding_divergent",
				"class %s is bound to both %s and %s; one slot class carries one topology fact",
				slot.class, previous.key(), seed.key()).
				WithRemedy("bind every slot of the class to one default checkpoint")
		}
		seeds[slot.class] = seed
	}
	// A class is skipped only when NO slot of it produced a seed.
	kept := []string{}
	for _, class := range skipped {
		if _, derived := seeds[class]; !derived {
			kept = append(kept, class)
		}
	}
	sort.Strings(kept)
	return seeds, dedupe(kept), nil
}

func dedupe(values []string) []string {
	out := values[:0]
	for i, value := range values {
		if i == 0 || values[i-1] != value {
			out = append(out, value)
		}
	}
	return out
}

// DeriveEvidence produces the publish-time slot-facts envelope: per slot class with
// a complete default binding, the shape-only code topology and the acceptable
// encodings set. A package with no seeded slots produces none, and that is the
// design: preflight for its slots answers "no facts" until a default is named.
func (p *Package) DeriveEvidence(ctx context.Context, org string, resolve SeedResolver) *exit.Error {
	if p.Root == "" || p.Descriptor == "" || p.Wheel == "" {
		return exit.Internalf("DeriveEvidence before Build")
	}
	slots, problem := declaredSlots(p.Descriptor)
	if problem != nil {
		return problem
	}
	if len(slots) == 0 {
		return nil
	}
	seeds, skipped, problem := slotSeeds(p.Tree, slots)
	if problem != nil {
		return problem
	}
	p.SlotFactsSkipped = skipped
	if len(seeds) == 0 {
		return nil
	}
	classes := make([]string, 0, len(seeds))
	for class := range seeds {
		classes = append(classes, class)
	}
	sort.Strings(classes)
	stage := filepath.Join(p.Root, "evidence-inputs")
	if err := os.MkdirAll(stage, 0o700); err != nil {
		return exit.Internalf("stage evidence inputs: %v", err)
	}
	inputs := map[string]DeriveInput{}
	requestSlots := make([]map[string]any, 0, len(classes))
	for i, class := range classes {
		seed := seeds[class]
		input, held := inputs[seed.key()]
		if !held {
			resolved, problem := resolve(ctx, seed.model, seed.release, seed.lane)
			if problem != nil {
				return problem
			}
			if len(resolved.Variants) == 0 {
				return exit.Named(exit.Unavailable, "derive_inputs_unavailable",
					"the hub advertises no active hardware variants to derive against")
			}
			inputs[seed.key()] = resolved
			input = resolved
		}
		configPath := filepath.Join(stage, fmt.Sprintf("config-%02d.json", i))
		if err := os.WriteFile(configPath, input.ConfigBytes, 0o600); err != nil {
			return exit.Internalf("write evidence config: %v", err)
		}
		variants := append([]string(nil), input.Variants...)
		sort.Strings(variants)
		requestSlots = append(requestSlots, map[string]any{
			"class":    class,
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
		"slots": requestSlots,
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
			WithRemedy("repair the slot classes or their default bindings, then publish again")
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
	entries, err := os.ReadDir(filepath.Join(outDir, "code-topologies"))
	if err != nil {
		return exit.Named(exit.Structural, "derive_evidence_invalid",
			"the evidence run produced no code-topologies documents")
	}
	for _, entry := range entries {
		if entry.IsDir() || !evidenceDocName.MatchString(entry.Name()) {
			return exit.Named(exit.Structural, "derive_evidence_invalid",
				"the evidence run produced an unexpected code-topologies entry %q", entry.Name())
		}
		evidence["evidence/code-topologies/"+entry.Name()] =
			filepath.Join(outDir, "code-topologies", entry.Name())
	}
	p.Evidence = evidence
	p.SlotFacts = classes
	return nil
}
