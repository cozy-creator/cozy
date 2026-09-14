package api

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/scratch"
)

func (s *Server) freezeMachineInputs(spec *orchestrator.Submission) (*scratch.Dir, *exit.Error) {
	if s.machineExecutions == nil || spec.LocalPackageDigest == "" {
		return nil, nil
	}
	jobs, problem := s.packages.JobsInstall(spec.InstallID)
	if problem != nil {
		return nil, problem
	}
	var entry *launch.Entrypoint
	for _, job := range jobs {
		if job.Name == spec.Entrypoint {
			entry = &launch.Entrypoint{Name: job.Name, Request: job.Request, Assets: job.Assets}
			break
		}
	}
	if entry == nil {
		return nil, exit.New(exit.Conflict, "captured root has no input declaration")
	}
	references, problem := launch.TreeInputRefs(entry, spec.Payload)
	if problem != nil {
		return nil, problem
	}
	sources := map[string]string{}
	for _, pair := range spec.Trees {
		name, directory, ok := strings.Cut(pair, "=")
		if !ok || name == "" || directory == "" || sources[name] != "" {
			return nil, exit.New(exit.Validation, "input tree aliases must be unique references and directories")
		}
		absolute, err := filepath.Abs(directory)
		if err != nil {
			return nil, exit.New(exit.Validation, "input tree directory is invalid")
		}
		sources[name] = absolute
	}
	wanted := map[string]int64{}
	for field, ref := range references {
		if sources[ref] == "" {
			return nil, exit.New(exit.Validation, "Tree field %s needs --input-tree %s=<directory>", field, ref)
		}
		policy, _ := launch.AssetSpec(entry, field)
		maximum := policy.MaxBytes
		if maximum <= 0 {
			maximum = inputasset.MaxRootInputBytes
		}
		if prior, ok := wanted[ref]; !ok || maximum < prior {
			wanted[ref] = maximum
		}
	}
	if len(wanted) != len(sources) {
		return nil, exit.New(exit.Validation, "input tree alias does not name a declared Tree argument")
	}
	if len(spec.Assets) == 0 && len(wanted) == 0 {
		return nil, nil
	}
	spec.RequestID = records.NewID("job")
	stage, problem := scratch.Named(s.layout.Tmp, spec.RequestID)
	if problem != nil {
		return nil, problem
	}
	fail := func(problem *exit.Error) (*scratch.Dir, *exit.Error) { stage.Release(); return nil, problem }
	captured := map[string]records.AssetBinding{}
	var total int64
	for index, binding := range spec.Assets {
		if binding.Native != nil || binding.Snapshot != nil {
			return fail(exit.New(exit.Validation, "root byte intake must capture its local input"))
		}
		digest := sha256.Sum256([]byte("file:" + binding.FieldPath))
		frozen, problem := inputasset.CaptureFile(filepath.Join(stage.Path, hex.EncodeToString(digest[:])), binding)
		if problem != nil {
			return fail(problem)
		}
		total += frozen.Snapshot.ContentBytes
		spec.Assets[index] = frozen
	}
	if total > inputasset.MaxRootInputBytes {
		return fail(exit.New(exit.Validation, "root byte inputs exceed 256 MiB"))
	}
	capturedBytes := total
	for alias, maximum := range wanted {
		digest := sha256.Sum256([]byte("tree:" + alias))
		binding, problem := inputasset.CaptureTree(filepath.Join(stage.Path, hex.EncodeToString(digest[:])), alias, sources[alias], min(maximum, inputasset.MaxRootInputBytes-capturedBytes))
		if problem != nil {
			return fail(problem)
		}
		capturedBytes += binding.Snapshot.ContentBytes
		binding.Snapshot.Reference = alias
		captured[alias] = binding
	}
	replacements := map[string]string{}
	for field, alias := range references {
		binding := captured[alias]
		binding.FieldPath = field
		total += binding.Snapshot.ContentBytes
		if total > inputasset.MaxRootInputBytes {
			return fail(exit.New(exit.Validation, "root byte inputs exceed 256 MiB"))
		}
		spec.Assets = append(spec.Assets, binding)
		replacements[field] = binding.Digest
	}
	raw, problem := launch.ReplaceTreeInputRefs(spec.Payload, replacements)
	if problem != nil {
		return fail(problem)
	}
	normalized, err := canonical.NormalizeJCS(raw)
	if err != nil {
		return fail(exit.New(exit.Validation, "captured input payload is not canonical"))
	}
	spec.Payload = normalized
	spec.Trees = nil
	sort.Slice(spec.Assets, func(i, j int) bool { return spec.Assets[i].FieldPath < spec.Assets[j].FieldPath })
	return stage, nil
}

func uncapturedRootBytes(assets []records.AssetBinding) bool {
	for _, asset := range assets {
		if asset.Snapshot == nil {
			return true
		}
	}
	return false
}

// An idempotent replay uses the already captured input, even if its source was
// edited or removed. Changed content references still change the request identity.
func replayMachineInputSnapshots(payload []byte, assets []records.AssetBinding, trees []string, recorded records.Request) ([]byte, []records.AssetBinding, *exit.Error) {
	replacements := map[string]string{}
	aliases := map[string]bool{}
	var document map[string]any
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	if decoder.Decode(&document) != nil {
		return nil, nil, exit.New(exit.Validation, "replayed input payload is unreadable")
	}
	for _, prior := range recorded.Assets {
		if prior.Snapshot == nil {
			continue
		}
		if prior.Snapshot.Reference != "" {
			aliases[prior.Snapshot.Reference] = true
			current := any(document)
			for _, part := range strings.Split(prior.FieldPath, ".") {
				fields, ok := current.(map[string]any)
				if !ok {
					return nil, nil, exit.New(exit.Conflict, "replayed Tree field changed")
				}
				current = fields[part]
			}
			if current == prior.Snapshot.Reference {
				replacements[prior.FieldPath] = prior.Digest
			} else if current != prior.Digest {
				return nil, nil, exit.New(exit.Conflict, "replayed Tree reference changed")
			}
		}
		found := false
		for index, asset := range assets {
			if asset.FieldPath == prior.FieldPath {
				if asset.Digest != prior.Digest || asset.Length != prior.Length || asset.MediaType != prior.MediaType || asset.Order != prior.Order {
					return nil, nil, exit.New(exit.Conflict, "replayed byte input changed")
				}
				assets[index] = prior
				found = true
				break
			}
		}
		if !found {
			assets = append(assets, prior)
		}
	}
	if len(aliases) > 0 {
		for _, pair := range trees {
			alias, _, ok := strings.Cut(pair, "=")
			if !ok || !aliases[alias] {
				return nil, nil, exit.New(exit.Conflict, "replayed Tree aliases changed")
			}
		}
	}
	if len(replacements) > 0 {
		var problem *exit.Error
		payload, problem = launch.ReplaceTreeInputRefs(payload, replacements)
		if problem != nil {
			return nil, nil, problem
		}
	}
	normalized, err := canonical.NormalizeJCS(payload)
	if err != nil {
		return nil, nil, exit.New(exit.Validation, "replayed input is not canonical")
	}
	sort.Slice(assets, func(i, j int) bool { return assets[i].FieldPath < assets[j].FieldPath })
	return normalized, assets, nil
}
