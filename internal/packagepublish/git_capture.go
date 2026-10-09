package packagepublish

import (
	"context"
	"encoding/json"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/wheel"
	"github.com/pelletier/go-toml/v2"
)

// retainLockedGitWheels changes only source transport inside an invocation-owned
// copy. The locked commit is built once here; workers install retained wheel bytes
// without Git, repository access, or another source build.
func retainLockedGitWheels(ctx context.Context, root string, selected *hostruntime.PythonInterpreter, extras []string) *exit.Error {
	lockPath := filepath.Join(root, "uv.lock")
	raw, err := os.ReadFile(lockPath)
	if err != nil {
		return exit.Internalf("cannot read captured dependency lock: %s", err)
	}
	var before map[string]any
	if err := toml.Unmarshal(raw, &before); err != nil {
		return exit.New(exit.Validation, "captured dependency lock is invalid")
	}
	entries, _ := before["package"].([]any)
	locked := map[string]map[string]any{}
	for _, item := range entries {
		entry, _ := item.(map[string]any)
		source, _ := entry["source"].(map[string]any)
		if _, ok := source["git"].(string); !ok {
			continue
		}
		name, _ := entry["name"].(string)
		name = normalizedProjectName(name)
		if name == "" || locked[name] != nil {
			return exit.New(exit.Validation, "captured Git dependency has ambiguous locked sources")
		}
		locked[name] = entry
	}
	if len(locked) == 0 {
		return nil
	}
	metadata, problem := readProjectDocument(filepath.Join(root, "pyproject.toml"))
	if problem != nil {
		return problem
	}
	scan := &dependencyCollector{byName: map[string]dependencyRecord{}, extras: map[string]map[string]bool{}, stack: map[string]bool{root: true}, scanOnly: true}
	scan.byName[normalizedProjectName(metadata.Project.Name)] = dependencyRecord{source: root, version: metadata.Project.Version}
	declaredExtras := append([]string(nil), extras...)
	for name := range metadata.Project.OptionalDependencies {
		declaredExtras = append(declaredExtras, name)
	}
	if problem := scan.collectProject(root, metadata, declaredExtras, true); problem != nil {
		return problem
	}
	for name, entry := range locked {
		pin, declared := scan.gitPins[name]
		source := entry["source"].(map[string]any)["git"].(string)
		parsed, parseErr := url.Parse(source)
		if !declared {
			return exit.Named(exit.Validation, "registry_dependency_git_undeclared", "%s is locked to a Git source this package does not declare", name)
		}
		if parseErr != nil || parsed.Fragment != pin.commit {
			return exit.Named(exit.Conflict, "private_dependency_git_lock_drift", "locked Git source for %s differs from its declared commit", name)
		}
		parsed.RawQuery, parsed.Fragment = "", ""
		if strings.TrimSuffix(parsed.String(), "/") != strings.TrimSuffix(pin.url, "/") {
			return exit.Named(exit.Conflict, "private_dependency_git_lock_drift", "locked Git repository for %s differs from its declaration", name)
		}
	}
	if selected == nil {
		python, problem := hostruntime.ProjectPython(ctx, root)
		if problem != nil {
			return problem
		}
		selected = &python
	}
	staging, err := os.MkdirTemp("", "cozy-captured-git-")
	if err != nil {
		return exit.Internalf("cannot stage captured Git wheels: %s", err)
	}
	defer os.RemoveAll(staging)
	// Export the already locked target selection; do not build Git dependencies
	// which only belong to an inactive extra or another environment.
	exported, problem := exportLockedRegistry(ctx, root, staging, *selected, extras)
	if problem != nil {
		return problem
	}
	var selection registryLock
	if toml.Unmarshal(exported, &selection) != nil {
		return exit.New(exit.Validation, "captured Git selection is invalid")
	}
	markers := make([]string, len(selection.Packages))
	for i, entry := range selection.Packages {
		markers[i] = entry.Marker
	}
	active, problem := readActiveRequirements(ctx, map[string]any{"markers": markers, "python": selected.Version})
	if problem != nil {
		return problem
	}
	if len(active.Markers) != len(selection.Packages) {
		return exit.New(exit.Validation, "captured Git marker selection is incomplete")
	}
	wanted := map[string]bool{}
	for i, entry := range selection.Packages {
		if active.Markers[i] && entry.VCS != nil {
			wanted[normalizedProjectName(entry.Name)] = true
		}
	}
	for name := range locked {
		if !wanted[name] {
			delete(locked, name)
		}
	}
	if len(locked) == 0 {
		return nil
	}
	names := make([]string, 0, len(locked))
	for name := range locked {
		names = append(names, name)
	}
	sort.Strings(names)
	builder := &dependencyCollector{ctx: ctx, stage: staging, python: selected.Executable, byName: map[string]dependencyRecord{}}
	versions, paths := map[string]string{}, map[string]string{}
	provenance := []map[string]string{}
	for _, name := range names {
		pin := scan.gitPins[name]
		if problem := builder.collectGit(requirement{name: name}, pin); problem != nil {
			return problem
		}
		built := builder.wheels[len(builder.wheels)-1]
		identity, problem := wheel.InspectIdentity(built.Path)
		if problem != nil {
			return problem
		}
		version, _ := locked[name]["version"].(string)
		if !wheel.SameVersion(identity.Version, version) {
			return exit.Named(exit.Conflict, "private_dependency_git_lock_drift", "Git dependency %s built version %s instead of locked %s", name, identity.Version, version)
		}
		rel := filepath.ToSlash(filepath.Join(".cozy-dependencies", "git", name, built.Filename))
		to := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(to), 0700); err != nil {
			return exit.Internalf("cannot retain captured Git wheel: %s", err)
		}
		if problem := copySnapshotFile(built.Path, to, MaxDependencyWheelBytes); problem != nil {
			return problem
		}
		digest, problem := dependencyDigest(to)
		if problem != nil {
			return problem
		}
		versions[name], paths[name] = version, rel
		provenance = append(provenance, map[string]string{"name": name, "repository": pin.url, "commit": pin.commit, "version": version, "wheel": rel, "digest": digest})
	}
	projectPath := filepath.Join(root, "pyproject.toml")
	raw, err = os.ReadFile(projectPath)
	if err != nil {
		return exit.Internalf("cannot read captured project: %s", err)
	}
	var document map[string]any
	if err := toml.Unmarshal(raw, &document); err != nil {
		return exit.New(exit.Validation, "captured project metadata is invalid")
	}
	nested := func(parent map[string]any, key string) map[string]any {
		if child, ok := parent[key].(map[string]any); ok {
			return child
		}
		child := map[string]any{}
		parent[key] = child
		return child
	}
	project := nested(document, "project")
	uv := nested(nested(document, "tool"), "uv")
	sources := nested(uv, "sources")
	// Root overrides keep a transitive package's identical Git requirement from
	// competing with the captured wheel. They pin the already-locked version.
	overrides, _ := uv["override-dependencies"].([]any)
	kept := make([]any, 0, len(overrides)+len(names))
	for _, value := range overrides {
		text, ok := value.(string)
		if !ok {
			return exit.New(exit.Validation, "captured dependency override is invalid")
		}
		req, problem := parseRequirement(text)
		if problem != nil {
			return problem
		}
		if versions[req.name] == "" {
			kept = append(kept, text)
		}
	}
	rewrite := func(values []any) ([]any, *exit.Error) {
		out := make([]any, 0, len(values))
		for _, value := range values {
			text, ok := value.(string)
			if !ok {
				return nil, exit.New(exit.Validation, "captured dependency declaration is invalid")
			}
			req, problem := parseRequirement(text)
			if problem != nil {
				return nil, problem
			}
			if version := versions[req.name]; version != "" && req.direct {
				text = req.name
				if len(req.extras) > 0 {
					text += "[" + strings.Join(req.extras, ",") + "]"
				}
				text += "==" + version
				if _, marker, ok := strings.Cut(req.raw, ";"); ok {
					text += ";" + marker
				}
			}
			out = append(out, text)
		}
		return out, nil
	}
	requirements, _ := project["dependencies"].([]any)
	requirements, problem = rewrite(requirements)
	if problem != nil {
		return problem
	}
	optional, _ := project["optional-dependencies"].(map[string]any)
	for key, value := range optional {
		values, _ := value.([]any)
		updated, problem := rewrite(values)
		if problem != nil {
			return problem
		}
		optional[key] = updated
	}
	for _, name := range names {
		for key := range sources {
			if normalizedProjectName(key) == name {
				delete(sources, key)
			}
		}
		sources[name] = map[string]any{"path": paths[name]}
		kept = append(kept, name+"=="+versions[name])
		direct := false
		for _, value := range requirements {
			req, problem := parseRequirement(value.(string))
			if problem != nil {
				return problem
			}
			direct = direct || req.name == name
		}
		if !direct {
			requirements = append(requirements, name+"=="+versions[name])
		}
	}
	project["dependencies"], uv["override-dependencies"] = requirements, kept
	raw, err = toml.Marshal(document)
	if err != nil {
		return exit.Internalf("cannot encode captured Git metadata: %s", err)
	}
	if err := os.WriteFile(projectPath, raw, 0600); err != nil {
		return exit.Internalf("cannot retain captured Git metadata: %s", err)
	}
	command := exec.CommandContext(ctx, "uv", "lock", "--no-progress", "--python", selected.Executable, "--no-python-downloads")
	command.Dir = root
	command.Env = config.Frozen().Tool()
	if output, err := command.CombinedOutput(); err != nil {
		detail := strings.Join(strings.Fields(string(output)), " ")
		if len(detail) > 2000 {
			detail = detail[len(detail)-2000:]
		}
		return exit.Named(exit.Validation, "private_dependency_git_lock_refused", "cannot retain locked Git wheel sources: %s", detail)
	}
	raw, err = os.ReadFile(lockPath)
	if err != nil {
		return exit.Internalf("cannot read retained Git lock: %s", err)
	}
	var after map[string]any
	if toml.Unmarshal(raw, &after) != nil {
		return exit.New(exit.Validation, "retained Git dependency lock is invalid")
	}
	if !sameGitCaptureClosure(before, after, normalizedProjectName(metadata.Project.Name), versions) {
		return exit.Named(exit.Conflict, "private_dependency_git_lock_drift", "retaining Git wheels changed another locked dependency")
	}
	receipt, err := json.Marshal(provenance)
	if err != nil {
		return exit.Internalf("cannot encode captured Git provenance: %s", err)
	}
	if err := os.WriteFile(filepath.Join(root, ".cozy-dependencies", "git-sources.json"), receipt, 0600); err != nil {
		return exit.Internalf("cannot retain captured Git provenance: %s", err)
	}
	return nil
}

func sameGitCaptureClosure(before, after map[string]any, project string, versions map[string]string) bool {
	normalize := func(doc map[string]any) []map[string]any {
		rows, _ := doc["package"].([]any)
		out := make([]map[string]any, 0, len(rows))
		for _, value := range rows {
			row, ok := value.(map[string]any)
			if !ok {
				return nil
			}
			copy := map[string]any{}
			for k, v := range row {
				copy[k] = v
			}
			name, _ := row["name"].(string)
			name = normalizedProjectName(name)
			if versions[name] != "" {
				delete(copy, "source")
				delete(copy, "wheels")
				delete(copy, "metadata")
			}
			if name == project {
				delete(copy, "metadata")
				// A transitive Git source becomes a direct source override so uv
				// can use its retained wheel. Only those existing closure members
				// may gain a root edge; all other dependency edges stay identical.
				if dependencies, ok := copy["dependencies"].([]any); ok {
					kept := make([]any, 0, len(dependencies))
					for _, dependency := range dependencies {
						row, _ := dependency.(map[string]any)
						name, _ := row["name"].(string)
						if versions[normalizedProjectName(name)] == "" {
							kept = append(kept, dependency)
						}
					}
					copy["dependencies"] = kept
				}
			}
			out = append(out, copy)
		}
		return out
	}
	return reflect.DeepEqual(normalize(before), normalize(after))
}
