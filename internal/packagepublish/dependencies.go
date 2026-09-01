package packagepublish

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/wheel"
)

const (
	MaxDependencyWheels     = 128
	MaxDependencyWheelBytes = 512 << 20
	maxWorkspaceAncestors   = 32
	maxWorkspacePatterns    = 64
	maxWorkspaceMembers     = 256
)

// DependencyWheel is one exact, separately built local distribution. Tensorhub
// derives its own facts from Path after upload; these fields are local checks
// and upload addressing, not caller-authored catalog identity.
type DependencyWheel struct {
	Filename string
	Path     string
}

type requirement struct {
	raw       string
	name      string
	extras    []string
	specifier pep440.Specifiers
	hasSpec   bool
	direct    bool
}

type localSource struct {
	path        string
	workspace   bool
	unsupported string
}

type dependencyRecord struct {
	source  string
	version string
}

type dependencyCollector struct {
	ctx      context.Context
	stage    string
	wheels   []DependencyWheel
	byName   map[string]dependencyRecord
	extras   map[string]map[string]bool
	stack    map[string]bool
	total    int64
	count    int
	registry bool
}

var requirementName = regexp.MustCompile(`^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?`)

func collectLocalDependencies(ctx context.Context, root string, document projectMetadata, stage string) ([]DependencyWheel, bool, *exit.Error) {
	canonical, problem := canonicalLocalPath(root)
	if problem != nil {
		return nil, false, problem
	}
	collector := &dependencyCollector{
		ctx: ctx, stage: stage, byName: map[string]dependencyRecord{}, extras: map[string]map[string]bool{},
		stack: map[string]bool{canonical: true},
	}
	if name := normalizedProjectName(document.Project.Name); name != "" {
		collector.byName[name] = dependencyRecord{source: canonical, version: document.Project.Version}
	}
	if problem := collector.collectProject(canonical, document, nil, true); problem != nil {
		return nil, false, problem
	}
	sort.Slice(collector.wheels, func(i, j int) bool {
		return collector.wheels[i].Filename < collector.wheels[j].Filename
	})
	return collector.wheels, collector.registry, nil
}

func (c *dependencyCollector) collectProject(root string, document projectMetadata, extras []string, includeBase bool) *exit.Error {
	sources, problem := localSources(document)
	if problem != nil {
		return problem
	}
	requirements := []string{}
	if includeBase {
		requirements = append(requirements, document.Project.Dependencies...)
	}
	for _, extra := range extras {
		optional, problem := optionalDependencyGroup(document, extra)
		if problem != nil {
			return problem
		}
		requirements = append(requirements, optional...)
	}
	for _, raw := range requirements {
		req, problem := parseRequirement(raw)
		if problem != nil {
			return problem
		}
		if req.direct {
			return exit.Named(exit.Validation, "project_dependency_direct_url_unsupported",
				"project dependency %q uses a direct URL", raw).
				WithRemedy("publish the distribution to an index or use a local path/workspace source")
		}
		source, exists := sources[req.name]
		if !exists {
			c.registry = true
			continue
		}
		if source.unsupported != "" {
			return exit.Named(exit.Validation, "project_dependency_source_unsupported",
				"%s uses unsupported %s source configuration", req.name, source.unsupported).
				WithRemedy("publish the distribution to an index or use a local path/workspace source")
		}
		if source.workspace {
			member, problem := workspaceMember(root, req.name)
			if problem != nil {
				return problem
			}
			if problem := c.collectDirectory(req, member); problem != nil {
				return problem
			}
			continue
		}
		candidate := source.path
		if !filepath.IsAbs(candidate) {
			candidate = filepath.Join(root, candidate)
		}
		canonical, problem := canonicalLocalPath(candidate)
		if problem != nil {
			return problem
		}
		info, err := os.Stat(canonical)
		if err != nil {
			return exit.Named(exit.NotFound, "local_dependency_absent",
				"local source for %s is unreadable: %s", req.name, canonical)
		}
		switch {
		case info.IsDir():
			if problem := c.collectDirectory(req, canonical); problem != nil {
				return problem
			}
		case info.Mode().IsRegular() && strings.HasSuffix(strings.ToLower(canonical), ".whl"):
			if problem := c.collectWheel(req, canonical); problem != nil {
				return problem
			}
		default:
			return exit.Named(exit.Validation, "local_dependency_source_invalid",
				"local source for %s must be a project directory or wheel: %s", req.name, canonical)
		}
	}
	return nil
}

func (c *dependencyCollector) collectDirectory(req requirement, source string) *exit.Error {
	canonical, problem := canonicalLocalPath(source)
	if problem != nil {
		return problem
	}
	if c.stack[canonical] {
		return exit.Named(exit.Validation, "local_dependency_cycle",
			"local dependency %s forms a cycle through %s", req.name, canonical).
			WithRemedy("remove the cyclic [tool.uv.sources] path or workspace edge")
	}
	document, problem := readProjectDocument(filepath.Join(canonical, "pyproject.toml"))
	if problem != nil {
		return problem
	}
	name := normalizedProjectName(strings.TrimSpace(document.Project.Name))
	version := strings.TrimSpace(document.Project.Version)
	if name == "" || version == "" || name != req.name {
		return exit.Named(exit.Validation, "local_dependency_identity_mismatch",
			"requirement %s resolves to local project %s==%s", req.raw, name, version).
			WithRemedy("make the [tool.uv.sources] key, requirement name, and local [project] identity agree")
	}
	if problem := req.accepts(version); problem != nil {
		return problem
	}
	if prior, exists := c.byName[name]; exists {
		if prior.source != canonical || prior.version != version {
			return duplicateDependency(name, prior, canonical, version)
		}
		newExtras := c.activateExtras(canonical, req.extras)
		if remoteBaseRoots[name] {
			return nil
		}
		if len(newExtras) == 0 {
			return nil
		}
		c.stack[canonical] = true
		problem := c.collectProject(canonical, document, newExtras, false)
		delete(c.stack, canonical)
		return problem
	}
	if c.count >= MaxDependencyWheels {
		return tooManyDependencies()
	}
	c.count++
	c.byName[name] = dependencyRecord{source: canonical, version: version}
	newExtras := c.activateExtras(canonical, req.extras)
	if !remoteBaseRoots[name] {
		c.stack[canonical] = true
		if problem := c.collectProject(canonical, document, newExtras, true); problem != nil {
			return problem
		}
		delete(c.stack, canonical)
	}

	out := filepath.Join(c.stage, "dependencies", fmt.Sprintf("%02d-%s", len(c.wheels)+1, name))
	built, problem := wheel.Build(wheel.Request{Context: c.ctx, Tree: canonical, OutDir: out})
	if problem != nil {
		return problem
	}
	identity, problem := wheel.InspectIdentity(built.Path)
	if problem != nil {
		return problem
	}
	if identity.Distribution != name || identity.Version != version {
		return exit.Named(exit.Validation, "local_dependency_identity_mismatch",
			"local project declares %s==%s but its wheel declares %s==%s",
			name, version, identity.Distribution, identity.Version)
	}
	return c.add(identity, built.Path)
}

func (c *dependencyCollector) collectWheel(req requirement, source string) *exit.Error {
	identity, problem := wheel.InspectIdentity(source)
	if problem != nil {
		return problem
	}
	if identity.Distribution != req.name {
		return exit.Named(exit.Validation, "local_dependency_identity_mismatch",
			"requirement %s resolves to wheel %s==%s", req.raw, identity.Distribution, identity.Version).
			WithRemedy("make the [tool.uv.sources] key, requirement name, and wheel identity agree")
	}
	if problem := req.accepts(identity.Version); problem != nil {
		return problem
	}
	canonical, problem := canonicalLocalPath(source)
	if problem != nil {
		return problem
	}
	if prior, exists := c.byName[identity.Distribution]; exists {
		if prior.source != canonical || prior.version != identity.Version {
			return duplicateDependency(identity.Distribution, prior, canonical, identity.Version)
		}
		return nil
	}
	if c.count >= MaxDependencyWheels {
		return tooManyDependencies()
	}
	c.count++
	c.byName[identity.Distribution] = dependencyRecord{source: canonical, version: identity.Version}
	return c.add(identity, canonical)
}

func (c *dependencyCollector) add(identity wheel.Identity, path string) *exit.Error {
	// Runtime is carried so Creator can build its independent local venv, then omitted
	// when Runtime authors the rental placement. Every other remote base root must stay
	// out of DependencyWheels entirely; TensorFS local custody is the exact source member
	// referenced by uv.lock, returned separately by Tensorhub's local install plan.
	if remoteBaseRoots[identity.Distribution] && identity.Distribution != "cozy-runtime" { //cozy:allow base distribution identity, not executable access
		return nil
	}
	if len(c.wheels) >= MaxDependencyWheels {
		return tooManyDependencies()
	}
	if identity.Length > MaxDependencyWheelBytes-c.total {
		return exit.Named(exit.Validation, "local_dependency_wheels_too_large",
			"local dependency wheels exceed %d B combined", MaxDependencyWheelBytes)
	}
	c.total += identity.Length
	c.wheels = append(c.wheels, DependencyWheel{Filename: identity.Filename, Path: path})
	return nil
}

func tooManyDependencies() *exit.Error {
	return exit.Named(exit.Validation, "local_dependency_count_exceeded",
		"package has more than %d local dependency wheels", MaxDependencyWheels)
}

func duplicateDependency(name string, prior dependencyRecord, source, version string) *exit.Error {
	return exit.Named(exit.Validation, "local_dependency_duplicate",
		"normalized dependency %s resolves to both %s (%s) and %s (%s)",
		name, prior.source, prior.version, source, version).
		WithRemedy("use one local source and version for each normalized distribution name")
}

func (c *dependencyCollector) activateExtras(source string, requested []string) []string {
	active := c.extras[source]
	if active == nil {
		active = map[string]bool{}
		c.extras[source] = active
	}
	var added []string
	for _, extra := range requested {
		if !active[extra] {
			active[extra] = true
			added = append(added, extra)
		}
	}
	return added
}

func optionalDependencyGroup(document projectMetadata, wanted string) ([]string, *exit.Error) {
	var matched []string
	found := false
	for name, requirements := range document.Project.OptionalDependencies {
		if normalizedProjectName(name) != wanted {
			continue
		}
		if found {
			return nil, exit.Named(exit.Validation, "local_dependency_extra_ambiguous",
				"local project declares more than one optional dependency group normalized as %s", wanted)
		}
		found = true
		matched = requirements
	}
	return matched, nil
}

func parseRequirement(raw string) (requirement, *exit.Error) {
	value := strings.TrimSpace(raw)
	match := requirementName.FindString(value)
	if match == "" {
		return requirement{}, invalidRequirement(raw)
	}
	req := requirement{raw: raw, name: normalizedProjectName(match)}
	rest := strings.TrimSpace(value[len(match):])
	if strings.HasPrefix(rest, "[") {
		end := strings.IndexByte(rest, ']')
		if end < 0 {
			return requirement{}, invalidRequirement(raw)
		}
		for _, value := range strings.Split(rest[1:end], ",") {
			extra := strings.TrimSpace(value)
			if extra == "" || requirementName.FindString(extra) != extra {
				return requirement{}, invalidRequirement(raw)
			}
			req.extras = append(req.extras, normalizedProjectName(extra))
		}
		sort.Strings(req.extras)
		for i := 1; i < len(req.extras); i++ {
			if req.extras[i] == req.extras[i-1] {
				return requirement{}, invalidRequirement(raw)
			}
		}
		rest = strings.TrimSpace(rest[end+1:])
	}
	if marker := strings.IndexByte(rest, ';'); marker >= 0 {
		rest = strings.TrimSpace(rest[:marker])
	}
	if strings.HasPrefix(rest, "@") || rest == "" {
		req.direct = strings.HasPrefix(rest, "@")
		return req, nil
	}
	if strings.HasPrefix(rest, "(") && strings.HasSuffix(rest, ")") {
		rest = strings.TrimSpace(rest[1 : len(rest)-1])
	}
	specifier, err := pep440.NewSpecifiers(rest)
	if err != nil {
		return requirement{}, invalidRequirement(raw)
	}
	req.specifier, req.hasSpec = specifier, true
	return req, nil
}

func invalidRequirement(raw string) *exit.Error {
	return exit.Named(exit.Validation, "project_requirement_invalid",
		"project dependency %q is not a supported PEP 508 requirement", raw)
}

func (r requirement) accepts(rawVersion string) *exit.Error {
	if !r.hasSpec {
		return nil
	}
	version, err := pep440.Parse(rawVersion)
	if err != nil || !r.specifier.Check(version) {
		return exit.Named(exit.Validation, "local_dependency_version_incompatible",
			"local dependency %s==%s does not satisfy %s", r.name, rawVersion, r.raw).
			WithRemedy("update the local project version or the standard [project].dependencies requirement")
	}
	return nil
}

func localSources(document projectMetadata) (map[string]localSource, *exit.Error) {
	out := map[string]localSource{}
	for rawName, rawSource := range document.Tool.UV.Sources {
		name := normalizedProjectName(strings.TrimSpace(rawName))
		if name == "" {
			return nil, exit.Named(exit.Validation, "local_dependency_source_invalid",
				"[tool.uv.sources] contains an empty distribution name")
		}
		if _, exists := out[name]; exists {
			return nil, exit.Named(exit.Validation, "local_dependency_source_duplicate",
				"[tool.uv.sources] names normalized distribution %s more than once", name)
		}
		source, local, problem := decodeLocalSource(name, rawSource)
		if problem != nil {
			return nil, problem
		}
		if local {
			out[name] = source
		}
	}
	return out, nil
}

func decodeLocalSource(name string, value any) (localSource, bool, *exit.Error) {
	if entries, ok := value.([]any); ok {
		var found *localSource
		for _, entry := range entries {
			source, local, problem := decodeLocalSource(name, entry)
			if problem != nil {
				return localSource{}, false, problem
			}
			if !local {
				continue
			}
			if found != nil && *found != source {
				return localSource{}, false, exit.Named(exit.Validation, "local_dependency_source_ambiguous",
					"%s has multiple conditional local sources", name)
			}
			copy := source
			found = &copy
		}
		if found == nil {
			return localSource{}, false, nil
		}
		return *found, true, nil
	}
	table, ok := value.(map[string]any)
	if !ok {
		return localSource{}, false, nil
	}
	pathValue, hasPath := table["path"].(string)
	workspace, hasWorkspace := table["workspace"].(bool)
	if hasPath && hasWorkspace && workspace {
		return localSource{}, false, exit.Named(exit.Validation, "local_dependency_source_invalid",
			"%s declares both path and workspace local sources", name)
	}
	if hasPath {
		pathValue = strings.TrimSpace(pathValue)
		if pathValue == "" {
			return localSource{}, false, exit.Named(exit.Validation, "local_dependency_source_invalid",
				"%s declares an empty local path", name)
		}
		return localSource{path: pathValue}, true, nil
	}
	if hasWorkspace && workspace {
		return localSource{workspace: true}, true, nil
	}
	if _, git := table["git"].(string); git {
		return localSource{unsupported: "Git"}, true, nil
	}
	if _, url := table["url"].(string); url {
		return localSource{unsupported: "URL"}, true, nil
	}
	return localSource{}, false, nil
}

func workspaceMember(start, name string) (string, *exit.Error) {
	for root, ancestors := start, 0; ; root, ancestors = filepath.Dir(root), ancestors+1 {
		if ancestors >= maxWorkspaceAncestors {
			return "", exit.Named(exit.Validation, "local_dependency_workspace_too_broad",
				"workspace lookup for %s crossed %d ancestors", name, maxWorkspaceAncestors)
		}
		path := filepath.Join(root, "pyproject.toml")
		document, problem := readProjectDocument(path)
		if problem == nil && len(document.Tool.UV.Workspace.Members) > 0 {
			members, problem := workspaceMembers(root, document)
			if problem != nil {
				return "", problem
			}
			var matches []string
			for _, member := range members {
				memberDocument, memberProblem := readProjectDocument(filepath.Join(member, "pyproject.toml"))
				if memberProblem == nil && normalizedProjectName(memberDocument.Project.Name) == name {
					matches = append(matches, member)
				}
			}
			if len(matches) == 1 {
				return matches[0], nil
			}
			if len(matches) > 1 {
				return "", exit.Named(exit.Validation, "local_dependency_workspace_duplicate",
					"workspace contains %d projects named %s", len(matches), name)
			}
		}
		parent := filepath.Dir(root)
		if parent == root {
			break
		}
	}
	return "", exit.Named(exit.Validation, "local_dependency_workspace_absent",
		"workspace source %s has no matching member in this project or an ancestor workspace", name)
}

func workspaceMembers(root string, document projectMetadata) ([]string, *exit.Error) {
	workspaceRoot, problem := canonicalLocalPath(root)
	if problem != nil {
		return nil, problem
	}
	patterns := len(document.Tool.UV.Workspace.Members) + len(document.Tool.UV.Workspace.Exclude)
	if patterns > maxWorkspacePatterns {
		return nil, exit.Named(exit.Validation, "local_dependency_workspace_too_broad",
			"workspace declares %d member/exclude patterns; limit is %d", patterns, maxWorkspacePatterns)
	}
	excluded := map[string]bool{}
	for _, pattern := range document.Tool.UV.Workspace.Exclude {
		if problem := safeWorkspacePattern(pattern); problem != nil {
			return nil, problem
		}
		matches, err := filepath.Glob(filepath.Join(root, pattern))
		if err != nil {
			return nil, exit.Named(exit.Validation, "local_dependency_workspace_invalid",
				"workspace exclude %q is invalid: %v", pattern, err)
		}
		for _, match := range matches {
			canonical, problem := canonicalLocalPath(match)
			if problem != nil {
				continue
			}
			if !withinWorkspace(workspaceRoot, canonical) {
				return nil, exit.Named(exit.Validation, "local_dependency_workspace_invalid",
					"workspace exclude %q resolves outside %s", pattern, workspaceRoot)
			}
			excluded[canonical] = true
		}
	}
	unique := map[string]bool{}
	for _, pattern := range document.Tool.UV.Workspace.Members {
		if problem := safeWorkspacePattern(pattern); problem != nil {
			return nil, problem
		}
		matches, err := filepath.Glob(filepath.Join(root, pattern))
		if err != nil {
			return nil, exit.Named(exit.Validation, "local_dependency_workspace_invalid",
				"workspace member %q is invalid: %v", pattern, err)
		}
		for _, match := range matches {
			canonical, problem := canonicalLocalPath(match)
			if problem != nil {
				continue
			}
			if !withinWorkspace(workspaceRoot, canonical) {
				return nil, exit.Named(exit.Validation, "local_dependency_workspace_invalid",
					"workspace member %q resolves outside %s", pattern, workspaceRoot)
			}
			if excluded[canonical] {
				continue
			}
			unique[canonical] = true
			if len(unique) > maxWorkspaceMembers {
				return nil, exit.Named(exit.Validation, "local_dependency_workspace_too_broad",
					"workspace expands past %d members", maxWorkspaceMembers)
			}
		}
	}
	members := make([]string, 0, len(unique))
	for member := range unique {
		members = append(members, member)
	}
	sort.Strings(members)
	return members, nil
}

func withinWorkspace(root, candidate string) bool {
	relative, err := filepath.Rel(root, candidate)
	return err == nil && relative != ".." && !strings.HasPrefix(filepath.ToSlash(relative), "../")
}

func safeWorkspacePattern(pattern string) *exit.Error {
	clean := filepath.Clean(strings.TrimSpace(pattern))
	if clean == "." || filepath.IsAbs(clean) {
		return exit.Named(exit.Validation, "local_dependency_workspace_invalid",
			"workspace pattern %q must be a relative member path", pattern)
	}
	for _, part := range strings.Split(filepath.ToSlash(clean), "/") {
		if part == ".." {
			return exit.Named(exit.Validation, "local_dependency_workspace_invalid",
				"workspace pattern %q escapes the workspace root", pattern)
		}
	}
	return nil
}

func canonicalLocalPath(value string) (string, *exit.Error) {
	abs, err := filepath.Abs(value)
	if err != nil {
		return "", exit.Named(exit.Validation, "local_dependency_path_invalid", "%s: %v", value, err)
	}
	canonical, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", exit.Named(exit.NotFound, "local_dependency_absent", "%s: %v", abs, err)
	}
	return filepath.Clean(canonical), nil
}
