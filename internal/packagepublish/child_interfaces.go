package packagepublish

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net/textproto"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/wheel"
	"github.com/pelletier/go-toml/v2"
)

// WithChildInterfaces prepares an owned copy whose invocable dependencies are
// exact lightweight interface wheels. The author's project and lock never change.
// Unrelated local libraries retain their original declarations and are captured
// by the ordinary snapshotter after this interface-only lock adjustment.
func WithChildInterfaces(ctx context.Context, parent *Package, replacements map[string]string, namespace NamespaceSource) (*Package, *exit.Error) {
	if len(replacements) == 0 {
		return parent, nil
	}
	return prepareUnpublishedCopy(ctx, parent, replacements, namespace)
}

// Both ordinary captured dependency resolution and interface substitution use
// the same bounded source capture, path rebasing and uv resolver.
func prepareUnpublishedCopy(ctx context.Context, parent *Package, replacements map[string]string, namespace NamespaceSource, extras ...string) (*Package, *exit.Error) {
	extras, problem := normalizedExtras(extras)
	if problem != nil {
		return nil, problem
	}
	dependencies, problem := LocalDependencySelections(parent.Tree, extras...)
	if problem != nil {
		return nil, problem
	}
	root, err := os.MkdirTemp("", "cozy-child-interfaces-")
	if err != nil {
		return nil, exit.Internalf("cannot stage captured interfaces: %s", err)
	}
	fail := func(problem *exit.Error) (*Package, *exit.Error) { _ = os.RemoveAll(root); return nil, problem }
	captured := &Package{Tree: parent.Tree, Files: map[string]string{}}
	for name, source := range parent.Files {
		to := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(to), 0o700); err != nil {
			return fail(exit.Internalf("cannot stage parent source: %s", err))
		}
		input, err := os.Open(source)
		if err != nil {
			return fail(exit.Internalf("cannot read parent source: %s", err))
		}
		output, err := os.OpenFile(to, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err != nil {
			input.Close()
			return fail(exit.Internalf("cannot capture parent source: %s", err))
		}
		n, copyErr := io.Copy(output, io.LimitReader(input, MaxSourceFileBytes+1))
		input.Close()
		closeErr := output.Close()
		if copyErr != nil || closeErr != nil || n > MaxSourceFileBytes {
			return fail(exit.New(exit.Conflict, "parent source changed while capturing child interfaces"))
		}
		captured.Files[name] = to
	}
	path := filepath.Join(root, "pyproject.toml")
	raw, err := os.ReadFile(path)
	if err != nil {
		return fail(exit.Internalf("cannot read parent metadata: %s", err))
	}
	var document map[string]any
	if err := toml.Unmarshal(raw, &document); err != nil {
		return fail(exit.New(exit.Validation, "parent metadata is not TOML"))
	}
	nested := func(parent map[string]any, key string) map[string]any {
		value, ok := parent[key].(map[string]any)
		if !ok {
			value = map[string]any{}
			parent[key] = value
		}
		return value
	}
	metadata, problem := readProjectDocument(filepath.Join(parent.Tree, "pyproject.toml"))
	if problem != nil {
		return fail(problem)
	}
	metadataIndexes := metadata.Tool.UV.Index
	requirements := append([]string(nil), metadata.Project.Dependencies...)
	if len(extras) > 0 {
		// A self-extra requirement lets uv evaluate the original optional
		// markers in their proper extra context. The selected set becomes part
		// of this immutable copied metadata, never the editable source.
		requirements = append(requirements, fmt.Sprintf("%s[%s]",
			normalizedProjectName(metadata.Project.Name), strings.Join(extras, ",")))
	}
	uv := nested(nested(document, "tool"), "uv")
	sources := nested(uv, "sources")
	// uv lock resolves every optional group, including inactive ones. Preserve
	// those declared source locations when the root project moves into its copy.
	declared, problem := localSources(metadata)
	if problem != nil {
		return fail(problem)
	}
	for name, source := range declared {
		path := source.path
		if source.workspace {
			path, problem = workspaceMember(parent.Tree, name)
			if problem != nil {
				return fail(problem)
			}
		} else if path != "" && !filepath.IsAbs(path) {
			path = filepath.Join(parent.Tree, path)
		}
		if path != "" {
			canonical, problem := canonicalLocalPath(path)
			if problem != nil {
				return fail(problem)
			}
			sources[name] = map[string]any{"path": canonical}
		}
	}
	for name, path := range dependencies {
		sources[name] = map[string]any{"path": path.Path}
	}
	names := make([]string, 0, len(replacements))
	for name := range replacements {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		path := replacements[name]
		identity, problem := wheel.InspectIdentity(path)
		if problem != nil {
			return fail(problem)
		}
		if identity.Distribution != name {
			return fail(exit.New(exit.Conflict, "captured interface replacement changed its distribution"))
		}
		sources[name] = map[string]any{"path": path}
		// Interface metadata carries the child's exact installed closure. Its
		// explicitly sourced dependencies must also be direct requirements, or uv
		// ignores their retained wheel/index overrides and falls back to PyPI.
		metadata, problem := wheel.Metadata(path)
		if problem != nil {
			return fail(problem)
		}
		headers, err := textproto.NewReader(bufio.NewReader(bytes.NewReader(metadata))).ReadMIMEHeader()
		if err != nil && err != io.EOF {
			return fail(exit.New(exit.Validation, "captured interface wheel metadata is malformed"))
		}
		for _, raw := range headers.Values("Requires-Dist") {
			req, problem := parseRequirement(raw)
			if problem != nil {
				return fail(problem)
			}
			source, _ := sources[req.name].(map[string]any)
			local, _ := source["path"].(string)
			index, _ := source["index"].(string)
			selectedVersion := ""
			if strings.HasSuffix(strings.ToLower(local), ".whl") {
				identity, problem := wheel.InspectIdentity(local)
				if problem != nil {
					return fail(problem)
				}
				if identity.Distribution != req.name {
					return fail(exit.New(exit.Conflict, "captured interface dependency changed its distribution"))
				}
				selectedVersion = identity.Version
			} else if index != "" {
				// Index sources remain scoped to their authored explicit table. The
				// captured interface's exact pin supplies the selected version; never
				// invent a fallback index or loosen its immutable child requirement.
				explicit := false
				for _, declared := range metadataIndexes {
					explicit = explicit || declared.Name == index && declared.Explicit
				}
				if !explicit {
					continue
				}
				var exact bool
				selectedVersion, exact = strings.CutPrefix(req.specifier.String(), "==")
				if !exact || strings.ContainsAny(selectedVersion, ",*") {
					return fail(exit.New(exit.Validation, "captured interface index dependency requires an exact selected version"))
				}
			} else {
				continue
			}
			if problem := req.accepts(selectedVersion); problem != nil {
				return fail(problem)
			}
			version, err := pep440.Parse(selectedVersion)
			if err != nil {
				return fail(exit.New(exit.Validation, "captured interface dependency has an invalid version"))
			}
			// The interface METADATA and exact source wheel retain the private
			// pin. The generated project declaration must follow the same portable
			// lower-bound policy as authored projects, without a local version tag.
			floor := req.name
			if len(req.extras) > 0 {
				floor += "[" + strings.Join(req.extras, ",") + "]"
			}
			floor += ">=" + version.Public()
			if _, marker, ok := strings.Cut(raw, ";"); ok {
				floor += ";" + marker
			}
			if !slices.Contains(requirements, floor) {
				requirements = append(requirements, floor)
			}
		}
		// uv applies root source overrides only to direct requirements. A callable
		// may be selected transitively, so declare its already-selected overlay here too.
		// Otherwise bindings advertise its exports while uv installs the original.
		// The exact source wheel above and uv.lock select its identity; the
		// dependency declaration need not invent an exact compatibility pin.
		if !slices.Contains(requirements, name) {
			requirements = append(requirements, name)
		}
	}
	if len(replacements) > 0 || len(extras) > 0 {
		nested(document, "project")["dependencies"] = requirements
	}
	delete(uv, "workspace") // all selected local members now have exact explicit paths
	raw, err = toml.Marshal(document)
	if err != nil {
		return fail(exit.Internalf("cannot encode captured interface metadata: %s", err))
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		return fail(exit.Internalf("cannot retain captured interface metadata: %s", err))
	}
	python, problem := hostruntime.ProjectPython(ctx, root)
	if problem != nil {
		return fail(problem)
	}
	if _, _, problem := writeNamedAccountIndex(root, namespace); problem != nil {
		return fail(problem)
	}
	command := exec.CommandContext(ctx, "uv", "lock", "--no-progress", "--python", python.Executable, "--no-python-downloads")
	command.Dir = root
	command.Env = config.Frozen().Tool()
	var log strings.Builder
	command.Stdout, command.Stderr = &log, &log
	if err := command.Run(); err != nil {
		detail := strings.TrimSpace(log.String())
		if len(detail) > 2000 {
			detail = detail[len(detail)-2000:]
		}
		return fail(exit.Named(exit.Structural, "child.interface_lock_refused", "cannot lock the captured interface dependencies: %s", detail))
	}
	prepared, problem := PrepareLocalFrom(root)
	if problem != nil {
		return fail(problem)
	}
	prepared.temporarySource = root
	return prepared, nil
}
