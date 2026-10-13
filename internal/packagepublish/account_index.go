package packagepublish

import (
	"bytes"
	"context"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/pelletier/go-toml/v2"
)

// AccountIndexName is the uv index a package source names for dependencies published by
// its own account: `[tool.uv.sources] qwen-image-2 = { index = "tensorhub" }`. Sources never
// declare its URL. Creator writes the command's Hub and caller into each owned project copy
// it locks, installs or publishes, so one source serves every Hub and account.
const AccountIndexName = "tensorhub"

var accountName = regexp.MustCompile(`^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$`)

// Namespace is one account on one Tensorhub: the namespace a publication lands in, the
// owner of org-relative model references, and the account index's package namespace.
type Namespace struct {
	Hub     string
	Account string
}

// NamespaceSource answers the command's namespace. Callers ask it only when a project
// needs it, so a project with no account dependencies never needs a signed-in caller.
type NamespaceSource func() (Namespace, *exit.Error)

// IndexURL is the account's simple index on this Hub.
func (n Namespace) IndexURL() (string, *exit.Error) {
	hub, err := url.Parse(n.Hub)
	if err != nil || hub.Host == "" || (hub.Scheme != "http" && hub.Scheme != "https") ||
		hub.User != nil || hub.RawQuery != "" || hub.Fragment != "" || !accountName.MatchString(n.Account) {
		return "", exit.Named(exit.Validation, "account_index_invalid",
			"the account index needs an HTTP(S) Tensorhub URL without credentials, query or fragment and a caller account")
	}
	return strings.TrimRight(n.Hub, "/") + "/v1/index/" + url.PathEscape(n.Account) + "/simple/", nil
}

// UsesAccountIndex reports whether an authored project resolves any dependency from the
// account index. Authored source never declares that index itself.
func UsesAccountIndex(project string) (bool, *exit.Error) {
	raw, err := os.ReadFile(filepath.Join(project, "pyproject.toml"))
	if err != nil {
		return false, exit.Named(exit.Validation, "project_metadata_unreadable", "cannot read pyproject.toml: %s", err)
	}
	uses, declared, problem := accountIndexDocument(raw)
	if problem != nil {
		return false, problem
	}
	if declared != "" {
		return false, declaredAccountIndex()
	}
	return uses, nil
}

// UsesAccountIndexDocument reads a frozen installation's metadata. Its reserved
// index may already be bound by Creator; only authored dependency selectors decide
// whether the machine needs the request's explicitly selected Hub authority.
func UsesAccountIndexDocument(raw []byte) (bool, *exit.Error) {
	uses, _, problem := accountIndexDocument(raw)
	return uses, problem
}

func accountIndexDocument(raw []byte) (bool, string, *exit.Error) {
	if len(raw) == 0 || len(raw) > maxProjectMetadataBytes {
		return false, "", exit.Named(exit.Validation, "project_metadata_unreadable", "pyproject.toml must be non-empty and at most %d bytes", maxProjectMetadataBytes)
	}
	var document projectMetadata
	if err := toml.Unmarshal(raw, &document); err != nil {
		return false, "", exit.Named(exit.Validation, "project_metadata_invalid", "pyproject.toml is not valid TOML: %v", err)
	}
	uses, declared := accountIndexUse(document)
	return uses, declared, nil
}

func declaredAccountIndex() *exit.Error {
	return exit.Named(exit.Validation, "account_index_declared",
		"pyproject.toml declares the reserved %q index", AccountIndexName).
		WithRemedy("delete that [[tool.uv.index]] table; Creator writes the %s index for the command's Tensorhub and account", AccountIndexName)
}

// accountIndexUse answers whether sources name the account index, and the URL a copy
// Creator already bound declares for it.
func accountIndexUse(document projectMetadata) (bool, string) {
	declared := ""
	for _, index := range document.Tool.UV.Index {
		if index.Name == AccountIndexName {
			declared = index.URL
		}
	}
	var names func(value any) bool
	names = func(value any) bool {
		switch source := value.(type) {
		case []any:
			for _, entry := range source {
				if names(entry) {
					return true
				}
			}
		case map[string]any:
			index, _ := source["index"].(string)
			return index == AccountIndexName
		}
		return false
	}
	for _, source := range document.Tool.UV.Sources {
		if names(source) {
			return true, declared
		}
	}
	return false, declared
}

// bindAccountIndex writes the account index into an owned project copy. A copy whose
// lock is already bound to that index is left alone; any other lock is relocked, seeded
// by itself. rebind requires the relock to keep every selected version and dependency
// edge: publication and editable installs change only where account rows come from.
func bindAccountIndex(ctx context.Context, project string, namespace NamespaceSource, python string, rebind bool) *exit.Error {
	selected, target, problem := writeNamedAccountIndex(project, namespace)
	if problem != nil || target == "" {
		return problem
	}
	lockPath := filepath.Join(project, "uv.lock")
	before, err := os.ReadFile(lockPath)
	if os.IsNotExist(err) && !rebind {
		return nil // the caller's own `uv lock` resolves a new project against this index
	}
	if err != nil || int64(len(before)) > maxLockBytes {
		return exit.Named(exit.Validation, "account_index_lock_absent",
			"a project with %s dependencies needs its bounded uv.lock", AccountIndexName).
			WithRemedy("run `cozy package lock` in the project and commit uv.lock")
	}
	bound, problem := lockBoundTo(before, target)
	if problem != nil || bound {
		return problem
	}
	command := exec.CommandContext(ctx, "uv", "lock", "--no-progress", "--python", python, "--no-python-downloads")
	command.Dir, command.Env = project, config.Frozen().Tool()
	var log strings.Builder
	command.Stdout, command.Stderr = &log, &log
	if err := command.Run(); err != nil {
		return exit.Named(exit.Validation, "account_index_lock_refused",
			"cannot lock %s dependencies against %s: %s", AccountIndexName, target, condensed(log.String())).
			WithRemedy("publish the locked %s dependencies to this Tensorhub as %s, or run `cozy package lock` and commit uv.lock", AccountIndexName, selected.Account)
	}
	if !rebind {
		return nil
	}
	after, err := os.ReadFile(lockPath)
	if err != nil {
		return exit.Internalf("cannot read the rebound lock: %s", err)
	}
	if drift := lockDrift(before, after); drift != "" {
		return exit.Named(exit.Validation, "account_index_lock_drift",
			"uv.lock does not rebind to %s unchanged: %s", target, drift).
			WithRemedy("publish the locked versions of your %s dependencies to this Tensorhub as %s, or run `cozy package lock` there and commit uv.lock",
				AccountIndexName, selected.Account)
	}
	return nil
}

// writeNamedAccountIndex writes the account index into an owned copy that names it and
// answers its URL; a project that does not name it is unchanged and answers "".
func writeNamedAccountIndex(project string, namespace NamespaceSource) (Namespace, string, *exit.Error) {
	document, problem := readProjectDocument(filepath.Join(project, "pyproject.toml"))
	if problem != nil {
		return Namespace{}, "", problem
	}
	uses, declared := accountIndexUse(document)
	if !uses {
		if declared != "" {
			return Namespace{}, "", declaredAccountIndex()
		}
		return Namespace{}, "", nil
	}
	if namespace == nil {
		return Namespace{}, "", exit.Named(exit.Validation, "account_index_namespace_missing",
			"this project resolves %s dependencies but no Tensorhub account was selected", AccountIndexName)
	}
	selected, problem := namespace()
	if problem != nil {
		return Namespace{}, "", problem
	}
	target, problem := selected.IndexURL()
	if problem != nil {
		return Namespace{}, "", problem
	}
	switch declared {
	case target: // an owned copy this command already bound
		return selected, target, nil
	case "":
		return selected, target, writeAccountIndex(filepath.Join(project, "pyproject.toml"), target)
	}
	return Namespace{}, "", declaredAccountIndex()
}

// writeAccountIndex appends the index table, keeping the authored text. A project whose
// tool.uv.index is an inline array gets a structured rewrite instead.
func writeAccountIndex(path, target string) *exit.Error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return exit.Internalf("cannot read owned project metadata: %s", err)
	}
	table := fmt.Sprintf("\n[[tool.uv.index]]\nname = %q\nurl = %q\nexplicit = true\n", AccountIndexName, target)
	appended := append(bytes.TrimRight(raw, "\n"), "\n"+table...)
	var probe map[string]any
	if toml.Unmarshal(appended, &probe) != nil {
		var document map[string]any
		if err := toml.Unmarshal(raw, &document); err != nil {
			return exit.New(exit.Validation, "owned project metadata is not TOML")
		}
		tool, _ := document["tool"].(map[string]any)
		uv, _ := tool["uv"].(map[string]any)
		if uv == nil {
			return exit.New(exit.Validation, "owned project metadata has no tool.uv table")
		}
		indexes, _ := uv["index"].([]any)
		uv["index"] = append(indexes, map[string]any{"name": AccountIndexName, "url": target, "explicit": true})
		if appended, err = toml.Marshal(document); err != nil {
			return exit.Internalf("cannot encode owned project metadata: %s", err)
		}
	}
	if err := os.WriteFile(path, appended, 0o600); err != nil {
		return exit.Internalf("cannot write owned project metadata: %s", err)
	}
	return nil
}

type lockDocument struct {
	Packages []map[string]any `toml:"package"`
}

// lockBoundTo answers whether every Tensorhub-index row in a lock already names target.
func lockBoundTo(raw []byte, target string) (bool, *exit.Error) {
	var lock lockDocument
	if err := toml.Unmarshal(raw, &lock); err != nil {
		return false, exit.Named(exit.Validation, "account_index_lock_invalid", "uv.lock is invalid TOML")
	}
	for _, pkg := range lock.Packages {
		if registry := lockRegistry(pkg); orgIndexNamespace(registry) != "" &&
			strings.TrimRight(registry, "/") != strings.TrimRight(target, "/") {
			return false, nil
		}
	}
	return true, nil
}

func lockRegistry(pkg map[string]any) string {
	source, _ := pkg["source"].(map[string]any)
	registry, _ := source["registry"].(string)
	return registry
}

// lockDrift names the first resolution a rebinding changed. Rows from a Tensorhub index
// may change location and wheel hashes only; every other row must be identical.
func lockDrift(before, after []byte) string {
	var old, bound lockDocument
	if toml.Unmarshal(before, &old) != nil || toml.Unmarshal(after, &bound) != nil {
		return "a lock is not TOML"
	}
	rows := func(lock lockDocument) map[string]map[string]any {
		out := map[string]map[string]any{}
		for _, pkg := range lock.Packages {
			name, _ := pkg["name"].(string)
			version, _ := pkg["version"].(string)
			out[name+" "+version] = pkg
		}
		return out
	}
	was, now := rows(old), rows(bound)
	keys := make([]string, 0, len(was)+len(now))
	for key := range was {
		keys = append(keys, key)
	}
	for key := range now {
		if _, ok := was[key]; !ok {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	for _, key := range keys {
		a, inBefore := was[key]
		b, inAfter := now[key]
		switch {
		case !inAfter:
			return key + " is no longer selected"
		case !inBefore:
			return key + " is newly selected"
		}
		account := orgIndexNamespace(lockRegistry(a)) != "" || orgIndexNamespace(lockRegistry(b)) != ""
		if account {
			if orgIndexNamespace(lockRegistry(a)) == "" || orgIndexNamespace(lockRegistry(b)) == "" ||
				!reflect.DeepEqual(a["dependencies"], b["dependencies"]) ||
				!reflect.DeepEqual(a["optional-dependencies"], b["optional-dependencies"]) {
				return key + " changed its source kind or dependencies"
			}
			continue
		}
		if !reflect.DeepEqual(withoutIndexes(a), withoutIndexes(b)) {
			return key + " changed"
		}
	}
	return ""
}

// withoutIndexes drops the Tensorhub index URL a project's own metadata records beside its
// account requirements; the index is exactly what a rebinding changes.
func withoutIndexes(pkg map[string]any) map[string]any {
	metadata, ok := pkg["metadata"].(map[string]any)
	if !ok {
		return pkg
	}
	out := make(map[string]any, len(pkg))
	for key, value := range pkg {
		out[key] = value
	}
	copied := make(map[string]any, len(metadata))
	for key, value := range metadata {
		copied[key] = value
	}
	for _, field := range []string{"requires-dist", "requires-dev"} {
		copied[field] = stripIndexes(copied[field])
	}
	out["metadata"] = copied
	return out
}

func stripIndexes(value any) any {
	switch typed := value.(type) {
	case []any:
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = stripIndexes(item)
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, item := range typed {
			if index, ok := item.(string); key == "index" && ok && orgIndexNamespace(index) != "" {
				continue
			}
			out[key] = stripIndexes(item)
		}
		return out
	}
	return value
}

func condensed(text string) string {
	text = strings.Join(strings.Fields(text), " ")
	if len(text) > 2000 {
		text = text[len(text)-2000:]
	}
	return text
}

// BindAccountIndex binds an owned source copy, such as an editable install's snapshot, to
// the namespace's account index without changing any locked version.
func BindAccountIndex(ctx context.Context, project string, namespace NamespaceSource) *exit.Error {
	document, problem := readProjectDocument(filepath.Join(project, "pyproject.toml"))
	if problem != nil {
		return problem
	}
	if uses, _ := accountIndexUse(document); !uses {
		return nil
	}
	python, problem := hostruntime.ProjectPython(ctx, project)
	if problem != nil {
		return problem
	}
	return bindAccountIndex(ctx, project, namespace, python.Executable, true)
}

// Locked is one written project lock.
type Locked struct {
	Path    string
	Index   string // the account index URL the lock resolved against, if the project names one
	Changed bool
}

// Lock writes the project's uv.lock against the namespace's account index. It resolves in
// an owned copy, so the authored pyproject never spells an index URL; the committed lock is
// the version pin publication rebinds to each target. Path sources must stay inside the
// project, as they must for publication.
func Lock(ctx context.Context, projectDir string, namespace NamespaceSource, upgrade []string) (Locked, *exit.Error) {
	tree, files, problem := boundedSourceTree(projectDir, []string{"pyproject.toml"})
	if problem != nil {
		return Locked{}, problem
	}
	document, problem := readProjectDocument(files["pyproject.toml"])
	if problem != nil {
		return Locked{}, problem
	}
	sources, problem := localSources(document)
	if problem != nil {
		return Locked{}, problem
	}
	canonicalTree, problem := canonicalLocalPath(tree)
	if problem != nil {
		return Locked{}, problem
	}
	for name, source := range sources {
		if source.path != "" && !withinPackageTree(canonicalTree, source.path) {
			return Locked{}, exit.Named(exit.Validation, "package_lock_source_outside",
				"%s is a path source outside the project, which cannot be locked for publication", name).
				WithRemedy("move %s inside the project, or publish it and depend on it by name", name)
		}
	}
	root, err := os.MkdirTemp("", "cozy-package-lock-")
	if err != nil {
		return Locked{}, exit.Internalf("cannot stage the lock project: %s", err)
	}
	defer os.RemoveAll(root)
	for name, source := range files {
		target := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
			return Locked{}, exit.Internalf("cannot stage the lock project: %s", err)
		}
		if problem := copySnapshotFile(source, target, SourceFileLimit(name)); problem != nil {
			return Locked{}, problem
		}
	}
	_, index, problem := writeNamedAccountIndex(root, namespace)
	if problem != nil {
		return Locked{}, problem
	}
	python, problem := hostruntime.ProjectPython(ctx, root)
	if problem != nil {
		return Locked{}, problem
	}
	args := []string{"lock", "--no-progress", "--python", python.Executable, "--no-python-downloads"}
	for _, name := range upgrade {
		args = append(args, "--upgrade-package", name)
	}
	command := exec.CommandContext(ctx, "uv", args...)
	command.Dir, command.Env = root, config.Frozen().Tool()
	var log strings.Builder
	command.Stdout, command.Stderr = &log, &log
	if err := command.Run(); err != nil {
		return Locked{}, exit.Named(exit.Validation, "package_lock_refused", "uv could not lock the project: %s", condensed(log.String()))
	}
	raw, err := os.ReadFile(filepath.Join(root, "uv.lock"))
	if err != nil || int64(len(raw)) > maxLockBytes {
		return Locked{}, exit.Named(exit.Structural, "package_lock_invalid", "uv did not write a bounded uv.lock")
	}
	path := filepath.Join(tree, "uv.lock")
	previous, _ := os.ReadFile(path)
	locked := Locked{Path: path, Index: index, Changed: !bytes.Equal(previous, raw)}
	if !locked.Changed {
		return locked, nil
	}
	staged := path + ".cozy-lock"
	if err := os.WriteFile(staged, raw, 0o644); err != nil {
		return Locked{}, exit.Internalf("cannot write uv.lock: %s", err)
	}
	if err := os.Rename(staged, path); err != nil {
		_ = os.Remove(staged)
		return Locked{}, exit.Internalf("cannot replace uv.lock: %s", err)
	}
	return locked, nil
}
