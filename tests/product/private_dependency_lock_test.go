package producttest

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestPrivateDependencyRefusesSourceMutationDuringResolution(t *testing.T) {
	root := t.TempDir()
	project, library := filepath.Join(root, "operation"), filepath.Join(root, "library")
	must(t, os.MkdirAll(project, 0o700))
	must(t, os.MkdirAll(library, 0o700))
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name = "private-operation-mutation-proof"
version = "0.0.1"
requires-python = ">=3.12,<3.13"
dependencies = ["private-mutating-library"]
[tool.uv.sources]
private-mutating-library = {path = "../library"}
`), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject = 'operation:app'\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "operation.py"), []byte("VALUE = 7\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(`[project]
name = "private-mutating-library"
version = "0.0.1"
dynamic = ["dependencies"]
[build-system]
requires = []
build-backend = "backend"
backend-path = ["."]
`), 0o600))
	marker := filepath.Join(project, "added_during_resolution.py")
	// uv runs this real PEP 517 backend to resolve dynamic dependency metadata.
	// Adding an authored file during resolution must invalidate the capture.
	backend := fmt.Sprintf(`from pathlib import Path
def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    Path(%q).write_text("CHANGED = True\n")
    name = "private_mutating_library-0.0.1.dist-info"
    target = Path(metadata_directory) / name
    target.mkdir()
    (target / "METADATA").write_text("Metadata-Version: 2.3\nName: private-mutating-library\nVersion: 0.0.1\n")
    return name
`, marker)
	must(t, os.WriteFile(filepath.Join(library, "backend.py"), []byte(backend), 0o600))
	pack, problem := packagepublish.PreparePrivateFrom(context.Background(), project)
	if pack != nil {
		pack.Close()
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatalf("real uv resolution did not execute the mutation arm: %v (%v)", err, problem)
	}
	if problem == nil || problem.ErrName() != "conflict" {
		t.Fatalf("source changed during resolution without a capture refusal: %v", problem)
	}
}

func TestPrivateDependencyLocksOnlyCapturedSource(t *testing.T) {
	root := t.TempDir()
	project, library := filepath.Join(root, "operation"), filepath.Join(root, "library")
	must(t, os.MkdirAll(project, 0o700))
	must(t, os.MkdirAll(library, 0o700))
	metadata := `[project]
name = "private-operation-lock-proof"
version = "0.0.1"
requires-python = ">=3.12,<3.13"
dependencies = ["private-lock-library"]
[tool.uv.sources]
private-lock-library = {path = "../library", editable = true}
`
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject = 'operation:app'\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "operation.py"), []byte("VALUE = 7\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte("[project]\nname = 'private-lock-library'\nversion = '0.0.1'\n"), 0o600))
	implementation := filepath.Join(library, "library.py")
	must(t, os.WriteFile(implementation, []byte("VALUE = 11\n"), 0o600))
	if _, problem := packagepublish.PrepareFrom(project); problem == nil || problem.ErrName() != "package_source_required_file_missing" {
		t.Fatalf("publication no longer requires an authored lock: %v", problem)
	}
	first, problem := packagepublish.PreparePrivateFrom(context.Background(), project)
	fatal(t, problem)
	defer first.Close()
	firstIdentity, _, _, problem := first.SourceIdentity()
	fatal(t, problem)
	second, problem := packagepublish.PreparePrivateFrom(context.Background(), project)
	fatal(t, problem)
	defer second.Close()
	secondIdentity, _, _, problem := second.SourceIdentity()
	fatal(t, problem)
	if first.Tree == second.Tree || firstIdentity != secondIdentity {
		t.Fatalf("temporary capture path changed source identity: %s / %s", firstIdentity, secondIdentity)
	}
	dependencies, problem := packagepublish.LocalDependencyPaths(first.Tree)
	fatal(t, problem)
	if dependencies["private-lock-library"] != library {
		t.Fatalf("relative library path changed meaning: %v", dependencies)
	}
	for _, dir := range []string{project, library} {
		if _, err := os.Stat(filepath.Join(dir, "uv.lock")); !os.IsNotExist(err) {
			t.Fatalf("resolution changed authored lock at %s: %v", dir, err)
		}
	}
	actual, err := os.ReadFile(filepath.Join(project, "pyproject.toml"))
	must(t, err)
	if string(actual) != metadata {
		t.Fatal("resolution rewrote authored dependency metadata")
	}
	if _, err := os.Stat(filepath.Join(first.Tree, "uv.lock")); err != nil {
		t.Fatalf("captured package has no resolved lock: %v", err)
	}
	must(t, os.WriteFile(implementation, []byte("VALUE = 12\n"), 0o600))
	edited, problem := packagepublish.PreparePrivateFrom(context.Background(), project)
	fatal(t, problem)
	defer edited.Close()
	editedIdentity, _, _, problem := edited.SourceIdentity()
	fatal(t, problem)
	if editedIdentity == firstIdentity {
		t.Fatal("same-version local library edit did not change captured identity")
	}
}

func TestPrivateSelectedExtrasAreCanonicalCapturedAndValidated(t *testing.T) {
	root := t.TempDir()
	project, library, alternate := filepath.Join(root, "operation"), filepath.Join(root, "library"), filepath.Join(root, "alternate")
	for _, dir := range []string{project, library, alternate} {
		must(t, os.MkdirAll(dir, 0700))
	}
	metadata := `[project]
name="private-extra-operation"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=[]
[project.optional-dependencies]
managed=["private-extra-library==0.0.1"]
Tool_Box=["private-extra-alternate==0.0.1"]
all=["private-extra-operation[managed,tool-box]"]
[tool.uv.sources]
private-extra-library={path="../library"}
private-extra-alternate={path="../alternate"}
`
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject='operation:app'\n"), 0600))
	must(t, os.WriteFile(filepath.Join(project, "operation.py"), []byte("VALUE=7\n"), 0600))
	for _, item := range []struct{ dir, name string }{{library, "private-extra-library"}, {alternate, "private-extra-alternate"}} {
		must(t, os.WriteFile(filepath.Join(item.dir, "pyproject.toml"), []byte(fmt.Sprintf("[project]\nname=%q\nversion='0.0.1'\n", item.name)), 0600))
		must(t, os.WriteFile(filepath.Join(item.dir, "helper.py"), []byte("VALUE=11\n"), 0600))
	}
	capture := func(extras ...string) (*packagepublish.Package, string) {
		t.Helper()
		pack, problem := packagepublish.PreparePrivateFrom(context.Background(), project, extras...)
		fatal(t, problem)
		t.Cleanup(pack.Close)
		identity, _, _, problem := pack.SourceIdentity()
		fatal(t, problem)
		return pack, identity
	}
	first, identity := capture("managed", "TOOL_box", "managed")
	_, repeat := capture("tool-box", "managed")
	if identity != repeat {
		t.Fatal("extra normalization/order changed immutable capture")
	}
	_, combined := capture("all")
	if combined != identity {
		t.Fatal("self-extra grouping changed effective captured requirements")
	}
	_, one := capture("managed")
	if one == identity {
		t.Fatal("different selected extras kept the same closure")
	}
	selected, problem := packagepublish.LocalDependencySelections(first.Tree)
	fatal(t, problem)
	if len(selected) != 2 || selected["private-extra-library"].Path != library || selected["private-extra-alternate"].Path != alternate {
		t.Fatalf("selected optional dependencies lost their original paths: %+v", selected)
	}
	for _, extra := range []string{"missing", "managed; injected"} {
		pack, problem := packagepublish.PreparePrivateFrom(context.Background(), project, extra)
		if pack != nil {
			pack.Close()
		}
		if problem == nil {
			t.Fatalf("unvalidated optional dependency group accepted: %q", extra)
		}
	}
	actual, err := os.ReadFile(filepath.Join(project, "pyproject.toml"))
	must(t, err)
	if string(actual) != metadata {
		t.Fatal("extra activation mutated editable source metadata")
	}
	if _, err := os.Stat(filepath.Join(project, "uv.lock")); !os.IsNotExist(err) {
		t.Fatal("extra activation wrote original lock")
	}
	must(t, os.WriteFile(filepath.Join(library, "helper.py"), []byte("VALUE=12\n"), 0600))
	_, edited := capture("managed")
	if edited == one {
		t.Fatal("selected same-version helper edit did not change captured closure")
	}
	changed := strings.Replace(metadata, "private-extra-library==0.0.1", "private-extra-library>=0.0.1", 1)
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(changed), 0600))
	_, repinned := capture("managed")
	if repinned == edited {
		t.Fatal("selected dependency requirement edit did not change capture")
	}
}
