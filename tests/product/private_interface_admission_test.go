package producttest

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestPrivateInterfaceRefusesOlderDependencyRuntimeBeforeGeneration(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the metadata-only peer is a POSIX shell script")
	}
	for _, version := range []string{"0.9.1", "invalid", "0.11.0", "0.12.0"} {
		t.Run(version, func(t *testing.T) {
			root := t.TempDir()
			tool, called := filepath.Join(root, "runtime"), filepath.Join(root, "generation-called")
			body := fmt.Sprintf("#!/bin/sh\nfor arg in \"$@\"; do\nif [ \"$arg\" = version ]; then\nprintf '%%s\\n' '{\"distribution\":\"%s\"}'\nexit 0\nfi\ndone\n: > %q\nexit 17\n", version, called)
			must(t, os.WriteFile(tool, []byte(body), 0o700))
			peer := launch.RuntimeCLI{Bin: tool, Dir: root, Home: root} //cozy:allow metadata-only admission peer
			_, problem := peer.InterfaceWheel(context.Background(), "interface.json", "local-library", "1.0.0", "implementation", "source.whl", "digest", root)
			_, generated := os.Stat(called)
			if version == "0.9.1" || version == "invalid" {
				if problem == nil || problem.ErrName() != "interface_runtime_below_floor" || !strings.Contains(problem.Remedy, "uv lock --upgrade-package cozy-runtime") || !os.IsNotExist(generated) {
					t.Fatalf("old dependency Runtime received ABI5 generation instead of its lock remedy: %v, generated=%v", problem, generated)
				}
			} else if generated != nil || problem == nil || problem.ErrName() == "interface_runtime_below_floor" {
				t.Fatalf("supported dependency Runtime did not reach generation: %v, generated=%v", problem, generated)
			}
		})
	}
}

func TestLocalApplicationDeclarationFailsBeforeCapture(t *testing.T) {
	root := t.TempDir()
	library := filepath.Join(root, "library")
	must(t, os.MkdirAll(library, 0o700))
	must(t, os.WriteFile(filepath.Join(root, "pyproject.toml"), []byte(`[project]
name="caller"
version="0.0.1"
dependencies=["tools"]
[tool.uv.sources]
tools={path="./library",editable=true}
`), 0o600))
	metadata := "[project]\nname='tools'\nversion='0.0.1'\n"
	path := filepath.Join(library, "pyproject.toml")
	must(t, os.WriteFile(path, []byte(metadata), 0o600))
	selected, problem := packagepublish.LocalDependencySelections(root)
	fatal(t, problem)
	if selected["tools"].Path != library {
		t.Fatal("ordinary Python library stopped being a dependency")
	}
	must(t, os.WriteFile(path, []byte(metadata+"[project.entry-points.'cozy.application']\ndefault='tools:app'\n"), 0o600))
	_, problem = packagepublish.LocalDependencySelections(root)
	if problem == nil || problem.ErrName() != "local_application_declaration_missing" || !strings.Contains(problem.Remedy, "[application]") {
		t.Fatalf("declared local application was silently treated as an ordinary library: %v", problem)
	}
	must(t, os.WriteFile(filepath.Join(library, "package.toml"), []byte("[application]\nobject='tools:app'\n"), 0o600))
	_, problem = packagepublish.LocalDependencySelections(root)
	fatal(t, problem)
}
