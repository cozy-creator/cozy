package producttest

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestLocalApplicationEntryPointUsesOrdinaryDependencyCapture(t *testing.T) {
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
	fatal(t, problem) // The installed wheel entry point supplies its application metadata.
	must(t, os.WriteFile(filepath.Join(library, "package.toml"), []byte("[application]\nobject='tools:app'\n"), 0o600))
	_, problem = packagepublish.LocalDependencySelections(root)
	fatal(t, problem)

	// Native facade providers may be supplied as exact wheel files. They are not
	// source directories and cannot be inspected through wheel/pyproject.toml.
	wheelPath := filepath.Join(root, "cozy_runtime-0.11.0-py3-none-any.whl")
	file, err := os.Create(wheelPath)
	must(t, err)
	wheel := zip.NewWriter(file)
	member, err := wheel.Create("cozy_runtime-0.11.0.dist-info/METADATA")
	must(t, err)
	_, err = member.Write([]byte("Metadata-Version: 2.3\nName: cozy-runtime\nVersion: 0.11.0\n"))
	must(t, err)
	must(t, wheel.Close())
	must(t, file.Close())
	must(t, os.WriteFile(filepath.Join(root, "pyproject.toml"), []byte(`[project]
name="caller"
version="0.0.1"
dependencies=["tools","cozy-runtime>=0.11.0"]
[tool.uv.sources]
tools={path="./library",editable=true}
cozy-runtime={path="./cozy_runtime-0.11.0-py3-none-any.whl"}
`), 0o600))
	selected, problem = packagepublish.LocalDependencySelections(root)
	fatal(t, problem)
	if selected["cozy-runtime"].Path != wheelPath { //cozy:allow immutable dependency wheel path, not a Runtime command
		t.Fatal("native facade wheel stopped being a dependency")
	}
}
