package producttest

import (
	"archive/zip"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

// Exercise real source admission, dependency discovery, snapshot copying and
// identity checks, stopping at the final metadata fence before any Python build.
func TestEditableSnapshotLargeWheelKeepsSourceIdentityFence(t *testing.T) {
	for _, location := range []string{"external", "inside"} {
		t.Run(location, func(t *testing.T) {
			root := t.TempDir()
			project := filepath.Join(root, "project")
			must(t, os.MkdirAll(project, 0o700))
			wheelDir := root
			if location == "inside" {
				wheelDir = project
			}
			wheelPath := filepath.Join(wheelDir, "snapshot_dependency-1.0-py3-none-any.whl")
			file, err := os.Create(wheelPath)
			must(t, err)
			// A sparse prefix makes the valid ZIP larger than the ordinary source
			// limit without checking in a giant fixture or allocating it in RAM.
			prefix := packagepublish.MaxSourceFileBytes + 1
			_, err = file.Seek(prefix, io.SeekStart)
			must(t, err)
			archive := zip.NewWriter(file)
			archive.SetOffset(prefix)
			for name, body := range map[string]string{
				"snapshot_dependency/__init__.py":            "VALUE = 1\n",
				"snapshot_dependency-1.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: snapshot-dependency\nVersion: 1.0\n",
				"snapshot_dependency-1.0.dist-info/WHEEL":    "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
			} {
				member, err := archive.Create(name)
				must(t, err)
				_, err = io.WriteString(member, body)
				must(t, err)
			}
			must(t, archive.Close())
			must(t, file.Close())
			rel, err := filepath.Rel(project, wheelPath)
			must(t, err)
			metadata := "[project]\nname='snapshot-root'\nversion='1.0'\nrequires-python='>=3.12'\ndependencies=['snapshot-dependency>=1,<2']\n[tool.uv.sources]\nsnapshot-dependency={path='" + filepath.ToSlash(rel) + "'}\n"
			must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0o600))
			must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject='snapshot_root:app'\n"), 0o600))
			must(t, os.WriteFile(filepath.Join(project, "uv.lock"), []byte("version = 1\n"), 0o600))
			pack, problem := packagepublish.PrepareLocalFrom(project)
			fatal(t, problem)
			defer pack.Close()
			digest, count, size, problem := pack.SourceIdentity()
			fatal(t, problem)
			// The deliberately stale release reaches the snapshot's final identity
			// fence only after every large dependency has copied and rehashed.
			local := &install.LocalSource{Package: "local/snapshot-root", Release: "2.0", Tree: project, Files: count, Bytes: size}
			_, problem = install.Run(home.Layout{Installs: filepath.Join(root, "installs")}, nil, install.Request{Snapshot: true, Local: local})
			if problem == nil || problem.Name != "local_package_source_changed" || !strings.Contains(problem.Error(), "package changed while capturing its invocation snapshot") {
				t.Fatalf("large admitted wheel did not reach the unchanged metadata identity fence: %v", problem)
			}
			// An ordinary member of the same size must still fail source admission.
			ordinary := filepath.Join(project, "ordinary.py")
			file, err = os.Create(ordinary)
			must(t, err)
			must(t, file.Truncate(prefix))
			must(t, file.Close())
			_, problem = packagepublish.PrepareLocalFrom(project)
			if problem == nil || problem.Name != "package_source_file_too_large" {
				t.Fatalf("ordinary oversized source was accepted: %v", problem)
			}
		})
	}
}
