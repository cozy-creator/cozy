package producttest

// cl-175 — package code is untrusted; analysis never executes it. Every reading of a package's
// surface is THIS host's cozy-runtime (>= hostruntime.ToolFloor) parsing source: at publish
// pre-flight, at install, and for a job's descriptor id. So a package whose module imports
// torch at the top installs and publishes on a host whose package environment has no torch —
// the fixture's venv is `cozy-runtime[media]` and this test adds nothing to it. A second
// top-level import names a module that exists nowhere, so the proof does not rest on what
// the host tool's own environment happens to carry. Serving the package would fail exactly
// where package code legitimately runs, in the worker, and nowhere earlier.

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestStaticDescribeNeverImportsPackageCode(t *testing.T) {
	root := filepath.Join(scratchBase, "static-describe")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		_ = os.RemoveAll(root)
	})
	const future = "from __future__ import annotations\n"
	project := weightlessFileVariant(t, weightlessProject(t), "weightless.py", func(source string) string {
		return strings.Replace(source, future,
			future+"\nimport torch  # noqa: F401 — absent from this package's environment\n"+
				"import cozy_cl175_module_that_exists_nowhere  # noqa: F401 — a static reading never runs this line\n", 1)
	})

	// (a) Install reads the editable tree's surface without importing weightless.py.
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("a module importing torch at the top did not install on a torch-less host [exit %d]\n%s", code, out)
	}
	install := activePackageInstall(t, root)
	python := home.VenvPython(filepath.Join(install.Dir, "venv"))
	if out, err := exec.Command(python, "-c", "import torch").CombinedOutput(); err == nil {
		t.Fatalf("the package environment %s carries torch; the proof needs it absent\n%s", python, out)
	}

	// (b) A job's descriptor id is read the same way, by the same host tool.
	facts, problem := launch.Read(install, root, childEnv(t, root))
	fatal(t, problem)
	job, problem := facts.Job("tile_job")
	fatal(t, problem)
	if job.DescriptorID == "" {
		t.Fatal("tile_job was read without a descriptor id")
	}

	// (c) Publish pre-flight: the committed interface equals the static reading and is
	// staged unchanged.
	committedPath := filepath.Join(project, packagepublish.CommittedInterfacePath)
	committed, err := os.ReadFile(committedPath)
	must(t, err)
	pack, problem := packagepublish.PrepareFrom(project)
	fatal(t, problem)
	fatal(t, pack.BuildForPublish(t.Context()))
	staged, err := os.ReadFile(pack.PackageInterface)
	pack.Close()
	must(t, err)
	if !bytes.Equal(staged, bytes.TrimSpace(committed)) {
		t.Fatalf("publication staged a different interface than the committed file\nstaged:\n%s\ncommitted:\n%s",
			staged, committed)
	}

	// (d) A committed file that no longer matches the tree is refused before upload, naming
	// the first difference; an absent one is refused by the same name.
	stale := strings.Replace(string(committed), `"name":"tile"`, `"name":"tiles"`, 1)
	if stale == string(committed) {
		t.Fatalf("the committed interface names no tile entrypoint:\n%s", committed)
	}
	must(t, os.WriteFile(committedPath, []byte(stale), 0o644))
	problem = publishStaging(t, project)
	if problem == nil || problem.Name != "package_publish.interface_stale" ||
		!strings.Contains(problem.Message, "entrypoints[") ||
		!strings.Contains(problem.Message, `committed "tiles", tree "tile"`) {
		t.Fatalf("a stale committed interface was not refused naming the first difference: %v", problem)
	}
	must(t, os.Remove(committedPath))
	problem = publishStaging(t, project)
	if problem == nil || problem.Name != "package_publish.interface_stale" ||
		!strings.Contains(problem.Message, packagepublish.CommittedInterfacePath+" is absent") {
		t.Fatalf("an absent committed interface was not refused by name: %v", problem)
	}
}

func publishStaging(t *testing.T, project string) *exit.Error {
	t.Helper()
	pack, problem := packagepublish.PrepareFrom(project)
	fatal(t, problem)
	defer pack.Close()
	return pack.BuildForPublish(t.Context())
}
