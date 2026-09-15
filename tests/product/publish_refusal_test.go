package producttest

// cl-084: publish refuses author-local uv.lock rows and
// nudges toward publishing when it auto-vendors an in-tree local dependency.
// Every arm drives the real publication staging path (PrepareFrom +
// BuildForPublish) over a real project tree; the frozen editable lock below is
// the exact marco-polo row shape (`cozy-runtime = { editable = "../../" }`)
// that motivated the refusal. An editable install of the same tree keeps its
// old legality: only publication is fenced.

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

const fixtureBuildSystem = `
[build-system]
requires = ["uv_build>=0.1,<1000"]
build-backend = "uv_build"
`

// The runtime name appears here only as frozen uv.lock fixture BYTES — the
// exact marco-polo row — never as an executed binary.
const runtimeDistribution = "cozy-runtime" //cozy:allow lock-row fixture bytes, nothing shells out

var editableRuntimeLock = fmt.Sprintf(`version = 1
revision = 3
requires-python = "==3.12.*"

[[package]]
name = "cozy-fixture-package"
version = "1.0.0"
source = { editable = "." }
dependencies = [
    { name = %[1]q },
]

[[package]]
name = %[1]q
version = "0.0.37"
source = { editable = "../../" }
`, runtimeDistribution)

const machinePathLock = `version = 1
revision = 3
requires-python = "==3.12.*"

[[package]]
name = "cozy-fixture-package"
version = "1.0.0"
source = { editable = "." }

[[package]]
name = "cozy-utils"
version = "0.2.0"
source = { directory = "/opt/checkouts/cozy-utils" }
`

const minimalFixtureLock = `version = 1
revision = 3
requires-python = "==3.12.*"

[[package]]
name = "cozy-fixture-package"
version = "1.0.0"
source = { editable = "." }
`

func fixturePyproject(dependencies ...string) string {
	quoted := make([]string, 0, len(dependencies))
	for _, dependency := range dependencies {
		quoted = append(quoted, fmt.Sprintf("%q", dependency))
	}
	return "[project]\n" +
		`name = "cozy-fixture-package"` + "\n" +
		`version = "1.0.0"` + "\n" +
		`requires-python = ">=3.12,<3.13"` + "\n" +
		"dependencies = [" + strings.Join(quoted, ", ") + "]\n\n" +
		"[project.entry-points.\"cozy.application\"]\n" +
		`default = "cozy_fixture_package:app"` + "\n" +
		fixtureBuildSystem
}

func fixtureTree(t *testing.T, pyproject, lock string) string {
	t.Helper()
	dir := t.TempDir()
	module := filepath.Join(dir, "src", "cozy_fixture_package")
	must(t, os.MkdirAll(module, 0o755))
	must(t, os.WriteFile(filepath.Join(module, "__init__.py"),
		[]byte("app = object()\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(dir, "pyproject.toml"), []byte(pyproject), 0o644))
	must(t, os.WriteFile(filepath.Join(dir, "package.toml"),
		[]byte("[application]\nobject = \"cozy_fixture_package:app\"\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(dir, "uv.lock"), []byte(lock), 0o644))
	return dir
}

func buildForPublish(t *testing.T, tree string) *packagepublish.Package {
	t.Helper()
	pack, problem := packagepublish.PrepareFrom(tree)
	fatal(t, problem)
	t.Cleanup(pack.Close)
	fatal(t, pack.BuildForPublish(t.Context()))
	return pack
}

func TestPublishRefusesAuthorLocalLockRows(t *testing.T) {
	tree := fixtureTree(t, fixturePyproject("cozy-runtime>=0.0,<2"), editableRuntimeLock)
	pack, problem := packagepublish.PrepareFrom(tree)
	fatal(t, problem)
	defer pack.Close()
	problem = pack.BuildForPublish(t.Context())
	if problem == nil || problem.Name != "package_publish.author_local_row" {
		t.Fatalf("editable cozy-runtime lock answered %v", problem)
	}
	want := `uv.lock row cozy-runtime 0.0.37 (editable = "../../") resolves outside the package tree and cannot travel`
	if problem.Message != want {
		t.Fatalf("refusal message = %q, want %q", problem.Message, want)
	}
	if !strings.Contains(problem.Remedy, "publish cozy-runtime to PyPI or your org index") ||
		!strings.Contains(problem.Remedy, "move it into the package tree") {
		t.Fatalf("refusal remedy = %q", problem.Remedy)
	}

	// A machine path that is not even editable is the same untransportable class.
	machine := fixtureTree(t, fixturePyproject("cozy-utils>=0.2,<1"), machinePathLock)
	machinePack, problem := packagepublish.PrepareFrom(machine)
	fatal(t, problem)
	defer machinePack.Close()
	problem = machinePack.BuildForPublish(t.Context())
	if problem == nil || problem.Name != "package_publish.author_local_row" ||
		!strings.Contains(problem.Message, `cozy-utils 0.2.0 (directory = "/opt/checkouts/cozy-utils")`) {
		t.Fatalf("machine-path lock answered %v", problem)
	}

	// The refusal is publish-only: the plain (editable/private) build of the
	// exact same tree never fires it.
	editable, problem := packagepublish.PrepareFrom(tree)
	fatal(t, problem)
	defer editable.Close()
	problem = editable.Build(t.Context())
	if problem != nil && strings.HasPrefix(problem.Name, "package_publish.") {
		t.Fatalf("non-publish build fired a publish refusal: %v", problem)
	}
}

// TestPublishStagesInTreeDependencies proves the auto-vendor path is unchanged
// for in-tree local deps: over a REAL `uv lock` of a tree with one in-tree path
// dependency, publication staging passes both cl-084 gates, builds the project
// and dependency wheels, and stops only at the PackageInterface step (this fixture
// deliberately has no cozy-runtime to describe it).
func TestPublishStagesInTreeDependencies(t *testing.T) {
	tree := fixtureTree(t, fixturePyproject("cozy-fixture-helper>=0.0.1")+`
[tool.uv.sources]
cozy-fixture-helper = { path = "libs/helper" }
`, minimalFixtureLock)
	writeHelperProject(t, filepath.Join(tree, "libs", "helper"),
		"cozy-fixture-helper", "cozy_fixture_helper", "0.0.1")
	lock := exec.Command("uv", "lock", "--offline")
	lock.Dir = tree
	if out, err := lock.CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v\n%s", err, out)
	}
	pack, problem := packagepublish.PrepareFrom(tree)
	fatal(t, problem)
	defer pack.Close()
	problem = pack.BuildForPublish(t.Context())
	if problem == nil || problem.Name != "package_interface_refused" {
		t.Fatalf("in-tree dependency tree stopped early: %v", problem)
	}
}

func writeHelperProject(t *testing.T, dir, distribution, module, version string) {
	t.Helper()
	must(t, os.MkdirAll(filepath.Join(dir, "src", module), 0o755))
	body := "[project]\n" +
		fmt.Sprintf("name = %q\n", distribution) +
		fmt.Sprintf("version = %q\n", version) +
		`requires-python = ">=3.12,<3.13"` + "\n" +
		"dependencies = []\n" +
		fixtureBuildSystem
	must(t, os.WriteFile(filepath.Join(dir, "pyproject.toml"), []byte(body), 0o644))
	must(t, os.WriteFile(filepath.Join(dir, "src", module, "__init__.py"),
		[]byte(fmt.Sprintf("__version__ = %q\n", version)), 0o644))
}

// TestPublishBuildAcceptsCompliantPackage is the green arm over the runtime
// peer fixture: an image-owned RANGE plus one in-tree local dependency builds
// the complete publication staging — PackageInterface included — vendors exactly the
// in-tree dep, keeps the roster wheel in source custody, and reports the
// th-113 publish nudge.
func TestPublishBuildAcceptsCompliantPackage(t *testing.T) {
	project := weightlessProject(t)

	pyprojectPath := filepath.Join(project, "pyproject.toml")
	metadata, err := os.ReadFile(pyprojectPath)
	must(t, err)
	pinned := regexp.MustCompile(`"cozy-runtime\[media\]>=([0-9][0-9a-zA-Z.]*),<1"`)
	match := pinned.FindStringSubmatch(string(metadata))
	if match == nil {
		t.Fatalf("weightless fixture no longer declares its bounded cozy-runtime range:\n%s", metadata)
	}
	ranged := fmt.Sprintf(`"cozy-runtime[media]>=%s,<2", "cozy-weightless-helper==0.0.1"`, match[1])
	rewritten := strings.Replace(string(metadata), match[0], ranged, 1)
	// [tool.uv.sources] is the fixture pyproject's last table.
	rewritten += "cozy-weightless-helper = { path = \"libs/helper\" }\n"
	must(t, os.WriteFile(pyprojectPath, []byte(rewritten), 0o644))

	writeHelperProject(t, filepath.Join(project, "libs", "helper"),
		"cozy-weightless-helper", "cozy_weightless_helper", "0.0.1")
	lock := exec.Command("uv", "lock")
	lock.Dir = project
	if out, err := lock.CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v\n%s", err, out)
	}

	pack := buildForPublish(t, project)
	if len(pack.Vendored) != 1 || pack.Vendored[0].Name != "cozy-weightless-helper" ||
		pack.Vendored[0].Version != "0.0.1" {
		t.Fatalf("vendored = %+v", pack.Vendored)
	}
	note := packagepublish.VendoredNote("acme", pack.Vendored[0])
	want := "Vendored cozy-weightless-helper 0.0.1 (unpublished). " +
		"Next: publish it and depend on acme/cozy-weightless-helper instead"
	if note != want {
		t.Fatalf("nudge = %q, want %q", note, want)
	}
	if len(pack.DependencyWheels) != 1 ||
		!strings.HasPrefix(pack.DependencyWheels[0].Filename, "cozy_weightless_helper-0.0.1-") {
		t.Fatalf("dependency wheels = %+v", pack.DependencyWheels)
	}
	if len(pack.Registry) != 0 {
		t.Fatalf("registry rows = %+v", pack.Registry)
	}
	if pack.PackageInterface == "" || pack.Wheel == "" {
		t.Fatalf("staging incomplete: wheel %q package interface %q", pack.Wheel, pack.PackageInterface)
	}
}
