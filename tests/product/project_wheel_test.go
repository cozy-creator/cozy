package producttest

// The defect: `cozy package install --editable` accepted a tree whose build backend
// produced a wheel holding nothing but .dist-info, `cozy run` worked locally from the
// source tree, and the same package failed minutes later on a rented pod as
// `runtime_preparation_failed`. The private revision Creator transferred carried a
// 1144-byte project wheel whose top_level.txt said `vendor`. Every arm below drives
// the real binary or the real staging code over a real tree from the real fixture
// builder; the broken arm is the exact pyproject shape the builder used to emit.

import (
	"archive/zip"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/privatepackage"
	"github.com/cozy-creator/cozy/internal/wheel"
)

const noImportRoots = "project_wheel_no_import_roots"

func TestProjectWheelImportRoots(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "project-wheel")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		_ = os.RemoveAll(root)
	})
	project := weightlessProject(t)
	broken := brokenWeightlessProject(t, project)

	// The old shape is refused where the human is: at install, before anything is
	// pinned, with the typed code, what the backend produced, and the fix.
	code, out := runCozy(t, root, "package", "install", broken, "--editable", "--json")
	if code != 1 || !strings.Contains(out, `"code":"`+noImportRoots+`"`) {
		t.Fatalf("empty project wheel was not refused at install [exit %d]\n%s", code, out)
	}
	for _, named := range []string{"cozy_weightless_package-1.0.0-py3-none-any.whl",
		"top_level.txt names vendor", "only its .dist-info metadata", "only-include"} {
		if !strings.Contains(out, named) {
			t.Errorf("install refusal omitted %q\n%s", named, out)
		}
	}
	if code, out := runCozy(t, root, "package", "list", "--json"); code != 0 ||
		strings.Contains(out, localWeightlessRef) {
		t.Fatalf("a refused editable install left a pin behind [exit %d]\n%s", code, out)
	}

	// The same fence on the path publish and private staging share.
	pack, problem := packagepublish.PrepareLocalFrom(broken)
	fatal(t, problem)
	defer pack.Close()
	if problem := pack.Build(t.Context()); problem == nil || problem.Name != noImportRoots {
		t.Fatalf("publish staging answered %v for the empty project wheel", problem)
	}
	built, problem := wheel.Build(wheel.Request{Context: t.Context(), Tree: broken, OutDir: filepath.Join(root, "broken-wheel")})
	fatal(t, problem)
	contents, problem := wheel.InspectContents(built.Path)
	fatal(t, problem)
	if len(contents.ImportRoots) != 0 || strings.Join(contents.TopLevel, ",") != "vendor" || len(contents.Members) != 0 {
		t.Fatalf("the backend's empty wheel measured as %+v", contents)
	}

	// The fixed fixture installs, and the private revision a rental would receive
	// carries a project wheel that installs weightless.py.
	code, out = runCozy(t, root, "package", "install", project, "--editable")
	if code != 0 {
		t.Fatalf("fixed fixture install [exit %d]\n%s", code, out)
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	revision, problem := privatepackage.Stage(t.Context(), layout, activePackageInstall(t, root))
	fatal(t, problem)
	var transferred []string
	for _, file := range revision.Files {
		if file.Kind != "project" {
			continue
		}
		if !strings.HasPrefix(file.Path, layout.PrivatePackages+string(os.PathSeparator)) {
			t.Fatalf("project wheel staged outside private packages: %s", file.Path)
		}
		contents, problem := wheel.InspectContents(file.Path)
		fatal(t, problem)
		if strings.Join(contents.ImportRoots, ",") != "weightless" {
			t.Fatalf("transferred project wheel %s has import roots %v, want [weightless]", file.Filename, contents.ImportRoots)
		}
		if record := wheelRecord(t, file.Path); !strings.Contains(record, "weightless.py,sha256=") {
			t.Fatalf("transferred project wheel RECORD carries no weightless.py:\n%s", record)
		}
		transferred = append(transferred, file.Filename)
	}
	if len(transferred) != 1 {
		t.Fatalf("private revision carried %d project wheels: %v", len(transferred), revision.Files)
	}
}

// brokenWeightlessProject is the fixture tree under the pyproject the builder used to
// emit: no [build-system], so setuptools guessed a flat layout and packaged nothing.
func brokenWeightlessProject(t *testing.T, project string) string {
	t.Helper()
	broken := filepath.Join(t.TempDir(), "broken")
	if out, err := exec.Command("cp", "-r", project, broken).CombinedOutput(); err != nil {
		t.Fatalf("copying the fixture: %v\n%s", err, out)
	}
	vendored, err := filepath.Glob(filepath.Join(broken, "vendor", "cozy_runtime-*.whl"))
	must(t, err)
	if len(vendored) != 1 {
		t.Fatalf("fixture vendors %d runtime wheels", len(vendored))
	}
	metadata, err := os.ReadFile(filepath.Join(project, "pyproject.toml"))
	must(t, err)
	dependencies := ""
	for _, line := range strings.Split(string(metadata), "\n") {
		if strings.HasPrefix(line, "dependencies = ") {
			dependencies = line
		}
	}
	if dependencies == "" {
		t.Fatalf("fixture pyproject declares no dependencies:\n%s", metadata)
	}
	must(t, os.WriteFile(filepath.Join(broken, "pyproject.toml"), []byte("[project]\n"+
		"name = \"cozy-weightless-package\"\nversion = \"1.0.0\"\n"+
		"requires-python = \">=3.12,<3.13\"\n"+dependencies+"\n\n"+
		"[tool.uv.sources]\ncozy-runtime = { path = \"vendor/"+filepath.Base(vendored[0])+"\" }\n"), 0o644))
	return broken
}

func wheelRecord(t *testing.T, path string) string {
	t.Helper()
	archive, err := zip.OpenReader(path)
	must(t, err)
	defer archive.Close()
	for _, member := range archive.File {
		if !strings.HasSuffix(member.Name, ".dist-info/RECORD") {
			continue
		}
		body, err := member.Open()
		must(t, err)
		defer body.Close()
		record, err := io.ReadAll(body)
		must(t, err)
		return string(record)
	}
	t.Fatalf("%s carries no RECORD", path)
	return ""
}
