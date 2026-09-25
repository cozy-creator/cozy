package producttest

// The defect class: `cozy package install --editable` accepted a tree, `cozy run`
// worked locally from the source tree, and the same package failed minutes later on
// a rented pod. Twice: first the build backend produced a wheel holding nothing but
// .dist-info (1144 bytes, top_level.txt said `vendor`), then the wheel carried the
// module but no `cozy.application` entry point — a worker discovers the application
// from the installed wheel, an editable run from package.toml, and only package.toml
// was declared. Every arm below drives the real binary or the real staging code over
// a real tree from the real fixture builder; each broken arm is the exact pyproject
// shape the builder used to emit.

import (
	"archive/zip"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/wheel"
)

const noImportRoots = "project_wheel_no_import_roots"

func TestProjectWheelImportRoots(t *testing.T) {
	root := filepath.Join(scratchBase, "project-wheel")
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

	// The second shape: the module is in the wheel, the application is not. And a
	// wheel whose entry point disagrees with package.toml is two applications, not one.
	unregistered := weightlessVariant(t, project, func(pyproject string) string {
		return strings.Replace(pyproject, applicationTable, "", 1)
	})
	code, out = runCozy(t, root, "package", "install", unregistered, "--editable", "--json")
	if code != 1 || !strings.Contains(out, `"code":"project_wheel_application_entrypoint_missing"`) {
		t.Fatalf("wheel without a cozy.application entry point was not refused at install [exit %d]\n%s", code, out)
	}
	for _, named := range []string{`[project.entry-points.\"cozy.application\"]`, `default = \"weightless:app\"`} {
		if !strings.Contains(out, named) {
			t.Errorf("entry point refusal omitted %q\n%s", named, out)
		}
	}
	disagreeing := weightlessVariant(t, project, func(pyproject string) string {
		return strings.Replace(pyproject, `default = "weightless:app"`, `default = "weightless:other"`, 1)
	})
	code, out = runCozy(t, root, "package", "install", disagreeing, "--editable", "--json")
	if code != 1 || !strings.Contains(out, `"code":"project_wheel_application_entrypoint_mismatch"`) ||
		!strings.Contains(out, "weightless:other") || !strings.Contains(out, "weightless:app") {
		t.Fatalf("entry point disagreeing with package.toml was not refused at install [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "package", "list", "--json"); code != 0 ||
		strings.Contains(out, localWeightlessRef) {
		t.Fatalf("a refused editable install left a pin behind [exit %d]\n%s", code, out)
	}

	// The fixed fixture installs, and the local revision a rental would receive
	// carries a project wheel that installs weightless.py and registers its application.
	code, out = runCozy(t, root, "package", "install", project, "--editable")
	if code != 0 {
		t.Fatalf("fixed fixture install [exit %d]\n%s", code, out)
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	revision, problem := localpackage.Stage(t.Context(), layout, activePackageInstall(t, root))
	fatal(t, problem)
	var transferred []string
	for _, file := range revision.Files {
		if file.Kind != "project" {
			continue
		}
		if !strings.HasPrefix(file.Path, layout.LocalPackages+string(os.PathSeparator)) {
			t.Fatalf("project wheel staged outside local packages: %s", file.Path)
		}
		contents, problem := wheel.InspectContents(file.Path)
		fatal(t, problem)
		if strings.Join(contents.ImportRoots, ",") != "weightless" {
			t.Fatalf("transferred project wheel %s has import roots %v, want [weightless]", file.Filename, contents.ImportRoots)
		}
		if record := wheelRecord(t, file.Path); !strings.Contains(record, "weightless.py,sha256=") {
			t.Fatalf("transferred project wheel RECORD carries no weightless.py:\n%s", record)
		}
		applications := contents.Group("cozy.application")
		if len(applications) != 1 || applications[0].Object != "weightless:app" {
			t.Fatalf("transferred project wheel registers cozy.application %+v, want one weightless:app", applications)
		}
		transferred = append(transferred, file.Filename)
	}
	if len(transferred) != 1 {
		t.Fatalf("local revision carried %d project wheels: %v", len(transferred), revision.Files)
	}
}

const applicationTable = "[project.entry-points.\"cozy.application\"]\ndefault = \"weightless:app\"\n\n"

// weightlessVariant is the fixture tree under a rewritten pyproject. The wheel fence
// runs before uv.lock is consulted, so the lock is left as built.
func weightlessVariant(t *testing.T, project string, rewrite func(pyproject string) string) string {
	t.Helper()
	return weightlessFileVariant(t, project, "pyproject.toml", rewrite)
}

// weightlessFileVariant is the fixture tree with one file rewritten.
func weightlessFileVariant(t *testing.T, project, name string, rewrite func(content string) string) string {
	t.Helper()
	variant := filepath.Join(t.TempDir(), "variant")
	if out, err := exec.Command("cp", "-r", project, variant).CombinedOutput(); err != nil {
		t.Fatalf("copying the fixture: %v\n%s", err, out)
	}
	content, err := os.ReadFile(filepath.Join(project, name))
	must(t, err)
	rewritten := rewrite(string(content))
	if rewritten == string(content) {
		t.Fatalf("the variant rewrite changed nothing:\n%s", content)
	}
	must(t, os.WriteFile(filepath.Join(variant, name), []byte(rewritten), 0o644))
	return variant
}

// brokenWeightlessProject is the fixture tree under the pyproject the builder first
// emitted: no [build-system] and no entry point, so setuptools guessed a flat layout
// and packaged nothing.
func brokenWeightlessProject(t *testing.T, project string) string {
	t.Helper()
	broken := weightlessVariant(t, project, func(pyproject string) string {
		start := strings.Index(pyproject, applicationTable)
		end := strings.Index(pyproject, "[tool.uv.sources]")
		if start < 0 || end < start {
			t.Fatalf("fixture pyproject lost its shape:\n%s", pyproject)
		}
		return pyproject[:start] + pyproject[end:]
	})
	// The historical tree predates the committed interface (cl-175): with two stray
	// top-level directories setuptools refuses outright instead of emitting the empty wheel
	// this arm exists to catch.
	must(t, os.RemoveAll(filepath.Join(broken, "metadata")))
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
