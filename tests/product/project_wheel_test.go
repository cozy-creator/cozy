package producttest

// Actual publication wheels must contain their declared import and application.
// Editable installations transport source and are built by their consuming uv environment.

import (
	"archive/zip"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/wheel"
)

const noImportRoots = "project_wheel_no_import_roots"

func TestProjectWheelImportRoots(t *testing.T) {
	project := weightlessProject(t)
	broken := brokenWeightlessProject(t, project)
	unregistered := weightlessVariant(t, project, func(pyproject string) string {
		return strings.Replace(pyproject, applicationTable, "", 1)
	})
	disagreeing := weightlessVariant(t, project, func(pyproject string) string {
		return strings.Replace(pyproject, `default = "weightless:app"`, `default = "weightless:other"`, 1)
	})
	for _, tc := range []struct{ name, tree, refusal string }{
		{"empty", broken, noImportRoots},
		{"unregistered", unregistered, "project_wheel_application_entrypoint_missing"},
		{"disagreeing", disagreeing, "project_wheel_application_entrypoint_mismatch"},
		{"valid", project, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pack, problem := packagepublish.PrepareLocalFrom(tc.tree)
			fatal(t, problem)
			defer pack.Close()
			problem = pack.Build(t.Context())
			if tc.refusal != "" {
				if problem == nil || problem.Name != tc.refusal {
					t.Fatalf("wheel publication answered %v, want %s", problem, tc.refusal)
				}
				return
			}
			fatal(t, problem)
			contents, problem := wheel.InspectContents(pack.Wheel)
			fatal(t, problem)
			if strings.Join(contents.ImportRoots, ",") != "weightless" {
				t.Fatalf("published wheel import roots: %v", contents.ImportRoots)
			}
			applications := contents.Group("cozy.application")
			if len(applications) != 1 || applications[0].Object != "weightless:app" {
				t.Fatalf("published application: %+v", applications)
			}
		})
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
