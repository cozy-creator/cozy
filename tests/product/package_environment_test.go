package producttest

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/wheel"
)

func TestPublishedPackageEnvironmentsKeepPythonAndTorchIndependent(t *testing.T) {
	type installed struct {
		python string
		torch  string
	}
	var environments []installed
	var oldProject, oldProjectWheel, newTorchWheel string
	for _, fixture := range []struct {
		name, python, torch string
	}{
		{name: "old", python: ">=3.11,<3.12", torch: "1.0.0"},
		{name: "new", python: ">=3.12,<3.13", torch: "2.0.0"},
	} {
		root := t.TempDir()
		torchProject := filepath.Join(root, "torch-"+fixture.torch)
		writePublishProject(t, torchProject, "torch", fixture.torch, nil, "", false)
		must(t, os.WriteFile(filepath.Join(torchProject, "torch", "__init__.py"),
			[]byte("VERSION = "+quote(fixture.torch)+"\n"), 0o644))

		project := filepath.Join(root, "package-"+fixture.name)
		writePublishProject(t, project, "package-"+fixture.name, "1.0.0",
			[]string{"torch==" + fixture.torch},
			"torch = { path = "+quote("../torch-"+fixture.torch)+" }\n", false)
		mutateFixtureFile(t, filepath.Join(project, "pyproject.toml"),
			"version = \"1.0.0\"\n", "version = \"1.0.0\"\nrequires-python = "+quote(fixture.python)+"\n")
		lockPublishProject(t, project)

		projectWheel, problem := wheel.Build(wheel.Request{Context: context.Background(),
			Tree: project, OutDir: filepath.Join(root, "project-wheel")})
		fatal(t, problem)
		torchWheel, problem := wheel.Build(wheel.Request{Context: context.Background(),
			Tree: torchProject, OutDir: filepath.Join(root, "torch-wheel")})
		fatal(t, problem)
		if fixture.name == "old" {
			oldProject, oldProjectWheel = project, projectWheel.Path
		} else {
			newTorchWheel = torchWheel.Path
		}
		must(t, os.RemoveAll(torchProject))

		environment := filepath.Join(root, "environment")
		receipt, problem := install.MaterializePublishedEnvironment(project, environment,
			install.PublishedWheel{Distribution: "package-" + fixture.name, Path: projectWheel.Path},
			[]install.PublishedWheel{{Distribution: "torch", Path: torchWheel.Path}})
		fatal(t, problem)
		python := filepath.Join(environment, "bin", "python")
		command := exec.Command(python, "-c",
			"import sys,torch; print(f'{sys.version_info.major}.{sys.version_info.minor} {torch.VERSION}')")
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("run %s package environment: %v\n%s", fixture.name, err, output)
		}
		parts := strings.Fields(string(output))
		if len(parts) != 2 || parts[1] != fixture.torch || !strings.HasPrefix(fixture.python, ">="+parts[0]) {
			t.Fatalf("%s environment ran %q, receipt=%+v", fixture.name, output, receipt)
		}
		environments = append(environments, installed{python: parts[0], torch: parts[1]})
	}
	if environments[0] == environments[1] {
		t.Fatalf("incompatible package environments collapsed together: %+v", environments)
	}
	_, problem := install.MaterializePublishedEnvironment(oldProject, filepath.Join(t.TempDir(), "bad"),
		install.PublishedWheel{Distribution: "package-old", Path: oldProjectWheel},
		[]install.PublishedWheel{{Distribution: "torch", Path: newTorchWheel}})
	if problem == nil || problem.Name != "package_requirement_incompatible" ||
		!strings.Contains(problem.Message, "torch==1.0.0") {
		t.Fatalf("true locked conflict did not name its exact requirement: %#v", problem)
	}
}

func TestModelPrefetchStatusMakesLocalReuseClearWithoutIdentityNoise(t *testing.T) {
	cases := []struct {
		models []install.PublishedModel
		want   string
	}{
		{want: "none"},
		{models: []install.PublishedModel{{}}, want: "downloaded"},
		{models: []install.PublishedModel{{Reused: true}}, want: "reused locally"},
		{models: []install.PublishedModel{{Reused: true}, {}}, want: "downloaded + local reuse"},
	}
	for _, tc := range cases {
		got := install.ModelPrefetchStatus(tc.models)
		if got != tc.want || strings.Contains(got, "sha256") {
			t.Fatalf("prefetch status = %q, want %q without digest noise", got, tc.want)
		}
	}
}

func quote(value string) string {
	return `"` + strings.ReplaceAll(value, `"`, `\"`) + `"`
}
