package producttest

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func lockedGitFixture(t *testing.T) string {
	t.Helper()
	repository, commit := serveGitProject(t)
	project := fixtureTree(t, fixturePyproject("cozy-fixture-pinned @ git+"+repository+"@"+commit), minimalFixtureLock)
	command := exec.CommandContext(t.Context(), "uv", "lock", "--directory", project)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("lock Git fixture: %v\n%s", err, output)
	}
	return project
}

func TestSelectedGitWheelCaptureDoesNotMisclassifyItAsRegistry(t *testing.T) {
	project := lockedGitFixture(t)
	selected := map[string]map[string]string{"caller": {"cozy-fixture-package": "1.0.0", "cozy-fixture-pinned": "0.3.0.dev0"}}
	got, problem := packagepublish.CaptureWheelDependencies(t.Context(), project, "cozy-fixture-package", "cozy-fixture-package==1.0.0\ncozy-fixture-pinned==0.3.0.dev0", t.TempDir(), selected)
	fatal(t, problem)
	dependency := got["cozy-fixture-pinned"]
	if dependency.Path == "" || dependency.Version != "0.3.0.dev0" || dependency.RegistryRequirement != "" || !strings.HasPrefix(dependency.Digest, "sha256:") {
		t.Fatalf("Git dependency is not carried as verified bytes: %+v", dependency)
	}
}

func TestCapturedGitSourceInstallsWithoutGitOrRepositoryAccess(t *testing.T) {
	project := lockedGitFixture(t)
	originalMetadata, err := os.ReadFile(filepath.Join(project, "pyproject.toml"))
	must(t, err)
	originalLock, err := os.ReadFile(filepath.Join(project, "uv.lock"))
	must(t, err)
	capture := filepath.Join(t.TempDir(), "captured")
	pack, problem := packagepublish.SnapshotSource(t.Context(), project, capture)
	fatal(t, problem)
	defer pack.Close()
	metadata, err := os.ReadFile(filepath.Join(capture, "pyproject.toml"))
	must(t, err)
	lock, err := os.ReadFile(filepath.Join(capture, "uv.lock"))
	must(t, err)
	if strings.Contains(string(metadata), "git+") || strings.Contains(string(lock), "git =") || !strings.Contains(string(lock), "cozy_fixture_pinned-0.3.0.dev0-") {
		t.Fatalf("captured installation still needs Git or lacks its wheel:\n%s\n%s", metadata, lock)
	}
	for name, want := range map[string][]byte{"pyproject.toml": originalMetadata, "uv.lock": originalLock} {
		got, err := os.ReadFile(filepath.Join(project, name))
		must(t, err)
		if string(got) != string(want) {
			t.Fatalf("capture changed author %s", name)
		}
	}
	// A receiving interpreter with an empty cache, no Git on PATH, and no network
	// installs the frozen dependency from the copied wheel, not the repository.
	uv, err := exec.LookPath("uv")
	must(t, err)
	python, err := exec.CommandContext(t.Context(), uv, "python", "find", "--no-python-downloads", "3.12").Output()
	must(t, err)
	environment := append([]string{}, os.Environ()...)
	for i := len(environment) - 1; i >= 0; i-- {
		if strings.HasPrefix(environment[i], "PATH=") || strings.HasPrefix(environment[i], "UV_CACHE_DIR=") {
			environment = append(environment[:i], environment[i+1:]...)
		}
	}
	environment = append(environment, "PATH="+t.TempDir(), "UV_CACHE_DIR="+t.TempDir(), "UV_PYTHON_DOWNLOADS=never")
	command := exec.CommandContext(t.Context(), uv, "sync", "--frozen", "--offline", "--no-install-project", "--no-dev", "--no-default-groups", "--python", strings.TrimSpace(string(python)), "--no-python-downloads")
	command.Dir = capture
	command.Env = environment
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("Git-free receiving sync: %v\n%s", err, output)
	}
	command = exec.CommandContext(t.Context(), filepath.Join(capture, ".venv", "bin", "python"), "-c", "import cozy_fixture_pinned; print(cozy_fixture_pinned.__version__)")
	if output, err := command.CombinedOutput(); err != nil || strings.TrimSpace(string(output)) != "0.3.0.dev0" {
		t.Fatalf("captured dependency did not execute: %v %s", err, output)
	}
	// Capturing the already portable tree does not rebuild or require Git either.
	repeated, problem := packagepublish.SnapshotSource(t.Context(), capture, filepath.Join(t.TempDir(), "repeated"))
	fatal(t, problem)
	defer repeated.Close()
}

func TestCapturedGitRejectsChangedLockedCommit(t *testing.T) {
	project := lockedGitFixture(t)
	p := filepath.Join(project, "uv.lock")
	raw, err := os.ReadFile(p)
	must(t, err)
	at := strings.Index(string(raw), "#")
	if at < 0 {
		t.Fatal("fixture has no locked Git commit")
	}
	changed := append([]byte(nil), raw...)
	copy(changed[at+1:at+41], strings.Repeat("0", 40))
	must(t, os.WriteFile(p, changed, 0600))
	pack, problem := packagepublish.SnapshotSource(t.Context(), project, filepath.Join(t.TempDir(), "capture"))
	if pack != nil {
		pack.Close()
	}
	if problem == nil || problem.Name != "private_dependency_git_lock_drift" {
		t.Fatalf("changed lock commit was not refused before capture: %v", problem)
	}
}

func TestCapturedGitSkipsInactiveExtra(t *testing.T) {
	repository, commit := serveGitProject(t)
	project := fixtureTree(t, fixturePyproject()+"\n[project.optional-dependencies]\nunused=['cozy-fixture-pinned @ git+"+repository+"@"+commit+"']\n", minimalFixtureLock)
	command := exec.CommandContext(t.Context(), "uv", "lock", "--directory", project)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("lock optional Git fixture: %v\n%s", err, output)
	}
	capture := filepath.Join(t.TempDir(), "captured")
	pack, problem := packagepublish.SnapshotSource(t.Context(), project, capture)
	fatal(t, problem)
	defer pack.Close()
	if _, err := os.Stat(filepath.Join(capture, ".cozy-dependencies", "git")); !os.IsNotExist(err) {
		t.Fatalf("inactive Git extra was built: %v", err)
	}
}

func TestCapturedGitDeclaredByLocalChildKeepsItsClosure(t *testing.T) {
	repository, commit := serveGitProject(t)
	project := fixtureTree(t, fixturePyproject("cozy-local-helper")+"\n[tool.uv.sources]\ncozy-local-helper={path='helper'}\n", minimalFixtureLock)
	child := filepath.Join(project, "helper")
	writeHelperProject(t, child, "cozy-local-helper", "cozy_local_helper", "1.0.0")
	p := filepath.Join(child, "pyproject.toml")
	raw, err := os.ReadFile(p)
	must(t, err)
	body := strings.Replace(string(raw), "dependencies = []", "dependencies = ['cozy-fixture-pinned @ git+"+repository+"@"+commit+"']", 1)
	must(t, os.WriteFile(p, []byte(body), 0600))
	command := exec.CommandContext(t.Context(), "uv", "lock", "--directory", project)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("lock child Git fixture: %v\n%s", err, output)
	}
	capture := filepath.Join(t.TempDir(), "captured")
	pack, problem := packagepublish.SnapshotSource(t.Context(), project, capture)
	fatal(t, problem)
	defer pack.Close()
	wheels, err := filepath.Glob(filepath.Join(capture, ".cozy-dependencies", "git", "cozy-fixture-pinned", "*.whl"))
	must(t, err)
	if len(wheels) != 1 {
		t.Fatalf("child Git dependency was not retained: %v", wheels)
	}
	current, err := os.ReadFile(p)
	must(t, err)
	if string(current) != body {
		t.Fatal("capture rewrote the authored local child")
	}
}

func TestUnretainedGitSourceIsNotReportedAsARegistryPlatformMismatch(t *testing.T) {
	project := lockedGitFixture(t)
	lock, err := os.ReadFile(filepath.Join(project, "uv.lock"))
	must(t, err)
	_, _, problem := packagepublish.CapturedRegistryRows(lock, "cozy-fixture-package==1.0.0\ncozy-fixture-pinned==0.3.0.dev0", "cozy-fixture-package", "1.0.0", nil)
	if problem == nil || problem.Name != "registry_dependency_git_undeclared" {
		t.Fatalf("unretained Git dependency was misclassified: %v", problem)
	}
}
