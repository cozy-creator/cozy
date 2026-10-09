package producttest

// A third-party git dependency pinned to a full commit publishes as the wheel built at
// that commit, through the real staging path (PrepareFrom + BuildForPublish). The
// repository is a real one, served on loopback by git's own smart-HTTP backend.

import (
	"net/http/cgi"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func serveGitProject(t *testing.T) (repository, commit string) {
	t.Helper()
	root := t.TempDir()
	git := func(dir string, args ...string) string {
		t.Helper()
		command := exec.Command("git", append([]string{"-C", dir, "-c", "user.name=Paul Fidika",
			"-c", "user.email=paul@fidika.com", "-c", "init.defaultBranch=main"}, args...)...)
		out, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	work := filepath.Join(root, "work")
	writeHelperProject(t, work, "cozy-fixture-pinned", "cozy_fixture_pinned", "0.3.0.dev0")
	git(work, "init", "--quiet")
	git(work, "add", ".")
	git(work, "commit", "--quiet", "-m", "fixture")
	commit = git(work, "rev-parse", "HEAD")
	served := filepath.Join(root, "served")
	git(root, "clone", "--quiet", "--bare", work, filepath.Join(served, "pinned.git"))
	// Public hosts serve any reachable commit by its id.
	git(filepath.Join(served, "pinned.git"), "config", "uploadpack.allowAnySHA1InWant", "true")
	server := httptest.NewServer(&cgi.Handler{
		Path: filepath.Join(git(root, "--exec-path"), "git-http-backend"),
		Env:  []string{"GIT_PROJECT_ROOT=" + served, "GIT_HTTP_EXPORT_ALL=1"},
	})
	t.Cleanup(server.Close)
	return server.URL + "/pinned.git", commit
}

func TestPublishCarriesAGitDependencyPinnedToAFullCommit(t *testing.T) {
	repository, commit := serveGitProject(t)
	project := fixtureTree(t, fixturePyproject("cozy-fixture-pinned @ git+"+repository+"@"+commit), minimalFixtureLock)
	// A real application, so staging runs to its end; describe reads source, not an environment.
	application, err := os.ReadFile(filepath.Join("testdata", "weightless", "weightless.py"))
	must(t, err)
	must(t, os.WriteFile(filepath.Join(project, "src", "cozy_fixture_package", "__init__.py"), application, 0o644))
	lock := exec.Command("uv", "lock")
	lock.Dir = project
	if out, err := lock.CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v\n%s", err, out)
	}

	// The build tag names the wheel's bytes: a published filename is frozen per account.
	named := regexp.MustCompile(`^cozy_fixture_pinned-0\.3\.0\.dev0-0[0-9a-f]{12}-py3-none-any\.whl$`)
	var carried []string
	for range 2 {
		pack := buildForPublish(t, project)
		var found string
		for _, wheel := range pack.DependencyWheels {
			if strings.HasPrefix(wheel.Filename, "cozy_fixture_pinned-") {
				found = wheel.Filename
			}
		}
		if !named.MatchString(found) {
			t.Fatalf("pinned commit was not carried as its content-named wheel: %+v", pack.DependencyWheels)
		}
		for _, row := range pack.Registry {
			if row.Name == "cozy-fixture-pinned" {
				t.Fatalf("git source also published as a registry row: %+v", row)
			}
		}
		for _, dependency := range pack.Vendored {
			if dependency.Name == "cozy-fixture-pinned" {
				t.Fatal("a pinned third-party commit was nudged toward publication")
			}
		}
		carried = append(carried, found)
	}
	if carried[0] != carried[1] {
		t.Fatalf("one commit built two different wheels: %v", carried)
	}
}

func TestPublishRefusesAGitDependencyThatCanMove(t *testing.T) {
	for reference, want := range map[string]string{
		"git+https://github.com/huggingface/diffusers@main":                                       "project_dependency_git_unpinned",
		"git+https://github.com/huggingface/diffusers@v0.40.0":                                    "project_dependency_git_unpinned",
		"git+https://github.com/huggingface/diffusers@578c9b2c":                                   "project_dependency_git_unpinned",
		"git+https://github.com/huggingface/diffusers":                                            "project_dependency_git_unpinned",
		"git+ssh://git@github.com/huggingface/diffusers@578c9b2c6636ab2424a0e56186268b83623656b2": "project_dependency_git_source_unsupported",
		"https://example.com/diffusers-0.41.0.tar.gz":                                             "project_dependency_direct_url_unsupported",
	} {
		tree := fixtureTree(t, fixturePyproject("diffusers @ "+reference), minimalFixtureLock)
		pack, problem := packagepublish.PrepareFrom(tree)
		fatal(t, problem)
		problem = pack.BuildForPublish(t.Context(), packagepublish.Namespace{Hub: "http://127.0.0.1:1", Account: "proof"})
		pack.Close()
		if problem == nil || problem.Name != want {
			t.Fatalf("%s answered %v, want %s", reference, problem, want)
		}
	}
	// A git source only the lock names (a transitive one) has no wheel to ride as.
	transitive := `lock-version = "1.0"
created-by = "uv"

[[packages]]
name = "diffusers"
version = "0.41.0.dev0"
vcs = { type = "git", url = "https://github.com/huggingface/diffusers", commit-id = "578c9b2c6636ab2424a0e56186268b83623656b2" }
`
	if _, problem := packagepublish.RegistryRowsFromLock([]byte(transitive), nil, ""); problem == nil ||
		problem.Name != "registry_dependency_git_undeclared" {
		t.Fatalf("an undeclared git source answered %v", problem)
	}
	// The uv source table is held to the same rule: only `rev` with a full commit.
	tree := fixtureTree(t, fixturePyproject("diffusers>=0.41.0.dev0")+`
[tool.uv.sources]
diffusers = { git = "https://github.com/huggingface/diffusers", branch = "main" }
`, minimalFixtureLock)
	pack, problem := packagepublish.PrepareFrom(tree)
	fatal(t, problem)
	defer pack.Close()
	problem = pack.BuildForPublish(t.Context(), packagepublish.Namespace{Hub: "http://127.0.0.1:1", Account: "proof"})
	if problem == nil || problem.Name != "project_dependency_git_unpinned" ||
		!strings.Contains(problem.Remedy, "git+https://github.com/huggingface/diffusers@<40-hex commit>") {
		t.Fatalf("a branch source answered %v", problem)
	}
}
