package producttest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/wheel"
)

func TestPublishedPackageDoesNotFeedRuntimeItsOwnWheel(t *testing.T) {
	homeDir, err := os.UserHomeDir()
	must(t, err)
	runtimeRepo := filepath.Join(homeDir, "cozy_v2", "cozy-runtime") //cozy:allow peer source; exact current Runtime wheel
	if _, err := os.Stat(filepath.Join(runtimeRepo, "pyproject.toml")); err != nil {
		t.Skipf("no cozy-runtime peer at %s: %v", runtimeRepo, err)
	}
	fixtureRoot := t.TempDir()
	project := filepath.Join(fixtureRoot, "source")
	build := exec.Command("python3", "tests/product/testdata/build-weightless.py",
		"--runtime-repo", runtimeRepo, "--runtime-sha", "HEAD", "--out", fixtureRoot,
		"--source-out", project, "--version", "1.0.4")
	build.Dir = "../.."
	build.Env = childEnv(t, runtimeRepo, "RUNTIME_REPO="+runtimeRepo)
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build current Runtime package fixture: %v\n%s", err, output)
	}
	must(t, os.Mkdir(filepath.Join(project, "weightless"), 0o755))
	must(t, os.Rename(filepath.Join(project, "weightless.py"),
		filepath.Join(project, "weightless", "__init__.py")))
	appendProjectTOML(t, project, `
[build-system]
requires = ["uv_build>=0.12.7,<0.13"]
build-backend = "uv_build"

[project.entry-points."cozy.application"]
default = "weightless:app"

[tool.uv.build-backend]
module-name = "weightless"
module-root = ""
`)
	lockPublishProject(t, project)
	pack, problem := packagepublish.PrepareFrom(project)
	fatal(t, problem)
	fatal(t, pack.Build(context.Background()))
	defer pack.Close()

	exact := func(path string) hub.ExactDocument {
		raw, err := os.ReadFile(path)
		must(t, err)
		digest := sha256.Sum256(raw)
		return hub.ExactDocument{CanonicalBytes: raw,
			Digest: "sha256:" + hex.EncodeToString(digest[:]), Length: int64(len(raw))}
	}
	type servedWheel struct {
		body []byte
	}
	served := map[string]servedWheel{}
	addWheel := func(path string, kind string) hub.PackageInstallDownload {
		body, err := os.ReadFile(path)
		must(t, err)
		fact, inspectProblem := wheel.InspectIdentity(path)
		fatal(t, inspectProblem)
		digest := sha256.Sum256(body)
		served[fact.Filename] = servedWheel{body: body}
		return hub.PackageInstallDownload{
			Digest: "sha256:" + hex.EncodeToString(digest[:]), Distribution: fact.Distribution,
			Kind: kind, Length: fact.Length, Path: fact.Filename, Tags: []string{"py3-none-any"},
			Version: fact.Version,
		}
	}
	downloads := []hub.PackageInstallDownload{addWheel(pack.Wheel, "project_wheel")}
	for _, dependency := range pack.DependencyWheels {
		downloads = append(downloads, addWheel(dependency.Path, "dependency_wheel"))
	}
	if len(downloads) < 2 || downloads[1].Distribution != "cozy-runtime" { //cozy:allow distribution assertion, not executable access
		t.Fatalf("fixture did not carry the exact Runtime dependency: %+v", downloads)
	}

	plan := hub.PackageDownloadPlan{
		Downloads: downloads, PackageConfig: exact(filepath.Join(project, "package.toml")),
		PackageDescriptor: exact(pack.Descriptor), Pyproject: exact(filepath.Join(project, "pyproject.toml")),
		Release: pack.Release, ReleaseDigest: "sha256:" + strings.Repeat("a", 64),
		UVLock: exact(filepath.Join(project, "uv.lock")),
	}
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path ==
			"/v1/packages/proof/cozy-weightless-package/download":
			for index := range plan.Downloads {
				plan.Downloads[index].URL = server.URL + "/files/" + plan.Downloads[index].Path
			}
			_ = json.NewEncoder(w).Encode(plan)
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/files/"):
			item, ok := served[strings.TrimPrefix(r.URL.Path, "/files/")]
			if !ok {
				http.NotFound(w, r)
				return
			}
			_, _ = w.Write(item.body)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	root := t.TempDir()
	code, output := runCozyDir(t, root, "", []string{"TENSORHUB_URL=" + server.URL},
		"package", "install", "proof/cozy-weightless-package", "--version", pack.Release,
		"--no-model-download", "--json")
	if code != 0 || !strings.Contains(output, `"package":"proof/cozy-weightless-package"`) {
		t.Fatalf("published package install [exit %d]\n%s", code, output)
	}
}

func TestPublishedPackageInstallsSourceCarriedTensorFSLocally(t *testing.T) {
	fixtureRoot := t.TempDir()
	runtimeSource := filepath.Join(fixtureRoot, "runtime")
	writeRuntimeFixture(t, runtimeSource)
	mutateFixtureFile(t, filepath.Join(runtimeSource, "cozy_runtime", "__init__.py"),
		`"entrypoints": [{"name": "proof",`,
		`"entrypoints": [{"models": [{"class": "proof", "component_use": {}, "path": "proof.models.model", "stamps": {}}], "name": "proof",`)
	appendProjectTOML(t, runtimeSource,
		"\n[project.optional-dependencies]\nmodel-execution = [\"tensorfs==0.0.6\"]\n")
	tensorFSSource := filepath.Join(fixtureRoot, "tensorfs")
	writePublishProject(t, tensorFSSource, "tensorfs", "0.0.6", nil, "", false)

	project := filepath.Join(fixtureRoot, "source")
	vendor := filepath.Join(project, "vendor")
	must(t, os.MkdirAll(vendor, 0o755))
	tensorFSWheel, problem := wheel.Build(wheel.Request{Context: context.Background(),
		Tree: tensorFSSource, OutDir: vendor})
	fatal(t, problem)
	writePublishProject(t, project, "custody-package", "1.0.0",
		[]string{"cozy-runtime[model-execution]==0.0.11", "tensorfs==0.0.6"},
		"cozy-runtime = { path = "+quote(runtimeSource)+", editable = true }\n"+
			"tensorfs = { path = "+quote("vendor/"+filepath.Base(tensorFSWheel.Path))+" }\n", true)
	lockPublishProject(t, project)
	pack, problem := packagepublish.PrepareFrom(project)
	fatal(t, problem)
	fatal(t, pack.Build(context.Background()))
	defer pack.Close()
	if len(pack.DependencyWheels) != 1 {
		t.Fatalf("TensorFS entered rental package dependencies: %+v", pack.DependencyWheels)
	}
	runtimeIdentity, problem := wheel.InspectIdentity(pack.DependencyWheels[0].Path)
	fatal(t, problem)
	if runtimeIdentity.Distribution != "cozy-runtime" { //cozy:allow distribution assertion, not executable access
		t.Fatalf("published package dependencies = %+v, want Runtime only", runtimeIdentity)
	}

	exact := func(path string) hub.ExactDocument {
		raw, err := os.ReadFile(path)
		must(t, err)
		sum := sha256.Sum256(raw)
		return hub.ExactDocument{CanonicalBytes: raw,
			Digest: "sha256:" + hex.EncodeToString(sum[:]), Length: int64(len(raw))}
	}
	type servedWheel struct{ body []byte }
	served := map[string]servedWheel{}
	addWheel := func(path, kind string) hub.PackageInstallDownload {
		body, err := os.ReadFile(path)
		must(t, err)
		identity, inspectProblem := wheel.InspectIdentity(path)
		fatal(t, inspectProblem)
		sum := sha256.Sum256(body)
		served[identity.Filename] = servedWheel{body: body}
		return hub.PackageInstallDownload{Digest: "sha256:" + hex.EncodeToString(sum[:]),
			Distribution: identity.Distribution, Kind: kind, Length: identity.Length,
			Path: identity.Filename, Tags: []string{"py3-none-any"}, Version: identity.Version}
	}
	downloads := []hub.PackageInstallDownload{addWheel(pack.Wheel, "project_wheel")}
	for _, dependency := range pack.DependencyWheels {
		downloads = append(downloads, addWheel(dependency.Path, "dependency_wheel"))
	}
	downloads = append(downloads, addWheel(tensorFSWheel.Path, "local_materialization_wheel"))
	plan := hub.PackageDownloadPlan{Downloads: downloads,
		PackageConfig:     exact(filepath.Join(project, "package.toml")),
		PackageDescriptor: exact(pack.Descriptor), Pyproject: exact(filepath.Join(project, "pyproject.toml")),
		Release: pack.Release, ReleaseDigest: "sha256:" + strings.Repeat("b", 64),
		UVLock: exact(filepath.Join(project, "uv.lock"))}
	// The published install must not reopen the author's vendor path.
	must(t, os.Remove(tensorFSWheel.Path))

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/packages/proof/custody-package/download":
			for index := range plan.Downloads {
				plan.Downloads[index].URL = server.URL + "/files/" + plan.Downloads[index].Path
			}
			_ = json.NewEncoder(w).Encode(plan)
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/files/"):
			item, ok := served[strings.TrimPrefix(r.URL.Path, "/files/")]
			if !ok {
				http.NotFound(w, r)
				return
			}
			_, _ = w.Write(item.body)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	root := t.TempDir()
	code, output := runCozyDir(t, root, "", []string{"TENSORHUB_URL=" + server.URL},
		"package", "install", "proof/custody-package", "--version", pack.Release,
		"--no-model-download", "--json")
	if code != 0 || !strings.Contains(output, `"package":"proof/custody-package"`) {
		t.Fatalf("source-carried TensorFS install [exit %d]\n%s", code, output)
	}
	installed := activeInstall(t, root, "proof/custody-package")
	probe := exec.Command(filepath.Join(installed.Dir, "venv", "bin", "python"),
		"-c", "import tensorfs; print(tensorfs.VALUE)")
	if out, err := probe.CombinedOutput(); err != nil || strings.TrimSpace(string(out)) != "1" {
		t.Fatalf("installed local TensorFS custody is absent: %v\n%s", err, out)
	}
}

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
			[]install.PublishedWheel{{Distribution: "torch", Path: torchWheel.Path}}, nil)
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
		[]install.PublishedWheel{{Distribution: "torch", Path: newTorchWheel}}, nil)
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
