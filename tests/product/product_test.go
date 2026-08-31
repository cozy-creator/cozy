package producttest

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/wheel"
)

const weightlessRef = "cozy/cozy-weightless-package"
const editableRuntimeFixtureSHA = "47a4f9e86aac8bd4fdb4f9d209ea07f0762b1adf"
const editableTensorFSFixtureSHA = "f645ee6a777cd32dca22144985099faa2276af4b"

func TestLiteralPayloadUsesOrdinaryScalarSyntax(t *testing.T) {
	entrypoint := &launch.Entrypoint{
		Name: "marco",
		Request: launch.Struct{Fields: []launch.Field{{
			Name: "message", Type: json.RawMessage(`{"literal":["marco"]}`), Wire: "required",
		}}},
	}
	for _, terms := range [][]string{{"message=marco"}, {"marco"}} {
		payload, problem := launch.ParsePayload(entrypoint, terms, "")
		if problem != nil || string(payload) != `{"message":"marco"}` {
			t.Fatalf("ParsePayload(%q) = %s, %v", terms, payload, problem)
		}
	}
	if _, problem := launch.ParsePayload(entrypoint, []string{"message=polo"}, ""); problem == nil || !strings.Contains(problem.Message, `must be one of: "marco"`) {
		t.Fatalf("wrong literal was not explained: %v", problem)
	}
}

func TestPackageHasOneActiveVersion(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.db"))
	fatal(t, problem)
	defer store.Close()
	install := func(id, version string, major int) records.PackageInstall {
		return records.PackageInstall{
			ID: id, Package: "cozy/example", Major: major, Version: version,
			SourceKind: "tensorhub", SourceRef: "cozy/example@" + version,
			SourceDigest: "sha256:" + strings.Repeat(id, 64/len(id)), Verified: true,
			Dir: filepath.Join(t.TempDir(), id),
		}
	}
	first := install("a", "1.0.0", 1)
	second := install("b", "2.0.0", 2)
	second.SelectionProfile = "torch2.13.0-cpu-cp314-linux-x86"
	second.PlacementSetDigest = "sha256:" + strings.Repeat("b", 64)
	if _, problem = store.Activate(first); problem != nil {
		t.Fatal(problem)
	}
	if superseded, problem := store.Activate(second); problem != nil || superseded != first.ID {
		t.Fatalf("replacement = %q, %v", superseded, problem)
	}
	pins, problem := store.Pins("cozy/example")
	if problem != nil || len(pins) != 1 || pins[0].InstallID != second.ID {
		t.Fatalf("active pins = %+v, %v", pins, problem)
	}
	_, active, problem := store.ActivePackage("cozy/example")
	if problem != nil || active == nil || active.SelectionProfile != second.SelectionProfile ||
		active.PlacementSetDigest != second.PlacementSetDigest {
		t.Fatalf("active selection = %+v, %v", active, problem)
	}
}

func TestModelManifestGrammar(t *testing.T) {
	root := t.TempDir()
	code, help := runCozy(t, root, "model", "publish", "--help")
	if code != 0 || !strings.Contains(help, "<manifest>") ||
		!strings.Contains(help, "--release") || !strings.Contains(help, "--lane") ||
		strings.Contains(strings.ToLower(help), "snapshot") {
		t.Fatalf("model publish did not expose only manifest/release/lane [exit %d]\n%s", code, help)
	}
	code, out := runCozy(t, root, "model", "publish", "acme/model",
		"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	if code != 2 || !strings.Contains(out, "--release") || !strings.Contains(out, "--lane") {
		t.Fatalf("model publish accepted missing release/lane [exit %d]\n%s", code, out)
	}
	code, help = runCozy(t, root, "model", "download", "--help")
	if code != 0 || strings.Contains(strings.ToLower(help), "snapshot") {
		t.Fatalf("model download retained snapshot vocabulary [exit %d]\n%s", code, help)
	}
}

func TestPackagePublishMetadataGrammar(t *testing.T) {
	root := t.TempDir()
	if code, help := runCozy(t, root, "run", "cozy/example/function", "--help"); code != 0 ||
		strings.Contains(help, "--version") || strings.Contains(help, "vN/function") ||
		!strings.Contains(help, "org/package[/function]") {
		t.Fatalf("run retained versioned target grammar [exit %d]\n%s", code, help)
	}
	if code, help := runCozy(t, root, "package", "publish", "--help"); code != 0 ||
		strings.Contains(help, "--release") || strings.Contains(help, "--dir") ||
		strings.Contains(help, "<package>") {
		t.Fatalf("package publish retained caller-authored identity [exit %d]\n%s", code, help)
	}
	if code, help := runCozy(t, root, "package", "install", "--help"); code != 0 ||
		!strings.Contains(help, "--version") || !strings.Contains(help, "explicit directory") ||
		strings.Contains(help, "--profile") || strings.Contains(help, "--major") ||
		strings.Contains(help, "--from") || strings.Contains(help, "--dir") ||
		strings.Contains(help, "--digest") || strings.Contains(help, "--allow-unsigned") ||
		strings.Contains(help, "--force") {
		t.Fatalf("package install exposed internal source/destination flags [exit %d]\n%s", code, help)
	}
	if code, out := runCozy(t, root, "package", "install", "cozy/example@1.2.3"); code != 2 ||
		!strings.Contains(out, "--version 1.2.3") {
		t.Fatalf("inline install version did not point to --version [exit %d]\n%s", code, out)
	}
	if code, help := runCozy(t, root, "package", "yank", "--help"); code != 0 ||
		!strings.Contains(help, "--version") || strings.Contains(help, "unyank") {
		t.Fatalf("package yank grammar changed [exit %d]\n%s", code, help)
	}
	if code, out := runCozy(t, root, "package", "yank", "cozy/example", "--version", "1.2"); code != 2 || !strings.Contains(out, "N.M.P") {
		t.Fatalf("package yank admitted a non-N.M.P release [exit %d]\n%s", code, out)
	}
	project := t.TempDir()
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name = "proof-package"
version = "1.0.0"
`), 0o644))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte(
		"[application]\nobject = \"proof_package:app\"\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(project, "uv.lock"), []byte("version = 1\n"), 0o644))
	code, out := runCozyDir(t, root, project, []string{"TENSORHUB_TOKEN=proof-token"},
		"package", "publish")
	if code != 1 || !strings.Contains(out, "must declare [tool.cozy] organization") {
		t.Fatalf("missing [tool.cozy] organization was not refused before build [exit %d]\n%s", code, out)
	}
}

func TestPackageYankUsesPermanentReleaseEndpoint(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete || r.URL.Path != "/v1/packages/proof/example/releases/1.2.3" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("X-Tensorhub-Reason") != "cozy package yank proof/example@1.2.3" {
			t.Errorf("package yank audit text = %q", r.Header.Get("X-Tensorhub-Reason"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w,
			`{"state":"yanked","release":"1.2.3","changed":true,"yanked_at":"2026-08-30T12:34:56Z"}`)
	}))
	defer server.Close()
	code, out := runCozyDir(t, t.TempDir(), ".", []string{
		"TENSORHUB_URL=" + server.URL, "TENSORHUB_TOKEN=proof-token",
	}, "package", "yank", "proof/example", "--version", "1.2.3")
	if code != 0 || !strings.Contains(out, "package: proof/example") ||
		!strings.Contains(out, "release: 1.2.3") || !strings.Contains(out, "status:  yanked") {
		t.Fatalf("package yank result changed [exit %d]\n%s", code, out)
	}
}

func TestPackageInstallExplicitDirectoryGrammar(t *testing.T) {
	root := t.TempDir()
	parent := t.TempDir()
	child := filepath.Join(parent, "child")
	must(t, os.Mkdir(child, 0o755))
	missing := filepath.Join(parent, "missing")
	for _, path := range []string{".", "..", "./missing", "../missing", missing} {
		code, out := runCozyDir(t, root, child, nil, "package", "install", path)
		if code != 1 || (!strings.Contains(out, "package source has no package.toml") &&
			!strings.Contains(out, "is not a directory")) || strings.Contains(out, "org/package") {
			t.Fatalf("explicit directory %q entered registry resolution [exit %d]\n%s", path, code, out)
		}
	}
}

func TestLocalPackageIdentityBindsSourceBytesWithoutBuilding(t *testing.T) {
	project := t.TempDir()
	writePublishProject(t, project, "local-identity", "1.0.0", nil, "", true)
	first, problem := packagepublish.PrepareFrom(project)
	fatal(t, problem)
	defer first.Close()
	firstDigest, files, bytes, problem := first.SourceIdentity()
	fatal(t, problem)
	replayDigest, replayFiles, replayBytes, problem := first.SourceIdentity()
	fatal(t, problem)
	if firstDigest != replayDigest || files != replayFiles || bytes != replayBytes {
		t.Fatalf("unchanged local build identity drifted: %s/%d/%d vs %s/%d/%d",
			firstDigest, files, bytes, replayDigest, replayFiles, replayBytes)
	}
	must(t, os.WriteFile(filepath.Join(project, "local_identity", "__init__.py"),
		[]byte("VALUE = 2\n"), 0o644))
	second, problem := packagepublish.PrepareFrom(project)
	fatal(t, problem)
	defer second.Close()
	secondDigest, _, _, problem := second.SourceIdentity()
	fatal(t, problem)
	if secondDigest == firstDigest {
		t.Fatalf("changed source retained local source identity %s", firstDigest)
	}
}

func TestRunAutoInstallsAMissingLocalPackage(t *testing.T) {
	root := t.TempDir()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/packages/proof/missing/download" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		_, _ = io.WriteString(w, `{"error":{"code":"package_download.no_compatible_release","message":"no non-yanked package release has a compatible active qualification","remedy":"publish a compatible release"}}`)
	}))
	defer server.Close()
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })

	code, out := runCozyDir(t, root, ".", []string{"TENSORHUB_URL=" + server.URL},
		"run", "proof/missing/generate")
	if code != 1 || !strings.Contains(out, "is not installed; installing it from Tensorhub") ||
		!strings.Contains(out, "no non-yanked package release") || strings.Contains(out, "is not installed on this host") {
		t.Fatalf("missing package did not enter automatic registry installation [exit %d]\n%s", code, out)
	}
}

func TestRentalCommandsSeparateInventoryFromCatalog(t *testing.T) {
	root := t.TempDir()
	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	problem = store.RecordRental(records.Rental{
		ID: "rnt-proof", MachineName: "studio", SKU: "h200",
		AcceleratorModel: "NVIDIA H200 SXM", HourlyRateUSDMicros: 6_000_000,
		State: "ready", Hub: "https://tensorhub.test",
	})
	fatal(t, problem)
	problem = store.RecordRental(records.Rental{
		ID: "rnt-failed", MachineName: "failed-gpu", SKU: "small",
		AcceleratorModel: "GPU", HourlyRateUSDMicros: 250_000,
		State: "failed", Hub: "https://tensorhub.test",
	})
	fatal(t, problem)
	store.Close()

	if code, out := runCozy(t, root, "rental"); code != 0 ||
		!strings.Contains(out, "studio") || !strings.Contains(out, "h200") ||
		!strings.Contains(out, "rentals: 2 remote machines running · $6.25/hour of $0.00/hour") {
		t.Fatalf("bare rental did not show current machines [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "rental", "list"); code != 2 ||
		!strings.Contains(out, "unexpected argument list") {
		t.Fatalf("retired rental list did not refuse [exit %d]\n%s", code, out)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/rental-skus" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[{"name":"h200","accelerator_model":"NVIDIA H200 SXM","compute_capability":"9.0","vram_gb":141,"price_usd_micros_per_hour":3990000}]`)
	}))
	defer server.Close()
	code, out := runCozyDir(t, root, ".", []string{"TENSORHUB_URL=" + server.URL}, "rental", "new")
	if code != 0 || !strings.Contains(out, "h200") || !strings.Contains(out, "NVIDIA H200 SXM") ||
		!strings.Contains(out, "sm_90") || !strings.Contains(out, "141 GB") || !strings.Contains(out, "$3.99/hr") {
		t.Fatalf("rental new did not show the SKU catalog [exit %d]\n%s", code, out)
	}
	if code, out := runCozyDir(t, root, ".", []string{"TENSORHUB_URL=" + server.URL},
		"rental", "new", "--name", "studio"); code != 2 ||
		!strings.Contains(out, "options require a GPU SKU") {
		t.Fatalf("catalog view silently accepted rental options [exit %d]\n%s", code, out)
	}
}

func TestRentalSpendConfigIsNestedAndExact(t *testing.T) {
	valid := t.TempDir()
	must(t, os.WriteFile(filepath.Join(valid, "config.yaml"), []byte(
		"port: 0\nrentals:\n  max_hourly_spend_usd: 4.125001\n"), 0o600))
	t.Cleanup(func() { _, _ = runCozy(t, valid, "down", "--all") })
	if code, out := runCozy(t, valid, "rental"); code != 0 ||
		!strings.Contains(out, "rentals: 0 remote machines running · $0.00/hour of $4.125001/hour") {
		t.Fatalf("nested rental ceiling was not exact [exit %d]\n%s", code, out)
	}

	overPrecise := t.TempDir()
	must(t, os.WriteFile(filepath.Join(overPrecise, "config.yaml"), []byte(
		"rentals:\n  max_hourly_spend_usd: 4.0000001\n"), 0o600))
	if code, out := runCozy(t, overPrecise, "rental"); code != 2 ||
		!strings.Contains(out, "at most six decimal places") {
		t.Fatalf("over-precise rental ceiling did not refuse [exit %d]\n%s", code, out)
	}

	deleted := t.TempDir()
	must(t, os.WriteFile(filepath.Join(deleted, "config.yaml"), []byte(
		"cloud:\n  max_hourly_spend_usd: 4\n"), 0o600))
	if code, out := runCozy(t, deleted, "rental"); code != 2 ||
		!strings.Contains(out, `unknown key "cloud"`) {
		t.Fatalf("deleted cloud config key did not refuse [exit %d]\n%s", code, out)
	}
}

func TestPackagePublishRefusesSilentlyOmittedPrivateFiles(t *testing.T) {
	project := t.TempDir()
	writePublishProject(t, project, "private-package", "1.0.0", nil, "", true)
	must(t, os.WriteFile(filepath.Join(project, ".env"), []byte("TOKEN=secret\n"), 0o600))
	pack, problem := packagepublish.PrepareFrom(project)
	if pack != nil {
		pack.Close()
	}
	if problem == nil || problem.Name != "package_source_file_refused" {
		t.Fatalf("private source file was silently skipped: %v", problem)
	}
}

func TestPackagePublishCommittedReplayStaysCompact(t *testing.T) {
	project := t.TempDir()
	writePublishProject(t, project, "replay-package", "1.0.0", nil, "", true)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/finalize"):
			_, _ = io.WriteString(w, `{"state":"committed","qualification_state":"qualified","release_digest":"sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"}`)
		case r.Method == http.MethodPost:
			_, _ = io.WriteString(w, `{"state":"committed","uploads":[]}`)
		default:
			http.Error(w, "unexpected method", http.StatusMethodNotAllowed)
		}
	}))
	defer server.Close()
	code, out := runCozyDir(t, t.TempDir(), project,
		[]string{"TENSORHUB_URL=" + server.URL, "TENSORHUB_TOKEN=proof-token"}, "package", "publish")
	if code != 0 || !strings.Contains(out, "status:  already published") ||
		!strings.Contains(out, "Checking proof/replay-package@1.0.0...") ||
		!strings.Contains(out, "Release already published; refreshing status...") ||
		!strings.Contains(out, "Committing exact package release...") ||
		strings.Contains(out, "qualification:") || strings.Contains(out, "changed:") {
		t.Fatalf("committed replay emitted verbose state [exit %d]\n%s", code, out)
	}
}

func TestPackagePublishBuildsBoundedLocalDependencyClosure(t *testing.T) {
	workspace := t.TempDir()
	projects := filepath.Join(workspace, "projects")
	must(t, os.MkdirAll(projects, 0o755))
	must(t, os.WriteFile(filepath.Join(workspace, "pyproject.toml"), []byte(
		"[tool.uv.workspace]\nmembers = [\"projects/*\"]\n"), 0o644))

	a := filepath.Join(projects, "local-a")
	b := filepath.Join(projects, "local-b")
	c := filepath.Join(projects, "local-c")
	platformCandidate := filepath.Join(projects, "platform-candidate")
	writePublishProject(t, a, "local-a", "1.0.0",
		[]string{"local-b>=2,<3", "local-b[images]>=2,<3", "cozy-runtime>=0.0.3"},
		"local-b = [{ workspace = true, marker = \"sys_platform == 'linux'\" }, { index = \"pypi\", marker = \"sys_platform != 'linux'\" }]\ncozy-runtime = { workspace = true, editable = true }\n", true)
	writePublishProject(t, b, "local-b", "2.1.0", nil,
		"local-c = { path = \"../local-c\", editable = true }\nabsent-local = { path = \"../absent-local\" }\n", false)
	appendProjectTOML(t, b, "\n[project.optional-dependencies]\nimages = [\"local-c==3.0.0\"]\nunused = [\"absent-local==1\"]\n")
	writePublishProject(t, c, "local-c", "3.0.0", nil, "", false)
	writePublishProject(t, platformCandidate, "cozy-runtime", "0.0.3", nil, "", false) //cozy:allow distribution fixture, not executable access

	pack, problem := preparePublishPackage(a)
	fatal(t, problem)
	defer pack.Close()
	wheels := map[string]string{}
	for _, dependency := range pack.DependencyWheels {
		identity, problem := wheel.InspectIdentity(dependency.Path)
		fatal(t, problem)
		wheels[identity.Distribution] = identity.Version
	}
	if len(pack.DependencyWheels) != 3 || wheels["local-b"] != "2.1.0" ||
		wheels["local-c"] != "3.0.0" || wheels["cozy-runtime"] != "0.0.3" { //cozy:allow distribution assertion, not executable access
		t.Fatalf("local dependency closure did not include the requested extra and base-name candidate: %+v", pack.DependencyWheels)
	}
	for _, dependency := range pack.DependencyWheels {
		if !strings.HasSuffix(dependency.Filename, ".whl") {
			t.Fatalf("dependency did not retain a wheel basename: %+v", dependency)
		}
		if info, err := os.Stat(dependency.Path); err != nil || !info.Mode().IsRegular() {
			t.Fatalf("dependency wheel is not a staged regular file: %+v err=%v", dependency, err)
		}
	}

	// Creator does not guess base ownership. It uploads Runtime as candidate
	// custody; Tensorhub's exact profile inventory must select the base copy and
	// omit this wheel from the eventual overlay.
	if wheels["cozy-runtime"] != "0.0.3" { //cozy:allow distribution assertion, not executable access
		t.Fatalf("local Runtime candidate was silently discarded: %+v", pack.DependencyWheels)
	}

	writePublishProject(t, a, "local-a", "1.0.0", []string{"local-b>=3"},
		"local-b = { workspace = true }\n", true)
	if incompatible, problem := preparePublishPackage(a); problem == nil || problem.Name != "local_dependency_version_incompatible" {
		if incompatible != nil {
			incompatible.Close()
		}
		t.Fatalf("local wheel outside the parent PEP 440 requirement did not refuse: %v", problem)
	}

	writePublishProject(t, a, "local-a", "1.0.0",
		[]string{"local-b[images]>=2,<3"},
		"local-b = { workspace = true }\n", true)
	writePublishProject(t, c, "local-c", "3.0.0", []string{"local-a==1.0.0"},
		"local-a = { path = \"../local-a\" }\n", false)
	if cycle, problem := preparePublishPackage(a); problem == nil || problem.Name != "local_dependency_cycle" {
		if cycle != nil {
			cycle.Close()
		}
		t.Fatalf("A->B->C->A did not refuse as a local dependency cycle: %v", problem)
	}

	conflictRoot := filepath.Join(t.TempDir(), "root")
	x1 := filepath.Join(filepath.Dir(conflictRoot), "x1")
	x2 := filepath.Join(filepath.Dir(conflictRoot), "x2")
	left := filepath.Join(filepath.Dir(conflictRoot), "left")
	right := filepath.Join(filepath.Dir(conflictRoot), "right")
	writePublishProject(t, conflictRoot, "conflict-root", "1.0.0",
		[]string{"left==1", "right==1"},
		"left = { path = \"../left\" }\nright = { path = \"../right\" }\n", true)
	writePublishProject(t, left, "left", "1", []string{"shared==1"},
		"shared = { path = \"../x1\" }\n", false)
	writePublishProject(t, right, "right", "1", []string{"shared==2"},
		"shared = { path = \"../x2\" }\n", false)
	writePublishProject(t, x1, "shared", "1", nil, "", false)
	writePublishProject(t, x2, "shared", "2", nil, "", false)
	if conflict, problem := preparePublishPackage(conflictRoot); problem == nil || problem.Name != "local_dependency_duplicate" {
		if conflict != nil {
			conflict.Close()
		}
		t.Fatalf("different sources for one normalized name did not refuse: %v", problem)
	}

	countRoot := filepath.Join(t.TempDir(), "root")
	writePublishProject(t, countRoot, "count-root", "1", []string{"count-01==1"},
		"count-01 = { path = \"../count-01\" }\n", true)
	countParent := filepath.Dir(countRoot)
	for i := 1; i <= packagepublish.MaxDependencyWheels+1; i++ {
		name := fmt.Sprintf("count-%02d", i)
		var dependencies []string
		var sources string
		if i <= packagepublish.MaxDependencyWheels {
			next := fmt.Sprintf("count-%02d", i+1)
			dependencies = []string{next + "==1"}
			sources = fmt.Sprintf("%s = { path = \"../%s\" }\n", next, next)
		}
		writePublishProject(t, filepath.Join(countParent, name), name, "1", dependencies, sources, false)
	}
	if counted, problem := preparePublishPackage(countRoot); problem == nil || problem.Name != "local_dependency_count_exceeded" {
		if counted != nil {
			counted.Close()
		}
		t.Fatalf("dependency closure above %d wheels was not refused: %v",
			packagepublish.MaxDependencyWheels, problem)
	}

	directRoot := filepath.Join(t.TempDir(), "direct")
	writePublishProject(t, directRoot, "direct-root", "1", []string{"foreign @ https://example.invalid/foreign.whl"}, "", true)
	if direct, problem := preparePublishPackage(directRoot); problem == nil || problem.Name != "project_dependency_direct_url_unsupported" {
		if direct != nil {
			direct.Close()
		}
		t.Fatalf("direct URL dependency did not refuse: %v", problem)
	}

	gitRoot := filepath.Join(t.TempDir(), "git")
	writePublishProject(t, gitRoot, "git-root", "1", []string{"foreign==1"},
		"foreign = { git = \"https://example.invalid/foreign.git\" }\n", true)
	if git, problem := preparePublishPackage(gitRoot); problem == nil || problem.Name != "project_dependency_source_unsupported" {
		if git != nil {
			git.Close()
		}
		t.Fatalf("VCS dependency source did not refuse: %v", problem)
	}
}

func preparePublishPackage(root string) (*packagepublish.Package, *exit.Error) {
	pack, problem := packagepublish.PrepareFrom(root)
	if problem != nil {
		return nil, problem
	}
	if problem := pack.Build(context.Background()); problem != nil {
		pack.Close()
		return pack, problem
	}
	return pack, nil
}

func writePublishProject(t *testing.T, root, name, version string, dependencies []string, sources string, publishable bool) {
	t.Helper()
	if dependencies == nil {
		dependencies = []string{}
	}
	must(t, os.MkdirAll(filepath.Join(root, strings.ReplaceAll(name, "-", "_")), 0o755))
	must(t, os.WriteFile(filepath.Join(root, strings.ReplaceAll(name, "-", "_"), "__init__.py"),
		[]byte("VALUE = 1\n"), 0o644))
	dependencyJSON, err := json.Marshal(dependencies)
	must(t, err)
	document := fmt.Sprintf(`[build-system]
requires = ["uv_build>=0.12.7,<0.13"]
build-backend = "uv_build"

[project]
name = %q
version = %q
dependencies = %s

[tool.uv.build-backend]
module-root = ""
`, name, version, dependencyJSON)
	if sources != "" {
		document += "\n[tool.uv.sources]\n" + sources
	}
	if publishable {
		document += "\n[tool.cozy]\norganization = \"proof\"\n"
		must(t, os.WriteFile(filepath.Join(root, "package.toml"), []byte(
			"[application]\nobject = \""+strings.ReplaceAll(name, "-", "_")+":app\"\n"), 0o644))
		must(t, os.WriteFile(filepath.Join(root, "uv.lock"), []byte("version = 1\n"), 0o644))
	}
	must(t, os.WriteFile(filepath.Join(root, "pyproject.toml"), []byte(document), 0o644))
}

func appendProjectTOML(t *testing.T, root, document string) {
	t.Helper()
	file, err := os.OpenFile(filepath.Join(root, "pyproject.toml"), os.O_APPEND|os.O_WRONLY, 0)
	must(t, err)
	_, err = file.WriteString(document)
	must(t, err)
	must(t, file.Close())
}

func TestDaemonWebLifecycle(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "daemon-web")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })

	code, help := runCozy(t, root)
	helpCode, explicitHelp := runCozy(t, root, "help")
	if helpCode != 0 || explicitHelp != help {
		t.Fatalf("cozy and cozy help rendered different root help [exit %d]\n%s", helpCode, explicitHelp)
	}
	for _, section := range []string{"Packages", "Models", "Authentication", "Runs", "Rentals", "Lifecycle"} {
		if code != 0 || strings.Count(help, "\n"+section+"\n") != 1 {
			t.Fatalf("bare cozy did not render one %q section [exit %d]\n%s", section, code, help)
		}
	}
	if strings.Contains(help, "\nResources\n") || strings.Contains(help, "\nWork\n") {
		t.Fatalf("bare cozy retained a combined command section\n%s", help)
	}
	if strings.Count(help, "run <org/package/function> [input]") != 1 || strings.Contains(help, "Primary command") {
		t.Fatalf("bare cozy did not state the primary run command in Runs\n%s", help)
	}
	for _, want := range []string{
		"Usage: cozy", "package install", "model download", "auth login", "run cancel",
		"rental new", "up", "down", "unload",
	} {
		if code != 0 || !strings.Contains(help, want) {
			t.Fatalf("bare cozy omitted %q [exit %d]\n%s", want, code, help)
		}
	}
	for _, retired := range []string{" exit ", "workflow", "video", "job submit"} {
		if strings.Contains(help, retired) {
			t.Fatalf("bare cozy retained %q\n%s", retired, help)
		}
	}
	if code, out := runCozy(t, root, "exit"); code != 2 || !strings.Contains(out, "unexpected argument exit") {
		t.Fatalf("retired exit command did not refuse [exit %d]\n%s", code, out)
	}

	env := childEnv(t, root)
	results := make(chan cozyResult, 2)
	for range 2 {
		go func() { results <- runCozyEnv(env, "up", "--json", "--full") }()
	}
	first, second := <-results, <-results
	if first.code != 0 || second.code != 0 {
		t.Fatalf("concurrent up did not converge: first=[%d] %s second=[%d] %s",
			first.code, first.output, second.code, second.output)
	}
	up := first.output
	var upDocument struct {
		URL     string `json:"url"`
		PID     int    `json:"pid"`
		Changed bool   `json:"changed"`
	}
	var secondDocument struct {
		URL     string `json:"url"`
		PID     int    `json:"pid"`
		Changed bool   `json:"changed"`
	}
	if err := json.Unmarshal([]byte(up), &upDocument); err != nil ||
		upDocument.URL == "" || upDocument.PID == 0 {
		t.Fatalf("up did not start one daemon: %v\n%s", err, up)
	}
	if err := json.Unmarshal([]byte(second.output), &secondDocument); err != nil ||
		upDocument.PID != secondDocument.PID || upDocument.URL != secondDocument.URL {
		t.Fatalf("concurrent up returned different daemon generations: %v\n%s\n%s",
			err, first.output, second.output)
	}
	if upDocument.Changed == secondDocument.Changed {
		t.Fatalf("concurrent up did not report one winner: first changed=%v second changed=%v\n%s\n%s",
			upDocument.Changed, secondDocument.Changed, first.output, second.output)
	}
	if strings.Contains(strings.ToLower(first.output+second.output), "lock") {
		t.Fatalf("up exposed its internal singleton mechanism\n%s\n%s", first.output, second.output)
	}
	url := upDocument.URL
	client := &http.Client{Timeout: 5 * time.Second}
	response, err := client.Get(url)
	if err != nil {
		t.Fatalf("up returned an unreachable web UI %q: %v", url, err)
	}
	page, readErr := io.ReadAll(response.Body)
	response.Body.Close()
	if readErr != nil || response.StatusCode != http.StatusOK ||
		!strings.Contains(string(page), "local generative workspace is running") {
		t.Fatalf("web stub is not ready at up return: status=%d err=%v\n%s",
			response.StatusCode, readErr, page)
	}
	if _, err := os.Stat(filepath.Join(root, "daemon.log")); !os.IsNotExist(err) {
		t.Fatalf("up created a persistent daemon log: %v", err)
	}
	if code, out := runCozy(t, root, "up", "--json", "--full"); code != 0 {
		t.Fatalf("repeated up failed [exit %d]\n%s", code, out)
	} else {
		var repeated struct {
			URL     string `json:"url"`
			PID     int    `json:"pid"`
			Changed bool   `json:"changed"`
		}
		if err := json.Unmarshal([]byte(out), &repeated); err != nil ||
			repeated.Changed || repeated.URL != url || repeated.PID != upDocument.PID {
			t.Fatalf("repeated up did not return the same healthy daemon with changed=false: %v\n%s",
				err, out)
		}
	}
	if code, out := runCozy(t, root, "down"); code != 0 ||
		!strings.Contains(out, "daemon:") || !strings.Contains(out, "stopped") {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "list", "--json"); code != 0 ||
		!strings.Contains(out, `"invocations":[]`) {
		t.Fatalf("stateful command did not auto-start the daemon [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "unload"); code != 0 || !strings.Contains(out, "No workers found.") {
		t.Fatalf("unload [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "up"); code != 0 || !strings.Contains(out, "changed: false") {
		t.Fatalf("unload stopped the daemon [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "down"); code != 0 ||
		!strings.Contains(out, "daemon:") || !strings.Contains(out, "stopped") {
		t.Fatalf("final down [exit %d]\n%s", code, out)
	}
}

func TestUpReportsLocalGPUCompatibility(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "up-gpu-compatibility")
	must(t, os.RemoveAll(root))
	toolDir := t.TempDir()
	nvidiaSMI := filepath.Join(toolDir, "nvidia-smi")
	must(t, os.WriteFile(nvidiaSMI, []byte(`#!/bin/sh
case "$1" in
  --query-gpu=*) printf '0,"NVIDIA, Test GPU",12288,24576,580.126.20,8.9\n' ;;
  *) printf '| NVIDIA-SMI 580.126.20 Driver Version: 580.126.20 CUDA Version: 13.0 |\n' ;;
esac
`), 0o755))
	env := childEnv(t, root, "PATH="+toolDir)
	t.Cleanup(func() { _ = runCozyEnv(env, "down", "--all") })

	result := runCozyEnv(env, "up", "--json", "--full")
	if result.code != 0 {
		t.Fatalf("up with NVIDIA GPU failed [exit %d]\n%s", result.code, result.output)
	}
	var document struct {
		GPUs       []string `json:"gpus"`
		GPUCount   int      `json:"gpu_count"`
		GPUDetails []struct {
			Model             string `json:"model"`
			VRAMFreeBytes     int64  `json:"vram_free_bytes"`
			VRAMTotalBytes    int64  `json:"vram_total_bytes"`
			DriverVersion     string `json:"driver_version"`
			DriverCUDAVersion string `json:"driver_cuda_version"`
			ComputeCapability string `json:"compute_capability"`
			SM                string `json:"sm"`
		} `json:"gpu_details"`
	}
	if err := json.Unmarshal([]byte(result.output), &document); err != nil ||
		document.GPUCount != 1 || len(document.GPUs) != 1 || len(document.GPUDetails) != 1 {
		t.Fatalf("up did not return one typed GPU: %v\n%s", err, result.output)
	}
	gpu := document.GPUDetails[0]
	if gpu.Model != "NVIDIA, Test GPU" || gpu.VRAMFreeBytes != 12<<30 ||
		gpu.VRAMTotalBytes != 24<<30 || gpu.DriverVersion != "580.126.20" ||
		gpu.DriverCUDAVersion != "13.0" || gpu.ComputeCapability != "8.9" || gpu.SM != "sm_89" {
		t.Fatalf("up GPU compatibility facts drifted: %+v", gpu)
	}
	for _, want := range []string{"NVIDIA, Test GPU", "12.0 / 24.0 GiB free", "driver 580.126.20", "driver CUDA 13.0", "sm_89"} {
		if !strings.Contains(document.GPUs[0], want) {
			t.Fatalf("up GPU summary omitted %q: %s", want, document.GPUs[0])
		}
	}
}

func TestRentalGPUCatalog(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/rental-skus" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[{"name":"h200","accelerator_model":"NVIDIA H200","compute_capability":"9.0","vram_gb":141,"price_usd_micros_per_hour":6000000},{"name":"rtx-4090","accelerator_model":"NVIDIA GeForce RTX 4090","compute_capability":"8.9","vram_gb":24,"price_usd_micros_per_hour":1250000}]`)
	}))
	defer server.Close()
	root := filepath.Join(os.TempDir(), "cozy-product-test", "rental-gpu-catalog")
	must(t, os.RemoveAll(root))
	env := childEnv(t, root, "TENSORHUB_URL="+server.URL)

	result := runCozyEnv(env, "rental", "new", "--json")
	if result.code != 0 {
		t.Fatalf("rental catalog failed [exit %d]\n%s", result.code, result.output)
	}
	var document struct {
		GPUs []map[string]string `json:"gpus"`
	}
	if err := json.Unmarshal([]byte(result.output), &document); err != nil || len(document.GPUs) != 2 {
		t.Fatalf("rental catalog was not a two-row GPU list: %v\n%s", err, result.output)
	}
	if document.GPUs[0]["name"] != "h200" || document.GPUs[0]["model"] != "NVIDIA H200" ||
		document.GPUs[0]["compute"] != "sm_90" || document.GPUs[0]["vram"] != "141 GB" || document.GPUs[0]["price"] != "$6/hr" ||
		document.GPUs[1]["price"] != "$1.25/hr" {
		t.Fatalf("rental catalog values drifted: %#v", document.GPUs)
	}
	if result := runCozyEnv(env, "rental", "new", "h200", "extra"); result.code != 2 || !strings.Contains(result.output, "unexpected argument") {
		t.Fatalf("generic rental accepted a second positional argument [exit %d]\n%s", result.code, result.output)
	}
}

func TestDefaultWebPortPreferenceAndFallback(t *testing.T) {
	held, err := net.Listen("tcp4", "127.0.0.1:8818") //cozy:allow product proof occupies the preferred loopback port to exercise fallback
	if err != nil {
		t.Skipf("localhost:8818 is already occupied outside this product test: %v", err)
	}

	fallbackRoot := filepath.Join(os.TempDir(), "cozy-product-test", "default-port-fallback")
	must(t, os.RemoveAll(fallbackRoot))
	code, out := runCozy(t, fallbackRoot, "up", "--json", "--full")
	if code != 0 {
		held.Close()
		t.Fatalf("up with occupied preferred port failed [exit %d]\n%s", code, out)
	}
	var fallback struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal([]byte(out), &fallback); err != nil || fallback.URL == "" ||
		strings.Contains(fallback.URL, ":8818") {
		held.Close()
		t.Fatalf("occupied 8818 did not select a fallback: %v\n%s", err, out)
	}
	if code, out := runCozy(t, fallbackRoot, "down"); code != 0 {
		held.Close()
		t.Fatalf("fallback daemon down [exit %d]\n%s", code, out)
	}
	must(t, held.Close())

	preferredRoot := filepath.Join(os.TempDir(), "cozy-product-test", "default-port-preferred")
	must(t, os.RemoveAll(preferredRoot))
	t.Cleanup(func() { _, _ = runCozy(t, preferredRoot, "down", "--all") })
	code, out = runCozy(t, preferredRoot, "up", "--json", "--full")
	if code != 0 {
		t.Fatalf("up on available preferred port failed [exit %d]\n%s", code, out)
	}
	var preferred struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal([]byte(out), &preferred); err != nil ||
		preferred.URL != "http://127.0.0.1:8818/" {
		t.Fatalf("available default did not bind 8818: %v\n%s", err, out)
	}
}

func TestDaemonStartupDiagnostic(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "daemon-startup-diagnostic")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })

	held, err := net.Listen("tcp4", "127.0.0.1:0") //cozy:allow product proof occupies one loopback port to exercise the real startup refusal
	must(t, err)
	port := held.Addr().(*net.TCPAddr).Port
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"),
		[]byte(fmt.Sprintf("port: %d\n", port)), 0o600))

	began := time.Now()
	code, out := runCozy(t, root, "up", "--json")
	if code != 1 {
		t.Fatalf("failed daemon startup exited %d, want operational 1\n%s", code, out)
	}
	if took := time.Since(began); took > 5*time.Second {
		t.Fatalf("failed daemon startup waited %s instead of relaying the exited child", took)
	}
	var document struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(out), &document); err != nil ||
		document.Error.Code != "daemon_startup_failed" ||
		!strings.Contains(document.Error.Message, fmt.Sprintf("127.0.0.1:%d", port)) ||
		!strings.Contains(document.Error.Message, "held by another process") {
		t.Fatalf("startup failure did not relay its child diagnostic: %v\n%s", err, out)
	}
	if strings.Contains(strings.ToLower(out), "lock") {
		t.Fatalf("startup failure exposed its internal singleton mechanism\n%s", out)
	}
	if len(out) > maxDaemonDiagnosticOutput {
		t.Fatalf("startup diagnostic is unbounded: %d bytes", len(out))
	}
	if _, err := os.Stat(filepath.Join(root, "daemon.log")); !os.IsNotExist(err) {
		t.Fatalf("failed startup created a persistent daemon log: %v", err)
	}

	must(t, held.Close())
	if code, out := runCozy(t, root, "up", "--json"); code != 0 {
		t.Fatalf("root did not recover after the failed child [exit %d]\n%s", code, out)
	}
}

const maxDaemonDiagnosticOutput = 18 << 10

func TestDevelopmentInstallRefreshesBeforeInvocation(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "editable-refresh")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		_ = os.RemoveAll(root)
	})

	project := weightlessProject(t)
	code, out := runCozy(t, root, "package", "install", project)
	if code != 0 {
		t.Fatalf("local directory package install [exit %d]\n%s", code, out)
	}
	if !strings.Contains(out, "source bytes are pinned locally") {
		t.Fatalf("local install omitted its unqualified identity note\n%s", out)
	}
	if code, out := runCozy(t, root, "package", "list", "--full", "--json"); code != 0 ||
		!strings.Contains(out, weightlessRef) || !strings.Contains(out, `"source":"local `) {
		t.Fatalf("package list omitted the install [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", weightlessRef); code != 0 ||
		!strings.Contains(out, "- tile") || !strings.Contains(out, "- refuse") {
		t.Fatalf("package-only run did not list functions [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", weightlessRef+"/v1.0.0/tile"); code != 2 ||
		!strings.Contains(out, weightlessRef+"/tile") || !strings.Contains(out, "installed release") {
		t.Fatalf("version-in-path remedy was not useful [exit %d]\n%s", code, out)
	}

	if code, out := runCozy(t, root, "run", weightlessRef+"/tile",
		"size=32", "seed=7", "--local", "--rental"); code != 2 ||
		!strings.Contains(out, "mutually exclusive") {
		t.Fatalf("local plus rental did not refuse [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", weightlessRef+"/tile",
		"size=32", "seed=7", "--machine", "local"); code != 2 ||
		!strings.Contains(out, "unknown flag") {
		t.Fatalf("deleted --machine did not refuse [exit %d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "run", weightlessRef+"/tile",
		"size=32", "seed=7", "--idempotency-key", "placement-proof", "--json")
	if code != 0 || !strings.Contains(out, `"revision":"first"`) {
		t.Fatalf("first editable invocation did not run source [exit %d]\n%s\n%s",
			code, out, productWorkerLogs(root))
	}
	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	requests, problem := store.RequestsOfKind("serving", "", 10)
	fatal(t, problem)
	store.Close()
	if len(requests) == 0 || requests[0].Rental || requests[0].Worker != "" {
		t.Fatalf("default local placement was not retained exactly: %+v", requests)
	}
	first := activePackageInstall(t, root)

	source := filepath.Join(project, "weightless.py")
	body, err := os.ReadFile(source)
	must(t, err)
	body = []byte(strings.Replace(string(body), `REVISION = "first"`, `REVISION = "second"`, 1))
	must(t, os.WriteFile(source, body, 0o644))
	code, out = runCozy(t, root, "run", weightlessRef+"/tile",
		"size=32", "seed=7", "--json")
	if code != 0 || !strings.Contains(out, `"revision":"second"`) {
		t.Fatalf("edited body was not live on the next invocation [exit %d]\n%s", code, out)
	}
	second := activePackageInstall(t, root)
	if second.ID == first.ID || second.SourceDigest == first.SourceDigest {
		t.Fatalf("body edit did not advance the editable generation: %#v -> %#v", first, second)
	}

	pyproject := filepath.Join(project, "pyproject.toml")
	metadata, err := os.ReadFile(pyproject)
	must(t, err)
	must(t, os.WriteFile(pyproject, append(metadata, []byte("\n# editable metadata refresh\n")...), 0o644))
	lock := filepath.Join(project, "uv.lock")
	lockBytes, err := os.ReadFile(lock)
	must(t, err)
	must(t, os.WriteFile(lock, append(lockBytes, []byte("\n# editable lock refresh\n")...), 0o644))
	code, out = runCozy(t, root, "run", weightlessRef+"/tile",
		"size=32", "seed=7", "--json")
	if code != 0 || !strings.Contains(out, `"revision":"second"`) {
		t.Fatalf("metadata/lock refresh did not remain runnable [exit %d]\n%s", code, out)
	}
	third := activePackageInstall(t, root)
	if third.ID == second.ID || third.LockDigest == second.LockDigest {
		t.Fatalf("metadata/lock edit did not atomically refresh the environment: %#v -> %#v", second, third)
	}

	goodMetadata, err := os.ReadFile(pyproject)
	must(t, err)
	must(t, os.WriteFile(pyproject, []byte("[project\n"), 0o644))
	code, out = runCozy(t, root, "run", weightlessRef+"/tile",
		"size=32", "seed=7", "--json")
	if code != 1 || !strings.Contains(out, `"code":"editable_refresh_failed"`) {
		t.Fatalf("failed edit did not return the typed refresh refusal [exit %d]\n%s", code, out)
	}
	failed := activePackageInstall(t, root)
	if failed.ID != third.ID || failed.SourceDigest != third.SourceDigest {
		t.Fatalf("failed refresh displaced the last good generation: %#v -> %#v", third, failed)
	}
	must(t, os.WriteFile(pyproject, goodMetadata, 0o644))
	code, out = runCozy(t, root, "run", weightlessRef+"/tile",
		"size=32", "seed=7", "--json")
	if code != 0 || !strings.Contains(out, `"revision":"second"`) {
		t.Fatalf("restored source did not reuse the last good generation [exit %d]\n%s", code, out)
	}
}

func TestModeledDevelopmentInstallRefreshesAndKeepsLastGoodSelection(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "modeled-editable-refresh")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		_ = os.RemoveAll(root)
	})

	project := modeledDevelopmentProject(t, root)
	code, out := runCozy(t, root, "package", "install", project, "--json", "--full")
	if code != 0 || !strings.Contains(out, `"package":"cozy/modeled-development-package"`) {
		t.Fatalf("modeled editable install failed [exit %d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "run", "cozy/modeled-development-package/render",
		"value=7", "--json")
	if code != 0 || !strings.Contains(out, `"value":7`) {
		t.Fatalf("modeled editable invocation failed [exit %d]\n%s\n%s",
			code, out, productWorkerLogs(root))
	}
	first := activeInstall(t, root, "cozy/modeled-development-package")

	source := filepath.Join(project, "src", "modeled_development_package", "__init__.py")
	body, err := os.ReadFile(source)
	must(t, err)
	body = []byte(strings.Replace(string(body), "return Result(payload.value)",
		"return Result(payload.value + 1)", 1))
	must(t, os.WriteFile(source, body, 0o644))
	code, out = runCozy(t, root, "run", "cozy/modeled-development-package/render",
		"value=7", "--json")
	if code != 0 || !strings.Contains(out, `"value":8`) {
		t.Fatalf("modeled source edit was not live on the next invocation [exit %d]\n%s\n%s",
			code, out, productWorkerLogs(root))
	}
	second := activeInstall(t, root, "cozy/modeled-development-package")
	if second.ID == first.ID || second.SourceDigest == first.SourceDigest {
		t.Fatalf("modeled body edit did not advance generation: %#v -> %#v", first, second)
	}

	binding := filepath.Join(project, "package.toml")
	goodBinding, err := os.ReadFile(binding)
	must(t, err)
	brokenBinding := strings.Replace(string(goodBinding), `release = "1.0.0"`,
		`release = "9.9.9"`, 1)
	must(t, os.WriteFile(binding, []byte(brokenBinding), 0o644))
	code, out = runCozy(t, root, "run", "cozy/modeled-development-package/render",
		"value=7", "--json")
	if code != 1 || !strings.Contains(out, `"code":"editable_refresh_failed"`) {
		t.Fatalf("absent modeled release did not return typed last-good refusal [exit %d]\n%s",
			code, out)
	}
	failed := activeInstall(t, root, "cozy/modeled-development-package")
	if failed.ID != second.ID || failed.SourceDigest != second.SourceDigest {
		t.Fatalf("failed modeled refresh displaced last good generation: %#v -> %#v", second, failed)
	}
	must(t, os.WriteFile(binding, goodBinding, 0o644))
	code, out = runCozy(t, root, "run", "cozy/modeled-development-package/render",
		"value=7", "--json")
	if code != 0 || !strings.Contains(out, `"value":8`) {
		t.Fatalf("restored modeled selection did not reuse last good generation [exit %d]\n%s",
			code, out)
	}
}

func productWorkerLogs(root string) string {
	paths, _ := filepath.Glob(filepath.Join(root, "workers", "*", "worker.log"))
	var out strings.Builder
	for _, path := range paths {
		data, _ := os.ReadFile(path)
		fmt.Fprintf(&out, "worker log %s:\n%s\n", filepath.Base(filepath.Dir(path)), data)
	}
	return out.String()
}

func activePackageInstall(t *testing.T, root string) records.PackageInstall {
	return activeInstall(t, root, weightlessRef)
}

func activeInstall(t *testing.T, root, packageRef string) records.PackageInstall {
	t.Helper()
	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	defer store.Close()
	_, install, problem := store.ActivePackage(packageRef)
	fatal(t, problem)
	if install == nil {
		t.Fatalf("editable package %s has no active install", packageRef)
	}
	return *install
}

func modeledDevelopmentProject(t *testing.T, root string) string {
	t.Helper()
	home, err := os.UserHomeDir()
	must(t, err)
	runtimeRepo := filepath.Join(home, "cozy_v2", "cozy-runtime") //cozy:allow peer source; exact test wheel
	tensorfsRepo := filepath.Join(home, "cozy_v2", "tensorfs")    //cozy:allow peer source; exact test wheel
	for _, repo := range []string{runtimeRepo, tensorfsRepo} {
		if _, err := os.Stat(filepath.Join(repo, "pyproject.toml")); err != nil {
			t.Skipf("no peer source at %s: %v", repo, err)
		}
	}
	dir := t.TempDir()
	project := filepath.Join(dir, "source")
	build := exec.Command("/usr/bin/nice", "-n", "19", "python3",
		"tests/product/testdata/build-modeled-editable.py",
		"--runtime-repo", runtimeRepo, "--runtime-sha", editableRuntimeFixtureSHA,
		"--tensorfs-repo", tensorfsRepo, "--tensorfs-sha", editableTensorFSFixtureSHA,
		"--out", dir, "--source-out", project, "--store", filepath.Join(root, "cas"))
	build.Dir = "../.."
	build.Env = childEnv(t, root)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building modeled editable fixture: %v\n%s", err, out)
	}
	return project
}

type cozyResult struct {
	code   int
	output string
}

func runCozyEnv(env []string, args ...string) cozyResult {
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = env
	data, _ := cmd.CombinedOutput()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	return cozyResult{code: code, output: string(data)}
}

func weightlessProject(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	must(t, err)
	repo := filepath.Join(home, "cozy_v2", "cozy-runtime") //cozy:allow peer source; the fixture builds the exact generation Runtime
	if _, err := os.Stat(filepath.Join(repo, "pyproject.toml")); err != nil {
		t.Skipf("no cozy-runtime peer at %s: %v", repo, err)
	}
	dir := t.TempDir()
	project := filepath.Join(dir, "source")
	build := exec.Command("/usr/bin/nice", "-n", "19", "python3",
		"tests/product/testdata/build-weightless.py", "--out", dir, "--source-out", project,
		"--runtime-sha", editableRuntimeFixtureSHA)
	build.Dir = "../.."
	build.Env = childEnv(t, repo, "RUNTIME_REPO="+repo)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building the weightless release: %v\n%s", err, out)
	}
	return project
}
