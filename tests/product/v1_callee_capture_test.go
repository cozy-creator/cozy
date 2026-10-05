package producttest

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	capturedwheel "github.com/cozy-creator/cozy/internal/wheel"
)

// Component-boundary proof: real CLI/daemon and machine on an isolated fixture root,
// candidate SDK wheels, three ordinary packages. It never installs a personal machine.
func TestV1CapturedPackageCallsNestedJobsOnTheRealMachine(t *testing.T) {
	if *machineHostBinary == "" || *privateChildRuntimeWheel == "" || *tensorfsFixtureWheel == "" {
		t.Skip("requires actual machine and paired Runtime/TensorFS fixture wheels")
	}
	root, err := os.MkdirTemp(os.TempDir(), "czcallee-")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Log("callee proof retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	project := t.TempDir()
	version := runtimeFixtureVersion(t, *privateChildRuntimeWheel)
	commonSources := "cozy-runtime = {path = " + strconv.Quote(*privateChildRuntimeWheel) + "}\ntensorfs = {path = " + strconv.Quote(*tensorfsFixtureWheel) + "}\n"
	for _, node := range []string{"leaf", "nested", "caller"} {
		dir := filepath.Join(project, node)
		must(t, os.MkdirAll(dir, 0o700))
		module := "callee_" + node + "_proof"
		dependencies := fmt.Sprintf("\"cozy-runtime>=%s\", \"tensorfs\"", version)
		sources := commonSources
		if node == "nested" {
			dependencies += ", \"callee-leaf-proof>=1.0.0\""
			sources += "callee-leaf-proof = {path = \"../leaf\"}\n"
		}
		if node == "caller" {
			dependencies += ", \"callee-nested-proof>=1.0.0\""
			sources += "callee-nested-proof = {path = \"../nested\"}\n"
		}
		metadata := fmt.Sprintf("[project]\nname=%q\nversion=\"1.0.0\"\nrequires-python=\">=3.12,<3.13\"\ndependencies=[%s]\n[tool.uv.sources]\n%s[project.entry-points.\"cozy.application\"]\ndefault=%q\n[build-system]\nrequires=[\"hatchling\"]\nbuild-backend=\"hatchling.build\"\n[tool.hatch.build.targets.wheel]\nonly-include=[%q]\n", "callee-"+node+"-proof", dependencies, sources, module+":app", module+".py")
		must(t, os.WriteFile(filepath.Join(dir, "pyproject.toml"), []byte(metadata), 0o600))
		must(t, os.WriteFile(filepath.Join(dir, "package.toml"), []byte(fmt.Sprintf("[application]\nobject=%q\n", module+":app")), 0o600))
		body := "import msgspec\nfrom cozy_runtime.author import App, Context, invocable\nclass Result(msgspec.Struct, frozen=True):\n    value: int\n"
		switch node {
		case "leaf":
			body += "@invocable(memoize=True)\nasync def compute(ctx: Context, *, value: int) -> Result:\n    return Result(value + 100)\napp=App()\napp.job(compute)\n"
		case "nested":
			body += "from callee_leaf_proof import compute as leaf\n@invocable(memoize=True)\nasync def compute(ctx: Context, *, value: int) -> Result:\n    result=await leaf(value=value)\n    return Result(result.value * 2)\napp=App()\napp.job(compute)\n"
		case "caller":
			body += "from callee_nested_proof import compute\nclass Input(msgspec.Struct):\n    value: int = 7\napp=App()\n@app.job\nasync def main(payload: Input) -> Result:\n    result=await compute(value=payload.value)\n    return Result(result.value)\n"
		}
		must(t, os.WriteFile(filepath.Join(dir, module+".py"), []byte(body), 0o600))
	}
	caller := filepath.Join(project, "caller")
	if out, err := exec.Command("uv", "lock", "--directory", caller, "--python", "3.12").CombinedOutput(); err != nil {
		t.Fatalf("fixture's authored lock could not select its SDK pair: %v: %s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", caller, "--editable"); code != 0 {
		t.Fatalf("ordinary caller installation failed [%d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "run", "local/callee-caller-proof/main", "--await", "--json"); code != 0 || !strings.Contains(out, `"value":214`) {
		t.Fatalf("ordinary nested package job did not return 214 [%d]: %s", code, out)
	}
	// Recreate the older source-only retention format from this run's immutable
	// installed snapshot, keeping its existing captured graph and install ID.
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByReference("1")
	fatal(t, problem)
	installed, problem := store.Install(request.InstallID)
	fatal(t, problem)
	legacyLayout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	legacyInstall := *installed
	legacyInstall.ProjectDir = legacyInstall.SourceRef
	legacy, problem := localpackage.Stage(t.Context(), legacyLayout, legacyInstall)
	fatal(t, problem)
	layout, problem := home.Open(root)
	fatal(t, problem)
	retained := filepath.Join(layout.LocalPackages, installed.ID)
	archive, err := os.ReadFile(legacy.Files[0].Path)
	must(t, err)
	must(t, os.WriteFile(filepath.Join(retained, "source.tar"), archive, 0o600))
	document, err := json.Marshal(legacy)
	must(t, err)
	must(t, os.WriteFile(filepath.Join(retained, "installation.json"), document, 0o600))
	if code, out := runCozy(t, root, "run", "local/callee-caller-proof/main", "--await", "--json"); code != 0 || !strings.Contains(out, `"value":214`) {
		t.Fatalf("old source-only multi-package capture did not replay [%d]: %s", code, out)
	}
	db, err := sql.Open("sqlite", "file:"+filepath.Join(root, "machine/root/var/lib/cozy/rust-machine/execution/executions.sqlite3")+"?mode=ro")
	must(t, err)
	defer db.Close()
	// Inspect the first run's three executed jobs. The legacy replay may reuse its
	// previously completed child through a memo, which has no live generation/executor.
	rows, err := db.Query("SELECT invocation FROM executions WHERE id <= 3 ORDER BY id")
	must(t, err)
	defer rows.Close()
	var packages []string
	generation := ""
	for rows.Next() {
		var raw []byte
		must(t, rows.Scan(&raw))
		var invocation struct {
			Package, Generation, Parent string
			Job                         bool
		}
		must(t, json.Unmarshal(raw, &invocation))
		if invocation.Parent == "" {
			generation = invocation.Generation
		}
		if !invocation.Job || invocation.Generation != generation {
			t.Fatalf("callee job escaped the captured environment or kind: %+v", invocation)
		}
		generation = invocation.Generation
		packages = append(packages, invocation.Package)
	}
	must(t, rows.Err())
	// A local generation labels its root by distribution; dispatched child jobs
	// retain the logical package identities recorded by the captured callees map.
	if strings.Join(packages, ",") != "callee-caller-proof,local/callee-nested-proof,local/callee-leaf-proof" {
		t.Fatalf("children lost their independent package identities: %v", packages)
	}
}

// Real retained wheel files establish root selection and published callee provenance.
func TestV1CaptureSelectsItsRootByIDAndPreservesCalleeOrganization(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	stage := func(id, name string, callees map[string]string) localpackage.Installation {
		inst := records.PackageInstall{ID: id, Package: "local/" + name, Version: "1.0.0", Python: "3.12"}
		wheel := installWheel(t, name, "1.0.0", "")
		revision, problem := localpackage.StageWheels(layout, inst, fixturePackageInterface, []string{wheel}, nil, callees)
		fatal(t, problem)
		return revision
	}
	root := stage("root-id", "caller", nil)
	child := stage("callee-id", "worker", map[string]string{"worker": "other-org/worker"})
	revision, problem := localpackage.CapturedRoot(localpackage.ExecutionCapture{Installations: []localpackage.Installation{child, root}}, root.ID)
	fatal(t, problem)
	if revision.ID != root.ID || len(revision.Files) != 2 || revision.SourceArchive != "" || revision.Callees["worker"] != "other-org/worker" {
		t.Fatalf("root or original callee identity was lost: %+v", revision)
	}
	for _, file := range revision.Files {
		if filepath.Ext(file.Filename) != ".whl" {
			t.Fatalf("multi-package source was mixed with the retained wheels: %+v", file)
		}
	}
}

func TestV1CaptureRefusesConflictingVersionsBeforeSubmission(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	root, problem := localpackage.StageWheels(layout, records.PackageInstall{ID: "root", Package: "local/caller", Version: "1.0.0"},
		fixturePackageInterface, []string{installWheel(t, "caller", "1.0.0", ""), installWheel(t, "worker", "2.0.0", "")}, nil)
	fatal(t, problem)
	child, problem := localpackage.StageWheels(layout, records.PackageInstall{ID: "child", Package: "local/worker", Version: "1.0.0"},
		fixturePackageInterface, []string{installWheel(t, "worker", "1.0.0", "")}, nil)
	fatal(t, problem)
	if _, problem := localpackage.CapturedRoot(localpackage.ExecutionCapture{Installations: []localpackage.Installation{root, child}}, root.ID); problem == nil {
		t.Fatal("two versions of the same distribution were admitted in one environment")
	}
}

func TestV1CaptureKeepsTheCallersSelectedWheelAndRemovesItsRegistryDuplicate(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	worker := installWheel(t, "worker", "1.0.0", "worker:app")
	selected, independent := filepath.Join(t.TempDir(), filepath.Base(worker)), filepath.Join(t.TempDir(), filepath.Base(worker))
	fatal(t, capturedwheel.PinDependencies(worker, selected, []string{"library>=1"}))
	fatal(t, capturedwheel.PinDependencies(worker, independent, []string{"library==2"}))
	root, problem := localpackage.StageWheels(layout, records.PackageInstall{ID: "root", Package: "local/caller", Version: "1.0.0"},
		fixturePackageInterface, []string{installWheel(t, "caller", "1.0.0", ""), selected},
		[]byte("worker @ https://example.invalid/worker.whl --hash=sha256:retained\nlibrary @ https://example.invalid/library.whl --hash=sha256:retained\n"))
	fatal(t, problem)
	child, problem := localpackage.StageWheels(layout, records.PackageInstall{ID: "child", Package: "local/worker", Version: "1.0.0"},
		fixturePackageInterface, []string{independent}, nil)
	fatal(t, problem)
	merged, problem := localpackage.CapturedRoot(localpackage.ExecutionCapture{Installations: []localpackage.Installation{root, child}}, root.ID)
	fatal(t, problem)
	if strings.Contains(string(merged.DependencyRequirements), "worker @") || !strings.Contains(string(merged.DependencyRequirements), "library @") {
		t.Fatal("supplied wheel competed with a registry source, or an unrelated requirement disappeared")
	}
	for _, file := range merged.Files {
		if strings.HasPrefix(file.Filename, "worker-") {
			for _, selected := range root.Files {
				if selected.Filename == file.Filename && selected.Digest != file.Digest {
					t.Fatal("child's independent dependency metadata overrode the caller's selected wheel")
				}
			}
		}
	}
}
