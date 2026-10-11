package producttest

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
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
	if code, out := runCozy(t, root, "package", "install", caller); code != 0 {
		t.Fatalf("ordinary caller installation failed [%d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "run", "local/callee-caller-proof/main", "--await", "--json"); code != 0 || !strings.Contains(out, `"value":214`) {
		t.Fatalf("ordinary nested package job did not return 214 [%d]: %s", code, out)
	}
	// Nothing was staged on this computer for it: the machine was sent the install's own files.
	if _, err := os.Stat(filepath.Join(root, "local-packages")); !os.IsNotExist(err) {
		t.Fatalf("the run staged a capture under local-packages: %v", err)
	}
	db, err := sql.Open("sqlite", "file:"+filepath.Join(root, "machine/root/var/lib/cozy/rust-machine/execution/executions.sqlite3")+"?mode=ro")
	must(t, err)
	defer db.Close()
	// The run's three executed jobs.
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

// Cut condition 21: a machine runs its own Runtime/TensorFS pair. A package with a local
// dependency installs there from its own lock although the lock chose another Runtime: the
// machine's pair replaces the locked one wherever the package's declared bounds admit it
// (run 4811 pinned the pair exactly, and the machine's own failed `uv pip check`).
func TestACapturedPackageInstallsOnAMachineWhoseRuntimeDiffers(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host")
	}
	root, err := os.MkdirTemp(os.TempDir(), "czsdk-")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Log("SDK skew proof retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	machineWheels, _ := filepath.Glob(filepath.Join(machineTemplateDir(t), "root/opt/cozy/machine/wheels/cozy_runtime-*.whl"))
	if len(machineWheels) != 1 {
		t.Fatalf("the test machine holds no single Runtime wheel: %v", machineWheels)
	}
	machine := runtimeFixtureVersion(t, machineWheels[0])
	project := t.TempDir()
	for _, node := range []string{"callee", "caller"} {
		dir := filepath.Join(project, node)
		must(t, os.MkdirAll(dir, 0o700))
		module := "skew_" + node
		dependencies, extra := `"cozy-runtime>=0.18.89", "msgspec>=0.19,<1"`, ""
		body := "import msgspec\nfrom cozy_runtime.author import App, Context, invocable\napp = App()\nclass Squared(msgspec.Struct):\n    square: int\n" +
			"@invocable()\nasync def square(ctx: Context, *, value: int) -> Squared:\n    return Squared(value * value)\napp.job(square)\n"
		if node == "caller" {
			// The lock picks a published Runtime other than the machine's.
			dependencies += `, "skew-callee>=1.0.0"`
			extra = fmt.Sprintf("[tool.uv]\nconstraint-dependencies = [\"cozy-runtime!=%s\"]\n[tool.uv.sources]\nskew-callee = {path = \"../callee\"}\n", machine)
			body = "import msgspec\nfrom cozy_runtime.author import App, Context\nfrom skew_callee import square\napp = App()\n" +
				"class Survey(msgspec.Struct):\n    values: list[int]\nclass Surveyed(msgspec.Struct):\n    squares: list[int]\n" +
				"async def survey(ctx: Context, payload: Survey) -> Surveyed:\n    return Surveyed([(await square(value=v)).square for v in payload.values])\napp.job(survey)\n"
		}
		metadata := fmt.Sprintf("[project]\nname=%q\nversion=\"1.0.0\"\nrequires-python=\">=3.12,<3.13\"\ndependencies=[%s]\n%s[project.entry-points.\"cozy.application\"]\ndefault=%q\n[build-system]\nrequires=[\"hatchling\"]\nbuild-backend=\"hatchling.build\"\n[tool.hatch.build.targets.wheel]\nonly-include=[%q]\n",
			"skew-"+node, dependencies, extra, module+":app", module+".py")
		must(t, os.WriteFile(filepath.Join(dir, "pyproject.toml"), []byte(metadata), 0o600))
		must(t, os.WriteFile(filepath.Join(dir, "package.toml"), []byte(fmt.Sprintf("[application]\nobject=%q\n", module+":app")), 0o600))
		must(t, os.WriteFile(filepath.Join(dir, module+".py"), []byte(body), 0o600))
	}
	caller := filepath.Join(project, "caller")
	if out, err := exec.Command("uv", "lock", "--directory", caller, "--python", "3.12").CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v: %s", err, out)
	}
	lock, err := os.ReadFile(filepath.Join(caller, "uv.lock"))
	must(t, err)
	locked := regexp.MustCompile(`name = "cozy-runtime"\nversion = "([^"]+)"`).FindSubmatch(lock)
	if locked == nil || string(locked[1]) == machine {
		t.Fatalf("the lock must choose a Runtime other than the machine's %s: %q", machine, locked)
	}
	if code, out := runCozy(t, root, "package", "install", caller); code != 0 {
		t.Fatalf("package install [%d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "run", "local/skew-caller/survey", "values:=[3,4]", "--await", "--json"); code != 0 || !strings.Contains(out, `"squares":[9,16]`) {
		t.Fatalf("the captured pair locked to Runtime %s did not run on a machine with %s [%d]: %s", locked[1], machine, code, out)
	}
	if _, err := os.Stat(filepath.Join(root, "local-packages")); !os.IsNotExist(err) {
		t.Fatalf("the run staged a capture under local-packages: %v", err)
	}
}
