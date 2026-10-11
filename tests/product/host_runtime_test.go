package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	pep440 "github.com/aquasecurity/go-pep440-version"

	"github.com/cozy-creator/cozy/internal/hostruntime"
)

// TestHostRuntimeAdmission: the host tool answers typed CLI verbs and speaks no machine protocol
// to this Cozy, so a Runtime of any protocol serves: the worker.v1 Runtime, the cohort Runtime
// of cozy.machine.v1, and one that names none. Only a release below ToolFloor does not. A host
// tool that cannot serve never stops the daemon (rentals and the Hub need none): `cozy up`
// names the upgrade, and only what needs the host tool refuses by name, with the reinstall.
// Every arm is the real binary against a real root; the tool is a stand-in that answers only
// `version`.
func TestHostRuntimeAdmission(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the stand-in runtimes are POSIX shell scripts")
	}
	install := "install cozy-runtime " + hostruntime.ToolFloor + " or newer"
	upNames := func(t *testing.T, root, path, code string, says ...string) {
		t.Helper()
		exit, out := runCozyPath(t, root, path, "up")
		if exit != 0 || !strings.Contains(out, "local runs refuse with "+code) || !strings.Contains(out, install) {
			t.Fatalf("`cozy up` did not start and name the upgrade [exit %d]\n%s", exit, out)
		}
		for _, want := range says {
			if !strings.Contains(out, want) {
				t.Fatalf("`cozy up` did not say %q\n%s", want, out)
			}
		}
		if exit, out := runCozyPath(t, root, path, "down"); exit != 0 {
			t.Fatalf("down [exit %d]\n%s", exit, out)
		}
	}

	// (a) The release predates static describe (cl-175).
	root, path := hostRuntimeRoot(t, "below-floor", stubRuntime(t, "0.4.0", runtimeWireProtocol))
	upNames(t, root, path, "host_runtime_below_floor", "release 0.4.0; this Cozy needs "+hostruntime.ToolFloor+" or newer")

	// (b) A tool that cannot say what it is.
	root, path = hostRuntimeRoot(t, "mute", "#!/bin/sh\necho 'usage: cozy-runtime <verb>' >&2\nexit 2\n")
	upNames(t, root, path, "host_runtime_unreadable", "exited 2: usage: cozy-runtime <verb>")

	// (c) The protocol a Runtime names for its machine is not this tool's contract.
	for name, protocol := range map[string]string{"worker": runtimeWireProtocol, "machine": "cozy.machine.v1",
		"older-minor": fmt.Sprintf("cozy.worker.v1+minor.%d", 0), "unnamed": ""} {
		root, path = hostRuntimeRoot(t, name, stubRuntime(t, hostruntime.ToolFloor, protocol))
		if code, out := runCozyPath(t, root, path, "up"); code != 0 || strings.Contains(out, "local runs refuse") {
			t.Fatalf("a host tool naming protocol %q was refused [exit %d]\n%s", protocol, code, out)
		}
		if code, out := runCozyPath(t, root, path, "down"); code != 0 {
			t.Fatalf("down [exit %d]\n%s", code, out)
		}
	}

	// (d) No tool at all starts the daemon: rentals and the hub need none, and a local
	// launch refuses `host_runtime_missing` for itself.
	root, path = hostRuntimeRoot(t, "missing", "")
	if code, out := runCozyPath(t, root, path, "up"); code != 0 {
		t.Fatalf("a host without cozy-runtime could not start its daemon [exit %d]\n%s", code, out)
	}
	if code, out := runCozyPath(t, root, path, "down"); code != 0 {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}

	// (e) This host's real tool, when it has one.
	if _, err := exec.LookPath("cozy-runtime"); err != nil { //cozy:allow the host's own tool, the fence's subject
		t.Log("this host has no cozy-runtime on PATH; the real-tool arm did not run")
		return
	}
	root, _ = hostRuntimeRoot(t, "real", "")
	if code, out := runCozy(t, root, "up"); code != 0 {
		t.Fatalf("this host's cozy-runtime was refused [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}
}

// stubRuntime answers `cozy-runtime --json version` the way the real tool does.
func stubRuntime(t *testing.T, release, protocol string) string {
	t.Helper()
	identity := map[string]string{"distribution": release}
	if protocol != "" {
		identity["wire_protocol"] = protocol
	}
	answer, err := json.Marshal(identity)
	must(t, err)
	return "#!/bin/sh\nprintf '%s\\n' '" + string(answer) + "'\n"
}

// hostRuntimeRoot is a fresh daemon root and a PATH holding only the stand-in tool (none
// when script is empty), so the daemon can find nothing else by that name.
func hostRuntimeRoot(t *testing.T, name, script string) (root, path string) {
	t.Helper()
	root = filepath.Join(scratchBase, "host-runtime-"+name)
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		_ = os.RemoveAll(root)
	})
	bin := t.TempDir()
	if script != "" {
		must(t, os.WriteFile(filepath.Join(bin, "cozy-runtime"), []byte(script), 0o700)) //cozy:allow a stand-in tool, not this host's
	}
	return root, bin
}

func runCozyPath(t *testing.T, root, path string, args ...string) (int, string) {
	t.Helper()
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	// Its own cache: a matching tool this Cozy installed for the user is not this test's.
	cmd.Env = childEnv(t, root, "PATH="+path, "XDG_CACHE_HOME="+filepath.Join(root, "cache"))
	data, _ := cmd.CombinedOutput()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	skipWithoutMachine(t, code, string(data))
	return code, string(data)
}

type refusal struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Remedy  string `json:"remedy"`
}

func refusalOf(t *testing.T, out string) refusal {
	t.Helper()
	var doc struct {
		Error refusal `json:"error"`
	}
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("not one refusal document: %v\n%s", err, out)
	}
	return doc.Error
}

// TestAnOldHostToolIsBroughtForward: a cozy-runtime on PATH older than the Runtime this Cozy is
// released with is never refused. This Cozy installs its own matching tool through uv and
// describes with it: a caller whose result is another local package's type describes (the old
// tool could not). Without uv it keeps the old tool and says what may be missing.
func TestAnOldHostToolIsBroughtForward(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the stand-in runtime is a POSIX shell script")
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv installs this Cozy's own tool")
	}
	cache, err := exec.Command(uv, "cache", "dir").Output()
	must(t, err)
	old := "#!/bin/sh\nif [ \"$2\" = version ]; then echo '{\"distribution\": \"0.18.84\"}'; exit 0; fi\n" +
		"echo 'this stand-in only answers version' >&2\nexit 2\n"
	stub := t.TempDir()
	must(t, os.WriteFile(filepath.Join(stub, "cozy-runtime"), []byte(old), 0o700)) //cozy:allow a stand-in tool, not this host's
	run := func(t *testing.T, root string, withUV bool, args ...string) (int, string) {
		t.Helper()
		path := stub + ":/usr/bin:/bin"
		if withUV {
			bin := t.TempDir()
			must(t, os.Symlink(uv, filepath.Join(bin, "uv")))
			path = stub + ":" + bin + ":/usr/bin:/bin"
		}
		cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
		cmd.Env = childEnv(t, root, "PATH="+path, "XDG_CACHE_HOME="+filepath.Join(root, "cache"),
			"UV_CACHE_DIR="+strings.TrimSpace(string(cache)))
		data, _ := cmd.CombinedOutput()
		return cmd.ProcessState.ExitCode(), string(data)
	}

	// Offline: the old tool stays, with the warning; nothing is refused.
	root, _ := hostRuntimeRoot(t, "old-offline", "")
	code, out := run(t, root, false, "up")
	if code != 0 || strings.Contains(out, "local runs refuse") ||
		!strings.Contains(out, "reading packages with cozy-runtime 0.18.84") ||
		!strings.Contains(out, "may be missing") {
		t.Fatalf("an old tool without uv was not kept with a warning [exit %d]\n%s", code, out)
	}
	_, _ = run(t, root, false, "down")

	// With uv: this Cozy's own tool, at its release, describes the two-package caller.
	root, _ = hostRuntimeRoot(t, "old-forward", "")
	project := filepath.Join(root, "project")
	child := filepath.Join(project, "child")
	must(t, os.MkdirAll(child, 0o755))
	write := func(path, body string) { must(t, os.WriteFile(path, []byte(body), 0o600)) }
	pyproject := func(name, module, dependencies, sources string) string {
		return fmt.Sprintf("[project]\nname=%q\nversion=\"0.1.0\"\nrequires-python=\">=3.12,<3.13\"\n"+
			"dependencies=[%s]\n%s[project.entry-points.\"cozy.application\"]\ndefault=\"%s:app\"\n"+
			"[build-system]\nrequires=[\"hatchling\"]\nbuild-backend=\"hatchling.build\"\n"+
			"[tool.hatch.build.targets.wheel]\nonly-include=[\"%s.py\"]\n", name, dependencies, sources, module, module)
	}
	write(filepath.Join(child, "pyproject.toml"), pyproject("labelled-child", "label_child", "", ""))
	write(filepath.Join(child, "package.toml"), "[application]\nobject=\"label_child:app\"\n")
	write(filepath.Join(child, "label_child.py"), "import msgspec\n"+
		"from cozy_runtime.author import App, Context, invocable\n"+
		"class Result(msgspec.Struct):\n    labels: list[str]\n"+
		"@invocable\nasync def inspect(ctx: Context, *, count: int) -> Result:\n    return Result([str(count)])\n"+
		"app=App()\napp.job(inspect)\n")
	write(filepath.Join(project, "pyproject.toml"), pyproject("labelled-parent", "label_parent",
		`"labelled-child>=0.1.0"`, "[tool.uv.sources]\nlabelled-child={path=\"./child\"}\n"))
	write(filepath.Join(project, "package.toml"), "[application]\nobject=\"label_parent:app\"\n")
	write(filepath.Join(project, "label_parent.py"), "from cozy_runtime.author import App, Context, invocable\n"+
		"from label_child import Result, inspect\n"+
		"@invocable\nasync def run(ctx: Context, *, count: int) -> Result:\n    return await inspect(count=count)\n"+
		"app=App()\napp.job(run)\n")
	lock := exec.Command(uv, "lock", "--no-progress", "--project", project)
	lock.Env = append(os.Environ(), "UV_CACHE_DIR="+strings.TrimSpace(string(cache)))
	if data, err := lock.CombinedOutput(); err != nil {
		t.Fatalf("lock fixture: %v\n%s", err, data)
	}
	code, out = run(t, root, true, "package", "install", project, "--json")
	if code != 0 {
		t.Fatalf("the caller was not described with this Cozy's own tool [exit %d]\n%s", code, out)
	}
	own := filepath.Join(root, "cache", "cozy", "host-runtime", "bin", "cozy-runtime")
	raw, err := exec.Command(own, "--json", "version").Output()
	var version struct{ Distribution string }
	if err != nil || json.Unmarshal(raw, &version) != nil {
		t.Fatalf("this Cozy's own tool is not installed at %s: %v %s", own, err, raw)
	}
	if found, err := pep440.Parse(version.Distribution); err != nil || found.LessThan(pep440.MustParse(hostruntime.ToolRelease)) {
		t.Fatalf("this Cozy's own tool is release %q, older than %s", version.Distribution, hostruntime.ToolRelease)
	}
	interfaces, _ := filepath.Glob(filepath.Join(root, "installs", "*", "documents", "package-interface.json"))
	described := ""
	for _, path := range interfaces {
		raw, _ := os.ReadFile(path)
		described += string(raw)
	}
	if !strings.Contains(described, `"labels"`) {
		t.Fatalf("the installed interface does not carry the callee's Result: %v\n%s", interfaces, out)
	}
	_, _ = run(t, root, true, "down")
}
