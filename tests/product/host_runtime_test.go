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

	"github.com/cozy-creator/cozy/internal/hostruntime"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// TestHostRuntimeWireFence is cl-086's follow-up as behaviour. The owner's host carried a
// 0.0.29 cozy-runtime (wire minor 16) under a minor-22 daemon: the worker launched, never
// came READY, and `cozy run` sat `queued` with nothing said. The daemon now asks the tool
// on PATH for its own identity before it exists, and refuses by name with the reinstall.
// Every arm is the real binary against a real root; the tool is a stand-in that answers
// only `version`, which is all the fence asks.
func TestHostRuntimeWireFence(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the stand-in runtimes are POSIX shell scripts")
	}
	install := "uv tool install --force 'cozy-runtime[media,model-execution]>=" + hostruntime.Floor + "'"

	// (a) An older minor cannot serve: `cozy up` refuses under the tool's own words, and
	// `cozy run` — which starts the same daemon — answers the same code instead of queuing.
	root, path := hostRuntimeRoot(t, "older", stubRuntime(t, "0.0.29", pb.WireMinor-6))
	code, out := runCozyPath(t, root, path, "up", "--json")
	refusal := refusalOf(t, out)
	if code == 0 || refusal.Code != "host_runtime_wire_mismatch" ||
		!strings.Contains(refusal.Message, fmt.Sprintf("release 0.0.29 and speaks cozy.worker.v1+minor.%d", pb.WireMinor-6)) ||
		!strings.Contains(refusal.Message, fmt.Sprintf("needs cozy.worker.v1+minor.%d or newer", pb.WireMinor)) ||
		!strings.Contains(refusal.Remedy, install) {
		t.Fatalf("an older host tool did not refuse `cozy up` by name [exit %d]\n%s", code, out)
	}
	if code, out := runCozyPath(t, root, path, "up"); code == 0 ||
		!strings.Contains(out, "Try: install cozy-runtime "+hostruntime.Floor+" or newer") {
		t.Fatalf("the human form of the refusal lost its remedy [exit %d]\n%s", code, out)
	}
	code, out = runCozyPath(t, root, path, "run", "fake/older/generate", "prompt=fox", "--json")
	if refusal := refusalOf(t, out); code == 0 || refusal.Code != "host_runtime_wire_mismatch" {
		t.Fatalf("`cozy run` under an older host tool did not refuse by name [exit %d]\n%s", code, out)
	}

	// (a′) The wire is right but the release predates static describe (cl-175): a tool
	// that would import a package to describe it is refused by name, with the floor.
	root, path = hostRuntimeRoot(t, "below-floor", stubRuntime(t, "0.4.0", pb.WireMinor))
	code, out = runCozyPath(t, root, path, "up", "--json")
	refusal = refusalOf(t, out)
	if code == 0 || refusal.Code != "host_runtime_below_floor" ||
		!strings.Contains(refusal.Message, "release 0.4.0; this Cozy needs "+hostruntime.Floor+" or newer") ||
		!strings.Contains(refusal.Remedy, install) {
		t.Fatalf("a host tool below the describe floor did not refuse `cozy up` by name [exit %d]\n%s", code, out)
	}

	// (b) A tool that cannot say what it is.
	root, path = hostRuntimeRoot(t, "mute", "#!/bin/sh\necho 'usage: cozy-runtime <verb>' >&2\nexit 2\n")
	code, out = runCozyPath(t, root, path, "up", "--json")
	if refusal := refusalOf(t, out); code == 0 || refusal.Code != "host_runtime_unreadable" ||
		!strings.Contains(refusal.Message, "exited 2: usage: cozy-runtime <verb>") ||
		!strings.Contains(refusal.Remedy, install) {
		t.Fatalf("a mute host tool did not refuse `cozy up` by name [exit %d]\n%s", code, out)
	}

	// (c) The wire is additive: a newer minor serves an older daemon.
	root, path = hostRuntimeRoot(t, "newer", stubRuntime(t, "9.9.9", pb.WireMinor+1))
	if code, out := runCozyPath(t, root, path, "up"); code != 0 {
		t.Fatalf("a newer host tool was refused [exit %d]\n%s", code, out)
	}
	if code, out := runCozyPath(t, root, path, "down"); code != 0 {
		t.Fatalf("down [exit %d]\n%s", code, out)
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
func stubRuntime(t *testing.T, release string, minor uint32) string {
	t.Helper()
	return "#!/bin/sh\nprintf '%s\\n' '" + stubIdentity(t, release, minor) + "'\n"
}

// stubIdentity is what `cozy-runtime --json version` prints: the document hostruntime.Path
// admits a tool by.
func stubIdentity(t *testing.T, release string, minor uint32) string {
	t.Helper()
	answer, err := json.Marshal(map[string]string{
		"distribution": release, "wire_protocol": fmt.Sprintf("cozy.worker.v1+minor.%d", minor),
	})
	must(t, err)
	return string(answer)
}

// hostRuntimeRoot is a fresh daemon root and a PATH holding only the stand-in tool (none
// when script is empty), so the daemon can find nothing else by that name.
func hostRuntimeRoot(t *testing.T, name, script string) (root, path string) {
	t.Helper()
	root = filepath.Join(os.TempDir(), "cozy-product-test", "host-runtime-"+name)
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
	cmd.Env = childEnv(t, root, "PATH="+path)
	data, _ := cmd.CombinedOutput()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
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
