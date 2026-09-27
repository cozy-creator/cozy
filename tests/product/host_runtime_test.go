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
// came READY, and `cozy run` sat `queued` with nothing said. A host tool that cannot serve
// no longer stops the daemon (rentals and the Hub need none): `cozy up` names the upgrade,
// and only what needs the host tool refuses by name, with the reinstall. Every arm is the
// real binary against a real root; the tool is a stand-in that answers only `version`.
func TestHostRuntimeWireFence(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the stand-in runtimes are POSIX shell scripts")
	}
	install := fmt.Sprintf("supporting cozy.worker.v1+minor.%d or newer", hostruntime.WireFloor)
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

	// (a) An older minor cannot serve: the daemon starts, and `cozy up` names the upgrade.
	root, path := hostRuntimeRoot(t, "older", stubRuntime(t, "0.0.29", hostruntime.WireFloor-1))
	upNames(t, root, path, "host_runtime_wire_mismatch",
		fmt.Sprintf("release 0.0.29 and speaks cozy.worker.v1+minor.%d", hostruntime.WireFloor-1),
		fmt.Sprintf("needs cozy.worker.v1+minor.%d or newer", hostruntime.WireFloor))

	// (a′) The wire is right but the release predates static describe (cl-175).
	root, path = hostRuntimeRoot(t, "below-floor", stubRuntime(t, "0.4.0", pb.WireMinor))
	upNames(t, root, path, "host_runtime_below_floor", "release 0.4.0; this Cozy needs "+hostruntime.ToolFloor+" or newer")

	// (b) A tool that cannot say what it is.
	root, path = hostRuntimeRoot(t, "mute", "#!/bin/sh\necho 'usage: cozy-runtime <verb>' >&2\nexit 2\n")
	upNames(t, root, path, "host_runtime_unreadable", "exited 2: usage: cozy-runtime <verb>")

	// (c) The wire is additive: a newer minor serves an older daemon.
	root, path = hostRuntimeRoot(t, "newer", stubRuntime(t, "9.9.9", pb.WireMinor+1))
	if code, out := runCozyPath(t, root, path, "up"); code != 0 {
		t.Fatalf("a newer host tool was refused [exit %d]\n%s", code, out)
	}
	if code, out := runCozyPath(t, root, path, "down"); code != 0 {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}

	// Publication is optional: a machine with the no-effects execution floor
	// remains usable even when this client vendors publication's newer schema.
	root, path = hostRuntimeRoot(t, "execution-floor", stubRuntime(t, "9.9.9", hostruntime.WireFloor))
	if code, out := runCozyPath(t, root, path, "up"); code != 0 {
		t.Fatalf("the no-effects Runtime floor was refused [exit %d]\n%s", code, out)
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

// A wire-compatible pre-native SDK must not inspect current Context metadata.
func TestHostRuntimeNativeAPIFloor(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("stand-in runtimes are POSIX shell scripts")
	}
	for _, arm := range []struct {
		name, release, refusal string
		minor                  uint32
	}{
		{"old-api", "0.17.2", "host_runtime_below_floor", 54},
		{"old-wire", "0.18.0", "host_runtime_wire_mismatch", 53},
		{"no-python-ensure", "0.18.13", "host_runtime_below_floor", 58},
		{"old-caller-compiler", "0.18.20", "host_runtime_below_floor", pb.WireMinor},
		{"native", hostruntime.ToolFloor, "", pb.WireMinor},
		{"source-dev", hostruntime.ToolFloor + "+dev.h687ee141", "", pb.WireMinor},
	} {
		t.Run(arm.name, func(t *testing.T) {
			root, path := hostRuntimeRoot(t, "native-floor-"+arm.name, stubRuntime(t, arm.release, arm.minor))
			code, out := runCozyPath(t, root, path, "up")
			if code != 0 {
				t.Fatalf("the daemon did not start over the host tool: %d %s", code, out)
			}
			if arm.refusal != "" && !strings.Contains(out, "local runs refuse with "+arm.refusal) {
				t.Fatalf("a stale host SDK was not named: %s", out)
			}
			if arm.refusal == "" && strings.Contains(out, "local runs refuse") {
				t.Fatalf("a native API cohort was named stale: %s", out)
			}
			if code, out := runCozyPath(t, root, path, "down"); code != 0 {
				t.Fatalf("down [exit %d]\n%s", code, out)
			}
		})
	}
}

// stubRuntime answers `cozy-runtime --json version` the way the real tool does.
func stubRuntime(t *testing.T, release string, minor uint32) string {
	t.Helper()
	answer, err := json.Marshal(map[string]string{
		"distribution": release, "wire_protocol": fmt.Sprintf("cozy.worker.v1+minor.%d", minor),
	})
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
