package launch

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// hostRuntimeInstall is the one remedy for a host tool this Cozy cannot drive. No release
// number is named because this repository holds no fact about which cozy-runtime release
// vendors its minor; the minor is the fact, and the tool's own `version` prints it.
var hostRuntimeInstall = fmt.Sprintf(
	"install a cozy-runtime release that vendors %s minor %d or newer: "+
		"uv tool install --force 'cozy-runtime[media,model-execution]' — then retry",
	wirePackage(), pb.WireMinor)

func wirePackage() string { return string(pb.File_cozy_worker_v1_worker_proto.Package()) }

// hostRuntimeVerdicts memoizes a tool's admission by resolved path: a version verb is one
// interpreter start, and the host is asked once per daemon, not once per launch. Only an
// admission is kept — a refused tool is re-asked on the next launch, so reinstalling it
// takes effect without a daemon restart.
var hostRuntimeVerdicts = struct {
	sync.Mutex
	admitted map[string]bool
}{admitted: map[string]bool{}}

// HostRuntime is the package-independent local worker control process. Package code runs
// through the separately selected venv interpreter.
//
// A cozy-runtime on PATH is not yet a worker this daemon can drive. It must vendor this
// daemon's wire package at this daemon's minor or newer — the minor is additive, so a
// newer tool serves an older daemon and an older tool cannot. cl-086's live run: a 0.0.29
// tool (minor 16) under a minor-22 daemon launched, never came READY, and the request sat
// `queued` with nothing said. The tool's own `version` verb is the fact, asked here.
func HostRuntime(env []string) (string, *exit.Error) {
	path, err := exec.LookPath("cozy-runtime")
	if err != nil {
		return "", exit.Named(exit.Structural, "host_runtime_missing",
			"this host has no cozy-runtime command on PATH").
			WithRemedy("%s", hostRuntimeInstall)
	}
	hostRuntimeVerdicts.Lock()
	admitted := hostRuntimeVerdicts.admitted[path]
	hostRuntimeVerdicts.Unlock()
	if admitted {
		return path, nil
	}
	if e := admitHostRuntime(path, env); e != nil {
		return "", e
	}
	hostRuntimeVerdicts.Lock()
	hostRuntimeVerdicts.admitted[path] = true
	hostRuntimeVerdicts.Unlock()
	return path, nil
}

// admitHostRuntime runs `cozy-runtime --json version` once and reads the identity the tool
// prints for itself: its distribution release and `wire_protocol`, the vendored protobuf
// package joined to its additive minor as `<package>+minor.<n>`.
func admitHostRuntime(path string, env []string) *exit.Error {
	cmd := exec.Command(path, "--json", "version")
	cmd.Env = env
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if cmd.ProcessState == nil {
		return unreadableHostRuntime(path, "cannot run it: %s", err)
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		said := strings.TrimSpace(stderr.String())
		if said == "" {
			said = strings.TrimSpace(stdout.String())
		}
		return unreadableHostRuntime(path, "`cozy-runtime --json version` exited %d: %s",
			code, condense(said))
	}
	var answer struct {
		Distribution string `json:"distribution"`
		WireProtocol string `json:"wire_protocol"`
	}
	if err := json.Unmarshal([]byte(stdout.String()), &answer); err != nil {
		return unreadableHostRuntime(path, "`cozy-runtime --json version` answered a document "+
			"this host cannot read: %s", err)
	}
	spoken, minorText, ok := strings.Cut(answer.WireProtocol, "+minor.")
	minor, err := strconv.ParseUint(minorText, 10, 32)
	if !ok || err != nil {
		return unreadableHostRuntime(path, "`cozy-runtime --json version` names wire_protocol %q, "+
			"not <package>+minor.<n>", answer.WireProtocol)
	}
	if spoken != wirePackage() || uint32(minor) < pb.WireMinor {
		return exit.Named(exit.Structural, "host_runtime_wire_mismatch",
			"cozy-runtime %s is release %s and speaks %s; this Cozy needs %s+minor.%d or newer",
			path, answer.Distribution, answer.WireProtocol, wirePackage(), pb.WireMinor).
			WithRemedy("%s", hostRuntimeInstall)
	}
	return nil
}

func unreadableHostRuntime(path, format string, args ...any) *exit.Error {
	return exit.Named(exit.Structural, "host_runtime_unreadable",
		"cozy-runtime %s did not identify itself: %s", path, fmt.Sprintf(format, args...)).
		WithRemedy("%s", hostRuntimeInstall)
}
