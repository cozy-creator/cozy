// Package hostruntime is THIS host's cozy-runtime tool: the package-independent worker control
// process and the ONE reader of package metadata on this host. Package code is untrusted and a
// reading of it never imports it (cl-175); it runs only through an install's own venv, as a
// worker. launch, install and packagepublish all ask here, and nothing here imports them.
package hostruntime

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"sync"

	pep440 "github.com/aquasecurity/go-pep440-version"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// ToolFloor is the first released Runtime with everything this Cozy drives: a lower bound.
// Distribution is the Runtime's Python distribution and executable name.
const Distribution = "cozy-runtime"

const ToolFloor = "0.18.67"

// PackageFloor is the first package SDK whose executor states its protocol revision and native
// interfaces in its hello.
const PackageFloor = "0.18.51"

var floor = pep440.MustParse(ToolFloor)

// InstallCommand installs this host's tools in one uv environment: cozy-runtime and the
// TensorFS `tfs` its dependencies resolve. install.sh and scripts/install.ps1 run it.
var InstallCommand = fmt.Sprintf(
	"uv tool install --force --python 3.12 --with-executables-from tensorfs 'cozy-runtime[media,model-execution]>=%s'",
	ToolFloor)

// hostRuntimeInstall is the one remedy for a host tool this Cozy cannot drive.
var hostRuntimeInstall = fmt.Sprintf("install cozy-runtime %s or newer: %s — then retry",
	ToolFloor, InstallCommand)

func wirePackage() string { return string(pb.File_cozy_worker_v1_worker_proto.Package()) }

// hostRuntimeVerdicts memoizes a tool's admission by resolved path: a version verb is one
// interpreter start, and the host is asked once per daemon, not once per launch. Only an
// admission is kept — a refused tool is re-asked on the next launch, so reinstalling it
// takes effect without a daemon restart.
var hostRuntimeVerdicts = struct {
	sync.Mutex
	admitted map[string]bool
}{admitted: map[string]bool{}}

// Path is the admitted tool: a cozy-runtime on PATH of this daemon's wire package, at
// ToolFloor or newer. The tool answers typed CLI verbs (describe, image preparation, builtin
// metadata) and never speaks the worker wire, so its wire minor is not compared: an older or
// newer tool serves this daemon. The tool's own `version` verb is the fact, asked here.
func Path(env []string) (string, *exit.Error) {
	path, err := exec.LookPath(Distribution)
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
	_, err = strconv.ParseUint(minorText, 10, 32)
	if !ok || err != nil {
		return unreadableHostRuntime(path, "`cozy-runtime --json version` names wire_protocol %q, "+
			"not <package>+minor.<n>", answer.WireProtocol)
	}
	if spoken != wirePackage() {
		return exit.Named(exit.Structural, "host_runtime_wire_mismatch",
			"cozy-runtime %s is release %s and speaks %s; this Cozy speaks %s",
			path, answer.Distribution, answer.WireProtocol, wirePackage()).
			WithRemedy("%s", hostRuntimeInstall)
	}
	release, err := pep440.Parse(answer.Distribution)
	if err != nil {
		return unreadableHostRuntime(path, "`cozy-runtime --json version` names distribution %q, "+
			"not a PEP 440 release", answer.Distribution)
	}
	if release.LessThan(floor) {
		return exit.Named(exit.Structural, "host_runtime_below_floor",
			"cozy-runtime %s is release %s; this Cozy needs %s or newer for native model ingestion and managed result collection",
			path, answer.Distribution, ToolFloor).
			WithRemedy("%s", hostRuntimeInstall)
	}
	return nil
}

func unreadableHostRuntime(path, format string, args ...any) *exit.Error {
	return exit.Named(exit.Structural, "host_runtime_unreadable",
		"cozy-runtime %s did not identify itself: %s", path, fmt.Sprintf(format, args...)).
		WithRemedy("%s", hostRuntimeInstall)
}

// Describe is the one reading of a package surface this host performs — at publish pre-flight,
// at install, and for a job's descriptor — by the tool at or above ToolFloor:
// `cozy-runtime describe` parses the source under dir and imports nothing. It returns the
// canonical PackageInterface bytes. `where` names the source when it is not under dir — an
// installed release's module lives in its venv, so the venv's interpreter is named for the
// reading to resolve the file; the interpreter is not run against the package.
func Describe(ctx context.Context, env []string, dir string, where ...string) ([]byte, *exit.Error) {
	runtimeBin, problem := Path(env)
	if problem != nil {
		return nil, problem
	}
	args := append([]string{"--json", "--dir", dir, "describe"}, where...)
	cmd := exec.CommandContext(ctx, runtimeBin, args...)
	cmd.Env = env
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if cmd.ProcessState == nil {
		return nil, exit.Internalf("cannot run %s: %s", runtimeBin, err)
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		return nil, RuntimeExit(code, "describe", "runtime_query_failed", stdout.String(), stderr.String())
	}
	return bytes.TrimSuffix([]byte(stdout.String()), []byte("\n")), nil
}

func condense(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 400 {
		return s[:400] + "…"
	}
	return s
}

// RuntimeExit is the ONE reading of a cozy-runtime child that exited non-zero. The exit
// matrix is SHARED (cozy-runtime-cli.md), so a runtime exit is already a cozy exit and the
// `--json` refusal it wrote is already the answer: code, name, message and remedy pass
// through as themselves. Nothing here re-labels. The runtime is the only layer that
// evaluated the wheels, the GPU, the CAS or the models, so it alone names the verdict —
// preparation, wheel, GPU, CAS and network refusals keep their names, and "does not fit"
// is said by a Fit and nobody else (model-code-fit §3). A child that died without a
// document (a signal, a traceback) is `untyped`, with the tail of what it wrote as the
// detail, because the end of a traceback is where the exception is.
func RuntimeExit(code int, verb, untyped, stdout, stderr string) *exit.Error {
	var doc struct {
		Error struct {
			Name    string `json:"name"`
			Message string `json:"message"`
			Remedy  string `json:"remedy"`
		} `json:"error"`
	}
	c := exit.Code(code)
	if !c.Valid() {
		c = exit.Internal
	}
	if json.Unmarshal([]byte(stderr), &doc) == nil && doc.Error.Message != "" {
		name := doc.Error.Name
		if name == "" {
			name = untyped
		}
		e := exit.Named(c, name, "%s", doc.Error.Message)
		if doc.Error.Remedy != "" {
			e.WithRemedy("%s", doc.Error.Remedy)
		}
		return e
	}
	said := strings.TrimSpace(stderr)
	if said == "" {
		said = strings.TrimSpace(stdout)
	}
	what := fmt.Sprintf("exited %d", code)
	if code < 0 {
		what = "was killed"
	}
	return exit.Named(c, untyped, "`cozy-runtime %s` %s without a typed refusal: %s",
		verb, what, tail(said))
}

// tail keeps the END of what a child wrote; a traceback names its exception last.
func tail(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > 400 {
		return "…" + string(r[len(r)-400:])
	}
	return s
}
