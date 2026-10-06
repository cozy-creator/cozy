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
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	pep440 "github.com/aquasecurity/go-pep440-version"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/flock"
)

// ToolFloor is the first released Runtime with everything this Cozy drives: a lower bound.
// Distribution is the Runtime's Python distribution and executable name.
const Distribution = "cozy-runtime"

const ToolFloor = "0.18.67"

// ToolRelease is the Runtime this Cozy is released with: what it reads packages with. A tool
// on PATH older than this is brought forward into this Cozy's own tool directory, never
// refused (newer type resolution, such as a callee package's types, lives in newer tools).
const ToolRelease = "0.19.0"

// PackageFloor is the first package SDK whose executor states its protocol revision and native
// interfaces in its hello.
const PackageFloor = "0.18.51"

var floor, release = pep440.MustParse(ToolFloor), pep440.MustParse(ToolRelease)

// InstallCommand installs this host's tool, cozy-runtime, in its own uv environment.
// install.sh and scripts/install.ps1 run it. TensorFS's `tfs` is not installed: only a
// machine writes its store.
var InstallCommand = fmt.Sprintf(
	"uv tool install --force --refresh-package cozy-runtime --python 3.12 '%s'", toolRequirement)

const toolRequirement = "cozy-runtime[media,model-execution]>=" + ToolRelease

// hostRuntimeInstall is the one remedy for a host tool this Cozy cannot drive.
var hostRuntimeInstall = fmt.Sprintf("install cozy-runtime %s or newer: %s — then retry",
	ToolFloor, InstallCommand)

// hostRuntimeVerdicts memoizes the tool Path chose for this process: a version verb is one
// interpreter start and an install one uv run, so the host is asked once per daemon. Only an
// admission is kept, so a reinstall takes effect without a daemon restart.
var hostRuntimeVerdicts = struct {
	sync.Mutex
	chosen map[string]string
}{chosen: map[string]string{}}

// Path is the tool this Cozy reads packages with: a cozy-runtime at ToolRelease or newer. The
// one on PATH when it is that new; else this Cozy's own copy, installed through uv the first
// time it is needed; else (offline, uv refused) the older one on PATH with a warning. Only a
// tool below ToolFloor, or none at all, with no own copy to be had, refuses. The tool answers
// typed CLI verbs and speaks no machine protocol to this Cozy, so the protocol it names is
// not read. Its own `version` verb is the fact, asked here.
func Path(env []string) (string, *exit.Error) {
	onPath, _ := exec.LookPath(Distribution)
	own := ownTool(env)
	key := onPath + "\x00" + own
	hostRuntimeVerdicts.Lock()
	chosen := hostRuntimeVerdicts.chosen[key]
	hostRuntimeVerdicts.Unlock()
	if chosen != "" {
		return chosen, nil
	}
	chosen, problem := choose(onPath, own, env)
	if problem != nil {
		return "", problem
	}
	hostRuntimeVerdicts.Lock()
	hostRuntimeVerdicts.chosen[key] = chosen
	hostRuntimeVerdicts.Unlock()
	return chosen, nil
}

func choose(onPath, own string, env []string) (string, *exit.Error) {
	var found pep440.Version
	var problem *exit.Error
	if onPath != "" {
		if found, problem = toolRelease(onPath, env); problem == nil && !found.LessThan(release) {
			return onPath, nil
		}
	}
	installed, why := ownCopy(own, env)
	if why == "" {
		return installed, nil
	}
	switch {
	case onPath == "":
		return "", exit.Named(exit.Structural, "host_runtime_missing",
			"this host has no cozy-runtime command on PATH, and this Cozy could not install its own: %s", why).
			WithRemedy("%s", hostRuntimeInstall)
	case problem != nil:
		return "", problem
	case found.LessThan(floor):
		return "", exit.Named(exit.Structural, "host_runtime_below_floor",
			"cozy-runtime %s is release %s; this Cozy needs %s or newer for native model ingestion and managed result collection, and could not install its own: %s",
			onPath, found.String(), ToolFloor, why).
			WithRemedy("%s", hostRuntimeInstall)
	}
	fmt.Fprintf(os.Stderr, "cozy: warning: reading packages with cozy-runtime %s (%s), older than this Cozy's %s, "+
		"because its own could not be installed (%s): type resolution added since, such as another "+
		"package's types, may be missing. %s\n", found.String(), onPath, ToolRelease, why, hostRuntimeInstall)
	return onPath, nil
}

// ownTool is where this Cozy keeps its own tool: a uv tool directory in the user's cache
// (XDG_CACHE_HOME of the environment it runs tools in), shared by every Cozy home on this host.
func ownTool(env []string) string {
	base := ""
	for _, pair := range env {
		if value, ok := strings.CutPrefix(pair, "XDG_CACHE_HOME="); ok && filepath.IsAbs(value) {
			base = value
		}
	}
	if base == "" {
		cache, err := os.UserCacheDir()
		if err != nil {
			cache = os.TempDir()
		}
		base = cache
	}
	return filepath.Join(base, "cozy", "host-runtime", "bin", Distribution)
}

// ownCopy is this Cozy's own tool at ToolRelease or newer, installed through uv when it is
// absent or older: its path, or why there is none.
func ownCopy(own string, env []string) (string, string) {
	if found, problem := toolRelease(own, env); problem == nil && !found.LessThan(release) {
		return own, ""
	}
	root := filepath.Dir(filepath.Dir(own))
	if err := os.MkdirAll(root, 0o700); err != nil {
		return "", err.Error()
	}
	lock, err := os.OpenFile(filepath.Join(root, ".lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return "", err.Error()
	}
	defer lock.Close()
	if err := flock.Block(lock); err != nil {
		return "", err.Error()
	}
	// Another Cozy may have installed it while this one waited.
	if found, problem := toolRelease(own, env); problem == nil && !found.LessThan(release) {
		return own, ""
	}
	cmd := exec.Command("uv", "tool", "install", "--force", "--python", "3.12", toolRequirement)
	cmd.Env = append(append([]string{}, env...), "UV_TOOL_DIR="+filepath.Join(root, "tools"),
		"UV_TOOL_BIN_DIR="+filepath.Dir(own))
	var output bytes.Buffer
	cmd.Stdout, cmd.Stderr = &output, &output
	fmt.Fprintf(os.Stderr, "cozy: installing cozy-runtime %s or newer for reading packages...\n", ToolRelease)
	if err := cmd.Run(); err != nil {
		return "", fmt.Sprintf("`uv tool install %s` failed: %s", toolRequirement, condense(output.String()))
	}
	found, problem := toolRelease(own, env)
	if problem != nil {
		return "", problem.Message
	}
	if found.LessThan(release) {
		return "", fmt.Sprintf("uv installed release %s", found.String())
	}
	return own, ""
}

// toolRelease runs `cozy-runtime --json version` and reads the release the tool prints for
// itself.
func toolRelease(path string, env []string) (pep440.Version, *exit.Error) {
	if _, err := os.Stat(path); err != nil {
		return pep440.Version{}, unreadableHostRuntime(path, "it is absent")
	}
	cmd := exec.Command(path, "--json", "version")
	cmd.Env = env
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if cmd.ProcessState == nil {
		return pep440.Version{}, unreadableHostRuntime(path, "cannot run it: %s", err)
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		said := strings.TrimSpace(stderr.String())
		if said == "" {
			said = strings.TrimSpace(stdout.String())
		}
		return pep440.Version{}, unreadableHostRuntime(path, "`cozy-runtime --json version` exited %d: %s",
			code, condense(said))
	}
	var answer struct {
		Distribution string `json:"distribution"`
	}
	if err := json.Unmarshal([]byte(stdout.String()), &answer); err != nil {
		return pep440.Version{}, unreadableHostRuntime(path, "`cozy-runtime --json version` answered a document "+
			"this host cannot read: %s", err)
	}
	found, err := pep440.Parse(answer.Distribution)
	if err != nil {
		return pep440.Version{}, unreadableHostRuntime(path, "`cozy-runtime --json version` names distribution %q, "+
			"not a PEP 440 release", answer.Distribution)
	}
	return found, nil
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
