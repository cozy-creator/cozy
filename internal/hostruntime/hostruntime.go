// Package hostruntime is THIS host's cozy-runtime tool: the package-independent worker control
// process and the ONE reader of package metadata on this host. Package code is untrusted and a
// reading of it never imports it (cl-175); it runs only through an install's own venv, as a
// worker. launch, install and packagepublish all ask here, and nothing here imports them.
package hostruntime

import (
	"bytes"
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
var InstallCommand = installCommand(ToolRelease)

func installCommand(minimum string) string {
	return fmt.Sprintf("uv tool install --force --refresh-package cozy-runtime --python 3.12 '%s'", requirement(minimum))
}

func requirement(minimum string) string { return "cozy-runtime[media,model-execution]>=" + minimum }

// hostRuntimeInstall is the one remedy for a host tool this Cozy cannot drive.
var hostRuntimeInstall = fmt.Sprintf("install cozy-runtime %s or newer: %s — then retry",
	ToolFloor, InstallCommand)

// tool is one cozy-runtime executable and the release it says it is.
type tool struct {
	path    string
	release pep440.Version
}

// hostRuntimeVerdicts memoizes the tool chosen per minimum for this process: a version verb is
// one interpreter start and an install one uv run, so the host is asked once per Calcifer. Only
// an admission is kept, so a reinstall takes effect without a Calcifer restart.
var hostRuntimeVerdicts = struct {
	sync.Mutex
	chosen map[string]tool
}{chosen: map[string]tool{}}

// Path is the tool this Cozy asks package-independent questions: a cozy-runtime at ToolRelease
// or newer (atLeast). The tool answers typed CLI verbs and speaks no machine protocol to this
// Cozy, so the protocol it names is not read. Its own `version` verb is the fact, asked here.
func Path(env []string) (string, *exit.Error) {
	chosen, problem := atLeast(env, release)
	return chosen.path, problem
}

// atLeast is a host tool at minimum or newer: the one on PATH when it is that new; else this
// Cozy's own copy, brought forward through uv when it is older; else (offline, uv refused) the
// newest one this host has, with a warning. Only a tool below ToolFloor, or none at all, with no
// own copy to be had, refuses.
func atLeast(env []string, minimum pep440.Version) (tool, *exit.Error) {
	onPath, _ := exec.LookPath(Distribution)
	own := ownTool(env)
	key := onPath + "\x00" + own + "\x00" + minimum.String()
	hostRuntimeVerdicts.Lock()
	chosen, ok := hostRuntimeVerdicts.chosen[key]
	hostRuntimeVerdicts.Unlock()
	if ok {
		return chosen, nil
	}
	chosen, problem := choose(onPath, own, minimum, env)
	if problem != nil {
		return tool{}, problem
	}
	hostRuntimeVerdicts.Lock()
	hostRuntimeVerdicts.chosen[key] = chosen
	hostRuntimeVerdicts.Unlock()
	return chosen, nil
}

func choose(onPath, own string, minimum pep440.Version, env []string) (tool, *exit.Error) {
	var best tool
	var problem *exit.Error
	if onPath != "" {
		var found pep440.Version
		if found, problem = toolRelease(onPath, env); problem == nil {
			if !found.LessThan(minimum) {
				return tool{onPath, found}, nil
			}
			best = tool{onPath, found}
		}
	}
	current, why := ownCopy(own, minimum, best, env)
	if why == "" {
		return current, nil
	}
	if current.path != "" && (best.path == "" || best.release.LessThan(current.release)) {
		best = current
	}
	switch {
	case best.path == "" && problem != nil:
		return tool{}, problem
	case best.path == "":
		return tool{}, exit.Named(exit.Structural, "host_runtime_missing",
			"this host has no cozy-runtime command on PATH, and this Cozy could not install its own: %s", why).
			WithRemedy("%s", hostRuntimeInstall)
	case best.release.LessThan(floor):
		return tool{}, exit.Named(exit.Structural, "host_runtime_below_floor",
			"cozy-runtime %s is release %s; this Cozy needs %s or newer for native model ingestion and managed result collection, and could not install its own: %s",
			best.path, best.release.String(), ToolFloor, why).
			WithRemedy("%s", hostRuntimeInstall)
	}
	fmt.Fprintf(os.Stderr, "cozy: warning: reading packages with cozy-runtime %s (%s), older than %s, "+
		"because a newer one could not be installed (%s): what was added since, such as another "+
		"package's types or a newer SDK's constructs, may be missing. Install it: %s\n",
		best.release.String(), best.path, minimum.String(), why, installCommand(minimum.String()))
	return best, nil
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

// ownCopy is this Cozy's own tool at minimum or newer, upgraded through uv when it is absent or
// older: it, or the copy there is (none when absent) and why it is not that new. uv installs the
// newest release at or above minimum, and nothing older replaces a newer copy, so it never moves
// below what another package needed. A one-line stage says so, naming what it replaces.
func ownCopy(own string, minimum pep440.Version, onPath tool, env []string) (tool, string) {
	current := func() tool {
		if found, problem := toolRelease(own, env); problem == nil {
			return tool{own, found}
		}
		return tool{}
	}
	if have := current(); have.path != "" && !have.release.LessThan(minimum) {
		return have, ""
	}
	root := filepath.Dir(filepath.Dir(own))
	if err := os.MkdirAll(root, 0o700); err != nil {
		return current(), err.Error()
	}
	lock, err := os.OpenFile(filepath.Join(root, ".lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return current(), err.Error()
	}
	defer lock.Close()
	if err := flock.Block(lock); err != nil {
		return current(), err.Error()
	}
	// Another Cozy may have brought it forward while this one waited.
	have := current()
	if have.path != "" && !have.release.LessThan(minimum) {
		return have, ""
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		return have, "uv is not on PATH"
	}
	if have.path == "" || (onPath.path != "" && have.release.LessThan(onPath.release)) {
		have = onPath
	}
	if have.path == "" {
		fmt.Fprintf(os.Stderr, "Installing cozy-runtime %s or newer to read packages...\n", minimum.String())
	} else {
		fmt.Fprintf(os.Stderr, "Upgrading cozy-runtime %s to %s or newer to read packages...\n",
			have.release.String(), minimum.String())
	}
	want := requirement(minimum.String())
	cmd := exec.Command(uv, "tool", "install", "--force", "--refresh-package", Distribution, "--python", "3.12", want)
	cmd.Env = append(append([]string{}, env...), "UV_TOOL_DIR="+filepath.Join(root, "tools"),
		"UV_TOOL_BIN_DIR="+filepath.Dir(own))
	var output bytes.Buffer
	cmd.Stdout, cmd.Stderr = &output, &output
	if err := cmd.Run(); err != nil {
		return current(), fmt.Sprintf("`uv tool install %s` failed: %s %s", want, err, condense(output.String()))
	}
	installed := current()
	if installed.path == "" {
		return tool{}, fmt.Sprintf("uv installed no working %s at %s", Distribution, own)
	}
	if installed.release.LessThan(minimum) {
		return installed, fmt.Sprintf("uv installed release %s", installed.release.String())
	}
	return installed, ""
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
