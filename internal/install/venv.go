package install

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
)

// EnvironmentReceipt is the environment record: exactly what produced this generation's venv.
type EnvironmentReceipt struct {
	Python     string
	UV         string
	LockDigest string
	Platform   string
	Extra      string
	Packages   int
	Closure    string
	Warnings   []string
}

// MaterializeEnvironment is the ONE code-executing step, and it runs only after the source has
// been verified. The lock is absolute: there is no relaxed fallback and no resolve
// that could write one — a lock that cannot satisfy this host refuses with uv's
// exact words.
//
// The spelling is `uv sync --locked`, not the design's `uv sync --frozen`. Observed
// on uv 0.12.7: `--frozen` skips the up-to-date check and happily
// installs a lock that does not match the release's pyproject, which is exactly the
// silent-different-closure outcome the rule exists to prevent. `--locked` refuses
// that, never writes uv.lock, and never falls back to a resolve; a matching lock
// needs no network at all (verified with UV_OFFLINE=1). The two flags are mutually
// exclusive in uv, so this is the stronger reading of one rule, not a second one.
func MaterializeEnvironment(sourceDir, venvDir string) (*EnvironmentReceipt, *exit.Error) {
	lock := filepath.Join(sourceDir, "uv.lock")
	lockDigest, err := fileDigest(lock)
	if err != nil {
		return nil, exit.Named(exit.Structural, "lock_missing",
			"the release carries no uv.lock at %s", lock).
			WithRemedy("a package release pins its whole closure; `uv sync --locked` has nothing to install without it")
	}

	env := &EnvironmentReceipt{
		LockDigest: "sha256:" + lockDigest,
		Platform:   runtime.GOOS + "/" + runtime.GOARCH,
		UV:         toolVersion("uv", "--version"),
	}
	env.Extra = pickCUDAExtra(sourceDir, &env.Warnings)

	args := []string{"sync", "--locked", "--no-progress"}
	if env.Extra != "" {
		args = append(args, "--extra", env.Extra)
	}
	cmd := exec.Command("uv", args...)
	cmd.Dir = sourceDir
	cmd.Env = config.Frozen().Tool(
		"UV_PROJECT_ENVIRONMENT=" + venvDir,
	)
	var out strings.Builder
	cmd.Stdout, cmd.Stderr = &out, &out
	if err := cmd.Run(); err != nil {
		return nil, exit.Named(exit.Structural, "locked_sync_refused",
			"`uv %s` refused for this host — the lock was not resolved, relaxed, or rewritten",
			strings.Join(args, " ")).
			WithRemedy("uv said: %s", condense(out.String())).
			WithNext("cozy help package install")
	}

	env.Python = pythonVersion(venvDir)
	env.Packages, env.Closure = closure(venvDir)
	return env, nil
}

// MaterializePublishedEnvironment recreates the frozen local environment from the exact
// published project metadata, then installs the exact project and custom wheels. Registry
// dependencies come from uv.lock; wheel paths replace only distributions whose published
// bytes are authoritative. Nothing is inherited from Creator's own Python environment.
func MaterializePublishedEnvironment(sourceDir, venvDir string, project PublishedWheel,
	dependencies []PublishedWheel,
) (*EnvironmentReceipt, *exit.Error) {
	lock := filepath.Join(sourceDir, "uv.lock")
	lockDigest, err := fileDigest(lock)
	if err != nil {
		return nil, exit.Named(exit.Structural, "lock_missing",
			"the published release carries no uv.lock").
			WithRemedy("publish the exact uv.lock that freezes the package environment")
	}
	env := &EnvironmentReceipt{
		LockDigest: "sha256:" + lockDigest,
		Platform:   runtime.GOOS + "/" + runtime.GOARCH,
		UV:         toolVersion("uv", "--version"),
	}
	if problem := runUV(sourceDir, config.Frozen().Tool(), "package_python_incompatible",
		"the package Python requirement cannot select an interpreter",
		"venv", "--no-progress", venvDir); problem != nil {
		return nil, problem
	}
	requirements := filepath.Join(filepath.Dir(venvDir), "locked-requirements.txt")
	// Published local/workspace sources are represented by their exact wheels, not by the
	// author's paths. Export the committed lock without reopening those unavailable paths;
	// the final pip check joins the wheel requirements back to this frozen registry closure.
	args := []string{"export", "--frozen", "--no-dev", "--no-emit-project",
		"--format", "requirements.txt", "--output-file", requirements, "--no-progress"}
	seen := map[string]bool{}
	for _, wheel := range dependencies {
		name := strings.TrimSpace(wheel.Distribution)
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		args = append(args, "--no-emit-package", name)
	}
	if problem := runUV(sourceDir, config.Frozen().Tool(), "locked_environment_refused",
		"the published lock cannot export its exact registry closure", args...); problem != nil {
		return nil, problem
	}
	if problem := runUV(sourceDir, config.Frozen().Tool(), "locked_environment_refused",
		"the exact registry closure is incompatible with the selected Python environment",
		"pip", "install", "--no-deps", "--require-hashes", "--python",
		home.VenvPython(venvDir), "--requirements", requirements); problem != nil {
		return nil, problem
	}
	wheels := append([]PublishedWheel{project}, dependencies...)
	args = []string{"pip", "install", "--offline", "--no-index", "--no-deps", "--no-build",
		"--python", home.VenvPython(venvDir)}
	for _, wheel := range wheels {
		args = append(args, wheel.Path)
	}
	if problem := runUV(sourceDir, config.Frozen().Tool(), "package_wheel_incompatible",
		"an exact published wheel is incompatible with the selected Python environment", args...); problem != nil {
		return nil, problem
	}
	if problem := runUV(sourceDir, config.Frozen().Tool(), "package_requirement_incompatible",
		"the installed package requirements are not satisfied", "pip", "check", "--python",
		home.VenvPython(venvDir)); problem != nil {
		return nil, problem
	}
	env.Python = pythonVersion(venvDir)
	env.Packages, env.Closure = closure(venvDir)
	return env, nil
}

func runUV(dir string, env []string, code, message string, args ...string) *exit.Error {
	cmd := exec.Command("uv", args...)
	cmd.Dir = dir
	cmd.Env = env
	var out strings.Builder
	cmd.Stdout, cmd.Stderr = &out, &out
	if err := cmd.Run(); err != nil {
		return exit.Named(exit.Validation, code, "%s: %s", message, condense(out.String())).
			WithRemedy("fix the named requirement or wheel and publish a new locked release").
			WithNext("cozy help package install")
	}
	return nil
}

// Disk measures one generation exactly once, at install: bytes only this generation
// holds, and bytes it shares with another venv through a hardlink. `cozy package list` reads
// these numbers back out of the record — it never walks 122k files.
func Disk(dir string) (exclusive, shared int64) {
	_ = filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		if hardlinked(info) {
			shared += info.Size()
			return nil
		}
		exclusive += info.Size()
		return nil
	})
	return
}

func fileDigest(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

var extraName = regexp.MustCompile(`^\s*(cu\d{2,4})\s*=`)

// pickCUDAExtra chooses the declared CUDA extra this host's driver can actually
// run: the highest declared cuXYZ at or below the driver's CUDA version. No
// accelerator, or nothing declared, selects no extra — never a guess.
func pickCUDAExtra(sourceDir string, warn *[]string) string {
	declared := declaredExtras(filepath.Join(sourceDir, "pyproject.toml"))
	if len(declared) == 0 {
		return ""
	}
	host := hostCUDA()
	if host == 0 {
		*warn = append(*warn, fmt.Sprintf(
			"no CUDA driver reported; installing without a CUDA extra (declared: %s)", strings.Join(declared, ", ")))
		return ""
	}
	best, bestN := "", 0
	for _, name := range declared {
		n, err := strconv.Atoi(strings.TrimPrefix(name, "cu"))
		if err != nil {
			continue
		}
		if n <= host && n > bestN {
			best, bestN = name, n
		}
	}
	if best == "" {
		*warn = append(*warn, fmt.Sprintf(
			"this host's CUDA %d.%d is below every declared extra (%s); installing without one",
			host/10, host%10, strings.Join(declared, ", ")))
	}
	return best
}

func declaredExtras(pyproject string) []string {
	f, err := os.Open(pyproject)
	if err != nil {
		return nil
	}
	defer f.Close()
	var out []string
	in := false
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if t := strings.TrimSpace(line); strings.HasPrefix(t, "[") {
			in = t == "[project.optional-dependencies]"
			continue
		}
		if in {
			if m := extraName.FindStringSubmatch(line); m != nil {
				out = append(out, m[1])
			}
		}
	}
	return out
}

var cudaVersion = regexp.MustCompile(`CUDA Version:\s*(\d+)\.(\d+)`)

// hostCUDA is the driver's maximum CUDA version as cuXYZ digits (13.0 -> 130).
func hostCUDA() int {
	cmd := exec.Command("nvidia-smi")
	// The allowlisted tool environment, like every other spawn: a probe that inherited
	// the full parent environment (TENSORHUB_TOKEN included) was the one production
	// spawn invisible to the env fence (cl-026).
	cmd.Env = config.Frozen().Tool()
	out, err := cmd.Output()
	if err != nil {
		return 0
	}
	m := cudaVersion.FindSubmatch(out)
	if m == nil {
		return 0
	}
	major, _ := strconv.Atoi(string(m[1]))
	minor, _ := strconv.Atoi(string(m[2]))
	return major*10 + minor
}

func pythonVersion(venvDir string) string {
	out := strings.TrimSpace(runOut(home.VenvPython(venvDir), "-V"))
	return strings.TrimSpace(strings.TrimPrefix(out, "Python"))
}

func closure(venvDir string) (int, string) {
	out := runOut("uv", "pip", "list", "--format=json", "--python", home.VenvPython(venvDir))
	var pkgs []struct{ Name, Version string }
	if json.Unmarshal([]byte(out), &pkgs) != nil {
		return 0, ""
	}
	lines := make([]string, 0, len(pkgs))
	for _, p := range pkgs {
		lines = append(lines, p.Name+"=="+p.Version)
	}
	sort.Strings(lines)
	return len(lines), strings.Join(lines, "\n")
}

func toolVersion(name string, args ...string) string {
	return strings.TrimSpace(strings.TrimPrefix(runOut(name, args...), name+" "))
}

func runOut(name string, args ...string) string {
	cmd := exec.Command(name, args...)
	cmd.Env = config.Frozen().Tool()
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return string(out)
}

// condense flattens a tool's output into one remedy line, keeping its own words.
func condense(s string) string {
	var kept []string
	for _, line := range strings.Split(s, "\n") {
		if t := strings.TrimSpace(line); t != "" {
			kept = append(kept, t)
		}
	}
	if len(kept) > 6 {
		kept = kept[len(kept)-6:]
	}
	return strings.Join(kept, " | ")
}
