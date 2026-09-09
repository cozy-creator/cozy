package install

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
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

// EnvironmentReceipt is the environment record: exactly what produced this install's venv.
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
	return materializeEnvironment(sourceDir, venvDir, true)
}

func materializeEnvironment(sourceDir, venvDir string, editable bool) (*EnvironmentReceipt, *exit.Error) {
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
	if !editable {
		args = append(args, "--no-editable")
	}
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
	if env.Python == "" || env.Packages == 0 {
		return nil, exit.New(exit.Structural, "installed environment has no exact Python/distribution metadata")
	}
	return env, nil
}

// MaterializePublishedEnvironment recreates the frozen environment from the exact
// published metadata alone (wire 30): registry dependencies from the committed lock's
// export, the release's own wheels hash-pinned against the org's public index. No wheel
// file is ever staged; every artifact must match a hash the export names
// (`--require-hashes`), so the indexes have no authority over bytes. The export it writes
// — index directives plus sorted exact rows — is retained at
// `<install>/locked-requirements.txt`: it is the exact document Runtime preparation
// consumes and re-consumes at model selection.
func MaterializePublishedEnvironment(sourceDir, venvDir string,
	published *PublishedSource,
) (*EnvironmentReceipt, *exit.Error) {
	lock := filepath.Join(sourceDir, "uv.lock")
	lockDigest, err := fileDigest(lock)
	if err != nil {
		return nil, exit.Named(exit.Structural, "lock_missing",
			"the published release carries no uv.lock").
			WithRemedy("publish the exact uv.lock that freezes the package environment")
	}
	if published.IndexURL == "" {
		return nil, exit.Internalf("published package source names no org index")
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
	appended := append([]PublishedWheel{published.ProjectWheel}, published.Wheels...)
	appended = append(appended, published.LocalWheels...)
	exported := filepath.Join(filepath.Dir(venvDir), ".locked-requirements-export.txt")
	defer os.Remove(exported)
	// Export the committed registry closure without the rows the release's own wheels
	// supply; those rows are re-added below as exact org-index pins.
	args := []string{"export", "--frozen", "--no-dev", "--no-emit-project",
		"--format", "requirements.txt", "--output-file", exported, "--no-progress"}
	seen := map[string]bool{}
	for _, wheel := range appended {
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
	requirements := filepath.Join(filepath.Dir(venvDir), "locked-requirements.txt")
	if problem := writeLockedRequirements(exported, requirements,
		published.IndexURL, appended); problem != nil {
		return nil, problem
	}
	if problem := runUV(sourceDir, config.Frozen().Tool(), "locked_environment_refused",
		"the exact locked closure is incompatible with the selected Python environment",
		"pip", "install", "--no-deps", "--require-hashes", "--python",
		home.VenvPython(venvDir), "--requirements", requirements); problem != nil {
		return nil, problem
	}
	if problem := runUV(sourceDir, config.Frozen().Tool(), "package_requirement_incompatible",
		"the installed package requirements are not satisfied", "pip", "check", "--python",
		home.VenvPython(venvDir)); problem != nil {
		return nil, problem
	}
	env.Python = pythonVersion(venvDir)
	env.Packages, env.Closure = closure(venvDir)
	if env.Python == "" || env.Packages == 0 {
		return nil, exit.New(exit.Structural, "installed environment has no exact Python/distribution metadata")
	}
	return env, nil
}

// runtimeScratchHome is the COZY_HOME every install-time cozy-runtime invocation gets:
// a per-user OS cache directory, never the Creator home. What the runtime derives there
// (its JIT-kernel proof cache) is reusable derived state, and the audited top-level
// `~/.cozy/jit-cache` — 19 empty directories — was exactly this scratch landing in the
// product home (cl-116).
func runtimeScratchHome() string {
	base, err := os.UserCacheDir()
	if err != nil {
		base = os.TempDir()
	}
	return filepath.Join(base, "cozy", "runtime-install")
}

// writeLockedRequirements merges the export's registry rows with the release's own exact
// wheel pins into the ONE document Runtime's reader admits: two https index directives,
// then hash-pinned rows sorted unique by normalized distribution.
func writeLockedRequirements(exported, target, indexURL string,
	wheels []PublishedWheel,
) *exit.Error {
	raw, err := os.ReadFile(exported)
	if err != nil {
		return exit.Internalf("cannot read the exported registry closure: %s", err)
	}
	rows := map[string]string{}
	pending := ""
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.HasSuffix(line, "\\") {
			pending += strings.TrimSuffix(line, "\\") + " "
			continue
		}
		row := strings.TrimSpace(pending + line)
		pending = ""
		if row == "" || strings.HasPrefix(row, "#") || strings.HasPrefix(row, "-") {
			continue
		}
		name := normalizedRequirementName(row)
		if name == "" || rows[name] != "" {
			return exit.Internalf("the exported registry closure row %q is not one exact pin", row)
		}
		rows[name] = strings.Join(strings.Fields(row), " ")
	}
	seen := map[string]bool{}
	for _, wheel := range wheels {
		name := normalizedRequirementName(wheel.Distribution)
		if name == "" || wheel.Version == "" ||
			!strings.HasPrefix(wheel.Digest, "sha256:") || rows[name] != "" || seen[name] {
			return exit.Internalf("published wheel fact %q is incomplete or duplicated",
				wheel.Distribution)
		}
		seen[name] = true
		rows[name] = name + "==" + wheel.Version + " --hash=" + wheel.Digest
	}
	names := make([]string, 0, len(rows))
	for name := range rows {
		names = append(names, name)
	}
	sort.Strings(names)
	var out strings.Builder
	out.WriteString("--index-url https://pypi.org/simple\n")
	out.WriteString("--extra-index-url " + indexURL + "\n")
	for _, name := range names {
		out.WriteString(rows[name] + "\n")
	}
	if err := os.WriteFile(target, []byte(out.String()), 0o600); err != nil {
		return exit.Internalf("cannot retain the locked-requirements export: %s", err)
	}
	return nil
}

var requirementName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*`)
var requirementNormalize = regexp.MustCompile(`[-_.]+`)

func normalizedRequirementName(row string) string {
	name := requirementName.FindString(strings.TrimSpace(row))
	return requirementNormalize.ReplaceAllString(strings.ToLower(name), "-")
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

// Disk measures one install exactly once, at install: the ALLOCATED bytes only this
// tree holds, and the allocated bytes it shares with a name outside it. `cozy package
// list` reads these numbers back out of the record — it never walks 122k files.
//
// Each inode is measured ONCE, however many names inside the tree reach it, and an
// inode whose every link lives inside the tree is EXCLUSIVE — its digest file plus a
// named view die together, so calling it shared double-counted ~26 GB on the audited
// home (cl-116). Only an inode with a link outside the walk is shared. Exclusive is
// therefore also the observed post-delete delta a removal reports.
func Disk(dir string) (exclusive, shared int64) {
	type counted struct {
		links     uint64
		seen      uint64
		allocated int64
	}
	inodes := map[inodeKey]*counted{}
	_ = filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		key, links, allocated, ok := inode(info)
		if !ok {
			exclusive += info.Size()
			return nil
		}
		if row := inodes[key]; row != nil {
			row.seen++
			return nil
		}
		inodes[key] = &counted{links: links, seen: 1, allocated: allocated}
		return nil
	})
	for _, row := range inodes {
		if row.seen >= row.links {
			exclusive += row.allocated
			continue
		}
		shared += row.allocated
	}
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
