package hostruntime

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/pelletier/go-toml/v2"
)

// PythonInventory reads the installed Runtime's versioned policy and actual
// executors. The control interpreter does not determine package compatibility.
type PythonInventory struct {
	ManagedRoot         string              `json:"managed_root"`
	Format              string              `json:"format"`
	ProvisionableMinors []string            `json:"provisionable_minors"`
	SupportedMinors     []string            `json:"supported_minors"`
	Interpreters        []PythonInterpreter `json:"interpreters"`
}
type PythonInterpreter struct {
	Executable string `json:"executable"`
	Version    string `json:"version"`
	ABI        string `json:"abi"`
}

func PythonExecutors(ctx context.Context) (PythonInventory, *exit.Error) {
	env := pythonEnvironment()
	bin, problem := Path(env)
	if problem != nil {
		return PythonInventory{}, problem
	}
	cmd := exec.CommandContext(ctx, bin, "--json", "python-interpreters")
	cmd.Env = env
	out, err := cmd.StdoutPipe()
	if err != nil || cmd.Start() != nil {
		return PythonInventory{}, exit.New(exit.Structural, "cannot inspect supported Python executors")
	}
	raw, readErr := io.ReadAll(io.LimitReader(out, (1<<20)+1))
	if len(raw) > 1<<20 {
		_ = cmd.Process.Kill()
	}
	err = cmd.Wait()
	var inventory PythonInventory
	if err != nil || readErr != nil || len(raw) > 1<<20 || json.Unmarshal(raw, &inventory) != nil || !filepath.IsAbs(inventory.ManagedRoot) || len(inventory.SupportedMinors) == 0 || inventory.Format != "cozy.python-interpreters/1" {
		return inventory, exit.Named(exit.Structural, "python_window_unavailable", "installed Runtime cannot report its supported Python executors and owned root").WithRemedy("upgrade cozy-runtime and retry")
	}
	return inventory, nil
}
func PythonMinor(version string) string {
	parts := strings.Split(version, ".")
	if len(parts) < 2 {
		return ""
	}
	return parts[0] + "." + parts[1]
}
func (inventory PythonInventory) Select(requires, explicit string) (PythonInterpreter, *exit.Error) {
	specifier := strings.TrimSpace(requires)
	if specifier == "" {
		specifier = ">=0"
	}
	bounds, err := pep440.NewSpecifiers(specifier)
	if err != nil {
		return PythonInterpreter{}, exit.Named(exit.Validation, "package_python_invalid", "invalid Requires-Python %q", requires)
	}
	supported := map[string]bool{}
	for _, minor := range inventory.SupportedMinors {
		supported[minor] = true
	}
	explicit = strings.TrimSpace(explicit)
	if explicit != "" && !supported[PythonMinor(explicit)] {
		return PythonInterpreter{}, exit.Named(exit.Validation, "package_python_unsupported", "project Python %s is outside the supported window %s", explicit, strings.Join(inventory.SupportedMinors, ", ")).WithRemedy("upgrade the package to a supported Python version")
	}
	candidates := []PythonInterpreter{}
	for _, candidate := range inventory.Interpreters {
		if _, err := pep440.Parse(candidate.Version); err == nil {
			candidates = append(candidates, candidate)
		}
	}
	sort.Slice(candidates, func(i, j int) bool {
		a, _ := pep440.Parse(candidates[i].Version)
		b, _ := pep440.Parse(candidates[j].Version)
		return a.LessThan(b)
	})
	for _, candidate := range candidates {
		version, e := pep440.Parse(candidate.Version)
		if e != nil || !supported[PythonMinor(candidate.Version)] || !filepath.IsAbs(candidate.Executable) || candidate.ABI != "cp"+strings.ReplaceAll(PythonMinor(candidate.Version), ".", "") || !bounds.Check(version) {
			continue
		}
		if explicit != "" && candidate.Version != explicit && PythonMinor(candidate.Version) != explicit {
			continue
		}
		return candidate, nil
	}
	return PythonInterpreter{}, exit.Named(exit.Validation, "package_python_unsupported", "no installed executor in supported Python window [%s] satisfies Requires-Python %q and project Python %q", strings.Join(inventory.SupportedMinors, ", "), requires, explicit).WithRemedy("upgrade the package or install a supported interpreter with uv python install")
}
func ProjectPython(ctx context.Context, directory string) (PythonInterpreter, *exit.Error) {
	raw, err := os.ReadFile(filepath.Join(directory, "pyproject.toml"))
	var document struct {
		Project struct {
			RequiresPython string `toml:"requires-python"`
		} `toml:"project"`
	}
	if err != nil || toml.Unmarshal(raw, &document) != nil {
		return PythonInterpreter{}, exit.New(exit.Validation, "cannot read project Python requirements")
	}
	var explicit string
	raw, err = os.ReadFile(filepath.Join(directory, ".python-version"))
	if err == nil {
		explicit = strings.TrimSpace(string(raw))
	} else if !os.IsNotExist(err) {
		return PythonInterpreter{}, exit.New(exit.Validation, "cannot read project Python selection")
	}
	return EnsurePython(ctx, document.Project.RequiresPython, explicit)
}

// EnsurePython delegates selection and missing-interpreter provisioning to the
// same Runtime policy used on rented workers. Exact captured patches stay exact.
func EnsurePython(ctx context.Context, requires, explicit string) (PythonInterpreter, *exit.Error) {
	env := pythonEnvironment()
	bin, problem := Path(env)
	if problem != nil {
		return PythonInterpreter{}, problem
	}
	if strings.TrimSpace(requires) == "" {
		requires = ">=0"
	}
	args := []string{"--json", "python-ensure", requires}
	if explicit != "" {
		args = append(args, explicit)
	}
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Env = env
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return PythonInterpreter{}, exit.Named(exit.Deadline, "python_provision_deadline", "Python preparation deadline exceeded")
	}
	if ctx.Err() != nil {
		return PythonInterpreter{}, exit.Named(exit.Canceled, "python_provision_canceled", "Python preparation canceled: %s", ctx.Err())
	}
	if cmd.ProcessState == nil {
		return PythonInterpreter{}, exit.New(exit.Structural, "cannot prepare package Python: %s", err)
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		// Runtime may report provisioning progress before its final typed refusal.
		refusal := strings.TrimSpace(stderr.String())
		for _, line := range strings.Split(refusal, "\n") {
			var doc struct {
				Error json.RawMessage `json:"error"`
			}
			if json.Unmarshal([]byte(line), &doc) == nil && len(doc.Error) > 0 {
				refusal = line
			}
		}
		return PythonInterpreter{}, RuntimeExit(code, "python-ensure", "python_provision_failed", stdout.String(), refusal)
	}
	var selected PythonInterpreter
	if stdout.Len() > 1<<20 || json.Unmarshal(stdout.Bytes(), &selected) != nil || !filepath.IsAbs(selected.Executable) || selected.ABI != "cp"+strings.ReplaceAll(PythonMinor(selected.Version), ".", "") {
		return PythonInterpreter{}, exit.Named(exit.Structural, "python_provision_invalid", "Runtime returned an invalid prepared Python interpreter")
	}
	// Validate the returned identity without duplicating Runtime's version window.
	inventory := PythonInventory{SupportedMinors: []string{PythonMinor(selected.Version)}, Interpreters: []PythonInterpreter{selected}}
	return inventory.Select(requires, explicit)
}

// Python CLI tools and local serving belong to the same Creator home, even
// though serve has a separate working-state home beneath it. Runtime owns the
// interpreter subdirectory; Creator only supplies its configured product home.
func pythonEnvironment() []string {
	cfg := config.Frozen()
	if cfg.Home == "" {
		return cfg.Tool()
	}
	return cfg.Tool("COZY_HOME=" + cfg.Home)
}
