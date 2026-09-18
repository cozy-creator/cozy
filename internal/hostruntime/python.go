package hostruntime

import (
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
	Format          string              `json:"format"`
	SupportedMinors []string            `json:"supported_minors"`
	Interpreters    []PythonInterpreter `json:"interpreters"`
}
type PythonInterpreter struct {
	Executable string `json:"executable"`
	Version    string `json:"version"`
	ABI        string `json:"abi"`
}

func PythonExecutors(ctx context.Context) (PythonInventory, *exit.Error) {
	env := config.Frozen().Tool()
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
	if err != nil || readErr != nil || len(raw) > 1<<20 || json.Unmarshal(raw, &inventory) != nil || len(inventory.SupportedMinors) == 0 || inventory.Format != "cozy.python-interpreters/1" {
		return inventory, exit.Named(exit.Structural, "python_window_unavailable", "installed Runtime cannot report its supported Python executor window").WithRemedy("upgrade cozy-runtime and retry")
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
		if e != nil || !supported[PythonMinor(candidate.Version)] || !filepath.IsAbs(candidate.Executable) || !bounds.Check(version) {
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
	inventory, problem := PythonExecutors(ctx)
	if problem != nil {
		return PythonInterpreter{}, problem
	}
	return inventory.Select(document.Project.RequiresPython, explicit)
}
