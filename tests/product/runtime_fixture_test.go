package producttest

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
)

// hostRuntimeIdentity is the host cozy-runtime's own report of what it is: its released
// distribution and the source commit its wheel embeds. A fixture package vendors exactly
// that Runtime, so the worker never runs a neighbour of the host's.
func hostRuntimeIdentity(env []string) (distribution, commit string, err error) {
	tool, problem := hostruntime.Path(env)
	if problem != nil {
		return "", "", problem
	}
	command := exec.Command(tool, "--json", "version")
	command.Env = env
	raw, err := command.Output()
	if err != nil {
		return "", "", fmt.Errorf("observe the host Runtime: %w", err)
	}
	var version struct {
		Distribution string `json:"distribution"`
		Commit       string `json:"commit"`
	}
	if json.Unmarshal(raw, &version) != nil || version.Distribution == "" ||
		!regexp.MustCompile(`^[0-9a-f]{12}$`).MatchString(version.Commit) {
		return "", "", fmt.Errorf("the host Runtime reports no released identity: %s", raw)
	}
	return version.Distribution, version.Commit, nil
}

// Runtime fixture declarations use the oldest Runtime this Creator speaks to. Exact local
// builds remain selected through their tool.uv.sources wheel path.
func runtimeFixtureVersion(t *testing.T, wheel string) string {
	t.Helper()
	if wheel == "" {
		return hostruntime.ToolFloor
	}
	parts := strings.Split(filepath.Base(wheel), "-")
	if len(parts) < 3 || parts[0] != "cozy_runtime" || parts[1] == "" || !strings.HasSuffix(wheel, ".whl") {
		t.Fatal("fixture requires a named Runtime wheel")
	}
	return strings.SplitN(parts[1], "+", 2)[0]
}
