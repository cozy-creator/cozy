package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Runtime's version verb observes the installed wheel's embedded source, not
// ambient Git. Resolve only that immutable prefix; never fall back to HEAD.
func qualifiedRuntimeFixtureSHA(repo string, env []string) (string, error) {
	tool, problem := hostruntime.Path(env)
	if problem != nil {
		return "", problem
	}
	command := exec.Command(tool, "--json", "version")
	command.Env = env
	raw, err := command.Output()
	if err != nil {
		return "", fmt.Errorf("observe qualified Runtime fixture provenance: %w", err)
	}
	var version struct {
		Commit string `json:"commit"`
	}
	if json.Unmarshal(raw, &version) != nil || !regexp.MustCompile(`^[0-9a-f]{12}$`).MatchString(version.Commit) {
		return "", fmt.Errorf("qualified Runtime fixture needs immutable build provenance; installed tool reported commit %q", version.Commit)
	}
	resolved, err := exec.Command("git", "-C", repo, "rev-parse", "--verify", version.Commit+"^{commit}").Output()
	if err != nil {
		return "", fmt.Errorf("qualified Runtime commit %s is absent or ambiguous in peer %s; fetch that exact history: %w", version.Commit, repo, err)
	}
	sha := strings.TrimSpace(string(resolved))
	if !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(sha) || !strings.HasPrefix(sha, version.Commit) {
		return "", fmt.Errorf("peer did not resolve qualified Runtime commit %s to its matching immutable object", version.Commit)
	}
	return sha, nil
}

func TestRuntimeFixtureRefusesMissingOrMutableProvenance(t *testing.T) {
	for _, commit := range []string{"", "unknown", "HEAD", "v0.4.0", "not-a-commit!"} {
		t.Run(fmt.Sprintf("commit_%q", commit), func(t *testing.T) {
			bin := t.TempDir()
			answer, err := json.Marshal(map[string]any{"distribution": hostruntime.ToolFloor, "wire_protocol": fmt.Sprintf("cozy.worker.v1+minor.%d", pb.WireMinor), "commit": commit})
			must(t, err)
			// An independently executable peer passes normal wire qualification but
			// has no immutable source authority. The fixture must stop before Git/build.
			must(t, os.WriteFile(filepath.Join(bin, "cozy-runtime"), []byte("#!/bin/sh\nprintf '%s\\n' '"+string(answer)+"'\n"), 0700)) //cozy:allow a stand-in provenance tool, not this host's
			t.Setenv("PATH", bin)
			_, err = qualifiedRuntimeFixtureSHA(filepath.Join(t.TempDir(), "no-repository"), childEnv(t, t.TempDir(), "PATH="+bin))
			if err == nil || !strings.Contains(err.Error(), "immutable build provenance") {
				t.Fatalf("missing/mutable source was not refused before Git/build: %v", err)
			}
		})
	}
}

// Runtime fixture declarations use a public compatibility floor. Exact local
// builds remain selected through their tool.uv.sources wheel path.
func runtimeFixtureVersion(t *testing.T, wheel string) string {
	t.Helper()
	if wheel == "" {
		return hostruntime.PackageFloor
	}
	parts := strings.Split(filepath.Base(wheel), "-")
	if len(parts) < 3 || parts[0] != "cozy_runtime" || parts[1] == "" || !strings.HasSuffix(wheel, ".whl") {
		t.Fatal("fixture requires a named Runtime wheel")
	}
	return strings.SplitN(parts[1], "+", 2)[0]
}
