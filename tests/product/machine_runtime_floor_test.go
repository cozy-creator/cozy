package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
)

// Runs need a machine whose Runtime keeps the run output log (0.18.73+). Run with
// -machine-runtime-wheel set to the published 0.18.69 and to 0.18.73: the newer machine's
// image lands in the outputs folder; the older one fails the run at once with one error naming
// the Runtime it needs, never a silent wait. A package change reaches either with no note: an
// older machine keeps no package cache, so there is nothing for it to forget.
func TestAMachineNamesTheRuntimeItNeeds(t *testing.T) {
	integration(t)
	if *privateScriptRuntimeWheel == "" || *machineHostBinary == "" {
		t.Skip("requires -script-runtime-wheel and -machine-host: a real Runtime on this computer's machine")
	}
	wheel, err := filepath.Abs(*privateScriptRuntimeWheel)
	must(t, err)
	version := runtimeFixtureVersion(t, wheel)
	hub := newMachineHub(t)
	hub.mux.HandleFunc("GET /v1/accounts/current", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"name":"proof"}`))
	})
	hub.mux.HandleFunc("DELETE /v1/packages/proof/skew/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"release":"1.0.0","state":"yanked","changed":true}`))
	})
	root, err := os.MkdirTemp("", "cozy-runtime-floor-")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.server.URL+
		"\ntensorhub_token: rental-idle-test\n"), 0o600))
	control := filepath.Join(t.TempDir(), "control")
	uv := func(args ...string) {
		t.Helper()
		if out, err := exec.Command("/usr/bin/nice", append([]string{"-n", "19", "uv"}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("uv: %v %s", err, out)
		}
	}
	uv("venv", control, "--python", "3.12")
	uv("pip", "install", "--python", filepath.Join(control, "bin", "python"), wheel)
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if !t.Failed() {
			must(t, removeAllForce(root))
		}
	})
	cozy := func(args ...string) ([]byte, error) {
		command := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
		command.Env = childEnv(t, root, "PATH="+path)
		return command.CombinedOutput()
	}
	script := filepath.Join(t.TempDir(), "picture.py")
	must(t, os.WriteFile(script, []byte(fmt.Sprintf(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime>=%s"]
# [tool.uv.sources]
# cozy-runtime={path=%q}
# ///
from typing import Annotated
from cozy_runtime.author import AssetBound, ImageAsset, ImageFrame, Outputs
def main(*, out: Outputs) -> Annotated[ImageAsset, AssetBound(media_types=("image/png",))]:
    return out.save_image(ImageFrame(8, 8, bytes([40, 120, 200]) * 64), format="png")
`, version, wheel)), 0o600))

	out, err := cozy("run", script, "--await", "--json")
	older := !versionAtLeast(version, 0, 18, 73)
	if older {
		if err == nil || !strings.Contains(string(out), "machine.runtime_update_required") || !strings.Contains(string(out), "Runtime 0.18.73 or newer") {
			t.Fatalf("a run on a Runtime %s machine did not fail naming the Runtime it needs: %v\n%s", version, err, out)
		}
	} else {
		if err != nil {
			t.Fatalf("the run on a %s machine failed: %v\n%s", version, err, out)
		}
		var result struct {
			Saved []struct {
				Path   string `json:"path"`
				Digest string `json:"digest"`
			} `json:"saved"`
		}
		// Its events stream first; the run's own JSON is the last line.
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		if json.Unmarshal([]byte(lines[len(lines)-1]), &result) != nil || len(result.Saved) != 1 {
			t.Fatalf("the run saved no image:\n%s", out)
		}
		saved := result.Saved[0]
		data, err := os.ReadFile(saved.Path)
		if err != nil || !strings.HasPrefix(saved.Path, filepath.Join(root, "outputs")+string(os.PathSeparator)) || digestOf(data) != saved.Digest {
			t.Fatalf("the image is not whole in the outputs folder: %s: %v", saved.Path, err)
		}
	}

	out, err = cozy("package", "yank", "proof/skew", "--version", "1.0.0")
	if err != nil || strings.Contains(string(out), "keeps what it read") || strings.Contains(string(out), "no machine was told") {
		t.Fatalf("a package change on a Runtime %s machine printed a note: %v\n%s", version, err, out)
	}
	t.Logf("Runtime %s machine: refused %v", version, older)
}

// versionAtLeast compares a released Runtime version with major.minor.patch.
func versionAtLeast(version string, major, minor, patch int) bool {
	var a, b, c int
	if _, err := fmt.Sscanf(strings.SplitN(version, "+", 2)[0], "%d.%d.%d", &a, &b, &c); err != nil {
		return false
	}
	if a != major {
		return a > major
	}
	if b != minor {
		return b > minor
	}
	return c >= patch
}
