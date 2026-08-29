package live

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

const weightlessRef = "cozy/weightless"

// TestProductPath drives only the public Kong surface: install a real weightless
// release, auto-start the controller on invoke, cross Runtime and the worker wire,
// accept an output, release idle residency, then exit cleanly.
func TestProductPath(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-live", "product")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))

	code, help := runCozy(t, root)
	for _, want := range []string{
		"Usage: cozy", "endpoint install", "model download", "invoke run",
		"rental new", "unload", "exit",
	} {
		if code != 0 || !strings.Contains(help, want) {
			t.Fatalf("bare cozy omitted %q [exit %d]\n%s", want, code, help)
		}
	}
	for _, retired := range []string{" up ", " down ", "workflow", "video", "job submit"} {
		if strings.Contains(help, retired) {
			t.Fatalf("bare cozy retained %q\n%s", retired, help)
		}
	}
	if code, out := runCozy(t, root, "up"); code != 2 || !strings.Contains(out, "cli.usage") {
		t.Fatalf("retired up command did not refuse [exit %d]\n%s", code, out)
	}

	archive := weightlessRelease(t)
	data, err := os.ReadFile(archive)
	must(t, err)
	digest := sha256.Sum256(data)
	code, out := runCozy(t, root, "endpoint", "install", weightlessRef,
		"--from", archive, "--digest", "sha256:"+hex.EncodeToString(digest[:]))
	if code != 0 {
		t.Fatalf("endpoint install [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "endpoint", "list", "--json"); code != 0 ||
		!strings.Contains(out, weightlessRef) {
		t.Fatalf("endpoint list omitted the install [exit %d]\n%s", code, out)
	}

	outDir := filepath.Join(root, "out")
	code, out = runCozy(t, root, "invoke", "run", weightlessRef+"/v1/tile",
		"size=32", "seed=7", "--out", outDir)
	if code != 0 {
		t.Fatalf("invoke run [exit %d]\n%s", code, out)
	}
	saved := filepath.Join(outDir, "image.png")
	image, err := os.ReadFile(saved)
	if err != nil || len(image) < 8 || string(image[1:4]) != "PNG" {
		t.Fatalf("invoke did not publish its declared PNG at %s: %v", saved, err)
	}

	code, listed := runCozy(t, root, "invoke", "list", "--json")
	if code != 0 {
		t.Fatalf("invoke list [exit %d]\n%s", code, listed)
	}
	var document map[string]any
	if err := json.Unmarshal([]byte(listed), &document); err != nil || document["ok"] != true {
		t.Fatalf("invoke list is not one successful JSON document: %v\n%s", err, listed)
	}
	if code, out := runCozy(t, root, "invoke", "run", weightlessRef+"/v1/nosuch"); code != 1 || !strings.Contains(out, "not_found") {
		t.Fatalf("operational refusal was not shell exit 1 with detail [exit %d]\n%s", code, out)
	}

	if code, out := runCozy(t, root, "unload"); code != 0 || !strings.Contains(out, "stopped") {
		t.Fatalf("unload [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "exit"); code != 0 || !strings.Contains(out, "controller: stopped") {
		t.Fatalf("exit [exit %d]\n%s", code, out)
	}
}

func weightlessRelease(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	must(t, err)
	repo := filepath.Join(home, "cozy_v2", "cozy-runtime") //cozy:allow peer source; the fixture builds it and never executes a host runtime
	if _, err := os.Stat(filepath.Join(repo, "pyproject.toml")); err != nil {
		t.Skipf("no cozy-runtime peer at %s: %v", repo, err)
	}
	dir := t.TempDir()
	build := exec.Command("/usr/bin/nice", "-n", "19", "python3",
		"internal/live/testdata/build-weightless.py", "--out", dir)
	build.Dir = "../.."
	build.Env = childEnv(t, repo, "RUNTIME_REPO="+repo)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building the weightless release: %v\n%s", err, out)
	}
	return filepath.Join(dir, "weightless-1.0.0.tar.gz")
}
