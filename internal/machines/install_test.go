package machines

import (
	"archive/zip"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Use real uv and installable wheels: a framework package installed outside the
// Runtime's requirements must survive upgrades and an installation that fails
// after uv has already replaced the selected pair.
func TestInstallPreservesBaseAndRestoresFailedPair(t *testing.T) {
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv is required")
	}
	t.Setenv("UV_NO_INDEX", "1")
	h := NewHost(filepath.Join(t.TempDir(), "machine"), "", nil)
	agent := filepath.Join(t.TempDir(), "cozy-machine")
	writeInstallFile(t, agent, []byte("first-agent"), 0o755)
	tensorfs := installWheel(t, "tensorfs", "0.3.78", "tfs")
	first := Source{Host: agent, RuntimeWheel: installWheel(t, "cozy_runtime", "0.18.85", "cozy-runtime-worker"), TensorFSWheel: tensorfs}
	if _, problem := h.Install(context.Background(), first, uv); problem != nil {
		t.Fatal(problem)
	}
	python := filepath.Join(h.python(), "bin/python")
	base := installWheel(t, "machine_base_framework", "2.14.0+cu130", "")
	installCommand(t, uv, "pip", "install", "--no-config", "--python", python, base)
	marker := filepath.Join(h.python(), "operator-settings")
	writeInstallFile(t, marker, []byte("keep"), 0o600)
	second := first
	second.RuntimeWheel = installWheel(t, "cozy_runtime", "0.18.86", "cozy-runtime-worker")
	if _, problem := h.Install(context.Background(), second, uv); problem != nil {
		t.Fatal(problem)
	}
	assertInstallVersion(t, python, "cozy-runtime", "0.18.86")
	assertInstallVersion(t, python, "machine-base-framework", "2.14.0+cu130")
	if raw, err := os.ReadFile(marker); err != nil || string(raw) != "keep" {
		t.Fatalf("existing Python environment was replaced: %q %v", raw, err)
	}
	before, err := os.ReadFile(h.path("installed.json"))
	if err != nil {
		t.Fatal(err)
	}
	third := second
	third.RuntimeWheel = installWheel(t, "cozy_runtime", "0.18.87", "cozy-runtime-worker")
	writeInstallFile(t, agent, []byte("candidate-agent"), 0o755)
	// Simulate an error after installation has already mutated the environment.
	failingUV := filepath.Join(t.TempDir(), "uv")
	writeInstallFile(t, failingUV, []byte(fmt.Sprintf("#!/bin/sh\n%q \"$@\" || exit $?\ncase \"$*\" in *cozy_runtime-0.18.87*) echo injected-install-failure >&2; exit 1;; esac\n", uv)), 0o755)
	if _, problem := h.Install(context.Background(), third, failingUV); problem == nil || !strings.Contains(problem.Error(), "restored the previous Runtime and TensorFS") {
		t.Fatalf("failed upgrade did not report restoration: %v", problem)
	}
	assertInstallVersion(t, python, "cozy-runtime", "0.18.86")
	assertInstallVersion(t, python, "tensorfs", "0.3.78")
	assertInstallVersion(t, python, "machine-base-framework", "2.14.0+cu130")
	after, err := os.ReadFile(h.path("installed.json"))
	if err != nil || string(before) != string(after) {
		t.Fatalf("failed upgrade changed the installation record: %v", err)
	}
	if raw, err := os.ReadFile(h.binary()); err != nil || string(raw) != "first-agent" {
		t.Fatalf("failed upgrade replaced the previous agent: %q %v", raw, err)
	}
	if _, err := os.Stat(filepath.Join(h.wheels(), filepath.Base(second.RuntimeWheel))); err != nil {
		t.Fatalf("failed upgrade discarded previous wheel: %v", err)
	}
	// Published installs still upgrade an existing environment, without resetting
	// its base. A local wheel index makes this independent of PyPI and its latest release.
	index := t.TempDir()
	for _, wheel := range []string{third.RuntimeWheel, tensorfs} {
		raw, err := os.ReadFile(wheel)
		if err != nil {
			t.Fatal(err)
		}
		writeInstallFile(t, filepath.Join(index, filepath.Base(wheel)), raw, 0o644)
	}
	t.Setenv("UV_FIND_LINKS", index)
	if _, problem := h.Install(context.Background(), Source{Host: agent}, uv); problem != nil {
		t.Fatal(problem)
	}
	assertInstallVersion(t, python, "cozy-runtime", "0.18.87")
	assertInstallVersion(t, python, "machine-base-framework", "2.14.0+cu130")
	if wheels, err := os.ReadDir(h.wheels()); err != nil || len(wheels) != 0 {
		t.Fatalf("published install kept obsolete SDK wheels: %v %v", wheels, err)
	}
	// An unusable environment must be retained for repair, never cleared by venv.
	if err := os.Remove(python); err != nil {
		t.Fatal(err)
	}
	if _, problem := h.Install(context.Background(), second, uv); problem == nil || !strings.Contains(problem.Error(), "it was preserved") {
		t.Fatalf("invalid existing environment was not preserved: %v", problem)
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatal(err)
	}
}

func installWheel(t *testing.T, name, version, entrypoint string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name+"-"+version+"-py3-none-any.whl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	w := zip.NewWriter(f)
	dist := name + "-" + version + ".dist-info/"
	files := map[string]string{
		name + ".py":      "def main(): pass\n",
		dist + "METADATA": "Metadata-Version: 2.1\nName: " + strings.ReplaceAll(name, "_", "-") + "\nVersion: " + version + "\nProvides-Extra: media\n",
		dist + "WHEEL":    "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
	}
	if name == "cozy_runtime" {
		files[dist+"METADATA"] += "Requires-Dist: tensorfs>=0.3.78\n"
	}
	if entrypoint != "" {
		files[dist+"entry_points.txt"] = "[console_scripts]\n" + entrypoint + " = " + name + ":main\n"
	}
	var record strings.Builder
	for path := range files {
		fmt.Fprintln(&record, path+",,")
	}
	files[dist+"RECORD"] = record.String() + dist + "RECORD,,\n"
	for path, body := range files {
		file, err := w.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := file.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func installCommand(t *testing.T, command string, args ...string) string {
	t.Helper()
	output, err := exec.Command(command, args...).CombinedOutput()
	if err != nil {
		t.Fatalf("%s: %v\n%s", command, err, output)
	}
	return strings.TrimSpace(string(output))
}

func assertInstallVersion(t *testing.T, python, name, version string) {
	t.Helper()
	got := installCommand(t, python, "-I", "-c", "import importlib.metadata as m,sys; print(m.version(sys.argv[1]))", name)
	if got != version {
		t.Fatalf("%s version = %s, want %s", name, got, version)
	}
}

func writeInstallFile(t *testing.T, path string, body []byte, mode os.FileMode) {
	t.Helper()
	if err := os.WriteFile(path, body, mode); err != nil {
		t.Fatal(err)
	}
}
