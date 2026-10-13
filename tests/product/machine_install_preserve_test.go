package producttest

import (
	"archive/zip"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/machines"
)

// Bootstrap uses actual uv and preserves machine-owned packages and settings.
// Replacement/rollback coverage executes the actual machine transaction in
// TestTheMachineUpdatesItsOwnRuntime and TestLocalInstallUsesMachineTransaction.
func TestBootstrapPreservesMachineBase(t *testing.T) {
	uv := offlineInstallerUV(t)
	h := machines.NewHost(filepath.Join(t.TempDir(), "machine"), "", nil)
	prefix := filepath.Join(h.Root(), "opt/cozy/python")
	installCommand(t, uv, "venv", "--no-config", "--no-project", "--python", "3.12", prefix)
	python := filepath.Join(prefix, "bin/python")
	installCommand(t, uv, "pip", "install", "--no-config", "--python", python, installWheel(t, "machine_base_framework", "2.14.0+cu130", ""))
	marker := filepath.Join(prefix, "operator-settings")
	writeInstallFile(t, marker, []byte("keep"), 0600)
	_, problem := h.Install(t.Context(), machines.Source{Host: bundledTestAgent(t, "dev"), RuntimeWheel: installWheel(t, "cozy_runtime", "0.18.87", "cozy-runtime-worker"), TensorFSWheel: installWheel(t, "tensorfs", "0.3.78", "tfs")}, uv)
	fatal(t, problem)
	assertInstallVersion(t, python, "machine-base-framework", "2.14.0+cu130")
	if raw, err := os.ReadFile(marker); err != nil || string(raw) != "keep" {
		t.Fatalf("machine settings changed: %q %v", raw, err)
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
