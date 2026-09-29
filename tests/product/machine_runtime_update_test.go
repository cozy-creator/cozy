package producttest

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/machines"
)

// A machine updates its own Runtime when `cozy rental update` asks it to: no SSH, no image
// script. Its daemon waits for a guarded restart, installs the pair, relaunches and checks the
// new Runtime answers. From Runtime 0.18.77 to 0.18.78 with local wheels; a broken build rolls
// back to 0.18.78; published versions by name. The machine keeps its pair as an image does,
// in var/lib/cozy/dev/current.
func TestTheMachineUpdatesItsOwnRuntime(t *testing.T) {
	from, to := publishedWheel(t, hostruntime.Distribution, "0.18.77"), publishedWheel(t, hostruntime.Distribution, "0.18.78")
	tensorfs := publishedWheel(t, "tensorfs", "0.3.78")
	h, root, _, _ := parityMachinesOn(t, machines.Source{Host: *machineHostBinary, RuntimeWheel: from, TensorFSWheel: tensorfs})
	provider := h.provider
	// The image layout: the kept pair under var/lib/cozy/dev, which opt/cozy/wheels follows.
	dev := filepath.Join(provider, "var/lib/cozy/dev")
	must(t, os.MkdirAll(dev, 0o755))
	must(t, os.Rename(filepath.Join(provider, "opt/cozy/wheels"), filepath.Join(dev, "initial")))
	must(t, os.Symlink("initial", filepath.Join(dev, "current")))
	must(t, os.Symlink("../../var/lib/cozy/dev/current", filepath.Join(provider, "opt/cozy/wheels")))
	installed := func() string {
		out, err := exec.Command(filepath.Join(provider, "opt/cozy/python/bin/python"), "-I", "-c",
			"import importlib.metadata as m; print(m.version('cozy-runtime'), m.version('tensorfs'))").Output()
		must(t, err)
		return strings.TrimSpace(string(out))
	}
	kept := func() string {
		entries, err := os.ReadDir(filepath.Join(dev, "current"))
		must(t, err)
		var names []string
		for _, entry := range entries {
			names = append(names, entry.Name())
		}
		return strings.Join(names, " ")
	}
	update := func(args ...string) (int, string) {
		t.Helper()
		return runCozy(t, root, append([]string{"rental", "update", "tessa", "--json"}, args...)...)
	}

	code, out := update("--runtime-wheel", to, "--tensorfs-wheel", tensorfs)
	if code != 0 || installed() != "0.18.78 0.3.78" || !strings.Contains(kept(), filepath.Base(to)) {
		t.Fatalf("the machine did not update to 0.18.78 [exit %d]: runs %s, keeps %s\n%s", code, installed(), kept(), out)
	}
	// The updated machine still serves its owner.
	if code, out := runCozy(t, root, "rental", "list", "--json"); code != 0 || !strings.Contains(out, "tessa") {
		t.Fatalf("the updated rental is not listed [exit %d]: %s", code, out)
	}

	code, out = update("--runtime-wheel", brokenBuild(t, to), "--tensorfs-wheel", tensorfs)
	if code == 0 || !strings.Contains(out, "exited before it was ready") || installed() != "0.18.78 0.3.78" || !strings.Contains(kept(), filepath.Base(to)) {
		t.Fatalf("a broken Runtime did not roll back to 0.18.78 [exit %d]: runs %s, keeps %s\n%s", code, installed(), kept(), out)
	}

	code, out = update("--runtime-version", "0.18.77", "--tensorfs-version", "0.3.78")
	if code != 0 || installed() != "0.18.77 0.3.78" || !strings.Contains(kept(), filepath.Base(from)) {
		t.Fatalf("the machine did not install the named versions [exit %d]: runs %s, keeps %s\n%s", code, installed(), kept(), out)
	}
}

// brokenBuild is a local build of wheel whose Runtime worker exits the moment it starts.
func brokenBuild(t *testing.T, wheel string) string {
	t.Helper()
	relabeled := localBuild(t, wheel, "broken")
	archive, err := zip.OpenReader(relabeled)
	must(t, err)
	defer archive.Close()
	path := filepath.Join(t.TempDir(), filepath.Base(relabeled))
	file, err := os.Create(path)
	must(t, err)
	out := zip.NewWriter(file)
	var record strings.Builder
	var recordName string
	for _, entry := range archive.File {
		if strings.HasSuffix(entry.Name, ".dist-info/RECORD") {
			recordName = entry.Name
			continue
		}
		reader, err := entry.Open()
		must(t, err)
		body, err := io.ReadAll(reader)
		reader.Close()
		must(t, err)
		if entry.Name == "cozy_runtime/cli/runtime_worker.py" {
			body = []byte("raise SystemExit('a broken Runtime build')\n")
		}
		header := entry.FileHeader
		header.Extra = nil // the writer adds its own; a copy would repeat it
		w, err := out.CreateHeader(&header)
		must(t, err)
		_, err = w.Write(body)
		must(t, err)
		sum := sha256.Sum256(body)
		fmt.Fprintf(&record, "%s,sha256=%s,%d\n", entry.Name, base64.RawURLEncoding.EncodeToString(sum[:]), len(body))
	}
	w, err := out.CreateHeader(&zip.FileHeader{Name: recordName, Method: zip.Deflate})
	must(t, err)
	_, err = w.Write([]byte(record.String() + recordName + ",,\n"))
	must(t, err)
	must(t, out.Close())
	must(t, file.Close())
	return path
}
