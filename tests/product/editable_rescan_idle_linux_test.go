package producttest

import (
	"archive/zip"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// A records write that changes no editable install reads no editable source. The daemon
// watched the records file and, on every write, re-read every editable package's project and
// the wheels it depends on: gigabytes a minute on the owner's laptop.
func TestRecordsWritesReadNoEditableSource(t *testing.T) {
	root, sources := t.TempDir(), t.TempDir()
	wheel := filepath.Join(sources, "heavy_dep-0.1.0-py3-none-any.whl")
	file, err := os.Create(wheel)
	must(t, err)
	archive := zip.NewWriter(file)
	for i := range 3000 {
		entry, err := archive.Create(fmt.Sprintf("heavy_dep/module_%04d.py", i))
		must(t, err)
		_, err = entry.Write([]byte("VALUE = 1\n"))
		must(t, err)
	}
	metadata, err := archive.Create("heavy_dep-0.1.0.dist-info/METADATA")
	must(t, err)
	_, err = metadata.Write([]byte("Metadata-Version: 2.1\nName: heavy-dep\nVersion: 0.1.0\n"))
	must(t, err)
	must(t, archive.Close())
	must(t, file.Close())
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	for i := range 10 {
		name := fmt.Sprintf("editable%d", i)
		source := filepath.Join(sources, name)
		must(t, os.MkdirAll(source, 0o755))
		must(t, os.WriteFile(filepath.Join(source, "pyproject.toml"), []byte("[project]\nname = \""+name+"\"\nversion = \"0.1.0\"\n"+
			"dependencies = [\"heavy-dep\"]\n\n[tool.uv.sources]\nheavy-dep = { path = \"../"+filepath.Base(wheel)+"\" }\n"), 0o644))
		writer, problem := home.LockWriter(layout)
		fatal(t, problem)
		_, problem = store.Activate(records.PackageInstall{ID: "editable-" + name, Package: "local/" + name, Version: "0.1.0",
			SourceKind: "local", SourceRef: source, ProjectDir: source, Dir: layout.InstallDir("editable-" + name)})
		writer.Unlock()
		fatal(t, problem)
	}
	daemon := startDaemonProcess(t, root)
	log := filepath.Join(root, "daemon.log")
	for deadline := time.Now().Add(time.Minute); ; time.Sleep(100 * time.Millisecond) {
		if raw, _ := os.ReadFile(log); strings.Count(string(raw), ": watching "+sources) == 10 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the daemon did not watch the editable sources\n%s", tail(log))
		}
	}
	time.Sleep(3 * time.Second)
	before := readBytes(t, daemon.cmd.Process.Pid)
	for i := range 30 {
		fatal(t, store.AppendPackageEvent("local/editable0", "proof.write", map[string]any{"write": i}))
		time.Sleep(50 * time.Millisecond)
	}
	time.Sleep(2 * time.Second)
	if read := readBytes(t, daemon.cmd.Process.Pid) - before; read > 4<<20 {
		t.Fatalf("30 records writes made the daemon read %d bytes", read)
	} else {
		t.Logf("30 records writes made the daemon read %d bytes", read)
	}
}

// readBytes is the bytes a process has read through read syscalls.
func readBytes(t *testing.T, pid int) int64 {
	t.Helper()
	raw, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/io")
	must(t, err)
	for _, line := range strings.Split(string(raw), "\n") {
		if value, ok := strings.CutPrefix(line, "rchar: "); ok {
			n, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64)
			must(t, err)
			return n
		}
	}
	t.Fatal("no rchar in /proc/<pid>/io")
	return 0
}
