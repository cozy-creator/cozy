package producttest

import (
	"bytes"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestCaptureSDKIdentityIncludesResourcesAndSourcelessBytecode(t *testing.T) {
	prefix := t.TempDir()
	launcher := filepath.Join(prefix, "bin", "admitted-tool")
	must(t, os.MkdirAll(filepath.Dir(launcher), 0700))
	must(t, os.WriteFile(launcher, []byte("admitted SDK launcher"), 0700))
	must(t, os.WriteFile(filepath.Join(prefix, "pyvenv.cfg"), []byte("version_info = 3.12.12\n"), 0600))
	site := filepath.Join(prefix, "lib", "python3.12", "site-packages")
	dist := "cozy_runtime-0.18.2.dist-info/"
	files := map[string]string{
		"cozy_runtime/__init__.py":                          "# Runtime source\n",
		"cozy_runtime/data/RECORD":                          "implementation resource",
		"cozy_runtime/data/INSTALLER":                       "implementation resource",
		"cozy_runtime/data/REQUESTED":                       "implementation resource",
		"cozy_runtime/data/direct_url.json":                 "implementation resource",
		"cozy_runtime/compiled.pyc":                         "sourceless implementation bytes",
		"cozy_runtime/__pycache__/ghost.cpython-312.pyc":    "sourceless cached implementation bytes",
		"cozy_runtime/__pycache__/__init__.cpython-312.pyc": "derived compilation",
		dist + "METADATA":                                   "Name: cozy-runtime\nVersion: 0.18.2\n",
		dist + "direct_url.json":                            `{"url":"file:///tmp/build-one.whl"}`,
	}
	write := func() {
		t.Helper()
		var names []string
		for name := range files {
			names = append(names, name)
		}
		sort.Strings(names)
		var record bytes.Buffer
		writer := csv.NewWriter(&record)
		for _, name := range names {
			raw := []byte(files[name])
			path := filepath.Join(site, filepath.FromSlash(name))
			must(t, os.MkdirAll(filepath.Dir(path), 0700))
			must(t, os.WriteFile(path, raw, 0600))
			sum := sha256.Sum256(raw)
			hash := "sha256=" + base64.RawURLEncoding.EncodeToString(sum[:])
			if strings.Contains(name, "/__pycache__/") {
				hash = "" // installed bytecode may have no RECORD hash
			}
			must(t, writer.Write([]string{name, hash, fmt.Sprint(len(raw))}))
		}
		must(t, writer.Write([]string{dist + "RECORD", "", ""}))
		writer.Flush()
		must(t, writer.Error())
		must(t, os.WriteFile(filepath.Join(site, filepath.FromSlash(dist+"RECORD")), record.Bytes(), 0600))
	}
	write()
	prior, ok := install.CaptureSDKIdentity(launcher)
	if !ok {
		t.Fatal("complete SDK byte inventory refused")
	}
	files[dist+"direct_url.json"] = `{"url":"file:///tmp/another-build.whl"}`
	files["cozy_runtime/__pycache__/__init__.cpython-312.pyc"] += " regenerated"
	write()
	if next, ok := install.CaptureSDKIdentity(launcher); !ok || next != prior {
		t.Fatal("installer location or source-derived bytecode changed implementation identity")
	}
	for _, name := range []string{"cozy_runtime/data/RECORD", "cozy_runtime/data/INSTALLER", "cozy_runtime/data/REQUESTED", "cozy_runtime/data/direct_url.json", "cozy_runtime/compiled.pyc", "cozy_runtime/__pycache__/ghost.cpython-312.pyc"} {
		files[name] += " changed"
		write()
		next, ok := install.CaptureSDKIdentity(launcher)
		if !ok || next == prior {
			t.Fatalf("SDK implementation bytes %s were excluded from identity", name)
		}
		prior = next
	}
	must(t, os.WriteFile(filepath.Join(site, "cozy_runtime", "data", "RECORD"), []byte("tampered without updating the wheel receipt"), 0600))
	if _, ok := install.CaptureSDKIdentity(launcher); ok {
		t.Fatal("SDK accepted a RECORD declaration whose actual file bytes changed")
	}
}

func TestCapturePinsMigrateAndPreserveOrdinaryInstallOwners(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	digest := func(char string) string { return "sha256:" + strings.Repeat(char, 64) }
	makeInstall := func(id string) records.PackageInstall {
		t.Helper()
		inst := records.PackageInstall{ID: id, Dir: layout.InstallDir(id), Package: "local/capture-proof", Version: "1.0.0", SourceKind: "local", SourceDigest: digest("a")}
		must(t, os.MkdirAll(inst.Dir, 0700))
		must(t, os.WriteFile(filepath.Join(inst.Dir, "keep"), []byte(id), 0600))
		fatal(t, store.RecordInstall(inst))
		return inst
	}
	old := makeInstall("capture-old")
	store.Close()
	db, err := sql.Open("sqlite", layout.DB)
	must(t, err)
	_, err = db.Exec(`DROP TABLE rental_idle; DROP TABLE rental_runtime_updates; DROP TABLE capture_pins; PRAGMA user_version=40`)
	must(t, err)
	if ordinary, problem := records.Open(layout.DB); problem == nil || problem.ErrName() != "records_schema_upgrade_required" {
		if ordinary != nil {
			ordinary.Close()
		}
		t.Fatalf("ordinary reader changed schema 40: %v", problem)
	}
	var version int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	if version != 40 {
		t.Fatalf("ordinary reader migrated live daemon database to %d", version)
	}
	store, problem = records.OpenForDaemon(layout.DB, "")
	fatal(t, problem)
	defer store.Close()
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	if version != 43 {
		t.Fatalf("daemon did not migrate capture ownership: %d", version)
	}
	db.Close()
	held, problem := store.Install(old.ID)
	fatal(t, problem)
	if held == nil || held.SourceDigest != old.SourceDigest {
		t.Fatal("migration changed the existing install")
	}
	next := makeInstall("capture-next")
	caller := digest("b")
	first := records.CapturePin{Caller: caller, InputDigest: digest("c"), InstallID: old.ID, RevisionDigest: digest("d")}
	_, problem = store.ReplaceCapturePin(first)
	fatal(t, problem)
	_, problem = install.Reclaim(layout, store, old.ID)
	fatal(t, problem)
	if _, err := os.Stat(old.Dir); err != nil {
		t.Fatal("completed capture was reclaimed", err)
	}
	// A failed replacement cannot erase the previous completed installation.
	bad := first
	bad.InstallID = "missing"
	if _, problem := store.ReplaceCapturePin(bad); problem == nil {
		t.Fatal("missing install accepted as completed capture")
	}
	actual, problem := store.CapturePin(caller)
	fatal(t, problem)
	if actual == nil || *actual != first {
		t.Fatal("failed replacement poisoned the completed pin")
	}
	_, _, problem = store.Submit(records.Request{ID: "capture-live", IdemKey: "capture-live", BodyDigest: digest("e"), Package: old.Package,
		Entrypoint: "main", Payload: []byte(`{}`), InstallID: old.ID})
	fatal(t, problem)
	second := records.CapturePin{Caller: caller, InputDigest: digest("f"), InstallID: next.ID, RevisionDigest: digest("1")}
	prior, problem := store.ReplaceCapturePin(second)
	fatal(t, problem)
	if prior != old.ID {
		t.Fatal("capture replacement lost prior owner")
	}
	_, problem = install.Reclaim(layout, store, old.ID)
	fatal(t, problem)
	if _, err := os.Stat(old.Dir); err != nil {
		t.Fatal("capture replacement deleted an active request's install", err)
	}
	used, problem := store.LocalPackageInUse(second.RevisionDigest, next.Package, next.Version, next.SourceDigest)
	fatal(t, problem)
	if !used {
		t.Fatal("completed capture failed to retain its revision")
	}
	_, problem = store.CancelQueuedRequest("capture-live", map[string]any{"status": "CANCELED"})
	fatal(t, problem)
	_, problem = install.Reclaim(layout, store, old.ID)
	fatal(t, problem)
	if _, err := os.Stat(old.Dir); !os.IsNotExist(err) {
		t.Fatal("superseded unowned capture was not reclaimed", err)
	}
	actual, problem = store.CapturePin(caller)
	fatal(t, problem)
	if actual == nil || *actual != second {
		t.Fatal("old request cleanup changed the current capture")
	}
}

func TestCompletedCaptureReusesUnchangedScriptAndMissesEditedDependency(t *testing.T) {
	integration(t)
	if *privateScriptRuntimeWheel == "" {
		t.Skip("requires the exact candidate Runtime wheel")
	}
	root, err := os.MkdirTemp("", "cozy-capture-reuse-")
	must(t, err)
	t.Cleanup(func() {
		path := ""
		for _, value := range childEnv(t, root) {
			if strings.HasPrefix(value, "PATH=") {
				path = strings.TrimPrefix(value, "PATH=")
			}
		}
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("capture reuse evidence retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	project := t.TempDir()
	library := filepath.Join(project, "arithmetic")
	must(t, os.MkdirAll(library, 0700))
	metadata := `[project]
name="capture-arithmetic"
version="1.0.0"
requires-python=">=3.12,<3.13"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["arithmetic.py"]
`
	metadataPath := filepath.Join(library, "pyproject.toml")
	module := filepath.Join(library, "arithmetic.py")
	must(t, os.WriteFile(metadataPath, []byte(metadata), 0600))
	must(t, os.WriteFile(module, []byte("def compute():\n    return 11\n"), 0600))
	script := filepath.Join(project, "run.py")
	resultPath := filepath.Join(root, "value.txt")
	source := fmt.Sprintf(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime>=%s","capture-arithmetic>=1.0.0"]
# [tool.uv.sources]
# cozy-runtime={path=%q}
# capture-arithmetic={path="./arithmetic",editable=true}
# ///
from pathlib import Path
from arithmetic import compute

def main():
    Path(%q).write_text(str(compute()))
`, runtimeFixtureVersion(t, *privateScriptRuntimeWheel), *privateScriptRuntimeWheel, resultPath)
	must(t, os.WriteFile(script, []byte(source), 0600))
	run := func(label, want string) time.Duration {
		t.Helper()
		start := time.Now()
		code, stdout, stderr := runCozyStreams(t, root, "run", script, "--await", "--json")
		took := time.Since(start)
		if code != 0 {
			t.Fatalf("%s failed [%d]: %s %s", label, code, stdout, stderr)
		}
		value, err := os.ReadFile(resultPath)
		must(t, err)
		if string(value) != want {
			t.Fatalf("%s used stale dependency bytes: %q, want %s", label, value, want)
		}
		return took
	}
	cold := run("cold", "11")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	caller, problem := install.CaptureCaller(script)
	fatal(t, problem)
	first, problem := store.CapturePin(caller)
	fatal(t, problem)
	if first == nil {
		t.Fatal("completed ordinary script did not retain a qualified capture")
	}
	warm := run("unchanged", "11")
	second, problem := store.CapturePin(caller)
	fatal(t, problem)
	if second == nil || *second != *first {
		t.Fatal("unchanged ordinary invocation rebuilt its captured installation")
	}
	before, problem := store.RequestByReference("1")
	fatal(t, problem)
	after, problem := store.RequestByReference("2")
	fatal(t, problem)
	if before == nil || after == nil || before.ID == after.ID || before.LocalPackageDigest != after.LocalPackageDigest {
		t.Fatal("capture reuse changed request identity or immutable revision")
	}
	must(t, os.WriteFile(module, []byte("def compute():\n    return 22\n"), 0600))
	edited := run("same-version dependency edit", "22")
	third, problem := store.CapturePin(caller)
	fatal(t, problem)
	if third == nil || third.InputDigest == first.InputDigest || third.InstallID == first.InstallID || third.RevisionDigest == first.RevisionDigest {
		t.Fatal("same-version dependency edit reused the old capture")
	}
	must(t, os.WriteFile(metadataPath, []byte(strings.Replace(metadata, "hatchling.build", "absent_capture_backend", 1)), 0600))
	if code, stdout, stderr := runCozyStreams(t, root, "run", script, "--await", "--json"); code == 0 {
		t.Fatalf("broken build backend unexpectedly captured: %s %s", stdout, stderr)
	}
	held, problem := store.CapturePin(caller)
	fatal(t, problem)
	if held == nil || *held != *third {
		t.Fatal("failed capture replaced the last completed pin")
	}
	must(t, os.WriteFile(metadataPath, []byte(metadata), 0600))
	restored := run("restored completed input", "22")
	held, problem = store.CapturePin(caller)
	fatal(t, problem)
	if held == nil || *held != *third {
		t.Fatal("failed capture damaged the previous reusable installation")
	}
	t.Logf("ordinary CLI wall times: cold=%s unchanged=%s edited-dependency=%s restored=%s; install %s reused as a new request", cold, warm, edited, restored, first.InstallID)
}
