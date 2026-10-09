package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// Default inventory follows the selected Hub; local captures remain visible.
// Global inventory is explicit and never contacts any Hub.
func TestPackageListDefaultsToSelectedHubAndLocalSources(t *testing.T) {
	root := t.TempDir()
	var reads atomic.Int32
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reads.Add(1)
		http.Error(w, "inventory must not contact a Hub", http.StatusInternalServerError)
	})
	a, b := httptest.NewServer(handler), httptest.NewServer(handler)
	defer a.Close()
	defer b.Close()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: a\nhubs:\n  a: "+a.URL+"\n  b: "+b.URL+"\n"), 0o600))
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	for id, row := range map[string]records.PackageInstall{
		"1111111111111111": {Package: "proof/hub-a", Hub: a.URL, SourceKind: "tensorhub"},
		"2222222222222222": {Package: "proof/hub-b", Hub: b.URL, SourceKind: "tensorhub"},
		"3333333333333333": {Package: "local/source", SourceKind: "local"},
	} {
		row.ID, row.Major, row.Version, row.Verified = id, 1, "1.0.0", true
		row.SourceRef, row.Dir = row.Package+"@1.0.0", layout.InstallDir(id)
		_, problem := store.Activate(row)
		fatal(t, problem)
	}
	store.Close()
	list := func(args ...string) map[string]string {
		t.Helper()
		code, out := runCozy(t, root, append([]string{"package", "list", "--json", "--full"}, args...)...)
		if code != 0 {
			t.Fatalf("list: %d %s", code, out)
		}
		var doc struct {
			Packages []struct{ Package, Hub, Scope string }
			Notes    []string
		}
		must(t, json.Unmarshal([]byte(out), &doc))
		if !strings.Contains(strings.Join(doc.Notes, " "), "Selected Hub for package references:") {
			t.Fatalf("list hides selected Hub: %s", out)
		}
		rows := map[string]string{}
		for _, row := range doc.Packages {
			rows[row.Package] = row.Hub + ":" + row.Scope
		}
		return rows
	}
	rows := list()
	if len(rows) != 2 || rows["proof/hub-a"] != "a:selected hub" || rows["proof/hub-b"] != "" || rows["local/source"] != ":local" {
		t.Fatalf("selected inventory scopes: %+v", rows)
	}
	rows = list("--tensorhub=b")
	if len(rows) != 2 || rows["proof/hub-b"] != "b:selected hub" || rows["local/source"] != ":local" {
		t.Fatalf("explicit Hub scope: %+v", rows)
	}
	rows = list("--all-hubs")
	if len(rows) != 3 || rows["proof/hub-b"] != "b:other hub" {
		t.Fatalf("explicit global inventory: %+v", rows)
	}
	if reads.Load() != 0 {
		t.Fatalf("inventory contacted Hub %d times", reads.Load())
	}
}

// A named removal cannot silently operate on another Hub's active or superseded
// installation. Explicit selection removes only that origin, without Hub traffic.
func TestPackageRemovePreservesOtherHubInstallations(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	const a, b = "http://127.0.0.1:1", "http://127.0.0.1:2"
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: a\nhubs:\n  a: "+a+"\n  b: "+b+"\n"), 0o600))
	prior := updateAllInstall(t, layout, store, 1, "proof/shared", "1.0.0", "tensorhub", a)
	active := updateAllInstall(t, layout, store, 2, "proof/shared", "1.0.1", "tensorhub", b)
	code, out := runCozy(t, root, "package", "remove", "proof/shared", "--json")
	if code == 0 || !strings.Contains(out, "package.other_hub") || !strings.Contains(out, "--tensorhub=b") {
		t.Fatalf("foreign removal did not refuse with explicit origin: %d %s", code, out)
	}
	_, got, problem := store.ActivePackage("proof/shared")
	fatal(t, problem)
	if got == nil || got.ID != active.ID {
		t.Fatal("foreign active install was unpinned")
	}
	if _, err := os.Stat(filepath.Join(active.Dir, "prior.py")); err != nil {
		t.Fatalf("foreign active files changed: %v", err)
	}
	code, out = runCozy(t, root, "package", "remove", "proof/shared", "--tensorhub=b", "--json")
	if code != 0 || !strings.Contains(out, `"changed":true`) {
		t.Fatalf("explicit Hub removal failed: %d %s", code, out)
	}
	if _, err := os.Stat(active.Dir); !os.IsNotExist(err) {
		t.Fatalf("selected Hub files survived removal: %v", err)
	}
	if _, err := os.Stat(filepath.Join(prior.Dir, "prior.py")); err != nil {
		t.Fatalf("another Hub's superseded files were reclaimed: %v", err)
	}
	retained, problem := store.Install(prior.ID)
	fatal(t, problem)
	if retained == nil {
		t.Fatal("another Hub's superseded record was removed")
	}
}

// A selected Hub's installed package must survive the default20-row presentation
// cap even when the home retains many earlier-alphabetic local diagnostics.
func TestPackageListPlacesSelectedHubBeforeLocalDiagnostics(t *testing.T) {
	root := t.TempDir()
	const current = "http://127.0.0.1:1"
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+current+"\n"), 0o600))
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	for i := range 24 {
		row := records.PackageInstall{ID: fmt.Sprintf("%016x", i+1), Package: fmt.Sprintf("local/diagnostic-%02d", i), Major: 1, Version: "1.0.0", SourceKind: "local", Verified: true}
		row.Dir = layout.InstallDir(row.ID)
		_, problem = store.Activate(row)
		fatal(t, problem)
	}
	row := records.PackageInstall{ID: "ffffffffffffffff", Package: "paul/minimax-h3", Major: 1, Version: "1.26.3", SourceKind: "tensorhub", Hub: current, Verified: true}
	row.Dir = layout.InstallDir(row.ID)
	_, problem = store.Activate(row)
	fatal(t, problem)
	other := row
	other.ID, other.Package, other.Hub = "eeeeeeeeeeeeeeee", "fidika/minimax-h3", "http://127.0.0.1:2"
	other.Dir = layout.InstallDir(other.ID)
	_, problem = store.Activate(other)
	fatal(t, problem)
	store.Close()
	code, out := runCozy(t, root, "package", "list", "--json")
	var doc struct {
		Packages []struct{ Package, Scope string }
		Omitted  int
	}
	if code != 0 || json.Unmarshal([]byte(out), &doc) != nil {
		t.Fatalf("list: %d %s", code, out)
	}
	if len(doc.Packages) != 20 || doc.Omitted != 5 || doc.Packages[0].Package != "paul/minimax-h3" || doc.Packages[0].Scope != "selected hub" || doc.Packages[1].Package != "local/diagnostic-00" {
		t.Fatalf("selected package hidden behind diagnostics: %s", out)
	}
	code, out = runCozy(t, root, "package", "list", "--all-hubs", "--json")
	if code != 0 || json.Unmarshal([]byte(out), &doc) != nil || len(doc.Packages) != 20 || doc.Omitted != 6 || doc.Packages[0].Package != row.Package || doc.Packages[1].Package != other.Package || doc.Packages[2].Package != "local/diagnostic-00" {
		t.Fatalf("explicit all-Hub inventory hid published packages: %d %s", code, out)
	}
}
