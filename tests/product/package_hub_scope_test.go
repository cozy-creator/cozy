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

// Inventory spans every hub by default, published installs before local captures so a page
// of captures never hides them; --tensorhub narrows to that hub and local sources. Listing
// never contacts a Hub.
func TestPackageListShowsEveryHubPublishedFirst(t *testing.T) {
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
	installs := map[string]records.PackageInstall{
		"1111111111111111": {Package: "proof/hub-a", Hub: a.URL, SourceKind: "tensorhub"},
		"2222222222222222": {Package: "proof/hub-b", Hub: b.URL, SourceKind: "tensorhub"},
	}
	for i := range 25 { // more local captures than one page, all sorting before proof/
		installs[fmt.Sprintf("3333333333333%03d", i)] = records.PackageInstall{Package: fmt.Sprintf("local/capture-%02d", i), SourceKind: "local"}
	}
	for id, row := range installs {
		row.ID, row.Major, row.Version, row.Verified = id, 1, "1.0.0", true
		row.SourceRef, row.Dir = row.Package+"@1.0.0", layout.InstallDir(id)
		_, problem := store.Activate(row)
		fatal(t, problem)
	}
	store.Close()
	list := func(args ...string) []string {
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
		if !strings.Contains(strings.Join(doc.Notes, " "), "org/name references resolve at hub ") {
			t.Fatalf("list hides the hub references resolve at: %s", out)
		}
		var rows []string
		for _, row := range doc.Packages {
			rows = append(rows, row.Package+"="+row.Hub+":"+row.Scope)
		}
		return rows
	}
	rows := list()
	if len(rows) != 27 || rows[0] != "proof/hub-a=a:current hub" || rows[1] != "proof/hub-b=b:other hub" || rows[2] != "local/capture-00=:local" {
		t.Fatalf("every-hub inventory: %v", rows)
	}
	rows = list("--tensorhub=b")
	if len(rows) != 26 || rows[0] != "proof/hub-b=b:current hub" {
		t.Fatalf("explicit hub narrowing: %v", rows)
	}
	code, out := runCozy(t, root, "package", "list")
	if code != 0 || !strings.Contains(out, "proof/hub-a") || !strings.Contains(out, "proof/hub-b") || !strings.Contains(out, "more not shown") {
		t.Fatalf("one page of local captures hid published installs: %d %s", code, out)
	}
	if reads.Load() != 0 {
		t.Fatalf("inventory contacted Hub %d times", reads.Load())
	}
}

// Each hub's installation of one org/name is its own. --tensorhub=a removes only a's; a hub
// with none refuses, naming the hub that has one; no flag removes every hub's.
func TestPackageRemoveUsesTheInstallationsHub(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	const a, b = "http://127.0.0.1:1", "http://127.0.0.1:2"
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: a\nhubs:\n  a: "+a+"\n  b: "+b+"\n"), 0o600))
	fromA := updateAllInstall(t, layout, store, 1, "proof/shared", "1.0.0", "tensorhub", a)
	fromB := updateAllInstall(t, layout, store, 2, "proof/shared", "1.0.1", "tensorhub", b)
	if code, out := runCozy(t, root, "package", "remove", "proof/shared", "--tensorhub=a", "--json"); code != 0 || !strings.Contains(out, `"changed":true`) {
		t.Fatalf("removing hub a's installation failed: %d %s", code, out)
	}
	if _, err := os.Stat(fromA.Dir); !os.IsNotExist(err) {
		t.Fatalf("hub a's files survived its removal: %v", err)
	}
	if _, got, problem := store.ActivePackage(b, "proof/shared"); problem != nil || got == nil || got.ID != fromB.ID {
		t.Fatalf("removing hub a's installation touched hub b's: %+v %v", got, problem)
	}
	code, out := runCozy(t, root, "package", "remove", "proof/shared", "--tensorhub=a", "--json")
	if code == 0 || !strings.Contains(out, "package.other_hub") || !strings.Contains(out, "installed from hub b ("+b+")") {
		t.Fatalf("removal at a hub without the package did not name the hub that has it: %d %s", code, out)
	}
	if code, out := runCozy(t, root, "package", "remove", "proof/shared", "--json"); code != 0 || !strings.Contains(out, `"changed":true`) {
		t.Fatalf("removal by name without a flag failed: %d %s", code, out)
	}
	if _, err := os.Stat(fromB.Dir); !os.IsNotExist(err) {
		t.Fatalf("hub b's files survived removal by name: %v", err)
	}
}

// A package this computer only ran (on a rental, with nothing installed here) is still
// traced to the hub it ran from, with no Hub asked but the one searched.
func TestAMissingPackageNamesTheHubItRanFrom(t *testing.T) {
	root := t.TempDir()
	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":{"code":"package.not_found","message":"no package","remedy":"publish its first immutable release"}}`))
	}))
	defer local.Close()
	const prod = "http://127.0.0.1:1"
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: local\nhubs:\n  local: "+local.URL+"\n  prod: "+prod+"\n"), 0o600))
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	_, _, problem = store.Submit(records.Request{ID: "req-ran-on-prod", IdemKey: "ran-on-prod", BodyDigest: "sha256:" + strings.Repeat("ab", 32),
		Package: "proof/ran", Entrypoint: "generate", Payload: []byte("{}"), Hub: prod})
	fatal(t, problem)
	store.Close()
	code, out := runCozy(t, root, "package", "info", "proof/ran")
	if code == 0 || !strings.Contains(out, "Error: no package proof/ran on hub local ("+local.URL+")") ||
		!strings.Contains(out, "Try: proof/ran ran from hub prod ("+prod+")") ||
		!strings.Contains(out, "Next: cozy package info proof/ran --tensorhub=prod\n") || strings.Contains(out, "publish its first") {
		t.Fatalf("a package this computer ran elsewhere was not traced to its hub [exit %d]\n%s", code, out)
	}
}
