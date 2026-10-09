package producttest

import (
	"encoding/json"
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

// Inventory remains global and offline, but cannot pass off another Hub's install
// as an invocation against the current Hub. An explicit override changes the scope.
func TestPackageListExplainsSelectedAndOtherHubInstalls(t *testing.T) {
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
	if len(rows) != 3 || rows["proof/hub-a"] != "a:selected hub" || rows["proof/hub-b"] != "b:other hub" || rows["local/source"] != ":local" {
		t.Fatalf("global inventory scopes: %+v", rows)
	}
	rows = list("--tensorhub=b")
	if len(rows) != 1 || rows["proof/hub-b"] != "b:selected hub" {
		t.Fatalf("explicit Hub scope: %+v", rows)
	}
	if reads.Load() != 0 {
		t.Fatalf("inventory contacted Hub %d times", reads.Load())
	}
}
