package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

// A run list that stops at its limit says how many matching runs it left out, and how to
// see them all; the count follows the filter.
func TestRunListSaysHowManyOlderRunsItLeftOut(t *testing.T) {
	o := hostOwner(t, "run-list-older")
	for index := 1; index <= 58; index++ {
		pkg := "proof/history"
		if index%10 == 0 {
			pkg = "proof/other"
		}
		id := fmt.Sprintf("req-older-%03d", index)
		_, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: "sha256:" + sixtyFour("1"),
			Package: pkg, Entrypoint: "run", Payload: []byte("{}")})
		fatal(t, problem)
	}
	defer publicationControlAPI(t, o)()

	listed := func(args ...string) (int, int) {
		t.Helper()
		code, out := runCozy(t, o.root, append([]string{"--json", "run", "list"}, args...)...)
		var document struct {
			Invocations []json.RawMessage `json:"invocations"`
			Omitted     int               `json:"omitted"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &document) != nil {
			t.Fatalf("run list %v [exit %d]\n%s", args, code, out)
		}
		return len(document.Invocations), document.Omitted
	}
	for _, c := range []struct {
		args          []string
		shown, hidden int
	}{{nil, 50, 8}, {[]string{"--package", "proof/history"}, 50, 3}, {[]string{"--limit", "10", "--package", "proof/other"}, 5, 0},
		{[]string{"--limit", "0"}, 58, 0}} {
		if shown, hidden := listed(c.args...); shown != c.shown || hidden != c.hidden {
			t.Fatalf("run list %v showed %d and left out %d, want %d and %d", c.args, shown, hidden, c.shown, c.hidden)
		}
	}
	if code, human := runCozy(t, o.root, "run", "list", "--no-watch"); code != 0 ||
		!strings.Contains(human, "8 more not shown. Use --limit 0 to show all.") {
		t.Fatalf("the human run list does not say what it left out [exit %d]\n%s", code, human)
	}
	if code, human := runCozy(t, o.root, "run", "list", "--no-watch", "--limit", "0"); code != 0 || strings.Contains(human, "not shown") {
		t.Fatalf("the whole history still claims hidden runs [exit %d]\n%s", code, human)
	}
}

// A search cut by --limit says how many matches it left out and which --limit shows them.
func TestPackageSearchSaysHowToSeeEveryMatch(t *testing.T) {
	var packages []hub.Resource
	for index := 1; index <= 30; index++ {
		packages = append(packages, hub.Resource{Org: "proof", Name: fmt.Sprintf("pkg-%02d", index), LatestRelease: "1.0.0"})
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/packages" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"packages": packages, "search": hub.ResourceSearch{Total: 30, Limit: 200}})
	}))
	defer server.Close()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\n"), 0o600))

	if code, out := runCozy(t, root, "package", "search", "--limit", "5"); code != 0 ||
		!strings.Contains(out, "25 more not shown. Use --limit 30 to show all.") || strings.Contains(out, "proof/pkg-06") {
		t.Fatalf("a limited search does not say how to see every match [exit %d]\n%s", code, out)
	}
	code, out := runCozy(t, root, "package", "search", "--limit", "30")
	if code != 0 || !strings.Contains(out, "proof/pkg-30") || strings.Contains(out, "not shown") {
		t.Fatalf("the --limit the search named does not show every match [exit %d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "--json", "package", "search", "--limit", "5")
	var document struct {
		Packages []json.RawMessage `json:"packages"`
		Omitted  int               `json:"omitted"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &document) != nil || len(document.Packages) != 5 || document.Omitted != 25 {
		t.Fatalf("the search document does not count what it left out [exit %d]\n%s", code, out)
	}
}
