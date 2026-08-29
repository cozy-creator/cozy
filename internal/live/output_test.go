package live

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	toon "github.com/toon-format/toon-go"
)

func TestCatalogOutputIsTheDomainAnswer(t *testing.T) {
	resources := []map[string]string{
		{"org": "cozy", "name": "marco-polo-derived", "created_at": "2026-08-29T05:39:35Z"},
		{"org": "cozy", "name": "marco-polo-cu130", "created_at": "2026-08-29T04:41:59Z"},
		{"org": "cozy", "name": "marco-polo-launch1", "created_at": "2026-08-29T04:22:28Z"},
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/endpoints" {
			writeHubError(w, http.StatusNotFound, "route.not_found", r.Method+" "+r.URL.Path)
			return
		}
		rows := resources
		if r.URL.Query().Get("q") == "missing" {
			rows = nil
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"endpoints": rows,
			"search": map[string]any{
				"total": len(rows), "limit": 200, "capped": false, "q": r.URL.Query().Get("q"),
			},
		})
	}))
	defer server.Close()
	home := t.TempDir()
	env := childEnv(t, home, "TENSORHUB_URL="+server.URL)
	run := func(args ...string) (int, string) {
		result := runCozyEnv(env, args...)
		return result.code, result.output
	}

	code, compact := run("endpoint", "search")
	if code != 0 || !strings.HasPrefix(compact, "endpoints[#3]:") ||
		strings.Contains(compact, "created") || strings.Contains(compact, "aggregates") ||
		strings.Contains(compact, "kind:") || strings.Contains(compact, "ok:") {
		t.Fatalf("default catalog output is not one scalar endpoint list [exit %d]\n%s", code, compact)
	}
	toonDocument, err := toon.Decode([]byte(compact))
	if err != nil {
		t.Fatalf("default output is not TOON: %v\n%s", err, compact)
	}

	code, encoded := run("endpoint", "search", "--json")
	var jsonDocument any
	if err := json.Unmarshal([]byte(encoded), &jsonDocument); code != 0 || err != nil {
		t.Fatalf("JSON catalog output [exit %d]: %v\n%s", code, err, encoded)
	}
	if !reflect.DeepEqual(toonDocument, jsonDocument) {
		t.Fatalf("TOON and JSON changed logical meaning\nTOON: %#v\nJSON: %#v", toonDocument, jsonDocument)
	}
	doc := jsonDocument.(map[string]any)
	if len(doc) != 1 || len(doc["endpoints"].([]any)) != 3 {
		t.Fatalf("catalog output carries renderer scaffolding: %#v", doc)
	}

	code, full := run("endpoint", "search", "--full", "--json")
	var fullDocument map[string]any
	if err := json.Unmarshal([]byte(full), &fullDocument); code != 0 || err != nil {
		t.Fatalf("full catalog output [exit %d]: %v\n%s", code, err, full)
	}
	first := fullDocument["endpoints"].([]any)[0].(map[string]any)
	if len(first) != 4 || first["ref"] != "cozy/marco-polo-derived" || first["created"] == "" {
		t.Fatalf("--full did not expose the catalog diagnostics: %#v", first)
	}

	code, limited := run("endpoint", "search", "--limit=2", "--json")
	var limitedDocument map[string]any
	if err := json.Unmarshal([]byte(limited), &limitedDocument); code != 0 || err != nil ||
		len(limitedDocument["endpoints"].([]any)) != 2 || limitedDocument["omitted"] != float64(1) {
		t.Fatalf("limited catalog output lost its one useful omission fact [exit %d]: %v\n%s", code, err, limited)
	}

	code, empty := run("endpoint", "search", "missing", "--json")
	var emptyDocument map[string]any
	if err := json.Unmarshal([]byte(empty), &emptyDocument); code != 0 || err != nil ||
		len(emptyDocument["endpoints"].([]any)) != 0 || len(emptyDocument) != 2 {
		t.Fatalf("empty search is not definitive and actionable [exit %d]: %v\n%s", code, err, empty)
	}
}

func TestNounGroupsAreFocusedHelp(t *testing.T) {
	for _, group := range []string{"endpoint", "model", "invoke", "rental"} {
		code, out := runCozy(t, t.TempDir(), group)
		if code != 0 || !strings.Contains(out, "Usage: cozy "+group+" <command>") ||
			strings.Contains(out, "cli.usage") {
			t.Errorf("cozy %s is not focused help [exit %d]\n%s", group, code, out)
		}
	}
	code, out := runCozy(t, t.TempDir(), "endpoint", "nope")
	if code != 2 || !strings.HasPrefix(out, "error:") ||
		strings.Contains(out, "kind: error") || strings.Contains(out, "ok: false") ||
		!strings.Contains(out, "cozy help endpoint") {
		t.Fatalf("typed usage error carries renderer scaffolding [exit %d]\n%s", code, out)
	}
}
