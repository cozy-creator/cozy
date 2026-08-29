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
		if r.Method != http.MethodGet || r.URL.Path != "/v1/packages" {
			writeHubError(w, http.StatusNotFound, "route.not_found", r.Method+" "+r.URL.Path)
			return
		}
		rows := resources
		if r.URL.Query().Get("q") == "missing" {
			rows = nil
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"packages": rows,
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

	code, compact := run("package", "search")
	if code != 0 || !strings.HasPrefix(compact, "packages[#3]:") ||
		strings.Contains(compact, "created") || strings.Contains(compact, "aggregates") ||
		strings.Contains(compact, "kind:") || strings.Contains(compact, "ok:") {
		t.Fatalf("default catalog output is not one scalar package list [exit %d]\n%s", code, compact)
	}
	toonDocument, err := toon.Decode([]byte(compact))
	if err != nil {
		t.Fatalf("default output is not TOON: %v\n%s", err, compact)
	}

	code, encoded := run("package", "search", "--json")
	var jsonDocument any
	if err := json.Unmarshal([]byte(encoded), &jsonDocument); code != 0 || err != nil {
		t.Fatalf("JSON catalog output [exit %d]: %v\n%s", code, err, encoded)
	}
	if !reflect.DeepEqual(toonDocument, jsonDocument) {
		t.Fatalf("TOON and JSON changed logical meaning\nTOON: %#v\nJSON: %#v", toonDocument, jsonDocument)
	}
	doc := jsonDocument.(map[string]any)
	if len(doc) != 1 || len(doc["packages"].([]any)) != 3 {
		t.Fatalf("catalog output carries renderer scaffolding: %#v", doc)
	}

	code, full := run("package", "search", "--full", "--json")
	var fullDocument map[string]any
	if err := json.Unmarshal([]byte(full), &fullDocument); code != 0 || err != nil {
		t.Fatalf("full catalog output [exit %d]: %v\n%s", code, err, full)
	}
	first := fullDocument["packages"].([]any)[0].(map[string]any)
	if len(first) != 4 || first["ref"] != "cozy/marco-polo-derived" || first["created"] == "" {
		t.Fatalf("--full did not expose the catalog diagnostics: %#v", first)
	}

	code, selected := run("package", "search", "--fields=ref,created", "--json")
	var selectedDocument map[string]any
	if err := json.Unmarshal([]byte(selected), &selectedDocument); code != 0 || err != nil {
		t.Fatalf("selected catalog output [exit %d]: %v\n%s", code, err, selected)
	}
	selectedRow := selectedDocument["packages"].([]any)[0].(map[string]any)
	if len(selectedRow) != 2 || selectedRow["ref"] == "" || selectedRow["created"] == "" {
		t.Fatalf("--fields did not select the exact requested facts: %#v", selectedRow)
	}

	code, limited := run("package", "search", "--limit=2", "--json")
	var limitedDocument map[string]any
	if err := json.Unmarshal([]byte(limited), &limitedDocument); code != 0 || err != nil ||
		len(limitedDocument["packages"].([]any)) != 2 || limitedDocument["omitted"] != float64(1) {
		t.Fatalf("limited catalog output lost its one useful omission fact [exit %d]: %v\n%s", code, err, limited)
	}

	code, empty := run("package", "search", "missing", "--json")
	var emptyDocument map[string]any
	if err := json.Unmarshal([]byte(empty), &emptyDocument); code != 0 || err != nil ||
		len(emptyDocument["packages"].([]any)) != 0 || len(emptyDocument) != 2 {
		t.Fatalf("empty search is not definitive and actionable [exit %d]: %v\n%s", code, err, empty)
	}

	result := runCozyEnv(env, "package", "search", "--fields=missing")
	if result.code != 2 || !strings.Contains(result.output, "output.field_unknown") ||
		!strings.Contains(result.output, "available fields: ref, created, org, name") {
		t.Fatalf("unknown output field did not refuse with its valid set [exit %d]\n%s", result.code, result.output)
	}
}

func TestNounGroupsAreFocusedHelp(t *testing.T) {
	for _, group := range []string{"package", "model", "invoke", "rental"} {
		code, out := runCozy(t, t.TempDir(), group)
		if code != 0 || !strings.Contains(out, "Usage: cozy "+group+" <command>") ||
			strings.Contains(out, "cli.usage") {
			t.Errorf("cozy %s is not focused help [exit %d]\n%s", group, code, out)
		}
	}
	code, out := runCozy(t, t.TempDir(), "package", "nope")
	if code != 2 || !strings.HasPrefix(out, "error:") ||
		strings.Contains(out, "kind: error") || strings.Contains(out, "ok: false") ||
		!strings.Contains(out, "cozy help package") {
		t.Fatalf("typed usage error carries renderer scaffolding [exit %d]\n%s", code, out)
	}
	code, stdout, stderr := runCozyStreams(t, t.TempDir(), "package", "nope")
	if code != 2 || stdout == "" || stderr != "" {
		t.Fatalf("structured error crossed streams [exit %d] stdout=%q stderr=%q", code, stdout, stderr)
	}
}
