package live

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/output"
)

func TestHumanOutputIsReadableAndMachineOutputStaysStructured(t *testing.T) {
	emit := func(document output.Document) string {
		t.Helper()
		var rendered bytes.Buffer
		if err := document.Emit(&rendered, output.Mode{Human: true}); err != nil {
			t.Fatal(err)
		}
		return rendered.String()
	}

	oneColumn := output.List{
		Name: "packages", Fields: []string{"ref"}, AllFields: []string{"ref"}, Total: 3,
		Rows:  []map[string]string{{"ref": "cozy/one"}, {"ref": "cozy/two"}},
		Notes: []string{"catalog results are capped"}, Next: []string{"cozy package install cozy/one"},
	}
	want := "cozy/one\ncozy/two\n1 more not shown. Use --full to show all.\n" +
		"Note: catalog results are capped\nNext: cozy package install cozy/one\n"
	if got := emit(oneColumn); got != want {
		t.Fatalf("one-column human list:\nwant %q\n got %q", want, got)
	}

	empty := output.List{
		Name: "models", Fields: []string{"model"}, AllFields: []string{"model"},
		Next: []string{"cozy model search"},
	}
	if got, want := emit(empty), "No models found.\nNext: cozy model search\n"; got != want {
		t.Fatalf("empty human list:\nwant %q\n got %q", want, got)
	}

	table := output.List{
		Name: "packages", Fields: []string{"package", "version"},
		AllFields: []string{"package", "version"},
		Rows: []map[string]string{
			{"package": "cozy/one", "version": "1.2.3"},
			{"package": "cozy/long", "version": "2.0.0"},
		},
		Aggregates: []output.Field{{K: "changed", V: true}},
	}
	want = "PACKAGE    VERSION\ncozy/one   1.2.3\ncozy/long  2.0.0\nchanged: true\n"
	if got := emit(table); got != want {
		t.Fatalf("human table:\nwant %q\n got %q", want, got)
	}

	record := output.Record{
		Fields: []output.Field{{K: "url", V: "http://127.0.0.1:2699/"}, {K: "changed", V: true}},
		Notes:  []string{"the daemon was already running"}, Next: []string{"cozy package search"},
	}
	want = "url:     http://127.0.0.1:2699/\nchanged: true\n" +
		"Note: the daemon was already running\nNext: cozy package search\n"
	if got := emit(record); got != want {
		t.Fatalf("human record:\nwant %q\n got %q", want, got)
	}

	var toonOut bytes.Buffer
	if err := oneColumn.Emit(&toonOut, output.Mode{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(toonOut.String(), "packages[#2]: cozy/one,cozy/two") ||
		strings.Contains(toonOut.String(), "No packages") {
		t.Fatalf("machine TOON changed shape: %q", toonOut.String())
	}
	var jsonOut bytes.Buffer
	if err := oneColumn.Emit(&jsonOut, output.Mode{JSON: true, Human: true, Color: true}); err != nil {
		t.Fatal(err)
	}
	var machine map[string]any
	if err := json.Unmarshal(jsonOut.Bytes(), &machine); err != nil || len(machine["packages"].([]any)) != 2 {
		t.Fatalf("JSON did not retain its exact structured document: %v\n%s", err, jsonOut.String())
	}
}

func TestHumanErrorIsConciseColoredAndKeepsMachineEnvelope(t *testing.T) {
	problem := output.NewError(output.Operational, "unavailable", "Tensorhub is unavailable.").
		WithRemedy("Try again later.").
		WithNext("cozy model search")
	var human bytes.Buffer
	if err := output.EmitError(&human, problem, output.Mode{Human: true, Color: true}); err != nil {
		t.Fatal(err)
	}
	want := "\x1b[1;31mError: Tensorhub is unavailable.\x1b[0m\n" +
		"Try: Try again later.\nNext: cozy model search\n"
	if human.String() != want {
		t.Fatalf("human error:\nwant %q\n got %q", want, human.String())
	}

	var toonOut bytes.Buffer
	if err := output.EmitError(&toonOut, problem, output.Mode{}); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"error:", "class: operational", "code: unavailable"} {
		if !strings.Contains(toonOut.String(), field) {
			t.Fatalf("machine TOON omitted %q: %q", field, toonOut.String())
		}
	}
	var jsonOut bytes.Buffer
	if err := output.EmitError(&jsonOut, problem, output.Mode{JSON: true, Human: true, Color: true}); err != nil {
		t.Fatal(err)
	}
	var machine struct {
		Error output.Error `json:"error"`
	}
	if err := json.Unmarshal(jsonOut.Bytes(), &machine); err != nil || machine.Error.Code != "unavailable" {
		t.Fatalf("JSON error lost its structured envelope: %v\n%s", err, jsonOut.String())
	}
}

func TestTensorhubUnavailableNamesTheAttemptedLocation(t *testing.T) {
	client := hub.New(config.Config{HubURL: "http://127.0.0.1:1"}, "live-output-test")
	ctx, cancel := hub.Context()
	defer cancel()
	_, _, problem := client.Models(ctx, "")
	if problem == nil {
		t.Fatal("closed Tensorhub address unexpectedly answered")
	}
	want := "Tensorhub at http://127.0.0.1:1 is unavailable. Try again later."
	if problem.Message != want || problem.Remedy != "Set TENSORHUB_URL to use a different Tensorhub." {
		t.Fatalf("unhelpful Tensorhub refusal: message=%q remedy=%q", problem.Message, problem.Remedy)
	}
}

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
	wantCompact := "cozy/marco-polo-derived\ncozy/marco-polo-cu130\ncozy/marco-polo-launch1\n"
	if code != 0 || compact != wantCompact ||
		strings.Contains(compact, "created") || strings.Contains(compact, "aggregates") ||
		strings.Contains(compact, "kind:") || strings.Contains(compact, "ok:") {
		t.Fatalf("default catalog output is not a readable package list [exit %d]\n%s", code, compact)
	}

	code, encoded := run("package", "search", "--json")
	var jsonDocument any
	if err := json.Unmarshal([]byte(encoded), &jsonDocument); code != 0 || err != nil {
		t.Fatalf("JSON catalog output [exit %d]: %v\n%s", code, err, encoded)
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
	if result.code != 2 || !strings.HasPrefix(result.output, "Error: unknown output field") ||
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
	if code != 2 || !strings.HasPrefix(out, "Error:") ||
		strings.Contains(out, "kind: error") || strings.Contains(out, "ok: false") ||
		!strings.Contains(out, "cozy help package") {
		t.Fatalf("typed usage error carries renderer scaffolding [exit %d]\n%s", code, out)
	}
	code, stdout, stderr := runCozyStreams(t, t.TempDir(), "package", "nope")
	if code != 2 || stdout == "" || stderr != "" {
		t.Fatalf("structured error crossed streams [exit %d] stdout=%q stderr=%q", code, stdout, stderr)
	}
}
