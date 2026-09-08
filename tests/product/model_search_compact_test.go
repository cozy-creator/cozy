package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
)

func TestModelSearchCompactColumnsPreserveFullData(t *testing.T) {
	model := hub.Resource{Org: "proof", Name: strings.TrimSuffix(strings.Repeat("model-", 8), "-"), Family: strings.Repeat("家族", 15)}
	release := "candidate-" + strings.Repeat("変換", 30)
	longLane := "fp8-" + strings.Repeat("量子化", 30)
	card := hub.ModelCard{Model: model, Releases: []hub.ModelReleaseSummary{
		{ReleaseSummary: hub.ReleaseSummary{Release: release}, Lanes: []hub.ModelLaneSummary{{Lane: longLane}}},
		{ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0"}, Lanes: []hub.ModelLaneSummary{{Lane: "bf16-full"}, {Lane: "bf16-adaln-pruned"}, {Lane: "fp8-adaln-pruned"}, {Lane: "mxfp8-adaln-pruned"}}},
	}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{"models": []hub.Resource{model}, "search": hub.ResourceSearch{Total: 1, Limit: 20}})
		case "/v1/models/" + model.Ref():
			_ = json.NewEncoder(w).Encode(card)
		default:
			t.Errorf("unexpected catalog route: %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\n"), 0600))
	code, compact := runCozy(t, root, "model", "search")
	if code != 0 || !utf8.ValidString(compact) {
		t.Fatalf("compact model search failed [%d]: %s", code, compact)
	}
	columns := regexp.MustCompile(` {2,}`)
	rows := 0
	for _, line := range strings.Split(compact, "\n") {
		if !strings.HasPrefix(line, "proof/") {
			continue
		}
		cells := columns.Split(line, -1)
		if len(cells) != 4 {
			t.Fatalf("expected four columns: %q", line)
		}
		for i, limit := range []int{28, 14, 24, 40} {
			if utf8.RuneCountInString(cells[i]) > limit {
				t.Errorf("column %d exceeds %d runes: %q", i, limit, cells[i])
			}
		}
		rows++
	}
	if rows != 2 || !strings.Contains(compact, "…") || !strings.Contains(compact, "--full") {
		t.Fatalf("compact search lost rows or the detail hint: %s", compact)
	}
	code, full := runCozy(t, root, "model", "search", "--full")
	if code != 0 || !strings.Contains(full, model.Ref()) || !strings.Contains(full, model.Family) || !strings.Contains(full, release) || !strings.Contains(full, longLane) {
		t.Fatalf("--full did not preserve complete columns: %d %s", code, full)
	}
	code, info := runCozy(t, root, "model", "info", model.Ref())
	if code != 0 || !strings.Contains(info, release) || !strings.Contains(info, longLane) {
		t.Fatalf("model info did not preserve complete detail: %d %s", code, info)
	}
	var documents []map[string]any
	for _, args := range [][]string{{"--json", "model", "search"}, {"--json", "model", "search", "--full"}} {
		code, encoded := runCozy(t, root, args...)
		var document map[string]any
		if code != 0 || json.Unmarshal([]byte(encoded), &document) != nil {
			t.Fatalf("invalid search JSON: %d %s", code, encoded)
		}
		documents = append(documents, document)
		models := document["models"].([]any)
		first := models[0].(map[string]any)
		if first["model"] != model.Ref() || first["family"] != model.Family || first["release"] != release || !reflect.DeepEqual(first["lanes"], []any{longLane}) {
			t.Fatalf("machine values were shortened: %+v", first)
		}
	}
	if !reflect.DeepEqual(documents[0], documents[1]) {
		t.Fatal("--full changed complete model-search JSON")
	}
}
