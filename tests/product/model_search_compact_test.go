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
	if rows != 1 || !strings.Contains(compact, "…") || !strings.Contains(compact, "--full") {
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
		if len(models) != 1 {
			t.Fatalf("expected one row per model: %+v", models)
		}
		first := models[0].(map[string]any)
		if first["model"] != model.Ref() || first["family"] != model.Family || first["release"] != release || !reflect.DeepEqual(first["lanes"], []any{longLane}) {
			t.Fatalf("machine values were shortened: %+v", first)
		}
	}
	if !reflect.DeepEqual(documents[0], documents[1]) {
		t.Fatal("--full changed complete model-search JSON")
	}
}

func TestModelSearchSelectsAvailableReleaseAndInfoKeepsLaneSizes(t *testing.T) {
	model := hub.Resource{Org: "proof", Name: "model", Family: "minimax-h3"}
	empty := hub.Resource{Org: "proof", Name: "unreleased"}
	manifest := "sha256:" + strings.Repeat("a", 64)
	card := hub.ModelCard{Model: model, Releases: []hub.ModelReleaseSummary{
		{ReleaseSummary: hub.ReleaseSummary{Release: "withdrawn", Yanked: true}},
		{ReleaseSummary: hub.ReleaseSummary{Release: "withdrawn-by-date", YankedAt: "2026-09-08T00:00:00Z"}},
		{},
		{ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0-rc.2"}, Lanes: []hub.ModelLaneSummary{
			{Lane: "bf16", ManifestID: manifest, Bytes: 5 << 30, Components: []string{"dit", "vae"}, ComponentBytes: map[string]int64{"dit": 4 << 30, "vae": 1 << 30}},
			{Lane: "fp8", ManifestID: manifest, Bytes: 3 << 30},
		}},
		{ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0-rc.1"}, Lanes: []hub.ModelLaneSummary{{Lane: "older-lane", ManifestID: manifest, Bytes: 7 << 30}}},
	}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{"models": []hub.Resource{model, empty}, "search": hub.ResourceSearch{Total: 2, Limit: 20}})
		case "/v1/models/" + model.Ref():
			_ = json.NewEncoder(w).Encode(card)
		case "/v1/models/" + empty.Ref():
			_ = json.NewEncoder(w).Encode(hub.ModelCard{Model: empty})
		default:
			t.Errorf("unexpected catalog route: %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\n"), 0600))
	for _, args := range [][]string{
		{"model", "search", "--json"},
		{"model", "search", "--json", "--full"},
		{"model", "search", model.Ref(), "--json"},
		{"model", "search", "--limit", "1", "--json"},
	} {
		code, encoded := runCozy(t, root, args...)
		var result struct {
			Models []struct {
				Model   string   `json:"model"`
				Release *string  `json:"release"`
				Lanes   []string `json:"lanes"`
			} `json:"models"`
		}
		if code != 0 || json.Unmarshal([]byte(encoded), &result) != nil || len(result.Models) == 0 {
			t.Fatalf("search %v failed: %d %s", args, code, encoded)
		}
		want := 2
		if strings.Contains(strings.Join(args, " "), model.Ref()) || strings.Contains(strings.Join(args, " "), "--limit") {
			want = 1
		}
		first := result.Models[0]
		if len(result.Models) != want || first.Model != model.Ref() || first.Release == nil || *first.Release != "1.0.0-rc.2" || !reflect.DeepEqual(first.Lanes, []string{"bf16", "fp8"}) {
			t.Fatalf("search must select one available release per model: %s", encoded)
		}
		if want == 2 && (result.Models[1].Model != empty.Ref() || result.Models[1].Release != nil || len(result.Models[1].Lanes) != 0) {
			t.Fatalf("unreleased model was misrepresented: %s", encoded)
		}
	}
	code, human := runCozy(t, root, "model", "search")
	if code != 0 || !strings.Contains(human, "Next: cozy model info "+model.Ref()+"\n") || strings.Contains(human, "older-lane") {
		t.Fatalf("search must link to the repository history: %d %s", code, human)
	}
	code, info := runCozy(t, root, "model", "info", model.Ref())
	if code != 0 || !strings.Contains(info, "older-lane") || !strings.Contains(info, "5.0GiB") || !strings.Contains(info, manifest) {
		t.Fatalf("info lost release history, lane size or checkpoint identity: %d %s", code, info)
	}
	code, encoded := runCozy(t, root, "model", "info", model.Ref()+"@1.0.0-rc.2", "--json")
	var detail struct {
		Releases []struct {
			Release string `json:"release"`
			Lanes   []struct {
				Bytes          int64            `json:"bytes"`
				ComponentBytes map[string]int64 `json:"component_bytes"`
			} `json:"lanes"`
		} `json:"releases"`
	}
	if code != 0 || json.Unmarshal([]byte(encoded), &detail) != nil || len(detail.Releases) != 1 || detail.Releases[0].Release != "1.0.0-rc.2" || len(detail.Releases[0].Lanes) != 2 {
		t.Fatalf("selected info must contain exactly one release: %d %s", code, encoded)
	}
	if lane := detail.Releases[0].Lanes[0]; lane.Bytes != 5<<30 || lane.ComponentBytes["dit"] != 4<<30 || lane.ComponentBytes["vae"] != 1<<30 {
		t.Fatalf("info must preserve exact byte counts: %s", encoded)
	}
}
