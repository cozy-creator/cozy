package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
)

// `cozy package info` lists a published package's releases from the Hub's package card:
// newest first by version, each with its publication time and yanked state.
func TestPackageInfoListsReleases(t *testing.T) {
	card := hub.PackageCard{
		Package: hub.Resource{Org: "proof", Name: "releases", LatestRelease: "1.10.0"},
		Releases: []hub.ReleaseSummary{
			{Release: "1.2.0", CutAt: "2026-09-26T10:00:00.123Z"},
			{Release: "1.10.0", CutAt: "2026-09-27T10:00:00Z"},
			{Release: "1.9.0", CutAt: "2026-09-27T09:00:00Z", Yanked: true, YankedAt: "2026-09-27T09:30:00Z"},
		},
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/packages/proof/releases", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(card)
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\n"), 0600))

	code, out := runCozy(t, root, "--json", "package", "info", "proof/releases")
	var listed struct {
		Releases []struct {
			Release   string `json:"release"`
			Published string `json:"published"`
			Yanked    string `json:"yanked"`
		} `json:"releases"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &listed) != nil {
		t.Fatalf("package info [exit %d]\n%s", code, out)
	}
	want := [][3]string{
		{"1.10.0", "2026-09-27T10:00:00Z", "no"},
		{"1.9.0", "2026-09-27T09:00:00Z", "yes"},
		{"1.2.0", "2026-09-26T10:00:00Z", "no"},
	}
	if len(listed.Releases) != len(want) {
		t.Fatalf("package info listed %d releases, want %d\n%s", len(listed.Releases), len(want), out)
	}
	for i, row := range listed.Releases {
		if got := [3]string{row.Release, row.Published, row.Yanked}; got != want[i] {
			t.Fatalf("release %d = %v, want %v\n%s", i, got, want[i], out)
		}
	}
}
