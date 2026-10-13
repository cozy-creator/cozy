package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// Audit reproduction, not a proposed desired-behavior regression test.
func TestAuditHistorySilentlyDropsLoggedOutHubs(t *testing.T) {
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: http://127.0.0.1:1\n"), 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	fatal(t, store.RecordRental(records.Rental{ID: "pr-audit-recorded", MachineName: "recorded", State: "released", AcceleratorCount: 1, AcceleratorModel: "RTX 5090", Hub: "http://127.0.0.1:2", RentedAt: "2026-10-01T00:00:00Z", HourlyRateUSDMicros: 4000000}))
	store.Close()
	code, out := runCozy(t, root, "rental", "list", "--all", "--json")
	t.Logf("exit=%d output=%s", code, out)
	var document map[string]any
	must(t, json.Unmarshal([]byte(out), &document))
	if code != 0 || strings.Contains(out, "pr-audit-recorded") || strings.Contains(out, "did not answer") {
		t.Fatalf("audit trigger changed: %d %s", code, out)
	}
	rows, ok := document["rental_history"].([]any)
	if !ok || len(rows) != 0 {
		t.Fatalf("expected observed empty-success bug: %s", out)
	}
}

// Both URL spellings name the same HTTP origin, but local provenance uses raw strings.
func TestAuditEquivalentHubSpellingChangesPackageScope(t *testing.T) {
	for _, c := range []struct{ name, selected, installed, scope string }{
		{"hostname-case", "http://LOCALHOST:8819", "http://localhost:8819", "other hub"},
		{"http-default-port", "http://localhost:80", "http://localhost", "other hub"},
		{"https-default-port", "https://example.test:443", "https://example.test", "other hub"},
		{"trailing-slash-control", "http://localhost:8819/", "http://localhost:8819", "current hub"},
	} {
		t.Run(c.name, func(t *testing.T) {
			root := t.TempDir()
			must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+c.selected+"\n"), 0600))
			layout, problem := home.Open(root)
			fatal(t, problem)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			row := records.PackageInstall{ID: "audit-same-origin", Package: "proof/shared", Major: 1, Version: "1.0.0", Verified: true,
				Hub: c.installed, SourceKind: "tensorhub", SourceRef: "proof/shared@1.0.0", Dir: layout.InstallDir("audit-same-origin")}
			_, problem = store.Activate(row)
			fatal(t, problem)
			store.Close()
			code, out := runCozy(t, root, "package", "list", "--json", "--full")
			t.Logf("exit=%d output=%s", code, out)
			if code != 0 || !strings.Contains(out, `"scope":"`+c.scope+`"`) {
				t.Fatalf("audit trigger changed: %d %s", code, out)
			}
		})
	}
}
