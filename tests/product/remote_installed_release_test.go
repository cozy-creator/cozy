package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// Real CLI submission must preserve the version chosen at install even when the
// catalog publishes a newer version. This catalog supplies metadata only and has
// no acquisition endpoint, so the proof cannot buy a machine or download weights.
func TestRentalRunPreservesInstalledPublishedRelease(t *testing.T) {
	for _, installed := range []bool{true, false} {
		name, want := "uninstalled uses latest", "2.10.0"
		if installed {
			name, want = "installed stays pinned", "2.9.0"
		}
		t.Run(name, func(t *testing.T) {
			iface := []byte(`{"format":"cozy.package.interface/1","application":"proof:app","entrypoints":[],"jobs":[{"name":"compute","models":[],"publishes":false,"request":{"fields":[]},"result":{"fields":[]},"weights_outputs":[]}]}`)
			parsed, problem := launch.DecodePackageInterface(iface)
			fatal(t, problem)
			mux := http.NewServeMux()
			mux.HandleFunc("GET /v1/packages/proof/releases", func(w http.ResponseWriter, _ *http.Request) {
				_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "releases"},
					Releases: []hub.ReleaseSummary{{Release: "2.9.0"}, {Release: "2.10.0"}}})
			})
			mux.HandleFunc("GET /v1/packages/proof/releases/releases/{release}", func(w http.ResponseWriter, r *http.Request) {
				var detail hub.PackageReleaseDetail
				detail.Release.Release = r.PathValue("release")
				detail.Release.PackageInterfaceDigest = parsed.Digest
				detail.Release.PackageInterfaceLength = int64(len(iface))
				detail.PackageInterface = iface
				detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25"}
				_ = json.NewEncoder(w).Encode(detail)
			})
			mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{"rentals":[]}`)) })
			mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`[]`)) })
			server := httptest.NewServer(mux)
			t.Cleanup(server.Close)
			root := t.TempDir()
			must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\ntensorhub_token: release-proof\nrentals:\n  max_hourly_spend_usd: 1\n  idle_release_s: 0\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			if installed {
				dir := filepath.Join(root, "installs", "inst-pinned")
				must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(dir)), 0700))
				must(t, os.WriteFile(launch.PackageInterfacePath(dir), iface, 0600))
				_, problem = store.Activate(records.PackageInstall{ID: "inst-pinned", Package: "proof/releases", Major: 2,
					Version: "2.9.0", SourceKind: "tensorhub", Dir: dir, PackageInterface: parsed.Digest,
					SourceDigest: "sha256:" + strings.Repeat("4", 64), Platform: "linux-x86"})
				fatal(t, problem)
			}
			store.Close()
			startDaemonProcess(t, root)
			code, out := runCozy(t, root, "run", "proof/releases/compute", "--rental-only", "--json", "--idempotency-key", "release-proof")
			if code != 0 {
				t.Fatalf("submission failed [exit %d]: %s", code, out)
			}
			store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			row, problem := store.RequestByIdempotencyKey("release-proof")
			fatal(t, problem)
			if row == nil || row.Release != want || row.InstallID != "" {
				t.Fatalf("remote request did not pin published release %s: %+v", want, row)
			}
		})
	}
}
