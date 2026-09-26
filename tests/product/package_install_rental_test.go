package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/alecthomas/kong"
	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

// Exercise the public grammar, real release resolver and authenticated daemon
// client against isolated peers. No worker, Python environment or weights exist.
func TestPackageInstallRental(t *testing.T) {
	for _, test := range []struct {
		name, version, selected, rental string
		refused                         bool
	}{
		{"latest-by-name", "", "2.0.0", "kirukiru", false},
		{"pinned-by-id", "1.2.3", "1.2.3", "rental-proof", false},
		{"changed-pin", "1.2.3", "2.0.0", "kirukiru", true},
		{"missing-release", "", "", "kirukiru", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			check := func(problem *exit.Error) {
				t.Helper()
				if problem != nil {
					t.Fatal(problem)
				}
			}
			var resolutions, preparations atomic.Int32
			hubPeer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != "POST" || r.URL.Path != "/v1/packages/paul/minimax-h3/download" || r.URL.Query().Get("release") != test.version {
					t.Errorf("unexpected Hub request (including model or byte download): %s %s", r.Method, r.URL)
					http.Error(w, "unexpected request", 500)
					return
				}
				resolutions.Add(1)
				_ = json.NewEncoder(w).Encode(hub.PackageDownloadPlan{Release: test.selected,
					Downloads: []hub.PackageInstallDownload{{Kind: "project_wheel", Path: "unused.whl"}}})
			}))
			defer hubPeer.Close()
			localPeer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == "GET" && r.URL.Path == "/" {
					return
				}
				if r.Method != "POST" || r.URL.Path != "/v1/local/rentals/rental-proof/prepare" || r.Header.Get("Authorization") == "" {
					t.Errorf("unexpected daemon request: %s %s", r.Method, r.URL)
					http.Error(w, "unexpected request", 500)
					return
				}
				var request api.RentalPackagePrepareRequest
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
				}
				if request.Package != "paul/minimax-h3" || request.Release != test.selected || len(request.Models) != 0 {
					t.Errorf("wrong preparation or implicit model selection: %+v", request)
				}
				preparations.Add(1)
				_ = json.NewEncoder(w).Encode(api.RentalPackagePrepareResult{Rental: "rental-proof", Package: request.Package, Release: request.Release, Status: "prepared"})
			}))
			defer localPeer.Close()
			layout, problem := home.Open(t.TempDir())
			check(problem)
			store, problem := records.Open(layout.DB)
			check(problem)
			defer store.Close()
			check(store.RecordRental(records.Rental{ID: "rental-proof", MachineName: "kirukiru", State: "ready", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, Hub: hubPeer.URL}))
			held, problem := daemon.Hold(layout, strings.TrimPrefix(localPeer.URL, "http://"), "")
			check(problem)
			defer held.Release()
			_, problem = api.Mint(layout)
			check(problem)
			var out, diagnostic bytes.Buffer
			var grammar cli.CLI
			parser, err := kong.New(&grammar, kong.Writers(&out, &diagnostic))
			if err != nil {
				t.Fatal(err)
			}
			args := []string{"package", "install", "paul/minimax-h3", "--rental=" + test.rental}
			if test.version != "" {
				args = append(args, "--version="+test.version)
			}
			parsed, err := parser.Parse(args)
			if err != nil {
				t.Fatal(err)
			}
			err = parsed.Run(&cli.Runtime{Cfg: config.Config{Home: layout.Root, HubURL: hubPeer.URL}, Out: &out, Err: &diagnostic, Mode: output.Mode{JSON: true, Full: true}})
			if test.refused {
				if err == nil || preparations.Load() != 0 {
					t.Fatalf("bad release reached preparation: %v, %d", err, preparations.Load())
				}
			} else if err != nil || preparations.Load() != 1 || !strings.Contains(out.String(), `"release":"`+test.selected+`"`) {
				t.Fatalf("rental install: %v; preparations=%d; output=%s", err, preparations.Load(), &out)
			}
			if resolutions.Load() != 1 {
				t.Fatalf("resolutions=%d", resolutions.Load())
			}
			_, installed, problem := store.ActivePackage("paul/minimax-h3")
			check(problem)
			if installed != nil {
				t.Fatal("rental installation changed local package inventory")
			}
		})
	}
}

func TestPackageInstallRentalRejectsEditableSource(t *testing.T) {
	for _, source := range []string{".", "./project", "paul/minimax-h3"} {
		t.Run(source, func(t *testing.T) {
			var out bytes.Buffer
			err := (&cli.PackageInstallCmd{Ref: source, Editable: true, Rental: "kirukiru"}).Run(&cli.Runtime{Out: &out, Err: &out})
			if err == nil || !strings.Contains(err.Error(), "requires a published org/name") {
				t.Fatalf("editable source was not refused before contacting a service: %v", err)
			}
		})
	}
}
