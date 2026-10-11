package producttest

import (
	"bytes"
	"encoding/json"
	"fmt"
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
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

// Exercise the public grammar and authenticated daemon client against isolated peers. The
// command asks the Hub nothing: the machine reads the release at its hub (the newest when
// none is named) and answers which it installed. No worker, Python environment or weights.
func TestPackageInstallRental(t *testing.T) {
	for _, test := range []struct {
		name, version, installed, rental string
		// older: a daemon from before package_verbs takes exact releases only, so the
		// command reads the newest at the hub itself.
		older bool
	}{
		{"latest-by-name", "", "2.0.0", "kirukiru", false},
		{"pinned-by-id", "1.2.3", "1.2.3", "rental-proof", false},
		{"older-daemon", "", "2.0.0", "kirukiru", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			check := func(problem *exit.Error) {
				t.Helper()
				if problem != nil {
					t.Fatal(problem)
				}
			}
			var preparations atomic.Int32
			hubPeer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if test.older && r.Method == "GET" && r.URL.Path == "/v1/packages/paul/minimax-h3" {
					_, _ = w.Write([]byte(`{"package":{"name":"minimax-h3"},"releases":[{"release":"1.2.3"},{"release":"2.0.0"}]}`))
					return
				}
				t.Errorf("a rental installation asked the Hub: %s %s", r.Method, r.URL)
				http.Error(w, "unexpected request", 500)
			}))
			defer hubPeer.Close()
			localPeer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == "GET" && r.URL.Path == "/" {
					return
				}
				if r.Method == "GET" && r.URL.Path == "/v1/capabilities" { // it carries v1 installs
					_, _ = fmt.Fprintf(w, `{"machine_v1":true,"package_verbs":%t}`, !test.older)
					return
				}
				if r.Method == "GET" && r.URL.Path == "/v1/local/rentals/rental-proof/installs/install-proof" { // the command follows it
					_ = json.NewEncoder(w).Encode(api.RentalInstallStatus{RentalInstall: records.RentalInstall{ID: "install-proof", RentalID: "rental-proof",
						State: "succeeded", Selection: records.RentalInstallSelection{Package: "paul/minimax-h3", Release: test.version},
						Result: json.RawMessage(`{"package":"paul/minimax-h3","release":"` + test.installed + `"}`)}})
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
				sent := test.version
				if test.older {
					sent = test.installed
				}
				if request.Package != "paul/minimax-h3" || request.Release != sent || request.Hub != hubPeer.URL || len(request.Models) != 0 {
					t.Errorf("wrong preparation or implicit model selection: %+v", request)
				}
				preparations.Add(1)
				_ = json.NewEncoder(w).Encode(api.RentalPackagePrepareResult{ID: "install-proof", RentalID: "rental-proof", Selection: request, State: "queued"})
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
			if err != nil || preparations.Load() != 1 || !strings.Contains(out.String(), `"status":"succeeded","target":"paul/minimax-h3@`+test.installed+`"`) {
				t.Fatalf("rental install: %v; preparations=%d; output=%s", err, preparations.Load(), &out)
			}
			_, installed, problem := store.ActivePackage(hubPeer.URL, "paul/minimax-h3")
			check(problem)
			if installed != nil {
				t.Fatal("rental installation changed local package inventory")
			}
		})
	}
}
