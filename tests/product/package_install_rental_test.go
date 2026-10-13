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
	"github.com/cozy-creator/cozy/internal/calcifer"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

// Exercise the public grammar and authenticated Calcifer client against isolated peers. The
// command asks the Hub nothing: the machine reads the release at its hub (the newest when
// none is named) and answers which it installed. No worker, Python environment or weights.
func TestPackageInstallRental(t *testing.T) {
	for _, test := range []struct{ name, version, rental string }{
		{"latest-by-name", "", "kirukiru"},
		{"pinned-by-id", "1.2.3", "rental-proof"},
	} {
		t.Run(test.name, func(t *testing.T) {
			check := func(problem *exit.Error) {
				t.Helper()
				if problem != nil {
					t.Fatal(problem)
				}
			}
			var operations atomic.Int32
			hubPeer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				t.Errorf("a rental installation asked the Hub: %s %s", r.Method, r.URL)
				http.Error(w, "unexpected request", 500)
			}))
			defer hubPeer.Close()
			localPeer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == "GET" && r.URL.Path == "/" {
					return
				}
				if r.Method == "GET" && r.URL.Path == "/v1/capabilities" {
					_, _ = w.Write([]byte(`{"machine_v1":true,"operations":true}`))
					return
				}
				if r.Method != "POST" || r.URL.Path != "/v1/local/machines/rental-proof/operations" || r.Header.Get("Authorization") == "" {
					t.Errorf("unexpected Calcifer request: %s %s", r.Method, r.URL)
					http.Error(w, "unexpected request", 500)
					return
				}
				var op records.Operation
				if err := json.NewDecoder(r.Body).Decode(&op); err != nil {
					t.Error(err)
				}
				if op.Package != "paul/minimax-h3" || op.Release != test.version || op.Hub != hubPeer.URL || len(op.Models) != 0 || op.Kind() != "install" {
					t.Errorf("wrong operation or implicit model selection: %+v", op)
				}
				operations.Add(1)
				w.WriteHeader(http.StatusAccepted)
				_ = json.NewEncoder(w).Encode(api.Handle{Number: 41, RequestID: "op-proof", Status: "queued"})
			}))
			defer localPeer.Close()
			layout, problem := home.Open(t.TempDir())
			check(problem)
			store, problem := records.Open(layout.DB)
			check(problem)
			defer store.Close()
			check(store.RecordRental(records.Rental{ID: "rental-proof", MachineName: "kirukiru", State: "ready", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, Hub: hubPeer.URL}))
			held, problem := calcifer.Hold(layout, strings.TrimPrefix(localPeer.URL, "http://"), "")
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
			args := []string{"package", "install", "paul/minimax-h3", "--rental=" + test.rental, "--no-wait"}
			if test.version != "" {
				args = append(args, "--version="+test.version)
			}
			parsed, err := parser.Parse(args)
			if err != nil {
				t.Fatal(err)
			}
			err = parsed.Run(&cli.Runtime{Cfg: config.Config{Home: layout.Root, HubURL: hubPeer.URL}, Out: &out, Err: &diagnostic, Mode: output.Mode{JSON: true, Full: true}})
			if err != nil || operations.Load() != 1 || !strings.Contains(out.String(), `"number":41`) || !strings.Contains(out.String(), `"kind":"install"`) {
				t.Fatalf("rental install: %v; operations=%d; output=%s", err, operations.Load(), &out)
			}
			_, installed, problem := store.ActivePackage(hubPeer.URL, "paul/minimax-h3")
			check(problem)
			if installed != nil {
				t.Fatal("rental installation changed local package inventory")
			}
		})
	}
}
