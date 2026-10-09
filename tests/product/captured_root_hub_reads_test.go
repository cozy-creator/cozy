package producttest

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/machines"
)

// A captured root naming its Model's repository without a lane has the CLI read that Model once
// and the machine resolve its release lane by name once (th-245): the warm run reads nothing at
// any Hub, nor does a new machine lifetime. The model's owner changing it from this computer
// moves the catalog revision, so the next run reads it again, even when the machine was stopped then.
func TestACapturedRootReadsItsModelOncePerBindingRevision(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: this computer's machine runs the call")
	}
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "czk")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if !t.Failed() {
			_ = removeAllForce(root)
		}
	})
	provisionMachine(t, root)
	resolved := seedProbe(t, h, root, machines.Local)

	var mu sync.Mutex
	var seen []string
	count := func(hub string, handler http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			authority := r.URL.Path == "/v1/worker/rental/authorized-keys" || hubAccessCall(r.Method+" "+r.URL.Path)
			if !strings.HasPrefix(r.URL.Path, "/v1/rentals") && !authority {
				mu.Lock()
				seen = append(seen, hub+" "+r.Method+" "+r.URL.Path)
				mu.Unlock()
			}
			handler.ServeHTTP(w, r)
		})
	}
	h.worker.Config.Handler = count("machine", h.worker.Config.Handler)
	account, probe := h.server.Config.Handler, probeModel(t, resolved)
	h.server.Config.Handler = count("account", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case probe(w, r):
		default:
			account.ServeHTTP(w, r)
		}
	}))

	if code, out := runCozy(t, root, "package", "install", probeProject(t, "proof/probe@1.0.0/bf16"), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	run := func(key string) string {
		t.Helper()
		mu.Lock()
		seen = nil
		mu.Unlock()
		code, out := runCozy(t, root, "run", parityPackage+"/touch", "value=1", "model.source=proof/probe", "--await", "--json")
		mu.Lock()
		defer mu.Unlock()
		if code != 0 || !strings.Contains(out, `"value":2`) {
			t.Fatalf("%s run [exit %d, read %q]\n%s", key, code, seen, out)
		}
		t.Logf("%s run read: %q", key, seen)
		return strings.Join(seen, "\n")
	}
	model := func(calls string) bool {
		if strings.Contains(calls, "machine GET /v1/models/") {
			t.Fatalf("the machine read a model at its Hub: %q", calls)
		}
		return strings.Contains(calls, "account GET /v1/models/proof/probe")
	}
	if !model(run("cold")) {
		t.Fatal("the cold run did not read its Model")
	}
	if calls := run("warm"); calls != "" {
		t.Fatalf("the warm run read: %q", calls)
	}

	// The model's owner changes it from this computer: the machine is told, reads the model
	// once more, then nothing again.
	h.mux.HandleFunc("DELETE /v1/models/proof/probe/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"release":"1.0.0","revision":2,"yanked":true,"changed":true}`))
	})
	if code, out := runCozy(t, root, "model", "yank", "proof/probe", "--release", "1.0.0"); code != 0 || strings.Contains(out, "keeps what it read") || strings.Contains(out, "no machine was told") {
		t.Fatalf("yank did not reach the machine [exit %d]\n%s", code, out)
	}
	if !model(run("changed")) {
		t.Fatal("the run after its model changed did not read it again")
	}
	if calls := run("warm again"); calls != "" {
		t.Fatalf("the warm run after the change read: %q", calls)
	}

	if code, out := runCozy(t, root, "machine", "stop"); code != 0 {
		t.Fatalf("machine stop [exit %d]\n%s", code, out)
	}
	if calls := run("new lifetime"); calls != "" {
		t.Fatalf("a new machine lifetime's first run read: %q", calls)
	}

	if code, out := runCozy(t, root, "machine", "stop"); code != 0 {
		t.Fatalf("machine stop [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "model", "yank", "proof/probe", "--release", "1.0.0"); code != 0 {
		t.Fatalf("yank while the machine is stopped [exit %d]\n%s", code, out)
	}
	if !model(run("changed while stopped")) {
		t.Fatal("the run after a change made while its machine was stopped did not read its Model again")
	}
}
