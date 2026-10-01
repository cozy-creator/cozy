package producttest

import (
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
)

// A captured root naming its Model's repository without a lane reads that Model at its
// machine's Hub once: the warm run and a restarted Runtime's first run read nothing at any
// Hub. The model's owner changing it from this computer tells the machine, and a new machine
// lifetime, which a change may have missed, reads it again.
func TestACapturedRootReadsItsModelOnceAcrossRuntimeRestarts(t *testing.T) {
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
	python := filepath.Join(root, "machine", "root", "opt", "cozy", "python", "bin", "python")
	seeded, err := exec.Command(python, "-I", "-c", seedCheckpoint, filepath.Join(root, "tensorfs")).CombinedOutput()
	if err != nil {
		t.Fatalf("seeding the machine's store: %v\n%s", err, seeded)
	}
	resolved := map[string]any{"model": "proof/probe", "release": "1.0.0", "lane": "bf16"}
	must(t, json.Unmarshal(seeded, &resolved))

	var mu sync.Mutex
	var seen []string
	count := func(hub string, handler http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			authority := r.URL.Path == "/v1/worker/rental/authorized-keys" || r.URL.Path == "/v1/execution-access"
			if !strings.HasPrefix(r.URL.Path, "/v1/rentals") && !authority {
				mu.Lock()
				seen = append(seen, hub+" "+r.Method+" "+r.URL.Path)
				mu.Unlock()
			}
			handler.ServeHTTP(w, r)
		})
	}
	doors := h.worker.Config.Handler
	h.worker.Config.Handler = count("machine", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/models/proof/probe":
			_, _ = w.Write([]byte(`{"releases":[{"release":"1.0.0","lanes":[{"lane":"bf16","bytes":1}]}]}`))
		case r.URL.Path == "/v1/models/resolve" && r.URL.Query().Get("ref") == "proof/probe@1.0.0" && r.URL.Query().Get("lane") == "bf16":
			_ = json.NewEncoder(w).Encode(resolved)
		default:
			doors.ServeHTTP(w, r)
		}
	}))
	// The caller's account, which a root of unpublished code carries as its owner: read once.
	h.mux.HandleFunc("GET /v1/accounts/current", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"name":"proof"}`))
	})
	h.server.Config.Handler = count("account", h.server.Config.Handler)

	if code, out := runCozy(t, root, "package", "install", probeProject(t, "proof/probe@1.0.0/bf16"), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	run := func(key string) string {
		t.Helper()
		mu.Lock()
		seen = nil
		mu.Unlock()
		code, out := runCozy(t, root, "run", parityPackage+"/touch", "value=1", "model.source=proof/probe", "--await", "--json")
		if code != 0 || !strings.Contains(out, `"value":2`) {
			t.Fatalf("%s run [exit %d]\n%s", key, code, out)
		}
		mu.Lock()
		defer mu.Unlock()
		t.Logf("%s run read: %q", key, seen)
		return strings.Join(seen, "\n")
	}
	model := func(calls string) bool {
		return strings.Contains(calls, "machine GET /v1/models/proof/probe") && strings.Contains(calls, "machine GET /v1/models/resolve")
	}
	if !model(run("cold")) {
		t.Fatal("the cold run did not read its Model at the machine's Hub")
	}
	if calls := run("warm"); calls != "" {
		t.Fatalf("the warm run read: %q", calls)
	}

	layout, problem := home.Open(root)
	fatal(t, problem)
	var agent struct {
		PID int `json:"pid"`
	}
	raw, err := os.ReadFile(filepath.Join(layout.Machine, "agent.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &agent))
	runtime := runtimeChild(agent.PID)
	if runtime == 0 {
		t.Fatal("the machine runs no Runtime")
	}
	must(t, syscall.Kill(runtime, syscall.SIGKILL))
	landed(t, "the machine to restart its Runtime", func() bool {
		next := runtimeChild(agent.PID)
		return next != 0 && next != runtime
	})
	if calls := run("restarted"); calls != "" {
		t.Fatalf("a restarted Runtime's first run read: %q", calls)
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
	if !model(run("new lifetime")) {
		t.Fatal("a new machine lifetime's first run did not read its Model again")
	}
}
