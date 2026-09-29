package producttest

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/secret"
)

// An owned server has no Hub lifecycle. Even without an installed Runtime it proves its
// identity and accepts scoped account access without surrendering its machine identity.
func TestIndependentAgentBootsOfflineAndRetainsScopedAccess(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host=<standalone cozy-machine>")
	}
	if machines.HostModule(*machineHostBinary) != machines.AgentModule {
		t.Fatal("machine fixture is not the independent agent")
	}
	dir, err := os.MkdirTemp("", "cozy-agent-")
	must(t, err)
	machine := machines.NewHost(dir, "", nil)
	t.Cleanup(func() {
		fatal(t, machine.Stop(context.Background()))
		if !t.Failed() {
			_ = os.RemoveAll(dir)
		}
	})
	binary := filepath.Join(machine.Root(), "usr/local/bin/cozy-machine")
	must(t, os.MkdirAll(filepath.Dir(binary), 0755))
	raw, err := os.ReadFile(*machineHostBinary)
	must(t, err)
	must(t, os.WriteFile(binary, raw, 0755))
	tfs, err := exec.LookPath("tfs")
	if err != nil {
		t.Skip("tfs initializes the isolated machine Store")
	}
	must(t, os.Symlink(tfs, filepath.Join(filepath.Dir(binary), "tfs")))
	must(t, os.WriteFile(filepath.Join(dir, "installed.json"), []byte(`{"host":{"name":"cozy-machine","module":"`+machines.AgentModule+`"},"host_pinned":true}`), 0600))
	// The previous registry entry is migration input only; its obsolete capability must
	// never leave this directory or reach the new process environment.
	must(t, os.WriteFile(filepath.Join(dir, "registration.json"), []byte(`{"hub":"https://old.invalid","id":"om-preserved","worker_token":"obsolete-local-worker-capability"}`), 0600))
	sentinel := filepath.Join(machine.Root(), "var/lib/tensorfs/retained-output")
	if output, err := exec.Command(tfs, "store", "init", filepath.Dir(sentinel)).CombinedOutput(); err != nil {
		t.Fatalf("initialize retained Store: %v: %s", err, output)
	}
	must(t, os.WriteFile(sentinel, []byte("completed output"), 0600))
	owner, problem := machine.Owner()
	fatal(t, problem)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	launch, problem := machine.Ensure(ctx, "", nil, true)
	fatal(t, problem)
	if launch.WorkerID != "om-preserved" || launch.Reads != "" {
		t.Fatal("offline adoption changed machine identity or required a Hub")
	}
	resolver := &machines.Resolver{Host: machine}
	connected, problem := resolver.Dial(ctx, machines.Local, "inspecting an offline machine")
	fatal(t, problem)
	if connected.HubID() != "" || !connected.Owned() {
		t.Fatal("owned machine authentication acquired a rental/registry identity")
	}
	_ = connected.Close()
	resolver.Forget(machines.Local)
	environment, err := os.ReadFile("/proc/" + strconv.Itoa(launch.PID) + "/environ")
	must(t, err)
	for _, forbidden := range []string{"COZY_WORKER_AUTH_TOKEN=", "TENSORHUB_ORIGIN=", "obsolete-local-worker-capability"} {
		if bytes.Contains(environment, []byte(forbidden)) {
			t.Error("legacy registration authority reached the independent process")
		}
	}
	if !bytes.Contains(environment, []byte("COZY_MACHINE_LIFETIME=persistent")) {
		t.Error("owned agent did not receive persistent lifetime")
	}
	var accesses atomic.Int32
	var registrations atomic.Int32
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/machines" {
			registrations.Add(1)
			http.Error(w, "registration forbidden", 500)
			return
		}
		if r.URL.Path != "/v1/execution-access" {
			http.NotFound(w, r)
			return
		}
		accesses.Add(1)
		if r.Header.Get("Authorization") != "Bearer account-only" {
			t.Error("account credential did not stay on the Hub request")
		}
		var body struct {
			Leaf string `json:"delegate_certificate_der_b64url"`
			TTL  int64  `json:"ttl_seconds"`
		}
		if json.NewDecoder(r.Body).Decode(&body) != nil || body.Leaf != base64.RawURLEncoding.EncodeToString(launch.Leaf) || body.TTL <= 0 || body.TTL > 604800 {
			t.Error("execution access was not bounded and pinned to the agent leaf")
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"token": "execution-only", "expires_at": time.Now().Add(time.Hour).UTC().Truncate(time.Second),
			"environment": map[string]string{"TENSORHUB_ORIGIN": server.URL, "TENSORHUB_PUBLIC_ORIGIN": server.URL}})
	}))
	defer server.Close()
	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("account-only")}, "test")
	withAccess, problem := machine.Ensure(ctx, server.URL, client, true)
	fatal(t, problem)
	if withAccess.PID != launch.PID || withAccess.WorkerID != launch.WorkerID || withAccess.Reads != server.URL {
		t.Fatal("attaching Hub access moved the machine")
	}
	_, problem = machine.Ensure(ctx, server.URL, nil, true)
	fatal(t, problem)
	if accesses.Load() != 1 || registrations.Load() != 0 {
		t.Fatalf("wanted one scoped access and no registration; got %d and %d", accesses.Load(), registrations.Load())
	}
	accessFile, err := os.ReadFile(filepath.Join(machine.Root(), "var/lib/cozy/machine/hub-access.json"))
	must(t, err)
	if bytes.Contains(accessFile, []byte("account-only")) || !bytes.Contains(accessFile, []byte("execution-only")) {
		t.Error("machine did not retain only delegated access")
	}
	fatal(t, machine.Stop(ctx))
	restarted := machines.NewHost(dir, "", nil)
	next, problem := restarted.Ensure(ctx, "", nil, true)
	fatal(t, problem)
	if next.WorkerID != launch.WorkerID || next.PID == launch.PID {
		t.Fatal("independent restart lost the stable local identity")
	}
	afterOwner, problem := restarted.Owner()
	fatal(t, problem)
	if owner.PublicKey() != afterOwner.PublicKey() {
		t.Error("restart changed the local owner key")
	}
	retained, err := os.ReadFile(sentinel)
	must(t, err)
	if string(retained) != "completed output" {
		t.Error("adoption or restart changed retained outputs")
	}
	cache, err := os.Stat(filepath.Join(dir, "execution-access.json"))
	must(t, err)
	if cache.Mode().Perm() != 0600 {
		t.Error("delegated access cache is not private")
	}
	log, err := os.ReadFile(filepath.Join(dir, "host.log"))
	must(t, err)
	if strings.Contains(string(log), "account-only") || strings.Contains(string(log), "execution-only") || strings.Contains(string(log), "obsolete-local-worker-capability") {
		t.Error("machine log contains a credential")
	}
}
