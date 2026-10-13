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
		principal := "account-first"
		switch r.Header.Get("Authorization") {
		case "Bearer account-only":
		case "Bearer second-account":
			principal = "account-second"
		default:
			t.Error("account credential did not stay on the Hub request")
		}
		var body struct {
			Leaf string `json:"delegate_certificate_der_b64url"`
			TTL  int64  `json:"ttl_seconds"`
		}
		if json.NewDecoder(r.Body).Decode(&body) != nil || body.Leaf != base64.RawURLEncoding.EncodeToString(launch.Leaf) || body.TTL <= 0 || body.TTL > 604800 {
			t.Error("execution access was not bounded and pinned to the agent leaf")
		}
		delegated := executionGrantToken(server.URL, principal, accesses.Load())
		_ = json.NewEncoder(w).Encode(map[string]any{"token": delegated, "expires_at": time.Now().Add(time.Hour).UTC().Truncate(time.Second),
			"environment": map[string]string{"TENSORHUB_ORIGIN": server.URL, "TENSORHUB_PUBLIC_ORIGIN": server.URL}})
	}))
	defer server.Close()
	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("account-only")}, "test")
	withAccess, problem := machine.Ensure(ctx, server.URL, client, true)
	fatal(t, problem)
	if withAccess.PID != launch.PID || withAccess.WorkerID != launch.WorkerID || withAccess.Reads != server.URL {
		t.Fatal("attaching Hub access moved the machine")
	}
	_, problem = machine.Ensure(ctx, server.URL, client, true)
	fatal(t, problem)
	if accesses.Load() != 1 || registrations.Load() != 0 {
		t.Fatalf("wanted one scoped access and no registration; got %d and %d", accesses.Load(), registrations.Load())
	}
	accessFile, err := os.ReadFile(filepath.Join(machine.Root(), "var/lib/cozy/machine/hub-access.json"))
	must(t, err)
	firstGrant := executionGrantToken(server.URL, "account-first", 1)
	if bytes.Contains(accessFile, []byte("account-only")) || !bytes.Contains(accessFile, []byte(firstGrant)) {
		t.Error("machine did not retain only delegated access")
	}
	for _, signedOut := range []*hub.Client{nil, hub.New(config.Config{HubURL: server.URL}, "signed-out")} {
		if _, problem := machine.Ensure(ctx, server.URL, signedOut, true); problem == nil || problem.ErrName() != "machine.execution_access_required" {
			t.Fatalf("signed-out access replayed a cached account grant: %v", problem)
		}
	}
	other := client.WithToken(secret.New("second-account"), "test")
	_, problem = machine.Ensure(ctx, server.URL, other, true)
	if problem == nil || problem.ErrName() != "machine.execution_access_cleanup_pending" {
		t.Fatalf("switching accounts did not preserve the existing machine principal: %v", problem)
	}
	accessFile, err = os.ReadFile(filepath.Join(machine.Root(), "var/lib/cozy/machine/hub-access.json"))
	must(t, err)
	if accesses.Load() != 2 || !bytes.Contains(accessFile, []byte(firstGrant)) {
		t.Fatal("switching accounts silently reused or replaced the preceding delegated grant")
	}
	_, problem = machine.Ensure(ctx, server.URL, client, true)
	fatal(t, problem)
	if accesses.Load() != 3 {
		t.Fatal("queued revocation replayed cached bytes instead of fresh same-principal authorization")
	}
	// Older caches named no credential. Refresh them under the current account even if
	// the leaf and expiry match, and allow a fresh token for the same delegated principal.
	cachePath := filepath.Join(dir, "execution-access.json")
	cached, err := os.ReadFile(cachePath)
	must(t, err)
	var legacy map[string]map[string]any
	must(t, json.Unmarshal(cached, &legacy))
	delete(legacy[server.URL], "credential_identity")
	cached, err = json.Marshal(legacy)
	must(t, err)
	must(t, os.WriteFile(cachePath, cached, 0600))
	_, problem = machine.Ensure(ctx, server.URL, client, true)
	fatal(t, problem)
	if accesses.Load() != 4 {
		t.Fatal("an unbound cache was reused without current account authorization")
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
	if strings.Contains(string(log), "account-only") || strings.Contains(string(log), firstGrant) || strings.Contains(string(log), "obsolete-local-worker-capability") {
		t.Error("machine log contains a credential")
	}
}

// The isolated Hub fixture supplies a syntactic delegated JWT. The real Hub verifies its
// authority; the agent only needs its stable issuer/account identity to guard replacement.
func executionGrantToken(issuer, principal string, serial int32) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"typ":"delegated-access+jwt"}`))
	body, _ := json.Marshal(map[string]any{"iss": issuer, "delegated_sub": principal, "permissions": []string{"cozy.execution-access"}, "jti": serial, "attributes": map[string]any{"execution_device_key_id": "fixture-device"}})
	return header + "." + base64.RawURLEncoding.EncodeToString(body) + "." + base64.RawURLEncoding.EncodeToString([]byte("fixture-signature"))
}
