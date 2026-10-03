package producttest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestScopedLogoutPreservesOtherHubAndNeverStartsStoppedAgent(t *testing.T) {
	server := httptest.NewServer(http.NotFoundHandler())
	origin := server.URL
	server.Close()
	root := t.TempDir()
	other := "https://other.example.test"
	agentOrigin := "https://agent-origin.example.test"
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+origin+"\n"), 0600))
	firstKey := writeMachineCredential(t, root, origin)
	otherKey := writeMachineCredential(t, root, other)
	path := filepath.Join(root, "machine", "execution-access.json")
	must(t, os.MkdirAll(filepath.Dir(path), 0700))
	cache := map[string]any{origin: map[string]any{"origin": agentOrigin, "token": "first-scoped-secret"}, other: map[string]any{"origin": other, "token": "other-scoped-secret", "future_binding": "keep"}}
	raw, err := json.Marshal(cache)
	must(t, err)
	must(t, os.WriteFile(path, raw, 0600))
	witness := filepath.Join(root, "agent-started")
	binary := filepath.Join(root, "machine/root/usr/local/bin/cozy-machine")
	must(t, os.MkdirAll(filepath.Dir(binary), 0755))
	must(t, os.WriteFile(binary, []byte("#!/bin/sh\ntouch "+witness+"\n"), 0755))
	code, out := runCozy(t, root, "auth", "logout", "--json")
	if code != 0 || !strings.Contains(out, "logged out") || !strings.Contains(out, "queued") || strings.Contains(out, "scoped-secret") {
		t.Fatal(code, out)
	}
	if _, err := os.Stat(firstKey); !os.IsNotExist(err) {
		t.Fatal("selected device key remains", err)
	}
	if _, err := os.Stat(otherKey); err != nil {
		t.Fatal("other Hub key was erased", err)
	}
	raw, err = os.ReadFile(path)
	must(t, err)
	var kept map[string]json.RawMessage
	must(t, json.Unmarshal(raw, &kept))
	if len(kept) != 1 || !strings.Contains(string(kept[other]), "future_binding") {
		t.Fatal("logout changed another Hub's cache")
	}
	pending, err := os.ReadFile(filepath.Join(root, "machine/execution-access-resets.json"))
	must(t, err)
	if !strings.Contains(string(pending), agentOrigin) || !strings.Contains(string(pending), `"pending":true`) {
		t.Fatal("actual agent origin was not queued")
	}
	if _, err := os.Stat(witness); !os.IsNotExist(err) {
		t.Fatal("logout started a stopped agent")
	}
}

type scopedAccessHub struct {
	hold             atomic.Bool
	entered, release chan struct{}
	once             sync.Once
	server, read     *httptest.Server
	calls            atomic.Int32
	expires          atomic.Int64
}

func newScopedAccessHub(t *testing.T) *scopedAccessHub {
	t.Helper()
	h := &scopedAccessHub{entered: make(chan struct{}), release: make(chan struct{})}
	h.expires.Store(3600)
	h.read = httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(h.read.Close)
	h.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/execution-access" {
			http.NotFound(w, r)
			return
		}
		if h.hold.Load() {
			h.once.Do(func() { close(h.entered) })
			select {
			case <-h.release:
			case <-r.Context().Done():
				return
			}
		}
		principal := "first"
		if r.Header.Get("Authorization") == "Bearer second-login" {
			principal = "second"
		}
		serial := h.calls.Add(1)
		_ = json.NewEncoder(w).Encode(map[string]any{"token": executionGrantToken(h.server.URL, principal, serial), "expires_at": time.Now().Add(time.Duration(h.expires.Load()) * time.Second).UTC(), "environment": map[string]string{"TENSORHUB_ORIGIN": h.read.URL}})
	}))
	t.Cleanup(h.server.Close)
	return h
}
func (h *scopedAccessHub) client(token string) *hub.Client {
	return hub.New(config.Config{HubURL: h.server.URL, HubToken: secret.New(token)}, "scoped access proof")
}

func scopedMachine(t *testing.T, paused bool) (string, *machines.Host, *machines.Launch) {
	t.Helper()
	if *machineHostBinary == "" || *machineRuntimeWheel == "" || *machineTensorFSWheel == "" {
		if *requireMachineHost {
			t.Fatal("scoped access proof requires actual agent and paired wheels")
		}
		t.Skip("requires actual agent and paired wheels")
	}
	root, err := os.MkdirTemp("", "cz-scope-")
	must(t, err)
	claimScratch(root)
	provisionMachine(t, root)
	h := machines.NewHost(home.Paths(root).Machine, "", nil)
	t.Cleanup(func() {
		_ = h.Stop(context.Background())
		if !t.Failed() {
			_ = removeAllForce(root)
		} else {
			t.Logf("scoped access evidence: %s", root)
		}
	})
	if paused {
		script := `from pathlib import Path
import sys
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents,worker_pb2 as pb
w=Workspace(Path(sys.argv[1]));e=Executions(w)
inv,digest=documents.identity(pb.InvocationSpec())
offer=pb.AttemptOffer(request_id='scoped-paused',attempt_ordinal=1,invocation_spec_digest=digest,invocation_spec_canonical_bytes=inv)
e.submit('cozy-local-client','scoped-paused',b'c'*32,offer,expected_execution_workspace_id=e.workspace_id)
e.control('cozy-local-client','scoped-paused','pause',1,'pause')
body,identity=documents.identity(pb.AttemptOutcomeBody(request_id='scoped-paused',attempt_ordinal=1,invocation_spec_digest=documents.spell(digest),status=pb.OUTCOME_STATUS_CANCELED,safe_message='paused proof'))
w.outcome('cozy-local-client',pb.AttemptOutcome(request_id='scoped-paused',attempt_ordinal=1,invocation_spec_digest=digest,outcome_id='paused-proof',outcome_digest=identity,outcome_canonical_bytes=body))
e.reconcile('cozy-local-client','scoped-paused')
assert e.status('cozy-local-client','scoped-paused').state=='paused'
`
		command := exec.Command(filepath.Join(h.Root(), "opt/cozy/python/bin/python"), "-I", "-c", script, filepath.Join(h.Root(), "var/lib/tensorfs"))
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("create actual paused journal: %v %s", err, output)
		}
	}
	launch, problem := h.Ensure(t.Context(), "", nil, true)
	fatal(t, problem)
	awaitScopedMachine(t, h, launch)
	return root, h, launch
}
func awaitScopedMachine(t *testing.T, h *machines.Host, launch *machines.Launch) {
	t.Helper()
	pin, problem := h.Pin()
	fatal(t, problem)
	owner, problem := h.Owner()
	fatal(t, problem)
	waitUntil(t, "the fixture Runtime is ready", func() bool {
		phase, err := runtimePhase(t, launch.Addr, pin, launch.WorkerID, owner)
		if err == nil && phase == "failed" {
			t.Fatal("fixture Runtime failed")
		}
		return err == nil && phase == "ready"
	})
}
func machineScopedFiles(t *testing.T, h *machines.Host) []byte {
	t.Helper()
	first, err := os.ReadFile(filepath.Join(h.Root(), "var/lib/cozy/machine/hub-access.json"))
	must(t, err)
	second, err := os.ReadFile(filepath.Join(h.Root(), "run/cozy/bootstrap/machine-hubs.json"))
	must(t, err)
	return append(first, second...)
}

func TestScopedAccessResetSwitchesAccountsAndRetainsOtherHub(t *testing.T) {
	root, h, launch := scopedMachine(t, false)
	a, b := newScopedAccessHub(t), newScopedAccessHub(t)
	first := a.client("first-login")
	_, problem := h.Ensure(t.Context(), a.server.URL, first, true)
	fatal(t, problem)
	_, problem = h.Ensure(t.Context(), a.server.URL, first, true)
	fatal(t, problem)
	if a.calls.Load() != 1 {
		t.Fatal("bound cached grant was not reused", a.calls.Load())
	}
	_, problem = h.Ensure(t.Context(), b.server.URL, b.client("first-login"), true)
	fatal(t, problem)
	pending, problem := h.ForgetExecutionAccess(t.Context(), a.server.URL)
	fatal(t, problem)
	if pending {
		t.Fatal("idle actual agent did not remove scoped access")
	}
	copies := string(machineScopedFiles(t, h))
	if strings.Contains(copies, executionGrantToken(a.server.URL, "first", 1)) || !strings.Contains(copies, executionGrantToken(b.server.URL, "first", 1)) {
		t.Fatal("reset erased the wrong Hub")
	}
	_, problem = h.Ensure(t.Context(), a.server.URL, a.client("second-login"), true)
	fatal(t, problem)
	// A cache from before device-bound grants must be refreshed, even when its
	// certificate, credential identity and expiry still match.
	cachePath := filepath.Join(home.Paths(root).Machine, "execution-access.json")
	raw, err := os.ReadFile(cachePath)
	must(t, err)
	var cache map[string]map[string]any
	must(t, json.Unmarshal(raw, &cache))
	entry := cache[a.server.URL]
	parts := strings.Split(entry["token"].(string), ".")
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	must(t, err)
	var claims map[string]any
	must(t, json.Unmarshal(payload, &claims))
	delete(claims, "attributes")
	payload, err = json.Marshal(claims)
	must(t, err)
	parts[1] = base64.RawURLEncoding.EncodeToString(payload)
	entry["token"] = strings.Join(parts, ".")
	raw, err = json.Marshal(cache)
	must(t, err)
	must(t, os.WriteFile(cachePath, raw, 0600))
	before := a.calls.Load()
	_, problem = h.Ensure(t.Context(), a.server.URL, a.client("second-login"), true)
	fatal(t, problem)
	if a.calls.Load() != before+1 {
		t.Fatal("unbound cached grant survived hard cut")
	}
	fatal(t, h.Stop(t.Context()))
	pending, problem = h.ForgetExecutionAccess(t.Context(), a.server.URL)
	fatal(t, problem)
	if !pending {
		t.Fatal("stopped-agent removal was not deferred")
	}
	status, problem := h.Status()
	fatal(t, problem)
	if status.Running {
		t.Fatal("logout restarted the agent")
	}
	launch, problem = h.Ensure(t.Context(), "", nil, true)
	fatal(t, problem)
	awaitScopedMachine(t, h, launch)
	_, problem = h.Ensure(t.Context(), "", nil, true)
	fatal(t, problem)
	copies = string(machineScopedFiles(t, h))
	if strings.Contains(copies, a.read.URL) || !strings.Contains(copies, b.read.URL) {
		t.Fatal("later use did not drain only the selected pending origin")
	}
}

func TestScopedAccountSwitchDefersWhileActualPausedWorkRemains(t *testing.T) {
	_, h, _ := scopedMachine(t, true)
	a := newScopedAccessHub(t)
	_, problem := h.Ensure(t.Context(), a.server.URL, a.client("first-login"), true)
	fatal(t, problem)
	before := string(machineScopedFiles(t, h))
	_, problem = h.Ensure(t.Context(), a.server.URL, a.client("second-login"), true)
	if problem == nil || problem.ErrName() != "machine.execution_access_busy" {
		t.Fatalf("paused work did not refuse account change: %v", problem)
	}
	if string(machineScopedFiles(t, h)) != before {
		t.Fatal("failed account switch changed agent credentials")
	}
	pending, problem := h.ForgetExecutionAccess(t.Context(), a.server.URL)
	if !pending || problem == nil || problem.ErrName() != "machine.execution_access_busy" {
		t.Fatal("busy logout was not deferred", pending, problem)
	}
	if string(machineScopedFiles(t, h)) != before {
		t.Fatal("busy logout changed agent credentials")
	}
	// Logging back in as the same account can safely refresh its revoked bearer
	// without changing the principal beneath the paused execution.
	_, problem = h.Ensure(t.Context(), a.server.URL, a.client("first-login"), true)
	fatal(t, problem)
	if !strings.Contains(string(machineScopedFiles(t, h)), executionGrantToken(a.server.URL, "first", 3)) {
		t.Fatal("same-account renewal was blocked by deferred removal")
	}
}

func TestScopedLogoutWinsAgainstAnInFlightAttachment(t *testing.T) {
	for _, existing := range []bool{false, true} {
		name := "initial"
		if existing {
			name = "account-switch"
		}
		t.Run(name, func(t *testing.T) { scopedLogoutRace(t, existing) })
	}
}

func scopedLogoutRace(t *testing.T, existing bool) {
	root, h, launch := scopedMachine(t, false)
	a := newScopedAccessHub(t)
	login := "first-login"
	if existing {
		_, problem := h.Ensure(t.Context(), a.server.URL, a.client(login), true)
		fatal(t, problem)
		login = "second-login"
	}
	a.hold.Store(true)
	done := make(chan *exit.Error, 1)
	go func() {
		_, problem := h.Ensure(t.Context(), a.server.URL, a.client(login), true)
		done <- problem
	}()
	select {
	case <-a.entered:
	case <-time.After(time.Minute):
		t.Fatal("authorization did not reach its fixture")
	}
	pending, problem := h.ForgetExecutionAccess(t.Context(), a.server.URL)
	fatal(t, problem)
	if !pending {
		t.Fatal("in-flight attachment did not defer cleanup")
	}
	close(a.release)
	select {
	case problem := <-done:
		if problem == nil || problem.ErrName() != "machine.execution_access_revoked" {
			t.Fatalf("logout was undone by attachment: %v", problem)
		}
	case <-time.After(time.Minute):
		t.Fatal("attachment did not settle")
	}
	if _, err := os.Stat(filepath.Join(home.Paths(root).Machine, "execution-access.json")); !os.IsNotExist(err) {
		t.Fatal("in-flight attachment restored client credentials")
	}
	// An account-switch refusal can leave the previous principal queued for
	// removal, but must never replace it with the in-flight new account.
	if existing {
		if strings.Contains(string(machineScopedFiles(t, h)), executionGrantToken(a.server.URL, "second", 2)) {
			t.Fatal("racing account switch installed the logged-out principal")
		}
		h.ResumeExecutionAccessCleanup(t.Context(), launch)
	}
	if strings.Contains(string(machineScopedFiles(t, h)), a.read.URL) {
		t.Fatal("in-flight attachment restored agent credentials")
	}
	awaitScopedMachine(t, h, launch)
}
