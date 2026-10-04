package machines

import (
	"cmp"
	"context"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"math/big"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/installkey"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/workertls"
)

// login is a signed-in account whose identity can rotate mid-attachment.
type login struct {
	identity atomic.Value
	rotate   func()
}

func (l *login) AccessToken(context.Context) (secret.Value, *exit.Error) {
	if l.rotate != nil {
		l.rotate()
	}
	return secret.New("user-bearer"), nil
}
func (l *login) Identity() string { return l.identity.Load().(string) }

// fakeHub answers POST /v1/execution-access as Tensorhub does: a device-bound
// delegated token for the presented leaf.
type fakeHub struct {
	*httptest.Server
	issued   atomic.Int32
	expires  time.Duration
	declared string
	subject  string
	unbound  bool
}

func newHub(t *testing.T) *fakeHub {
	h := &fakeHub{expires: time.Hour, subject: "acct-1"}
	h.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/execution-access" || r.Header.Get("Authorization") != "Bearer user-bearer" {
			http.Error(w, "unexpected", http.StatusBadRequest)
			return
		}
		var body struct {
			Leaf string `json:"delegate_certificate_der_b64url"`
		}
		if json.NewDecoder(r.Body).Decode(&body) != nil || body.Leaf == "" {
			http.Error(w, "leaf required", http.StatusBadRequest)
			return
		}
		n := h.issued.Add(1)
		attributes := map[string]string{"execution_device_key_id": "device-1"}
		if h.unbound {
			attributes = map[string]string{}
		}
		claims, _ := json.Marshal(map[string]any{"iss": h.URL, "delegated_sub": h.subject, "permissions": []string{"cozy.execution-access"}, "attributes": attributes, "n": n})
		token := base64.RawURLEncoding.EncodeToString([]byte(`{"typ":"delegated-access+jwt"}`)) + "." + base64.RawURLEncoding.EncodeToString(claims) + ".sig"
		_ = json.NewEncoder(w).Encode(map[string]any{"token": token, "expires_at": time.Now().Add(h.expires).UTC().Format(time.RFC3339), "environment": map[string]string{"TENSORHUB_ORIGIN": cmp.Or(h.declared, h.URL), "TENSORHUB_UNKNOWN": "dropped"}, "future_field": true})
	}))
	t.Cleanup(h.Close)
	return h
}

// fakeMachine is the Go agent's POST/DELETE /v1/hubs/access contract behind a pinned leaf.
type fakeMachine struct {
	*httptest.Server
	worker  string
	keys    []ed25519.PublicKey
	mu      sync.Mutex
	held    map[string]string // origin key -> principal
	posts   int
	deletes int
	reply   func(origin string, expires int64) map[string]any
	busy    bool
	absent  bool
	pem     []byte
}

func leaf(t *testing.T) (tls.Certificate, []byte) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: workertls.ServerName}, DNSNames: []string{workertls.ServerName}, NotBefore: time.Now().Add(-time.Minute), NotAfter: time.Now().Add(30 * 24 * time.Hour)}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	return tls.Certificate{Certificate: [][]byte{der}, PrivateKey: key}, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
}

func principal(token string) string {
	parts := strings.Split(token, ".")
	raw, _ := base64.RawURLEncoding.DecodeString(parts[1])
	var claims struct {
		Issuer  string `json:"iss"`
		Subject string `json:"delegated_sub"`
	}
	_ = json.Unmarshal(raw, &claims)
	return claims.Issuer + "\x00" + claims.Subject
}

func newMachine(t *testing.T, owner ed25519.PublicKey) *fakeMachine {
	m := &fakeMachine{worker: "worker-1", keys: []ed25519.PublicKey{owner}, held: map[string]string{}}
	cert, pemBytes := leaf(t)
	m.pem = pemBytes
	m.Server = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if m.absent || r.URL.Path != "/v1/hubs/access" {
			http.NotFound(w, r)
			return
		}
		token, _ := strings.CutPrefix(r.Header.Get("Authorization"), "Cozy-Cap ")
		grant, err := capability.Verify(token, m.worker, m.keys, time.Now(), "")
		if err != nil || !grant.Permits("hub-access") {
			http.Error(w, "a hub-access capability is required", http.StatusForbidden)
			return
		}
		var body struct {
			Origin    string `json:"origin"`
			Token     string `json:"token"`
			ExpiresAt int64  `json:"expires_at"`
		}
		if json.NewDecoder(r.Body).Decode(&body) != nil {
			http.Error(w, "bad", http.StatusBadRequest)
			return
		}
		m.mu.Lock()
		defer m.mu.Unlock()
		key := accessOrigin(strings.TrimSuffix(body.Origin, "/"))
		if r.Method == http.MethodDelete {
			m.deletes++
			if m.busy {
				w.WriteHeader(http.StatusConflict)
				_, _ = w.Write([]byte(`{"error":{"code":"machine_busy"}}`))
				return
			}
			delete(m.held, key)
			w.WriteHeader(http.StatusNoContent)
			return
		}
		m.posts++
		if body.ExpiresAt <= time.Now().Unix() {
			http.Error(w, "Hub access grant has expired", http.StatusBadRequest)
			return
		}
		if held, ok := m.held[key]; ok && held != principal(body.Token) {
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(`{"error":{"code":"hub_access_principal_conflict"}}`))
			return
		}
		m.held[key] = principal(body.Token)
		reply := map[string]any{"origin": body.Origin, "expires_at": body.ExpiresAt}
		if m.reply != nil {
			reply = m.reply(body.Origin, body.ExpiresAt)
		}
		_ = json.NewEncoder(w).Encode(reply)
	}))
	m.TLS = &tls.Config{Certificates: []tls.Certificate{cert}}
	m.StartTLS()
	t.Cleanup(m.Close)
	return m
}

type scope struct {
	cache   accessCache
	target  accessTarget
	account *hub.Client
	login   *login
	hub     *fakeHub
	machine *fakeMachine
	owner   installkey.Key
}

func newScope(t *testing.T) *scope {
	dir := t.TempDir()
	owner, problem := installkey.Ensure(dir)
	if problem != nil {
		t.Fatal(problem)
	}
	h := newHub(t)
	m := newMachine(t, owner.Public())
	pin, err := workertls.ParsePin(m.pem)
	if err != nil {
		t.Fatal(err)
	}
	l := &login{}
	l.identity.Store("device:acct-1")
	s := &scope{cache: accessCache{dir: filepath.Join(dir, "endpoints", "endpoint-x")}, login: l, hub: h, machine: m, owner: owner,
		account: hub.New(config.Config{HubURL: h.URL}, "test").WithTokenSource(l)}
	s.target = accessTarget{addr: strings.TrimPrefix(m.URL, "https://"), worker: m.worker, leaf: pin.DER(),
		pin:   func() (*workertls.Pin, *exit.Error) { return pin, nil },
		owner: func() (installkey.Key, *exit.Error) { return s.owner, nil }}
	return s
}

func (s *scope) attach(t *testing.T) (string, *exit.Error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	resumeAccessCleanup(ctx, s.cache, s.target)
	return attachAccess(ctx, s.cache, s.target, s.hub.URL, s.account)
}

func named(t *testing.T, problem *exit.Error, name string) {
	t.Helper()
	if problem == nil || problem.ErrName() != name {
		t.Fatalf("want %s, got %v", name, problem)
	}
}

func TestEndpointAccessIsCachedAcrossRuns(t *testing.T) {
	s := newScope(t)
	for run := 0; run < 3; run++ {
		reads, problem := s.attach(t)
		if problem != nil || accessOrigin(reads) != accessOrigin(s.hub.URL) {
			t.Fatalf("run %d: reads %q, %v", run, reads, problem)
		}
	}
	// The machine is told on every connection; Tensorhub is asked once.
	if s.hub.issued.Load() != 1 || s.machine.posts != 3 {
		t.Fatalf("hub issued %d, machine posts %d", s.hub.issued.Load(), s.machine.posts)
	}
}

func TestEndpointAccessAcknowledgementIsSemantic(t *testing.T) {
	s := newScope(t)
	s.machine.reply = func(origin string, expires int64) map[string]any {
		// The agent answers its retained spelling, plus fields this client does not know.
		spelled := strings.Replace(origin, "http://127.0.0.1", "HTTP://127.0.0.1", 1) + "/"
		return map[string]any{"origin": spelled, "expires_at": expires, "principal": "x", "generation": 9}
	}
	if _, problem := s.attach(t); problem != nil {
		t.Fatal(problem)
	}
	s.machine.reply = func(origin string, expires int64) map[string]any {
		return map[string]any{"origin": "https://other.example", "expires_at": expires}
	}
	s.login.identity.Store("device:acct-1-renewed") // force a fresh grant
	_, problem := s.attach(t)
	if problem == nil || problem.Code != exit.Conflict {
		t.Fatalf("another acknowledged origin must refuse, got %v", problem)
	}
	s.machine.reply = func(origin string, expires int64) map[string]any {
		return map[string]any{"origin": origin, "expires_at": expires + 1}
	}
	if _, problem := s.attach(t); problem == nil || problem.Code != exit.Conflict {
		t.Fatalf("another acknowledged expiry must refuse, got %v", problem)
	}
}

func TestEndpointAccessRefreshesExpiringAndRotatedGrants(t *testing.T) {
	s := newScope(t)
	s.hub.expires = 30 * time.Second // inside the one-minute refresh margin
	if _, problem := s.attach(t); problem != nil {
		t.Fatal(problem)
	}
	if _, problem := s.attach(t); problem != nil {
		t.Fatal(problem)
	}
	if s.hub.issued.Load() != 2 {
		t.Fatalf("an expiring grant must be renewed, issued %d", s.hub.issued.Load())
	}
	s.hub.expires = time.Hour
	s.login.identity.Store("device:acct-1-rotated")
	if _, problem := s.attach(t); problem != nil {
		t.Fatal(problem)
	}
	if s.hub.issued.Load() != 3 {
		t.Fatalf("a rotated login must not reuse the earlier grant, issued %d", s.hub.issued.Load())
	}
	// A Hub that answers already-expired access is refused before the machine sees it.
	s.hub.expires = -time.Minute
	s.login.identity.Store("device:acct-1-again")
	posts := s.machine.posts
	_, problem := s.attach(t)
	named(t, problem, "hub.execution_access_invalid")
	if s.machine.posts != posts {
		t.Fatal("expired access reached the machine")
	}
}

func TestEndpointAccessLoginChangeDuringAttachmentRefuses(t *testing.T) {
	s := newScope(t)
	s.login.rotate = func() { s.login.identity.Store("device:someone-else") }
	_, problem := s.attach(t)
	named(t, problem, "machine.execution_account_changed")
	if s.machine.posts != 0 {
		t.Fatal("access from a changed login reached the machine")
	}
}

func TestEndpointAccessRefusesUnboundAccess(t *testing.T) {
	s := newScope(t)
	s.hub.unbound = true
	_, problem := s.attach(t)
	named(t, problem, "hub.execution_access_device_key_required")
}

func TestEndpointAccessWrongKeyWorkerOrPinRefuses(t *testing.T) {
	s := newScope(t)
	other, problem := installkey.Ensure(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	right := s.owner
	s.owner = other
	_, problem = s.attach(t)
	named(t, problem, "machine.execution_access_refused")

	s.owner = right
	s.target.worker = "worker-2"
	_, problem = s.attach(t)
	named(t, problem, "machine.execution_access_refused")

	s.target.worker = s.machine.worker
	_, wrong := leaf(t)
	pin, err := workertls.ParsePin(wrong)
	if err != nil {
		t.Fatal(err)
	}
	s.target.pin = func() (*workertls.Pin, *exit.Error) { return pin, nil }
	_, problem = s.attach(t)
	if problem == nil || problem.Code != exit.Unavailable {
		t.Fatalf("a machine presenting another leaf must not be reached, got %v", problem)
	}
	if s.machine.posts != 0 {
		t.Fatalf("refused capabilities were accepted %d times", s.machine.posts)
	}
}

func TestEndpointAccessOtherPrincipalIsResetThenAttached(t *testing.T) {
	s := newScope(t)
	if _, problem := s.attach(t); problem != nil {
		t.Fatal(problem)
	}
	s.hub.subject = "acct-2"
	s.login.identity.Store("device:acct-2")
	s.machine.busy = true
	_, problem := s.attach(t)
	named(t, problem, "machine.execution_access_busy")
	// Accepted work finished: the queued removal runs first, then the new account attaches.
	s.machine.busy = false
	if _, problem := s.attach(t); problem != nil {
		t.Fatal(problem)
	}
	if s.machine.held[accessOrigin(s.hub.URL)] != s.hub.URL+"\x00acct-2" {
		t.Fatalf("machine holds %q", s.machine.held)
	}
}

func TestEndpointAccessMachineWithoutRouteIsNamed(t *testing.T) {
	s := newScope(t)
	s.machine.absent = true
	_, problem := s.attach(t)
	named(t, problem, "machine.agent_update_required")
}

func TestLogoutErasesEndpointGrantAndQueuesRemoval(t *testing.T) {
	s := newScope(t)
	if _, problem := s.attach(t); problem != nil {
		t.Fatal(problem)
	}
	host := NewHost(filepath.Dir(filepath.Dir(s.cache.dir)), "", nil)
	if _, problem := host.ForgetExecutionAccess(context.Background(), s.hub.URL); problem != nil {
		t.Fatal(problem)
	}
	var cached executionAccess
	_ = s.cache.state(context.Background(), func(state *accessState) *exit.Error {
		cached = cachedAccess(state, s.hub.URL)
		return nil
	})
	if cached.Token != "" {
		t.Fatal("logout left the endpoint grant cached")
	}
	if _, problem := s.attach(t); problem != nil {
		t.Fatal(problem)
	}
	if s.machine.deletes != 1 || s.hub.issued.Load() != 2 {
		t.Fatalf("next connection must remove then re-authorize: deletes %d issued %d", s.machine.deletes, s.hub.issued.Load())
	}
}

// A machine serving cozy.machine.v1 is never given a standing grant, so a logout that finds
// none cached owes it no removal and asks it nothing.
func TestLogoutWithNoAttachedGrantOwesNoRemoval(t *testing.T) {
	s := newScope(t)
	host := NewHost(filepath.Dir(filepath.Dir(s.cache.dir)), "", nil)
	pending, problem := host.ForgetExecutionAccess(context.Background(), s.hub.URL)
	if problem != nil || pending {
		t.Fatalf("a logout with nothing attached deferred a removal: %v %v", pending, problem)
	}
	resumeAccessCleanup(context.Background(), s.cache, s.target)
	if s.machine.deletes != 0 {
		t.Fatalf("the machine was asked to remove access it never held: %d", s.machine.deletes)
	}
}

func TestRemoteEndpointReadsDeclaredOriginOfLoopbackHub(t *testing.T) {
	s := newScope(t)
	pin, _ := s.target.pin()
	// A loopback endpoint address is a tunnel as often as this computer.
	s.target = endpointTarget(machineendpoint.Endpoint{Address: s.target.addr, WorkerID: s.target.worker}, pin, s.owner)
	s.hub.declared = "https://public.example.test"
	reads, problem := s.attach(t)
	if problem != nil || reads != "https://public.example.test" {
		t.Fatalf("a remote machine must read the declared origin, got %q %v", reads, problem)
	}
	s.target.local = true
	s.cache = accessCache{dir: t.TempDir()}
	reads, problem = s.attach(t)
	if problem != nil || accessOrigin(reads) != accessOrigin(s.hub.URL) {
		t.Fatalf("a machine on this computer reads a loopback Hub at loopback, got %q %v", reads, problem)
	}
}
