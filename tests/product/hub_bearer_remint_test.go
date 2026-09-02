package producttest

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
)

// mintingSource is the accountauth shape: a cached bearer that only changes when
// the client tells it the hub refused the current one.
type mintingSource struct {
	token       atomic.Pointer[string]
	invalidated atomic.Int32
}

func (s *mintingSource) AccessToken(context.Context) (secret.Value, *exit.Error) {
	return secret.New(*s.token.Load()), nil
}

func (s *mintingSource) Invalidate() {
	s.invalidated.Add(1)
	fresh := "fresh"
	s.token.Store(&fresh)
}

// staleSource has no Invalidate: the client must surface the refusal untouched.
type staleSource struct{}

func (staleSource) AccessToken(context.Context) (secret.Value, *exit.Error) {
	return secret.New("stale"), nil
}

// hubAccepting answers 401 to every bearer but `accept`, counting requests.
func hubAccepting(accept string, requests *atomic.Int32) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if r.Header.Get("Authorization") != "Bearer "+accept {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"code":"invalid_credentials","message":"authenticate with cozy auth login"}}`))
			return
		}
		_, _ = w.Write([]byte(`{"name":"paul"}`))
	}))
}

func remintClient(url string, source hub.TokenSource) *hub.Client {
	return hub.New(config.Config{HubURL: url}, "producttest").WithTokenSource(source)
}

// A hub that restarted forgets our session without expiring our copy of it. The
// client re-mints from the machine key once and replays; the caller never sees
// the 401 that killed run 97 one minute after an air rebuild bounced the dev hub.
func TestUnauthorizedBearerIsRemintedOnce(t *testing.T) {
	var requests atomic.Int32
	server := hubAccepting("fresh", &requests)
	defer server.Close()
	source := &mintingSource{}
	stale := "stale"
	source.token.Store(&stale)

	account, e := remintClient(server.URL, source).CurrentAccount(context.Background())
	if e != nil {
		t.Fatalf("the reminted call failed: %s", e)
	}
	if account.Name != "paul" {
		t.Fatalf("the reminted call answered account %q", account.Name)
	}
	if got := source.invalidated.Load(); got != 1 {
		t.Fatalf("the source was invalidated %d times", got)
	}
	if got := requests.Load(); got != 2 {
		t.Fatalf("the hub saw %d requests", got)
	}
}

func TestUnauthorizedBearerWithoutRemintSurfaces(t *testing.T) {
	var requests atomic.Int32
	server := hubAccepting("fresh", &requests)
	defer server.Close()

	_, e := remintClient(server.URL, staleSource{}).CurrentAccount(context.Background())
	if e == nil || e.ErrName() != "invalid_credentials" {
		t.Fatalf("expected the refusal to surface, got %v", e)
	}
	if got := requests.Load(); got != 1 {
		t.Fatalf("the hub saw %d requests", got)
	}
}

// A machine key the hub truly revoked keeps refusing: exactly one replay, no loop.
func TestPersistentUnauthorizedStopsAfterOneReplay(t *testing.T) {
	var requests atomic.Int32
	server := hubAccepting("never", &requests)
	defer server.Close()
	source := &mintingSource{}
	stale := "stale"
	source.token.Store(&stale)

	_, e := remintClient(server.URL, source).CurrentAccount(context.Background())
	if e == nil || e.ErrName() != "invalid_credentials" {
		t.Fatalf("expected the refusal to surface, got %v", e)
	}
	if got := requests.Load(); got != 2 {
		t.Fatalf("the hub saw %d requests", got)
	}
}
