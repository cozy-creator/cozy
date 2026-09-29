package producttest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
)

type executionCredential struct {
	key    string
	bearer secret.Value
}

func (s *executionCredential) Identity() string { return s.key }
func (s *executionCredential) AccessToken(context.Context) (secret.Value, *exit.Error) {
	return s.bearer, nil
}

func TestExecutionAccessIdentityFollowsDeviceKeyNotShortBearer(t *testing.T) {
	source := &executionCredential{key: "key:first", bearer: secret.New("short-first")}
	client := hub.New(config.Config{HubURL: "https://hub.invalid"}, "test").WithTokenSource(source)
	first := client.CredentialIdentity()
	source.bearer = secret.New("short-refreshed")
	if first == "" || client.CredentialIdentity() != first {
		t.Fatal("refreshing a short bearer changed the delegated authority's identity")
	}
	source.key = "key:second"
	if client.CredentialIdentity() == first {
		t.Fatal("another device login retained the preceding delegated identity")
	}
	source.key = ""
	if client.CredentialIdentity() != "" {
		t.Fatal("signing out retained authority from a cached short bearer")
	}
	firstToken := client.WithToken(secret.New("operator-first"), "test")
	secondToken := client.WithToken(secret.New("operator-second"), "test")
	if firstToken.CredentialIdentity() == secondToken.CredentialIdentity() || firstToken.CredentialIdentity() == "" {
		t.Fatal("operator credentials shared delegated authority")
	}
}

func TestCachedAccountSessionFollowsPersistedDeviceKeyChanges(t *testing.T) {
	var logins atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/auth/device-keys/login/begin":
			var request struct {
				Key string `json:"device_key_id"`
			}
			if json.NewDecoder(r.Body).Decode(&request) != nil || request.Key == "" {
				t.Error("login lost the persisted device key")
			}
			logins.Add(1)
			_ = json.NewEncoder(w).Encode(map[string]any{"challenge_id": request.Key,
				"challenge": base64.RawURLEncoding.EncodeToString(make([]byte, 32)), "expires_at": time.Now().Add(time.Minute)})
		case "/v1/auth/device-keys/login/finish":
			var request struct {
				Key string `json:"challenge_id"`
			}
			_ = json.NewDecoder(r.Body).Decode(&request)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"token_set":  map[string]any{"access_token": "bearer-" + request.Key, "token_type": "Bearer", "expires_in": 600},
				"device_key": map[string]any{"id": request.Key}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	root := t.TempDir()
	path := writeMachineCredential(t, root, server.URL)
	manager := accountauth.New(config.Config{Home: root, HubURL: server.URL})
	for range 2 {
		session, problem := manager.Authenticate(t.Context())
		fatal(t, problem)
		if !session.AccessToken.Equal("bearer-device-1") || logins.Load() != 1 {
			t.Fatal("the current key did not reuse its short bearer")
		}
	}
	raw, err := os.ReadFile(path)
	must(t, err)
	var changed map[string]any
	must(t, json.Unmarshal(raw, &changed))
	changed["device_key_id"] = "device-2"
	raw, err = json.Marshal(changed)
	must(t, err)
	must(t, os.WriteFile(path, raw, 0600))
	session, problem := manager.Authenticate(t.Context())
	fatal(t, problem)
	if session.DeviceKeyID != "device-2" || !session.AccessToken.Equal("bearer-device-2") || logins.Load() != 2 {
		t.Fatal("a new persisted login retained the preceding account's bearer")
	}
	must(t, os.Remove(path))
	if _, problem := manager.Authenticate(t.Context()); problem == nil || problem.ErrName() != "auth.machine_key_missing" {
		t.Fatalf("a signed-out daemon retained its cached account session: %v", problem)
	}
}
