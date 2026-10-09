package producttest

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
)

// Logging out erases this account's local device credential
// whatever Tensorhub answers: an empty 2xx is a confirmation, and a refused or unreachable
// revocation is a note.
func TestLogoutAlwaysErasesTheLocalCredential(t *testing.T) {
	for _, row := range []struct {
		name   string
		revoke func(http.ResponseWriter)
		note   bool
	}{
		{"empty-ok", func(w http.ResponseWriter) { w.WriteHeader(http.StatusOK) }, false},
		{"no-content", func(w http.ResponseWriter) { w.WriteHeader(http.StatusNoContent) }, false},
		{"refused", func(w http.ResponseWriter) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":{"code":"internal","message":"revocation store unavailable"}}`))
		}, true},
	} {
		t.Run(row.name, func(t *testing.T) {
			revoked := 0
			mux := http.NewServeMux()
			mux.HandleFunc("POST /v1/auth/v1/device-keys/login/begin", func(w http.ResponseWriter, _ *http.Request) {
				challenge := make([]byte, 32)
				_, _ = rand.Read(challenge)
				_ = json.NewEncoder(w).Encode(map[string]string{"challenge_id": "challenge-1",
					"challenge": base64.RawURLEncoding.EncodeToString(challenge), "expires_at": time.Now().Add(time.Minute).Format(time.RFC3339Nano)})
			})
			mux.HandleFunc("POST /v1/auth/v1/device-keys/login/finish", func(w http.ResponseWriter, _ *http.Request) {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"token_set":  map[string]any{"access_token": "machine-token", "token_type": "Bearer", "expires_in": 600},
					"device_key": map[string]any{"id": "device-1"}})
			})
			mux.HandleFunc("DELETE /v1/auth/v1/me/sign-in-keys/{id}", func(w http.ResponseWriter, r *http.Request) {
				if r.PathValue("id") != "device-1" || r.Header.Get("Authorization") != "Bearer machine-token" {
					t.Errorf("unexpected revocation: %s %s", r.PathValue("id"), r.Header.Get("Authorization"))
				}
				revoked++
				row.revoke(w)
			})
			server := httptest.NewServer(mux)
			defer server.Close()
			root := t.TempDir()
			must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\n"), 0600))
			credential := writeMachineCredential(t, root, server.URL)
			code, out := runCozy(t, root, "auth", "logout", "--json")
			if code != 0 || !strings.Contains(out, `"status":"logged out"`) || revoked != 1 {
				t.Fatalf("logout failed [exit %d, %d revocations]: %s", code, revoked, out)
			}
			if _, err := os.Stat(credential); !os.IsNotExist(err) {
				t.Fatalf("%s survived logout: %v", credential, err)
			}
			if row.note != strings.Contains(out, "did not confirm") {
				t.Fatalf("revocation note mismatch: %s", out)
			}
		})
	}
	t.Run("hub-unreachable", func(t *testing.T) {
		server := httptest.NewServer(http.NotFoundHandler())
		origin := server.URL
		server.Close()
		root := t.TempDir()
		must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+origin+"\n"), 0600))
		credential := writeMachineCredential(t, root, origin)
		if code, out := runCozy(t, root, "auth", "logout", "--json"); code != 0 || !strings.Contains(out, `"status":"logged out"`) {
			t.Fatalf("an unreachable hub blocked local logout [exit %d]: %s", code, out)
		}
		if _, err := os.Stat(credential); !os.IsNotExist(err) {
			t.Fatalf("%s survived logout: %v", credential, err)
		}
	})
}

func writeMachineCredential(t *testing.T, root, origin string) string {
	t.Helper()
	seed := make([]byte, 32)
	_, err := rand.Read(seed)
	must(t, err)
	raw, err := json.Marshal(map[string]any{"version": 1, "hub": origin, "email": "owner@example.com",
		"device_key_id": "device-1", "private_key": base64.RawURLEncoding.EncodeToString(seed)})
	must(t, err)
	sum := sha256.Sum256([]byte(origin))
	path := filepath.Join(root, "auth", hex.EncodeToString(sum[:])+".json")
	must(t, os.MkdirAll(filepath.Dir(path), 0700))
	must(t, os.WriteFile(path, raw, 0600))
	return path
}
