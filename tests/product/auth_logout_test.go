package producttest

import (
	"crypto/ed25519"
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

	"golang.org/x/crypto/ssh"

	"github.com/cozy-creator/cozy/internal/config"
)

// Logging out erases this machine's local credential, this install's key and the execution
// access it minted whatever Tensorhub answers: an empty 2xx is a confirmation, and a refused or unreachable
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
			mux.HandleFunc("POST /v1/auth/device-keys/login/begin", func(w http.ResponseWriter, _ *http.Request) {
				challenge := make([]byte, 32)
				_, _ = rand.Read(challenge)
				_ = json.NewEncoder(w).Encode(map[string]string{"challenge_id": "challenge-1",
					"challenge": base64.RawURLEncoding.EncodeToString(challenge), "expires_at": time.Now().Add(time.Minute).Format(time.RFC3339Nano)})
			})
			mux.HandleFunc("POST /v1/auth/device-keys/login/finish", func(w http.ResponseWriter, _ *http.Request) {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"token_set":  map[string]any{"access_token": "machine-token", "token_type": "Bearer", "expires_in": 600},
					"device_key": map[string]any{"id": "device-1"}})
			})
			mux.HandleFunc("DELETE /v1/auth/device-keys/{id}", func(w http.ResponseWriter, r *http.Request) {
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
			credential, access := writeMachineCredential(t, root, server.URL), writeExecutionAccess(t, root, server.URL)
			code, out := runCozy(t, root, "auth", "logout", "--json")
			if code != 0 || !strings.Contains(out, `"status":"logged out"`) || revoked != 1 {
				t.Fatalf("logout failed [exit %d, %d revocations]: %s", code, revoked, out)
			}
			// The revoked key cannot be registered again: this install's key goes with the login.
			for _, path := range []string{credential, access, filepath.Join(root, "id_ed25519"), filepath.Join(root, "id_ed25519.pub")} {
				if _, err := os.Stat(path); !os.IsNotExist(err) {
					t.Fatalf("%s survived logout: %v", path, err)
				}
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
		credential, access := writeMachineCredential(t, root, origin), writeExecutionAccess(t, root, origin)
		if code, out := runCozy(t, root, "auth", "logout", "--json"); code != 0 || !strings.Contains(out, `"status":"logged out"`) {
			t.Fatalf("an unreachable hub blocked local logout [exit %d]: %s", code, out)
		}
		for _, path := range []string{credential, access} {
			if _, err := os.Stat(path); !os.IsNotExist(err) {
				t.Fatalf("%s survived logout: %v", path, err)
			}
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

// writeExecutionAccess is the grant a run under this login left in the machine's cache.
func writeExecutionAccess(t *testing.T, root, origin string) string {
	t.Helper()
	raw, err := json.Marshal(map[string]any{origin: map[string]any{"origin": origin, "token": "execution-token",
		"expires_at": time.Now().Add(7 * 24 * time.Hour).Unix(), "credential_identity": "key:device-1"}})
	must(t, err)
	path := filepath.Join(root, "machine", "execution-access.json")
	must(t, os.MkdirAll(filepath.Dir(path), 0700))
	must(t, os.WriteFile(path, raw, 0600))
	return path
}

// A login made before install keys carries over: its device key becomes this install's key,
// so the Hub's registration still names it and nobody logs in again.
func TestALoginBeforeInstallKeysBecomesTheInstallKey(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/auth/device-keys/login/begin", func(w http.ResponseWriter, _ *http.Request) {
		challenge := make([]byte, 32)
		_, _ = rand.Read(challenge)
		_ = json.NewEncoder(w).Encode(map[string]string{"challenge_id": "challenge-1",
			"challenge": base64.RawURLEncoding.EncodeToString(challenge), "expires_at": time.Now().Add(time.Minute).Format(time.RFC3339Nano)})
	})
	mux.HandleFunc("POST /v1/auth/device-keys/login/finish", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token_set":  map[string]any{"access_token": "machine-token", "token_type": "Bearer", "expires_in": 600},
			"device_key": map[string]any{"id": "device-1"}})
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\n"), 0600))
	raw, err := os.ReadFile(writeMachineCredential(t, root, server.URL))
	must(t, err)
	var login struct {
		PrivateKey string `json:"private_key"`
	}
	must(t, json.Unmarshal(raw, &login))
	seed, err := base64.RawURLEncoding.DecodeString(login.PrivateKey)
	must(t, err)
	if code, out := runCozy(t, root, "auth", "--json"); code != 0 || !strings.Contains(out, `"status":"logged in"`) {
		t.Fatalf("auth status [exit %d]: %s", code, out)
	}
	public, err := ssh.NewPublicKey(ed25519.NewKeyFromSeed(seed).Public())
	must(t, err)
	installed, err := os.ReadFile(filepath.Join(root, "id_ed25519.pub"))
	must(t, err)
	if !strings.HasPrefix(string(installed), strings.TrimSpace(string(ssh.MarshalAuthorizedKey(public)))+" ") {
		t.Fatalf("the install key is not the login's device key: %s", installed)
	}
}
