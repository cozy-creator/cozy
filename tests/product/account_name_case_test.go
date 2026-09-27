package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"
)

// accountHub is Tensorhub's login and account-name contract: device-key
// enrollment, then one immutable lowercase account name per user.
type accountHub struct {
	mu         sync.Mutex
	challenges map[string][]byte
	keys       map[string]ed25519.PublicKey
	owners     map[string]string // account name -> user token
	accounts   map[string]string // user token -> account name
	registered []string
}

func newAccountHub(t *testing.T) *httptest.Server {
	hub := &accountHub{challenges: map[string][]byte{}, keys: map[string]ed25519.PublicKey{},
		owners: map[string]string{}, accounts: map[string]string{}}
	mux := http.NewServeMux()
	reply := func(w http.ResponseWriter, status int, body any) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(body)
	}
	problem := func(code, message string) map[string]any {
		return map[string]any{"error": map[string]string{"code": code, "message": message}}
	}
	mux.HandleFunc("POST /v1/auth/device-keys/enroll/begin", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			PublicKey string `json:"public_key"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		public, err := base64.RawURLEncoding.DecodeString(body.PublicKey)
		if err != nil || len(public) != ed25519.PublicKeySize {
			reply(w, 400, problem("invalid_request", "bad public key"))
			return
		}
		challenge := make([]byte, 32)
		_, _ = rand.Read(challenge)
		id := base64.RawURLEncoding.EncodeToString(challenge[:8])
		hub.mu.Lock()
		hub.challenges[id], hub.keys[id] = challenge, public
		hub.mu.Unlock()
		reply(w, 202, map[string]string{"enrollment_id": id, "challenge": base64.RawURLEncoding.EncodeToString(challenge),
			"expires_at": time.Now().Add(10 * time.Minute).UTC().Format(time.RFC3339Nano)})
	})
	mux.HandleFunc("POST /v1/auth/device-keys/enroll/finish", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			ID        string `json:"enrollment_id"`
			Code      string `json:"code"`
			Signature string `json:"signature"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		hub.mu.Lock()
		challenge, public := hub.challenges[body.ID], hub.keys[body.ID]
		hub.mu.Unlock()
		signature, _ := base64.RawURLEncoding.DecodeString(body.Signature)
		message := append([]byte("authkit.device-key-enrollment/1\x00"), challenge...)
		if body.Code != "123456" || public == nil || !ed25519.Verify(public, message, signature) {
			reply(w, 400, problem("invalid_code", "enrollment proof refused"))
			return
		}
		reply(w, 200, map[string]any{
			"token_set":  map[string]any{"access_token": "user-" + body.ID, "token_type": "Bearer", "expires_in": 900},
			"device_key": map[string]string{"id": "key-" + body.ID, "label": "proof"},
		})
	})
	mux.HandleFunc("GET /v1/accounts/current", func(w http.ResponseWriter, r *http.Request) {
		hub.mu.Lock()
		name, ok := hub.accounts[r.Header.Get("Authorization")]
		hub.mu.Unlock()
		if !ok {
			reply(w, 409, problem("account.name_required", "choose an account name"))
			return
		}
		reply(w, 200, map[string]string{"name": name})
	})
	mux.HandleFunc("PUT /v1/accounts/{account}", func(w http.ResponseWriter, r *http.Request) {
		name, user := r.PathValue("account"), r.Header.Get("Authorization")
		hub.mu.Lock()
		defer hub.mu.Unlock()
		hub.registered = append(hub.registered, name)
		if name != strings.ToLower(name) {
			reply(w, 422, problem("account.name_invalid", "account names are stored lowercase"))
			return
		}
		if owner, taken := hub.owners[name]; taken && owner != user {
			reply(w, 409, problem("account.name_unavailable", "that account name is already registered"))
			return
		}
		hub.owners[name], hub.accounts[user] = user, name
		reply(w, 200, map[string]string{"name": name})
	})
	mux.HandleFunc("GET /v1/models/{org}/{name}", func(w http.ResponseWriter, r *http.Request) {
		if r.PathValue("org") != "paul" || r.PathValue("name") != "sdxl" {
			reply(w, 404, problem("model.not_found", "no model "+r.PathValue("org")+"/"+r.PathValue("name")))
			return
		}
		reply(w, 200, map[string]any{"model": map[string]string{"org": "paul", "name": "sdxl",
			"created_at": "2026-09-26T00:00:00Z"}, "releases": []any{}})
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return server
}

// The owner's report: `cozy auth login` refused "Fidika" at the account-name
// prompt. Any case is accepted and becomes the one lowercase account; the same
// name in another case from another user is "already registered".
func TestAuthLoginAccountNameIgnoresCase(t *testing.T) {
	hub := newAccountHub(t)
	login := func(email, typed string) (int, string) {
		t.Helper()
		cmd := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "auth", "login", email, "--tensorhub="+hub.URL)
		cmd.Env = childEnv(t, t.TempDir())
		cmd.Stdin = strings.NewReader("123456\n" + typed + "\n") //cozy:stdin-value test login code and account name
		var out bytes.Buffer
		cmd.Stdout, cmd.Stderr = &out, &out
		_ = cmd.Run()
		return cmd.ProcessState.ExitCode(), out.String()
	}
	code, out := login("fidika@example.test", "Fidika")
	if code != 0 || !strings.Contains(out, "fidika") || strings.Contains(out, "Fidika") {
		t.Fatalf("mixed-case account name: exit %d\n%s", code, out)
	}
	code, out = login("other@example.test", "FIDIKA")
	if code == 0 || !strings.Contains(out, "already registered") {
		t.Fatalf("another user took fidika in a different case: exit %d\n%s", code, out)
	}
}

// A ref names its org and resource in any case: `Paul/SDXL` reads paul/sdxl.
func TestModelRefIgnoresCase(t *testing.T) {
	hub := newAccountHub(t)
	code, out := runCozy(t, t.TempDir(), "model", "search", "Paul/SDXL", "--tensorhub="+hub.URL)
	if code != 0 || !strings.Contains(out, "paul/sdxl") {
		t.Fatalf("mixed-case model ref: exit %d\n%s", code, out)
	}
}
