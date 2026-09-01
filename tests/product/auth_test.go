package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
)

const authTestDeviceID = "98bf8322-6a6d-45dd-ae23-f989509f1ec8"

var authBase64 = base64.RawURLEncoding

func TestEmailMachineLoginAndAutomaticReauthentication(t *testing.T) {
	challengeA := bytes.Repeat([]byte{0x11}, 32)
	challengeB := bytes.Repeat([]byte{0x22}, 32)
	expires := time.Now().UTC().Add(time.Hour).Format(time.RFC3339Nano)
	var mu sync.Mutex
	var public ed25519.PublicKey
	enrollBegins := 0
	loginBegins := 0
	accountAttempts := 0
	accountRegistrations := 0
	accountName := ""
	foreignModelRequests := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/auth/device-keys/enroll/begin":
			mu.Lock()
			enrollBegins++
			mu.Unlock()
			var body struct {
				Email     string `json:"email"`
				PublicKey string `json:"public_key"`
			}
			if !decodeAuthBody(t, r, &body) {
				return
			}
			decoded, err := authBase64.DecodeString(body.PublicKey)
			if body.Email != "person@example.com" || err != nil || len(decoded) != ed25519.PublicKeySize {
				t.Errorf("invalid enrollment declaration: email=%q public=%q error=%v", body.Email, body.PublicKey, err)
				return
			}
			public = append(ed25519.PublicKey(nil), decoded...)
			writeAuthJSON(t, w, http.StatusAccepted, map[string]any{
				"enrollment_id": "enroll-1", "challenge": authBase64.EncodeToString(challengeA),
				"expires_at": expires,
			})
		case "/v1/auth/device-keys/enroll/finish":
			var body map[string]string
			if !decodeAuthBody(t, r, &body) {
				return
			}
			if body["enrollment_id"] != "enroll-1" || body["code"] != "123456" ||
				!verifyAuthSignature(public, "authkit.device-key-enrollment/1", challengeA, body["signature"]) {
				t.Errorf("invalid enrollment proof: %+v", body)
				return
			}
			writeAuthToken(t, w, "first-access-token", expires)
		case "/v1/auth/device-keys/login/begin":
			var body map[string]string
			if !decodeAuthBody(t, r, &body) {
				return
			}
			if body["device_key_id"] != authTestDeviceID {
				t.Errorf("login device id = %q", body["device_key_id"])
				return
			}
			mu.Lock()
			loginBegins++
			mu.Unlock()
			writeAuthJSON(t, w, http.StatusAccepted, map[string]any{
				"challenge_id": "challenge-1", "challenge": authBase64.EncodeToString(challengeB),
				"expires_at": expires,
			})
		case "/v1/auth/device-keys/login/finish":
			var body map[string]string
			if !decodeAuthBody(t, r, &body) {
				return
			}
			if body["challenge_id"] != "challenge-1" ||
				!verifyAuthSignature(public, "authkit.device-key-login/1", challengeB, body["signature"]) {
				t.Errorf("invalid login proof: %+v", body)
				return
			}
			writeAuthToken(t, w, "second-access-token", expires)
		case "/v1/auth/device-keys/revoke-others":
			if r.Method != http.MethodPost || r.Header.Get("Authorization") != "Bearer first-access-token" {
				t.Errorf("revoke others = %s auth %q", r.Method, r.Header.Get("Authorization"))
				return
			}
			var body map[string]any
			if !decodeAuthBody(t, r, &body) || len(body) != 0 {
				t.Errorf("revoke others body = %+v", body)
				return
			}
			writeAuthJSON(t, w, http.StatusOK, map[string]bool{"ok": true})
		case "/v1/auth/device-keys/" + authTestDeviceID:
			if r.Method != http.MethodDelete || r.Header.Get("Authorization") != "Bearer second-access-token" {
				t.Errorf("logout = %s auth %q", r.Method, r.Header.Get("Authorization"))
				return
			}
			writeAuthJSON(t, w, http.StatusOK, map[string]bool{"ok": true})
		case "/v1/accounts/current":
			if authorization := r.Header.Get("Authorization"); authorization != "Bearer first-access-token" && authorization != "Bearer second-access-token" {
				t.Errorf("current-account read carried %q", authorization)
				return
			}
			mu.Lock()
			name := accountName
			mu.Unlock()
			if name == "" {
				writeAuthJSON(t, w, http.StatusConflict, map[string]any{"error": map[string]string{
					"code": "account.name_required", "message": "choose an account name",
					"remedy": "finish Tensorhub registration",
				}})
				return
			}
			writeAuthJSON(t, w, http.StatusOK, map[string]string{"name": name})
		case "/v1/accounts/local":
			if r.Method != http.MethodPut || r.Header.Get("Authorization") != "Bearer first-access-token" {
				t.Errorf("reserved account registration = %s auth %q", r.Method, r.Header.Get("Authorization"))
				return
			}
			mu.Lock()
			accountAttempts++
			mu.Unlock()
			writeAuthJSON(t, w, http.StatusUnprocessableEntity, map[string]any{"error": map[string]string{
				"code": "account.name_invalid", "message": "choose a valid, non-reserved account name",
				"remedy": "choose another account name",
			}})
		case "/v1/accounts/paul":
			if r.Method != http.MethodPut || r.Header.Get("Authorization") != "Bearer second-access-token" {
				t.Errorf("account registration = %s auth %q", r.Method, r.Header.Get("Authorization"))
				return
			}
			var body map[string]any
			if !decodeAuthBody(t, r, &body) || len(body) != 0 {
				t.Errorf("account registration body = %+v", body)
				return
			}
			mu.Lock()
			accountName = "paul"
			accountAttempts++
			accountRegistrations++
			mu.Unlock()
			writeAuthJSON(t, w, http.StatusOK, map[string]string{"name": "paul"})
		case "/v1/rentals":
			if r.Header.Get("Authorization") != "Bearer second-access-token" {
				t.Errorf("protected hub request carried %q", r.Header.Get("Authorization"))
				return
			}
			writeAuthJSON(t, w, http.StatusAccepted, map[string]any{
				"rental_id": "rental-auth-proof", "state": "pending_acquisition",
				"hourly_rate_usd_micros": int64(100_000),
			})
		case "/v1/models/foreign/model", "/v1/models/foreign/model/releases/stable":
			mu.Lock()
			foreignModelRequests++
			mu.Unlock()
			http.Error(w, "foreign model request crossed the account fence", http.StatusInternalServerError)
		default:
			t.Errorf("unexpected auth route %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	root := t.TempDir()
	before := runAuthCozy(t, root, server.URL, "", "auth", "--json")
	if before.code != 0 || !strings.Contains(before.stdout, `"status":"not logged in"`) || before.stderr != "" {
		t.Fatalf("status before login [exit %d]\nstdout: %s\nstderr: %s", before.code, before.stdout, before.stderr)
	}
	first := runAuthCozy(t, root, server.URL, "123456\nlocal\n", "auth", "login", "person@example.com", "--json")
	if first.code == 0 || !strings.Contains(first.stdout, `"code":"account.name_invalid"`) ||
		!strings.Contains(first.stderr, "A verification code was sent") ||
		!strings.Contains(first.stderr, "Tensorhub account name:") {
		t.Fatalf("reserved account registration [exit %d]\nstdout: %s\nstderr: %s", first.code, first.stdout, first.stderr)
	}

	files, err := filepath.Glob(filepath.Join(root, "auth", "*.json"))
	if err != nil || len(files) != 1 {
		t.Fatalf("machine credential files = %v, %v", files, err)
	}
	info, err := os.Stat(files[0])
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm() != 0o600 {
		t.Fatalf("machine credential mode = %#o", info.Mode().Perm())
	}
	stored, err := os.ReadFile(files[0])
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(stored), "first-access-token") || strings.Contains(string(stored), "second-access-token") {
		t.Fatal("short access token was persisted")
	}

	incomplete := runAuthCozy(t, root, server.URL, "", "auth", "--json")
	if incomplete.code != 0 || !strings.Contains(incomplete.stdout, `"status":"logged in"`) ||
		!strings.Contains(incomplete.stdout, `"email":"person@example.com"`) ||
		!strings.Contains(incomplete.stdout, `"account":"not registered"`) ||
		!strings.Contains(incomplete.stdout, `"cozy auth login <email>"`) || incomplete.stderr != "" {
		t.Fatalf("incomplete account status [exit %d]\nstdout: %s\nstderr: %s",
			incomplete.code, incomplete.stdout, incomplete.stderr)
	}

	retried := runAuthCozy(t, root, server.URL, "paul\n", "auth", "login", "person@example.com", "--json")
	if retried.code != 0 || !strings.Contains(retried.stdout, `"status":"authenticated"`) ||
		!strings.Contains(retried.stdout, `"account":"paul"`) ||
		!strings.Contains(retried.stderr, "Tensorhub account name:") ||
		strings.Contains(retried.stderr, "verification code") {
		t.Fatalf("account registration retry [exit %d]\nstdout: %s\nstderr: %s",
			retried.code, retried.stdout, retried.stderr)
	}

	second := runAuthCozy(t, root, server.URL, "", "auth", "login", "person@example.com", "--json")
	if second.code != 0 || !strings.Contains(second.stdout, `"status":"authenticated"`) ||
		!strings.Contains(second.stdout, `"account":"paul"`) || second.stderr != "" {
		t.Fatalf("automatic login [exit %d]\nstdout: %s\nstderr: %s", second.code, second.stdout, second.stderr)
	}
	status := runAuthCozy(t, root, server.URL, "", "auth", "--json")
	if status.code != 0 || !strings.Contains(status.stdout, `"status":"logged in"`) ||
		!strings.Contains(status.stdout, `"email":"person@example.com"`) ||
		!strings.Contains(status.stdout, `"account":"paul"`) || status.stderr != "" {
		t.Fatalf("status after login [exit %d]\nstdout: %s\nstderr: %s", status.code, status.stdout, status.stderr)
	}
	foreign := runAuthCozy(t, root, server.URL, "", "model", "publish", "foreign/model",
		"--release", "stable", "--lane", "bf16=sha256:"+strings.Repeat("a", 64))
	if foreign.code != 2 || !strings.Contains(foreign.stdout, "logged in as Tensorhub account paul") ||
		!strings.Contains(foreign.stdout, "publish as paul/model") {
		t.Fatalf("machine account accepted foreign publication [exit %d]\nstdout: %s\nstderr: %s",
			foreign.code, foreign.stdout, foreign.stderr)
	}
	manager := accountauth.New(config.Config{Home: root, HubURL: server.URL})
	hubClient := hub.New(config.Config{HubURL: server.URL}, "cozy-product-auth-test").WithTokenSource(manager)
	requestBody, problem := hub.RentalRequestBytes("cpu", strings.Repeat("1", 64),
		authBase64.EncodeToString(bytes.Repeat([]byte{1}, ed25519.PublicKeySize)))
	if problem != nil {
		t.Fatal(problem)
	}
	if rental, problem := hubClient.Rent(context.Background(), requestBody, "auth proof", "auth-proof"); problem != nil || rental.ID != "rental-auth-proof" {
		t.Fatalf("authenticated hub mutation = %+v, %v", rental, problem)
	}
	revoked := runAuthCozy(t, root, server.URL, "123456\n", "auth", "revoke-other-machines", "--json")
	if revoked.code != 0 || !strings.Contains(revoked.stdout, `"status":"other machines revoked"`) ||
		!strings.Contains(revoked.stderr, "A verification code was sent") {
		t.Fatalf("revoke other machines [exit %d]\nstdout: %s\nstderr: %s",
			revoked.code, revoked.stdout, revoked.stderr)
	}
	loggedOut := runAuthCozy(t, root, server.URL, "", "auth", "logout", "--json")
	if loggedOut.code != 0 || !strings.Contains(loggedOut.stdout, `"status":"logged out"`) || loggedOut.stderr != "" {
		t.Fatalf("logout [exit %d]\nstdout: %s\nstderr: %s", loggedOut.code, loggedOut.stdout, loggedOut.stderr)
	}
	afterLogout := runAuthCozy(t, root, server.URL, "", "auth", "--json")
	if afterLogout.code != 0 || !strings.Contains(afterLogout.stdout, `"status":"not logged in"`) || afterLogout.stderr != "" {
		t.Fatalf("status after logout [exit %d]\nstdout: %s\nstderr: %s",
			afterLogout.code, afterLogout.stdout, afterLogout.stderr)
	}
	mu.Lock()
	defer mu.Unlock()
	if enrollBegins != 2 || loginBegins != 8 {
		t.Fatalf("auth begin calls = enroll %d login %d, want 2/8", enrollBegins, loginBegins)
	}
	if foreignModelRequests != 0 {
		t.Fatalf("foreign model requests = %d, want 0", foreignModelRequests)
	}
	if accountAttempts != 2 || accountRegistrations != 1 || accountName != "paul" {
		t.Fatalf("account registration = attempts %d successes %d name %q",
			accountAttempts, accountRegistrations, accountName)
	}
}

type authCozyResult struct {
	code           int
	stdout, stderr string
}

func runAuthCozy(t *testing.T, root, hubURL, stdin string, args ...string) authCozyResult {
	t.Helper()
	cmd := exec.Command(cozyBin, args...)
	// A stale operator token must not shadow a valid machine key.
	cmd.Env = childEnv(t, root, "TENSORHUB_URL="+hubURL, "TENSORHUB_TOKEN=stale-static-token")
	cmd.Stdin = strings.NewReader(stdin) //cozy:stdin-value test drives the real email-code prompt
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	_ = cmd.Run()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	return authCozyResult{code: code, stdout: stdout.String(), stderr: stderr.String()}
}

func decodeAuthBody(t *testing.T, r *http.Request, out any) bool {
	t.Helper()
	if err := json.NewDecoder(r.Body).Decode(out); err != nil {
		t.Errorf("decode %s: %v", r.URL.Path, err)
		return false
	}
	return true
}

func writeAuthJSON(t *testing.T, w http.ResponseWriter, status int, value any) {
	t.Helper()
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(value); err != nil {
		t.Errorf("write auth answer: %v", err)
	}
}

func writeAuthToken(t *testing.T, w http.ResponseWriter, token, expires string) {
	t.Helper()
	writeAuthJSON(t, w, http.StatusOK, map[string]any{
		"access_token": token, "token_type": "Bearer", "expires_at": expires,
		"device_key": map[string]string{
			"id": authTestDeviceID, "label": "", "created_at": time.Now().UTC().Format(time.RFC3339Nano),
		},
	})
}

func verifyAuthSignature(public ed25519.PublicKey, domain string, challenge []byte, encoded string) bool {
	signature, err := authBase64.DecodeString(encoded)
	message := append(append([]byte(domain), 0), challenge...)
	return err == nil && ed25519.Verify(public, message, signature)
}
