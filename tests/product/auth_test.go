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
	loginBegins := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/auth/device-keys/enroll/begin":
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
		case "/v1/auth/me":
			if authorization := r.Header.Get("Authorization"); authorization != "Bearer first-access-token" && authorization != "Bearer second-access-token" {
				t.Errorf("current-user read carried %q", authorization)
				return
			}
			writeAuthJSON(t, w, http.StatusOK, map[string]any{
				"id": "140e338a-ebd5-48c5-a124-703f2457195a", "email": "person@example.com",
				"email_verified": true, "entitlements": []string{},
				"availability": []map[string]any{{"action": "update_username", "allowed": true}},
			})
		case "/v1/rentals":
			if r.Header.Get("Authorization") != "Bearer second-access-token" {
				t.Errorf("protected hub request carried %q", r.Header.Get("Authorization"))
				return
			}
			writeAuthJSON(t, w, http.StatusAccepted, map[string]string{
				"rental_id": "rental-auth-proof", "state": "pending_acquisition",
			})
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
	first := runAuthCozy(t, root, server.URL, "123456\n", "auth", "login", "person@example.com", "--json")
	if first.code != 0 || !strings.Contains(first.stdout, `"status":"registered"`) ||
		strings.Contains(first.stdout, "personal_org") ||
		!strings.Contains(first.stderr, "A verification code was sent") {
		t.Fatalf("first login [exit %d]\nstdout: %s\nstderr: %s", first.code, first.stdout, first.stderr)
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

	second := runAuthCozy(t, root, server.URL, "", "auth", "login", "person@example.com", "--json")
	if second.code != 0 || !strings.Contains(second.stdout, `"status":"authenticated"`) || second.stderr != "" {
		t.Fatalf("automatic login [exit %d]\nstdout: %s\nstderr: %s", second.code, second.stdout, second.stderr)
	}
	status := runAuthCozy(t, root, server.URL, "", "auth", "--json")
	if status.code != 0 || !strings.Contains(status.stdout, `"status":"logged in"`) ||
		!strings.Contains(status.stdout, `"email":"person@example.com"`) || status.stderr != "" {
		t.Fatalf("status after login [exit %d]\nstdout: %s\nstderr: %s", status.code, status.stdout, status.stderr)
	}
	manager := accountauth.New(config.Config{Home: root, HubURL: server.URL})
	hubClient := hub.New(config.Config{HubURL: server.URL}, "cozy-product-auth-test").WithTokenSource(manager)
	requestBody, problem := hub.RentalRequestBytes("proof/marco@release", "cpu", strings.Repeat("1", 64),
		authBase64.EncodeToString(bytes.Repeat([]byte{1}, ed25519.PublicKeySize)))
	if problem != nil {
		t.Fatal(problem)
	}
	if rental, problem := hubClient.Rent(context.Background(), requestBody, "auth proof", "auth-proof"); problem != nil || rental.ID != "rental-auth-proof" {
		t.Fatalf("authenticated hub mutation = %+v, %v", rental, problem)
	}
	mu.Lock()
	defer mu.Unlock()
	if loginBegins != 3 {
		t.Fatalf("login begin calls = %d, want 3", loginBegins)
	}
}

type authCozyResult struct {
	code           int
	stdout, stderr string
}

func runAuthCozy(t *testing.T, root, hubURL, stdin string, args ...string) authCozyResult {
	t.Helper()
	cmd := exec.Command(cozyBin, args...)
	cmd.Env = childEnv(t, root, "TENSORHUB_URL="+hubURL)
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
