package cli

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/config"
)

func TestDaemonHubCallsReuseOneMachineLogin(t *testing.T) {
	enrollmentChallenge := bytes.Repeat([]byte{0x11}, 32)
	loginChallenge := bytes.Repeat([]byte{0x22}, 32)
	expires := time.Now().UTC().Add(15 * time.Minute).Format(time.RFC3339Nano)
	const (
		deviceID        = "device-auth-cache-proof"
		enrollmentToken = "enrollment-access-token"
		loginToken      = "daemon-access-token"
		rentalID        = "pr-auth-cache-proof"
		baseDigest      = "sha256:2222222222222222222222222222222222222222222222222222222222222222"
	)
	var publicMu sync.Mutex
	var public ed25519.PublicKey
	var loginBegins atomic.Int32
	var loginFinishes atomic.Int32
	publicKey := func() ed25519.PublicKey {
		publicMu.Lock()
		defer publicMu.Unlock()
		return append(ed25519.PublicKey(nil), public...)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/v1/auth/device-keys/enroll/begin":
			var body struct {
				PublicKey string `json:"public_key"`
			}
			decodeAuthCacheBody(t, r, &body)
			decoded, err := base64.RawURLEncoding.DecodeString(body.PublicKey)
			if err != nil || len(decoded) != ed25519.PublicKeySize {
				t.Errorf("invalid enrollment public key: %v", err)
				return
			}
			publicMu.Lock()
			public = append(ed25519.PublicKey(nil), decoded...)
			publicMu.Unlock()
			writeAuthCacheJSON(t, w, http.StatusAccepted, map[string]string{
				"enrollment_id": "enrollment-1",
				"challenge":     base64.RawURLEncoding.EncodeToString(enrollmentChallenge),
				"expires_at":    expires,
			})
		case r.URL.Path == "/v1/auth/device-keys/enroll/finish":
			var body map[string]string
			decodeAuthCacheBody(t, r, &body)
			if body["enrollment_id"] != "enrollment-1" || body["code"] != "123456" ||
				!verifyAuthCacheSignature(publicKey(), "authkit.device-key-enrollment/1",
					enrollmentChallenge, body["signature"]) {
				t.Errorf("invalid enrollment proof")
				return
			}
			writeAuthCacheToken(t, w, enrollmentToken, deviceID, expires)
		case r.URL.Path == "/v1/auth/device-keys/login/begin":
			loginBegins.Add(1)
			writeAuthCacheJSON(t, w, http.StatusAccepted, map[string]string{
				"challenge_id": "login-1",
				"challenge":    base64.RawURLEncoding.EncodeToString(loginChallenge),
				"expires_at":   expires,
			})
		case r.URL.Path == "/v1/auth/device-keys/login/finish":
			var body map[string]string
			decodeAuthCacheBody(t, r, &body)
			if body["challenge_id"] != "login-1" ||
				!verifyAuthCacheSignature(publicKey(), "authkit.device-key-login/1",
					loginChallenge, body["signature"]) {
				t.Errorf("invalid login proof")
				return
			}
			loginFinishes.Add(1)
			writeAuthCacheToken(t, w, loginToken, deviceID, expires)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rental-skus":
			writeAuthCacheJSON(t, w, http.StatusOK, []map[string]any{{
				"name": "cpu", "accelerator_model": "CPU",
				"price_usd_micros_per_hour": int64(100_000),
			}})
		case r.Method == http.MethodPost && r.URL.Path == "/v1/rentals":
			requireAuthCacheToken(t, r, loginToken)
			writeRental := map[string]any{
				"rental_id": rentalID, "state": "pending_acquisition",
				"hourly_rate_usd_micros":     int64(100_000),
				"wheelhouse_manifest_digest": baseDigest,
			}
			writeAuthCacheJSON(t, w, http.StatusAccepted, writeRental)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rentals/"+rentalID:
			requireAuthCacheToken(t, r, loginToken)
			writeAuthCacheJSON(t, w, http.StatusOK, map[string]any{
				"rental_id": rentalID, "state": "pending_acquisition",
				"hourly_rate_usd_micros":     int64(100_000),
				"wheelhouse_manifest_digest": baseDigest,
			})
		case r.Method == http.MethodDelete && r.URL.Path == "/v1/rentals/"+rentalID:
			requireAuthCacheToken(t, r, loginToken)
			w.WriteHeader(http.StatusNoContent)
		default:
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	root := t.TempDir()
	cfg := config.Config{Home: root, HubURL: server.URL}
	enroller := accountauth.New(cfg)
	enrollment, _, problem := enroller.BeginEnrollment(context.Background(), "person@example.com")
	if problem != nil {
		t.Fatal(problem)
	}
	if _, problem := enroller.FinishEnrollment(context.Background(), enrollment, "123456"); problem != nil {
		t.Fatal(problem)
	}

	ctx := &Context{Cfg: cfg, AccountAuth: accountauth.New(cfg)}
	if _, problem := client(ctx).RentalSKUs(context.Background()); problem != nil {
		t.Fatal(problem)
	}
	if _, problem := client(ctx).Rent(context.Background(), []byte(`{}`), "", "operation-1"); problem != nil {
		t.Fatal(problem)
	}
	if _, problem := client(ctx).Rental(context.Background(), rentalID); problem != nil {
		t.Fatal(problem)
	}
	if problem := client(ctx).Release(context.Background(), rentalID, ""); problem != nil {
		t.Fatal(problem)
	}
	if got := loginBegins.Load(); got != 1 {
		t.Fatalf("machine login begins = %d, want 1", got)
	}
	if got := loginFinishes.Load(); got != 1 {
		t.Fatalf("machine login finishes = %d, want 1", got)
	}

	credentialFiles, err := filepath.Glob(filepath.Join(root, "auth", "*.json"))
	if err != nil || len(credentialFiles) != 1 {
		t.Fatalf("credential files = %v, error = %v", credentialFiles, err)
	}
	stored, err := os.ReadFile(credentialFiles[0])
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(stored), enrollmentToken) || strings.Contains(string(stored), loginToken) {
		t.Fatal("short access token was persisted")
	}
}

func decodeAuthCacheBody(t *testing.T, r *http.Request, out any) {
	t.Helper()
	if err := json.NewDecoder(r.Body).Decode(out); err != nil {
		t.Errorf("decode %s: %v", r.URL.Path, err)
	}
}

func writeAuthCacheJSON(t *testing.T, w http.ResponseWriter, status int, value any) {
	t.Helper()
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(value); err != nil {
		t.Errorf("write response: %v", err)
	}
}

func writeAuthCacheToken(t *testing.T, w http.ResponseWriter, token, deviceID, expires string) {
	t.Helper()
	writeAuthCacheJSON(t, w, http.StatusOK, map[string]any{
		"access_token": token, "token_type": "Bearer", "expires_at": expires,
		"device_key": map[string]string{
			"id": deviceID, "label": "", "created_at": time.Now().UTC().Format(time.RFC3339Nano),
		},
	})
}

func verifyAuthCacheSignature(public ed25519.PublicKey, domain string, challenge []byte,
	encoded string,
) bool {
	signature, err := base64.RawURLEncoding.DecodeString(encoded)
	message := append(append([]byte(domain), 0), challenge...)
	return err == nil && ed25519.Verify(public, message, signature)
}

func requireAuthCacheToken(t *testing.T, r *http.Request, token string) {
	t.Helper()
	if got := r.Header.Get("Authorization"); got != "Bearer "+token {
		t.Errorf("authorization = %q", got)
	}
}
