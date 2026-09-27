package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestPublishedRentalPreparationUsesPersistedMachineKey(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	challenge := bytes.Repeat([]byte{17}, 32)
	var token atomic.Value
	token.Store("")
	var minted, prepared, publicationReads, publicationRequests atomic.Int64
	var origin string
	proof := startPublishedMachineHost(t, func(h *fakeRentalHub) {
		origin = h.server.URL
		fallback := h.server.Config.Handler
		h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			switch r.URL.Path {
			case "/v1/auth/device-keys/login/begin":
				var body map[string]string
				if json.NewDecoder(r.Body).Decode(&body) != nil || body["device_key_id"] != "prepare-machine" {
					http.Error(w, "invalid machine", 401)
					return
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"challenge_id": "prepare-login", "challenge": base64.RawURLEncoding.EncodeToString(challenge), "expires_at": time.Now().Add(time.Minute).UTC().Format(time.RFC3339)})
			case "/v1/auth/device-keys/login/finish":
				var body map[string]string
				if json.NewDecoder(r.Body).Decode(&body) != nil {
					http.Error(w, "invalid login", 401)
					return
				}
				signature, err := base64.RawURLEncoding.DecodeString(body["signature"])
				signed := append([]byte("authkit.device-key-login/1\x00"), challenge...)
				if err != nil || body["challenge_id"] != "prepare-login" || !ed25519.Verify(public, signed, signature) {
					http.Error(w, "invalid signature", 401)
					return
				}
				minted.Add(1)
				_ = json.NewEncoder(w).Encode(map[string]any{"token_set": map[string]any{"access_token": token.Load().(string), "token_type": "Bearer", "expires_in": 3600}, "device_key": map[string]string{"id": "prepare-machine"}})
			case "/v1/machine-authorizations":
				if r.Header.Get("Authorization") != "Bearer "+token.Load().(string) || minted.Load() == 0 {
					http.Error(w, "no machine authentication", 401)
					return
				}
				publicationRequests.Add(1)
				w.WriteHeader(http.StatusForbidden)
				_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"code": "publication.auth_probe", "message": "controlled stop before granting publication"}})
			default:
				if r.URL.Path == "/v1/rentals/rental-private-child-host" && r.Header.Get("Authorization") == "Bearer "+token.Load().(string) {
					publicationReads.Add(1)
				}
				if strings.HasSuffix(r.URL.Path, "/prepare-facts") {
					if r.Header.Get("Authorization") != "Bearer "+token.Load().(string) || minted.Load() == 0 {
						http.Error(w, "no machine authentication", 401)
						return
					}
					prepared.Add(1)
				}
				fallback.ServeHTTP(w, r)
			}
		})
	})
	configPath := filepath.Join(proof.Layout.Root, "config.yaml")
	configBytes, err := os.ReadFile(configPath)
	must(t, err)
	var lines []string
	for _, line := range strings.Split(string(configBytes), "\n") {
		if strings.HasPrefix(line, "tensorhub_token: ") {
			token.Store(strings.TrimPrefix(line, "tensorhub_token: "))
			continue
		}
		lines = append(lines, line)
	}
	if token.Load().(string) == "" {
		t.Fatal("fixture omitted its test authority")
	}
	must(t, os.WriteFile(configPath, []byte(strings.Join(lines, "\n")), 0600))
	key := sha256.Sum256([]byte(origin))
	directory := filepath.Join(proof.Layout.Root, "auth")
	must(t, os.MkdirAll(directory, 0700))
	credential, err := json.Marshal(map[string]any{"version": 1, "hub": origin, "email": "prepare@example.test", "device_key_id": "prepare-machine", "private_key": base64.RawURLEncoding.EncodeToString(private.Seed())})
	must(t, err)
	must(t, os.WriteFile(filepath.Join(directory, hex.EncodeToString(key[:])+".json"), credential, 0600))
	code, output := runCozyPath(t, proof.Layout.Root, proof.Path, "run", proof.Package.Package+"/main", "--rental", "child-host", "--await", "--json", "--idempotency-key", "machine-key-prepare")
	if code != 0 {
		t.Fatalf("machine-key-only published CLI [%d]: %s", code, output)
	}
	request, problem := proof.Store.RequestByIdempotencyKey("machine-key-prepare")
	fatal(t, problem)
	link, problem := proof.Store.MachineExecution(request.ID)
	fatal(t, problem)
	if request.State != "succeeded" || link == nil || !link.Collected || minted.Load() == 0 || prepared.Load() == 0 {
		t.Fatalf("machine-key preparation was not exercised: state=%s minted=%d prepares=%d", request.State, minted.Load(), prepared.Load())
	}
	t.Logf("machine-key-only published request %s completed and collected; minted=%d authenticated prepare-facts=%d", request.ID, minted.Load(), prepared.Load())
	before := publicationReads.Load()
	code, output = runCozyPath(t, proof.Layout.Root, proof.Path, "run", proof.Package.Package+"/main", "--rental", "child-host", "--allow-upload", "proof/checkpoint", "--await", "--json", "--idempotency-key", "machine-key-publication")
	if code == 0 || !strings.Contains(output, "controlled stop before granting publication") || publicationReads.Load() <= before || publicationRequests.Load() != 1 {
		t.Fatalf("publication machine-key bridge [%d], reads=%d grants=%d: %s", code, publicationReads.Load()-before, publicationRequests.Load(), output)
	}
	request, problem = proof.Store.RequestByIdempotencyKey("machine-key-publication")
	fatal(t, problem)
	link, problem = proof.Store.MachineExecution(request.ID)
	fatal(t, problem)
	if link == nil || len(link.Receipt) != 0 {
		t.Fatal("controlled refusal admitted an execution")
	}
	t.Log("authenticated rental readback and scoped publication request reached the server; no grant, execution or upload occurred")
}
