package producttest

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/secret"
)

// Version comes from the authenticated launch receipt. An old agent remains observable,
// but it must receive no delegated credential before its principal guard is available.
func TestOldAgentCannotReceiveDelegatedAuthority(t *testing.T) {
	for _, release := range []struct {
		version, module string
		safe            bool
	}{{"", machines.AgentModule, false}, {"0.1.0", machines.AgentModule, false}, {"0.1.1", machines.AgentModule, true},
		{"0.2.0+dev", machines.AgentModule, true}, {"0.1.14", "github.com/cozy-creator/cozy", false}} {
		t.Run(release.version, func(t *testing.T) {
			version := release.version
			var minted, attached atomic.Int32
			var account *httptest.Server
			account = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				minted.Add(1)
				_ = json.NewEncoder(w).Encode(map[string]any{"token": "scoped-fixture", "expires_at": time.Now().Add(time.Hour),
					"environment": map[string]string{"TENSORHUB_ORIGIN": account.URL}})
			}))
			defer account.Close()
			key := make([]byte, 32)
			var machine *httptest.Server
			machine = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/bootstrap/receipt" {
					payload, _ := json.Marshal(map[string]any{"pod_boot_id": "fixture-boot", "machine_version": version,
						"worker_internal_port":       machine.Listener.Addr().(*net.TCPAddr).Port,
						"tls_certificate_der_base64": base64.StdEncoding.EncodeToString(machine.TLS.Certificates[0].Certificate[0])})
					mac := hmac.New(sha256.New, key)
					_, _ = mac.Write([]byte("cozy.pod-readiness/1\x00"))
					_, _ = mac.Write(payload)
					_ = json.NewEncoder(w).Encode(map[string]any{"payload": payload, "hmac_sha256": hex.EncodeToString(mac.Sum(nil))})
					return
				}
				if r.URL.Path != "/v1/hubs/access" {
					t.Errorf("unexpected machine operation %s", r.URL.Path)
					http.NotFound(w, r)
					return
				}
				attached.Add(1)
				var grant struct {
					Origin  string `json:"origin"`
					Expires int64  `json:"expires_at"`
				}
				_ = json.NewDecoder(r.Body).Decode(&grant)
				_ = json.NewEncoder(w).Encode(map[string]any{"origin": grant.Origin, "expires_at": grant.Expires})
			}))
			defer machine.Close()
			dir := t.TempDir()
			host := machines.NewHost(dir, "", nil)
			binary := filepath.Join(host.Root(), "usr/local/bin/cozy-machine")
			self, err := os.Executable()
			must(t, err)
			must(t, os.MkdirAll(filepath.Dir(binary), 0755))
			must(t, os.Symlink(self, binary))
			port := machine.Listener.Addr().(*net.TCPAddr).Port
			record, err := json.Marshal(map[string]any{"pid": os.Getpid(), "worker_id": "fixture-machine", "worker_port": port,
				"media_port": port, "receipt_key": base64.RawURLEncoding.EncodeToString(key)})
			must(t, err)
			must(t, os.WriteFile(filepath.Join(dir, "host.json"), record, 0600))
			installed, err := json.Marshal(map[string]any{"host": map[string]string{"name": "cozy-machine", "module": release.module}, "host_pinned": true})
			must(t, err)
			must(t, os.WriteFile(filepath.Join(dir, "installed.json"), installed, 0600))
			launch, problem := host.Ensure(t.Context(), "", nil, true)
			fatal(t, problem)
			if launch.AgentVersion != version {
				t.Fatal("offline observation lost the attested agent version")
			}
			client := hub.New(config.Config{HubURL: account.URL, HubToken: secret.New("account-fixture")}, "test")
			_, problem = host.Ensure(t.Context(), account.URL, client, true)
			if !release.safe {
				if problem == nil || problem.ErrName() != "machine.agent_update_required" || minted.Load() != 0 || attached.Load() != 0 {
					t.Fatalf("old agent received delegated authority: %v; minted=%d attached=%d", problem, minted.Load(), attached.Load())
				}
			} else {
				fatal(t, problem)
				if minted.Load() != 1 || attached.Load() != 1 {
					t.Fatal("a compatible agent could not receive scoped access")
				}
			}
		})
	}
}
