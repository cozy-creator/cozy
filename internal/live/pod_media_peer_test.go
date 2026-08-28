package live

// This is the Creator client's independent media-protocol peer. The production
// server is owned and verified in Tensorhub; retaining it here would restore a
// second implementation. These remote-owner tests need only the one health
// exchange the client performs before it sends a worker placement.

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/media"
	"github.com/cozy-creator/cozy-creator/internal/mediawire"
	"github.com/cozy-creator/cozy-creator/internal/secret"
)

func serveMediaPeer(t *testing.T, certPath, keyPath, token string, revision int) string {
	t.Helper()
	certificate, err := tls.LoadX509KeyPair(certPath, keyPath)
	must(t, err)
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/health" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer "+token {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(mediawire.Health{
			Service: mediawire.Service, ContractRev: &revision,
		})
	})
	peer := httptest.NewUnstartedServer(handler)
	peer.TLS = &tls.Config{Certificates: []tls.Certificate{certificate}, MinVersion: tls.VersionTLS13}
	peer.StartTLS()
	t.Cleanup(peer.Close)
	return strings.TrimPrefix(peer.URL, "https://")
}

func TestMediaClientRefusesContractSkew(t *testing.T) {
	root := t.TempDir()
	certPath, keyPath := podTLS(t, root)
	tokenText := "contract-skew-token"
	addr := serveMediaPeer(t, certPath, keyPath, tokenText, mediawire.ContractRev+1)
	client, problem := media.Dial(media.Spec{
		Addr: addr, Token: secret.New(tokenText), CACert: certPath,
	}, time.Second, 1<<20)
	fatal(t, problem)
	problem = client.Health()
	if problem == nil || problem.ErrName() != "media_contract_mismatch" {
		t.Fatalf("contract-skew health = %#v, want media_contract_mismatch", problem)
	}
}

// podTLS mints the same self-signed leaf shape used by the pod. The retained
// remote-placement tests use it for both their independent worker and media
// peers, so the real client still exercises exact certificate pinning.
func podTLS(t *testing.T, dir string) (string, string) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	must(t, err)
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "pod-supervisor"},
		NotBefore:    time.Now().Add(-5 * time.Minute),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames:     []string{"cozy-worker"},
		IPAddresses:  []net.IP{net.ParseIP("127.0.0.1")},
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	must(t, err)
	keyDER, err := x509.MarshalECPrivateKey(key)
	must(t, err)
	certPath := filepath.Join(dir, "tls.crt")
	keyPath := filepath.Join(dir, "tls.key")
	must(t, os.WriteFile(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o444))
	must(t, os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o400))
	return certPath, keyPath
}
