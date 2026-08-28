package live

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"io"
	"math/big"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/mediawire"
	"github.com/cozy-creator/cozy-creator/internal/podmedia"
)

// TestPodMediaGrant is the pod's byte plane, run for real. It moved here with cl-036: the
// plane used to be a second binary (`cozy-media`) that `scripts/verify-hardening.sh`
// launched with an argv grant, and the merged `cozy-pod` has no argv to launch it with —
// it binds the plane in-process from the validated environment grant. The arms did not
// change shape, only their door: the same adversary digest sets, and the same admission
// matrix, now driven through the real `podmedia.Bind`/`Serve` on a real socket. The
// serving arms are STRONGER here than they were in bash, because the pod's own posture —
// off-loopback with a real TLS leaf — is what is exercised rather than a plaintext
// loopback stand-in.
//
// The renter token is minted in this test and never leaves it: the plane is given only its
// SHA-256, which is the whole point of the grant's shape.
func TestPodMediaGrant(t *testing.T) {
	token := "renter-token-live-proof"
	sum := sha256.Sum256([]byte(token))
	digest := hex.EncodeToString(sum[:])

	root := t.TempDir()
	receipt := filepath.Join(root, "readiness-envelope.json")
	envelope, err := json.Marshal(map[string]any{"payload": []byte("opaque"), "hmac_sha256": digest})
	must(t, err)
	must(t, os.WriteFile(receipt, envelope, 0o400))

	cert, key := podTLS(t, root)
	plane, err := podmedia.Bind(podmedia.Options{
		Listen:           "127.0.0.1:0",
		Root:             filepath.Join(root, "media"),
		Plans:            filepath.Join(root, "plans"),
		TokenHashes:      []string{"sha256:" + digest},
		Cert:             cert,
		Key:              key,
		BootstrapReceipt: receipt,
	})
	must(t, err)
	addr := plane.Addr()
	served := make(chan error, 1)
	go func() { served <- plane.Serve() }()
	t.Cleanup(func() {
		_ = plane.Close()
		select {
		case err := <-served:
			if err != nil {
				t.Errorf("the media plane ended with %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Error("the media plane did not stop when its listener closed")
		}
	})

	client := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true}, //cozy:allow the pod's leaf is minted at boot and is not a trusted-CA chain; the receipt's HMAC is what authenticates this peer
	}}
	get := func(t *testing.T, path, bearer string) *http.Response {
		t.Helper()
		req, err := http.NewRequest("GET", "https://"+addr+path, nil)
		must(t, err)
		if bearer != "" {
			req.Header.Set("Authorization", "Bearer "+bearer)
		}
		resp, err := client.Do(req)
		must(t, err)
		return resp
	}

	// The admission matrix. A digest set authenticates the matching bearer and NOBODY else;
	// there is no unauthenticated renter route.
	for _, arm := range []struct {
		name, bearer string
		want         int
	}{
		{"the matching bearer is admitted", token, http.StatusOK},
		{"a non-matching bearer is refused", "not-the-renters-token", http.StatusUnauthorized},
		{"no bearer is refused", "", http.StatusUnauthorized},
	} {
		t.Run(arm.name, func(t *testing.T) {
			resp := get(t, "/v1/health", arm.bearer)
			defer resp.Body.Close()
			if resp.StatusCode != arm.want {
				t.Fatalf("%s: got HTTP %d, want %d", arm.name, resp.StatusCode, arm.want)
			}
		})
	}

	// The health answer is the CONTRACT AND NOTHING ELSE (rev 2). It is checked against the
	// ONE declaration rather than against a second spelling of the field names here — the
	// fence forbids that spelling for the same reason this plane has a revision at all. So:
	// the served bytes must equal the declared document exactly, and it must carry exactly
	// the two fields that have a reader. A field added to the answer reddens this arm, which
	// is the point — an unread field is how a wire shape both ends must agree on grows.
	t.Run("the health answer publishes only the contract", func(t *testing.T) {
		resp := get(t, "/v1/health", token)
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		must(t, err)
		declared, err := json.Marshal(mediawire.Ours())
		must(t, err)
		if strings.TrimSpace(string(body)) != string(declared) {
			t.Fatalf("the health answer is not the declared document:\n got %s\nwant %s",
				strings.TrimSpace(string(body)), declared)
		}
		var said map[string]any
		must(t, json.Unmarshal(body, &said))
		if len(said) != 2 {
			t.Fatalf("the health answer carries %d fields, want exactly 2 (%s)", len(said), body)
		}
	})

	// The readiness envelope is served as EXACT BYTES and carries no capability: it is
	// the one route Tensorhub reads before it trusts the TLS peer, and it authenticates
	// the bytes with the attempt HMAC rather than the connection.
	t.Run("the readiness envelope is served byte-identically", func(t *testing.T) {
		resp := get(t, "/v1/bootstrap/receipt", "")
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("the receipt route answered HTTP %d", resp.StatusCode)
		}
		got := make([]byte, mediawire.MaxReceiptBytes+1)
		n, _ := resp.Body.Read(got)
		if string(got[:n]) != string(envelope) {
			t.Fatalf("the receipt was rewritten in flight:\n got %s\nwant %s", got[:n], envelope)
		}
	})
}

// TestPodMediaFailsClosed is the other half: a grant the plane cannot authenticate yields
// NO PLANE. Since cl-036 that is strictly stronger than it was — the plane binds inside
// PID 1 before any child exists, so a set that refuses to bind is a pod that does not boot
// rather than a child that dies after the adapter is already running.
func TestPodMediaFailsClosed(t *testing.T) {
	good := strings.Repeat("a", 64)
	for _, arm := range []struct {
		name   string
		hashes []string
	}{
		{"an absent digest set", nil},
		{"an empty digest set", []string{}},
		{"an empty entry", []string{""}},
		{"a non-hex digest", []string{"sha256:not-hex"}},
		{"an unprefixed digest", []string{good}},
		{"a truncated digest", []string{"sha256:" + good[:63]}},
		{"an uppercase digest", []string{"sha256:" + strings.ToUpper(good)}},
		{"an unsorted set", []string{"sha256:" + strings.Repeat("b", 64), "sha256:" + good}},
		{"a duplicated digest", []string{"sha256:" + good, "sha256:" + good}},
		{"a 17-digest set", oversizedSet()},
	} {
		t.Run(arm.name, func(t *testing.T) {
			plane, err := podmedia.Bind(podmedia.Options{
				Listen: "127.0.0.1:0", Root: t.TempDir(), TokenHashes: arm.hashes,
			})
			if err == nil {
				_ = plane.Close()
				t.Fatalf("%s bound anyway on %s — a malformed grant must authenticate "+
					"NOBODY, not everybody", arm.name, plane.Addr())
			}
		})
	}
}

func oversizedSet() []string {
	out := make([]string, 17)
	for i := range out {
		out[i] = "sha256:" + hex.EncodeToString([]byte{byte(i)}) + strings.Repeat("a", 62)
	}
	return out
}

// podTLS mints the same shape of self-signed leaf the pod mints at boot — an ECDSA P-256
// server leaf for `cozy-worker`, valid from now — so the serving arms above run against a
// real TLS handshake rather than a plaintext stand-in.
func podTLS(t *testing.T, dir string) (string, string) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	must(t, err)
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "cozy-pod"},
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
