package main

// THE POD'S FIXED FILESYSTEM LAYOUT, and the credentials minted into it at boot.
//
// These paths are the image's, not a configuration surface: the entrypoint takes no
// arguments and reads no configuration file, and the adapter learns every path it needs
// from the closed environment this process hands it. Nothing here is discovered.

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"time"
)

const (
	stateDir            = "/run/cozy/bootstrap"
	mediaRoot           = "/var/lib/cozy/media"
	plansDir            = "/run/cozy/binding-plans"
	podBootIDPath       = stateDir + "/pod-boot-id"
	certificatePath     = stateDir + "/tls.crt"
	privateKeyPath      = stateDir + "/tls.key"
	receiptPayloadPath  = stateDir + "/readiness-payload"
	receiptEnvelopePath = stateDir + "/readiness-envelope.json"
	tlsServerName       = "cozy-worker"
)

func prepareState(cfg config) error {
	for _, dir := range []string{stateDir, mediaRoot, plansDir} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return fmt.Errorf("create private runtime directory: %w", err)
		}
		if err := os.Chmod(dir, 0o700); err != nil {
			return fmt.Errorf("protect private runtime directory: %w", err)
		}
	}
	for _, path := range []string{receiptPayloadPath, receiptEnvelopePath} {
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("remove stale readiness handoff: %w", err)
		}
	}
	if err := mintPodBootID(); err != nil {
		return err
	}
	return mintCertificate(cfg.leaseExpiry)
}

func mintPodBootID() error {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return fmt.Errorf("mint pod_boot_id: %w", err)
	}
	// Every entrypoint boot is a new container boot observation. Replace any stale
	// handoff atomically; only this process and its child reuse it.
	return atomicWrite(podBootIDPath, []byte(base64.RawURLEncoding.EncodeToString(raw)+"\n"), 0o400)
}

// mintCertificate mints the pod's TLS leaf AGAINST THE RENTAL LEASE, and this is the whole
// reason COZY_RENTAL_LEASE_EXPIRY_UNIX is not called a boot or receipt deadline. The
// certificate must outlive the lease and nothing else: every renter request for the pod's
// whole paid life is served under it, so a value shortened to some plausible readiness
// timeout would not fail the boot — it would expire the serving cert mid-rental, long after
// anyone was watching. NotAfter is the lease plus one hour of clock slack, never less.
//
// The private key stays on the filesystem
// because the adapter's worker listener serves with it too; that this process now also
// holds the readiness HMAC key changes nothing an attacker could use, since owning this
// address space already meant owning the very credential the receipt exists to bind.
func mintCertificate(leaseExpiry time.Time) error {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return fmt.Errorf("mint TLS key: %w", err)
	}
	serialLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serial, err := rand.Int(rand.Reader, serialLimit)
	if err != nil {
		return fmt.Errorf("mint TLS serial: %w", err)
	}
	now := time.Now()
	template := certificateTemplate(serial, now, leaseExpiry)
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		return fmt.Errorf("mint TLS certificate: %w", err)
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return fmt.Errorf("encode TLS key: %w", err)
	}
	if err := atomicWrite(privateKeyPath, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o400); err != nil {
		return err
	}
	return atomicWrite(certificatePath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o444)
}

func certificateTemplate(serial *big.Int, now, leaseExpiry time.Time) *x509.Certificate {
	return &x509.Certificate{
		SerialNumber: serial,
		Subject:      pkix.Name{CommonName: "cozy-pod"},
		NotBefore:    now.Add(-5 * time.Minute),
		NotAfter:     leaseExpiry.Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames:     []string{tlsServerName},
	}
}

func atomicWrite(target string, data []byte, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
		return err
	}
	temp, err := os.CreateTemp(filepath.Dir(target), ".atomic-*")
	if err != nil {
		return err
	}
	tempName := temp.Name()
	defer os.Remove(tempName)
	if err := temp.Chmod(mode); err != nil {
		temp.Close()
		return err
	}
	if _, err := temp.Write(data); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Sync(); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tempName, target); err != nil {
		return err
	}
	return syncDirectory(filepath.Dir(target))
}

func syncDirectory(path string) error {
	dir, err := os.Open(path)
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}
