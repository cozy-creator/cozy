package workertls

import (
	"bytes"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"os"
)

// Pin is one worker's exact certificate: the leaf the pod presents must be byte-for-byte
// the certificate pinned at rental time. A root pool would also admit anything that
// certificate had SIGNED; a pin admits one thing.
type Pin struct{ der []byte }

// DER is the pinned leaf's exact bytes: the identity a publication grant binds.
func (p *Pin) DER() []byte { return append([]byte(nil), p.der...) }

func (p *Pin) Digest() []byte {
	sum := sha256.Sum256(p.der)
	return append([]byte(nil), sum[:]...)
}

// LoadPin reads the pinned PEM and keeps its first CERTIFICATE block's DER.
func LoadPin(path string) (*Pin, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading the pinned worker cert: %w", err)
	}
	for {
		var block *pem.Block
		block, data = pem.Decode(data)
		if block == nil {
			return nil, fmt.Errorf("%s holds no usable certificate to pin", path)
		}
		if block.Type != "CERTIFICATE" {
			continue
		}
		if _, err := x509.ParseCertificate(block.Bytes); err != nil {
			return nil, fmt.Errorf("%s: pinned certificate is unreadable: %w", path, err)
		}
		return &Pin{der: block.Bytes}, nil
	}
}

// TLSConfig verifies ONLY the pin: chain building and hostname checks are skipped because
// the pinned leaf is self-signed and dialled by a provider-read-back IP; ServerName is
// kept so the pod's SNI routing sees the name its certificate carries.
func (p *Pin) TLSConfig() *tls.Config {
	return &tls.Config{
		MinVersion:         tls.VersionTLS12,
		ServerName:         ServerName,
		InsecureSkipVerify: true, // replaced by the exact-bytes check below
		VerifyPeerCertificate: func(raw [][]byte, _ [][]*x509.Certificate) error {
			if len(raw) == 0 {
				return fmt.Errorf("the worker presented no certificate")
			}
			if !bytes.Equal(raw[0], p.der) {
				return fmt.Errorf("the worker's certificate is not the one pinned for this rental")
			}
			return nil
		},
	}
}
