// Package machineendpoint describes a pinned machine target, never its authority.
package machineendpoint

import (
	"crypto/sha256"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
)

const Format = "cozy.machine.endpoint/1"

type Endpoint struct {
	Format         string `json:"format"`
	Address        string `json:"address"`
	WorkerID       string `json:"worker_id"`
	WorkerBootID   string `json:"worker_boot_id"`
	CertificatePEM string `json:"tls_certificate_pem"`
	WorkspaceID    string `json:"execution_workspace_id"`
}

func Read(path string) (*Endpoint, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, 65537))
	if err != nil {
		return nil, err
	}
	if len(data) > 65536 {
		return nil, fmt.Errorf("machine endpoint record exceeds 64 KiB")
	}
	var endpoint Endpoint
	if err = json.Unmarshal(data, &endpoint); err != nil {
		return nil, err
	}
	return &endpoint, endpoint.Validate()
}

func (e Endpoint) Validate() error {
	if e.Format != Format {
		return fmt.Errorf("machine endpoint format is not %s", Format)
	}
	host, port, err := net.SplitHostPort(e.Address)
	if err != nil || host == "" || strings.ContainsAny(host, "/\x00\r\n") {
		return fmt.Errorf("machine address requires host:port")
	}
	n, err := strconv.Atoi(port)
	if err != nil || n < 1 || n > 65535 {
		return fmt.Errorf("machine endpoint port is invalid")
	}
	for name, value := range map[string]string{"worker_id": e.WorkerID, "worker_boot_id": e.WorkerBootID, "execution_workspace_id": e.WorkspaceID} {
		if value == "" || len(value) > 256 || strings.TrimSpace(value) != value {
			return fmt.Errorf("machine endpoint %s is invalid", name)
		}
	}
	_, err = e.Leaf()
	return err
}

func (e Endpoint) Leaf() ([]byte, error) {
	block, rest := pem.Decode([]byte(e.CertificatePEM))
	if block == nil || block.Type != "CERTIFICATE" || len(strings.TrimSpace(string(rest))) != 0 {
		return nil, fmt.Errorf("machine endpoint requires one pinned TLS leaf")
	}
	if _, err := x509.ParseCertificate(block.Bytes); err != nil {
		return nil, err
	}
	return block.Bytes, nil
}

func (e Endpoint) Name() string {
	leaf, _ := e.Leaf()
	h := sha256.New()
	_, _ = h.Write([]byte(e.Address + "\x00" + e.WorkerID + "\x00" + e.WorkspaceID + "\x00"))
	_, _ = h.Write(leaf)
	return "endpoint-" + hex.EncodeToString(h.Sum(nil))[:24]
}

func IsName(name string) bool { return strings.HasPrefix(name, "endpoint-") }
