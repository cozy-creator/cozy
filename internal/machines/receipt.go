package machines

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/tls"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/cozy-creator/cozy/internal/machinev1"
)

// The readiness receipt, read as Tensorhub reads a machine's: the envelope from Status,
// authenticated under the key the launcher minted, naming the TLS leaf that served it.
const (
	ReadinessReceiptDomain = "cozy.pod-readiness/1\x00"
	maxReceiptBytes        = 64 << 10
)

// ReceiptGPU is one device the Runtime measured, as the receipt names it.
type ReceiptGPU struct {
	Index    int    `json:"device_index"`
	Name     string `json:"device_name"`
	UUID     string `json:"device_uuid"`
	Memory   uint64 `json:"memory_bytes"`
	PCIBusID string `json:"pci_bus_id"`
}

type receipt struct {
	PodBootID               string       `json:"pod_boot_id"`
	WorkerInternalPort      int          `json:"worker_internal_port"`
	TLSCertificateDERBase64 string       `json:"tls_certificate_der_base64"`
	RuntimeGPUs             []ReceiptGPU `json:"runtime_gpus"`
	MachineVersion          string       `json:"machine_version"`
	MachineCapabilities     []string     `json:"machine_capabilities"`
}

// receiptRefusal is a receipt that answered and did not verify: waiting cannot fix it.
type receiptRefusal struct{ reason string }

func (r *receiptRefusal) Error() string { return r.reason }

// readReceipt reads the machine's sealed receipt from its Status, which it answers without a
// capability, authenticates it under the key this launch minted, and checks it names the TLS
// leaf that served it.
func readReceipt(ctx context.Context, workerPort int, key []byte) (receipt, []byte, error) {
	// The leaf is unknown until the receipt names it; it is compared with the served one below.
	tlsConfig := &tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS12}
	frame, served, err := machinev1.Identity(ctx, "127.0.0.1:"+strconv.Itoa(workerPort), tlsConfig)
	if err != nil {
		return receipt{}, nil, err
	}
	if len(frame.GetReceipt()) == 0 {
		return receipt{}, nil, fmt.Errorf("the machine has not sealed its receipt yet")
	}
	if len(frame.GetReceipt()) > maxReceiptBytes || len(served) == 0 {
		return receipt{}, nil, &receiptRefusal{"the receipt is oversized or not served over TLS"}
	}
	var envelope struct {
		Payload    []byte `json:"payload"`
		HMACSHA256 string `json:"hmac_sha256"`
	}
	if err := json.Unmarshal(frame.GetReceipt(), &envelope); err != nil || len(envelope.Payload) == 0 {
		return receipt{}, nil, &receiptRefusal{"the receipt envelope is malformed"}
	}
	want, err := hex.DecodeString(envelope.HMACSHA256)
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(ReadinessReceiptDomain))
	mac.Write(envelope.Payload)
	if err != nil || !hmac.Equal(mac.Sum(nil), want) {
		return receipt{}, nil, &receiptRefusal{"the receipt is not authenticated by this launch's key"}
	}
	var out receipt
	if err := json.Unmarshal(envelope.Payload, &out); err != nil || out.PodBootID == "" {
		return receipt{}, nil, &receiptRefusal{"the receipt names no boot"}
	}
	leaf, err := base64.StdEncoding.DecodeString(out.TLSCertificateDERBase64)
	if err != nil || !bytes.Equal(leaf, served) {
		return receipt{}, nil, &receiptRefusal{"the machine serves a leaf the receipt does not name"}
	}
	return out, leaf, nil
}
