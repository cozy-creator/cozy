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
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/cozy-creator/cozy/internal/host"
)

// The readiness receipt, read as Tensorhub reads a pod's: the envelope over the media plane,
// authenticated under the key the launcher minted, naming the TLS leaf that served it.
const (
	receiptDomain   = host.ReceiptDomain
	maxReceiptBytes = 64 << 10
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
}

// receiptRefusal is a receipt that answered and did not verify: waiting cannot fix it.
type receiptRefusal struct{ reason string }

func (r *receiptRefusal) Error() string { return r.reason }

// runtimeGone is a Host whose Runtime cannot start: no readiness will come.
type runtimeGone struct{ reason string }

func (r *runtimeGone) Error() string { return r.reason }

var receiptClient = &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{
	// The leaf is unknown until the receipt names it; it is compared with the served one below.
	TLSClientConfig: &tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS12},
}}

func readReceipt(ctx context.Context, mediaPort int, key []byte) (receipt, []byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://127.0.0.1:"+strconv.Itoa(mediaPort)+"/v1/bootstrap/receipt", nil)
	if err != nil {
		return receipt{}, nil, err
	}
	response, err := receiptClient.Do(request)
	if err != nil {
		return receipt{}, nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, maxReceiptBytes+1))
	if err != nil {
		return receipt{}, nil, err
	}
	if response.StatusCode != http.StatusOK {
		var answer struct {
			Error struct{ Code, Message string }
		}
		if json.Unmarshal(body, &answer) == nil && answer.Error.Code == "machine.runtime_gone" {
			return receipt{}, nil, &runtimeGone{answer.Error.Message}
		}
		return receipt{}, nil, fmt.Errorf("receipt answered %d", response.StatusCode)
	}
	if len(body) > maxReceiptBytes || response.TLS == nil || len(response.TLS.PeerCertificates) == 0 {
		return receipt{}, nil, &receiptRefusal{"the receipt is oversized or not served over TLS"}
	}
	var envelope struct {
		Payload    []byte `json:"payload"`
		HMACSHA256 string `json:"hmac_sha256"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil || len(envelope.Payload) == 0 {
		return receipt{}, nil, &receiptRefusal{"the receipt envelope is malformed"}
	}
	want, err := hex.DecodeString(envelope.HMACSHA256)
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(receiptDomain))
	mac.Write(envelope.Payload)
	if err != nil || !hmac.Equal(mac.Sum(nil), want) {
		return receipt{}, nil, &receiptRefusal{"the receipt is not authenticated by this launch's key"}
	}
	var out receipt
	if err := json.Unmarshal(envelope.Payload, &out); err != nil || out.PodBootID == "" {
		return receipt{}, nil, &receiptRefusal{"the receipt names no boot"}
	}
	leaf, err := base64.StdEncoding.DecodeString(out.TLSCertificateDERBase64)
	if err != nil || !bytes.Equal(leaf, response.TLS.PeerCertificates[0].Raw) {
		return receipt{}, nil, &receiptRefusal{"the media plane serves a leaf the receipt does not name"}
	}
	return out, leaf, nil
}
