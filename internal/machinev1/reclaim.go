package machinev1

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

var ErrReclaimUnavailable = errors.New("this machine does not support owner idle memory reclaim")
var ErrReclaimBusy = errors.New("GPU work or an unknown startup reservation prevents idle memory reclaim")

type IdleMemoryReclaim struct {
	ExecutorsBefore          uint64   `json:"executors_before"`
	ExecutorsEnded           uint64   `json:"executors_ended"`
	ExecutorsUnconfirmed     uint64   `json:"executors_unconfirmed"`
	HoldingsBefore           uint64   `json:"holdings_before"`
	HoldingsRevoked          uint64   `json:"holdings_revoked"`
	HoldingsReleased         uint64   `json:"holdings_released"`
	RootExportBytesReleased  uint64   `json:"root_export_bytes_released"`
	RootExportBytesRemaining uint64   `json:"root_export_bytes_remaining"`
	ReadersRemaining         uint64   `json:"readers_remaining"`
	Warnings                 []string `json:"warnings"`
}

// ReclaimIdleMemory is explicit owner maintenance over the same pinned machine.
// It submits no work and controls no run. The capability never leaves this request.
func (c *Client) ReclaimIdleMemory(ctx context.Context) (IdleMemoryReclaim, error) {
	var result IdleMemoryReclaim
	token, err := c.Cap("", nil, 5*time.Minute)
	if err != nil {
		return result, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://"+c.address+"/v1/machine/memory/reclaim", nil)
	if err != nil {
		return result, err
	}
	request.Header.Set("Authorization", "Cozy-Cap "+token)
	response, err := c.http.Do(request)
	if err != nil {
		return result, err
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(response.Body, (64<<10)+1))
	if err != nil {
		return result, err
	}
	if len(raw) > 64<<10 {
		return result, errors.New("owner memory reclaim answer exceeds 64 KiB")
	}
	switch response.StatusCode {
	case http.StatusNotFound, http.StatusNotImplemented:
		return result, ErrReclaimUnavailable
	case http.StatusConflict:
		return result, ErrReclaimBusy
	case http.StatusOK:
		if err := json.Unmarshal(raw, &result); err != nil {
			return result, fmt.Errorf("owner memory reclaim answer is unreadable: %w", err)
		}
		return result, nil
	default:
		return result, fmt.Errorf("owner memory reclaim refused: HTTP %d", response.StatusCode)
	}
}
