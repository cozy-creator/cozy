// Package publication owns explicit checkpoint/release effects outside CLI parsing.
package publication

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

// ReleaseRequest is caller intent. A nil revision reads and freezes the current one.
type ReleaseRequest struct {
	Destination      string            `json:"destination"`
	Release          string            `json:"release"`
	Lanes            map[string]string `json:"lanes"`
	ExpectedRevision *int64            `json:"expected_revision"`
}

// ReleaseIntent is recorded durably before mutation. It never rebases on later state.
type ReleaseIntent struct {
	Request          ReleaseRequest    `json:"request"`
	BaselineRevision int64             `json:"baseline_revision"`
	Baseline         map[string]string `json:"baseline"`
	Desired          map[string]string `json:"desired"`
}

type ReleaseReceipt struct {
	Destination string            `json:"destination"`
	Release     string            `json:"release"`
	Revision    int64             `json:"revision"`
	Lanes       map[string]string `json:"lanes"`
	Observation string            `json:"observation"`
}

type Releases interface {
	ModelRelease(context.Context, hub.Ref, string) (hub.ModelRelease, *exit.Error)
	UpdateModelRelease(context.Context, hub.Ref, string, int64, map[string]string, []string, string, string) (hub.ModelRelease, *exit.Error)
}
