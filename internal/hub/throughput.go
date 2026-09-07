package hub

import (
	"context"
	"net/http"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// ModelThroughput is one measured (release, lane, sku) of a model under its reference
// workload (th-179): seconds per run and the pod's prepare time. The route answers the
// newest row per key; no row means unmeasured.
//
//	GET /v1/models/{org}/{name}/throughput -> {model, throughput: [row…]}
type ModelThroughput struct {
	Model             string    `json:"model"`
	Release           string    `json:"release"`
	Lane              string    `json:"lane"`
	SKU               string    `json:"sku"`
	Workload          string    `json:"workload"`
	PayloadSHA256     string    `json:"payload_sha256"`
	Runs              int       `json:"runs"`
	MedianS           float64   `json:"median_s"`
	MinS              float64   `json:"min_s"`
	MaxS              float64   `json:"max_s"`
	PrepareS          float64   `json:"prepare_s"`
	RuntimeVersion    string    `json:"runtime_version"`
	WorkerImageDigest string    `json:"worker_image_digest"`
	MeasuredAt        time.Time `json:"measured_at"`
}

// ModelThroughput reads the model's published throughput rows. Public. A hub without
// the route, or without the model, has measured nothing: no rows, no error.
func (c *Client) ModelThroughput(ctx context.Context, ref Ref) ([]ModelThroughput, *exit.Error) {
	var out struct {
		Throughput []ModelThroughput `json:"throughput"`
	}
	e := c.do(ctx, call{method: http.MethodGet, path: resourcePath("models", ref) + "/throughput"}, &out)
	if e != nil && e.Code == exit.NotFound {
		return nil, nil
	}
	return out.Throughput, e
}
