package hub

import (
	"context"
	"net/http"
	"net/url"

	"github.com/cozy-creator/cozy/internal/exit"
)

type RuntimeUpdateWheel struct {
	Version  string `json:"version"`
	Filename string `json:"filename"`
	URL      string `json:"url"`
	Digest   string `json:"digest"`
	Length   int64  `json:"length"`
}

type RuntimeUpdateTarget struct {
	RentalID      string `json:"rental_id"`
	WorkerBootID  string `json:"worker_boot_id"`
	ImageDigest   string `json:"image_digest"`
	RuntimeUpdate struct {
		Runtime  RuntimeUpdateWheel `json:"runtime"`
		TensorFS RuntimeUpdateWheel `json:"tensorfs"`
	} `json:"runtime_update"`
}

func (c *Client) RentalRuntimeUpdateTarget(ctx context.Context, rental string) (RuntimeUpdateTarget, *exit.Error) {
	var result RuntimeUpdateTarget
	if problem := validateRentalID(rental); problem != nil {
		return result, problem
	}
	problem := c.do(ctx, call{method: http.MethodGet, path: "/v1/rentals/" + url.PathEscape(rental) + "/runtime-update-target", auth: true}, &result)
	if problem == nil && (result.RentalID != rental || result.WorkerBootID == "" || result.ImageDigest == "" || result.RuntimeUpdate.Runtime.Version == "" || result.RuntimeUpdate.TensorFS.Version == "") {
		problem = exit.New(exit.Structural, "Tensorhub returned an incomplete approved Runtime update target")
	}
	return result, problem
}
