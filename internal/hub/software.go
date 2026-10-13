package hub

import (
	"context"
	"net/http"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Software is the Runtime and TensorFS the Hub's machines run: what its current worker image
// bakes, which a rental already runs and a local machine follows, or, Pinned, a rollback or
// forward fix every machine boot moves to. Both empty: none.
type Software struct {
	Runtime  string `json:"runtime"`
	TensorFS string `json:"tensorfs"`
	Pinned   bool   `json:"pinned"`
}

// Software reads the target. A Hub that predates it names none.
func (c *Client) Software(ctx context.Context) (Software, *exit.Error) {
	var out Software
	problem := c.do(ctx, call{method: http.MethodGet, path: "/v1/software"}, &out)
	if problem != nil && problem.Code == exit.NotFound {
		return Software{}, nil
	}
	return out, problem
}
