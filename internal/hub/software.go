package hub

import (
	"context"
	"net/http"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Software is the Hub's target software: the published Runtime and TensorFS versions a
// controller brings each machine boot to. Both empty: none.
type Software struct {
	Runtime  string `json:"runtime"`
	TensorFS string `json:"tensorfs"`
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
